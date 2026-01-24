"""
Semantic Extraction Strategy.

Semantic embedding-based extraction using Vietnamese sentence embeddings (PhoBERT).
Uses semantic similarity to match questions with predefined object descriptions.
"""
import logging
from typing import Dict, List, Set, Tuple, Optional, Any

from ..base import ExtractionStrategy
from ..models import QueryAnalysisResult
from ..constants import (
    SEMANTIC_OBJECT_DESCRIPTIONS,
    SEMANTIC_SIMILARITY_THRESHOLD,
    NATIVE_SBERT_MODELS,
)
from .keyword import KeywordExtractionStrategy

logger = logging.getLogger(__name__)

class SemanticExtractionStrategy(ExtractionStrategy):
    """
    Semantic embedding-based extraction using Vietnamese sentence embeddings.
    
    Uses semantic similarity to match questions with predefined object descriptions.
    More powerful but requires model loading.
    
    Supported models:
    - vinai/phobert-base: Raw PhoBERT model (will be wrapped with pooling layer)
    - dangvantuan/vietnamese-embedding: Pre-trained sentence-transformers model
    - keepitreal/vietnamese-sbert: Vietnamese SBERT model
    - AITeamVN/Vietnamese_Embedding: Fine-tuned from BGE-M3
    
    Note: For raw Hugging Face models like vinai/phobert-base, we explicitly
    configure the Transformer + Pooling modules to avoid warnings.
    """
    
    def __init__(self, model_name: str = "vinai/phobert-base"):
        self._model_name = model_name
        self._model = None
        self._embeddings_cache: Dict[str, Any] = {}
        self._object_embeddings: Optional[Dict[str, Any]] = None
        self._keyword_strategy = KeywordExtractionStrategy()
        self._init_model()
        self._precompute_object_embeddings()
    
    @property
    def name(self) -> str:
        return "semantic"
    
    def _init_model(self):
        """Initialize the embedding model.
        
        For raw Hugging Face models (like vinai/phobert-base), we explicitly
        create the SentenceTransformer with Transformer and Pooling modules.
        This avoids the warning about no sentence-transformers model found.
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            # Check if this is a native sentence-transformers model
            if self._model_name in NATIVE_SBERT_MODELS:
                # Load directly - it has the proper config
                self._model = SentenceTransformer(self._model_name)
                logger.info(f"Loaded native sentence-transformers model: {self._model_name}")
            else:
                # For raw HuggingFace models like vinai/phobert-base,
                # explicitly configure the modules to avoid warnings
                self._model = self._load_huggingface_model_as_sbert()
                
        except ImportError:
            logger.warning(
                "sentence-transformers not available. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.warning(f"Failed to load model {self._model_name}: {e}")
    
    def _load_huggingface_model_as_sbert(self):
        """Load a raw HuggingFace model and wrap it for sentence embeddings.
        
        This creates a proper SentenceTransformer by explicitly defining:
        1. Transformer module: loads the HuggingFace model
        2. Pooling module: applies mean pooling over token embeddings
        
        Returns:
            SentenceTransformer: The wrapped model ready for sentence embeddings
        """
        from sentence_transformers import SentenceTransformer, models
        
        # Load the transformer model from HuggingFace
        word_embedding_model = models.Transformer(self._model_name)
        
        # Add mean pooling layer
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        
        # Create the SentenceTransformer with explicit modules
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        logger.info(f"Loaded HuggingFace model as SentenceTransformer: {self._model_name}")
        
        return model
    
    def _precompute_object_embeddings(self):
        """Precompute embeddings for all object descriptions.
        
        This is done once at initialization for faster inference later.
        """
        if self._model is None:
            return
        
        try:
            import numpy as np
            
            self._object_embeddings = {}
            descriptions = []
            object_types = []
            
            for obj_type, (desc, _) in SEMANTIC_OBJECT_DESCRIPTIONS.items():
                descriptions.append(desc)
                object_types.append(obj_type)
            
            # Batch encode all descriptions
            embeddings = self._model.encode(descriptions, convert_to_numpy=True)
            
            # Store embeddings by object type
            for i, obj_type in enumerate(object_types):
                self._object_embeddings[obj_type] = embeddings[i]
            
            logger.info(f"Precomputed embeddings for {len(self._object_embeddings)} object types")
            
        except Exception as e:
            logger.warning(f"Failed to precompute object embeddings: {e}")
            self._object_embeddings = None
    
    def _compute_similarity(self, query_embedding, object_embedding) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            query_embedding: Embedding of the question
            object_embedding: Embedding of the object description
            
        Returns:
            Cosine similarity score (0-1)
        """
        import numpy as np
        
        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        object_norm = object_embedding / (np.linalg.norm(object_embedding) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(query_norm, object_norm)
        
        return float(similarity)
    
    def _find_semantic_matches(self, question: str) -> List[Tuple[str, float]]:
        """Find semantically similar objects for a question.
        
        Args:
            question: Vietnamese question text
            
        Returns:
            List of (object_type, similarity_score) tuples sorted by score
        """
        if self._model is None or self._object_embeddings is None:
            return []
        
        try:
            # Get or compute question embedding
            if question in self._embeddings_cache:
                query_embedding = self._embeddings_cache[question]
            else:
                query_embedding = self._model.encode(question, convert_to_numpy=True)
                self._embeddings_cache[question] = query_embedding
            
            # Compute similarity with all object descriptions
            similarities = []
            for obj_type, obj_embedding in self._object_embeddings.items():
                similarity = self._compute_similarity(query_embedding, obj_embedding)
                if similarity >= SEMANTIC_SIMILARITY_THRESHOLD:
                    similarities.append((obj_type, similarity))
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities
            
        except Exception as e:
            logger.warning(f"Semantic matching failed: {e}")
            return []
    
    def extract(self, question: str) -> QueryAnalysisResult:
        """Extract using semantic similarity combined with keyword matching.
        
        The extraction process:
        1. Run keyword extraction for fast baseline results
        2. Compute semantic similarity with object descriptions
        3. Merge semantic matches with keyword matches
        4. Boost confidence based on semantic agreement
        """
        # Fall back to keyword extraction if model not available
        if self._model is None:
            return self._keyword_strategy.extract(question)
        
        # Get keyword-based results as baseline
        keyword_result = self._keyword_strategy.extract(question)
        
        # Find semantic matches
        semantic_matches = self._find_semantic_matches(question)
        
        # Merge target objects from both methods
        target_objects: Set[str] = set(keyword_result.target_objects)
        yolo_classes: Set[str] = set(keyword_result.yolo_classes)
        semantic_objects_found: List[str] = []
        max_semantic_score = 0.0
        
        for obj_type, score in semantic_matches:
            target_objects.add(obj_type)
            semantic_objects_found.append(f"{obj_type}:{score:.2f}")
            max_semantic_score = max(max_semantic_score, score)
            
            # Add corresponding YOLO classes
            if obj_type in SEMANTIC_OBJECT_DESCRIPTIONS:
                _, yolo = SEMANTIC_OBJECT_DESCRIPTIONS[obj_type]
                yolo_classes.update(yolo)
        
        # Calculate confidence boost based on semantic matching
        # - If semantic matches agree with keywords, boost confidence
        # - If semantic found new objects, moderate boost
        keyword_objects = set(keyword_result.target_objects)
        semantic_objects = {obj for obj, _ in semantic_matches}
        
        overlap = len(keyword_objects & semantic_objects)
        if overlap > 0:
            # Strong agreement between keyword and semantic
            confidence_boost = 0.2 + (max_semantic_score * 0.2)
        elif len(semantic_matches) > 0:
            # New semantic matches found
            confidence_boost = max_semantic_score * 0.15
        else:
            # Slight boost just for using semantic
            confidence_boost = 0.05
        
        final_confidence = min(1.0, keyword_result.confidence + confidence_boost)
        
        # Combine keywords found with semantic matches info
        keywords_found = keyword_result.keywords_found.copy()
        if semantic_objects_found:
            keywords_found.append(f"[semantic: {', '.join(semantic_objects_found[:3])}]")
        
        return QueryAnalysisResult(
            original_question=question,
            translated_question=None,
            target_objects=list(target_objects),
            question_intent=keyword_result.question_intent,
            yolo_classes=list(yolo_classes),
            keywords_found=keywords_found,
            confidence=final_confidence,
            temporal_hints=keyword_result.temporal_hints
        )