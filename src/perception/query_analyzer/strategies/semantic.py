"""
Semantic Extraction Strategy.

Uses sentence transformers or PhoBERT for semantic matching.
"""
import logging
from typing import List, Tuple, Optional

from ..base import ExtractionStrategy
from ..models import QueryAnalysisResult
from ..constants import (
    QuestionIntent,
    SEMANTIC_OBJECT_DESCRIPTIONS,
    SEMANTIC_SIMILARITY_THRESHOLD,
    NATIVE_SBERT_MODELS,
)
from .keyword import KeywordExtractionStrategy

logger = logging.getLogger(__name__)


class SemanticExtractionStrategy(ExtractionStrategy):
    """
    Semantic extraction using sentence embeddings.
    
    Uses Vietnamese sentence transformers or PhoBERT to semantically
    match questions with predefined object descriptions.
    """
    
    def __init__(self, model_name: str = "keepitreal/vietnamese-sbert"):
        """
        Initialize semantic strategy.
        
        Args:
            model_name: HuggingFace model for Vietnamese embeddings
        """
        self._model_name = model_name
        self._model = None
        self._keyword_strategy = KeywordExtractionStrategy()
        self._object_embeddings = None
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading semantic model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            
            # Pre-compute object description embeddings
            self._precompute_embeddings()
            
        except ImportError:
            logger.warning("sentence_transformers not installed. Semantic matching disabled.")
            self._model = None
        except Exception as e:
            logger.warning(f"Failed to load semantic model: {e}")
            self._model = None
    
    def _precompute_embeddings(self):
        """Pre-compute embeddings for object descriptions."""
        if self._model is None:
            return
        
        descriptions = []
        self._object_keys = []
        
        for obj_type, (description, _) in SEMANTIC_OBJECT_DESCRIPTIONS.items():
            descriptions.append(description)
            self._object_keys.append(obj_type)
        
        self._object_embeddings = self._model.encode(descriptions, convert_to_tensor=True)
        logger.info(f"Pre-computed embeddings for {len(descriptions)} object types")
    
    @property
    def name(self) -> str:
        return "semantic"
    
    def _semantic_match(self, question: str) -> List[Tuple[str, float]]:
        """Find semantically similar object types."""
        self._load_model()
        
        if self._model is None or self._object_embeddings is None:
            return []
        
        try:
            from sentence_transformers import util
            
            # Encode the question
            question_embedding = self._model.encode(question, convert_to_tensor=True)
            
            # Compute similarities
            similarities = util.cos_sim(question_embedding, self._object_embeddings)[0]
            
            # Find matches above threshold
            matches = []
            for i, score in enumerate(similarities):
                if score >= SEMANTIC_SIMILARITY_THRESHOLD:
                    matches.append((self._object_keys[i], float(score)))
            
            # Sort by score
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches
            
        except Exception as e:
            logger.warning(f"Semantic matching failed: {e}")
            return []
    
    def extract(self, question: str) -> QueryAnalysisResult:
        """Extract information using semantic matching."""
        # Get keyword-based results as baseline
        keyword_result = self._keyword_strategy.extract(question)
        
        # Add semantic matches
        semantic_matches = self._semantic_match(question)
        
        # Combine results
        target_objects = list(keyword_result.target_objects)
        yolo_classes = set(keyword_result.yolo_classes)
        
        for obj_type, score in semantic_matches:
            if obj_type not in target_objects:
                target_objects.append(obj_type)
            # Get YOLO classes for this object type
            if obj_type in SEMANTIC_OBJECT_DESCRIPTIONS:
                _, classes = SEMANTIC_OBJECT_DESCRIPTIONS[obj_type]
                yolo_classes.update(classes)
        
        # Adjust confidence based on semantic matches
        confidence = keyword_result.confidence
        if semantic_matches:
            # Boost confidence based on best semantic match
            best_score = semantic_matches[0][1] if semantic_matches else 0
            confidence = min(1.0, confidence + best_score * 0.3)
        
        return QueryAnalysisResult(
            original_question=question,
            translated_question=None,
            target_objects=target_objects,
            question_intent=keyword_result.question_intent,
            yolo_classes=list(yolo_classes),
            keywords_found=keyword_result.keywords_found,
            confidence=confidence,
            temporal_hints=keyword_result.temporal_hints,
        )
