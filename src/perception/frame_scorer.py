"""
Frame Scorer for Query-Aware Frame Selection.

Scores video frames by relevance to a given question using multiple strategies:
1. Question-Frame Similarity (QFS) - CLIP-based semantic similarity
2. Detection-Guided Scoring (DGS) - Boost frames with detected target objects
3. Inter-Frame Distinctiveness (IFD) - Reduce redundancy

Based on SOTA approaches:
- VidF4 (ECCV 2024): QFS + QFM + IFD scoring mechanisms
- Q-Frame (CVPR 2025): CLIP-based training-free frame selection

Supports multiple scoring strategies (configurable as plugins):
- CLIPScorer: Use CLIP for Question-Frame Similarity
- MCLIPScorer: Use Multilingual CLIP
- DetectionScorer: Use YOLO detections for boosting
- CombinedScorer: Weighted combination of strategies

Usage:
    from src.perception.frame_scorer import FrameScorer, ScoringConfig
    
    scorer = FrameScorer(strategy="clip")
    scores = scorer.score_frames(frames, question, target_classes)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ScoringConfig:
    """
    Configuration for frame scoring.
    
    Attributes:
        strategy: Scoring strategy ("clip", "mclip", "detection", "combined")
        clip_model: CLIP model name (e.g., "ViT-L/14", "ViT-B/32")
        device: Compute device ("cuda", "cpu", "auto")
        batch_size: Batch size for frame encoding
        alpha: Weight for Question-Frame Similarity (QFS)
        beta: Weight for Detection-Guided Scoring (DGS)
        gamma: Weight for Inter-Frame Distinctiveness (IFD)
        use_translation: Whether to translate Vietnamese to English for CLIP
        cache_embeddings: Whether to cache frame embeddings
    """
    strategy: str = "clip"
    clip_model: str = "ViT-L/14"
    device: str = "auto"
    batch_size: int = 16
    alpha: float = 0.5  # QFS weight
    beta: float = 0.3   # Detection boost weight
    gamma: float = 0.2  # Distinctiveness weight
    use_translation: bool = True  # Default: translate Vietnamese for CLIP
    cache_embeddings: bool = True

@dataclass
class FrameScore:
    """
    Score result for a single frame.
    
    Attributes:
        frame_idx: Frame index in the video
        qfs_score: Question-Frame Similarity score
        detection_score: Detection-guided boost score
        ifd_score: Inter-Frame Distinctiveness score
        final_score: Combined weighted score
        detections_found: List of detected object classes
    """
    frame_idx: int
    qfs_score: float = 0.0
    detection_score: float = 0.0
    ifd_score: float = 0.0
    final_score: float = 0.0
    detections_found: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "frame_idx": self.frame_idx,
            "qfs_score": self.qfs_score,
            "detection_score": self.detection_score,
            "ifd_score": self.ifd_score,
            "final_score": self.final_score,
            "detections_found": self.detections_found
        }

# Scoring Strategies
class ScoringStrategy(ABC):
    """Abstract base class for frame scoring strategies."""
    
    @abstractmethod
    def score(
        self,
        frames: List[np.ndarray],
        question: str,
        target_classes: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Score frames by relevance to question.
        
        Args:
            frames: List of video frames (numpy arrays, HWC format)
            question: Question text (Vietnamese or English)
            target_classes: Optional list of target object classes
            
        Returns:
            Array of scores, one per frame
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass

class CLIPScoringStrategy(ScoringStrategy):
    """
    CLIP-based Question-Frame Similarity scoring.
    
    Uses OpenAI CLIP or OpenCLIP to compute semantic similarity between question text and video frames.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-L/14",
        device: str = "auto",
        use_translation: bool = True
    ):
        self._model_name = model_name
        self._device = self._resolve_device(device)
        self._use_translation = use_translation
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._translator = None
        self._init_model()
        
        if use_translation:
            self._init_translator()
    
    @property
    def name(self) -> str:
        return "clip"
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _init_model(self):
        """Initialize CLIP model."""
        try:
            import torch
            import clip
            
            logger.info(f"Loading CLIP model: {self._model_name}")
            self._model, self._preprocess = clip.load(
                self._model_name, 
                device=self._device
            )
            self._model.eval()
            logger.info(f"CLIP model loaded on {self._device}")
            
        except ImportError:
            logger.warning(
                "OpenAI CLIP not available. Trying OpenCLIP..."
            )
            try:
                import open_clip
                self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                    self._model_name.replace("/", "-"),
                    pretrained='openai'
                )
                self._tokenizer = open_clip.get_tokenizer(
                    self._model_name.replace("/", "-")
                )
                import torch
                self._model = self._model.to(self._device)
                self._model.eval()
                logger.info(f"OpenCLIP model loaded on {self._device}")
            except ImportError:
                logger.error(
                    "Neither CLIP nor OpenCLIP available. "
                    "Install with: pip install git+https://github.com/openai/CLIP.git "
                    "or pip install open-clip-torch"
                )
    
    def _init_translator(self):
        """Initialize translation for Vietnamese using deep_translator."""
        try:
            from deep_translator import GoogleTranslator
            self._translator = GoogleTranslator(source='vi', target='en')
            logger.info("Translation enabled for Vietnamese -> English (deep_translator)")
        except ImportError:
            logger.warning(
                "deep_translator not available for translation. "
                "Install with: pip install deep_translator"
            )
            self._translator = None
    
    def _translate(self, text: str) -> str:
        """Translate Vietnamese to English using deep_translator."""
        if self._translator is None:
            return text
        
        try:
            # Simple heuristic: if text contains Vietnamese characters, translate
            if any('\u00c0' <= c <= '\u1ef9' for c in text):
                translated = self._translator.translate(text)
                logger.debug(f"Translated: '{text}' -> '{translated}'")
                return translated
            return text
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text
    
    def score(
        self,
        frames: List[np.ndarray],
        question: str,
        target_classes: Optional[List[str]] = None
    ) -> np.ndarray:
        """Score frames using CLIP Question-Frame Similarity."""
        if self._model is None:
            logger.warning("CLIP model not loaded, returning uniform scores")
            return np.ones(len(frames))
        
        import torch
        from PIL import Image
        
        # Translate question if needed
        if self._use_translation:
            question = self._translate(question)
        
        # Encode question text
        if hasattr(self, '_tokenizer') and self._tokenizer is not None:
            # OpenCLIP
            text_tokens = self._tokenizer([question]).to(self._device)
            with torch.no_grad():
                text_features = self._model.encode_text(text_tokens)
        else:
            # OpenAI CLIP
            import clip
            text_tokens = clip.tokenize([question], truncate=True).to(self._device)
            with torch.no_grad():
                text_features = self._model.encode_text(text_tokens)
        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Encode frames in batches
        all_scores = []
        batch_size = 16
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            
            # Preprocess frames
            processed_frames = []
            for frame in batch_frames:
                # Convert BGR to RGB if needed
                if frame.shape[-1] == 3:
                    frame_rgb = frame[..., ::-1] if frame.dtype == np.uint8 else frame
                else:
                    frame_rgb = frame
                
                pil_image = Image.fromarray(frame_rgb.astype(np.uint8))
                processed = self._preprocess(pil_image)
                processed_frames.append(processed)
            
            image_input = torch.stack(processed_frames).to(self._device)
            
            with torch.no_grad():
                image_features = self._model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
                similarities = (image_features @ text_features.T).squeeze(-1)
                all_scores.extend(similarities.cpu().numpy().tolist())
        
        scores = np.array(all_scores)
        
        # Normalize scores to [0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return scores

class MultilingualCLIPScoringStrategy(ScoringStrategy):
    """
    Multilingual CLIP (M-CLIP) scoring for direct Vietnamese support.
    
    Uses multilingual CLIP variants that support Vietnamese text directly
    without translation. Combines M-CLIP text encoder with OpenCLIP image encoder.
    """
    
    def __init__(
        self,
        model_name: str = "M-CLIP/XLM-Roberta-Large-Vit-L-14",
        device: str = "auto"
    ):
        self._model_name = model_name
        self._device = self._resolve_device(device)
        self._text_model = None
        self._tokenizer = None
        self._image_model = None
        self._image_preprocess = None
        self._init_model()
    
    @property
    def name(self) -> str:
        return "mclip"
    
    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _init_model(self):
        """Initialize Multilingual CLIP text encoder and OpenCLIP image encoder."""
        try:
            import torch
            from multilingual_clip import pt_multilingual_clip
            import transformers
            
            logger.info(f"Loading M-CLIP text model: {self._model_name}")
            
            # Load M-CLIP text encoder
            self._text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(
                self._model_name
            )
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self._model_name
            )
            self._text_model = self._text_model.to(self._device)
            self._text_model.eval()
            
            # Load matching OpenCLIP image encoder
            # M-CLIP/XLM-Roberta-Large-Vit-L-14 uses ViT-L/14 image encoder
            try:
                import open_clip
                
                # Determine which image model to use based on M-CLIP model name
                if "Vit-L-14" in self._model_name:
                    clip_model_name = "ViT-L-14"
                elif "Vit-B-32" in self._model_name:
                    clip_model_name = "ViT-B-32"
                else:
                    clip_model_name = "ViT-L-14"  # Default
                
                logger.info(f"Loading OpenCLIP image model: {clip_model_name}")
                self._image_model, _, self._image_preprocess = open_clip.create_model_and_transforms(
                    clip_model_name,
                    pretrained='openai'
                )
                self._image_model = self._image_model.to(self._device)
                self._image_model.eval()
                
            except ImportError:
                # Fallback to OpenAI CLIP
                logger.info("OpenCLIP not available, trying OpenAI CLIP for image encoder")
                import clip
                
                if "Vit-L-14" in self._model_name:
                    clip_model_name = "ViT-L/14"
                elif "Vit-B-32" in self._model_name:
                    clip_model_name = "ViT-B/32"
                else:
                    clip_model_name = "ViT-L/14"
                
                self._image_model, self._image_preprocess = clip.load(
                    clip_model_name,
                    device=self._device
                )
                self._image_model.eval()
            
            logger.info(f"M-CLIP model initialized on {self._device}")
            
        except ImportError as e:
            logger.warning(
                f"Failed to load M-CLIP: {e}. "
                "Install with: pip install multilingual-clip open-clip-torch"
            )
        except Exception as e:
            logger.error(f"Error initializing M-CLIP: {e}")
    
    def score(
        self,
        frames: List[np.ndarray],
        question: str,
        target_classes: Optional[List[str]] = None
    ) -> np.ndarray:
        """Score frames using Multilingual CLIP."""
        if self._text_model is None or self._image_model is None:
            logger.warning("M-CLIP model not fully loaded, returning uniform scores")
            return np.ones(len(frames))
        
        import torch
        from PIL import Image
        
        # Encode question text using M-CLIP text encoder
        # Note: M-CLIP forward() expects (text, tokenizer) not tokenized tensors
        with torch.no_grad():
            text_features = self._text_model.forward(
                question,
                self._tokenizer
            )
            text_features = text_features.to(self._device)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Encode frames using CLIP image encoder
        all_scores = []
        batch_size = 16
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            
            # Preprocess frames
            processed_frames = []
            for frame in batch_frames:
                # Ensure RGB format
                if frame.shape[-1] == 3:
                    frame_rgb = frame
                else:
                    frame_rgb = frame
                
                pil_image = Image.fromarray(frame_rgb.astype(np.uint8))
                processed = self._image_preprocess(pil_image)
                processed_frames.append(processed)
            
            image_input = torch.stack(processed_frames).to(self._device)
            
            with torch.no_grad():
                image_features = self._image_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
                similarities = (image_features @ text_features.T).squeeze(-1)
                all_scores.extend(similarities.cpu().numpy().tolist())
        
        scores = np.array(all_scores)
        
        # Normalize scores to [0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return scores

class DetectionScoringStrategy(ScoringStrategy):
    """
    Detection-guided scoring using YOLO models.
    
    Boosts frames that contain detected objects matching the target classes
    extracted from the question.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.25,
        device: str = "auto"
    ):
        self._model_path = model_path
        self._confidence = confidence
        self._device = device
        self._model = None
        
        if model_path:
            self._init_model()
    
    @property
    def name(self) -> str:
        return "detection"
    
    def _init_model(self):
        """Initialize YOLO model."""
        try:
            from ultralytics import YOLO
            
            logger.info(f"Loading YOLO model: {self._model_path}")
            self._model = YOLO(self._model_path)
            logger.info("YOLO model loaded")
            
        except ImportError:
            logger.error(
                "Ultralytics not available. "
                "Install with: pip install ultralytics"
            )
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
    
    def set_model(self, model):
        """Set pre-loaded YOLO model."""
        self._model = model
    
    def score(
        self,
        frames: List[np.ndarray],
        question: str,
        target_classes: Optional[List[str]] = None
    ) -> np.ndarray:
        """Score frames based on detected objects."""
        if self._model is None:
            logger.warning("YOLO model not loaded, returning uniform scores")
            return np.ones(len(frames))
        
        if target_classes is None or len(target_classes) == 0:
            # No target classes, score by total detection confidence
            target_classes = []
        
        scores = []
        
        for frame in frames:
            # Run detection
            results = self._model.predict(
                frame, 
                conf=self._confidence,
                verbose=False
            )
            
            if len(results) == 0 or results[0].boxes is None:
                scores.append(0.0)
                continue
            
            boxes = results[0].boxes
            class_names = results[0].names
            
            if len(boxes) == 0:
                scores.append(0.0)
                continue
            
            # Calculate score based on detections
            total_score = 0.0
            
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = class_names.get(cls_id, "").lower()
                
                # Boost if matches target class
                if target_classes:
                    for target in target_classes:
                        if target.lower() in cls_name or cls_name in target.lower():
                            total_score += conf * 2.0  # Double boost for target match
                            break
                    else:
                        total_score += conf * 0.5  # Smaller boost for other detections
                else:
                    # No specific targets, use confidence directly
                    total_score += conf
            
            scores.append(total_score)
        
        scores = np.array(scores)
        
        # Normalize
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores

class DistinctivenessStrategy:
    """
    Inter-Frame Distinctiveness (IFD) scoring.
    
    Reduces redundancy by scoring frames based on their distinctiveness
    from neighboring frames.
    """
    
    def __init__(self, window_size: int = 3):
        self._window_size = window_size
    
    def compute(self, frame_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute distinctiveness scores for frames.
        
        Args:
            frame_embeddings: Frame embeddings (N x D)
            
        Returns:
            Distinctiveness scores (N,)
        """
        n_frames = len(frame_embeddings)
        if n_frames <= 1:
            return np.ones(n_frames)
        
        # Normalize embeddings
        norms = np.linalg.norm(frame_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings_norm = frame_embeddings / norms
        
        scores = []
        
        for i in range(n_frames):
            # Get window around current frame
            start = max(0, i - self._window_size)
            end = min(n_frames, i + self._window_size + 1)
            
            # Exclude current frame
            neighbors_idx = [j for j in range(start, end) if j != i]
            
            if len(neighbors_idx) == 0:
                scores.append(1.0)
                continue
            
            # Compute similarity to neighbors
            current = embeddings_norm[i]
            neighbors = embeddings_norm[neighbors_idx]
            
            similarities = np.dot(neighbors, current)
            avg_similarity = np.mean(similarities)
            
            # Distinctiveness = 1 - similarity
            distinctiveness = 1.0 - avg_similarity
            scores.append(distinctiveness)
        
        return np.array(scores)
    
    def compute_from_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Compute distinctiveness directly from frame pixels.
        
        Uses histogram comparison for speed.
        """
        import cv2
        
        n_frames = len(frames)
        if n_frames <= 1:
            return np.ones(n_frames)
        
        # Compute histograms
        histograms = []
        for frame in frames:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)
        
        histograms = np.array(histograms)
        
        return self.compute(histograms)

# Combined Frame Scorer
class FrameScorer:
    """
    Main frame scoring interface with multiple strategies.
    
    Combines Question-Frame Similarity (QFS), Detection-Guided Scoring (DGS),
    and Inter-Frame Distinctiveness (IFD) with configurable weights.
    
    Usage:
        config = ScoringConfig(strategy="combined", alpha=0.5, beta=0.3, gamma=0.2)
        scorer = FrameScorer(config)
        scores = scorer.score_frames(frames, question, target_classes)
    """
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        """
        Initialize FrameScorer.
        
        Args:
            config: Scoring configuration
        """
        self.config = config or ScoringConfig()
        
        # Initialize strategies based on config
        self._qfs_scorer: Optional[ScoringStrategy] = None
        self._detection_scorer: Optional[DetectionScoringStrategy] = None
        self._ifd_scorer = DistinctivenessStrategy()
        
        self._init_strategies()
        
        # Embedding cache
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    def _init_strategies(self):
        """Initialize scoring strategies based on config."""
        # Initialize QFS strategy
        if self.config.strategy in ["clip", "combined"]:
            self._qfs_scorer = CLIPScoringStrategy(
                model_name=self.config.clip_model,
                device=self.config.device,
                use_translation=self.config.use_translation
            )
        elif self.config.strategy == "mclip":
            self._qfs_scorer = MultilingualCLIPScoringStrategy(
                device=self.config.device
            )
        
        # Detection scorer (optional, can be set later)
        if self.config.strategy in ["detection", "combined"]:
            self._detection_scorer = DetectionScoringStrategy(
                device=self.config.device
            )
    
    def set_yolo_model(self, model):
        """Set YOLO model for detection-guided scoring."""
        if self._detection_scorer is None:
            self._detection_scorer = DetectionScoringStrategy(
                device=self.config.device
            )
        self._detection_scorer.set_model(model)
    
    def score_frames(
        self,
        frames: List[np.ndarray],
        question: str,
        target_classes: Optional[List[str]] = None,
        return_detailed: bool = False
    ) -> Union[np.ndarray, List[FrameScore]]:
        """
        Score all frames by relevance to the question.
        
        Args:
            frames: List of video frames (numpy arrays)
            question: Question text
            target_classes: Optional target object classes from query analysis
            return_detailed: If True, return detailed FrameScore objects
            
        Returns:
            Array of scores or list of FrameScore objects
        """
        n_frames = len(frames)
        
        # Initialize scores
        qfs_scores = np.zeros(n_frames)
        detection_scores = np.zeros(n_frames)
        ifd_scores = np.zeros(n_frames)
        
        # Compute Question-Frame Similarity
        if self._qfs_scorer is not None and self.config.alpha > 0:
            logger.info("Computing Question-Frame Similarity scores...")
            qfs_scores = self._qfs_scorer.score(frames, question, target_classes)
        
        # Compute Detection-Guided Scores
        if self._detection_scorer is not None and self.config.beta > 0:
            logger.info("Computing Detection-Guided scores...")
            detection_scores = self._detection_scorer.score(
                frames, question, target_classes
            )
        
        # Compute Inter-Frame Distinctiveness
        if self.config.gamma > 0:
            logger.info("Computing Inter-Frame Distinctiveness scores...")
            ifd_scores = self._ifd_scorer.compute_from_frames(frames)
        
        # Combine scores with weights
        final_scores = (
            self.config.alpha * qfs_scores +
            self.config.beta * detection_scores +
            self.config.gamma * ifd_scores
        )
        
        if return_detailed:
            return [
                FrameScore(
                    frame_idx=i,
                    qfs_score=float(qfs_scores[i]),
                    detection_score=float(detection_scores[i]),
                    ifd_score=float(ifd_scores[i]),
                    final_score=float(final_scores[i])
                )
                for i in range(n_frames)
            ]
        
        return final_scores
    
    def score_single_frame(
        self,
        frame: np.ndarray,
        question: str,
        target_classes: Optional[List[str]] = None
    ) -> FrameScore:
        """Score a single frame."""
        scores = self.score_frames(
            [frame], question, target_classes, return_detailed=True
        )
        return scores[0]

def get_available_strategies() -> List[str]:
    """Get list of available scoring strategies."""
    return ["clip", "mclip", "detection", "combined"]


def create_scorer(
    strategy: str = "clip",
    clip_model: str = "ViT-L/14",
    device: str = "auto",
    use_translation: bool = True,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2
) -> FrameScorer:
    """
    Factory function to create a FrameScorer.
    
    Args:
        strategy: Scoring strategy
        clip_model: CLIP model name
        device: Compute device
        use_translation: Whether to translate Vietnamese for CLIP
        alpha: QFS weight
        beta: Detection weight
        gamma: Distinctiveness weight
        
    Returns:
        Configured FrameScorer
    """
    config = ScoringConfig(
        strategy=strategy,
        clip_model=clip_model,
        device=device,
        use_translation=use_translation,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )
    return FrameScorer(config)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Frame Scorer Demo")
    print("=" * 60)
    
    # Create scorer with default config
    config = ScoringConfig(
        strategy="clip",
        use_translation=True,
        alpha=0.7,
        beta=0.0,  # No detection for demo
        gamma=0.3
    )
    
    scorer = FrameScorer(config)
    
    # Create dummy frames for testing
    dummy_frames = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(5)
    ]
    
    question = "Biển báo tốc độ tối đa là bao nhiêu?"
    
    print(f"\nQuestion: {question}")
    print(f"Number of frames: {len(dummy_frames)}")
    print(f"Strategy: {config.strategy}")
    print(f"Weights: alpha={config.alpha}, beta={config.beta}, gamma={config.gamma}")
    
    # Score frames
    scores = scorer.score_frames(dummy_frames, question, return_detailed=True)
    
    print("\nFrame Scores:")
    for score in scores:
        print(f"  Frame {score.frame_idx}: "
              f"QFS={score.qfs_score:.3f}, "
              f"IFD={score.ifd_score:.3f}, "
              f"Final={score.final_score:.3f}")