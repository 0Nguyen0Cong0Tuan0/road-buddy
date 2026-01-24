"""
Query-Aware Keyframe Selector.

Selects the most relevant keyframes from a video based on the question
using a combination of semantic similarity, object detection, and
frame distinctiveness.

This is the main orchestrator that combines:
1. QueryAnalyzer: Parses questions to identify target objects
2. FrameScorer: Scores frames by relevance using CLIP/YOLO
3. Selection Algorithm: Selects top-k diverse frames

Configurable options:
- Number of keyframes
- Frame sampling rate
- Scoring weights (alpha, beta, gamma)
- Selection strategy (top-k, diverse-top-k, temporal-weighted)
- YOLO detection mode (all frames vs selected only)

Usage:
    from src.perception.keyframe_selector import KeyframeSelector, KeyframeSelectorConfig
    
    config = KeyframeSelectorConfig(num_keyframes=8, yolo_mode="selected_only")
    selector = KeyframeSelector(config)
    
    keyframes = selector.select(video_path, question)
    for kf in keyframes:
        print(f"Frame {kf.frame_idx}: score={kf.score:.3f}")
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
from enum import Enum
import numpy as np
import logging
import time
import decord
import cv2

from .query_analyzer import QueryAnalyzer, QueryAnalysisResult
from .frame_scorer import FrameScorer, ScoringConfig, FrameScore
from .results import FrameDetections

logger = logging.getLogger(__name__)

class YOLOMode(Enum):
    """YOLO detection mode."""
    ALL_FRAMES = "all_frames"  # Run on all sampled frames (more accurate, slower)
    SELECTED_ONLY = "selected_only"  # Run only on selected keyframes (faster)
    NONE = "none"  # No YOLO detection (fastest)

class SelectionStrategy(Enum):
    """Keyframe selection strategy."""
    TOP_K = "top_k"  # Simply select top k scores
    DIVERSE_TOP_K = "diverse_top_k"  # Select diverse frames with high scores
    TEMPORAL_WEIGHTED = "temporal_weighted"  # Weight by temporal position

@dataclass
class KeyframeSelectorConfig:
    """
    Configuration for keyframe selection.
    
    Attributes:
        num_keyframes: Number of keyframes to select (default: 8)
        sample_fps: Frame sampling rate in FPS (default: 2.0)
        max_frames: Maximum frames to sample from video (default: 64)
        
        query_strategy: Query analysis strategy ("keyword", "translation", "semantic")
        scoring_strategy: Frame scoring strategy ("clip", "mclip", "detection", "combined")
        selection_strategy: Selection algorithm ("top_k", "diverse_top_k", "temporal_weighted")
        
        yolo_mode: When to run YOLO ("all_frames", "selected_only", "none")
        yolo_model_path: Path to YOLO model weights
        yolo_confidence: YOLO confidence threshold
        
        use_translation: Translate Vietnamese for CLIP (default: True)
        clip_model: CLIP model name (default: "ViT-L/14")
        
        alpha: Question-Frame Similarity weight (default: 0.5)
        beta: Detection boost weight (default: 0.3)
        gamma: Distinctiveness weight (default: 0.2)
        
        diversity_threshold: Minimum distance between selected frames (default: 0.5)
        temporal_decay: Decay factor for temporal weighting (default: 0.1)
    """
    # Frame selection
    num_keyframes: int = 8
    sample_fps: float = 2.0
    max_frames: int = 64
    
    # Strategy selection
    query_strategy: str = "keyword"  # Default per user request
    scoring_strategy: str = "clip"
    selection_strategy: str = "diverse_top_k"
    
    # YOLO settings
    yolo_mode: str = "selected_only"  # Default per user request
    yolo_model_path: Optional[str] = None
    yolo_confidence: float = 0.25
    
    # CLIP settings
    use_translation: bool = True  # Default per user request
    clip_model: str = "ViT-L/14"
    
    # Scoring weights
    alpha: float = 0.5  # QFS weight
    beta: float = 0.3   # Detection boost weight
    gamma: float = 0.2  # Distinctiveness weight
    
    # Selection parameters
    diversity_threshold: float = 0.5
    temporal_decay: float = 0.1
    
    # Device
    device: str = "auto"

@dataclass
class KeyframeResult:
    """
    Result for a selected keyframe.
    
    Attributes:
        frame_idx: Original frame index in the video
        timestamp: Timestamp in seconds
        frame: The frame image (numpy array)
        score: Selection score
        score_details: Detailed scoring breakdown
        detections: YOLO detections (if available)
        query_info: Query analysis results
    """
    frame_idx: int
    timestamp: float
    frame: np.ndarray
    score: float
    score_details: Optional[FrameScore] = None
    detections: Optional[FrameDetections] = None
    query_info: Optional[QueryAnalysisResult] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (without frame data)."""
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "score": self.score,
            "score_details": self.score_details.to_dict() if self.score_details else None,
            "detections": self.detections.to_dict() if self.detections else None,
        }

@dataclass
class KeyframeSelectionResult:
    """
    Complete result of keyframe selection.
    
    Attributes:
        keyframes: List of selected KeyframeResult objects
        query_analysis: Query analysis result
        total_frames_sampled: Number of frames that were sampled and scored
        processing_time: Time taken for selection in seconds
        config: Configuration used
    """
    keyframes: List[KeyframeResult]
    query_analysis: QueryAnalysisResult
    total_frames_sampled: int
    processing_time: float
    config: KeyframeSelectorConfig
    
    @property
    def num_keyframes(self) -> int:
        return len(self.keyframes)
    
    @property
    def frames(self) -> List[np.ndarray]:
        """Get just the frame images."""
        return [kf.frame for kf in self.keyframes]
    
    @property
    def timestamps(self) -> List[float]:
        """Get timestamps of selected frames."""
        return [kf.timestamp for kf in self.keyframes]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (without frame data)."""
        return {
            "num_keyframes": self.num_keyframes,
            "total_frames_sampled": self.total_frames_sampled,
            "processing_time": self.processing_time,
            "query_analysis": self.query_analysis.to_dict(),
            "keyframes": [kf.to_dict() for kf in self.keyframes],
        }

class KeyframeSelector:
    """
    Main keyframe selector that orchestrates query-aware frame selection.
    
    Workflow:
    1. Analyze question to extract target objects
    2. Sample frames from video at specified rate
    3. Score frames using CLIP/detection scoring
    4. Select top-k diverse keyframes
    5. Optionally run YOLO on selected frames
    
    Usage:
        selector = KeyframeSelector()
        result = selector.select("video.mp4", "Biển báo tốc độ là bao nhiêu?")
        
        for kf in result.keyframes:
            print(f"Frame {kf.frame_idx} @ {kf.timestamp:.2f}s: {kf.score:.3f}")
    """
    
    def __init__(self, config: Optional[KeyframeSelectorConfig] = None):
        """
        Initialize KeyframeSelector.
        
        Args:
            config: Configuration options
        """
        self.config = config or KeyframeSelectorConfig()
        
        # Initialize components
        self._query_analyzer = QueryAnalyzer(
            strategy=self.config.query_strategy,
            translator="deep_translator"
        )
        
        self._frame_scorer = FrameScorer(
            ScoringConfig(
                strategy=self.config.scoring_strategy,
                clip_model=self.config.clip_model,
                device=self.config.device,
                use_translation=self.config.use_translation,
                alpha=self.config.alpha,
                beta=self.config.beta,
                gamma=self.config.gamma
            )
        )
        
        # YOLO model (lazy loaded)
        self._yolo_model = None
        
        logger.info(
            f"KeyframeSelector initialized: "
            f"query={self.config.query_strategy}, "
            f"scoring={self.config.scoring_strategy}, "
            f"selection={self.config.selection_strategy}, "
            f"yolo_mode={self.config.yolo_mode}"
        )
    
    def _load_yolo_model(self):
        """Lazy load YOLO model."""
        if self._yolo_model is not None:
            return
        
        if self.config.yolo_model_path is None:
            logger.warning("No YOLO model path specified")
            return
        
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model: {self.config.yolo_model_path}")
            self._yolo_model = YOLO(self.config.yolo_model_path)
            
            # Also set in frame scorer if using detection scoring
            self._frame_scorer.set_yolo_model(self._yolo_model)
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
    
    def set_yolo_model(self, model):
        """Set pre-loaded YOLO model."""
        self._yolo_model = model
        self._frame_scorer.set_yolo_model(model)
    
    def _sample_frames(
        self, 
        video_path: str
    ) -> Tuple[List[np.ndarray], List[float], Dict]:
        """
        Sample frames from video at specified rate.
        
        Returns:
            Tuple of (frames, timestamps, metadata)
        """
        try:
            decord.bridge.set_bridge('native')
            
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)
            fps = vr.get_avg_fps()
            duration = total_frames / fps
            
            logger.info(
                f"Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s"
            )
            
            # Calculate frame indices to sample
            sample_interval = int(fps / self.config.sample_fps)
            sample_interval = max(1, sample_interval)
            
            frame_indices = list(range(0, total_frames, sample_interval))
            
            # Limit to max_frames
            if len(frame_indices) > self.config.max_frames:
                # Uniformly subsample
                step = len(frame_indices) / self.config.max_frames
                frame_indices = [
                    frame_indices[int(i * step)] 
                    for i in range(self.config.max_frames)
                ]
            
            # Sample frames
            frames_array = vr.get_batch(frame_indices).asnumpy()
            frames = [frames_array[i] for i in range(len(frames_array))]
            
            # Calculate timestamps
            timestamps = [idx / fps for idx in frame_indices]
            
            metadata = {
                "total_frames": total_frames,
                "fps": fps,
                "duration": duration,
                "sampled_frames": len(frames)
            }
            
            return frames, timestamps, metadata
            
        except ImportError:
            logger.warning("Decord not available, using OpenCV")
            return self._sample_frames_opencv(video_path)
    
    def _sample_frames_opencv(
        self, 
        video_path: str
    ) -> Tuple[List[np.ndarray], List[float], Dict]:
        """Fallback frame sampling using OpenCV."""
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        sample_interval = int(fps / self.config.sample_fps)
        sample_interval = max(1, sample_interval)
        
        frames = []
        timestamps = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                timestamps.append(frame_idx / fps)
                
                if len(frames) >= self.config.max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        
        metadata = {
            "total_frames": total_frames,
            "fps": fps,
            "duration": duration,
            "sampled_frames": len(frames)
        }
        
        return frames, timestamps, metadata
    
    def _select_top_k(
        self,
        scores: np.ndarray,
        k: int
    ) -> List[int]:
        """Simple top-k selection."""
        indices = np.argsort(scores)[::-1][:k]
        return sorted(indices.tolist())  # Sort by frame order
    
    def _select_diverse_top_k(
        self,
        scores: np.ndarray,
        frames: List[np.ndarray],
        k: int
    ) -> List[int]:
        """
        Select top-k frames while maintaining diversity.
        
        Uses a greedy approach: iteratively select the highest scoring
        frame that is sufficiently different from already selected frames.
        """
        n_frames = len(scores)
        if n_frames <= k:
            return list(range(n_frames))
        
        # Compute frame similarity matrix using histograms        
        histograms = []
        for frame in frames:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)
        
        histograms = np.array(histograms)
        
        # Greedy selection
        selected = []
        remaining = list(range(n_frames))
        
        while len(selected) < k and remaining:
            # Find best next frame
            best_idx = None
            best_score = -float('inf')
            
            for idx in remaining:
                # Check diversity with selected frames
                if selected:
                    # Compute minimum distance to selected frames
                    min_dist = float('inf')
                    for sel_idx in selected:
                        dist = np.linalg.norm(
                            histograms[idx] - histograms[sel_idx]
                        )
                        min_dist = min(min_dist, dist)
                    
                    # Penalize if too similar
                    if min_dist < self.config.diversity_threshold:
                        continue
                
                # Check score
                if scores[idx] > best_score:
                    best_score = scores[idx]
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                # If no diverse frame found, just take the highest score
                remaining.sort(key=lambda x: scores[x], reverse=True)
                if remaining:
                    selected.append(remaining.pop(0))
        
        return sorted(selected)
    
    def _select_temporal_weighted(
        self,
        scores: np.ndarray,
        timestamps: List[float],
        k: int
    ) -> List[int]:
        """
        Select frames with temporal weighting.
        
        Gives slight preference to frames in the middle of the video
        where the key information often appears.
        """
        n_frames = len(scores)
        duration = timestamps[-1] if timestamps else 1.0
        
        # Create temporal weights (Gaussian centered at middle)
        mid_time = duration / 2
        temporal_weights = np.array([
            np.exp(-self.config.temporal_decay * ((t - mid_time) / duration) ** 2)
            for t in timestamps
        ])
        
        # Combine with scores
        weighted_scores = scores * temporal_weights
        
        return self._select_top_k(weighted_scores, k)
    
    def _run_yolo_detection(
        self,
        frames: List[np.ndarray],
        frame_indices: List[int]
    ) -> Dict[int, FrameDetections]:
        """Run YOLO detection on frames."""
        if self._yolo_model is None:
            self._load_yolo_model()
        
        if self._yolo_model is None:
            return {}
        
        from .results import parse_yolo_results
        
        detections = {}
        
        for i, frame in enumerate(frames):
            results = self._yolo_model.predict(
                frame,
                conf=self.config.yolo_confidence,
                verbose=False
            )
            
            parsed = parse_yolo_results(results, start_frame_idx=frame_indices[i])
            if parsed:
                detections[frame_indices[i]] = parsed[0]
        
        return detections
    
    def select(
        self,
        video_path: str,
        question: str
    ) -> KeyframeSelectionResult:
        """
        Select keyframes from video based on question.
        
        Args:
            video_path: Path to video file
            question: Vietnamese traffic question
            
        Returns:
            KeyframeSelectionResult with selected frames and metadata
        """
        start_time = time.time()
        
        # Step 1: Analyze question
        logger.info(f"Analyzing question: {question[:50]}...")
        query_analysis = self._query_analyzer.analyze(question)
        
        logger.info(
            f"Query analysis: targets={query_analysis.target_objects}, "
            f"intent={query_analysis.question_intent.value}"
        )
        
        # Step 2: Sample frames from video
        logger.info(f"Sampling frames from: {video_path}")
        frames, timestamps, metadata = self._sample_frames(video_path)
        
        logger.info(f"Sampled {len(frames)} frames from {metadata['duration']:.1f}s video")
        
        # Step 3: Score frames (with optional YOLO for scoring)
        logger.info("Scoring frames...")
        
        # Only load YOLO for scoring if mode is ALL_FRAMES
        if self.config.yolo_mode == "all_frames" and self.config.beta > 0:
            self._load_yolo_model()
        
        # Get question for CLIP (translated if configured)
        query_text = query_analysis.translated_question or question
        
        detailed_scores = self._frame_scorer.score_frames(
            frames, 
            query_text,
            target_classes=query_analysis.yolo_classes,
            return_detailed=True
        )
        
        scores = np.array([s.final_score for s in detailed_scores])
        
        # Step 4: Select keyframes
        logger.info(f"Selecting {self.config.num_keyframes} keyframes...")
        
        if self.config.selection_strategy == "top_k":
            selected_indices = self._select_top_k(scores, self.config.num_keyframes)
        elif self.config.selection_strategy == "diverse_top_k":
            selected_indices = self._select_diverse_top_k(
                scores, frames, self.config.num_keyframes
            )
        elif self.config.selection_strategy == "temporal_weighted":
            selected_indices = self._select_temporal_weighted(
                scores, timestamps, self.config.num_keyframes
            )
        else:
            selected_indices = self._select_top_k(scores, self.config.num_keyframes)
        
        # Step 5: Run YOLO on selected frames (if mode is selected_only)
        detections = {}
        if self.config.yolo_mode == "selected_only":
            logger.info("Running YOLO on selected frames...")
            selected_frames = [frames[i] for i in selected_indices]
            detections = self._run_yolo_detection(selected_frames, selected_indices)
        
        # Step 6: Build results
        keyframes = []
        for idx in selected_indices:
            kf = KeyframeResult(
                frame_idx=idx,
                timestamp=timestamps[idx],
                frame=frames[idx],
                score=float(scores[idx]),
                score_details=detailed_scores[idx],
                detections=detections.get(idx),
                query_info=query_analysis
            )
            keyframes.append(kf)
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Selected {len(keyframes)} keyframes in {processing_time:.2f}s"
        )
        
        return KeyframeSelectionResult(
            keyframes=keyframes,
            query_analysis=query_analysis,
            total_frames_sampled=len(frames),
            processing_time=processing_time,
            config=self.config
        )
    
    def select_from_frames(self, frames: List[np.ndarray], question: str, timestamps: Optional[List[float]] = None) -> KeyframeSelectionResult:
        """Select keyframes from pre-loaded frames."""
        start_time = time.time()
        
        # Generate timestamps if not provided
        if timestamps is None:
            timestamps = list(range(len(frames)))
        
        # Analyze question
        query_analysis = self._query_analyzer.analyze(question)
        
        # Score frames
        query_text = query_analysis.translated_question or question
        detailed_scores = self._frame_scorer.score_frames(
            frames,
            query_text,
            target_classes=query_analysis.yolo_classes,
            return_detailed=True
        )
        
        scores = np.array([s.final_score for s in detailed_scores])
        
        # Select keyframes
        if self.config.selection_strategy == "top_k":
            selected_indices = self._select_top_k(scores, self.config.num_keyframes)
        elif self.config.selection_strategy == "diverse_top_k":
            selected_indices = self._select_diverse_top_k(
                scores, frames, self.config.num_keyframes
            )
        else:
            selected_indices = self._select_top_k(scores, self.config.num_keyframes)
        
        # Run YOLO if configured
        detections = {}
        if self.config.yolo_mode == "selected_only":
            selected_frames = [frames[i] for i in selected_indices]
            detections = self._run_yolo_detection(selected_frames, selected_indices)
        
        # Build results
        keyframes = []
        for idx in selected_indices:
            kf = KeyframeResult(
                frame_idx=idx,
                timestamp=float(timestamps[idx]),
                frame=frames[idx],
                score=float(scores[idx]),
                score_details=detailed_scores[idx],
                detections=detections.get(idx),
                query_info=query_analysis
            )
            keyframes.append(kf)
        
        processing_time = time.time() - start_time
        
        return KeyframeSelectionResult(
            keyframes=keyframes,
            query_analysis=query_analysis,
            total_frames_sampled=len(frames),
            processing_time=processing_time,
            config=self.config
        )

def create_selector(
    num_keyframes: int = 8,
    query_strategy: str = "keyword",
    scoring_strategy: str = "clip",
    yolo_mode: str = "selected_only",
    use_translation: bool = True,
    yolo_model_path: Optional[str] = None,
    **kwargs
) -> KeyframeSelector:
    """
    Factory function to create a KeyframeSelector.
    
    Args:
        num_keyframes: Number of keyframes to select
        query_strategy: Query analysis strategy
        scoring_strategy: Frame scoring strategy
        yolo_mode: YOLO detection mode
        use_translation: Translate Vietnamese for CLIP
        yolo_model_path: Path to YOLO model
        **kwargs: Additional config options
        
    Returns:
        Configured KeyframeSelector
    """
    config = KeyframeSelectorConfig(
        num_keyframes=num_keyframes,
        query_strategy=query_strategy,
        scoring_strategy=scoring_strategy,
        yolo_mode=yolo_mode,
        use_translation=use_translation,
        yolo_model_path=yolo_model_path,
        **kwargs
    )
    return KeyframeSelector(config)