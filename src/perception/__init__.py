"""
Perception Module for Object Detection, Tracking, and Query-Guided Frame Selection.

This module provides:
1. YOLO-based object detection and tracking using finetuned models
2. Query-guided keyframe selection for Video QA tasks

Components:
    - PerceptionEngine: Main detection/tracking interface
    - ModelRegistry: Registry of available finetuned models
    - Detection, FrameDetections: Result dataclasses
    - parse_yolo_results: Result parsing utility
    
    Query-Guided Components (NEW):
    - QueryAnalyzer: Parse questions to extract target objects
    - FrameScorer: Score frames by relevance using CLIP/YOLO
    - KeyframeSelector: Select query-aware keyframes

Usage:
    # Detection/Tracking
    from src.perception import PerceptionEngine, get_model
    engine = PerceptionEngine(config)
    detections = engine.detect_and_parse(frames)
    
    # Query-Guided Keyframe Selection
    from src.perception import KeyframeSelector, KeyframeSelectorConfig
    selector = KeyframeSelector(KeyframeSelectorConfig(num_keyframes=8))
    result = selector.select("video.mp4", "Biển báo tốc độ là bao nhiêu?")
"""
from .detector import PerceptionEngine
from .model_registry import (
    ModelRegistry,
    ModelInfo,
    get_model,
)
from .results import (
    Detection,
    FrameDetections,
    parse_yolo_results,
    aggregate_detections,
    detections_to_annotations,
)

# Query-Guided Perception Components
from .query_analyzer import (
    QueryAnalyzer,
    QueryAnalysisResult,
    QuestionIntent,
)
from .frame_scorer import (
    FrameScorer,
    ScoringConfig,
    FrameScore,
)
from .keyframe_selector import (
    KeyframeSelector,
    KeyframeSelectorConfig,
    KeyframeResult,
    KeyframeSelectionResult,
)

__all__ = [
    # Main engine
    "PerceptionEngine",
    # Model registry
    "ModelRegistry",
    "ModelInfo", 
    "get_model",
    # Results
    "Detection",
    "FrameDetections",
    "parse_yolo_results",
    "aggregate_detections",
    "detections_to_annotations",
    # Query-Guided Perception
    "QueryAnalyzer",
    "QueryAnalysisResult",
    "QuestionIntent",
    "FrameScorer",
    "ScoringConfig",
    "FrameScore",
    "KeyframeSelector",
    "KeyframeSelectorConfig",
    "KeyframeResult",
    "KeyframeSelectionResult",
]
