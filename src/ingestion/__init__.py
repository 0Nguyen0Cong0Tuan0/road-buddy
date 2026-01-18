"""Video Ingestion Module for RoadBuddy.

This module provides GPU-accelerated video loading, temporal sampling,
and batch processing capabilities for the RoadBuddy traffic law QA system.

Design Principle: NO resizing/cropping during ingestion to preserve image quality.
Only temporal sampling is performed - downstream models receive native resolution frames.

Main Components:
- RoadVideoLoader: Core video loading with Decord/OpenCV backends
- FrameSampler: Intelligent frame sampling strategies (uniform, adaptive, temporal)
- BatchVideoProcessor: Batch processing for datasets
- Utility functions: Validation, metadata extraction

Sampling Strategies:
- Uniform: Evenly spaced frames
- Adaptive: Frame count scales with video duration (longer = more frames)
- FPS-based: Sample at target FPS rate
- Temporal chunks: Sample from video segments

Example Usage:
    >>> from src.ingestion import RoadVideoLoader, FrameSampler
    >>> from src.configs import DecordConfig
    >>> 
    >>> config = DecordConfig(video_path='video.mp4', device='gpu')
    >>> loader = RoadVideoLoader(config)
    >>> 
    >>> # Adaptive sampling: longer videos get more frames
    >>> frames = loader.sample_adaptive(min_frames=8, max_frames=64)
    >>> 
    >>> # Or use FrameSampler directly for indices
    >>> sampler = FrameSampler()
    >>> indices = sampler.sample_adaptive(loader.total_frames, loader.fps)
"""

from .loader import RoadVideoLoader
from .sampler import (
    FrameSampler,
    sample_video_adaptive,
    sample_video_uniform
)
from .processor import BatchVideoProcessor, ProcessingStats, extract_keyframes
from .utils import (
    validate_video,
    get_video_info,
    estimate_memory,
    convert_timestamp_to_frame,
    convert_frame_to_timestamp,
    expand_bbox,
    calculate_iou
)


__version__ = "1.1.0"
__all__ = [
    # Core classes
    "RoadVideoLoader",
    "FrameSampler",
    "BatchVideoProcessor",
    "ProcessingStats",
    
    # Sampling utilities
    "sample_video_adaptive",
    "sample_video_uniform",
    
    # Processing functions
    "extract_keyframes",
    
    # Validation & info
    "validate_video",
    "get_video_info",
    "estimate_memory",
    
    # Time/Frame conversion
    "convert_timestamp_to_frame",
    "convert_frame_to_timestamp",
    
    # Image operations (available for post-detection use)
    "expand_bbox",
    "calculate_iou",
]