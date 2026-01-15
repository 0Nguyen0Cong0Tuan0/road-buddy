"""Video Ingestion Module for RoadBuddy.

This module provides GPU-accelerated video loading, temporal sampling,
and batch processing capabilities for the RoadBuddy traffic law QA system.

Main Components:
- RoadVideoLoader: Core video loading with Decord/OpenCV backends
- BatchVideoProcessor: Batch processing for datasets
- Utility functions: Validation, metadata extraction, bbox operations

Example Usage:
    >>> from src.ingestion import RoadVideoLoader, DecordConfig
    >>> 
    >>> config = DecordConfig(video_path='video.mp4', device='gpu')
    >>> loader = RoadVideoLoader(config)
    >>> frames = loader.sample_uniform(8)  # Get 8 evenly spaced frames
    >>> metadata = loader.get_metadata()
"""

from .loader import RoadVideoLoader
from .processor import BatchVideoProcessor, ProcessingStats, extract_keyframes
from .utils import (
    validate_video,
    get_video_info,
    estimate_memory,
    convert_timestamp_to_frame,
    convert_frame_to_timestamp,
    crop_and_resize,
    expand_bbox,
    calculate_iou,
    create_video_thumbnail
)

__version__ = "1.0.0"
__all__ = [
    # Core classes
    "RoadVideoLoader",
    "BatchVideoProcessor",
    "ProcessingStats",
    
    # Processing functions
    "extract_keyframes",
    
    # Validation & info
    "validate_video",
    "get_video_info",
    "estimate_memory",
    
    # Time/Frame conversion
    "convert_timestamp_to_frame",
    "convert_frame_to_timestamp",
    
    # Image operations
    "crop_and_resize",
    "expand_bbox",
    "calculate_iou",
    "create_video_thumbnail",
]
