# Utility Functions for Video Ingestion

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import subprocess
import json


def validate_video(video_path: str) -> bool:
    """Validate video file integrity and format.
    
    Args:
        video_path: Path to video file
        
    Returns:
        bool: True if video is valid, False otherwise
    """
    path = Path(video_path)
    
    # Check existence
    if not path.exists():
        logging.warning(f"Video file not found: {video_path}")
        return False
    
    if not path.is_file():
        logging.warning(f"Path is not a file: {video_path}")
        return False
    
    # Check size (shouldn't be empty)
    if path.stat().st_size == 0:
        logging.warning(f"Video file is empty: {video_path}")
        return False
    
    # Check extension
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    if path.suffix.lower() not in valid_extensions:
        logging.warning(f"Invalid video extension: {path.suffix}")
        return False
    
    return True


def get_video_info(video_path: str, use_ffprobe: bool = True) -> Optional[Dict[str, Any]]:
    """Extract video metadata quickly without loading the video.
    
    This uses ffprobe if available for fast metadata extraction without
    loading the entire video into memory.
    
    Args:
        video_path: Path to video file
        use_ffprobe: Whether to use ffprobe (faster) or fallback to cv2
        
    Returns:
        dict: Video metadata or None if failed
        
    Example:
        >>> info = get_video_info('video.mp4')
        >>> print(f"Duration: {info['duration']}s, FPS: {info['fps']}")
    """
    path = Path(video_path)
    
    if not path.exists():
        logging.error(f"Video not found: {video_path}")
        return None
    
    # Try ffprobe first (fastest)
    if use_ffprobe:
        try:
            return _get_video_info_ffprobe(str(path))
        except Exception as e:
            logging.warning(f"ffprobe failed, falling back to OpenCV: {e}")
    
    # Fallback to OpenCV
    try:
        return _get_video_info_opencv(str(path))
    except Exception as e:
        logging.error(f"Failed to get video info: {e}")
        return None


def _get_video_info_ffprobe(video_path: str) -> Dict[str, Any]:
    """Get video info using ffprobe (fast, no video loading).
    
    Args:
        video_path: Path to video file
        
    Returns:
        dict: Video metadata
    """
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    
    probe_data = json.loads(result.stdout)
    
    # Extract video stream
    video_stream = None
    for stream in probe_data.get('streams', []):
        if stream.get('codec_type') == 'video':
            video_stream = stream
            break
    
    if not video_stream:
        raise ValueError("No video stream found")
    
    # Parse metadata
    format_data = probe_data.get('format', {})
    
    # Calculate FPS
    fps_str = video_stream.get('r_frame_rate', '0/1')
    num, den = map(int, fps_str.split('/'))
    fps = num / den if den > 0 else 0
    
    return {
        "width": int(video_stream.get('width', 0)),
        "height": int(video_stream.get('height', 0)),
        "fps": fps,
        "duration": float(format_data.get('duration', 0)),
        "total_frames": int(video_stream.get('nb_frames', 0)),
        "codec": video_stream.get('codec_name', 'unknown'),
        "bitrate": int(format_data.get('bit_rate', 0)),
        "size_mb": int(format_data.get('size', 0)) / (1024 * 1024),
        "format": format_data.get('format_name', 'unknown')
    }


def _get_video_info_opencv(video_path: str) -> Dict[str, Any]:
    """Get video info using OpenCV (slower, loads video).
    
    Args:
        video_path: Path to video file
        
    Returns:
        dict: Video metadata
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        path = Path(video_path)
        size_mb = path.stat().st_size / (1024 * 1024)
        
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "duration": duration,
            "total_frames": total_frames,
            "size_mb": size_mb,
            "codec": "unknown",
            "format": path.suffix
        }
    finally:
        cap.release()


def estimate_memory(
    video_info: Dict[str, Any],
    num_frames: Optional[int] = None,
    batch_size: int = 1,
    dtype: str = 'float32') -> float:
    """Estimate memory usage for video processing.
    
    Args:
        video_info: Video metadata dict from get_video_info()
        num_frames: Number of frames to load (None = all frames)
        batch_size: Batch size for processing
        dtype: Data type ('float32', 'float16', 'uint8')
        
    Returns:
        float: Estimated memory in MB
        
    Example:
        >>> info = get_video_info('video.mp4')
        >>> mem = estimate_memory(info, num_frames=8, batch_size=4)
        >>> print(f"Estimated memory: {mem:.2f} MB")
    """
    height = video_info['height']
    width = video_info['width']
    
    # Determine bytes per pixel based on dtype
    bytes_per_value = {
        'float32': 4,
        'float16': 2,
        'uint8': 1
    }.get(dtype, 4)
    
    # 3 channels (RGB)
    bytes_per_frame = height * width * 3 * bytes_per_value
    
    # Calculate total frames to load
    if num_frames is None:
        num_frames = video_info['total_frames']
    
    # Account for batch processing (only batch_size frames in memory at once)
    frames_in_memory = min(num_frames, batch_size)
    
    total_bytes = bytes_per_frame * frames_in_memory
    
    return total_bytes / (1024 * 1024)  # Convert to MB


def convert_timestamp_to_frame(timestamp: float, fps: float) -> int:
    """Convert timestamp in seconds to frame index.
    
    Args:
        timestamp: Time in seconds
        fps: Frames per second
        
    Returns:
        int: Frame index
        
    Example:
        >>> frame_idx = convert_timestamp_to_frame(2.5, 30)  # 2.5 seconds at 30 FPS
        >>> print(frame_idx)  # 75
    """
    return int(timestamp * fps)


def convert_frame_to_timestamp(frame_idx: int, fps: float) -> float:
    """Convert frame index to timestamp in seconds.
    
    Args:
        frame_idx: Frame index
        fps: Frames per second
        
    Returns:
        float: Timestamp in seconds
        
    Example:
        >>> timestamp = convert_frame_to_timestamp(75, 30)
        >>> print(f"{timestamp:.2f}s")  # 2.50s
    """
    return frame_idx / fps

def expand_bbox(
    bbox: Tuple[int, int, int, int],
    expansion_ratio: float,
    image_width: int,
    image_height: int
) -> Tuple[int, int, int, int]:
    """Expand bounding box by a ratio while staying within image bounds.
    
    Args:
        bbox: Original bounding box (x1, y1, x2, y2)
        expansion_ratio: Expansion ratio (e.g., 0.2 for 20% expansion)
        image_width: Image width
        image_height: Image height
        
    Returns:
        tuple: Expanded bounding box (x1, y1, x2, y2)
        
    Example:
        >>> bbox = (100, 100, 200, 200)
        >>> expanded = expand_bbox(bbox, 0.2, 1920, 1080)
    """
    x1, y1, x2, y2 = bbox
    
    width = x2 - x1
    height = y2 - y1
    
    # Calculate expansion amounts
    expand_w = int(width * expansion_ratio / 2)
    expand_h = int(height * expansion_ratio / 2)
    
    # Apply expansion
    x1_new = max(0, x1 - expand_w)
    y1_new = max(0, y1 - expand_h)
    x2_new = min(image_width, x2 + expand_w)
    y2_new = min(image_height, y2 + expand_h)
    
    return (x1_new, y1_new, x2_new, y2_new)


def calculate_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box (x1, y1, x2, y2)
        bbox2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        float: IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0  # No intersection
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0