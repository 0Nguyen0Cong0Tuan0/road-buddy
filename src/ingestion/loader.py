# Video Loading with Decord

import logging
import os
from typing import Iterator, Dict, Any, List, Optional, Tuple, Union
import torch
import numpy as np
from pathlib import Path

try:
    from decord import VideoReader, cpu, gpu
    from decord import bridge
    bridge.set_bridge('torch')
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    logging.warning("Decord not available. Install via: pip install decord")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available. Install via: pip install opencv-python")


class RoadVideoLoader:
    """GPU-accelerated video loader using Decord with fallback to OpenCV.
    
    Features:
    - GPU-accelerated decoding via Decord
    - Temporal sampling strategies (uniform, FPS-based, index-based)
    - Frame preprocessing (resize, normalize, format conversion)
    - Robust error handling with CPU/OpenCV fallbacks
    - Video validation and metadata extraction
    
    Args:
        config: Configuration object with video loading parameters
        
    Attributes:
        total_frames (int): Total number of frames in video
        fps (float): Frames per second of the video
        duration (float): Duration in seconds
    """

    def __init__(self, config):
        """Initialize video loader with configuration."""
        self.cfg = config
        self.video_path = Path(config.video_path)
        
        # Validate video file
        self._validate_video()
        
        # Try Decord first, fallback to OpenCV if needed
        self.backend = None
        self.reader = None
        
        try:
            self._init_decord()
            self.backend = 'decord'
        except Exception as e:
            logging.warning(f"Decord initialization failed: {e}. Falling back to OpenCV.")
            if CV2_AVAILABLE:
                self._init_opencv()
                self.backend = 'opencv'
            else:
                raise RuntimeError("Both Decord and OpenCV are unavailable. Cannot load video.")
        
        logging.info(
            f"Video loaded with {self.backend}: {self.total_frames} frames "
            f"at {self.fps:.2f} fps, duration {self.duration:.2f}s"
        )

    def _validate_video(self):
        """Validate video file exists and has valid format."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        
        if not self.video_path.is_file():
            raise ValueError(f"Path is not a file: {self.video_path}")
        
        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        if self.video_path.suffix.lower() not in valid_extensions:
            logging.warning(
                f"Video extension {self.video_path.suffix} may not be supported. "
                f"Valid extensions: {valid_extensions}"
            )

    def _init_decord(self):
        """Initialize Decord video reader."""
        if not DECORD_AVAILABLE:
            raise ImportError("Decord library is not installed.")
        
        self.ctx = self._get_context(self.cfg.device, self.cfg.ctx_id)
        
        self.reader = VideoReader(
            str(self.video_path),
            ctx=self.ctx,
            width=self.cfg.width,
            height=self.cfg.height,
            num_threads=self.cfg.num_threads
        )
        
        self.total_frames = len(self.reader)
        self.fps = self.reader.get_avg_fps()
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
    def _init_opencv(self):
        """Initialize OpenCV video reader as fallback."""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV library is not installed.")
        
        self.reader = cv2.VideoCapture(str(self.video_path))
        
        if not self.reader.isOpened():
            raise RuntimeError(f"OpenCV failed to open video: {self.video_path}")
        
        self.total_frames = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.reader.get(cv2.CAP_PROP_FPS)
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

    def _get_context(self, device_str: str, device_id: int):
        """Get Decord device context (GPU or CPU)."""
        if 'gpu' in device_str.lower() or 'cuda' in device_str.lower():
            if torch.cuda.is_available():
                return gpu(device_id)
            else:
                logging.warning("CUDA not available, falling back to CPU.")
                return cpu(0)
        return cpu(0)

    # ==================== Streaming Methods ====================

    def stream_batches(self) -> Iterator[torch.Tensor]:
        """Yield batches of video frames on target device.
        
        Yields:
            torch.Tensor: Batch of frames with shape (B, C, H, W), normalized to [0, 1]
        """
        batch_size = self.cfg.batch_size

        for i in range(0, self.total_frames, batch_size):
            end_idx = min(i + batch_size, self.total_frames)
            indices = list(range(i, end_idx))

            try:
                batch_tensor = self._get_batch(indices)
                yield batch_tensor
            except Exception as e:
                logging.error(f"Error decoding batch at frames {i}-{end_idx}: {e}")
                continue

    def _get_batch(self, indices: List[int]) -> torch.Tensor:
        """Get a batch of frames by indices.
        
        Args:
            indices: List of frame indices to extract
            
        Returns:
            torch.Tensor: Frames with shape (B, C, H, W), normalized to [0, 1]
        """
        if self.backend == 'decord':
            batch = self.reader.get_batch(indices)
            # Convert from (B, H, W, C) to (B, C, H, W) and normalize
            return batch.permute(0, 3, 1, 2).float() / 255.0
        else:  # opencv
            frames = []
            for idx in indices:
                frame = self._get_frame_opencv(idx)
                frames.append(frame)
            return torch.stack(frames)

    # ==================== Single Frame Extraction ====================
    
    def get_frame(self, frame_idx: int) -> torch.Tensor:
        """Get a single frame by index.
        
        Args:
            frame_idx: Frame index (0 to total_frames-1)
            
        Returns:
            torch.Tensor: Frame with shape (C, H, W), normalized to [0, 1]
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise IndexError(
                f"Frame index {frame_idx} out of range [0, {self.total_frames})"
            )
        
        if self.backend == 'decord':
            frame = self.reader[frame_idx]
            return frame.permute(2, 0, 1).float() / 255.0
        else:
            return self._get_frame_opencv(frame_idx)

    def _get_frame_opencv(self, frame_idx: int) -> torch.Tensor:
        """Get frame using OpenCV backend.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            torch.Tensor: Frame with shape (C, H, W), normalized to [0, 1]
        """
        self.reader.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.reader.read()
        
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx}")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to tensor (H, W, C) -> (C, H, W)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        return frame

    # ==================== Temporal Sampling Methods ====================

    def sample_uniform(self, num_frames: int) -> torch.Tensor:
        """Sample frames uniformly across the video.
        
        Args:
            num_frames: Number of frames to sample
            
        Returns:
            torch.Tensor: Sampled frames with shape (N, C, H, W)
            
        Example:
            >>> loader = RoadVideoLoader(config)
            >>> frames = loader.sample_uniform(8)  # Get 8 evenly spaced frames
        """
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        
        if num_frames > self.total_frames:
            logging.warning(
                f"Requested {num_frames} frames but video only has {self.total_frames}. "
                f"Using all frames."
            )
            num_frames = self.total_frames
        
        # Calculate evenly spaced indices
        indices = np.linspace(0, self.total_frames - 1, num_frames, dtype=int).tolist()
        
        return self.sample_indices(indices)

    def sample_fps(self, target_fps: float) -> torch.Tensor:
        """Sample frames at a target FPS.
        
        Args:
            target_fps: Target frames per second
            
        Returns:
            torch.Tensor: Sampled frames with shape (N, C, H, W)
            
        Example:
            >>> loader = RoadVideoLoader(config)
            >>> frames = loader.sample_fps(1.0)  # Sample at 1 FPS
        """
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        
        # Calculate frame interval
        interval = max(1, int(self.fps / target_fps))
        
        # Generate indices
        indices = list(range(0, self.total_frames, interval))
        
        logging.info(
            f"Sampling at {target_fps} FPS (interval={interval}): "
            f"{len(indices)} frames from {self.total_frames}"
        )
        
        return self.sample_indices(indices)

    def sample_indices(self, indices: List[int]) -> torch.Tensor:
        """Sample specific frame indices.
        
        Args:
            indices: List of frame indices to extract
            
        Returns:
            torch.Tensor: Sampled frames with shape (N, C, H, W)
            
        Raises:
            IndexError: If any index is out of range
        """
        if not indices:
            raise ValueError("indices list is empty")
        
        # Validate indices
        for idx in indices:
            if idx < 0 or idx >= self.total_frames:
                raise IndexError(
                    f"Frame index {idx} out of range [0, {self.total_frames})"
                )
        
        return self._get_batch(indices)

    # ==================== Preprocessing Methods ====================

    def preprocess_frame(
        self,
        frame: torch.Tensor,
        resize: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ) -> torch.Tensor:
        """Preprocess a single frame.
        
        Args:
            frame: Input frame tensor (C, H, W) in range [0, 1]
            resize: Target size (height, width), None to skip
            normalize: Whether to apply normalization
            mean: Mean for normalization (ImageNet default)
            std: Standard deviation for normalization (ImageNet default)
            
        Returns:
            torch.Tensor: Preprocessed frame
        """
        # Resize if needed
        if resize is not None:
            import torch.nn.functional as F
            # Add batch dimension for resize
            frame = frame.unsqueeze(0)
            frame = F.interpolate(
                frame,
                size=resize,
                mode='bilinear',
                align_corners=False
            )
            frame = frame.squeeze(0)
        
        # Normalize if needed
        if normalize:
            mean_tensor = torch.tensor(mean).view(3, 1, 1)
            std_tensor = torch.tensor(std).view(3, 1, 1)
            frame = (frame - mean_tensor) / std_tensor
        
        return frame

    def preprocess_batch(
        self,
        frames: torch.Tensor,
        resize: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ) -> torch.Tensor:
        """Preprocess a batch of frames.
        
        Args:
            frames: Input frames tensor (B, C, H, W) in range [0, 1]
            resize: Target size (height, width), None to skip
            normalize: Whether to apply normalization
            mean: Mean for normalization
            std: Standard deviation for normalization
            
        Returns:
            torch.Tensor: Preprocessed frames
        """
        # Resize if needed
        if resize is not None:
            import torch.nn.functional as F
            frames = F.interpolate(
                frames,
                size=resize,
                mode='bilinear',
                align_corners=False
            )
        
        # Normalize if needed
        if normalize:
            mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
            std_tensor = torch.tensor(std).view(1, 3, 1, 1)
            frames = (frames - mean_tensor) / std_tensor
        
        return frames

    # ==================== Metadata and Utilities ====================

    def get_metadata(self) -> Dict[str, Any]:
        """Return comprehensive video metadata.
        
        Returns:
            dict: Video metadata including fps, frames, duration, resolution
        """
        if self.backend == 'decord':
            height, width = self.reader[0].shape[:2]
        else:  # opencv
            height = int(self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(self.reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        return {
            "backend": self.backend,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration": self.duration,
            "width": width,
            "height": height,
            "path": str(self.video_path),
            "size_mb": self.video_path.stat().st_size / (1024 * 1024)
        }

    def estimate_memory(self, num_frames: int, batch_size: int = 1) -> float:
        """Estimate memory usage for loading frames.
        
        Args:
            num_frames: Number of frames to load
            batch_size: Batch size for processing
            
        Returns:
            float: Estimated memory in MB
        """
        if self.backend == 'decord':
            height, width = self.reader[0].shape[:2]
        else:
            height = int(self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(self.reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # 3 channels, 4 bytes per float32
        bytes_per_frame = height * width * 3 * 4
        
        # Account for batch processing
        total_bytes = bytes_per_frame * min(num_frames, batch_size)
        
        return total_bytes / (1024 * 1024)  # Convert to MB

    def __len__(self) -> int:
        """Return total number of frames."""
        return self.total_frames

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RoadVideoLoader(path='{self.video_path.name}', "
            f"backend={self.backend}, frames={self.total_frames}, "
            f"fps={self.fps:.2f}, duration={self.duration:.2f}s)"
        )

    def __del__(self):
        """Cleanup resources."""
        if self.backend == 'opencv' and self.reader is not None:
            self.reader.release()