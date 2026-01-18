# Frame Sampling Strategies for Video Ingestion
#
# This module provides various sampling strategies to extract frames from videos.
# Key design principle: NO resizing/cropping - only temporal sampling to preserve image quality.

import logging
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class FrameSampler:
    """Frame sampling strategies for video processing.
    
    Design Philosophy:
    - Longer videos should yield more frames to capture more information
    - Sampling should be deterministic and reproducible
    - No image transformation (resize/crop) to preserve quality for downstream models
    
    Available Strategies:
    - Uniform: Evenly spaced frames across the video
    - Adaptive: Frame count scales with video duration
    - FPS-based: Sample at a target frames-per-second rate
    - Temporal chunks: Sample from video segments for better temporal coverage
    
    Example:
        >>> sampler = FrameSampler()
        >>> indices = sampler.sample_adaptive(
        ...     total_frames=1800,  # 1 minute video at 30fps
        ...     fps=30.0,
        ...     min_frames=8,
        ...     max_frames=64
        ... )
        >>> print(f"Selected {len(indices)} frames")
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize sampler.
        
        Args:
            seed: Random seed for reproducibility (used in stochastic sampling)
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    # ==================== Core Sampling Methods ====================
    
    def sample_uniform(
        self,
        total_frames: int,
        num_frames: int
    ) -> List[int]:
        """Sample frames uniformly across the video.
        
        Creates evenly spaced indices from start to end of video.
        
        Args:
            total_frames: Total number of frames in video
            num_frames: Number of frames to sample
            
        Returns:
            List of frame indices
            
        Example:
            >>> sampler = FrameSampler()
            >>> indices = sampler.sample_uniform(1000, 10)
            >>> # Returns [0, 111, 222, 333, 444, 555, 666, 777, 888, 999]
        """
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        
        if num_frames >= total_frames:
            logger.warning(
                f"Requested {num_frames} frames but video has {total_frames}. "
                f"Returning all frames."
            )
            return list(range(total_frames))
        
        # Calculate evenly spaced indices
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        
        return indices
    
    def sample_adaptive(
        self,
        total_frames: int,
        fps: float,
        min_frames: int = 4,
        max_frames: int = 64,
        frames_per_second: float = 0.5
    ) -> List[int]:
        """Adaptive sampling based on video duration.
        
        Longer videos get more frames, with configurable bounds.
        Formula: num_frames = clip(duration * frames_per_second, min_frames, max_frames)
        
        Args:
            total_frames: Total number of frames in video
            fps: Video frames per second
            min_frames: Minimum number of frames to return
            max_frames: Maximum number of frames to return
            frames_per_second: Target sampling rate (frames per second of video duration)
            
        Returns:
            List of frame indices
            
        Example:
            >>> sampler = FrameSampler()
            >>> # 30 second video at 30fps = 900 frames
            >>> indices = sampler.sample_adaptive(
            ...     total_frames=900, fps=30.0,
            ...     min_frames=8, max_frames=64, frames_per_second=0.5
            ... )
            >>> # duration = 30s, num_frames = min(max(30*0.5, 8), 64) = 15 frames
        """
        if total_frames <= 0:
            raise ValueError(f"total_frames must be positive, got {total_frames}")
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        
        # Calculate video duration
        duration = total_frames / fps
        
        # Calculate adaptive frame count
        target_frames = int(duration * frames_per_second)
        
        # Clamp to bounds
        num_frames = max(min_frames, min(target_frames, max_frames))
        
        # Ensure we don't exceed total frames
        num_frames = min(num_frames, total_frames)
        
        logger.debug(
            f"Adaptive sampling: duration={duration:.1f}s, "
            f"target={target_frames}, actual={num_frames}"
        )
        
        return self.sample_uniform(total_frames, num_frames)
    
    def sample_fps(
        self,
        total_frames: int,
        video_fps: float,
        target_fps: float = 1.0
    ) -> List[int]:
        """Sample frames at a target FPS rate.
        
        Args:
            total_frames: Total number of frames in video
            video_fps: Original video FPS
            target_fps: Target sampling FPS
            
        Returns:
            List of frame indices
            
        Example:
            >>> sampler = FrameSampler()
            >>> # Sample 1 frame per second from 30fps video
            >>> indices = sampler.sample_fps(900, video_fps=30.0, target_fps=1.0)
            >>> # Returns 30 frames (one every 30 frames)
        """
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        if video_fps <= 0:
            raise ValueError(f"video_fps must be positive, got {video_fps}")
        
        # Calculate frame interval
        interval = max(1, int(video_fps / target_fps))
        
        # Generate indices
        indices = list(range(0, total_frames, interval))
        
        logger.debug(
            f"FPS sampling: interval={interval}, "
            f"sampled {len(indices)} from {total_frames} frames"
        )
        
        return indices
    
    def sample_temporal_chunks(
        self,
        total_frames: int,
        num_chunks: int = 4,
        frames_per_chunk: int = 2
    ) -> List[int]:
        """Sample frames from temporal chunks for better coverage.
        
        Divides video into chunks and samples from each chunk,
        ensuring frames are spread across the temporal extent.
        
        Good for videos where important events may occur at different times.
        
        Args:
            total_frames: Total number of frames in video
            num_chunks: Number of temporal chunks to divide video into
            frames_per_chunk: Number of frames to sample from each chunk
            
        Returns:
            List of frame indices, sorted in temporal order
            
        Example:
            >>> sampler = FrameSampler()
            >>> # Sample 2 frames from each of 4 chunks
            >>> indices = sampler.sample_temporal_chunks(1000, num_chunks=4, frames_per_chunk=2)
            >>> # Returns 8 frames spread across video
        """
        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks}")
        if frames_per_chunk <= 0:
            raise ValueError(f"frames_per_chunk must be positive, got {frames_per_chunk}")
        
        total_requested = num_chunks * frames_per_chunk
        if total_requested >= total_frames:
            logger.warning(
                f"Requested {total_requested} frames but video has {total_frames}. "
                f"Returning uniform sample."
            )
            return self.sample_uniform(total_frames, min(total_requested, total_frames))
        
        indices = []
        chunk_size = total_frames // num_chunks
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = chunk_start + chunk_size if chunk_idx < num_chunks - 1 else total_frames
            
            # Sample uniformly within chunk
            chunk_frames = min(frames_per_chunk, chunk_end - chunk_start)
            if chunk_frames > 0:
                chunk_indices = np.linspace(
                    chunk_start, chunk_end - 1, chunk_frames, dtype=int
                ).tolist()
                indices.extend(chunk_indices)
        
        # Remove duplicates and sort
        indices = sorted(set(indices))
        
        return indices
    
    # ==================== Utility Methods ====================

    
    def calculate_frame_count(
        self,
        duration: float,
        frames_per_second: float = 0.5,
        min_frames: int = 4,
        max_frames: int = 64
    ) -> int:
        """Calculate recommended frame count for a video duration.
        
        Args:
            duration: Video duration in seconds
            frames_per_second: Target sampling rate
            min_frames: Minimum frames
            max_frames: Maximum frames
            
        Returns:
            Recommended number of frames
            
        Example:
            >>> sampler = FrameSampler()
            >>> count = sampler.calculate_frame_count(60.0)  # 1 minute video
            >>> print(count)  # 30 frames (60 * 0.5, clamped to 4-64)
        """
        target = int(duration * frames_per_second)
        return max(min_frames, min(target, max_frames))
    
    def get_sampling_info(
        self,
        total_frames: int,
        fps: float,
        indices: List[int]
    ) -> dict:
        """Get information about a sampling result.
        
        Args:
            total_frames: Total frames in video
            fps: Video FPS
            indices: Sampled frame indices
            
        Returns:
            Dict with sampling statistics
        """
        if not indices:
            return {
                "num_sampled": 0,
                "coverage_ratio": 0.0,
                "avg_interval": 0,
                "sampling_fps": 0.0
            }
        
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate average interval between samples
        if len(indices) > 1:
            intervals = [indices[i+1] - indices[i] for i in range(len(indices) - 1)]
            avg_interval = sum(intervals) / len(intervals)
        else:
            avg_interval = total_frames
        
        return {
            "num_sampled": len(indices),
            "total_frames": total_frames,
            "coverage_ratio": len(indices) / total_frames if total_frames > 0 else 0,
            "avg_interval": avg_interval,
            "avg_interval_seconds": avg_interval / fps if fps > 0 else 0,
            "sampling_fps": len(indices) / duration if duration > 0 else 0,
            "first_frame": indices[0],
            "last_frame": indices[-1],
            "video_duration": duration
        }


# ==================== Convenience Functions ====================

def sample_video_adaptive(
    total_frames: int,
    fps: float,
    min_frames: int = 8,
    max_frames: int = 64,
    frames_per_second: float = 0.5
) -> List[int]:
    """Convenience function for adaptive sampling.
    
    Args:
        total_frames: Total frames in video
        fps: Video FPS
        min_frames: Minimum frames to sample
        max_frames: Maximum frames to sample
        frames_per_second: Target sampling rate
        
    Returns:
        List of frame indices
    """
    sampler = FrameSampler()
    return sampler.sample_adaptive(
        total_frames=total_frames,
        fps=fps,
        min_frames=min_frames,
        max_frames=max_frames,
        frames_per_second=frames_per_second
    )


def sample_video_uniform(total_frames: int, num_frames: int) -> List[int]:
    """Convenience function for uniform sampling.
    
    Args:
        total_frames: Total frames in video
        num_frames: Number of frames to sample
        
    Returns:
        List of frame indices
    """
    sampler = FrameSampler()
    return sampler.sample_uniform(total_frames, num_frames)
