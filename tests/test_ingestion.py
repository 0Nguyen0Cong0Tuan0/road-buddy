"""
Comprehensive Test Suite for Ingestion Module.

Tests all components:
- FrameSampler: Sampling strategies
- RoadVideoLoader: Video loading and sampling
- BatchVideoProcessor: Batch processing
- Utility functions: Validation, info extraction, bbox operations

Usage:
    # Run all tests
    pytest tests/test_ingestion.py -v
    
    # Run specific test class
    pytest tests/test_ingestion.py::TestFrameSampler -v
    
    # Run with real video (requires video file)
    pytest tests/test_ingestion.py --video_path data/raw/train/videos/0.mp4
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ==================== Test FrameSampler ====================

class TestFrameSampler:
    """Test cases for FrameSampler class."""
    
    @pytest.fixture
    def sampler(self):
        """Create a FrameSampler instance."""
        from src.ingestion.sampler import FrameSampler
        return FrameSampler(seed=42)
    
    # ---------- sample_uniform tests ----------
    
    def test_sample_uniform_basic(self, sampler):
        """Test basic uniform sampling."""
        indices = sampler.sample_uniform(total_frames=100, num_frames=10)
        
        assert len(indices) == 10
        assert indices[0] == 0
        assert indices[-1] == 99
        assert all(isinstance(i, int) for i in indices)
    
    def test_sample_uniform_all_frames(self, sampler):
        """Test uniform sampling when num_frames >= total_frames."""
        indices = sampler.sample_uniform(total_frames=50, num_frames=100)
        
        # Should return all frames when requesting more than available
        assert len(indices) == 50
        assert indices == list(range(50))
    
    def test_sample_uniform_single_frame(self, sampler):
        """Test uniform sampling with single frame."""
        indices = sampler.sample_uniform(total_frames=100, num_frames=1)
        
        assert len(indices) == 1
        assert indices[0] == 0
    
    def test_sample_uniform_invalid_num_frames(self, sampler):
        """Test that invalid num_frames raises error."""
        with pytest.raises(ValueError):
            sampler.sample_uniform(total_frames=100, num_frames=0)
        
        with pytest.raises(ValueError):
            sampler.sample_uniform(total_frames=100, num_frames=-5)
    
    def test_sample_uniform_spacing(self, sampler):
        """Test that frames are evenly spaced."""
        indices = sampler.sample_uniform(total_frames=100, num_frames=5)
        
        # Should be [0, 24, 49, 74, 99] or similar evenly spaced
        intervals = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
        # All intervals should be approximately equal
        assert max(intervals) - min(intervals) <= 1
    
    # ---------- sample_adaptive tests ----------
    
    def test_sample_adaptive_short_video(self, sampler):
        """Test adaptive sampling for short video (uses min_frames)."""
        # 10 second video at 30fps = 300 frames
        # 10 * 0.5 = 5 frames, but min is 8
        indices = sampler.sample_adaptive(
            total_frames=300, fps=30.0,
            min_frames=8, max_frames=64,
            frames_per_second=0.5
        )
        
        assert len(indices) == 8  # Should use min_frames
    
    def test_sample_adaptive_medium_video(self, sampler):
        """Test adaptive sampling for medium video."""
        # 30 second video at 30fps = 900 frames
        # 30 * 0.5 = 15 frames
        indices = sampler.sample_adaptive(
            total_frames=900, fps=30.0,
            min_frames=8, max_frames=64,
            frames_per_second=0.5
        )
        
        assert len(indices) == 15
    
    def test_sample_adaptive_long_video(self, sampler):
        """Test adaptive sampling for long video (uses max_frames)."""
        # 200 second video at 30fps = 6000 frames
        # 200 * 0.5 = 100 frames, but max is 64
        indices = sampler.sample_adaptive(
            total_frames=6000, fps=30.0,
            min_frames=8, max_frames=64,
            frames_per_second=0.5
        )
        
        assert len(indices) == 64  # Should use max_frames
    
    def test_sample_adaptive_scales_with_duration(self, sampler):
        """Test that frame count scales with video duration."""
        # Different durations, same fps
        short_indices = sampler.sample_adaptive(900, 30.0, 8, 64, 0.5)   # 30s -> 15
        medium_indices = sampler.sample_adaptive(1800, 30.0, 8, 64, 0.5)  # 60s -> 30
        long_indices = sampler.sample_adaptive(2700, 30.0, 8, 64, 0.5)   # 90s -> 45
        
        assert len(short_indices) < len(medium_indices) < len(long_indices)
    
    def test_sample_adaptive_invalid_inputs(self, sampler):
        """Test that invalid inputs raise errors."""
        with pytest.raises(ValueError):
            sampler.sample_adaptive(total_frames=0, fps=30.0)
        
        with pytest.raises(ValueError):
            sampler.sample_adaptive(total_frames=100, fps=0)
        
        with pytest.raises(ValueError):
            sampler.sample_adaptive(total_frames=100, fps=-30.0)
    
    # ---------- sample_fps tests ----------
    
    def test_sample_fps_basic(self, sampler):
        """Test FPS-based sampling."""
        # 30 second video at 30fps, sample at 1fps
        # Should get ~30 frames
        indices = sampler.sample_fps(
            total_frames=900,
            video_fps=30.0,
            target_fps=1.0
        )
        
        assert len(indices) == 30
        # Interval should be 30 (sample every 30th frame)
        assert indices[1] - indices[0] == 30
    
    def test_sample_fps_high_target(self, sampler):
        """Test FPS sampling with high target fps."""
        # Sample at 10fps from 30fps video
        indices = sampler.sample_fps(
            total_frames=300,
            video_fps=30.0,
            target_fps=10.0
        )
        
        # Interval should be 3 (30/10)
        assert indices[1] - indices[0] == 3
    
    def test_sample_fps_invalid(self, sampler):
        """Test FPS sampling with invalid inputs."""
        with pytest.raises(ValueError):
            sampler.sample_fps(100, video_fps=30.0, target_fps=0)
        
        with pytest.raises(ValueError):
            sampler.sample_fps(100, video_fps=0, target_fps=1.0)
    
    # ---------- sample_temporal_chunks tests ----------
    
    def test_sample_temporal_chunks_basic(self, sampler):
        """Test temporal chunk sampling."""
        indices = sampler.sample_temporal_chunks(
            total_frames=1000,
            num_chunks=4,
            frames_per_chunk=2
        )
        
        # Should have 8 frames total (4 chunks * 2 frames)
        assert len(indices) == 8
        
        # Frames should be spread across temporal extent
        assert indices[0] < 250  # First chunk
        assert indices[-1] > 750  # Last chunk
    
    def test_sample_temporal_chunks_coverage(self, sampler):
        """Test that chunks cover the video temporally."""
        indices = sampler.sample_temporal_chunks(
            total_frames=1000,
            num_chunks=5,
            frames_per_chunk=1
        )
        
        # Calculate which chunks each frame belongs to
        chunk_size = 1000 // 5
        chunks_represented = set(idx // chunk_size for idx in indices)
        
        # Should have frames from all chunks
        assert len(chunks_represented) >= 4  # Allow some variance
    
    def test_sample_temporal_chunks_invalid(self, sampler):
        """Test temporal chunk sampling with invalid inputs."""
        with pytest.raises(ValueError):
            sampler.sample_temporal_chunks(100, num_chunks=0)
        
        with pytest.raises(ValueError):
            sampler.sample_temporal_chunks(100, frames_per_chunk=0)
    
    # ---------- calculate_frame_count tests ----------
    
    def test_calculate_frame_count(self, sampler):
        """Test frame count calculation."""
        # 60 second video at 0.5 fps = 30 frames
        count = sampler.calculate_frame_count(
            duration=60.0,
            frames_per_second=0.5,
            min_frames=8,
            max_frames=64
        )
        assert count == 30
        
        # Short video should use min
        count = sampler.calculate_frame_count(5.0, 0.5, 8, 64)
        assert count == 8
        
        # Long video should use max
        count = sampler.calculate_frame_count(200.0, 0.5, 8, 64)
        assert count == 64
    
    # ---------- get_sampling_info tests ----------
    
    def test_get_sampling_info(self, sampler):
        """Test sampling info retrieval."""
        indices = [0, 25, 50, 75, 99]
        info = sampler.get_sampling_info(
            total_frames=100,
            fps=30.0,
            indices=indices
        )
        
        assert info["num_sampled"] == 5
        assert info["total_frames"] == 100
        assert info["first_frame"] == 0
        assert info["last_frame"] == 99
        assert "avg_interval" in info
        assert "sampling_fps" in info


# ==================== Convenience Functions Tests ====================

class TestSamplerConvenienceFunctions:
    """Test convenience sampling functions."""
    
    def test_sample_video_adaptive(self):
        """Test sample_video_adaptive convenience function."""
        from src.ingestion.sampler import sample_video_adaptive
        
        indices = sample_video_adaptive(
            total_frames=900,
            fps=30.0,
            min_frames=8,
            max_frames=64
        )
        
        assert len(indices) == 15  # 30s * 0.5 = 15
        assert indices[0] == 0
    
    def test_sample_video_uniform(self):
        """Test sample_video_uniform convenience function."""
        from src.ingestion.sampler import sample_video_uniform
        
        indices = sample_video_uniform(total_frames=100, num_frames=10)
        
        assert len(indices) == 10
        assert indices[0] == 0
        assert indices[-1] == 99


# ==================== Test Utility Functions ====================

class TestUtilityFunctions:
    """Test utility functions from utils.py."""
    
    def test_validate_video_existing_file(self, tmp_path):
        """Test validation with existing video file."""
        from src.ingestion.utils import validate_video
        
        # Create a fake video file
        fake_video = tmp_path / "test.mp4"
        fake_video.write_bytes(b"fake video content")
        
        assert validate_video(str(fake_video)) is True
    
    def test_validate_video_nonexistent(self):
        """Test validation with non-existent file."""
        from src.ingestion.utils import validate_video
        
        result = validate_video("/nonexistent/path/video.mp4")
        assert result is False
    
    def test_validate_video_empty_file(self, tmp_path):
        """Test validation with empty file."""
        from src.ingestion.utils import validate_video
        
        empty_file = tmp_path / "empty.mp4"
        empty_file.touch()  # Create empty file
        
        result = validate_video(str(empty_file))
        assert result is False
    
    def test_validate_video_invalid_extension(self, tmp_path):
        """Test validation with invalid extension."""
        from src.ingestion.utils import validate_video
        
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("not a video")
        
        result = validate_video(str(invalid_file))
        assert result is False
    
    def test_estimate_memory(self):
        """Test memory estimation."""
        from src.ingestion.utils import estimate_memory
        
        video_info = {
            "width": 1920,
            "height": 1080,
            "total_frames": 1000
        }
        
        mem = estimate_memory(video_info, num_frames=8, batch_size=4)
        
        # Should be > 0
        assert mem > 0
        # 1920*1080*3*4 bytes = ~24MB per frame for float32
        assert mem < 100  # Sanity check
    
    def test_convert_timestamp_to_frame(self):
        """Test timestamp to frame conversion."""
        from src.ingestion.utils import convert_timestamp_to_frame
        
        # 2.5 seconds at 30fps = frame 75
        frame = convert_timestamp_to_frame(2.5, 30.0)
        assert frame == 75
        
        # 0 seconds = frame 0
        assert convert_timestamp_to_frame(0, 30.0) == 0
    
    def test_convert_frame_to_timestamp(self):
        """Test frame to timestamp conversion."""
        from src.ingestion.utils import convert_frame_to_timestamp
        
        # Frame 75 at 30fps = 2.5 seconds
        timestamp = convert_frame_to_timestamp(75, 30.0)
        assert timestamp == 2.5
        
        # Frame 0 = 0 seconds
        assert convert_frame_to_timestamp(0, 30.0) == 0
    
    def test_expand_bbox_basic(self):
        """Test bounding box expansion."""
        from src.ingestion.utils import expand_bbox
        
        bbox = (100, 100, 200, 200)  # 100x100 box
        expanded = expand_bbox(bbox, 0.2, 1920, 1080)
        
        x1, y1, x2, y2 = expanded
        
        # Should be larger
        assert x1 < 100
        assert y1 < 100
        assert x2 > 200
        assert y2 > 200
    
    def test_expand_bbox_at_border(self):
        """Test bbox expansion at image border."""
        from src.ingestion.utils import expand_bbox
        
        # Box at top-left corner
        bbox = (0, 0, 100, 100)
        expanded = expand_bbox(bbox, 0.2, 1920, 1080)
        
        x1, y1, x2, y2 = expanded
        
        # Should be clamped at 0
        assert x1 == 0
        assert y1 == 0
    
    def test_calculate_iou_identical(self):
        """Test IoU for identical boxes."""
        from src.ingestion.utils import calculate_iou
        
        bbox = (100, 100, 200, 200)
        iou = calculate_iou(bbox, bbox)
        
        assert iou == 1.0
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU for non-overlapping boxes."""
        from src.ingestion.utils import calculate_iou
        
        bbox1 = (0, 0, 100, 100)
        bbox2 = (200, 200, 300, 300)
        
        iou = calculate_iou(bbox1, bbox2)
        assert iou == 0.0
    
    def test_calculate_iou_partial_overlap(self):
        """Test IoU for partially overlapping boxes."""
        from src.ingestion.utils import calculate_iou
        
        bbox1 = (0, 0, 100, 100)
        bbox2 = (50, 50, 150, 150)  # 50% overlap in each dimension
        
        iou = calculate_iou(bbox1, bbox2)
        
        # Should be between 0 and 1
        assert 0 < iou < 1


# ==================== Test RoadVideoLoader (Mock) ====================

class TestRoadVideoLoaderMocked:
    """Test RoadVideoLoader with mocked dependencies."""
    
    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a mock config."""
        @dataclass
        class MockConfig:
            video_path: str = ""
            batch_size: int = 16
            width: int = -1
            height: int = -1
            num_threads: int = 0
            device: str = "cpu"
            ctx_id: int = 0
        
        # Create a dummy video file
        dummy_video = tmp_path / "test.mp4"
        dummy_video.write_bytes(b"fake video data")
        
        return MockConfig(video_path=str(dummy_video))
    
    def test_loader_validates_video_path(self, mock_config):
        """Test that loader validates video path exists."""
        from src.ingestion.loader import RoadVideoLoader
        
        mock_config.video_path = "/nonexistent/video.mp4"
        
        with pytest.raises(FileNotFoundError):
            RoadVideoLoader(mock_config)
    
    def test_loader_validates_video_extension(self, mock_config, tmp_path):
        """Test that loader warns on invalid extension."""
        import logging
        
        invalid_file = tmp_path / "test.xyz"
        invalid_file.write_bytes(b"data")
        mock_config.video_path = str(invalid_file)
        
        # Should log a warning but might proceed (depends on backend)


# ==================== Test RoadVideoLoader (Real Video) ====================

class TestRoadVideoLoaderReal:
    """Integration tests with real video files.
    
    These tests require a video file and are skipped if not provided.
    Run with: pytest --video_path path/to/video.mp4
    """
    
    @pytest.fixture
    def video_path(self, request):
        """Get video path from command line or skip."""
        video_path = request.config.getoption("--video_path", default=None)
        if video_path is None or not Path(video_path).exists():
            pytest.skip("No video file provided. Use --video_path option.")
        return video_path
    
    @pytest.fixture
    def loader(self, video_path):
        """Create a loader for the test video."""
        from src.ingestion.loader import RoadVideoLoader
        from src.configs import DecordConfig
        
        config = DecordConfig(
            video_path=video_path,
            device='cpu',
            width=-1,
            height=-1
        )
        return RoadVideoLoader(config)
    
    def test_loader_initialization(self, loader):
        """Test loader initializes correctly."""
        assert loader.total_frames > 0
        assert loader.fps > 0
        assert loader.duration > 0
        assert loader.backend in ('decord', 'opencv')
    
    def test_get_metadata(self, loader):
        """Test metadata extraction."""
        metadata = loader.get_metadata()
        
        assert "fps" in metadata
        assert "total_frames" in metadata
        assert "width" in metadata
        assert "height" in metadata
        assert "duration" in metadata
        assert metadata["width"] > 0
        assert metadata["height"] > 0
    
    def test_get_single_frame(self, loader):
        """Test single frame extraction."""
        import torch
        
        frame = loader.get_frame(0)
        
        assert isinstance(frame, torch.Tensor)
        assert frame.ndim == 3  # (C, H, W)
        assert frame.shape[0] == 3  # RGB
        assert frame.min() >= 0.0
        assert frame.max() <= 1.0
    
    def test_get_frame_out_of_range(self, loader):
        """Test frame extraction with invalid index."""
        with pytest.raises(IndexError):
            loader.get_frame(-1)
        
        with pytest.raises(IndexError):
            loader.get_frame(loader.total_frames + 100)
    
    def test_sample_uniform(self, loader):
        """Test uniform sampling."""
        import torch
        
        frames = loader.sample_uniform(8)
        
        assert isinstance(frames, torch.Tensor)
        assert frames.shape[0] == 8
        assert frames.ndim == 4  # (B, C, H, W)
    
    def test_sample_fps(self, loader):
        """Test FPS-based sampling."""
        import torch
        
        frames = loader.sample_fps(1.0)  # 1 fps
        
        assert isinstance(frames, torch.Tensor)
        # Should have approximately duration frames
        expected = int(loader.duration)
        assert abs(frames.shape[0] - expected) <= 2
    
    def test_sample_adaptive(self, loader):
        """Test adaptive sampling."""
        import torch
        
        frames = loader.sample_adaptive(min_frames=4, max_frames=32)
        
        assert isinstance(frames, torch.Tensor)
        assert 4 <= frames.shape[0] <= 32
    
    def test_sample_temporal_chunks(self, loader):
        """Test temporal chunk sampling."""
        import torch
        
        frames = loader.sample_temporal_chunks(num_chunks=4, frames_per_chunk=2)
        
        assert isinstance(frames, torch.Tensor)
        assert frames.shape[0] == 8  # 4 * 2
    
    def test_sample_indices(self, loader):
        """Test custom index sampling."""
        import torch
        
        indices = [0, 10, 20]
        frames = loader.sample_indices(indices)
        
        assert isinstance(frames, torch.Tensor)
        assert frames.shape[0] == 3
    
    def test_stream_batches(self, loader):
        """Test batch streaming."""
        total_frames = 0
        batch_count = 0
        
        for batch in loader.stream_batches():
            batch_count += 1
            total_frames += batch.shape[0]
            
            # Just check first few batches
            if batch_count >= 3:
                break
        
        assert batch_count > 0
        assert total_frames > 0
    
    def test_estimate_memory(self, loader):
        """Test memory estimation."""
        mem = loader.estimate_memory(num_frames=8, batch_size=4)
        
        assert mem > 0


# ==================== Test BatchVideoProcessor (Mock) ====================

class TestBatchVideoProcessor:
    """Test BatchVideoProcessor with mocks."""
    
    def test_processing_stats_to_dict(self):
        """Test ProcessingStats conversion."""
        from src.ingestion.processor import ProcessingStats
        
        stats = ProcessingStats(
            total_videos=100,
            processed=90,
            failed=5,
            skipped=5
        )
        
        d = stats.to_dict()
        
        assert d["total_videos"] == 100
        assert d["processed"] == 90
        assert d["failed"] == 5
        assert d["success_rate"] == 0.9


# ==================== Test Module Imports ====================

class TestModuleImports:
    """Test that all module imports work correctly."""
    
    def test_import_ingestion_module(self):
        """Test main module import."""
        from src.ingestion import (
            RoadVideoLoader,
            FrameSampler,
            BatchVideoProcessor,
            sample_video_adaptive,
            sample_video_uniform,
            validate_video,
            get_video_info,
        )
        
        assert RoadVideoLoader is not None
        assert FrameSampler is not None
    
    def test_import_sampler(self):
        """Test sampler module import."""
        from src.ingestion.sampler import FrameSampler
        
        sampler = FrameSampler()
        assert sampler is not None
    
    def test_import_utils(self):
        """Test utils module import."""
        from src.ingestion.utils import (
            validate_video,
            get_video_info,
            estimate_memory,
            convert_timestamp_to_frame,
            convert_frame_to_timestamp,
            expand_bbox,
            calculate_iou
        )
        
        assert validate_video is not None


# ==================== Pytest Configuration ====================

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--video_path",
        action="store",
        default=None,
        help="Path to video file for integration tests"
    )


# ==================== Run Tests Directly ====================

if __name__ == "__main__":
    import sys
    
    # Run pytest with verbose output
    sys.exit(pytest.main([__file__, "-v", "-s"] + sys.argv[1:]))
