"""
Comprehensive CLI Test Script for Video Ingestion Module.

This script tests all components of the ingestion module:
- RoadVideoLoader: Video loading and frame extraction
- FrameSampler: Sampling strategies  
- Utils: Utility functions
- BatchVideoProcessor: Batch processing

Usage:
    # Basic test with video
    python tests/test_ingestion.py path/to/video.mp4
    
    # With custom output directory
    python tests/test_ingestion.py video.mp4 --output_dir ./test_output
    
    # Test specific components
    python tests/test_ingestion.py video.mp4 --test loader
    python tests/test_ingestion.py video.mp4 --test sampler
    python tests/test_ingestion.py video.mp4 --test utils
    python tests/test_ingestion.py video.mp4 --test all
    
    # Don't save output images
    python tests/test_ingestion.py video.mp4 --no_save
    
    # Custom sampling parameters
    python tests/test_ingestion.py video.mp4 --min_frames 4 --max_frames 32
    
    # Verbose output
    python tests/test_ingestion.py video.mp4 -v
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional: for saving images
try:
    import torchvision
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_subheader(text: str):
    """Print a subsection header."""
    print(f"\n{Colors.CYAN}--- {text} ---{Colors.END}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}[PASS]{Colors.END} {text}")


def print_fail(text: str):
    """Print failure message."""
    print(f"{Colors.RED}[FAIL]{Colors.END} {text}")



def print_info(text: str):
    """Print info message."""
    print(f"{Colors.YELLOW}  INFO:{Colors.END} {text}")


def print_result(name: str, value: Any):
    """Print a name-value result."""
    print(f"  {name}: {value}")


class IngestionTester:
    """Comprehensive tester for ingestion module."""
    
    def __init__(
        self,
        video_path: str,
        output_dir: Optional[Path] = None,
        save_images: bool = True,
        min_frames: int = 8,
        max_frames: int = 64,
        verbose: bool = False
    ):
        self.video_path = Path(video_path)
        self.output_dir = output_dir
        self.save_images = save_images and TORCHVISION_AVAILABLE
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.verbose = verbose
        
        self.results: Dict[str, bool] = {}
        self.errors: List[str] = []
        
        if self.save_images and output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_all_tests(self) -> bool:
        """Run all test suites."""
        print_header("INGESTION MODULE TEST SUITE")
        print(f"Video: {self.video_path}")
        print(f"Output: {self.output_dir or 'None (not saving)'}")
        print(f"Sampling: min={self.min_frames}, max={self.max_frames}")
        
        start_time = time.time()
        
        # Run test suites
        self.test_utils()
        self.test_sampler()
        self.test_loader()
        
        elapsed = time.time() - start_time
        
        # Print summary
        self._print_summary(elapsed)
        
        return all(self.results.values())
    
    def test_utils(self) -> bool:
        """Test utility functions."""
        print_header("TESTING: Utility Functions (utils.py)")
        
        from src.ingestion.utils import (
            validate_video,
            get_video_info,
            estimate_memory,
            convert_timestamp_to_frame,
            convert_frame_to_timestamp,
            expand_bbox,
            calculate_iou
        )
        
        all_passed = True
        
        # Test 1: Video validation
        print_subheader("Test 1: validate_video()")
        try:
            result = validate_video(str(self.video_path))
            if result:
                print_success(f"Video validated: {self.video_path.name}")
                self.results['utils.validate_video'] = True
            else:
                print_fail(f"Video validation failed")
                self.results['utils.validate_video'] = False
                all_passed = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['utils.validate_video'] = False
            all_passed = False
        
        # Test 2: Get video info
        print_subheader("Test 2: get_video_info()")
        try:
            info = get_video_info(str(self.video_path), use_ffprobe=False)
            if info:
                print_success("Video info extracted")
                print_result("Resolution", f"{info['width']}x{info['height']}")
                print_result("FPS", f"{info['fps']:.2f}")
                print_result("Duration", f"{info['duration']:.2f}s")
                print_result("Total frames", info['total_frames'])
                print_result("Size", f"{info['size_mb']:.2f} MB")
                self.results['utils.get_video_info'] = True
                self.video_info = info
            else:
                print_fail("Could not extract video info")
                self.results['utils.get_video_info'] = False
                all_passed = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['utils.get_video_info'] = False
            all_passed = False
        
        # Test 3: Memory estimation
        print_subheader("Test 3: estimate_memory()")
        try:
            if hasattr(self, 'video_info'):
                mem_8 = estimate_memory(self.video_info, num_frames=8, batch_size=4)
                mem_32 = estimate_memory(self.video_info, num_frames=32, batch_size=8)
                print_success("Memory estimation working")
                print_result("8 frames (float32)", f"{mem_8:.2f} MB")
                print_result("32 frames (float32)", f"{mem_32:.2f} MB")
                self.results['utils.estimate_memory'] = True
            else:
                print_fail("Skipped - no video info")
                self.results['utils.estimate_memory'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['utils.estimate_memory'] = False
            all_passed = False
        
        # Test 4: Timestamp/frame conversion
        print_subheader("Test 4: Timestamp <-> Frame conversion")
        try:
            fps = self.video_info.get('fps', 30.0) if hasattr(self, 'video_info') else 30.0
            
            # Test roundtrip
            timestamp = 2.5
            frame = convert_timestamp_to_frame(timestamp, fps)
            recovered = convert_frame_to_timestamp(frame, fps)
            
            error = abs(timestamp - recovered)
            if error < 0.1:
                print_success("Conversion roundtrip OK")
                print_result("Timestamp -> Frame", f"{timestamp}s -> frame {frame}")
                print_result("Frame -> Timestamp", f"frame {frame} -> {recovered:.3f}s")
                print_result("Roundtrip error", f"{error:.4f}s")
                self.results['utils.timestamp_conversion'] = True
            else:
                print_fail(f"Roundtrip error too high: {error}")
                self.results['utils.timestamp_conversion'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['utils.timestamp_conversion'] = False
            all_passed = False
        
        # Test 5: Bounding box operations
        print_subheader("Test 5: Bounding box operations")
        try:
            # Test expand_bbox
            bbox = (100, 100, 200, 200)
            expanded = expand_bbox(bbox, 0.2, 1920, 1080)
            
            # Test calculate_iou
            iou_same = calculate_iou(bbox, bbox)
            iou_overlap = calculate_iou((0, 0, 100, 100), (50, 50, 150, 150))
            iou_none = calculate_iou((0, 0, 50, 50), (200, 200, 250, 250))
            
            if iou_same == 1.0 and iou_none == 0.0 and 0 < iou_overlap < 1:
                print_success("Bounding box operations OK")
                print_result("Expanded bbox", f"{bbox} -> {expanded}")
                print_result("IoU (identical)", f"{iou_same:.3f}")
                print_result("IoU (overlap)", f"{iou_overlap:.3f}")
                print_result("IoU (no overlap)", f"{iou_none:.3f}")
                self.results['utils.bbox_operations'] = True
            else:
                print_fail("IoU calculation incorrect")
                self.results['utils.bbox_operations'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['utils.bbox_operations'] = False
            all_passed = False
        
        return all_passed
    
    def test_sampler(self) -> bool:
        """Test FrameSampler."""
        print_header("TESTING: FrameSampler (sampler.py)")
        
        from src.ingestion.sampler import (
            FrameSampler,
            sample_video_adaptive,
            sample_video_uniform
        )
        
        all_passed = True
        sampler = FrameSampler(seed=42)
        
        # Get video info for testing
        fps = self.video_info.get('fps', 30.0) if hasattr(self, 'video_info') else 30.0
        total_frames = self.video_info.get('total_frames', 900) if hasattr(self, 'video_info') else 900
        
        # Test 1: Uniform sampling
        print_subheader("Test 1: sample_uniform()")
        try:
            indices = sampler.sample_uniform(total_frames=total_frames, num_frames=10)
            
            if len(indices) == 10 and indices[0] == 0 and indices[-1] == total_frames - 1:
                print_success("Uniform sampling OK")
                print_result("Requested", 10)
                print_result("Returned", len(indices))
                print_result("First/Last", f"{indices[0]} / {indices[-1]}")
                if self.verbose:
                    print_result("All indices", indices)
                self.results['sampler.uniform'] = True
            else:
                print_fail("Uniform sampling incorrect")
                self.results['sampler.uniform'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['sampler.uniform'] = False
            all_passed = False
        
        # Test 2: Adaptive sampling
        print_subheader("Test 2: sample_adaptive()")
        try:
            indices = sampler.sample_adaptive(
                total_frames=total_frames,
                fps=fps,
                min_frames=self.min_frames,
                max_frames=self.max_frames,
                frames_per_second=0.5
            )
            
            duration = total_frames / fps
            expected = int(duration * 0.5)
            expected = max(self.min_frames, min(expected, self.max_frames))
            
            if len(indices) == expected:
                print_success("Adaptive sampling OK")
                print_result("Duration", f"{duration:.1f}s")
                print_result("Expected frames", expected)
                print_result("Actual frames", len(indices))
                self.results['sampler.adaptive'] = True
            else:
                print_fail(f"Expected {expected} frames, got {len(indices)}")
                self.results['sampler.adaptive'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['sampler.adaptive'] = False
            all_passed = False
        
        # Test 3: FPS sampling
        print_subheader("Test 3: sample_fps()")
        try:
            target_fps = 1.0
            indices = sampler.sample_fps(
                total_frames=total_frames,
                video_fps=fps,
                target_fps=target_fps
            )
            
            expected = int(total_frames / fps)  # Duration in seconds
            tolerance = 2
            
            if abs(len(indices) - expected) <= tolerance:
                print_success("FPS sampling OK")
                print_result("Target FPS", target_fps)
                print_result("Expected frames", f"~{expected}")
                print_result("Actual frames", len(indices))
                self.results['sampler.fps'] = True
            else:
                print_fail(f"Expected ~{expected} frames, got {len(indices)}")
                self.results['sampler.fps'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['sampler.fps'] = False
            all_passed = False
        
        # Test 4: Temporal chunks
        print_subheader("Test 4: sample_temporal_chunks()")
        try:
            num_chunks = 4
            frames_per_chunk = 2
            indices = sampler.sample_temporal_chunks(
                total_frames=total_frames,
                num_chunks=num_chunks,
                frames_per_chunk=frames_per_chunk
            )
            
            expected = num_chunks * frames_per_chunk
            
            if len(indices) == expected:
                print_success("Temporal chunk sampling OK")
                print_result("Chunks", num_chunks)
                print_result("Frames/chunk", frames_per_chunk)
                print_result("Total frames", len(indices))
                print_result("Coverage", f"frames {indices[0]} to {indices[-1]}")
                self.results['sampler.temporal_chunks'] = True
            else:
                print_fail(f"Expected {expected} frames, got {len(indices)}")
                self.results['sampler.temporal_chunks'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['sampler.temporal_chunks'] = False
            all_passed = False
        
        # Test 5: Convenience functions
        print_subheader("Test 5: Convenience functions")
        try:
            indices_adaptive = sample_video_adaptive(total_frames, fps)
            indices_uniform = sample_video_uniform(total_frames, 10)
            
            if len(indices_uniform) == 10 and len(indices_adaptive) >= self.min_frames:
                print_success("Convenience functions OK")
                print_result("sample_video_adaptive", f"{len(indices_adaptive)} frames")
                print_result("sample_video_uniform", f"{len(indices_uniform)} frames")
                self.results['sampler.convenience'] = True
            else:
                print_fail("Convenience function output incorrect")
                self.results['sampler.convenience'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['sampler.convenience'] = False
            all_passed = False
        
        return all_passed
    
    def test_loader(self) -> bool:
        """Test RoadVideoLoader."""
        print_header("TESTING: RoadVideoLoader (loader.py)")
        
        from src.ingestion.loader import RoadVideoLoader
        from src.configs import DecordConfig
        
        all_passed = True
        
        # Test 1: Initialization
        print_subheader("Test 1: Loader initialization")
        try:
            config = DecordConfig(
                video_path=str(self.video_path),
                batch_size=16,
                device='cpu',
                width=-1,
                height=-1
            )
            loader = RoadVideoLoader(config)
            
            print_success(f"Loader initialized with {loader.backend} backend")
            print_result("Total frames", loader.total_frames)
            print_result("FPS", f"{loader.fps:.2f}")
            print_result("Duration", f"{loader.duration:.2f}s")
            self.results['loader.init'] = True
            self.loader = loader
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['loader.init'] = False
            return False
        
        # Test 2: Metadata
        print_subheader("Test 2: get_metadata()")
        try:
            metadata = loader.get_metadata()
            
            required_keys = ['fps', 'total_frames', 'width', 'height', 'duration']
            if all(k in metadata for k in required_keys):
                print_success("Metadata extraction OK")
                print_result("Resolution", f"{metadata['width']}x{metadata['height']}")
                print_result("Backend", metadata['backend'])
                self.results['loader.metadata'] = True
            else:
                print_fail(f"Missing keys in metadata")
                self.results['loader.metadata'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['loader.metadata'] = False
            all_passed = False
        
        # Test 3: Single frame extraction
        print_subheader("Test 3: get_frame()")
        try:
            import torch
            frame = loader.get_frame(0)
            
            if (isinstance(frame, torch.Tensor) and 
                frame.ndim == 3 and 
                frame.shape[0] == 3 and
                frame.min() >= 0 and frame.max() <= 1):
                print_success("Single frame extraction OK")
                print_result("Shape", f"{tuple(frame.shape)}")
                print_result("Dtype", str(frame.dtype))
                print_result("Range", f"[{frame.min():.3f}, {frame.max():.3f}]")
                self.results['loader.get_frame'] = True
                
                if self.save_images and self.output_dir:
                    path = self.output_dir / "test_single_frame.png"
                    torchvision.utils.save_image(frame, str(path))
                    print_info(f"Saved: {path}")
            else:
                print_fail("Invalid frame format")
                self.results['loader.get_frame'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['loader.get_frame'] = False
            all_passed = False
        
        # Test 4: Uniform sampling
        print_subheader("Test 4: sample_uniform()")
        try:
            frames = loader.sample_uniform(8)
            
            if frames.shape[0] == 8 and frames.ndim == 4:
                print_success("Uniform sampling OK")
                print_result("Shape", f"{tuple(frames.shape)}")
                print_result("Memory", f"{frames.numel() * 4 / 1024**2:.2f} MB")
                self.results['loader.sample_uniform'] = True
                
                if self.save_images and self.output_dir:
                    path = self.output_dir / "test_uniform_sampling.png"
                    torchvision.utils.save_image(frames, str(path), nrow=4)
                    print_info(f"Saved: {path}")
            else:
                print_fail("Invalid output shape")
                self.results['loader.sample_uniform'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['loader.sample_uniform'] = False
            all_passed = False
        
        # Test 5: Adaptive sampling
        print_subheader("Test 5: sample_adaptive()")
        try:
            frames = loader.sample_adaptive(
                min_frames=self.min_frames,
                max_frames=self.max_frames
            )
            
            if self.min_frames <= frames.shape[0] <= self.max_frames:
                print_success("Adaptive sampling OK")
                print_result("Frames extracted", frames.shape[0])
                print_result("Shape", f"{tuple(frames.shape)}")
                self.results['loader.sample_adaptive'] = True
                
                if self.save_images and self.output_dir:
                    path = self.output_dir / "test_adaptive_sampling.png"
                    nrow = min(8, frames.shape[0])
                    torchvision.utils.save_image(frames, str(path), nrow=nrow)
                    print_info(f"Saved: {path}")
            else:
                print_fail(f"Frame count {frames.shape[0]} outside bounds [{self.min_frames}, {self.max_frames}]")
                self.results['loader.sample_adaptive'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['loader.sample_adaptive'] = False
            all_passed = False
        
        # Test 6: FPS sampling
        print_subheader("Test 6: sample_fps()")
        try:
            frames = loader.sample_fps(1.0)  # 1 FPS
            
            expected = int(loader.duration)
            if abs(frames.shape[0] - expected) <= 2:
                print_success("FPS sampling OK")
                print_result("Target FPS", 1.0)
                print_result("Frames extracted", frames.shape[0])
                print_result("Expected", f"~{expected}")
                self.results['loader.sample_fps'] = True
            else:
                print_fail(f"Expected ~{expected} frames, got {frames.shape[0]}")
                self.results['loader.sample_fps'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['loader.sample_fps'] = False
            all_passed = False
        
        # Test 7: Temporal chunks
        print_subheader("Test 7: sample_temporal_chunks()")
        try:
            frames = loader.sample_temporal_chunks(num_chunks=4, frames_per_chunk=2)
            
            if frames.shape[0] == 8:
                print_success("Temporal chunk sampling OK")
                print_result("Chunks", 4)
                print_result("Frames/chunk", 2)
                print_result("Total frames", frames.shape[0])
                self.results['loader.sample_temporal_chunks'] = True
                
                if self.save_images and self.output_dir:
                    path = self.output_dir / "test_temporal_chunks.png"
                    torchvision.utils.save_image(frames, str(path), nrow=4)
                    print_info(f"Saved: {path}")
            else:
                print_fail(f"Expected 8 frames, got {frames.shape[0]}")
                self.results['loader.sample_temporal_chunks'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['loader.sample_temporal_chunks'] = False
            all_passed = False
        
        # Test 8: Custom indices
        print_subheader("Test 8: sample_indices()")
        try:
            indices = [0, loader.total_frames // 2, loader.total_frames - 1]
            frames = loader.sample_indices(indices)
            
            if frames.shape[0] == 3:
                print_success("Custom index sampling OK")
                print_result("Indices", indices)
                print_result("Frames extracted", frames.shape[0])
                self.results['loader.sample_indices'] = True
            else:
                print_fail(f"Expected 3 frames, got {frames.shape[0]}")
                self.results['loader.sample_indices'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['loader.sample_indices'] = False
            all_passed = False
        
        # Test 9: Batch streaming
        print_subheader("Test 9: stream_batches()")
        try:
            batch_count = 0
            total_frames = 0
            
            for batch in loader.stream_batches():
                batch_count += 1
                total_frames += batch.shape[0]
                if batch_count >= 3:  # Only test first 3 batches
                    break
            
            if batch_count == 3 and total_frames > 0:
                print_success("Batch streaming OK")
                print_result("Batches tested", batch_count)
                print_result("Frames in 3 batches", total_frames)
                self.results['loader.stream_batches'] = True
            else:
                print_fail("Streaming failed")
                self.results['loader.stream_batches'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['loader.stream_batches'] = False
            all_passed = False
        
        # Test 10: Memory estimation
        print_subheader("Test 10: estimate_memory()")
        try:
            mem = loader.estimate_memory(num_frames=8, batch_size=4)
            
            if mem > 0:
                print_success("Memory estimation OK")
                print_result("8 frames (batch=4)", f"{mem:.2f} MB")
                self.results['loader.estimate_memory'] = True
            else:
                print_fail("Memory estimation returned 0")
                self.results['loader.estimate_memory'] = False
        except Exception as e:
            print_fail(f"Exception: {e}")
            self.results['loader.estimate_memory'] = False
            all_passed = False
        
        return all_passed
    
    def _print_summary(self, elapsed: float):
        """Print test summary."""
        print_header("TEST SUMMARY")
        
        passed = sum(1 for v in self.results.values() if v)
        failed = sum(1 for v in self.results.values() if not v)
        total = len(self.results)
        
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Tests run: {total}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.END}")
        print(f"{Colors.RED}Failed: {failed}{Colors.END}")
        
        if failed > 0:
            print(f"\n{Colors.RED}Failed tests:{Colors.END}")
            for name, result in self.results.items():
                if not result:
                    print(f"  - {name}")
        
        if self.save_images and self.output_dir:
            saved = list(self.output_dir.glob("*.png"))
            print(f"\nSaved {len(saved)} images to: {self.output_dir}")
        
        print()
        if failed == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}All tests passed!{Colors.END}")
        else:
            print(f"{Colors.RED}{Colors.BOLD}{failed} test(s) failed.{Colors.END}")


def main():
    parser = argparse.ArgumentParser(
        description='Test video ingestion module',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_ingestion.py data/raw/train/videos/0.mp4
  python tests/test_ingestion.py video.mp4 --output_dir ./test_output
  python tests/test_ingestion.py video.mp4 --test loader --no_save
  python tests/test_ingestion.py video.mp4 --min_frames 4 --max_frames 32 -v
        """
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to video file for testing'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Directory to save output images (default: tests/test_output/<timestamp>)'
    )
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Disable saving output images'
    )
    parser.add_argument(
        '--test', '-t',
        type=str,
        choices=['all', 'loader', 'sampler', 'utils'],
        default='all',
        help='Which component to test (default: all)'
    )
    parser.add_argument(
        '--min_frames',
        type=int,
        default=8,
        help='Minimum frames for adaptive sampling (default: 8)'
    )
    parser.add_argument(
        '--max_frames',
        type=int,
        default=64,
        help='Maximum frames for adaptive sampling (default: 64)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate video path
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"{Colors.RED}ERROR: Video not found: {args.video_path}{Colors.END}")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif not args.no_save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "test_output" / f"{video_path.stem}_{timestamp}"
    else:
        output_dir = None
    
    # Create tester
    tester = IngestionTester(
        video_path=str(video_path),
        output_dir=output_dir,
        save_images=not args.no_save,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        verbose=args.verbose
    )
    
    # Run tests
    if args.test == 'all':
        success = tester.run_all_tests()
    elif args.test == 'utils':
        success = tester.test_utils()
    elif args.test == 'sampler':
        success = tester.test_sampler()
    elif args.test == 'loader':
        success = tester.test_loader()
    else:
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
