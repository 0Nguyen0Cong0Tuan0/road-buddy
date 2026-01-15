"""
Standalone test script for video loader module.

Usage:
    python test_loader.py <video_path>
    
Example:
    python test_loader.py data\\raw\\train\\videos\\0b990034_129_clip_007_0046_0055_N.mp4
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loader import RoadVideoLoader
from src.configs import DecordConfig


def test_loader(video_path: str):
    """Test video loader with a real video file."""
    print("=" * 70)
    print("TESTING VIDEO LOADER")
    print("=" * 70)
    print(f"\nVideo path: {video_path}\n")
    
    # Test 1: Initialization
    print("Test 1: Initializing video loader...")
    try:
        config = DecordConfig(
            video_path=video_path,
            batch_size=16,
            device='cpu',  # Change to 'gpu' if you have CUDA
            ctx_id=0,
            width=-1,
            height=-1,
            num_threads=0
        )
        loader = RoadVideoLoader(config)
        print(f"✓ SUCCESS - Backend: {loader.backend}")
        print(f"  - Total frames: {loader.total_frames}")
        print(f"  - FPS: {loader.fps:.2f}")
        print(f"  - Duration: {loader.duration:.2f}s")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 2: Get metadata
    print("\nTest 2: Extracting metadata...")
    try:
        metadata = loader.get_metadata()
        print(f"✓ SUCCESS")
        print(f"  - Resolution: {metadata['width']}x{metadata['height']}")
        print(f"  - Size: {metadata['size_mb']:.2f} MB")
        print(f"  - Backend: {metadata['backend']}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 3: Get single frame
    print("\nTest 3: Extracting single frame...")
    try:
        frame = loader.get_frame(0)
        print(f"✓ SUCCESS")
        print(f"  - Frame shape: {frame.shape}")
        print(f"  - Frame dtype: {frame.dtype}")
        print(f"  - Value range: [{frame.min():.3f}, {frame.max():.3f}]")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 4: Uniform sampling
    print("\nTest 4: Uniform sampling (8 frames)...")
    try:
        frames = loader.sample_uniform(8)
        print(f"✓ SUCCESS")
        print(f"  - Frames shape: {frames.shape}")
        print(f"  - Memory usage: ~{frames.numel() * 4 / (1024**2):.2f} MB")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 5: FPS sampling
    print("\nTest 5: FPS-based sampling (1 FPS)...")
    try:
        frames = loader.sample_fps(1.0)
        print(f"✓ SUCCESS")
        print(f"  - Frames extracted: {frames.shape[0]}")
        print(f"  - Expected frames: ~{int(loader.duration)}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 6: Custom indices sampling
    print("\nTest 6: Custom indices sampling...")
    try:
        max_idx = min(loader.total_frames - 1, 50)
        indices = [0, max_idx // 4, max_idx // 2, 3 * max_idx // 4, max_idx]
        frames = loader.sample_indices(indices)
        print(f"✓ SUCCESS")
        print(f"  - Indices: {indices}")
        print(f"  - Frames extracted: {frames.shape[0]}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 7: Preprocessing
    print("\nTest 7: Frame preprocessing...")
    try:
        frame = loader.get_frame(0)
        preprocessed = loader.preprocess_frame(
            frame,
            resize=(224, 224),
            normalize=True
        )
        print(f"✓ SUCCESS")
        print(f"  - Original shape: {frame.shape}")
        print(f"  - Preprocessed shape: {preprocessed.shape}")
        print(f"  - Normalized (not in [0,1]): {not (0 <= preprocessed.min() <= preprocessed.max() <= 1)}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 8: Batch streaming
    print("\nTest 8: Batch streaming...")
    try:
        batch_count = 0
        total_frames = 0
        for batch in loader.stream_batches():
            batch_count += 1
            total_frames += batch.shape[0]
            if batch_count == 1:  # Only show first batch details
                print(f"  - First batch shape: {batch.shape}")
        
        print(f"✓ SUCCESS")
        print(f"  - Total batches: {batch_count}")
        print(f"  - Total frames streamed: {total_frames}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 9: Memory estimation
    print("\nTest 9: Memory estimation...")
    try:
        mem_8_frames = loader.estimate_memory(8, batch_size=4)
        mem_all_frames = loader.estimate_memory(loader.total_frames, batch_size=16)
        print(f"✓ SUCCESS")
        print(f"  - 8 frames (batch=4): {mem_8_frames:.2f} MB")
        print(f"  - All frames (batch=16): {mem_all_frames:.2f} MB")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(description='Test video loader with real video')
    parser.add_argument('video_path', type=str, help='Path to video file')
    
    args = parser.parse_args()
    
    # Validate video exists
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Run tests
    success = test_loader(str(video_path))
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
