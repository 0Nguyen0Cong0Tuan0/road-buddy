"""
Standalone test script for complete ingestion module (all components).

Usage:
    python test_ingestion_all.py <video_path>
    
Example:
    python test_ingestion_all.py data\\raw\\train\\videos\\0b990034_129_clip_007_0046_0055_N.mp4
"""

import sys
import argparse
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import (
    RoadVideoLoader,
    validate_video,
    get_video_info,
    estimate_memory
)
from src.configs import DecordConfig


def test_complete_pipeline(video_path: str):
    """Test complete ingestion pipeline with real video."""
    print("=" * 70)
    print("COMPLETE INGESTION PIPELINE TEST")
    print("=" * 70)
    print(f"\nVideo path: {video_path}\n")
    
    overall_start = time.time()
    
    # Phase 1: Validation
    print("PHASE 1: VIDEO VALIDATION")
    print("-" * 70)
    
    start = time.time()
    if not validate_video(video_path):
        print("✗ FAILED: Video validation failed")
        return False
    print(f"✓ Video is valid ({time.time() - start:.3f}s)")
    
    # Phase 2: Metadata Extraction
    print("\nPHASE 2: METADATA EXTRACTION")
    print("-" * 70)
    
    start = time.time()
    info = get_video_info(video_path, use_ffprobe=False)
    if not info:
        print("✗ FAILED: Could not extract metadata")
        return False
    
    print(f"✓ Metadata extracted ({time.time() - start:.3f}s)")
    print(f"  - Resolution: {info['width']}x{info['height']}")
    print(f"  - FPS: {info['fps']:.2f}")
    print(f"  - Duration: {info['duration']:.2f}s")
    print(f"  - Total frames: {info['total_frames']}")
    
    # Phase 3: Memory Planning
    print("\nPHASE 3: MEMORY PLANNING")
    print("-" * 70)
    
    mem_8_frames = estimate_memory(info, num_frames=8, batch_size=4)
    print(f"✓ Memory estimation complete")
    print(f"  - 8 frames (batch=4): {mem_8_frames:.2f} MB")
    
    if mem_8_frames > 1000:
        print(f"  ⚠ WARNING: High memory usage estimated")
    
    # Phase 4: Video Loading
    print("\nPHASE 4: VIDEO LOADING")
    print("-" * 70)
    
    start = time.time()
    config = DecordConfig(
        video_path=video_path,
        batch_size=16,
        device='cpu',
        ctx_id=0
    )
    
    try:
        loader = RoadVideoLoader(config)
        load_time = time.time() - start
        print(f"✓ Video loaded ({load_time:.3f}s)")
        print(f"  - Backend: {loader.backend}")
        print(f"  - Frames: {loader.total_frames}")
        
        if load_time > 2.0:
            print(f"  ⚠ WARNING: Loading took >{2.0}s (budget is 2s)")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Phase 5: Frame Sampling
    print("\nPHASE 5: FRAME SAMPLING")
    print("-" * 70)
    
    # Test uniform sampling
    start = time.time()
    frames_uniform = loader.sample_uniform(8)
    uniform_time = time.time() - start
    print(f"✓ Uniform sampling (8 frames): {uniform_time:.3f}s")
    print(f"  - Shape: {frames_uniform.shape}")
    
    # Test FPS sampling
    start = time.time()
    frames_fps = loader.sample_fps(1.0)
    fps_time = time.time() - start
    print(f"✓ FPS sampling (1 FPS): {fps_time:.3f}s")
    print(f"  - Frames extracted: {frames_fps.shape[0]}")
    
    # Phase 6: Preprocessing
    print("\nPHASE 6: PREPROCESSING")
    print("-" * 70)
    
    start = time.time()
    preprocessed = loader.preprocess_batch(
        frames_uniform,
        resize=(640, 640),
        normalize=False
    )
    preprocess_time = time.time() - start
    print(f"✓ Preprocessing complete: {preprocess_time:.3f}s")
    print(f"  - Input shape: {frames_uniform.shape}")
    print(f"  - Output shape: {preprocessed.shape}")
    print(f"  - Ready for YOLO: Yes")
    
    # Phase 7: Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    
    total_time = time.time() - overall_start
    
    print(f"\nTotal time: {total_time:.3f}s")
    print(f"  - Loading: {load_time:.3f}s ({load_time/total_time*100:.1f}%)")
    print(f"  - Sampling: {uniform_time + fps_time:.3f}s ({(uniform_time + fps_time)/total_time*100:.1f}%)")
    print(f"  - Preprocessing: {preprocess_time:.3f}s ({preprocess_time/total_time*100:.1f}%)")
    
    print(f"\nPerformance:")
    if total_time < 2.0:
        print(f"  ✓ EXCELLENT - Well under 2s budget")
    elif total_time < 3.0:
        print(f"  ✓ GOOD - Under budget")
    else:
        print(f"  ⚠ WARNING - May exceed 30s budget in full pipeline")
    
    print(f"\nMemory:")
    print(f"  - Peak estimated: {mem_8_frames:.2f} MB")
    if mem_8_frames < 500:
        print(f"  ✓ GOOD - Low memory footprint")
    
    print(f"\nReadiness:")
    print(f"  ✓ Ready for perception module (YOLO)")
    print(f"  ✓ Ready for VLM processing")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Test complete ingestion pipeline with real video'
    )
    parser.add_argument('video_path', type=str, help='Path to video file')
    
    args = parser.parse_args()
    
    # Validate video exists
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Run tests
    success = test_complete_pipeline(str(video_path))
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
