"""
Standalone test script for video utilities module.

Usage:
    python tests\test_utils.py <video_path> [--output_dir OUTPUT_DIR] [--no_save]
    
Example:
    python tests\test_utils.py data\raw\train\videos\0.mp4
    python tests\test_utils.py data\raw\train\videos\0.mp4 --output_dir ./debug_output
    python tests\test_utils.py data\raw\train\videos\0.mp4 --no_save
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.utils import (
    validate_video,
    get_video_info,
    estimate_memory,
    convert_timestamp_to_frame,
    convert_frame_to_timestamp,
)


def test_utils(video_path: str, output_dir: Path, save_images: bool = True):
    """Test utility functions with a real video file."""
    print(f"\nVideo path: {video_path}")
    print(f"Output dir: {output_dir}")
    print(f"Save images: {save_images}\n")
    
    if save_images:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created: {output_dir}\n")
    
    video_name = Path(video_path).stem
    
    # Test 1: Video validation
    print("Test 1: Video validation...")
    try:
        is_valid = validate_video(video_path)
        if is_valid:
            print(f"SUCCESS - Video is valid")
        else:
            print(f"FAILED - Video validation failed")
            return False
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 2: Get video info
    print("\nTest 2: Extracting video info...")
    try:
        info = get_video_info(video_path, use_ffprobe=False)
        if info:
            print(f"SUCCESS")
            print(f"- Resolution: {info['width']}x{info['height']}")
            print(f"- FPS: {info['fps']:.2f}")
            print(f"- Duration: {info['duration']:.2f}s")
            print(f"- Total frames: {info['total_frames']}")
            print(f"- Size: {info['size_mb']:.2f} MB")
        else:
            print(f"FAILED - Could not extract info")
            return False
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    video_info = info
    
    # Test 3: Memory estimation
    print("\nTest 3: Memory estimation...")
    try:
        mem_8_frames = estimate_memory(video_info, num_frames=8, batch_size=4)
        mem_all = estimate_memory(video_info, num_frames=None, batch_size=16)
        
        print(f"SUCCESS")
        print(f"- 8 frames (float32): {mem_8_frames:.2f} MB")
        print(f"- All frames (float32): {mem_all:.2f} MB")
        
        mem_fp16 = estimate_memory(video_info, num_frames=8, dtype='float16')
        mem_uint8 = estimate_memory(video_info, num_frames=8, dtype='uint8')
        print(f"- 8 frames (float16): {mem_fp16:.2f} MB")
        print(f"- 8 frames (uint8): {mem_uint8:.2f} MB")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 4: Timestamp/Frame conversions
    print("\nTest 4: Timestamp/Frame conversions...")
    try:
        fps = video_info['fps']
        
        timestamp = 2.5
        frame_idx = convert_timestamp_to_frame(timestamp, fps)
        recovered_timestamp = convert_frame_to_timestamp(frame_idx, fps)
        
        print(f"SUCCESS")
        print(f"- {timestamp}s -> frame {frame_idx}")
        print(f"- frame {frame_idx} -> {recovered_timestamp:.2f}s")
        print(f"- Roundtrip error: {abs(timestamp - recovered_timestamp):.4f}s")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    if save_images:
        print(f"\nAll saved images are in: {output_dir.absolute()}")
        saved_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
        print(f"- {len(saved_files)} image files")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test video utilities with real video')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output images')
    parser.add_argument('--no_save', action='store_true',
                        help='Disable saving images')
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {args.video_path}")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "debug_output" / f"{video_path.stem}_utils_{timestamp}"
    
    save_images = not args.no_save
    success = test_utils(str(video_path), output_dir, save_images)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()