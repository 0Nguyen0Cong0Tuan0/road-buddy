"""
Standalone test script for video utilities module.

Usage:
    python test_utils.py <video_path>
    
Example:
    python test_utils.py data\\raw\\train\\videos\\0b990034_129_clip_007_0046_0055_N.mp4
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.utils import (
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


def test_utils(video_path: str):
    """Test utility functions with a real video file."""
    print("=" * 70)
    print("TESTING VIDEO UTILITIES")
    print("=" * 70)
    print(f"\nVideo path: {video_path}\n")
    
    # Test 1: Video validation
    print("Test 1: Video validation...")
    try:
        is_valid = validate_video(video_path)
        if is_valid:
            print(f"✓ SUCCESS - Video is valid")
        else:
            print(f"✗ FAILED - Video validation failed")
            return False
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 2: Get video info
    print("\nTest 2: Extracting video info...")
    try:
        info = get_video_info(video_path, use_ffprobe=False)  # Use OpenCV
        if info:
            print(f"✓ SUCCESS")
            print(f"  - Resolution: {info['width']}x{info['height']}")
            print(f"  - FPS: {info['fps']:.2f}")
            print(f"  - Duration: {info['duration']:.2f}s")
            print(f"  - Total frames: {info['total_frames']}")
            print(f"  - Size: {info['size_mb']:.2f} MB")
        else:
            print(f"✗ FAILED - Could not extract info")
            return False
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Store info for later tests
    video_info = info
    
    # Test 3: Memory estimation
    print("\nTest 3: Memory estimation...")
    try:
        mem_8_frames = estimate_memory(video_info, num_frames=8, batch_size=4)
        mem_all = estimate_memory(video_info, num_frames=None, batch_size=16)
        
        print(f"✓ SUCCESS")
        print(f"  - 8 frames (float32): {mem_8_frames:.2f} MB")
        print(f"  - All frames (float32): {mem_all:.2f} MB")
        
        # Test different dtypes
        mem_fp16 = estimate_memory(video_info, num_frames=8, dtype='float16')
        mem_uint8 = estimate_memory(video_info, num_frames=8, dtype='uint8')
        print(f"  - 8 frames (float16): {mem_fp16:.2f} MB")
        print(f"  - 8 frames (uint8): {mem_uint8:.2f} MB")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 4: Timestamp/Frame conversions
    print("\nTest 4: Timestamp/Frame conversions...")
    try:
        fps = video_info['fps']
        
        # Convert timestamp to frame
        timestamp = 2.5
        frame_idx = convert_timestamp_to_frame(timestamp, fps)
        
        # Convert back
        recovered_timestamp = convert_frame_to_timestamp(frame_idx, fps)
        
        print(f"✓ SUCCESS")
        print(f"  - {timestamp}s → frame {frame_idx}")
        print(f"  - frame {frame_idx} → {recovered_timestamp:.2f}s")
        print(f"  - Roundtrip error: {abs(timestamp - recovered_timestamp):.4f}s")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 5: Crop and resize (torch tensor)
    print("\nTest 5: Crop and resize (torch tensor)...")
    try:
        # Create sample frame
        frame = torch.rand(3, video_info['height'], video_info['width'])
        
        # Define bbox (center region)
        h, w = video_info['height'], video_info['width']
        bbox = (w//4, h//4, 3*w//4, 3*h//4)
        
        # Crop and resize
        cropped = crop_and_resize(frame, bbox, target_size=(224, 224))
        
        print(f"✓ SUCCESS")
        print(f"  - Original shape: {frame.shape}")
        print(f"  - Bbox: {bbox}")
        print(f"  - Cropped shape: {cropped.shape}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 6: Crop and resize (numpy array)
    print("\nTest 6: Crop and resize (numpy array)...")
    try:
        # Create sample frame (H, W, C)
        frame = np.random.rand(video_info['height'], video_info['width'], 3).astype(np.float32)
        
        bbox = (100, 100, 300, 300)
        cropped = crop_and_resize(frame, bbox, target_size=(224, 224))
        
        print(f"✓ SUCCESS")
        print(f"  - Original shape: {frame.shape}")
        print(f"  - Cropped shape: {cropped.shape}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 7: Expand bbox
    print("\nTest 7: Expand bounding box...")
    try:
        original_bbox = (100, 100, 200, 200)
        expanded = expand_bbox(
            original_bbox,
            expansion_ratio=0.2,
            image_width=video_info['width'],
            image_height=video_info['height']
        )
        
        print(f"✓ SUCCESS")
        print(f"  - Original bbox: {original_bbox}")
        print(f"  - Expanded bbox: {expanded}")
        print(f"  - Expansion: 20%")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 8: Calculate IoU
    print("\nTest 8: Calculate IoU...")
    try:
        bbox1 = (100, 100, 200, 200)
        bbox2 = (150, 150, 250, 250)  # Overlapping
        bbox3 = (300, 300, 400, 400)  # No overlap
        
        iou_overlap = calculate_iou(bbox1, bbox2)
        iou_no_overlap = calculate_iou(bbox1, bbox3)
        iou_same = calculate_iou(bbox1, bbox1)
        
        print(f"✓ SUCCESS")
        print(f"  - IoU (overlapping): {iou_overlap:.3f}")
        print(f"  - IoU (no overlap): {iou_no_overlap:.3f}")
        print(f"  - IoU (same bbox): {iou_same:.3f}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 9: Create thumbnail
    print("\nTest 9: Create video thumbnail...")
    try:
        output_dir = Path(__file__).parent / "temp"
        output_dir.mkdir(exist_ok=True)
        
        thumbnail_path = str(output_dir / "test_thumbnail.jpg")
        
        success = create_video_thumbnail(
            video_path,
            thumbnail_path,
            frame_idx=None,  # Middle frame
            size=(320, 180)
        )
        
        if success and Path(thumbnail_path).exists():
            print(f"✓ SUCCESS")
            print(f"  - Thumbnail saved: {thumbnail_path}")
            print(f"  - Size: {Path(thumbnail_path).stat().st_size / 1024:.2f} KB")
            
            # Cleanup
            Path(thumbnail_path).unlink()
            output_dir.rmdir()
        else:
            print(f"✗ FAILED - Could not create thumbnail")
            return False
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(description='Test video utilities with real video')
    parser.add_argument('video_path', type=str, help='Path to video file')
    
    args = parser.parse_args()
    
    # Validate video exists
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Run tests
    success = test_utils(str(video_path))
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
