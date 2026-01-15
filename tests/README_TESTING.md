# Testing Ingestion Module

This directory contains standalone test scripts for testing the ingestion module with real video files.

## Quick Start

### Test All Components (Recommended)
```bash
python tests/test_ingestion_all.py data\raw\train\videos\0b990034_129_clip_007_0046_0055_N.mp4
```

### Test Individual Components

**Test Video Loader:**
```bash
python tests/test_loader.py data\raw\train\videos\0b990034_129_clip_007_0046_0055_N.mp4
```

**Test Utilities:**
```bash
python tests/test_utils.py data\raw\train\videos\0b990034_129_clip_007_0046_0055_N.mp4
```

## Test Scripts

### `test_ingestion_all.py` - Complete Pipeline Test
Tests the entire ingestion pipeline end-to-end:
- Video validation
- Metadata extraction
- Memory planning
- Video loading
- Frame sampling (uniform, FPS-based)
- Preprocessing
- Performance metrics

**Output includes:**
- Timing breakdown
- Memory usage
- Performance assessment
- Readiness for downstream modules

### `test_loader.py` - Video Loader Tests
Tests all video loader functionality:
- Initialization (Decord/OpenCV backends)
- Single frame extraction
- Batch streaming
- Uniform sampling
- FPS-based sampling
- Custom indices sampling
- Preprocessing
- Memory estimation

**9 comprehensive tests**

### `test_utils.py` - Utility Functions Tests
Tests all utility functions:
- Video validation
- Metadata extraction (ffprobe/OpenCV)
- Memory estimation (different dtypes)
- Timestamp/frame conversions
- Crop and resize (torch/numpy)
- Bounding box expansion
- IoU calculation
- Thumbnail creation

**9 comprehensive tests**

## Example Output

```
==================================================================
COMPLETE INGESTION PIPELINE TEST
==================================================================

Video path: data\raw\train\videos\0b990034_129_clip_007_0046_0055_N.mp4

PHASE 1: VIDEO VALIDATION
------------------------------------------------------------------
✓ Video is valid (0.002s)

PHASE 2: METADATA EXTRACTION
------------------------------------------------------------------
✓ Metadata extracted (0.125s)
  - Resolution: 1920x1080
  - FPS: 30.00
  - Duration: 9.00s
  - Total frames: 270

PHASE 3: MEMORY PLANNING
------------------------------------------------------------------
✓ Memory estimation complete
  - 8 frames (batch=4): 47.81 MB

PHASE 4: VIDEO LOADING
------------------------------------------------------------------
✓ Video loaded (0.342s)
  - Backend: decord
  - Frames: 270

PHASE 5: FRAME SAMPLING
------------------------------------------------------------------
✓ Uniform sampling (8 frames): 0.089s
  - Shape: torch.Size([8, 3, 1080, 1920])
✓ FPS sampling (1 FPS): 0.067s
  - Frames extracted: 9

PHASE 6: PREPROCESSING
------------------------------------------------------------------
✓ Preprocessing complete: 0.123s
  - Input shape: torch.Size([8, 3, 1080, 1920])
  - Output shape: torch.Size([8, 3, 640, 640])
  - Ready for YOLO: Yes

==================================================================
PIPELINE SUMMARY
==================================================================

Total time: 0.748s
  - Loading: 0.342s (45.7%)
  - Sampling: 0.156s (20.9%)
  - Preprocessing: 0.123s (16.4%)

Performance:
  ✓ EXCELLENT - Well under 2s budget

Memory:
  - Peak estimated: 47.81 MB
  ✓ GOOD - Low memory footprint

Readiness:
  ✓ Ready for perception module (YOLO)
  ✓ Ready for VLM processing

==================================================================
ALL TESTS PASSED ✓
==================================================================
```

## Requirements

Make sure these are installed:
```bash
pip install torch numpy opencv-python decord
```

## Troubleshooting

**"Module not found" error:**
- Make sure you run from the project root directory
- The scripts automatically add `src/` to Python path

**"Decord not available" warning:**
- Tests will automatically fall back to OpenCV
- Install decord for GPU-accelerated loading: `pip install decord`

**CUDA errors:**
- Tests default to CPU mode
- To test GPU, edit the test files and change `device='cpu'` to `device='gpu'`
