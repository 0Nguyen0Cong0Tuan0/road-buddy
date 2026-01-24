# **RoadBuddy Autonomous Assistant**

**Hybrid AI Architecture for Vietnamese Traffic Law Question Answering**

[![ython 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## **Overview**
RoadBuddy implements the **"Lenses & Law"** architecture. The system processes dashcam video (5-15 seconds) and answers legal questions about Vietnamese traffic regulations within a 30-second inference budget on a single RTX 3090/A30 GPU.

### **Key features**

- **Modular pipeline**: Decoupled Perception → Retrieval → Reasoning
- **High-performance video ingestion**: GPU-accelerated decoding via Decord
- **Smart keyframe selection**: Optical flow-based sampling
- **Legal RAG**: Hybrid search over Law 36/2024 and QCVN 41:2024
- **Cloud VLM**: Gemini Pro Vision API (gemini-2.0-flash)
- **Latency monitoring**: Real-time tracking of 30s constraint

## **Architecture**

```

Video Input → Decord Loader → YOLO11 Detection → Keyframe Selection
                                                         ↓
Legal Answer ← vLLM Reasoning ← Qdrant RAG ← Visual Summary
```

## **Quick start**

### **Prerequisites**

- NVIDIA GPU (RTX 3090 or A30 recommended)
- Docker with NVIDIA Container Toolkit
- 24GB+ VRAM
- CUDA 12.1+

### **Installation**

```bash
# Clone repository
git clone https://github.com/0Nguyen0Cong0Tuan0/roadbuddy-core.git
cd roadbuddy-core

# Build Docker image
docker build-t roadbuddy:v1 -f docker/Dockerfile .

# Start services
docker-compose -f docker/docker-compose.yml up -d
```

### **Running inference**

```bash
# Process test set
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    -e INPUT_CSV=/app/data/public_test.csv \
    -e OUTPUT_CSV=/app/outputs/submission.csv \
    roadbuddy:v1 \
```

## **Performance**

## **Testing**

```bash
# Run latency tests
python tests/test_latency.py

# Run evaluation
python scripts/evaluate.py
```

## **Documentation**

See `docs/readme.md` for detailed architecture documentation.

## **License**

MIT License - see LICENSE file for details.

## **Acknowledgments**

- Zalo AI Challenge 2025 organizers
- Qwen Team (Alibaba Cloud)
- Ultralytics YOLO
- vLLM Team