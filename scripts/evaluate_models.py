#!/usr/bin/env python3
"""
Model Comparison Evaluation Script

Compares different model variants used in the Road Buddy VQA pipeline:
- CLIP: ViT-B/32 vs ViT-L/14
- VLM: qwen2.5-vl-7b vs qwen2.5-vl-7b-awq
- VLM Backend: transformers vs vllm

Usage:
    python scripts/evaluate_models.py --comparison clip --samples 10
    python scripts/evaluate_models.py --comparison vlm --samples 10
    python scripts/evaluate_models.py --comparison all --samples 5
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.dataset_loader import load_dataset, RoadBuddyDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Result from a single model evaluation."""
    model_name: str
    accuracy: float
    avg_time_ms: float
    total_samples: int
    correct_count: int
    

@dataclass
class ComparisonResult:
    """Result from comparing two models."""
    comparison_type: str
    model_a: ModelResult
    model_b: ModelResult
    winner: str
    accuracy_diff: float
    speed_diff_pct: float
    timestamp: str


def evaluate_clip_model(
    clip_model: str,
    dataset: RoadBuddyDataset,
    max_samples: int = 10
) -> ModelResult:
    """Evaluate a CLIP model variant."""
    from src.perception.keyframe_selector import KeyframeSelector, KeyframeSelectorConfig
    from src.perception.frame_scorer import ScoringConfig
    
    # Create config with specific CLIP model
    scoring_config = ScoringConfig(
        strategy="clip",
        clip_model=clip_model,
        device="auto"
    )
    
    selector_config = KeyframeSelectorConfig(
        num_keyframes=8,
        scoring_config=scoring_config,
        yolo_mode="none"
    )
    
    selector = KeyframeSelector(selector_config)
    
    times = []
    samples = list(dataset)[:max_samples]
    
    logger.info(f"Evaluating CLIP model: {clip_model} on {len(samples)} samples")
    
    for sample in samples:
        start = time.time()
        try:
            result = selector.select(sample.video_path, sample.question)
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
    
    # Unload model
    selector.unload_models()
    
    avg_time = sum(times) / len(times) if times else 0
    
    return ModelResult(
        model_name=clip_model,
        accuracy=0.0,  # CLIP doesn't produce final answers, just keyframes
        avg_time_ms=avg_time,
        total_samples=len(samples),
        correct_count=0
    )


def evaluate_vlm_model(
    vlm_model: str,
    backend: str,
    dataset: RoadBuddyDataset,
    max_samples: int = 10
) -> ModelResult:
    """Evaluate a VLM model variant."""
    from src.perception.keyframe_selector import KeyframeSelector, KeyframeSelectorConfig
    from src.perception.frame_scorer import ScoringConfig
    from src.reasoning.vlm_client import create_vlm_client
    from src.evaluation.metrics import extract_answer_letter
    import numpy as np
    
    # Use smaller CLIP for VLM tests to save memory
    scoring_config = ScoringConfig(
        strategy="clip",
        clip_model="ViT-B/32",
        device="auto"
    )
    
    selector_config = KeyframeSelectorConfig(
        num_keyframes=8,
        scoring_config=scoring_config,
        yolo_mode="none"
    )
    
    selector = KeyframeSelector(selector_config)
    
    # Create VLM client
    vlm_client = create_vlm_client(
        model_name=vlm_model,
        backend=backend,
        max_tokens=256,
        temperature=0.1
    )
    
    # Load VLM model
    if hasattr(vlm_client, 'load_model'):
        vlm_client.load_model()
    
    times = []
    correct = 0
    samples = list(dataset)[:max_samples]
    
    logger.info(f"Evaluating VLM: {vlm_model} ({backend}) on {len(samples)} samples")
    
    for sample in samples:
        start = time.time()
        try:
            # Select keyframes
            kf_result = selector.select(sample.video_path, sample.question)
            
            # Build prompt
            choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(sample.choices)])
            prompt = f"Question: {sample.question}\n\nChoices:\n{choices_text}\n\nAnswer with only the letter (A, B, C, or D)."
            
            # Generate answer
            response = vlm_client.generate(kf_result.frames, prompt)
            predicted = extract_answer_letter(response.text)
            
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            
            if predicted == sample.answer:
                correct += 1
                
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
    
    # Unload models
    selector.unload_models()
    if hasattr(vlm_client, 'unload_model'):
        vlm_client.unload_model()
    
    avg_time = sum(times) / len(times) if times else 0
    accuracy = correct / len(samples) if samples else 0
    
    return ModelResult(
        model_name=f"{vlm_model}_{backend}",
        accuracy=accuracy,
        avg_time_ms=avg_time,
        total_samples=len(samples),
        correct_count=correct
    )


def compare_models(
    result_a: ModelResult,
    result_b: ModelResult,
    comparison_type: str
) -> ComparisonResult:
    """Compare two model results."""
    accuracy_diff = result_a.accuracy - result_b.accuracy
    
    if result_b.avg_time_ms > 0:
        speed_diff = ((result_a.avg_time_ms - result_b.avg_time_ms) / result_b.avg_time_ms) * 100
    else:
        speed_diff = 0
    
    # Determine winner (prefer accuracy, then speed)
    if result_a.accuracy > result_b.accuracy:
        winner = result_a.model_name
    elif result_b.accuracy > result_a.accuracy:
        winner = result_b.model_name
    elif result_a.avg_time_ms < result_b.avg_time_ms:
        winner = result_a.model_name
    else:
        winner = result_b.model_name
    
    return ComparisonResult(
        comparison_type=comparison_type,
        model_a=result_a,
        model_b=result_b,
        winner=winner,
        accuracy_diff=accuracy_diff,
        speed_diff_pct=speed_diff,
        timestamp=datetime.now().isoformat()
    )


def run_clip_comparison(dataset: RoadBuddyDataset, max_samples: int) -> ComparisonResult:
    """Compare CLIP models: ViT-B/32 vs ViT-L/14."""
    logger.info("=" * 60)
    logger.info("CLIP Model Comparison: ViT-B/32 vs ViT-L/14")
    logger.info("=" * 60)
    
    result_b32 = evaluate_clip_model("ViT-B/32", dataset, max_samples)
    result_l14 = evaluate_clip_model("ViT-L/14", dataset, max_samples)
    
    return compare_models(result_b32, result_l14, "clip")


def run_vlm_comparison(dataset: RoadBuddyDataset, max_samples: int) -> ComparisonResult:
    """Compare VLM models: qwen2.5-vl-7b vs qwen2.5-vl-7b-awq."""
    logger.info("=" * 60)
    logger.info("VLM Model Comparison: FP16 vs AWQ")
    logger.info("=" * 60)
    
    # Note: Running both might require significant GPU memory
    # Run AWQ first as it uses less memory
    result_awq = evaluate_vlm_model("qwen2.5-vl-7b-awq", "vllm", dataset, max_samples)
    result_fp16 = evaluate_vlm_model("qwen2.5-vl-7b", "vllm", dataset, max_samples)
    
    return compare_models(result_awq, result_fp16, "vlm")


def run_backend_comparison(dataset: RoadBuddyDataset, max_samples: int) -> ComparisonResult:
    """Compare backends: transformers vs vllm."""
    logger.info("=" * 60)
    logger.info("Backend Comparison: transformers vs vllm")
    logger.info("=" * 60)
    
    result_transformers = evaluate_vlm_model("qwen2.5-vl-7b-awq", "transformers", dataset, max_samples)
    result_vllm = evaluate_vlm_model("qwen2.5-vl-7b-awq", "vllm", dataset, max_samples)
    
    return compare_models(result_transformers, result_vllm, "backend")


def print_comparison_report(comparison: ComparisonResult):
    """Print a formatted comparison report."""
    print("\n" + "=" * 60)
    print(f"COMPARISON: {comparison.comparison_type.upper()}")
    print("=" * 60)
    
    print(f"\n{'Model':<30} {'Accuracy':<12} {'Avg Time (ms)':<15}")
    print("-" * 60)
    
    a = comparison.model_a
    b = comparison.model_b
    
    print(f"{a.model_name:<30} {a.accuracy*100:>6.1f}%      {a.avg_time_ms:>10.1f}")
    print(f"{b.model_name:<30} {b.accuracy*100:>6.1f}%      {b.avg_time_ms:>10.1f}")
    
    print("-" * 60)
    print(f"Winner: {comparison.winner}")
    print(f"Accuracy difference: {comparison.accuracy_diff*100:+.1f}%")
    print(f"Speed difference: {comparison.speed_diff_pct:+.1f}%")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Model Comparison Evaluation")
    parser.add_argument(
        "--comparison",
        type=str,
        choices=["clip", "vlm", "backend", "all"],
        default="clip",
        help="Type of comparison to run"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--train_json",
        type=str,
        default="data/raw/train/train.json",
        help="Path to training JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_dataset(args.train_json)
    logger.info(f"Loaded {len(dataset)} samples")
    
    results = []
    
    if args.comparison in ["clip", "all"]:
        result = run_clip_comparison(dataset, args.samples)
        print_comparison_report(result)
        results.append(asdict(result))
    
    if args.comparison in ["vlm", "all"]:
        result = run_vlm_comparison(dataset, args.samples)
        print_comparison_report(result)
        results.append(asdict(result))
    
    if args.comparison in ["backend", "all"]:
        result = run_backend_comparison(dataset, args.samples)
        print_comparison_report(result)
        results.append(asdict(result))
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")
    else:
        # Default output
        output_path = project_root / "outputs" / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
