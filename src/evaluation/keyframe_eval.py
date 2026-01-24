"""
Keyframe Evaluation Metrics.

Evaluates keyframe selection quality against ground truth support_frames.
"""
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class KeyframeEvalResult:
    """
    Result of keyframe evaluation for a single sample.
    
    Attributes:
        sample_id: Sample identifier
        predicted_timestamps: List of predicted keyframe timestamps
        ground_truth_timestamps: List of ground truth timestamps
        tolerance: Tolerance window in seconds
        recall: Recall score (0-1)
        precision: Precision score (0-1)
        min_distance: Minimum distance from predicted to ground truth
        hit: Whether any prediction is within tolerance
    """
    sample_id: str
    predicted_timestamps: List[float]
    ground_truth_timestamps: List[float]
    tolerance: float
    recall: float
    precision: float
    min_distance: float
    hit: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "num_predicted": len(self.predicted_timestamps),
            "num_ground_truth": len(self.ground_truth_timestamps),
            "tolerance": self.tolerance,
            "recall": self.recall,
            "precision": self.precision,
            "min_distance": self.min_distance,
            "hit": self.hit,
        }

def evaluate_keyframe_selection(predicted_timestamps: List[float], ground_truth_timestamps: List[float], tolerance: float = 1.0, sample_id: str = "") -> KeyframeEvalResult:
    """Evaluate keyframe selection against ground truth."""
    if not predicted_timestamps or not ground_truth_timestamps:
        return KeyframeEvalResult(
            sample_id=sample_id,
            predicted_timestamps=predicted_timestamps,
            ground_truth_timestamps=ground_truth_timestamps,
            tolerance=tolerance,
            recall=0.0,
            precision=0.0,
            min_distance=float('inf'),
            hit=False,
        )
    
    pred = np.array(predicted_timestamps)
    gt = np.array(ground_truth_timestamps)
    
    # Compute distance matrix
    distances = np.abs(pred[:, np.newaxis] - gt[np.newaxis, :])
    
    # Find minimum distance for each ground truth
    min_dist_per_gt = np.min(distances, axis=0)
    
    # Count hits (predictions within tolerance of ground truth)
    gt_hits = np.sum(min_dist_per_gt <= tolerance)
    
    # Recall: fraction of ground truths that have a nearby prediction
    recall = gt_hits / len(gt)
    
    # Find minimum distance for each prediction
    min_dist_per_pred = np.min(distances, axis=1)
    
    # Precision: fraction of predictions that are near a ground truth
    pred_hits = np.sum(min_dist_per_pred <= tolerance)
    precision = pred_hits / len(pred)
    
    # Overall minimum distance
    min_distance = float(np.min(distances))
    
    # Hit: any ground truth captured
    hit = gt_hits > 0
    
    return KeyframeEvalResult(
        sample_id=sample_id,
        predicted_timestamps=predicted_timestamps,
        ground_truth_timestamps=ground_truth_timestamps,
        tolerance=tolerance,
        recall=recall,
        precision=precision,
        min_distance=min_distance,
        hit=hit,
    )

def evaluate_batch(predictions: List[List[float]], ground_truths: List[List[float]], sample_ids: Optional[List[str]] = None, tolerance: float = 1.0) -> Dict[str, Any]:
    """Evaluate keyframe selection on a batch of samples."""
    if sample_ids is None:
        sample_ids = [f"sample_{i}" for i in range(len(predictions))]
    
    results = []
    for pred, gt, sid in zip(predictions, ground_truths, sample_ids):
        result = evaluate_keyframe_selection(pred, gt, tolerance, sid)
        results.append(result)
    
    # Aggregate metrics
    recalls = [r.recall for r in results]
    precisions = [r.precision for r in results]
    min_distances = [r.min_distance for r in results if r.min_distance < float('inf')]
    hits = [r.hit for r in results]
    
    return {
        "num_samples": len(results),
        "mean_recall": float(np.mean(recalls)),
        "mean_precision": float(np.mean(precisions)),
        "mean_min_distance": float(np.mean(min_distances)) if min_distances else float('inf'),
        "hit_rate": float(np.mean(hits)),
        "tolerance": tolerance,
        "per_sample_results": [r.to_dict() for r in results],
    }

def recall_at_k(predicted_timestamps: List[float], ground_truth_timestamps: List[float], k: int, tolerance: float = 1.0) -> float:
    """Compute Recall@K metric."""
    top_k_pred = predicted_timestamps[:k]
    result = evaluate_keyframe_selection(top_k_pred, ground_truth_timestamps, tolerance)
    return result.recall

def compute_recall_at_k_sweep(predicted_timestamps: List[float], ground_truth_timestamps: List[float], k_values: List[int] = [1, 3, 5, 8], tolerance: float = 1.0) -> Dict[int, float]:
    """Compute Recall@K for multiple K values."""
    results = {}
    for k in k_values:
        results[k] = recall_at_k(predicted_timestamps, ground_truth_timestamps, k, tolerance)
    return results

def print_eval_summary(batch_results: Dict[str, Any]):
    print("Keyframe evaluation summary")
    print(f"Samples Evaluated: {batch_results['num_samples']}")
    print(f"Tolerance: {batch_results['tolerance']}s")
    print(f"Mean Recall: {batch_results['mean_recall']:.3f}")
    print(f"Mean Precision: {batch_results['mean_precision']:.3f}")
    print(f"Hit Rate: {batch_results['hit_rate']:.3f}")
    print(f"Mean Min Distance: {batch_results['mean_min_distance']:.3f}s")