"""
Evaluation Package for Road Buddy VQA.

Provides dataset loading, keyframe evaluation, and VQA metrics.
"""
from .dataset_loader import (
    VQASample,
    RoadBuddyDataset,
    load_dataset,
    print_dataset_stats,
)

from .keyframe_eval import (
    KeyframeEvalResult,
    evaluate_keyframe_selection,
    evaluate_batch,
    recall_at_k,
    compute_recall_at_k_sweep,
    print_eval_summary,
)

from .metrics import (
    compute_accuracy,
    compute_accuracy_by_letter,
    compute_f1,
    compute_confusion_matrix,
    compute_all_metrics,
    print_metrics,
)

__all__ = [
    # Dataset
    "VQASample",
    "RoadBuddyDataset",
    "load_dataset",
    "print_dataset_stats",
    # Keyframe eval
    "KeyframeEvalResult",
    "evaluate_keyframe_selection",
    "evaluate_batch",
    "recall_at_k",
    "compute_recall_at_k_sweep",
    "print_eval_summary",
    # Metrics
    "compute_accuracy",
    "compute_accuracy_by_letter",
    "compute_f1",
    "compute_confusion_matrix",
    "compute_all_metrics",
    "print_metrics",
]
