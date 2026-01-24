"""
VQA Evaluation Metrics.

Provides accuracy, F1, and confusion matrix metrics for MCQ evaluation.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

def compute_accuracy(predictions: List[str],ground_truths: List[str]) -> float:
    """Compute accuracy for VQA predictions."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    return correct / len(predictions)

def compute_accuracy_by_letter(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compute accuracy by comparing letter only. More lenient matching that ignores answer text."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    if len(predictions) == 0:
        return 0.0
    
    def get_letter(s: str) -> str:
        s = s.strip()
        if s and s[0].upper() in "ABCD":
            return s[0].upper()
        return s
    
    correct = sum(
        get_letter(p) == get_letter(g) 
        for p, g in zip(predictions, ground_truths)
    )
    return correct / len(predictions)

def compute_f1(predictions: List[str], ground_truths: List[str], average: str = "macro") -> float:
    """Compute F1 score for VQA predictions.""" 
    # Encode labels
    all_labels = list(set(predictions + ground_truths))
    encoder = LabelEncoder()
    encoder.fit(all_labels)
    
    y_pred = encoder.transform(predictions)
    y_true = encoder.transform(ground_truths)
    
    return f1_score(y_true, y_pred, average=average, zero_division=0)

def compute_confusion_matrix(predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
    """Compute confusion matrix for VQA predictions."""
    # Extract letters only
    def get_letter(s: str) -> str:
        s = s.strip()
        if s and s[0].upper() in "ABCD":
            return s[0].upper()
        return "?"
    
    pred_letters = [get_letter(p) for p in predictions]
    gt_letters = [get_letter(g) for g in ground_truths]
    
    labels = sorted(set(pred_letters + gt_letters))
    
    matrix = {}
    for true_label in labels:
        matrix[true_label] = {pred_label: 0 for pred_label in labels}
    
    for pred, gt in zip(pred_letters, gt_letters):
        if gt in matrix:
            matrix[gt][pred] += 1
    
    return {
        "labels": labels,
        "matrix": matrix,
        "total": len(predictions),
    }

def compute_per_category_accuracy(predictions: List[str], ground_truths: List[str], categories: List[str]) -> Dict[str, float]:
    """Compute accuracy per category/question type."""
    category_results = {}
    
    for pred, gt, cat in zip(predictions, ground_truths, categories):
        if cat not in category_results:
            category_results[cat] = {"correct": 0, "total": 0}
        
        category_results[cat]["total"] += 1
        if pred == gt:
            category_results[cat]["correct"] += 1
    
    return {
        cat: data["correct"] / data["total"] if data["total"] > 0 else 0.0
        for cat, data in category_results.items()
    }

def compute_binary_vs_mcq_accuracy(predictions: List[str], ground_truths: List[str], num_choices: List[int]) -> Dict[str, float]:
    """Compute accuracy separately for binary and MCQ questions."""
    binary_correct = 0
    binary_total = 0
    mcq_correct = 0
    mcq_total = 0
    
    for pred, gt, nc in zip(predictions, ground_truths, num_choices):
        if nc == 2:
            binary_total += 1
            if pred == gt:
                binary_correct += 1
        else:
            mcq_total += 1
            if pred == gt:
                mcq_correct += 1
    
    return {
        "binary_accuracy": binary_correct / binary_total if binary_total > 0 else 0.0,
        "mcq_accuracy": mcq_correct / mcq_total if mcq_total > 0 else 0.0,
        "binary_count": binary_total,
        "mcq_count": mcq_total,
    }

def compute_all_metrics(predictions: List[str], ground_truths: List[str], num_choices: Optional[List[int]] = None) -> Dict[str, Any]:
    """Compute all VQA metrics."""
    metrics = {
        "accuracy": compute_accuracy(predictions, ground_truths),
        "accuracy_by_letter": compute_accuracy_by_letter(predictions, ground_truths),
        "f1_macro": compute_f1(predictions, ground_truths, "macro"),
        "total_samples": len(predictions),
        "correct": sum(p == g for p, g in zip(predictions, ground_truths)),
    }
    
    if num_choices:
        binary_mcq_metrics = compute_binary_vs_mcq_accuracy(predictions, ground_truths, num_choices)
        metrics.update(binary_mcq_metrics)
    
    return metrics

def print_metrics(metrics: Dict[str, Any]):
    print("VQA evaluation metrics")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Accuracy (by letter): {metrics['accuracy_by_letter']:.2%}")
    print(f"F1 (macro): {metrics['f1_macro']:.3f}")
    
    if 'binary_accuracy' in metrics:
        print(f"Binary Questions: {metrics['binary_count']}")
        print(f"  Accuracy: {metrics['binary_accuracy']:.2%}")
        print(f"MCQ Questions: {metrics['mcq_count']}")
        print(f"  Accuracy: {metrics['mcq_accuracy']:.2%}")
    
def print_confusion_matrix(cm: Dict[str, Any]):
    labels = cm['labels']
    matrix = cm['matrix']
    
    print("\nConfusion Matrix:")
    print("True \\ Pred", end="\t")
    for label in labels:
        print(label, end="\t")
    print()
    
    for true_label in labels:
        print(true_label, end="\t\t")
        for pred_label in labels:
            print(matrix[true_label][pred_label], end="\t")
        print()