"""
Dataset Builder for Training.

Converts train.json to HuggingFace dataset format for training.
"""
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import random

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

@dataclass
class TrainingSample:
    """
    Single training sample for VLM fine-tuning.
    
    Attributes:
        id: Sample ID
        question: Question text
        choices: List of choices
        answer: Correct answer (full text)
        answer_letter: Answer letter (A, B, C, D)
        video_path: Path to video
        support_frames: Ground truth keyframe timestamps
        prompt: Formatted prompt for training
        response: Expected response for training
    """
    id: str
    question: str
    choices: List[str]
    answer: str
    answer_letter: str
    video_path: str
    support_frames: List[float]
    prompt: str
    response: str

def build_prompt_response_pair(question: str, choices: List[str], answer: str, include_reasoning: bool = False) -> Tuple[str, str]:
    """Build prompt-response pair for training."""
    choices_text = "\n".join(choices)
    
    prompt = f"""Xem các hình ảnh và trả lời câu hỏi trắc nghiệm.

Câu hỏi: {question}

Lựa chọn:
{choices_text}

Trả lời:"""
    
    # Extract letter from answer
    answer_letter = answer[0] if answer and answer[0] in "ABCD" else "A"
    
    if include_reasoning:
        response = f"Dựa vào hình ảnh, đáp án đúng là {answer}"
    else:
        response = answer_letter
    
    return prompt, response

def load_train_json(json_path: str) -> List[Dict[str, Any]]:
    """Load raw data from train.json."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get("data", data)

def build_training_samples(json_path: str, include_reasoning: bool = False) -> List[TrainingSample]:
    """Build training samples from train.json."""
    raw_data = load_train_json(json_path)
    samples = []
    
    for item in raw_data:
        question = item["question"]
        choices = item.get("choices", [])
        answer = item.get("answer", "")
        
        prompt, response = build_prompt_response_pair(
            question, choices, answer, include_reasoning
        )
        
        answer_letter = answer[0] if answer and answer[0] in "ABCD" else "A"
        
        sample = TrainingSample(
            id=item["id"],
            question=question,
            choices=choices,
            answer=answer,
            answer_letter=answer_letter,
            video_path=item.get("video_path", ""),
            support_frames=item.get("support_frames", []),
            prompt=prompt,
            response=response,
        )
        samples.append(sample)
    
    logger.info(f"Built {len(samples)} training samples")
    return samples

def split_samples(samples: List[TrainingSample], train_ratio: float = 0.9, seed: int = 42) -> Tuple[List[TrainingSample], List[TrainingSample]]:
    """Split samples into train/validation."""
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train = shuffled[:split_idx]
    val = shuffled[split_idx:]
    
    logger.info(f"Split: {len(train)} train, {len(val)} val")
    return train, val


def samples_to_hf_format(samples: List[TrainingSample]) -> List[Dict[str, Any]]:
    """Convert samples to HuggingFace datasets format."""
    return [
        {
            "id": s.id,
            "prompt": s.prompt,
            "response": s.response,
            "question": s.question,
            "choices": s.choices,
            "answer": s.answer,
            "answer_letter": s.answer_letter,
            "video_path": s.video_path,
            "support_frames": s.support_frames,
        }
        for s in samples
    ]

def build_hf_dataset(json_path: str, train_ratio: float = 0.9, seed: int = 42):
    """Build HuggingFace Dataset from train.json."""
    samples = build_training_samples(json_path)
    train_samples, val_samples = split_samples(samples, train_ratio, seed)
    
    train_data = samples_to_hf_format(train_samples)
    val_data = samples_to_hf_format(val_samples)
    
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
    })
    
    logger.info(f"Built HuggingFace dataset: {dataset}")
    return dataset