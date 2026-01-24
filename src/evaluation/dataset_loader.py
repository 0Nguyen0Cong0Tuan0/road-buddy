"""
Dataset Loader for Road Buddy VQA.

Loads and manages the train.json dataset with video paths and ground truth.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import random

logger = logging.getLogger(__name__)

@dataclass
class VQASample:
    id: str
    question: str
    choices: List[str]
    answer: str
    support_frames: List[float]
    video_path: str
    video_abs_path: Optional[Path] = None
    
    @property
    def num_choices(self) -> int:
        return len(self.choices)
    
    @property
    def answer_index(self) -> int:
        for i, choice in enumerate(self.choices):
            if choice == self.answer:
                return i
        return -1
    
    @property
    def answer_letter(self) -> str:
        if self.answer and len(self.answer) > 0:
            return self.answer[0]
        return ""
    
    @property
    def is_binary(self) -> bool:
        return self.num_choices == 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "choices": self.choices,
            "answer": self.answer,
            "answer_index": self.answer_index,
            "support_frames": self.support_frames,
            "video_path": self.video_path,
            "num_choices": self.num_choices,
            "is_binary": self.is_binary,
        }

class RoadBuddyDataset:
    """Dataset class for Road Buddy VQA. Loads train.json and provides access to samples."""
    def __init__(self, json_path: str, data_root: Optional[str] = None, validate_videos: bool = False):
        self.json_path = Path(json_path)
        # data_root should be parent.parent since video_path in json includes "train/videos/..."
        self.data_root = Path(data_root) if data_root else self.json_path.parent.parent
        self.validate_videos = validate_videos
        
        self.samples: List[VQASample] = []
        self._load_data()
    
    def _load_data(self):
        """Load data from JSON file."""
        logger.info(f"Loading dataset from {self.json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_samples = data.get("data", data)
        
        missing_videos = 0
        
        for item in raw_samples:
            video_rel_path = item.get("video_path", "")
            video_abs_path = self.data_root / video_rel_path
            
            if self.validate_videos and not video_abs_path.exists():
                missing_videos += 1
                continue
            
            sample = VQASample(
                id=item["id"],
                question=item["question"],
                choices=item.get("choices", []),
                answer=item.get("answer", ""),
                support_frames=item.get("support_frames", []),
                video_path=video_rel_path,
                video_abs_path=video_abs_path if video_abs_path.exists() else None,
            )
            self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} samples")
        
        if missing_videos > 0:
            logger.warning(f"Skipped {missing_videos} samples with missing videos")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> VQASample:
        return self.samples[idx]
    
    def __iter__(self) -> Iterator[VQASample]:
        return iter(self.samples)
    
    def get_by_id(self, sample_id: str) -> Optional[VQASample]:
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        return None
    
    def filter_by_num_choices(self, num_choices: int) -> List[VQASample]:
        return [s for s in self.samples if s.num_choices == num_choices]
    
    def get_binary_questions(self) -> List[VQASample]:
        return self.filter_by_num_choices(2)
    
    def get_mcq_questions(self) -> List[VQASample]:
        return [s for s in self.samples if s.num_choices > 2]
    
    def split(self, train_ratio: float = 0.8, seed: int = 42) -> tuple:
        """Split dataset into train/validation sets."""
        random.seed(seed)
        samples = self.samples.copy()
        random.shuffle(samples)
        
        split_idx = int(len(samples) * train_ratio)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        logger.info(f"Split: {len(train_samples)} train, {len(val_samples)} val")
        
        return train_samples, val_samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        num_binary = len(self.get_binary_questions())
        num_mcq = len(self.get_mcq_questions())

        unique_videos = set(s.video_path for s in self.samples)
        choice_counts = {}
        for s in self.samples:
            nc = s.num_choices
            choice_counts[nc] = choice_counts.get(nc, 0) + 1

        total_support_frames = sum(len(s.support_frames) for s in self.samples)
        avg_support_frames = total_support_frames / len(self.samples) if self.samples else 0
        
        return {
            "total_samples": len(self.samples),
            "binary_questions": num_binary,
            "mcq_questions": num_mcq,
            "unique_videos": len(unique_videos),
            "choice_distribution": choice_counts,
            "total_support_frames": total_support_frames,
            "avg_support_frames_per_sample": avg_support_frames,
        }
    
    def to_huggingface_format(self) -> List[Dict[str, Any]]:
        """Convert to HuggingFace datasets format."""
        return [s.to_dict() for s in self.samples]

def load_dataset(json_path: str = "data/raw/train/train.json", validate_videos: bool = False) -> RoadBuddyDataset:
    """Load the Road Buddy VQA dataset."""
    return RoadBuddyDataset(json_path, validate_videos=validate_videos)

def print_dataset_stats(dataset: RoadBuddyDataset):
    stats = dataset.get_statistics()
    
    print("\nDataset statistics")
    print(f"Total Samples: {stats['total_samples']}")
    print(f"Binary Questions: {stats['binary_questions']}")
    print(f"MCQ Questions: {stats['mcq_questions']}")
    print(f"Unique Videos: {stats['unique_videos']}")
    print(f"Avg Support Frames: {stats['avg_support_frames_per_sample']:.2f}")
    print("\nChoice Distribution:")
    for nc, count in sorted(stats['choice_distribution'].items()):
        print(f"  {nc} choices: {count} samples")