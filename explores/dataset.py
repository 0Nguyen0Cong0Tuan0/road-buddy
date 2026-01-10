from pathlib import Path


current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

TRAIN_DIR = parent_dir / "data" / "train"
TRAIN_VIDEO = TRAIN_DIR / "videos"
TRAIN_JSON = TRAIN_DIR / "train.json"
import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class RoadBuddyDataset(Dataset):
    def __init__(self, annotation_file, root_dir, transform=None, frame_count=16, height=224, width=224):
        """
        Args:
            annotation_file (str): Path to the JSON file containing metadata.
            root_dir (str): The project root directory (absolute path).
            transform (callable, optional): Optional transform to be applied on a sample.
            frame_count (int): Number of frames to extract from the video.
            height (int): Target height for resizing.
            width (int): Target width for resizing.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.frame_count = frame_count
        self.height = height
        self.width = width

        # Load annotations
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # Basic validation to ensure data is loaded
        if not self.data:
            raise ValueError(f"Annotation file {annotation_file} is empty or invalid.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # KEY FIX: The JSON likely contains "train/videos/..." 
        # We must join it with project_root to get the absolute path.
        # Adjust 'filename' key if your JSON uses 'video_path' or 'id'
        relative_path = item.get('filename', item.get('video_path'))
        
        if not relative_path:
            raise KeyError(f"Could not find 'filename' or 'video_path' in annotation entry: {item}")

        video_path = os.path.join(self.root_dir, relative_path)

        try:
            frames = self._load_video(video_path)
        except Exception as e:
            print(f"Error loading video at index {idx}: {video_path}")
            raise e

        if self.transform:
            frames = self.transform(frames)

        # Return dictionary compatible with most pipelines
        return {
            "video": frames,     # Tensor shape: (T, C, H, W) or (T, H, W, C) depending on transforms
            "label": item.get('label', -1), # Placeholder if labels aren't numeric yet
            "metadata": item     # Pass full metadata for debugging
        }

    def _load_video(self, video_path):
        """
        Reads a video and extracts a fixed number of frames using uniform sampling.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Handle edge case: empty video or read error
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video has 0 frames: {video_path}")

        # Uniform Temporal Subsampling: Pick 'frame_count' indices evenly spaced
        indices = np.linspace(0, total_frames - 1, self.frame_count).astype(int)
        
        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            if i in indices:
                # Resize and Color Convert (BGR -> RGB)
                frame = cv2.resize(frame, (self.width, self.height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
                # Optimization: Stop if we have enough frames
                if len(frames) >= self.frame_count:
                    break
        
        cap.release()

        # Padding: If video is too short, pad with the last frame
        while len(frames) < self.frame_count:
            frames.append(frames[-1] if frames else np.zeros((self.height, self.width, 3), dtype=np.uint8))

        # Convert to Tensor (T, H, W, C) -> PyTorch usually expects (C, T, H, W) for 3D CNNs
        # For now, we return (T, H, W, C) as a numpy array, let transforms handle Permute
        return torch.from_numpy(np.array(frames))