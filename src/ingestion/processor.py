# Batch Video Processor for Dataset Preparation

import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import torch
from tqdm import tqdm
import json
import time

from .loader import RoadVideoLoader
from .utils import validate_video, get_video_info


@dataclass
class ProcessingStats:
    """Statistics for batch processing."""
    total_videos: int = 0
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    total_time: float = 0.0
    errors: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_videos": self.total_videos,
            "processed": self.processed,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": self.processed / self.total_videos if self.total_videos > 0 else 0,
            "total_time": self.total_time,
            "avg_time_per_video": self.total_time / self.processed if self.processed > 0 else 0,
            "errors": self.errors
        }

class BatchVideoProcessor:
    """Process batches of videos for offline dataset preparation.
    
    Features:
    - Multi-threaded/GPU batched processing
    - Progress tracking with tqdm
    - Resume capability for interrupted processing
    - Error logging and recovery
    - Flexible processing functions
    
    Args:
        config: Configuration object
        num_workers: Number of parallel workers (default: 4)
        checkpoint_path: Path to save/load checkpoint for resuming
        
    Example:
        >>> processor = BatchVideoProcessor(config, num_workers=4)
        >>> results = processor.process_dataset(
        ...     csv_path='data/videos.csv',
        ...     video_col='video_path',
        ...     process_fn=extract_keyframes
        ... )
    """
    
    def __init__(
        self,
        config,
        num_workers: int = 4,
        checkpoint_path: Optional[str] = None
    ):
        """Initialize batch processor."""
        self.config = config
        self.num_workers = num_workers
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.stats = ProcessingStats()
        self.processed_ids = set()
        
        # Load checkpoint if exists
        if self.checkpoint_path and self.checkpoint_path.exists():
            self._load_checkpoint()
            logging.info(f"Loaded checkpoint: {len(self.processed_ids)} videos already processed")
    
    def process_dataset(
        self,
        csv_path: str,
        video_col: str,
        process_fn: Callable,
        id_col: Optional[str] = None,
        output_dir: Optional[str] = None,
        save_interval: int = 100
    ) -> Dict[str, Any]:
        """Process all videos in a dataset CSV.
        
        Args:
            csv_path: Path to CSV file with video paths
            video_col: Column name containing video paths
            process_fn: Function to process each video (takes RoadVideoLoader, returns dict)
            id_col: Column name for unique video ID (for resume capability)
            output_dir: Directory to save processed outputs
            save_interval: Save checkpoint every N videos
            
        Returns:
            dict: Processing statistics and results
        """
        # Load dataset
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded dataset: {len(df)} videos from {csv_path}")
        
        # Filter already processed if resuming
        if id_col and self.processed_ids:
            original_len = len(df)
            df = df[~df[id_col].isin(self.processed_ids)]
            logging.info(f"Resuming: {original_len - len(df)} videos already processed")
        
        self.stats.total_videos = len(df)
        
        # Create output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Process videos
        results = []
        start_time = time.time()
        
        with tqdm(total=len(df), desc="Processing videos") as pbar:
            for idx, row in df.iterrows():
                video_path = row[video_col]
                video_id = row[id_col] if id_col else f"video_{idx}"
                
                try:
                    # Validate video first
                    if not validate_video(video_path):
                        logging.warning(f"Skipping invalid video: {video_path}")
                        self.stats.skipped += 1
                        pbar.update(1)
                        continue
                    
                    # Create loader config
                    video_config = self._create_video_config(video_path)
                    
                    # Process video
                    result = self._process_single_video(
                        video_config=video_config,
                        process_fn=process_fn,
                        video_id=video_id,
                        metadata=row.to_dict()
                    )
                    
                    if result:
                        results.append(result)
                        self.stats.processed += 1
                        
                        if id_col:
                            self.processed_ids.add(video_id)
                    
                    # Save checkpoint periodically
                    if self.checkpoint_path and self.stats.processed % save_interval == 0:
                        self._save_checkpoint()
                
                except Exception as e:
                    logging.error(f"Error processing {video_path}: {e}")
                    self.stats.failed += 1
                    self.stats.errors.append({
                        "video_id": video_id,
                        "video_path": str(video_path),
                        "error": str(e)
                    })
                
                pbar.update(1)
        
        self.stats.total_time = time.time() - start_time
        
        # Final checkpoint
        if self.checkpoint_path:
            self._save_checkpoint()
        
        # Save results
        if output_dir:
            self._save_results(results, output_path)
        
        return {
            "results": results,
            "stats": self.stats.to_dict()
        }
    
    def _process_single_video(
        self,
        video_config,
        process_fn: Callable,
        video_id: str,
        metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process a single video.
        
        Args:
            video_config: Configuration for video loader
            process_fn: Processing function
            video_id: Unique video identifier
            metadata: Additional metadata
            
        Returns:
            dict: Processing result or None if failed
        """
        try:
            # Load video
            loader = RoadVideoLoader(video_config)
            
            # Process with custom function
            result = process_fn(loader)
            
            # Add metadata
            result.update({
                "video_id": video_id,
                "metadata": metadata,
                "video_info": loader.get_metadata()
            })
            
            return result
            
        except Exception as e:
            logging.error(f"Failed to process video {video_id}: {e}")
            return None
    
    def _create_video_config(self, video_path: str):
        """Create video config from base config."""
        from dataclasses import replace
        return replace(self.config, video_path=str(video_path))
    
    def _save_checkpoint(self):
        """Save checkpoint for resume capability."""
        if not self.checkpoint_path:
            return
        
        checkpoint = {
            "processed_ids": list(self.processed_ids),
            "stats": self.stats.to_dict()
        }
        
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logging.debug(f"Checkpoint saved: {len(self.processed_ids)} videos")
    
    def _load_checkpoint(self):
        """Load checkpoint to resume processing."""
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return
        
        with open(self.checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        self.processed_ids = set(checkpoint.get("processed_ids", []))
        
        # Restore stats
        stats_dict = checkpoint.get("stats", {})
        self.stats.processed = stats_dict.get("processed", 0)
        self.stats.failed = stats_dict.get("failed", 0)
        self.stats.skipped = stats_dict.get("skipped", 0)
    
    def _save_results(self, results: List[Dict], output_path: Path):
        """Save processing results to disk."""
        results_file = output_path / "processing_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Results saved to {results_file}")
        
        # Also save stats
        stats_file = output_path / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2)
        
        logging.info(f"Stats saved to {stats_file}")


# ==================== Example Processing Functions ====================

def extract_keyframes(loader: RoadVideoLoader, num_frames: int = 8) -> Dict[str, Any]:
    """Example processing function: Extract keyframes.
    
    Args:
        loader: Video loader instance
        num_frames: Number of keyframes to extract
        
    Returns:
        dict: Keyframes and metadata
    """
    frames = loader.sample_uniform(num_frames)
    
    return {
        "keyframes": frames,
        "num_keyframes": len(frames),
        "sampling_method": "uniform"
    }

