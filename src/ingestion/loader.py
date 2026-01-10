# Video Loading with Decord

import logging
import os
from typing import Iterator, Dict, Any
import torch
import numpy as np

try:
    from decord import VideoReader, cpu, gpu
    from decord import bridge
    bridge.set_bridge('torch')
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    logging.warning("Decord not available. Install via: pip install decord")

class RoadVideoLoader:
    """GPU-accelerated video loader using Decord."""

    def __init__(self, config):
        if not DECORD_AVAILABLE:
            raise ImportError("Decord library is not installed.")
        
        self.cfg = config

        if not os.path.exists(config.video_path):
            raise FileNotFoundError(f"Video path does not found: {config.video_path}")
        
        self.ctx = self._get_context(config.device, config.ctx_id)

        logging.info(f"Initializing Decord VideoReader for {config.video_path} on {self.ctx}")
        self.reader = VideoReader(
            config.video_path,
            ctx=self.ctx,
            width=config.width,
            height=config.height,
            num_threads=config.num_threads
        )   

    def _get_context(self, device_str: str, device_id: int):
        if 'gpu' in device_str.lower() or 'cuda' in device_str.lower():
            if torch.cuda.is_available():
                return gpu(device_id)
            else:
                logging.warning("CUDA not available, falling back to CPU.")
                return cpu(0)
        return cpu(0)