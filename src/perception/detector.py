"""
Perception Engine for YOLO-based Object Detection.

Provides a unified interface for running object detection and tracking
using YOLO models, with support for finetuned models.

Usage:
    from src.perception.detector import PerceptionEngine
    from config.settings import PerceptionConfig
    
    config = PerceptionConfig(model_path="models/finetune/yolo11n_bdd100k/weights/best.pt")
    engine = PerceptionEngine(config)
    
    # Run detection
    results = engine.detect(frames)
    
    # Parse results
    detections = engine.parse_results(results)
"""
import torch
import os
from typing import List, Optional, Union
import logging
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

from .results import FrameDetections, Detection, parse_yolo_results

class PerceptionEngine:
    """
    Thread-safe wrapper around YOLO for object detection and tracking.
    
    Supports both pretrained and finetuned YOLO models with configurable
    inference parameters.
    
    Attributes:
        model: Loaded YOLO model
        device: Device for inference ('cpu', 'cuda', '0', etc.)
        cfg: Configuration object with inference parameters
    """

    def __init__(self, config, warmup: bool = True):
        """
        Initialize PerceptionEngine with configuration.
        
        Args:
            config: YOLOConfig or compatible config object with:
                - model_path: Path to model weights
                - device: Device for inference
                - confidence: Detection confidence threshold
                - iou_threshold: NMS IOU threshold
                - imgsz: Input image size
                - classes: Optional class filter
                - half: Use FP16 inference
                - tracker_config: Tracker configuration file
            warmup: Whether to run warmup inference
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics is required. Install via: pip install ultralytics"
            )
        
        self.cfg = config
        self._setup_device()
        self._load_model()
        
        if warmup:
            self._warmup()
    
    def _setup_device(self):
        """Configure inference device."""
        device = getattr(self.cfg, 'device', '0')
        
        if device in ('cuda', '0', 0) and not torch.cuda.is_available():
            logging.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        elif device == 'auto':
            self.device = '0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
    
    def _load_model(self):
        """Load YOLO model from config path."""
        model_path = self.cfg.model_path
        
        # Create directory if needed
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        if not os.path.exists(model_path):
            # Try downloading if it looks like a model name
            model_name = getattr(self.cfg, 'model_name', None)
            if model_name:
                logging.warning(
                    f"Model not found at {model_path}. Attempting to download..."
                )
                self._download_model(model_name, model_path)
            else:
                raise FileNotFoundError(f"Model not found at: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")

        logging.info(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
    
        if self.device != 'cpu':
            self.model.to(self.device)
            logging.info(f"Model moved to {self.device}")
        
        # Store class names from model
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        
    def _download_model(self, model_name: str, save_path: str):
        """Download a pretrained model."""
        try:
            logging.info(f"Downloading {model_name}...")
            temp_model = YOLO(model_name) 
            temp_model.save(save_path)
            logging.info(f"Model saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
    
    def _warmup(self):
        """Warm up GPU with dummy inference."""
        logging.info("Warming up the model...")
        
        # Use imgsz from config (not input_size)
        imgsz = getattr(self.cfg, 'imgsz', 640)
        dummy_input = torch.zeros(1, 3, imgsz, imgsz)

        if self.device != 'cpu':
            dummy_input = dummy_input.to(self.device)
        
        self.model(dummy_input, verbose=False)
        logging.info("Model warm-up complete.")
    
    def detect(
        self,
        frames: Union[torch.Tensor, np.ndarray, List],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        classes: Optional[List[int]] = None
    ) -> List:
        """
        Run object detection on input frames.
        
        Args:
            frames: Input frames as tensor (N, C, H, W), numpy array, or list
            conf: Override confidence threshold (uses config default if None)
            iou: Override IOU threshold (uses config default if None)
            classes: Override class filter (uses config default if None)
            
        Returns:
            List of YOLO result objects
        """
        results = self.model(
            source=frames,
            conf=conf or self.cfg.confidence,
            iou=iou or self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            device=self.device,
            verbose=False,
            classes=classes if classes is not None else self.cfg.classes,
            half=getattr(self.cfg, 'half', False)
        )

        return results

    def track(
        self,
        frames: Union[torch.Tensor, np.ndarray, List],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        persist: bool = True
    ) -> List:
        """
        Run object tracking on input frames.
        
        Args:
            frames: Input frames as tensor, numpy array, or list
            conf: Override confidence threshold
            iou: Override IOU threshold
            persist: Whether to persist tracks across calls
            
        Returns:
            List of YOLO result objects with tracking IDs
        """
        tracker = getattr(self.cfg, 'tracker_config', 'botsort.yaml')
        
        results = self.model.track(
            source=frames,
            conf=conf or self.cfg.confidence,
            iou=iou or self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            device=self.device,
            persist=persist,
            tracker=tracker,
            verbose=False,
            classes=self.cfg.classes,
            half=getattr(self.cfg, 'half', False)
        )

        return results
    
    def parse_results(
        self,
        results: List,
        start_frame_idx: int = 0
    ) -> List[FrameDetections]:
        """
        Parse YOLO results into structured FrameDetections.
        
        Args:
            results: List of YOLO result objects
            start_frame_idx: Starting frame index for numbering
            
        Returns:
            List of FrameDetections with parsed detection data
        """
        return parse_yolo_results(results, start_frame_idx)
    
    def detect_and_parse(
        self,
        frames: Union[torch.Tensor, np.ndarray, List],
        start_frame_idx: int = 0,
        **kwargs
    ) -> List[FrameDetections]:
        """
        Run detection and return parsed results in one call.
        
        Args:
            frames: Input frames
            start_frame_idx: Starting frame index
            **kwargs: Additional arguments passed to detect()
            
        Returns:
            List of FrameDetections
        """
        results = self.detect(frames, **kwargs)
        return self.parse_results(results, start_frame_idx)

    def export_tensorrt(self, output_path: str):
        """
        Export model to TensorRT for production deployment.
        
        Args:
            output_path: Path to save the TensorRT engine
        """
        logging.info("Exporting to TensorRT...")
        self.model.export(
            format="engine",
            save_path=output_path,
            imgsz=self.cfg.imgsz,
            device=self.device,
            half=getattr(self.cfg, 'half', False),
            dynamic=True
        )
        logging.info(f"TensorRT engine saved to {output_path}")
    
    def get_class_names(self) -> dict:
        """Get the class name mapping from the model."""
        return self.class_names
    
    def __repr__(self) -> str:
        return (
            f"PerceptionEngine("
            f"model={self.cfg.model_path}, "
            f"device={self.device}, "
            f"classes={len(self.class_names)})"
        )