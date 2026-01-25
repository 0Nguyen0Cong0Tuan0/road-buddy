"""
Model Registry for Finetuned YOLO Models.

Provides a centralized registry and factory for loading finetuned YOLO models.
Supports models trained on Road Lane and BDD100K datasets.

Usage:
    from src.perception.model_registry import ModelRegistry, get_model
    
    # List available models
    models = ModelRegistry.list_models()
    
    # Get model info
    info = ModelRegistry.get_model_info("yolo11n_road_lane")
    
    # Load a model
    model = get_model("yolo11n_road_lane")
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logging.warning("Ultralytics not available. Install via: pip install ultralytics")

def _get_project_root() -> Path:
    """Get project root directory."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / 'src').exists() or (parent / 'setup.py').exists():
            return parent
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = _get_project_root()
MODELS_DIR = PROJECT_ROOT / "models"
FINETUNE_DIR = MODELS_DIR / "finetune"
INITIAL_DIR = MODELS_DIR / "initial"

@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    path: Path
    base_model: str
    dataset: str
    task: str = "detect"
    description: str = ""
    
    @property
    def exists(self) -> bool:
        """Check if model weights file exists."""
        return self.path.exists()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": str(self.path),
            "base_model": self.base_model,
            "dataset": self.dataset,
            "task": self.task,
            "description": self.description,
            "exists": self.exists
        }

class ModelRegistry:
    """
    Registry of available finetuned YOLO models.
    
    Provides discovery and loading of models from the models/finetune directory.
    """
    
    # Registered models with their metadata
    _MODELS: Dict[str, ModelInfo] = {
        "yolo11n_road_lane": ModelInfo(
            name="yolo11n_road_lane",
            path=FINETUNE_DIR / "yolo11n_road_lane" / "weights" / "best.pt",
            base_model="yolo11n",
            dataset="road_lane",
            description="YOLO11n finetuned on Road Lane dataset for lane detection"
        ),
        "yolo11n_bdd100k": ModelInfo(
            name="yolo11n_bdd100k",
            path=FINETUNE_DIR / "yolo11n_bdd100k" / "weights" / "best.pt",
            base_model="yolo11n",
            dataset="bdd100k",
            description="YOLO11n finetuned on BDD100K dataset for traffic object detection"
        ),
        "yolo11l_road_lane": ModelInfo(
            name="yolo11l_road_lane",
            path=FINETUNE_DIR / "yolo11l_road_lane" / "weights" / "best.pt",
            base_model="yolo11l",
            dataset="road_lane",
            description="YOLO11l finetuned on Road Lane dataset for lane detection"
        ),
        "yolo11l_bdd100k": ModelInfo(
            name="yolo11l_bdd100k",
            path=FINETUNE_DIR / "yolo11l_bdd100k" / "weights" / "best.pt",
            base_model="yolo11l",
            dataset="bdd100k",
            description="YOLO11l finetuned on BDD100K dataset for traffic object detection"
        ),
        "yolo11n_unified": ModelInfo(
            name="yolo11n_unified",
            path=FINETUNE_DIR / "yolo11n_unified" / "weights" / "best.pt",
            base_model="yolo11n",
            dataset="unified",
            description="YOLO11n finetuned on combined Road Lane and BDD100K datasets"
        ),
        "yolo11l_unified": ModelInfo(
            name="yolo11l_unified",
            path=FINETUNE_DIR / "yolo11l_unified" / "weights" / "best.pt",
            base_model="yolo11l",
            dataset="unified",
            description="YOLO11l finetuned on combined Road Lane and BDD100K datasets"
        ),
    }
    
    # Initial/pretrained models (not finetuned)
    _INITIAL_MODELS: Dict[str, ModelInfo] = {
        "yolo11n": ModelInfo(
            name="yolo11n",
            path=INITIAL_DIR / "yolo11n.pt",
            base_model="yolo11n",
            dataset="coco",
            description="YOLO11n pretrained on COCO"
        ),
        "yolo11l": ModelInfo(
            name="yolo11l",
            path=INITIAL_DIR / "yolo11l.pt",
            base_model="yolo11l",
            dataset="coco",
            description="YOLO11l pretrained on COCO"
        ),
        "yolo26n": ModelInfo(
            name="yolo26n",
            path=INITIAL_DIR / "yolo26n.pt",
            base_model="yolo26n",
            dataset="coco",
            description="YOLO26n pretrained on COCO"
        ),
    }
    
    @classmethod
    def list_models(cls, include_initial: bool = False) -> List[str]:
        """
        List all available model names.
        
        Args:
            include_initial: Include pretrained/initial models
            
        Returns:
            List of model names
        """
        models = list(cls._MODELS.keys())
        if include_initial:
            models.extend(cls._INITIAL_MODELS.keys())
        return models
    
    @classmethod
    def list_available_models(cls, include_initial: bool = False) -> List[str]:
        """
        List models that have weights files present.
        
        Args:
            include_initial: Include pretrained/initial models
            
        Returns:
            List of model names with existing weight files
        """
        available = []
        for name, info in cls._MODELS.items():
            if info.exists:
                available.append(name)
        
        if include_initial:
            for name, info in cls._INITIAL_MODELS.items():
                if info.exists:
                    available.append(name)
        
        return available
    
    @classmethod
    def get_model_info(cls, name: str) -> Optional[ModelInfo]:
        """
        Get information about a model.
        
        Args:
            name: Model name
            
        Returns:
            ModelInfo or None if not found
        """
        if name in cls._MODELS:
            return cls._MODELS[name]
        if name in cls._INITIAL_MODELS:
            return cls._INITIAL_MODELS[name]
        return None
    
    @classmethod
    def get_model_path(cls, name: str) -> Optional[Path]:
        """
        Get the path to a model's weights file.
        
        Args:
            name: Model name
            
        Returns:
            Path to weights file or None if not found
        """
        info = cls.get_model_info(name)
        return info.path if info else None
    
    @classmethod
    def get_models_by_dataset(cls, dataset: str) -> List[str]:
        """
        Get models trained on a specific dataset.
        
        Args:
            dataset: Dataset name (e.g., "road_lane", "bdd100k")
            
        Returns:
            List of model names
        """
        return [
            name for name, info in cls._MODELS.items()
            if info.dataset == dataset
        ]
    
    @classmethod
    def get_models_by_base(cls, base_model: str) -> List[str]:
        """
        Get models based on a specific architecture.
        
        Args:
            base_model: Base model name (e.g., "yolo11n", "yolo11l")
            
        Returns:
            List of model names
        """
        return [
            name for name, info in cls._MODELS.items()
            if info.base_model == base_model
        ]

def get_model(name: str, device: str = "auto") -> "YOLO":
    """Factory function to load a model by name."""
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics is required. Install via: pip install ultralytics")
    
    info = ModelRegistry.get_model_info(name)
    if info is None:
        available = ModelRegistry.list_models(include_initial=True)
        raise ValueError(
            f"Model '{name}' not found. Available models: {available}"
        )
    
    if not info.exists:
        raise FileNotFoundError(
            f"Model weights not found at: {info.path}"
        )
    
    logging.info(f"Loading model '{name}' from {info.path}")
    model = YOLO(str(info.path))
    
    # Move to device if specified
    if device != "auto":
        model.to(device)
    
    return model