"""
Device Utilities for GPU/CPU Fallback.

Provides unified device detection and management for all modules.
Automatically falls back to CPU when GPU is not available.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from transformers import BitsAndBytesConfig
import os
import torch

logger = logging.getLogger(__name__)

def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA GPU: {gpu_name}")
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Using Apple Silicon MPS")
        return "mps"
    else:
        logger.info("No GPU available, using CPU")
        return "cpu"

def get_device_info() -> Dict[str, Any]:
    """Get detailed device information."""
    info = {
        "device": get_device(),
        "cuda_available": False,
        "mps_available": False,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "cpu_count": os.cpu_count(),
    }
    
    info["cuda_available"] = torch.cuda.is_available()
    info["mps_available"] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info["gpu_count"] = torch.cuda.device_count()
    
    return info

@dataclass
class DeviceConfig:
    """
    Device configuration with automatic GPU/CPU fallback.
    
    Attributes:
        preferred_device: Preferred device ("auto", "cuda", "cpu", "mps")
        use_fp16: Use half precision on GPU
        use_quantization: Use INT8/INT4 quantization
        max_memory_gb: Maximum GPU memory to use (for multi-GPU)
    """
    preferred_device: str = "auto"
    use_fp16: bool = True
    use_quantization: bool = False
    quantization_bits: int = 4
    max_memory_gb: Optional[float] = None
    
    @property
    def device(self) -> str:
        """Get actual device based on preference and availability."""
        if self.preferred_device == "auto":
            return get_device()
        
        # Validate requested device exists
        if self.preferred_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        elif self.preferred_device == "mps":
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                logger.warning("MPS requested but not available, falling back to CPU")
                return "cpu"
        
        return self.preferred_device
    
    @property
    def is_gpu(self) -> bool:
        return self.device in ("cuda", "mps")
    
    @property
    def is_cpu(self) -> bool:
        return self.device == "cpu"
    
    @property
    def dtype(self):
        if self.is_gpu and self.use_fp16:
            return torch.float16
        return torch.float32
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for loading HuggingFace models."""
        kwargs = {}
        
        if self.is_gpu:
            kwargs["device_map"] = "auto"
            
            if self.use_fp16:
                kwargs["torch_dtype"] = torch.float16
            
            if self.use_quantization:
                if self.quantization_bits == 4:
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=kwargs.get("torch_dtype"),
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                elif self.quantization_bits == 8:
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
        
            if self.max_memory_gb:
                kwargs["max_memory"] = {0: f"{self.max_memory_gb}GB"}
        else:
            # CPU mode
            kwargs["device_map"] = "cpu"
        
        return kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "preferred_device": self.preferred_device,
            "actual_device": self.device,
            "is_gpu": self.is_gpu,
            "use_fp16": self.use_fp16,
            "use_quantization": self.use_quantization,
            "quantization_bits": self.quantization_bits,
        }

def get_clip_device_config() -> DeviceConfig:
    """Get optimized config for CLIP models."""
    return DeviceConfig(
        preferred_device="auto",
        use_fp16=True,
        use_quantization=False,
    )

def get_yolo_device_config() -> DeviceConfig:
    """Get optimized config for YOLO models."""
    return DeviceConfig(
        preferred_device="auto",
        use_fp16=True,
        use_quantization=False,
    )

def get_vlm_device_config(model_size: str = "7b") -> DeviceConfig:
    """Get optimized config for VLM models."""
    device = get_device()
    
    if device == "cpu":
        # CPU mode: recommend API or very small models
        return DeviceConfig(
            preferred_device="cpu",
            use_fp16=False,
            use_quantization=False,
        )
    
    # GPU mode: use quantization for larger models
    use_quantization = model_size in ("7b", "13b")
    
    return DeviceConfig(
        preferred_device="auto",
        use_fp16=True,
        use_quantization=use_quantization,
        quantization_bits=4,
    )

def print_device_info():
    info = get_device_info()
    print("Device information\n")

    print(f"Active Device: {info['device'].upper()}")
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"MPS Available: {info['mps_available']}")
    print(f"CPU Cores: {info['cpu_count']}")
    
    if info['gpu_name']:
        print(f"GPU Name: {info['gpu_name']}")
        print(f"GPU Memory: {info['gpu_memory_gb']:.1f} GB")