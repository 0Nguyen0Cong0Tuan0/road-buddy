"""
VLM Client for Road Buddy VQA.

Supports both Gemini API and local Qwen2.5-VL models.

Usage:
    from src.reasoning.vlm_client import create_vlm_client
    
    # Use Gemini API (fast, requires API key)
    client = create_vlm_client(model_name="gemini-2.0-flash")
    
    # Use local Qwen model (no API needed, requires GPU)
    client = create_vlm_client(model_name="qwen2.5-vl-7b")
    
    response = client.generate(frames, prompt)
    print(response.text)
"""
import os
import logging
from io import BytesIO
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Load config from centralized models.yaml
try:
    from config.settings import get_vlm_config, get_vllm_config
    _vlm_yaml = get_vlm_config()
    _vllm_yaml = get_vllm_config()
except ImportError:
    _vlm_yaml = {}
    _vllm_yaml = {}

# Configuration
@dataclass
class VLMConfig:
    """Configuration for VLM client. Defaults loaded from config/models.yaml."""
    model: str = field(default_factory=lambda: _vlm_yaml.get("default", "qwen2.5-vl-7b-awq"))
    api_key: Optional[str] = None
    max_tokens: int = field(default_factory=lambda: _vlm_yaml.get("max_tokens", 256))
    temperature: float = field(default_factory=lambda: _vlm_yaml.get("temperature", 0.1))
    device: str = "auto"
    use_quantization: bool = True
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("GOOGLE_API_KEY")

@dataclass
class VLMResponse:
    """Response from VLM generation."""
    text: str
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
        }

# Base Client
class BaseVLMClient(ABC):
    """Abstract base class for VLM clients."""
    
    @abstractmethod
    def generate(self, frames: List[np.ndarray], prompt: str, **kwargs) -> VLMResponse:
        """Generate response from frames and prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if client is ready."""
        pass

# Gemini Client
class GeminiVLMClient(BaseVLMClient):
    """
    Gemini Vision-Language Model client (API-based).
    
    Uses Google's genai SDK for multimodal inference.
    Supports: gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro
    """
    
    SUPPORTED_MODELS = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite", 
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
    ]
    
    def __init__(self, config: Optional[VLMConfig] = None):
        self.config = config or VLMConfig()
        self._client = None
        self._available = False
        self._initialize()
    
    def _initialize(self):
        """Initialize Google GenAI client."""
        if not self.config.api_key:
            logger.warning("No API key provided. Set GOOGLE_API_KEY env var.")
            return
        
        try:
            from google import genai
            self._genai = genai
            self._client = genai.Client(api_key=self.config.api_key)
            self._available = True
            logger.info(f"Gemini client initialized: {self.config.model}")
        except ImportError:
            logger.error("google-genai not installed. Run: pip install google-genai")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
    
    def _encode_frame(self, frame: np.ndarray):
        """Encode numpy frame to Gemini-compatible Part."""
        from google.genai import types
        
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        
        image = Image.fromarray(frame.astype(np.uint8))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")
    
    def generate(self, frames: List[np.ndarray], prompt: str, **kwargs) -> VLMResponse:
        if not self._available:
            return VLMResponse(text="", model=self.config.model)
        
        # Retry configuration
        max_retries = kwargs.get("max_retries", 3)
        base_delay = kwargs.get("base_delay", 60)  # Start with 60s for rate limits
        
        for attempt in range(max_retries + 1):
            try:
                from google.genai import types
                
                contents = [self._encode_frame(f) for f in frames]
                contents.append(prompt)
                
                response = self._client.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        max_output_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                    )
                )
                
                text = response.text if response.text else ""
                usage = {}
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = {
                        "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                        "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                        "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0),
                    }
                
                return VLMResponse(text=text, model=self.config.model, usage=usage)
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error (429)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries:
                        # Extract retry delay from error if available
                        import re
                        import time
                        
                        delay_match = re.search(r'retry in (\d+)', error_str.lower())
                        if delay_match:
                            delay = int(delay_match.group(1)) + 5  # Add buffer
                        else:
                            delay = base_delay * (2 ** attempt)  # Exponential backoff
                        
                        logger.warning(f"Rate limited. Waiting {delay}s before retry ({attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                
                logger.error(f"Gemini generation failed: {e}")
                return VLMResponse(text="", model=self.config.model)
    
    def is_available(self) -> bool:
        return self._available
    
    def __repr__(self) -> str:
        return f"GeminiVLMClient(model={self.config.model}, available={self._available})"


# Qwen2.5-VL Client
class Qwen2VLClient(BaseVLMClient):
    """
    Qwen2.5-VL local model client.
    
    Runs Qwen2.5-VL-7B-Instruct locally using HuggingFace Transformers.
    Supports CPU and GPU inference with optional quantization.
    """
    
    # Default model ID, but can be overridden by config
    DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    def __init__(self, config: Optional[VLMConfig] = None):
        self.config = config or VLMConfig(model="qwen2.5-vl-7b")
        
        # Use model from config, or default
        self._model_id = self._resolve_model_id(self.config.model)
        
        self._model = None
        self._processor = None
        self._available = False
        self._device = None
        self._initialize()
    
    def _resolve_model_id(self, model_name: str) -> str:
        """Resolve model name to HuggingFace model ID."""
        # If already a full HF path, use as-is
        if model_name.startswith("Qwen/"):
            return model_name
        
        # Map short names to full HF paths
        model_map = {
            "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
            "qwen2.5-vl-7b-instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
            "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
            "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
            "qwen2.5-vl-2b": "Qwen/Qwen2.5-VL-3B-Instruct",  # 2B doesn't exist, use 3B
            "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        }
        
        return model_map.get(model_name.lower(), self.DEFAULT_MODEL_ID)
    
    def _initialize(self):
        """Load Qwen2.5-VL model."""
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            
            # Determine device
            if self.config.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.config.device
            
            logger.info(f"Loading {self._model_id} on {self._device}...")
            
            # Model loading kwargs
            model_kwargs = {
                "trust_remote_code": True,
            }
            
            # Add quantization for GPU
            if self._device == "cuda" and self.config.use_quantization:
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    model_kwargs["device_map"] = "auto"
                    logger.info("Using 4-bit quantization")
                except ImportError:
                    logger.warning("bitsandbytes not available, loading without quantization")
                    model_kwargs["torch_dtype"] = torch.float16
                    model_kwargs["device_map"] = "auto"
            elif self._device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
            else:
                # CPU - use float32
                model_kwargs["torch_dtype"] = torch.float32
            
            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self._model_id, 
                trust_remote_code=True
            )
            
            # Load model using AutoModelForImageTextToText (fixes deprecation)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._model_id,
                **model_kwargs
            )
            
            if self._device == "cpu" and "device_map" not in model_kwargs:
                self._model = self._model.to("cpu")
                logger.warning("Running on CPU - inference will be slow")
            
            self._available = True
            logger.info(f"{self._model_id} loaded successfully on {self._device}")
            
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            logger.error("Install: pip install transformers qwen-vl-utils accelerate")
        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-VL: {e}")
    
    def _prepare_messages(self, frames: List[np.ndarray], prompt: str, max_frames: int = 4, max_image_size: int = 448) -> List[Dict]:
        """
        Prepare messages in Qwen chat format.
        
        Args:
            frames: List of video frames
            prompt: Text prompt
            max_frames: Maximum frames to include (to avoid token limit)
            max_image_size: Maximum image dimension (smaller = fewer tokens)
        """
        content = []
        
        # Limit number of frames to avoid token overflow
        if len(frames) > max_frames:
            # Sample evenly spaced frames
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]
            logger.info(f"Reduced frames from {len(frames)} to {max_frames} to fit token limit")
        
        # Add images (resized to reduce token count)
        for frame in frames:
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            
            image = Image.fromarray(frame.astype(np.uint8))
            
            # Resize to reduce token count
            h, w = image.size
            if max(h, w) > max_image_size:
                scale = max_image_size / max(h, w)
                new_size = (int(h * scale), int(w * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            content.append({"type": "image", "image": image})
        
        # Add text
        content.append({"type": "text", "text": prompt})
        
        return [{"role": "user", "content": content}]
    
    def generate(self, frames: List[np.ndarray], prompt: str, **kwargs) -> VLMResponse:
        if not self._available:
            return VLMResponse(text="", model="qwen2.5-vl-7b")
        
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            
            messages = self._prepare_messages(frames, prompt)
            
            # Apply chat template
            text = self._processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process images
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Tokenize
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Move to device
            inputs = inputs.to(self._model.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature if self.config.temperature > 0 else None,
                    do_sample=self.config.temperature > 0,
                )
            
            # Decode output (remove input tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return VLMResponse(
                text=output_text,
                model="qwen2.5-vl-7b",
                usage={
                    "prompt_tokens": inputs.input_ids.shape[1],
                    "completion_tokens": generated_ids_trimmed[0].shape[0],
                    "total_tokens": inputs.input_ids.shape[1] + generated_ids_trimmed[0].shape[0],
                },
                finish_reason="stop",
            )
            
        except Exception as e:
            logger.error(f"Qwen generation failed: {e}")
            return VLMResponse(text="", model="qwen2.5-vl-7b")
    
    def is_available(self) -> bool:
        return self._available
    
    def __repr__(self) -> str:
        return f"Qwen2VLClient(device={self._device}, available={self._available})"


# Qwen2.5-VL vLLM Client
class Qwen2VLLMClient(BaseVLMClient):
    """
    Qwen2.5-VL client using vLLM backend (High Performance).
    
    Requires 'vllm' package: pip install vllm
    """
    
    def __init__(self, config: Optional[VLMConfig] = None):
        self.config = config or VLMConfig(model="qwen2.5-vl-7b")
        self._llm = None
        self._available = False
        self._model_loaded = False
        # NOTE: Lazy loading - vLLM is initialized on first load_model() call
        # This allows CLIP to be unloaded first, freeing GPU memory
    
    def _initialize(self):
        try:
            from vllm import LLM, SamplingParams
            import torch
            
            logger.info(f"Initializing vLLM with model: {self.config.model}")
            
            # Handle model aliases for vLLM loading
            model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
            if "awq" in self.config.model.lower():
                model_path = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
                logger.info("Using AWQ-quantized model for reduced memory usage")

            # Load vLLM parameters from centralized config
            gpu_memory_utilization = _vllm_yaml.get("gpu_memory_utilization", 0.85)
            max_model_len = _vllm_yaml.get("max_model_len", 4096)
            enforce_eager = _vllm_yaml.get("enforce_eager", True)
            limit_images = _vllm_yaml.get("limit_images", 10)

            self._llm = LLM(
                model=model_path,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": limit_images},
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
            )
            
            self._available = True
            self._model_loaded = True
            logger.info(f"vLLM initialized successfully with model: {model_path}")
            
        except ImportError:
            logger.error("vLLM not installed. Run: pip install vllm")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")

    def generate(self, frames: List[np.ndarray], prompt: str, **kwargs) -> VLMResponse:
        if not self._available:
             return VLMResponse(text="", model=self.config.model)

        try:
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stop_token_ids=None
            )
            
            # Prepare inputs for vLLM
            # vLLM expects a prompt string and multi-modal data
            # Construct a conversation format if the model expects it, 
            # currently vLLM's chat support is evolving, using the standard apply_chat_template logic internally if possible
            # But LLM.generate takes prompt_token_ids or prompt.
            # For Qwen2-VL, we often construct the raw prompt with special tokens.
            
            # However, vLLM now supports "chat" mode via `llm.chat` in newer versions or we can use the tokenizer.
            # Let's try to use the tokenizer from vLLM's internal HF model to format the prompt.
            
            tokenizer = self._llm.get_tokenizer()
            
            messages = []
            for frame in frames:
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                elif frame.shape[-1] == 4:
                    frame = frame[:, :, :3]
                image = Image.fromarray(frame.astype(np.uint8))
                messages.append({"type": "image", "image": image})
            
            messages.append({"type": "text", "text": prompt})
            
            chat_message = [{"role": "user", "content": messages}]
            
            # Apply chat template
            prompt_text = tokenizer.apply_chat_template(
                chat_message, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # vLLM generate expects 'multi_modal_data'
            # We need to extract images back out for the multi_modal_data dict
            # Qwen2-VL supports multiple images.
            # vLLM expects: multi_modal_data={"image": [img1, img2, ...]} or single image.
            
            multi_modal_data = {"image": [msg["image"] for msg in messages if msg["type"] == "image"]}
            if len(multi_modal_data["image"]) == 1:
                multi_modal_data["image"] = multi_modal_data["image"][0] # Unwrap if single
            elif len(multi_modal_data["image"]) == 0:
                multi_modal_data = None
                
            outputs = self._llm.generate(
                {
                    "prompt": prompt_text,
                    "multi_modal_data": multi_modal_data,
                },
                sampling_params=sampling_params,
                use_tqdm=False
            )
            
            generated_text = outputs[0].outputs[0].text
            
            usage = {
                "prompt_tokens": len(outputs[0].prompt_token_ids),
                "completion_tokens": len(outputs[0].outputs[0].token_ids),
                "total_tokens": len(outputs[0].prompt_token_ids) + len(outputs[0].outputs[0].token_ids)
            }
            
            return VLMResponse(
                text=generated_text, 
                model=self.config.model,
                usage=usage,
                finish_reason=outputs[0].outputs[0].finish_reason
            )

        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            return VLMResponse(text="", model=self.config.model)

    def is_available(self) -> bool:
        return self._available
    
    def load_model(self):
        """Load vLLM model (called lazily)."""
        if not self._model_loaded:
            self._initialize()
    
    def unload_model(self):
        """Unload vLLM model to free GPU memory."""
        if self._llm is not None:
            logger.info("Unloading vLLM model...")
            import gc
            import torch
            
            # vLLM doesn't have a clean unload, but we can delete the reference
            del self._llm
            self._llm = None
            self._available = False
            self._model_loaded = False
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            logger.info("vLLM model unloaded")

# Model name mappings
MODEL_ALIASES = {
    # Gemini models
    "gemini": "gemini-2.0-flash",
    "gemini-flash": "gemini-2.0-flash",
    "gemini-pro": "gemini-1.5-pro",
    # Qwen models
    "qwen": "qwen2.5-vl-7b",
    "qwen2.5-vl": "qwen2.5-vl-7b",
    "qwen2.5-vl-7b": "qwen2.5-vl-7b",
    "qwen2-vl-7b": "qwen2.5-vl-7b",
    "local": "qwen2.5-vl-7b",
    # AWQ quantized models (recommended for memory efficiency)
    "qwen2.5-vl-7b-awq": "qwen2.5-vl-7b-awq",
    "qwen-awq": "qwen2.5-vl-7b-awq",
}


def create_vlm_client(
    model_name: str = "qwen2.5-vl-7b",  # Local model by default
    device: str = "auto",
    api_key: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.1,
    use_quantization: bool = True,
    backend: str = "transformers", # "transformers" or "vllm"
    **kwargs
) -> BaseVLMClient:
    """
    Create VLM client.
    
    Args:
        model_name: Model to use:
            - "qwen2.5-vl-7b" or "local" (local model, default)
            - "gemini-2.0-flash" (API, fast, requires GOOGLE_API_KEY)
        device: Device for local models ("auto", "cuda", "cpu")
        api_key: Google API key (for Gemini)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        use_quantization: Use 4-bit quantization for local models (transformers backend only)
        backend: Backend engine: "transformers" (default) or "vllm" (faster)
    
    Returns:
        VLM client instance
    """
    # Resolve aliases
    resolved_name = MODEL_ALIASES.get(model_name.lower(), model_name)
    
    config = VLMConfig(
        model=resolved_name,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        device=device,
        use_quantization=use_quantization,
    )
    
    if resolved_name.startswith("gemini"):
        return GeminiVLMClient(config)
    elif "qwen" in resolved_name.lower() or resolved_name == "local":
        if backend == "vllm":
            return Qwen2VLLMClient(config)
        else:
            return Qwen2VLClient(config)
    else:
        logger.warning(f"Unknown model '{model_name}', defaulting to Gemini")
        return GeminiVLMClient(config)