from dataclasses import dataclass, field
from typing import List, Optional, Any
from hydra.core.config_store import ConfigStore 
from omegaconf import MISSING

# 1. Dataset configuration schema
@dataclass
class QdrantConfig:
    """Configuration for the Qdrant Vector Database."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "roadbuddy_memory"
    use_hybrid: bool = True

    # Dense model for semantic search
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Sparse model for keyword-based search (SPLADE or BM25)
    sparse_model_name: str = "prithivida/Splade_PP_en_v1"
    api_key: Optional[str] = None

# 2. Perception configuration schema
@dataclass
class YOLOConfig:
    """Configuration for the YOLO11 Perception Engine."""
    model_path: str = "yolo11n.pt" # Path to the YOLO11 model weightss
    task: str = "detect"           # Options: "detect", "segment", "pose", "obb"
    confidence: float = 0.25       # Confidence threshold for detections
    iou_threshold: float = 0.45    # NMS Intersection over Union threshold
    device: str = "0"              # CUDA device index or "cpu"
    imgsz: int = 640               # Input image size (pixels)
    classes: Optional[List[int]] = None # Filter specific classes (e.g., 0 for person)
    half: bool = False              # Use FP16 inference
    tracker_config: str = "botsort.yaml" # Tracker configuration file

# 3. Ingestion Configuration Schema
@dataclass
class DecordConfig:
    """Configuration for Video Ingestion via Decord."""
    video_path: str = MISSING       # Must be provided at runtime
    batch_size: int = 16            # Number of frames to process at once
    width: int = -1                 # -1 implies original width
    height: int = -1                # -1 implies original height
    num_threads: int = 0            # 0 implies auto-detection
    device: str = "gpu"             # 'cpu' or 'gpu'
    ctx_id: int = 0                 # GPU device index for decoding

# 4. Reasoning Configuration Schema
@dataclass
class VLLMConfig:
    """Configuration for the vLLM Cognitive Engine."""
    api_base: str = "http://localhost:8000/v1"
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    api_key: str = "EMPTY"          # vLLM often uses dummy keys
    max_tokens: int = 128           # Limit output length for latency
    temperature: float = 0.1        # Low temperature for deterministic advice
    system_prompt: str = "You are a safety-critical autonomous driving assistant."

# 5. Root Configuration Schema
@dataclass
class RoadBuddyConfig:
    """Master Configuration Object."""
    db: QdrantConfig = field(default_factory=QdrantConfig)
    perception: YOLOConfig = field(default_factory=YOLOConfig)
    ingestion: DecordConfig = field(default_factory=DecordConfig)
    reasoning: VLLMConfig = field(default_factory=VLLMConfig)

    # Global settings
    debug: bool = False
    output_dir: str = "outputs"
    seed: int = 42

# Registering the configs allows Hydra to discover them by name
def register_configs():
    cs = ConfigStore.instance()
    # Register the root config
    cs.store(name="base_config", node=RoadBuddyConfig)
    
    # Register component groups
    cs.store(group="db", name="local_qdrant", node=QdrantConfig)
    cs.store(group="perception", name="yolo11n", node=YOLOConfig)
    cs.store(group="ingestion", name="decord_gpu", node=DecordConfig)
    cs.store(group="reasoning", name="vllm_docker", node=VLLMConfig)