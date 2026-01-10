from dataclasses import dataclass, field
from typing import List, Optional
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

@dataclass
class QdrantConfig:
    """Configuration for the Qdrant Vector Database."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "roadbuddy_memory"
    use_hybrid: bool = True
    dense_model_name: str = "BAAI/bge-m3"
    sparse_model_name: str = "prithivida/Splade_PP_en_v1"
    api_key: Optional[str] = None
    embedding_dim: int = 1024  # bge-m3 dimension

@dataclass
class YOLOConfig:
    """Configuration for YOLO11 Perception Engine."""
    model_path: str = "yolov11n.pt"
    task: str = "detect"
    confidence: float = 0.25
    iou_threshold: float = 0.45
    device: str = "0"
    imgsz: int = 640
    classes: Optional[List[int]] = None
    half: bool = False
    tracker_config: str = "botsort.yaml"

@dataclass
class DecordConfig:
    """Configuration for Video Ingestion via Decord."""
    video_path: str = MISSING
    batch_size: int = 16
    width: int = -1
    height: int = -1
    num_threads: int = 0
    device: str = "gpu" 
    ctx_id: int = 0

@dataclass
class VLLMConfig:
    """Configuration for VLLM Congitive Reasoning Engine."""
    api_base: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    api_key: str = "EMPTY"
    max_tokens: int = 2048
    temperature: float = 0.1
    system_prompt: str = (
        "You are a legal expert in Vietnamese Road Traffic Law. "
        "Answer based ONLY on the provided context from Law 36/2024 and QCVN 41:2024. "
        "Cite specific Articles and Clauses."
    )

@dataclass
class KeyframeConfig:
    """Configuration for Dymamic Keyframe Selection."""
    base_fps: int = 1 # Base sampling rate
    max_keyframes: int = 8
    optical_flow_threshold: float = 5.0
    similarity_threshold: float = 0.95
    detection_trigger: bool = True

@dataclass
class RoadBuddyConfig:
    """Master Configuration Object."""
    db: QdrantConfig = field(default_factory=QdrantConfig)
    perception: YOLOConfig = field(default_factory=YOLOConfig)
    ingestion: DecordConfig = field(default_factory=DecordConfig)
    reasoning: VLLMConfig = field(default_factory=VLLMConfig)
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)

    # Global settings
    debug: bool = False
    output_dir: str = "outputs"
    seed: int = 42
    max_latency: float = 28.0  # 2 seconds buffer for 30s limit

def register_configs():
    """Register all configurations to Hydra's ConfigStore."""
    