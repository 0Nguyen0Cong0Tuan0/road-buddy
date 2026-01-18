# Hydra-compatible configurations for RoadBuddy
#
# This file provides Hydra/OmegaConf compatible dataclasses.
# For centralized path management, see: config/settings.py

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

try:
    from hydra.core.config_store import ConfigStore
    from omegaconf import MISSING
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    MISSING = None

# Try to import centralized settings for defaults
try:
    from config.settings import get_config, PATHS
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
    PATHS = None


# ==================== Path Resolution ====================

def _get_project_root() -> Path:
    """Get project root directory."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / 'src').exists() or (parent / 'setup.py').exists():
            return parent
    return Path(__file__).resolve().parent.parent

PROJECT_ROOT = _get_project_root()


# ==================== Database Configuration ====================

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


# ==================== Perception Configuration ====================

@dataclass
class YOLOConfig:
    """Configuration for YOLO11 Perception Engine."""
    model_path: str = str(PROJECT_ROOT / "models" / "yolo11n.pt")
    task: str = "detect"
    confidence: float = 0.25
    iou_threshold: float = 0.45
    device: str = "0"
    imgsz: int = 640
    classes: Optional[List[int]] = None
    half: bool = False
    tracker_config: str = "botsort.yaml"


# ==================== Ingestion Configuration ====================

@dataclass
class SamplingConfig:
    """Configuration for frame sampling strategies.
    
    Design: Scale frame count with video duration.
    Longer videos = more frames (within bounds).
    """
    strategy: str = "adaptive"        # "uniform", "adaptive", "fps", "temporal_chunks"
    min_frames: int = 8               # Minimum frames for any video
    max_frames: int = 64              # Maximum frames for any video
    frames_per_second: float = 0.5    # Target rate for adaptive sampling
    target_fps: float = 1.0           # For FPS-based sampling
    num_chunks: int = 4               # For temporal chunk sampling
    frames_per_chunk: int = 2         # Frames per chunk


@dataclass
class DecordConfig:
    """Configuration for Video Ingestion via Decord.
    
    Note: width and height are kept at -1 (native resolution) by design.
    We do NOT resize during ingestion to preserve image quality.
    """
    video_path: str = MISSING if HYDRA_AVAILABLE else ""
    batch_size: int = 16
    width: int = -1               # -1 = native resolution (do not change)
    height: int = -1              # -1 = native resolution (do not change)
    num_threads: int = 0          # 0 = auto
    device: str = "gpu"           # "gpu" or "cpu"
    ctx_id: int = 0               # GPU device ID


# ==================== Reasoning Configuration ====================

@dataclass
class VLLMConfig:
    """Configuration for VLLM Cognitive Reasoning Engine."""
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


# ==================== Keyframe Configuration ====================

@dataclass
class KeyframeConfig:
    """Configuration for Dynamic Keyframe Selection.
    
    Note: Consider using SamplingConfig.adaptive strategy instead,
    which automatically scales frames with video duration.
    """
    base_fps: int = 1               # Base sampling rate
    max_keyframes: int = 8
    optical_flow_threshold: float = 5.0
    similarity_threshold: float = 0.95
    detection_trigger: bool = True


# ==================== Master Configuration ====================

@dataclass
class RoadBuddyConfig:
    """Master Configuration Object.
    
    For centralized path management, use:
        from config.settings import get_config, PATHS
    """
    db: QdrantConfig = field(default_factory=QdrantConfig)
    perception: YOLOConfig = field(default_factory=YOLOConfig)
    ingestion: DecordConfig = field(default_factory=DecordConfig)
    reasoning: VLLMConfig = field(default_factory=VLLMConfig)
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    # Global settings
    debug: bool = False
    output_dir: str = str(PROJECT_ROOT / "outputs")
    seed: int = 42
    max_latency: float = 28.0  # 2 seconds buffer for 30s limit


# ==================== Hydra Registration ====================

def register_configs():
    """Register all configurations to Hydra's ConfigStore."""
    if not HYDRA_AVAILABLE:
        return
    
    cs = ConfigStore.instance()
    cs.store(name="roadbuddy_config", node=RoadBuddyConfig)
    cs.store(group="db", name="qdrant", node=QdrantConfig)
    cs.store(group="perception", name="yolo", node=YOLOConfig)
    cs.store(group="ingestion", name="decord", node=DecordConfig)
    cs.store(group="reasoning", name="vllm", node=VLLMConfig)
    cs.store(group="keyframe", name="keyframe", node=KeyframeConfig)
    cs.store(group="sampling", name="sampling", node=SamplingConfig)
