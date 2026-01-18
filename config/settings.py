# Centralized Configuration for RoadBuddy Challenge
#
# This file is the single source of truth for all paths and configurations.
# Import settings from here instead of hardcoding paths in individual modules.

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


# ==================== Path Resolution ====================

def get_project_root() -> Path:
    """Get the project root directory.
    
    Works regardless of where the script is run from.
    Looks for markers like 'src' directory or 'setup.py'.
    """
    current = Path(__file__).resolve()
    
    # Walk up until we find the project root
    for parent in [current] + list(current.parents):
        # Check for common project markers
        if (parent / 'src').exists() or (parent / 'setup.py').exists():
            return parent
    
    # Fallback to parent of config directory
    return Path(__file__).resolve().parent.parent


# ==================== Path Configuration ====================

@dataclass
class PathConfig:
    """All project paths centralized in one place.
    
    Usage:
        >>> from config.settings import get_path_config
        >>> paths = get_path_config()
        >>> print(paths.models_dir)  # /path/to/project/models
    """
    # Core directories (auto-resolved from project root)
    project_root: Path = field(default_factory=get_project_root)
    
    # Source code
    @property
    def src_dir(self) -> Path:
        return self.project_root / "src"
    
    @property
    def config_dir(self) -> Path:
        return self.project_root / "config"
    
    # Data directories
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"
    
    @property
    def train_videos_dir(self) -> Path:
        return self.raw_data_dir / "train" / "videos"
    
    @property
    def test_videos_dir(self) -> Path:
        return self.raw_data_dir / "test" / "videos"
    
    @property
    def custom_data_dir(self) -> Path:
        return self.data_dir / "custom train data"
    
    # Model directories
    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"
    
    @property
    def perception_models_dir(self) -> Path:
        return self.models_dir / "perception"
    
    @property
    def yolo_model_path(self) -> Path:
        return self.models_dir / "yolo11n.pt"
    
    # Output directories
    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"
    
    @property
    def runs_dir(self) -> Path:
        return self.project_root / "runs"
    
    @property
    def logs_dir(self) -> Path:
        return self.outputs_dir / "logs"
    
    @property
    def cache_dir(self) -> Path:
        return self.project_root / ".cache"
    
    # Documentation and exploration
    @property
    def docs_dir(self) -> Path:
        return self.project_root / "docs"
    
    @property
    def explores_dir(self) -> Path:
        return self.project_root / "explores"
    
    # Tests
    @property
    def tests_dir(self) -> Path:
        return self.project_root / "tests"
    
    def ensure_dirs_exist(self) -> None:
        """Create output directories if they don't exist."""
        dirs_to_create = [
            self.outputs_dir,
            self.logs_dir,
            self.runs_dir,
            self.cache_dir
        ]
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def __str__(self) -> str:
        return f"PathConfig(project_root={self.project_root})"


# ==================== Ingestion Configuration ====================

@dataclass
class IngestionConfig:
    """Video ingestion settings.
    
    Design: NO resizing/cropping during ingestion.
    Only temporal sampling is performed.
    """
    # Sampling strategy
    sampling_strategy: str = "adaptive"  # "uniform", "adaptive", "fps", "temporal_chunks"
    
    # Adaptive sampling parameters
    min_frames: int = 8           # Minimum frames for any video
    max_frames: int = 64          # Maximum frames for any video
    frames_per_second: float = 0.5  # Target sampling rate (0.5 = 1 frame per 2 seconds)
    
    # FPS-based sampling
    target_fps: float = 1.0       # For fps sampling strategy
    
    # Temporal chunk sampling
    num_chunks: int = 4           # Number of temporal segments
    frames_per_chunk: int = 2     # Frames per segment
    
    # Video loading
    batch_size: int = 16          # Batch size for streaming
    num_threads: int = 0          # 0 = auto
    device: str = "gpu"           # "gpu" or "cpu"
    ctx_id: int = 0               # GPU device ID
    
    # Native resolution (no resizing)
    width: int = -1               # -1 = native resolution
    height: int = -1              # -1 = native resolution


# ==================== Perception Configuration ====================

@dataclass
class PerceptionConfig:
    """Object detection model settings."""
    # Model selection
    model_name: str = "yolo11n"
    model_path: Optional[str] = None  # Will be resolved from paths
    
    # Detection parameters
    task: str = "detect"
    confidence: float = 0.25
    iou_threshold: float = 0.45
    
    # Device settings
    device: str = "0"             # "0" for GPU:0, "cpu" for CPU
    half: bool = False            # FP16 inference
    
    # Input settings
    imgsz: int = 640              # Input image size
    
    # Filtering
    classes: Optional[List[int]] = None  # None = all classes
    
    # Tracking
    tracker_config: str = "botsort.yaml"


# ==================== Reasoning Configuration ====================

@dataclass
class ReasoningConfig:
    """VLM reasoning engine settings."""
    # API settings
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    
    # Generation settings
    max_tokens: int = 2048
    temperature: float = 0.1
    
    # System prompt
    system_prompt: str = (
        "You are a legal expert in Vietnamese Road Traffic Law. "
        "Answer based ONLY on the provided context from Law 36/2024 and QCVN 41:2024. "
        "Cite specific Articles and Clauses."
    )


# ==================== Database Configuration ====================

@dataclass
class DatabaseConfig:
    """Vector database settings (Qdrant)."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "roadbuddy_memory"
    
    # Embedding models
    use_hybrid: bool = True
    dense_model_name: str = "BAAI/bge-m3"
    sparse_model_name: str = "prithivida/Splade_PP_en_v1"
    embedding_dim: int = 1024     # bge-m3 dimension
    
    # Connection
    api_key: Optional[str] = None


# ==================== Master Configuration ====================

@dataclass
class ProjectConfig:
    """Master configuration for the RoadBuddy project.
    
    Usage:
        >>> from config.settings import get_config
        >>> config = get_config()
        >>> print(config.paths.models_dir)
        >>> print(config.ingestion.min_frames)
    """
    # Sub-configurations
    paths: PathConfig = field(default_factory=PathConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Global settings
    debug: bool = False
    seed: int = 42
    max_latency: float = 28.0     # 2 seconds buffer for 30s limit
    
    def __post_init__(self):
        """Resolve model paths after initialization."""
        if self.perception.model_path is None:
            self.perception.model_path = str(self.paths.yolo_model_path)


# ==================== Configuration Accessors ====================

# Singleton instances
_path_config: Optional[PathConfig] = None
_project_config: Optional[ProjectConfig] = None


def get_path_config() -> PathConfig:
    """Get the path configuration singleton.
    
    Returns:
        PathConfig instance with all project paths
    """
    global _path_config
    if _path_config is None:
        _path_config = PathConfig()
    return _path_config


def get_config() -> ProjectConfig:
    """Get the master configuration singleton.
    
    Returns:
        ProjectConfig instance with all settings
    """
    global _project_config
    if _project_config is None:
        _project_config = ProjectConfig()
    return _project_config


def reset_config() -> None:
    """Reset configuration singletons (useful for testing)."""
    global _path_config, _project_config
    _path_config = None
    _project_config = None


# ==================== For Convenient Import ====================

# Pre-create singletons on module load
PATHS = get_path_config()
CONFIG = get_config()


# ==================== Quick Access Constants ====================

# These can be imported directly: from config.settings import PROJECT_ROOT
PROJECT_ROOT = PATHS.project_root
DATA_DIR = PATHS.data_dir
MODELS_DIR = PATHS.models_dir
OUTPUTS_DIR = PATHS.outputs_dir


if __name__ == "__main__":
    # Quick test
    config = get_config()
    print(f"Project Root: {config.paths.project_root}")
    print(f"Data Dir: {config.paths.data_dir}")
    print(f"Models Dir: {config.paths.models_dir}")
    print(f"YOLO Model: {config.perception.model_path}")
    print(f"Ingestion Strategy: {config.ingestion.sampling_strategy}")
    print(f"Min Frames: {config.ingestion.min_frames}")
    print(f"Max Frames: {config.ingestion.max_frames}")
