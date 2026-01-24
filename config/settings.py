import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
load_dotenv()

# Path Resolution
def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    
    for parent in [current] + list(current.parents):
        if (parent / 'src').exists() or (parent / 'setup.py').exists():
            return parent
    return Path(__file__).resolve().parent.parent

# Path
@dataclass
class PathConfig:
    """All project paths centralized in one place."""
    # Core directories (auto-resolved from project root)
    project_root: Path = field(default_factory=get_project_root)
    
    @property
    def src_dir(self) -> Path:
        return self.project_root / "src"
    
    @property
    def config_dir(self) -> Path:
        return self.project_root / "config"

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
    
    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"
    
    @property
    def perception_models_dir(self) -> Path:
        return self.models_dir / "perception"
    
    @property
    def yolo_model_path(self) -> Path:
        return self.models_dir / "yolo11n.pt"
    
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
    
    @property
    def docs_dir(self) -> Path:
        return self.project_root / "docs"
    
    @property
    def explores_dir(self) -> Path:
        return self.project_root / "explores"
    
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

# Perception
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


# Query-Guided Perception

@dataclass
class QueryGuidedConfig:
    """Configuration for Query-Guided Perception Module.
    
    This module enables question-aware keyframe selection using:
    1. Query analysis to extract target objects from questions
    2. CLIP-based frame scoring for semantic relevance
    3. Multi-criteria keyframe selection
    
    Strategies are configurable as plugins - all options available,
    user selects defaults but can switch at runtime.
    """
    
    # Frame Selection Settings
    num_keyframes: int = 8        # Default per user request
    sample_fps: float = 2.0       # Frames per second to sample
    max_frames: int = 64          # Max frames to score
    
    # Strategy Selection (Plugins)
    # Query analysis strategy: "keyword" | "translation" | "semantic"
    query_strategy: str = "keyword"  # Default: fast keyword-based
    
    # Frame scoring strategy: "clip" | "mclip" | "detection" | "combined"
    scoring_strategy: str = "clip"  # Default: CLIP-based
    
    # Frame selection: "top_k" | "diverse_top_k" | "temporal_weighted"
    selection_strategy: str = "diverse_top_k"  # Select diverse high-scoring frames
    
    # YOLO Settings
    # YOLO detection mode: "all_frames" | "selected_only" | "none"
    yolo_mode: str = "selected_only"  # Default per user request (faster)
    yolo_model_name: str = "yolo11n_unified"  # Default fine-tuned model
    yolo_confidence: float = 0.25
    
    # CLIP Settings
    use_translation: bool = True  # Default per user request: translate Vietnamese→English
    translator: str = "googletrans"  # "googletrans" | "deep_translator"
    clip_model: str = "ViT-L/14"  # CLIP model to use
    
    # Scoring Weights
    # Combined score = alpha * QFS + beta * Detection + gamma * Distinctiveness
    alpha: float = 0.5           # Question-Frame Similarity weight
    beta: float = 0.3            # Detection boost weight
    gamma: float = 0.2           # Inter-frame distinctiveness weight
    
    # Selection Parameters
    diversity_threshold: float = 0.5  # Min distance between selected frames
    temporal_decay: float = 0.1       # Decay for temporal weighting
    
    # Semantic Strategy
    semantic_model: str = "vinai/phobert-base"  # For semantic extraction


# Reasoning
@dataclass
class ReasoningConfig:
    """VLM reasoning engine settings (Gemini API)."""
    # API settings - loaded from environment variable GOOGLE_API_KEY
    api_key: Optional[str] = field(default_factory=lambda: os.environ.get("GOOGLE_API_KEY"))
    
    # Model settings (Gemini)
    model_name: str = "gemini-2.0-flash"  # Options: gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro
    
    # Generation settings
    max_tokens: int = 2048
    temperature: float = 0.1
    
    # System prompt
    system_prompt: str = (
        "You are a legal expert in Vietnamese Road Traffic Law. "
        "Cite specific Articles and Clauses."
    )

# Master Configuration
@dataclass
class ProjectConfig:
    """Master configuration for the RoadBuddy project."""
    # Sub-configurations
    paths: PathConfig = field(default_factory=PathConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    query_guided: QueryGuidedConfig = field(default_factory=QueryGuidedConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    
    # Global settings
    debug: bool = False
    seed: int = 42
    max_latency: float = 28.0     # 2 seconds buffer for 30s limit
    
    def __post_init__(self):
        """Resolve model paths after initialization."""
        if self.perception.model_path is None:
            self.perception.model_path = str(self.paths.yolo_model_path)


# Configuration Accessors
_path_config: Optional[PathConfig] = None
_project_config: Optional[ProjectConfig] = None


def get_path_config() -> PathConfig:
    """Get the path configuration singleton."""
    global _path_config
    if _path_config is None:
        _path_config = PathConfig()
    return _path_config


def get_config() -> ProjectConfig:
    """Get the master configuration singleton."""
    global _project_config
    if _project_config is None:
        _project_config = ProjectConfig()
    return _project_config

# Pre-create singletons on module load
PATHS = get_path_config()
CONFIG = get_config()

PROJECT_ROOT = PATHS.project_root
DATA_DIR = PATHS.data_dir
MODELS_DIR = PATHS.models_dir
OUTPUTS_DIR = PATHS.outputs_dir