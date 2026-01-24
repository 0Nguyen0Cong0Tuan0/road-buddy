"""
Constants and Vocabulary for Vietnamese Traffic Query Analysis.

This module contains all dictionaries, patterns, and constants used by
the query analyzer strategies.
"""
from typing import Dict, List, Tuple
from enum import Enum

class QuestionIntent(Enum):
    """Types of questions in Vietnamese traffic QA."""
    EXISTENCE = "existence"  # "Có biển báo không?" (Is there a sign?)
    DIRECTION = "direction"  # "Hướng nào?" (Which direction?)
    VALUE = "value"  # "Tốc độ bao nhiêu?" (What speed?)
    PERMISSION = "permission"  # "Có được phép không?" (Is it allowed?)
    IDENTIFICATION = "identification"  # "Biển báo gì?" (What sign?)
    COUNTING = "counting"  # "Có bao nhiêu?" (How many?)
    COMPARISON = "comparison"  # "Đúng hay sai?" (True or false?)
    TEMPORAL = "temporal"  # "Xuất hiện đầu tiên?" (Which appears first?)
    UNKNOWN = "unknown"

# Vietnamese Traffic Keywords -> Target Objects Mapping
VIETNAMESE_TRAFFIC_KEYWORDS: Dict[str, Dict] = {
    # === SIGNS ===
    "biển báo": {
        "objects": ["traffic_sign"],
        "yolo_classes": ["traffic sign"],
    },
    "biển cấm": {
        "objects": ["prohibitory_sign"],
        "yolo_classes": ["traffic sign"],
    },
    "biển báo cấm": {
        "objects": ["prohibitory_sign"],
        "yolo_classes": ["traffic sign"],
    },
    "biển chỉ dẫn": {
        "objects": ["direction_sign", "guide_sign"],
        "yolo_classes": ["traffic sign"],
    },
    "biển hiệu lệnh": {
        "objects": ["mandatory_sign", "command_sign"],
        "yolo_classes": ["traffic sign"],
    },
    "biển nguy hiểm": {
        "objects": ["warning_sign", "danger_sign"],
        "yolo_classes": ["traffic sign"],
    },
    "biển báo nguy hiểm": {
        "objects": ["warning_sign", "danger_sign"],
        "yolo_classes": ["traffic sign"],
    },
    "biển cảnh báo": {
        "objects": ["warning_sign"],
        "yolo_classes": ["traffic sign"],
    },
    "biển phụ": {
        "objects": ["supplementary_sign"],
        "yolo_classes": ["traffic sign"],
    },
    
    # === SPECIFIC SIGNS ===
    "tốc độ tối đa": {
        "objects": ["speed_limit_sign", "max_speed_sign"],
        "yolo_classes": ["traffic sign"],
    },
    "tốc độ tối thiểu": {
        "objects": ["min_speed_sign"],
        "yolo_classes": ["traffic sign"],
    },
    "giới hạn tốc độ": {
        "objects": ["speed_limit_sign"],
        "yolo_classes": ["traffic sign"],
    },
    "cấm dừng": {
        "objects": ["no_stopping_sign"],
        "yolo_classes": ["traffic sign"],
    },
    "cấm đỗ": {
        "objects": ["no_parking_sign"],
        "yolo_classes": ["traffic sign"],
    },
    "cấm dừng đỗ": {
        "objects": ["no_stopping_sign", "no_parking_sign"],
        "yolo_classes": ["traffic sign"],
    },
    "cấm quay đầu": {
        "objects": ["no_uturn_sign"],
        "yolo_classes": ["traffic sign"],
    },
    "cấm rẽ phải": {
        "objects": ["no_right_turn_sign"],
        "yolo_classes": ["traffic sign"],
    },
    "cấm rẽ trái": {
        "objects": ["no_left_turn_sign"],
        "yolo_classes": ["traffic sign"],
    },
    "cấm đi ngược chiều": {
        "objects": ["no_entry_sign"],
        "yolo_classes": ["traffic sign"],
    },
    
    # === TRAFFIC LIGHTS ===
    "đèn đỏ": {
        "objects": ["traffic_light_red", "red_light"],
        "yolo_classes": ["traffic light"],
    },
    "đèn xanh": {
        "objects": ["traffic_light_green", "green_light"],
        "yolo_classes": ["traffic light"],
    },
    "đèn vàng": {
        "objects": ["traffic_light_yellow", "yellow_light"],
        "yolo_classes": ["traffic light"],
    },
    "đèn giao thông": {
        "objects": ["traffic_light"],
        "yolo_classes": ["traffic light"],
    },
    "đèn tín hiệu": {
        "objects": ["traffic_light"],
        "yolo_classes": ["traffic light"],
    },
    
    # === LANES & ROADS ===
    "làn đường": {
        "objects": ["lane", "lane_marking"],
        "yolo_classes": ["road"],
    },
    "làn xe": {
        "objects": ["lane", "vehicle_lane"],
        "yolo_classes": ["road"],
    },
    "làn ngoài cùng bên phải": {
        "objects": ["rightmost_lane"],
        "yolo_classes": ["road"],
    },
    "làn ngoài cùng bên trái": {
        "objects": ["leftmost_lane"],
        "yolo_classes": ["road"],
    },
    "vạch kẻ đường": {
        "objects": ["road_marking", "lane_marking"],
        "yolo_classes": ["road"],
    },
    "vạch kẻ": {
        "objects": ["road_marking"],
        "yolo_classes": ["road"],
    },
    
    # === DIRECTIONS ===
    "rẽ phải": {
        "objects": ["right_turn", "direction_right"],
        "yolo_classes": ["traffic sign"],
    },
    "rẽ trái": {
        "objects": ["left_turn", "direction_left"],
        "yolo_classes": ["traffic sign"],
    },
    "đi thẳng": {
        "objects": ["go_straight", "direction_straight"],
        "yolo_classes": ["traffic sign"],
    },
    "quay đầu": {
        "objects": ["uturn", "direction_uturn"],
        "yolo_classes": ["traffic sign"],
    },
    "hướng đi": {
        "objects": ["direction", "direction_sign"],
        "yolo_classes": ["traffic sign"],
    },
    
    # === VEHICLES ===
    "xe ô tô": {
        "objects": ["car", "automobile"],
        "yolo_classes": ["car"],
    },
    "ô tô": {
        "objects": ["car", "automobile"],
        "yolo_classes": ["car"],
    },
    "xe máy": {
        "objects": ["motorcycle", "motorbike"],
        "yolo_classes": ["motorcycle"],
    },
    "xe mô tô": {
        "objects": ["motorcycle"],
        "yolo_classes": ["motorcycle"],
    },
    "xe gắn máy": {
        "objects": ["motorcycle", "scooter"],
        "yolo_classes": ["motorcycle"],
    },
    "xe tải": {
        "objects": ["truck"],
        "yolo_classes": ["truck"],
    },
    "xe khách": {
        "objects": ["bus", "passenger_bus"],
        "yolo_classes": ["bus"],
    },
    
    # === HIGHWAYS ===
    "cao tốc": {
        "objects": ["highway", "expressway"],
        "yolo_classes": ["road"],
    },
    "đường cao tốc": {
        "objects": ["highway", "expressway"],
        "yolo_classes": ["road"],
    },
    
    # === OTHER ===
    "nút giao": {
        "objects": ["intersection", "junction"],
        "yolo_classes": ["road"],
    },
    "ngã tư": {
        "objects": ["intersection", "crossroad"],
        "yolo_classes": ["road"],
    },
    "trạm xăng": {
        "objects": ["gas_station"],
        "yolo_classes": ["traffic sign"],
    },
}

# Question Intent Patterns (Ordered by Priority)

# ORDER MATTERS: more specific patterns first!
INTENT_PATTERNS_ORDERED: List[Tuple[QuestionIntent, List[str]]] = [
    # Most specific patterns first
    (QuestionIntent.TEMPORAL, [
        r"đầu tiên",
        r"cuối cùng",
        r"hiện tại",
        r"lúc nào",
    ]),
    (QuestionIntent.PERMISSION, [
        r"được phép",
        r"có được phép",
        r"cho phép",
        r"có được.*đi",  # "có được đi"
        r"có thể.*được",
    ]),
    (QuestionIntent.VALUE, [
        r"bao nhiêu",
        r"là bao",
        r"tốc độ.*là",
        r"mấy km",
    ]),
    (QuestionIntent.COMPARISON, [
        r"đúng hay sai",
        r"phải không$",  # Must be at end of question
        r"đúng không$",
    ]),
    (QuestionIntent.COUNTING, [
        r"có mấy",
        r"mấy loại",
        r"bao nhiêu loại",
    ]),
    (QuestionIntent.DIRECTION, [
        r"hướng nào",
        r"đi hướng",
        r"phải đi.*hướng",
        r"được đi.*hướng",
        r"rẽ.*hướng",
    ]),
    (QuestionIntent.IDENTIFICATION, [
        r"biển.*gì",
        r"là biển",
        r"loại.*nào",
        r"những.*nào",
    ]),
    # Most general pattern last (catch-all)
    (QuestionIntent.EXISTENCE, [
        r"có.*không\?*$",  # General existence at end
        r"có xuất hiện",
        r"xuất hiện.*không",
    ]),
]

# Dict version for backwards compatibility
INTENT_PATTERNS: Dict[QuestionIntent, List[str]] = {
    intent: patterns for intent, patterns in INTENT_PATTERNS_ORDERED
}

# Temporal Keywords
TEMPORAL_KEYWORDS: Dict[str, str] = {
    "đầu tiên": "first",
    "cuối cùng": "last",
    "hiện tại": "current",
    "đang": "current",
    "trước": "before",
    "sau": "after",
}


# Semantic Object Descriptions for PhoBERT Matching

# Vietnamese traffic object descriptions for semantic matching
# Each entry: object_type -> (Vietnamese description, YOLO classes)
SEMANTIC_OBJECT_DESCRIPTIONS: Dict[str, Tuple[str, List[str]]] = {
    # Signs
    "traffic_sign": ("biển báo giao thông trên đường", ["traffic sign"]),
    "speed_limit_sign": ("biển báo giới hạn tốc độ tối đa cho phép", ["traffic sign"]),
    "min_speed_sign": ("biển báo tốc độ tối thiểu phải đạt", ["traffic sign"]),
    "no_stopping_sign": ("biển cấm dừng xe trên đường", ["traffic sign"]),
    "no_parking_sign": ("biển cấm đỗ xe", ["traffic sign"]),
    "no_entry_sign": ("biển cấm đi vào đường một chiều", ["traffic sign"]),
    "no_uturn_sign": ("biển cấm quay đầu xe", ["traffic sign"]),
    "no_left_turn_sign": ("biển cấm rẽ trái", ["traffic sign"]),
    "no_right_turn_sign": ("biển cấm rẽ phải", ["traffic sign"]),
    "warning_sign": ("biển cảnh báo nguy hiểm phía trước", ["traffic sign"]),
    "direction_sign": ("biển chỉ dẫn hướng đi", ["traffic sign"]),
    "mandatory_sign": ("biển hiệu lệnh bắt buộc", ["traffic sign"]),
    
    # Traffic lights
    "traffic_light": ("đèn tín hiệu giao thông", ["traffic light"]),
    "red_light": ("đèn đỏ yêu cầu dừng lại", ["traffic light"]),
    "green_light": ("đèn xanh cho phép đi", ["traffic light"]),
    "yellow_light": ("đèn vàng chuẩn bị dừng", ["traffic light"]),
    
    # Lanes and roads
    "lane": ("làn đường cho xe đi", ["road"]),
    "lane_marking": ("vạch kẻ đường phân làn", ["road"]),
    "rightmost_lane": ("làn ngoài cùng bên phải", ["road"]),
    "leftmost_lane": ("làn ngoài cùng bên trái", ["road"]),
    "highway": ("đường cao tốc", ["road"]),
    
    # Vehicles
    "car": ("xe ô tô con", ["car"]),
    "motorcycle": ("xe máy xe mô tô", ["motorcycle"]),
    "truck": ("xe tải chở hàng", ["truck"]),
    "bus": ("xe buýt xe khách", ["bus"]),
    
    # Other
    "intersection": ("ngã tư nút giao thông", ["road"]),
    "pedestrian_crossing": ("vạch qua đường cho người đi bộ", ["road"]),
}

# Similarity threshold for semantic matching
SEMANTIC_SIMILARITY_THRESHOLD: float = 0.5

# Native sentence-transformers models (no wrapping needed)
NATIVE_SBERT_MODELS = {
    "dangvantuan/vietnamese-embedding",
    "keepitreal/vietnamese-sbert",
    "AITeamVN/Vietnamese_Embedding",
}