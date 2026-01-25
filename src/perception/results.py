"""
Detection Result Utilities.

Provides dataclasses and utilities for parsing and manipulating YOLO detection results.

Usage:
    from src.perception.results import Detection, FrameDetections, parse_yolo_results
    
    # Parse YOLO results
    detections = parse_yolo_results(yolo_results)
    
    # Access detections
    for frame in detections:
        for det in frame.detections:
            print(f"{det.class_name}: {det.confidence:.2f}")
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

@dataclass
class Detection:
    """
    Single object detection result.
    
    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in pixels
        confidence: Detection confidence score (0-1)
        class_id: Class ID (integer)
        class_name: Human-readable class name
        track_id: Optional tracking ID for multi-object tracking
    """
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    
    @property
    def x1(self) -> float:
        """Left x coordinate."""
        return self.bbox[0]
    
    @property
    def y1(self) -> float:
        """Top y coordinate."""
        return self.bbox[1]
    
    @property
    def x2(self) -> float:
        """Right x coordinate."""
        return self.bbox[2]
    
    @property
    def y2(self) -> float:
        """Bottom y coordinate."""
        return self.bbox[3]
    
    @property
    def width(self) -> float:
        """Bounding box width."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Bounding box height."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """Bounding box area in pixels squared."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Center point (x, y) of bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "center": self.center
        }
    
    def to_yolo_format(self, img_width: int, img_height: int) -> Tuple[int, float, float, float, float]:
        """Convert to YOLO annotation format (class_id, x_center, y_center, width, height)."""
        cx, cy = self.center
        w, h = self.width, self.height
        return (
            self.class_id,
            cx / img_width,
            cy / img_height,
            w / img_width,
            h / img_height
        )


@dataclass  
class FrameDetections:
    """
    All detections for a single frame.
    
    Attributes:
        frame_idx: Frame index in the video
        detections: List of Detection objects
        image_size: Optional (width, height) of the frame
    """
    frame_idx: int
    detections: List[Detection] = field(default_factory=list)
    image_size: Optional[Tuple[int, int]] = None
    
    @property
    def num_detections(self) -> int:
        """Number of detections in this frame."""
        return len(self.detections)
    
    @property
    def class_counts(self) -> Dict[str, int]:
        """Count of detections per class."""
        counts: Dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts
    
    @property
    def unique_classes(self) -> List[str]:
        """List of unique class names detected."""
        return list(set(det.class_name for det in self.detections))
    
    def filter_by_class(self, class_name: str) -> List[Detection]:
        """Get detections of a specific class."""
        return [det for det in self.detections if det.class_name == class_name]
    
    def filter_by_confidence(self, min_confidence: float) -> List[Detection]:
        """Get detections above a confidence threshold."""
        return [det for det in self.detections if det.confidence >= min_confidence]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frame_idx": self.frame_idx,
            "num_detections": self.num_detections,
            "detections": [det.to_dict() for det in self.detections],
            "class_counts": self.class_counts,
            "image_size": self.image_size
        }

def parse_yolo_results(results: List, start_frame_idx: int = 0) -> List[FrameDetections]:
    """Parse YOLO results into structured FrameDetections."""
    frame_detections = []
    
    for idx, result in enumerate(results):
        frame_idx = start_frame_idx + idx
        
        # Get image size
        img_size = None
        if hasattr(result, 'orig_shape'):
            h, w = result.orig_shape
            img_size = (w, h)
        
        detections = []
        
        # Parse boxes if available
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            
            # Get class names mapping
            names = result.names if hasattr(result, 'names') else {}
            
            for i in range(len(boxes)):
                # Extract bbox (xyxy format)
                xyxy = boxes.xyxy[i].cpu().numpy()
                bbox = tuple(float(x) for x in xyxy)
                
                # Extract confidence
                conf = float(boxes.conf[i].cpu().numpy())
                
                # Extract class info
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = names.get(cls_id, str(cls_id))
                
                # Extract track ID if available
                track_id = None
                if hasattr(boxes, 'id') and boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())
                
                detection = Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    track_id=track_id
                )
                detections.append(detection)
        
        frame_detections.append(FrameDetections(
            frame_idx=frame_idx,
            detections=detections,
            image_size=img_size
        ))
    
    return frame_detections

def aggregate_detections(frame_detections: List[FrameDetections]) -> Dict[str, Any]:
    """Aggregate statistics across multiple frames."""
    total_detections = 0
    class_counts: Dict[str, int] = {}
    confidence_scores: List[float] = []
    
    for frame in frame_detections:
        total_detections += frame.num_detections
        
        for det in frame.detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
            confidence_scores.append(det.confidence)
    
    return {
        "num_frames": len(frame_detections),
        "total_detections": total_detections,
        "avg_detections_per_frame": total_detections / max(1, len(frame_detections)),
        "class_counts": class_counts,
        "unique_classes": list(class_counts.keys()),
        "avg_confidence": np.mean(confidence_scores) if confidence_scores else 0.0,
        "min_confidence": np.min(confidence_scores) if confidence_scores else 0.0,
        "max_confidence": np.max(confidence_scores) if confidence_scores else 0.0,
    }

def detections_to_annotations(frame_detections: FrameDetections, img_width: int, img_height: int) -> List[str]:
    """Convert frame detections to YOLO annotation format."""
    annotations = []
    for det in frame_detections.detections:
        cls_id, cx, cy, w, h = det.to_yolo_format(img_width, img_height)
        annotations.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return annotations
