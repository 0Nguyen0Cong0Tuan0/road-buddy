"""
Data Models for Query Analysis.

This module contains the data classes and result types used by
the query analyzer.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .constants import QuestionIntent

@dataclass
class QueryAnalysisResult:
    """
    Result of analyzing a Vietnamese traffic question.
    
    Attributes:
        original_question: The original Vietnamese question
        translated_question: English translation (if applicable)
        target_objects: List of target object types to look for
        question_intent: The type/intent of the question
        yolo_classes: Mapped YOLO class names for detection
        keywords_found: Vietnamese keywords that were matched
        confidence: Confidence score of the analysis (0-1)
        temporal_hints: Any temporal information (first, last, current)
    """
    original_question: str
    translated_question: Optional[str] = None
    target_objects: List[str] = field(default_factory=list)
    question_intent: QuestionIntent = QuestionIntent.UNKNOWN
    yolo_classes: List[str] = field(default_factory=list)
    keywords_found: List[str] = field(default_factory=list)
    confidence: float = 0.0
    temporal_hints: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "original_question": self.original_question,
            "translated_question": self.translated_question,
            "target_objects": self.target_objects,
            "question_intent": self.question_intent.value,
            "yolo_classes": self.yolo_classes,
            "keywords_found": self.keywords_found,
            "confidence": self.confidence,
            "temporal_hints": self.temporal_hints
        }