"""
Keyword Extraction Strategy.

Fast rule-based keyword extraction for Vietnamese traffic questions.
Uses a comprehensive dictionary of Vietnamese traffic terms to identify
target objects, question intent, and map to YOLO classes.
"""
import re
import logging
from typing import List, Set

from ..base import ExtractionStrategy
from ..models import QueryAnalysisResult
from ..constants import (
    VIETNAMESE_TRAFFIC_KEYWORDS,
    INTENT_PATTERNS_ORDERED,
    TEMPORAL_KEYWORDS,
    QuestionIntent,
)

logger = logging.getLogger(__name__)

class KeywordExtractionStrategy(ExtractionStrategy):
    """
    Fast rule-based keyword extraction for Vietnamese traffic questions.
    
    Uses a comprehensive dictionary of Vietnamese traffic terms to identify
    target objects, question intent, and map to YOLO classes.
    
    This is the fastest strategy and serves as the baseline for other strategies.
    """
    
    def __init__(self):
        self._keywords = VIETNAMESE_TRAFFIC_KEYWORDS
        self._intent_patterns_ordered = INTENT_PATTERNS_ORDERED
        self._temporal_keywords = TEMPORAL_KEYWORDS
    
    @property
    def name(self) -> str:
        return "keyword"
    
    def extract(self, question: str) -> QueryAnalysisResult:
        """Extract target objects and intent using keyword matching."""
        question_lower = question.lower()
        
        # Find matching keywords
        target_objects: Set[str] = set()
        yolo_classes: Set[str] = set()
        keywords_found: List[str] = []
        
        # Sort keywords by length (longer first) to match more specific terms first
        sorted_keywords = sorted(
            self._keywords.keys(), 
            key=len, 
            reverse=True
        )
        
        for keyword in sorted_keywords:
            if keyword in question_lower:
                keywords_found.append(keyword)
                target_objects.update(self._keywords[keyword]["objects"])
                yolo_classes.update(self._keywords[keyword]["yolo_classes"])
        
        # Detect question intent
        intent = self._detect_intent(question_lower)
        
        # Extract temporal hints
        temporal_hints = self._extract_temporal_hints(question_lower)
        
        # Calculate confidence based on matches
        confidence = min(1.0, len(keywords_found) * 0.3 + 0.1)
        
        return QueryAnalysisResult(
            original_question=question,
            translated_question=None,
            target_objects=list(target_objects),
            question_intent=intent,
            yolo_classes=list(yolo_classes),
            keywords_found=keywords_found,
            confidence=confidence,
            temporal_hints=temporal_hints
        )
    
    def _detect_intent(self, question: str) -> QuestionIntent:
        """Detect the intent/type of the question. Uses ordered patterns for priority."""
        for intent, patterns in self._intent_patterns_ordered:
            for pattern in patterns:
                if re.search(pattern, question):
                    return intent
        return QuestionIntent.UNKNOWN
    
    def _extract_temporal_hints(self, question: str) -> List[str]:
        """Extract temporal hints from the question."""
        hints = []
        for vn_keyword, en_hint in self._temporal_keywords.items():
            if vn_keyword in question:
                hints.append(en_hint)
        return hints