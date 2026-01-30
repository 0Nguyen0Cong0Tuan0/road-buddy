"""
Keyword Extraction Strategy.

Fast rule-based extraction using Vietnamese traffic vocabulary.
"""
import re
import logging
from typing import List, Set

from ..base import ExtractionStrategy
from ..models import QueryAnalysisResult
from ..constants import (
    QuestionIntent,
    VIETNAMESE_TRAFFIC_KEYWORDS,
    INTENT_PATTERNS_ORDERED,
    TEMPORAL_KEYWORDS,
)

logger = logging.getLogger(__name__)


class KeywordExtractionStrategy(ExtractionStrategy):
    """
    Fast rule-based keyword extraction for Vietnamese traffic questions.
    
    Uses predefined vocabulary to extract:
    - Target objects (signs, lights, lanes, vehicles)
    - Question intent (existence, direction, value, permission)
    - Temporal hints (first, last, current)
    """
    
    def __init__(self):
        """Initialize keyword extractor."""
        self._compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        compiled = {}
        for intent, patterns in INTENT_PATTERNS_ORDERED:
            compiled[intent] = [re.compile(p, re.IGNORECASE) for p in patterns]
        return compiled
    
    @property
    def name(self) -> str:
        return "keyword"
    
    def extract(self, question: str) -> QueryAnalysisResult:
        """Extract information from Vietnamese question using keywords."""
        question_lower = question.lower()
        
        # Extract target objects
        target_objects: List[str] = []
        yolo_classes: Set[str] = set()
        keywords_found: List[str] = []
        
        for keyword, info in VIETNAMESE_TRAFFIC_KEYWORDS.items():
            if keyword in question_lower:
                target_objects.extend(info["objects"])
                yolo_classes.update(info["yolo_classes"])
                keywords_found.append(keyword)
        
        # Determine question intent (check in priority order)
        intent = QuestionIntent.UNKNOWN
        for check_intent, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(question_lower):
                    intent = check_intent
                    break
            if intent != QuestionIntent.UNKNOWN:
                break
        
        # Extract temporal hints
        temporal_hints = []
        for vn_keyword, en_meaning in TEMPORAL_KEYWORDS.items():
            if vn_keyword in question_lower:
                temporal_hints.append(en_meaning)
        
        # Calculate confidence based on matches
        confidence = 0.0
        if keywords_found:
            confidence += 0.4
        if intent != QuestionIntent.UNKNOWN:
            confidence += 0.4
        if temporal_hints:
            confidence += 0.2
        
        return QueryAnalysisResult(
            original_question=question,
            translated_question=None,
            target_objects=list(set(target_objects)),  # Deduplicate
            question_intent=intent,
            yolo_classes=list(yolo_classes),
            keywords_found=keywords_found,
            confidence=confidence,
            temporal_hints=temporal_hints,
        )
