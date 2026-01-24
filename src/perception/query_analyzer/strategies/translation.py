"""
Translation Extraction Strategy.

Translation-based extraction for CLIP compatibility.
Translates Vietnamese questions to English for use with English CLIP models.
Uses keyword extraction as fallback for object detection.
"""
import logging
from typing import Optional

from ..base import ExtractionStrategy
from ..models import QueryAnalysisResult
from .keyword import KeywordExtractionStrategy

logger = logging.getLogger(__name__)

class TranslationExtractionStrategy(ExtractionStrategy):
    """
    Translation-based extraction for CLIP compatibility.
    
    Translates Vietnamese questions to English for use with English CLIP models.
    Uses keyword extraction as fallback for object detection.
    
    Uses deep_translator (GoogleTranslator) for reliable translation.
    """
    
    def __init__(self, translator: str = "deep_translator"):
        """Initialize with deep_translator (translator param kept for API compatibility)."""
        self._translator = None
        self._keyword_strategy = KeywordExtractionStrategy()
        self._init_translator()
    
    @property
    def name(self) -> str:
        return "translation"
    
    def _init_translator(self):
        """Initialize the deep_translator backend."""
        try:
            from deep_translator import GoogleTranslator
            self._translator = GoogleTranslator(source='vi', target='en')
            logger.info("Translation enabled using deep_translator (Vietnamese -> English)")
        except ImportError as e:
            logger.warning(
                f"deep_translator not available: {e}. "
                "Install with: pip install deep_translator"
            )
    
    def extract(self, question: str) -> QueryAnalysisResult:
        """Extract information with translation support."""
        # First, use keyword extraction for object/class mapping
        keyword_result = self._keyword_strategy.extract(question)
        
        # Then translate for CLIP
        translated = self._translate(question)
        
        return QueryAnalysisResult(
            original_question=question,
            translated_question=translated,
            target_objects=keyword_result.target_objects,
            question_intent=keyword_result.question_intent,
            yolo_classes=keyword_result.yolo_classes,
            keywords_found=keyword_result.keywords_found,
            confidence=keyword_result.confidence if translated else keyword_result.confidence * 0.5,
            temporal_hints=keyword_result.temporal_hints
        )
    
    def _translate(self, text: str) -> Optional[str]:
        """Translate Vietnamese to English using deep_translator."""
        if self._translator is None:
            return None
        
        try:
            return self._translator.translate(text)
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return None