"""
Translation Extraction Strategy.

Translates Vietnamese questions to English for better CLIP compatibility.
"""
import logging
from typing import Optional

from ..base import ExtractionStrategy
from ..models import QueryAnalysisResult
from ..constants import QuestionIntent
from .keyword import KeywordExtractionStrategy

logger = logging.getLogger(__name__)


class TranslationExtractionStrategy(ExtractionStrategy):
    """
    Extraction strategy that translates Vietnamese to English.
    
    Uses deep_translator for translation, then applies keyword extraction.
    Useful when working with English-only CLIP models.
    """
    
    def __init__(self, translator: str = "deep_translator"):
        """
        Initialize translation strategy.
        
        Args:
            translator: Translation backend to use
        """
        self._translator_name = translator
        self._translator = None
        self._keyword_strategy = KeywordExtractionStrategy()
    
    def _get_translator(self):
        """Lazy load translator."""
        if self._translator is None:
            try:
                from deep_translator import GoogleTranslator
                self._translator = GoogleTranslator(source='vi', target='en')
                logger.info("Loaded GoogleTranslator for vi->en")
            except ImportError:
                logger.warning("deep_translator not installed. Translation disabled.")
                self._translator = None
        return self._translator
    
    @property
    def name(self) -> str:
        return "translation"
    
    def translate(self, text: str) -> Optional[str]:
        """Translate Vietnamese text to English."""
        translator = self._get_translator()
        if translator is None:
            return None
        
        try:
            translated = translator.translate(text)
            return translated
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return None
    
    def extract(self, question: str) -> QueryAnalysisResult:
        """Extract information with translation support."""
        # First, extract using keyword strategy (for Vietnamese-specific patterns)
        keyword_result = self._keyword_strategy.extract(question)
        
        # Translate the question
        translated = self.translate(question)
        
        # Update result with translation
        return QueryAnalysisResult(
            original_question=question,
            translated_question=translated,
            target_objects=keyword_result.target_objects,
            question_intent=keyword_result.question_intent,
            yolo_classes=keyword_result.yolo_classes,
            keywords_found=keyword_result.keywords_found,
            confidence=keyword_result.confidence,
            temporal_hints=keyword_result.temporal_hints,
        )
