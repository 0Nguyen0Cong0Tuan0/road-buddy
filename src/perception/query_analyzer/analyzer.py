"""
Main Query Analyzer Interface.

This module provides the main QueryAnalyzer class that orchestrates
the different extraction strategies.
"""
import logging
from typing import List, Dict

from .base import ExtractionStrategy
from .models import QueryAnalysisResult
from .constants import VIETNAMESE_TRAFFIC_KEYWORDS
from .strategies import (
    KeywordExtractionStrategy,
    TranslationExtractionStrategy,
    SemanticExtractionStrategy,
)

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    """
    Main interface for analyzing Vietnamese traffic questions.
    
    Supports multiple extraction strategies (plugins):
    - "keyword": Fast rule-based extraction (default)
    - "translation": Translates to English for CLIP
    - "semantic": Uses PhoBERT/embeddings for semantic matching
    
    Usage:
        from src.perception.query_analyzer import QueryAnalyzer
        
        analyzer = QueryAnalyzer(strategy="keyword")
        result = analyzer.analyze("Biển báo tốc độ tối đa là bao nhiêu?")
        print(result.target_objects)  # ['speed_limit_sign']
        print(result.question_intent)  # 'value_query'
    """
    
    STRATEGIES = {
        "keyword": KeywordExtractionStrategy,
        "translation": TranslationExtractionStrategy,
        "semantic": SemanticExtractionStrategy,
    }
    
    def __init__(
        self, 
        strategy: str = "keyword",
        translator: str = "deep_translator",
        semantic_model: str = "vinai/phobert-base"
    ):
        """
        Initialize QueryAnalyzer.
        
        Args:
            strategy: Extraction strategy ("keyword", "translation", "semantic")
            translator: Translation backend for translation strategy
            semantic_model: Model name for semantic strategy
        """
        self._strategy_name = strategy
        self._strategy = self._create_strategy(
            strategy, translator, semantic_model
        )
        
        logger.info(f"QueryAnalyzer initialized with strategy: {strategy}")
    
    def _create_strategy(
        self, 
        strategy: str,
        translator: str,
        semantic_model: str
    ) -> ExtractionStrategy:
        """Create the extraction strategy."""
        if strategy == "keyword":
            return KeywordExtractionStrategy()
        elif strategy == "translation":
            return TranslationExtractionStrategy(translator=translator)
        elif strategy == "semantic":
            return SemanticExtractionStrategy(model_name=semantic_model)
        else:
            logger.warning(f"Unknown strategy: {strategy}. Using keyword.")
            return KeywordExtractionStrategy()
    
    def analyze(self, question: str) -> QueryAnalysisResult:
        """
        Analyze a Vietnamese traffic question.
        
        Args:
            question: Vietnamese question text
            
        Returns:
            QueryAnalysisResult with extracted information
        """
        return self._strategy.extract(question)
    
    def analyze_batch(self, questions: List[str]) -> List[QueryAnalysisResult]:
        """Analyze multiple questions."""
        return [self.analyze(q) for q in questions]
    
    @property
    def strategy_name(self) -> str:
        """Get current strategy name."""
        return self._strategy_name
    
    def switch_strategy(
        self, 
        strategy: str,
        translator: str = "deep_translator",
        semantic_model: str = "vinai/phobert-base"
    ):
        """Switch to a different extraction strategy."""
        self._strategy_name = strategy
        self._strategy = self._create_strategy(strategy, translator, semantic_model)
        logger.info(f"Switched to strategy: {strategy}")

def get_available_strategies() -> List[str]:
    """Get list of available extraction strategies."""
    return list(QueryAnalyzer.STRATEGIES.keys())


def get_yolo_class_mapping() -> Dict[str, List[str]]:
    """Get mapping from keywords to YOLO classes."""
    return {
        keyword: info["yolo_classes"]
        for keyword, info in VIETNAMESE_TRAFFIC_KEYWORDS.items()
    }