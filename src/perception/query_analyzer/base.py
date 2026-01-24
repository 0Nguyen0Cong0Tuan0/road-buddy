"""
Base Strategy Interface for Query Extraction.

This module defines the abstract base class that all extraction
strategies must implement.
"""
from abc import ABC, abstractmethod

from .models import QueryAnalysisResult

class ExtractionStrategy(ABC):
    """Abstract base class for query extraction strategies."""
    
    @abstractmethod
    def extract(self, question: str) -> QueryAnalysisResult:
        """Extract information from a question.
        
        Args:
            question: Vietnamese question text
            
        Returns:
            QueryAnalysisResult with extracted information
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name identifier."""
        pass