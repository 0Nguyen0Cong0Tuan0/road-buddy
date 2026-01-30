"""
Answer Extractor for VLM Responses.

Parses VLM output to extract the answer choice.
"""
import re
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    """
    Result of answer extraction.
    
    Attributes:
        letter: Extracted letter (A, B, C, D)
        full_answer: Full answer text if matched
        confidence: Confidence in extraction
        method: Method used for extraction
    """
    letter: str
    full_answer: Optional[str] = None
    confidence: float = 1.0
    method: str = "unknown"
    
    @property
    def is_valid(self) -> bool:
        """Check if extraction is valid."""
        return self.letter in "ABCD"


def extract_answer_letter(text: str) -> Optional[str]:
    """Extract answer letter from text."""
    text = text.strip()
    
    # Pattern 1: Just the letter at start
    if text and text[0].upper() in "ABCD":
        # Check it's not part of a word
        if len(text) == 1 or not text[1].isalpha():
            return text[0].upper()
    
    # Pattern 2: "Đáp án: X" or "Answer: X"
    patterns = [
        r"(?:đáp án|answer|trả lời|câu trả lời)[:\s]*([ABCD])",
        r"\b([ABCD])\.",  # "A." format
        r"\b([ABCD])\s*$",  # Letter at end
        r"^([ABCD])\b",  # Letter at start
        r"chọn\s*([ABCD])",  # "chọn A"
        r"là\s*([ABCD])",  # "là B"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return None

def extract_answer(response_text: str, choices: List[str], default: str = "A") -> ExtractionResult:
    """
    Extract answer from VLM response.
    
    Tries multiple extraction strategies:
    1. Direct letter extraction
    2. Full answer text matching
    3. Keyword matching
    """
    response_text = response_text.strip()
    
    # Strategy 1: Direct letter extraction
    letter = extract_answer_letter(response_text)
    if letter:
        full_answer = None
        for choice in choices:
            if choice.startswith(letter):
                full_answer = choice
                break
        
        return ExtractionResult(
            letter=letter,
            full_answer=full_answer,
            confidence=0.9,
            method="letter_extraction"
        )
    
    # Strategy 2: Exact answer text matching
    for choice in choices:
        if choice.lower() in response_text.lower():
            letter = choice[0].upper() if choice else default
            return ExtractionResult(
                letter=letter,
                full_answer=choice,
                confidence=0.85,
                method="text_matching"
            )
    
    # Strategy 3: Keyword matching (answer content without letter)
    for choice in choices:
        # Remove "A. ", "B. " prefix
        content = re.sub(r"^[ABCD]\.\s*", "", choice)
        if content.lower() in response_text.lower():
            letter = choice[0].upper() if choice else default
            return ExtractionResult(
                letter=letter,
                full_answer=choice,
                confidence=0.75,
                method="content_matching"
            )
    
    # Strategy 4: Binary heuristics (for Yes/No questions)
    response_lower = response_text.lower()
    
    positive_keywords = ["có", "đúng", "phải", "yes", "true", "correct"]
    negative_keywords = ["không", "sai", "no", "false", "wrong"]
    
    # Check if choices are binary
    if len(choices) == 2:
        choice_texts = [c.lower() for c in choices]
        
        # Check for positive match
        if any(kw in response_lower for kw in positive_keywords):
            for i, choice_text in enumerate(choice_texts):
                if any(kw in choice_text for kw in positive_keywords):
                    letter = choices[i][0].upper()
                    return ExtractionResult(
                        letter=letter,
                        full_answer=choices[i],
                        confidence=0.7,
                        method="binary_heuristic"
                    )
        
        # Check for negative match
        if any(kw in response_lower for kw in negative_keywords):
            for i, choice_text in enumerate(choice_texts):
                if any(kw in choice_text for kw in negative_keywords):
                    letter = choices[i][0].upper()
                    return ExtractionResult(
                        letter=letter,
                        full_answer=choices[i],
                        confidence=0.7,
                        method="binary_heuristic"
                    )
    
    # Fallback: return default with low confidence
    logger.warning(f"Could not extract answer from: {response_text[:100]}...")
    
    full_answer = None
    for choice in choices:
        if choice.startswith(default):
            full_answer = choice
            break
    
    return ExtractionResult(
        letter=default,
        full_answer=full_answer,
        confidence=0.3,
        method="default_fallback"
    )

def batch_extract_answers(responses: List[str], choices_list: List[List[str]], default: str = "A") -> List[ExtractionResult]:
    """Extract answers from multiple responses."""
    results = []
    for response, choices in zip(responses, choices_list):
        result = extract_answer(response, choices, default)
        results.append(result)
    return results