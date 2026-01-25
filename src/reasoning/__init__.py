"""
Reasoning Package for Road Buddy VQA.

Provides VLM integration (Gemini API + local Qwen2.5-VL), prompt building, and answer extraction.
"""
from .vlm_client import (
    VLMConfig,
    VLMResponse,
    GeminiVLMClient,
    Qwen2VLClient,
    create_vlm_client,
)

from .prompt_builder import (
    PromptStyle,
    PromptTemplate,
    build_mcq_prompt,
    build_prompt_from_sample,
    format_choices,
    format_context,
)

from .answer_extractor import (
    ExtractionResult,
    extract_answer,
    extract_answer_letter,
    batch_extract_answers,
)

__all__ = [
    # VLM Clients (Gemini API + Local Qwen)
    "VLMConfig",
    "VLMResponse",
    "GeminiVLMClient",
    "Qwen2VLClient",
    "create_vlm_client",
    # Prompt Builder
    "PromptStyle",
    "PromptTemplate",
    "build_mcq_prompt",
    "build_prompt_from_sample",
    "format_choices",
    "format_context",
    # Answer Extractor
    "ExtractionResult",
    "extract_answer",
    "extract_answer_letter",
    "batch_extract_answers",
]
