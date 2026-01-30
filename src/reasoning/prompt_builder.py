"""
Prompt Builder for Vietnamese Traffic MCQ.

Builds optimized prompts for VLM answer generation.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class PromptStyle(Enum):
    """Prompt style options."""
    SIMPLE = "simple"
    DETAILED = "detailed"
    FEW_SHOT = "few_shot"
    COT = "chain_of_thought"

@dataclass
class PromptTemplate:
    """
    Prompt template configuration.
    
    Attributes:
        style: Prompt style
        language: Response language ("vi" or "en")
        include_context: Include detection context
        include_reasoning: Ask for reasoning
    """
    style: PromptStyle = PromptStyle.SIMPLE
    language: str = "vi"
    include_context: bool = True
    include_reasoning: bool = False


# Prompt Templates
SIMPLE_TEMPLATE_VI = """Xem các hình ảnh từ video giao thông và trả lời câu hỏi trắc nghiệm sau.

Câu hỏi: {question}

Các lựa chọn:
{choices}

{context}

Trả lời bằng một chữ cái (A, B, C hoặc D):"""


SIMPLE_TEMPLATE_EN = """Look at these frames from a traffic video and answer the multiple choice question.

Question: {question}

Choices:
{choices}

{context}

Answer with a single letter (A, B, C, or D):"""


DETAILED_TEMPLATE_VI = """Bạn là một chuyên gia về luật giao thông đường bộ Việt Nam.

Xem các hình ảnh từ video và trả lời câu hỏi trắc nghiệm bên dưới.

## THÔNG TIN PHÁT HIỆN
{context}

## CÂU HỎI
{question}

## CÁC LỰA CHỌN
{choices}

## HƯỚNG DẪN
- Phân tích biển báo, đèn tín hiệu, vạch kẻ đường trong hình
- Chỉ chọn một đáp án đúng nhất
- Trả lời bằng chữ cái tương ứng (A, B, C hoặc D)

## TRẢ LỜI
Đáp án:"""


FEW_SHOT_TEMPLATE_VI = """Bạn là chuyên gia luật giao thông. Dưới đây là một số ví dụ về cách trả lời:

Ví dụ 1:
Câu hỏi: Video có đèn đỏ không?
Lựa chọn: A. Có, B. Không
Trả lời: A

Ví dụ 2:
Câu hỏi: Tốc độ tối đa là bao nhiêu?
Lựa chọn: A. 40 km/h, B. 50 km/h, C. 60 km/h, D. 70 km/h
Trả lời: C

Bây giờ trả lời câu hỏi sau dựa trên hình ảnh:

{context}

Câu hỏi: {question}

Lựa chọn:
{choices}

Trả lời:"""


COT_TEMPLATE_VI = """Xem các hình ảnh và phân tích từng bước để trả lời câu hỏi.

{context}

Câu hỏi: {question}

Lựa chọn:
{choices}

Hãy suy nghĩ từng bước:
1. Quan sát các yếu tố trong hình (biển báo, đèn, vạch kẻ)
2. Xác định thông tin liên quan đến câu hỏi
3. So sánh với các lựa chọn
4. Đưa ra câu trả lời

Phân tích:"""

def format_choices(choices: List[str]) -> str:
    return "\n".join(choices)

def format_context(detections: Optional[List[str]] = None, target_objects: Optional[List[str]] = None) -> str:
    if not detections and not target_objects:
        return ""
    
    parts = []

    if target_objects:
        parts.append(f"Đối tượng cần tìm: {', '.join(target_objects)}")
    if detections:
        parts.append(f"Đã phát hiện: {', '.join(detections)}")
    
    return "\n".join(parts)

def build_mcq_prompt(question: str, choices: List[str], context: str = "", template: Optional[PromptTemplate] = None) -> str:
    """Build MCQ prompt for VLM."""
    if template is None:
        template = PromptTemplate()
    
    choices_text = format_choices(choices)
    
    # Select template based on style and language
    if template.style == PromptStyle.SIMPLE:
        if template.language == "vi":
            tmpl = SIMPLE_TEMPLATE_VI
        else:
            tmpl = SIMPLE_TEMPLATE_EN
    elif template.style == PromptStyle.DETAILED:
        tmpl = DETAILED_TEMPLATE_VI
    elif template.style == PromptStyle.FEW_SHOT:
        tmpl = FEW_SHOT_TEMPLATE_VI
    elif template.style == PromptStyle.COT:
        tmpl = COT_TEMPLATE_VI
    else:
        tmpl = SIMPLE_TEMPLATE_VI
    
    # Format context section
    context_section = ""
    if template.include_context and context:
        context_section = f"\nThông tin bổ sung:\n{context}\n"
    
    prompt = tmpl.format(
        question=question,
        choices=choices_text,
        context=context_section,
    )
    
    return prompt.strip()

def build_prompt_from_sample(
    sample,  # VQASample
    detections: Optional[List[str]] = None, target_objects: Optional[List[str]] = None, template: Optional[PromptTemplate] = None) -> str:
    """Build prompt from VQASample."""
    context = format_context(detections, target_objects)
    
    return build_mcq_prompt(
        question=sample.question,
        choices=sample.choices,
        context=context,
        template=template,
    )