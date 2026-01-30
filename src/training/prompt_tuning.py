"""
Prompt Tuning Strategy (Strategy A - Zero Training).

Experiments with different prompt templates to find optimal configuration.
No model training required - just prompt engineering.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import numpy as np
from src.reasoning.prompt_builder import build_prompt_from_sample, PromptStyle, PromptTemplate
from src.reasoning.answer_extractor import extract_answer

logger = logging.getLogger(__name__)

@dataclass
class PromptExperimentResult:
    """
    Result from a prompt experiment.
    
    Attributes:
        template_name: Name of prompt template
        accuracy: Accuracy on test samples
        avg_confidence: Average extraction confidence
        avg_time_ms: Average inference time in ms
        num_samples: Number of samples tested
        per_sample_results: Detailed per-sample results
    """
    template_name: str
    accuracy: float
    avg_confidence: float
    avg_time_ms: float
    num_samples: int
    per_sample_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_name": self.template_name,
            "accuracy": self.accuracy,
            "avg_confidence": self.avg_confidence,
            "avg_time_ms": self.avg_time_ms,
            "num_samples": self.num_samples,
        }

class PromptTuner:
    """
    Prompt tuning for VLM-based VQA.
    
    Experiments with different prompt templates to find optimal configuration
    without any model fine-tuning.
    """
    
    def __init__(self, vlm_client, templates: Optional[Dict[str, Any]] = None):
        self.vlm_client = vlm_client
        self.templates = templates or self._get_default_templates()
    
    def _get_default_templates(self) -> Dict[str, Any]:
        """Get default template configurations."""
        
        return {
            "simple_vi": PromptTemplate(style=PromptStyle.SIMPLE, language="vi"),
            "simple_en": PromptTemplate(style=PromptStyle.SIMPLE, language="en"),
            "detailed_vi": PromptTemplate(style=PromptStyle.DETAILED, language="vi"),
            "few_shot_vi": PromptTemplate(style=PromptStyle.FEW_SHOT, language="vi"),
            "cot_vi": PromptTemplate(style=PromptStyle.COT, language="vi"),
        }
    
    def evaluate_single_sample(self, sample, frames: List[np.ndarray], template) -> Dict[str, Any]:
        """Evaluate single sample with template."""
        
        prompt = build_prompt_from_sample(sample, template=template)
        
        start_time = time.perf_counter()
        response = self.vlm_client.generate(frames, prompt)
        inference_time = (time.perf_counter() - start_time) * 1000

        extraction = extract_answer(response.text, sample.choices)
        correct = extraction.letter == sample.answer_letter
        
        return {
            "sample_id": sample.id,
            "predicted": extraction.letter,
            "ground_truth": sample.answer_letter,
            "correct": correct,
            "confidence": extraction.confidence,
            "extraction_method": extraction.method,
            "inference_time_ms": inference_time,
            "response_text": response.text[:200],
        }
    
    def evaluate_template(self, template_name: str, template, samples: List, frames_dict: Dict[str, List[np.ndarray]], max_samples: Optional[int] = None) -> PromptExperimentResult:
        """Evaluate a template on multiple samples."""
        if max_samples:
            samples = samples[:max_samples]
        
        results = []
        
        for sample in samples:
            frames = frames_dict.get(sample.id, [])
            
            if not frames:
                logger.warning(f"No frames for sample {sample.id}")
                continue
            
            try:
                result = self.evaluate_single_sample(sample, frames, template)
                results.append(result)
            except Exception as e:
                logger.error(f"Error on sample {sample.id}: {e}")
        
        if not results:
            return PromptExperimentResult(
                template_name=template_name,
                accuracy=0.0,
                avg_confidence=0.0,
                avg_time_ms=0.0,
                num_samples=0,
            )
        
        # Aggregate metrics
        accuracy = np.mean([r["correct"] for r in results])
        avg_confidence = np.mean([r["confidence"] for r in results])
        avg_time = np.mean([r["inference_time_ms"] for r in results])
        
        return PromptExperimentResult(
            template_name=template_name,
            accuracy=accuracy,
            avg_confidence=avg_confidence,
            avg_time_ms=avg_time,
            num_samples=len(results),
            per_sample_results=results,
        )
    
    def run_all_experiments(self, samples: List, frames_dict: Dict[str, List[np.ndarray]], max_samples: Optional[int] = None) -> Dict[str, PromptExperimentResult]:
        """Run experiments with all templates."""
        results = {}
        
        for name, template in self.templates.items():
            logger.info(f"Evaluating template: {name}")
            result = self.evaluate_template(
                name, template, samples, frames_dict, max_samples
            )
            results[name] = result
            logger.info(f"  Accuracy: {result.accuracy:.2%}")
        
        return results
    
    def get_best_template(self, results: Dict[str, PromptExperimentResult]) -> str:
        """Get best performing template name."""
        best_name = max(results.keys(), key=lambda k: results[k].accuracy)
        return best_name

def print_experiment_summary(results: Dict[str, PromptExperimentResult]):
    print("Prompt tuning experiment results")
    print(f"{'Template':<20} {'Accuracy':<12} {'Confidence':<12} {'Time (ms)':<12}")
    
    sorted_results = sorted(
        results.items(), 
        key=lambda x: x[1].accuracy, 
        reverse=True
    )
    
    for name, result in sorted_results:
        print(f"{name:<20} {result.accuracy:>10.2%} {result.avg_confidence:>10.2f} {result.avg_time_ms:>10.1f}")