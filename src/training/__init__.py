"""
Training Package for Road Buddy VQA.

Provides both training strategies:
- Strategy A: Prompt Tuning (zero training)
- Strategy B: LoRA Fine-tuning (light training)
"""

from .dataset_builder import (
    TrainingSample,
    build_training_samples,
    split_samples,
    samples_to_hf_format,
    build_hf_dataset,
    build_prompt_response_pair,
)

from .prompt_tuning import (
    PromptExperimentResult,
    PromptTuner,
    print_experiment_summary,
)

from .lora_trainer import (
    LoRAConfig,
    LoRATrainer,
    get_cpu_optimized_config,
    get_gpu_optimized_config,
)

__all__ = [
    # Dataset Builder
    "TrainingSample",
    "build_training_samples",
    "split_samples",
    "samples_to_hf_format",
    "build_hf_dataset",
    "build_prompt_response_pair",
    # Prompt Tuning (Strategy A)
    "PromptExperimentResult",
    "PromptTuner",
    "print_experiment_summary",
    # LoRA Trainer (Strategy B)
    "LoRAConfig",
    "LoRATrainer",
    "get_cpu_optimized_config",
    "get_gpu_optimized_config",
]
