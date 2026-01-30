"""
LoRA Fine-tuning Strategy (Strategy B - Light Training).

Fine-tunes VLM using LoRA (Low-Rank Adaptation) for efficient training.
Supports both GPU (fast) and CPU (slow but works) modes.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType, PeftModel
from src.utils.device import get_device, DeviceConfig

logger = logging.getLogger(__name__)

@dataclass
class LoRAConfig:
    """
    Configuration for LoRA fine-tuning.
    
    Attributes:
        model_name: Base model to fine-tune
        device: Device ("auto", "cuda", "cpu")
        lora_r: LoRA rank (lower = fewer params)
        lora_alpha: LoRA alpha
        lora_dropout: Dropout rate
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Training batch size
        gradient_accumulation_steps: Steps before gradient update
        use_gradient_checkpointing: Save memory with checkpointing
        output_dir: Directory to save model
    """
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"  # Local model for fine-tuning
    device: str = "auto"
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Training parameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    use_8bit_optimizer: bool = True
    
    # Output
    output_dir: str = "models/lora"
    save_steps: int = 100
    logging_steps: int = 10
    
    @property
    def effective_batch_size(self) -> int:
        """Get effective batch size with gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

class LoRATrainer:
    """
    LoRA fine-tuning trainer for VLM.
    
    Supports:
    - GPU training with quantization
    - CPU training (slow but works)
    - Gradient checkpointing for memory efficiency
    """
    
    def __init__(self, config: LoRAConfig):
        """Initialize LoRA trainer."""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self._setup_device()
    
    def _setup_device(self):
        """Setup device configuration."""
        if self.config.device == "auto":
            self.device = get_device()
        else:
            self.device = self.config.device
        
        self.device_config = DeviceConfig(
            preferred_device=self.device,
            use_quantization=self.device != "cpu",
            quantization_bits=4,
        )
        
        logger.info(f"LoRA Trainer using device: {self.device}")
    
    def load_base_model(self):
        """Load base model for fine-tuning.""" 
        model_kwargs = self.device_config.get_model_kwargs()
        
        logger.info(f"Loading base model: {self.config.model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare for k-bit training if using quantization
        if self.device_config.use_quantization:
            self.model = prepare_model_for_kbit_training(self.model)
        
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info("Base model loaded successfully")
    
    def setup_lora(self):
        """Setup LoRA configuration."""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(
            p.numel() for p in self.peft_model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        
        logger.info(f"LoRA setup complete")
        logger.info(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def train(self, train_dataset, val_dataset=None, resume_from_checkpoint: Optional[str] = None):
        """Train model with LoRA."""
        if self.peft_model is None:
            self.load_base_model()
            self.setup_lora()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=self.config.save_steps if val_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=bool(val_dataset),
            fp16=self.device == "cuda",
            optim="adamw_8bit" if self.config.use_8bit_optimizer and self.device == "cuda" else "adamw_torch",
            report_to=["tensorboard"],
            gradient_checkpointing=self.config.use_gradient_checkpointing,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        logger.info("Starting LoRA training...")
        logger.info(f"  Epochs: {self.config.num_epochs}")
        logger.info(f"  Effective batch size: {self.config.effective_batch_size}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        self.save_model()
        
        logger.info("Training complete!")
    
    def save_model(self, path: Optional[str] = None):
        """Save LoRA weights."""
        save_path = path or self.config.output_dir
        
        if self.peft_model:
            self.peft_model.save_pretrained(save_path)
            logger.info(f"LoRA weights saved to: {save_path}")
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights from checkpoint."""
        if self.model is None:
            self.load_base_model()
        
        self.peft_model = PeftModel.from_pretrained(
            self.model,
            path,
        )
        logger.info(f"LoRA weights loaded from: {path}")

def get_cpu_optimized_config() -> LoRAConfig:
    """Get LoRA config optimized for CPU training."""
    return LoRAConfig(
        lora_r=8,  # Smaller rank
        lora_alpha=16,
        batch_size=1,
        gradient_accumulation_steps=16,
        use_gradient_checkpointing=True,
        use_8bit_optimizer=False,  # 8-bit not available on CPU
        num_epochs=1,  # Shorter training
    )

def get_gpu_optimized_config() -> LoRAConfig:
    """Get LoRA config optimized for GPU training."""
    return LoRAConfig(
        lora_r=16,
        lora_alpha=32,
        batch_size=4,
        gradient_accumulation_steps=4,
        use_gradient_checkpointing=True,
        use_8bit_optimizer=True,
        num_epochs=3,
    )