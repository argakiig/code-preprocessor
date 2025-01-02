"""GPU configuration module for optimizing model training on specific hardware."""

import gc
from dataclasses import dataclass
from typing import Any, Dict

import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig, TrainingArguments


def setup_gpu_training() -> None:
    """Setup GPU for training with extreme memory optimizations."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")


@dataclass
class GPU4070Config:
    """Configuration for RTX 4070 with extreme memory constraints."""

    output_dir: str
    training_args: TrainingArguments
    quant_config: BitsAndBytesConfig
    lora_config: LoraConfig
    model_config: Dict[str, Any]

    def __init__(self, output_dir: str) -> None:
        """Initialize GPU configuration for RTX 4070.

        Args:
            output_dir: Directory for saving model outputs and checkpoints
        """
        self.output_dir = output_dir

        # Training arguments with extreme memory optimization
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            max_steps=1000,  # Limit training steps
            learning_rate=12e-4,
            fp16=False,
            bf16=True,
            bf16_full_eval=True,
            tf32=True,
            optim="adamw_torch_fused",  # Memory efficient optimizer
            save_strategy="steps",
            save_steps=100,
            logging_steps=10,
            remove_unused_columns=True,
            dataloader_num_workers=1,
            group_by_length=True,
            length_column_name="length",
            report_to="wandb",
            ddp_find_unused_parameters=False,
            torch_compile=False,
            seed=42,
            label_smoothing_factor=0.1,
            gradient_accumulation_steps=4,
            eval_strategy="steps",
            eval_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        # 4-bit quantization config with extreme memory savings
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=False,  # Disable CPU offloading
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True,
        )

        # Minimal LoRA config for memory efficiency
        self.lora_config = LoraConfig(
            r=8,  # Small rank
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
        )

        # Model loading config without CPU offloading
        self.model_config = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",  # Let transformers handle device mapping
            "use_cache": False,
            "max_memory": {
                0: "10GiB",  # Strict GPU memory limit
            },
            "low_cpu_mem_usage": True,
        }

    def prepare_model(self, model: Any) -> Any:
        """Prepare model for training with proper device handling.

        Args:
            model: The model to prepare for training

        Returns:
            The prepared model
        """
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()

        # Move model to GPU directly
        model = model.cuda()

        # Enable memory efficient features
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        if hasattr(model, "config"):
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False

        # Enable training mode
        model.train()

        return model
