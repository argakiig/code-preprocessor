"""Module for training language models on code with optimized configurations.

This module provides functionality for training language models specifically on code data,
with support for:
- Quantization and LoRA for efficient training
- Custom tokenization for code
- Automatic dataset preparation
- GPU memory optimization
- Sequence-to-sequence training setup
"""

import gc
import logging
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
)

from ..tokenizers import RustTokenizer

logger = logging.getLogger(__name__)


class RustProcessingConfig(PretrainedConfig):
    """Configuration class for Rust code processing.

    This class extends PretrainedConfig to provide Rust-specific model configuration
    options. It ensures compatibility with the HuggingFace training pipeline while
    maintaining Rust-specific processing requirements.
    """

    model_type: str = "rust_code"


class ModelTrainer:
    """Trainer for code models using either HuggingFace models or local models.

    This class handles the complete training pipeline for code language models, including:
    - Model initialization with optional quantization and LoRA
    - Dataset preparation and tokenization
    - Training loop management
    - Memory optimization for GPU training
    - Model saving and evaluation

    The trainer supports both full model training and parameter-efficient training
    methods like LoRA, making it suitable for various hardware configurations.
    """

    def __init__(
        self,
        *,  # Force keyword-only arguments
        model_name_or_path: str,
        tokenizer: RustTokenizer,
        output_dir: str,
        sequence_length: int = 256,
        training_args: Optional[TrainingArguments] = None,
        quant_config: Optional[Dict[str, Any]] = None,
        lora_config: Optional[LoraConfig] = None,
        model_config: Optional[Dict[str, Any]] = None,
        eval_dataset: Optional[Dataset] = None,
        train_dataset: Optional[Dataset] = None,
    ) -> None:
        """Initialize the model trainer with specified configuration.

        Args:
            model_name_or_path: Path or name of the model to load
            tokenizer: The tokenizer to use for processing text
            output_dir: Directory for saving outputs
            sequence_length: Maximum sequence length for training
            training_args: Optional HuggingFace training arguments
            quant_config: Optional quantization configuration for reduced memory usage
            lora_config: Optional LoRA configuration for parameter-efficient training
            model_config: Optional model configuration overrides
            eval_dataset: Optional evaluation dataset
            train_dataset: Optional training dataset

        Raises:
            Exception: If model loading fails or configuration is invalid
        """
        self.output_dir = os.path.expanduser(output_dir)
        self.tokenizer = tokenizer
        self.max_length = sequence_length
        self.training_args = training_args
        self.eval_dataset = eval_dataset
        self.train_dataset = train_dataset

        # Load model with quantization
        logger.info("Loading model from %s", model_name_or_path)

        # First, load the model config
        model_config = model_config or {}
        model_config.update(
            {
                "torch_dtype": torch.bfloat16,
                "device_map": {"": 0},  # Force everything to GPU 0
                "trust_remote_code": True,
                "use_cache": False,  # Disable KV cache during training
            }
        )

        # Load the model with quantization if specified
        if quant_config:
            model_config["quantization_config"] = quant_config

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_config)

            # Apply LoRA if config provided
            if lora_config:
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=True,
                )
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()

            # Prepare model for training
            self.prepare_model(self.model)

        except Exception as e:
            logger.error("Failed to load model: %s", str(e))
            raise

        # Initialize data collator for sequence-to-sequence LM
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer.hf_tokenizer,
            model=self.model,
            padding=True,
            pad_to_multiple_of=8,  # For tensor cores
            return_tensors="pt",
        )

        # Initialize trainer with proper arguments
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer.hf_tokenizer,
        )

    @staticmethod
    def _prepare_sample_static(
        example: Dict[str, Any], tokenizer: RustTokenizer, max_length: int
    ) -> Dict[str, Tensor]:
        """Prepare a single sample for training by combining documentation and code.

        This method handles the core preprocessing of each example, including:
        - Adding special tokens for documentation and code sections
        - Truncating sequences to max_length
        - Creating attention masks and labels
        - Handling padding and empty samples

        Args:
            example: Dictionary containing the sample data
            tokenizer: The tokenizer to use for processing
            max_length: Maximum sequence length to use

        Returns:
            Dictionary containing processed tensors ready for training
        """
        try:
            # Initialize empty list for input tokens
            input_ids: List[int] = []

            # Add documentation if present
            if example.get("doc_tokens") and isinstance(example["doc_tokens"], (list, tuple)):
                input_ids.extend([tokenizer.tokenizer.token_to_id(tokenizer.SPECIAL_TOKENS["doc"])])
                input_ids.extend(example["doc_tokens"])

            # Add code tokens - ensure they exist and are proper type
            content_tokens = example.get("content_tokens", [])
            if isinstance(content_tokens, (list, tuple)):
                input_ids.extend(
                    [tokenizer.tokenizer.token_to_id(tokenizer.SPECIAL_TOKENS["code"])]
                )
                input_ids.extend(content_tokens)

            # Ensure we have some content
            if not input_ids:
                # Return a properly formatted empty sample
                empty_tensor = torch.tensor(
                    [tokenizer.tokenizer.token_to_id(tokenizer.SPECIAL_TOKENS["pad"])] * max_length
                )
                return {
                    "input_ids": empty_tensor,
                    "attention_mask": torch.zeros_like(empty_tensor),
                    "labels": torch.full_like(empty_tensor, -100),
                }

            # Truncate if needed
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]

            # Create attention mask for actual content
            attention_mask = [1] * len(input_ids)

            # Pad if needed
            pad_token_id = tokenizer.tokenizer.token_to_id(tokenizer.SPECIAL_TOKENS["pad"])
            padding_length = max_length - len(input_ids)

            if padding_length > 0:
                input_ids.extend([pad_token_id] * padding_length)
                attention_mask.extend([0] * padding_length)

            # Convert to tensors
            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
            attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
            labels = input_ids_tensor.clone()
            labels[attention_mask_tensor == 0] = -100  # Mask padding tokens in loss

            prepared_sample = {
                "input_ids": input_ids_tensor,
                "attention_mask": attention_mask_tensor,
                "labels": labels,
                "decoder_input_ids": input_ids_tensor,  # Add decoder inputs for seq2seq
                "decoder_attention_mask": attention_mask_tensor,  # Add decoder attention mask
            }

            return prepared_sample

        except Exception as e:
            logger.error("Error preparing sample: %s", str(e))
            logger.error("Sample content: %s", example)
            # Return a properly formatted empty sample
            empty_tensor = torch.tensor(
                [tokenizer.tokenizer.token_to_id(tokenizer.SPECIAL_TOKENS["pad"])] * max_length
            )
            return {
                "input_ids": empty_tensor,
                "attention_mask": torch.zeros_like(empty_tensor),
                "labels": torch.full_like(empty_tensor, -100),
            }

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for training by combining documentation and code.

        This method transforms the raw dataset into a format suitable for training by:
        - Processing each example with _prepare_sample_static
        - Handling batched processing for efficiency
        - Removing unnecessary columns
        - Managing memory usage during processing

        Args:
            dataset: The raw dataset to process

        Returns:
            Processed dataset ready for training
        """  # noqa: D202

        # Create a pickleable transform function
        def transform_fn(examples: Dict[str, Any]) -> Dict[str, Tensor]:
            return self._prepare_sample_static(examples, self.tokenizer, self.max_length)

        return dataset.map(
            transform_fn,
            remove_columns=dataset.column_names,
            desc="Preparing dataset for training",
            batch_size=100,  # Process in smaller batches
            num_proc=1,  # Single process for stability
            load_from_cache_file=False,  # Disable cache during debugging
        )

    def prepare_model(self, model: Any) -> Any:
        """Prepare model for training with optimized memory usage.

        This method configures the model for efficient training by:
        - Clearing GPU cache
        - Enabling gradient checkpointing if available
        - Disabling caching during training
        - Setting proper training modes

        Args:
            model: The model to prepare

        Returns:
            Prepared model ready for training
        """
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()

        # Enable memory efficient features
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        # Enable training mode
        model.train()

        return model

    def train(self) -> None:
        """Train the model on the prepared dataset.

        This method handles the complete training process including:
        - Dataset preparation
        - Progress logging
        - Error handling
        - Memory management during training

        Raises:
            Exception: If training fails or datasets are invalid
        """
        logger.info("Starting training")

        try:
            # Prepare datasets
            logger.info("Preparing training dataset")
            if self.train_dataset is not None:
                self.train_dataset = self.prepare_dataset(self.train_dataset)
                # Debug: Log first sample from training dataset
                if len(self.train_dataset) > 0:
                    logger.debug("First training sample keys: %s", self.train_dataset[0].keys())

            if self.eval_dataset is not None:
                logger.info("Preparing evaluation dataset")
                self.eval_dataset = self.prepare_dataset(self.eval_dataset)
                # Debug: Log first sample from eval dataset
                if len(self.eval_dataset) > 0:
                    logger.debug("First eval sample keys: %s", self.eval_dataset[0].keys())

            # Update trainer with datasets
            self.trainer.train_dataset = self.train_dataset
            self.trainer.eval_dataset = self.eval_dataset

            # Debug: Log data collator output for first batch
            if self.train_dataset is not None and len(self.train_dataset) > 0:
                first_batch = [self.train_dataset[0]]
                collated = self.data_collator(first_batch)
                logger.debug("Collated batch keys: %s", collated.keys())

            # Start training
            logger.info("Starting training")
            self.trainer.train()

        except Exception as e:
            logger.error("Training failed: %s", str(e))
            raise

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the evaluation dataset."""
        if not self.eval_dataset:
            raise ValueError("No evaluation dataset provided")

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
        )

        # Run evaluation
        metrics: Dict[str, float] = trainer.evaluate()
        return metrics

    def save(self, output_dir: Optional[str] = None) -> None:
        """Save the model and tokenizer."""
        save_dir = os.path.expanduser(output_dir) if output_dir else self.output_dir

        # Save model
        self.model.save_pretrained(save_dir)

        # Save tokenizer
        self.tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

        logger.info("Model and tokenizer saved to %s", save_dir)
