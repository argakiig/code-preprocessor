"""Configuration management for the code preprocessor.

This module provides configuration management through:
- A strongly typed configuration dataclass
- Default values for all settings
- Path expansion and validation
- Directory structure management
- Configuration validation rules
"""

import os
from dataclasses import dataclass
from typing import Optional

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_SPLIT,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_WANDB_PROJECT,
)


@dataclass
class PreprocessorConfig:
    """Configuration for the code preprocessor.

    This class manages all configuration settings for the preprocessor:
    - Model configuration (vocab size, sequence length)
    - Training settings (batch size, epochs)
    - Data processing options (workers, sampling)
    - File paths and directories
    - Logging configuration
    - Experiment tracking settings
    - GPU settings (device, precision, memory efficiency)

    The class provides:
    - Type checking through dataclass
    - Automatic path expansion
    - Configuration validation
    - Default values for optional settings
    - Computed properties for derived paths

    Attributes:
        code_path: Path to the source code directory to process
        output_dir: Directory to save outputs (datasets, models)
        model_name: Name/path of the model to use or fine-tune
        epochs: Number of training epochs
        vocab_size: Size of tokenizer vocabulary
        batch_size: Training batch size
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_sequence_length: Maximum sequence length for training
        num_workers: Number of data loading workers
        cache_dir: Optional cache directory for datasets
        log_level: Logging level (DEBUG, INFO, etc.)
        log_file: Optional log file path
        skip_dataset: Whether to skip dataset creation
        skip_tokenizer: Whether to skip tokenizer creation
        num_samples: Optional number of samples to use
        seed: Random seed for reproducibility
        eval_split: Evaluation split ratio (0-1)
        wandb_project: Weights & Biases project name
        gpu_device: GPU device to use (e.g., "cuda")
        gpu_precision: Training precision ("fp16", "bf16", etc.)
        gpu_memory_efficient: Whether to use memory-efficient training
    """

    code_path: str
    output_dir: str = DEFAULT_OUTPUT_DIR
    model_name: str = DEFAULT_MODEL_NAME
    epochs: int = DEFAULT_EPOCHS
    vocab_size: int = DEFAULT_VOCAB_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    max_sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    num_workers: int = DEFAULT_NUM_WORKERS
    cache_dir: Optional[str] = None
    log_level: str = DEFAULT_LOG_LEVEL
    log_file: Optional[str] = None
    skip_dataset: bool = False
    skip_tokenizer: bool = False
    num_samples: Optional[int] = None
    seed: int = DEFAULT_SEED
    eval_split: float = DEFAULT_EVAL_SPLIT
    wandb_project: str = DEFAULT_WANDB_PROJECT
    gpu_device: str = "cuda"
    gpu_precision: str = "fp16"
    gpu_memory_efficient: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        This method is automatically called after dataclass initialization to:
        - Validate all configuration values
        - Expand user paths in file paths

        Raises:
            ValueError: If any configuration values are invalid
            FileNotFoundError: If required paths don't exist
        """
        self.validate()
        self._expand_paths()

    def validate(self) -> None:
        """Validate configuration values.

        This method checks:
        - Positive values for numeric settings
        - Valid ranges for ratios and splits
        - Existence of required paths
        - Non-negative worker counts

        Raises:
            ValueError: If any values are invalid
            FileNotFoundError: If code_path doesn't exist
        """
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")

        if not os.path.exists(os.path.expanduser(self.code_path)):
            raise FileNotFoundError(f"Code path not found: {self.code_path}")

        if self.eval_split <= 0 or self.eval_split >= 1:
            raise ValueError("eval_split must be between 0 and 1")

        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

    def _expand_paths(self) -> None:
        """Expand user paths (~) in path fields.

        This method expands all path fields to their absolute paths,
        handling user home directory expansion (~) for:
        - code_path
        - output_dir
        - cache_dir (if specified)
        - log_file (if specified)
        """
        self.code_path = os.path.expanduser(self.code_path)
        self.output_dir = os.path.expanduser(self.output_dir)
        if self.cache_dir:
            self.cache_dir = os.path.expanduser(self.cache_dir)
        if self.log_file:
            self.log_file = os.path.expanduser(self.log_file)

    @property
    def dataset_dir(self) -> str:
        """Get the dataset directory path.

        Returns:
            Path to the directory where datasets will be stored,
            as a subdirectory of output_dir
        """
        return os.path.join(self.output_dir, "dataset")

    @property
    def model_dir(self) -> str:
        """Get the model directory path.

        Returns:
            Path to the directory where models will be stored,
            as a subdirectory of output_dir
        """
        return os.path.join(self.output_dir, "model")

    @property
    def tokenizer_path(self) -> str:
        """Get the tokenizer file path.

        Returns:
            Path to the tokenizer.json file within the dataset directory
        """
        return os.path.join(self.dataset_dir, "tokenizer.json")
