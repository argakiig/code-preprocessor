"""Configuration management for the code preprocessor."""
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

    Attributes:
        code_path: Path to the source code directory
        output_dir: Directory to save outputs
        model_name: Name/path of the model to use
        epochs: Number of training epochs
        vocab_size: Size of tokenizer vocabulary
        batch_size: Training batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_sequence_length: Maximum sequence length for training
        num_workers: Number of data loading workers
        cache_dir: Optional cache directory
        log_level: Logging level
        log_file: Optional log file path
        skip_dataset: Whether to skip dataset creation
        skip_tokenizer: Whether to skip tokenizer creation
        num_samples: Optional number of samples to use
        seed: Random seed
        eval_split: Evaluation split ratio
        wandb_project: Weights & Biases project name
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

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()
        self._expand_paths()

    def validate(self) -> None:
        """Validate configuration values."""
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
        """Expand user paths (~) in path fields."""
        self.code_path = os.path.expanduser(self.code_path)
        self.output_dir = os.path.expanduser(self.output_dir)
        if self.cache_dir:
            self.cache_dir = os.path.expanduser(self.cache_dir)
        if self.log_file:
            self.log_file = os.path.expanduser(self.log_file)

    @property
    def dataset_dir(self) -> str:
        """Get the dataset directory path."""
        return os.path.join(self.output_dir, "dataset")

    @property
    def model_dir(self) -> str:
        """Get the model directory path."""
        return os.path.join(self.output_dir, "model")

    @property
    def tokenizer_path(self) -> str:
        """Get the tokenizer file path."""
        return os.path.join(self.dataset_dir, "tokenizer.json")
