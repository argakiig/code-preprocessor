"""Configuration loading utilities for the code preprocessor.

This module provides utilities for:
- Loading YAML configuration files
- Creating typed configuration objects
- Handling configuration overrides
- Validating configuration values
- Managing default configurations
"""

import os
from typing import Any, Dict, Optional

import yaml

from ..config import PreprocessorConfig
from .logging import get_logger

logger = get_logger(__name__)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    This function handles:
    - File existence validation
    - YAML parsing with error handling
    - Safe loading of YAML content

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the parsed configuration

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML content is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config: Dict[str, Any] = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise


def create_config_from_yaml(
    config_path: str,
    code_path: Optional[str] = None,
    override_args: Optional[Dict[str, Any]] = None,
) -> PreprocessorConfig:
    """Create a PreprocessorConfig from a YAML file.

    This function handles the complete configuration loading process:
    - Loading the base YAML configuration
    - Extracting configuration sections (model, training, paths, etc.)
    - Applying command-line overrides
    - Validating required fields
    - Creating a typed configuration object

    The configuration structure supports:
    - Model settings (name, vocab size, sequence length)
    - Training parameters (batch size, epochs, etc.)
    - Path configurations (code path, output dir)
    - Logging settings (level, file location)
    - Weights & Biases integration

    Args:
        config_path: Path to the YAML configuration file
        code_path: Optional code path to override the one in config
        override_args: Optional dictionary of arguments to override

    Returns:
        PreprocessorConfig instance with validated settings

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is invalid
        ValueError: If required fields are missing or invalid
    """
    # Load YAML config
    yaml_config = load_yaml_config(config_path)

    # Extract config sections
    model_config = yaml_config.get("model", {})
    training_config = yaml_config.get("training", {})
    paths_config = yaml_config.get("paths", {})
    logging_config = yaml_config.get("logging", {})
    wandb_config = yaml_config.get("wandb", {})

    # Create base config dict with type hints and defaults
    config_dict = {
        # Paths configuration
        "code_path": code_path or paths_config.get("code_path"),
        "output_dir": paths_config.get("output_dir"),
        "cache_dir": paths_config.get("cache_dir"),
        # Model configuration
        "model_name": model_config.get("name"),
        "vocab_size": model_config.get("vocab_size"),
        "max_sequence_length": model_config.get("max_sequence_length"),
        # Training configuration
        "batch_size": training_config.get("batch_size"),
        "num_workers": training_config.get("num_workers"),
        "epochs": training_config.get("epochs"),
        "gradient_accumulation_steps": training_config.get("gradient_accumulation_steps"),
        "eval_split": training_config.get("eval_split"),
        "seed": training_config.get("seed"),
        # Logging configuration
        "log_level": logging_config.get("level"),
        "log_file": logging_config.get("file"),
        # Weights & Biases configuration
        "wandb_project": wandb_config.get("project"),
    }

    # Override with any provided arguments
    if override_args:
        config_dict.update(override_args)

    # Validate required fields
    if not config_dict["code_path"]:
        raise ValueError("code_path must be provided either in config or as argument")

    # Create and return config, filtering out None values
    return PreprocessorConfig(**{k: v for k, v in config_dict.items() if v is not None})
