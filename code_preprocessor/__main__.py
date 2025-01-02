"""Main entry point for the code preprocessor.

This module serves as the entry point for the code preprocessing pipeline:
- Handles command line argument parsing
- Loads configuration from YAML files
- Sets up logging and monitoring
- Initializes Weights & Biases tracking
- Creates necessary output directories
"""

import logging
import os
from typing import Any, Dict

import wandb

from .utils.config_loader import create_config_from_yaml
from .utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> Dict[str, Any]:
    """Parse command line arguments for the code preprocessor.

    This function sets up the argument parser with:
    - Required config file path
    - Optional code directory path
    - Help messages for each argument

    Returns:
        Dictionary containing the parsed arguments with their values
    """
    import argparse

    parser = argparse.ArgumentParser(description="Code preprocessor for training language models")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--code-path", type=str, help="Path to code directory to process")
    args = parser.parse_args()
    return vars(args)


def main() -> None:
    """Main entry point for the code preprocessing pipeline.

    This function orchestrates the preprocessing pipeline by:
    - Parsing command line arguments
    - Loading and validating configuration
    - Setting up logging infrastructure
    - Initializing experiment tracking with W&B
    - Creating output directories

    The function handles all initialization steps and ensures
    proper error handling and logging throughout the process.

    Raises:
        Exception: If initialization fails for any reason
    """
    try:
        # Parse arguments
        args = parse_args()

        # Load and create config
        config = create_config_from_yaml(args["config"], args.get("code_path"))

        # Setup logging
        setup_logging(config.log_level, config.log_file)

        # Initialize Weights & Biases if project is specified
        wandb.init(project=config.wandb_project)

        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)

    except Exception as e:
        logger.error("Failed to initialize: %s", str(e))
        raise


if __name__ == "__main__":
    main()
