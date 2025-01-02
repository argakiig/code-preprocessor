"""Main entry point for the code preprocessor.

This module serves as the entry point for the code preprocessing pipeline:
- Handles command line argument parsing
- Loads configuration from YAML files
- Sets up logging and monitoring
- Initializes Weights & Biases tracking
- Creates necessary output directories
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict

import wandb

from .processor import CodePreprocessor
from .tokenizers.rust_tokenizer import RustTokenizer
from .training.gpu_config import GPU4070Config, setup_gpu_training
from .training.model_trainer import ModelTrainer
from .utils.config_loader import create_config_from_yaml
from .utils.logging import setup_logging
from .utils.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


def parse_args() -> Dict[str, Any]:
    """Parse command line arguments.

    Returns:
        Dictionary of parsed arguments
    """  # noqa: D202

    parser = argparse.ArgumentParser(description="Train a code model")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--code-path", type=str, help="Path to the code directory")
    parser.add_argument("--output-dir", type=str, help="Directory to save outputs")
    parser.add_argument("--vocab-size", type=int, help="Size of tokenizer vocabulary")
    parser.add_argument("--max-sequence-length", type=int, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--num-workers", type=int, help="Number of workers")
    parser.add_argument("--cache-dir", type=str, help="Cache directory")
    parser.add_argument(
        "--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    parser.add_argument("--log-file", type=str, help="Log file path")
    parser.add_argument("--model-name", type=str, help="Model name/path")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset creation")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer creation")
    parser.add_argument("--num-samples", type=int, help="Number of samples to use")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--eval-split", type=float, help="Evaluation split ratio")
    parser.add_argument("--wandb-project", type=str, help="Weights & Biases project")

    args = parser.parse_args()
    return {k: v for k, v in vars(args).items() if v is not None}


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

        # Setup resource manager
        resource_manager = ResourceManager()

        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.dataset_dir, exist_ok=True)
        os.makedirs(config.model_dir, exist_ok=True)

        # Initialize processor
        processor = CodePreprocessor(config.output_dir)

        dataset = None
        tokenizer = None

        try:
            # Create or load dataset
            if not config.skip_dataset:
                logger.info("Processing code from %s", config.code_path)
                dataset = processor.process_local_directory(config.code_path)
            else:
                logger.info("Loading existing dataset")
                dataset = processor.load_dataset(os.path.join(config.dataset_dir, "tokenized"))

            # Create or load tokenizer
            if not config.skip_tokenizer:
                logger.info("Training new tokenizer")
                tokenizer = RustTokenizer(vocab_size=config.vocab_size)
                tokenizer.train(dataset)
            else:
                logger.info("Loading existing tokenizer")
                tokenizer = RustTokenizer(tokenizer_path=config.tokenizer_path)

            # Tokenize dataset
            logger.info("Tokenizing dataset")
            tokenized_dataset = tokenizer.prepare_for_training(dataset)

            # Save datasets and tokenizer
            logger.info("Saving processed data")
            processor.save_dataset(dataset, "raw")
            processor.save_dataset(tokenized_dataset, "tokenized")
            tokenizer.save(config.tokenizer_path)

            # Sample dataset if requested
            if config.num_samples:
                logger.info("Sampling %d examples from dataset", config.num_samples)
                tokenized_dataset = tokenized_dataset.shuffle(seed=config.seed)
                tokenized_dataset = tokenized_dataset.select(range(config.num_samples))

            # Split dataset
            logger.info("Splitting dataset into train and eval")
            split = tokenized_dataset.train_test_split(
                test_size=config.eval_split, seed=config.seed
            )
            train_dataset = split["train"]
            eval_dataset = split["test"].select(range(100))

            # Set up GPU training
            setup_gpu_training()
            gpu_config = GPU4070Config(
                output_dir=config.model_dir,
                precision=config.gpu_precision,
                memory_efficient=config.gpu_memory_efficient,
            )
            # Initialize trainer
            trainer = ModelTrainer(
                model_name_or_path=config.model_name,
                tokenizer=tokenizer,
                output_dir=config.model_dir,
                sequence_length=config.max_sequence_length,
                training_args=gpu_config.training_args,
                quant_config=gpu_config.quant_config,
                lora_config=gpu_config.lora_config,
                model_config=gpu_config.model_config,
                eval_dataset=eval_dataset,
                train_dataset=train_dataset,
            )

            # Train model
            logger.info("Starting training")
            trainer.train()

            # Save final model
            trainer.save()

            logger.info("Training complete! Files saved in:")
            logger.info("- Dataset: %s", config.dataset_dir)
            logger.info("- Model: %s", config.model_dir)
            logger.info("- Tokenizer: %s", config.tokenizer_path)

        finally:
            # Clean up resources
            resource_manager.cleanup()

    except Exception as e:
        logger.error("Training failed: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
