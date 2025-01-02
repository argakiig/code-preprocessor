"""Module for preprocessing source code into datasets for model training."""

import logging
import os
import random

from datasets import Dataset
from tqdm import tqdm

from .parsers.rust_parser import RustParser

logger = logging.getLogger(__name__)


class CodePreprocessor:
    """Preprocessor for converting source code into training datasets.

    This class handles the preprocessing of source code files into structured
    datasets suitable for training language models. It supports parsing code
    into blocks, extracting documentation, and creating HuggingFace datasets.
    """

    def __init__(self, output_dir: str) -> None:
        """Initialize the code preprocessor.

        Args:
            output_dir: Directory where processed datasets will be saved
        """
        self.output_dir = os.path.expanduser(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.parser = RustParser()

    def process_local_directory(self, directory: str) -> Dataset:
        """Process a local directory and create a Hugging Face dataset."""
        directory_path = os.path.abspath(directory)

        data = []
        for root, _, files in tqdm(list(os.walk(directory_path)), desc="Processing files"):
            for file in files:
                if file.endswith(".rs"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, directory_path)

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Parse the file into blocks
                        blocks = self.parser.parse(content)

                        # Add each block as a separate entry
                        for block in blocks:
                            entry = {"path": rel_path, "type": "rs", **block.to_dict()}
                            data.append(entry)

                    except Exception as e:
                        logger.error("Error processing %s: %s", file_path, str(e))

        # Display random samples from the dataset
        if data:
            self.display_sample(data)

        return Dataset.from_list(data)

    def display_sample(self, dataset: Dataset, num_samples: int = 3) -> None:
        """Display a random sample of entries from the dataset."""
        logger.info("\nRandom sample of dataset entries:")
        logger.info("=" * 80)

        sample_size = min(num_samples, len(dataset))

        for entry in random.sample(dataset, sample_size):
            logger.info("\nFile: %s", entry["path"])

            if entry.get("doc"):
                logger.debug("Documentation:")
                logger.debug(entry["doc"])

            if entry.get("content"):
                logger.debug("Content:")
                logger.debug(entry["content"])

            if entry.get("blocks"):
                for block in entry["blocks"]:
                    logger.debug(str(block))
                logger.debug("-" * 80)

    def save_dataset(self, dataset: Dataset, name: str) -> None:
        """Save a dataset to disk."""
        save_path = os.path.join(self.output_dir, name)
        dataset.save_to_disk(save_path)

    def load_dataset(self, path: str) -> Dataset:
        """Load a dataset from disk."""
        return Dataset.load_from_disk(path)
