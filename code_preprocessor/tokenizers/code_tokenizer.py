"""Base tokenizer module providing common functionality for code tokenization.

This module provides a base tokenizer implementation that:
- Handles special tokens for code and documentation
- Supports BPE tokenization with configurable vocabulary
- Manages tokenizer training and persistence
- Provides dataset preparation utilities
- Ensures consistent token handling across languages
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from datasets import Dataset
from tokenizers import Tokenizer, models
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent deadlock warnings


class BaseCodeTokenizer(ABC):
    """Base tokenizer class for all programming language tokenizers.

    This abstract base class provides common tokenization functionality:
    - Special token management (PAD, UNK, BOS, EOS, etc.)
    - BPE-based tokenization with configurable vocabulary
    - Dataset preparation for training
    - Token encoding and decoding
    - Model persistence (save/load)

    Language-specific tokenizers should inherit from this class and
    implement the _configure_tokenizer method to add their specific
    tokenization rules.
    """

    SPECIAL_TOKENS = {
        "pad": "[PAD]",  # Padding token for batch processing
        "unk": "[UNK]",  # Unknown token for OOV words
        "bos": "[BOS]",  # Beginning of sequence
        "eos": "[EOS]",  # End of sequence
        "sep": "[SEP]",  # Separator for paired sequences
        "doc": "[DOC]",  # Start of documentation
        "code": "[CODE]",  # Start of code
    }

    def __init__(self, vocab_size: int = 50000, tokenizer_path: Optional[str] = None) -> None:
        """Initialize the base tokenizer.

        This method either loads a pre-trained tokenizer or creates a new one with:
        - BPE-based tokenization
        - Special token configuration
        - Post-processing setup

        Args:
            vocab_size: Size of the vocabulary to use (default: 50000)
            tokenizer_path: Optional path to load a pre-trained tokenizer
        """
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            return

        self.tokenizer = Tokenizer(models.BPE())
        self._configure_tokenizer()
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(self.SPECIAL_TOKENS.values()),
            show_progress=True,
        )
        self._configure_post_processor()

    @abstractmethod
    def _configure_tokenizer(self) -> None:
        """Configure language-specific tokenizer settings.

        This method should be implemented by each language-specific tokenizer
        to configure:
        - Pre-tokenization rules
        - Token normalization
        - Special character handling
        - Language-specific patterns
        """

    def _configure_post_processor(self) -> None:
        """Configure post-processing for special tokens.

        This method sets up:
        - Special token training
        - Token ID assignment
        - Template processing for single and paired sequences
        - Special token placement rules
        """
        # Train on special tokens first to ensure they have IDs
        self.tokenizer.train_from_iterator(
            [self.SPECIAL_TOKENS["bos"], self.SPECIAL_TOKENS["eos"], self.SPECIAL_TOKENS["sep"]],
            trainer=self.trainer,
        )

        special_tokens = [
            (self.SPECIAL_TOKENS["bos"], self.tokenizer.token_to_id(self.SPECIAL_TOKENS["bos"])),
            (self.SPECIAL_TOKENS["eos"], self.tokenizer.token_to_id(self.SPECIAL_TOKENS["eos"])),
            (self.SPECIAL_TOKENS["sep"], self.tokenizer.token_to_id(self.SPECIAL_TOKENS["sep"])),
        ]

        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{self.SPECIAL_TOKENS['bos']} $A {self.SPECIAL_TOKENS['eos']}",
            pair=(
                f"{self.SPECIAL_TOKENS['bos']} $A {self.SPECIAL_TOKENS['sep']} "
                f"$B {self.SPECIAL_TOKENS['eos']}"
            ),
            special_tokens=[(str(t), id) for t, id in special_tokens],  # Ensure string type
        )

    def train(self, dataset: Dataset) -> None:
        """Train the tokenizer on a dataset.

        This method:
        - Extracts text from documentation and code content
        - Trains the BPE tokenizer on the combined text
        - Updates the vocabulary and token mappings

        Args:
            dataset: Dataset containing code and documentation
        """
        texts = []
        for example in dataset:
            if "doc" in example and example["doc"]:
                texts.append(example["doc"])
            if "content" in example and example["content"]:
                texts.append(example["content"])

        self.tokenizer.train_from_iterator(texts, trainer=self.trainer)

    def prepare_for_training(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for training by tokenizing content.

        This method processes the dataset by:
        - Tokenizing documentation and code separately
        - Preserving block type information
        - Maintaining original content for reference
        - Optimizing processing with multiprocessing

        Args:
            dataset: Raw dataset with code and documentation

        Returns:
            Processed dataset with tokenized content
        """

        def tokenize_entry(entry: Dict) -> Dict:
            # Tokenize documentation if present
            doc_tokens = None
            if entry.get("doc"):
                doc_tokens = self.encode(entry["doc"])

            # Tokenize code content
            content_tokens = self.encode(entry["content"])

            return {
                "doc_tokens": doc_tokens,
                "content_tokens": content_tokens,
                "block_type": entry.get("block_type"),
                "original_content": entry.get("content", ""),
                "original_doc": entry.get("doc", ""),
            }

        return dataset.map(
            tokenize_entry,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
            num_proc=16,
            load_from_cache_file=True,
        )

    def save(self, path: str) -> None:
        """Save tokenizer to disk.

        Args:
            path: Path where the tokenizer should be saved
        """
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> "BaseCodeTokenizer":
        """Load tokenizer from disk.

        Args:
            path: Path to the saved tokenizer

        Returns:
            Loaded tokenizer instance
        """
        instance = cls(tokenizer_path=path)
        return instance

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        result = self.tokenizer.encode(text).ids
        return [int(id) for id in result]  # Ensure we return List[int]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids to text.

        Args:
            token_ids: List of token IDs to decode

        Returns:
            Decoded text
        """
        result = self.tokenizer.decode(token_ids)
        return str(result)  # Ensure we return str

    def encode_pair(self, text1: str, text2: str) -> List[int]:
        """Encode a pair of texts with separator token.

        Args:
            text1: First text to encode
            text2: Second text to encode

        Returns:
            List of token IDs for the combined sequence
        """
        result = self.tokenizer.encode(text1, text2).ids
        return [int(id) for id in result]  # Ensure we return List[int]
