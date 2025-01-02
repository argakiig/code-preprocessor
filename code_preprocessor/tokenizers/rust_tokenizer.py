"""Rust-specific tokenizer implementation for processing Rust source code.

This module provides a specialized tokenizer for Rust code that:
- Handles Rust-specific syntax patterns
- Supports byte-level tokenization
- Integrates with HuggingFace's transformers library
- Manages special tokens for code and documentation
- Provides efficient pre-tokenization rules
"""

from typing import Optional

from tokenizers import decoders, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from .code_tokenizer import BaseCodeTokenizer


class RustTokenizer(BaseCodeTokenizer):
    """Tokenizer specifically designed for Rust source code.

    This tokenizer extends the BaseCodeTokenizer with Rust-specific features:
    - Custom pre-tokenization rules for Rust syntax (::, ->, etc.)
    - Byte-level encoding for handling Unicode and special characters
    - Integration with HuggingFace's fast tokenizer implementation
    - Special token handling for code blocks and documentation
    - Configurable vocabulary size
    """

    def __init__(self, vocab_size: int = 50000, tokenizer_path: Optional[str] = None) -> None:
        """Initialize the Rust tokenizer with specified configuration.

        Args:
            vocab_size: Size of the vocabulary to use, defaults to 50000
            tokenizer_path: Optional path to load a pre-trained tokenizer.
                          If not provided, a new tokenizer will be trained.
        """
        super().__init__(vocab_size, tokenizer_path)
        self._hf_tokenizer: Optional[PreTrainedTokenizerFast] = None

    def _configure_tokenizer(self) -> None:
        """Configure Rust-specific tokenizer settings.

        This method sets up:
        - Byte-level pre-tokenization for Unicode support
        - Whitespace and punctuation handling
        - Rust-specific token splitting (::, ->, .)
        - Proper decoding settings for byte-level tokens
        """
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.ByteLevel(add_prefix_space=False),
                pre_tokenizers.Whitespace(),
                pre_tokenizers.Punctuation(),
                # Split on Rust-specific patterns
                pre_tokenizers.Split("::", behavior="isolated"),
                pre_tokenizers.Split("->", behavior="isolated"),
                pre_tokenizers.Split(".", behavior="isolated"),
            ]
        )
        self.tokenizer.decoder = decoders.ByteLevel()

    @property
    def hf_tokenizer(self) -> PreTrainedTokenizerFast:
        """Get HuggingFace compatible tokenizer.

        This property provides a HuggingFace-compatible tokenizer that:
        - Handles special tokens (pad, eos, bos, unk)
        - Uses right-side padding
        - Maintains compatibility with transformers library
        - Caches the tokenizer instance for efficiency

        Returns:
            A HuggingFace PreTrainedTokenizerFast instance
        """
        if self._hf_tokenizer is None:
            self._hf_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=self.tokenizer,
                pad_token=self.SPECIAL_TOKENS["pad"],
                eos_token=self.SPECIAL_TOKENS["eos"],
                bos_token=self.SPECIAL_TOKENS["bos"],
                unk_token=self.SPECIAL_TOKENS["unk"],
                mask_token=None,  # Not used for causal LM
                padding_side="right",
            )
        return self._hf_tokenizer
