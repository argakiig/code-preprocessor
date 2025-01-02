"""Rust-specific tokenizer implementation for processing Rust source code."""

from typing import Optional

from tokenizers import decoders, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from .code_tokenizer import BaseCodeTokenizer


class RustTokenizer(BaseCodeTokenizer):
    """Tokenizer specifically designed for Rust source code."""

    def __init__(self, vocab_size: int = 50000, tokenizer_path: Optional[str] = None) -> None:
        """Initialize the Rust tokenizer.

        Args:
            vocab_size: Size of the vocabulary to use
            tokenizer_path: Optional path to load a pre-trained tokenizer
        """
        super().__init__(vocab_size, tokenizer_path)
        self._hf_tokenizer: Optional[PreTrainedTokenizerFast] = None

    def _configure_tokenizer(self) -> None:
        """Configure Rust-specific tokenizer settings."""
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
        """Get HuggingFace compatible tokenizer."""
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
