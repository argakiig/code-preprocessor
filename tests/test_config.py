"""Tests for configuration functionality."""
import os
from typing import TYPE_CHECKING, Any, Dict, Union

import pytest

from code_preprocessor.config import PreprocessorConfig
from code_preprocessor.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEQUENCE_LENGTH,
)

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def test_config_initialization(tmp_path: Any) -> None:
    """Test that configuration is initialized correctly with valid values.

    Args:
        tmp_path: Pytest fixture providing temporary directory.
    """
    code_path = str(tmp_path)
    os.makedirs(code_path, exist_ok=True)

    config = PreprocessorConfig(code_path=code_path)
    assert config.code_path == code_path
    assert config.batch_size == DEFAULT_BATCH_SIZE  # Default value from constants
    assert config.max_sequence_length == DEFAULT_SEQUENCE_LENGTH  # Default value from constants
    assert config.num_workers == DEFAULT_NUM_WORKERS
    assert config.epochs == DEFAULT_EPOCHS
    assert config.skip_dataset is False
    assert config.skip_tokenizer is False
    assert config.gpu_memory_efficient is True


def test_config_validation_invalid_path() -> None:
    """Test that appropriate error is raised when code path is invalid."""
    with pytest.raises(FileNotFoundError):
        PreprocessorConfig(code_path="/nonexistent/path")


@pytest.mark.parametrize(  # type: ignore[misc]
    "field,value,error_msg",
    [
        ("vocab_size", 0, "vocab_size must be positive"),
        ("batch_size", 0, "batch_size must be positive"),
        ("max_sequence_length", 0, "max_sequence_length must be positive"),
        ("eval_split", 1.5, "eval_split must be between 0 and 1"),
        ("eval_split", 0.0, "eval_split must be between 0 and 1"),
        ("eval_split", 1.0, "eval_split must be between 0 and 1"),
        ("num_workers", -1, "num_workers must be non-negative"),
        ("epochs", 0, "epochs must be positive"),
    ],
)
def test_config_validation_invalid_values_parametrized(
    tmp_path: Any, field: str, value: Union[int, float], error_msg: str
) -> None:
    """Test that appropriate errors are raised for invalid config values.

    Args:
        tmp_path: Pytest fixture providing temporary directory.
        field: Configuration field to test.
        value: Invalid value to test.
        error_msg: Expected error message.
    """
    code_path = str(tmp_path)
    os.makedirs(code_path, exist_ok=True)

    kwargs: Dict[str, Union[str, int, float]] = {"code_path": code_path, field: value}
    with pytest.raises(ValueError, match=error_msg):
        PreprocessorConfig(**kwargs)  # type: ignore[arg-type]


def test_config_path_expansion(tmp_path: Any, monkeypatch: "MonkeyPatch") -> None:
    """Test that paths are expanded correctly.

    Args:
        tmp_path: Pytest fixture providing temporary directory.
        monkeypatch: Pytest fixture for modifying environment.
    """
    code_path = str(tmp_path)
    os.makedirs(code_path, exist_ok=True)

    # Mock home directory for testing path expansion
    fake_home = str(tmp_path)
    monkeypatch.setenv("HOME", fake_home)

    config = PreprocessorConfig(
        code_path=code_path, output_dir="~/output", cache_dir="~/cache", log_file="~/test.log"
    )

    assert config.output_dir == os.path.join(fake_home, "output")
    assert config.cache_dir == os.path.join(fake_home, "cache")
    assert config.log_file == os.path.join(fake_home, "test.log")


def test_config_optional_paths(tmp_path: Any) -> None:
    """Test that optional paths can be None.

    Args:
        tmp_path: Pytest fixture providing temporary directory.
    """
    code_path = str(tmp_path)
    os.makedirs(code_path, exist_ok=True)

    config = PreprocessorConfig(code_path=code_path)
    assert config.cache_dir is None
    assert config.log_file is None


def test_config_derived_paths(tmp_path: Any) -> None:
    """Test that derived paths are constructed correctly.

    Args:
        tmp_path: Pytest fixture providing temporary directory.
    """
    code_path = str(tmp_path)
    os.makedirs(code_path, exist_ok=True)
    output_dir = str(tmp_path / "output")

    config = PreprocessorConfig(code_path=code_path, output_dir=output_dir)

    assert config.dataset_dir == os.path.join(output_dir, "dataset")
    assert config.model_dir == os.path.join(output_dir, "model")
    assert config.tokenizer_path == os.path.join(output_dir, "dataset", "tokenizer.json")
