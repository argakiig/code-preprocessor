"""Common test fixtures and configuration for code_preprocessor tests."""

import os
from typing import TYPE_CHECKING, Any, Generator

import pytest

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture(scope="function")
def mock_env(monkeypatch: "MonkeyPatch") -> None:
    """Set up a clean test environment with minimal environment variables.

    Args:
        monkeypatch: Pytest fixture for modifying environment variables.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Any) -> Generator[str, None, None]:
    """Create a temporary directory with test data structure.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Yields:
        str: Path to the temporary test data directory.
    """
    data_dir = tmp_path / "test_data"
    os.makedirs(data_dir, exist_ok=True)

    # Create some dummy code files for testing
    with open(data_dir / "test.py", "w", encoding="utf-8") as f:
        f.write("def test(): pass\n")

    yield str(data_dir)
