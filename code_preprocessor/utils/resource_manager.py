"""Resource management utilities for the code preprocessor."""
import os
import tempfile
from contextlib import contextmanager
from typing import Generator, Optional, Set

from .logging import get_logger

logger = get_logger(__name__)


class ResourceManager:
    """Base class for managing resources and cleanup.

    Provides functionality for tracking and cleaning up temporary files
    and other resources that need proper disposal.
    """

    def __init__(self) -> None:
        """Initialize the resource manager."""
        self._temp_files: Set[str] = set()
        self._temp_dirs: Set[str] = set()

    def register_temp_file(self, filepath: str) -> None:
        """Register a temporary file for cleanup.

        Args:
            filepath: Path to the temporary file
        """
        self._temp_files.add(os.path.abspath(filepath))

    def register_temp_dir(self, dirpath: str) -> None:
        """Register a temporary directory for cleanup.

        Args:
            dirpath: Path to the temporary directory
        """
        self._temp_dirs.add(os.path.abspath(dirpath))

    @contextmanager
    def temp_file(
        self, suffix: Optional[str] = None, prefix: Optional[str] = None, dir: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Context manager for creating and cleaning up a temporary file.

        Args:
            suffix: File name suffix
            prefix: File name prefix
            dir: Directory to create the file in

        Yields:
            Path to the temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=prefix, dir=dir)
        try:
            self.register_temp_file(temp_file.name)
            yield temp_file.name
        finally:
            temp_file.close()

    @contextmanager
    def temp_dir(
        self, suffix: Optional[str] = None, prefix: Optional[str] = None, dir: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Context manager for creating and cleaning up a temporary directory.

        Args:
            suffix: Directory name suffix
            prefix: Directory name prefix
            dir: Parent directory

        Yields:
            Path to the temporary directory
        """
        temp_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        try:
            self.register_temp_dir(temp_dir)
            yield temp_dir
        finally:
            pass  # Cleanup happens in cleanup() method

    def cleanup_temp_file(self) -> None:
        """Clean up all registered temporary files.

        Attempts to remove each registered temporary file, logging any failures.
        """
        for filepath in self._temp_files:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.debug(f"Removed temporary file: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {filepath}: {e}")

    def cleanup_temp_dir(self) -> None:
        """Clean up all registered temporary directories.

        Attempts to remove each registered temporary directory, logging any failures.
        """
        for dirpath in self._temp_dirs:
            try:
                if os.path.exists(dirpath):
                    os.rmdir(dirpath)
                    logger.debug(f"Removed temporary directory: {dirpath}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary directory {dirpath}: {e}")

    def cleanup(self) -> None:
        """Clean up all registered temporary files and directories."""
        # Clean up temporary files
        self.cleanup_temp_file()

        # Clean up temporary directories
        self.cleanup_temp_dir()

        # Clear the sets
        self._temp_files.clear()
        self._temp_dirs.clear()

    def __del__(self) -> None:
        """Ensure cleanup on object destruction."""
        self.cleanup()
