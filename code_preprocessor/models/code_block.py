"""Module containing the CodeBlock class for representing code blocks with documentation.

This module provides a data structure for representing code blocks that:
- Maintains code and documentation together
- Tracks source location information
- Supports hierarchical code organization
- Provides serialization capabilities
- Handles content cleaning and formatting
"""

from typing import Dict, Optional


class CodeBlock:
    """Represents a block of code with its documentation and location information.

    The CodeBlock class serves as a fundamental data structure that:
    - Encapsulates code content and its associated documentation
    - Maintains source file location information
    - Supports hierarchical organization through full_path
    - Provides clean content handling and formatting
    - Enables serialization for dataset creation

    This class is used throughout the codebase to represent parsed code
    segments and their metadata in a consistent format.
    """

    def __init__(
        self,
        block_type: str,
        name: str,
        full_path: str,
        content: str,
        start_line: int,
        end_line: int,
        doc: Optional[str] = None,
        doc_start_line: Optional[int] = None,
        doc_end_line: Optional[int] = None,
    ) -> None:
        """Initialize a CodeBlock instance.

        Args:
            block_type: Type of the code block (e.g., function, class, trait)
            name: Name of the code block (e.g., function name, class name)
            full_path: Full path to the code
                block in the module hierarchy (e.g., crate::module::item)
            content: The actual code content without documentation
            start_line: Starting line number in the source file (1-based)
            end_line: Ending line number in the source file (1-based)
            doc: Optional documentation string (comments, docstrings)
            doc_start_line: Optional starting line number of documentation (1-based)
            doc_end_line: Optional ending line number of documentation (1-based)
        """
        self.block_type = block_type
        self.name = name
        self.full_path = full_path
        self.content = self._clean_content(content, doc)
        self.start_line = start_line
        self.end_line = end_line
        self.doc = doc.strip() if doc else None
        self.doc_start_line = doc_start_line
        self.doc_end_line = doc_end_line

    def _clean_content(self, content: str, doc: Optional[str]) -> str:
        """Remove documentation from the content if it exists at the start.

        This method ensures that documentation is not duplicated in the content
        field when it's already stored in the doc field.

        Args:
            content: The raw content that might contain documentation
            doc: The documentation string to remove if present

        Returns:
            Clean content with documentation removed if it was at the start
        """
        if doc and content.startswith(doc):
            return content[len(doc) :].lstrip()
        return content

    def to_dict(self) -> Dict:
        """Convert the code block to a dictionary representation.

        This method serializes all CodeBlock fields into a dictionary format,
        suitable for:
        - Dataset creation
        - JSON serialization
        - Data transfer between components

        Returns:
            Dictionary containing all CodeBlock fields
        """
        return {
            "block_type": self.block_type,
            "name": self.name,
            "full_path": self.full_path,
            "content": self.content,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "doc": self.doc,
            "doc_start_line": self.doc_start_line,
            "doc_end_line": self.doc_end_line,
        }

    def __str__(self) -> str:
        """Return a string representation of the code block.

        This method creates a human-readable representation that includes:
        - Block type and name
        - Documentation preview (if present)
        - Content preview
        - Line number information

        The previews are truncated for readability when too long.

        Returns:
            A formatted string containing type, name, documentation, and content previews
        """
        doc_info = ""
        if self.doc is not None:
            doc_preview = self.doc[:100] + "..." if len(self.doc) > 100 else self.doc
            doc_info = f"\nDoc ({self.doc_start_line}-{self.doc_end_line}): {doc_preview}"

        content_preview = (
            f"\nContent: {self.content[:200]}..."
            if len(self.content) > 200
            else f"\nContent: {self.content}"
        )
        return f"Type: {self.block_type}, Name: {self.full_path}{doc_info}{content_preview}"
