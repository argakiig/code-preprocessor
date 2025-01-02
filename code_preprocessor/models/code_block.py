"""Module containing the CodeBlock class for representing code blocks with documentation."""

from typing import Dict, Optional


class CodeBlock:
    """Represents a block of code with its documentation and location information."""

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
            block_type: Type of the code block (e.g., function, class)
            name: Name of the code block
            full_path: Full path to the code block in the module hierarchy
            content: The actual code content
            start_line: Starting line number in the source file
            end_line: Ending line number in the source file
            doc: Optional documentation string
            doc_start_line: Optional starting line number of documentation
            doc_end_line: Optional ending line number of documentation
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
        """Remove documentation from the content if it exists at the start."""
        if doc and content.startswith(doc):
            return content[len(doc) :].lstrip()
        return content

    def to_dict(self) -> Dict:
        """Convert the code block to a dictionary representation."""
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
