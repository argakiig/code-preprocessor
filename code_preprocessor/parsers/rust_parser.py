"""Module for parsing Rust source code into structured code blocks."""

import re
from typing import Dict, List, Optional, Tuple

from ..models.code_block import CodeBlock


class RustParser:
    """Parser class that handles Rust code processing.

    This class is responsible for:
    1. Processing Rust source files
    2. Extracting code blocks and documentation
    3. Parsing Rust syntax patterns

    Example:
        ```python
        parser = RustParser()
        blocks = parser.parse(content)
        ```
    """

    # Rust code patterns
    PATTERNS = [
        # Enum definitions
        (
            r"(?P<doc>(?://[!/][^\n]*\n|/\*[!*][\s\S]*?\*/)\s*)?(?P<code>pub\s+enum\s+"
            r"(?P<n>\w+)[\s\S]*?(?=\n\n|\Z))",
            "enum",
        ),
        # Impl blocks
        (
            r"(?P<doc>(?://[!/][^\n]*\n|/\*[!*][\s\S]*?\*/)\s*)?(?P<code>impl\s+"
            r"(?P<n>\w+)[\s\S]*?(?=\n\n|\Z))",
            "impl",
        ),
        # Functions
        (
            r"(?P<doc>(?://[!/][^\n]*\n|/\*[!*][\s\S]*?\*/)\s*)?(?P<code>(?:pub\s+)?fn\s+"
            r"(?P<n>\w+)[\s\S]*?(?=\n\n|\Z))",
            "function",
        ),
        # Structs
        (
            r"(?P<doc>(?://[!/][^\n]*\n|/\*[!*][\s\S]*?\*/)\s*)?(?P<code>pub\s+struct\s+"
            r"(?P<n>\w+)[\s\S]*?(?=\n\n|\Z))",
            "struct",
        ),
        # Traits
        (
            r"(?P<doc>(?://[!/][^\n]*\n|/\*[!*][\s\S]*?\*/)\s*)?(?P<code>pub\s+trait\s+"
            r"(?P<n>\w+)[\s\S]*?(?=\n\n|\Z))",
            "trait",
        ),
    ]

    def __init__(self) -> None:
        """Initialize the RustParser.

        Compiles regex patterns and initializes caches for better performance.
        """
        # Compile patterns once during initialization
        self.compiled_patterns = [
            (re.compile(pattern, re.MULTILINE), block_type) for pattern, block_type in self.PATTERNS
        ]
        # Add LRU cache for module path extraction
        self._module_path_cache: Dict[str, str] = {}

    @staticmethod
    def _get_line_numbers(
        content: str, match: re.Match, group_name: Optional[str] = None
    ) -> Tuple[int, int]:
        """Get the start and end line numbers for a match or a specific group."""
        if group_name and group_name in match.groupdict():
            start_idx = match.start(group_name)
            end_idx = match.end(group_name)
        else:
            start_idx = match.start()
            end_idx = match.end()

        # Count newlines before the match start to get start line
        start_line = content.count("\n", 0, start_idx) + 1
        # Count newlines before the match end to get end line
        end_line = content.count("\n", 0, end_idx) + 1
        return start_line, end_line

    @staticmethod
    def _extract_module_path(content: str, current_pos: int) -> str:
        """Extract the module path from the code.

        Looks at mod declarations and use statements before the current position.
        """
        # Find all mod declarations before this position
        mod_pattern = r"(?:pub\s+)?mod\s+(\w+)\s*{?"
        mods = [m.group(1) for m in re.finditer(mod_pattern, content[:current_pos])]

        # Find crate name or module path from use statements
        use_pattern = r"use\s+((?:crate|self|super)(?:::\w+)*|\w+(?:::\w+)*)"
        uses = [m.group(1) for m in re.finditer(use_pattern, content[:current_pos])]

        # If we have explicit crate/module references, use those
        if uses:
            base_path = uses[-1]  # Use the most recent use statement
        else:
            base_path = "crate"  # Default to crate root

        # Build the full path by joining all module segments
        if mods:
            return f"{base_path}::{':'.join(mods)}"
        return base_path

    def parse(self, content: str) -> List[CodeBlock]:
        """Parse Rust source code content into a list of code blocks.

        Args:
            content: The Rust source code content to parse

        Returns:
            A list of CodeBlock instances representing the parsed code blocks
        """
        blocks = []
        for pattern, block_type in self.compiled_patterns:
            for match in pattern.finditer(content):
                doc = match.group("doc")
                name = match.group("n")
                code = match.group("code")

                # Get the module path up to this point
                module_path = self._extract_module_path(content, match.start())
                full_path = f"{module_path}::{name}"

                # Get line numbers for both doc and code
                doc_start = doc_end = None
                if doc:
                    doc_start, doc_end = self._get_line_numbers(content, match, "doc")
                code_start, code_end = self._get_line_numbers(content, match, "code")

                blocks.append(
                    CodeBlock(
                        block_type=block_type,
                        name=name,
                        full_path=full_path,
                        content=code,
                        start_line=code_start,
                        end_line=code_end,
                        doc=doc,
                        doc_start_line=doc_start,
                        doc_end_line=doc_end,
                    )
                )

        return blocks
