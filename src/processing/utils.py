"""
Utility functions for GitLab RAG processing.

This module provides helper functions used across different processors
for extracting documentation, parsing code, and normalizing text.
"""

import re
import os
from typing import Dict, Tuple, List, Any


def extract_docstring(
    content: str, pos: int, style: str = "auto"
) -> Tuple[str, Dict[str, Any]]:
    """
    Extract docstring from code starting at a specific position.

    Supports multiple docstring formats including:
    - Python triple-quoted docstrings (Google, NumPy, reST)
    - R roxygen tags (#')
    - Julia triple-quoted docstrings

    Args:
        content: Source code content
        pos: Position to start searching for docstring
        style: Docstring style to use for parsing (auto, python, r, julia)

    Returns:
        Tuple of (raw_docstring, parsed_metadata)
    """
    if style == "auto":
        # Try to detect language from surrounding code
        if "#'" in content[:pos]:
            style = "r"
        elif '"""' in content[:pos] or "'''" in content[:pos]:
            # Could be Python or Julia - look for indicators
            if (
                "function " in content[pos : pos + 100]
                or "struct " in content[pos : pos + 100]
            ):
                style = "julia"
            else:
                style = "python"

    if style == "python":
        return extract_python_docstring(content, pos)
    elif style == "r":
        return extract_roxygen_docs(content, pos)
    elif style == "julia":
        return extract_julia_docstring(content, pos)
    else:
        return "", {}


def extract_python_docstring(content: str, pos: int) -> Tuple[str, Dict[str, Any]]:
    """
    Extract and parse Python docstrings from a function or class definition.

    Args:
        content: Full source code
        pos: Position where function/class starts

    Returns:
        Tuple of (raw_docstring, metadata_dict)
    """
    # Find the body start (position after the colon)
    body_start = content.find(":", pos)
    if body_start == -1:
        return "", {}

    # Look for triple quotes after the function/class definition
    # This handles both ''' and """ style docstrings
    docstring_pattern = r"[\s\n]+([\']{3}|[\"]{3})(.*?)(\1)"
    match = re.search(
        docstring_pattern, content[body_start : body_start + 2000], re.DOTALL
    )

    if not match:
        return "", {}

    # Extract the raw docstring
    raw_docstring = match.group(2).strip()

    # Parse the docstring based on its style
    metadata = {}

    # Try to detect docstring style
    if re.search(r"Args:|Returns:|Raises:|Example:", raw_docstring):
        # Google style
        metadata = _parse_google_docstring(raw_docstring)
    elif re.search(r"Parameters\s*\n\s*----------", raw_docstring):
        # NumPy style
        metadata = _parse_numpy_docstring(raw_docstring)
    elif re.search(r":param\s+\w+:", raw_docstring):
        # reST style
        metadata = _parse_rest_docstring(raw_docstring)
    else:
        # Simple docstring, just split into summary and description
        metadata = _parse_simple_docstring(raw_docstring)

    return raw_docstring, metadata


def extract_roxygen_docs(content: str, pos: int) -> Tuple[str, Dict[str, Any]]:
    """
    Extract and parse roxygen2 documentation for an R function.

    Args:
        content: Full source code
        pos: Position where function starts

    Returns:
        Tuple of (raw_docstring, metadata_dict)
    """
    # Find roxygen lines preceding the function position
    lines = content[:pos].split("\n")
    roxygen_lines = []

    # Start from the lines immediately preceding the function and go backwards
    for line in reversed(lines):
        line = line.rstrip()  # Keep leading spaces for indentation
        if line.startswith("#'"):
            # Found a roxygen line, add it to our collection
            roxygen_lines.insert(0, line)
        elif line and not line.startswith("#"):
            # Found a non-comment, non-empty line, stop collecting
            break

    if not roxygen_lines:
        return "", {}

    # Combine lines into a single string
    roxygen_text = "\n".join(roxygen_lines)

    # Parse structured metadata from roxygen tags
    metadata = _parse_roxygen_metadata(roxygen_lines)

    return roxygen_text, metadata


def extract_julia_docstring(content: str, pos: int) -> Tuple[str, Dict[str, Any]]:
    """
    Extract and parse Julia docstrings.

    Args:
        content: Full source code
        pos: Position where function/type starts

    Returns:
        Tuple of (raw_docstring, metadata_dict)
    """
    # Find a docstring before the function definition
    # Look for a triple-quoted string before the function
    docstring_pattern = (
        r'(""".*?""")\s*(?:function|struct|abstract\s+type|mutable\s+struct)'
    )

    # Look in the content leading up to the function position
    content_before = content[:pos]
    match = re.search(docstring_pattern, content_before, re.DOTALL)

    if not match:
        return "", {}

    # Extract the raw docstring (without the triple quotes)
    raw_docstring = match.group(1).strip('"""').strip()

    # Parse the docstring into metadata
    metadata = _parse_julia_docstring(raw_docstring)

    return raw_docstring, metadata


def _parse_simple_docstring(docstring: str) -> Dict[str, Any]:
    """Parse a simple docstring into summary and description."""
    lines = docstring.split("\n")

    # First line is summary/title
    summary = lines[0].strip() if lines else ""

    # Rest is description (skip blank line if present)
    description_lines = []
    for i in range(1, len(lines)):
        if i == 1 and not lines[i].strip():
            continue  # Skip first blank line
        description_lines.append(lines[i])

    description = "\n".join(description_lines).strip()

    return {"summary": summary, "description": description if description else None}


def _parse_google_docstring(docstring: str) -> Dict[str, Any]:
    """Parse a Google-style docstring into structured metadata."""
    # Initialize metadata
    metadata = {}

    # Split docstring into lines
    lines = docstring.split("\n")

    # Extract summary and description
    summary = lines[0].strip() if lines else ""
    metadata["summary"] = summary

    # Find the start of the first section
    section_indices = [
        i
        for i, line in enumerate(lines)
        if re.match(r"^\s*(Args|Returns|Raises|Examples|Note|Yields):", line)
    ]

    # Extract description
    if section_indices:
        first_section = section_indices[0]
        if first_section > 1:
            description_lines = lines[1:first_section]
            if description_lines and not description_lines[0].strip():
                description_lines = description_lines[1:]  # Skip first blank line
            description = "\n".join(description_lines).strip()
            if description:
                metadata["description"] = description
    else:
        # No sections found, everything after summary is description
        if len(lines) > 1:
            description_lines = lines[1:]
            if description_lines and not description_lines[0].strip():
                description_lines = description_lines[1:]  # Skip first blank line
            description = "\n".join(description_lines).strip()
            if description:
                metadata["description"] = description

    # Parse each section
    current_section = None
    section_content = []

    for i, line in enumerate(lines):
        # Check for section headers
        section_match = re.match(
            r"^\s*(Args|Returns|Raises|Examples|Note|Yields):", line
        )
        if section_match:
            # Save previous section
            if current_section and section_content:
                section_content = [
                    this_line.strip()
                    for this_line in section_content
                    if this_line.strip()
                ]
                if section_content:
                    if current_section == "Args":
                        metadata["parameters"] = _parse_google_args(
                            "\n".join(section_content)
                        )
                    elif current_section == "Returns":
                        metadata["returns"] = "\n".join(section_content).strip()
                    elif current_section == "Raises":
                        metadata["raises"] = _parse_google_raises(
                            "\n".join(section_content)
                        )
                    elif current_section == "Examples":
                        metadata["examples"] = "\n".join(section_content).strip()
                    elif current_section == "Note":
                        metadata["note"] = "\n".join(section_content).strip()
                    elif current_section == "Yields":
                        metadata["yields"] = "\n".join(section_content).strip()

            # Start new section
            current_section = section_match.group(1)
            section_content = []
        elif current_section:
            # Add line to current section
            section_content.append(line.strip())

    # Save the last section
    if current_section and section_content:
        section_content = [
            this_line.strip() for this_line in section_content if this_line.strip()
        ]
        if section_content:
            if current_section == "Args":
                metadata["parameters"] = _parse_google_args("\n".join(section_content))
            elif current_section == "Returns":
                metadata["returns"] = "\n".join(section_content).strip()
            elif current_section == "Raises":
                metadata["raises"] = _parse_google_raises("\n".join(section_content))
            elif current_section == "Examples":
                metadata["examples"] = "\n".join(section_content).strip()
            elif current_section == "Note":
                metadata["note"] = "\n".join(section_content).strip()
            elif current_section == "Yields":
                metadata["yields"] = "\n".join(section_content).strip()

    return metadata


def _parse_google_args(args_text: str) -> Dict[str, str]:
    """Parse the Args section of a Google-style docstring."""
    parameters = {}
    param_pattern = r"(\w+)(?:\s*\([\w\s,]+\))?\s*:\s*(.*?)(?=\n\w+\s*:|$)"

    for match in re.finditer(param_pattern, args_text, re.DOTALL):
        param_name = match.group(1)
        param_desc = match.group(2).strip()
        parameters[param_name] = param_desc

    return parameters


def _parse_google_raises(raises_text: str) -> Dict[str, str]:
    """Parse the Raises section of a Google-style docstring."""
    exceptions = {}
    exception_pattern = r"(\w+):\s*(.*?)(?=\n\w+:|$)"

    for match in re.finditer(exception_pattern, raises_text, re.DOTALL):
        exception_name = match.group(1)
        exception_desc = match.group(2).strip()
        exceptions[exception_name] = exception_desc

    return exceptions


def _parse_numpy_docstring(docstring: str) -> Dict[str, Any]:
    """Parse a NumPy-style docstring into structured metadata."""
    # Implementation details omitted for brevity
    # Would parse sections demarcated by underlines (------)
    return _parse_simple_docstring(docstring)  # Simplified for now


def _parse_rest_docstring(docstring: str) -> Dict[str, Any]:
    """Parse a reST-style docstring into structured metadata."""
    # Implementation details omitted for brevity
    # Would parse directives like :param: and :returns:
    metadata = _parse_simple_docstring(docstring)

    # Extract parameters
    param_matches = re.finditer(
        r":param\s+(\w+):\s*(.+?)(?=\n:[a-z]+|$)", docstring, re.DOTALL
    )
    parameters = {}

    for match in param_matches:
        param_name = match.group(1)
        param_desc = match.group(2).strip()
        parameters[param_name] = param_desc

    if parameters:
        metadata["parameters"] = parameters

    # Extract return value
    return_match = re.search(r":returns?:\s*(.+?)(?=\n:[a-z]+|$)", docstring, re.DOTALL)
    if return_match:
        metadata["returns"] = return_match.group(1).strip()

    return metadata


def _parse_roxygen_metadata(roxygen_lines: List[str]) -> Dict[str, Any]:
    """Parse roxygen2 documentation lines into structured metadata."""
    metadata = {}

    # Clean up the lines for processing
    clean_lines = []
    for line in roxygen_lines:
        # Remove the #' prefix and one space if present
        if line.startswith("#' "):
            clean_lines.append(line[3:])
        elif line == "#'":
            clean_lines.append("")  # Empty line
        else:
            clean_lines.append(line[2:])  # Just remove #'

    # Process explicit tags first
    explicit_tags = {}
    i = 0
    while i < len(clean_lines):
        line = clean_lines[i]
        if line.startswith("@"):
            # This is an explicit tag
            tag_parts = line[1:].split(maxsplit=1)
            tag_name = tag_parts[0]
            tag_content = tag_parts[1] if len(tag_parts) > 1 else ""

            # Collect continuation lines
            j = i + 1
            while (
                j < len(clean_lines)
                and not clean_lines[j].startswith("@")
                and clean_lines[j].strip()
            ):
                tag_content += " " + clean_lines[j].strip()
                j += 1

            # Store the tag
            if tag_name == "param":
                # Handle parameters separately
                param_parts = tag_content.split(maxsplit=1)
                param_name = param_parts[0]
                param_desc = param_parts[1] if len(param_parts) > 1 else ""

                if "params" not in explicit_tags:
                    explicit_tags["params"] = {}
                explicit_tags["params"][param_name] = param_desc
            else:
                explicit_tags[tag_name] = tag_content

            i = j
        else:
            i += 1

    # Now handle implicit content (title, description, details)
    # First, filter out lines that are part of explicit tags
    implicit_lines = []
    i = 0
    while i < len(clean_lines):
        if clean_lines[i].startswith("@"):
            # Skip this tag and its continuation lines
            i += 1
            while (
                i < len(clean_lines)
                and not clean_lines[i].startswith("@")
                and clean_lines[i].strip()
            ):
                i += 1
        else:
            implicit_lines.append(clean_lines[i])
            i += 1

    # Process implicit content
    if implicit_lines:
        # Group lines into paragraphs (separated by blank lines)
        paragraphs = []
        current_paragraph = []

        for line in implicit_lines:
            if line.strip():
                current_paragraph.append(line)
            elif current_paragraph:  # Only append if we have content
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []

        # Add the last paragraph if it exists
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))

        # Extract title, description, and details
        if paragraphs:
            # Title is the first sentence of the first paragraph
            first_para = paragraphs[0]

            # Find the first sentence
            sentence_match = re.match(r"([^.!?]+[.!?])", first_para)

            if sentence_match:
                # We found a sentence ending with punctuation
                title = sentence_match.group(1).strip()
                # Check if there's more content in the first paragraph
                rest_of_para = first_para[len(sentence_match.group(0)) :].strip()

                if rest_of_para:
                    # The rest of the first paragraph is part of the description
                    if len(paragraphs) > 1:
                        # Combine with next paragraph for full description
                        description = rest_of_para + " " + paragraphs[1]
                        details = (
                            " ".join(paragraphs[2:]) if len(paragraphs) > 2 else ""
                        )
                    else:
                        # Just the rest is the description
                        description = rest_of_para
                        details = ""
                else:
                    # First paragraph is just the title
                    description = paragraphs[1] if len(paragraphs) > 1 else ""
                    details = " ".join(paragraphs[2:]) if len(paragraphs) > 2 else ""
            else:
                # No sentence ending found, treat whole first paragraph as title
                title = first_para
                description = paragraphs[1] if len(paragraphs) > 1 else ""
                details = " ".join(paragraphs[2:]) if len(paragraphs) > 2 else ""

            # Store in metadata
            if title and "title" not in explicit_tags:
                metadata["title"] = title
            if description and "description" not in explicit_tags:
                metadata["description"] = description
            if details and "details" not in explicit_tags:
                metadata["details"] = details

    # Combine explicit tags and implicit content
    metadata.update(explicit_tags)

    return metadata


def _parse_julia_docstring(docstring: str) -> Dict[str, Any]:
    """Parse a Julia docstring into structured metadata."""
    # Split by lines
    lines = docstring.split("\n")

    # First line is the summary
    summary = lines[0].strip() if lines else ""
    metadata = {"summary": summary}

    # Rest is description
    if len(lines) > 1:
        description_lines = []
        for i in range(1, len(lines)):
            if lines[i].strip():  # Skip empty lines at the beginning
                description_lines = lines[i:]
                break

        if description_lines:
            metadata["description"] = "\n".join(description_lines).strip()

    # Look for parameter descriptions (format: `param -- description`)
    param_pattern = r"\s*`([^`]+)`\s*(?:--|—)\s*(.+)"
    parameters = {}

    for line in lines:
        param_match = re.match(param_pattern, line)
        if param_match:
            param_name = param_match.group(1).strip()
            param_desc = param_match.group(2).strip()
            parameters[param_name] = param_desc

    if parameters:
        metadata["parameters"] = parameters

    # Look for return descriptions
    return_patterns = [r"(?:Returns|Return value):\s*(.+)", r"@return\s+(.+)"]

    for pattern in return_patterns:
        match = re.search(pattern, docstring, re.IGNORECASE)
        if match:
            metadata["returns"] = match.group(1).strip()
            break

    return metadata


def extract_functions(content: str, language: str = "auto") -> List[Dict[str, Any]]:
    """
    Extract functions or methods from code content.

    Args:
        content: Source code content
        language: Language of the code (auto, python, r, julia)

    Returns:
        List of extracted functions with metadata
    """
    if language == "auto":
        # Try to detect language from file content
        if re.search(r"def\s+\w+\s*\(", content):
            language = "python"
        elif re.search(r"\w+\s*<-\s*function\s*\(", content):
            language = "r"
        elif re.search(r"function\s+\w+\s*\(", content):
            language = "julia"
        else:
            # Default to Python if can't determine
            language = "python"

    if language == "python":
        return extract_python_functions(content)
    elif language == "r":
        return extract_r_functions(content)
    elif language == "julia":
        return extract_julia_functions(content)
    else:
        return []


def extract_python_functions(content: str) -> List[Dict[str, Any]]:
    """Extract Python functions and methods from code content."""
    functions = []

    # Match both regular and class method definitions
    patterns = [
        r"def\s+(\w+)\s*\(([^)]*)\)\s*:",  # Regular functions
        r"def\s+(\w+)\s*\(self(?:,\s*[^)]*)*\)\s*:",  # Class methods
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, content, re.DOTALL):
            func_name = match.group(1)
            func_pos = match.start()

            # Skip dunder methods
            if func_name.startswith("__") and func_name.endswith("__"):
                continue

            # Extract docstring
            docstring, metadata = extract_python_docstring(content, func_pos)

            # Find the function content - simplified for brevity
            lines = content[func_pos:].split("\n")
            func_end = func_pos
            current_line_offset = 0
            indent_level = None

            for i, line in enumerate(lines):
                if i == 0:
                    current_line_offset += len(line) + 1
                    continue

                line_stripped = line.lstrip()
                if not line_stripped:
                    current_line_offset += len(line) + 1
                    continue

                current_indent = len(line) - len(line_stripped)

                if indent_level is None:
                    indent_level = current_indent
                elif current_indent <= 0 or current_indent < indent_level:
                    func_end = func_pos + current_line_offset
                    break

                current_line_offset += len(line) + 1

            if func_end == func_pos:
                func_end = len(content)

            functions.append(
                {
                    "name": func_name,
                    "position": func_pos,
                    "content": content[func_pos:func_end],
                    "docstring": docstring,
                    "metadata": metadata,
                }
            )

    return functions


def extract_r_functions(content: str) -> List[Dict[str, Any]]:
    """Extract R functions from code content."""
    functions = []

    # Pattern for R functions
    pattern = r"(\w+)\s*<-\s*function\s*\(([^)]*)\)"

    for match in re.finditer(pattern, content, re.DOTALL):
        func_name = match.group(1)
        func_pos = match.start()

        # Extract roxygen documentation
        roxygen_text, roxygen_metadata = extract_roxygen_docs(content, func_pos)

        # Find function content
        remaining = content[func_pos:]
        brace_match = re.search(r"\{", remaining)
        if not brace_match:
            continue

        brace_pos = brace_match.start()
        brace_count = 1
        end_pos = brace_pos + 1

        while brace_count > 0 and end_pos < len(remaining):
            if remaining[end_pos] == "{":
                brace_count += 1
            elif remaining[end_pos] == "}":
                brace_count -= 1
            end_pos += 1

        if brace_count == 0:
            func_content = remaining[:end_pos]

            functions.append(
                {
                    "name": func_name,
                    "position": func_pos,
                    "content": func_content,
                    "docstring": roxygen_text,
                    "metadata": roxygen_metadata,
                }
            )

    return functions


def extract_julia_functions(content: str) -> List[Dict[str, Any]]:
    """Extract Julia functions from code content."""
    functions = []

    # Patterns for Julia functions
    patterns = [
        r"function\s+(\w+(?:\.\w+)*)\s*\(([^)]*)\)",  # Standard function definition
        r"(\w+(?:\.\w+)*)\s*\(([^)]*)\)\s*=\s*",  # Short-form function definition
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, content, re.DOTALL):
            func_name = match.group(1)
            func_pos = match.start()

            # Extract docstring
            docstring, metadata = extract_julia_docstring(content, func_pos)

            # Find the end of the function (simplified)
            if "function" in pattern:
                # Full function definition - find matching "end"
                remaining = content[func_pos:]
                nesting = 1
                end_pos = 0

                while nesting > 0 and end_pos < len(remaining):
                    func_match = re.search(r"\bfunction\b", remaining[end_pos:])
                    end_match = re.search(r"\bend\b", remaining[end_pos:])

                    if not end_match:
                        end_pos = len(remaining)
                        break

                    if func_match and func_match.start() < end_match.start():
                        nesting += 1
                        end_pos += func_match.start() + 8
                    else:
                        nesting -= 1
                        if nesting == 0:
                            end_pos += end_match.start() + 3
                            break
                        else:
                            end_pos += end_match.start() + 3

                func_content = remaining[:end_pos]
            else:
                # One-line function - find end of line
                line_end = content[func_pos:].find("\n")
                if line_end == -1:
                    func_content = content[func_pos:]
                else:
                    func_content = content[func_pos : func_pos + line_end]

            functions.append(
                {
                    "name": func_name,
                    "position": func_pos,
                    "content": func_content,
                    "docstring": docstring,
                    "metadata": metadata,
                }
            )

    return functions


def normalize_text(text: str) -> str:
    """
    Normalize text for better indexing and retrieval.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)

    # Remove common noise markers
    text = re.sub(r"#\s*@ts-\w+", "", text)  # TypeScript markers

    # Remove very common logging imports
    text = re.sub(r"import\s+logging.*", "", text)

    # Normalize empty docstrings
    text = re.sub(r'"""\s*"""', "", text)
    text = re.sub(r"'''\s*'''", "", text)

    return text.strip()


def detect_language_from_path(file_path: str) -> str:
    """
    Detect programming language from file path.

    Args:
        file_path: Path to the file

    Returns:
        Language name (lowercase)
    """
    ext = os.path.splitext(file_path.lower())[1]

    if ext == ".py":
        return "python"
    elif ext == ".r":
        return "r"
    elif ext == ".jl":
        return "julia"
    elif ext == ".rmd":
        return "rmarkdown"
    elif ext in [".c", ".cpp", ".cc", ".cxx", ".h", ".hpp"]:
        return "cpp"
    elif ext in [".js", ".jsx", ".ts", ".tsx"]:
        return "javascript"
    elif ext == ".java":
        return "java"
    elif ext == ".rb":
        return "ruby"
    elif ext in [".html", ".htm"]:
        return "html"
    elif ext == ".css":
        return "css"
    elif ext == ".php":
        return "php"
    elif ext == ".go":
        return "go"
    elif ext == ".rs":
        return "rust"
    elif ext == ".swift":
        return "swift"
    elif ext == ".cs":
        return "csharp"
    elif ext == ".sh":
        return "shell"
    else:
        return "unknown"
