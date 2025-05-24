"""
Document chunking for GitLab RAG system.

This module handles splitting documents into appropriately sized chunks for
embedding and retrieval, with specialized strategies for different document types.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any
import logging

from src.processing.base_processor import Document


@dataclass
class Chunk:
    """
    Represents a chunk of text ready for embedding.

    A chunk is a portion of a larger document that has been split for
    more effective embedding and retrieval.

    Attributes:
        content (str): The text content of the chunk
        metadata (Dict[str, Any]): Metadata about the chunk and its source document
        chunk_id (str): Unique identifier for the chunk
    """

    content: str
    metadata: Dict[str, Any]
    chunk_id: str


class BaseChunker:
    """
    Base class for document chunking strategies.

    This class defines the interface for all chunking strategies and provides
    common functionality.
    """

    def __init__(
        self,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50,
        max_chunk_size: int = 1024,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_overlap: Number of tokens/characters to overlap between chunks
            min_chunk_size: Minimum size of a chunk to be included
            max_chunk_size: Maximum size of a chunk
        """
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_chunks(self, document: Document) -> List[Chunk]:
        """
        Split a document into chunks.

        Args:
            document: Document to split

        Returns:
            List of Chunk objects
        """
        raise NotImplementedError("Subclasses must implement create_chunks")

    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """
        Generate a unique ID for a chunk.

        Args:
            doc_id: ID of the source document
            chunk_index: Index of the chunk within the document

        Returns:
            Unique chunk ID
        """
        return f"{doc_id}_chunk_{chunk_index}"


class SentenceChunker(BaseChunker):
    """
    Chunker that splits documents by sentences.

    This chunker is suitable for prose text like issue descriptions, comments,
    and wiki pages.
    """

    def create_chunks(self, document: Document) -> List[Chunk]:
        """
        Split a document into chunks based on sentence boundaries.

        Args:
            document: Document to split

        Returns:
            List of Chunk objects
        """
        content = document.content
        doc_id = document.doc_id
        metadata = document.metadata.copy()

        # Split by sentences using regex
        sentences = re.split(r"(?<=[.!?])\s+", content)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence would exceed max_chunk_size,
            # finalize the current chunk and start a new one
            if (
                current_length + sentence_length > self.max_chunk_size
                and current_length > 0
            ):
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunk_id = self._generate_chunk_id(doc_id, len(chunks))
                    chunk_metadata = {
                        **metadata,
                        "chunk_index": len(chunks),
                        "chunk_type": "sentence",
                        "parent_id": doc_id,
                    }
                    chunks.append(
                        Chunk(
                            content=chunk_text,
                            metadata=chunk_metadata,
                            chunk_id=chunk_id,
                        )
                    )

                # Start a new chunk, potentially with overlap
                overlap_sentences = []
                overlap_length = 0

                # Add sentences from the end of the previous chunk until we reach desired overlap
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s) + 1  # +1 for space
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_length

            # Add the current sentence to the chunk
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space

        # Add the last chunk if it's not empty
        if current_chunk and len(" ".join(current_chunk)) >= self.min_chunk_size:
            chunk_text = " ".join(current_chunk)
            chunk_id = self._generate_chunk_id(doc_id, len(chunks))
            chunk_metadata = {
                **metadata,
                "chunk_index": len(chunks),
                "chunk_type": "sentence",
                "parent_id": doc_id,
            }
            chunks.append(
                Chunk(content=chunk_text, metadata=chunk_metadata, chunk_id=chunk_id)
            )

        return chunks


class CodeChunker(BaseChunker):
    """
    Chunker specialized for code documents.

    This chunker attempts to keep functions, classes, and other code units intact
    while respecting maximum chunk sizes. It handles special cases like:

    - Python function and class definitions
    - R functions with roxygen documentation
    - Julia functions and types with docstrings

    The chunker ensures documentation stays with its associated function/class.
    """

    def create_chunks(self, document: Document) -> List[Chunk]:
        """
        Split a code document into chunks, trying to maintain logical units.

        Args:
            document: Document to split

        Returns:
            List of Chunk objects
        """
        # If this is already a function or class, treat it as a single chunk if possible
        doc_type = document.metadata.get("type", "")
        if doc_type in [
            "python_function",
            "python_class",
            "r_function",
            "julia_function",
            "julia_type",
        ]:
            content = document.content

            # If the content is small enough, return it as a single chunk
            if len(content) <= self.max_chunk_size:
                chunk_id = self._generate_chunk_id(document.doc_id, 0)
                chunk_metadata = {
                    **document.metadata,
                    "chunk_index": 0,
                    "chunk_type": "code_unit",
                    "parent_id": document.doc_id,
                }
                return [
                    Chunk(content=content, metadata=chunk_metadata, chunk_id=chunk_id)
                ]

        # For larger code files or functions that exceed max size,
        # try to split on logical boundaries
        return self._split_code_document(document)

    def _split_code_document(self, document: Document) -> List[Chunk]:
        """
        Split a code document using logical boundaries like functions, classes, and blank lines.

        Args:
            document: Document to split

        Returns:
            List of Chunk objects
        """
        content = document.content
        doc_id = document.doc_id
        metadata = document.metadata.copy()
        language = metadata.get("language", "unknown")

        chunks = []

        # Different splitting strategies based on language
        if language in ["python", "julia", "r", "cpp", "c", "java", "csharp"]:
            # Try to split on function/class definitions and blank lines
            lines = content.split("\n")

            current_chunk = []
            current_length = 0

            for line in lines:
                line_length = len(line) + 1  # +1 for newline

                # Check if this line starts a new logical unit (function, class, etc.)
                is_new_unit = self._is_code_unit_boundary(line, language, current_chunk)

                # If adding this line would exceed max_chunk_size or we're at a
                # logical boundary and have enough content, create a new chunk
                if (
                    current_length + line_length > self.max_chunk_size
                    and current_length > 0
                ) or (is_new_unit and current_length >= self.min_chunk_size):
                    chunk_text = "\n".join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunk_id = self._generate_chunk_id(doc_id, len(chunks))
                        chunk_metadata = {
                            **metadata,
                            "chunk_index": len(chunks),
                            "chunk_type": "code",
                            "parent_id": doc_id,
                        }
                        chunks.append(
                            Chunk(
                                content=chunk_text,
                                metadata=chunk_metadata,
                                chunk_id=chunk_id,
                            )
                        )

                    # If we're at a logical boundary, start fresh
                    if is_new_unit:
                        current_chunk = [line]
                        current_length = line_length
                    else:
                        # Otherwise add overlap
                        overlap_lines = []
                        overlap_length = 0

                        # Look back a certain number of lines or until we find a logical boundary
                        for i in range(
                            len(current_chunk) - 1, max(0, len(current_chunk) - 10), -1
                        ):
                            if (
                                overlap_length + len(current_chunk[i]) + 1
                                <= self.chunk_overlap
                            ):
                                overlap_lines.insert(0, current_chunk[i])
                                overlap_length += len(current_chunk[i]) + 1
                            else:
                                break

                        current_chunk = overlap_lines + [line]
                        current_length = overlap_length + line_length
                else:
                    # Add the current line to the chunk
                    current_chunk.append(line)
                    current_length += line_length

            # Add the last chunk if it's not empty
            if current_chunk and len("\n".join(current_chunk)) >= self.min_chunk_size:
                chunk_text = "\n".join(current_chunk)
                chunk_id = self._generate_chunk_id(doc_id, len(chunks))
                chunk_metadata = {
                    **metadata,
                    "chunk_index": len(chunks),
                    "chunk_type": "code",
                    "parent_id": doc_id,
                }
                chunks.append(
                    Chunk(
                        content=chunk_text, metadata=chunk_metadata, chunk_id=chunk_id
                    )
                )
        else:
            # For unknown languages, fall back to a simple line-based chunking
            chunks = self._split_by_lines(document)

        return chunks

    def _is_code_unit_boundary(
        self, line: str, language: str, current_chunk: List[str]
    ) -> bool:
        """
        Check if a line represents the start of a new logical code unit.

        Args:
            line: The line to check
            language: Programming language
            current_chunk: Lines in the current chunk

        Returns:
            True if this line starts a new logical unit
        """
        # Skip empty or whitespace-only lines
        if not line.strip():
            return False

        # Python function or class definition
        if language == "python":
            # Check for function or class definition
            if re.match(r"^\s*(def|class)\s+\w+", line):
                # Only consider it a boundary if we have enough content already
                return len(current_chunk) > 5
            return False

        # R function definition or roxygen documentation
        elif language == "r":
            # Check for roxygen comment that starts a new documentation block
            if line.strip().startswith("#'"):
                # Look back to see if there's a gap between this and previous roxygen
                empty_lines_before = True
                for prev_line in reversed(
                    current_chunk[-3:] if len(current_chunk) >= 3 else current_chunk
                ):
                    if prev_line.strip().startswith("#'"):
                        empty_lines_before = False
                        break
                    elif prev_line.strip():
                        # Found non-empty, non-roxygen line
                        empty_lines_before = True
                        break

                # If there's a gap, this is a new roxygen block - treat as boundary
                if empty_lines_before and len(current_chunk) > 5:
                    return True

            # Check for function definition
            if re.search(r"\w+\s*<-\s*function\s*\(", line):
                # Don't treat as boundary if preceded by roxygen (already handled by roxygen boundary)
                for prev_line in reversed(
                    current_chunk[-5:] if len(current_chunk) >= 5 else current_chunk
                ):
                    if prev_line.strip().startswith("#'"):
                        return False
                    elif prev_line.strip() and not prev_line.strip().startswith("#"):
                        break

                return len(current_chunk) > 5

            return False

        # Julia function or type definition
        elif language == "julia":
            if re.match(r"^\s*function\s+\w+", line) or re.match(
                r"^\s*(struct|mutable\s+struct|abstract\s+type)\s+\w+", line
            ):
                # Check if preceded by docstring (similar to R roxygen check)
                for prev_line in reversed(
                    current_chunk[-5:] if len(current_chunk) >= 5 else current_chunk
                ):
                    if '"""' in prev_line:
                        return False
                    elif prev_line.strip() and not prev_line.strip().startswith("#"):
                        break

                return len(current_chunk) > 5

            # Check for docstring start
            if '"""' in line and len(current_chunk) > 5:
                # Look back to see if there's a gap
                empty_lines_before = True
                for prev_line in reversed(
                    current_chunk[-3:] if len(current_chunk) >= 3 else current_chunk
                ):
                    if '"""' in prev_line:
                        empty_lines_before = False
                        break
                    elif prev_line.strip():
                        empty_lines_before = True
                        break

                return empty_lines_before

            return False

        # C/C++/Java style function detection
        elif language in ["cpp", "c", "java", "csharp"]:
            # Function definitions like: type name() or void name()
            if re.match(r"^\s*(\w+\s+)+\w+\s*\([^)]*\)\s*({)?", line):
                return len(current_chunk) > 5
            # Class definitions
            if re.match(r"^\s*(class|struct|enum)\s+\w+", line):
                return len(current_chunk) > 5

        # Default: not a boundary
        return False

    def _split_by_lines(self, document: Document) -> List[Chunk]:
        """
        Fall back to simple line-based chunking for unknown languages.

        Args:
            document: Document to split

        Returns:
            List of Chunk objects
        """
        content = document.content
        doc_id = document.doc_id
        metadata = document.metadata.copy()

        lines = content.split("\n")
        chunks = []

        current_chunk = []
        current_length = 0

        for line in lines:
            line_length = len(line) + 1  # +1 for newline

            # If adding this line would exceed max_chunk_size, create a new chunk
            if (
                current_length + line_length > self.max_chunk_size
                and current_length > 0
            ):
                chunk_text = "\n".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunk_id = self._generate_chunk_id(doc_id, len(chunks))
                    chunk_metadata = {
                        **metadata,
                        "chunk_index": len(chunks),
                        "chunk_type": "code_lines",
                        "parent_id": doc_id,
                    }
                    chunks.append(
                        Chunk(
                            content=chunk_text,
                            metadata=chunk_metadata,
                            chunk_id=chunk_id,
                        )
                    )

                # Start a new chunk with overlap
                overlap_lines = []
                overlap_length = 0

                # Add lines from the end of the previous chunk until we reach desired overlap
                for i in range(
                    len(current_chunk) - 1, max(0, len(current_chunk) - 5), -1
                ):
                    if overlap_length + len(current_chunk[i]) + 1 <= self.chunk_overlap:
                        overlap_lines.insert(0, current_chunk[i])
                        overlap_length += len(current_chunk[i]) + 1
                    else:
                        break

                current_chunk = overlap_lines
                current_length = overlap_length

            # Add the current line to the chunk
            current_chunk.append(line)
            current_length += line_length

        # Add the last chunk if it's not empty
        if current_chunk and len("\n".join(current_chunk)) >= self.min_chunk_size:
            chunk_text = "\n".join(current_chunk)
            chunk_id = self._generate_chunk_id(doc_id, len(chunks))
            chunk_metadata = {
                **metadata,
                "chunk_index": len(chunks),
                "chunk_type": "code_lines",
                "parent_id": doc_id,
            }
            chunks.append(
                Chunk(content=chunk_text, metadata=chunk_metadata, chunk_id=chunk_id)
            )

        return chunks


class IssueChunker(BaseChunker):
    """
    Chunker specialized for issues and merge requests.

    This chunker ensures that issue titles are included in each chunk and
    handles comments appropriately.
    """

    def create_chunks(self, document: Document) -> List[Chunk]:
        """
        Split an issue or merge request document into chunks.

        Args:
            document: Document to split

        Returns:
            List of Chunk objects
        """
        doc_type = document.metadata.get("type", "")

        # For issue and merge request comments, treat them as single chunks if possible
        if doc_type in ["issue_comment", "mr_comment"]:
            content = document.content

            # If the content is small enough, return it as a single chunk
            if len(content) <= self.max_chunk_size:
                chunk_id = self._generate_chunk_id(document.doc_id, 0)
                chunk_metadata = {
                    **document.metadata,
                    "chunk_index": 0,
                    "chunk_type": "comment",
                    "parent_id": document.doc_id,
                }
                return [
                    Chunk(content=content, metadata=chunk_metadata, chunk_id=chunk_id)
                ]
            else:
                # For longer comments, use the sentence chunker with special handling
                return self._split_comment(document)

        # For issues and merge requests, we want to keep the title with each chunk
        elif doc_type in ["issue", "merge_request"]:
            return self._split_issue_or_mr(document)

        # Fall back to sentence chunking for unknown types
        return SentenceChunker(
            chunk_overlap=self.chunk_overlap,
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=self.max_chunk_size,
        ).create_chunks(document)

    def _split_comment(self, document: Document) -> List[Chunk]:
        """
        Split a comment into chunks, preserving context about the parent issue/MR.

        Args:
            document: Comment document to split

        Returns:
            List of Chunk objects
        """
        content = document.content
        doc_id = document.doc_id
        metadata = document.metadata.copy()

        # Extract header (first line with author and context info)
        lines = content.split("\n", 1)
        header = lines[0]
        body = lines[1] if len(lines) > 1 else ""

        # Use the sentence chunker on the body
        sentence_chunks = []
        if body:
            temp_doc = Document(content=body, metadata=metadata, doc_id=doc_id)

            # Use slightly smaller chunk size to account for the header
            adjusted_size = self.max_chunk_size - len(header) - 2  # -2 for newlines
            sentence_chunker = SentenceChunker(
                chunk_overlap=self.chunk_overlap,
                min_chunk_size=self.min_chunk_size,
                max_chunk_size=adjusted_size,
            )
            sentence_chunks = sentence_chunker.create_chunks(temp_doc)

        # Add the header to each chunk
        result_chunks = []
        for i, chunk in enumerate(sentence_chunks):
            chunk_content = f"{header}\n\n{chunk.content}"
            chunk_id = self._generate_chunk_id(doc_id, i)
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "chunk_type": "comment",
                "parent_id": doc_id,
            }
            result_chunks.append(
                Chunk(content=chunk_content, metadata=chunk_metadata, chunk_id=chunk_id)
            )

        # If no chunks were created, create one for the header
        if not result_chunks and header:
            chunk_id = self._generate_chunk_id(doc_id, 0)
            chunk_metadata = {
                **metadata,
                "chunk_index": 0,
                "chunk_type": "comment",
                "parent_id": doc_id,
            }
            result_chunks.append(
                Chunk(content=header, metadata=chunk_metadata, chunk_id=chunk_id)
            )

        return result_chunks

    def _split_issue_or_mr(self, document: Document) -> List[Chunk]:
        """
        Split an issue or merge request, ensuring the title is included in each chunk.

        Args:
            document: Issue or MR document to split

        Returns:
            List of Chunk objects
        """
        content = document.content
        doc_id = document.doc_id
        metadata = document.metadata.copy()

        # Try to extract the title (first heading)
        title_match = re.match(r"# (.*?)(\n|$)", content)

        if title_match:
            title = title_match.group(0)  # Include the # for formatting
            body = content[len(title_match.group(0)) :]
        else:
            # Fallback if no title is found
            title = f"# {metadata.get('title', 'Untitled')}"
            body = content

        # If the content is small enough, return it as a single chunk
        if len(content) <= self.max_chunk_size:
            chunk_id = self._generate_chunk_id(doc_id, 0)
            chunk_metadata = {
                **metadata,
                "chunk_index": 0,
                "chunk_type": "issue_mr",
                "parent_id": doc_id,
            }
            return [Chunk(content=content, metadata=chunk_metadata, chunk_id=chunk_id)]

        # Split the body using the sentence chunker
        temp_doc = Document(content=body, metadata=metadata, doc_id=doc_id)

        # Use slightly smaller chunk size to account for the title
        adjusted_size = self.max_chunk_size - len(title) - 2  # -2 for newlines
        sentence_chunker = SentenceChunker(
            chunk_overlap=self.chunk_overlap,
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=adjusted_size,
        )
        body_chunks = sentence_chunker.create_chunks(temp_doc)

        # Add the title to each chunk
        result_chunks = []
        for i, chunk in enumerate(body_chunks):
            chunk_content = f"{title}\n\n{chunk.content}"
            chunk_id = self._generate_chunk_id(doc_id, i)
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "chunk_type": "issue_mr",
                "parent_id": doc_id,
            }
            result_chunks.append(
                Chunk(content=chunk_content, metadata=chunk_metadata, chunk_id=chunk_id)
            )

        return result_chunks


class WikiChunker(BaseChunker):
    """
    Chunker specialized for wiki pages.

    This chunker splits wiki pages by section, keeping headings with their content
    and handling hierarchical structure.
    """

    def create_chunks(self, document: Document) -> List[Chunk]:
        """
        Split a wiki page into chunks based on section headers.

        Args:
            document: Wiki page document to split

        Returns:
            List of Chunk objects
        """
        doc_type = document.metadata.get("type", "")

        # If this is already a wiki section, treat it as a single chunk if possible
        if doc_type == "wiki_section":
            content = document.content

            # If the content is small enough, return it as a single chunk
            if len(content) <= self.max_chunk_size:
                chunk_id = self._generate_chunk_id(document.doc_id, 0)
                chunk_metadata = {
                    **document.metadata,
                    "chunk_index": 0,
                    "chunk_type": "wiki_section",
                    "parent_id": document.doc_id,
                }
                return [
                    Chunk(content=content, metadata=chunk_metadata, chunk_id=chunk_id)
                ]

        # For wiki pages, split by headings
        if doc_type in ["wiki_page", "wiki_section"]:
            return self._split_by_headings(document)

        # Fall back to sentence chunking for unknown types
        return SentenceChunker(
            chunk_overlap=self.chunk_overlap,
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=self.max_chunk_size,
        ).create_chunks(document)

    def _split_by_headings(self, document: Document) -> List[Chunk]:
        """
        Split a wiki page into chunks based on markdown headings.

        Args:
            document: Wiki page document to split

        Returns:
            List of Chunk objects
        """
        content = document.content
        doc_id = document.doc_id
        metadata = document.metadata.copy()

        # If content is small enough, return it as a single chunk
        if len(content) <= self.max_chunk_size:
            chunk_id = self._generate_chunk_id(doc_id, 0)
            chunk_metadata = {
                **metadata,
                "chunk_index": 0,
                "chunk_type": "wiki",
                "parent_id": doc_id,
            }
            return [Chunk(content=content, metadata=chunk_metadata, chunk_id=chunk_id)]

        # Extract the title (if present)
        title_match = re.match(r"# (.*?)(\n|$)", content)
        wiki_title = title_match.group(0) if title_match else ""

        # Find all headings (## or more)
        heading_pattern = r"^(#{2,6})\s+(.+)$"
        lines = content.split("\n")

        chunks = []
        current_section = []
        current_length = 0
        current_heading = wiki_title if wiki_title else None

        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            heading_match = re.match(heading_pattern, line)

            # If this is a heading, it may start a new section
            if heading_match:
                # Save the current section if it's not empty
                if current_section and current_length >= self.min_chunk_size:
                    section_text = "\n".join(current_section)
                    chunk_id = self._generate_chunk_id(doc_id, len(chunks))

                    # Add the wiki title at the top if we have one and it's not already there
                    if wiki_title and not section_text.startswith(wiki_title):
                        section_text = f"{wiki_title}\n\n{section_text}"

                    chunk_metadata = {
                        **metadata,
                        "chunk_index": len(chunks),
                        "chunk_type": "wiki_section",
                        "parent_id": doc_id,
                        "section_heading": current_heading,
                    }
                    chunks.append(
                        Chunk(
                            content=section_text,
                            metadata=chunk_metadata,
                            chunk_id=chunk_id,
                        )
                    )

                # Start a new section with this heading
                current_heading = line
                current_section = [line]
                current_length = line_length
            else:
                # Check if adding this line would exceed max_chunk_size
                if (
                    current_length + line_length > self.max_chunk_size
                    and current_length > 0
                ):
                    section_text = "\n".join(current_section)
                    chunk_id = self._generate_chunk_id(doc_id, len(chunks))

                    # Add the wiki title at the top if we have one and it's not already there
                    if wiki_title and not section_text.startswith(wiki_title):
                        section_text = f"{wiki_title}\n\n{section_text}"

                    chunk_metadata = {
                        **metadata,
                        "chunk_index": len(chunks),
                        "chunk_type": "wiki_section",
                        "parent_id": doc_id,
                        "section_heading": current_heading,
                    }
                    chunks.append(
                        Chunk(
                            content=section_text,
                            metadata=chunk_metadata,
                            chunk_id=chunk_id,
                        )
                    )

                    # Start a new section with the same heading
                    current_section = [current_heading] if current_heading else []
                    current_length = len(current_heading) + 1 if current_heading else 0

                # Add the current line to the section
                current_section.append(line)
                current_length += line_length

        # Add the last section if it's not empty
        if current_section and current_length >= self.min_chunk_size:
            section_text = "\n".join(current_section)
            chunk_id = self._generate_chunk_id(doc_id, len(chunks))

            # Add the wiki title at the top if we have one and it's not already there
            if wiki_title and not section_text.startswith(wiki_title):
                section_text = f"{wiki_title}\n\n{section_text}"

            chunk_metadata = {
                **metadata,
                "chunk_index": len(chunks),
                "chunk_type": "wiki_section",
                "parent_id": doc_id,
                "section_heading": current_heading,
            }
            chunks.append(
                Chunk(content=section_text, metadata=chunk_metadata, chunk_id=chunk_id)
            )

        return chunks


class ChunkerFactory:
    """
    Factory for creating appropriate chunkers based on document type.
    """

    @staticmethod
    def create_chunker(document: Document, **kwargs) -> BaseChunker:
        """
        Create the appropriate chunker for a document based on its type.

        Args:
            document: Document to create a chunker for
            **kwargs: Additional parameters to pass to the chunker

        Returns:
            Appropriate chunker instance
        """
        doc_type = document.metadata.get("type", "")
        language = document.metadata.get("language", "")

        # Code files and functions
        if doc_type in [
            "code_file",
            "python_function",
            "python_class",
            "r_function",
            "julia_function",
            "julia_type",
        ] or language in ["python", "r", "julia", "javascript", "cpp"]:
            return CodeChunker(**kwargs)

        # Issues and merge requests
        elif doc_type in ["issue", "merge_request", "issue_comment", "mr_comment"]:
            return IssueChunker(**kwargs)

        # Wiki pages
        elif doc_type in ["wiki_page", "wiki_section"]:
            return WikiChunker(**kwargs)

        # Default to sentence chunking for other types
        else:
            return SentenceChunker(**kwargs)


def chunk_documents(
    documents: List[Document],
    chunk_overlap: int = 50,
    min_chunk_size: int = 50,
    max_chunk_size: int = 1024,
) -> List[Chunk]:
    """
    Split a list of documents into chunks using appropriate chunking strategies.

    Args:
        documents: List of documents to chunk
        chunk_overlap: Number of tokens/characters to overlap between chunks
        min_chunk_size: Minimum size of a chunk to be included
        max_chunk_size: Maximum size of a chunk

    Returns:
        List of chunks across all documents
    """
    all_chunks = []

    for document in documents:
        chunker = ChunkerFactory.create_chunker(
            document,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )

        chunks = chunker.create_chunks(document)
        all_chunks.extend(chunks)

    return all_chunks
