"""
Julia file processor for GitLab RAG system.

This module processes Julia files, extracting functions, types, and docstrings
to create structured documents for the RAG system.
"""

import re
from pathlib import Path
from typing import List, Optional, Union

from src.processing.base_processor import BaseProcessor, Document
from src.processing.utils import extract_julia_docstring, extract_julia_functions


class JuliaProcessor(BaseProcessor):
    """
    Processor for Julia files in GitLab projects.

    Extracts Julia functions, types, and their docstrings, creating
    structured documents suitable for retrieval.
    """

    def __init__(
        self,
        gitlab_backup_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        language: str = "julia",
        min_content_length: int = 30,
    ):
        """
        Initialize the Julia processor.

        Args:
            gitlab_backup_dir: Path to the GitLab backup directory
            output_dir: Path to the output directory for processed documents
            language: Programming language to process (should be 'julia')
            min_content_length: Minimum length of content to consider
        """
        super().__init__(gitlab_backup_dir, output_dir)
        self.language = language
        self.min_content_length = min_content_length

    def process(self, project_name: str) -> List[Document]:
        """
        Process Julia files in a project.

        Args:
            project_name: Name of the project to process

        Returns:
            List of Document objects
        """
        project_dir = self.gitlab_backup_dir / project_name
        if not project_dir.exists():
            self.logger.warning(f"Project directory does not exist: {project_dir}")
            return []

        julia_files = list(project_dir.rglob("*.jl"))
        self.logger.info(f"Found {len(julia_files)} Julia files in {project_name}")

        documents = []
        processed_files = 0

        for jl_file in julia_files:
            # Skip files in .git directory
            if ".git" in str(jl_file):
                continue

            try:
                relative_path = jl_file.relative_to(project_dir)
                content, error = self.read_file(jl_file)

                if error:
                    self.logger.warning(f"Error reading {jl_file}: {error}")
                    continue

                # Skip empty or very small files
                if not content or len(content) < self.min_content_length:
                    continue

                # Create a document for the whole file
                file_doc_id = self.generate_doc_id(
                    project_name, "julia_file", str(relative_path).replace("/", "_")
                )

                file_doc = Document(
                    content=content,
                    metadata={
                        "project": project_name,
                        "path": str(relative_path),
                        "type": "code_file",
                        "language": "julia",
                        "filename": jl_file.name,
                    },
                    doc_id=file_doc_id,
                )
                documents.append(file_doc)

                # Extract Julia functions
                functions = extract_julia_functions(content)

                # Process each function
                for func in functions:
                    func_name = func["name"]
                    func_content = func["content"]
                    docstring = func["docstring"]
                    metadata = func["metadata"]

                    # Skip very short functions with no meaningful docstrings
                    if len(func_content) < self.min_content_length and not docstring:
                        continue

                    # Generate a unique ID for the function
                    func_id = self.generate_doc_id(
                        project_name,
                        "julia_function",
                        f"{relative_path}_{func_name}".replace("/", "_"),
                    )

                    # Create a document for the function
                    func_doc = Document(
                        content=func_content,
                        metadata={
                            "project": project_name,
                            "path": str(relative_path),
                            "type": "julia_function",
                            "name": func_name,
                            "language": "julia",
                            "docstring": metadata,
                            "parent_file": jl_file.name,
                        },
                        doc_id=func_id,
                    )
                    documents.append(func_doc)

                # Extract Julia types and structs
                self._extract_julia_types(
                    content, project_name, relative_path, documents
                )

                processed_files += 1
                if processed_files % 50 == 0:
                    self.logger.info(
                        f"Processed {processed_files}/{len(julia_files)} Julia files in {project_name}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error processing {jl_file}: {str(e)}", exc_info=True
                )

        self.logger.info(
            f"Processed {len(julia_files)} Julia files in {project_name}, extracted {len(documents)} documents"
        )
        return documents

    def _extract_julia_types(
        self,
        content: str,
        project_name: str,
        relative_path: Path,
        documents: List[Document],
    ) -> None:
        """
        Extract Julia types (structs, mutable structs, abstract types) from code content.

        Args:
            content: Source code content
            project_name: Name of the project
            relative_path: Relative path of the file
            documents: List to add extracted documents to
        """
        # Patterns for Julia types
        type_patterns = [
            (r"struct\s+(\w+)", "struct"),
            (r"mutable\s+struct\s+(\w+)", "mutable struct"),
            (r"abstract\s+type\s+(\w+)", "abstract type"),
        ]

        for pattern, kind in type_patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                if kind == "abstract type":
                    type_name = match.group(1)
                else:
                    type_name = match.group(1)

                type_pos = match.start()

                # Extract the type's docstring
                docstring, docstring_metadata = extract_julia_docstring(
                    content, type_pos
                )

                # Find the end of the type definition
                next_content = content[type_pos:]
                # Look for matching "end" keyword
                end_match = re.search(r"\bend\b", next_content)

                if not end_match:
                    # No "end" found, use a reasonable limit
                    type_end = type_pos + min(1000, len(next_content))
                else:
                    type_end = type_pos + end_match.end()

                # Extract the entire type definition
                type_content = content[type_pos:type_end]

                # Skip small types with no docstring
                if len(type_content) < self.min_content_length and not docstring:
                    continue

                # Generate a unique ID for the type
                type_id = self.generate_doc_id(
                    project_name,
                    "julia_type",
                    f"{relative_path}_{type_name}".replace("/", "_"),
                )

                # Create a document for the type
                type_doc = Document(
                    content=type_content,
                    metadata={
                        "project": project_name,
                        "path": str(relative_path),
                        "type": "julia_type",
                        "name": type_name,
                        "language": "julia",
                        "kind": kind,
                        "docstring": docstring_metadata,
                        "parent_file": relative_path.name,
                    },
                    doc_id=type_id,
                )
                documents.append(type_doc)
