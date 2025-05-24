"""
Python file processor for GitLab RAG system.

This module processes Python files, extracting functions, classes, and docstrings
to create structured documents for the RAG system.
"""

import re
from pathlib import Path
from typing import List, Optional, Union

from src.processing.base_processor import BaseProcessor, Document
from src.processing.utils import extract_python_docstring, extract_python_functions


class PythonProcessor(BaseProcessor):
    """
    Processor for Python files in GitLab projects.

    Extracts Python functions, classes, and their docstrings, creating
    structured documents suitable for retrieval.
    """

    def __init__(
        self,
        gitlab_backup_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        language: str = "python",
        min_content_length: int = 30,
    ):
        """
        Initialize the Python processor.

        Args:
            gitlab_backup_dir: Path to the GitLab backup directory
            output_dir: Path to the output directory for processed documents
            language: Programming language to process (should be 'python')
            min_content_length: Minimum length of content to consider
        """
        super().__init__(gitlab_backup_dir, output_dir)
        self.language = language
        self.min_content_length = min_content_length

    def process(self, project_name: str) -> List[Document]:
        """
        Process Python files in a project.

        Args:
            project_name: Name of the project to process

        Returns:
            List of Document objects
        """
        project_dir = self.gitlab_backup_dir / project_name
        if not project_dir.exists():
            self.logger.warning(f"Project directory does not exist: {project_dir}")
            return []

        python_files = list(project_dir.rglob("*.py"))
        self.logger.info(f"Found {len(python_files)} Python files in {project_name}")

        documents = []
        processed_files = 0

        for py_file in python_files:
            # Skip files in .git directory
            if ".git" in str(py_file):
                continue

            try:
                relative_path = py_file.relative_to(project_dir)
                content, error = self.read_file(py_file)

                if error:
                    self.logger.warning(f"Error reading {py_file}: {error}")
                    continue

                # Skip empty or very small files
                if not content or len(content) < self.min_content_length:
                    continue

                # Create a document for the whole file
                file_doc_id = self.generate_doc_id(
                    project_name, "python_file", str(relative_path).replace("/", "_")
                )

                file_doc = Document(
                    content=content,
                    metadata={
                        "project": project_name,
                        "path": str(relative_path),
                        "type": "code_file",
                        "language": "python",
                        "filename": py_file.name,
                    },
                    doc_id=file_doc_id,
                )
                documents.append(file_doc)

                # Extract Python functions
                functions = extract_python_functions(content)

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
                        "python_function",
                        f"{relative_path}_{func_name}".replace("/", "_"),
                    )

                    # Create a document for the function
                    func_doc = Document(
                        content=func_content,
                        metadata={
                            "project": project_name,
                            "path": str(relative_path),
                            "type": "python_function",
                            "name": func_name,
                            "language": "python",
                            "docstring": metadata,
                            "parent_file": py_file.name,
                        },
                        doc_id=func_id,
                    )
                    documents.append(func_doc)

                # Extract Python classes
                self._extract_python_classes(
                    content, project_name, relative_path, documents
                )

                processed_files += 1
                if processed_files % 50 == 0:
                    self.logger.info(
                        f"Processed {processed_files}/{len(python_files)} Python files in {project_name}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error processing {py_file}: {str(e)}", exc_info=True
                )

        self.logger.info(
            f"Processed {len(python_files)} Python files in {project_name}, extracted {len(documents)} documents"
        )
        return documents

    def _extract_python_classes(
        self,
        content: str,
        project_name: str,
        relative_path: Path,
        documents: List[Document],
    ) -> None:
        """
        Extract Python classes from code content.

        Args:
            content: Source code content
            project_name: Name of the project
            relative_path: Relative path of the file
            documents: List to add extracted documents to
        """
        class_pattern = r"class\s+(\w+)(?:\(([^)]*)\))?\s*:"
        for match in re.finditer(class_pattern, content, re.DOTALL):
            class_name = match.group(1)
            class_pos = match.start()

            # Extract the class's docstring
            docstring, docstring_metadata = extract_python_docstring(content, class_pos)

            # Find the end of the class
            class_line_start = content.rfind("\n", 0, class_pos) + 1
            indentation = class_pos - class_line_start

            lines = content[class_pos:].split("\n")
            class_end = class_pos
            current_line_offset = 0

            for i, line in enumerate(lines):
                if i == 0:  # Skip the class definition line
                    current_line_offset += len(line) + 1
                    continue

                if line.strip() and len(line) - len(line.lstrip()) <= indentation:
                    # This line has the same or less indentation
                    class_end = class_pos + current_line_offset
                    break

                current_line_offset += len(line) + 1

            if class_end == class_pos:
                class_end = len(content)

            class_content = content[class_pos:class_end]

            # Skip small classes with no docstring
            if len(class_content) < self.min_content_length and not docstring:
                continue

            # Generate a unique ID for the class
            class_id = self.generate_doc_id(
                project_name,
                "python_class",
                f"{relative_path}_{class_name}".replace("/", "_"),
            )

            # Create a document for the class
            class_doc = Document(
                content=class_content,
                metadata={
                    "project": project_name,
                    "path": str(relative_path),
                    "type": "python_class",
                    "name": class_name,
                    "language": "python",
                    "docstring": docstring_metadata,
                    "parent_file": relative_path.name,
                },
                doc_id=class_id,
            )
            documents.append(class_doc)
