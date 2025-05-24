"""
R file processor for GitLab RAG system.

This module processes R files, extracting functions and roxygen documentation
to create structured documents for the RAG system.
"""

from pathlib import Path
from typing import List, Optional, Union

from src.processing.base_processor import BaseProcessor, Document
from src.processing.utils import extract_r_functions


class RProcessor(BaseProcessor):
    """
    Processor for R files in GitLab projects.

    Extracts R functions and their roxygen documentation, creating
    structured documents suitable for retrieval.
    """

    def __init__(
        self,
        gitlab_backup_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        language: str = "r",
        min_content_length: int = 30,
    ):
        """
        Initialize the R processor.

        Args:
            gitlab_backup_dir: Path to the GitLab backup directory
            output_dir: Path to the output directory for processed documents
            language: Programming language to process (should be 'r')
            min_content_length: Minimum length of content to consider
        """
        super().__init__(gitlab_backup_dir, output_dir)
        self.language = language
        self.min_content_length = min_content_length

    def process(self, project_name: str) -> List[Document]:
        """
        Process R files in a project.

        Prioritizes the standard R/ directory, but also processes R files
        located elsewhere in the project.

        Args:
            project_name: Name of the project to process

        Returns:
            List of Document objects
        """
        project_dir = self.gitlab_backup_dir / project_name
        if not project_dir.exists():
            self.logger.warning(f"Project directory does not exist: {project_dir}")
            return []

        documents = []

        # First process the R/ directory, which is standard for R packages
        r_dir = project_dir / "R"
        if r_dir.exists() and r_dir.is_dir():
            r_dir_files = list(r_dir.glob("*.R"))
            self.logger.info(
                f"Found {len(r_dir_files)} R files in R/ directory of {project_name}"
            )

            for r_file in r_dir_files:
                docs = self._process_r_file(r_file, project_dir, project_name)
                documents.extend(docs)

        # Then find and process other R files in the project
        all_r_files = set(project_dir.rglob("*.R"))
        r_dir_files = (
            set(r_dir.glob("*.R")) if r_dir.exists() and r_dir.is_dir() else set()
        )
        other_r_files = all_r_files - r_dir_files

        if other_r_files:
            self.logger.info(
                f"Found {len(other_r_files)} additional R files in {project_name}"
            )

            for r_file in other_r_files:
                # Skip files in .git directory
                if ".git" in str(r_file):
                    continue

                docs = self._process_r_file(r_file, project_dir, project_name)
                documents.extend(docs)

        self.logger.info(
            f"Processed {len(all_r_files)} R files in {project_name}, extracted {len(documents)} documents"
        )
        return documents

    def _process_r_file(
        self, r_file: Path, project_dir: Path, project_name: str
    ) -> List[Document]:
        """
        Process a single R file.

        Args:
            r_file: Path to the R file
            project_dir: Path to the project directory
            project_name: Name of the project

        Returns:
            List of Document objects
        """
        documents = []

        try:
            relative_path = r_file.relative_to(project_dir)
            content, error = self.read_file(r_file)

            if error:
                self.logger.warning(f"Error reading {r_file}: {error}")
                return []

            # Skip empty or very small files
            if not content or len(content) < self.min_content_length:
                return []

            # Create a document for the whole file
            file_doc_id = self.generate_doc_id(
                project_name, "r_file", str(relative_path).replace("/", "_")
            )

            file_doc = Document(
                content=content,
                metadata={
                    "project": project_name,
                    "path": str(relative_path),
                    "type": "code_file",
                    "language": "r",
                    "filename": r_file.name,
                },
                doc_id=file_doc_id,
            )
            documents.append(file_doc)

            # Extract R functions
            functions = extract_r_functions(content)

            # Process each function
            for func in functions:
                func_name = func["name"]
                func_content = func["content"]
                roxygen_text = func["docstring"]
                roxygen_metadata = func["metadata"]

                # Skip very short functions with no meaningful documentation
                if len(func_content) < self.min_content_length and not roxygen_text:
                    continue

                # Generate a unique ID for the function
                func_id = self.generate_doc_id(
                    project_name,
                    "r_function",
                    f"{relative_path}_{func_name}".replace("/", "_"),
                )

                # Create a document for the function
                func_doc = Document(
                    content=roxygen_text + "\n" + func_content
                    if roxygen_text
                    else func_content,
                    metadata={
                        "project": project_name,
                        "path": str(relative_path),
                        "type": "r_function",
                        "name": func_name,
                        "language": "r",
                        "roxygen": roxygen_metadata,
                        "parent_file": r_file.name,
                    },
                    doc_id=func_id,
                )
                documents.append(func_doc)

        except Exception as e:
            self.logger.error(f"Error processing {r_file}: {str(e)}", exc_info=True)

        return documents
