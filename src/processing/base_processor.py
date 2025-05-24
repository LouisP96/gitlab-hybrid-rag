"""
Base Processor for GitLab RAG system.

This module defines the BaseProcessor class that all specialized processors
will inherit from, establishing a common interface and shared functionality
for processing different types of files and metadata from a GitLab instance.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple


@dataclass
class Document:
    """
    Represents a processed document with content and metadata.

    This is the standard output format for all processors, ensuring consistent
    document structure throughout the RAG pipeline.

    Attributes:
        content (str): The text content of the document
        metadata (Dict[str, Any]): Metadata about the document
        doc_id (str): Unique identifier for the document
    """

    content: str
    metadata: Dict[str, Any]
    doc_id: str


class BaseProcessor(ABC):
    """
    Base class for all processors in the GitLab RAG system.

    This abstract class defines the interface that all specialized processors
    must implement. It also provides common utility methods for document handling,
    error management, and standardized output.
    """

    def __init__(
        self,
        gitlab_backup_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the processor.

        Args:
            gitlab_backup_dir: Path to the GitLab backup directory
            output_dir: Path to the output directory for processed documents
            log_level: Logging level for the processor
        """
        self.gitlab_backup_dir = Path(gitlab_backup_dir)
        self.output_dir = Path(output_dir) if output_dir else None

        # Set up logging
        self.logger = self._setup_logger(log_level)

        # Validate paths
        if not self.gitlab_backup_dir.exists():
            raise ValueError(
                f"GitLab backup directory does not exist: {self.gitlab_backup_dir}"
            )

        if self.output_dir and not self.output_dir.exists():
            self.logger.info(f"Creating output directory: {self.output_dir}")
            self.output_dir.mkdir(parents=True)

    def _setup_logger(self, log_level: int) -> logging.Logger:
        """
        Set up logging for the processor.

        Args:
            log_level: Logging level

        Returns:
            Logger instance
        """
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(log_level)

        # Create console handler if no handlers exist
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @abstractmethod
    def process(self, project_name: str) -> List[Document]:
        """
        Process a project and return a list of documents.

        This is the main method that must be implemented by all subclasses.
        It should process the specified project and return a list of Document
        objects.

        Args:
            project_name: Name of the project to process

        Returns:
            List of Document objects
        """
        pass

    def process_all_projects(self) -> Dict[str, List[Document]]:
        """
        Process all projects in the GitLab backup directory.

        Returns:
            Dictionary mapping project names to lists of Document objects
        """
        projects = self._get_projects()
        result = {}

        self.logger.info(f"Processing {len(projects)} projects")
        for project in projects:
            try:
                self.logger.info(f"Processing project: {project}")
                documents = self.process(project)
                result[project] = documents
                self.logger.info(
                    f"Processed {len(documents)} documents for project {project}"
                )
            except Exception as e:
                self.logger.error(f"Error processing project {project}: {str(e)}")

        return result

    def _get_projects(self) -> List[str]:
        """
        Get a list of all projects in the GitLab backup directory.

        Returns:
            List of project names
        """
        # Find all directories that don't end with _metadata
        project_dirs = [
            d.name
            for d in self.gitlab_backup_dir.iterdir()
            if d.is_dir() and not d.name.endswith("_metadata")
        ]
        return project_dirs

    def save_documents(self, documents: List[Document], project_name: str) -> None:
        """
        Save processed documents to the output directory.

        Args:
            documents: List of Document objects to save
            project_name: Name of the project
        """
        if not self.output_dir:
            self.logger.warning("No output directory specified, skipping document save")
            return

        project_output_dir = self.output_dir / project_name
        project_output_dir.mkdir(exist_ok=True)

        self.logger.info(
            f"Saving {len(documents)} documents for project {project_name}"
        )
        for doc in documents:
            doc_path = project_output_dir / f"{doc.doc_id}.json"
            self._save_document(doc, doc_path)

    def _save_document(self, document: Document, path: Path) -> None:
        """
        Save a single document to a file.

        Args:
            document: Document object to save
            path: Path where to save the document
        """
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "content": document.content,
                    "metadata": document.metadata,
                    "doc_id": document.doc_id,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def read_file(
        self, path: Union[str, Path], encoding: str = "utf-8"
    ) -> Tuple[str, Optional[Exception]]:
        """
        Read a file with error handling.

        Args:
            path: Path to the file
            encoding: Encoding to use

        Returns:
            Tuple of (content, exception) where exception is None if successful
        """
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read(), None
        except UnicodeDecodeError:
            try:
                with open(path, "r", encoding=encoding, errors="replace") as f:
                    content = f.read()
                    self.logger.warning(
                        f"File {path} had encoding issues, used replacement characters"
                    )
                    return content, None
            except Exception as e:
                return "", e
        except Exception as e:
            return "", e

    def generate_doc_id(self, project_name: str, doc_type: str, unique_id: str) -> str:
        """
        Generate a unique document ID.

        Args:
            project_name: Name of the project
            doc_type: Type of document (e.g., 'code_file', 'issue', 'function')
            unique_id: Unique identifier within the document type

        Returns:
            Unique document ID
        """
        return f"{project_name}_{doc_type}_{unique_id}"
