"""
Factory for creating appropriate processors based on file types.

This module provides a factory pattern implementation for dynamically
creating the right processor for a given file or metadata type.
"""

from pathlib import Path
from typing import Dict, Type, Optional, Union, List

from src.processing.base_processor import BaseProcessor
from src.processing.code_processors.python_processor import PythonProcessor
from src.processing.code_processors.r_processor import RProcessor
from src.processing.code_processors.julia_processor import JuliaProcessor
from src.processing.code_processors.rmarkdown_processor import RmarkdownProcessor
from src.processing.code_processors.generic_code_processor import GenericCodeProcessor
from src.processing.code_processors.documentation_processor import (
    DocumentationProcessor,
)
from src.processing.metadata_processors.issue_processor import IssueProcessor
from src.processing.metadata_processors.merge_request_processor import (
    MergeRequestProcessor,
)
from src.processing.metadata_processors.wiki_processor import WikiProcessor
from src.processing.metadata_processors.project_metadata_processor import (
    ProjectMetadataProcessor,
)


class ProcessorFactory:
    """
    Factory for creating appropriate processors for different file types.
    """

    def __init__(
        self,
        gitlab_backup_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the processor factory.

        Args:
            gitlab_backup_dir: Path to the GitLab backup directory
            output_dir: Path to the output directory for processed documents
        """
        self.gitlab_backup_dir = Path(gitlab_backup_dir)
        self.output_dir = Path(output_dir) if output_dir else None

        # Register specialized code processors
        self._specialized_code_processor_map: Dict[str, Type[BaseProcessor]] = {
            "python": PythonProcessor,
            "r": RProcessor,
            "julia": JuliaProcessor,
            "rmarkdown": RmarkdownProcessor,
        }

        # Register generic processors for other file types
        self._generic_processor_map: Dict[str, Type[BaseProcessor]] = {
            "documentation": DocumentationProcessor,
            "generic_code": GenericCodeProcessor,
        }

        # Register processors for metadata types
        self._metadata_processor_map: Dict[str, Type[BaseProcessor]] = {
            "issues": IssueProcessor,
            "merge_requests": MergeRequestProcessor,
            "wiki": WikiProcessor,
            "project_metadata": ProjectMetadataProcessor,
        }

    def create_code_processor(self, language: str) -> BaseProcessor:
        """
        Create a code processor for the specified language.

        Args:
            language: Programming language

        Returns:
            Appropriate code processor instance
        """
        processor_class = self._specialized_code_processor_map.get(
            language, GenericCodeProcessor
        )
        return processor_class(
            gitlab_backup_dir=self.gitlab_backup_dir,
            output_dir=self.output_dir,
            language=language,
        )

    def create_metadata_processor(self, metadata_type: str) -> BaseProcessor:
        """
        Create a metadata processor for the specified type.

        Args:
            metadata_type: Type of metadata to process

        Returns:
            Appropriate metadata processor instance
        """
        processor_class = self._metadata_processor_map.get(metadata_type)
        if not processor_class:
            raise ValueError(
                f"No processor registered for metadata type: {metadata_type}"
            )

        return processor_class(
            gitlab_backup_dir=self.gitlab_backup_dir, output_dir=self.output_dir
        )

    def create_generic_processor(self, processor_type: str) -> BaseProcessor:
        """
        Create a generic processor for handling various file types.

        Args:
            processor_type: Type of generic processor to create

        Returns:
            Appropriate generic processor instance
        """
        processor_class = self._generic_processor_map.get(processor_type)
        if not processor_class:
            raise ValueError(
                f"No processor registered for generic type: {processor_type}"
            )

        return processor_class(
            gitlab_backup_dir=self.gitlab_backup_dir, output_dir=self.output_dir
        )

    def create_all_processors(self) -> List[BaseProcessor]:
        """
        Create instances of all registered processors.

        Returns:
            List of all processor instances
        """
        processors = []

        # Create specialized code processors
        for language, processor_class in self._specialized_code_processor_map.items():
            processors.append(
                processor_class(
                    gitlab_backup_dir=self.gitlab_backup_dir,
                    output_dir=self.output_dir,
                    language=language,
                )
            )

        # Create generic processors
        for processor_type, processor_class in self._generic_processor_map.items():
            # For GenericCodeProcessor, set include_all_files to ensure it processes
            # files that aren't handled by specialized processors
            if processor_type == "generic_code":
                processors.append(
                    processor_class(
                        gitlab_backup_dir=self.gitlab_backup_dir,
                        output_dir=self.output_dir,
                        include_all_files=False,
                    )
                )
            else:
                processors.append(
                    processor_class(
                        gitlab_backup_dir=self.gitlab_backup_dir,
                        output_dir=self.output_dir,
                    )
                )

        # Create metadata processors
        for metadata_type, processor_class in self._metadata_processor_map.items():
            processors.append(
                processor_class(
                    gitlab_backup_dir=self.gitlab_backup_dir, output_dir=self.output_dir
                )
            )

        return processors

    def create_all_code_processors(self) -> List[BaseProcessor]:
        """
        Create instances of all code processors (specialized and generic).

        Returns:
            List of code processor instances
        """
        processors = []

        # Create specialized code processors
        for language, processor_class in self._specialized_code_processor_map.items():
            processors.append(
                processor_class(
                    gitlab_backup_dir=self.gitlab_backup_dir,
                    output_dir=self.output_dir,
                    language=language,
                )
            )

        # Add generic code processor
        processors.append(
            GenericCodeProcessor(
                gitlab_backup_dir=self.gitlab_backup_dir,
                output_dir=self.output_dir,
                include_all_files=False,
            )
        )

        # Add documentation processor
        processors.append(
            DocumentationProcessor(
                gitlab_backup_dir=self.gitlab_backup_dir, output_dir=self.output_dir
            )
        )

        return processors

    def create_all_metadata_processors(self) -> List[BaseProcessor]:
        """
        Create instances of all registered metadata processors.

        Returns:
            List of metadata processor instances
        """
        return [
            self.create_metadata_processor(meta_type)
            for meta_type in self._metadata_processor_map.keys()
        ]
