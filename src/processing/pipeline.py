"""
Processing pipeline for GitLab RAG system.

This module orchestrates the full processing workflow, handling multiple projects
and combining the outputs of different processors.
"""

import logging
from pathlib import Path
from typing import Dict, List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from src.processing.base_processor import Document
from src.processing.processor_factory import ProcessorFactory


class ProcessingPipeline:
    """
    Orchestrates the processing of GitLab projects for RAG.

    This class coordinates the execution of all processors on each project,
    handling parallel processing and result aggregation.
    """

    def __init__(
        self,
        gitlab_backup_dir: Union[str, Path],
        output_dir: Union[str, Path],
        max_workers: int = 4,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the processing pipeline.

        Args:
            gitlab_backup_dir: Path to the GitLab backup directory
            output_dir: Path to the output directory for processed documents
            max_workers: Maximum number of worker threads
            log_level: Logging level
        """
        self.gitlab_backup_dir = Path(gitlab_backup_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers

        # Set up logging
        self.logger = self._setup_logger(log_level)

        # Create output directory if it doesn't exist
        if not self.output_dir.exists():
            self.logger.info(f"Creating output directory: {self.output_dir}")
            self.output_dir.mkdir(parents=True)

        # Initialize processor factory
        self.processor_factory = ProcessorFactory(
            gitlab_backup_dir=self.gitlab_backup_dir, output_dir=self.output_dir
        )

    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Set up logging for the pipeline."""
        logger = logging.getLogger("ProcessingPipeline")
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

    def process_project(self, project_name: str) -> Dict[str, List[Document]]:
        """
        Process a single project using all registered processors.

        Args:
            project_name: Name of the project to process

        Returns:
            Dictionary mapping processor names to lists of documents
        """
        self.logger.info(f"Processing project: {project_name}")

        # Get all processors
        processors = self.processor_factory.create_all_processors()

        results = {}
        for processor in processors:
            processor_name = processor.__class__.__name__
            try:
                self.logger.info(f"Running {processor_name} on {project_name}")
                documents = processor.process(project_name)

                if documents:
                    self.logger.info(
                        f"{processor_name} extracted {len(documents)} documents from {project_name}"
                    )
                    results[processor_name] = documents

                    # Save documents if output directory is specified
                    processor_output_dir = (
                        self.output_dir / project_name / processor_name.lower()
                    )
                    processor_output_dir.mkdir(parents=True, exist_ok=True)
                    self._save_documents(documents, processor_output_dir)
                else:
                    self.logger.info(
                        f"{processor_name} extracted no documents from {project_name}"
                    )
            except Exception as e:
                self.logger.error(
                    f"Error processing {project_name} with {processor_name}: {str(e)}",
                    exc_info=True,
                )

        return results

    def process_all_projects(
        self, parallel: bool = True
    ) -> Dict[str, Dict[str, List[Document]]]:
        """
        Process all projects in the GitLab backup directory.

        Args:
            parallel: Whether to process projects in parallel

        Returns:
            Nested dictionary mapping project names to processor names to document lists
        """
        projects = self._get_projects()
        self.logger.info(f"Found {len(projects)} projects to process")

        results = {}
        if parallel and len(projects) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_project = {
                    executor.submit(self.process_project, project): project
                    for project in projects
                }

                for future in as_completed(future_to_project):
                    project = future_to_project[future]
                    try:
                        project_results = future.result()
                        results[project] = project_results
                        self.logger.info(f"Completed processing project: {project}")
                    except Exception as e:
                        self.logger.error(
                            f"Error processing project {project}: {str(e)}",
                            exc_info=True,
                        )
        else:
            for project in projects:
                try:
                    project_results = self.process_project(project)
                    results[project] = project_results
                except Exception as e:
                    self.logger.error(
                        f"Error processing project {project}: {str(e)}", exc_info=True
                    )

        # Generate summary
        self._generate_processing_summary(results)

        return results

    def _get_projects(self) -> List[str]:
        """Get a list of all projects in the GitLab backup directory."""
        project_dirs = [
            d.name
            for d in self.gitlab_backup_dir.iterdir()
            if d.is_dir() and not d.name.endswith("_metadata")
        ]
        return project_dirs

    def _save_documents(self, documents: List[Document], output_dir: Path) -> None:
        """
        Save documents to JSON files.

        Args:
            documents: List of documents to save
            output_dir: Directory to save the documents to
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for doc in documents:
            doc_path = output_dir / f"{doc.doc_id}.json"
            with open(doc_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "doc_id": doc.doc_id,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    def _generate_processing_summary(
        self, results: Dict[str, Dict[str, List[Document]]]
    ) -> None:
        """
        Generate a summary of processing results.

        Args:
            results: Nested dictionary mapping projects to processors to documents
        """
        summary = {
            "total_projects": len(results),
            "total_documents": 0,
            "documents_by_project": {},
            "documents_by_processor": {},
        }

        for project, project_results in results.items():
            project_doc_count = 0

            for processor, documents in project_results.items():
                doc_count = len(documents)
                project_doc_count += doc_count

                # Update processor counts
                if processor not in summary["documents_by_processor"]:
                    summary["documents_by_processor"][processor] = 0
                summary["documents_by_processor"][processor] += doc_count

            # Update project counts
            summary["documents_by_project"][project] = project_doc_count
            summary["total_documents"] += project_doc_count

        # Save summary to file
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(
            f"Processing summary: {summary['total_documents']} documents extracted from {summary['total_projects']} projects"
        )

        # Log top projects and processors
        top_projects = sorted(
            summary["documents_by_project"].items(), key=lambda x: x[1], reverse=True
        )[:5]
        top_processors = sorted(
            summary["documents_by_processor"].items(), key=lambda x: x[1], reverse=True
        )

        self.logger.info("Top projects by document count:")
        for project, count in top_projects:
            self.logger.info(f"  {project}: {count} documents")

        self.logger.info("Documents by processor type:")
        for processor, count in top_processors:
            self.logger.info(f"  {processor}: {count} documents")
