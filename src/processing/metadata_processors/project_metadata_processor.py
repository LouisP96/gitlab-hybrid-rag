"""
Project metadata processor for GitLab RAG system.

This module processes GitLab project metadata, including milestones, releases,
and pipeline information, creating structured documents for the RAG system.
"""

import json
from pathlib import Path
from typing import List, Optional, Union

from src.processing.base_processor import BaseProcessor, Document


class ProjectMetadataProcessor(BaseProcessor):
    """
    Processor for GitLab project metadata.

    Extracts project metadata from the GitLab metadata directory, including
    milestones, releases, and pipeline information.
    """

    def __init__(
        self,
        gitlab_backup_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        min_content_length: int = 30,
    ):
        """
        Initialize the project metadata processor.

        Args:
            gitlab_backup_dir: Path to the GitLab backup directory
            output_dir: Path to the output directory for processed documents
            min_content_length: Minimum length of content to consider
        """
        super().__init__(gitlab_backup_dir, output_dir)
        self.min_content_length = min_content_length

    def process(self, project_name: str) -> List[Document]:
        """
        Process project metadata for a project.

        Args:
            project_name: Name of the project to process

        Returns:
            List of Document objects
        """
        # Check if metadata directory exists
        metadata_dir = self.gitlab_backup_dir / f"{project_name}_metadata"
        if not metadata_dir.exists():
            self.logger.warning(f"Metadata directory does not exist for {project_name}")
            return []

        documents = []

        # Process milestones
        milestone_docs = self._process_milestones(metadata_dir, project_name)
        documents.extend(milestone_docs)

        # Process releases
        release_docs = self._process_releases(metadata_dir, project_name)
        documents.extend(release_docs)

        # Process pipelines
        pipeline_docs = self._process_pipelines(metadata_dir, project_name)
        documents.extend(pipeline_docs)

        self.logger.info(
            f"Processed metadata for {project_name}, extracted {len(documents)} documents"
        )
        return documents

    def _process_milestones(
        self, metadata_dir: Path, project_name: str
    ) -> List[Document]:
        """
        Process milestones for a project.

        Args:
            metadata_dir: Path to the metadata directory
            project_name: Name of the project

        Returns:
            List of Document objects
        """
        milestones_file = metadata_dir / "milestones.json"
        if not milestones_file.exists():
            self.logger.info(f"No milestones.json found for {project_name}")
            return []

        try:
            with open(milestones_file, "r", encoding="utf-8") as f:
                milestones = json.load(f)

            self.logger.info(f"Found {len(milestones)} milestones for {project_name}")

            documents = []

            for milestone in milestones:
                milestone_id = milestone.get("id")
                title = milestone.get("title", "")
                description = milestone.get("description", "")
                state = milestone.get("state", "")
                due_date = milestone.get("due_date", "")
                start_date = milestone.get("start_date", "")

                # Skip milestones with no meaningful description
                if not description or len(description) < self.min_content_length:
                    description = (
                        f"Milestone: {title} (no detailed description provided)"
                    )

                # Create content for the milestone
                content = f"# Milestone: {title}\n\n"
                content += f"**State:** {state}\n"

                if start_date:
                    content += f"**Start date:** {start_date}\n"

                if due_date:
                    content += f"**Due date:** {due_date}\n"

                content += f"\n## Description\n\n{description}"

                # Create document for the milestone
                milestone_doc_id = self.generate_doc_id(
                    project_name, "milestone", f"milestone_{milestone_id}"
                )

                milestone_doc = Document(
                    content=content,
                    metadata={
                        "project": project_name,
                        "type": "milestone",
                        "milestone_id": milestone_id,
                        "title": title,
                        "state": state,
                        "due_date": due_date,
                        "start_date": start_date,
                    },
                    doc_id=milestone_doc_id,
                )
                documents.append(milestone_doc)

            return documents

        except Exception as e:
            self.logger.error(
                f"Error processing milestones for {project_name}: {str(e)}",
                exc_info=True,
            )
            return []

    def _process_releases(
        self, metadata_dir: Path, project_name: str
    ) -> List[Document]:
        """
        Process releases for a project.

        Args:
            metadata_dir: Path to the metadata directory
            project_name: Name of the project

        Returns:
            List of Document objects
        """
        releases_file = metadata_dir / "releases.json"
        if not releases_file.exists():
            self.logger.info(f"No releases.json found for {project_name}")
            return []

        try:
            with open(releases_file, "r", encoding="utf-8") as f:
                releases = json.load(f)

            self.logger.info(f"Found {len(releases)} releases for {project_name}")

            documents = []

            for release in releases:
                tag_name = release.get("tag_name", "")
                name = release.get("name", tag_name)
                description = release.get("description", "")
                created_at = release.get("created_at", "")

                # Skip releases with no meaningful description
                if not description or len(description) < self.min_content_length:
                    description = f"Release {name} (tag: {tag_name}) - No detailed description provided"

                # Create content for the release
                content = f"# Release: {name}\n\n"
                content += f"**Tag:** {tag_name}\n"
                content += f"**Created at:** {created_at}\n"

                content += f"\n## Description\n\n{description}"

                # Create document for the release
                release_doc_id = self.generate_doc_id(
                    project_name, "release", f"release_{tag_name.replace('.', '_')}"
                )

                release_doc = Document(
                    content=content,
                    metadata={
                        "project": project_name,
                        "type": "release",
                        "tag_name": tag_name,
                        "name": name,
                        "created_at": created_at,
                    },
                    doc_id=release_doc_id,
                )
                documents.append(release_doc)

            return documents

        except Exception as e:
            self.logger.error(
                f"Error processing releases for {project_name}: {str(e)}", exc_info=True
            )
            return []

    def _process_pipelines(
        self, metadata_dir: Path, project_name: str
    ) -> List[Document]:
        """
        Process CI/CD pipelines for a project.

        Args:
            metadata_dir: Path to the metadata directory
            project_name: Name of the project

        Returns:
            List of Document objects
        """
        pipelines_file = metadata_dir / "pipelines.json"
        if not pipelines_file.exists():
            self.logger.info(f"No pipelines.json found for {project_name}")
            return []

        try:
            with open(pipelines_file, "r", encoding="utf-8") as f:
                pipelines = json.load(f)

            self.logger.info(f"Found {len(pipelines)} pipelines for {project_name}")

            documents = []

            # Process only recent pipelines (e.g., last 20) to avoid too many documents
            recent_pipelines = pipelines[:20] if len(pipelines) > 20 else pipelines

            for pipeline in recent_pipelines:
                pipeline_id = pipeline.get("id")
                status = pipeline.get("status", "")
                ref = pipeline.get("ref", "")
                sha = pipeline.get("sha", "")
                created_at = pipeline.get("created_at", "")

                # Look for detailed pipeline info
                details_file = metadata_dir / f"pipeline_{pipeline_id}_details.json"
                if details_file.exists():
                    try:
                        with open(details_file, "r", encoding="utf-8") as f:
                            details = json.load(f)

                        # Create content for the pipeline
                        content = f"# Pipeline #{pipeline_id}\n\n"
                        content += f"**Status:** {status}\n"
                        content += f"**Branch:** {ref}\n"
                        content += f"**Commit:** {sha}\n"
                        content += f"**Created:** {created_at}\n\n"

                        # Add any additional details
                        content += "## Additional Details\n\n"
                        for key, value in details.items():
                            if key not in ["id", "status", "ref", "sha", "created_at"]:
                                if isinstance(value, str) and value:
                                    content += f"**{key}:** {value}\n"

                        # Create document for the pipeline
                        pipeline_doc_id = self.generate_doc_id(
                            project_name, "pipeline", f"pipeline_{pipeline_id}"
                        )

                        pipeline_doc = Document(
                            content=content,
                            metadata={
                                "project": project_name,
                                "type": "pipeline",
                                "pipeline_id": pipeline_id,
                                "status": status,
                                "ref": ref,
                                "sha": sha,
                                "created_at": created_at,
                            },
                            doc_id=pipeline_doc_id,
                        )
                        documents.append(pipeline_doc)

                        # Process pipeline jobs
                        jobs_docs = self._process_pipeline_jobs(
                            metadata_dir, pipeline_id, project_name
                        )
                        documents.extend(jobs_docs)

                    except Exception as e:
                        self.logger.error(
                            f"Error processing pipeline {pipeline_id}: {str(e)}",
                            exc_info=True,
                        )

            return documents

        except Exception as e:
            self.logger.error(
                f"Error processing pipelines for {project_name}: {str(e)}",
                exc_info=True,
            )
            return []

    def _process_pipeline_jobs(
        self, metadata_dir: Path, pipeline_id: int, project_name: str
    ) -> List[Document]:
        """
        Process jobs for a pipeline.

        Args:
            metadata_dir: Path to the metadata directory
            pipeline_id: ID of the pipeline
            project_name: Name of the project

        Returns:
            List of Document objects
        """
        jobs_file = metadata_dir / f"pipeline_{pipeline_id}_jobs.json"
        if not jobs_file.exists():
            return []

        try:
            with open(jobs_file, "r", encoding="utf-8") as f:
                jobs = json.load(f)

            documents = []

            for job in jobs:
                job_id = job.get("id")
                job_name = job.get("name", "")
                job_status = job.get("status", "")
                stage = job.get("stage", "")

                # Create content for the job
                content = f"# Pipeline Job: {job_name}\n\n"
                content += f"**Status:** {job_status}\n"
                content += f"**Stage:** {stage}\n"
                content += f"**Pipeline ID:** {pipeline_id}\n"

                # Add any additional job information
                for key, value in job.items():
                    if key not in ["id", "name", "status", "stage", "pipeline"]:
                        if isinstance(value, str) and value:
                            content += f"**{key}:** {value}\n"

                # Create document for the job
                job_doc_id = self.generate_doc_id(
                    project_name, "pipeline_job", f"pipeline_{pipeline_id}_job_{job_id}"
                )

                job_doc = Document(
                    content=content,
                    metadata={
                        "project": project_name,
                        "type": "pipeline_job",
                        "job_id": job_id,
                        "pipeline_id": pipeline_id,
                        "name": job_name,
                        "status": job_status,
                        "stage": stage,
                    },
                    doc_id=job_doc_id,
                )
                documents.append(job_doc)

            return documents

        except Exception as e:
            self.logger.error(
                f"Error processing jobs for pipeline {pipeline_id}: {str(e)}",
                exc_info=True,
            )
            return []
