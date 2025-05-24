"""
Merge request processor for GitLab RAG system.

This module processes GitLab merge requests and their discussions, creating
structured documents for the RAG system.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from src.processing.base_processor import BaseProcessor, Document


class MergeRequestProcessor(BaseProcessor):
    """
    Processor for GitLab merge requests and discussions.

    Extracts merge request data from the GitLab metadata directory, including
    merge request descriptions and comments.
    """

    def __init__(
        self,
        gitlab_backup_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        min_content_length: int = 30,
    ):
        """
        Initialize the merge request processor.

        Args:
            gitlab_backup_dir: Path to the GitLab backup directory
            output_dir: Path to the output directory for processed documents
            min_content_length: Minimum length of content to consider
        """
        super().__init__(gitlab_backup_dir, output_dir)
        self.min_content_length = min_content_length

    def process(self, project_name: str) -> List[Document]:
        """
        Process merge requests for a project.

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

        # Check if merge_requests.json exists
        mrs_file = metadata_dir / "merge_requests.json"
        if not mrs_file.exists():
            self.logger.info(f"No merge_requests.json found for {project_name}")
            return []

        try:
            with open(mrs_file, "r", encoding="utf-8") as f:
                merge_requests = json.load(f)

            self.logger.info(
                f"Found {len(merge_requests)} merge requests for {project_name}"
            )

            documents = []

            # Process each merge request
            for mr in merge_requests:
                mr_docs = self._process_merge_request(mr, metadata_dir, project_name)
                documents.extend(mr_docs)

            self.logger.info(
                f"Processed {len(merge_requests)} merge requests for {project_name}, extracted {len(documents)} documents"
            )
            return documents

        except Exception as e:
            self.logger.error(
                f"Error processing merge requests for {project_name}: {str(e)}",
                exc_info=True,
            )
            return []

    def _process_merge_request(
        self, mr: Dict[str, Any], metadata_dir: Path, project_name: str
    ) -> List[Document]:
        """
        Process a single merge request and its discussions.

        Args:
            mr: Merge request data
            metadata_dir: Path to the metadata directory
            project_name: Name of the project

        Returns:
            List of Document objects
        """
        documents = []

        try:
            # Extract basic merge request information
            mr_id = mr.get("iid")
            mr_title = mr.get("title", "")
            mr_description = mr.get("description", "")
            mr_state = mr.get("state", "")
            mr_created_at = mr.get("created_at", "")
            mr_merged_at = mr.get("merged_at", "")
            mr_closed_at = mr.get("closed_at", "")
            mr_author = mr.get("author", {}).get("name", "Unknown")
            mr_assignees = [a.get("name", "") for a in mr.get("assignees", [])]
            mr_source_branch = mr.get("source_branch", "")
            mr_target_branch = mr.get("target_branch", "")
            mr_labels = [label for label in mr.get("labels", [])]

            # Create merge request content
            mr_content = f"# {mr_title}\n\n"

            # Add metadata
            mr_content += f"**State:** {mr_state}\n"
            mr_content += f"**Created by:** {mr_author}\n"
            mr_content += f"**Created at:** {mr_created_at}\n"
            mr_content += f"**Branches:** {mr_source_branch} → {mr_target_branch}\n"

            if mr_merged_at:
                mr_content += f"**Merged at:** {mr_merged_at}\n"
            elif mr_closed_at:
                mr_content += f"**Closed at:** {mr_closed_at}\n"

            if mr_assignees:
                mr_content += f"**Assignees:** {', '.join(mr_assignees)}\n"

            if mr_labels:
                mr_content += f"**Labels:** {', '.join(mr_labels)}\n"

            mr_content += f"\n## Description\n\n{mr_description}"

            # Create document for the merge request
            mr_doc_id = self.generate_doc_id(
                project_name, "merge_request", f"mr_{mr_id}"
            )

            mr_doc = Document(
                content=mr_content,
                metadata={
                    "project": project_name,
                    "type": "merge_request",
                    "mr_id": mr_id,
                    "title": mr_title,
                    "state": mr_state,
                    "created_at": mr_created_at,
                    "merged_at": mr_merged_at,
                    "closed_at": mr_closed_at,
                    "author": mr_author,
                    "source_branch": mr_source_branch,
                    "target_branch": mr_target_branch,
                    "labels": mr_labels,
                },
                doc_id=mr_doc_id,
            )
            documents.append(mr_doc)

            # Process merge request discussions if they exist
            discussion_file = metadata_dir / f"mr_{mr_id}_discussions.json"
            if discussion_file.exists():
                discussion_docs = self._process_mr_discussions(
                    discussion_file, mr_id, project_name
                )
                documents.extend(discussion_docs)

        except Exception as e:
            self.logger.error(
                f"Error processing merge request {mr.get('iid', 'unknown')}: {str(e)}",
                exc_info=True,
            )

        return documents

    def _process_mr_discussions(
        self, discussion_file: Path, mr_id: int, project_name: str
    ) -> List[Document]:
        """
        Process discussions for a merge request.

        Args:
            discussion_file: Path to the discussion file
            mr_id: ID of the merge request
            project_name: Name of the project

        Returns:
            List of Document objects
        """
        documents = []

        try:
            with open(discussion_file, "r", encoding="utf-8") as f:
                discussions = json.load(f)

            for discussion in discussions:
                discussion_id = discussion.get("id", "")
                notes = discussion.get("notes", [])

                for note_idx, note in enumerate(notes):
                    author = note.get("author", {}).get("name", "Unknown")
                    body = note.get("body", "")
                    created_at = note.get("created_at", "")

                    # Skip very short comments
                    if not body or len(body) < self.min_content_length:
                        continue

                    # Get position information (for code comments)
                    position = note.get("position", {})
                    is_code_comment = position is not None and len(position) > 0
                    file_path = (
                        position.get("new_path", "") if is_code_comment else None
                    )

                    # Create content for this note
                    if is_code_comment and file_path:
                        note_content = f"Code comment by {author} on {created_at} in file {file_path}:\n\n{body}"
                    else:
                        note_content = f"Comment by {author} on {created_at}:\n\n{body}"

                    # Create document for this note
                    note_doc_id = self.generate_doc_id(
                        project_name,
                        "mr_comment",
                        f"mr_{mr_id}_discussion_{discussion_id}_note_{note_idx}",
                    )

                    metadata = {
                        "project": project_name,
                        "type": "mr_comment",
                        "mr_id": mr_id,
                        "discussion_id": discussion_id,
                        "note_idx": note_idx,
                        "author": author,
                        "created_at": created_at,
                        "is_code_comment": is_code_comment,
                    }

                    # Add position information for code comments
                    if is_code_comment:
                        metadata["file_path"] = file_path
                        metadata["position"] = position

                    note_doc = Document(
                        content=note_content, metadata=metadata, doc_id=note_doc_id
                    )
                    documents.append(note_doc)

        except Exception as e:
            self.logger.error(
                f"Error processing discussions for merge request {mr_id}: {str(e)}",
                exc_info=True,
            )

        return documents
