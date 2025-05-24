"""
Issue processor for GitLab RAG system.

This module processes GitLab issues and their discussions, creating
structured documents for the RAG system.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from src.processing.base_processor import BaseProcessor, Document


class IssueProcessor(BaseProcessor):
    """
    Processor for GitLab issues and discussions.

    Extracts issue data from the GitLab metadata directory, including
    issue descriptions and comments.
    """

    def __init__(
        self,
        gitlab_backup_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        min_content_length: int = 30,
    ):
        """
        Initialize the issue processor.

        Args:
            gitlab_backup_dir: Path to the GitLab backup directory
            output_dir: Path to the output directory for processed documents
            min_content_length: Minimum length of content to consider
        """
        super().__init__(gitlab_backup_dir, output_dir)
        self.min_content_length = min_content_length

    def process(self, project_name: str) -> List[Document]:
        """
        Process issues for a project.

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

        # Check if issues.json exists
        issues_file = metadata_dir / "issues.json"
        if not issues_file.exists():
            self.logger.info(f"No issues.json found for {project_name}")
            return []

        try:
            with open(issues_file, "r", encoding="utf-8") as f:
                issues = json.load(f)

            self.logger.info(f"Found {len(issues)} issues for {project_name}")

            documents = []

            # Process each issue
            for issue in issues:
                issue_docs = self._process_issue(issue, metadata_dir, project_name)
                documents.extend(issue_docs)

            self.logger.info(
                f"Processed {len(issues)} issues for {project_name}, extracted {len(documents)} documents"
            )
            return documents

        except Exception as e:
            self.logger.error(
                f"Error processing issues for {project_name}: {str(e)}", exc_info=True
            )
            return []

    def _process_issue(
        self, issue: Dict[str, Any], metadata_dir: Path, project_name: str
    ) -> List[Document]:
        """
        Process a single issue and its discussions.

        Args:
            issue: Issue data
            metadata_dir: Path to the metadata directory
            project_name: Name of the project

        Returns:
            List of Document objects
        """
        documents = []

        try:
            # Extract basic issue information
            issue_id = issue.get("iid")
            issue_title = issue.get("title", "")
            issue_description = issue.get("description", "")
            issue_state = issue.get("state", "")
            issue_created_at = issue.get("created_at", "")
            issue_closed_at = issue.get("closed_at", "")
            issue_author = issue.get("author", {}).get("name", "Unknown")
            issue_assignees = [a.get("name", "") for a in issue.get("assignees", [])]
            issue_labels = [label for label in issue.get("labels", [])]

            # Create issue content
            issue_content = f"# {issue_title}\n\n"

            # Add metadata
            issue_content += f"**State:** {issue_state}\n"
            issue_content += f"**Created by:** {issue_author}\n"
            issue_content += f"**Created at:** {issue_created_at}\n"

            if issue_closed_at:
                issue_content += f"**Closed at:** {issue_closed_at}\n"

            if issue_assignees:
                issue_content += f"**Assignees:** {', '.join(issue_assignees)}\n"

            if issue_labels:
                issue_content += f"**Labels:** {', '.join(issue_labels)}\n"

            issue_content += f"\n## Description\n\n{issue_description}"

            # Create document for the issue
            issue_doc_id = self.generate_doc_id(
                project_name, "issue", f"issue_{issue_id}"
            )

            issue_doc = Document(
                content=issue_content,
                metadata={
                    "project": project_name,
                    "type": "issue",
                    "issue_id": issue_id,
                    "title": issue_title,
                    "state": issue_state,
                    "created_at": issue_created_at,
                    "closed_at": issue_closed_at,
                    "author": issue_author,
                    "labels": issue_labels,
                },
                doc_id=issue_doc_id,
            )
            documents.append(issue_doc)

            # Process issue discussions if they exist
            discussion_file = metadata_dir / f"issue_{issue_id}_discussions.json"
            if discussion_file.exists():
                discussion_docs = self._process_issue_discussions(
                    discussion_file, issue_id, project_name, issue_title
                )
                documents.extend(discussion_docs)

        except Exception as e:
            self.logger.error(
                f"Error processing issue {issue.get('iid', 'unknown')}: {str(e)}",
                exc_info=True,
            )

        return documents

    def _process_issue_discussions(
        self, discussion_file: Path, issue_id: int, project_name: str, issue_title: str
    ) -> List[Document]:
        """
        Process discussions for an issue.

        Args:
            discussion_file: Path to the discussion file
            issue_id: ID of the issue
            project_name: Name of the project
            issue_title: Title of the parent issue

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

                    # Create content for this note - include issue title for context
                    note_content = f"Comment by {author} on issue '#{issue_id}: {issue_title}' ({created_at}):\n\n{body}"

                    # Create document for this note
                    note_doc_id = self.generate_doc_id(
                        project_name,
                        "issue_comment",
                        f"issue_{issue_id}_discussion_{discussion_id}_note_{note_idx}",
                    )

                    note_doc = Document(
                        content=note_content,
                        metadata={
                            "project": project_name,
                            "type": "issue_comment",
                            "issue_id": issue_id,
                            "issue_title": issue_title,
                            "discussion_id": discussion_id,
                            "note_idx": note_idx,
                            "author": author,
                            "created_at": created_at,
                        },
                        doc_id=note_doc_id,
                    )
                    documents.append(note_doc)

        except Exception as e:
            self.logger.error(
                f"Error processing discussions for issue {issue_id}: {str(e)}",
                exc_info=True,
            )

        return documents
