"""
RMarkdown file processor for GitLab RAG system.

This module processes RMarkdown (.Rmd) files, extracting text chunks, code blocks,
and structured content to create documents for the RAG system.
"""

import re
from pathlib import Path
from typing import List, Optional, Union, Tuple

from src.processing.base_processor import BaseProcessor, Document


class RmarkdownProcessor(BaseProcessor):
    """
    Processor for RMarkdown files in GitLab projects.

    Extracts both text and code chunks from RMarkdown files, creating
    structured documents suitable for retrieval.
    """

    def __init__(
        self,
        gitlab_backup_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        language: str = "rmarkdown",
        min_text_length: int = 100,
    ):
        """
        Initialize the RMarkdown processor.

        Args:
            gitlab_backup_dir: Path to the GitLab backup directory
            output_dir: Path to the output directory for processed documents
            language: Programming language to process (should be 'rmarkdown')
            min_text_length: Minimum length of text chunks to consider
        """
        super().__init__(gitlab_backup_dir, output_dir)
        self.language = language
        self.min_text_length = min_text_length

    def process(self, project_name: str) -> List[Document]:
        """
        Process RMarkdown files in a project.

        Args:
            project_name: Name of the project to process

        Returns:
            List of Document objects
        """
        project_dir = self.gitlab_backup_dir / project_name
        if not project_dir.exists():
            self.logger.warning(f"Project directory does not exist: {project_dir}")
            return []

        rmd_files = list(project_dir.rglob("*.Rmd"))
        self.logger.info(f"Found {len(rmd_files)} RMarkdown files in {project_name}")

        documents = []

        for rmd_file in rmd_files:
            # Skip files in .git directory
            if ".git" in str(rmd_file):
                continue

            try:
                relative_path = rmd_file.relative_to(project_dir)
                content, error = self.read_file(rmd_file)

                if error:
                    self.logger.warning(f"Error reading {rmd_file}: {error}")
                    continue

                # Skip empty files
                if not content:
                    continue

                # Create a document for the whole file
                file_doc_id = self.generate_doc_id(
                    project_name, "rmd_file", str(relative_path).replace("/", "_")
                )

                file_doc = Document(
                    content=content,
                    metadata={
                        "project": project_name,
                        "path": str(relative_path),
                        "type": "code_file",
                        "language": "rmarkdown",
                        "filename": rmd_file.name,
                    },
                    doc_id=file_doc_id,
                )
                documents.append(file_doc)

                # Extract YAML header
                yaml_header = self._extract_yaml_header(content)
                if yaml_header:
                    header_doc_id = self.generate_doc_id(
                        project_name, "rmd_header", str(relative_path).replace("/", "_")
                    )

                    header_doc = Document(
                        content=yaml_header,
                        metadata={
                            "project": project_name,
                            "path": str(relative_path),
                            "type": "rmd_header",
                            "language": "rmarkdown",
                            "parent_file": rmd_file.name,
                        },
                        doc_id=header_doc_id,
                    )
                    documents.append(header_doc)

                # Split content into chunks (code and markdown)
                chunks = self._split_rmd_content(content)

                for i, (chunk_type, chunk_content, chunk_language) in enumerate(chunks):
                    # Skip very small chunks
                    if (
                        chunk_type == "markdown"
                        and len(chunk_content) < self.min_text_length
                    ):
                        continue

                    chunk_doc_id = self.generate_doc_id(
                        project_name,
                        f"rmd_{chunk_type}",
                        f"{relative_path}_chunk_{i}".replace("/", "_"),
                    )

                    chunk_doc = Document(
                        content=chunk_content,
                        metadata={
                            "project": project_name,
                            "path": str(relative_path),
                            "type": f"rmd_{chunk_type}",
                            "chunk_index": i,
                            "language": chunk_language,
                            "parent_file": rmd_file.name,
                        },
                        doc_id=chunk_doc_id,
                    )
                    documents.append(chunk_doc)

            except Exception as e:
                self.logger.error(
                    f"Error processing {rmd_file}: {str(e)}", exc_info=True
                )

        self.logger.info(
            f"Processed {len(rmd_files)} RMarkdown files in {project_name}, extracted {len(documents)} documents"
        )
        return documents

    def _extract_yaml_header(self, content: str) -> Optional[str]:
        """
        Extract YAML header from RMarkdown content.

        Args:
            content: RMarkdown content

        Returns:
            Extracted YAML header or None if not found
        """
        yaml_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.search(yaml_pattern, content, re.DOTALL)

        if match:
            return match.group(0)
        return None

    def _split_rmd_content(self, content: str) -> List[Tuple[str, str, str]]:
        """
        Split RMarkdown content into chunks of markdown and code.

        Args:
            content: RMarkdown content

        Returns:
            List of tuples (chunk_type, chunk_content, language)
            where chunk_type is either 'markdown' or 'code'
        """
        # Remove YAML header if present
        content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)

        # Pattern to match R code chunks
        code_chunk_pattern = r"```\{([a-zA-Z0-9_]+)([^}]*)\}(.*?)```"

        # Find all code chunks
        code_matches = list(re.finditer(code_chunk_pattern, content, re.DOTALL))

        if not code_matches:
            # No code chunks found, return the whole content as markdown
            return [("markdown", content, "markdown")]

        chunks = []
        last_end = 0

        for match in code_matches:
            # Add markdown content before this code chunk
            if match.start() > last_end:
                markdown_content = content[last_end : match.start()]
                if markdown_content.strip():
                    chunks.append(("markdown", markdown_content, "markdown"))

            # Add this code chunk
            code_lang = match.group(1)  # Language (e.g., r, python)
            code_content = match.group(3)

            if code_content.strip():
                chunks.append(("code", code_content, code_lang))

            last_end = match.end()

        # Add any remaining markdown content
        if last_end < len(content):
            markdown_content = content[last_end:]
            if markdown_content.strip():
                chunks.append(("markdown", markdown_content, "markdown"))

        return chunks
