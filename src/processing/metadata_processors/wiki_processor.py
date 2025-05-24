"""
Wiki processor for GitLab RAG system.

This module processes GitLab wiki pages, creating structured documents
for the RAG system.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from src.processing.base_processor import BaseProcessor, Document


class WikiProcessor(BaseProcessor):
    """
    Processor for GitLab wiki pages.

    Extracts wiki content from the GitLab metadata directory, creating
    structured documents for each wiki page.
    """

    def __init__(
        self,
        gitlab_backup_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        min_content_length: int = 50,
    ):
        """
        Initialize the wiki processor.

        Args:
            gitlab_backup_dir: Path to the GitLab backup directory
            output_dir: Path to the output directory for processed documents
            min_content_length: Minimum length of content to consider
        """
        super().__init__(gitlab_backup_dir, output_dir)
        self.min_content_length = min_content_length

    def process(self, project_name: str) -> List[Document]:
        """
        Process wiki pages for a project.

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

        # Check if wiki directory exists
        wiki_dir = metadata_dir / "wiki"
        if not wiki_dir.exists():
            self.logger.info(f"No wiki directory found for {project_name}")
            return []

        try:
            # Check for wiki pages list
            wiki_list_file = wiki_dir / "wiki_pages_list.json"
            if not wiki_list_file.exists():
                self.logger.info(f"No wiki pages list found for {project_name}")
                return []

            with open(wiki_list_file, "r", encoding="utf-8") as f:
                wiki_pages = json.load(f)

            self.logger.info(f"Found {len(wiki_pages)} wiki pages for {project_name}")

            documents = []

            # Process each wiki page
            for page in wiki_pages:
                page_docs = self._process_wiki_page(page, wiki_dir, project_name)
                documents.extend(page_docs)

            self.logger.info(
                f"Processed {len(wiki_pages)} wiki pages for {project_name}, extracted {len(documents)} documents"
            )
            return documents

        except Exception as e:
            self.logger.error(
                f"Error processing wiki pages for {project_name}: {str(e)}",
                exc_info=True,
            )
            return []

    def _process_wiki_page(
        self, page: Dict[str, Any], wiki_dir: Path, project_name: str
    ) -> List[Document]:
        """
        Process a single wiki page.

        Args:
            page: Wiki page data
            wiki_dir: Path to the wiki directory
            project_name: Name of the project

        Returns:
            List of Document objects
        """
        documents = []

        try:
            # Extract basic wiki page information
            slug = page.get("slug", "")
            title = page.get("title", "")
            format = page.get("format", "markdown")

            if not slug:
                return []

            # Try to find the content file (prefer .md file for easier processing)
            content_file = wiki_dir / f"{slug}.md"
            if not content_file.exists():
                # Try the JSON file instead
                content_file = wiki_dir / f"{slug}.json"
                if not content_file.exists():
                    self.logger.warning(
                        f"No content file found for wiki page '{title}' in {project_name}"
                    )
                    return []

            # Read the content
            if content_file.suffix == ".md":
                content, error = self.read_file(content_file)
                if error:
                    self.logger.warning(f"Error reading {content_file}: {error}")
                    return []
            else:  # JSON file
                try:
                    with open(content_file, "r", encoding="utf-8") as f:
                        page_data = json.load(f)
                    content = page_data.get("content", "")
                except Exception as e:
                    self.logger.error(
                        f"Error parsing JSON for wiki page '{title}': {str(e)}"
                    )
                    return []

            # Skip empty or very short content
            if not content or len(content) < self.min_content_length:
                return []

            # Create document for the wiki page
            wiki_doc_id = self.generate_doc_id(
                project_name, "wiki_page", f"wiki_{slug}"
            )

            wiki_doc = Document(
                content=content,
                metadata={
                    "project": project_name,
                    "type": "wiki_page",
                    "title": title,
                    "slug": slug,
                    "format": format,
                },
                doc_id=wiki_doc_id,
            )
            documents.append(wiki_doc)

            # For longer wiki pages, split into sections for better retrieval
            if len(content) > 2000:
                section_docs = self._split_wiki_page_into_sections(
                    content, slug, title, project_name
                )
                documents.extend(section_docs)

        except Exception as e:
            self.logger.error(
                f"Error processing wiki page '{page.get('title', 'unknown')}': {str(e)}",
                exc_info=True,
            )

        return documents

    def _split_wiki_page_into_sections(
        self, content: str, slug: str, title: str, project_name: str
    ) -> List[Document]:
        """
        Split a wiki page into sections based on headers.

        Args:
            content: Wiki page content
            slug: Wiki page slug
            title: Wiki page title
            project_name: Name of the project

        Returns:
            List of Document objects (one per section)
        """
        import re

        documents = []

        # Find all headers (markdown style)
        header_pattern = r"^(#{1,6})\s+(.+)$"
        lines = content.split("\n")

        sections = []
        current_section = {"title": title, "level": 0, "content": []}

        for line in lines:
            header_match = re.match(header_pattern, line)
            if header_match:
                # Start a new section if we already have content
                if current_section["content"]:
                    sections.append(current_section)

                # Create a new section
                header_level = len(header_match.group(1))
                header_text = header_match.group(2).strip()

                current_section = {
                    "title": header_text,
                    "level": header_level,
                    "content": [line],  # Include the header line in the content
                }
            else:
                # Add to current section
                current_section["content"].append(line)

        # Add the last section
        if current_section["content"]:
            sections.append(current_section)

        # Create documents for each section
        for i, section in enumerate(sections):
            section_content = "\n".join(section["content"])

            # Skip very short sections
            if len(section_content) < self.min_content_length:
                continue

            section_title = section["title"]
            section_level = section["level"]

            # Generate a unique ID for this section
            section_id = self.generate_doc_id(
                project_name, "wiki_section", f"wiki_{slug}_section_{i}"
            )

            section_doc = Document(
                content=section_content,
                metadata={
                    "project": project_name,
                    "type": "wiki_section",
                    "wiki_title": title,
                    "section_title": section_title,
                    "section_level": section_level,
                    "section_index": i,
                    "slug": slug,
                },
                doc_id=section_id,
            )
            documents.append(section_doc)

        return documents
