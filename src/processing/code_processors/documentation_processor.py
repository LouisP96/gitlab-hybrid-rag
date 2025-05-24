"""
Documentation file processor for GitLab RAG system.

This module processes documentation files like README.md, DESCRIPTION, LICENSE,
and other non-code text files that provide important context about projects.
"""

from pathlib import Path
from typing import List, Optional, Union

from src.processing.base_processor import BaseProcessor, Document


class DocumentationProcessor(BaseProcessor):
    """
    Processor for documentation and text files.

    Handles README files, markdown documents, project configuration files,
    and other important text files that aren't handled by specialized code processors.
    """

    # Documentation file patterns to process
    DOCUMENTATION_PATTERNS = [
        # Common README files
        "README*",
        "*.md",
        "*.markdown",
        "*.txt",
        # R package files
        "DESCRIPTION",
        "NAMESPACE",
        "NEWS",
        "CITATION",
        # License files
        "LICENSE*",
        "COPYING*",
        # Configuration files
        "*.config",
        "*.conf",
        "*.cfg",
        "*.ini",
        "*.toml",
        # Package specifications
        "requirements.txt",
        "environment.yml",
        "setup.py",
        "package.json",
        # Documentation directories (will match files inside these)
        "docs/*",
        "doc/*",
        "man/*",
    ]

    # Files to exclude even if they match patterns (typically large generated files)
    EXCLUDE_PATTERNS = [
        # Generated documentation
        "site/*",
        "build/*",
        "dist/*",
        # Lock files
        "package-lock.json",
        "yarn.lock",
        # Large data files that may have documentation extensions
        "*.min.js",
        "*.min.css",
    ]

    def __init__(
        self,
        gitlab_backup_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        min_content_length: int = 10,  # Lower threshold for docs like LICENSE
    ):
        """
        Initialize the documentation processor.

        Args:
            gitlab_backup_dir: Path to the GitLab backup directory
            output_dir: Path to the output directory for processed documents
            min_content_length: Minimum length of content to consider
        """
        super().__init__(gitlab_backup_dir, output_dir)
        self.min_content_length = min_content_length

    def process(self, project_name: str) -> List[Document]:
        """
        Process documentation files in a project.

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
        doc_files = set()

        # Find all files matching documentation patterns
        for pattern in self.DOCUMENTATION_PATTERNS:
            if "/*" in pattern:  # Handle directory patterns
                dir_pattern, file_pattern = pattern.split("/*", 1)
                for dir_path in project_dir.glob(dir_pattern):
                    if dir_path.is_dir():
                        if (
                            file_pattern
                        ):  # If there's a specific file pattern after the directory
                            doc_files.update(dir_path.glob(file_pattern))
                        else:  # Get all files in the directory
                            doc_files.update(dir_path.glob("*"))
            else:
                doc_files.update(project_dir.glob(pattern))

        # Remove excluded files
        for exclude_pattern in self.EXCLUDE_PATTERNS:
            if "/*" in exclude_pattern:  # Handle directory exclusions
                dir_pattern, file_pattern = exclude_pattern.split("/*", 1)
                for dir_path in project_dir.glob(dir_pattern):
                    if dir_path.is_dir():
                        for exclude_file in dir_path.glob(file_pattern or "*"):
                            if exclude_file in doc_files:
                                doc_files.remove(exclude_file)
            else:
                for exclude_file in project_dir.glob(exclude_pattern):
                    if exclude_file in doc_files:
                        doc_files.remove(exclude_file)

        # Remove directories from the file list
        doc_files = {f for f in doc_files if f.is_file()}

        self.logger.info(
            f"Found {len(doc_files)} documentation files to process in {project_name}"
        )

        # Process each file
        processed_files = 0
        for doc_file in doc_files:
            # Skip files in .git directory
            if ".git" in str(doc_file):
                continue

            try:
                relative_path = doc_file.relative_to(project_dir)
                content, error = self.read_file(doc_file)

                if error:
                    self.logger.warning(f"Error reading {doc_file}: {error}")
                    continue

                # Skip empty or very small files
                if not content or len(content) < self.min_content_length:
                    continue

                # Determine doc type based on filename and path
                doc_type = self._determine_doc_type(doc_file)

                # Create a document for the file
                file_doc_id = self.generate_doc_id(
                    project_name, doc_type, str(relative_path).replace("/", "_")
                )

                file_doc = Document(
                    content=content,
                    metadata={
                        "project": project_name,
                        "path": str(relative_path),
                        "type": "documentation",
                        "doc_type": doc_type,
                        "filename": doc_file.name,
                    },
                    doc_id=file_doc_id,
                )
                documents.append(file_doc)

                processed_files += 1
                if processed_files % 50 == 0:
                    self.logger.info(
                        f"Processed {processed_files}/{len(doc_files)} documentation files in {project_name}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error processing {doc_file}: {str(e)}", exc_info=True
                )

        self.logger.info(
            f"Processed {processed_files} documentation files in {project_name}, extracted {len(documents)} documents"
        )
        return documents

    def _determine_doc_type(self, file_path: Path) -> str:
        """
        Determine the type of documentation file.

        Args:
            file_path: Path to the documentation file

        Returns:
            Documentation type identifier
        """
        filename = file_path.name.lower()
        parent_dir = file_path.parent.name.lower()
        suffix = file_path.suffix.lower()

        # Check for README files
        if filename.startswith("readme"):
            return "readme"

        # Check for license files
        if filename.startswith("license") or filename.startswith("copying"):
            return "license"

        # Check for R package files
        if filename in ["description", "namespace", "citation"]:
            return "r_package_config"

        # Check for markdown
        if suffix in [".md", ".markdown"]:
            if parent_dir in ["doc", "docs"]:
                return "project_docs"
            return "markdown"

        # Check for man pages
        if parent_dir == "man":
            return "manual"

        # Check for package specs
        if filename in [
            "requirements.txt",
            "environment.yml",
            "setup.py",
            "package.json",
        ]:
            return "package_spec"

        # Check for configurations
        if suffix in [".config", ".conf", ".cfg", ".ini", ".toml", ".yaml", ".yml"]:
            return "config"

        # Default case
        return "documentation"
