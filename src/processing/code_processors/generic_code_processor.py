"""
Generic code file processor for GitLab RAG system.

This module processes code files in languages without specialized processors,
creating basic documents for the RAG system.
"""

from pathlib import Path
from typing import List, Optional, Union, Set

from src.processing.base_processor import BaseProcessor, Document
from src.processing.utils import detect_language_from_path


class GenericCodeProcessor(BaseProcessor):
    """
    Processor for code files in languages without specialized processors.

    Handles various programming languages by creating document representations
    of the source files without deep parsing of language structures.
    """

    # Extensions of files to process - excludes those handled by specialized processors
    SUPPORTED_EXTENSIONS = {
        # C/C++
        ".cpp",
        ".hpp",
        ".h",
        ".c",
        ".cc",
        # Java
        ".java",
        # C#
        ".cs",
        # Go
        ".go",
        # Rust
        ".rs",
        # Swift
        ".swift",
        # Ruby
        ".rb",
        # PHP
        ".php",
        # Shell
        ".sh",
        ".bash",
        # Other code-related files
        ".sql",
        ".proto",
        ".gradle",
        ".cmake",
        ".groovy",
        ".kt",
        ".lua",
        ".m",
        ".pl",
        ".ps1",
        ".scala",
        ".vb",
        ".tf",
        ".dart",
    }

    # Excluded extensions (already covered by specialized processors)
    EXCLUDED_EXTENSIONS = {
        ".py",
        ".r",
        ".R",
        ".jl",
        ".rmd",
        ".Rmd",
        ".md",
        ".markdown",
        ".txt",
        ".html",
        ".htm",
        ".xml", 
        ".csv",
        ".tsv",
        ".json",
        ".rds",
        ".RDS",
        ".Rds",
        ".RData",
        ".yaml",
        ".yml", 
        ".err",
        ".ttf",
        ".TTF",
        ".dia",
        ".pickle",
        ".h5",
        ".ser",
        ".soln",
        ".JPG",
        ".woff",
        ".woff2",
        ".pdf",
        ".swo",
        ".sqlite3",
        ".odt",
        ".ods",
        ".eot",
        ".otf",
        ".rda",
        ".rpm",
        ".feather",
        ".mp4",
        ".ogv",
    }

    # Binary file extensions to exclude
    BINARY_EXTENSIONS = {
        # Binary formats
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".zip",
        ".tar",
        ".gz",
        ".rar",
        ".7z",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".svg",
        ".ico",
        ".pyc",
        ".pyo",
        ".pyd",
        ".o",
        ".so",
        ".dll",
        ".exe",
        ".bin",
        ".dat",
        ".db",
        ".sqlite",
        ".class",
    }

    def __init__(
        self,
        gitlab_backup_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        language: Optional[str] = None,
        min_content_length: int = 30,
        include_all_files: bool = False,
        max_file_size_mb: float = 5.0,
    ):
        """
        Initialize the generic code processor.

        Args:
            gitlab_backup_dir: Path to the GitLab backup directory
            output_dir: Path to the output directory for processed documents
            language: Optional specific language to process
            min_content_length: Minimum length of content to consider
            include_all_files: Whether to include all files that aren't handled by other processors
        """
        super().__init__(gitlab_backup_dir, output_dir)
        self.language = language  # If specified, only process files of this language
        self.min_content_length = min_content_length
        self.include_all_files = include_all_files
        self.max_file_size_mb = max_file_size_mb

    def process(self, project_name: str) -> List[Document]:
        """
        Process code files in a project.

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
        code_files = []

        # Get code files based on supported extensions
        if self.language:
            # Process only files of the specified language
            extensions = self._get_extensions_for_language(self.language)
            self.logger.info(
                f"Processing only {self.language} files with extensions: {extensions}"
            )

            for ext in extensions:
                code_files.extend(list(project_dir.glob(f"**/*{ext}")))

        elif self.include_all_files:
            # Process all files except those with excluded or binary extensions
            all_files = list(project_dir.glob("**/*"))
            code_files = [
                f
                for f in all_files
                if f.is_file()
                and f.suffix not in self.EXCLUDED_EXTENSIONS
                and f.suffix not in self.BINARY_EXTENSIONS
            ]

            self.logger.info("Processing all non-excluded files")

        else:
            # Process files with supported extensions
            for ext in self.SUPPORTED_EXTENSIONS:
                code_files.extend(list(project_dir.glob(f"**/*{ext}")))

        # Remove duplicates and sort
        code_files = sorted(set(code_files))

        self.logger.info(f"Found {len(code_files)} files to process in {project_name}")

        # Process each file
        processed_files = 0
        for code_file in code_files:
            # Skip files in .git directory
            if ".git" in str(code_file):
                continue

            try:
                file_size_mb = code_file.stat().st_size / (1024 * 1024)  # Size in MB
                max_size_mb = self.max_file_size_mb  # Maximum file size in MB
                if file_size_mb > max_size_mb:
                    self.logger.info(
                        f"Skipping large file ({file_size_mb:.2f} MB): {code_file}"
                    )
                    continue
                relative_path = code_file.relative_to(project_dir)
                content, error = self.read_file(code_file)

                if error:
                    self.logger.warning(f"Error reading {code_file}: {error}")
                    continue

                # Skip empty or very small files
                if not content or len(content) < self.min_content_length:
                    continue

                # Check if this might be a binary file despite the extension
                if self._appears_to_be_binary(content):
                    self.logger.debug(f"Skipping likely binary file: {code_file}")
                    continue

                # Detect language from file extension
                language = detect_language_from_path(str(code_file))

                # Create a document for the whole file
                file_doc_id = self.generate_doc_id(
                    project_name,
                    f"{language}_file",
                    str(relative_path).replace("/", "_"),
                )

                file_doc = Document(
                    content=content,
                    metadata={
                        "project": project_name,
                        "path": str(relative_path),
                        "type": "code_file",
                        "language": language,
                        "filename": code_file.name,
                    },
                    doc_id=file_doc_id,
                )
                documents.append(file_doc)

                processed_files += 1
                if processed_files % 100 == 0:
                    self.logger.info(
                        f"Processed {processed_files}/{len(code_files)} files in {project_name}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error processing {code_file}: {str(e)}", exc_info=True
                )

        self.logger.info(
            f"Processed {processed_files} files in {project_name}, extracted {len(documents)} documents"
        )
        return documents

    def _get_extensions_for_language(self, language: str) -> Set[str]:
        """
        Get file extensions for a specific programming language.

        Args:
            language: Programming language name

        Returns:
            Set of file extensions
        """
        language = language.lower()

        language_extensions = {
            "python": {".py"},
            "r": {".r", ".R"},
            "julia": {".jl"},
            "rmarkdown": {".rmd", ".Rmd"},
            "cpp": {".cpp", ".hpp", ".h", ".c", ".cc"},
            "javascript": {".js", ".jsx"},
            "typescript": {".ts", ".tsx"},
            "html": {".html", ".htm"},
            "css": {".css", ".scss", ".sass"},
            "java": {".java"},
            "csharp": {".cs"},
            "go": {".go"},
            "rust": {".rs"},
            "swift": {".swift"},
            "ruby": {".rb"},
            "php": {".php"},
            "sql": {".sql"},
            "shell": {".sh", ".bash"},
        }

        return language_extensions.get(
            language, {language.startswith(".") and language or f".{language}"}
        )

    def _appears_to_be_binary(self, content: str) -> bool:
        """
        Check if content appears to be binary despite file extension.

        Args:
            content: File content

        Returns:
            True if the content appears to be binary
        """
        # Check for null bytes which indicate binary content
        if "\0" in content:
            return True

        # Check ratio of printable to non-printable characters
        printable_chars = sum(c.isprintable() or c.isspace() for c in content[:1000])
        if printable_chars < 0.75 * min(1000, len(content)):
            return True

        return False
