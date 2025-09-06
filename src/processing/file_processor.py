import logging
from pathlib import Path
from typing import List

from .document import Document
from .file_utils import should_process_file


class FileProcessor:
    """
    Processes code and documentation files.
    """
    
    # Code file extensions
    CODE_EXTENSIONS = {
        '.py', '.r', '.R', '.jl', '.cpp', '.hpp', '.h', '.c', '.cc',
        '.java', '.cs', '.go', '.rs', '.swift', '.rb', '.php', '.sh',
        '.bash', '.sql', '.js', '.jsx', '.ts', '.tsx', '.css',
        '.scss', '.sass', '.kt', '.lua', '.m', '.pl', '.ps1', '.scala',
        '.vb', '.tf', '.dart', '.rmd', '.Rmd'
    }
    
    # Documentation file patterns
    DOC_PATTERNS = [
        'README*', 'readme*', '*.md', '*.txt', 
        'CHANGELOG*', 'LICENSE*', 'CONTRIBUTING*'
    ]
    
    def __init__(self, gitlab_backup_dir: Path, max_file_size_mb: float = 5.0):
        self.gitlab_backup_dir = Path(gitlab_backup_dir)
        self.max_file_size_mb = max_file_size_mb
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process_code(self, project_name: str) -> List[Document]:
        """Process all code files in a project."""
        project_dir = self.gitlab_backup_dir / project_name
        if not project_dir.exists():
            return []
        
        # Find code files
        code_files = []
        for ext in self.CODE_EXTENSIONS:
            code_files.extend(project_dir.rglob(f'*{ext}'))
        
        return self._process_files(code_files, project_name, 'code', self._detect_language)
    
    def process_docs(self, project_name: str) -> List[Document]:
        """Process documentation files in a project."""
        project_dir = self.gitlab_backup_dir / project_name
        if not project_dir.exists():
            return []
        
        # Find documentation files
        doc_files = []
        for pattern in self.DOC_PATTERNS:
            doc_files.extend(project_dir.rglob(pattern))
        doc_files = list(set(doc_files))  # Remove duplicates
        
        return self._process_files(doc_files, project_name, 'documentation', self._classify_doc_type)
    
    def _process_files(self, files: List[Path], project_name: str, file_type: str, classifier_func) -> List[Document]:
        """Common file processing logic."""
        project_dir = self.gitlab_backup_dir / project_name
        documents = []
        
        self.logger.info(f"Found {len(files)} {file_type} files in {project_name}")
        
        for file_path in files:
            should_process, content = should_process_file(
                file_path, self.max_file_size_mb, logger=self.logger
            )
            if not should_process:
                continue
            
            try:
                relative_path = file_path.relative_to(project_dir)
                classification = classifier_func(file_path)
                
                doc_id = f"{project_name}_{file_type}_{str(relative_path).replace('/', '_')}"
                
                metadata = {
                    'project': project_name,
                    'path': str(relative_path),
                    'type': file_type,
                    'filename': file_path.name
                }
                
                if file_type == 'code':
                    metadata['language'] = classification
                else:
                    metadata['doc_type'] = classification
                
                documents.append(Document(
                    content=content,
                    metadata=metadata,
                    doc_id=doc_id
                ))
                
            except Exception as e:
                self.logger.warning(f"Error processing {file_path}: {e}")
        
        self.logger.info(f"Processed {len(documents)} {file_type} files for {project_name}")
        return documents
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        extension = file_path.suffix
        lang_map = {
            '.py': 'python',
            '.r': 'r', '.R': 'r',
            '.jl': 'julia',
            '.rmd': 'rmarkdown', '.Rmd': 'rmarkdown',
            '.cpp': 'cpp', '.hpp': 'cpp', '.c': 'c', '.h': 'c',
            '.java': 'java',
            '.js': 'javascript', '.jsx': 'javascript',
            '.ts': 'typescript', '.tsx': 'typescript',
            '.css': 'css', '.scss': 'scss', '.sass': 'sass',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.sh': 'shell', '.bash': 'shell',
            '.sql': 'sql'
        }
        return lang_map.get(extension.lower(), 'unknown')
    
    def _classify_doc_type(self, file_path: Path) -> str:
        """Classify document type based on filename."""
        filename_lower = file_path.name.lower()
        
        if filename_lower.startswith('readme'):
            return 'readme'
        elif filename_lower.startswith('changelog'):
            return 'changelog'
        elif filename_lower.startswith('license'):
            return 'license'
        elif filename_lower.startswith('contributing'):
            return 'contributing'
        elif filename_lower.endswith('.md'):
            return 'markdown'
        elif filename_lower.endswith('.txt'):
            return 'text'
        else:
            return 'other'