"""
Processing pipeline for GitLab RAG system.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from .document import Document
from .file_processor import FileProcessor
from .metadata_processor import MetadataProcessor


class ProcessingPipeline:
    """
    Pipeline that coordinates file and metadata processors.
    """
    
    def __init__(self, gitlab_backup_dir: Path, output_dir: Path, max_file_size_mb: float = 5.0):
        self.gitlab_backup_dir = Path(gitlab_backup_dir)
        self.output_dir = Path(output_dir)
        self.max_file_size_mb = max_file_size_mb
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # initialise processors
        self.file_processor = FileProcessor(gitlab_backup_dir, max_file_size_mb)
        self.metadata_processor = MetadataProcessor(gitlab_backup_dir)
    
    def process_project(self, project_name: str) -> Dict[str, List[Document]]:
        """Process a single project with all processors."""
        self.logger.info(f"Processing project: {project_name}")
        
        results = {}
        
        # Process with each processor
        try:
            code_docs = self.file_processor.process_code(project_name)
            if code_docs:
                results['code'] = code_docs
                self._save_documents(code_docs, project_name, 'code')
        except Exception as e:
            self.logger.error(f"Error in code processor for {project_name}: {e}")
        
        try:
            doc_docs = self.file_processor.process_docs(project_name)
            if doc_docs:
                results['documentation'] = doc_docs
                self._save_documents(doc_docs, project_name, 'documentation')
        except Exception as e:
            self.logger.error(f"Error in documentation processor for {project_name}: {e}")
        
        try:
            metadata_docs = self.metadata_processor.process(project_name)
            if metadata_docs:
                results['metadata'] = metadata_docs
                self._save_documents(metadata_docs, project_name, 'metadata')
        except Exception as e:
            self.logger.error(f"Error in metadata processor for {project_name}: {e}")
        
        total_docs = sum(len(docs) for docs in results.values())
        self.logger.info(f"Processed {project_name}: {total_docs} documents")
        
        return results
    
    def process_all_projects(self) -> Dict[str, Dict[str, List[Document]]]:
        """Process all projects."""
        projects = self._get_projects()
        self.logger.info(f"Found {len(projects)} projects to process")
        
        all_results = {}
        
        for project_name in projects:
            try:
                project_results = self.process_project(project_name)
                if project_results:
                    all_results[project_name] = project_results
            except Exception as e:
                self.logger.error(f"Error processing project {project_name}: {e}")
        
        
        return all_results
    
    def _get_projects(self) -> List[str]:
        """Get list of project directories."""
        projects = []
        for item in self.gitlab_backup_dir.iterdir():
            if item.is_dir() and not item.name.endswith('_metadata'):
                projects.append(item.name)
        return sorted(projects)
    
    def _save_documents(self, documents: List[Document], project_name: str, processor_type: str):
        """Save documents to JSON files."""
        project_dir = self.output_dir / project_name / processor_type
        project_dir.mkdir(parents=True, exist_ok=True)
        
        for doc in documents:
            doc_path = project_dir / f"{doc.doc_id}.json"
            with open(doc_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'doc_id': doc.doc_id
                }, f, ensure_ascii=False, indent=2)
    
