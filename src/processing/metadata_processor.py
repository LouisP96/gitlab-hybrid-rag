"""
Metadata processor for GitLab issues, merge requests, and wikis.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from .document import Document


class MetadataProcessor:
    """
    Processes GitLab metadata: issues, merge requests, and wikis.
    Creates simple text documents from the JSON metadata.
    """
    
    def __init__(self, gitlab_backup_dir: Path):
        self.gitlab_backup_dir = Path(gitlab_backup_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process(self, project_name: str) -> List[Document]:
        """Process all metadata for a project."""
        metadata_dir = self.gitlab_backup_dir / f"{project_name}_metadata"
        if not metadata_dir.exists():
            return []
        
        documents = []
        
        # Process issues (with discussions)
        documents.extend(self._process_issues(metadata_dir, project_name))
        
        # Process merge requests (with discussions)
        documents.extend(self._process_merge_requests(metadata_dir, project_name))
        
        # Process wikis
        documents.extend(self._process_wikis(metadata_dir, project_name))
        
        # Process milestones
        documents.extend(self._process_milestones(metadata_dir, project_name))
        
        # Process releases
        documents.extend(self._process_releases(metadata_dir, project_name))
        
        self.logger.info(f"Processed {len(documents)} metadata documents for {project_name}")
        return documents
    
    def _process_issues(self, metadata_dir: Path, project_name: str) -> List[Document]:
        """Process issues.json file."""
        issues_file = metadata_dir / "issues.json"
        if not issues_file.exists():
            return []
        
        try:
            with open(issues_file, 'r', encoding='utf-8') as f:
                issues = json.load(f)
            
            documents = []
            for issue in issues:
                title = issue.get('title', '')
                description = issue.get('description', '')
                iid = issue.get('iid', 0)
                state = issue.get('state', '')
                
                if not title and not description:
                    continue
                
                content = f"Issue #{iid}: {title}\n\n{description}"
                
                doc_id = f"{project_name}_issue_{iid}"
                document = Document(
                    content=content,
                    metadata={
                        'project': project_name,
                        'type': 'issue',
                        'issue_id': iid,
                        'title': title,
                        'state': state
                    },
                    doc_id=doc_id
                )
                documents.append(document)
                
                # Process issue discussions/comments
                discussion_docs = self._process_issue_discussions(
                    metadata_dir, project_name, iid, title
                )
                documents.extend(discussion_docs)
            
            self.logger.info(f"Processed {len(documents)} issue documents for {project_name}")
            return documents
            
        except Exception as e:
            self.logger.warning(f"Error processing issues for {project_name}: {e}")
            return []
    
    def _process_merge_requests(self, metadata_dir: Path, project_name: str) -> List[Document]:
        """Process merge_requests.json file."""
        mr_file = metadata_dir / "merge_requests.json"
        if not mr_file.exists():
            return []
        
        try:
            with open(mr_file, 'r', encoding='utf-8') as f:
                merge_requests = json.load(f)
            
            documents = []
            for mr in merge_requests:
                title = mr.get('title', '')
                description = mr.get('description', '')
                iid = mr.get('iid', 0)
                state = mr.get('state', '')
                
                if not title and not description:
                    continue
                
                content = f"Merge Request #{iid}: {title}\n\n{description}"
                
                doc_id = f"{project_name}_mr_{iid}"
                document = Document(
                    content=content,
                    metadata={
                        'project': project_name,
                        'type': 'merge_request',
                        'mr_id': iid,
                        'title': title,
                        'state': state
                    },
                    doc_id=doc_id
                )
                documents.append(document)
                
                # Process MR discussions/comments
                discussion_docs = self._process_mr_discussions(
                    metadata_dir, project_name, iid, title
                )
                documents.extend(discussion_docs)
            
            self.logger.info(f"Processed {len(documents)} merge request documents for {project_name}")
            return documents
            
        except Exception as e:
            self.logger.warning(f"Error processing merge requests for {project_name}: {e}")
            return []
    
    def _process_wikis(self, metadata_dir: Path, project_name: str) -> List[Document]:
        """Process wiki markdown files."""
        wiki_dir = metadata_dir / "wiki"
        if not wiki_dir.exists():
            return []
        
        documents = []
        
        # Find all .md wiki files
        wiki_files = list(wiki_dir.glob("*.md"))
        
        for wiki_file in wiki_files:
            try:
                content = wiki_file.read_text(encoding='utf-8')
                if not content or len(content) < 50:
                    continue
                
                page_name = wiki_file.stem
                doc_id = f"{project_name}_wiki_{page_name}"
                
                document = Document(
                    content=content,
                    metadata={
                        'project': project_name,
                        'type': 'wiki',
                        'page_name': page_name
                    },
                    doc_id=doc_id
                )
                documents.append(document)
                
            except Exception as e:
                self.logger.warning(f"Error processing wiki file {wiki_file}: {e}")
        
        self.logger.info(f"Processed {len(documents)} wiki pages for {project_name}")
        return documents
    
    def _process_issue_discussions(self, metadata_dir: Path, project_name: str, 
                                  issue_iid: int, issue_title: str) -> List[Document]:
        """Process discussion/comments for a specific issue."""
        discussion_file = metadata_dir / f"issue_{issue_iid}_discussions.json"
        if not discussion_file.exists():
            return []
        
        try:
            with open(discussion_file, 'r', encoding='utf-8') as f:
                discussions = json.load(f)
            
            documents = []
            for discussion in discussions:
                notes = discussion.get('notes', [])
                
                for note in notes:
                    body = note.get('body', '')
                    author = note.get('author', {}).get('name', 'Unknown')
                    created_at = note.get('created_at', '')
                    
                    if not body or len(body) < 10:  # Skip very short comments
                        continue
                    
                    content = f"Issue #{issue_iid} Comment: {issue_title}\nAuthor: {author}\nDate: {created_at}\n\n{body}"
                    
                    doc_id = f"{project_name}_issue_{issue_iid}_comment_{note.get('id', 'unknown')}"
                    
                    document = Document(
                        content=content,
                        metadata={
                            'project': project_name,
                            'type': 'issue_comment',
                            'issue_id': issue_iid,
                            'issue_title': issue_title,
                            'author': author,
                            'created_at': created_at
                        },
                        doc_id=doc_id
                    )
                    documents.append(document)
            
            return documents
            
        except Exception as e:
            self.logger.warning(f"Error processing issue {issue_iid} discussions for {project_name}: {e}")
            return []
    
    def _process_mr_discussions(self, metadata_dir: Path, project_name: str, 
                               mr_iid: int, mr_title: str) -> List[Document]:
        """Process discussion/comments for a specific merge request."""
        discussion_file = metadata_dir / f"mr_{mr_iid}_discussions.json"
        if not discussion_file.exists():
            return []
        
        try:
            with open(discussion_file, 'r', encoding='utf-8') as f:
                discussions = json.load(f)
            
            documents = []
            for discussion in discussions:
                notes = discussion.get('notes', [])
                
                for note in notes:
                    body = note.get('body', '')
                    author = note.get('author', {}).get('name', 'Unknown')
                    created_at = note.get('created_at', '')
                    
                    if not body or len(body) < 10:  # Skip very short comments
                        continue
                    
                    content = f"Merge Request #{mr_iid} Comment: {mr_title}\nAuthor: {author}\nDate: {created_at}\n\n{body}"
                    
                    doc_id = f"{project_name}_mr_{mr_iid}_comment_{note.get('id', 'unknown')}"
                    
                    document = Document(
                        content=content,
                        metadata={
                            'project': project_name,
                            'type': 'mr_comment',
                            'mr_id': mr_iid,
                            'mr_title': mr_title,
                            'author': author,
                            'created_at': created_at
                        },
                        doc_id=doc_id
                    )
                    documents.append(document)
            
            return documents
            
        except Exception as e:
            self.logger.warning(f"Error processing MR {mr_iid} discussions for {project_name}: {e}")
            return []
    
    def _process_milestones(self, metadata_dir: Path, project_name: str) -> List[Document]:
        """Process milestones.json file."""
        milestones_file = metadata_dir / "milestones.json"
        if not milestones_file.exists():
            return []
        
        try:
            with open(milestones_file, 'r', encoding='utf-8') as f:
                milestones = json.load(f)
            
            documents = []
            for milestone in milestones:
                title = milestone.get('title', '')
                description = milestone.get('description', '')
                state = milestone.get('state', '')
                due_date = milestone.get('due_date', '')
                milestone_id = milestone.get('id', 0)
                
                if not title and not description:
                    continue
                
                content = f"Milestone: {title}\nState: {state}"
                if due_date:
                    content += f"\nDue Date: {due_date}"
                if description:
                    content += f"\n\nDescription:\n{description}"
                
                doc_id = f"{project_name}_milestone_{milestone_id}"
                document = Document(
                    content=content,
                    metadata={
                        'project': project_name,
                        'type': 'milestone',
                        'milestone_id': milestone_id,
                        'title': title,
                        'state': state,
                        'due_date': due_date
                    },
                    doc_id=doc_id
                )
                documents.append(document)
            
            self.logger.info(f"Processed {len(documents)} milestones for {project_name}")
            return documents
            
        except Exception as e:
            self.logger.warning(f"Error processing milestones for {project_name}: {e}")
            return []
    
    def _process_releases(self, metadata_dir: Path, project_name: str) -> List[Document]:
        """Process releases.json file."""
        releases_file = metadata_dir / "releases.json"
        if not releases_file.exists():
            return []
        
        try:
            with open(releases_file, 'r', encoding='utf-8') as f:
                releases = json.load(f)
            
            documents = []
            for release in releases:
                tag_name = release.get('tag_name', '')
                name = release.get('name', '')
                description = release.get('description', '')
                released_at = release.get('released_at', '')
                
                if not tag_name and not name and not description:
                    continue
                
                title = name or tag_name or 'Unnamed Release'
                content = f"Release: {title}"
                if released_at:
                    content += f"\nReleased: {released_at}"
                if description:
                    content += f"\n\nRelease Notes:\n{description}"
                
                doc_id = f"{project_name}_release_{tag_name or name}".replace('/', '_')
                document = Document(
                    content=content,
                    metadata={
                        'project': project_name,
                        'type': 'release',
                        'tag_name': tag_name,
                        'name': name,
                        'released_at': released_at
                    },
                    doc_id=doc_id
                )
                documents.append(document)
            
            self.logger.info(f"Processed {len(documents)} releases for {project_name}")
            return documents
            
        except Exception as e:
            self.logger.warning(f"Error processing releases for {project_name}: {e}")
            return []