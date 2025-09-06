"""
Simple document chunking for GitLab RAG system.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

from src.processing.document import Document


@dataclass
class Chunk:
    """
    Represents a chunk of text ready for embedding.
    
    Attributes:
        content (str): The text content of the chunk
        metadata (Dict[str, Any]): Metadata about the chunk and its source document
        chunk_id (str): Unique identifier for the chunk
    """
    content: str
    metadata: Dict[str, Any]
    chunk_id: str


def chunk_text(
    text: str, 
    max_size: int = 1024, 
    overlap: int = 50, 
    min_size: int = 50
) -> List[str]:
    """
    Split text into chunks with overlap.
    
    Args:
        text: Text to chunk
        max_size: Maximum chunk size in characters
        overlap: Number of characters to overlap between chunks
        min_size: Minimum chunk size to include
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find end position
        end = start + max_size
        
        # If this would be the last chunk, take everything remaining
        if end >= len(text):
            chunk = text[start:]
        else:
            chunk = text[start:end]
            
            # Try to break at sentence boundary if possible
            last_sentence = max(
                chunk.rfind('. '),
                chunk.rfind('! '),
                chunk.rfind('? ')
            )
            
            if last_sentence > len(chunk) // 2:  # Only break if we're past halfway
                chunk = chunk[:last_sentence + 1]
                end = start + len(chunk)
        
        # Add chunk if it meets minimum size
        if len(chunk.strip()) >= min_size:
            chunks.append(chunk.strip())
        
        # Break if we've reached the end
        if end >= len(text):
            break
            
        # Move start position for next chunk (with overlap)
        start = max(start + 1, end - overlap)
    
    return chunks


def chunk_documents(
    documents: List[Document],
    max_chunk_size: int = 1024,
    chunk_overlap: int = 50,
    min_chunk_size: int = 50,
) -> List[Chunk]:
    """
    Split documents into chunks for embedding.
    
    Args:
        documents: List of documents to chunk
        max_chunk_size: Maximum chunk size in characters
        chunk_overlap: Characters to overlap between chunks
        min_chunk_size: Minimum chunk size to include
        
    Returns:
        List of chunks across all documents
    """
    all_chunks = []
    
    for document in documents:
        # Split document content into text chunks
        text_chunks = chunk_text(
            document.content,
            max_size=max_chunk_size,
            overlap=chunk_overlap,
            min_size=min_chunk_size
        )
        
        # Convert text chunks to Chunk objects
        for i, chunk_content in enumerate(text_chunks):
            chunk_id = f"{document.doc_id}_chunk_{i}"
            
            chunk_metadata = {
                **document.metadata,
                'chunk_index': i,
                'parent_id': document.doc_id,
                'total_doc_chunks': len(text_chunks)
            }
            
            chunk = Chunk(
                content=chunk_content,
                metadata=chunk_metadata,
                chunk_id=chunk_id
            )
            all_chunks.append(chunk)
    
    return all_chunks