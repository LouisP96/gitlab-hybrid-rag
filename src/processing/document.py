from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Document:
    """
    Represents a processed document with content and metadata.
    
    Attributes:
        content (str): The text content of the document
        metadata (Dict[str, Any]): Metadata about the document  
        doc_id (str): Unique identifier for the document
    """
    content: str
    metadata: Dict[str, Any]
    doc_id: str