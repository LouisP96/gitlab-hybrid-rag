import logging
from pathlib import Path
from typing import Optional, Tuple


def should_process_file(
    file_path: Path, 
    max_size_mb: float, 
    min_content_length: int = 50,
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, Optional[str]]:
    """
    Check if a file should be processed based on size and basic validation.
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum file size in MB
        min_content_length: Minimum content length after reading
        logger: Logger instance for warnings
        
    Returns:
        Tuple of (should_process, content_or_none)
        - should_process: True if file should be processed
        - content_or_none: File content if should process, None otherwise
    """
    if not logger:
        logger = logging.getLogger(__name__)
    
    # Skip .git directories
    if '.git' in str(file_path):
        return False, None
    
    try:
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            logger.info(f"Skipping large file ({file_size_mb:.2f} MB): {file_path}")
            return False, None
        
        # Read file content
        content = read_file_with_fallback(file_path)
        
        # Check content length
        if not content or len(content) < min_content_length:
            return False, None
        
        return True, content
        
    except Exception as e:
        logger.warning(f"Error checking file {file_path}: {e}")
        return False, None


def read_file_with_fallback(file_path: Path) -> str:
    """
    Read file content with encoding fallback.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File content as string, empty string if failed
    """
    try:
        return file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return file_path.read_text(encoding='latin1')
        except:
            return ""