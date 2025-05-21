"""
Helpers Module

This module provides general utility functions used throughout the application.
It includes functions for URL handling, file operations, text processing,
data conversion, and other common tasks.
"""

import os
import re
import json
import uuid
import hashlib
import datetime
import urllib.parse
from typing import Any, Dict, List, Optional, Union, Tuple, Set, TypeVar, Callable
from pathlib import Path

# Import logger
from apps.utils.logger import get_logger

# Set up logger
logger = get_logger(__name__)

# Type variables for generic functions
T = TypeVar('T')
U = TypeVar('U')

"""Helper functions for the application."""

import uuid
import json
import os
import time
from typing import Any, Dict, List, Optional, Union

def generate_uuid() -> str:
    """Generate a unique UUID."""
    return str(uuid.uuid4())

def read_json_file(file_path: str) -> Dict[str, Any]:
    """Read a JSON file and return its contents as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json_file(file_path: str, data: Dict[str, Any]) -> None:
    """Write a dictionary to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def get_timestamp() -> float:
    """Get the current timestamp."""
    return time.time()

def format_timestamp(timestamp: float) -> str:
    """Format a timestamp as a human-readable string."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

def merge_dictionaries(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries."""
    result = dict1.copy()
    result.update(dict2)
    return result

def chunk_text(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks of specified size."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def safe_get(obj: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Safely get a nested value from a dictionary using dot notation."""
    keys = path.split('.')
    current = obj
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current

# -----------------------------------------------------------------------------
# URL Handling
# -----------------------------------------------------------------------------

def sanitize_url(url: str) -> str:
    """
    Sanitize a URL by ensuring it has a scheme and is properly encoded.
    
    Args:
        url: The URL to sanitize
        
    Returns:
        Sanitized URL
    """
    # Add scheme if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Parse and rebuild to ensure proper encoding
    parsed = urllib.parse.urlparse(url)
    
    # Rebuild with proper encoding
    return urllib.parse.urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            urllib.parse.quote(parsed.path),
            parsed.params,
            parsed.query,
            parsed.fragment
        )
    )


def get_domain(url: str) -> str:
    """
    Extract the domain from a URL.
    
    Args:
        url: The URL to extract domain from
        
    Returns:
        Domain name
    """
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc
    
    # Remove www. prefix if present
    if domain.startswith('www.'):
        domain = domain[4:]
    
    return domain


def url_to_filename(url: str, max_length: int = 100) -> str:
    """
    Convert a URL to a safe filename.
    
    Args:
        url: The URL to convert
        max_length: Maximum length of the filename
        
    Returns:
        Safe filename derived from the URL
    """
    # Extract domain and path
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    
    # Remove www. prefix if present
    if domain.startswith('www.'):
        domain = domain[4:]
    
    # Replace special characters in path
    path = re.sub(r'[^a-zA-Z0-9]', '_', path)
    path = re.sub(r'_+', '_', path)  # Replace multiple underscores with one
    
    # Combine domain and path
    filename = f"{domain}{path}"
    
    # Truncate if too long
    if len(filename) > max_length:
        # Use hash for uniqueness if truncated
        hash_suffix = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = filename[:max_length-9] + '_' + hash_suffix
    
    return filename


# -----------------------------------------------------------------------------
# File Operations
# -----------------------------------------------------------------------------

def safe_read_file(file_path: Union[str, Path], default: str = "", encoding: str = "utf-8") -> str:
    """
    Safely read a file with error handling.
    
    Args:
        file_path: Path to the file
        default: Default value to return if file cannot be read
        encoding: File encoding
        
    Returns:
        File contents or default value
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return default


def safe_write_file(file_path: Union[str, Path], content: str, 
                   encoding: str = "utf-8", create_dirs: bool = True) -> bool:
    """
    Safely write content to a file with error handling.
    
    Args:
        file_path: Path to the file
        content: Content to write
        encoding: File encoding
        create_dirs: Whether to create parent directories if they don't exist
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert to Path object
        path = Path(file_path)
        
        # Create parent directories if they don't exist
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
        
        return True
    except Exception as e:
        logger.error(f"Error writing to file {file_path}: {str(e)}")
        return False


def ensure_directory(directory: Union[str, Path]) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {str(e)}")
        return False


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get the extension of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (without the dot)
    """
    return os.path.splitext(str(file_path))[1].lstrip('.')


def is_binary_file(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is binary.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is binary, False otherwise
    """
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\0' in chunk
    except Exception as e:
        logger.error(f"Error checking if file is binary {file_path}: {str(e)}")
        return False


# -----------------------------------------------------------------------------
# Text Processing
# -----------------------------------------------------------------------------

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length-len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
    
    # Replace multiple underscores with one
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip('. ')
    
    # Ensure filename is not empty
    if not sanitized:
        sanitized = "unnamed_file"
    
    return sanitized


def extract_text_between(text: str, start_marker: str, end_marker: str) -> str:
    """
    Extract text between two markers.
    
    Args:
        text: Text to extract from
        start_marker: Start marker
        end_marker: End marker
        
    Returns:
        Extracted text or empty string if markers not found
    """
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return ""
    
    start_idx += len(start_marker)
    end_idx = text.find(end_marker, start_idx)
    
    if end_idx == -1:
        return ""
    
    return text[start_idx:end_idx]


def clean_html(html: str) -> str:
    """
    Remove HTML tags from text.
    
    Args:
        html: HTML text
        
    Returns:
        Text without HTML tags
    """
    # Simple regex to remove HTML tags
    clean = re.sub(r'<[^>]+>', '', html)
    
    # Replace multiple whitespace with single space
    clean = re.sub(r'\s+', ' ', clean)
    
    # Decode HTML entities
    clean = clean.replace('&nbsp;', ' ')
    clean = clean.replace('&amp;', '&')
    clean = clean.replace('&lt;', '<')
    clean = clean.replace('&gt;', '>')
    clean = clean.replace('&quot;', '"')
    clean = clean.replace('&#39;', "'")
    
    return clean.strip()


# -----------------------------------------------------------------------------
# Data Conversion
# -----------------------------------------------------------------------------

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with error handling.
    
    Args:
        json_str: JSON string to parse
        default: Default value to return if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Error parsing JSON: {str(e)}")
        return default if default is not None else {}


def safe_json_dumps(obj: Any, default: str = "{}", indent: Optional[int] = None) -> str:
    """
    Safely convert object to JSON string with error handling.
    
    Args:
        obj: Object to convert
        default: Default value to return if conversion fails
        indent: Indentation level
        
    Returns:
        JSON string or default value
    """
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error converting to JSON: {str(e)}")
        return default


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested dictionaries
        sep: Separator for keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


# -----------------------------------------------------------------------------
# Date and Time
# -----------------------------------------------------------------------------

def get_iso_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        Current timestamp in ISO format
    """
    return datetime.datetime.now().isoformat()


def format_timestamp(timestamp: Union[str, datetime.datetime], 
                    format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a timestamp.
    
    Args:
        timestamp: Timestamp to format (ISO string or datetime object)
        format_str: Format string
        
    Returns:
        Formatted timestamp
    """
    if isinstance(timestamp, str):
        try:
            dt = datetime.datetime.fromisoformat(timestamp)
        except ValueError:
            # Try parsing with dateutil if available
            try:
                from dateutil import parser
                dt = parser.parse(timestamp)
            except (ImportError, ValueError):
                logger.error(f"Could not parse timestamp: {timestamp}")
                return timestamp
    else:
        dt = timestamp
    
    return dt.strftime(format_str)


# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------

def generate_session_id() -> str:
    """
    Generate a unique session ID.
    
    Returns:
        Unique session ID
    """
    return str(uuid.uuid4())


def chunk_list(lst: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def retry_operation(operation: Callable[[], T], max_retries: int = 3, 
                   delay: float = 1.0, backoff_factor: float = 2.0,
                   exceptions: Tuple[Exception, ...] = (Exception,)) -> T:
    """
    Retry an operation with exponential backoff.
    
    Args:
        operation: Function to retry
        max_retries: Maximum number of retries
        delay: Initial delay between retries (seconds)
        backoff_factor: Factor to increase delay after each retry
        exceptions: Exceptions to catch and retry
        
    Returns:
        Result of the operation
        
    Raises:
        The last exception if all retries fail
    """
    import time
    
    retries = 0
    current_delay = delay
    last_exception = None
    
    while retries <= max_retries:
        try:
            return operation()
        except exceptions as e:
            last_exception = e
            retries += 1
            
            if retries > max_retries:
                break
            
            logger.warning(f"Retry {retries}/{max_retries} after error: {str(e)}")
            time.sleep(current_delay)
            current_delay *= backoff_factor
    
    if last_exception:
        raise last_exception
    
    # This should never happen, but just in case
    raise RuntimeError("Retry operation failed without an exception")


async def async_retry_with_backoff(operation: Callable[[], T], max_retries: int = 3, 
                           initial_delay: float = 1.0, backoff_factor: float = 2.0,
                           exceptions: Tuple[Exception, ...] = (Exception,)) -> T:
    """
    Retry an async operation with exponential backoff.
    
    Args:
        operation: Async function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay between retries (seconds)
        backoff_factor: Factor to increase delay after each retry
        exceptions: Exceptions to catch and retry
        
    Returns:
        Result of the operation
        
    Raises:
        The last exception if all retries fail
    """
    import asyncio
    import logging

    logger = logging.getLogger(__name__)
    
    retries = 0
    current_delay = initial_delay
    last_exception = None
    
    while retries <= max_retries:
        try:
            return await operation()
        except exceptions as e:
            last_exception = e
            retries += 1
            
            if retries > max_retries:
                break
            
            logger.warning(f"Async Retry {retries}/{max_retries} after error: {str(e)}")
            await asyncio.sleep(current_delay)
            current_delay *= backoff_factor
    
    if last_exception:
        raise last_exception
    
    # This should never happen, but just in case
    raise RuntimeError("Async Retry operation failed without an exception")


def is_valid_uuid(val: str) -> bool:
    """
    Check if a string is a valid UUID.
    
    Args:
        val: String to check
        
    Returns:
        True if string is a valid UUID, False otherwise
    """
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False


# Example usage
if __name__ == "__main__":
    # URL handling
    print(sanitize_url("example.com/path with spaces"))
    print(get_domain("https://www.example.com/path"))
    print(url_to_filename("https://www.example.com/path/to/page.html"))
    
    # File operations
    print(ensure_directory("./test_dir"))
    print(safe_write_file("./test_dir/test.txt", "Hello, world!"))
    print(safe_read_file("./test_dir/test.txt"))
    
    # Text processing
    print(truncate_text("This is a long text that needs to be truncated", 20))
    print(sanitize_filename("file:with?invalid*chars.txt"))
    
    # Data conversion
    print(safe_json_dumps({"key": "value", "nested": {"key": "value"}}))
    print(flatten_dict({"a": 1, "b": {"c": 2, "d": {"e": 3}}}))
    
    # Date and time
    print(get_iso_timestamp())
    print(format_timestamp(get_iso_timestamp()))
    
    # Miscellaneous
    print(generate_session_id())
    print(chunk_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3))
    
    # Retry example
    def flaky_operation():
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise ConnectionError("Random failure")
        return "Success!"
    
    try:
        result = retry_operation(flaky_operation, max_retries=5, exceptions=(ConnectionError,))
        print(f"Retry result: {result}")
    except Exception as e:
        print(f"Retry failed: {e}")


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """
    Extract important keywords from text.
    
    This is a simple implementation that uses word frequency.
    For production use, consider using NLP libraries like NLTK or spaCy.
    
    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of keywords
    """
    # Convert to lowercase and split into words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Common English stop words to filter out
    stop_words = {
        'the', 'and', 'is', 'in', 'it', 'to', 'of', 'for', 'with', 'on', 'that', 'this',
        'are', 'as', 'be', 'by', 'from', 'has', 'have', 'not', 'was', 'were', 'will',
        'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall', 'such',
        'when', 'where', 'which', 'who', 'why', 'how', 'what', 'than', 'then', 'there',
        'these', 'those', 'they', 'their', 'them', 'some', 'all', 'any', 'but', 'if',
        'or', 'because', 'also', 'other', 'use', 'used', 'using', 'very'
    }
    
    # Filter out stop words
    filtered_words = [word for word in words if word not in stop_words]
    
    # Count word frequency
    word_counts = {}
    for word in filtered_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency (descending)
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Return top keywords
    return [word for word, count in sorted_words[:max_keywords]]


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """
    Create a simple extractive summary of text.
    
    This is a basic implementation. For production use,
    consider using NLP libraries or LLMs for better summarization.
    
    Args:
        text: Text to summarize
        max_sentences: Maximum number of sentences in summary
        
    Returns:
        Summarized text
    """
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # If text is already short, return it
    if len(sentences) <= max_sentences:
        return text
    
    # Simple scoring: prefer sentences with keywords and early position
    keywords = extract_keywords(text, max_keywords=10)
    
    # Score each sentence
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        # Position score (earlier sentences get higher scores)
        position_score = 1.0 - (i / len(sentences))
        
        # Keyword score (sentences with more keywords get higher scores)
        keyword_count = sum(1 for keyword in keywords if keyword.lower() in sentence.lower())
        keyword_score = keyword_count / len(keywords) if keywords else 0
        
        # Length score (prefer medium-length sentences)
        words = len(sentence.split())
        length_score = 1.0 - abs(words - 20) / 20 if words <= 40 else 0
        
        # Combined score
        total_score = position_score * 0.3 + keyword_score * 0.5 + length_score * 0.2
        sentence_scores.append((sentence, total_score))
    
    # Sort by score (descending)
    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    
    # Get top sentences and sort them by original position
    top_sentences = sorted_sentences[:max_sentences]
    original_order = [(i, sentence) for i, (sentence, _) in enumerate(sentences) 
                     if sentence in [s for s, _ in top_sentences]]
    original_order.sort(key=lambda x: x[0])
    
    # Join sentences in original order
    summary = ' '.join(sentence for _, sentence in original_order)
    
    return summary


def download_file(url: str, local_path: Union[str, Path], 
                 timeout: int = 30, chunk_size: int = 8192) -> bool:
    """
    Download a file from a URL.
    
    Args:
        url: URL to download from
        local_path: Local path to save the file
        timeout: Timeout in seconds
        chunk_size: Size of chunks to download
        
    Returns:
        True if download was successful, False otherwise
    """
    try:
        import requests
        
        # Create directory if it doesn't exist
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download file
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        
        return True
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {str(e)}")
        return False


def is_url_accessible(url: str, timeout: int = 5) -> bool:
    """
    Check if a URL is accessible.
    
    Args:
        url: URL to check
        timeout: Timeout in seconds
        
    Returns:
        True if URL is accessible, False otherwise
    """
    try:
        import requests
        
        response = requests.head(url, timeout=timeout)
        return response.status_code < 400
    except Exception:
        return False


def is_valid_url(url: str) -> bool:
    """
    Check if a string is a valid URL.

    Args:
        url: The URL to check.

    Returns:
        True if the URL is valid, False otherwise.
    """
    import re
    # Basic regex for URL validation
    url_regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https:// or ftp://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|)' # domain...
        r'(?:/?|[/?]\S+)$'
        , re.IGNORECASE)
    return re.match(url_regex, url) is not None


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any], 
               overwrite: bool = True) -> Dict[str, Any]:
    """
    Merge two dictionaries recursively.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (values from this dict will be merged into dict1)
        overwrite: Whether to overwrite values in dict1 with values from dict2
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_dicts(result[key], value, overwrite)
        elif key not in result or overwrite:
            # Add or overwrite value
            result[key] = value
    
    return result


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted file size (e.g., "1.23 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    
    for unit in ['KB', 'MB', 'GB', 'TB']:
        size_bytes /= 1024
        if size_bytes < 1024:
            break
    
    return f"{size_bytes:.2f} {unit}"


def get_mime_type(file_path: Union[str, Path]) -> str:
    """
    Get MIME type of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MIME type or 'application/octet-stream' if unknown
    """
    import mimetypes
    
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or 'application/octet-stream'


def create_directory_structure(base_dir: Union[str, Path], structure: Dict[str, Any]) -> bool:
    """
    Create a directory structure from a dictionary.
    
    Args:
        base_dir: Base directory
        structure: Directory structure as a nested dictionary
                  Keys are directory/file names, values are either:
                  - None (for empty directories)
                  - Dict (for nested directories)
                  - String (for file content)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        base_path = Path(base_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        
        for name, content in structure.items():
            path = base_path / name
            
            if content is None:
                # Create empty directory
                path.mkdir(exist_ok=True)
            elif isinstance(content, dict):
                # Create nested directory structure
                create_directory_structure(path, content)
            elif isinstance(content, str):
                # Create file with content
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                logger.warning(f"Unsupported content type for {path}: {type(content)}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating directory structure: {str(e)}")
        return False


def find_files(directory: Union[str, Path], pattern: str, 
              recursive: bool = True) -> List[Path]:
    """
    Find files matching a pattern in a directory.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern to match
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    path = Path(directory)
    
    if recursive:
        return list(path.glob(f"**/{pattern}"))
    else:
        return list(path.glob(pattern))


def safe_delete(path: Union[str, Path], recursive: bool = False) -> bool:
    """
    Safely delete a file or directory.
    
    Args:
        path: Path to delete
        recursive: Whether to delete directories recursively
        
    Returns:
        True if deletion was successful, False otherwise
    """
    try:
        path = Path(path)
        
        if not path.exists():
            return True
        
        if path.is_file():
            path.unlink()
            return True
        
        if path.is_dir():
            if recursive:
                import shutil
                shutil.rmtree(path)
                return True
            else:
                # Only delete if directory is empty
                if not any(path.iterdir()):
                    path.rmdir()
                    return True
                else:
                    logger.error(f"Directory not empty and recursive=False: {path}")
                    return False
        
        return False
    except Exception as e:
        logger.error(f"Error deleting {path}: {str(e)}")
        return False


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        File hash as hexadecimal string
    """
    try:
        hash_func = getattr(hashlib, algorithm)()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {str(e)}")
        return ""


def is_valid_email(email: str) -> bool:
    """
    Check if a string is a valid email address.
    
    Args:
        email: Email address to check
        
    Returns:
        True if email is valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def slugify(text: str) -> str:
    """
    Convert text to a URL-friendly slug.
    
    Args:
        text: Text to convert
        
    Returns:
        URL-friendly slug
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace spaces with hyphens
    text = re.sub(r'\s+', '-', text)
    
    # Remove special characters
    text = re.sub(r'[^a-z0-9\-]', '', text)
    
    # Remove duplicate hyphens
    text = re.sub(r'-+', '-', text)
    
    # Remove leading/trailing hyphens
    text = text.strip('-')
    
    return text
