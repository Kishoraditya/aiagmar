"""
File Manager Agent

This agent organizes and stores research materials, such as summaries and images, in a structured file system.
It relies on Filesystem MCP for secure file operations, ensuring organized storage.
"""

import os
import uuid
import json
import time
import re
import base64
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Literal, TypedDict, cast
from urllib.parse import urlparse
import requests

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

# Import MCP clients
from apps.mcps.filesystem_mcp import FilesystemMCP
from apps.mcps.memory_mcp import MemoryMCP

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("file_manager_agent")


# -----------------------------------------------------------------------------
# Type Definitions
# -----------------------------------------------------------------------------

class FileInfo(TypedDict):
    """Structure for file information."""
    path: str
    name: str
    type: str  # "file" or "directory"
    size: Optional[int]
    created: Optional[str]
    modified: Optional[str]
    content_type: Optional[str]
    metadata: Optional[Dict[str, Any]]


class DirectoryInfo(TypedDict):
    """Structure for directory information."""
    path: str
    name: str
    files: List[FileInfo]
    subdirectories: List[str]
    total_files: int
    total_size: int


class SaveFileRequest(TypedDict):
    """Structure for save file requests."""
    content: str
    path: str
    metadata: Optional[Dict[str, Any]]
    overwrite: Optional[bool]
    namespace: Optional[str]
    session_id: Optional[str]


class SaveUrlContentRequest(TypedDict):
    """Structure for save URL content requests."""
    url: str
    path: str
    metadata: Optional[Dict[str, Any]]
    overwrite: Optional[bool]
    namespace: Optional[str]
    session_id: Optional[str]


class OrganizeFilesRequest(TypedDict):
    """Structure for organize files requests."""
    source_directory: str
    target_directory: Optional[str]
    organization_type: Optional[str]  # "date", "type", "custom"
    custom_rules: Optional[Dict[str, Any]]
    namespace: Optional[str]
    session_id: Optional[str]


class SearchFilesRequest(TypedDict):
    """Structure for search files requests."""
    query: str
    directory: Optional[str]
    file_types: Optional[List[str]]
    recursive: Optional[bool]
    namespace: Optional[str]
    session_id: Optional[str]


class FileOperationResponse(TypedDict):
    """Structure for file operation responses."""
    success: bool
    file_info: Optional[FileInfo]
    message: str
    execution_stats: Dict[str, Any]
    errors: List[Dict[str, Any]]


class DirectoryOperationResponse(TypedDict):
    """Structure for directory operation responses."""
    success: bool
    directory_info: Optional[DirectoryInfo]
    message: str
    execution_stats: Dict[str, Any]
    errors: List[Dict[str, Any]]


class SearchFilesResponse(TypedDict):
    """Structure for search files responses."""
    success: bool
    results: List[FileInfo]
    query: str
    message: str
    execution_stats: Dict[str, Any]
    errors: List[Dict[str, Any]]


class OrganizeFilesResponse(TypedDict):
    """Structure for organize files responses."""
    success: bool
    organized_files: List[Dict[str, str]]  # old_path -> new_path
    message: str
    execution_stats: Dict[str, Any]
    errors: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# MCP Client Management
# -----------------------------------------------------------------------------

class MCPClientManager:
    """Manages connections to MCP services used by the File Manager Agent."""
    
    def __init__(self):
        """Initialize MCP client manager."""
        self.filesystem_mcp = None
        self.memory_mcp = None
    
    def get_filesystem_mcp(self) -> FilesystemMCP:
        """Get or create Filesystem MCP client."""
        if self.filesystem_mcp is None:
            workspace_dir = os.environ.get("WORKSPACE_DIR", "./workspace")
            self.filesystem_mcp = FilesystemMCP(workspace_dir=workspace_dir)
        return self.filesystem_mcp
    
    def get_memory_mcp(self) -> MemoryMCP:
        """Get or create Memory MCP client."""
        if self.memory_mcp is None:
            storage_path = os.environ.get("MEMORY_STORAGE_PATH", "./memory_storage")
            self.memory_mcp = MemoryMCP(storage_path=storage_path)
        return self.memory_mcp
    
    def close_all(self):
        """Close all MCP clients."""
        if self.filesystem_mcp:
            self.filesystem_mcp.close()
        if self.memory_mcp:
            self.memory_mcp.close()


# Create a singleton instance
mcp_manager = MCPClientManager()


# -----------------------------------------------------------------------------
# File Manager Agent Core Functions
# -----------------------------------------------------------------------------

def create_llm(model: str = "gpt-4o", temperature: float = 0.2):
    """Create a language model instance."""
    return ChatOpenAI(model=model, temperature=temperature)


def get_file_extension(path: str) -> str:
    """Get the file extension from a path."""
    _, ext = os.path.splitext(path)
    return ext.lower()


def get_content_type(path: str) -> str:
    """Determine content type based on file extension."""
    ext = get_file_extension(path)
    
    # Map extensions to content types
    content_types = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
        ".json": "application/json",
        ".xml": "application/xml",
        ".csv": "text/csv",
        ".pdf": "application/pdf",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xls": "application/vnd.ms-excel",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".ppt": "application/vnd.ms-powerpoint",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".mp3": "audio/mpeg",
        ".mp4": "video/mp4",
        ".wav": "audio/wav",
        ".zip": "application/zip",
        ".tar": "application/x-tar",
        ".gz": "application/gzip",
        ".py": "text/x-python",
        ".js": "text/javascript",
        ".css": "text/css",
    }
    
    return content_types.get(ext, "application/octet-stream")


def download_from_url(url: str) -> Tuple[bytes, str]:
    """
    Download content from a URL.
    
    Args:
        url: URL to download from
        
    Returns:
        Tuple of (content_bytes, content_type)
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Get content type from response headers
        content_type = response.headers.get("Content-Type", "application/octet-stream")
        if ";" in content_type:
            content_type = content_type.split(";")[0].strip()
        
        return response.content, content_type
    
    except Exception as e:
        logger.error(f"Error downloading from URL {url}: {e}")
        raise


def suggest_file_organization(files: List[FileInfo], organization_type: str = "type") -> Dict[str, List[str]]:
    """
    Suggest file organization based on file information.
    
    Args:
        files: List of file information
        organization_type: Type of organization ("date", "type", "custom")
        
    Returns:
        Dictionary mapping category names to lists of file paths
    """
    organization: Dict[str, List[str]] = {}
    
    if organization_type == "date":
        # Organize by date (year/month)
        for file in files:
            if file["type"] == "file" and file.get("modified"):
                try:
                    date = datetime.fromisoformat(file["modified"].replace("Z", "+00:00"))
                    category = f"{date.year}/{date.month:02d}"
                    
                    if category not in organization:
                        organization[category] = []
                    
                    organization[category].append(file["path"])
                except (ValueError, TypeError):
                    # If date parsing fails, put in "unknown" category
                    if "unknown_date" not in organization:
                        organization["unknown_date"] = []
                    organization["unknown_date"].append(file["path"])
    
    elif organization_type == "type":
        # Organize by file type
        for file in files:
            if file["type"] == "file":
                ext = get_file_extension(file["path"])
                category = ext[1:] if ext else "unknown"
                
                # Group similar types
                if category in ["jpg", "jpeg", "png", "gif", "svg"]:
                    category = "images"
                elif category in ["doc", "docx", "txt", "md", "pdf"]:
                    category = "documents"
                elif category in ["mp3", "wav", "ogg"]:
                    category = "audio"
                elif category in ["mp4", "avi", "mov"]:
                    category = "videos"
                elif category in ["zip", "tar", "gz", "rar"]:
                    category = "archives"
                elif category in ["py", "js", "html", "css", "java", "cpp"]:
                    category = "code"
                
                if category not in organization:
                    organization[category] = []
                
                organization[category].append(file["path"])
    
    else:  # custom or fallback
        # Simple organization by extension
        for file in files:
            if file["type"] == "file":
                ext = get_file_extension(file["path"])
                category = ext[1:] if ext else "unknown"
                
                if category not in organization:
                    organization[category] = []
                
                organization[category].append(file["path"])
    
    return organization


def generate_file_metadata(file_path: str, content: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate metadata for a file using LLM.
    
    Args:
        file_path: Path to the file
        content: Optional file content
        
    Returns:
        Dictionary of metadata
    """
    # If no content provided, try to read it
    if content is None:
        try:
            fs_mcp = mcp_manager.get_filesystem_mcp()
            content = fs_mcp.read_file(file_path)
        except Exception as e:
            logger.warning(f"Could not read file content for metadata generation: {e}")
            content = ""
    
    # Basic metadata
    metadata = {
        "filename": os.path.basename(file_path),
        "extension": get_file_extension(file_path),
        "content_type": get_content_type(file_path),
        "generated_at": datetime.now().isoformat(),
    }
    
    # For text-based files, generate additional metadata using LLM
    if content and metadata["content_type"].startswith(("text/", "application/json", "application/xml")):
        try:
            # Truncate content if too long
            if len(content) > 10000:
                content = content[:10000] + "... [content truncated]"
            
            llm = create_llm(temperature=0.1)
            
            # Create prompt for metadata generation
            prompt_template = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are an expert at analyzing files and generating useful metadata. 
                Examine the file content and generate a JSON object with the following fields:
                - summary: A brief summary of the file content (1-2 sentences)
                - topics: An array of 3-5 main topics or themes in the content
                - keywords: An array of 5-10 relevant keywords
                - language: The detected language of the content
                - content_type: More specific content type if detectable
                
                Return only the JSON object without any explanation or formatting."""),
                HumanMessage(content="File path: {file_path}\n\nFile content:\n{content}")
            ])
            
            # Execute the prompt
            chain = prompt_template | llm | JsonOutputParser()
            llm_metadata = chain.invoke({
                "file_path": file_path,
                "content": content
            })
            
            # Merge LLM-generated metadata
            metadata.update(llm_metadata)
        
        except Exception as e:
            logger.warning(f"Error generating enhanced metadata: {e}")
    
    return metadata


def store_file_metadata(file_info: FileInfo, namespace: str, session_id: Optional[str] = None) -> str:
    """
    Store file metadata in memory.
    
    Args:
        file_info: File information to store
        namespace: Namespace for storage
        session_id: Optional session ID
        
    Returns:
        Memory key where metadata is stored
    """
    memory_mcp = mcp_manager.get_memory_mcp()
    
    # Generate a key
    file_id = str(uuid.uuid4())
    key = f"file_{session_id or ''}_{file_id}"
    
    # Store the metadata
    memory_mcp.store_memory(
        key=key,
        value=json.dumps(file_info),
        namespace=namespace
    )
    
    return key


# -----------------------------------------------------------------------------
# File Manager Agent Class
# -----------------------------------------------------------------------------

class FileManagerAgent:
    """
    File Manager Agent that organizes and stores research materials in a structured file system.
    """
    
    def __init__(self):
        """Initialize the File Manager Agent."""
        pass
    
    def save_file(self, request: SaveFileRequest) -> FileOperationResponse:
        """
        Save content to a file.
        
        Args:
            request: Save file request
            
        Returns:
            File operation response
        """
        start_time = time.time()
        
        # Initialize response
        response: FileOperationResponse = {
            "success": False,
            "file_info": None,
            "message": "",
            "execution_stats": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 0
            },
            "errors": []
        }
        
        try:
            # Get Filesystem MCP client
            fs_mcp = mcp_manager.get_filesystem_mcp()
            
            # Check if file exists and handle overwrite
            file_exists = False
            try:
                file_exists = fs_mcp.file_exists(request["path"])
            except Exception as e:
                logger.debug(f"Error checking if file exists: {e}")
            
            if file_exists and not request.get("overwrite", False):
                raise ValueError(f"File already exists at {request['path']} and overwrite is not enabled")
            
            # Create parent directory if it doesn't exist
            parent_dir = os.path.dirname(request["path"])
            if parent_dir:
                try:
                    fs_mcp.create_directory(parent_dir)
                except Exception as e:
                    # Directory might already exist
                    logger.debug(f"Note when creating directory: {e}")
            
            # Write the file
            logger.info(f"Writing file to {request['path']}")
            fs_mcp.write_file(request["path"], request["content"])
            
            # Generate metadata if not provided
            metadata = request.get("metadata", {})
            if not metadata:
                metadata = generate_file_metadata(request["path"], request["content"])
            
            # Create file info
            file_info: FileInfo = {
                "path": request["path"],
                "name": os.path.basename(request["path"]),
                "type": "file",
                "size": len(request["content"].encode('utf-8')),
                "created": datetime.now().isoformat(),
                "modified": datetime.now().isoformat(),
                "content_type": get_content_type(request["path"]),
                "metadata": metadata
            }
            
            # Store metadata in memory if namespace provided
            if request.get("namespace"):
                try:
                    store_file_metadata(
                        file_info=file_info,
                        namespace=request["namespace"],
                        session_id=request.get("session_id")
                    )
                except Exception as e:
                    logger.error(f"Error storing file metadata: {e}")
                    response["errors"].append({
                        "type": "metadata_storage_error",
                        "error": str(e)
                    })
            
            # Update response
            response["success"] = True
            response["file_info"] = file_info
            response["message"] = f"File successfully saved to {request['path']}"
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            
            return response
        
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            response["success"] = False
            response["message"] = f"Error saving file: {str(e)}"
            response["errors"].append({
                "type": "file_save_error",
                "error": str(e)
            })
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            return response
    
    def save_url_content(self, request: SaveUrlContentRequest) -> FileOperationResponse:
        """
        Download content from a URL and save it to a file.
        
        Args:
            request: Save URL content request
            
        Returns:
            File operation response
        """
        start_time = time.time()
        
        # Initialize response
        response: FileOperationResponse = {
            "success": False,
            "file_info": None,
            "message": "",
            "execution_stats": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 0
            },
            "errors": []
        }
        
        try:
            # Get Filesystem MCP client
            fs_mcp = mcp_manager.get_filesystem_mcp()
            
            # Check if file exists and handle overwrite
            file_exists = False
            try:
                file_exists = fs_mcp.file_exists(request["path"])
            except Exception as e:
                logger.debug(f"Error checking if file exists: {e}")
            
            if file_exists and not request.get("overwrite", False):
                raise ValueError(f"File already exists at {request['path']} and overwrite is not enabled")
            
            # Create parent directory if it doesn't exist
            parent_dir = os.path.dirname(request["path"])
            if parent_dir:
                try:
                    fs_mcp.create_directory(parent_dir)
                except Exception as e:
                    # Directory might already exist
                    logger.debug(f"Note when creating directory: {e}")
            
            # Download content from URL
            logger.info(f"Downloading content from {request['url']}")
            content_bytes, content_type = download_from_url(request["url"])
            
            # Determine how to save the content based on content type
            if content_type.startswith(("text/", "application/json", "application/xml", "application/javascript")):
                # Text-based content
                try:
                    content_str = content_bytes.decode('utf-8')
                    fs_mcp.write_file(request["path"], content_str)
                except UnicodeDecodeError:
                    # If decoding fails, save as base64
                    content_str = base64.b64encode(content_bytes).decode('utf-8')
                    fs_mcp.write_file(request["path"], content_str)
            else:
                # Binary content (saved as base64)
                content_str = base64.b64encode(content_bytes).decode('utf-8')
                fs_mcp.write_file(request["path"], content_str)
            
            # Generate metadata if not provided
            metadata = request.get("metadata", {})
            if not metadata:
                metadata = {
                    "source_url": request["url"],
                    "content_type": content_type,
                    "download_time": datetime.now().isoformat()
                }
                
                # For text content, try to generate enhanced metadata
                if content_type.startswith(("text/", "application/json", "application/xml")):
                    try:
                        enhanced_metadata = generate_file_metadata(
                            file_path=request["path"],
                            content=content_bytes.decode('utf-8', errors='ignore')
                        )
                        metadata.update(enhanced_metadata)
                    except Exception as e:
                        logger.warning(f"Error generating enhanced metadata: {e}")
            
            # Create file info
            file_info: FileInfo = {
                "path": request["path"],
                "name": os.path.basename(request["path"]),
                "type": "file",
                "size": len(content_bytes),
                "created": datetime.now().isoformat(),
                "modified": datetime.now().isoformat(),
                "content_type": content_type,
                "metadata": metadata
            }
            
            # Store metadata in memory if namespace provided
            if request.get("namespace"):
                try:
                    store_file_metadata(
                        file_info=file_info,
                        namespace=request["namespace"],
                        session_id=request.get("session_id")
                    )
                except Exception as e:
                    logger.error(f"Error storing file metadata: {e}")
                    response["errors"].append({
                        "type": "metadata_storage_error",
                        "error": str(e)
                    })
            
            # Update response
            response["success"] = True
            response["file_info"] = file_info
            response["message"] = f"Content from {request['url']} successfully saved to {request['path']}"
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            
            return response
        
        except Exception as e:
            logger.error(f"Error saving URL content: {e}")
            response["success"] = False
            response["message"] = f"Error saving URL content: {str(e)}"
            response["errors"].append({
                "type": "url_content_save_error",
                "error": str(e)
            })
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            return response
    
    def read_file(self, path: str) -> FileOperationResponse:
        """
        Read content from a file.
        
        Args:
            path: Path to the file
            
        Returns:
            File operation response with content in the message field
        """
        start_time = time.time()
        
        # Initialize response
        response: FileOperationResponse = {
            "success": False,
            "file_info": None,
            "message": "",
            "execution_stats": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 0
            },
            "errors": []
        }
        
        try:
            # Get Filesystem MCP client
            fs_mcp = mcp_manager.get_filesystem_mcp()
            
            # Check if file exists
            if not fs_mcp.file_exists(path):
                raise FileNotFoundError(f"File not found at {path}")
            
            # Read the file
            logger.info(f"Reading file from {path}")
            content = fs_mcp.read_file(path)
            
            # Create file info
            file_info: FileInfo = {
                "path": path,
                "name": os.path.basename(path),
                "type": "file",
                "size": len(content.encode('utf-8')),
                "created": None,  # Not available from MCP
                "modified": None,  # Not available from MCP
                "content_type": get_content_type(path),
                "metadata": None
            }
            
            # Update response
            response["success"] = True
            response["file_info"] = file_info
            response["message"] = content
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            
            return response
        
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            response["success"] = False
            response["message"] = f"Error reading file: {str(e)}"
            response["errors"].append({
                "type": "file_read_error",
                "error": str(e)
            })
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            return response
    
    def list_directory(self, directory: str, recursive: bool = False) -> DirectoryOperationResponse:
        """
        List files and directories in a directory.
        
        Args:
            directory: Path to the directory
            recursive: Whether to list files recursively
            
        Returns:
            Directory operation response
        """
        start_time = time.time()
        
        # Initialize response
        response: DirectoryOperationResponse = {
            "success": False,
            "directory_info": None,
            "message": "",
            "execution_stats": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 0
            },
            "errors": []
        }
        
        try:
            # Get Filesystem MCP client
            fs_mcp = mcp_manager.get_filesystem_mcp()
            
            # List the directory
            logger.info(f"Listing directory {directory} (recursive={recursive})")
            listing = fs_mcp.list_directory(directory, recursive=recursive)
            
            # Parse the listing
            files: List[FileInfo] = []
            subdirectories: List[str] = []
            total_size = 0
            
            # Process the listing text
            lines = listing.strip().split('\n')
            for line in lines:
                if not line.strip():
                    continue
                
                # Try to determine if it's a file or directory
                is_directory = False
                if line.endswith('/') or '[DIR]' in line:
                    is_directory = True
                
                # Extract the path
                path = line.strip()
                if '[DIR]' in path:
                    path = path.split('[DIR]')[1].strip()
                
                # Normalize path
                path = path.rstrip('/')
                
                if is_directory:
                    subdirectories.append(path)
                else:
                    # Create file info
                    file_info: FileInfo = {
                        "path": path,
                        "name": os.path.basename(path),
                        "type": "file",
                        "size": None,  # Not available from listing
                        "created": None,  # Not available from listing
                        "modified": None,  # Not available from listing
                        "content_type": get_content_type(path),
                        "metadata": None
                    }
                    files.append(file_info)
            
            # Create directory info
            directory_info: DirectoryInfo = {
                "path": directory,
                "name": os.path.basename(directory) or directory,
                "files": files,
                "subdirectories": subdirectories,
                "total_files": len(files),
                "total_size": total_size
            }
            
            # Update response
            response["success"] = True
            response["directory_info"] = directory_info
            response["message"] = f"Successfully listed directory {directory} with {len(files)} files and {len(subdirectories)} subdirectories"
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            
            return response
        
        except Exception as e:
            logger.error(f"Error listing directory: {e}")
            response["success"] = False
            response["message"] = f"Error listing directory: {str(e)}"
            response["errors"].append({
                "type": "directory_list_error",
                "error": str(e)
            })
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            return response
    
    def create_directory(self, path: str) -> DirectoryOperationResponse:
        """
        Create a directory.
        
        Args:
            path: Path to the directory
            
        Returns:
            Directory operation response
        """
        start_time = time.time()
        
        # Initialize response
        response: DirectoryOperationResponse = {
            "success": False,
            "directory_info": None,
            "message": "",
            "execution_stats": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 0
            },
            "errors": []
        }
        
        try:
            # Get Filesystem MCP client
            fs_mcp = mcp_manager.get_filesystem_mcp()
            
            # Create the directory
            logger.info(f"Creating directory {path}")
            fs_mcp.create_directory(path)
            
            # Create directory info
            directory_info: DirectoryInfo = {
                "path": path,
                "name": os.path.basename(path) or path,
                "files": [],
                "subdirectories": [],
                "total_files": 0,
                "total_size": 0
            }
            
            # Update response
            response["success"] = True
            response["directory_info"] = directory_info
            response["message"] = f"Successfully created directory {path}"
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            
            return response
        
        except Exception as e:
            logger.error(f"Error creating directory: {e}")
            response["success"] = False
            response["message"] = f"Error creating directory: {str(e)}"
            response["errors"].append({
                "type": "directory_create_error",
                "error": str(e)
            })
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            return response
    
    def delete_file(self, path: str) -> FileOperationResponse:
        """
        Delete a file.
        
        Args:
            path: Path to the file
            
        Returns:
            File operation response
        """
        start_time = time.time()
        
        # Initialize response
        response: FileOperationResponse = {
            "success": False,
            "file_info": None,
            "message": "",
            "execution_stats": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 0
            },
            "errors": []
        }
        
        try:
            # Get Filesystem MCP client
            fs_mcp = mcp_manager.get_filesystem_mcp()
            
            # Check if file exists
            if not fs_mcp.file_exists(path):
                raise FileNotFoundError(f"File not found at {path}")
            
            # Create file info before deletion
            file_info: FileInfo = {
                "path": path,
                "name": os.path.basename(path),
                "type": "file",
                "size": None,
                "created": None,
                "modified": None,
                "content_type": get_content_type(path),
                "metadata": None
            }
            
            # Delete the file
            logger.info(f"Deleting file {path}")
            fs_mcp.delete_file(path)
            
            # Update response
            response["success"] = True
            response["file_info"] = file_info
            response["message"] = f"Successfully deleted file {path}"
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            
            return response
        
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            response["success"] = False
            response["message"] = f"Error deleting file: {str(e)}"
            response["errors"].append({
                "type": "file_delete_error",
                "error": str(e)
            })
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            return response
    
    def search_files(self, request: SearchFilesRequest) -> SearchFilesResponse:
        """
        Search for files matching a pattern.
        
        Args:
            request: Search files request
            
        Returns:
            Search files response
        """
        start_time = time.time()
        
        # Initialize response
        response: SearchFilesResponse = {
            "success": False,
            "results": [],
            "query": request["query"],
            "message": "",
            "execution_stats": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 0
            },
            "errors": []
        }
        
        try:
            # Get Filesystem MCP client
            fs_mcp = mcp_manager.get_filesystem_mcp()
            
            # Set default values
            directory = request.get("directory", ".")
            recursive = request.get("recursive", True)
            
            # Search for files
            logger.info(f"Searching for files matching '{request['query']}' in {directory}")
            search_results = fs_mcp.search_files(
                pattern=request["query"],
                path=directory,
                recursive=recursive
            )
            
            # Parse the search results
            results: List[FileInfo] = []
            
            # Process the search results text
            lines = search_results.strip().split('\n')
            for line in lines:
                if not line.strip():
                    continue
                
                # Extract the path
                path = line.strip()
                
                # Filter by file types if specified
                if request.get("file_types"):
                    ext = get_file_extension(path)
                    if ext[1:] not in request["file_types"]:
                        continue
                
                # Create file info
                file_info: FileInfo = {
                    "path": path,
                    "name": os.path.basename(path),
                    "type": "file",
                    "size": None,  # Not available from search results
                    "created": None,  # Not available from search results
                    "modified": None,  # Not available from search results
                    "content_type": get_content_type(path),
                    "metadata": None
                }
                results.append(file_info)
            
            # Update response
            response["success"] = True
            response["results"] = results
            response["message"] = f"Found {len(results)} files matching '{request['query']}'"
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            
            return response
        
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            response["success"] = False
            response["message"] = f"Error searching files: {str(e)}"
            response["errors"].append({
                "type": "file_search_error",
                "error": str(e)
            })
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            return response
    
    def organize_files(self, request: OrganizeFilesRequest) -> OrganizeFilesResponse:
        """
        Organize files in a directory according to specified rules.
        
        Args:
            request: Organize files request
            
        Returns:
            Organize files response
        """
        start_time = time.time()
        
        # Initialize response
        response: OrganizeFilesResponse = {
            "success": False,
            "organized_files": [],
            "message": "",
            "execution_stats": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 0
            },
            "errors": []
        }
        
        try:
            # Get Filesystem MCP client
            fs_mcp = mcp_manager.get_filesystem_mcp()
            
            # Set default values
            source_directory = request["source_directory"]
            target_directory = request.get("target_directory", source_directory)
            organization_type = request.get("organization_type", "type")
            
            # List files in the source directory
            logger.info(f"Listing files in {source_directory} for organization")
            dir_response = self.list_directory(source_directory, recursive=False)
            
            if not dir_response["success"]:
                raise ValueError(f"Failed to list directory: {dir_response['message']}")
            
            if not dir_response["directory_info"]:
                raise ValueError("Directory info not available")
            
            files = dir_response["directory_info"]["files"]
            
            # Determine organization structure
            if request.get("custom_rules"):
                # Use custom rules
                organization = request["custom_rules"]
            else:
                # Use built-in organization logic
                organization = suggest_file_organization(files, organization_type)
            
            # Create target directories
            organized_files = []
            for category, file_paths in organization.items():
                category_dir = os.path.join(target_directory, category)
                
                # Create category directory
                try:
                    fs_mcp.create_directory(category_dir)
                except Exception as e:
                    logger.warning(f"Note when creating directory {category_dir}: {e}")
                
                # Move files to category directory
                for file_path in file_paths:
                    try:
                        # Determine new path
                        file_name = os.path.basename(file_path)
                        new_path = os.path.join(category_dir, file_name)
                        
                        # Read file content
                        content = fs_mcp.read_file(file_path)
                        
                        # Write to new location
                        fs_mcp.write_file(new_path, content)
                        
                        # Delete original file
                        fs_mcp.delete_file(file_path)
                        
                        # Record the move
                        organized_files.append({
                            "old_path": file_path,
                            "new_path": new_path
                        })
                        
                        logger.info(f"Moved {file_path} to {new_path}")
                    
                    except Exception as e:
                        logger.error(f"Error moving file {file_path}: {e}")
                        response["errors"].append({
                            "type": "file_move_error",
                            "error": f"Error moving {file_path}: {str(e)}"
                        })
            
            # Update response
            response["success"] = True
            response["organized_files"] = organized_files
            response["message"] = f"Successfully organized {len(organized_files)} files into {len(organization)} categories"
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            
            return response
        
        except Exception as e:
            logger.error(f"Error organizing files: {e}")
            response["success"] = False
            response["message"] = f"Error organizing files: {str(e)}"
            response["errors"].append({
                "type": "file_organization_error",
                "error": str(e)
            })
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            return response
    
    def create_research_directory(self, topic: str, session_id: Optional[str] = None) -> DirectoryOperationResponse:
        """
        Create a structured directory for research on a specific topic.
        
        Args:
            topic: Research topic
            session_id: Optional session ID
            
        Returns:
            Directory operation response
        """
        start_time = time.time()
        
        # Initialize response
        response: DirectoryOperationResponse = {
            "success": False,
            "directory_info": None,
            "message": "",
            "execution_stats": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 0
            },
            "errors": []
        }
        
        try:
            # Get Filesystem MCP client
            fs_mcp = mcp_manager.get_filesystem_mcp()
            
            # Create a sanitized topic name for the directory
            sanitized_topic = re.sub(r'[^\w\s-]', '', topic).strip().lower()
            sanitized_topic = re.sub(r'[-\s]+', '_', sanitized_topic)
            
            # Create a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create the main research directory
            research_dir = f"research_{sanitized_topic}_{timestamp}"
            
            # Create subdirectories
            subdirectories = [
                "sources",        # For storing source materials
                "summaries",      # For storing summaries
                "images",         # For storing images
                "notes",          # For storing notes
                "final_output"    # For storing final research output
            ]
            
            # Create the main directory
            logger.info(f"Creating research directory structure for topic: {topic}")
            fs_mcp.create_directory(research_dir)
            
            # Create subdirectories
            for subdir in subdirectories:
                subdir_path = os.path.join(research_dir, subdir)
                fs_mcp.create_directory(subdir_path)
            
            # Create a README file with information about the research
            readme_content = f"""# Research: {topic}

Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Session ID: {session_id or "N/A"}

## Directory Structure

- sources/: Source materials and references
- summaries/: Summaries of research findings
- images/: Images, diagrams, and visual aids
- notes/: Research notes and observations
- final_output/: Final research output and reports

## Research Topic

{topic}
"""
            
            readme_path = os.path.join(research_dir, "README.md")
            fs_mcp.write_file(readme_path, readme_content)
            
            # Create directory info
            directory_info: DirectoryInfo = {
                "path": research_dir,
                "name": research_dir,
                "files": [{
                    "path": readme_path,
                    "name": "README.md",
                    "type": "file",
                    "size": len(readme_content.encode('utf-8')),
                    "created": datetime.now().isoformat(),
                    "modified": datetime.now().isoformat(),
                    "content_type": "text/markdown",
                    "metadata": None
                }],
                "subdirectories": [os.path.join(research_dir, subdir) for subdir in subdirectories],
                "total_files": 1,
                "total_size": len(readme_content.encode('utf-8'))
            }
            
            # Update response
            response["success"] = True
            response["directory_info"] = directory_info
            response["message"] = f"Successfully created research directory structure at {research_dir}"
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            
            return response
        
        except Exception as e:
            logger.error(f"Error creating research directory: {e}")
            response["success"] = False
            response["message"] = f"Error creating research directory: {str(e)}"
            response["errors"].append({
                "type": "research_directory_create_error",
                "error": str(e)
            })
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            return response
    
    def save_research_summary(self, content: str, topic: str, research_dir: str, 
                             metadata: Optional[Dict[str, Any]] = None,
                             namespace: Optional[str] = None,
                             session_id: Optional[str] = None) -> FileOperationResponse:
        """
        Save a research summary to the appropriate location.
        
        Args:
            content: Summary content
            topic: Research topic
            research_dir: Research directory
            metadata: Optional metadata
            namespace: Optional namespace for storing metadata
            session_id: Optional session ID
            
        Returns:
            File operation response
        """
        # Create a sanitized topic name for the filename
        sanitized_topic = re.sub(r'[^\w\s-]', '', topic).strip().lower()
        sanitized_topic = re.sub(r'[-\s]+', '_', sanitized_topic)
        
        # Create a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create the file path
        file_name = f"summary_{sanitized_topic}_{timestamp}.md"
        file_path = os.path.join(research_dir, "summaries", file_name)
        
        # Generate metadata if not provided
        if metadata is None:
            metadata = {
                "topic": topic,
                "created_at": datetime.now().isoformat(),
                "session_id": session_id,
                "type": "research_summary"
            }
        
        # Save the file
        request: SaveFileRequest = {
            "content": content,
            "path": file_path,
            "metadata": metadata,
            "overwrite": False,
            "namespace": namespace,
            "session_id": session_id
        }
        
        return self.save_file(request)
    
    def save_research_image(self, image_url: str, description: str, topic: str, 
                           research_dir: str, metadata: Optional[Dict[str, Any]] = None,
                           namespace: Optional[str] = None,
                           session_id: Optional[str] = None) -> FileOperationResponse:
        """
        Save a research image to the appropriate location.
        
        Args:
            image_url: URL of the image
            description: Description of the image
            topic: Research topic
            research_dir: Research directory
            metadata: Optional metadata
            namespace: Optional namespace for storing metadata
            session_id: Optional session ID
            
        Returns:
            File operation response
        """
        # Create a sanitized topic name for the filename
        sanitized_topic = re.sub(r'[^\w\s-]', '', topic).strip().lower()
        sanitized_topic = re.sub(r'[-\s]+', '_', sanitized_topic)
        
        # Create a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine file extension from URL
        url_path = urlparse(image_url).path
        ext = os.path.splitext(url_path)[1]
        if not ext:
            ext = ".jpg"  # Default extension
        
        # Create the file path
        file_name = f"image_{sanitized_topic}_{timestamp}{ext}"
        file_path = os.path.join(research_dir, "images", file_name)
        
        # Generate metadata if not provided
        if metadata is None:
            metadata = {
                "topic": topic,
                "description": description,
                "source_url": image_url,
                "created_at": datetime.now().isoformat(),
                "session_id": session_id,
                "type": "research_image"
            }
        
        # Save the image
        request: SaveUrlContentRequest = {
            "url": image_url,
            "path": file_path,
            "metadata": metadata,
            "overwrite": False,
            "namespace": namespace,
            "session_id": session_id
        }
        
        return self.save_url_content(request)
    
    def cleanup(self):
        """Clean up resources used by the File Manager Agent."""
        # Close all MCP clients
        mcp_manager.close_all()


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the File Manager Agent")
    parser.add_argument("--mode", type=str, choices=["save", "read", "list", "create-dir", "delete", "search", "organize", "research-dir"], 
                      default="list", help="Operation mode")
    parser.add_argument("--path", type=str, help="File or directory path")
    parser.add_argument("--content", type=str, help="Content to write to file")
    parser.add_argument("--url", type=str, help="URL to download content from")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--recursive", action="store_true", help="Recursive operation")
    parser.add_argument("--source-dir", type=str, help="Source directory for organization")
    parser.add_argument("--target-dir", type=str, help="Target directory for organization")
    parser.add_argument("--org-type", type=str, choices=["date", "type", "custom"], default="type", 
                      help="Organization type")
    parser.add_argument("--topic", type=str, help="Research topic")
    parser.add_argument("--workspace", type=str, help="Workspace directory")
    parser.add_argument("--namespace", type=str, default="default", help="Namespace for storing in memory")
    parser.add_argument("--session-id", type=str, help="Session ID for continuity")
    args = parser.parse_args()
    
    # Set workspace directory if provided
    if args.workspace:
        os.environ["WORKSPACE_DIR"] = os.path.abspath(args.workspace)
    
    # Create the File Manager Agent
    agent = FileManagerAgent()
    
    try:
        # Execute the requested operation
        if args.mode == "save":
            if args.url:
                # Save content from URL
                if not args.path:
                    parser.error("--path is required for save mode with URL")
                
                request: SaveUrlContentRequest = {
                    "url": args.url,
                    "path": args.path,
                    "metadata": None,
                    "overwrite": True,
                    "namespace": args.namespace,
                    "session_id": args.session_id
                }
                
                response = agent.save_url_content(request)
                print(json.dumps(response, indent=2))
            
            elif args.content:
                # Save content from command line
                if not args.path:
                    parser.error("--path is required for save mode")
                
                request: SaveFileRequest = {
                    "content": args.content,
                    "path": args.path,
                    "metadata": None,
                    "overwrite": True,
                    "namespace": args.namespace,
                    "session_id": args.session_id
                }
                
                response = agent.save_file(request)
                print(json.dumps(response, indent=2))
            
            else:
                parser.error("Either --content or --url is required for save mode")
        
        elif args.mode == "read":
            # Read file content
            if not args.path:
                parser.error("--path is required for read mode")
            
            response = agent.read_file(args.path)
            print(response["message"])
        
        elif args.mode == "list":
            # List directory contents
            path = args.path or "."
            response = agent.list_directory(path, recursive=args.recursive)
            print(json.dumps(response, indent=2))
        
        elif args.mode == "create-dir":
            # Create directory
            if not args.path:
                parser.error("--path is required for create-dir mode")
            
            response = agent.create_directory(args.path)
            print(json.dumps(response, indent=2))
        
        elif args.mode == "delete":
            # Delete file
            if not args.path:
                parser.error("--path is required for delete mode")
            
            response = agent.delete_file(args.path)
            print(json.dumps(response, indent=2))
        
        elif args.mode == "search":
            # Search for files
            if not args.query:
                parser.error("--query is required for search mode")
            
            request: SearchFilesRequest = {
                "query": args.query,
                "directory": args.path or ".",
                "file_types": None,
                "recursive": args.recursive,
                "namespace": args.namespace,
                "session_id": args.session_id
            }
            
            response = agent.search_files(request)
            print(json.dumps(response, indent=2))
        
        elif args.mode == "organize":
            # Organize files
            if not args.source_dir:
                parser.error("--source-dir is required for organize mode")
            
            request: OrganizeFilesRequest = {
                "source_directory": args.source_dir,
                "target_directory": args.target_dir,
                "organization_type": args.org_type,
                "custom_rules": None,
                "namespace": args.namespace,
                "session_id": args.session_id
            }
            
            response = agent.organize_files(request)
            print(json.dumps(response, indent=2))
        
        elif args.mode == "research-dir":
            # Create research directory
            if not args.topic:
                parser.error("--topic is required for research-dir mode")
            
            response = agent.create_research_directory(args.topic, session_id=args.session_id)
            print(json.dumps(response, indent=2))
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    finally:
        # Clean up
        agent.cleanup()


def run_interactive_mode():
    """Run the File Manager Agent in interactive mode."""
    print("File Manager Agent - Interactive Mode")
    print("=====================================")
    
    # Create the File Manager Agent
    agent = FileManagerAgent()
    
    # Get session information
    namespace = input("Enter namespace (default: 'default'): ") or "default"
    session_id = input("Enter session ID (optional): ") or str(uuid.uuid4())
    
    try:
        while True:
            print("\nAvailable operations:")
            print("1. Save file")
            print("2. Save content from URL")
            print("3. Read file")
            print("4. List directory")
            print("5. Create directory")
            print("6. Delete file")
            print("7. Search files")
            print("8. Organize files")
            print("9. Create research directory")
            print("0. Exit")
            
            choice = input("\nEnter choice (0-9): ")
            
            if choice == "0":
                break
            
            elif choice == "1":  # Save file
                path = input("Enter file path: ")
                print("Enter content (end with an empty line):")
                content_lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    content_lines.append(line)
                
                content = "\n".join(content_lines)
                
                # Save the file
                request: SaveFileRequest = {
                    "content": content,
                    "path": path,
                    "metadata": None,
                    "overwrite": True,
                    "namespace": namespace,
                    "session_id": session_id
                }
                
                response = agent.save_file(request)
                
                if response["success"]:
                    print(f"File saved successfully to {path}")
                else:
                    print(f"Error saving file: {response['message']}")
            
            elif choice == "2":  # Save content from URL
                url = input("Enter URL: ")
                path = input("Enter file path: ")
                
                # Save the content
                request: SaveUrlContentRequest = {
                    "url": url,
                    "path": path,
                    "metadata": None,
                    "overwrite": True,
                    "namespace": namespace,
                    "session_id": session_id
                }
                
                response = agent.save_url_content(request)
                
                if response["success"]:
                    print(f"Content from {url} saved successfully to {path}")
                else:
                    print(f"Error saving content: {response['message']}")
            
            elif choice == "3":  # Read file
                path = input("Enter file path: ")
                
                response = agent.read_file(path)
                
                if response["success"]:
                    print("\nFile content:")
                    print("-------------")
                    print(response["message"])
                else:
                    print(f"Error reading file: {response['message']}")
            
            elif choice == "4":  # List directory
                path = input("Enter directory path (default: '.'): ") or "."
                recursive = input("List recursively? (y/n, default: n): ").lower() == "y"
                
                response = agent.list_directory(path, recursive=recursive)
                
                if response["success"] and response["directory_info"]:
                    print(f"\nDirectory: {response['directory_info']['path']}")
                    print(f"Files: {response['directory_info']['total_files']}")
                    
                    print("\nFiles:")
                    for file in response["directory_info"]["files"]:
                        print(f"- {file['path']}")
                    
                    print("\nSubdirectories:")
                    for subdir in response["directory_info"]["subdirectories"]:
                        print(f"- {subdir}")
                else:
                    print(f"Error listing directory: {response['message']}")
            
            elif choice == "5":  # Create directory
                path = input("Enter directory path: ")
                
                response = agent.create_directory(path)
                
                if response["success"]:
                    print(f"Directory created successfully at {path}")
                else:
                    print(f"Error creating directory: {response['message']}")
            
            elif choice == "6":  # Delete file
                path = input("Enter file path: ")
                
                response = agent.delete_file(path)
                
                if response["success"]:
                    print(f"File deleted successfully: {path}")
                else:
                    print(f"Error deleting file: {response['message']}")
            
            elif choice == "7":  # Search files
                query = input("Enter search query: ")
                directory = input("Enter directory to search (default: '.'): ") or "."
                recursive = input("Search recursively? (y/n, default: y): ").lower() != "n"
                
                request: SearchFilesRequest = {
                    "query": query,
                    "directory": directory,
                    "file_types": None,
                    "recursive": recursive,
                    "namespace": namespace,
                    "session_id": session_id
                }
                
                response = agent.search_files(request)
                
                if response["success"]:
                    print(f"\nFound {len(response['results'])} results for '{query}':")
                    for file in response["results"]:
                        print(f"- {file['path']}")
                else:
                    print(f"Error searching files: {response['message']}")
            
            elif choice == "8":  # Organize files
                source_dir = input("Enter source directory: ")
                target_dir = input("Enter target directory (default: same as source): ") or source_dir
                org_type = input("Enter organization type (date/type/custom, default: type): ") or "type"
                
                if org_type not in ["date", "type", "custom"]:
                    print("Invalid organization type. Using 'type'.")
                    org_type = "type"
                
                request: OrganizeFilesRequest = {
                    "source_directory": source_dir,
                    "target_directory": target_dir,
                    "organization_type": org_type,
                    "custom_rules": None,
                    "namespace": namespace,
                    "session_id": session_id
                }
                
                response = agent.organize_files(request)
                
                if response["success"]:
                    print(f"\nOrganized {len(response['organized_files'])} files:")
                    for move in response["organized_files"]:
                        print(f"- {move['old_path']} -> {move['new_path']}")
                else:
                    print(f"Error organizing files: {response['message']}")
            
            elif choice == "9":  # Create research directory
                topic = input("Enter research topic: ")
                
                response = agent.create_research_directory(topic, session_id=session_id)
                
                if response["success"] and response["directory_info"]:
                    print(f"\nCreated research directory: {response['directory_info']['path']}")
                    print("Subdirectories:")
                    for subdir in response["directory_info"]["subdirectories"]:
                        print(f"- {subdir}")
                else:
                    print(f"Error creating research directory: {response['message']}")
            
            else:
                print("Invalid choice. Please try again.")
    
    except KeyboardInterrupt:
        print("\nExiting interactive mode...")
    
    except Exception as e:
        print(f"Error in interactive mode: {e}")
    
    finally:
        # Clean up
        agent.cleanup()


if __name__ == "__main__" and len(sys.argv) == 1:
    # If no arguments provided, run in interactive mode
    run_interactive_mode()
