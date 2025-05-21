"""
Mock MCP (Model Context Protocol) implementations for testing purposes.
"""

import json
import os
import uuid
from typing import Dict, List, Any, Optional, Union
from unittest.mock import MagicMock, patch

# Import MCP classes
from apps.mcps.brave_search_mcp import BraveSearchMCP
from apps.mcps.everart_mcp import EverArtMCP
from apps.mcps.fetch_mcp import FetchMCP
from apps.mcps.filesystem_mcp import FilesystemMCP
from apps.mcps.memory_mcp import MemoryMCP


class MockBraveSearchMCP:
    """Mock implementation of the Brave Search MCP for testing."""
    
    def __init__(self, api_key: Optional[str] = "mock_api_key", use_docker: bool = False):
        """
        Initialize the mock Brave Search MCP.
        
        Args:
            api_key: Mock API key for Brave Search
            use_docker: Whether to simulate using Docker (has no effect in mock)
        """
        self.api_key = api_key
        self.use_docker = use_docker
        self.process = MagicMock()
        self.process.poll.return_value = None  # Process is running
        
        # Track method calls
        self.web_search_called = False
        self.web_search_args = None
        self.web_search_kwargs = None
        self.web_search_result = self._generate_mock_web_search_results()
        
        self.local_search_called = False
        self.local_search_args = None
        self.local_search_kwargs = None
        self.local_search_result = self._generate_mock_local_search_results()
        
        self.list_tools_called = False
        self.list_tools_result = [
            {
                "name": "brave_web_search",
                "description": "Performs a web search using the Brave Search API, ideal for general queries, news, articles, and online content."
            },
            {
                "name": "brave_local_search",
                "description": "Searches for local businesses and places using Brave's Local Search API."
            }
        ]
        
        self.close_called = False
    
    def _generate_mock_web_search_results(self, count: int = 3) -> str:
        """Generate mock web search results."""
        results = []
        for i in range(1, count + 1):
            results.append(
                f"Title: Mock Search Result {i}\n"
                f"Description: This is a mock search result description for testing purposes. Result #{i}.\n"
                f"URL: https://example.com/result{i}"
            )
        return "\n\n".join(results)
    
    def _generate_mock_local_search_results(self, count: int = 2) -> str:
        """Generate mock local search results."""
        results = []
        for i in range(1, count + 1):
            results.append(
                f"Name: Mock Local Business {i}\n"
                f"Address: {i}23 Main St, Example City, EX {i}2345\n"
                f"Phone: (555) {i}23-4567\n"
                f"Rating: {4.0 + (i / 10)} ({i * 10} reviews)\n"
                f"Price Range: {'$' * min(i, 4)}\n"
                f"Hours: Mon-Fri 9AM-5PM, Sat-Sun 10AM-4PM\n"
                f"Description: This is a mock local business description for testing purposes. Business #{i}."
            )
        return "\n---\n".join(results)
    
    def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock sending a request to the MCP server."""
        if method == "listTools":
            self.list_tools_called = True
            return {"tools": self.list_tools_result}
        
        if method == "callTool":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            
            if tool_name == "brave_web_search":
                self.web_search_called = True
                self.web_search_args = (arguments.get("query", ""),)
                self.web_search_kwargs = {
                    "count": arguments.get("count", 10),
                    "offset": arguments.get("offset", 0)
                }
                return {
                    "content": [{"type": "text", "text": self.web_search_result}],
                    "isError": False
                }
            
            if tool_name == "brave_local_search":
                self.local_search_called = True
                self.local_search_args = (arguments.get("query", ""),)
                self.local_search_kwargs = {
                    "count": arguments.get("count", 5)
                }
                return {
                    "content": [{"type": "text", "text": self.local_search_result}],
                    "isError": False
                }
        
        # Unknown method or tool
        return {
            "content": [{"type": "text", "text": f"Unknown method or tool: {method}"}],
            "isError": True
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools in the MCP server."""
        self.list_tools_called = True
        return self.list_tools_result
    
    def web_search(self, query: str, count: int = 10, offset: int = 0) -> str:
        """
        Perform a web search using Brave Search.
        
        Args:
            query: Search query
            count: Number of results (1-20, default 10)
            offset: Pagination offset (max 9, default 0)
            
        Returns:
            Search results as a formatted string
        """
        self.web_search_called = True
        self.web_search_args = (query,)
        self.web_search_kwargs = {"count": count, "offset": offset}
        
        # Validate inputs
        if not query:
            raise RuntimeError("Query cannot be empty")
        if len(query) > 400:
            raise RuntimeError("Query too long (max 400 chars)")
        if count < 1 or count > 20:
            raise RuntimeError("Count out of range (1-20)")
        if offset < 0 or offset > 9:
            raise RuntimeError("Offset out of range (0-9)")
        
        return self.web_search_result
    
    def set_web_search_result(self, result: str):
        """Set the result to be returned by web_search()."""
        self.web_search_result = result
    
    def local_search(self, query: str, count: int = 5) -> str:
        """
        Search for local businesses and places using Brave's Local Search API.
        
        Args:
            query: Local search query
            count: Number of results (1-20, default 5)
            
        Returns:
            Local search results as a formatted string
        """
        self.local_search_called = True
        self.local_search_args = (query,)
        self.local_search_kwargs = {"count": count}
        
        # Validate inputs
        if not query:
            raise RuntimeError("Query cannot be empty")
        if len(query) > 400:
            raise RuntimeError("Query too long (max 400 chars)")
        if count < 1 or count > 20:
            raise RuntimeError("Count out of range (1-20)")
        
        return self.local_search_result
    
    def set_local_search_result(self, result: str):
        """Set the result to be returned by local_search()."""
        self.local_search_result = result
    
    def close(self):
        """Close the MCP server process."""
        self.close_called = True
        self.process.poll.return_value = 0  # Process has exited
    
    def __enter__(self):
        """Enter the context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.close()


class MockEverArtMCP:
    """Mock implementation of the EverArt MCP for testing."""
    
    def __init__(self, api_key: Optional[str] = "mock_api_key", use_docker: bool = False):
        """
        Initialize the mock EverArt MCP.
        
        Args:
            api_key: Mock API key for EverArt
            use_docker: Whether to simulate using Docker (has no effect in mock)
        """
        self.api_key = api_key
        self.use_docker = use_docker
        self.process = MagicMock()
        self.process.poll.return_value = None  # Process is running
        
        # Track method calls
        self.generate_image_called = False
        self.generate_image_args = None
        self.generate_image_kwargs = None
        self.generate_image_result = self._generate_mock_image_result()
        
        self.enhance_image_called = False
        self.enhance_image_args = None
        self.enhance_image_kwargs = None
        self.enhance_image_result = self._generate_mock_enhanced_image_result()
        
        self.describe_image_called = False
        self.describe_image_args = None
        self.describe_image_kwargs = None
        self.describe_image_result = self._generate_mock_image_description()
        
        self.list_tools_called = False
        self.list_tools_result = [
            {
                "name": "everart_generate_image",
                "description": "Generate images using EverArt AI"
            },
            {
                "name": "everart_enhance_image",
                "description": "Enhance or modify an existing image using EverArt AI"
            },
            {
                "name": "everart_describe_image",
                "description": "Generate a detailed description of an image using EverArt AI"
            }
        ]
        
        self.close_called = False
    
    def _generate_mock_image_result(self) -> str:
        """Generate a mock image generation result."""
        return (
            "Image generated successfully!\n"
            "URL: https://example.com/generated_image.jpg\n"
            "Prompt: A serene mountain landscape with a lake at sunset\n"
            "Style: oil painting\n"
            "Aspect Ratio: 16:9"
        )
    
    def _generate_mock_enhanced_image_result(self) -> str:
        """Generate a mock image enhancement result."""
        return (
            "Image enhanced successfully!\n"
            "URL: https://example.com/enhanced_image.jpg\n"
            "Original: https://example.com/original_image.jpg\n"
            "Prompt: Make the colors more vibrant and add a dramatic sky\n"
            "Strength: 0.7"
        )
    
    def _generate_mock_image_description(self) -> str:
        """Generate a mock image description."""
        return (
            "Image Description:\n"
            "The image shows a serene mountain landscape with a lake at sunset.\n"
            "The mountains are reflected in the still water of the lake.\n"
            "The sky is painted in vibrant orange and purple hues.\n"
            "There are some pine trees in the foreground silhouetted against the sunset."
        )
    
    def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock sending a request to the MCP server."""
        if method == "listTools":
            self.list_tools_called = True
            return {"tools": self.list_tools_result}
        
        if method == "callTool":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            
            if tool_name == "everart_generate_image":
                self.generate_image_called = True
                self.generate_image_args = (arguments.get("prompt", ""),)
                self.generate_image_kwargs = {
                    "style": arguments.get("style"),
                    "aspect_ratio": arguments.get("aspect_ratio", "1:1"),
                    "num_images": arguments.get("num_images", 1)
                }
                return {
                    "content": [{"type": "text", "text": self.generate_image_result}],
                    "isError": False
                }
            
            if tool_name == "everart_enhance_image":
                self.enhance_image_called = True
                self.enhance_image_args = (arguments.get("image_url", ""), arguments.get("prompt", ""))
                self.enhance_image_kwargs = {
                    "strength": arguments.get("strength", 0.5)
                }
                return {
                    "content": [{"type": "text", "text": self.enhance_image_result}],
                    "isError": False
                }
            
            if tool_name == "everart_describe_image":
                self.describe_image_called = True
                self.describe_image_args = (arguments.get("image_url", ""),)
                self.describe_image_kwargs = {
                    "detail_level": arguments.get("detail_level", "medium")
                }
                return {
                    "content": [{"type": "text", "text": self.describe_image_result}],
                    "isError": False
                }
        
        # Unknown method or tool
        return {
            "content": [{"type": "text", "text": f"Unknown method or tool: {method}"}],
            "isError": True
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools in the MCP server."""
        self.list_tools_called = True
        return self.list_tools_result
    
    def generate_image(self, prompt: str, style: Optional[str] = None, 
                      aspect_ratio: str = "1:1", num_images: int = 1) -> str:
        """
        Generate images using EverArt AI.
        
        Args:
            prompt: Text description of the image to generate
            style: Optional style to apply (e.g., "realistic", "anime", "oil painting")
            aspect_ratio: Aspect ratio of the generated image (default: "1:1")
            num_images: Number of images to generate (default: 1)
            
        Returns:
            Generated image URLs as a formatted string
        """
        self.generate_image_called = True
        self.generate_image_args = (prompt,)
        self.generate_image_kwargs = {
            "style": style,
            "aspect_ratio": aspect_ratio,
            "num_images": num_images
        }
        
        # Validate inputs
        if not prompt:
            raise RuntimeError("Prompt cannot be empty")
        if len(prompt) > 1000:
            raise RuntimeError("Prompt too long (max 1000 chars)")
        if num_images < 1 or num_images > 4:
            raise RuntimeError("Number of images out of range (1-4)")
        
        return self.generate_image_result
    
    def set_generate_image_result(self, result: str):
        """Set the result to be returned by generate_image()."""
        self.generate_image_result = result
    
    def enhance_image(self, image_url: str, prompt: str, 
                     strength: float = 0.5) -> str:
        """
        Enhance or modify an existing image using EverArt AI.
        
        Args:
            image_url: URL of the source image to enhance
            prompt: Text description of the desired modifications
            strength: Strength of the enhancement (0.0-1.0, default: 0.5)
            
        Returns:
            Enhanced image URL as a formatted string
        """
        self.enhance_image_called = True
        self.enhance_image_args = (image_url, prompt)
        self.enhance_image_kwargs = {"strength": strength}
        
        # Validate inputs
        if not image_url:
            raise RuntimeError("Image URL cannot be empty")
        if not prompt:
            raise RuntimeError("Prompt cannot be empty")
        if strength < 0.0 or strength > 1.0:
            raise RuntimeError("Strength out of range (0.0-1.0)")
        
        return self.enhance_image_result
    
    def set_enhance_image_result(self, result: str):
        """Set the result to be returned by enhance_image()."""
        self.enhance_image_result = result
    
    def describe_image(self, image_url: str, detail_level: str = "medium") -> str:
        """
        Generate a detailed description of an image using EverArt AI.
        
        Args:
            image_url: URL of the image to describe
            detail_level: Level of detail in the description ("low", "medium", "high")
            
        Returns:
            Image description as a formatted string
        """
        self.describe_image_called = True
        self.describe_image_args = (image_url,)
        self.describe_image_kwargs = {"detail_level": detail_level}
        
        # Validate inputs
        if not image_url:
            raise RuntimeError("Image URL cannot be empty")
        if detail_level not in ["low", "medium", "high"]:
            raise RuntimeError("Invalid detail level (must be 'low', 'medium', or 'high')")
        
        return self.describe_image_result
    
    def set_describe_image_result(self, result: str):
        """Set the result to be returned by describe_image()."""
        self.describe_image_result = result
    
    def close(self):
        """Close the MCP server process."""
        self.close_called = True
        self.process.poll.return_value = 0  # Process has exited
    
    def __enter__(self):
        """Enter the context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.close()


class MockFetchMCP:
    """Mock implementation of the Fetch MCP for testing."""
    
    def __init__(self, use_docker: bool = False):
        """
        Initialize the mock Fetch MCP.
        
        Args:
            use_docker: Whether to simulate using Docker (has no effect in mock)
        """
        self.use_docker = use_docker
        self.process = MagicMock()
        self.process.poll.return_value = None  # Process is running
        
        # Track method calls
        self.fetch_url_called = False
        self.fetch_url_args = None
        self.fetch_url_kwargs = None
        self.fetch_url_result = self._generate_mock_html_content()
        
        self.fetch_text_called = False
        self.fetch_text_args = None
        self.fetch_text_kwargs = None
        self.fetch_text_result = self._generate_mock_text_content()
        
        self.fetch_html_called = False
        self.fetch_html_args = None
        self.fetch_html_kwargs = None
        self.fetch_html_result = self._generate_mock_html_content()
        
        self.list_tools_called = False
        self.list_tools_result = [
            {
                "name": "fetch_url",
                "description": "Fetch content from a URL"
            },
            {
                "name": "fetch_text",
                "description": "Fetch the text content from a URL, removing HTML tags"
            },
            {
                "name": "fetch_html",
                "description": "Fetch the raw HTML content from a URL"
            }
        ]
        
        self.close_called = False
    
    def _generate_mock_html_content(self) -> str:
        """Generate mock HTML content."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mock Webpage</title>
            <meta charset="UTF-8">
        </head>
        <body>
            <h1>Mock Webpage Title</h1>
            <p>This is a mock webpage content for testing purposes.</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
                <li>Item 3</li>
            </ul>
            <div class="footer">
                <p>Mock footer content</p>
            </div>
        </body>
        </html>
        """
    
    def _generate_mock_text_content(self) -> str:
        """Generate mock text content."""
        return """
        Mock Webpage Title
        
        This is a mock webpage content for testing purposes.
        
        * Item 1
        * Item 2
        * Item 3
        
        Mock footer content
        """
    
    def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock sending a request to the MCP server."""
        if method == "listTools":
            self.list_tools_called = True
            return {"tools": self.list_tools_result}
        
        if method == "callTool":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            
            if tool_name == "fetch_url":
                self.fetch_url_called = True
                self.fetch_url_args = (arguments.get("url", ""),)
                self.fetch_url_kwargs = {
                    "selector": arguments.get("selector"),
                    "timeout": arguments.get("timeout", 30000),
                    "wait_for": arguments.get("waitFor")
                }
                return {
                    "content": [{"type": "text", "text": self.fetch_url_result}],
                    "isError": False
                }
            
            if tool_name == "fetch_text":
                self.fetch_text_called = True
                self.fetch_text_args = (arguments.get("url", ""),)
                self.fetch_text_kwargs = {
                    "timeout": arguments.get("timeout", 30000),
                    "wait_for": arguments.get("waitFor")
                }
                return {
                    "content": [{"type": "text", "text": self.fetch_text_result}],
                    "isError": False
                }
            
            if tool_name == "fetch_html":
                self.fetch_html_called = True
                self.fetch_html_args = (arguments.get("url", ""),)
                self.fetch_html_kwargs = {
                    "timeout": arguments.get("timeout", 30000),
                    "wait_for": arguments.get("waitFor")
                }
                return {
                    "content": [{"type": "text", "text": self.fetch_html_result}],
                    "isError": False
                }
        
        # Unknown method or tool
        return {
            "content": [{"type": "text", "text": f"Unknown method or tool: {method}"}],
            "isError": True
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools in the MCP server."""
        self.list_tools_called = True
        return self.list_tools_result
    
    def fetch_url(self, url: str, selector: Optional[str] = None, 
                 timeout: int = 30000, wait_for: Optional[str] = None) -> str:
        """
        Fetch content from a URL.
        
        Args:
            url: The URL to fetch
            selector: Optional CSS selector to extract specific content
            timeout: Timeout in milliseconds (default: 30000)
            wait_for: Optional selector to wait for before extracting content
            
        Returns:
            The fetched content as text
        """
        self.fetch_url_called = True
        self.fetch_url_args = (url,)
        self.fetch_url_kwargs = {
            "selector": selector,
            "timeout": timeout,
            "wait_for": wait_for
        }
        
        # Validate inputs
        if not url:
            raise RuntimeError("URL cannot be empty")
        if not url.startswith(("http://", "https://")):
            raise RuntimeError("Invalid URL scheme (must be http:// or https://)")
        if timeout < 0:
            raise RuntimeError("Timeout cannot be negative")
        
        return self.fetch_url_result
    
    def set_fetch_url_result(self, result: str):
        """Set the result to be returned by fetch_url()."""
        self.fetch_url_result = result
    
    def fetch_html(self, url: str, timeout: int = 30000, 
                  wait_for: Optional[str] = None) -> str:
        """
        Fetch the raw HTML content from a URL.
        
        Args:
            url: The URL to fetch
            timeout: Timeout in milliseconds (default: 30000)
            wait_for: Optional selector to wait for before extracting content
            
        Returns:
            The raw HTML content
        """
        self.fetch_html_called = True
        self.fetch_html_args = (url,)
        self.fetch_html_kwargs = {
            "timeout": timeout,
            "wait_for": wait_for
        }
        
        # Validate inputs
        if not url:
            raise RuntimeError("URL cannot be empty")
        if not url.startswith(("http://", "https://")):
            raise RuntimeError("Invalid URL scheme (must be http:// or https://)")
        if timeout < 0:
            raise RuntimeError("Timeout cannot be negative")
        
        return self.fetch_html_result
    
    def set_fetch_html_result(self, result: str):
        """Set the result to be returned by fetch_html()."""
        self.fetch_html_result = result
    
    def fetch_text(self, url: str, timeout: int = 30000, 
                  wait_for: Optional[str] = None) -> str:
        """
        Fetch the text content from a URL, removing HTML tags.
        
        Args:
            url: The URL to fetch
            timeout: Timeout in milliseconds (default: 30000)
            wait_for: Optional selector to wait for before extracting content
            
        Returns:
            The text content with HTML tags removed
        """
        self.fetch_text_called = True
        self.fetch_text_args = (url,)
        self.fetch_text_kwargs = {
            "timeout": timeout,
            "wait_for": wait_for
        }
        
        # Validate inputs
        if not url:
            raise RuntimeError("URL cannot be empty")
        if not url.startswith(("http://", "https://")):
            raise RuntimeError("Invalid URL scheme (must be http:// or https://)")
        if timeout < 0:
            raise RuntimeError("Timeout cannot be negative")
        
        return self.fetch_text_result
    
    def set_fetch_text_result(self, result: str):
        """Set the result to be returned by fetch_text()."""
        self.fetch_text_result = result
    
    def close(self):
        """Close the MCP server process."""
        self.close_called = True
        self.process.poll.return_value = 0  # Process has exited
    
    def __enter__(self):
        """Enter the context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.close()


class MockFilesystemMCP:
    """Mock implementation of the Filesystem MCP for testing."""
    
    def __init__(self, workspace_dir: Optional[str] = "/mock/workspace", use_docker: bool = False):
        """
        Initialize the mock Filesystem MCP.
        
        Args:
            workspace_dir: Mock workspace directory
            use_docker: Whether to simulate using Docker (has no effect in mock)
        """
        self.workspace_dir = workspace_dir
        self.use_docker = use_docker
        self.process = MagicMock()
        self.process.poll.return_value = None  # Process is running
        
        # Mock file system state
        self.files = {
            "document.txt": "This is a mock document content.",
            "research/report.txt": "Mock research report content.",
            "images/image1.jpg": "Mock image content (binary).",
            "data/data.json": '{"key": "value", "number": 42}'
        }
        
        # Track method calls
        self.read_file_called = False
        self.read_file_args = None
        self.read_file_kwargs = None
        self.read_file_result = "Mock file content"
        
        self.write_file_called = False
        self.write_file_args = None
        self.write_file_kwargs = None
        self.write_file_result = "File written successfully"
        
        self.list_directory_called = False
        self.list_directory_args = None
        self.list_directory_kwargs = None
        self.list_directory_result = self._generate_mock_directory_listing()
        
        self.create_directory_called = False
        self.create_directory_args = None
        self.create_directory_kwargs = None
        self.create_directory_result = "Directory created successfully"
        
        self.delete_file_called = False
        self.delete_file_args = None
        self.delete_file_kwargs = None
        self.delete_file_result = "File deleted successfully"
        
        self.file_exists_called = False
        self.file_exists_args = None
        self.file_exists_kwargs = None
        self.file_exists_result = True
        
        self.search_files_called = False
        self.search_files_args = None
        self.search_files_kwargs = None
        self.search_files_result = self._generate_mock_search_results()
        
        self.list_tools_called = False
        self.list_tools_result = [
            {
                "name": "read_file",
                "description": "Read the contents of a file"
            },
            {
                "name": "write_file",
                "description": "Write content to a file"
            },
            {
                "name": "list_directory",
                "description": "List files and directories in a directory"
            },
            {
                "name": "create_directory",
                "description": "Create a directory"
            },
            {
                "name": "delete_file",
                "description": "Delete a file"
            },
            {
                "name": "file_exists",
                "description": "Check if a file exists"
            },
            {
                "name": "search_files",
                "description": "Search for files matching a pattern"
            }
        ]
        
        self.close_called = False
    
    def _generate_mock_directory_listing(self) -> str:
        """Generate a mock directory listing."""
        return """
        document.txt
        research/
        ├── report.txt
        ├── data.csv
        images/
        ├── image1.jpg
        ├── image2.jpg
        data/
        └── data.json
        """
    
    def _generate_mock_search_results(self) -> str:
        """Generate mock search results."""
        return """
        Found 3 files matching pattern '*.txt':
        document.txt
        research/report.txt
        research/data.csv
        """
    
    def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock sending a request to the MCP server."""
        if method == "listTools":
            self.list_tools_called = True
            return {"tools": self.list_tools_result}
        
        if method == "callTool":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            
            if tool_name == "read_file":
                self.read_file_called = True
                self.read_file_args = (arguments.get("path", ""),)
                self.read_file_kwargs = {}
                
                # Check if file exists in mock filesystem
                path = arguments.get("path", "")
                if path in self.files:
                    content = self.files[path]
                else:
                    return {
                        "content": [{"type": "text", "text": f"Error: File not found: {path}"}],
                        "isError": True
                    }
                
                return {
                    "content": [{"type": "text", "text": content}],
                    "isError": False
                }
            
            if tool_name == "write_file":
                self.write_file_called = True
                self.write_file_args = (arguments.get("path", ""), arguments.get("content", ""))
                self.write_file_kwargs = {}
                
                # Update mock filesystem
                path = arguments.get("path", "")
                content = arguments.get("content", "")
                self.files[path] = content
                
                return {
                    "content": [{"type": "text", "text": self.write_file_result}],
                    "isError": False
                }
            
            if tool_name == "list_directory":
                self.list_directory_called = True
                self.list_directory_args = (arguments.get("path", "."),)
                self.list_directory_kwargs = {
                    "recursive": arguments.get("recursive", False)
                }
                return {
                    "content": [{"type": "text", "text": self.list_directory_result}],
                    "isError": False
                }
            
            if tool_name == "create_directory":
                self.create_directory_called = True
                self.create_directory_args = (arguments.get("path", ""),)
                self.create_directory_kwargs = {}
                return {
                    "content": [{"type": "text", "text": self.create_directory_result}],
                    "isError": False
                }
            
            if tool_name == "delete_file":
                self.delete_file_called = True
                self.delete_file_args = (arguments.get("path", ""),)
                self.delete_file_kwargs = {}
                
                # Update mock filesystem
                path = arguments.get("path", "")
                if path in self.files:
                    del self.files[path]
                
                return {
                    "content": [{"type": "text", "text": self.delete_file_result}],
                    "isError": False
                }
            
            if tool_name == "file_exists":
                self.file_exists_called = True
                self.file_exists_args = (arguments.get("path", ""),)
                self.file_exists_kwargs = {}
                
                # Check if file exists in mock filesystem
                path = arguments.get("path", "")
                exists = path in self.files
                
                return {
                    "content": [{"type": "text", "text": f"File exists: {exists}"}],
                    "isError": False
                }
            
            if tool_name == "search_files":
                self.search_files_called = True
                self.search_files_args = (arguments.get("pattern", ""),)
                self.search_files_kwargs = {
                    "path": arguments.get("path", "."),
                    "recursive": arguments.get("recursive", True)
                }
                return {
                    "content": [{"type": "text", "text": self.search_files_result}],
                    "isError": False
                }
        
        # Unknown method or tool
        return {
            "content": [{"type": "text", "text": f"Unknown method or tool: {method}"}],
            "isError": True
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools in the MCP server."""
        self.list_tools_called = True
        return self.list_tools_result
    
    def read_file(self, path: str) -> str:
        """
        Read the contents of a file.
        
        Args:
            path: Path to the file, relative to the workspace directory
            
        Returns:
            The file contents as a string
        """
        self.read_file_called = True
        self.read_file_args = (path,)
        self.read_file_kwargs = {}
        
        # Check if file exists in mock filesystem
        if path in self.files:
            return self.files[path]
        else:
            raise RuntimeError(f"File not found: {path}")
    
    def set_read_file_result(self, result: str):
        """Set the result to be returned by read_file()."""
        self.read_file_result = result
    
    def write_file(self, path: str, content: str) -> str:
        """
        Write content to a file.
        
        Args:
            path: Path to the file, relative to the workspace directory
            content: Content to write to the file
            
        Returns:
            Confirmation message
        """
        self.write_file_called = True
        self.write_file_args = (path, content)
        self.write_file_kwargs = {}
        
        # Update mock filesystem
        self.files[path] = content
        
        return self.write_file_result
    
    def set_write_file_result(self, result: str):
        """Set the result to be returned by write_file()."""
        self.write_file_result = result
    
    def list_directory(self, path: str = ".", recursive: bool = False) -> str:
        """
        List files and directories in a directory.
        
        Args:
            path: Path to the directory, relative to the workspace directory
            recursive: Whether to list files recursively
            
        Returns:
            Directory listing as a formatted string
        """
        self.list_directory_called = True
        self.list_directory_args = (path,)
        self.list_directory_kwargs = {"recursive": recursive}
        
        return self.list_directory_result
    
    def set_list_directory_result(self, result: str):
        """Set the result to be returned by list_directory()."""
        self.list_directory_result = result
    
    def create_directory(self, path: str) -> str:
        """
        Create a directory.
        
        Args:
            path: Path to the directory, relative to the workspace directory
            
        Returns:
            Confirmation message
        """
        self.create_directory_called = True
        self.create_directory_args = (path,)
        self.create_directory_kwargs = {}
        
        return self.create_directory_result
    
    def set_create_directory_result(self, result: str):
        """Set the result to be returned by create_directory()."""
        self.create_directory_result = result
    
    def delete_file(self, path: str) -> str:
        """
        Delete a file.
        
        Args:
            path: Path to the file, relative to the workspace directory
            
        Returns:
            Confirmation message
        """
        self.delete_file_called = True
        self.delete_file_args = (path,)
        self.delete_file_kwargs = {}
        
        # Update mock filesystem
        if path in self.files:
            del self.files[path]
        
        return self.delete_file_result
    
    def set_delete_file_result(self, result: str):
        """Set the result to be returned by delete_file()."""
        self.delete_file_result = result
    
    def file_exists(self, path: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            path: Path to the file, relative to the workspace directory
            
        Returns:
            True if the file exists, False otherwise
        """
        self.file_exists_called = True
        self.file_exists_args = (path,)
        self.file_exists_kwargs = {}
        
        # Check if file exists in mock filesystem
        exists = path in self.files
        
        return exists if exists else self.file_exists_result
    
    def set_file_exists_result(self, result: bool):
        """Set the result to be returned by file_exists()."""
        self.file_exists_result = result
    
    def search_files(self, pattern: str, path: str = ".", recursive: bool = True) -> str:
        """
        Search for files matching a pattern.
        
        Args:
            pattern: Search pattern (glob or regex)
            path: Path to search in, relative to the workspace directory
            recursive: Whether to search recursively
            
        Returns:
            Search results as a formatted string
        """
        self.search_files_called = True
        self.search_files_args = (pattern,)
        self.search_files_kwargs = {"path": path, "recursive": recursive}
        
        return self.search_files_result
    
    def set_search_files_result(self, result: str):
        """Set the result to be returned by search_files()."""
        self.search_files_result = result
    
    def close(self):
        """Close the MCP server process."""
        self.close_called = True
        self.process.poll.return_value = 0  # Process has exited
    
    def __enter__(self):
        """Enter the context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.close()


class MockMemoryMCP:
    """Mock implementation of the Memory MCP for testing."""
    
    def __init__(self, storage_path: Optional[str] = None, use_docker: bool = False):
        """
        Initialize the mock Memory MCP.
        
        Args:
            storage_path: Optional path to store memory data (has no effect in mock)
            use_docker: Whether to simulate using Docker (has no effect in mock)
        """
        self.storage_path = storage_path
        self.use_docker = use_docker
        self.process = MagicMock()
        self.process.poll.return_value = None  # Process is running
        
        # Mock memory storage
        self.memories = {
            "default": {
                "greeting": "Hello, world!",
                "user_name": "Test User",
                "last_query": "What is the capital of France?",
                "search_history": "Python programming, AI research, Machine learning tutorials"
            },
            "research": {
                "topic": "Artificial Intelligence",
                "sources": "https://example.com/ai, https://example.org/ml",
                "notes": "AI is a rapidly evolving field with applications in various domains."
            }
        }
        
        # Track method calls
        self.store_memory_called = False
        self.store_memory_args = None
        self.store_memory_kwargs = None
        self.store_memory_result = "Memory stored successfully"
        
        self.retrieve_memory_called = False
        self.retrieve_memory_args = None
        self.retrieve_memory_kwargs = None
        self.retrieve_memory_result = "Retrieved memory content"
        
        self.list_memories_called = False
        self.list_memories_args = None
        self.list_memories_kwargs = None
        self.list_memories_result = "greeting\nuser_name\nlast_query\nsearch_history"
        
        self.delete_memory_called = False
        self.delete_memory_args = None
        self.delete_memory_kwargs = None
        self.delete_memory_result = "Memory deleted successfully"
        
        self.search_memories_called = False
        self.search_memories_args = None
        self.search_memories_kwargs = None
        self.search_memories_result = "greeting: Hello, world!\nuser_name: Test User"
        
        self.clear_namespace_called = False
        self.clear_namespace_args = None
        self.clear_namespace_kwargs = None
        self.clear_namespace_result = "Namespace cleared successfully"
        
        self.list_tools_called = False
        self.list_tools_result = [
            {
                "name": "store_memory",
                "description": "Store a memory item"
            },
            {
                "name": "retrieve_memory",
                "description": "Retrieve a memory item by key"
            },
            {
                "name": "list_memories",
                "description": "List all memories in a namespace"
            },
            {
                "name": "delete_memory",
                "description": "Delete a memory item"
            },
            {
                "name": "search_memories",
                "description": "Search for memories by content"
            },
            {
                "name": "clear_namespace",
                "description": "Clear all memories in a namespace"
            }
        ]
        
        self.close_called = False
    
    def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock sending a request to the MCP server."""
        if method == "listTools":
            self.list_tools_called = True
            return {"tools": self.list_tools_result}
        
        if method == "callTool":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            
            if tool_name == "store_memory":
                self.store_memory_called = True
                self.store_memory_args = (
                    arguments.get("key", ""),
                    arguments.get("value", "")
                )
                self.store_memory_kwargs = {
                    "namespace": arguments.get("namespace", "default")
                }
                
                # Update mock memory storage
                key = arguments.get("key", "")
                value = arguments.get("value", "")
                namespace = arguments.get("namespace", "default")
                
                if namespace not in self.memories:
                    self.memories[namespace] = {}
                
                self.memories[namespace][key] = value
                
                return {
                    "content": [{"type": "text", "text": self.store_memory_result}],
                    "isError": False
                }
            
            if tool_name == "retrieve_memory":
                self.retrieve_memory_called = True
                self.retrieve_memory_args = (arguments.get("key", ""),)
                self.retrieve_memory_kwargs = {
                    "namespace": arguments.get("namespace", "default")
                }
                
                # Retrieve from mock memory storage
                key = arguments.get("key", "")
                namespace = arguments.get("namespace", "default")
                
                if namespace not in self.memories:
                    return {
                        "content": [{"type": "text", "text": f"Error: Namespace not found: {namespace}"}],
                        "isError": True
                    }
                
                if key not in self.memories[namespace]:
                    return {
                        "content": [{"type": "text", "text": f"Error: Key not found: {key}"}],
                        "isError": True
                    }
                
                value = self.memories[namespace][key]
                
                return {
                    "content": [{"type": "text", "text": value}],
                    "isError": False
                }
            
            if tool_name == "list_memories":
                self.list_memories_called = True
                self.list_memories_args = tuple()
                self.list_memories_kwargs = {
                    "namespace": arguments.get("namespace", "default")
                }
                
                # List from mock memory storage
                namespace = arguments.get("namespace", "default")
                
                if namespace not in self.memories:
                    return {
                        "content": [{"type": "text", "text": f"Error: Namespace not found: {namespace}"}],
                        "isError": True
                    }
                
                keys = list(self.memories[namespace].keys())
                result = "\n".join(keys)
                
                return {
                    "content": [{"type": "text", "text": result}],
                    "isError": False
                }
            
            if tool_name == "delete_memory":
                self.delete_memory_called = True
                self.delete_memory_args = (arguments.get("key", ""),)
                self.delete_memory_kwargs = {
                    "namespace": arguments.get("namespace", "default")
                }
                
                # Delete from mock memory storage
                key = arguments.get("key", "")
                namespace = arguments.get("namespace", "default")
                
                if namespace not in self.memories:
                    return {
                        "content": [{"type": "text", "text": f"Error: Namespace not found: {namespace}"}],
                        "isError": True
                    }
                
                if key not in self.memories[namespace]:
                    return {
                        "content": [{"type": "text", "text": f"Error: Key not found: {key}"}],
                        "isError": True
                    }
                
                del self.memories[namespace][key]
                
                return {
                    "content": [{"type": "text", "text": self.delete_memory_result}],
                    "isError": False
                }
            
            if tool_name == "search_memories":
                self.search_memories_called = True
                self.search_memories_args = (arguments.get("query", ""),)
                self.search_memories_kwargs = {
                    "namespace": arguments.get("namespace", "default")
                }
                
                # Search in mock memory storage
                query = arguments.get("query", "").lower()
                namespace = arguments.get("namespace", "default")
                
                if namespace not in self.memories:
                    return {
                        "content": [{"type": "text", "text": f"Error: Namespace not found: {namespace}"}],
                        "isError": True
                    }
                
                results = []
                for key, value in self.memories[namespace].items():
                    if query in key.lower() or query in str(value).lower():
                        results.append(f"{key}: {value}")
                
                result = "\n".join(results) if results else "No matching memories found."
                
                return {
                    "content": [{"type": "text", "text": result}],
                    "isError": False
                }
            
            if tool_name == "clear_namespace":
                self.clear_namespace_called = True
                self.clear_namespace_args = (arguments.get("namespace", ""),)
                self.clear_namespace_kwargs = {}
                
                # Clear namespace in mock memory storage
                namespace = arguments.get("namespace", "")
                
                if namespace not in self.memories:
                    return {
                        "content": [{"type": "text", "text": f"Error: Namespace not found: {namespace}"}],
                        "isError": True
                    }
                
                self.memories[namespace] = {}
                
                return {
                    "content": [{"type": "text", "text": self.clear_namespace_result}],
                    "isError": False
                }
        
        # Unknown method or tool
        return {
            "content": [{"type": "text", "text": f"Unknown method or tool: {method}"}],
            "isError": True
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools in the MCP server."""
        self.list_tools_called = True
        return self.list_tools_result
    
    def store_memory(self, key: str, value: str, namespace: str = "default") -> str:
        """
        Store a memory item.
        
        Args:
            key: The key to store the memory under
            value: The value to store
            namespace: Optional namespace to organize memories (default: "default")
            
        Returns:
            Confirmation message
        """
        self.store_memory_called = True
        self.store_memory_args = (key, value)
        self.store_memory_kwargs = {"namespace": namespace}
        
        # Validate inputs
        if not key:
            raise RuntimeError("Key cannot be empty")
        
        # Update mock memory storage
        if namespace not in self.memories:
            self.memories[namespace] = {}
        
        self.memories[namespace][key] = value
        
        return self.store_memory_result
    
    def set_store_memory_result(self, result: str):
        """Set the result to be returned by store_memory()."""
        self.store_memory_result = result
    
    def retrieve_memory(self, key: str, namespace: str = "default") -> str:
        """
        Retrieve a memory item by key.
        
        Args:
            key: The key of the memory to retrieve
            namespace: The namespace where the memory is stored (default: "default")
            
        Returns:
            The stored memory value
        """
        self.retrieve_memory_called = True
        self.retrieve_memory_args = (key,)
        self.retrieve_memory_kwargs = {"namespace": namespace}
        
        # Validate inputs
        if not key:
            raise RuntimeError("Key cannot be empty")
        
        # Retrieve from mock memory storage
        if namespace not in self.memories:
            raise RuntimeError(f"Namespace not found: {namespace}")
        
        if key not in self.memories[namespace]:
            raise RuntimeError(f"Key not found: {key}")
        
        return self.memories[namespace][key]
    
    def set_retrieve_memory_result(self, result: str):
        """Set the result to be returned by retrieve_memory()."""
        self.retrieve_memory_result = result
    
    def list_memories(self, namespace: str = "default") -> str:
        """
        List all memories in a namespace.
        
        Args:
            namespace: The namespace to list memories from (default: "default")
            
        Returns:
            List of memory keys as a formatted string
        """
        self.list_memories_called = True
        self.list_memories_args = tuple()
        self.list_memories_kwargs = {"namespace": namespace}
        
        # Validate inputs
        if namespace not in self.memories:
            raise RuntimeError(f"Namespace not found: {namespace}")
        
        # List from mock memory storage
        keys = list(self.memories[namespace].keys())
        result = "\n".join(keys)
        
        return result if keys else self.list_memories_result
    
    def set_list_memories_result(self, result: str):
        """Set the result to be returned by list_memories()."""
        self.list_memories_result = result
    
    def delete_memory(self, key: str, namespace: str = "default") -> str:
        """
        Delete a memory item.
        
        Args:
            key: The key of the memory to delete
            namespace: The namespace where the memory is stored (default: "default")
            
        Returns:
            Confirmation message
        """
        self.delete_memory_called = True
        self.delete_memory_args = (key,)
        self.delete_memory_kwargs = {"namespace": namespace}
        
        # Validate inputs
        if not key:
            raise RuntimeError("Key cannot be empty")
        
        # Delete from mock memory storage
        if namespace not in self.memories:
            raise RuntimeError(f"Namespace not found: {namespace}")
        
        if key not in self.memories[namespace]:
            raise RuntimeError(f"Key not found: {key}")
        
        del self.memories[namespace][key]
        
        return self.delete_memory_result
    
    def set_delete_memory_result(self, result: str):
        """Set the result to be returned by delete_memory()."""
        self.delete_memory_result = result
    
    def search_memories(self, query: str, namespace: str = "default") -> str:
        """
        Search for memories by content.
        
        Args:
            query: The search query
            namespace: The namespace to search in (default: "default")
            
        Returns:
            Search results as a formatted string
        """
        self.search_memories_called = True
        self.search_memories_args = (query,)
        self.search_memories_kwargs = {"namespace": namespace}
        
        # Validate inputs
        if not query:
            raise RuntimeError("Query cannot be empty")
        
        if namespace not in self.memories:
            raise RuntimeError(f"Namespace not found: {namespace}")
        
        # Search in mock memory storage
        query = query.lower()
        results = []
        for key, value in self.memories[namespace].items():
            if query in key.lower() or query in str(value).lower():
                results.append(f"{key}: {value}")
        
        result = "\n".join(results) if results else "No matching memories found."
        
        return result if results else self.search_memories_result
    
    def set_search_memories_result(self, result: str):
        """Set the result to be returned by search_memories()."""
        self.search_memories_result = result
    
    def clear_namespace(self, namespace: str) -> str:
        """
        Clear all memories in a namespace.
        
        Args:
            namespace: The namespace to clear
            
        Returns:
            Confirmation message
        """
        self.clear_namespace_called = True
        self.clear_namespace_args = (namespace,)
        self.clear_namespace_kwargs = {}
        
        # Validate inputs
        if not namespace:
            raise RuntimeError("Namespace cannot be empty")
        
        if namespace not in self.memories:
            raise RuntimeError(f"Namespace not found: {namespace}")
        
        # Clear namespace in mock memory storage
        self.memories[namespace] = {}
        
        return self.clear_namespace_result
    
    def set_clear_namespace_result(self, result: str):
        """Set the result to be returned by clear_namespace()."""
        self.clear_namespace_result = result
    
    def close(self):
        """Close the MCP server process."""
        self.close_called = True
        self.process.poll.return_value = 0  # Process has exited
    
    def __enter__(self):
        """Enter the context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.close()


# Factory class for creating mock MCP instances
class MockMCPFactory:
    """Factory for creating mock MCP instances."""
    
    @staticmethod
    def create_brave_search_mcp(api_key: Optional[str] = "mock_api_key", use_docker: bool = False) -> MockBraveSearchMCP:
        """Create a mock BraveSearchMCP instance."""
        return MockBraveSearchMCP(api_key=api_key, use_docker=use_docker)
    
    @staticmethod
    def create_everart_mcp(api_key: Optional[str] = "mock_api_key", use_docker: bool = False) -> MockEverArtMCP:
        """Create a mock EverArtMCP instance."""
        return MockEverArtMCP(api_key=api_key, use_docker=use_docker)
    
    @staticmethod
    def create_fetch_mcp(use_docker: bool = False) -> MockFetchMCP:
        """Create a mock FetchMCP instance."""
        return MockFetchMCP(use_docker=use_docker)
    
    @staticmethod
    def create_filesystem_mcp(workspace_dir: Optional[str] = "/mock/workspace", use_docker: bool = False) -> MockFilesystemMCP:
        """Create a mock FilesystemMCP instance."""
        return MockFilesystemMCP(workspace_dir=workspace_dir, use_docker=use_docker)
    
    @staticmethod
    def create_memory_mcp(storage_path: Optional[str] = None, use_docker: bool = False) -> MockMemoryMCP:
        """Create a mock MemoryMCP instance."""
        return MockMemoryMCP(storage_path=storage_path, use_docker=use_docker)


# Patch functions for testing
def patch_mcps():
    """Patch all MCP classes with their mock counterparts."""
    patches = [
        patch('apps.mcps.brave_search_mcp.BraveSearchMCP', MockBraveSearchMCP),
        patch('apps.mcps.everart_mcp.EverArtMCP', MockEverArtMCP),
        patch('apps.mcps.fetch_mcp.FetchMCP', MockFetchMCP),
        patch('apps.mcps.filesystem_mcp.FilesystemMCP', MockFilesystemMCP),
        patch('apps.mcps.memory_mcp.MemoryMCP', MockMemoryMCP)
    ]
    
    for p in patches:
        p.start()
    
    return patches

def stop_patches(patches):
    """Stop all patches."""
    for p in patches:
        p.stop()


# If this module is run directly, perform a simple test
if __name__ == "__main__":
    # Create mock MCPs
    brave_search = MockBraveSearchMCP()
    everart = MockEverArtMCP()
    fetch = MockFetchMCP()
    filesystem = MockFilesystemMCP()
    memory = MockMemoryMCP()
    
    # Test BraveSearchMCP
    result = brave_search.web_search("Python programming")
    print(f"BraveSearchMCP web_search result: {result[:100]}...")
    
    # Test EverArtMCP
    result = everart.generate_image("A beautiful sunset over mountains")
    print(f"EverArtMCP generate_image result: {result[:100]}...")
    
    # Test FetchMCP
    result = fetch.fetch_url("https://example.com")
    print(f"FetchMCP fetch_url result: {result[:100]}...")
    
    # Test FilesystemMCP
    result = filesystem.list_directory()
    print(f"FilesystemMCP list_directory result: {result[:100]}...")
    
    # Test MemoryMCP
    memory.store_memory("test_key", "test_value")
    result = memory.retrieve_memory("test_key")
    print(f"MemoryMCP retrieve_memory result: {result}")
    
    # Test patching
    patches = patch_mcps()
    
    # Create instances of the patched classes
    from apps.mcps.brave_search_mcp import BraveSearchMCP
    from apps.mcps.everart_mcp import EverArtMCP
    from apps.mcps.fetch_mcp import FetchMCP
    from apps.mcps.filesystem_mcp import FilesystemMCP
    from apps.mcps.memory_mcp import MemoryMCP
    
    patched_brave = BraveSearchMCP()
    patched_everart = EverArtMCP()
    patched_fetch = FetchMCP()
    patched_filesystem = FilesystemMCP()
    patched_memory = MemoryMCP()
    
    # Verify that the patched instances are actually our mocks
    print(f"Patched BraveSearchMCP is a mock: {isinstance(patched_brave, MockBraveSearchMCP)}")
    print(f"Patched EverArtMCP is a mock: {isinstance(patched_everart, MockEverArtMCP)}")
    print(f"Patched FetchMCP is a mock: {isinstance(patched_fetch, MockFetchMCP)}")
    print(f"Patched FilesystemMCP is a mock: {isinstance(patched_filesystem, MockFilesystemMCP)}")
    print(f"Patched MemoryMCP is a mock: {isinstance(patched_memory, MockMemoryMCP)}")
    
    # Stop patches
    stop_patches(patches)
