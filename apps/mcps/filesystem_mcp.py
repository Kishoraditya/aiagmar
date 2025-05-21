#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import uuid
from typing import Dict, List, Optional, Union, Any, BinaryIO

class FilesystemMCP:
    """Python wrapper for the Filesystem MCP server for file operations."""
    
    def __init__(self, workspace_dir: Optional[str] = None, use_docker: bool = False):
        """
        Initialize the Filesystem MCP client.
        
        Args:
            workspace_dir: Optional directory to use as the workspace root.
                           If not provided, the current working directory is used.
            use_docker: Whether to use Docker or NPX to run the server.
        """
        self.workspace_dir = workspace_dir or os.getcwd()
        self.process = None
        self.use_docker = use_docker
        self._start_server()
    
    def _start_server(self):
        """Start the Filesystem MCP server as a subprocess."""
        env = os.environ.copy()
        
        if self.use_docker:
            cmd = [
                "docker", "run", "-i", "--rm",
                "-v", f"{self.workspace_dir}:/workspace",
                "-e", "WORKSPACE_DIR=/workspace",
                "mcp/filesystem"
            ]
        else:
            env["WORKSPACE_DIR"] = self.workspace_dir
            cmd = ["npx", "-y", "@modelcontextprotocol/server-filesystem"]
        
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1  # Line buffered
        )
    
    def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a request to the MCP server and get the response.
        
        Args:
            method: The MCP method name
            params: The parameters for the method
            
        Returns:
            The response from the server
        """
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("MCP server is not running")
        
        request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params
        }
        
        # Send the request
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str)
        self.process.stdin.flush()
        
        # Read the response
        response_str = self.process.stdout.readline()
        response = json.loads(response_str)
        
        if "error" in response:
            raise RuntimeError(f"MCP server error: {response['error']}")
        
        return response["result"]
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools in the MCP server."""
        response = self._send_request("listTools", {})
        return response["tools"]
    
    def read_file(self, path: str) -> str:
        """
        Read the contents of a file.
        
        Args:
            path: Path to the file, relative to the workspace directory
            
        Returns:
            The file contents as a string
        """
        response = self._send_request("callTool", {
            "name": "read_file",
            "arguments": {
                "path": path
            }
        })
        
        if response["isError"]:
            raise RuntimeError(f"Read file error: {response['content'][0]['text']}")
        
        return response["content"][0]["text"]
    
    def write_file(self, path: str, content: str) -> str:
        """
        Write content to a file.
        
        Args:
            path: Path to the file, relative to the workspace directory
            content: Content to write to the file
            
        Returns:
            Confirmation message
        """
        response = self._send_request("callTool", {
            "name": "write_file",
            "arguments": {
                "path": path,
                "content": content
            }
        })
        
        if response["isError"]:
            raise RuntimeError(f"Write file error: {response['content'][0]['text']}")
        
        return response["content"][0]["text"]
    
    def list_directory(self, path: str = ".", recursive: bool = False) -> str:
        """
        List files and directories in a directory.
        
        Args:
            path: Path to the directory, relative to the workspace directory
            recursive: Whether to list files recursively
            
        Returns:
            Directory listing as a formatted string
        """
        arguments = {
            "path": path
        }
        
        if recursive:
            arguments["recursive"] = True
        
        response = self._send_request("callTool", {
            "name": "list_directory",
            "arguments": arguments
        })
        
        if response["isError"]:
            raise RuntimeError(f"List directory error: {response['content'][0]['text']}")
        
        return response["content"][0]["text"]
    
    def create_directory(self, path: str) -> str:
        """
        Create a directory.
        
        Args:
            path: Path to the directory, relative to the workspace directory
            
        Returns:
            Confirmation message
        """
        response = self._send_request("callTool", {
            "name": "create_directory",
            "arguments": {
                "path": path
            }
        })
        
        if response["isError"]:
            raise RuntimeError(f"Create directory error: {response['content'][0]['text']}")
        
        return response["content"][0]["text"]
    
    def delete_file(self, path: str) -> str:
        """
        Delete a file.
        
        Args:
            path: Path to the file, relative to the workspace directory
            
        Returns:
            Confirmation message
        """
        response = self._send_request("callTool", {
            "name": "delete_file",
            "arguments": {
                "path": path
            }
        })
        
        if response["isError"]:
            raise RuntimeError(f"Delete file error: {response['content'][0]['text']}")
        
        return response["content"][0]["text"]
    
    def file_exists(self, path: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            path: Path to the file, relative to the workspace directory
            
        Returns:
            True if the file exists, False otherwise
        """
        response = self._send_request("callTool", {
            "name": "file_exists",
            "arguments": {
                "path": path
            }
        })
        
        if response["isError"]:
            raise RuntimeError(f"File exists error: {response['content'][0]['text']}")
        
        # Parse the response to determine if the file exists
        return "exists" in response["content"][0]["text"].lower() and "true" in response["content"][0]["text"].lower()
    
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
        arguments = {
            "pattern": pattern,
            "path": path,
            "recursive": recursive
        }
        
        response = self._send_request("callTool", {
            "name": "search_files",
            "arguments": arguments
        })
        
        if response["isError"]:
            raise RuntimeError(f"Search files error: {response['content'][0]['text']}")
        
        return response["content"][0]["text"]
    
    def close(self):
        """Close the MCP server process."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage
if __name__ == "__main__":
    # Get workspace directory from command line or use current directory
    workspace_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    
    with FilesystemMCP(workspace_dir=workspace_dir) as fs:
        # List available tools
        print("Available tools:")
        tools = fs.list_tools()
        for tool in tools:
            print(f"- {tool['name']}: {tool['description'][:100]}...")
        
        # Example list directory
        print("\nList directory example:")
        listing = fs.list_directory(".")
        print(listing)
        
        # Example write and read file
        test_file = "test_file.txt"
        print(f"\nWriting to {test_file}:")
        fs.write_file(test_file, "Hello, world!")
        print(f"Reading from {test_file}:")
        content = fs.read_file(test_file)
        print(content)
        
        # Clean up
        print(f"\nDeleting {test_file}:")
        fs.delete_file(test_file)
