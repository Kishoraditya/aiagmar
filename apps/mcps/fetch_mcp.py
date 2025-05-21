#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import uuid
from typing import Dict, List, Optional, Union, Any

class FetchMCP:
    """Python wrapper for the Fetch MCP server for retrieving web content."""
    
    def __init__(self, use_docker: bool = False):
        """
        Initialize the Fetch MCP client.
        
        Args:
            use_docker: Whether to use Docker or NPX to run the server.
        """
        self.process = None
        self.use_docker = use_docker
        self._start_server()
    
    def _start_server(self):
        """Start the Fetch MCP server as a subprocess."""
        env = os.environ.copy()
        
        if self.use_docker:
            cmd = [
                "docker", "run", "-i", "--rm",
                "mcp/fetch"
            ]
        else:
            cmd = ["npx", "-y", "@modelcontextprotocol/server-fetch"]
        
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
        arguments = {
            "url": url,
            "timeout": timeout
        }
        
        if selector:
            arguments["selector"] = selector
            
        if wait_for:
            arguments["waitFor"] = wait_for
        
        response = self._send_request("callTool", {
            "name": "fetch_url",
            "arguments": arguments
        })
        
        if response["isError"]:
            raise RuntimeError(f"Fetch error: {response['content'][0]['text']}")
        
        return response["content"][0]["text"]
    
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
        arguments = {
            "url": url,
            "timeout": timeout
        }
            
        if wait_for:
            arguments["waitFor"] = wait_for
        
        response = self._send_request("callTool", {
            "name": "fetch_html",
            "arguments": arguments
        })
        
        if response["isError"]:
            raise RuntimeError(f"Fetch HTML error: {response['content'][0]['text']}")
        
        return response["content"][0]["text"]
    
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
        arguments = {
            "url": url,
            "timeout": timeout
        }
            
        if wait_for:
            arguments["waitFor"] = wait_for
        
        response = self._send_request("callTool", {
            "name": "fetch_text",
            "arguments": arguments
        })
        
        if response["isError"]:
            raise RuntimeError(f"Fetch text error: {response['content'][0]['text']}")
        
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
    # Get URL from command line
    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    
    with FetchMCP() as fetch:
        # List available tools
        print("Available tools:")
        tools = fetch.list_tools()
        for tool in tools:
            print(f"- {tool['name']}: {tool['description'][:100]}...")
        
        # Example fetch URL
        print("\nFetch URL example:")
        content = fetch.fetch_url(url)
        print(f"Content length: {len(content)} characters")
        print(content[:500] + "..." if len(content) > 500 else content)
        
        # Example fetch text
        print("\nFetch text example:")
        text = fetch.fetch_text(url)
        print(f"Text length: {len(text)} characters")
        print(text[:500] + "..." if len(text) > 500 else text)
