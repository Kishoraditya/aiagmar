#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import uuid
from typing import Dict, List, Optional, Union, Any

class BraveSearchMCP:
    """Python wrapper for the Brave Search MCP server."""
    
    def __init__(self, api_key: Optional[str] = None, use_docker: bool = False):
        """
        Initialize the Brave Search MCP client.
        
        Args:
            api_key: Brave Search API key. If not provided, will try to use BRAVE_API_KEY env var.
            use_docker: Whether to use Docker or NPX to run the server.
        """
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")
        if not self.api_key:
            raise ValueError("Brave API key is required. Provide it as an argument or set BRAVE_API_KEY environment variable.")
        
        self.process = None
        self.use_docker = use_docker
        self._start_server()
    
    def _start_server(self):
        """Start the Brave Search MCP server as a subprocess."""
        env = os.environ.copy()
        env["BRAVE_API_KEY"] = self.api_key
        
        if self.use_docker:
            cmd = [
                "docker", "run", "-i", "--rm",
                "-e", f"BRAVE_API_KEY={self.api_key}",
                "mcp/brave-search"
            ]
        else:
            cmd = ["npx", "-y", "@modelcontextprotocol/server-brave-search"]
        
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
    
    def web_search(self, query: str, count: int = 10, offset: int = 0) -> str:
        """
        Perform a web search using Brave Search.
        
        Args:
            query: Search query (max 400 chars, 50 words)
            count: Number of results (1-20, default 10)
            offset: Pagination offset (max 9, default 0)
            
        Returns:
            Search results as a formatted string
        """
        response = self._send_request("callTool", {
            "name": "brave_web_search",
            "arguments": {
                "query": query,
                "count": count,
                "offset": offset
            }
        })
        
        if response["isError"]:
            raise RuntimeError(f"Web search error: {response['content'][0]['text']}")
        
        return response["content"][0]["text"]
    
    def local_search(self, query: str, count: int = 5) -> str:
        """
        Search for local businesses and places using Brave's Local Search API.
        
        Args:
            query: Local search query (e.g. 'pizza near Central Park')
            count: Number of results (1-20, default 5)
            
        Returns:
            Local search results as a formatted string
        """
        response = self._send_request("callTool", {
            "name": "brave_local_search",
            "arguments": {
                "query": query,
                "count": count
            }
        })
        
        if response["isError"]:
            raise RuntimeError(f"Local search error: {response['content'][0]['text']}")
        
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
    # Get API key from environment or command line
    api_key = os.environ.get("BRAVE_API_KEY") or (sys.argv[1] if len(sys.argv) > 1 else None)
    
    with BraveSearchMCP(api_key=api_key) as brave:
        # List available tools
        print("Available tools:")
        tools = brave.list_tools()
        for tool in tools:
            print(f"- {tool['name']}: {tool['description'][:100]}...")
        
        # Example web search
        print("\nWeb search example:")
        results = brave.web_search("Python programming language", count=3)
        print(results)
        
        # Example local search
        print("\nLocal search example:")
        results = brave.local_search("coffee shops in San Francisco", count=2)
        print(results)
