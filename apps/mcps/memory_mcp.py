#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import uuid
from typing import Dict, List, Optional, Union, Any

class MemoryMCP:
    """Python wrapper for the Memory MCP server for storing and retrieving information."""
    
    def __init__(self, storage_path: Optional[str] = None, use_docker: bool = False):
        """
        Initialize the Memory MCP client.
        
        Args:
            storage_path: Optional path to store memory data.
                         If not provided, an in-memory store will be used.
            use_docker: Whether to use Docker or NPX to run the server.
        """
        self.storage_path = storage_path
        self.process = None
        self.use_docker = use_docker
        self._start_server()
    
    def _start_server(self):
        """Start the Memory MCP server as a subprocess."""
        env = os.environ.copy()
        
        if self.use_docker:
            cmd = ["docker", "run", "-i", "--rm"]
            
            if self.storage_path:
                # Mount the storage path if provided
                cmd.extend(["-v", f"{self.storage_path}:/data"])
                cmd.extend(["-e", "MEMORY_STORAGE_PATH=/data"])
            
            cmd.append("mcp/memory")
        else:
            if self.storage_path:
                env["MEMORY_STORAGE_PATH"] = self.storage_path
            
            cmd = ["npx", "-y", "@modelcontextprotocol/server-memory"]
        
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
        response = self._send_request("callTool", {
            "name": "store_memory",
            "arguments": {
                "key": key,
                "value": value,
                "namespace": namespace
            }
        })
        
        if response["isError"]:
            raise RuntimeError(f"Store memory error: {response['content'][0]['text']}")
        
        return response["content"][0]["text"]
    
    def retrieve_memory(self, key: str, namespace: str = "default") -> str:
        """
        Retrieve a memory item by key.
        
        Args:
            key: The key of the memory to retrieve
            namespace: The namespace where the memory is stored (default: "default")
            
        Returns:
            The stored memory value
        """
        response = self._send_request("callTool", {
            "name": "retrieve_memory",
            "arguments": {
                "key": key,
                "namespace": namespace
            }
        })
        
        if response["isError"]:
            raise RuntimeError(f"Retrieve memory error: {response['content'][0]['text']}")
        
        return response["content"][0]["text"]
    
    def list_memories(self, namespace: str = "default") -> str:
        """
        List all memories in a namespace.
        
        Args:
            namespace: The namespace to list memories from (default: "default")
            
        Returns:
            List of memory keys as a formatted string
        """
        response = self._send_request("callTool", {
            "name": "list_memories",
            "arguments": {
                "namespace": namespace
            }
        })
        
        if response["isError"]:
            raise RuntimeError(f"List memories error: {response['content'][0]['text']}")
        
        return response["content"][0]["text"]
    
    def delete_memory(self, key: str, namespace: str = "default") -> str:
        """
        Delete a memory item.
        
        Args:
            key: The key of the memory to delete
            namespace: The namespace where the memory is stored (default: "default")
            
        Returns:
            Confirmation message
        """
        response = self._send_request("callTool", {
            "name": "delete_memory",
            "arguments": {
                "key": key,
                "namespace": namespace
            }
        })
        
        if response["isError"]:
            raise RuntimeError(f"Delete memory error: {response['content'][0]['text']}")
        
        return response["content"][0]["text"]
    
    def search_memories(self, query: str, namespace: str = "default") -> str:
        """
        Search for memories by content.
        
        Args:
            query: The search query
            namespace: The namespace to search in (default: "default")
            
        Returns:
            Search results as a formatted string
        """
        response = self._send_request("callTool", {
            "name": "search_memories",
            "arguments": {
                "query": query,
                "namespace": namespace
            }
        })
        
        if response["isError"]:
            raise RuntimeError(f"Search memories error: {response['content'][0]['text']}")
        
        return response["content"][0]["text"]
    
    def clear_namespace(self, namespace: str) -> str:
        """
        Clear all memories in a namespace.
        
        Args:
            namespace: The namespace to clear
            
        Returns:
            Confirmation message
        """
        response = self._send_request("callTool", {
            "name": "clear_namespace",
            "arguments": {
                "namespace": namespace
            }
        })
        
        if response["isError"]:
            raise RuntimeError(f"Clear namespace error: {response['content'][0]['text']}")
        
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
    # Get storage path from command line (optional)
    storage_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    with MemoryMCP(storage_path=storage_path) as memory:
        # List available tools
        print("Available tools:")
        tools = memory.list_tools()
        for tool in tools:
            print(f"- {tool['name']}: {tool['description'][:100]}...")
        
        # Example store and retrieve
        print("\nStoring memory:")
        memory.store_memory("greeting", "Hello, world!")
        print("Memory stored.")
        
        print("\nRetrieving memory:")
        value = memory.retrieve_memory("greeting")
        print(f"Retrieved value: {value}")
        
        # Example list memories
        print("\nListing all memories:")
        memories = memory.list_memories()
        print(memories)
        
        # Example search
        print("\nSearching memories:")
        results = memory.search_memories("Hello")
        print(results)
        
        # Example delete
        print("\nDeleting memory:")
        memory.delete_memory("greeting")
        print("Memory deleted.")
