#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import uuid
from typing import Dict, List, Optional, Union, Any

class EverArtMCP:
    """Python wrapper for the EverArt MCP server."""
    
    def __init__(self, api_key: Optional[str] = None, use_docker: bool = False):
        """
        Initialize the EverArt MCP client.
        
        Args:
            api_key: EverArt API key. If not provided, will try to use EVERART_API_KEY env var.
            use_docker: Whether to use Docker or NPX to run the server.
        """
        self.api_key = api_key or os.environ.get("EVERART_API_KEY")
        if not self.api_key:
            raise ValueError("EverArt API key is required. Provide it as an argument or set EVERART_API_KEY environment variable.")
        
        self.process = None
        self.use_docker = use_docker
        self._start_server()
    
    def _start_server(self):
        """Start the EverArt MCP server as a subprocess."""
        env = os.environ.copy()
        env["EVERART_API_KEY"] = self.api_key
        
        if self.use_docker:
            cmd = [
                "docker", "run", "-i", "--rm",
                "-e", f"EVERART_API_KEY={self.api_key}",
                "mcp/everart"
            ]
        else:
            cmd = ["npx", "-y", "@modelcontextprotocol/server-everart"]
        
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
        arguments = {
            "prompt": prompt,
            "num_images": num_images,
            "aspect_ratio": aspect_ratio
        }
        
        if style:
            arguments["style"] = style
        
        response = self._send_request("callTool", {
            "name": "everart_generate_image",
            "arguments": arguments
        })
        
        if response["isError"]:
            raise RuntimeError(f"Image generation error: {response['content'][0]['text']}")
        
        return response["content"][0]["text"]
    
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
        response = self._send_request("callTool", {
            "name": "everart_enhance_image",
            "arguments": {
                "image_url": image_url,
                "prompt": prompt,
                "strength": strength
            }
        })
        
        if response["isError"]:
            raise RuntimeError(f"Image enhancement error: {response['content'][0]['text']}")
        
        return response["content"][0]["text"]
    
    def describe_image(self, image_url: str, detail_level: str = "medium") -> str:
        """
        Generate a detailed description of an image using EverArt AI.
        
        Args:
            image_url: URL of the image to describe
            detail_level: Level of detail in the description ("low", "medium", "high")
            
        Returns:
            Image description as a formatted string
        """
        response = self._send_request("callTool", {
            "name": "everart_describe_image",
            "arguments": {
                "image_url": image_url,
                "detail_level": detail_level
            }
        })
        
        if response["isError"]:
            raise RuntimeError(f"Image description error: {response['content'][0]['text']}")
        
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
    api_key = os.environ.get("EVERART_API_KEY") or (sys.argv[1] if len(sys.argv) > 1 else None)
    
    with EverArtMCP(api_key=api_key) as everart:
        # List available tools
        print("Available tools:")
        tools = everart.list_tools()
        for tool in tools:
            print(f"- {tool['name']}: {tool['description'][:100]}...")
        
        # Example image generation
        print("\nImage generation example:")
        results = everart.generate_image(
            "A serene mountain landscape with a lake at sunset",
            style="oil painting",
            aspect_ratio="16:9"
        )
        print(results)
        
        # Example image description (assuming we have an image URL)
        if len(sys.argv) > 2:
            image_url = sys.argv[2]
            print("\nImage description example:")
            description = everart.describe_image(image_url)
            print(description)
