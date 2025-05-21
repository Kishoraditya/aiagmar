"""
Unit tests for the EverArtMCP class.
"""

import os
import json
import pytest
import subprocess
from unittest.mock import patch, MagicMock, call
from apps.mcps.everart_mcp import EverArtMCP
from apps.utils.exceptions import ValidationError

# Test API key for testing
TEST_API_KEY = "test_everart_api_key"

class TestEverArtMCP:
    """Test suite for EverArtMCP class."""

    @pytest.fixture
    def mock_process(self):
        """Fixture to create a mock subprocess."""
        mock = MagicMock()
        mock.poll.return_value = None  # Process is running
        mock.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "tools": [
                    {
                        "name": "everart_generate_image",
                        "description": "Generates images using EverArt AI"
                    },
                    {
                        "name": "everart_enhance_image",
                        "description": "Enhances or modifies an existing image"
                    },
                    {
                        "name": "everart_describe_image",
                        "description": "Generates a detailed description of an image"
                    }
                ]
            }
        })
        return mock

    @pytest.fixture
    def everart_mcp(self, monkeypatch):
        """Fixture to create an EverArtMCP instance with mocked subprocess."""
        # Set environment variable for API key
        monkeypatch.setenv("EVERART_API_KEY", TEST_API_KEY)
        
        # Mock the subprocess.Popen to avoid actual process creation
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Process is running
            mock_popen.return_value = mock_process
            
            # Create the EverArtMCP instance
            mcp = EverArtMCP()
            
            # Replace the process with our controlled mock
            mcp.process = mock_process
            
            yield mcp
            
            # Clean up
            mcp.close()

    def test_init_with_api_key_param(self):
        """Test initialization with API key as parameter."""
        with patch("subprocess.Popen") as mock_popen:
            mcp = EverArtMCP(api_key=TEST_API_KEY)
            assert mcp.api_key == TEST_API_KEY
            mock_popen.assert_called_once()

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization with API key from environment variable."""
        monkeypatch.setenv("EVERART_API_KEY", TEST_API_KEY)
        
        with patch("subprocess.Popen") as mock_popen:
            mcp = EverArtMCP()
            assert mcp.api_key == TEST_API_KEY
            mock_popen.assert_called_once()

    def test_init_no_api_key(self, monkeypatch):
        """Test initialization with no API key raises error."""
        # Ensure environment variable is not set
        monkeypatch.delenv("EVERART_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="EverArt API key is required"):
            EverArtMCP()

    def test_init_with_docker(self):
        """Test initialization with Docker option."""
        with patch("subprocess.Popen") as mock_popen:
            mcp = EverArtMCP(api_key=TEST_API_KEY, use_docker=True)
            
            # Check that Docker command was used
            args, kwargs = mock_popen.call_args
            cmd = args[0]
            assert "docker" in cmd
            assert "run" in cmd
            assert "mcp/everart" in cmd

    def test_start_server_npx(self):
        """Test starting server with NPX."""
        with patch("subprocess.Popen") as mock_popen:
            mcp = EverArtMCP(api_key=TEST_API_KEY, use_docker=False)
            
            # Check that NPX command was used
            args, kwargs = mock_popen.call_args
            cmd = args[0]
            assert "npx" in cmd
            assert "@modelcontextprotocol/server-everart" in cmd

    def test_send_request_success(self, everart_mcp, mock_process):
        """Test sending a request successfully."""
        # Setup mock response
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {"test": "data"}
        })
        everart_mcp.process = mock_process
        
        # Send request
        result = everart_mcp._send_request("test_method", {"param": "value"})
        
        # Verify result
        assert result == {"test": "data"}
        
        # Verify request was sent correctly
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "test_method" in written_data
        assert "param" in written_data
        assert "value" in written_data

    def test_send_request_error(self, everart_mcp, mock_process):
        """Test sending a request that returns an error."""
        # Setup mock error response
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "error": {"code": -32000, "message": "Test error"}
        })
        everart_mcp.process = mock_process
        
        # Send request and expect error
        with pytest.raises(RuntimeError, match="MCP server error"):
            everart_mcp._send_request("test_method", {"param": "value"})

    def test_send_request_process_not_running(self, everart_mcp):
        """Test sending a request when process is not running."""
        # Set process to None to simulate not running
        everart_mcp.process = None
        
        # Send request and expect error
        with pytest.raises(RuntimeError, match="MCP server is not running"):
            everart_mcp._send_request("test_method", {"param": "value"})

    def test_list_tools(self, everart_mcp, mock_process):
        """Test listing available tools."""
        # Setup mock response
        tools_response = {
            "tools": [
                {
                    "name": "everart_generate_image",
                    "description": "Generates images using EverArt AI"
                },
                {
                    "name": "everart_enhance_image",
                    "description": "Enhances or modifies an existing image"
                },
                {
                    "name": "everart_describe_image",
                    "description": "Generates a detailed description of an image"
                }
            ]
        }
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": tools_response
        })
        everart_mcp.process = mock_process
        
        # Get tools
        tools = everart_mcp.list_tools()
        
        # Verify tools
        assert len(tools) == 3
        assert tools[0]["name"] == "everart_generate_image"
        assert tools[1]["name"] == "everart_enhance_image"
        assert tools[2]["name"] == "everart_describe_image"

    def test_generate_image(self, everart_mcp, mock_process):
        """Test image generation functionality."""
        # Setup mock response for successful image generation
        image_result = "Generated image URL: https://everart.ai/images/test123.jpg"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": image_result}],
                "isError": False
            }
        })
        everart_mcp.process = mock_process
        
        # Generate image
        result = everart_mcp.generate_image(
            prompt="a beautiful sunset over mountains",
            style="oil painting",
            aspect_ratio="16:9",
            num_images=2
        )
        
        # Verify result
        assert result == image_result
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "everart_generate_image" in written_data
        assert "a beautiful sunset over mountains" in written_data
        assert "oil painting" in written_data
        assert "16:9" in written_data
        assert "2" in written_data  # num_images

    def test_generate_image_error(self, everart_mcp, mock_process):
        """Test image generation with error response."""
        # Setup mock response for failed image generation
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Image generation failed: Invalid prompt"}],
                "isError": True
            }
        })
        everart_mcp.process = mock_process
        
        # Generate image and expect error
        with pytest.raises(RuntimeError, match="Image generation error"):
            everart_mcp.generate_image("invalid prompt")

    def test_enhance_image(self, everart_mcp, mock_process):
        """Test image enhancement functionality."""
        # Setup mock response for successful image enhancement
        enhance_result = "Enhanced image URL: https://everart.ai/images/enhanced123.jpg"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": enhance_result}],
                "isError": False
            }
        })
        everart_mcp.process = mock_process
        
        # Enhance image
        result = everart_mcp.enhance_image(
            image_url="https://example.com/original.jpg",
            prompt="make it more vibrant",
            strength=0.7
        )
        
        # Verify result
        assert result == enhance_result
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "everart_enhance_image" in written_data
        assert "https://example.com/original.jpg" in written_data
        assert "make it more vibrant" in written_data
        assert "0.7" in written_data  # strength

    def test_enhance_image_error(self, everart_mcp, mock_process):
        """Test image enhancement with error response."""
        # Setup mock response for failed image enhancement
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Image enhancement failed: Invalid image URL"}],
                "isError": True
            }
        })
        everart_mcp.process = mock_process
        
        # Enhance image and expect error
        with pytest.raises(RuntimeError, match="Image enhancement error"):
            everart_mcp.enhance_image(
                image_url="invalid-url",
                prompt="make it better"
            )

    def test_describe_image(self, everart_mcp, mock_process):
        """Test image description functionality."""
        # Setup mock response for successful image description
        description_result = "The image shows a serene mountain landscape at sunset with vibrant orange and purple hues."
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": description_result}],
                "isError": False
            }
        })
        everart_mcp.process = mock_process
        
        # Describe image
        result = everart_mcp.describe_image(
            image_url="https://example.com/image.jpg",
            detail_level="high"
        )
        
        # Verify result
        assert result == description_result
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "everart_describe_image" in written_data
        assert "https://example.com/image.jpg" in written_data
        assert "high" in written_data  # detail_level

    def test_describe_image_error(self, everart_mcp, mock_process):
        """Test image description with error response."""
        # Setup mock response for failed image description
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Image description failed: Could not access image"}],
                "isError": True
            }
        })
        everart_mcp.process = mock_process
        
        # Describe image and expect error
        with pytest.raises(RuntimeError, match="Image description error"):
            everart_mcp.describe_image(
                image_url="https://example.com/nonexistent.jpg"
            )

    def test_close(self):
        """Test closing the MCP server process."""
        with patch("subprocess.Popen") as mock_popen:
            # Create a mock process
            mock_process = MagicMock()
            mock_popen.return_value = mock_process
            
            # Create MCP and then close it
            mcp = EverArtMCP(api_key=TEST_API_KEY)
            mcp.close()
            
            # Verify process was terminated
            mock_process.terminate.assert_called_once()

    def test_close_process_not_running(self):
        """Test closing when process is not running."""
        with patch("subprocess.Popen") as mock_popen:
            # Create a mock process that's not running
            mock_process = MagicMock()
            mock_process.poll.return_value = 0  # Process has exited
            mock_popen.return_value = mock_process
            
            # Create MCP and then close it
            mcp = EverArtMCP(api_key=TEST_API_KEY)
            mcp.close()
            
            # Verify terminate was not called
            mock_process.terminate.assert_not_called()

    def test_context_manager(self):
        """Test using EverArtMCP as a context manager."""
        with patch("subprocess.Popen") as mock_popen:
            # Create a mock process
            mock_process = MagicMock()
            mock_popen.return_value = mock_process
            
            # Use as context manager
            with EverArtMCP(api_key=TEST_API_KEY) as mcp:
                pass
            
            # Verify process was terminated on exit
            mock_process.terminate.assert_called_once()

    def test_process_timeout_on_close(self):
        """Test handling timeout when closing process."""
        with patch("subprocess.Popen") as mock_popen:
            # Create a mock process that times out on wait
            mock_process = MagicMock()
            mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=5)
            mock_popen.return_value = mock_process
            
            # Create MCP and then close it
            mcp = EverArtMCP(api_key=TEST_API_KEY)
            mcp.close()
            
            # Verify process was killed after timeout
            mock_process.terminate.assert_called_once()
            mock_process.kill.assert_called_once()
            
    def test_generate_image_minimal_params(self, everart_mcp, mock_process):
        """Test image generation with minimal parameters."""
        # Setup mock response
        image_result = "Generated image URL: https://everart.ai/images/minimal123.jpg"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": image_result}],
                "isError": False
            }
        })
        everart_mcp.process = mock_process
        
        # Generate image with only required parameters
        result = everart_mcp.generate_image(prompt="a simple landscape")
        
        # Verify result
        assert result == image_result
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "everart_generate_image" in written_data
        assert "a simple landscape" in written_data
        assert "1:1" in written_data  # default aspect_ratio
        assert "1" in written_data  # default num_images

    def test_generate_image_with_style(self, everart_mcp, mock_process):
        """Test image generation with style parameter."""
        # Setup mock response
        image_result = "Generated image URL: https://everart.ai/images/styled123.jpg"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": image_result}],
                "isError": False
            }
        })
        everart_mcp.process = mock_process
        
        # Generate image with style
        result = everart_mcp.generate_image(
            prompt="a cityscape",
            style="anime"
        )
        
        # Verify result
        assert result == image_result
        
        # Verify style parameter was included
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "anime" in written_data

    def test_describe_image_default_detail(self, everart_mcp, mock_process):
        """Test image description with default detail level."""
        # Setup mock response
        description_result = "The image shows a landscape with mountains and trees."
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": description_result}],
                "isError": False
            }
        })
        everart_mcp.process = mock_process
        
        # Describe image with default detail level
        result = everart_mcp.describe_image(
            image_url="https://example.com/image.jpg"
        )
        
        # Verify result
        assert result == description_result
        
        # Verify default detail level was used
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "medium" in written_data  # default detail_level

    def test_enhance_image_default_strength(self, everart_mcp, mock_process):
        """Test image enhancement with default strength."""
        # Setup mock response
        enhance_result = "Enhanced image URL: https://everart.ai/images/default-strength123.jpg"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": enhance_result}],
                "isError": False
            }
        })
        everart_mcp.process = mock_process
        
        # Enhance image with default strength
        result = everart_mcp.enhance_image(
            image_url="https://example.com/original.jpg",
            prompt="add more details"
        )
        
        # Verify result
        assert result == enhance_result
        
        # Verify default strength was used
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "0.5" in written_data  # default strength

    def test_multiple_requests(self, everart_mcp, mock_process):
        """Test sending multiple requests in sequence."""
        # Setup mock responses for two different requests
        mock_process.stdout.readline.side_effect = [
            json.dumps({
                "jsonrpc": "2.0",
                "id": "test-id-1",
                "result": {
                    "content": [{"type": "text", "text": "First result"}],
                    "isError": False
                }
            }),
            json.dumps({
                "jsonrpc": "2.0",
                "id": "test-id-2",
                "result": {
                    "content": [{"type": "text", "text": "Second result"}],
                    "isError": False
                }
            })
        ]
        everart_mcp.process = mock_process
        
        # Send first request
        result1 = everart_mcp.generate_image("first prompt")
        assert result1 == "First result"
        
        # Send second request
        result2 = everart_mcp.generate_image("second prompt")
        assert result2 == "Second result"
        
        # Verify both requests were sent
        assert mock_process.stdin.write.call_count == 2
        
        # Verify different prompts were used
        call_args_list = mock_process.stdin.write.call_args_list
        assert "first prompt" in call_args_list[0][0][0]
        assert "second prompt" in call_args_list[1][0][0]

    def test_invalid_aspect_ratio(self, everart_mcp, mock_process):
        """Test image generation with invalid aspect ratio format."""
        # Setup mock response for error
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Invalid aspect ratio format"}],
                "isError": True
            }
        })
        everart_mcp.process = mock_process
        
        # Generate image with invalid aspect ratio
        with pytest.raises(RuntimeError, match="Image generation error"):
            everart_mcp.generate_image(
                prompt="test image",
                aspect_ratio="invalid"
            )

    def test_invalid_detail_level(self, everart_mcp, mock_process):
        """Test image description with invalid detail level."""
        # Setup mock response for error
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Invalid detail level"}],
                "isError": True
            }
        })
        everart_mcp.process = mock_process
        
        # Describe image with invalid detail level
        with pytest.raises(RuntimeError, match="Image description error"):
            everart_mcp.describe_image(
                image_url="https://example.com/image.jpg",
                detail_level="ultra"  # Invalid level
            )
