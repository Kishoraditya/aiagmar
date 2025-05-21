"""
Unit tests for the FetchMCP class.
"""

import os
import json
import pytest
import subprocess
from unittest.mock import patch, MagicMock, call
from apps.mcps.fetch_mcp import FetchMCP
from apps.utils.exceptions import ValidationError

class TestFetchMCP:
    """Test suite for FetchMCP class."""

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
                        "name": "fetch_url",
                        "description": "Fetches content from a URL"
                    },
                    {
                        "name": "fetch_html",
                        "description": "Fetches raw HTML content from a URL"
                    },
                    {
                        "name": "fetch_text",
                        "description": "Fetches text content from a URL, removing HTML tags"
                    }
                ]
            }
        })
        return mock

    @pytest.fixture
    def fetch_mcp(self):
        """Fixture to create a FetchMCP instance with mocked subprocess."""
        # Mock the subprocess.Popen to avoid actual process creation
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Process is running
            mock_popen.return_value = mock_process
            
            # Create the FetchMCP instance
            mcp = FetchMCP()
            
            # Replace the process with our controlled mock
            mcp.process = mock_process
            
            yield mcp
            
            # Clean up
            mcp.close()

    def test_init_default(self):
        """Test initialization with default parameters."""
        with patch("subprocess.Popen") as mock_popen:
            mcp = FetchMCP()
            mock_popen.assert_called_once()
            
            # Verify NPX is used by default
            args, kwargs = mock_popen.call_args
            cmd = args[0]
            assert "npx" in cmd
            assert "@modelcontextprotocol/server-fetch" in cmd

    def test_init_with_docker(self):
        """Test initialization with Docker option."""
        with patch("subprocess.Popen") as mock_popen:
            mcp = FetchMCP(use_docker=True)
            
            # Check that Docker command was used
            args, kwargs = mock_popen.call_args
            cmd = args[0]
            assert "docker" in cmd
            assert "run" in cmd
            assert "mcp/fetch" in cmd

    def test_start_server_npx(self):
        """Test starting server with NPX."""
        with patch("subprocess.Popen") as mock_popen:
            mcp = FetchMCP(use_docker=False)
            
            # Check that NPX command was used
            args, kwargs = mock_popen.call_args
            cmd = args[0]
            assert "npx" in cmd
            assert "@modelcontextprotocol/server-fetch" in cmd

    def test_send_request_success(self, fetch_mcp, mock_process):
        """Test sending a request successfully."""
        # Setup mock response
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {"test": "data"}
        })
        fetch_mcp.process = mock_process
        
        # Send request
        result = fetch_mcp._send_request("test_method", {"param": "value"})
        
        # Verify result
        assert result == {"test": "data"}
        
        # Verify request was sent correctly
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "test_method" in written_data
        assert "param" in written_data
        assert "value" in written_data

    def test_send_request_error(self, fetch_mcp, mock_process):
        """Test sending a request that returns an error."""
        # Setup mock error response
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "error": {"code": -32000, "message": "Test error"}
        })
        fetch_mcp.process = mock_process
        
        # Send request and expect error
        with pytest.raises(RuntimeError, match="MCP server error"):
            fetch_mcp._send_request("test_method", {"param": "value"})

    def test_send_request_process_not_running(self, fetch_mcp):
        """Test sending a request when process is not running."""
        # Set process to None to simulate not running
        fetch_mcp.process = None
        
        # Send request and expect error
        with pytest.raises(RuntimeError, match="MCP server is not running"):
            fetch_mcp._send_request("test_method", {"param": "value"})

    def test_list_tools(self, fetch_mcp, mock_process):
        """Test listing available tools."""
        # Setup mock response
        tools_response = {
            "tools": [
                {
                    "name": "fetch_url",
                    "description": "Fetches content from a URL"
                },
                {
                    "name": "fetch_html",
                    "description": "Fetches raw HTML content from a URL"
                },
                {
                    "name": "fetch_text",
                    "description": "Fetches text content from a URL, removing HTML tags"
                }
            ]
        }
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": tools_response
        })
        fetch_mcp.process = mock_process
        
        # Get tools
        tools = fetch_mcp.list_tools()
        
        # Verify tools
        assert len(tools) == 3
        assert tools[0]["name"] == "fetch_url"
        assert tools[1]["name"] == "fetch_html"
        assert tools[2]["name"] == "fetch_text"

    def test_fetch_url(self, fetch_mcp, mock_process):
        """Test fetch_url functionality."""
        # Setup mock response for successful fetch
        fetch_result = "<html><body><h1>Test Page</h1><p>This is a test page.</p></body></html>"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": fetch_result}],
                "isError": False
            }
        })
        fetch_mcp.process = mock_process
        
        # Fetch URL
        result = fetch_mcp.fetch_url(
            url="https://example.com",
            selector="body",
            timeout=5000,
            wait_for="h1"
        )
        
        # Verify result
        assert result == fetch_result
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "fetch_url" in written_data
        assert "https://example.com" in written_data
        assert "body" in written_data  # selector
        assert "5000" in written_data  # timeout
        assert "h1" in written_data  # wait_for

    def test_fetch_url_minimal_params(self, fetch_mcp, mock_process):
        """Test fetch_url with minimal parameters."""
        # Setup mock response
        fetch_result = "<html><body><h1>Test Page</h1><p>This is a test page.</p></body></html>"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": fetch_result}],
                "isError": False
            }
        })
        fetch_mcp.process = mock_process
        
        # Fetch URL with only required parameters
        result = fetch_mcp.fetch_url(url="https://example.com")
        
        # Verify result
        assert result == fetch_result
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "fetch_url" in written_data
        assert "https://example.com" in written_data
        assert "30000" in written_data  # default timeout

    def test_fetch_url_error(self, fetch_mcp, mock_process):
        """Test fetch_url with error response."""
        # Setup mock response for failed fetch
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Fetch failed: 404 Not Found"}],
                "isError": True
            }
        })
        fetch_mcp.process = mock_process
        
        # Fetch URL and expect error
        with pytest.raises(RuntimeError, match="Fetch error"):
            fetch_mcp.fetch_url("https://example.com/nonexistent")

    def test_fetch_html(self, fetch_mcp, mock_process):
        """Test fetch_html functionality."""
        # Setup mock response for successful HTML fetch
        html_result = "<html><body><h1>Test Page</h1><p>This is a test page.</p></body></html>"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": html_result}],
                "isError": False
            }
        })
        fetch_mcp.process = mock_process
        
        # Fetch HTML
        result = fetch_mcp.fetch_html(
            url="https://example.com",
            timeout=5000,
            wait_for="h1"
        )
        
        # Verify result
        assert result == html_result
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "fetch_html" in written_data
        assert "https://example.com" in written_data
        assert "5000" in written_data  # timeout
        assert "h1" in written_data  # wait_for

    def test_fetch_html_minimal_params(self, fetch_mcp, mock_process):
        """Test fetch_html with minimal parameters."""
        # Setup mock response
        html_result = "<html><body><h1>Test Page</h1><p>This is a test page.</p></body></html>"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": html_result}],
                "isError": False
            }
        })
        fetch_mcp.process = mock_process
        
        # Fetch HTML with only required parameters
        result = fetch_mcp.fetch_html(url="https://example.com")
        
        # Verify result
        assert result == html_result
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "fetch_html" in written_data
        assert "https://example.com" in written_data
        assert "30000" in written_data  # default timeout

    def test_fetch_html_error(self, fetch_mcp, mock_process):
        """Test fetch_html with error response."""
        # Setup mock response for failed HTML fetch
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Fetch HTML failed: Connection timeout"}],
                "isError": True
            }
        })
        fetch_mcp.process = mock_process
        
        # Fetch HTML and expect error
        with pytest.raises(RuntimeError, match="Fetch HTML error"):
            fetch_mcp.fetch_html("https://example.com")

    def test_fetch_text(self, fetch_mcp, mock_process):
        """Test fetch_text functionality."""
        # Setup mock response for successful text fetch
        text_result = "Test Page\n\nThis is a test page."
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": text_result}],
                "isError": False
            }
        })
        fetch_mcp.process = mock_process
        
        # Fetch text
        result = fetch_mcp.fetch_text(
            url="https://example.com",
            timeout=5000,
            wait_for="h1"
        )
        
        # Verify result
        assert result == text_result
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "fetch_text" in written_data
        assert "https://example.com" in written_data
        assert "5000" in written_data  # timeout
        assert "h1" in written_data  # wait_for

    def test_fetch_text_minimal_params(self, fetch_mcp, mock_process):
        """Test fetch_text with minimal parameters."""
        # Setup mock response
        text_result = "Test Page\n\nThis is a test page."
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": text_result}],
                "isError": False
            }
        })
        fetch_mcp.process = mock_process
        
        # Fetch text with only required parameters
        result = fetch_mcp.fetch_text(url="https://example.com")
        
        # Verify result
        assert result == text_result
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "fetch_text" in written_data
        assert "https://example.com" in written_data
        assert "30000" in written_data  # default timeout

    def test_fetch_text_error(self, fetch_mcp, mock_process):
        """Test fetch_text with error response."""
        # Setup mock response for failed text fetch
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Fetch text failed: Invalid URL"}],
                "isError": True
            }
        })
        fetch_mcp.process = mock_process
        
        # Fetch text and expect error
        with pytest.raises(RuntimeError, match="Fetch text error"):
            fetch_mcp.fetch_text("invalid-url")

    def test_close(self):
        """Test closing the MCP server process."""
        with patch("subprocess.Popen") as mock_popen:
            # Create a mock process
            mock_process = MagicMock()
            mock_popen.return_value = mock_process
            
            # Create MCP and then close it
            mcp = FetchMCP()
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
            mcp = FetchMCP()
            mcp.close()
            
            # Verify terminate was not called
            mock_process.terminate.assert_not_called()

    def test_context_manager(self):
        """Test using FetchMCP as a context manager."""
        with patch("subprocess.Popen") as mock_popen:
            # Create a mock process
            mock_process = MagicMock()
            mock_popen.return_value = mock_process
            
            # Use as context manager
            with FetchMCP() as mcp:
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
            mcp = FetchMCP()
            mcp.close()
            
            # Verify process was killed after timeout
            mock_process.terminate.assert_called_once()
            mock_process.kill.assert_called_once()

    def test_multiple_requests(self, fetch_mcp, mock_process):
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
        fetch_mcp.process = mock_process
        
        # Send first request
        result1 = fetch_mcp.fetch_url("https://example.com/first")
        assert result1 == "First result"
        
        # Send second request
        result2 = fetch_mcp.fetch_url("https://example.com/second")
        assert result2 == "Second result"
        
        # Verify both requests were sent
        assert mock_process.stdin.write.call_count == 2
        
        # Verify different URLs were used
        call_args_list = mock_process.stdin.write.call_args_list
        assert "https://example.com/first" in call_args_list[0][0][0]
        assert "https://example.com/second" in call_args_list[1][0][0]

    def test_fetch_url_with_selector(self, fetch_mcp, mock_process):
        """Test fetch_url with CSS selector."""
        # Setup mock response
        fetch_result = "<p>This is a test paragraph.</p>"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": fetch_result}],
                "isError": False
            }
        })
        fetch_mcp.process = mock_process
        
        # Fetch URL with selector
        result = fetch_mcp.fetch_url(
            url="https://example.com",
            selector="p"
        )
        
        # Verify result
        assert result == fetch_result
        
        # Verify selector was included in request
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "p" in written_data  # selector

    def test_fetch_url_with_wait_for(self, fetch_mcp, mock_process):
        """Test fetch_url with wait_for parameter."""
        # Setup mock response
        fetch_result = "<div class='dynamic-content'>Loaded content</div>"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": fetch_result}],
                "isError": False
            }
        })
        fetch_mcp.process = mock_process
        
        # Fetch URL with wait_for
        result = fetch_mcp.fetch_url(
            url="https://example.com",
            wait_for=".dynamic-content"
        )
        
        # Verify result
        assert result == fetch_result
        
        # Verify wait_for was included in request
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert ".dynamic-content" in written_data  # wait_for

    def test_fetch_url_with_custom_timeout(self, fetch_mcp, mock_process):
        """Test fetch_url with custom timeout."""
        # Setup mock response
        fetch_result = "<html><body>Content</body></html>"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": fetch_result}],
                "isError": False
            }
        })
        fetch_mcp.process = mock_process
        
        # Fetch URL with custom timeout
        result = fetch_mcp.fetch_url(
            url="https://example.com",
            timeout=60000  # 60 seconds
        )
        
        # Verify result
        assert result == fetch_result
        
        # Verify timeout was included in request
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "60000" in written_data  # timeout

    def test_invalid_url(self, fetch_mcp, mock_process):
        """Test with invalid URL format."""
        # Setup mock response for error
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Invalid URL format"}],
                "isError": True
            }
        })
        fetch_mcp.process = mock_process
        
        # Fetch with invalid URL
        with pytest.raises(RuntimeError, match="Fetch error"):
            fetch_mcp.fetch_url("not-a-valid-url")
