"""
Unit tests for the BraveSearchMCP class.
"""

import os
import json
import pytest
import subprocess
from unittest.mock import patch, MagicMock, call
from apps.mcps.brave_search_mcp import BraveSearchMCP
from apps.utils.exceptions import ConfigurationError, APIError

# Test API key for testing
TEST_API_KEY = "test_api_key"

class TestBraveSearchMCP:
    """Test suite for BraveSearchMCP class."""

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
                        "name": "brave_web_search",
                        "description": "Performs a web search using the Brave Search API"
                    },
                    {
                        "name": "brave_local_search",
                        "description": "Searches for local businesses and places"
                    }
                ]
            }
        })
        return mock

    @pytest.fixture
    def brave_search_mcp(self, monkeypatch):
        """Fixture to create a BraveSearchMCP instance with mocked subprocess."""
        # Set environment variable for API key
        monkeypatch.setenv("BRAVE_API_KEY", TEST_API_KEY)
        
        # Mock the subprocess.Popen to avoid actual process creation
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Process is running
            mock_popen.return_value = mock_process
            
            # Create the BraveSearchMCP instance
            mcp = BraveSearchMCP()
            
            # Replace the process with our controlled mock
            mcp.process = mock_process
            
            yield mcp
            
            # Clean up
            mcp.close()

    def test_init_with_api_key_param(self):
        """Test initialization with API key as parameter."""
        with patch("subprocess.Popen") as mock_popen:
            mcp = BraveSearchMCP(api_key=TEST_API_KEY)
            assert mcp.api_key == TEST_API_KEY
            mock_popen.assert_called_once()

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization with API key from environment variable."""
        monkeypatch.setenv("BRAVE_API_KEY", TEST_API_KEY)
        
        with patch("subprocess.Popen") as mock_popen:
            mcp = BraveSearchMCP()
            assert mcp.api_key == TEST_API_KEY
            mock_popen.assert_called_once()

    def test_init_no_api_key(self, monkeypatch):
        """Test initialization with no API key raises error."""
        # Ensure environment variable is not set
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="Brave API key is required"):
            BraveSearchMCP()

    def test_init_with_docker(self):
        """Test initialization with Docker option."""
        with patch("subprocess.Popen") as mock_popen:
            mcp = BraveSearchMCP(api_key=TEST_API_KEY, use_docker=True)
            
            # Check that Docker command was used
            args, kwargs = mock_popen.call_args
            cmd = args[0]
            assert "docker" in cmd
            assert "run" in cmd
            assert "mcp/brave-search" in cmd

    def test_start_server_npx(self):
        """Test starting server with NPX."""
        with patch("subprocess.Popen") as mock_popen:
            mcp = BraveSearchMCP(api_key=TEST_API_KEY, use_docker=False)
            
            # Check that NPX command was used
            args, kwargs = mock_popen.call_args
            cmd = args[0]
            assert "npx" in cmd
            assert "@modelcontextprotocol/server-brave-search" in cmd

    def test_send_request_success(self, brave_search_mcp, mock_process):
        """Test sending a request successfully."""
        # Setup mock response
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {"test": "data"}
        })
        brave_search_mcp.process = mock_process
        
        # Send request
        result = brave_search_mcp._send_request("test_method", {"param": "value"})
        
        # Verify result
        assert result == {"test": "data"}
        
        # Verify request was sent correctly
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "test_method" in written_data
        assert "param" in written_data
        assert "value" in written_data

    def test_send_request_error(self, brave_search_mcp, mock_process):
        """Test sending a request that returns an error."""
        # Setup mock error response
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "error": {"code": -32000, "message": "Test error"}
        })
        brave_search_mcp.process = mock_process
        
        # Send request and expect error
        with pytest.raises(RuntimeError, match="MCP server error"):
            brave_search_mcp._send_request("test_method", {"param": "value"})

    def test_send_request_process_not_running(self, brave_search_mcp):
        """Test sending a request when process is not running."""
        # Set process to None to simulate not running
        brave_search_mcp.process = None
        
        # Send request and expect error
        with pytest.raises(RuntimeError, match="MCP server is not running"):
            brave_search_mcp._send_request("test_method", {"param": "value"})

    def test_list_tools(self, brave_search_mcp, mock_process):
        """Test listing available tools."""
        # Setup mock response
        tools_response = {
            "tools": [
                {
                    "name": "brave_web_search",
                    "description": "Performs a web search using the Brave Search API"
                },
                {
                    "name": "brave_local_search",
                    "description": "Searches for local businesses and places"
                }
            ]
        }
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": tools_response
        })
        brave_search_mcp.process = mock_process
        
        # Get tools
        tools = brave_search_mcp.list_tools()
        
        # Verify tools
        assert len(tools) == 2
        assert tools[0]["name"] == "brave_web_search"
        assert tools[1]["name"] == "brave_local_search"

    def test_web_search(self, brave_search_mcp, mock_process):
        """Test web search functionality."""
        # Setup mock response for successful search
        search_result = "Title: Test Result\nDescription: This is a test result\nURL: https://example.com"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": search_result}],
                "isError": False
            }
        })
        brave_search_mcp.process = mock_process
        
        # Perform search
        result = brave_search_mcp.web_search("test query", count=5, offset=1)
        
        # Verify result
        assert result == search_result
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "brave_web_search" in written_data
        assert "test query" in written_data
        assert "5" in written_data  # count
        assert "1" in written_data  # offset

    def test_web_search_error(self, brave_search_mcp, mock_process):
        """Test web search with error response."""
        # Setup mock response for failed search
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Search failed: Rate limit exceeded"}],
                "isError": True
            }
        })
        brave_search_mcp.process = mock_process
        
        # Perform search and expect error
        with pytest.raises(RuntimeError, match="Web search error"):
            brave_search_mcp.web_search("test query")

    def test_local_search(self, brave_search_mcp, mock_process):
        """Test local search functionality."""
        # Setup mock response for successful local search
        search_result = "Name: Test Business\nAddress: 123 Test St, Test City\nPhone: 555-1234"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": search_result}],
                "isError": False
            }
        })
        brave_search_mcp.process = mock_process
        
        # Perform local search
        result = brave_search_mcp.local_search("coffee shops", count=3)
        
        # Verify result
        assert result == search_result
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "brave_local_search" in written_data
        assert "coffee shops" in written_data
        assert "3" in written_data  # count

    def test_local_search_error(self, brave_search_mcp, mock_process):
        """Test local search with error response."""
        # Setup mock response for failed local search
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Local search failed: Invalid location"}],
                "isError": True
            }
        })
        brave_search_mcp.process = mock_process
        
        # Perform local search and expect error
        with pytest.raises(RuntimeError, match="Local search error"):
            brave_search_mcp.local_search("coffee shops")

    def test_close(self):
        """Test closing the MCP server process."""
        with patch("subprocess.Popen") as mock_popen:
            # Create a mock process
            mock_process = MagicMock()
            mock_popen.return_value = mock_process
            
            # Create MCP and then close it
            mcp = BraveSearchMCP(api_key=TEST_API_KEY)
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
            mcp = BraveSearchMCP(api_key=TEST_API_KEY)
            mcp.close()
            
            # Verify terminate was not called
            mock_process.terminate.assert_not_called()

    def test_context_manager(self):
        """Test using BraveSearchMCP as a context manager."""
        with patch("subprocess.Popen") as mock_popen:
            # Create a mock process
            mock_process = MagicMock()
            mock_popen.return_value = mock_process
            
            # Use as context manager
            with BraveSearchMCP(api_key=TEST_API_KEY) as mcp:
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
            mcp = BraveSearchMCP(api_key=TEST_API_KEY)
            mcp.close()
            
            # Verify process was killed after timeout
            mock_process.terminate.assert_called_once()
            mock_process.kill.assert_called_once()
