"""
Unit tests for the MemoryMCP class.
"""

import os
import json
import pytest
import subprocess
from unittest.mock import patch, MagicMock, call
from apps.mcps.memory_mcp import MemoryMCP
from apps.utils.exceptions import ValidationError

class TestMemoryMCP:
    """Test suite for MemoryMCP class."""

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
                        "name": "store_memory",
                        "description": "Stores a memory item"
                    },
                    {
                        "name": "retrieve_memory",
                        "description": "Retrieves a memory item by key"
                    },
                    {
                        "name": "list_memories",
                        "description": "Lists all memories in a namespace"
                    },
                    {
                        "name": "delete_memory",
                        "description": "Deletes a memory item"
                    },
                    {
                        "name": "search_memories",
                        "description": "Searches for memories by content"
                    },
                    {
                        "name": "clear_namespace",
                        "description": "Clears all memories in a namespace"
                    }
                ]
            }
        })
        return mock

    @pytest.fixture
    def memory_mcp(self):
        """Fixture to create a MemoryMCP instance with mocked subprocess."""
        # Mock the subprocess.Popen to avoid actual process creation
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Process is running
            mock_popen.return_value = mock_process
            
            # Create the MemoryMCP instance
            mcp = MemoryMCP(storage_path="/test/storage")
            
            # Replace the process with our controlled mock
            mcp.process = mock_process
            
            yield mcp
            
            # Clean up
            mcp.close()

    def test_init_default(self):
        """Test initialization with default parameters."""
        with patch("subprocess.Popen") as mock_popen:
            mcp = MemoryMCP()
            
            # Verify storage_path is None by default
            assert mcp.storage_path is None
            
            mock_popen.assert_called_once()
            
            # Verify NPX is used by default
            args, kwargs = mock_popen.call_args
            cmd = args[0]
            assert "npx" in cmd
            assert "@modelcontextprotocol/server-memory" in cmd

    def test_init_with_storage_path(self):
        """Test initialization with custom storage path."""
        with patch("subprocess.Popen") as mock_popen:
            mcp = MemoryMCP(storage_path="/custom/storage")
            
            # Verify storage_path is set correctly
            assert mcp.storage_path == "/custom/storage"
            
            # Verify storage_path is passed to the server
            env = mock_popen.call_args[1]["env"]
            assert env["MEMORY_STORAGE_PATH"] == "/custom/storage"

    def test_init_with_docker(self):
        """Test initialization with Docker option."""
        with patch("subprocess.Popen") as mock_popen:
            mcp = MemoryMCP(storage_path="/test/storage", use_docker=True)
            
            # Check that Docker command was used
            args, kwargs = mock_popen.call_args
            cmd = args[0]
            assert "docker" in cmd
            assert "run" in cmd
            assert "mcp/memory" in cmd
            
            # Verify storage is mounted correctly
            assert "-v" in cmd
            assert "/test/storage:/data" in " ".join(cmd)
            assert "-e" in cmd
            assert "MEMORY_STORAGE_PATH=/data" in " ".join(cmd)

    def test_start_server_npx_with_storage(self):
        """Test starting server with NPX and storage path."""
        with patch("subprocess.Popen") as mock_popen:
            mcp = MemoryMCP(storage_path="/test/storage", use_docker=False)
            
            # Check that NPX command was used
            args, kwargs = mock_popen.call_args
            cmd = args[0]
            assert "npx" in cmd
            assert "@modelcontextprotocol/server-memory" in cmd
            
            # Verify storage_path is set in environment
            env = kwargs["env"]
            assert env["MEMORY_STORAGE_PATH"] == "/test/storage"

    def test_start_server_npx_without_storage(self):
        """Test starting server with NPX and no storage path."""
        with patch("subprocess.Popen") as mock_popen:
            mcp = MemoryMCP(use_docker=False)
            
            # Check that NPX command was used
            args, kwargs = mock_popen.call_args
            cmd = args[0]
            assert "npx" in cmd
            assert "@modelcontextprotocol/server-memory" in cmd
            
            # Verify MEMORY_STORAGE_PATH is not set
            env = kwargs["env"]
            assert "MEMORY_STORAGE_PATH" not in env

    def test_send_request_success(self, memory_mcp, mock_process):
        """Test sending a request successfully."""
        # Setup mock response
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {"test": "data"}
        })
        memory_mcp.process = mock_process
        
        # Send request
        result = memory_mcp._send_request("test_method", {"param": "value"})
        
        # Verify result
        assert result == {"test": "data"}
        
        # Verify request was sent correctly
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "test_method" in written_data
        assert "param" in written_data
        assert "value" in written_data

    def test_send_request_error(self, memory_mcp, mock_process):
        """Test sending a request that returns an error."""
        # Setup mock error response
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "error": {"code": -32000, "message": "Test error"}
        })
        memory_mcp.process = mock_process
        
        # Send request and expect error
        with pytest.raises(RuntimeError, match="MCP server error"):
            memory_mcp._send_request("test_method", {"param": "value"})

    def test_send_request_process_not_running(self, memory_mcp):
        """Test sending a request when process is not running."""
        # Set process to None to simulate not running
        memory_mcp.process = None
        
        # Send request and expect error
        with pytest.raises(RuntimeError, match="MCP server is not running"):
            memory_mcp._send_request("test_method", {"param": "value"})

    def test_list_tools(self, memory_mcp, mock_process):
        """Test listing available tools."""
        # Setup mock response
        tools_response = {
            "tools": [
                {
                    "name": "store_memory",
                    "description": "Stores a memory item"
                },
                {
                    "name": "retrieve_memory",
                    "description": "Retrieves a memory item by key"
                }
            ]
        }
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": tools_response
        })
        memory_mcp.process = mock_process
        
        # Get tools
        tools = memory_mcp.list_tools()
        
        # Verify tools
        assert len(tools) == 2
        assert tools[0]["name"] == "store_memory"
        assert tools[1]["name"] == "retrieve_memory"

    def test_store_memory(self, memory_mcp, mock_process):
        """Test store_memory functionality."""
        # Setup mock response for successful memory storage
        success_message = "Memory stored successfully with key: test_key"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": success_message}],
                "isError": False
            }
        })
        memory_mcp.process = mock_process
        
        # Store memory
        result = memory_mcp.store_memory("test_key", "test value", namespace="test_namespace")
        
        # Verify result
        assert result == success_message
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "store_memory" in written_data
        assert "test_key" in written_data
        assert "test value" in written_data
        assert "test_namespace" in written_data

    def test_store_memory_default_namespace(self, memory_mcp, mock_process):
        """Test store_memory with default namespace."""
        # Setup mock response
        success_message = "Memory stored successfully with key: test_key"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": success_message}],
                "isError": False
            }
        })
        memory_mcp.process = mock_process
        
        # Store memory with default namespace
        result = memory_mcp.store_memory("test_key", "test value")
        
        # Verify result
        assert result == success_message
        
        # Verify default namespace was used
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "default" in written_data  # default namespace

    def test_store_memory_error(self, memory_mcp, mock_process):
        """Test store_memory with error response."""
        # Setup mock response for failed memory storage
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Failed to store memory: invalid key"}],
                "isError": True
            }
        })
        memory_mcp.process = mock_process
        
        # Store memory and expect error
        with pytest.raises(RuntimeError, match="Store memory error"):
            memory_mcp.store_memory("", "test value")

    def test_retrieve_memory(self, memory_mcp, mock_process):
        """Test retrieve_memory functionality."""
        # Setup mock response for successful memory retrieval
        memory_value = "This is the stored value"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": memory_value}],
                "isError": False
            }
        })
        memory_mcp.process = mock_process
        
        # Retrieve memory
        result = memory_mcp.retrieve_memory("test_key", namespace="test_namespace")
        
        # Verify result
        assert result == memory_value
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "retrieve_memory" in written_data
        assert "test_key" in written_data
        assert "test_namespace" in written_data

    def test_retrieve_memory_default_namespace(self, memory_mcp, mock_process):
        """Test retrieve_memory with default namespace."""
        # Setup mock response
        memory_value = "This is the stored value"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": memory_value}],
                "isError": False
            }
        })
        memory_mcp.process = mock_process
        
        # Retrieve memory with default namespace
        result = memory_mcp.retrieve_memory("test_key")
        
        # Verify result
        assert result == memory_value
        
        # Verify default namespace was used
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "default" in written_data  # default namespace

    def test_retrieve_memory_error(self, memory_mcp, mock_process):
        """Test retrieve_memory with error response."""
        # Setup mock response for failed memory retrieval
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Memory not found: nonexistent_key"}],
                "isError": True
            }
        })
        memory_mcp.process = mock_process
        
        # Retrieve memory and expect error
        with pytest.raises(RuntimeError, match="Retrieve memory error"):
            memory_mcp.retrieve_memory("nonexistent_key")

    def test_list_memories(self, memory_mcp, mock_process):
        """Test list_memories functionality."""
        # Setup mock response for successful memory listing
        memory_list = "key1\nkey2\nkey3"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": memory_list}],
                "isError": False
            }
        })
        memory_mcp.process = mock_process
        
        # List memories
        result = memory_mcp.list_memories(namespace="test_namespace")
        
        # Verify result
        assert result == memory_list
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "list_memories" in written_data
        assert "test_namespace" in written_data

    def test_list_memories_default_namespace(self, memory_mcp, mock_process):
        """Test list_memories with default namespace."""
        # Setup mock response
        memory_list = "key1\nkey2\nkey3"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": memory_list}],
                "isError": False
            }
        })
        memory_mcp.process = mock_process
        
        # List memories with default namespace
        result = memory_mcp.list_memories()
        
        # Verify result
        assert result == memory_list
        
        # Verify default namespace was used
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "default" in written_data  # default namespace

    def test_list_memories_error(self, memory_mcp, mock_process):
        """Test list_memories with error response."""
        # Setup mock response for failed memory listing
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Namespace not found: nonexistent"}],
                "isError": True
            }
        })
        memory_mcp.process = mock_process
        
        # List memories and expect error
        with pytest.raises(RuntimeError, match="List memories error"):
            memory_mcp.list_memories(namespace="nonexistent")

    def test_delete_memory(self, memory_mcp, mock_process):
        """Test delete_memory functionality."""
        # Setup mock response for successful memory deletion
        success_message = "Memory deleted: test_key"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": success_message}],
                "isError": False
            }
        })
        memory_mcp.process = mock_process
        
        # Delete memory
        result = memory_mcp.delete_memory("test_key", namespace="test_namespace")
        
        # Verify result
        assert result == success_message
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "delete_memory" in written_data
        assert "test_key" in written_data
        assert "test_namespace" in written_data

    def test_delete_memory_default_namespace(self, memory_mcp, mock_process):
        """Test delete_memory with default namespace."""
        # Setup mock response
        success_message = "Memory deleted: test_key"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": success_message}],
                "isError": False
            }
        })
        memory_mcp.process = mock_process
        
        # Delete memory with default namespace
        result = memory_mcp.delete_memory("test_key")
        
        # Verify result
        assert result == success_message
        
        # Verify default namespace was used
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "default" in written_data  # default namespace

    def test_delete_memory_error(self, memory_mcp, mock_process):
        """Test delete_memory with error response."""
        # Setup mock response for failed memory deletion
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Memory not found: nonexistent_key"}],
                "isError": True
            }
        })
        memory_mcp.process = mock_process
        
        # Delete memory and expect error
        with pytest.raises(RuntimeError, match="Delete memory error"):
            memory_mcp.delete_memory("nonexistent_key")

    def test_search_memories(self, memory_mcp, mock_process):
        """Test search_memories functionality."""
        # Setup mock response for successful memory search
        search_results = "key1: value containing search term\nkey2: another matching value"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": search_results}],
                "isError": False
            }
        })
        memory_mcp.process = mock_process
        
        # Search memories
        result = memory_mcp.search_memories("search term", namespace="test_namespace")
        
        # Verify result
        assert result == search_results
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "search_memories" in written_data
        assert "search term" in written_data
        assert "test_namespace" in written_data

    def test_search_memories_default_namespace(self, memory_mcp, mock_process):
        """Test search_memories with default namespace."""
        # Setup mock response
        search_results = "key1: value containing search term"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": search_results}],
                "isError": False
            }
        })
        memory_mcp.process = mock_process
        
        # Search memories with default namespace
        result = memory_mcp.search_memories("search term")
        
        # Verify result
        assert result == search_results
        
        # Verify default namespace was used
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "default" in written_data  # default namespace

    def test_search_memories_error(self, memory_mcp, mock_process):
        """Test search_memories with error response."""
        # Setup mock response for failed memory search
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Search failed: invalid query"}],
                "isError": True
            }
        })
        memory_mcp.process = mock_process
        
        # Search memories and expect error
        with pytest.raises(RuntimeError, match="Search memories error"):
            memory_mcp.search_memories("")

    def test_clear_namespace(self, memory_mcp, mock_process):
        """Test clear_namespace functionality."""
        # Setup mock response for successful namespace clearing
        success_message = "Namespace cleared: test_namespace"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": success_message}],
                "isError": False
            }
        })
        memory_mcp.process = mock_process
        
        # Clear namespace
        result = memory_mcp.clear_namespace("test_namespace")
        
        # Verify result
        assert result == success_message
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "clear_namespace" in written_data
        assert "test_namespace" in written_data

    def test_clear_namespace_error(self, memory_mcp, mock_process):
        """Test clear_namespace with error response."""
        # Setup mock response for failed namespace clearing
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Namespace not found: nonexistent"}],
                "isError": True
            }
        })
        memory_mcp.process = mock_process
        
        # Clear namespace and expect error
        with pytest.raises(RuntimeError, match="Clear namespace error"):
            memory_mcp.clear_namespace("nonexistent")

    def test_close(self):
        """Test closing the MCP server process."""
        with patch("subprocess.Popen") as mock_popen:
            # Create a mock process
            mock_process = MagicMock()
            mock_popen.return_value = mock_process
            
            # Create MCP and then close it
            mcp = MemoryMCP()
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
            mcp = MemoryMCP()
            mcp.close()
            
            # Verify terminate was not called
            mock_process.terminate.assert_not_called()

    def test_context_manager(self):
        """Test using MemoryMCP as a context manager."""
        with patch("subprocess.Popen") as mock_popen:
            # Create a mock process
            mock_process = MagicMock()
            mock_popen.return_value = mock_process
            
            # Use as context manager
            with MemoryMCP() as mcp:
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
            mcp = MemoryMCP()
            mcp.close()
            
            # Verify process was killed after timeout
            mock_process.terminate.assert_called_once()
            mock_process.kill.assert_called_once()

    def test_memory_workflow(self, memory_mcp, mock_process):
        """Test a complete memory workflow: store, retrieve, list, search, delete."""
        # Setup mock responses for each operation
        mock_process.stdout.readline.side_effect = [
            # store_memory response
            json.dumps({
                "jsonrpc": "2.0",
                "id": "test-id-1",
                "result": {
                    "content": [{"type": "text", "text": "Memory stored successfully with key: test_key"}],
                    "isError": False
                }
            }),
            # retrieve_memory response
            json.dumps({
                "jsonrpc": "2.0",
                "id": "test-id-2",
                "result": {
                    "content": [{"type": "text", "text": "test value"}],
                    "isError": False
                }
            }),
            # list_memories response
            json.dumps({
                "jsonrpc": "2.0",
                "id": "test-id-3",
                "result": {
                    "content": [{"type": "text", "text": "test_key"}],
                    "isError": False
                }
            }),
            # search_memories response
            json.dumps({
                "jsonrpc": "2.0",
                "id": "test-id-4",
                "result": {
                    "content": [{"type": "text", "text": "test_key: test value"}],
                    "isError": False
                }
            }),
            # delete_memory response
            json.dumps({
                "jsonrpc": "2.0",
                "id": "test-id-5",
                "result": {
                    "content": [{"type": "text", "text": "Memory deleted: test_key"}],
                    "isError": False
                }
            })
        ]
        memory_mcp.process = mock_process
        
        # Execute workflow
        store_result = memory_mcp.store_memory("test_key", "test value")
        assert "successfully" in store_result
        
        retrieve_result = memory_mcp.retrieve_memory("test_key")
        assert retrieve_result == "test value"
        
        list_result = memory_mcp.list_memories()
        assert "test_key" in list_result
        
        search_result = memory_mcp.search_memories("test")
        assert "test_key: test value" in search_result
        
        delete_result = memory_mcp.delete_memory("test_key")
        assert "deleted" in delete_result
        
        # Verify all requests were sent
        assert mock_process.stdin.write.call_count == 5
