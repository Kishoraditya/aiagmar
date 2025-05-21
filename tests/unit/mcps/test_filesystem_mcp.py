"""
Unit tests for the FilesystemMCP class.
"""

import os
import json
import pytest
import subprocess
from unittest.mock import patch, MagicMock, call
from apps.mcps.filesystem_mcp import FilesystemMCP
from apps.utils.exceptions import ValidationError

class TestFilesystemMCP:
    """Test suite for FilesystemMCP class."""

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
                        "name": "read_file",
                        "description": "Reads the contents of a file"
                    },
                    {
                        "name": "write_file",
                        "description": "Writes content to a file"
                    },
                    {
                        "name": "list_directory",
                        "description": "Lists files and directories"
                    },
                    {
                        "name": "create_directory",
                        "description": "Creates a directory"
                    },
                    {
                        "name": "delete_file",
                        "description": "Deletes a file"
                    },
                    {
                        "name": "file_exists",
                        "description": "Checks if a file exists"
                    },
                    {
                        "name": "search_files",
                        "description": "Searches for files matching a pattern"
                    }
                ]
            }
        })
        return mock

    @pytest.fixture
    def filesystem_mcp(self):
        """Fixture to create a FilesystemMCP instance with mocked subprocess."""
        # Mock the subprocess.Popen to avoid actual process creation
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Process is running
            mock_popen.return_value = mock_process
            
            # Create the FilesystemMCP instance
            mcp = FilesystemMCP(workspace_dir="/test/workspace")
            
            # Replace the process with our controlled mock
            mcp.process = mock_process
            
            yield mcp
            
            # Clean up
            mcp.close()

    def test_init_default(self):
        """Test initialization with default parameters."""
        with patch("subprocess.Popen") as mock_popen, patch("os.getcwd") as mock_getcwd:
            mock_getcwd.return_value = "/current/dir"
            mcp = FilesystemMCP()
            
            # Verify workspace_dir is current directory by default
            assert mcp.workspace_dir == "/current/dir"
            
            mock_popen.assert_called_once()
            
            # Verify NPX is used by default
            args, kwargs = mock_popen.call_args
            cmd = args[0]
            assert "npx" in cmd
            assert "@modelcontextprotocol/server-filesystem" in cmd

    def test_init_with_workspace_dir(self):
        """Test initialization with custom workspace directory."""
        with patch("subprocess.Popen") as mock_popen:
            mcp = FilesystemMCP(workspace_dir="/custom/workspace")
            
            # Verify workspace_dir is set correctly
            assert mcp.workspace_dir == "/custom/workspace"
            
            # Verify workspace_dir is passed to the server
            env = mock_popen.call_args[1]["env"]
            assert env["WORKSPACE_DIR"] == "/custom/workspace"

    def test_init_with_docker(self):
        """Test initialization with Docker option."""
        with patch("subprocess.Popen") as mock_popen:
            mcp = FilesystemMCP(workspace_dir="/test/workspace", use_docker=True)
            
            # Check that Docker command was used
            args, kwargs = mock_popen.call_args
            cmd = args[0]
            assert "docker" in cmd
            assert "run" in cmd
            assert "mcp/filesystem" in cmd
            
            # Verify workspace is mounted correctly
            assert "-v" in cmd
            assert "/test/workspace:/workspace" in " ".join(cmd)

    def test_start_server_npx(self):
        """Test starting server with NPX."""
        with patch("subprocess.Popen") as mock_popen:
            mcp = FilesystemMCP(workspace_dir="/test/workspace", use_docker=False)
            
            # Check that NPX command was used
            args, kwargs = mock_popen.call_args
            cmd = args[0]
            assert "npx" in cmd
            assert "@modelcontextprotocol/server-filesystem" in cmd
            
            # Verify workspace_dir is set in environment
            env = kwargs["env"]
            assert env["WORKSPACE_DIR"] == "/test/workspace"

    def test_send_request_success(self, filesystem_mcp, mock_process):
        """Test sending a request successfully."""
        # Setup mock response
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {"test": "data"}
        })
        filesystem_mcp.process = mock_process
        
        # Send request
        result = filesystem_mcp._send_request("test_method", {"param": "value"})
        
        # Verify result
        assert result == {"test": "data"}
        
        # Verify request was sent correctly
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "test_method" in written_data
        assert "param" in written_data
        assert "value" in written_data

    def test_send_request_error(self, filesystem_mcp, mock_process):
        """Test sending a request that returns an error."""
        # Setup mock error response
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "error": {"code": -32000, "message": "Test error"}
        })
        filesystem_mcp.process = mock_process
        
        # Send request and expect error
        with pytest.raises(RuntimeError, match="MCP server error"):
            filesystem_mcp._send_request("test_method", {"param": "value"})

    def test_send_request_process_not_running(self, filesystem_mcp):
        """Test sending a request when process is not running."""
        # Set process to None to simulate not running
        filesystem_mcp.process = None
        
        # Send request and expect error
        with pytest.raises(RuntimeError, match="MCP server is not running"):
            filesystem_mcp._send_request("test_method", {"param": "value"})

    def test_list_tools(self, filesystem_mcp, mock_process):
        """Test listing available tools."""
        # Setup mock response
        tools_response = {
            "tools": [
                {
                    "name": "read_file",
                    "description": "Reads the contents of a file"
                },
                {
                    "name": "write_file",
                    "description": "Writes content to a file"
                }
            ]
        }
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": tools_response
        })
        filesystem_mcp.process = mock_process
        
        # Get tools
        tools = filesystem_mcp.list_tools()
        
        # Verify tools
        assert len(tools) == 2
        assert tools[0]["name"] == "read_file"
        assert tools[1]["name"] == "write_file"

    def test_read_file(self, filesystem_mcp, mock_process):
        """Test read_file functionality."""
        # Setup mock response for successful file read
        file_content = "This is the content of the test file."
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": file_content}],
                "isError": False
            }
        })
        filesystem_mcp.process = mock_process
        
        # Read file
        result = filesystem_mcp.read_file("test.txt")
        
        # Verify result
        assert result == file_content
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "read_file" in written_data
        assert "test.txt" in written_data

    def test_read_file_error(self, filesystem_mcp, mock_process):
        """Test read_file with error response."""
        # Setup mock response for failed file read
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "File not found: test.txt"}],
                "isError": True
            }
        })
        filesystem_mcp.process = mock_process
        
        # Read file and expect error
        with pytest.raises(RuntimeError, match="Read file error"):
            filesystem_mcp.read_file("test.txt")

    def test_write_file(self, filesystem_mcp, mock_process):
        """Test write_file functionality."""
        # Setup mock response for successful file write
        success_message = "File written successfully: test.txt"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": success_message}],
                "isError": False
            }
        })
        filesystem_mcp.process = mock_process
        
        # Write file
        result = filesystem_mcp.write_file("test.txt", "This is test content.")
        
        # Verify result
        assert result == success_message
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "write_file" in written_data
        assert "test.txt" in written_data
        assert "This is test content." in written_data

    def test_write_file_error(self, filesystem_mcp, mock_process):
        """Test write_file with error response."""
        # Setup mock response for failed file write
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Permission denied: test.txt"}],
                "isError": True
            }
        })
        filesystem_mcp.process = mock_process
        
        # Write file and expect error
        with pytest.raises(RuntimeError, match="Write file error"):
            filesystem_mcp.write_file("test.txt", "This is test content.")

    def test_list_directory(self, filesystem_mcp, mock_process):
        """Test list_directory functionality."""
        # Setup mock response for successful directory listing
        listing = "file1.txt\nfile2.txt\nsubdir/"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": listing}],
                "isError": False
            }
        })
        filesystem_mcp.process = mock_process
        
        # List directory
        result = filesystem_mcp.list_directory("testdir", recursive=True)
        
        # Verify result
        assert result == listing
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "list_directory" in written_data
        assert "testdir" in written_data
        assert "recursive" in written_data
        assert "true" in written_data.lower()  # recursive=True

    def test_list_directory_default_params(self, filesystem_mcp, mock_process):
        """Test list_directory with default parameters."""
        # Setup mock response
        listing = "file1.txt\nfile2.txt"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": listing}],
                "isError": False
            }
        })
        filesystem_mcp.process = mock_process
        
        # List directory with default parameters
        result = filesystem_mcp.list_directory()
        
        # Verify result
        assert result == listing
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "list_directory" in written_data
        assert "." in written_data  # default path

    def test_list_directory_error(self, filesystem_mcp, mock_process):
        """Test list_directory with error response."""
        # Setup mock response for failed directory listing
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Directory not found: nonexistent"}],
                "isError": True
            }
        })
        filesystem_mcp.process = mock_process
        
        # List directory and expect error
        with pytest.raises(RuntimeError, match="List directory error"):
            filesystem_mcp.list_directory("nonexistent")

    def test_create_directory(self, filesystem_mcp, mock_process):
        """Test create_directory functionality."""
        # Setup mock response for successful directory creation
        success_message = "Directory created: newdir"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": success_message}],
                "isError": False
            }
        })
        filesystem_mcp.process = mock_process
        
        # Create directory
        result = filesystem_mcp.create_directory("newdir")
        
        # Verify result
        assert result == success_message
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "create_directory" in written_data
        assert "newdir" in written_data

    def test_create_directory_error(self, filesystem_mcp, mock_process):
        """Test create_directory with error response."""
        # Setup mock response for failed directory creation
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Failed to create directory: already exists"}],
                "isError": True
            }
        })
        filesystem_mcp.process = mock_process
        
        # Create directory and expect error
        with pytest.raises(RuntimeError, match="Create directory error"):
            filesystem_mcp.create_directory("existing_dir")

    def test_delete_file(self, filesystem_mcp, mock_process):
        """Test delete_file functionality."""
        # Setup mock response for successful file deletion
        success_message = "File deleted: test.txt"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": success_message}],
                "isError": False
            }
        })
        filesystem_mcp.process = mock_process
        
        # Delete file
        result = filesystem_mcp.delete_file("test.txt")
        
        # Verify result
        assert result == success_message
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "delete_file" in written_data
        assert "test.txt" in written_data

    def test_delete_file_error(self, filesystem_mcp, mock_process):
        """Test delete_file with error response."""
        # Setup mock response for failed file deletion
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "File not found: nonexistent.txt"}],
                "isError": True
            }
        })
        filesystem_mcp.process = mock_process
        
        # Delete file and expect error
        with pytest.raises(RuntimeError, match="Delete file error"):
            filesystem_mcp.delete_file("nonexistent.txt")

    def test_file_exists_true(self, filesystem_mcp, mock_process):
        """Test file_exists when file exists."""
        # Setup mock response for file exists
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "File exists: true"}],
                "isError": False
            }
        })
        filesystem_mcp.process = mock_process
        
        # Check if file exists
        result = filesystem_mcp.file_exists("existing.txt")
        
        # Verify result
        assert result is True
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "file_exists" in written_data
        assert "existing.txt" in written_data

    def test_file_exists_false(self, filesystem_mcp, mock_process):
        """Test file_exists when file does not exist."""
        # Setup mock response for file does not exist
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "File exists: false"}],
                "isError": False
            }
        })
        filesystem_mcp.process = mock_process
        
        # Check if file exists
        result = filesystem_mcp.file_exists("nonexistent.txt")
        
        # Verify result
        assert result is False
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "file_exists" in written_data
        assert "nonexistent.txt" in written_data

    def test_file_exists_error(self, filesystem_mcp, mock_process):
        """Test file_exists with error response."""
        # Setup mock response for error
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Error checking file existence"}],
                "isError": True
            }
        })
        filesystem_mcp.process = mock_process
        
        # Check file existence and expect error
        with pytest.raises(RuntimeError, match="File exists error"):
            filesystem_mcp.file_exists("problematic.txt")

    def test_search_files(self, filesystem_mcp, mock_process):
        """Test search_files functionality."""
        # Setup mock response for successful file search
        search_results = "file1.txt\nsubdir/file2.txt"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": search_results}],
                "isError": False
            }
        })
        filesystem_mcp.process = mock_process
        
        # Search files
        result = filesystem_mcp.search_files("*.txt", path="testdir", recursive=True)
        
        # Verify result
        assert result == search_results
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "search_files" in written_data
        assert "*.txt" in written_data
        assert "testdir" in written_data
        assert "true" in written_data.lower()  # recursive=True

    def test_search_files_default_params(self, filesystem_mcp, mock_process):
        """Test search_files with default parameters."""
        # Setup mock response
        search_results = "file1.txt\nfile2.txt"
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": search_results}],
                "isError": False
            }
        })
        filesystem_mcp.process = mock_process
        
        # Search files with default parameters
        result = filesystem_mcp.search_files("*.txt")
        
        # Verify result
        assert result == search_results
        
        # Verify correct parameters were sent
        mock_process.stdin.write.assert_called_once()
        written_data = mock_process.stdin.write.call_args[0][0]
        assert "search_files" in written_data
        assert "*.txt" in written_data
        assert "." in written_data  # default path
        assert "true" in written_data.lower()  # default recursive=True

    def test_search_files_error(self, filesystem_mcp, mock_process):
        """Test search_files with error response."""
        # Setup mock response for failed file search
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Invalid search pattern"}],
                "isError": True
            }
        })
        filesystem_mcp.process = mock_process
        
        # Search files and expect error
        with pytest.raises(RuntimeError, match="Search files error"):
            filesystem_mcp.search_files("[invalid")

    def test_close(self):
        """Test closing the MCP server process."""
        with patch("subprocess.Popen") as mock_popen:
            # Create a mock process
            mock_process = MagicMock()
            mock_popen.return_value = mock_process
            
            # Create MCP and then close it
            mcp = FilesystemMCP()
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
            mcp = FilesystemMCP()
            mcp.close()
            
            # Verify terminate was not called
            mock_process.terminate.assert_not_called()

    def test_context_manager(self):
        """Test using FilesystemMCP as a context manager."""
        with patch("subprocess.Popen") as mock_popen:
            # Create a mock process
            mock_process = MagicMock()
            mock_popen.return_value = mock_process
            
            # Use as context manager
            with FilesystemMCP() as mcp:
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
            mcp = FilesystemMCP()
            mcp.close()
            
            # Verify process was killed after timeout
            mock_process.terminate.assert_called_once()
            mock_process.kill.assert_called_once()

    def test_multiple_requests(self, filesystem_mcp, mock_process):
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
        filesystem_mcp.process = mock_process
        
        # Send first request
        result1 = filesystem_mcp.read_file("file1.txt")
        assert result1 == "First result"
        
        # Send second request
        result2 = filesystem_mcp.read_file("file2.txt")
        assert result2 == "Second result"
        
        # Verify both requests were sent
        assert mock_process.stdin.write.call_count == 2
        
        # Verify different file paths were used
        call_args_list = mock_process.stdin.write.call_args_list
        assert "file1.txt" in call_args_list[0][0][0]
        assert "file2.txt" in call_args_list[1][0][0]

    def test_read_write_workflow(self, filesystem_mcp, mock_process):
        """Test a workflow of writing and then reading a file."""
        # Setup mock responses for write and read operations
        mock_process.stdout.readline.side_effect = [
            json.dumps({
                "jsonrpc": "2.0",
                "id": "test-id-1",
                "result": {
                    "content": [{"type": "text", "text": "File written successfully: test.txt"}],
                    "isError": False
                }
            }),
            json.dumps({
                "jsonrpc": "2.0",
                "id": "test-id-2",
                "result": {
                    "content": [{"type": "text", "text": "This is test content."}],
                    "isError": False
                }
            })
        ]
        filesystem_mcp.process = mock_process
        
        # Write file
        write_result = filesystem_mcp.write_file("test.txt", "This is test content.")
        assert "successfully" in write_result
        
        # Read file
        read_result = filesystem_mcp.read_file("test.txt")
        assert read_result == "This is test content."
        
        # Verify both requests were sent
        assert mock_process.stdin.write.call_count == 2
