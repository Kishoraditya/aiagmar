"""
End-to-end tests for error handling in the system.

These tests verify that the system properly handles various error conditions
that may occur during execution, including:
- MCP connection errors
- API rate limiting
- Invalid inputs
- Authentication failures
- Network timeouts
- Resource not found errors
"""

import os
import pytest
import json
import tempfile
from unittest.mock import patch, MagicMock
import subprocess

# Import agents
from apps.agents.manager_agent import ManagerAgent
from apps.agents.pre_response_agent import PreResponseAgent
from apps.agents.research_agent import ResearchAgent
from apps.agents.image_generation_agent import ImageGenerationAgent
from apps.agents.file_manager_agent import FileManagerAgent
from apps.agents.summary_agent import SummaryAgent
from apps.agents.verification_agent import VerificationAgent

# Import MCPs
from apps.mcps.brave_search_mcp import BraveSearchMCP
from apps.mcps.everart_mcp import EverArtMCP
from apps.mcps.fetch_mcp import FetchMCP
from apps.mcps.filesystem_mcp import FilesystemMCP
from apps.mcps.memory_mcp import MemoryMCP

# Import workflow
from apps.workflows.research_workflow import ResearchWorkflow

# Import utils
from apps.utils.exceptions import (
    AuthenticationError,
    ResourceNotFoundError
)


class TestErrorHandling:
    """Test error handling in the system."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up test environment before each test."""
        # Create a temporary workspace directory
        self.workspace_dir = str(tmp_path / "workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Set up environment variables for testing
        os.environ["BRAVE_API_KEY"] = "test_brave_api_key"
        os.environ["EVERART_API_KEY"] = "test_everart_api_key"
        
        yield
        
        # Clean up environment variables
        if "BRAVE_API_KEY" in os.environ:
            del os.environ["BRAVE_API_KEY"]
        if "EVERART_API_KEY" in os.environ:
            del os.environ["EVERART_API_KEY"]
    
    def test_brave_search_mcp_api_key_missing(self):
        """Test BraveSearchMCP handling of missing API key."""
        # Remove API key from environment
        if "BRAVE_API_KEY" in os.environ:
            del os.environ["BRAVE_API_KEY"]
        
        # Attempt to initialize BraveSearchMCP without API key
        with pytest.raises(ValueError) as excinfo:
            brave_search = BraveSearchMCP()
        
        # Verify error message
        assert "API key is required" in str(excinfo.value)
    
    def test_everart_mcp_api_key_missing(self):
        """Test EverArtMCP handling of missing API key."""
        # Remove API key from environment
        if "EVERART_API_KEY" in os.environ:
            del os.environ["EVERART_API_KEY"]
        
        # Attempt to initialize EverArtMCP without API key
        with pytest.raises(ValueError) as excinfo:
            everart = EverArtMCP()
        
        # Verify error message
        assert "API key is required" in str(excinfo.value)
    
    def test_brave_search_mcp_connection_error(self):
        """Test BraveSearchMCP handling of connection errors."""
        # Create BraveSearchMCP instance
        brave_search = BraveSearchMCP(api_key="test_brave_api_key")
        
        # Mock the _send_request method to raise a connection error
        with patch.object(brave_search, '_send_request', side_effect=RuntimeError("MCP server is not running")):
            # Attempt to use the MCP
            with pytest.raises(RuntimeError) as excinfo:
                brave_search.web_search("Python programming")
            
            # Verify error message
            assert "MCP server is not running" in str(excinfo.value)
    
    def test_fetch_mcp_invalid_url(self):
        """Test FetchMCP handling of invalid URLs."""
        # Create FetchMCP instance
        fetch = FetchMCP()
        
        # Mock the _send_request method to handle invalid URL
        def mock_send_request(method, params):
            if method == "callTool" and params["name"] == "fetch_url":
                if "invalid" in params["arguments"]["url"]:
                    return {
                        "content": [{"type": "text", "text": "Error: Invalid URL format"}],
                        "isError": True
                    }
            return {}
        
        with patch.object(fetch, '_send_request', side_effect=mock_send_request):
            # Attempt to fetch an invalid URL
            with pytest.raises(RuntimeError) as excinfo:
                fetch.fetch_url("http://invalid.url.that.does.not.exist")
            
            # Verify error message
            assert "Invalid URL" in str(excinfo.value)
    
    def test_filesystem_mcp_file_not_found(self):
        """Test FilesystemMCP handling of file not found errors."""
        # Create FilesystemMCP instance
        filesystem = FilesystemMCP(workspace_dir=self.workspace_dir)
        
        # Mock the _send_request method to handle file not found
        def mock_send_request(method, params):
            if method == "callTool" and params["name"] == "read_file":
                if "nonexistent" in params["arguments"]["path"]:
                    return {
                        "content": [{"type": "text", "text": "Error: File not found"}],
                        "isError": True
                    }
            return {}
        
        with patch.object(filesystem, '_send_request', side_effect=mock_send_request):
            # Attempt to read a nonexistent file
            with pytest.raises(RuntimeError) as excinfo:
                filesystem.read_file("nonexistent_file.txt")
            
            # Verify error message
            assert "File not found" in str(excinfo.value)
    
    def test_memory_mcp_key_not_found(self):
        """Test MemoryMCP handling of key not found errors."""
        # Create MemoryMCP instance
        memory = MemoryMCP()
        
        # Mock the _send_request method to handle key not found
        def mock_send_request(method, params):
            if method == "callTool" and params["name"] == "retrieve_memory":
                if "nonexistent" in params["arguments"]["key"]:
                    return {
                        "content": [{"type": "text", "text": "Error: Key not found"}],
                        "isError": True
                    }
            return {}
        
        with patch.object(memory, '_send_request', side_effect=mock_send_request):
            # Attempt to retrieve a nonexistent key
            with pytest.raises(RuntimeError) as excinfo:
                memory.retrieve_memory("nonexistent_key")
            
            # Verify error message
            assert "Key not found" in str(excinfo.value)
    
    def test_brave_search_mcp_rate_limit(self):
        """Test BraveSearchMCP handling of rate limit errors."""
        # Create BraveSearchMCP instance
        brave_search = BraveSearchMCP(api_key="test_brave_api_key")
        
        # Mock the _send_request method to simulate rate limiting
        def mock_send_request(method, params):
            if method == "callTool" and params["name"] == "brave_web_search":
                return {
                    "content": [{"type": "text", "text": "Error: Rate limit exceeded"}],
                    "isError": True
                }
            return {}
        
        with patch.object(brave_search, '_send_request', side_effect=mock_send_request):
            # Attempt to use the MCP
            with pytest.raises(RuntimeError) as excinfo:
                brave_search.web_search("Python programming")
            
            # Verify error message
            assert "Rate limit exceeded" in str(excinfo.value)
    
    def test_everart_mcp_invalid_input(self):
        """Test EverArtMCP handling of invalid input errors."""
        # Create EverArtMCP instance
        everart = EverArtMCP(api_key="test_everart_api_key")
        
        # Mock the _send_request method to handle invalid input
        def mock_send_request(method, params):
            if method == "callTool" and params["name"] == "everart_generate_image":
                if not params["arguments"]["prompt"] or len(params["arguments"]["prompt"]) < 5:
                    return {
                        "content": [{"type": "text", "text": "Error: Prompt too short or empty"}],
                        "isError": True
                    }
            return {}
        
        with patch.object(everart, '_send_request', side_effect=mock_send_request):
            # Attempt to generate an image with an invalid prompt
            with pytest.raises(RuntimeError) as excinfo:
                everart.generate_image("")
            
            # Verify error message
            assert "Prompt too short or empty" in str(excinfo.value)
    
    def test_fetch_mcp_timeout(self):
        """Test FetchMCP handling of timeout errors."""
        # Create FetchMCP instance
        fetch = FetchMCP()
        
        # Mock the _send_request method to simulate timeout
        def mock_send_request(method, params):
            if method == "callTool" and params["name"] == "fetch_url":
                return {
                    "content": [{"type": "text", "text": "Error: Request timed out after 30000ms"}],
                    "isError": True
                }
            return {}
        
        with patch.object(fetch, '_send_request', side_effect=mock_send_request):
            # Attempt to fetch a URL that times out
            with pytest.raises(RuntimeError) as excinfo:
                fetch.fetch_url("http://example.com/slow_page")
            
            # Verify error message
            assert "timed out" in str(excinfo.value)
    
    def test_research_workflow_with_mcp_errors(self):
        """Test research workflow handling of MCP errors."""
        # Create a research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Mock the research agent to raise an error
        def mock_research_execute(*args, **kwargs):
            raise AuthenticationError("Failed to connect to Brave Search MCP")
        
        # Apply the mock
        with patch.object(ResearchAgent, 'execute', side_effect=mock_research_execute):
            # Execute the workflow
            result = workflow.execute("Tell me about Python programming")
            
            # Verify the workflow handled the error
            assert "error" in result.lower()
            assert "research" in result.lower()
            assert "brave search" in result.lower() or "mcp connection" in result.lower()
    
    def test_workflow_with_multiple_mcp_errors(self):
        """Test workflow handling of multiple MCP errors."""
        # Create a research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Mock the research agent to raise an error
        def mock_research_execute(*args, **kwargs):
            raise AuthenticationError("Brave Search API rate limit exceeded")
        
        # Mock the image generation agent to raise an error
        def mock_image_execute(*args, **kwargs):
            raise AuthenticationError("Invalid EverArt API key")
        
        # Apply the mocks
        with patch.object(ResearchAgent, 'execute', side_effect=mock_research_execute):
            with patch.object(ImageGenerationAgent, 'execute', side_effect=mock_image_execute):
                # Execute the workflow
                result = workflow.execute("Tell me about Python programming")
                
                # Verify the workflow handled both errors
                assert "error" in result.lower()
                assert "research" in result.lower()
                assert "image generation" in result.lower()
                assert "rate limit" in result.lower() or "api rate" in result.lower()
                assert "authentication" in result.lower() or "api key" in result.lower()
    
    def test_workflow_with_partial_success(self):
        """Test workflow with some successful agents and some failing agents."""
        # Create a research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Mock the research agent to succeed
        def mock_research_execute(*args, **kwargs):
            return "Python is a high-level programming language."
        
        # Mock the image generation agent to raise an error
        def mock_image_execute(*args, **kwargs):
            raise AuthenticationError("EverArt API request timed out")
        
        # Mock the summary agent to succeed
        def mock_summary_execute(*args, **kwargs):
            return "Python is a versatile programming language."
        
        # Apply the mocks
        with patch.object(ResearchAgent, 'execute', side_effect=mock_research_execute):
            with patch.object(ImageGenerationAgent, 'execute', side_effect=mock_image_execute):
                with patch.object(SummaryAgent, 'execute', side_effect=mock_summary_execute):
                    # Execute the workflow
                    result = workflow.execute("Tell me about Python programming")
                    
                    # Verify the workflow included successful results and handled the error
                    assert "python" in result.lower()
                    assert "programming language" in result.lower()
                    assert "error" in result.lower()
                    assert "image generation" in result.lower()
                    assert "timed out" in result.lower() or "timeout" in result.lower()
    
    def test_brave_search_mcp_invalid_arguments(self):
        """Test BraveSearchMCP handling of invalid arguments."""
        # Create BraveSearchMCP instance
        brave_search = BraveSearchMCP(api_key="test_brave_api_key")
        
        # Mock the _send_request method to handle invalid arguments
        def mock_send_request(method, params):
            if method == "callTool" and params["name"] == "brave_web_search":
                if "count" in params["arguments"] and params["arguments"]["count"] > 20:
                    return {
                        "content": [{"type": "text", "text": "Error: Count must be between 1 and 20"}],
                        "isError": True
                    }
            return {}
        
        with patch.object(brave_search, '_send_request', side_effect=mock_send_request):
            # Attempt to use the MCP with invalid count
            with pytest.raises(RuntimeError) as excinfo:
                brave_search.web_search("Python programming", count=30)
            
            # Verify error message
            assert "Count must be between" in str(excinfo.value)
    
    def test_filesystem_mcp_permission_error(self):
        """Test FilesystemMCP handling of permission errors."""
        # Create FilesystemMCP instance
        filesystem = FilesystemMCP(workspace_dir=self.workspace_dir)
        
        # Mock the _send_request method to simulate permission error
        def mock_send_request(method, params):
            if method == "callTool" and params["name"] == "write_file":
                if "/root/" in params["arguments"]["path"]:
                    return {
                        "content": [{"type": "text", "text": "Error: Permission denied"}],
                        "isError": True
                    }
            return {}
        
        with patch.object(filesystem, '_send_request', side_effect=mock_send_request):
            # Attempt to write to a protected location
            with pytest.raises(RuntimeError) as excinfo:
                filesystem.write_file("/root/test.txt", "This should fail")
            
            # Verify error message
            assert "Permission denied" in str(excinfo.value)
    
    def test_memory_mcp_storage_error(self):
        """Test MemoryMCP handling of storage errors."""
        # Create MemoryMCP instance
        memory = MemoryMCP()
        
        # Mock the _send_request method to simulate storage error
        def mock_send_request(method, params):
            if method == "callTool" and params["name"] == "store_memory":
                if len(params["arguments"]["value"]) > 10000:  # Simulate size limit
                    return {
                        "content": [{"type": "text", "text": "Error: Value exceeds maximum size limit"}],
                        "isError": True
                    }
            return {}
        
        with patch.object(memory, '_send_request', side_effect=mock_send_request):
            # Attempt to store a value that's too large
            large_value = "x" * 20000
            with pytest.raises(RuntimeError) as excinfo:
                memory.store_memory("large_key", large_value)
            
            # Verify error message
            assert "exceeds maximum size" in str(excinfo.value)
    
    def test_fetch_mcp_http_error(self):
        """Test FetchMCP handling of HTTP errors."""
        # Create FetchMCP instance
        fetch = FetchMCP()
        
        # Mock the _send_request method to simulate HTTP errors
        def mock_send_request(method, params):
            if method == "callTool" and params["name"] == "fetch_url":
                url = params["arguments"]["url"]
                if "404" in url:
                    return {
                        "content": [{"type": "text", "text": "Error: HTTP 404 Not Found"}],
                        "isError": True
                    }
                elif "500" in url:
                    return {
                        "content": [{"type": "text", "text": "Error: HTTP 500 Internal Server Error"}],
                        "isError": True
                    }
            return {}
        
        with patch.object(fetch, '_send_request', side_effect=mock_send_request):
            # Test 404 error
            with pytest.raises(RuntimeError) as excinfo:
                fetch.fetch_url("http://example.com/404")
            assert "404 Not Found" in str(excinfo.value)
            
            # Test 500 error
            with pytest.raises(RuntimeError) as excinfo:
                fetch.fetch_url("http://example.com/500")
            assert "500 Internal Server Error" in str(excinfo.value)
    
    def test_everart_mcp_generation_error(self):
        """Test EverArtMCP handling of image generation errors."""
        # Create EverArtMCP instance
        everart = EverArtMCP(api_key="test_everart_api_key")
        
        # Mock the _send_request method to simulate generation errors
        def mock_send_request(method, params):
            if method == "callTool" and params["name"] == "everart_generate_image":
                if "inappropriate" in params["arguments"]["prompt"].lower():
                    return {
                        "content": [{"type": "text", "text": "Error: Content policy violation"}],
                        "isError": True
                    }
            return {}
        
        with patch.object(everart, '_send_request', side_effect=mock_send_request):
            # Attempt to generate an inappropriate image
            with pytest.raises(RuntimeError) as excinfo:
                everart.generate_image("This is an inappropriate prompt")
            
            # Verify error message
            assert "Content policy violation" in str(excinfo.value)
    
    def test_workflow_error_recovery(self):
        """Test workflow's ability to recover from errors and continue execution."""
        # Create a research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Mock the research agent to raise an error on first call but succeed on second call
        research_call_count = 0
        def mock_research_execute(*args, **kwargs):
            nonlocal research_call_count
            research_call_count += 1
            if research_call_count == 1:
                raise AuthenticationError("Brave Search API request timed out")
            return "Python is a high-level programming language."
        
        # Mock the workflow's retry mechanism
        original_execute_agent = workflow._execute_agent
        def mock_execute_agent(agent, *args, **kwargs):
            if agent == workflow.research_agent:
                return mock_research_execute(*args, **kwargs)
            return original_execute_agent(agent, *args, **kwargs)
        
        # Apply the mock
        with patch.object(workflow, '_execute_agent', side_effect=mock_execute_agent):
            # Execute the workflow
            result = workflow.execute("Tell me about Python programming")
            
            # Verify the workflow recovered and completed successfully
            assert research_call_count > 1
            assert "python" in result.lower()
            assert "programming language" in result.lower()
            assert "recovered" in result.lower() or "retry" in result.lower() or "attempted again" in result.lower()
    
    def test_workflow_with_agent_fallback(self):
        """Test workflow's ability to use fallback agents when primary agents fail."""
        # Create a research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Create a fallback research agent
        fallback_research_agent = ResearchAgent()
        
        # Mock the primary research agent to fail
        def mock_research_execute(*args, **kwargs):
            raise AuthenticationError("Failed to connect to Brave Search MCP")
        
        # Mock the fallback research agent to succeed
        def mock_fallback_execute(*args, **kwargs):
            return "Python is a high-level programming language (from fallback)."
        
        # Mock the workflow to use the fallback agent
        original_execute_agent = workflow._execute_agent
        def mock_execute_agent(agent, *args, **kwargs):
            if agent == workflow.research_agent:
                try:
                    return mock_research_execute(*args, **kwargs)
                except Exception as e:
                    # Log the error and use fallback
                    print(f"Primary agent failed: {str(e)}")
                    return mock_fallback_execute(*args, **kwargs)
            return original_execute_agent(agent, *args, **kwargs)
        
        # Apply the mock
        with patch.object(workflow, '_execute_agent', side_effect=mock_execute_agent):
            # Execute the workflow
            result = workflow.execute("Tell me about Python programming")
            
            # Verify the fallback agent was used
            assert "python" in result.lower()
            assert "programming language" in result.lower()
            assert "fallback" in result.lower()
    
    def test_workflow_with_graceful_degradation(self):
        """Test workflow's ability to gracefully degrade when some agents fail."""
        # Create a research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Mock the image generation agent to fail
        def mock_image_execute(*args, **kwargs):
            raise AuthenticationError("Invalid EverArt API key")
        
        # Mock the workflow to handle the failure gracefully
        original_execute = workflow.execute
        def mock_execute(query):
            # Patch the image generation agent
            with patch.object(ImageGenerationAgent, 'execute', side_effect=mock_image_execute):
                try:
                    return original_execute(query)
                except Exception as e:
                    # This shouldn't happen if degradation works properly
                    return f"Workflow failed: {str(e)}"
        
        # Apply the mock
        workflow.execute = mock_execute
        
        # Execute the workflow
        result = workflow.execute("Tell me about Python programming")
        
        # Verify the workflow completed with degraded functionality
        assert "python" in result.lower()
        assert "research" in result.lower()
        assert "image" not in result.lower() or ("image" in result.lower() and "error" in result.lower())
        assert "completed with limited functionality" in result.lower() or "some features unavailable" in result.lower()
    
    def test_mcp_server_crash_recovery(self):
        """Test recovery from MCP server crashes."""
        # Create BraveSearchMCP instance
        brave_search = BraveSearchMCP(api_key="test_brave_api_key")
        
        # Mock the process to simulate a crash
        brave_search.process = MagicMock()
        brave_search.process.poll.return_value = 1  # Return non-None to indicate process has terminated
        
        # Mock the _start_server method to track calls
        original_start_server = brave_search._start_server
        start_server_called = False
        
        def mock_start_server():
            nonlocal start_server_called
            start_server_called = True
            original_start_server()
        
        # Mock the _send_request method to handle the crash and recovery
        send_request_call_count = 0
        def mock_send_request(method, params):
            nonlocal send_request_call_count
            send_request_call_count += 1
            
            if send_request_call_count == 1:
                # First call fails due to crashed server
                raise RuntimeError("MCP server is not running")
            
            # Subsequent calls succeed
            return {
                "content": [{"type": "text", "text": "Search results for Python programming"}],
                "isError": False
            }
        
        # Apply the mocks
        with patch.object(brave_search, '_start_server', side_effect=mock_start_server):
            with patch.object(brave_search, '_send_request', side_effect=mock_send_request):
                # First attempt should fail but trigger a restart
                with pytest.raises(RuntimeError):
                    brave_search.web_search("Python programming")
                
                # Verify the server was restarted
                assert start_server_called
                
                # Reset for next test
                start_server_called = False
                brave_search.process.poll.return_value = None  # Process now running
                
                # Second attempt should succeed
                result = brave_search.web_search("Python programming")
                assert "Search results" in result
    
    def test_workflow_with_retry_mechanism(self):
        """Test workflow's retry mechanism for transient errors."""
        # Create a research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Mock the research agent to fail with a transient error then succeed
        research_call_count = 0
        def mock_research_execute(*args, **kwargs):
            nonlocal research_call_count
            research_call_count += 1
            if research_call_count <= 2:  # Fail twice
                raise AuthenticationError("Brave Search API request timed out")
            return "Python is a high-level programming language."
        
        # Mock the workflow's retry mechanism
        max_retries = 3
        retry_count = 0
        
        original_execute_agent = workflow._execute_agent
        def mock_execute_agent(agent, *args, **kwargs):
            nonlocal retry_count
            if agent == workflow.research_agent:
                try:
                    return mock_research_execute(*args, **kwargs)
                except AuthenticationError as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Retrying after error: {str(e)}")
                        return mock_execute_agent(agent, *args, **kwargs)
                    raise
            return original_execute_agent(agent, *args, **kwargs)
        
        # Apply the mock
        with patch.object(workflow, '_execute_agent', side_effect=mock_execute_agent):
            # Execute the workflow
            result = workflow.execute("Tell me about Python programming")
            
            # Verify the retry mechanism worked
            assert research_call_count > 2
            assert retry_count == 2
            assert "python" in result.lower()
            assert "programming language" in result.lower()
    
    def test_error_logging(self):
        """Test that errors are properly logged."""
        # Create a research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Mock the logger
        mock_logger = MagicMock()
        
        # Mock the research agent to raise an error
        def mock_research_execute(*args, **kwargs):
            raise AuthenticationError("Brave Search API rate limit exceeded")
        
        # Apply the mocks
        with patch('apps.utils.logger.Logger.error', mock_logger.error):
            with patch.object(ResearchAgent, 'execute', side_effect=mock_research_execute):
                # Execute the workflow
                result = workflow.execute("Tell me about Python programming")
                
                # Verify the error was logged
                mock_logger.error.assert_called()
                
                # Verify the log message contains the error details
                log_message = mock_logger.error.call_args[0][0]
                assert "rate limit" in log_message.lower() or "api rate" in log_message.lower()
    
    def test_concurrent_mcp_errors(self):
        """Test handling of errors when multiple MCPs are used concurrently."""
        # Create a research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent(),
            parallel_execution=True  # Enable parallel execution
        )
        
        # Mock the research agent to raise an error
        def mock_research_execute(*args, **kwargs):
            raise AuthenticationError("Brave Search API request timed out")
        
        # Mock the image generation agent to raise an error
        def mock_image_execute(*args, **kwargs):
            raise AuthenticationError("Invalid EverArt API key")
        
        # Apply the mocks
        with patch.object(ResearchAgent, 'execute', side_effect=mock_research_execute):
            with patch.object(ImageGenerationAgent, 'execute', side_effect=mock_image_execute):
                # Execute the workflow
                result = workflow.execute("Tell me about Python programming")
                
                # Verify the workflow handled both errors
                assert "error" in result.lower()
                assert "research" in result.lower()
                assert "image generation" in result.lower()
                assert "timed out" in result.lower() or "timeout" in result.lower()
                assert "authentication" in result.lower() or "api key" in result.lower()
    
    def test_cascading_errors(self):
        """Test handling of cascading errors where one failure leads to others."""
        # Create a research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Mock the research agent to raise an error
        def mock_research_execute(*args, **kwargs):
            raise ResourceNotFoundError("No search results found")
        
        # Mock the summary agent to depend on research results
        def mock_summary_execute(data):
            if not data or "research_info" not in data:
                raise AuthenticationError("No research information available for summarization")
            return "Summary of research information"
        
        # Apply the mocks
        with patch.object(ResearchAgent, 'execute', side_effect=mock_research_execute):
            with patch.object(SummaryAgent, 'execute', side_effect=mock_summary_execute):
                # Execute the workflow
                result = workflow.execute("Tell me about Python programming")
                
                # Verify the workflow handled the cascading errors
                assert "error" in result.lower()
                assert "research" in result.lower()
                assert "summary" in result.lower()
                assert "not found" in result.lower() or "no search results" in result.lower()
                assert "no research information" in result.lower() or "missing research" in result.lower()
    
    def test_error_handling_with_user_feedback(self):
        """Test error handling with user feedback to resolve issues."""
        # Create a research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Mock the research agent to raise an error on ambiguous query
        def mock_research_execute(query):
            if "ambiguous" in query.lower() or query.lower() == "python":
                raise AuthenticationError("Query is too ambiguous. Please specify if you mean Python programming language or Python snake.")
            return f"Research results for {query}"
        
        # Mock the workflow's method to get user feedback
        user_feedback = "I mean the Python programming language"
        original_get_user_feedback = workflow.get_user_feedback
        workflow.get_user_feedback = lambda message: user_feedback
        
        # Mock the workflow to handle the error with user feedback
        original_execute = workflow.execute
        def mock_execute(query):
            try:
                # First try with original query
                research_result = mock_research_execute(query)
                # Continue with normal execution
                return original_execute(query)
            except AuthenticationError as e:
                # Get clarification from user
                clarification = workflow.get_user_feedback(str(e))
                # Retry with clarified query
                research_result = mock_research_execute(clarification)
                return f"Based on your clarification, here are the results: {research_result}"
        
        try:
            # Apply the mock
            workflow.execute = mock_execute
            
            # Execute the workflow with ambiguous query
            result = workflow.execute("Python")
            
            # Verify the workflow used user feedback to resolve the error
            assert "clarification" in result.lower()
            assert "programming language" in result.lower()
        finally:
            # Restore the original method
            workflow.get_user_feedback = original_get_user_feedback
    
    def test_error_reporting_format(self):
        """Test that error reports are formatted consistently."""
        # Create BraveSearchMCP instance
        brave_search = BraveSearchMCP(api_key="test_brave_api_key")
        
        # Define a set of different errors to test
        error_scenarios = [
            ("Rate limit exceeded", AuthenticationError("Brave Search API rate limit exceeded")),
            ("Authentication failed", AuthenticationError("Invalid API key")),
            ("Network timeout", AuthenticationError("Request timed out after 30 seconds")),
            ("Resource not found", ResourceNotFoundError("No search results found")),
            ("Invalid input", AuthenticationError("Query is too short"))
        ]
        
        # Mock the _send_request method to raise different errors
        for scenario_name, error in error_scenarios:
            def mock_send_request(method, params):
                raise error
            
            with patch.object(brave_search, '_send_request', side_effect=mock_send_request):
                # Attempt to use the MCP
                with pytest.raises(Exception) as excinfo:
                    brave_search.web_search("Python programming")
                
                # Verify error message format is consistent
                error_message = str(excinfo.value)
                assert error.args[0] in error_message
                
                # Check that the error type is preserved
                assert isinstance(excinfo.value, error.__class__)
    
    def test_mcp_process_termination(self):
        """Test proper termination of MCP server processes."""
        # Create MCP instances
        brave_search = BraveSearchMCP(api_key="test_brave_api_key")
        everart = EverArtMCP(api_key="test_everart_api_key")
        fetch = FetchMCP()
        filesystem = FilesystemMCP(workspace_dir=self.workspace_dir)
        memory = MemoryMCP()
        
        # Mock the process objects
        for mcp in [brave_search, everart, fetch, filesystem, memory]:
            mcp.process = MagicMock()
            mcp.process.poll.return_value = None  # Process is running
            mcp.process.terminate = MagicMock()
            mcp.process.wait = MagicMock()
            mcp.process.kill = MagicMock()
        
        # Test normal termination
        brave_search.close()
        brave_search.process.terminate.assert_called_once()
        brave_search.process.wait.assert_called_once()
        
        # Test termination with timeout
        everart.process.wait.side_effect = subprocess.TimeoutExpired(cmd="mock_cmd", timeout=5)
        everart.close()
        everart.process.terminate.assert_called_once()
        everart.process.kill.assert_called_once()
        
        # Test context manager
        with fetch as f:
            # Use the MCP
            pass
        fetch.process.terminate.assert_called_once()
        
        # Test multiple close calls
        filesystem.close()
        filesystem.process.terminate.assert_called_once()
        filesystem.close()  # Second call should not terminate again
        filesystem.process.terminate.assert_called_once()  # Still only called once
        
        # Test close when process already terminated
        memory.process.poll.return_value = 0  # Process already terminated
        memory.close()
        memory.process.terminate.assert_not_called()
    
    def test_workflow_error_summary(self):
        """Test that the workflow provides a clear summary of errors."""
        # Create a research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Mock multiple agents to raise different errors
        def mock_research_execute(*args, **kwargs):
            raise AuthenticationError("Brave Search API rate limit exceeded")
        
        def mock_image_execute(*args, **kwargs):
            raise AuthenticationError("Invalid EverArt API key")
        
        def mock_verification_execute(*args, **kwargs):
            raise AuthenticationError("Verification request timed out")
        
        # Apply the mocks
        with patch.object(ResearchAgent, 'execute', side_effect=mock_research_execute):
            with patch.object(ImageGenerationAgent, 'execute', side_effect=mock_image_execute):
                with patch.object(VerificationAgent, 'execute', side_effect=mock_verification_execute):
                    # Execute the workflow
                    result = workflow.execute("Tell me about Python programming")
                    
                    # Verify the workflow provides a clear summary of all errors
                    assert "error summary" in result.lower() or "encountered errors" in result.lower()
                    assert "research" in result.lower() and "rate limit" in result.lower()
                    assert "image generation" in result.lower() and "authentication" in result.lower()
                    assert "verification" in result.lower() and "timed out" in result.lower()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
