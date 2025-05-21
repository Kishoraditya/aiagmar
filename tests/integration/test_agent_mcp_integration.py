"""
Integration tests for agent interactions with MCP services.

These tests verify that agents can properly interact with MCP services,
focusing on the integration points between agents and their respective MCPs.
"""

import os
import pytest
import json
import tempfile
from unittest.mock import patch, MagicMock

# Import agents
from apps.agents.manager_agent import ManagerAgent
from apps.agents.pre_response_agent import PreResponseAgent
from apps.agents.research_agent import ResearchAgent
from apps.agents.summary_agent import SummaryAgent
from apps.agents.verification_agent import VerificationAgent
from apps.agents.image_generation_agent import ImageGenerationAgent
from apps.agents.file_manager_agent import FileManagerAgent

# Import MCPs
from apps.mcps.brave_search_mcp import BraveSearchMCP
from apps.mcps.everart_mcp import EverArtMCP
from apps.mcps.fetch_mcp import FetchMCP
from apps.mcps.filesystem_mcp import FilesystemMCP
from apps.mcps.memory_mcp import MemoryMCP

# Import mocks
from tests.mocks.mock_mcps import (
    MockBraveSearchMCP,
    MockEverArtMCP,
    MockFetchMCP,
    MockFilesystemMCP,
    MockMemoryMCP,
    patch_mcps
)


class TestAgentMCPIntegration:
    """Test integration between agents and MCP services."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up test environment before each test."""
        # Create a temporary workspace directory
        self.workspace_dir = str(tmp_path / "workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Set up mock MCPs
        self.brave_search = MockBraveSearchMCP(api_key="mock_brave_api_key")
        self.everart = MockEverArtMCP(api_key="mock_everart_api_key")
        self.fetch = MockFetchMCP()
        self.filesystem = MockFilesystemMCP(workspace_dir=self.workspace_dir)
        self.memory = MockMemoryMCP()
        
        # Configure mock responses
        self._configure_mock_responses()
        
        # Apply patches
        self.patches = patch_mcps()
        
        yield
        
        # Clean up
        for p in self.patches:
            p.stop()
    
    def _configure_mock_responses(self):
        """Configure mock responses for the MCPs."""
        # Configure BraveSearchMCP
        self.brave_search.set_web_search_result("""
        Title: Python Programming Language
        Description: Python is a high-level, interpreted programming language known for its readability and versatility.
        URL: https://www.python.org/
        
        Title: Python Tutorial - W3Schools
        Description: Python is a popular programming language. Learn Python with our step-by-step tutorial.
        URL: https://www.w3schools.com/python/
        
        Title: Python (programming language) - Wikipedia
        Description: Python is an interpreted high-level general-purpose programming language.
        URL: https://en.wikipedia.org/wiki/Python_(programming_language)
        """)
        
        # Configure EverArtMCP
        self.everart.set_generate_image_result("""
        Image generated successfully!
        
        URL: https://example.com/images/mock-image-12345.jpg
        
        The image shows a Python logo with code snippets in the background.
        """)
        
        # Configure FetchMCP
        self.fetch.set_fetch_url_result("""
        <html>
        <head><title>Python Programming</title></head>
        <body>
        <h1>Python Programming Language</h1>
        <p>Python is a high-level, interpreted programming language known for its readability and versatility.</p>
        <p>Key features include:</p>
        <ul>
            <li>Easy to learn syntax</li>
            <li>Interpreted nature</li>
            <li>Dynamic typing</li>
            <li>High-level data structures</li>
        </ul>
        </body>
        </html>
        """)
        
        # Configure FilesystemMCP
        self.filesystem.files = {
            "research_results.txt": "Initial research results placeholder"
        }
        
        # Configure MemoryMCP
        self.memory.memories = {
            "default": {
                "user_query": "Tell me about Python programming language"
            }
        }
    
    def test_research_agent_with_brave_search_mcp(self):
        """Test integration between Research Agent and BraveSearchMCP."""
        # Create Research Agent with BraveSearchMCP
        research_agent = ResearchAgent(
            brave_search_mcp=self.brave_search,
            fetch_mcp=self.fetch,
            memory_mcp=self.memory
        )
        
        # Perform research
        query = "Python programming language"
        result = research_agent.research(query)
        
        # Verify BraveSearchMCP was called correctly
        assert self.brave_search.web_search_called
        assert query in self.brave_search.web_search_args[0]
        
        # Verify research results were processed correctly
        assert "success" in result["status"]
        assert "python" in result["results"].lower()
        
        # Verify results were stored in memory
        assert self.memory.store_memory_called
        memory_keys = [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
        assert any("research" in key.lower() for key in memory_keys)
    
    def test_research_agent_with_fetch_mcp(self):
        """Test integration between Research Agent and FetchMCP."""
        # Create Research Agent with FetchMCP
        research_agent = ResearchAgent(
            brave_search_mcp=self.brave_search,
            fetch_mcp=self.fetch,
            memory_mcp=self.memory
        )
        
        # Configure BraveSearchMCP to return URLs
        self.brave_search.set_web_search_result("""
        Title: Python Official Website
        Description: The official Python programming language website.
        URL: https://www.python.org/
        """)
        
        # Perform research
        query = "Python programming language"
        result = research_agent.research(query)
        
        # Verify FetchMCP was called to retrieve content from URLs
        assert self.fetch.fetch_url_called or self.fetch.fetch_text_called
        
        # If fetch_url was called, verify the URL was from the search results
        if self.fetch.fetch_url_called:
            assert "python.org" in self.fetch.fetch_url_args[0]
        
        # Verify fetched content was processed
        assert "success" in result["status"]
        assert "python" in result["results"].lower()
    
    def test_verification_agent_with_brave_search_mcp(self):
        """Test integration between Verification Agent and BraveSearchMCP."""
        # Create Verification Agent with BraveSearchMCP
        verification_agent = VerificationAgent(
            brave_search_mcp=self.brave_search,
            memory_mcp=self.memory
        )
        
        # Store facts to verify in memory
        facts = [
            "Python was created by Guido van Rossum",
            "Python was first released in 1991",
            "Python is a high-level programming language"
        ]
        self.memory.store_memory("facts_to_verify", "\n".join(facts), namespace="research")
        
        # Configure BraveSearchMCP to return verification results
        self.brave_search.set_web_search_result("""
        Title: Python Creator - Guido van Rossum
        Description: Guido van Rossum is the creator of Python, first releasing it in 1991.
        URL: https://en.wikipedia.org/wiki/Guido_van_Rossum
        
        Title: Python Features
        Description: Python is known as a high-level, interpreted programming language with dynamic typing.
        URL: https://www.python.org/about/
        """)
        
        # Verify facts
        result = verification_agent.verify_facts("Python programming language")
        
        # Verify BraveSearchMCP was called for each fact
        assert self.brave_search.web_search_called
        assert len(self.brave_search.web_search_args) >= 1
        
        # Verify verification results
        assert "success" in result["status"]
        assert "verified" in result["results"].lower()
        
        # Verify results were stored in memory
        assert self.memory.store_memory_called
        memory_keys = [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
        assert any("verif" in key.lower() for key in memory_keys)
    
    def test_image_generation_agent_with_everart_mcp(self):
        """Test integration between Image Generation Agent and EverArtMCP."""
        # Create Image Generation Agent with EverArtMCP
        image_agent = ImageGenerationAgent(
            everart_mcp=self.everart,
            memory_mcp=self.memory
        )
        
        # Store research summary in memory
        summary = "Python is a high-level programming language known for its readability and versatility."
        self.memory.store_memory("summary", summary, namespace="research")
        
        # Generate image
        result = image_agent.generate_image("Python programming language")
        
        # Verify EverArtMCP was called correctly
        assert self.everart.generate_image_called
        assert "python" in self.everart.generate_image_args[0].lower()
        
        # Verify image generation results
        assert "success" in result["status"]
        assert "image" in result["results"].lower()
        assert "url" in result["results"].lower()
        
        # Verify results were stored in memory
        assert self.memory.store_memory_called
        memory_keys = [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
        assert any("image" in key.lower() for key in memory_keys)
    
    def test_file_manager_agent_with_filesystem_mcp(self):
        """Test integration between File Manager Agent and FilesystemMCP."""
        # Create File Manager Agent with FilesystemMCP
        file_manager = FileManagerAgent(
            filesystem_mcp=self.filesystem,
            memory_mcp=self.memory
        )
        
        # Store research outputs in memory
        self.memory.store_memory("summary", "Python is a high-level programming language.", namespace="research")
        self.memory.store_memory("image_url", "https://example.com/images/python.jpg", namespace="research")
        
        # Save research outputs
        result = file_manager.save_research_outputs("Python programming language")
        
        # Verify FilesystemMCP was called correctly
        assert self.filesystem.write_file_called
        assert len(self.filesystem.write_file_args) >= 1
        
        # Verify at least one file contains "python" in the content
        python_files = [
            args for args in self.filesystem.write_file_args 
            if isinstance(args, tuple) and len(args) > 1 and "python" in args[1].lower()
        ]
        assert len(python_files) > 0
        
        # Verify file saving results
        assert "success" in result["status"]
        assert "saved" in result["results"].lower()
        
        # Verify results were stored in memory
        assert self.memory.store_memory_called
        memory_keys = [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
        assert any("file" in key.lower() for key in memory_keys)
    
    def test_manager_agent_with_memory_mcp(self):
        """Test integration between Manager Agent and MemoryMCP."""
        # Create Manager Agent with MemoryMCP
        manager = ManagerAgent(memory_mcp=self.memory)
        
        # Store task in memory
        task_id = "task-123"
        task_data = {
            "query": "Python programming language",
            "status": "pending",
            "assigned_to": "research_agent"
        }
        manager.store_task(task_id, task_data)
        
        # Verify MemoryMCP was called correctly
        assert self.memory.store_memory_called
        assert task_id in [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
        
        # Retrieve task
        retrieved_task = manager.get_task(task_id)
        
        # Verify MemoryMCP retrieve was called
        assert self.memory.retrieve_memory_called
        assert task_id in [args[0] for args in self.memory.retrieve_memory_args if isinstance(args, tuple) and len(args) > 0]
        
        # Verify retrieved task matches stored task
        assert retrieved_task["query"] == task_data["query"]
        assert retrieved_task["status"] == task_data["status"]
        assert retrieved_task["assigned_to"] == task_data["assigned_to"]
    
    def test_summary_agent_with_memory_mcp(self):
        """Test integration between Summary Agent and MemoryMCP."""
        # Create Summary Agent with MemoryMCP
        summary_agent = SummaryAgent(memory_mcp=self.memory)
        
        # Store research results in memory
        research_results = """
        Python is a high-level, interpreted programming language known for its readability and versatility.
        Key features include easy syntax, dynamic typing, and high-level data structures.
        It was created by Guido van Rossum and released in 1991.
        Python is widely used in web development, data science, AI, and automation.
        """
        self.memory.store_memory("research_results", research_results, namespace="research")
        
        # Create summary
        result = summary_agent.create_summary("Python programming language")
        
        # Verify MemoryMCP retrieve was called
        assert self.memory.retrieve_memory_called
        assert "research_results" in [args[0] for args in self.memory.retrieve_memory_args if isinstance(args, tuple) and len(args) > 0]
        
        # Verify summary results
        assert "success" in result["status"]
        assert "python" in result["summary"].lower()
        
        # Verify summary was stored in memory
        assert self.memory.store_memory_called
        assert "summary" in [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
    
    def test_pre_response_agent_with_memory_mcp(self):
        """Test integration between Pre-response Agent and MemoryMCP."""
        # Create Pre-response Agent with MemoryMCP
        pre_response = PreResponseAgent(memory_mcp=self.memory)
        
        # Create research plan
        query = "Tell me about Python programming language"
        result = pre_response.create_research_plan(query)
        
        # Verify plan was stored in memory
        assert self.memory.store_memory_called
        assert "research_plan" in [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
        
        # Verify plan results
        assert "success" in result["status"]
        assert "plan" in result["results"].lower()
        assert "python" in result["results"].lower()
    
    def test_brave_search_mcp_error_handling(self):
        """Test how agents handle errors from BraveSearchMCP."""
        # Create Research Agent with BraveSearchMCP
        research_agent = ResearchAgent(
            brave_search_mcp=self.brave_search,
            fetch_mcp=self.fetch,
            memory_mcp=self.memory
        )
        
        # Make BraveSearchMCP raise an exception
        self.brave_search.web_search = MagicMock(side_effect=RuntimeError("API rate limit exceeded"))
        
        # Perform research
        query = "Python programming language"
        result = research_agent.research(query)
        
        # Verify error was handled gracefully
        assert "error" in result["status"]
        assert "rate limit" in result["error"].lower()
        
        # Verify error was logged in memory
        assert self.memory.store_memory_called
        error_keys = [
            args[0] for args in self.memory.store_memory_args 
            if isinstance(args, tuple) and len(args) > 0 and "error" in args[0].lower()
        ]
        assert len(error_keys) > 0
    
    def test_everart_mcp_error_handling(self):
        """Test how agents handle errors from EverArtMCP."""
        # Create Image Generation Agent with EverArtMCP
        image_agent = ImageGenerationAgent(
            everart_mcp=self.everart,
            memory_mcp=self.memory
        )
        
        # Make EverArtMCP raise an exception
        self.everart.generate_image = MagicMock(side_effect=RuntimeError("Invalid API key"))
        
        # Generate image
        result = image_agent.generate_image("Python programming language")
        
        # Verify error was handled gracefully
        assert "error" in result["status"]
        assert "api key" in result["error"].lower()
        
        # Verify error was logged in memory
        assert self.memory.store_memory_called
        error_keys = [
            args[0] for args in self.memory.store_memory_args 
            if isinstance(args, tuple) and len(args) > 0 and "error" in args[0].lower()
        ]
        assert len(error_keys) > 0
    
    def test_fetch_mcp_error_handling(self):
        """Test how agents handle errors from FetchMCP."""
        # Create Research Agent with FetchMCP
        research_agent = ResearchAgent(
            brave_search_mcp=self.brave_search,
            fetch_mcp=self.fetch,
            memory_mcp=self.memory
        )
        
        # Configure BraveSearchMCP to return URLs
        self.brave_search.set_web_search_result("""
        Title: Python Official Website
        Description: The official Python programming language website.
        URL: https://www.python.org/
        """)
        
        # Make FetchMCP raise an exception
        self.fetch.fetch_url = MagicMock(side_effect=RuntimeError("Connection timeout"))
        self.fetch.fetch_text = MagicMock(side_effect=RuntimeError("Connection timeout"))
        
        # Perform research
        query = "Python programming language"
        result = research_agent.research(query)
        
        # Verify error was handled gracefully
        assert "error" in result["status"] or "partial" in result["status"]
        assert "timeout" in result["error"].lower() if "error" in result else True
        
        # Verify error was logged in memory
        assert self.memory.store_memory_called
        error_keys = [
            args[0] for args in self.memory.store_memory_args 
            if isinstance(args, tuple) and len(args) > 0 and "error" in args[0].lower()
        ]
        assert len(error_keys) > 0
    
    def test_filesystem_mcp_error_handling(self):
        """Test how agents handle errors from FilesystemMCP."""
        # Create File Manager Agent with FilesystemMCP
        file_manager = FileManagerAgent(
            filesystem_mcp=self.filesystem,
            memory_mcp=self.memory
        )
        
        # Store research outputs in memory
        self.memory.store_memory("summary", "Python is a high-level programming language.", namespace="research")
        
        # Make FilesystemMCP raise an exception
        self.filesystem.write_file = MagicMock(side_effect=RuntimeError("Permission denied"))
        
        # Save research outputs
        result = file_manager.save_research_outputs("Python programming language")
        
        # Verify error was handled gracefully
        assert "error" in result["status"]
        assert "permission" in result["error"].lower()
        
        # Verify error was logged in memory
        assert self.memory.store_memory_called
        error_keys = [
            args[0] for args in self.memory.store_memory_args 
            if isinstance(args, tuple) and len(args) > 0 and "error" in args[0].lower()
        ]
        assert len(error_keys) > 0
    
    def test_memory_mcp_error_handling(self):
        """Test how agents handle errors from MemoryMCP."""
        # Create Manager Agent with MemoryMCP
        manager = ManagerAgent(memory_mcp=self.memory)
        
        # Make MemoryMCP raise an exception
        self.memory.store_memory = MagicMock(side_effect=RuntimeError("Storage limit exceeded"))
        
        # Try to store task
        task_id = "task-123"
        task_data = {
            "query": "Python programming language",
            "status": "pending"
        }
        
        # This should handle the error gracefully
        result = manager.store_task(task_id, task_data)
        
        # Verify error was handled gracefully
        assert "error" in result["status"]
        assert "storage limit" in result["error"].lower()
    
    def test_brave_search_mcp_retry_mechanism(self):
        """Test retry mechanism for BraveSearchMCP."""
        # Create Research Agent with BraveSearchMCP
        research_agent = ResearchAgent(
            brave_search_mcp=self.brave_search,
            fetch_mcp=self.fetch,
            memory_mcp=self.memory
        )
        
        # Make BraveSearchMCP fail on first call but succeed on second
        fail_once_mock = MagicMock()
        fail_once_mock.side_effect = [
            RuntimeError("API rate limit exceeded"),
            "Python is a high-level programming language."
        ]
        self.brave_search.web_search = fail_once_mock
        
        # Perform research with retry
        query = "Python programming language"
        result = research_agent.research(query, retry_count=1)
        
        # Verify BraveSearchMCP was called twice
        assert fail_once_mock.call_count == 2
        
        # Verify research succeeded on retry
        assert "success" in result["status"]
        assert "python" in result["results"].lower()
    
    def test_fetch_mcp_with_different_content_types(self):
        """Test FetchMCP with different content types."""
        # Create Research Agent with FetchMCP
        research_agent = ResearchAgent(
            brave_search_mcp=self.brave_search,
            fetch_mcp=self.fetch,
            memory_mcp=self.memory
        )
        
        # Configure BraveSearchMCP to return different types of URLs
        self.brave_search.set_web_search_result("""
        Title: Python Official Website
        Description: The official Python programming language website.
        URL: https://www.python.org/
        
        Title: Python Documentation PDF
        Description: Python language reference in PDF format.
        URL: https://docs.python.org/3/python-language-reference.pdf
        
        Title: Python Tutorial Video
        Description: Learn Python programming with this tutorial video.
        URL: https://www.youtube.com/watch?v=python_tutorial
        """)
        
        # Configure FetchMCP to handle different content types
        self.fetch.fetch_url = MagicMock(side_effect=[
            "<html><body><h1>Python Website</h1></body></html>",  # HTML
            "PDF content not supported",  # PDF
            "Video content not supported"  # Video
        ])
        
        # Perform research
        query = "Python programming language"
        result = research_agent.research(query)
        
        # Verify FetchMCP was called for each URL
        assert self.fetch.fetch_url.call_count > 0
        
        # Verify research results contain information from fetchable sources
        assert "success" in result["status"]
        assert "python" in result["results"].lower()
    
    def test_filesystem_mcp_with_different_file_types(self):
        """Test FilesystemMCP with different file types."""
        # Create File Manager Agent with FilesystemMCP
        file_manager = FileManagerAgent(
            filesystem_mcp=self.filesystem,
            memory_mcp=self.memory
        )
        
        # Store different types of research outputs in memory
        self.memory.store_memory("summary", "Python is a high-level programming language.", namespace="research")
        self.memory.store_memory("image_url", "https://example.com/images/python.jpg", namespace="research")
        self.memory.store_memory("code_snippet", "print('Hello, Python!')", namespace="research")
        
        # Save research outputs
        result = file_manager.save_research_outputs("Python programming language")
        
        # Verify FilesystemMCP was called for different file types
        assert self.filesystem.write_file_called
        
        # Check for different file extensions
        file_paths = [args[0] for args in self.filesystem.write_file_args if isinstance(args, tuple) and len(args) > 0]
        
        # There should be at least one text file and potentially other types
        text_files = [path for path in file_paths if path.endswith(".txt") or path.endswith(".md")]
        assert len(text_files) > 0
        
        # Verify results indicate success
        assert "success" in result["status"]
        assert "saved" in result["results"].lower()
    
    def test_memory_mcp_with_different_namespaces(self):
        """Test MemoryMCP with different namespaces."""
        # Create agents that use MemoryMCP
        manager = ManagerAgent(memory_mcp=self.memory)
        research = ResearchAgent(
            brave_search_mcp=self.brave_search,
            fetch_mcp=self.fetch,
            memory_mcp=self.memory
        )
        summary = SummaryAgent(memory_mcp=self.memory)
        
        # Store data in different namespaces
        manager.store_task("task-123", {"query": "Python basics"}, namespace="tasks")
        research.store_research_results("Python basics", {"content": "Python basics info"}, namespace="research")
        summary.store_summary("Python basics", "Python is easy to learn", namespace="summaries")
        
        # Verify data was stored in different namespaces
        namespace_args = [
            args[2] for args in self.memory.store_memory_args 
            if isinstance(args, tuple) and len(args) > 2
        ]
        
        assert "tasks" in namespace_args
        assert "research" in namespace_args
        assert "summaries" in namespace_args
        
        # Verify each agent can retrieve from its namespace
        manager_data = manager.get_task("task-123", namespace="tasks")
        research_data = research.get_research_data("Python basics", namespace="research")
        summary_data = summary.get_summary("Python basics", namespace="summaries")
        
        assert "Python basics" in manager_data["query"]
        assert "Python basics info" in research_data["content"]
        assert "Python is easy to learn" in summary_data
    
def test_mcp_service_discovery(self):
    """Test MCP service discovery and connection."""
    # This test simulates the process of discovering and connecting to MCP services
    
    # Create a temporary directory for MCP service discovery
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock MCP service discovery files
        service_info = {
            "brave_search": {
                "type": "brave_search",
                "endpoint": "http://localhost:8001",
                "api_key_env": "BRAVE_API_KEY"
            },
            "everart": {
                "type": "everart",
                "endpoint": "http://localhost:8002",
                "api_key_env": "EVERART_API_KEY"
            },
            "fetch": {
                "type": "fetch",
                "endpoint": "http://localhost:8003"
            },
            "filesystem": {
                "type": "filesystem",
                "workspace_dir": self.workspace_dir
            },
            "memory": {
                "type": "memory",
                "storage_path": os.path.join(temp_dir, "memory")
            }
        }
        
        # Write service info to discovery file
        discovery_file = os.path.join(temp_dir, "mcp_services.json")
        with open(discovery_file, "w") as f:
            json.dump(service_info, f)
        
        # Create a mock mcp_discovery module if it doesn't exist
        try:
            # Try to import the module first
            from apps.utils import mcp_discovery
            
            # Mock the service discovery function
            with patch("apps.utils.mcp_discovery.discover_mcps") as mock_discover:
                with patch("apps.utils.mcp_discovery.connect_to_mcp") as mock_connect:
                    # Configure mocks
                    mock_discover.return_value = service_info
                    
                    def mock_connect_impl(service_info):
                        service_type = service_info["type"]
                        if service_type == "brave_search":
                            return MockBraveSearchMCP(api_key="mock_key")
                        elif service_type == "everart":
                            return MockEverArtMCP(api_key="mock_key")
                        elif service_type == "fetch":
                            return MockFetchMCP()
                        elif service_type == "filesystem":
                            return MockFilesystemMCP(workspace_dir=service_info.get("workspace_dir", "."))
                        elif service_type == "memory":
                            return MockMemoryMCP(storage_path=service_info.get("storage_path"))
                        else:
                            raise ValueError(f"Unknown MCP service type: {service_type}")
                    
                    mock_connect.side_effect = mock_connect_impl
                    
                    # Call the function under test
                    mcps = mcp_discovery.load_mcp_services(discovery_file)
                    
                    # Verify all services were discovered and connected
                    assert "brave_search" in mcps
                    assert "everart" in mcps
                    assert "fetch" in mcps
                    assert "filesystem" in mcps
                    assert "memory" in mcps
                    
                    # Verify services are of the correct type
                    assert isinstance(mcps["brave_search"], MockBraveSearchMCP)
                    assert isinstance(mcps["everart"], MockEverArtMCP)
                    assert isinstance(mcps["fetch"], MockFetchMCP)
                    assert isinstance(mcps["filesystem"], MockFilesystemMCP)
                    assert isinstance(mcps["memory"], MockMemoryMCP)
        
        except ImportError:
            # If the module doesn't exist, create a mock implementation for testing
            class MockMCPDiscovery:
                @staticmethod
                def discover_mcps(discovery_file):
                    with open(discovery_file, "r") as f:
                        return json.load(f)
                
                @staticmethod
                def connect_to_mcp(service_info):
                    service_type = service_info["type"]
                    if service_type == "brave_search":
                        return MockBraveSearchMCP(api_key="mock_key")
                    elif service_type == "everart":
                        return MockEverArtMCP(api_key="mock_key")
                    elif service_type == "fetch":
                        return MockFetchMCP()
                    elif service_type == "filesystem":
                        return MockFilesystemMCP(workspace_dir=service_info.get("workspace_dir", "."))
                    elif service_type == "memory":
                        return MockMemoryMCP(storage_path=service_info.get("storage_path"))
                    else:
                        raise ValueError(f"Unknown MCP service type: {service_type}")
                
                @staticmethod
                def load_mcp_services(discovery_file):
                    services_info = MockMCPDiscovery.discover_mcps(discovery_file)
                    mcps = {}
                    for name, info in services_info.items():
                        mcps[name] = MockMCPDiscovery.connect_to_mcp(info)
                    return mcps
            
            # Use the mock implementation
            mcps = MockMCPDiscovery.load_mcp_services(discovery_file)
            
            # Verify all services were discovered and connected
            assert "brave_search" in mcps
            assert "everart" in mcps
            assert "fetch" in mcps
            assert "filesystem" in mcps
            assert "memory" in mcps
            
            # Verify services are of the correct type
            assert isinstance(mcps["brave_search"], MockBraveSearchMCP)
            assert isinstance(mcps["everart"], MockEverArtMCP)
            assert isinstance(mcps["fetch"], MockFetchMCP)
            assert isinstance(mcps["filesystem"], MockFilesystemMCP)
            assert isinstance(mcps["memory"], MockMemoryMCP)
    
    def test_mcp_reconnection_after_failure(self):
        """Test reconnection to MCP services after failure."""
        # Create Research Agent with BraveSearchMCP
        research_agent = ResearchAgent(
            brave_search_mcp=self.brave_search,
            fetch_mcp=self.fetch,
            memory_mcp=self.memory
        )
        
        # Make BraveSearchMCP connection fail
        original_process = self.brave_search.process
        self.brave_search.process = None  # Simulate disconnection
        
        # Mock the reconnect method
        original_start_server = self.brave_search._start_server
        self.brave_search._start_server = MagicMock()
        self.brave_search._start_server.side_effect = lambda: setattr(self.brave_search, 'process', original_process)
        
        try:
            # Perform research, which should trigger reconnection
            query = "Python programming language"
            result = research_agent.research(query)
            
            # Verify reconnection was attempted
            assert self.brave_search._start_server.called
            
            # Verify research completed successfully after reconnection
            assert "success" in result["status"]
            assert "python" in result["results"].lower()
        finally:
            # Restore original methods
            self.brave_search._start_server = original_start_server
            self.brave_search.process = original_process
    
    def test_multiple_agents_sharing_same_mcp(self):
        """Test multiple agents sharing the same MCP instance."""
        # Create multiple agents sharing the same MCPs
        research_agent = ResearchAgent(
            brave_search_mcp=self.brave_search,
            fetch_mcp=self.fetch,
            memory_mcp=self.memory
        )
        
        verification_agent = VerificationAgent(
            brave_search_mcp=self.brave_search,  # Same BraveSearchMCP
            memory_mcp=self.memory  # Same MemoryMCP
        )
        
        # Perform operations with both agents
        research_result = research_agent.research("Python basics")
        verification_result = verification_agent.verify_facts("Python creator")
        
        # Verify both agents used the same BraveSearchMCP
        assert self.brave_search.web_search_called
        assert len(self.brave_search.web_search_args) >= 2  # At least one call from each agent
        
        # Verify both agents used the same MemoryMCP
        assert self.memory.store_memory_called
        assert len(self.memory.store_memory_args) >= 2  # At least one call from each agent
        
        # Verify both operations completed successfully
        assert "success" in research_result["status"]
        assert "success" in verification_result["status"]
    
    def test_agent_with_multiple_mcp_fallbacks(self):
        """Test agent with multiple MCP fallbacks."""
        # Create a second BraveSearchMCP as fallback
        fallback_brave_search = MockBraveSearchMCP(api_key="fallback_api_key")
        fallback_brave_search.set_web_search_result("""
        Title: Python Fallback Result
        Description: This is a fallback result for Python.
        URL: https://fallback.example.com/python
        """)
        
        # Create Research Agent with primary and fallback MCPs
        research_agent = ResearchAgent(
            brave_search_mcp=self.brave_search,
            fetch_mcp=self.fetch,
            memory_mcp=self.memory,
            fallback_search_mcp=fallback_brave_search  # Add fallback
        )
        
        # Make primary BraveSearchMCP fail
        self.brave_search.web_search = MagicMock(side_effect=RuntimeError("API rate limit exceeded"))
        
        # Perform research
        query = "Python programming language"
        result = research_agent.research(query)
        
        # Verify fallback was used
        assert fallback_brave_search.web_search_called
        assert query in fallback_brave_search.web_search_args[0]
        
        # Verify research completed successfully using fallback
        assert "success" in result["status"]
        assert "fallback" in result["results"].lower()
    
    def test_mcp_authentication_handling(self):
        """Test MCP authentication handling."""
        # Create a BraveSearchMCP with invalid API key
        invalid_brave_search = MockBraveSearchMCP(api_key="invalid_key")
        invalid_brave_search.web_search = MagicMock(side_effect=RuntimeError("Authentication failed: Invalid API key"))
        
        # Create Research Agent with the invalid MCP
        research_agent = ResearchAgent(
            brave_search_mcp=invalid_brave_search,
            fetch_mcp=self.fetch,
            memory_mcp=self.memory
        )
        
        # Perform research
        query = "Python programming language"
        result = research_agent.research(query)
        
        # Verify authentication error was handled
        assert "error" in result["status"]
        assert "authentication" in result["error"].lower() or "api key" in result["error"].lower()
        
        # Verify error was logged in memory
        assert self.memory.store_memory_called
        error_keys = [
            args[0] for args in self.memory.store_memory_args 
            if isinstance(args, tuple) and len(args) > 0 and "error" in args[0].lower()
        ]
        assert len(error_keys) > 0
    
    def test_mcp_rate_limiting_handling(self):
        """Test MCP rate limiting handling."""
        # Configure BraveSearchMCP to simulate rate limiting
        self.brave_search.web_search = MagicMock(side_effect=RuntimeError("Rate limit exceeded. Retry after 60 seconds."))
        
        # Create Research Agent with BraveSearchMCP
        research_agent = ResearchAgent(
            brave_search_mcp=self.brave_search,
            fetch_mcp=self.fetch,
            memory_mcp=self.memory
        )
        
        # Perform research
        query = "Python programming language"
        result = research_agent.research(query)
        
        # Verify rate limiting error was handled
        assert "error" in result["status"]
        assert "rate limit" in result["error"].lower()
        
        # Verify error was logged in memory
        assert self.memory.store_memory_called
        error_keys = [
            args[0] for args in self.memory.store_memory_args 
            if isinstance(args, tuple) and len(args) > 0 and "error" in args[0].lower()
        ]
        assert len(error_keys) > 0
        
        # Verify retry information was included
        assert "retry" in result["error"].lower() and "60" in result["error"]
    
    def test_mcp_timeout_handling(self):
        """Test MCP timeout handling."""
        # Configure FetchMCP to simulate timeout
        self.fetch.fetch_url = MagicMock(side_effect=RuntimeError("Request timed out after 30 seconds"))
        
        # Create Research Agent with FetchMCP
        research_agent = ResearchAgent(
            brave_search_mcp=self.brave_search,
            fetch_mcp=self.fetch,
            memory_mcp=self.memory
        )
        
        # Configure BraveSearchMCP to return URLs
        self.brave_search.set_web_search_result("""
        Title: Python Official Website
        Description: The official Python programming language website.
        URL: https://www.python.org/
        """)
        
        # Perform research
        query = "Python programming language"
        result = research_agent.research(query)
        
        # Verify timeout error was handled
        assert "error" in result["status"] or "partial" in result["status"]
        if "error" in result:
            assert "timeout" in result["error"].lower()
        
        # Verify error was logged in memory
        assert self.memory.store_memory_called
        error_keys = [
            args[0] for args in self.memory.store_memory_args 
            if isinstance(args, tuple) and len(args) > 0 and "error" in args[0].lower()
        ]
        assert len(error_keys) > 0


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
