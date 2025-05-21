"""
Mock agent implementations for testing purposes.
"""

import json
import os
import uuid
from typing import Dict, List, Any, Optional, Union
from unittest.mock import MagicMock, patch

# Import base agent class and other necessary components
from apps.agents.base_agent import BaseAgent
from apps.mcps.brave_search_mcp import BraveSearchMCP
from apps.mcps.everart_mcp import EverArtMCP
from apps.mcps.fetch_mcp import FetchMCP
from apps.mcps.filesystem_mcp import FilesystemMCP
from apps.mcps.memory_mcp import MemoryMCP


class MockBaseAgent(BaseAgent):
    """Mock implementation of the BaseAgent class for testing."""
    
    def __init__(self, agent_id: Optional[str] = None, agent_type: str = "mock"):
        """Initialize the mock base agent."""
        super().__init__(agent_id=agent_id or str(uuid.uuid4()), agent_type=agent_type)
        self.execute_called = False
        self.execute_args = None
        self.execute_kwargs = None
        self.execute_result = "Mock execution result"
    
    def execute(self, *args, **kwargs):
        """Mock execution method."""
        self.execute_called = True
        self.execute_args = args
        self.execute_kwargs = kwargs
        return self.execute_result
    
    def set_execute_result(self, result: Any):
        """Set the result to be returned by execute()."""
        self.execute_result = result


class MockManagerAgent(MockBaseAgent):
    """Mock implementation of the Manager Agent for testing."""
    
    def __init__(self, agent_id: Optional[str] = None):
        """Initialize the mock manager agent."""
        super().__init__(agent_id=agent_id, agent_type="manager")
        self.memory_mcp = MagicMock(spec=MemoryMCP)
        self.delegate_task_called = False
        self.delegate_task_args = None
        self.delegate_task_kwargs = None
        self.delegate_task_result = "Task delegated successfully"
        
        # Mock the memory MCP methods
        self.memory_mcp.store_memory.return_value = "Memory stored successfully"
        self.memory_mcp.retrieve_memory.return_value = "Retrieved memory content"
        self.memory_mcp.list_memories.return_value = "memory1\nmemory2\nmemory3"
    
    def delegate_task(self, agent_type: str, task: str, *args, **kwargs):
        """Mock task delegation method."""
        self.delegate_task_called = True
        self.delegate_task_args = (agent_type, task) + args
        self.delegate_task_kwargs = kwargs
        return self.delegate_task_result
    
    def set_delegate_task_result(self, result: Any):
        """Set the result to be returned by delegate_task()."""
        self.delegate_task_result = result
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Mock method to get task status."""
        return {
            "task_id": task_id,
            "status": "completed",
            "result": "Task result",
            "agent_type": "mock_agent",
            "timestamp": "2023-05-15T12:30:45"
        }
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Mock method to get workflow status."""
        return {
            "workflow_id": str(uuid.uuid4()),
            "status": "in_progress",
            "tasks": [
                {"task_id": "task1", "status": "completed", "agent_type": "research"},
                {"task_id": "task2", "status": "in_progress", "agent_type": "summary"},
                {"task_id": "task3", "status": "pending", "agent_type": "verification"}
            ],
            "progress": 33,
            "start_time": "2023-05-15T12:00:00",
            "estimated_completion": "2023-05-15T13:00:00"
        }


class MockResearchAgent(MockBaseAgent):
    """Mock implementation of the Research Agent for testing."""
    
    def __init__(self, agent_id: Optional[str] = None):
        """Initialize the mock research agent."""
        super().__init__(agent_id=agent_id, agent_type="research")
        self.brave_search_mcp = MagicMock(spec=BraveSearchMCP)
        self.fetch_mcp = MagicMock(spec=FetchMCP)
        self.search_called = False
        self.search_args = None
        self.search_kwargs = None
        self.search_result = "Search results"
        self.fetch_content_called = False
        self.fetch_content_args = None
        self.fetch_content_kwargs = None
        self.fetch_content_result = "Fetched content"
        
        # Mock the Brave Search MCP methods
        self.brave_search_mcp.web_search.return_value = """
        Title: Example Search Result 1
        Description: This is an example search result description.
        URL: https://example.com/result1
        
        Title: Example Search Result 2
        Description: Another example search result description.
        URL: https://example.com/result2
        """
        
        self.brave_search_mcp.local_search.return_value = """
        Name: Example Local Business
        Address: 123 Main St, Example City, EX 12345
        Phone: (555) 123-4567
        Rating: 4.5 (42 reviews)
        Price Range: $$
        Hours: Mon-Fri 9AM-5PM, Sat-Sun 10AM-4PM
        Description: This is an example local business description.
        """
        
        # Mock the Fetch MCP methods
        self.fetch_mcp.fetch_url.return_value = "<html><body><h1>Example Content</h1><p>This is example content fetched from a URL.</p></body></html>"
        self.fetch_mcp.fetch_text.return_value = "Example Content\n\nThis is example content fetched from a URL."
    
    def search(self, query: str, *args, **kwargs):
        """Mock search method."""
        self.search_called = True
        self.search_args = (query,) + args
        self.search_kwargs = kwargs
        return self.search_result
    
    def set_search_result(self, result: str):
        """Set the result to be returned by search()."""
        self.search_result = result
    
    def fetch_content(self, url: str, *args, **kwargs):
        """Mock fetch content method."""
        self.fetch_content_called = True
        self.fetch_content_args = (url,) + args
        self.fetch_content_kwargs = kwargs
        return self.fetch_content_result
    
    def set_fetch_content_result(self, result: str):
        """Set the result to be returned by fetch_content()."""
        self.fetch_content_result = result


class MockSummaryAgent(MockBaseAgent):
    """Mock implementation of the Summary Agent for testing."""
    
    def __init__(self, agent_id: Optional[str] = None):
        """Initialize the mock summary agent."""
        super().__init__(agent_id=agent_id, agent_type="summary")
        self.memory_mcp = MagicMock(spec=MemoryMCP)
        self.summarize_called = False
        self.summarize_args = None
        self.summarize_kwargs = None
        self.summarize_result = "Summarized content"
        
        # Mock the memory MCP methods
        self.memory_mcp.store_memory.return_value = "Memory stored successfully"
        self.memory_mcp.retrieve_memory.return_value = "Retrieved memory content"
    
    def summarize(self, content: str, *args, **kwargs):
        """Mock summarize method."""
        self.summarize_called = True
        self.summarize_args = (content,) + args
        self.summarize_kwargs = kwargs
        return self.summarize_result
    
    def set_summarize_result(self, result: str):
        """Set the result to be returned by summarize()."""
        self.summarize_result = result


class MockVerificationAgent(MockBaseAgent):
    """Mock implementation of the Verification Agent for testing."""
    
    def __init__(self, agent_id: Optional[str] = None):
        """Initialize the mock verification agent."""
        super().__init__(agent_id=agent_id, agent_type="verification")
        self.brave_search_mcp = MagicMock(spec=BraveSearchMCP)
        self.verify_called = False
        self.verify_args = None
        self.verify_kwargs = None
        self.verify_result = {
            "verified": True,
            "confidence": 0.85,
            "sources": [
                {"url": "https://example.com/source1", "relevance": 0.9},
                {"url": "https://example.com/source2", "relevance": 0.8}
            ],
            "notes": "Information verified from multiple reliable sources."
        }
        
        # Mock the Brave Search MCP methods
        self.brave_search_mcp.web_search.return_value = """
        Title: Verification Source 1
        Description: This source confirms the information.
        URL: https://example.com/source1
        
        Title: Verification Source 2
        Description: This source also confirms the information.
        URL: https://example.com/source2
        """
    
    def verify(self, claim: str, *args, **kwargs):
        """Mock verify method."""
        self.verify_called = True
        self.verify_args = (claim,) + args
        self.verify_kwargs = kwargs
        return self.verify_result
    
    def set_verify_result(self, result: Dict[str, Any]):
        """Set the result to be returned by verify()."""
        self.verify_result = result


class MockPreResponseAgent(MockBaseAgent):
    """Mock implementation of the Pre-response Agent for testing."""
    
    def __init__(self, agent_id: Optional[str] = None):
        """Initialize the mock pre-response agent."""
        super().__init__(agent_id=agent_id, agent_type="pre_response")
        self.memory_mcp = MagicMock(spec=MemoryMCP)
        self.clarify_query_called = False
        self.clarify_query_args = None
        self.clarify_query_kwargs = None
        self.clarify_query_result = {
            "clarified_query": "Clarified query",
            "additional_context": "Additional context",
            "user_confirmed": True
        }
        self.present_plan_called = False
        self.present_plan_args = None
        self.present_plan_kwargs = None
        self.present_plan_result = {
            "plan_presented": True,
            "user_approved": True,
            "modifications": None
        }
        
        # Mock the memory MCP methods
        self.memory_mcp.store_memory.return_value = "Memory stored successfully"
        self.memory_mcp.retrieve_memory.return_value = "Retrieved memory content"
    
    def clarify_query(self, query: str, *args, **kwargs):
        """Mock clarify query method."""
        self.clarify_query_called = True
        self.clarify_query_args = (query,) + args
        self.clarify_query_kwargs = kwargs
        return self.clarify_query_result
    
    def set_clarify_query_result(self, result: Dict[str, Any]):
        """Set the result to be returned by clarify_query()."""
        self.clarify_query_result = result
    
    def present_plan(self, plan: Dict[str, Any], *args, **kwargs):
        """Mock present plan method."""
        self.present_plan_called = True
        self.present_plan_args = (plan,) + args
        self.present_plan_kwargs = kwargs
        return self.present_plan_result
    
    def set_present_plan_result(self, result: Dict[str, Any]):
        """Set the result to be returned by present_plan()."""
        self.present_plan_result = result


class MockImageGenerationAgent(MockBaseAgent):
    """Mock implementation of the Image Generation Agent for testing."""
    
    def __init__(self, agent_id: Optional[str] = None):
        """Initialize the mock image generation agent."""
        super().__init__(agent_id=agent_id, agent_type="image_generation")
        self.everart_mcp = MagicMock(spec=EverArtMCP)
        self.generate_image_called = False
        self.generate_image_args = None
        self.generate_image_kwargs = None
        self.generate_image_result = {
            "image_url": "https://example.com/generated_image.jpg",
            "prompt": "Generated image prompt",
            "style": "realistic",
            "aspect_ratio": "16:9"
        }
        
        # Mock the EverArt MCP methods
        self.everart_mcp.generate_image.return_value = """
        Image generated successfully!
        URL: https://example.com/generated_image.jpg
        Prompt: Generated image prompt
        Style: realistic
        Aspect Ratio: 16:9
        """
        
        self.everart_mcp.enhance_image.return_value = """
        Image enhanced successfully!
        URL: https://example.com/enhanced_image.jpg
        Original: https://example.com/original_image.jpg
        Prompt: Enhanced image prompt
        Strength: 0.7
        """
        
        self.everart_mcp.describe_image.return_value = """
        Image Description:
        The image shows a serene mountain landscape with a lake at sunset.
        The mountains are reflected in the still water of the lake.
        The sky is painted in vibrant orange and purple hues.
        There are some pine trees in the foreground silhouetted against the sunset.
        """
    
    def generate_image(self, prompt: str, *args, **kwargs):
        """Mock generate image method."""
        self.generate_image_called = True
        self.generate_image_args = (prompt,) + args
        self.generate_image_kwargs = kwargs
        return self.generate_image_result
    
    def set_generate_image_result(self, result: Dict[str, Any]):
        """Set the result to be returned by generate_image()."""
        self.generate_image_result = result
    
    def enhance_image(self, image_url: str, prompt: str, *args, **kwargs):
        """Mock enhance image method."""
        return {
            "enhanced_image_url": "https://example.com/enhanced_image.jpg",
            "original_image_url": image_url,
            "prompt": prompt,
            "strength": kwargs.get("strength", 0.5)
        }
    
    def describe_image(self, image_url: str, *args, **kwargs):
        """Mock describe image method."""
        return {
            "description": "The image shows a serene mountain landscape with a lake at sunset.",
            "tags": ["mountain", "lake", "sunset", "landscape", "nature"],
            "colors": ["orange", "blue", "purple", "green"],
            "detail_level": kwargs.get("detail_level", "medium")
        }


class MockFileManagerAgent(MockBaseAgent):
    """Mock implementation of the File Manager Agent for testing."""
    
    def __init__(self, agent_id: Optional[str] = None):
        """Initialize the mock file manager agent."""
        super().__init__(agent_id=agent_id, agent_type="file_manager")
        self.filesystem_mcp = MagicMock(spec=FilesystemMCP)
        self.save_file_called = False
        self.save_file_args = None
        self.save_file_kwargs = None
        self.save_file_result = {
            "file_path": "research/document.txt",
            "success": True,
            "size": 1024
        }
        self.read_file_called = False
        self.read_file_args = None
        self.read_file_kwargs = None
        self.read_file_result = "File content"
        self.organize_files_called = False
        self.organize_files_args = None
        self.organize_files_kwargs = None
        self.organize_files_result = {
            "organized": True,
            "file_count": 5,
            "directory_structure": {
                "research": ["document1.txt", "document2.txt"],
                "images": ["image1.jpg", "image2.jpg"],
                "summaries": ["summary.txt"]
            }
        }
        
        # Mock the Filesystem MCP methods
        self.filesystem_mcp.read_file.return_value = "Mock file content"
        self.filesystem_mcp.write_file.return_value = "File written successfully"
        self.filesystem_mcp.list_directory.return_value = """
        research/
        ├── document1.txt
        ├── document2.txt
        images/
        ├── image1.jpg
        ├── image2.jpg
        summaries/
        └── summary.txt
        """
        self.filesystem_mcp.create_directory.return_value = "Directory created successfully"
        self.filesystem_mcp.delete_file.return_value = "File deleted successfully"
        self.filesystem_mcp.file_exists.return_value = True
        self.filesystem_mcp.search_files.return_value = """
        Found 2 files matching pattern '*.txt':
        research/document1.txt
        research/document2.txt
        """
    
    def save_file(self, content: str, file_path: str, *args, **kwargs):
        """Mock save file method."""
        self.save_file_called = True
        self.save_file_args = (content, file_path) + args
        self.save_file_kwargs = kwargs
        return self.save_file_result
    
    def set_save_file_result(self, result: Dict[str, Any]):
        """Set the result to be returned by save_file()."""
        self.save_file_result = result
    
    def read_file(self, file_path: str, *args, **kwargs):
        """Mock read file method."""
        self.read_file_called = True
        self.read_file_args = (file_path,) + args
        self.read_file_kwargs = kwargs
        return self.read_file_result
    
    def set_read_file_result(self, result: str):
        """Set the result to be returned by read_file()."""
        self.read_file_result = result
    
    def organize_files(self, directory: str, *args, **kwargs):
        """Mock organize files method."""
        self.organize_files_called = True
        self.organize_files_args = (directory,) + args
        self.organize_files_kwargs = kwargs
        return self.organize_files_result
    
    def set_organize_files_result(self, result: Dict[str, Any]):
        """Set the result to be returned by organize_files()."""
        self.organize_files_result = result
    
    def create_directory(self, directory_path: str):
        """Mock create directory method."""
        return {"created": True, "path": directory_path}
    
    def delete_file(self, file_path: str):
        """Mock delete file method."""
        return {"deleted": True, "path": file_path}
    
    def search_files(self, pattern: str, directory: str = "."):
        """Mock search files method."""
        return {
            "pattern": pattern,
            "directory": directory,
            "matches": [
                f"{directory}/file1.txt",
                f"{directory}/file2.txt"
            ]
        }


class MockMCPFactory:
    """Factory class for creating mock MCP instances."""
    
    @staticmethod
    def create_brave_search_mcp(api_key: Optional[str] = None):
        """Create a mock BraveSearchMCP instance."""
        mock_mcp = MagicMock(spec=BraveSearchMCP)
        mock_mcp.api_key = api_key or "mock_brave_api_key"
        mock_mcp.web_search.return_value = """
        Title: Example Search Result 1
        Description: This is an example search result description.
        URL: https://example.com/result1
        
        Title: Example Search Result 2
        Description: Another example search result description.
        URL: https://example.com/result2
        """
        mock_mcp.local_search.return_value = """
        Name: Example Local Business
        Address: 123 Main St, Example City, EX 12345
        Phone: (555) 123-4567
        Rating: 4.5 (42 reviews)
        Price Range: $$
        Hours: Mon-Fri 9AM-5PM, Sat-Sun 10AM-4PM
        Description: This is an example local business description.
        """
        mock_mcp.list_tools.return_value = [
            {
                "name": "brave_web_search",
                "description": "Performs a web search using the Brave Search API"
            },
            {
                "name": "brave_local_search",
                "description": "Searches for local businesses and places using Brave's Local Search API"
            }
        ]
        return mock_mcp
    
    @staticmethod
    def create_everart_mcp(api_key: Optional[str] = None):
        """Create a mock EverArtMCP instance."""
        mock_mcp = MagicMock(spec=EverArtMCP)
        mock_mcp.api_key = api_key or "mock_everart_api_key"
        mock_mcp.generate_image.return_value = """
        Image generated successfully!
        URL: https://example.com/generated_image.jpg
        Prompt: Generated image prompt
        Style: realistic
        Aspect Ratio: 16:9
        """
        mock_mcp.enhance_image.return_value = """
        Image enhanced successfully!
        URL: https://example.com/enhanced_image.jpg
        Original: https://example.com/original_image.jpg
        Prompt: Enhanced image prompt
        Strength: 0.7
        """
        mock_mcp.describe_image.return_value = """
        Image Description:
        The image shows a serene mountain landscape with a lake at sunset.
        The mountains are reflected in the still water of the lake.
        The sky is painted in vibrant orange and purple hues.
        There are some pine trees in the foreground silhouetted against the sunset.
        """
        mock_mcp.list_tools.return_value = [
            {
                "name": "everart_generate_image",
                "description": "Generate images using EverArt AI"
            },
            {
                "name": "everart_enhance_image",
                "description": "Enhance or modify an existing image using EverArt AI"
            },
            {
                "name": "everart_describe_image",
                "description": "Generate a detailed description of an image using EverArt AI"
            }
        ]
        return mock_mcp
    
    @staticmethod
    def create_fetch_mcp():
        """Create a mock FetchMCP instance."""
        mock_mcp = MagicMock(spec=FetchMCP)
        mock_mcp.fetch_url.return_value = "<html><body><h1>Example Content</h1><p>This is example content fetched from a URL.</p></body></html>"
        mock_mcp.fetch_text.return_value = "Example Content\n\nThis is example content fetched from a URL."
        mock_mcp.fetch_html.return_value = "<html><body><h1>Example Content</h1><p>This is example content fetched from a URL.</p></body></html>"
        mock_mcp.list_tools.return_value = [
            {
                "name": "fetch_url",
                "description": "Fetch content from a URL"
            },
            {
                "name": "fetch_text",
                "description": "Fetch the text content from a URL, removing HTML tags"
            },
            {
                "name": "fetch_html",
                "description": "Fetch the raw HTML content from a URL"
            }
        ]
        return mock_mcp
    
    @staticmethod
    def create_filesystem_mcp(workspace_dir: Optional[str] = None):
        """Create a mock FilesystemMCP instance."""
        mock_mcp = MagicMock(spec=FilesystemMCP)
        mock_mcp.workspace_dir = workspace_dir or "/mock/workspace"
        mock_mcp.read_file.return_value = "Mock file content"
        mock_mcp.write_file.return_value = "File written successfully"
        mock_mcp.list_directory.return_value = """
        research/
        ├── document1.txt
        ├── document2.txt
        images/
        ├── image1.jpg
        ├── image2.jpg
        summaries/
        └── summary.txt
        """
        mock_mcp.create_directory.return_value = "Directory created successfully"
        mock_mcp.delete_file.return_value = "File deleted successfully"
        mock_mcp.file_exists.return_value = True
        mock_mcp.search_files.return_value = """
        Found 2 files matching pattern '*.txt':
        research/document1.txt
        research/document2.txt
        """
        mock_mcp.list_tools.return_value = [
            {
                "name": "read_file",
                "description": "Read the contents of a file"
            },
            {
                "name": "write_file",
                "description": "Write content to a file"
            },
            {
                "name": "list_directory",
                "description": "List files and directories in a directory"
            },
            {
                "name": "create_directory",
                "description": "Create a directory"
            },
            {
                "name": "delete_file",
                "description": "Delete a file"
            },
            {
                "name": "file_exists",
                "description": "Check if a file exists"
            },
            {
                "name": "search_files",
                "description": "Search for files matching a pattern"
            }
        ]
        return mock_mcp
    
    @staticmethod
    def create_memory_mcp(storage_path: Optional[str] = None):
        """Create a mock MemoryMCP instance."""
        mock_mcp = MagicMock(spec=MemoryMCP)
        mock_mcp.storage_path = storage_path or "/mock/memory"
        mock_mcp.store_memory.return_value = "Memory stored successfully"
        mock_mcp.retrieve_memory.return_value = "Retrieved memory content"
        mock_mcp.list_memories.return_value = "memory1\nmemory2\nmemory3"
        mock_mcp.delete_memory.return_value = "Memory deleted successfully"
        mock_mcp.search_memories.return_value = "memory1\nmemory3"
        mock_mcp.clear_namespace.return_value = "Namespace cleared successfully"
        mock_mcp.list_tools.return_value = [
            {
                "name": "store_memory",
                "description": "Store a memory item"
            },
            {
                "name": "retrieve_memory",
                "description": "Retrieve a memory item by key"
            },
            {
                "name": "list_memories",
                "description": "List all memories in a namespace"
            },
            {
                "name": "delete_memory",
                "description": "Delete a memory item"
            },
            {
                "name": "search_memories",
                "description": "Search for memories by content"
            },
            {
                "name": "clear_namespace",
                "description": "Clear all memories in a namespace"
            }
        ]
        return mock_mcp


class MockAgentFactory:
    """Factory class for creating mock agent instances."""
    
    @staticmethod
    def create_manager_agent():
        """Create a mock manager agent."""
        return MockManagerAgent()
    
    @staticmethod
    def create_research_agent():
        """Create a mock research agent."""
        return MockResearchAgent()
    
    @staticmethod
    def create_summary_agent():
        """Create a mock summary agent."""
        return MockSummaryAgent()
    
    @staticmethod
    def create_verification_agent():
        """Create a mock verification agent."""
        return MockVerificationAgent()
    
    @staticmethod
    def create_pre_response_agent():
        """Create a mock pre-response agent."""
        return MockPreResponseAgent()
    
    @staticmethod
    def create_image_generation_agent():
        """Create a mock image generation agent."""
        return MockImageGenerationAgent()
    
    @staticmethod
    def create_file_manager_agent():
        """Create a mock file manager agent."""
        return MockFileManagerAgent()
    
    @staticmethod
    def create_agent(agent_type: str):
        """Create a mock agent of the specified type."""
        agent_map = {
            "manager": MockAgentFactory.create_manager_agent,
            "research": MockAgentFactory.create_research_agent,
            "summary": MockAgentFactory.create_summary_agent,
            "verification": MockAgentFactory.create_verification_agent,
            "pre_response": MockAgentFactory.create_pre_response_agent,
            "image_generation": MockAgentFactory.create_image_generation_agent,
            "file_manager": MockAgentFactory.create_file_manager_agent
        }
        
        if agent_type not in agent_map:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        return agent_map[agent_type]()


# Helper functions for patching agent and MCP classes in tests

def patch_all_agents():
    """Patch all agent classes with their mock counterparts."""
    patches = [
        patch('apps.agents.manager_agent.ManagerAgent', MockManagerAgent),
        patch('apps.agents.research_agent.ResearchAgent', MockResearchAgent),
        patch('apps.agents.summary_agent.SummaryAgent', MockSummaryAgent),
        patch('apps.agents.verification_agent.VerificationAgent', MockVerificationAgent),
        patch('apps.agents.pre_response_agent.PreResponseAgent', MockPreResponseAgent),
        patch('apps.agents.image_generation_agent.ImageGenerationAgent', MockImageGenerationAgent),
        patch('apps.agents.file_manager_agent.FileManagerAgent', MockFileManagerAgent)
    ]
    
    for p in patches:
        p.start()
    
    return patches

def patch_all_mcps():
    """Patch all MCP classes with their mock counterparts."""
    patches = [
        patch('apps.mcps.brave_search_mcp.BraveSearchMCP', MagicMock(spec=BraveSearchMCP)),
        patch('apps.mcps.everart_mcp.EverArtMCP', MagicMock(spec=EverArtMCP)),
        patch('apps.mcps.fetch_mcp.FetchMCP', MagicMock(spec=FetchMCP)),
        patch('apps.mcps.filesystem_mcp.FilesystemMCP', MagicMock(spec=FilesystemMCP)),
        patch('apps.mcps.memory_mcp.MemoryMCP', MagicMock(spec=MemoryMCP))
    ]
    
    for p in patches:
        p.start()
    
    return patches

def stop_patches(patches):
    """Stop all patches."""
    for p in patches:
        p.stop()


# Mock workflow for testing
class MockResearchWorkflow:
    """Mock implementation of the Research Workflow for testing."""
    
    def __init__(self):
        """Initialize the mock research workflow."""
        self.manager_agent = MockManagerAgent()
        self.pre_response_agent = MockPreResponseAgent()
        self.research_agent = MockResearchAgent()
        self.summary_agent = MockSummaryAgent()
        self.verification_agent = MockVerificationAgent()
        self.image_generation_agent = MockImageGenerationAgent()
        self.file_manager_agent = MockFileManagerAgent()
        
        self.execute_called = False
        self.execute_args = None
        self.execute_kwargs = None
        self.execute_result = {
            "workflow_id": str(uuid.uuid4()),
            "status": "completed",
            "query": "mock query",
            "results": {
                "summary": "Mock summary of research findings",
                "sources": [
                    {"url": "https://example.com/source1", "title": "Source 1"},
                    {"url": "https://example.com/source2", "title": "Source 2"}
                ],
                "images": [
                    {"url": "https://example.com/image1.jpg", "caption": "Image 1"},
                    {"url": "https://example.com/image2.jpg", "caption": "Image 2"}
                ],
                "files": [
                    {"path": "research/document1.txt", "type": "text"},
                    {"path": "images/image1.jpg", "type": "image"}
                ],
                "verification": {
                    "verified": True,
                    "confidence": 0.85,
                    "notes": "Information verified from multiple reliable sources."
                }
            },
            "execution_time": 10.5
        }
    
    def execute(self, query: str, *args, **kwargs):
        """Mock execute method."""
        self.execute_called = True
        self.execute_args = (query,) + args
        self.execute_kwargs = kwargs
        return self.execute_result
    
    def set_execute_result(self, result: Dict[str, Any]):
        """Set the result to be returned by execute()."""
        self.execute_result = result
    
    def get_status(self):
        """Get the current status of the workflow."""
        return {
            "workflow_id": self.execute_result["workflow_id"],
            "status": self.execute_result["status"],
            "progress": 100 if self.execute_result["status"] == "completed" else 50,
            "current_step": "completed" if self.execute_result["status"] == "completed" else "research",
            "execution_time": self.execute_result["execution_time"]
        }


# Mock response objects for testing HTTP requests
class MockResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self, json_data=None, text=None, status_code=200, headers=None):
        """Initialize the mock response."""
        self.json_data = json_data
        self.text = text or ""
        self.status_code = status_code
        self.headers = headers or {}
        self.ok = 200 <= status_code < 300
    
    def json(self):
        """Return the JSON data."""
        if self.json_data is None:
            raise ValueError("No JSON data available")
        return self.json_data
    
    def raise_for_status(self):
        """Raise an exception if the status code indicates an error."""
        if not self.ok:
            raise Exception(f"HTTP Error: {self.status_code}")


# Mock context managers for testing
class MockContextManager:
    """Mock context manager for testing."""
    
    def __init__(self, return_value=None, exception=None):
        """Initialize the mock context manager."""
        self.return_value = return_value
        self.exception = exception
        self.entered = False
        self.exited = False
        self.exit_args = None
    
    def __enter__(self):
        """Enter the context manager."""
        self.entered = True
        if self.exception:
            raise self.exception
        return self.return_value
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.exited = True
        self.exit_args = (exc_type, exc_val, exc_tb)
        return False  # Don't suppress exceptions


# Mock subprocess for testing
class MockSubprocess:
    """Mock subprocess for testing."""
    
    def __init__(self, stdout=None, stderr=None, returncode=0):
        """Initialize the mock subprocess."""
        self.stdout = stdout or ""
        self.stderr = stderr or ""
        self.returncode = returncode
        self.stdin = MagicMock()
        self.args = None
        self.kwargs = None
    
    def communicate(self, input=None):
        """Mock communicate method."""
        return self.stdout, self.stderr
    
    def poll(self):
        """Mock poll method."""
        return self.returncode
    
    def wait(self, timeout=None):
        """Mock wait method."""
        return self.returncode


# Mock file-like objects for testing
class MockFile:
    """Mock file-like object for testing."""
    
    def __init__(self, content="", name="mock_file.txt"):
        """Initialize the mock file."""
        self.content = content
        self.name = name
        self.closed = False
        self.mode = "r"
        self.encoding = "utf-8"
        self.position = 0
    
    def read(self, size=None):
        """Mock read method."""
        if size is None:
            result = self.content[self.position:]
            self.position = len(self.content)
        else:
            result = self.content[self.position:self.position + size]
            self.position += min(size, len(self.content) - self.position)
        return result
    
    def write(self, data):
        """Mock write method."""
        if "r" in self.mode and "+" not in self.mode:
            raise IOError("File not open for writing")
        self.content = self.content[:self.position] + data + self.content[self.position:]
        self.position += len(data)
        return len(data)
    
    def seek(self, position, whence=0):
        """Mock seek method."""
        if whence == 0:  # SEEK_SET
            self.position = position
        elif whence == 1:  # SEEK_CUR
            self.position += position
        elif whence == 2:  # SEEK_END
            self.position = len(self.content) + position
        self.position = max(0, min(self.position, len(self.content)))
    
    def tell(self):
        """Mock tell method."""
        return self.position
    
    def close(self):
        """Mock close method."""
        self.closed = True
    
    def __enter__(self):
        """Enter the context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.close()
        return False  # Don't suppress exceptions


# Utility functions for creating mock data
def create_mock_search_results(count=5):
    """Create mock search results."""
    results = []
    for i in range(1, count + 1):
        results.append({
            "title": f"Mock Search Result {i}",
            "description": f"This is mock search result {i} description.",
            "url": f"https://example.com/result{i}"
        })
    return results

def create_mock_local_results(count=3):
    """Create mock local search results."""
    results = []
    for i in range(1, count + 1):
        results.append({
            "name": f"Mock Local Business {i}",
            "address": f"{i}23 Main St, Example City, EX {i}2345",
            "phone": f"(555) {i}23-4567",
            "rating": 4.0 + (i / 10),
            "reviews": i * 10,
            "price_range": "$" * i,
            "hours": "Mon-Fri 9AM-5PM, Sat-Sun 10AM-4PM",
            "description": f"This is mock local business {i} description."
        })
    return results

def create_mock_image_results(count=2):
    """Create mock image generation results."""
    results = []
    for i in range(1, count + 1):
        results.append({
            "url": f"https://example.com/image{i}.jpg",
            "prompt": f"Mock image prompt {i}",
            "style": "realistic" if i % 2 == 0 else "artistic",
            "aspect_ratio": "16:9" if i % 2 == 0 else "1:1"
        })
    return results

def create_mock_file_structure():
    """Create a mock file structure."""
    return {
        "research": {
            "document1.txt": "Content of document 1",
            "document2.txt": "Content of document 2",
            "notes": {
                "note1.txt": "Content of note 1",
                "note2.txt": "Content of note 2"
            }
        },
        "images": {
            "image1.jpg": b"Binary content of image 1",
            "image2.jpg": b"Binary content of image 2"
        },
        "summaries": {
            "summary.txt": "Summary content"
        }
    }


# If this module is run directly, perform a simple test
if __name__ == "__main__":
    # Create mock agents
    manager = MockManagerAgent()
    research = MockResearchAgent()
    summary = MockSummaryAgent()
    verification = MockVerificationAgent()
    pre_response = MockPreResponseAgent()
    image_generation = MockImageGenerationAgent()
    file_manager = MockFileManagerAgent()
    
    # Test manager agent
    result = manager.execute("Test task")
    print(f"Manager agent execute result: {result}")
    
    # Test research agent
    result = research.search("Test query")
    print(f"Research agent search result: {result}")
    
    # Test summary agent
    result = summary.summarize("Test content to summarize")
    print(f"Summary agent summarize result: {result}")
    
    # Test verification agent
    result = verification.verify("Test claim to verify")
    print(f"Verification agent verify result: {result}")
    
    # Test pre-response agent
    result = pre_response.clarify_query("Test query to clarify")
    print(f"Pre-response agent clarify_query result: {result}")
    
    # Test image generation agent
    result = image_generation.generate_image("Test prompt for image generation")
    print(f"Image generation agent generate_image result: {result}")
    
    # Test file manager agent
    result = file_manager.save_file("Test content", "test_file.txt")
    print(f"File manager agent save_file result: {result}")
    
    print("All mock agents tested successfully!")
