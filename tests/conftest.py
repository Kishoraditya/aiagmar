import os
import pytest
import uuid
import json
import shutil
from unittest.mock import MagicMock, patch

from apps.workflows.research_workflow import ResearchWorkflow
from apps.agents.manager_agent import ManagerAgent
from apps.agents.pre_response_agent import PreResponseAgent
from apps.agents.research_agent import ResearchAgent
from apps.agents.image_generation_agent import ImageGenerationAgent
from apps.agents.file_manager_agent import FileManagerAgent
from apps.agents.summary_agent import SummaryAgent
from apps.agents.verification_agent import VerificationAgent

from apps.mcps.brave_search_mcp import BraveSearchMCP
from apps.mcps.everart_mcp import EverArtMCP
from apps.mcps.fetch_mcp import FetchMCP
from apps.mcps.filesystem_mcp import FilesystemMCP
from apps.mcps.memory_mcp import MemoryMCP

from apps.utils.logger import Logger


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory."""
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="session")
def sample_queries(test_data_dir):
    """Load sample queries from JSON file."""
    with open(os.path.join(test_data_dir, "sample_queries.json"), "r") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def sample_responses(test_data_dir):
    """Load sample responses from JSON file."""
    with open(os.path.join(test_data_dir, "sample_responses.json"), "r") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def expected_results(test_data_dir):
    """Load expected results from JSON file."""
    with open(os.path.join(test_data_dir, "expected_results.json"), "r") as f:
        return json.load(f)


@pytest.fixture(scope="function")
def temp_workspace_dir():
    """Create a temporary workspace directory for testing."""
    # Create a unique workspace directory for each test
    workspace_dir = os.path.join(os.path.dirname(__file__), "temp_workspace", str(uuid.uuid4()))
    os.makedirs(workspace_dir, exist_ok=True)
    
    yield workspace_dir
    
    # Clean up workspace directory
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)


@pytest.fixture(scope="function")
def mock_brave_search_mcp():
    """Create a mocked BraveSearchMCP instance."""
    mock_mcp = MagicMock(spec=BraveSearchMCP)
    mock_mcp.web_search.return_value = "Mock web search results"
    mock_mcp.local_search.return_value = "Mock local search results"
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


@pytest.fixture(scope="function")
def mock_everart_mcp():
    """Create a mocked EverArtMCP instance."""
    mock_mcp = MagicMock(spec=EverArtMCP)
    mock_mcp.generate_image.return_value = "https://example.com/mock-image.jpg"
    mock_mcp.enhance_image.return_value = "https://example.com/mock-enhanced-image.jpg"
    mock_mcp.describe_image.return_value = "A mock image description"
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


@pytest.fixture(scope="function")
def mock_fetch_mcp():
    """Create a mocked FetchMCP instance."""
    mock_mcp = MagicMock(spec=FetchMCP)
    mock_mcp.fetch_url.return_value = "Mock content from website"
    mock_mcp.fetch_html.return_value = "<html><body>Mock HTML content</body></html>"
    mock_mcp.fetch_text.return_value = "Mock text content from website"
    mock_mcp.list_tools.return_value = [
        {
            "name": "fetch_url",
            "description": "Fetch content from a URL"
        },
        {
            "name": "fetch_html",
            "description": "Fetch the raw HTML content from a URL"
        },
        {
            "name": "fetch_text",
            "description": "Fetch the text content from a URL, removing HTML tags"
        }
    ]
    return mock_mcp


@pytest.fixture(scope="function")
def mock_filesystem_mcp(temp_workspace_dir):
    """Create a mocked FilesystemMCP instance."""
    mock_mcp = MagicMock(spec=FilesystemMCP)
    mock_mcp.read_file.return_value = "Mock file content"
    mock_mcp.write_file.return_value = "File written successfully"
    mock_mcp.list_directory.return_value = "file1.txt\nfile2.txt\ndir1/"
    mock_mcp.create_directory.return_value = "Directory created successfully"
    mock_mcp.delete_file.return_value = "File deleted successfully"
    mock_mcp.file_exists.return_value = True
    mock_mcp.search_files.return_value = "file1.txt\ndir1/file3.txt"
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


@pytest.fixture(scope="function")
def mock_memory_mcp():
    """Create a mocked MemoryMCP instance."""
    mock_mcp = MagicMock(spec=MemoryMCP)
    mock_mcp.store_memory.return_value = "Memory stored successfully"
    mock_mcp.retrieve_memory.return_value = "Mock memory content"
    mock_mcp.list_memories.return_value = "key1\nkey2\nkey3"
    mock_mcp.delete_memory.return_value = "Memory deleted successfully"
    mock_mcp.search_memories.return_value = "key1: value1\nkey3: value3"
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


@pytest.fixture(scope="function")
def mock_manager_agent(mock_memory_mcp):
    """Create a mocked ManagerAgent instance."""
    agent = ManagerAgent()
    agent.memory_mcp = mock_memory_mcp
    agent.execute = MagicMock(return_value="Manager agent response")
    return agent


@pytest.fixture(scope="function")
def mock_pre_response_agent(mock_memory_mcp):
    """Create a mocked PreResponseAgent instance."""
    agent = PreResponseAgent()
    agent.memory_mcp = mock_memory_mcp
    agent.execute = MagicMock(return_value="Pre-response agent response")
    agent.needs_clarification = MagicMock(return_value=False)
    agent.clarify_query = MagicMock(return_value="Clarified query")
    return agent


@pytest.fixture(scope="function")
def mock_research_agent(mock_brave_search_mcp, mock_fetch_mcp, mock_memory_mcp):
    """Create a mocked ResearchAgent instance."""
    agent = ResearchAgent()
    agent.brave_search_mcp = mock_brave_search_mcp
    agent.fetch_mcp = mock_fetch_mcp
    agent.memory_mcp = mock_memory_mcp
    agent.execute = MagicMock(return_value="Research agent response")
    return agent


@pytest.fixture(scope="function")
def mock_image_generation_agent(mock_everart_mcp, mock_memory_mcp):
    """Create a mocked ImageGenerationAgent instance."""
    agent = ImageGenerationAgent()
    agent.everart_mcp = mock_everart_mcp
    agent.memory_mcp = mock_memory_mcp
    agent.execute = MagicMock(return_value="Image generation agent response")
    return agent


@pytest.fixture(scope="function")
def mock_file_manager_agent(mock_filesystem_mcp, mock_memory_mcp):
    """Create a mocked FileManagerAgent instance."""
    agent = FileManagerAgent()
    agent.filesystem_mcp = mock_filesystem_mcp
    agent.memory_mcp = mock_memory_mcp
    agent.execute = MagicMock(return_value="File manager agent response")
    return agent


@pytest.fixture(scope="function")
def mock_summary_agent(mock_memory_mcp):
    """Create a mocked SummaryAgent instance."""
    agent = SummaryAgent()
    agent.memory_mcp = mock_memory_mcp
    agent.execute = MagicMock(return_value="Summary agent response")
    return agent


@pytest.fixture(scope="function")
def mock_verification_agent(mock_brave_search_mcp, mock_memory_mcp):
    """Create a mocked VerificationAgent instance."""
    agent = VerificationAgent()
    agent.brave_search_mcp = mock_brave_search_mcp
    agent.memory_mcp = mock_memory_mcp
    agent.execute = MagicMock(return_value="Verification agent response")
    return agent


@pytest.fixture(scope="function")
def mock_research_workflow(
    mock_manager_agent,
    mock_pre_response_agent,
    mock_research_agent,
    mock_image_generation_agent,
    mock_file_manager_agent,
    mock_summary_agent,
    mock_verification_agent
):
    """Create a mocked ResearchWorkflow instance."""
    workflow = ResearchWorkflow(
        manager_agent=mock_manager_agent,
        pre_response_agent=mock_pre_response_agent,
        research_agent=mock_research_agent,
        image_generation_agent=mock_image_generation_agent,
        file_manager_agent=mock_file_manager_agent,
        summary_agent=mock_summary_agent,
        verification_agent=mock_verification_agent
    )
    
    # Preserve the original execute method but wrap it with a MagicMock to track calls
    original_execute = workflow.execute
    workflow.execute = MagicMock(side_effect=original_execute)
    
    return workflow


@pytest.fixture(scope="function")
def mock_subprocess_popen():
    """Mock subprocess.Popen for testing MCPs."""
    with patch('subprocess.Popen') as mock_popen:
        # Mock the process
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline.return_value = json.dumps({
            "jsonrpc": "2.0",
            "id": "test-id",
            "result": {
                "content": [{"type": "text", "text": "Mock MCP response"}],
                "isError": False
            }
        })
        mock_popen.return_value = mock_process
        
        yield mock_popen


@pytest.fixture(scope="function")
def real_brave_search_mcp(mock_subprocess_popen):
    """Create a real BraveSearchMCP instance with mocked subprocess."""
    mcp = BraveSearchMCP(api_key="test_brave_api_key")
    yield mcp
    mcp.close()


@pytest.fixture(scope="function")
def real_everart_mcp(mock_subprocess_popen):
    """Create a real EverArtMCP instance with mocked subprocess."""
    mcp = EverArtMCP(api_key="test_everart_api_key")
    yield mcp
    mcp.close()


@pytest.fixture(scope="function")
def real_fetch_mcp(mock_subprocess_popen):
    """Create a real FetchMCP instance with mocked subprocess."""
    mcp = FetchMCP()
    yield mcp
    mcp.close()


@pytest.fixture(scope="function")
def real_filesystem_mcp(mock_subprocess_popen, temp_workspace_dir):
    """Create a real FilesystemMCP instance with mocked subprocess."""
    mcp = FilesystemMCP(workspace_dir=temp_workspace_dir)
    yield mcp
    mcp.close()


@pytest.fixture(scope="function")
def real_memory_mcp(mock_subprocess_popen):
    """Create a real MemoryMCP instance with mocked subprocess."""
    mcp = MemoryMCP()
    yield mcp
    mcp.close()


@pytest.fixture(scope="function")
def logger():
    """Create a Logger instance for testing."""
    return Logger("test")


@pytest.fixture(scope="function")
def sample_query():
    """Return a sample query for testing."""
    return "What is quantum computing?"


@pytest.fixture(scope="function")
def sample_research_data():
    """Return sample research data for testing."""
    return {
        "query": "What is quantum computing?",
        "sources": [
            {
                "title": "Introduction to Quantum Computing",
                "url": "https://example.com/quantum-computing",
                "content": "Quantum computing is a type of computation that harnesses quantum mechanical phenomena."
            },
            {
                "title": "Quantum Computing Explained",
                "url": "https://example.org/quantum-explained",
                "content": "Unlike classical computers, quantum computers use qubits which can represent multiple states simultaneously."
            }
        ],
        "images": [
            "https://example.com/quantum-image1.jpg",
            "https://example.com/quantum-image2.jpg"
        ],
        "summary": "Quantum computing uses quantum mechanics principles to perform calculations. Unlike classical computers that use bits (0 or 1), quantum computers use qubits that can exist in multiple states simultaneously through superposition and entanglement.",
        "verification": {
            "verified": True,
            "confidence": 0.85,
            "corroborating_sources": 3
        }
    }


@pytest.fixture(scope="function")
def mock_mcp_response():
    """Return a mock MCP response for testing."""
    return {
        "jsonrpc": "2.0",
        "id": "test-id",
        "result": {
            "content": [{"type": "text", "text": "Mock MCP response"}],
            "isError": False
        }
    }


@pytest.fixture(scope="function")
def mock_mcp_error_response():
    """Return a mock MCP error response for testing."""
    return {
        "jsonrpc": "2.0",
        "id": "test-id",
        "result": {
            "content": [{"type": "text", "text": "Error: API rate limit exceeded"}],
            "isError": True
        }
    }


@pytest.fixture(scope="function")
def mock_http_response():
    """Create a mock HTTP response for testing."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.ok = True
    mock_response.json.return_value = {"data": "mock data"}
    mock_response.text = "Mock response text"
    return mock_response


@pytest.fixture(scope="function")
def mock_http_error_response():
    """Create a mock HTTP error response for testing."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.ok = False
    mock_response.raise_for_status.side_effect = Exception("HTTP Error 404: Not Found")
    return mock_response


@pytest.fixture(scope="function")
def patch_requests():
    """Patch requests.get and requests.post for testing."""
    with patch('requests.get') as mock_get, patch('requests.post') as mock_post:
        yield mock_get, mock_post


@pytest.fixture(scope="function")
def patch_httpx():
    """Patch httpx.AsyncClient for testing."""
    with patch('httpx.AsyncClient') as mock_client:
        yield mock_client


@pytest.fixture(scope="function")
def mock_agent_response():
    """Create a mock agent response for testing."""
    return {
        "status": "success",
        "message": "Task completed successfully",
        "data": {
            "result": "Mock agent result",
            "confidence": 0.9,
            "processing_time": 1.5
        }
    }


@pytest.fixture(scope="function")
def mock_workflow_context():
    """Create a mock workflow context for testing."""
    return {
        "query": "What is quantum computing?",
        "user_id": "test_user",
        "session_id": "test_session",
        "preferences": {
            "detail_level": "medium",
            "include_images": True,
            "max_sources": 5
        },
        "constraints": {
            "time_limit": 30,
            "max_tokens": 2000
        },
        "history": [
            {"role": "user", "content": "Tell me about AI"},
            {"role": "system", "content": "AI or artificial intelligence refers to..."}
        ]
    }


@pytest.fixture(scope="function")
def mock_protocol_message():
    """Create a mock protocol message for testing."""
    return {
        "id": "msg_123456",
        "timestamp": "2023-06-15T10:30:00Z",
        "sender": "agent_a",
        "recipient": "agent_b",
        "protocol": "a2a",
        "type": "request",
        "content": {
            "capability": "research",
            "payload": {
                "query": "quantum computing",
                "max_results": 5
            }
        }
    }


@pytest.fixture(scope="function")
def setup_test_environment(temp_workspace_dir):
    """Set up a complete test environment with all necessary components."""
    # Create test files in the workspace
    os.makedirs(os.path.join(temp_workspace_dir, "research"), exist_ok=True)
    os.makedirs(os.path.join(temp_workspace_dir, "images"), exist_ok=True)
    
    with open(os.path.join(temp_workspace_dir, "test_file.txt"), "w") as f:
        f.write("This is a test file for testing file operations.")
    
    # Set up environment variables for testing
    os.environ["TEST_MODE"] = "true"
    os.environ["WORKSPACE_DIR"] = temp_workspace_dir
    os.environ["BRAVE_API_KEY"] = "test_brave_api_key"
    os.environ["EVERART_API_KEY"] = "test_everart_api_key"
    
    yield temp_workspace_dir
    
    # Clean up environment variables
    for var in ["TEST_MODE", "WORKSPACE_DIR", "BRAVE_API_KEY", "EVERART_API_KEY"]:
        if var in os.environ:
            del os.environ[var]


@pytest.fixture(scope="function")
def mock_all_mcps(
    mock_brave_search_mcp,
    mock_everart_mcp,
    mock_fetch_mcp,
    mock_filesystem_mcp,
    mock_memory_mcp
):
    """Return all mocked MCPs in a dictionary."""
    return {
        "brave_search": mock_brave_search_mcp,
        "everart": mock_everart_mcp,
        "fetch": mock_fetch_mcp,
        "filesystem": mock_filesystem_mcp,
        "memory": mock_memory_mcp
    }


@pytest.fixture(scope="function")
def mock_all_agents(
    mock_manager_agent,
    mock_pre_response_agent,
    mock_research_agent,
    mock_image_generation_agent,
    mock_file_manager_agent,
    mock_summary_agent,
    mock_verification_agent
):
    """Return all mocked agents in a dictionary."""
    return {
        "manager": mock_manager_agent,
        "pre_response": mock_pre_response_agent,
        "research": mock_research_agent,
        "image_generation": mock_image_generation_agent,
        "file_manager": mock_file_manager_agent,
        "summary": mock_summary_agent,
        "verification": mock_verification_agent
    }


@pytest.fixture(scope="function")
def real_all_mcps(
    real_brave_search_mcp,
    real_everart_mcp,
    real_fetch_mcp,
    real_filesystem_mcp,
    real_memory_mcp
):
    """Return all real MCPs (with mocked subprocess) in a dictionary."""
    return {
        "brave_search": real_brave_search_mcp,
        "everart": real_everart_mcp,
        "fetch": real_fetch_mcp,
        "filesystem": real_filesystem_mcp,
        "memory": real_memory_mcp
    }


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--run-integration", action="store_true", default=False, help="run integration tests"
    )
    parser.addoption(
        "--run-e2e", action="store_true", default=False, help="run end-to-end tests"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on command line options."""
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    skip_e2e = pytest.mark.skip(reason="need --run-e2e option to run")
    
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
        if "integration" in item.keywords and not config.getoption("--run-integration"):
            item.add_marker(skip_integration)
        if "e2e" in item.keywords and not config.getoption("--run-e2e"):
            item.add_marker(skip_e2e)
