"""
Constants Module

This module defines constant values used throughout the application.
These constants include status codes, error messages, default values,
file paths, and other unchanging values.
"""

import os
from enum import Enum, auto
from pathlib import Path

# -----------------------------------------------------------------------------
# Application Constants
# -----------------------------------------------------------------------------

# Application information
APP_NAME = "AIAGMAR"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = "AI Agent for Multi-Agent Research"
APP_AUTHOR = "Kishoraditya"
APP_REPOSITORY = "github.com/Kishoraditya/aiagmar"

# Base directories
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
APPS_DIR = ROOT_DIR / "apps"
SERVICES_DIR = ROOT_DIR / "services"
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
TEMP_DIR = ROOT_DIR / "temp"
OUTPUT_DIR = ROOT_DIR / "output"

# Ensure directories exist
for directory in [DATA_DIR, LOGS_DIR, TEMP_DIR, OUTPUT_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# -----------------------------------------------------------------------------
# API Constants
# -----------------------------------------------------------------------------

# API settings
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
DEFAULT_API_WORKERS = 4
DEFAULT_API_TIMEOUT = 60
DEFAULT_API_PREFIX = "/api/v1"

# Rate limiting
DEFAULT_RATE_LIMIT = 100  # requests per minute
DEFAULT_RATE_LIMIT_PERIOD = 60  # seconds

# CORS settings
DEFAULT_CORS_ORIGINS = ["*"]

# -----------------------------------------------------------------------------
# MCP Constants
# -----------------------------------------------------------------------------

# MCP types
class MCPType(Enum):
    BRAVE_SEARCH = "brave_search"
    EVERART = "everart"
    FETCH = "fetch"
    FILESYSTEM = "filesystem"
    MEMORY = "memory"

# MCP server commands
MCP_COMMANDS = {
    MCPType.BRAVE_SEARCH: {
        "docker": ["docker", "run", "-i", "--rm", "-e", "BRAVE_API_KEY", "mcp/brave-search"],
        "npx": ["npx", "-y", "@modelcontextprotocol/server-brave-search"],
    },
    MCPType.EVERART: {
        "docker": ["docker", "run", "-i", "--rm", "-e", "EVERART_API_KEY", "mcp/everart"],
        "npx": ["npx", "-y", "@modelcontextprotocol/server-everart"],
    },
    MCPType.FETCH: {
        "docker": ["docker", "run", "-i", "--rm", "mcp/fetch"],
        "npx": ["npx", "-y", "@modelcontextprotocol/server-fetch"],
    },
    MCPType.FILESYSTEM: {
        "docker": ["docker", "run", "-i", "--rm", "-v", "{workspace_dir}:/workspace", "-e", "WORKSPACE_DIR=/workspace", "mcp/filesystem"],
        "npx": ["npx", "-y", "@modelcontextprotocol/server-filesystem"],
    },
    MCPType.MEMORY: {
        "docker": ["docker", "run", "-i", "--rm", "-v", "{storage_path}:/data", "-e", "MEMORY_STORAGE_PATH=/data", "mcp/memory"],
        "npx": ["npx", "-y", "@modelcontextprotocol/server-memory"],
    },
}

# MCP default settings
DEFAULT_MCP_SETTINGS = {
    MCPType.BRAVE_SEARCH: {
        "enabled": True,
        "use_docker": False,
    },
    MCPType.EVERART: {
        "enabled": True,
        "use_docker": False,
    },
    MCPType.FETCH: {
        "enabled": True,
        "use_docker": False,
    },
    MCPType.FILESYSTEM: {
        "enabled": True,
        "use_docker": False,
        "workspace_dir": str(OUTPUT_DIR),
    },
    MCPType.MEMORY: {
        "enabled": True,
        "use_docker": False,
        "storage_path": str(DATA_DIR / "memory"),
    },
}

# -----------------------------------------------------------------------------
# Agent Constants
# -----------------------------------------------------------------------------

# Agent types
class AgentType(Enum):
    MANAGER = "manager"
    PRE_RESPONSE = "pre_response"
    RESEARCH = "research"
    SUMMARY = "summary"
    VERIFICATION = "verification"
    IMAGE_GENERATION = "image_generation"
    FILE_MANAGER = "file_manager"

# Agent roles and descriptions
AGENT_DESCRIPTIONS = {
    AgentType.MANAGER: "Coordinates the workflow, deciding which agents to call based on the user's query and managing the overall process.",
    AgentType.PRE_RESPONSE: "Interacts with the user, clarifies queries if needed, and presents the research plan before execution.",
    AgentType.RESEARCH: "Performs web searches to find relevant articles and fetches content for further processing.",
    AgentType.SUMMARY: "Summarizes the fetched content using the language model's capabilities, providing concise insights.",
    AgentType.VERIFICATION: "Verifies facts by searching for additional sources, ensuring accuracy and reliability of information.",
    AgentType.IMAGE_GENERATION: "Generates images or diagrams based on the research findings, enhancing visual representation.",
    AgentType.FILE_MANAGER: "Organizes and stores research materials, such as summaries and images, in a structured file system.",
}

# Agent dependencies (which MCPs each agent relies on)
AGENT_DEPENDENCIES = {
    AgentType.MANAGER: [MCPType.MEMORY],
    AgentType.PRE_RESPONSE: [MCPType.MEMORY],
    AgentType.RESEARCH: [MCPType.BRAVE_SEARCH, MCPType.FETCH],
    AgentType.SUMMARY: [MCPType.MEMORY],
    AgentType.VERIFICATION: [MCPType.BRAVE_SEARCH],
    AgentType.IMAGE_GENERATION: [MCPType.EVERART],
    AgentType.FILE_MANAGER: [MCPType.FILESYSTEM],
}

# Default agent settings
DEFAULT_AGENT_SETTINGS = {
    AgentType.MANAGER: {
        "enabled": True,
        "max_tasks": 10,
    },
    AgentType.PRE_RESPONSE: {
        "enabled": True,
    },
    AgentType.RESEARCH: {
        "enabled": True,
        "max_sources": 5,
        "max_content_length": 100000,
    },
    AgentType.SUMMARY: {
        "enabled": True,
        "max_summary_length": 5000,
    },
    AgentType.VERIFICATION: {
        "enabled": True,
        "verification_threshold": 0.7,
    },
    AgentType.IMAGE_GENERATION: {
        "enabled": True,
        "max_images": 3,
    },
    AgentType.FILE_MANAGER: {
        "enabled": True,
        "max_file_size": 10485760,  # 10MB
    },
}

# -----------------------------------------------------------------------------
# Workflow Constants
# -----------------------------------------------------------------------------

# Workflow types
class WorkflowType(Enum):
    RESEARCH = "research"

# Research depth options
class ResearchDepth(Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"

# Output format options
class OutputFormat(Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    JSON = "json"

# Default workflow settings
DEFAULT_WORKFLOW_SETTINGS = {
    WorkflowType.RESEARCH: {
        "enabled": True,
        "default_max_sources": 5,
        "default_include_images": True,
        "default_verify_facts": True,
        "default_research_depth": ResearchDepth.STANDARD.value,
        "default_output_format": OutputFormat.MARKDOWN.value,
    },
}

# Research depth settings
RESEARCH_DEPTH_SETTINGS = {
    ResearchDepth.QUICK: {
        "max_sources": 3,
        "max_content_length": 50000,
        "verification_threshold": 0.6,
    },
    ResearchDepth.STANDARD: {
        "max_sources": 5,
        "max_content_length": 100000,
        "verification_threshold": 0.7,
    },
    ResearchDepth.DEEP: {
        "max_sources": 10,
        "max_content_length": 200000,
        "verification_threshold": 0.8,
    },
}

# -----------------------------------------------------------------------------
# Protocol Constants
# -----------------------------------------------------------------------------

# Protocol types
class ProtocolType(Enum):
    A2A = "a2a"
    ANP = "anp"
    ACP = "acp"

# Protocol descriptions
PROTOCOL_DESCRIPTIONS = {
    ProtocolType.A2A: "Agent-to-Agent Protocol for peer-to-peer agent communication",
    ProtocolType.ANP: "Agent Network Protocol for open-network discovery and collaboration",
    ProtocolType.ACP: "Agent Communication Protocol for REST-native, multimodal messaging",
}

# -----------------------------------------------------------------------------
# HTTP Status Codes
# -----------------------------------------------------------------------------

class HTTPStatus(Enum):
    # Success codes
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    
    # Client error codes
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    TOO_MANY_REQUESTS = 429
    
    # Server error codes
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503

# -----------------------------------------------------------------------------
# Error Constants
# -----------------------------------------------------------------------------

# Error codes
class ErrorCode(Enum):
    # General errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    
    # Authentication errors
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    
    # Resource errors
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
    
    # API errors
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    API_ERROR = "API_ERROR"
    
    # MCP errors
    MCP_ERROR = "MCP_ERROR"
    MCP_NOT_AVAILABLE = "MCP_NOT_AVAILABLE"
    
    # Agent errors
    AGENT_ERROR = "AGENT_ERROR"
    AGENT_NOT_AVAILABLE = "AGENT_NOT_AVAILABLE"
    
    # Workflow errors
    WORKFLOW_ERROR = "WORKFLOW_ERROR"
    WORKFLOW_NOT_AVAILABLE = "WORKFLOW_NOT_AVAILABLE"

# Error messages
ERROR_MESSAGES = {
    ErrorCode.UNKNOWN_ERROR: "An unknown error occurred",
    ErrorCode.VALIDATION_ERROR: "Validation error: {details}",
    ErrorCode.CONFIGURATION_ERROR: "Configuration error: {details}",
    
    ErrorCode.AUTHENTICATION_ERROR: "Authentication failed: {details}",
    ErrorCode.AUTHORIZATION_ERROR: "Authorization failed: {details}",
    
    ErrorCode.RESOURCE_NOT_FOUND: "Resource not found: {details}",
    ErrorCode.RESOURCE_ALREADY_EXISTS: "Resource already exists: {details}",
    
    ErrorCode.RATE_LIMIT_EXCEEDED: "Rate limit exceeded. Try again in {retry_after} seconds",
    ErrorCode.API_ERROR: "API error: {details}",
    
    ErrorCode.MCP_ERROR: "MCP error: {details}",
    ErrorCode.MCP_NOT_AVAILABLE: "MCP not available: {details}",
    
    ErrorCode.AGENT_ERROR: "Agent error: {details}",
    ErrorCode.AGENT_NOT_AVAILABLE: "Agent not available: {details}",
    
    ErrorCode.WORKFLOW_ERROR: "Workflow error: {details}",
    ErrorCode.WORKFLOW_NOT_AVAILABLE: "Workflow not available: {details}",
}

# -----------------------------------------------------------------------------
# File Constants
# -----------------------------------------------------------------------------

# File extensions
class FileExtension(Enum):
    # Text files
    TXT = "txt"
    MD = "md"
    HTML = "html"
    JSON = "json"
    YAML = "yaml"
    CSV = "csv"
    
    # Image files
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    SVG = "svg"
    
    # Document files
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"

# MIME types
MIME_TYPES = {
    FileExtension.TXT: "text/plain",
    FileExtension.MD: "text/markdown",
    FileExtension.HTML: "text/html",
    FileExtension.JSON: "application/json",
    FileExtension.YAML: "application/yaml",
    FileExtension.CSV: "text/csv",
    
    FileExtension.JPG: "image/jpeg",
    FileExtension.JPEG: "image/jpeg",
    FileExtension.PNG: "image/png",
    FileExtension.GIF: "image/gif",
    FileExtension.SVG: "image/svg+xml",
    
    FileExtension.PDF: "application/pdf",
    FileExtension.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    FileExtension.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    FileExtension.PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

# -----------------------------------------------------------------------------
# Miscellaneous Constants
# -----------------------------------------------------------------------------

# Default timeout values (in seconds)
DEFAULT_TIMEOUT = 30
LONG_TIMEOUT = 120
SHORT_TIMEOUT = 10

# Default retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_RETRY_BACKOFF = 2.0

# Default pagination settings
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# Default cache settings
DEFAULT_CACHE_TTL = 3600  # 1 hour

# User agent for HTTP requests
USER_AGENT = f"{APP_NAME}/{APP_VERSION} (+https://{APP_REPOSITORY})"

# Default headers for HTTP requests
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}

# -----------------------------------------------------------------------------
# Environment Constants
# -----------------------------------------------------------------------------

# Environment types
class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

# Current environment
ENVIRONMENT = Environment(os.environ.get("AIAGMAR_ENVIRONMENT", "development").lower())

# Debug mode
DEBUG = os.environ.get("AIAGMAR_DEBUG", "false").lower() in ("true", "1", "yes")

# -----------------------------------------------------------------------------
# Logging Constants
# -----------------------------------------------------------------------------

# Log levels
class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

# Log formats
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"

# Default log settings
DEFAULT_LOG_LEVEL = LogLevel.INFO.value
DEFAULT_LOG_TO_CONSOLE = True
DEFAULT_LOG_TO_FILE = True
DEFAULT_LOG_FILENAME = "aiagmar.log"
DEFAULT_LOG_MAX_BYTES = 10485760  # 10MB
DEFAULT_LOG_BACKUP_COUNT = 5

# -----------------------------------------------------------------------------
# Memory Constants
# -----------------------------------------------------------------------------

# Memory namespaces
class MemoryNamespace(Enum):
    DEFAULT = "default"
    USER = "user"
    RESEARCH = "research"
    WORKFLOW = "workflow"
    AGENT = "agent"
    SYSTEM = "system"

# Default memory settings
DEFAULT_MEMORY_STORAGE_PATH = str(DATA_DIR / "memory")

# -----------------------------------------------------------------------------
# Research Constants
# -----------------------------------------------------------------------------

# Search engines
class SearchEngine(Enum):
    BRAVE = "brave"
    GOOGLE = "google"
    BING = "bing"

# Default search settings
DEFAULT_SEARCH_ENGINE = SearchEngine.BRAVE.value
DEFAULT_SEARCH_COUNT = 10
DEFAULT_SEARCH_OFFSET = 0

# Content types
class ContentType(Enum):
    ARTICLE = "article"
    NEWS = "news"
    BLOG = "blog"
    FORUM = "forum"
    ACADEMIC = "academic"
    REFERENCE = "reference"
    OTHER = "other"

# Source reliability levels
class SourceReliability(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

# -----------------------------------------------------------------------------
# Image Generation Constants
# -----------------------------------------------------------------------------

# Image styles
class ImageStyle(Enum):
    REALISTIC = "realistic"
    CARTOON = "cartoon"
    SKETCH = "sketch"
    PAINTING = "painting"
    ABSTRACT = "abstract"
    ANIME = "anime"
    PIXEL_ART = "pixel_art"
    WATERCOLOR = "watercolor"
    OIL_PAINTING = "oil_painting"
    DIGITAL_ART = "digital_art"

# Image aspect ratios
class ImageAspectRatio(Enum):
    SQUARE = "1:1"
    PORTRAIT = "2:3"
    LANDSCAPE = "3:2"
    WIDESCREEN = "16:9"
    ULTRAWIDE = "21:9"
    PANORAMA = "3:1"

# Default image generation settings
DEFAULT_IMAGE_STYLE = ImageStyle.REALISTIC.value
DEFAULT_IMAGE_ASPECT_RATIO = ImageAspectRatio.SQUARE.value
DEFAULT_IMAGE_COUNT = 1

# -----------------------------------------------------------------------------
# File Manager Constants
# -----------------------------------------------------------------------------

# Default file structure
DEFAULT_FILE_STRUCTURE = {
    "research": {
        "sources": None,
        "summaries": None,
        "images": None,
        "output": None,
    },
    "temp": None,
    "logs": None,
}

# Default file naming patterns
DEFAULT_SOURCE_FILE_PATTERN = "source_{index}_{domain}.{ext}"
DEFAULT_SUMMARY_FILE_PATTERN = "summary_{index}_{topic}.{ext}"
DEFAULT_IMAGE_FILE_PATTERN = "image_{index}_{description}.{ext}"
DEFAULT_OUTPUT_FILE_PATTERN = "research_{topic}_{timestamp}.{ext}"

# -----------------------------------------------------------------------------
# API Endpoint Constants
# -----------------------------------------------------------------------------

# API endpoints
class APIEndpoint(Enum):
    # Research workflow endpoints
    RESEARCH = "/research"
    RESEARCH_STATUS = "/research/{research_id}/status"
    RESEARCH_RESULT = "/research/{research_id}/result"
    
    # Agent endpoints
    AGENTS = "/agents"
    AGENT = "/agents/{agent_id}"
    AGENT_TASK = "/agents/{agent_id}/task"
    
    # MCP endpoints
    MCPS = "/mcps"
    MCP = "/mcps/{mcp_id}"
    
    # System endpoints
    HEALTH = "/health"
    CONFIG = "/config"
    LOGS = "/logs"

# -----------------------------------------------------------------------------
# Task Status Constants
# -----------------------------------------------------------------------------

# Task status
class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Task priority
class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# -----------------------------------------------------------------------------
# Security Constants
# -----------------------------------------------------------------------------

# Authentication methods
class AuthMethod(Enum):
    NONE = "none"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH = "oauth"

# Default security settings
DEFAULT_AUTH_METHOD = AuthMethod.NONE.value
DEFAULT_API_KEY_HEADER = "X-API-Key"
DEFAULT_JWT_HEADER = "Authorization"
DEFAULT_JWT_PREFIX = "Bearer"
DEFAULT_TOKEN_EXPIRY = 86400  # 24 hours

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def get_mcp_command(mcp_type: MCPType, use_docker: bool = False, **kwargs) -> list:
    """
    Get the command to start an MCP server.
    
    Args:
        mcp_type: Type of MCP server
        use_docker: Whether to use Docker or NPX
        **kwargs: Additional arguments to format the command
        
    Returns:
        Command as a list of strings
    """
    command_type = "docker" if use_docker else "npx"
    command = MCP_COMMANDS[mcp_type][command_type].copy()
    
    # Format command with kwargs
    formatted_command = []
    for item in command:
        if "{" in item and "}" in item:
            formatted_command.append(item.format(**kwargs))
        else:
            formatted_command.append(item)
    
    return formatted_command


def get_error_message(error_code: ErrorCode, **kwargs) -> str:
    """
    Get an error message for an error code.
    
    Args:
        error_code: Error code
        **kwargs: Arguments to format the error message
        
    Returns:
        Formatted error message
    """
    message_template = ERROR_MESSAGES.get(error_code, ERROR_MESSAGES[ErrorCode.UNKNOWN_ERROR])
    return message_template.format(**kwargs)


def get_mime_type(file_extension: FileExtension) -> str:
    """
    Get the MIME type for a file extension.
    
    Args:
        file_extension: File extension
        
    Returns:
        MIME type
    """
    return MIME_TYPES.get(file_extension, "application/octet-stream")


def get_agent_description(agent_type: AgentType) -> str:
    """
    Get the description for an agent type.
    
    Args:
        agent_type: Agent type
        
    Returns:
        Agent description
    """
    return AGENT_DESCRIPTIONS.get(agent_type, "")


def get_agent_dependencies(agent_type: AgentType) -> list:
    """
    Get the MCP dependencies for an agent type.
    
    Args:
        agent_type: Agent type
        
    Returns:
        List of MCP types
    """
    return AGENT_DEPENDENCIES.get(agent_type, [])


def get_protocol_description(protocol_type: ProtocolType) -> str:
    """
    Get the description for a protocol type.
    
    Args:
        protocol_type: Protocol type
        
    Returns:
        Protocol description
    """
    return PROTOCOL_DESCRIPTIONS.get(protocol_type, "")


def get_research_depth_settings(depth: ResearchDepth) -> dict:
    """
    Get the settings for a research depth.
    
    Args:
        depth: Research depth
        
    Returns:
        Research depth settings
    """
    return RESEARCH_DEPTH_SETTINGS.get(depth, RESEARCH_DEPTH_SETTINGS[ResearchDepth.STANDARD])


# Example usage
if __name__ == "__main__":
    # Print application information
    print(f"{APP_NAME} v{APP_VERSION}")
    print(f"Description: {APP_DESCRIPTION}")
    print(f"Repository: {APP_REPOSITORY}")
    
    # Print directory paths
    print(f"Root directory: {ROOT_DIR}")
    print(f"Apps directory: {APPS_DIR}")
    print(f"Services directory: {SERVICES_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Logs directory: {LOGS_DIR}")
    print(f"Temp directory: {TEMP_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Print MCP command example
    brave_search_command = get_mcp_command(
        MCPType.BRAVE_SEARCH,
        use_docker=True,
        BRAVE_API_KEY="your_api_key_here"
    )
    print(f"Brave Search MCP command: {brave_search_command}")
    
    # Print error message example
    error_message = get_error_message(
        ErrorCode.RESOURCE_NOT_FOUND,
        details="User with ID 123 not found"
    )
    print(f"Error message: {error_message}")
    
    # Print agent description example
    research_agent_description = get_agent_description(AgentType.RESEARCH)
    print(f"Research agent description: {research_agent_description}")
    
    # Print agent dependencies example
    research_agent_dependencies = get_agent_dependencies(AgentType.RESEARCH)
    print(f"Research agent dependencies: {research_agent_dependencies}")
    
    # Print research depth settings example
    deep_research_settings = get_research_depth_settings(ResearchDepth.DEEP)
    print(f"Deep research settings: {deep_research_settings}")
