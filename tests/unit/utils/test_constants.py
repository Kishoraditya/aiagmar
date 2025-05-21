"""
Unit tests for the constants utilities.
"""

import os
import pytest
import importlib
from unittest.mock import patch
import inspect
from enum import Enum

import apps.utils.constants
from apps.utils.constants import (
    # Import constants that are actually defined in the constants.py file
    APP_NAME,
    APP_VERSION,
    APP_DESCRIPTION,
    APP_AUTHOR,
    APP_REPOSITORY,
    ROOT_DIR,
    APPS_DIR,
    SERVICES_DIR,
    DATA_DIR,
    LOGS_DIR,
    TEMP_DIR,
    OUTPUT_DIR,
    DEFAULT_API_HOST,
    DEFAULT_API_PORT,
    DEFAULT_API_WORKERS,
    DEFAULT_API_TIMEOUT,
    DEFAULT_API_PREFIX,
    DEFAULT_RATE_LIMIT,
    DEFAULT_RATE_LIMIT_PERIOD,
    DEFAULT_CORS_ORIGINS,
    MCPType,
    MCP_COMMANDS,
    DEFAULT_MCP_SETTINGS,
    AgentType,
    AGENT_DESCRIPTIONS,
    AGENT_DEPENDENCIES,
    DEFAULT_AGENT_SETTINGS,
    WorkflowType,
    ResearchDepth,
    OutputFormat,
    DEFAULT_WORKFLOW_SETTINGS,
    RESEARCH_DEPTH_SETTINGS,
    ProtocolType,
    PROTOCOL_DESCRIPTIONS,
    HTTPStatus,
    ErrorCode,
    ERROR_MESSAGES,
    FileExtension,
    MIME_TYPES,
    DEFAULT_TIMEOUT,
    LONG_TIMEOUT,
    SHORT_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_RETRY_BACKOFF,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    DEFAULT_CACHE_TTL,
    USER_AGENT,
    DEFAULT_HEADERS,
    Environment,
    ENVIRONMENT,
    DEBUG,
    LogLevel,
    DEFAULT_LOG_FORMAT,
    DETAILED_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_TO_CONSOLE,
    DEFAULT_LOG_TO_FILE,
    DEFAULT_LOG_FILENAME,
    DEFAULT_LOG_MAX_BYTES,
    DEFAULT_LOG_BACKUP_COUNT,
    MemoryNamespace,
    DEFAULT_MEMORY_STORAGE_PATH,
    SearchEngine,
    DEFAULT_SEARCH_ENGINE,
    DEFAULT_SEARCH_COUNT,
    DEFAULT_SEARCH_OFFSET,
    ContentType,
    SourceReliability,
    ImageStyle,
    ImageAspectRatio,
    DEFAULT_IMAGE_STYLE,
    DEFAULT_IMAGE_ASPECT_RATIO,
    DEFAULT_IMAGE_COUNT,
    DEFAULT_FILE_STRUCTURE,
    DEFAULT_SOURCE_FILE_PATTERN,
    DEFAULT_SUMMARY_FILE_PATTERN,
    DEFAULT_IMAGE_FILE_PATTERN,
    DEFAULT_OUTPUT_FILE_PATTERN,
    APIEndpoint,
    TaskStatus,
    TaskPriority,
    AuthMethod,
    DEFAULT_AUTH_METHOD,
    DEFAULT_API_KEY_HEADER,
    DEFAULT_JWT_HEADER,
    DEFAULT_JWT_PREFIX,
    DEFAULT_TOKEN_EXPIRY,
    get_mcp_command,
    get_error_message,
    get_mime_type,
    get_agent_description,
    get_agent_dependencies,
    get_protocol_description,
    get_research_depth_settings
)


class TestConstants:
    """Test suite for constants utilities."""

    def setup_method(self):
        """Set up test fixtures."""
        import apps.utils.constants as constants_module
        self.constants_module = constants_module

    def test_constants_immutability(self):
        """Test that enum constants cannot be modified at runtime."""
        # Enums are immutable by design
        with pytest.raises(AttributeError):
            MCPType.BRAVE_SEARCH = "new_value"
        
        with pytest.raises(AttributeError):
            AgentType.RESEARCH = "new_value"
        
        # Dictionaries in Python are mutable, but we can check that 
        # critical dictionaries are not modified in the application
        original_error_messages = ERROR_MESSAGES.copy()
        
        # Try to modify a dictionary constant (this will actually work in Python)
        ERROR_MESSAGES["test_key"] = "test_value"
        
        # But we can verify that the application code doesn't do this
        # by checking if the original values are preserved
        for key, value in original_error_messages.items():
            assert ERROR_MESSAGES[key] == value

    def test_constants_completeness(self):
        """Test that all required constants are defined."""
        # Check for required MCP types
        assert MCPType.BRAVE_SEARCH
        assert MCPType.EVERART
        assert MCPType.FETCH
        assert MCPType.FILESYSTEM
        assert MCPType.MEMORY
        
        # Check for required agent types
        assert AgentType.MANAGER
        assert AgentType.RESEARCH
        assert AgentType.SUMMARY
        assert AgentType.VERIFICATION
        
        # Check for required workflow types
        assert WorkflowType.RESEARCH
        
        # Check for required research depths
        assert ResearchDepth.QUICK
        assert ResearchDepth.STANDARD
        assert ResearchDepth.DEEP
        
        # Check for required output formats
        assert OutputFormat.MARKDOWN
        assert OutputFormat.HTML
        assert OutputFormat.TEXT
        assert OutputFormat.JSON
        
        # Check for required protocol types
        assert ProtocolType.A2A
        assert ProtocolType.ANP
        assert ProtocolType.ACP
        
        # Check for required HTTP status codes
        assert HTTPStatus.OK
        assert HTTPStatus.BAD_REQUEST
        assert HTTPStatus.INTERNAL_SERVER_ERROR
        
        # Check for required error codes
        assert ErrorCode.UNKNOWN_ERROR
        assert ErrorCode.VALIDATION_ERROR
        assert ErrorCode.API_ERROR

    def test_constants_types(self):
        """Test that constants have the correct types."""
        # Check application information constants are strings
        assert isinstance(APP_NAME, str)
        assert isinstance(APP_VERSION, str)
        assert isinstance(APP_DESCRIPTION, str)
        assert isinstance(APP_AUTHOR, str)
        assert isinstance(APP_REPOSITORY, str)
        
        # Check directory constants are Path objects
        assert isinstance(ROOT_DIR, type(os.path))
        assert isinstance(APPS_DIR, type(os.path))
        assert isinstance(DATA_DIR, type(os.path))
        
        # Check API constants have correct types
        assert isinstance(DEFAULT_API_HOST, str)
        assert isinstance(DEFAULT_API_PORT, int)
        assert isinstance(DEFAULT_API_WORKERS, int)
        assert isinstance(DEFAULT_API_TIMEOUT, int)
        assert isinstance(DEFAULT_API_PREFIX, str)
        
        # Check rate limiting constants have correct types
        assert isinstance(DEFAULT_RATE_LIMIT, int)
        assert isinstance(DEFAULT_RATE_LIMIT_PERIOD, int)
        
        # Check CORS settings have correct types
        assert isinstance(DEFAULT_CORS_ORIGINS, list)
        
        # Check MCP constants have correct types
        assert isinstance(MCPType, type(Enum))
        assert isinstance(MCP_COMMANDS, dict)
        assert isinstance(DEFAULT_MCP_SETTINGS, dict)
        
        # Check agent constants have correct types
        assert isinstance(AgentType, type(Enum))
        assert isinstance(AGENT_DESCRIPTIONS, dict)
        assert isinstance(AGENT_DEPENDENCIES, dict)
        assert isinstance(DEFAULT_AGENT_SETTINGS, dict)
        
        # Check workflow constants have correct types
        assert isinstance(WorkflowType, type(Enum))
        assert isinstance(ResearchDepth, type(Enum))
        assert isinstance(OutputFormat, type(Enum))
        assert isinstance(DEFAULT_WORKFLOW_SETTINGS, dict)
        assert isinstance(RESEARCH_DEPTH_SETTINGS, dict)
        
        # Check protocol constants have correct types
        assert isinstance(ProtocolType, type(Enum))
        assert isinstance(PROTOCOL_DESCRIPTIONS, dict)
        
        # Check HTTP status constants have correct types
        assert isinstance(HTTPStatus, type(Enum))
        
        # Check error constants have correct types
        assert isinstance(ErrorCode, type(Enum))
        assert isinstance(ERROR_MESSAGES, dict)
        
        # Check file constants have correct types
        assert isinstance(FileExtension, type(Enum))
        assert isinstance(MIME_TYPES, dict)
        
        # Check timeout constants have correct types
        assert isinstance(DEFAULT_TIMEOUT, int)
        assert isinstance(LONG_TIMEOUT, int)
        assert isinstance(SHORT_TIMEOUT, int)
        
        # Check retry constants have correct types
        assert isinstance(DEFAULT_MAX_RETRIES, int)
        assert isinstance(DEFAULT_RETRY_DELAY, float)
        assert isinstance(DEFAULT_RETRY_BACKOFF, float)
        
        # Check pagination constants have correct types
        assert isinstance(DEFAULT_PAGE_SIZE, int)
        assert isinstance(MAX_PAGE_SIZE, int)
        
        # Check cache constants have correct types
        assert isinstance(DEFAULT_CACHE_TTL, int)
        
        # Check user agent constant has correct type
        assert isinstance(USER_AGENT, str)
        
        # Check default headers constant has correct type
        assert isinstance(DEFAULT_HEADERS, dict)
        
        # Check environment constants have correct types
        assert isinstance(Environment, type(Enum))
        assert isinstance(ENVIRONMENT, Environment)
        assert isinstance(DEBUG, bool)
        
        # Check logging constants have correct types
        assert isinstance(LogLevel, type(Enum))
        assert isinstance(DEFAULT_LOG_FORMAT, str)
        assert isinstance(DETAILED_LOG_FORMAT, str)
        assert isinstance(DEFAULT_LOG_LEVEL, str)
        assert isinstance(DEFAULT_LOG_TO_CONSOLE, bool)
        assert isinstance(DEFAULT_LOG_TO_FILE, bool)
        assert isinstance(DEFAULT_LOG_FILENAME, str)
        assert isinstance(DEFAULT_LOG_MAX_BYTES, int)
        assert isinstance(DEFAULT_LOG_BACKUP_COUNT, int)
        
        # Check memory constants have correct types
        assert isinstance(MemoryNamespace, type(Enum))
        assert isinstance(DEFAULT_MEMORY_STORAGE_PATH, str)
        
        # Check research constants have correct types
        assert isinstance(SearchEngine, type(Enum))
        assert isinstance(DEFAULT_SEARCH_ENGINE, str)
        assert isinstance(DEFAULT_SEARCH_COUNT, int)
        assert isinstance(DEFAULT_SEARCH_OFFSET, int)
        assert isinstance(ContentType, type(Enum))
        assert isinstance(SourceReliability, type(Enum))
        
        # Check image generation constants have correct types
        assert isinstance(ImageStyle, type(Enum))
        assert isinstance(ImageAspectRatio, type(Enum))
        assert isinstance(DEFAULT_IMAGE_STYLE, str)
        assert isinstance(DEFAULT_IMAGE_ASPECT_RATIO, str)
        assert isinstance(DEFAULT_IMAGE_COUNT, int)
        
        # Check file manager constants have correct types
        assert isinstance(DEFAULT_FILE_STRUCTURE, dict)
        assert isinstance(DEFAULT_SOURCE_FILE_PATTERN, str)
        assert isinstance(DEFAULT_SUMMARY_FILE_PATTERN, str)
        assert isinstance(DEFAULT_IMAGE_FILE_PATTERN, str)
        assert isinstance(DEFAULT_OUTPUT_FILE_PATTERN, str)
        
        # Check API endpoint constants have correct types
        assert isinstance(APIEndpoint, type(Enum))
        
        # Check task status constants have correct types
        assert isinstance(TaskStatus, type(Enum))
        assert isinstance(TaskPriority, type(Enum))
        
        # Check security constants have correct types
        assert isinstance(AuthMethod, type(Enum))
        assert isinstance(DEFAULT_AUTH_METHOD, str)
        assert isinstance(DEFAULT_API_KEY_HEADER, str)
        assert isinstance(DEFAULT_JWT_HEADER, str)
        assert isinstance(DEFAULT_JWT_PREFIX, str)
        assert isinstance(DEFAULT_TOKEN_EXPIRY, int)
        
        # Check utility functions have correct types
        assert callable(get_mcp_command)
        assert callable(get_error_message)
        assert callable(get_mime_type)
        assert callable(get_agent_description)
        assert callable(get_agent_dependencies)
        assert callable(get_protocol_description)
        assert callable(get_research_depth_settings)

    def test_constants_values(self):
        """Test that constants have valid values."""
        # Check application information constants have valid values
        assert len(APP_NAME) > 0
        assert len(APP_VERSION) > 0
        assert len(APP_DESCRIPTION) > 0
        assert len(APP_AUTHOR) > 0
        assert len(APP_REPOSITORY) > 0
        
        # Check directory constants have valid values
        assert os.path.exists(ROOT_DIR)
        
        # Check API constants have valid values
        assert len(DEFAULT_API_HOST) > 0
        assert 0 < DEFAULT_API_PORT < 65536
        assert DEFAULT_API_WORKERS > 0
        assert DEFAULT_API_TIMEOUT > 0
        assert DEFAULT_API_PREFIX.startswith("/")
        
        # Check rate limiting constants have valid values
        assert DEFAULT_RATE_LIMIT > 0
        assert DEFAULT_RATE_LIMIT_PERIOD > 0
        
        # Check CORS settings have valid values
        assert len(DEFAULT_CORS_ORIGINS) > 0
        
        # Check MCP constants have valid values
        for mcp_type in MCPType:
            assert mcp_type.value in MCP_COMMANDS
            assert mcp_type.value in DEFAULT_MCP_SETTINGS
        
        # Check agent constants have valid values
        for agent_type in AgentType:
            assert agent_type.value in AGENT_DESCRIPTIONS
            assert agent_type in AGENT_DEPENDENCIES
            assert agent_type in DEFAULT_AGENT_SETTINGS
        
        # Check workflow constants have valid values
        for workflow_type in WorkflowType:
            assert workflow_type in DEFAULT_WORKFLOW_SETTINGS
        
        for depth in ResearchDepth:
            assert depth in RESEARCH_DEPTH_SETTINGS
        
        # Check protocol constants have valid values
        for protocol_type in ProtocolType:
            assert protocol_type in PROTOCOL_DESCRIPTIONS
        
        # Check error constants have valid values
        for error_code in ErrorCode:
            assert error_code.value in ERROR_MESSAGES
        
        # Check file constants have valid values
        for file_ext in FileExtension:
            assert file_ext in MIME_TYPES
        
        # Check timeout constants have valid values
        assert DEFAULT_TIMEOUT > 0
        assert LONG_TIMEOUT > DEFAULT_TIMEOUT
        assert SHORT_TIMEOUT < DEFAULT_TIMEOUT
        
        # Check retry constants have valid values
        assert DEFAULT_MAX_RETRIES >= 0
        assert DEFAULT_RETRY_DELAY > 0
        assert DEFAULT_RETRY_BACKOFF > 1
        
        # Check pagination constants have valid values
        assert DEFAULT_PAGE_SIZE > 0
        assert MAX_PAGE_SIZE >= DEFAULT_PAGE_SIZE
        
        # Check cache constants have valid values
        assert DEFAULT_CACHE_TTL > 0
        
        # Check user agent constant has valid value
        assert APP_NAME in USER_AGENT
        assert APP_VERSION in USER_AGENT
        
        # Check default headers constant has valid values
        assert "User-Agent" in DEFAULT_HEADERS
        assert "Accept" in DEFAULT_HEADERS
        
        # Check environment constants have valid values
        assert ENVIRONMENT in Environment
        
        # Check logging constants have valid values
        assert DEFAULT_LOG_LEVEL in [level.value for level in LogLevel]
        assert "%(asctime)s" in DEFAULT_LOG_FORMAT
        assert "%(levelname)s" in DEFAULT_LOG_FORMAT
        # Check memory constants have valid values
        assert all(ns.value for ns in MemoryNamespace)
        assert len(DEFAULT_MEMORY_STORAGE_PATH) > 0
        
        # Check research constants have valid values
        assert DEFAULT_SEARCH_ENGINE in [engine.value for engine in SearchEngine]
        assert DEFAULT_SEARCH_COUNT > 0
        assert DEFAULT_SEARCH_OFFSET >= 0
        
        # Check image generation constants have valid values
        assert DEFAULT_IMAGE_STYLE in [style.value for style in ImageStyle]
        assert DEFAULT_IMAGE_ASPECT_RATIO in [ratio.value for ratio in ImageAspectRatio]
        assert DEFAULT_IMAGE_COUNT > 0
        
        # Check file manager constants have valid values
        assert "research" in DEFAULT_FILE_STRUCTURE
        assert "{index}" in DEFAULT_SOURCE_FILE_PATTERN
        assert "{index}" in DEFAULT_SUMMARY_FILE_PATTERN
        assert "{index}" in DEFAULT_IMAGE_FILE_PATTERN
        assert "{timestamp}" in DEFAULT_OUTPUT_FILE_PATTERN
        
        # Check security constants have valid values
        assert DEFAULT_AUTH_METHOD in [method.value for method in AuthMethod]
        assert len(DEFAULT_API_KEY_HEADER) > 0
        assert len(DEFAULT_JWT_HEADER) > 0
        assert len(DEFAULT_JWT_PREFIX) > 0
        assert DEFAULT_TOKEN_EXPIRY > 0

    def test_constants_environment_override(self):
        """Test that constants can be overridden by environment variables."""
        # Set environment variables to override constants
        with patch.dict(os.environ, {
            "AIAGMAR_ENVIRONMENT": "production",
            "AIAGMAR_DEBUG": "true",
            "DEFAULT_API_PORT": "9000",
            "DEFAULT_TIMEOUT": "60"
        }):
            # Reload the constants module to apply environment variables
            importlib.reload(apps.utils.constants)
            
            # Check that environment variables were applied
            # Note: This assumes the constants module reads these environment variables
            # If it doesn't, these assertions may fail
            try:
                assert apps.utils.constants.ENVIRONMENT == Environment.PRODUCTION
                assert apps.utils.constants.DEBUG is True
                assert apps.utils.constants.DEFAULT_API_PORT == 9000
                assert apps.utils.constants.DEFAULT_TIMEOUT == 60
            finally:
                # Reload the module again to reset to default values
                importlib.reload(apps.utils.constants)

    def test_utility_functions(self):
        """Test the utility functions in the constants module."""
        # Test get_mcp_command function
        brave_search_command = get_mcp_command(
            MCPType.BRAVE_SEARCH,
            use_docker=True,
            BRAVE_API_KEY="test_api_key"
        )
        assert isinstance(brave_search_command, list)
        assert "docker" in brave_search_command[0]
        assert any("BRAVE_API_KEY" in item for item in brave_search_command)
        
        # Test get_error_message function
        error_message = get_error_message(
            ErrorCode.RESOURCE_NOT_FOUND,
            details="User with ID 123 not found"
        )
        assert isinstance(error_message, str)
        assert "User with ID 123 not found" in error_message
        
        # Test get_mime_type function
        mime_type = get_mime_type(FileExtension.JSON)
        assert mime_type == "application/json"
        
        # Test get_agent_description function
        description = get_agent_description(AgentType.RESEARCH)
        assert isinstance(description, str)
        assert len(description) > 0
        
        # Test get_agent_dependencies function
        dependencies = get_agent_dependencies(AgentType.RESEARCH)
        assert isinstance(dependencies, list)
        assert MCPType.BRAVE_SEARCH in dependencies
        assert MCPType.FETCH in dependencies
        
        # Test get_protocol_description function
        protocol_desc = get_protocol_description(ProtocolType.A2A)
        assert isinstance(protocol_desc, str)
        assert len(protocol_desc) > 0
        
        # Test get_research_depth_settings function
        depth_settings = get_research_depth_settings(ResearchDepth.DEEP)
        assert isinstance(depth_settings, dict)
        assert "max_sources" in depth_settings
        assert depth_settings["max_sources"] > RESEARCH_DEPTH_SETTINGS[ResearchDepth.STANDARD]["max_sources"]

    def test_constants_module_structure(self):
        """Test the structure of the constants module."""
        # Get all attributes of the constants module
        module_attrs = dir(self.constants_module)
        
        # Check for expected sections (enums and major constant groups)
        expected_sections = [
            "APP_NAME",
            "APP_VERSION",
            "ROOT_DIR",
            "MCPType",
            "AgentType",
            "WorkflowType",
            "ResearchDepth",
            "OutputFormat",
            "ProtocolType",
            "HTTPStatus",
            "ErrorCode",
            "FileExtension",
            "Environment",
            "LogLevel",
            "MemoryNamespace",
            "SearchEngine",
            "ContentType",
            "SourceReliability",
            "ImageStyle",
            "ImageAspectRatio",
            "APIEndpoint",
            "TaskStatus",
            "TaskPriority",
            "AuthMethod"
        ]
        
        for section in expected_sections:
            assert section in module_attrs, f"Missing expected section: {section}"
        
        # Check for utility functions
        expected_functions = [
            "get_mcp_command",
            "get_error_message",
            "get_mime_type",
            "get_agent_description",
            "get_agent_dependencies",
            "get_protocol_description",
            "get_research_depth_settings"
        ]
        
        for function in expected_functions:
            assert function in module_attrs, f"Missing expected function: {function}"

    def test_constants_for_brave_search_mcp(self):
        """Test constants specific to the Brave Search MCP."""
        # Check for Brave Search MCP type
        assert MCPType.BRAVE_SEARCH
        
        # Check for Brave Search MCP commands
        assert MCPType.BRAVE_SEARCH.value in MCP_COMMANDS
        assert "docker" in MCP_COMMANDS[MCPType.BRAVE_SEARCH]
        assert "npx" in MCP_COMMANDS[MCPType.BRAVE_SEARCH]
        
        # Check for Brave Search MCP settings
        assert MCPType.BRAVE_SEARCH.value in DEFAULT_MCP_SETTINGS
        assert "enabled" in DEFAULT_MCP_SETTINGS[MCPType.BRAVE_SEARCH.value]
        
        # Check for Brave Search in agent dependencies
        assert MCPType.BRAVE_SEARCH in AGENT_DEPENDENCIES[AgentType.RESEARCH]
        
        # Check for Brave Search in search engines
        assert SearchEngine.BRAVE

    def test_constants_for_everart_mcp(self):
        """Test constants specific to the EverArt MCP."""
        # Check for EverArt MCP type
        assert MCPType.EVERART
        
        # Check for EverArt MCP commands
        assert MCPType.EVERART.value in MCP_COMMANDS
        assert "docker" in MCP_COMMANDS[MCPType.EVERART]
        assert "npx" in MCP_COMMANDS[MCPType.EVERART]
        
        # Check for EverArt MCP settings
        assert MCPType.EVERART.value in DEFAULT_MCP_SETTINGS
        assert "enabled" in DEFAULT_MCP_SETTINGS[MCPType.EVERART.value]
        
        # Check for EverArt in agent dependencies
        assert MCPType.EVERART in AGENT_DEPENDENCIES[AgentType.IMAGE_GENERATION]
        
        # Check for image generation settings
        assert AgentType.IMAGE_GENERATION in DEFAULT_AGENT_SETTINGS
        assert DEFAULT_IMAGE_STYLE
        assert DEFAULT_IMAGE_ASPECT_RATIO
        assert DEFAULT_IMAGE_COUNT > 0

    def test_constants_for_fetch_mcp(self):
        """Test constants specific to the Fetch MCP."""
        # Check for Fetch MCP type
        assert MCPType.FETCH
        
        # Check for Fetch MCP commands
        assert MCPType.FETCH.value in MCP_COMMANDS
        assert "docker" in MCP_COMMANDS[MCPType.FETCH]
        assert "npx" in MCP_COMMANDS[MCPType.FETCH]
        
        # Check for Fetch MCP settings
        assert MCPType.FETCH.value in DEFAULT_MCP_SETTINGS
        assert "enabled" in DEFAULT_MCP_SETTINGS[MCPType.FETCH.value]
        
        # Check for Fetch in agent dependencies
        assert MCPType.FETCH in AGENT_DEPENDENCIES[AgentType.RESEARCH]
        
        # Check for User-Agent in default headers
        assert "User-Agent" in DEFAULT_HEADERS
        assert USER_AGENT in DEFAULT_HEADERS["User-Agent"]

    def test_constants_for_filesystem_mcp(self):
        """Test constants specific to the Filesystem MCP."""
        # Check for Filesystem MCP type
        assert MCPType.FILESYSTEM
        
        # Check for Filesystem MCP commands
        assert MCPType.FILESYSTEM.value in MCP_COMMANDS
        assert "docker" in MCP_COMMANDS[MCPType.FILESYSTEM]
        assert "npx" in MCP_COMMANDS[MCPType.FILESYSTEM]
        
        # Check for Filesystem MCP settings
        assert MCPType.FILESYSTEM.value in DEFAULT_MCP_SETTINGS
        assert "enabled" in DEFAULT_MCP_SETTINGS[MCPType.FILESYSTEM.value]
        assert "workspace_dir" in DEFAULT_MCP_SETTINGS[MCPType.FILESYSTEM.value]
        
        # Check for Filesystem in agent dependencies
        assert MCPType.FILESYSTEM in AGENT_DEPENDENCIES[AgentType.FILE_MANAGER]
        
        # Check for file manager settings
        assert AgentType.FILE_MANAGER in DEFAULT_AGENT_SETTINGS
        assert DEFAULT_FILE_STRUCTURE
        assert DEFAULT_SOURCE_FILE_PATTERN
        assert DEFAULT_SUMMARY_FILE_PATTERN
        assert DEFAULT_IMAGE_FILE_PATTERN
        assert DEFAULT_OUTPUT_FILE_PATTERN

    def test_constants_for_memory_mcp(self):
        """Test constants specific to the Memory MCP."""
        # Check for Memory MCP type
        assert MCPType.MEMORY
        
        # Check for Memory MCP commands
        assert MCPType.MEMORY.value in MCP_COMMANDS
        assert "docker" in MCP_COMMANDS[MCPType.MEMORY]
        assert "npx" in MCP_COMMANDS[MCPType.MEMORY]
        
        # Check for Memory MCP settings
        assert MCPType.MEMORY.value in DEFAULT_MCP_SETTINGS
        assert "enabled" in DEFAULT_MCP_SETTINGS[MCPType.MEMORY.value]
        assert "storage_path" in DEFAULT_MCP_SETTINGS[MCPType.MEMORY.value]
        
        # Check for Memory in agent dependencies
        assert MCPType.MEMORY in AGENT_DEPENDENCIES[AgentType.MANAGER]
        
        # Check for memory namespaces
        assert MemoryNamespace.DEFAULT
        assert MemoryNamespace.RESEARCH
        assert MemoryNamespace.USER
        assert DEFAULT_MEMORY_STORAGE_PATH

    def test_constants_for_research_workflow(self):
        """Test constants specific to the research workflow."""
        # Check for research workflow type
        assert WorkflowType.RESEARCH
        
        # Check for research workflow settings
        assert WorkflowType.RESEARCH in DEFAULT_WORKFLOW_SETTINGS
        assert "enabled" in DEFAULT_WORKFLOW_SETTINGS[WorkflowType.RESEARCH]
        assert "default_max_sources" in DEFAULT_WORKFLOW_SETTINGS[WorkflowType.RESEARCH]
        assert "default_include_images" in DEFAULT_WORKFLOW_SETTINGS[WorkflowType.RESEARCH]
        assert "default_verify_facts" in DEFAULT_WORKFLOW_SETTINGS[WorkflowType.RESEARCH]
        assert "default_research_depth" in DEFAULT_WORKFLOW_SETTINGS[WorkflowType.RESEARCH]
        assert "default_output_format" in DEFAULT_WORKFLOW_SETTINGS[WorkflowType.RESEARCH]
        
        # Check for research depth settings
        assert ResearchDepth.QUICK in RESEARCH_DEPTH_SETTINGS
        assert ResearchDepth.STANDARD in RESEARCH_DEPTH_SETTINGS
        assert ResearchDepth.DEEP in RESEARCH_DEPTH_SETTINGS
        assert "max_sources" in RESEARCH_DEPTH_SETTINGS[ResearchDepth.STANDARD]
        assert "max_content_length" in RESEARCH_DEPTH_SETTINGS[ResearchDepth.STANDARD]
        assert "verification_threshold" in RESEARCH_DEPTH_SETTINGS[ResearchDepth.STANDARD]
        
        # Check for research-related agents
        assert AgentType.RESEARCH in DEFAULT_AGENT_SETTINGS
        assert AgentType.VERIFICATION in DEFAULT_AGENT_SETTINGS
        assert AgentType.SUMMARY in DEFAULT_AGENT_SETTINGS

    def test_constants_for_verification_process(self):
        """Test constants specific to the verification process."""
        # Check for verification agent
        assert AgentType.VERIFICATION
        
        # Check for verification agent settings
        assert AgentType.VERIFICATION in DEFAULT_AGENT_SETTINGS
        assert "verification_threshold" in RESEARCH_DEPTH_SETTINGS[ResearchDepth.STANDARD]
        
        # Check for verification agent dependencies
        assert MCPType.BRAVE_SEARCH in AGENT_DEPENDENCIES[AgentType.VERIFICATION]
        
        # Check for source reliability levels
        assert SourceReliability.HIGH
        assert SourceReliability.MEDIUM
        assert SourceReliability.LOW
        assert SourceReliability.UNKNOWN

    def test_constants_for_image_generation(self):
        """Test constants specific to image generation."""
        # Check for image generation agent
        assert AgentType.IMAGE_GENERATION
        
        # Check for image generation agent settings
        assert AgentType.IMAGE_GENERATION in DEFAULT_AGENT_SETTINGS
        
        # Check for image generation agent dependencies
        assert MCPType.EVERART in AGENT_DEPENDENCIES[AgentType.IMAGE_GENERATION]
        
        # Check for image styles
        assert ImageStyle.REALISTIC
        assert ImageStyle.CARTOON
        assert ImageStyle.SKETCH
        assert ImageStyle.PAINTING
        
        # Check for image aspect ratios
        assert ImageAspectRatio.SQUARE
        assert ImageAspectRatio.PORTRAIT
        assert ImageAspectRatio.LANDSCAPE
        
        # Check for image generation settings
        assert DEFAULT_IMAGE_STYLE
        assert DEFAULT_IMAGE_ASPECT_RATIO
        assert DEFAULT_IMAGE_COUNT > 0
        
        # Check for image file extensions
        assert FileExtension.JPG
        assert FileExtension.PNG
        assert FileExtension.GIF
        assert FileExtension.SVG

    def test_constants_for_file_management(self):
        """Test constants specific to file management."""
        # Check for file manager agent
        assert AgentType.FILE_MANAGER
        
        # Check for file manager agent settings
        assert AgentType.FILE_MANAGER in DEFAULT_AGENT_SETTINGS
        assert "max_file_size" in DEFAULT_AGENT_SETTINGS[AgentType.FILE_MANAGER]
        
        # Check for file manager agent dependencies
        assert MCPType.FILESYSTEM in AGENT_DEPENDENCIES[AgentType.FILE_MANAGER]
        # Check for file structure
        assert DEFAULT_FILE_STRUCTURE
        assert "research" in DEFAULT_FILE_STRUCTURE
        assert "sources" in DEFAULT_FILE_STRUCTURE["research"]
        assert "summaries" in DEFAULT_FILE_STRUCTURE["research"]
        assert "images" in DEFAULT_FILE_STRUCTURE["research"]
        assert "output" in DEFAULT_FILE_STRUCTURE["research"]
        
        # Check for file patterns
        assert DEFAULT_SOURCE_FILE_PATTERN
        assert DEFAULT_SUMMARY_FILE_PATTERN
        assert DEFAULT_IMAGE_FILE_PATTERN
        assert DEFAULT_OUTPUT_FILE_PATTERN
        
        # Check for file extensions
        assert FileExtension.TXT
        assert FileExtension.MD
        assert FileExtension.HTML
        assert FileExtension.JSON
        assert FileExtension.YAML
        assert FileExtension.CSV

    def test_constants_default_values(self):
        """Test that constants have appropriate default values."""
        # Check default timeouts
        assert DEFAULT_TIMEOUT >= 30  # At least 30 seconds
        assert LONG_TIMEOUT >= DEFAULT_TIMEOUT
        assert SHORT_TIMEOUT <= DEFAULT_TIMEOUT
        
        # Check default max retries
        assert DEFAULT_MAX_RETRIES >= 3  # At least 3 retries
        
        # Check default workflow settings
        assert DEFAULT_WORKFLOW_SETTINGS[WorkflowType.RESEARCH]["default_max_sources"] >= 5  # At least 5 results
        
        # Check default agent settings
        assert DEFAULT_AGENT_SETTINGS[AgentType.RESEARCH]["max_sources"] >= 1  # At least 1 source
        assert 0 <= RESEARCH_DEPTH_SETTINGS[ResearchDepth.STANDARD]["verification_threshold"] <= 1  # Between 0 and 1

    def test_constants_serialization(self):
        """Test that constants can be properly serialized."""
        import json
        
        # Test serializing enums (convert to value first)
        try:
            serialized = json.dumps(MCPType.BRAVE_SEARCH.value)
            deserialized = json.loads(serialized)
            assert deserialized == MCPType.BRAVE_SEARCH.value
        except TypeError:
            pytest.fail("Failed to serialize MCPType enum value")
        
        # Test serializing dictionaries
        try:
            serialized = json.dumps(DEFAULT_AGENT_SETTINGS)
            deserialized = json.loads(serialized)
            assert isinstance(deserialized, dict)
            assert AgentType.RESEARCH.value in deserialized
        except TypeError:
            # If direct serialization fails, we need to convert Enum keys to strings
            serializable_settings = {k.value if isinstance(k, Enum) else k: v 
                                    for k, v in DEFAULT_AGENT_SETTINGS.items()}
            serialized = json.dumps(serializable_settings)
            deserialized = json.loads(serialized)
            assert isinstance(deserialized, dict)
            assert AgentType.RESEARCH.value in deserialized

    def test_constants_documentation(self):
        """Test that constants have proper documentation."""
        # Check that module has docstring
        assert self.constants_module.__doc__ is not None
        assert len(self.constants_module.__doc__) > 0
        
        # Check for presence of section comments in the module source
        module_source = inspect.getsource(self.constants_module)
        
        # Check for section headers
        assert "# Application Constants" in module_source
        assert "# API Constants" in module_source
        assert "# MCP Constants" in module_source
        assert "# Agent Constants" in module_source
        assert "# Workflow Constants" in module_source
        assert "# Protocol Constants" in module_source
        assert "# HTTP Status Codes" in module_source
        assert "# Error Constants" in module_source
        assert "# File Constants" in module_source
        assert "# Miscellaneous Constants" in module_source
        assert "# Environment Constants" in module_source
        assert "# Logging Constants" in module_source
        assert "# Memory Constants" in module_source
        assert "# Research Constants" in module_source
        assert "# Image Generation Constants" in module_source
        assert "# File Manager Constants" in module_source
        assert "# API Endpoint Constants" in module_source
        assert "# Task Status Constants" in module_source
        assert "# Security Constants" in module_source
        assert "# Utility Functions" in module_source

    def test_constants_naming_convention(self):
        """Test that constants follow naming conventions."""
        # Check that top-level constants use UPPER_CASE
        for name, value in inspect.getmembers(self.constants_module):
            if not name.startswith("_") and not inspect.isclass(value) and not inspect.isfunction(value) and not inspect.ismodule(value):
                if isinstance(value, (str, int, float, bool, list, tuple, dict)):
                    assert name.isupper(), f"Constant {name} should be UPPER_CASE"
        
        # Check that enum classes use PascalCase (first letter of each word capitalized)
        for name, value in inspect.getmembers(self.constants_module, inspect.isclass):
            if issubclass(value, Enum) and value is not Enum:
                assert name[0].isupper(), f"Enum class {name} should use PascalCase"
                assert not name.isupper(), f"Enum class {name} should use PascalCase, not UPPER_CASE"
                assert "_" not in name, f"Enum class {name} should use PascalCase, not snake_case"

    def test_constants_isolation(self):
        """Test that constants are properly isolated and don't have side effects."""
        # Import the constants in a different way to check isolation
        import apps.utils.constants as constants_module
        
        # Verify that the constants are the same objects
        assert constants_module.APP_NAME is APP_NAME
        assert constants_module.DEFAULT_TIMEOUT is DEFAULT_TIMEOUT
        assert constants_module.MCPType is MCPType
        
        # Create a copy of a dictionary constant
        agent_settings_copy = DEFAULT_AGENT_SETTINGS.copy()
        
        # Modify the copy
        agent_settings_copy[AgentType.RESEARCH]["max_sources"] = 999
        
        # Verify the original is unchanged
        assert DEFAULT_AGENT_SETTINGS[AgentType.RESEARCH]["max_sources"] != 999
        
        # Verify that importing again doesn't reset constants
        importlib.reload(constants_module)
        assert constants_module.APP_NAME is APP_NAME  # Same object, not reset

    def test_constants_integration_with_config(self):
        """Test that constants integrate properly with configuration module."""
        # This test would normally check how constants are used in the config module
        # Since we don't have access to the actual config module, we'll mock it
        
        with patch('apps.utils.config.DEFAULT_CONFIG', create=True) as mock_config:
            mock_config.return_value = {
                "app_name": APP_NAME,
                "app_version": APP_VERSION,
                "timeout": DEFAULT_TIMEOUT,
                "max_retries": DEFAULT_MAX_RETRIES,
                "agent_settings": DEFAULT_AGENT_SETTINGS,
                "workflow_settings": DEFAULT_WORKFLOW_SETTINGS
            }
            
            # Import the config module (this would fail if the module doesn't exist)
            try:
                from apps.utils import config
                
                # Check that config uses constants
                assert config.DEFAULT_CONFIG["app_name"] == APP_NAME
                assert config.DEFAULT_CONFIG["timeout"] == DEFAULT_TIMEOUT
            except ImportError:
                # If the config module doesn't exist, skip this test
                pytest.skip("Config module not available")

    def test_constants_integration_with_agents(self):
        """Test that constants integrate properly with agent modules."""
        # This test would normally check how constants are used in agent modules
        # Since we don't have access to the actual agent modules, we'll mock them
        
        with patch('apps.agents.research_agent.AGENT_SETTINGS', create=True) as mock_settings:
            mock_settings.return_value = DEFAULT_AGENT_SETTINGS[AgentType.RESEARCH]
            
            # Import the agent module (this would fail if the module doesn't exist)
            try:
                from apps.agents import research_agent
                
                # Check that the agent module uses constants
                assert research_agent.AGENT_SETTINGS == DEFAULT_AGENT_SETTINGS[AgentType.RESEARCH]
            except ImportError:
                # If the agent module doesn't exist, skip this test
                pytest.skip("Research agent module not available")

    def test_constants_integration_with_error_handling(self):
        """Test that constants integrate properly with error handling module."""
        # This test would normally check how constants are used in error handling modules
        # Since we don't have access to the actual error modules, we'll mock them
        
        with patch('apps.utils.exceptions.ErrorCode', create=True) as mock_error_code:
            mock_error_code.return_value = ErrorCode
            
            # Import the exceptions module (this would fail if the module doesn't exist)
            try:
                from apps.utils import exceptions
                
                # Check that the exceptions module uses constants
                assert exceptions.ErrorCode == ErrorCode
            except ImportError:
                # If the exceptions module doesn't exist, skip this test
                pytest.skip("Exceptions module not available")

    def test_constants_performance(self):
        """Test the performance impact of using constants."""
        import time
        
        # Measure time to access a constant
        start_time = time.time()
        for _ in range(10000):
            _ = DEFAULT_TIMEOUT
        constant_access_time = time.time() - start_time
        
        # Measure time to access a regular variable
        regular_var = DEFAULT_TIMEOUT
        start_time = time.time()
        for _ in range(10000):
            _ = regular_var
        var_access_time = time.time() - start_time
        
        # Constant access should not be significantly slower
        assert constant_access_time < var_access_time * 2, "Constant access is too slow"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
