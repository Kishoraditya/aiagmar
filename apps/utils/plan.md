# plan

ased on the codebase we've worked on so far, let's plan the utility modules, examples, and integration points.

Utility Modules
utils/logger.py
Purpose: Centralized logging configuration for all components
Contents:
Custom logger setup with different levels (DEBUG, INFO, WARNING, ERROR)
Log formatting with timestamps, module names, and log levels
File and console handlers
Context-aware logging that includes session IDs and agent names
2. utils/config.py
Purpose: Configuration management for the entire system
Contents:
Environment variable loading (API keys, paths, etc.)
Configuration file parsing (YAML/JSON)
Default configurations for agents and workflows
Dynamic configuration updates
Configuration validation
3. utils/decorators.py
Purpose: Reusable decorators for common patterns
Contents:
@timed_execution - Measure and log execution time
@retry - Automatic retry logic for API calls
@validate_input - Input validation for functions
@rate_limit - Rate limiting for API calls
@cache_result - Caching for expensive operations
4. utils/exceptions.py
Purpose: Custom exception classes for the system
Contents:
AgentError - Base exception for all agent errors
MCPConnectionError - MCP connection issues
ValidationError - Input validation failures
ResourceNotFoundError - Missing resources
WorkflowError - Workflow execution issues
Exception handling utilities
5. utils/constants.py
Purpose: System-wide constants
Contents:
API endpoint URLs
Default timeouts
File paths and directory structures
Status codes and messages
Content types and MIME types
6. utils/helpers.py
Purpose: General utility functions
Contents:
URL parsing and sanitization
File operations (safe read/write)
Text processing (truncation, sanitization)
Data conversion utilities
Date/time formatting
JSON handling utilities
7. utils/validation.py
Purpose: Input validation utilities
Contents:
Schema validation functions
Type checking utilities
URL validation
File path validation
Content validation (size limits, format checks)

This implementation provides:

Centralized Logging Configuration: A setup_logging function to configure logging for the entire application.

Context-Aware Logging: The ContextAdapter and get_logger function allow adding context information like session IDs and agent names to log messages.

LoggerMixin: A mixin class that can be added to any class to provide logging capabilities.

Flexible Output Options: Support for logging to both console and files, with configurable log rotation.

Environment Variable Configuration: Logging settings can be controlled via environment variables.

Detailed Formatting: Option for detailed log format that includes context information.

This logging system will be used throughout the application to provide consistent, context-aware logging that makes it easier to trace execution flows and debug issues.

This comprehensive helpers.py module provides a wide range of utility functions that will be useful throughout the application, including:

URL Handling: Functions for sanitizing URLs, extracting domains, and converting URLs to filenames.
File Operations: Safe file reading/writing, directory management, and file type detection.
Text Processing: Text truncation, HTML cleaning, keyword extraction, and summarization.
Data Conversion: Safe JSON handling, dictionary flattening, and merging.
Date and Time: Timestamp formatting and generation.
File Management: File downloading, hash calculation, MIME type detection, and directory structure creation.
Miscellaneous: Session ID generation, retry logic, validation functions, and more.
These utilities will help standardize common operations across the application, reduce code duplication, and improve error handling.

This comprehensive configuration module provides:

Flexible Configuration Sources: Loads configuration from environment variables, YAML/JSON files, and provides sensible defaults.

Hierarchical Configuration: Supports nested configuration values with dot notation access.

Type Conversion: Automatically converts environment variables to appropriate types (boolean, integer, float, JSON).

Validation: Validates configuration values to ensure they are valid.

Helper Functions: Provides convenient functions for common configuration tasks.

Environment Detection: Easy methods to check if running in development, testing, or production.

API Key Management: Centralized access to API keys from environment variables or configuration files.

MCP and Agent Configuration: Specific helpers for MCP and agent configuration.

The module is designed to be flexible and extensible, allowing for easy configuration of all aspects of the application from a single source.

his comprehensive constants.py module provides a centralized location for all constant values used throughout the application, including:

Application Information: Name, version, description, and repository.
Directory Paths: Root, apps, services, data, logs, temp, and output directories.
API Constants: Default settings for the API server, rate limiting, and CORS.
MCP Constants: MCP types, commands, and default settings.
Agent Constants: Agent types, descriptions, dependencies, and default settings.
Workflow Constants: Workflow types, research depth options, output formats, and default settings.
Protocol Constants: Protocol types and descriptions.
HTTP Status Codes: Standard HTTP status codes.
Error Constants: Error codes and messages.
File Constants: File extensions and MIME types.
Miscellaneous Constants: Timeout values, retry settings, pagination settings, etc.
Environment Constants: Environment types and debug mode.
Logging Constants: Log levels, formats, and default settings.
Memory Constants: Memory namespaces and default settings.
Research Constants: Search engines, content types, and source reliability levels.
Image Generation Constants: Image styles, aspect ratios, and default settings.
File Manager Constants: Default file structure and naming patterns.
API Endpoint Constants: API endpoints.
Task Status Constants: Task status and priority.
Security Constants: Authentication methods and default settings.
The module also includes utility functions to work with these constants, making it easy to access and use them throughout the application.

This comprehensive decorators.py module provides a wide range of useful decorators for the application, including:

Performance Decorators:

timed: Measures and logs the execution time of a function
async_timed: Measures and logs the execution time of an async function
Retry Decorators:

retry: Retries a function on specified exceptions
async_retry: Retries an async function on specified exceptions
Caching Decorators:

cached: Caches function results with TTL and size limits
async_cached: Caches async function results with TTL and size limits
Logging Decorators:

log_call: Logs function calls and results
async_log_call: Logs async function calls and results
Rate Limiting Decorators:

rate_limit: Limits the rate of function calls
async_rate_limit: Limits the rate of async function calls
Validation Decorators:

validate_args: Validates function arguments
validate_return: Validates function return values
async_validate_args: Validates async function arguments
async_validate_return: Validates async function return values
Error Handling Decorators:

handle_exceptions: Handles exceptions in functions
async_handle_exceptions: Handles exceptions in async functions
Miscellaneous Decorators:

deprecated: Marks functions as deprecated
singleton: Creates singleton classes
memoize: Memoizes function results (cache with no expiration)
async_memoize: Memoizes async function results
synchronized: Synchronizes function execution with a lock
Each decorator is well-documented with docstrings and includes type hints for better IDE support. The module also includes examples of how to use each decorator.

This comprehensive exceptions.py module defines a hierarchy of custom exceptions for the application, including:

Base Exception:

BaseError: Base class for all custom exceptions with error code, status code, and details
General Exceptions:

ValidationError: Raised when validation fails
ConfigurationError: Raised when there is a configuration error
Authentication and Authorization Exceptions:

AuthenticationError: Raised when authentication fails
AuthorizationError: Raised when authorization fails
Resource Exceptions:

ResourceNotFoundError: Raised when a resource is not found
ResourceAlreadyExistsError: Raised when a resource already exists
API Exceptions:

RateLimitExceededError: Raised when rate limit is exceeded
APIError: Raised when there is an API error
MCP Exceptions:

MCPError: Raised when there is an MCP error
MCPNotAvailableError: Raised when an MCP is not available
Agent Exceptions:

AgentError: Raised when there is an agent error
AgentNotAvailableError: Raised when an agent is not available
Workflow Exceptions:

WorkflowError: Raised when there is a workflow error
WorkflowNotAvailableError: Raised when a workflow is not available
Each exception includes:

A descriptive error message
An error code from the ErrorCode enum
An HTTP status code from the HTTPStatus enum
Optional additional details
The exceptions can be converted to dictionaries for API responses and have a string representation for logging. This structured approach to exceptions makes error handling more consistent and informative

This comprehensive validation.py module provides a wide range of validation utilities for the application, including:

Basic Validation Functions:

validate_required: Validates that a value is not None or empty
validate_type: Validates that a value is of the expected type
validate_enum: Validates that a value is a valid enum value
validate_range: Validates that a numeric value is within a specified range
validate_length: Validates that a value's length is within a specified range
validate_regex: Validates that a string matches a regular expression pattern
validate_email: Validates that a string is a valid email address
validate_url: Validates that a string is a valid URL
validate_uuid: Validates that a string is a valid UUID
validate_date: Validates that a string is a valid date in the specified format
validate_json: Validates that a string is valid JSON
Application-Specific Validation Functions:

validate_search_query: Validates a search query
validate_search_engine: Validates a search engine
validate_search_count: Validates a search count
validate_search_offset: Validates a search offset
validate_image_prompt: Validates an image generation prompt
validate_image_style: Validates an image style
validate_image_aspect_ratio: Validates an image aspect ratio
validate_image_count: Validates an image count
validate_file_path: Validates a file path
validate_file_extension: Validates a file extension
validate_research_depth: Validates a research depth
validate_source_reliability: Validates a source reliability
validate_task_status: Validates a task status
validate_task_priority: Validates a task priority
validate_memory_key: Validates a memory key
validate_memory_namespace: Validates a memory namespace
Validation Decorator Functions:

validate_with: Creates a validator function that applies a validation function
validate_dict_keys: Creates a validator function that validates dictionary keys
validate_list_items: Creates a validator function that validates list items
Composite Validation Functions:

validate_search_params: Validates search parameters
validate_image_params: Validates image generation parameters
validate_memory_params: Validates memory parameters
Schema Validation Functions:

validate_schema: Validates data against a JSON schema
validate_model: Validates data against a Pydantic model
Each validation function raises a ValidationError with a descriptive message and details when validation fails. This structured approach to validation makes input validation more consistent and informative throughout the application.
