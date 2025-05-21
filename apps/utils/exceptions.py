"""
Exceptions Module

This module defines custom exceptions used throughout the application.
These exceptions provide more specific error information and can be
caught and handled appropriately by the application.
"""

from typing import Any, Dict, Optional, Union
from apps.utils.constants import ErrorCode, HTTPStatus

class BaseError(Exception):
    """Base class for all custom exceptions in the application."""
    
    def __init__(
        self, 
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        status_code: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            error_code: Error code
            status_code: HTTP status code
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception to a dictionary.
        
        Returns:
            Dictionary representation of the exception
        """
        return {
            "error": self.error_code.value,
            "message": self.message,
            "status_code": self.status_code.value,
            "details": self.details
        }
    
    def __str__(self) -> str:
        """
        Get string representation of the exception.
        
        Returns:
            String representation
        """
        details_str = f", details: {self.details}" if self.details else ""
        return f"{self.error_code.value}: {self.message}{details_str}"


# -----------------------------------------------------------------------------
# General Exceptions
# -----------------------------------------------------------------------------

class ValidationError(BaseError):
    """Exception raised when validation fails."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Validation error details
            status_code: HTTP status code
        """
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=status_code,
            details=details
        )


class ConfigurationError(BaseError):
    """Exception raised when there is a configuration error."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Configuration error details
            status_code: HTTP status code
        """
        super().__init__(
            message=message,
            error_code=ErrorCode.CONFIGURATION_ERROR,
            status_code=status_code,
            details=details
        )


# -----------------------------------------------------------------------------
# Authentication and Authorization Exceptions
# -----------------------------------------------------------------------------

class AuthenticationError(BaseError):
    """Exception raised when authentication fails."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None,
        status_code: HTTPStatus = HTTPStatus.UNAUTHORIZED
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Authentication error details
            status_code: HTTP status code
        """
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHENTICATION_ERROR,
            status_code=status_code,
            details=details
        )


class AuthorizationError(BaseError):
    """Exception raised when authorization fails."""
    
    def __init__(
        self,
        message: str = "Authorization failed",
        details: Optional[Dict[str, Any]] = None,
        status_code: HTTPStatus = HTTPStatus.FORBIDDEN
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Authorization error details
            status_code: HTTP status code
        """
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHORIZATION_ERROR,
            status_code=status_code,
            details=details
        )


# -----------------------------------------------------------------------------
# Resource Exceptions
# -----------------------------------------------------------------------------

class ResourceNotFoundError(BaseError):
    """Exception raised when a resource is not found."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[Union[str, int]] = None,
        status_code: HTTPStatus = HTTPStatus.NOT_FOUND
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            resource_type: Type of resource
            resource_id: ID of resource
            status_code: HTTP status code
        """
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id
        
        super().__init__(
            message=message,
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            status_code=status_code,
            details=details
        )


class ResourceAlreadyExistsError(BaseError):
    """Exception raised when a resource already exists."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[Union[str, int]] = None,
        status_code: HTTPStatus = HTTPStatus.CONFLICT
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            resource_type: Type of resource
            resource_id: ID of resource
            status_code: HTTP status code
        """
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id
        
        super().__init__(
            message=message,
            error_code=ErrorCode.RESOURCE_ALREADY_EXISTS,
            status_code=status_code,
            details=details
        )


# -----------------------------------------------------------------------------
# API Exceptions
# -----------------------------------------------------------------------------

class RateLimitExceededError(BaseError):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        status_code: HTTPStatus = HTTPStatus.TOO_MANY_REQUESTS
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            status_code: HTTP status code
        """
        details = {}
        if retry_after is not None:
            details["retry_after"] = retry_after
        
        super().__init__(
            message=message,
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            status_code=status_code,
            details=details
        )


class APIError(BaseError):
    """Exception raised when there is an API error."""
    
    def __init__(
        self,
        message: str,
        api_name: Optional[str] = None,
        status_code: HTTPStatus = HTTPStatus.BAD_GATEWAY
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            api_name: Name of the API
            status_code: HTTP status code
        """
        details = {}
        if api_name:
            details["api_name"] = api_name
        
        super().__init__(
            message=message,
            error_code=ErrorCode.API_ERROR,
            status_code=status_code,
            details=details
        )


# -----------------------------------------------------------------------------
# MCP Exceptions
# -----------------------------------------------------------------------------

class MCPError(BaseError):
    """Exception raised when there is an MCP error."""
    
    def __init__(
        self,
        message: str,
        mcp_name: Optional[str] = None,
        status_code: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            mcp_name: Name of the MCP
            status_code: HTTP status code
        """
        details = {}
        if mcp_name:
            details["mcp_name"] = mcp_name
        
        super().__init__(
            message=message,
            error_code=ErrorCode.MCP_ERROR,
            status_code=status_code,
            details=details
        )


class MCPNotAvailableError(BaseError):
    """Exception raised when an MCP is not available."""
    
    def __init__(
        self,
        message: str = "MCP not available",
        mcp_name: Optional[str] = None,
        status_code: HTTPStatus = HTTPStatus.SERVICE_UNAVAILABLE
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            mcp_name: Name of the MCP
            status_code: HTTP status code
        """
        details = {}
        if mcp_name:
            details["mcp_name"] = mcp_name
        
        super().__init__(
            message=message,
            error_code=ErrorCode.MCP_NOT_AVAILABLE,
            status_code=status_code,
            details=details
        )


# -----------------------------------------------------------------------------
# Agent Exceptions
# -----------------------------------------------------------------------------

class AgentError(BaseError):
    """Exception raised when there is an agent error."""
    
    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        status_code: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            agent_name: Name of the agent
            status_code: HTTP status code
        """
        details = {}
        if agent_name:
            details["agent_name"] = agent_name
        
        super().__init__(
            message=message,
            error_code=ErrorCode.AGENT_ERROR,
            status_code=status_code,
            details=details
        )


class AgentNotAvailableError(BaseError):
    """Exception raised when an agent is not available."""
    
    def __init__(
        self,
        message: str = "Agent not available",
        agent_name: Optional[str] = None,
        status_code: HTTPStatus = HTTPStatus.SERVICE_UNAVAILABLE
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            agent_name: Name of the agent
            status_code: HTTP status code
        """
        details = {}
        if agent_name:
            details["agent_name"] = agent_name
        
        super().__init__(
            message=message,
            error_code=ErrorCode.AGENT_NOT_AVAILABLE,
            status_code=status_code,
            details=details
        )


# -----------------------------------------------------------------------------
# Workflow Exceptions
# -----------------------------------------------------------------------------

class WorkflowError(BaseError):
    """Exception raised when there is a workflow error."""
    
    def __init__(
        self,
        message: str,
        workflow_name: Optional[str] = None,
        status_code: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            workflow_name: Name of the workflow
            status_code: HTTP status code
        """
        details = {}
        if workflow_name:
            details["workflow_name"] = workflow_name
        
        super().__init__(
            message=message,
            error_code=ErrorCode.WORKFLOW_ERROR,
            status_code=status_code,
            details=details
        )


class WorkflowNotAvailableError(BaseError):
    """Exception raised when a workflow is not available."""
    
    def __init__(
        self,
        message: str = "Workflow not available",
        workflow_name: Optional[str] = None,
        status_code: HTTPStatus = HTTPStatus.SERVICE_UNAVAILABLE
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            workflow_name: Name of the workflow
            status_code: HTTP status code
        """
        details = {}
        if workflow_name:
            details["workflow_name"] = workflow_name
        
        super().__init__(
            message=message,
            error_code=ErrorCode.WORKFLOW_NOT_AVAILABLE,
            status_code=status_code,
            details=details
        )

"""Custom exceptions for the application."""

class AgentCommunicationError(AgentError):
    """Exception raised when communication between agents fails."""
    pass

class FileOperationError(MCPError):
    """Exception raised for errors during file operations."""
    pass

class ImageGenerationError(MCPError):
    """Exception raised for errors during image generation."""
    pass

class QueryClarificationError(AgentError):
    """Exception raised when query clarification fails."""
    pass

class PlanPresentationError(AgentError):
    """Exception raised when plan presentation fails."""
    pass

class TimeoutError(BaseError):
    """Exception raised when an operation times out."""
    pass


# Example usage
if __name__ == "__main__":
    # Example of ValidationError
    try:
        raise ValidationError(
            message="Invalid input",
            details={"field": "username", "error": "Username must be at least 3 characters"}
        )
    except BaseError as e:
        print(f"Exception: {e}")
        print(f"Dictionary: {e.to_dict()}")
    
    # Example of ResourceNotFoundError
    try:
        raise ResourceNotFoundError(
            message="User not found",
            resource_type="user",
            resource_id="123"
        )
    except BaseError as e:
        print(f"\nException: {e}")
        print(f"Dictionary: {e.to_dict()}")
    
    # Example of MCPError
    try:
        raise MCPError(
            message="Failed to connect to MCP server",
            mcp_name="brave_search"
        )
    except BaseError as e:
        print(f"\nException: {e}")
        print(f"Dictionary: {e.to_dict()}")
    
    # Example of AgentError
    try:
        raise AgentError(
            message="Agent failed to process request",
            agent_name="research_agent"
        )
    except BaseError as e:
        print(f"\nException: {e}")
        print(f"Dictionary: {e.to_dict()}")
