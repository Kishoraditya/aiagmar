"""
Logger Module

This module provides a centralized logging configuration for the entire application.
It includes custom formatters, handlers, and convenience functions for logging.
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from typing import Optional, Dict, Any, Union

# Default log directory
DEFAULT_LOG_DIR = os.path.join(os.getcwd(), "logs")

# Log levels
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Detailed log format with context
DETAILED_LOG_FORMAT = "%(asctime)s - [%(levelname)s] - %(name)s - [%(session_id)s] - %(message)s"

import logging
import os
import sys
from typing import Optional

class Logger:
    """Logger class for consistent logging across the application."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Add console handler if not already added
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO):
    """Set up a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class ContextAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context information to log records.
    This allows adding session_id, agent_name, and other context to logs.
    """
    
    def process(self, msg, kwargs):
        """Process the log message and add context information."""
        # Add context from extra dict to the message
        if 'extra' not in kwargs:
            kwargs['extra'] = self.extra
        else:
            # Merge extra dictionaries
            for key, value in self.extra.items():
                if key not in kwargs['extra']:
                    kwargs['extra'][key] = value
        
        return msg, kwargs


def setup_logging(
    log_level: str = "info",
    log_dir: Optional[str] = None,
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_file_name: Optional[str] = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
    detailed_format: bool = False
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (debug, info, warning, error, critical)
        log_dir: Directory to store log files
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_file_name: Name of the log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        detailed_format: Whether to use detailed log format with context
    """
    # Convert log level string to logging level
    level = LOG_LEVELS.get(log_level.lower(), logging.INFO)
    
    # Create log directory if it doesn't exist
    if log_to_file:
        log_dir = log_dir or DEFAULT_LOG_DIR
        os.makedirs(log_dir, exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    log_format = DETAILED_LOG_FORMAT if detailed_format else DEFAULT_LOG_FORMAT
    formatter = logging.Formatter(log_format)
    
    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file:
        if not log_file_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_name = f"aiagmar_{timestamp}.log"
        
        log_file_path = os.path.join(log_dir, log_file_name)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log the configuration
    logging.info(f"Logging configured with level: {log_level}")
    if log_to_file:
        logging.info(f"Log file: {log_file_path if 'log_file_path' in locals() else 'None'}")


def get_logger(
    name: str,
    session_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    **extra_context
) -> logging.LoggerAdapter:
    """
    Get a logger with context information.
    
    Args:
        name: Logger name (usually __name__)
        session_id: Session ID for tracking requests
        agent_name: Name of the agent using the logger
        **extra_context: Additional context to include in logs
        
    Returns:
        Logger adapter with context
    """
    logger = logging.getLogger(name)
    
    # Prepare context
    context = {
        'session_id': session_id or 'no_session',
    }
    
    # Add agent name if provided
    if agent_name:
        context['agent_name'] = agent_name
    
    # Add any extra context
    context.update(extra_context)
    
    # Create and return adapter
    return ContextAdapter(logger, context)


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.
    
    Usage:
        class MyClass(LoggerMixin):
            def __init__(self):
                self.setup_logger()
                self.logger.info("Initialized")
    """
    
    def setup_logger(
        self,
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        **extra_context
    ) -> None:
        """
        Set up a logger for this class.
        
        Args:
            session_id: Session ID for tracking requests
            agent_name: Name of the agent using the logger
            **extra_context: Additional context to include in logs
        """
        # Use class name as logger name if not specified
        name = self.__class__.__module__ + '.' + self.__class__.__name__
        
        # Use class name as agent name if not specified and class is an agent
        if agent_name is None and 'Agent' in self.__class__.__name__:
            agent_name = self.__class__.__name__
        
        # Get logger with context
        self.logger = get_logger(
            name=name,
            session_id=session_id,
            agent_name=agent_name,
            **extra_context
        )


# Initialize logging with default settings
setup_logging(
    log_level=os.environ.get("LOG_LEVEL", "info"),
    log_to_console=os.environ.get("LOG_TO_CONSOLE", "true").lower() == "true",
    log_to_file=os.environ.get("LOG_TO_FILE", "true").lower() == "true",
    detailed_format=True
)


# Example usage
if __name__ == "__main__":
    # Basic usage
    logger = get_logger(__name__, session_id="test-session")
    logger.info("This is an info message")
    logger.error("This is an error message")
    
    # Usage with context
    context_logger = get_logger(
        __name__,
        session_id="context-session",
        agent_name="TestAgent",
        workflow="research",
        user_id="user123"
    )
    context_logger.info("Message with context")
    
    # Usage with mixin
    class TestClass(LoggerMixin):
        def __init__(self):
            self.setup_logger(session_id="mixin-session")
            self.logger.info("Initialized TestClass")
    
    test = TestClass()
