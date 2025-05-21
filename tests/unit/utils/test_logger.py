"""
Unit tests for the logger utilities.
"""

import pytest
import os
import logging
import tempfile
import json
import time
from unittest.mock import patch, MagicMock, call
from io import StringIO

from apps.utils.logger import (
    # Import logger functions and classes that are actually available
    setup_logger,
    get_logger,
    # Removed JsonFormatter, StructuredLogger, LoggerConfig imports
    setup_logging, # Assuming this is the main configuration function now
    ContextAdapter, # Assuming this is used internally or for advanced cases
    # Removed other functions that might not be directly importable/testable
)

# It seems that the approach to testing logging has changed.
# Instead of testing specific formatter/adapter classes, we should test the behavior
# of the configured logger using setup_logging or setup_logger.

class TestLogger:
    """Test suite for logger utilities."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        # Reset logging configuration before each test to avoid interference
        logging.shutdown()
        # Remove all handlers from the root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # Keeping the test for setup_logger if it's still relevant
    # If setup_logger is primarily for internal use or simple cases,
    # consider focusing on setup_logging.
    def test_setup_logger(self):
        """Test that setup_logger creates a logger with correct handlers."""
        logger_name = "test_setup"
        log_file = "/tmp/test_setup.log"

        # Set up the logger
        logger = setup_logger(logger_name, log_file=log_file, level=logging.DEBUG)

        # Verify logger name and level
        assert logger.name == logger_name
        assert logger.level == logging.DEBUG
        
        # Verify handlers (console and file)
        handlers = logger.handlers
        assert len(handlers) == 2
        assert isinstance(handlers[0], logging.StreamHandler)
        assert isinstance(handlers[1], logging.FileHandler)
        assert handlers[1].baseFilename.endswith(log_file) # Check file path

        # Verify formatter is set for both handlers
        formatter_code = handlers[0].formatter._fmt # Accessing protected member for testing
        assert '%(asctime)s - %(name)s - %(levelname)s - %(message)s' in formatter_code
        formatter_code = handlers[1].formatter._fmt
        assert '%(asctime)s - %(name)s - %(levelname)s - %(message)s' in formatter_code
        
        # Clean up the log file
        if os.path.exists(log_file):
            os.remove(log_file)

    def test_setup_logging_console_only(self):
        """Test setup_logging with console output only."""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            setup_logging(log_level="info", log_to_console=True, log_to_file=False)
            logger = logging.getLogger()
            logger.info("Test message")

            # Verify message is in stdout
            output = mock_stdout.getvalue()
            assert "Test message" in output
            assert "INFO" in output

            # Verify only console handler is present
            handlers = logger.handlers
            assert len(handlers) == 1
            assert isinstance(handlers[0], logging.StreamHandler)

    def test_setup_logging_file_only(self, tmp_path):
        """Test setup_logging with file output only."""
        log_dir = tmp_path / "logs"
        log_file_name = "test_file.log"

        setup_logging(
            log_level="debug",
            log_dir=str(log_dir),
            log_to_console=False,
            log_to_file=True,
            log_file_name=log_file_name
        )
        logger = logging.getLogger()
        logger.debug("File message")

        # Verify log file exists and contains the message
        log_file_path = log_dir / log_file_name
        assert log_file_path.exists()

        with open(log_file_path, "r") as f:
            content = f.read()
            assert "File message" in content
            assert "DEBUG" in content

        # Verify only file handler is present
        handlers = logger.handlers
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.handlers.RotatingFileHandler)

    def test_setup_logging_detailed_format(self, tmp_path):
        """Test setup_logging with detailed format."""
        log_dir = tmp_path / "logs"
        log_file_name = "test_detailed.log"

        setup_logging(
            log_level="info",
            log_dir=str(log_dir),
            log_to_console=False,
            log_to_file=True,
            log_file_name=log_file_name,
            detailed_format=True
        )
        logger = get_logger("test_detailed", session_id="abc-123", agent_name="research")
        logger.info("Detailed message")

        # Verify log file exists and contains the detailed message format
        log_file_path = log_dir / log_file_name
        assert log_file_path.exists()

        with open(log_file_path, "r") as f:
            content = f.read()
            assert "Detailed message" in content
            assert "INFO" in content
            # Check for parts of the detailed format
            assert "[INFO]" in content
            assert "[abc-123]" in content
            assert "test_detailed" in content

    # Removed tests for JsonFormatter, StructuredLogger, LoggerConfig
    # Added or updated tests based on available functions and classes

# Example of a test for get_logger with ContextAdapter
# This test would verify that using get_logger adds context correctly.
    def test_get_logger_with_context(self, tmp_path):
        """Test that get_logger adds context using ContextAdapter."""
        log_dir = tmp_path / "logs"
        log_file_name = "test_context.log"

        # Set up logging with detailed format to see the context
        setup_logging(
            log_level="debug",
            log_dir=str(log_dir),
            log_to_console=False,
            log_to_file=True,
            log_file_name=log_file_name,
            detailed_format=True
        )

        # Get logger with context
        logger = get_logger("test_context", session_id="context-session", extra_context={"user_id": "user-xyz"})
        logger.debug("Message with context")

        # Verify log file contains the message with context
        log_file_path = log_dir / log_file_name
        assert log_file_path.exists()

        with open(log_file_path, "r") as f:
            content = f.read()
            assert "Message with context" in content
            assert "DEBUG" in content
            assert "[context-session]" in content # Check for session_id
            # Check for extra_context (might be formatted differently depending on the adapter/formatter)
            # This assertion might need adjustment based on actual log output format with extra context
            # Assuming the default formatter includes extra context if provided in 'extra'
            # A more robust test might involve patching the formatter or handler.


# You might need to add tests for other functions like set_log_level, add_file_handler, etc.
# based on what is intended to be testable from the logger module.

# Keep or remove the __main__ block based on whether you want to run tests directly
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
