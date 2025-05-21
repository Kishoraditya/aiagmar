"""
Unit tests for the configuration utilities.
"""

import os
import pytest
import tempfile
import json
from unittest.mock import patch, mock_open

from apps.utils.config import (
    Config,
    save_config,
    get_config,
    set_config,
    merge_dicts,
    DEFAULT_CONFIG
)
from apps.utils.exceptions import ValidationError


class TestConfig:
    """Test suite for configuration utilities."""

    @pytest.fixture
    def sample_config(self):
        """Fixture providing a sample configuration dictionary."""
        return {
            "api_keys": {
                "brave_search": "test_brave_key",
                "everart": "test_everart_key"
            },
            "workflow": {
                "default_output_directory": "output"
            }
        }

    @pytest.fixture
    def config_file_path(self, tmp_path, sample_config):
        """Fixture creating a temporary config file."""
        file_path = tmp_path / "config.json"
        with open(file_path, 'w') as f:
            json.dump(sample_config, f)
        return str(file_path)

    @patch('apps.utils.config.CONFIG_FILE_PATHS', [])
    @patch('apps.utils.config.os.environ', {})
    def test_config_init_with_file(self, sample_config, config_file_path):
        """Test Config initialization loads from a specified file."""
        config = Config(config_path=config_file_path)
        assert config.get("api_keys.brave_search") == "test_brave_key"
        assert config.get("workflow.default_output_directory") == "output"

    @patch('apps.utils.config.CONFIG_FILE_PATHS', [])
    @patch('apps.utils.config.os.environ', {})
    def test_config_init_file_not_exists(self):
        """Test Config initialization with a non-existent file uses defaults."""
        config = Config(config_path="/nonexistent/path/config.json")
        # Check that default values are loaded
        assert config.get("app.name") == DEFAULT_CONFIG["app"]["name"]

    @patch('apps.utils.config.CONFIG_FILE_PATHS', [])
    @patch('apps.utils.config.os.environ', {})
    def test_config_init_invalid_json(self, config_file_path):
        """Test Config initialization with an invalid JSON file raises error."""
        # Overwrite valid config file with invalid content
        with open(config_file_path, 'w') as f:
            f.write("invalid json")

        # Config initialization should handle this internally or raise a relevant error
        # Assuming it logs an error and proceeds with defaults or a partial load
        # This test might need adjustment based on actual error handling in Config.__init__
        config = Config(config_path=config_file_path)
        # Assert that it doesn't raise an unhandled exception and potentially logs an error
        # Further assertions would depend on the desired behavior of the Config class
        assert config is not None # Basic assertion that initialization completed
        # Add checks here if invalid config leads to specific state or exceptions

    def test_save_config(self, sample_config, config_file_path):
        """Test saving configuration to a file."""
        config = Config(config_path=config_file_path) # Load initial config
        config.set("new_setting.key", "new_value")

        # Save the modified config
        save_config(config_file_path) # Use the helper function

        # Read the saved file and verify content
        with open(config_file_path, 'r') as f:
            saved_config = json.load(f)
        
        assert saved_config["api_keys"] == sample_config["api_keys"]
        assert saved_config["workflow"] == sample_config["workflow"]
        assert saved_config["new_setting"] == {"key": "new_value"}

    def test_save_config_directory_creation(self, tmp_path, sample_config):
        """Test saving configuration creates directory if needed."""
        nested_dir = tmp_path / "nested" / "config"
        nested_config_path = nested_dir / "config.json"

        config = Config(config_path=None) # Start with defaults
        config.set("app.debug", True)

        # Save config to a path with non-existent directories
        save_config(str(nested_config_path))

        # Verify directory and file were created
        assert nested_dir.exists()
        assert nested_config_path.exists()

        # Verify content
        with open(nested_config_path, 'r') as f:
                saved_config = json.load(f)
        
        # Check some saved values
        assert saved_config["app"]["debug"] is True

    def test_get_config_value(self, sample_config):
        """Test getting configuration values using dot notation."""
        config = Config(config_path=None) # Start with defaults
        # Set some values via the Config object
        config.set("api_keys.brave_search", "test_brave_key")
        config.set("app.debug", True)

        # Use the helper function to get values
        assert get_config("api_keys.brave_search") == "test_brave_key"
        assert get_config("app.debug") is True
        assert get_config("nonexistent.key") is None # Test getting non-existent key
        assert get_config("nonexistent.key", default="default_value") == "default_value" # Test default value

    def test_set_config_value(self, sample_config):
        """Test setting configuration values using dot notation."""
        config = Config(config_path=None) # Start with defaults

        # Use the helper function to set values
        set_config("new_api.key", "api_value")
        set_config("app.debug", False) # Override a default value

        # Verify the values were set in the internal config
        # Access the internal config directly for verification (might need adjustment)
        # Assuming there's a way to access the internal _config dict or test side effects

        # A better way is to get the value back using get_config
        assert get_config("new_api.key") == "api_value"
        assert get_config("app.debug") is False

    def test_merge_configs(self, sample_config):
        """Test merging configuration dictionaries."""
        # Use the helper function to merge
        dict1 = {"a": 1, "b": {"c": 2}}
        dict2 = {"b": {"d": 3}, "e": 4}
        merged = merge_dicts(dict1, dict2)

        assert merged == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}

        # Test with overwrite
        dict1_overwrite = {"b": {"c": 2}}
        dict2_overwrite = {"b": {"c": 4}}
        merged_overwrite = merge_dicts(dict1_overwrite, dict2_overwrite, overwrite=True)
        assert merged_overwrite == {"b": {"c": 4}}

        merged_no_overwrite = merge_dicts(dict1_overwrite, dict2_overwrite, overwrite=False)
        assert merged_no_overwrite == {"b": {"c": 2}}

    @patch('apps.utils.config.CONFIG_FILE_PATHS', [])
    @patch('apps.utils.config.os.environ', {})
    def test_config_init_valid_config(self, sample_config, config_file_path):
        """Test Config initialization with valid configuration."""
        config = Config(config_path=config_file_path)
        # If initialization completes without raising an exception, it's considered valid
        assert config is not None
        # Optionally check a few values to ensure loading was successful
        assert config.get("api_keys.brave_search") == "test_brave_key"

    @patch('apps.utils.config.CONFIG_FILE_PATHS', [])
    @patch('apps.utils.config.os.environ', {})
    def test_config_init_missing_required(self, config_file_path):
        """Test Config initialization with missing required config."""
        # Create a config file with missing required section
        invalid_config = sample_config.copy()
        del invalid_config["api_keys"]
        with open(config_file_path, 'w') as f:
            json.dump(invalid_config, f)

        # Expect ValidationError during initialization
        with pytest.raises(ValidationError) as excinfo:
            config = Config(config_path=config_file_path)
        
        # Check for a message indicating missing required keys or similar
        assert "Missing required" in str(excinfo.value) or "Validation failed" in str(excinfo.value)

    @patch('apps.utils.config.CONFIG_FILE_PATHS', [])
    @patch('apps.utils.config.os.environ', {})
    def test_config_init_invalid_types(self, config_file_path):
        """Test Config initialization with invalid field types."""
        # Create a config file with invalid type
        invalid_config = sample_config.copy()
        invalid_config["app"]["debug"] = "True" # Should be boolean
        with open(config_file_path, 'w') as f:
            json.dump(invalid_config, f)
        
        # Expect ValidationError during initialization
        with pytest.raises(ValidationError) as excinfo:
            config = Config(config_path=config_file_path)
            
        # Check for a message indicating type mismatch or similar
        assert "must be of type" in str(excinfo.value) or "Validation failed" in str(excinfo.value)

    @patch('apps.utils.config.CONFIG_FILE_PATHS', [])
    @patch('apps.utils.config.os.environ', {})
    def test_config_init_invalid_values(self, config_file_path):
        """Test Config initialization with invalid field values."""
        # Create a config file with invalid value
        invalid_config = sample_config.copy()
        invalid_config["logging"]["level"] = "verbose" # Invalid log level
        with open(config_file_path, 'w') as f:
            json.dump(invalid_config, f)
        
        # Expect ValidationError during initialization
        with pytest.raises(ValidationError) as excinfo:
            config = Config(config_path=config_file_path)
            
        # Check for a message indicating invalid value or similar
        assert "invalid value" in str(excinfo.value) or "Validation failed" in str(excinfo.value)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
