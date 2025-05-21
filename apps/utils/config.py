"""
Configuration Module

This module provides a centralized configuration system for the application.
It handles loading configuration from environment variables, configuration files,
and provides defaults for all settings.
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set, TypeVar, cast

# Import helpers
from apps.utils.helpers import safe_json_loads, merge_dicts, safe_read_file
from apps.utils.logger import get_logger

# Set up logger
logger = get_logger(__name__)

# Type for configuration values
ConfigValue = Union[str, int, float, bool, List[Any], Dict[str, Any], None]

# Default configuration
DEFAULT_CONFIG = {
    # General settings
    "app": {
        "name": "AIAGMAR",
        "version": "0.1.0",
        "debug": False,
        "environment": "development",
        "temp_dir": "temp",
        "max_retries": 3,
        "timeout": 30,
    },
    
    # Logging settings
    "logging": {
        "level": "info",
        "to_console": True,
        "to_file": True,
        "log_dir": "logs",
        "detailed_format": True,
    },
    
    # API settings
    "api": {
        "host": "127.0.0.1",
        "port": 8000,
        "workers": 4,
        "cors_origins": ["*"],
        "rate_limit": 100,
        "rate_limit_period": 60,
    },
    
    # MCP settings
    "mcps": {
        "brave_search": {
            "enabled": True,
            "use_docker": False,
        },
        "everart": {
            "enabled": True,
            "use_docker": False,
        },
        "fetch": {
            "enabled": True,
            "use_docker": False,
        },
        "filesystem": {
            "enabled": True,
            "use_docker": False,
        },
        "memory": {
            "enabled": True,
            "use_docker": False,
            "storage_path": "memory_storage",
        },
    },
    
    # Agent settings
    "agents": {
        "manager": {
            "enabled": True,
            "max_tasks": 10,
        },
        "pre_response": {
            "enabled": True,
        },
        "research": {
            "enabled": True,
            "max_sources": 5,
            "max_content_length": 100000,
        },
        "summary": {
            "enabled": True,
            "max_summary_length": 5000,
        },
        "verification": {
            "enabled": True,
            "verification_threshold": 0.7,
        },
        "image_generation": {
            "enabled": True,
            "max_images": 3,
        },
        "file_manager": {
            "enabled": True,
            "max_file_size": 10485760,  # 10MB
        },
    },
    
    # Workflow settings
    "workflows": {
        "research": {
            "enabled": True,
            "default_max_sources": 5,
            "default_include_images": True,
            "default_verify_facts": True,
            "default_research_depth": "standard",
            "default_output_format": "markdown",
        },
    },
    
    # Security settings
    "security": {
        "api_key_required": False,
        "allowed_ips": [],
        "rate_limiting": True,
    },
}

# Environment variable prefix
ENV_PREFIX = "AIAGMAR_"

# Config file paths to check (in order)
CONFIG_FILE_PATHS = [
    "./config.yaml",
    "./config.yml",
    "./config.json",
    "~/.aiagmar/config.yaml",
    "~/.aiagmar/config.json",
]


class Config:
    """
    Configuration manager for the application.
    
    This class handles loading configuration from environment variables,
    configuration files, and provides defaults for all settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to a configuration file
        """
        self._config: Dict[str, Any] = {}
        self._loaded_files: List[str] = []
        
        # Load default configuration
        self._config = DEFAULT_CONFIG.copy()
        
        # Load configuration from files
        self._load_from_files(config_path)
        
        # Load configuration from environment variables
        self._load_from_env()
        
        # Validate configuration
        self._validate()
        
        # Log configuration source
        self._log_config_source()
    
    def _load_from_files(self, config_path: Optional[str] = None) -> None:
        """
        Load configuration from files.
        
        Args:
            config_path: Optional path to a configuration file
        """
        # Check specified config file first
        if config_path:
            if self._load_file(config_path):
                return
        
        # Check default config file paths
        for path in CONFIG_FILE_PATHS:
            expanded_path = os.path.expanduser(path)
            if self._load_file(expanded_path):
                return
    
    def _load_file(self, file_path: str) -> bool:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            True if file was loaded successfully, False otherwise
        """
        if not os.path.isfile(file_path):
            return False
        
        try:
            content = safe_read_file(file_path)
            if not content:
                return False
            
            config_data: Dict[str, Any] = {}
            
            # Parse based on file extension
            if file_path.endswith(('.yaml', '.yml')):
                config_data = yaml.safe_load(content)
            elif file_path.endswith('.json'):
                config_data = json.loads(content)
            else:
                logger.warning(f"Unsupported config file format: {file_path}")
                return False
            
            if not isinstance(config_data, dict):
                logger.error(f"Invalid config file format: {file_path}")
                return False
            
            # Merge with existing configuration
            self._config = merge_dicts(self._config, config_data)
            self._loaded_files.append(file_path)
            logger.info(f"Loaded configuration from {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading config file {file_path}: {str(e)}")
            return False
    
    def _load_from_env(self) -> None:
        """
        Load configuration from environment variables.
        
        Environment variables should be prefixed with AIAGMAR_ and use double underscores
        to indicate nesting. For example, AIAGMAR_APP__DEBUG=true would set app.debug to True.
        """
        for key, value in os.environ.items():
            if not key.startswith(ENV_PREFIX):
                continue
            
            # Remove prefix and split by double underscore
            config_key = key[len(ENV_PREFIX):].lower()
            parts = config_key.split('__')
            
            # Skip invalid keys
            if not all(parts):
                continue
            
            # Convert value to appropriate type
            typed_value = self._convert_env_value(value)
            
            # Update configuration
            self._set_nested_value(self._config, parts, typed_value)
    
    def _convert_env_value(self, value: str) -> ConfigValue:
        """
        Convert environment variable string to appropriate type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Converted value
        """
        # Check for boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Check for null
        if value.lower() in ('null', 'none'):
            return None
        
        # Check for integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Check for float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Check for JSON
        if value.startswith('{') or value.startswith('['):
            try:
                return safe_json_loads(value, default=value)
            except Exception:
                pass
        
        # Default to string
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], keys: List[str], value: ConfigValue) -> None:
        """
        Set a nested value in the configuration dictionary.
        
        Args:
            config: Configuration dictionary
            keys: List of keys representing the path to the value
            value: Value to set
        """
        if len(keys) == 1:
            config[keys[0]] = value
            return
        
        key = keys[0]
        if key not in config or not isinstance(config[key], dict):
            config[key] = {}
        
        self._set_nested_value(config[key], keys[1:], value)
    
    def _validate(self) -> None:
        """
        Validate the configuration.
        
        This method checks for required values and ensures types are correct.
        """
        # Ensure log level is valid
        log_level = self.get('logging.level', 'info').lower()
        if log_level not in ('debug', 'info', 'warning', 'error', 'critical'):
            logger.warning(f"Invalid log level: {log_level}, using 'info'")
            self._set_nested_value(self._config, ['logging', 'level'], 'info')
        
        # Ensure environment is valid
        env = self.get('app.environment', 'development').lower()
        if env not in ('development', 'testing', 'production'):
            logger.warning(f"Invalid environment: {env}, using 'development'")
            self._set_nested_value(self._config, ['app', 'environment'], 'development')
        
        # Ensure API port is valid
        port = self.get('api.port', 8000)
        if not isinstance(port, int) or port < 1 or port > 65535:
            logger.warning(f"Invalid API port: {port}, using 8000")
            self._set_nested_value(self._config, ['api', 'port'], 8000)
    
    def _log_config_source(self) -> None:
        """Log the sources of configuration."""
        sources = []
        
        if self._loaded_files:
            sources.append(f"files: {', '.join(self._loaded_files)}")
        
        env_vars = [key for key in os.environ if key.startswith(ENV_PREFIX)]
        if env_vars:
            sources.append(f"environment variables: {len(env_vars)} variables")
        
        if sources:
            logger.info(f"Configuration loaded from {' and '.join(sources)}")
        else:
            logger.info("Using default configuration")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (dot-separated for nested values)
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        parts = key.split('.')
        value = self._config
        
        for part in parts:
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (dot-separated for nested values)
            value: Value to set
        """
        parts = key.split('.')
        self._set_nested_value(self._config, parts, value)
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get the entire configuration as a dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self._config.copy()
    
    def save(self, file_path: str) -> bool:
        """
        Save the current configuration to a file.
        
        Args:
            file_path: Path to save the configuration to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Write configuration to file
            if file_path.endswith(('.yaml', '.yml')):
                with open(file_path, 'w') as f:
                    yaml.dump(self._config, f, default_flow_style=False)
            elif file_path.endswith('.json'):
                with open(file_path, 'w') as f:
                    json.dump(self._config, f, indent=2)
            else:
                logger.error(f"Unsupported config file format: {file_path}")
                return False
            
            logger.info(f"Configuration saved to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {str(e)}")
            return False
    
    def reload(self) -> None:
        """Reload the configuration from files and environment variables."""
        # Reset to default configuration
        self._config = DEFAULT_CONFIG.copy()
        self._loaded_files = []
        
        # Reload from files and environment variables
        self._load_from_files()
        self._load_from_env()
        
        # Validate configuration
        self._validate()
        
        logger.info("Configuration reloaded")
    
    def __getitem__(self, key: str) -> Any:
        """
        Get a configuration value using dictionary syntax.
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value
            
        Raises:
            KeyError: If key is not found
        """
        value = self.get(key, None)
        if value is None:
            raise KeyError(f"Configuration key not found: {key}")
        return value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dictionary syntax.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """
        Check if a configuration key exists.
        
        Args:
            key: Configuration key
            
        Returns:
            True if key exists, False otherwise
        """
        return self.get(key, None) is not None


# Create a global configuration instance
config = Config()


# Helper functions to access the global configuration
def get_config(key: str, default: Any = None) -> Any:
    """
    Get a configuration value from the global configuration.
    
    Args:
        key: Configuration key
        default: Default value if key is not found
        
    Returns:
        Configuration value or default
    """
    return config.get(key, default)


def set_config(key: str, value: Any) -> None:
    """
    Set a configuration value in the global configuration.
    
    Args:
        key: Configuration key
        value: Value to set
    """
    config.set(key, value)


def reload_config() -> None:
    """Reload the global configuration."""
    config.reload()


def save_config(file_path: str) -> bool:
    """
    Save the global configuration to a file.
    
    Args:
        file_path: Path to save the configuration to
        
    Returns:
        True if successful, False otherwise
    """
    return config.save(file_path)


def get_all_config() -> Dict[str, Any]:
    """
    Get the entire global configuration as a dictionary.
    
    Returns:
        Configuration dictionary
    """
    return config.as_dict()


def is_development() -> bool:
    """
    Check if the application is running in development mode.
    
    Returns:
        True if in development mode, False otherwise
    """
    return config.get('app.environment', 'development').lower() == 'development'


def is_production() -> bool:
    """
    Check if the application is running in production mode.
    
    Returns:
        True if in production mode, False otherwise
    """
    return config.get('app.environment', 'development').lower() == 'production'


def is_testing() -> bool:
    """
    Check if the application is running in testing mode.
    
    Returns:
        True if in testing mode, False otherwise
    """
    return config.get('app.environment', 'development').lower() == 'testing'


def is_debug_enabled() -> bool:
    """
    Check if debug mode is enabled.
    
    Returns:
        True if debug mode is enabled, False otherwise
    """
    return config.get('app.debug', False)


def get_temp_dir() -> str:
    """
    Get the temporary directory path.
    
    Returns:
        Temporary directory path
    """
    temp_dir = config.get('app.temp_dir', 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def get_api_keys() -> Dict[str, str]:
    """
    Get all API keys from the configuration.
    
    Returns:
        Dictionary of API keys
    """
    api_keys = {}
    
    # Check environment variables first
    for key, value in os.environ.items():
        if key.endswith('_API_KEY'):
            service = key.replace('_API_KEY', '').lower()
            api_keys[service] = value
    
    # Check configuration
    config_api_keys = config.get('api_keys', {})
    if isinstance(config_api_keys, dict):
        api_keys.update(config_api_keys)
    
    return api_keys


def get_api_key(service: str) -> Optional[str]:
    """
    Get an API key for a specific service.
    
    Args:
        service: Service name
        
    Returns:
        API key or None if not found
    """
    # Check environment variables first
    env_key = f"{service.upper()}_API_KEY"
    if env_key in os.environ:
        return os.environ[env_key]
    
    # Check configuration
    api_keys = config.get('api_keys', {})
    if isinstance(api_keys, dict) and service.lower() in api_keys:
        return api_keys[service.lower()]
    
    return None


def is_mcp_enabled(mcp_name: str) -> bool:
    """
    Check if an MCP is enabled.
    
    Args:
        mcp_name: MCP name
        
    Returns:
        True if MCP is enabled, False otherwise
    """
    return config.get(f'mcps.{mcp_name}.enabled', True)


def should_use_docker(mcp_name: str) -> bool:
    """
    Check if Docker should be used for an MCP.
    
    Args:
        mcp_name: MCP name
        
    Returns:
        True if Docker should be used, False otherwise
    """
    return config.get(f'mcps.{mcp_name}.use_docker', False)


def is_agent_enabled(agent_name: str) -> bool:
    """
    Check if an agent is enabled.
    
    Args:
        agent_name: Agent name
        
    Returns:
        True if agent is enabled, False otherwise
    """
    return config.get(f'agents.{agent_name}.enabled', True)


def is_workflow_enabled(workflow_name: str) -> bool:
    """
    Check if a workflow is enabled.
    
    Args:
        workflow_name: Workflow name
        
    Returns:
        True if workflow is enabled, False otherwise
    """
    return config.get(f'workflows.{workflow_name}.enabled', True)


# Example usage
if __name__ == "__main__":
    # Print current configuration
    print("Current configuration:")
    import pprint
    pprint.pprint(get_all_config())
    
    # Check if we're in development mode
    print(f"Development mode: {is_development()}")
    
    # Get API key
    brave_api_key = get_api_key('brave')
    print(f"Brave API key: {brave_api_key if brave_api_key else 'Not found'}")
    
    # Check if MCP is enabled
    print(f"Brave Search MCP enabled: {is_mcp_enabled('brave_search')}")
    
    # Check if agent is enabled
    print(f"Research agent enabled: {is_agent_enabled('research')}")
    
    # Save configuration to file
    save_config('config_example.yaml')
