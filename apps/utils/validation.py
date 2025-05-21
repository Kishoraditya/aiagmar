"""
Validation Module

This module provides utilities for validating data throughout the application.
It includes functions for validating common data types, input parameters,
and specialized validation for specific application domains.
"""

import re
import json
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, Type, Union, cast

from apps.utils.exceptions import ValidationError
from apps.utils.constants import (
    FileExtension, ImageAspectRatio, ImageStyle, ResearchDepth, 
    SearchEngine, SourceReliability, TaskPriority, TaskStatus
)
"""Validation utilities for the application."""

from typing import Any, Dict, List, Optional, Union, Callable
from apps.utils.exceptions import ValidationError

def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
    """Validate that all required fields are present in the data."""
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")

def validate_field_type(data: Dict[str, Any], field: str, expected_type: type) -> None:
    """Validate that a field has the expected type."""
    if field in data and not isinstance(data[field], expected_type):
        raise ValidationError(f"Field '{field}' must be of type {expected_type.__name__}")

def validate_string_length(value: str, min_length: int = 0, max_length: Optional[int] = None) -> None:
    """Validate that a string has the expected length."""
    if len(value) < min_length:
        raise ValidationError(f"String must be at least {min_length} characters long")
    
    if max_length is not None and len(value) > max_length:
        raise ValidationError(f"String must be at most {max_length} characters long")

def validate_numeric_range(value: Union[int, float], min_value: Optional[Union[int, float]] = None, 
                          max_value: Optional[Union[int, float]] = None) -> None:
    """Validate that a number is within the expected range."""
    if min_value is not None and value < min_value:
        raise ValidationError(f"Value must be at least {min_value}")
    
    if max_value is not None and value > max_value:
        raise ValidationError(f"Value must be at most {max_value}")

def validate_with_custom_function(value: Any, validation_func: Callable[[Any], bool], 
                                 error_message: str = "Validation failed") -> None:
    """Validate a value using a custom validation function."""
    if not validation_func(value):
        raise ValidationError(error_message)

def validate_email(email: str) -> None:
    """Validate that a string is a valid email address."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValidationError("Invalid email address")

def validate_url(url: str) -> None:
    """Validate that a string is a valid URL."""
    import re
    pattern = r'^(http|https)://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'
    if not re.match(pattern, url):
        raise ValidationError("Invalid URL")

# -----------------------------------------------------------------------------
# Basic Validation Functions
# -----------------------------------------------------------------------------

def validate_required(value: Any, field_name: str) -> None:
    """
    Validate that a value is not None or empty.
    
    Args:
        value: Value to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        raise ValidationError(f"{field_name} is required", {"field": field_name})
    
    if isinstance(value, (str, list, dict, set, tuple)) and len(value) == 0:
        raise ValidationError(f"{field_name} cannot be empty", {"field": field_name})


def validate_type(value: Any, expected_type: Type, field_name: str) -> None:
    """
    Validate that a value is of the expected type.
    
    Args:
        value: Value to validate
        expected_type: Expected type
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    if value is not None and not isinstance(value, expected_type):
        raise ValidationError(
            f"{field_name} must be of type {expected_type.__name__}",
            {"field": field_name, "expected_type": expected_type.__name__, "actual_type": type(value).__name__}
        )


def validate_enum(value: Any, enum_class: Type, field_name: str) -> None:
    """
    Validate that a value is a valid enum value.
    
    Args:
        value: Value to validate
        enum_class: Enum class
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    if value is not None:
        try:
            # Try to convert string to enum value
            if isinstance(value, str):
                enum_class(value)
            # Check if value is a valid enum value
            elif value not in [e.value for e in enum_class]:
                raise ValueError()
        except ValueError:
            valid_values = [e.value for e in enum_class]
            raise ValidationError(
                f"{field_name} must be one of: {', '.join(valid_values)}",
                {"field": field_name, "valid_values": valid_values}
            )


def validate_range(value: Union[int, float], min_value: Optional[Union[int, float]] = None, 
                  max_value: Optional[Union[int, float]] = None, field_name: str = "value") -> None:
    """
    Validate that a numeric value is within a specified range.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return
    
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"{field_name} must be a number",
            {"field": field_name}
        )
    
    if min_value is not None and value < min_value:
        raise ValidationError(
            f"{field_name} must be at least {min_value}",
            {"field": field_name, "min_value": min_value}
        )
    
    if max_value is not None and value > max_value:
        raise ValidationError(
            f"{field_name} must be at most {max_value}",
            {"field": field_name, "max_value": max_value}
        )


def validate_length(value: Union[str, List, Dict, Set, Tuple], min_length: Optional[int] = None,
                   max_length: Optional[int] = None, field_name: str = "value") -> None:
    """
    Validate that a value's length is within a specified range.
    
    Args:
        value: Value to validate
        min_length: Minimum allowed length (inclusive)
        max_length: Maximum allowed length (inclusive)
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return
    
    if not hasattr(value, "__len__"):
        raise ValidationError(
            f"{field_name} must have a length",
            {"field": field_name}
        )
    
    length = len(value)
    
    if min_length is not None and length < min_length:
        raise ValidationError(
            f"{field_name} must be at least {min_length} characters long",
            {"field": field_name, "min_length": min_length}
        )
    
    if max_length is not None and length > max_length:
        raise ValidationError(
            f"{field_name} must be at most {max_length} characters long",
            {"field": field_name, "max_length": max_length}
        )


def validate_regex(value: str, pattern: Union[str, Pattern], field_name: str) -> None:
    """
    Validate that a string matches a regular expression pattern.
    
    Args:
        value: Value to validate
        pattern: Regular expression pattern
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return
    
    if not isinstance(value, str):
        raise ValidationError(
            f"{field_name} must be a string",
            {"field": field_name}
        )
    
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    
    if not pattern.match(value):
        raise ValidationError(
            f"{field_name} does not match the required pattern",
            {"field": field_name, "pattern": pattern.pattern}
        )


def validate_email(email: str, field_name: str = "email") -> None:
    """
    Validate that a string is a valid email address.
    
    Args:
        email: Email address to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    if email is None:
        return
    
    # Simple email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    try:
        validate_regex(email, pattern, field_name)
    except ValidationError:
        raise ValidationError(
            f"{field_name} is not a valid email address",
            {"field": field_name}
        )


def validate_url(url: str, field_name: str = "url") -> None:
    """
    Validate that a string is a valid URL.
    
    Args:
        url: URL to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    if url is None:
        return
    
    # URL regex pattern
    pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
    
    try:
        validate_regex(url, pattern, field_name)
    except ValidationError:
        raise ValidationError(
            f"{field_name} is not a valid URL",
            {"field": field_name}
        )


def validate_uuid(value: str, field_name: str = "id") -> None:
    """
    Validate that a string is a valid UUID.
    
    Args:
        value: UUID to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return
    
    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        raise ValidationError(
            f"{field_name} is not a valid UUID",
            {"field": field_name}
        )


def validate_date(value: str, format: str = "%Y-%m-%d", field_name: str = "date") -> None:
    """
    Validate that a string is a valid date in the specified format.
    
    Args:
        value: Date string to validate
        format: Date format string
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return
    
    try:
        datetime.strptime(value, format)
    except (ValueError, TypeError):
        raise ValidationError(
            f"{field_name} is not a valid date in format {format}",
            {"field": field_name, "format": format}
        )


def validate_json(value: str, field_name: str = "json") -> None:
    """
    Validate that a string is valid JSON.
    
    Args:
        value: JSON string to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return
    
    try:
        json.loads(value)
    except (json.JSONDecodeError, TypeError):
        raise ValidationError(
            f"{field_name} is not valid JSON",
            {"field": field_name}
        )


def validate_search_query(query: str, field_name: str = "query") -> None:
    """
    Validate a search query.
    
    Args:
        query: Search query to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    validate_required(query, field_name)
    validate_type(query, str, field_name)
    validate_length(query, min_length=1, max_length=400, field_name=field_name)
    
    # Count words
    word_count = len(query.split())
    if word_count > 50:
        raise ValidationError(
            f"{field_name} must contain at most 50 words",
            {"field": field_name, "word_count": word_count}
        )


def validate_search_engine(engine: str, field_name: str = "search_engine") -> None:
    """
    Validate a search engine.
    
    Args:
        engine: Search engine to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    validate_enum(engine, SearchEngine, field_name)


def validate_search_count(count: int, field_name: str = "count") -> None:
    """
    Validate a search count.
    
    Args:
        count: Search count to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    validate_type(count, int, field_name)
    validate_range(count, min_value=1, max_value=20, field_name=field_name)


def validate_search_offset(offset: int, field_name: str = "offset") -> None:
    """
    Validate a search offset.
    
    Args:
        offset: Search offset to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    validate_type(offset, int, field_name)
    validate_range(offset, min_value=0, max_value=9, field_name=field_name)


def validate_image_prompt(prompt: str, field_name: str = "prompt") -> None:
    """
    Validate an image generation prompt.
    
    Args:
        prompt: Image prompt to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    validate_required(prompt, field_name)
    validate_type(prompt, str, field_name)
    validate_length(prompt, min_length=1, max_length=1000, field_name=field_name)


def validate_image_style(style: str, field_name: str = "style") -> None:
    """
    Validate an image style.
    
    Args:
        style: Image style to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    validate_enum(style, ImageStyle, field_name)


def validate_image_aspect_ratio(aspect_ratio: str, field_name: str = "aspect_ratio") -> None:
    """
    Validate an image aspect ratio.
    
    Args:
        aspect_ratio: Image aspect ratio to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    validate_enum(aspect_ratio, ImageAspectRatio, field_name)


def validate_image_count(count: int, field_name: str = "count") -> None:
    """
    Validate an image count.
    
    Args:
        count: Image count to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    validate_type(count, int, field_name)
    validate_range(count, min_value=1, max_value=4, field_name=field_name)


def validate_file_path(path: str, field_name: str = "path") -> None:
    """
    Validate a file path.
    
    Args:
        path: File path to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    validate_required(path, field_name)
    validate_type(path, str, field_name)
    
    # Check for path traversal attempts
    if ".." in path or path.startswith("/") or path.startswith("\\"):
        raise ValidationError(
            f"{field_name} contains invalid path components",
            {"field": field_name}
        )


def validate_file_extension(extension: str, field_name: str = "extension") -> None:
    """
    Validate a file extension.
    
    Args:
        extension: File extension to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    validate_enum(extension, FileExtension, field_name)


def validate_research_depth(depth: str, field_name: str = "depth") -> None:
    """
    Validate a research depth.
    
    Args:
        depth: Research depth to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    validate_enum(depth, ResearchDepth, field_name)


def validate_source_reliability(reliability: str, field_name: str = "reliability") -> None:
    """
    Validate a source reliability.
    
    Args:
        reliability: Source reliability to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    validate_enum(reliability, SourceReliability, field_name)


def validate_task_status(status: str, field_name: str = "status") -> None:
    """
    Validate a task status.
    
    Args:
        status: Task status to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    validate_enum(status, TaskStatus, field_name)


def validate_task_priority(priority: str, field_name: str = "priority") -> None:
    """
    Validate a task priority.
    
    Args:
        priority: Task priority to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    validate_enum(priority, TaskPriority, field_name)


def validate_memory_key(key: str, field_name: str = "key") -> None:
    """
    Validate a memory key.
    
    Args:
        key: Memory key to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    validate_required(key, field_name)
    validate_type(key, str, field_name)
    validate_length(key, min_length=1, max_length=100, field_name=field_name)
    
    # Check for valid characters
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', key):
        raise ValidationError(
            f"{field_name} contains invalid characters (only alphanumeric, underscore, hyphen, and dot are allowed)",
            {"field": field_name}
        )


def validate_memory_namespace(namespace: str, field_name: str = "namespace") -> None:
    """
    Validate a memory namespace.
    
    Args:
        namespace: Memory namespace to validate
        field_name: Name of the field for error message
        
    Raises:
        ValidationError: If validation fails
    """
    validate_required(namespace, field_name)
    validate_type(namespace, str, field_name)
    validate_length(namespace, min_length=1, max_length=50, field_name=field_name)
    
    # Check for valid characters
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', namespace):
        raise ValidationError(
            f"{field_name} contains invalid characters (only alphanumeric, underscore, hyphen, and dot are allowed)",
            {"field": field_name}
        )


def validate_dict_schema(field_name: str, data: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """
    Validate that a dictionary conforms to a specified schema.

    Args:
        field_name: The name of the dictionary field being validated.
        data: The dictionary data to validate.
        schema: The schema to validate against.

    Raises:
        ValidationError: If the dictionary does not conform to the schema.
    """
    if not isinstance(data, dict):
        raise ValidationError(f"Field '{field_name}' must be a dictionary", {"field": field_name, "expected_type": "dict", "actual_type": type(data).__name__})

    for key, rules in schema.items():
        is_required = rules.get("required", False)
        field_value = data.get(key)
        nested_field_name = f"{field_name}.{key}"

        if is_required and field_value is None:
            raise ValidationError(f"Field '{nested_field_name}' is required", {"field": nested_field_name})

        if field_value is not None:
            expected_type = rules.get("type")
            if expected_type and not isinstance(field_value, expected_type):
                raise ValidationError(f"Field '{nested_field_name}' must be of type {expected_type.__name__}", {"field": nested_field_name, "expected_type": expected_type.__name__, "actual_type": type(field_value).__name__})

            # Handle nested dictionaries
            nested_schema = rules.get("nested")
            if nested_schema and isinstance(field_value, dict):
                validate_dict_schema(nested_field_name, field_value, nested_schema)

            # Handle list items
            item_type = rules.get("item_type")
            if item_type and isinstance(field_value, list):
                for i, item in enumerate(field_value):
                    if not isinstance(item, item_type):
                        raise ValidationError(f"Field '{nested_field_name}[{i}]' must be of type {item_type.__name__}", {"field": f"{nested_field_name}[{i}]", "expected_type": item_type.__name__, "actual_type": type(item).__name__})

            # Handle custom validator function
            validator_func = rules.get("validator")
            if validator_func:
                try:
                    validator_func(field_value)
                except ValidationError as e:
                    raise ValidationError(f"Field '{nested_field_name}' validation failed: {e}", {"field": nested_field_name, "validation_error": str(e)})

            # Handle range validation
            min_value = rules.get("min_value")
            max_value = rules.get("max_value")
            if isinstance(field_value, (int, float)):
                if min_value is not None and field_value < min_value:
                    raise ValidationError(f"Field '{nested_field_name}' must be at least {min_value}", {"field": nested_field_name, "min_value": min_value})
                if max_value is not None and field_value > max_value:
                    raise ValidationError(f"Field '{nested_field_name}' must be at most {max_value}", {"field": nested_field_name, "max_value": max_value})

            # Handle length validation
            min_length = rules.get("min_length")
            max_length = rules.get("max_length")
            if isinstance(field_value, (str, list, dict, set, tuple)):
                if min_length is not None and len(field_value) < min_length:
                    raise ValidationError(f"Field '{nested_field_name}' must have length at least {min_length}", {"field": nested_field_name, "min_length": min_length})
                if max_length is not None and len(field_value) > max_length:
                    raise ValidationError(f"Field '{nested_field_name}' must have length at most {max_length}", {"field": nested_field_name, "max_length": max_length})


# -----------------------------------------------------------------------------
# Validation Decorator Functions
# -----------------------------------------------------------------------------

def validate_with(validator: Callable[[Any], None]) -> Callable[[Any], Any]:
    """
    Create a validator function that applies a validation function.
    
    Args:
        validator: Validation function to apply
        
    Returns:
        Validator function
    """
    def validate(value: Any) -> Any:
        validator(value)
        return value
    
    return validate


def validate_dict_keys(required_keys: List[str], optional_keys: Optional[List[str]] = None) -> Callable[[Dict], Dict]:
    """
    Create a validator function that validates dictionary keys.
    
    Args:
        required_keys: List of required keys
        optional_keys: List of optional keys
        
    Returns:
        Validator function
    """
    def validate(data: Dict) -> Dict:
        if not isinstance(data, dict):
            raise ValidationError("Expected a dictionary")
        
        # Check for required keys
        for key in required_keys:
            if key not in data:
                raise ValidationError(f"Missing required key: {key}")
        
        # Check for unknown keys
        allowed_keys = set(required_keys)
        if optional_keys:
            allowed_keys.update(optional_keys)
        
        unknown_keys = set(data.keys()) - allowed_keys
        if unknown_keys:
            raise ValidationError(f"Unknown keys: {', '.join(unknown_keys)}")
        
        return data
    
    return validate


def validate_list_items(item_validator: Callable[[Any], None]) -> Callable[[List], List]:
    """
    Create a validator function that validates list items.
    
    Args:
        item_validator: Validation function for list items
        
    Returns:
        Validator function
    """
    def validate(data: List) -> List:
        if not isinstance(data, list):
            raise ValidationError("Expected a list")
        
        for i, item in enumerate(data):
            try:
                item_validator(item)
            except ValidationError as e:
                raise ValidationError(f"Invalid item at index {i}: {str(e)}")
        
        return data
    
    return validate


# -----------------------------------------------------------------------------
# Composite Validation Functions
# -----------------------------------------------------------------------------

def validate_search_params(query: str, engine: str = None, count: int = None, offset: int = None) -> None:
    """
    Validate search parameters.
    
    Args:
        query: Search query
        engine: Search engine
        count: Number of results
        offset: Pagination offset
        
    Raises:
        ValidationError: If validation fails
    """
    validate_search_query(query)
    
    if engine is not None:
        validate_search_engine(engine)
    
    if count is not None:
        validate_search_count(count)
    
    if offset is not None:
        validate_search_offset(offset)


def validate_image_params(prompt: str, style: str = None, aspect_ratio: str = None, count: int = None) -> None:
    """
    Validate image generation parameters.
    
    Args:
        prompt: Image prompt
        style: Image style
        aspect_ratio: Image aspect ratio
        count: Number of images
        
    Raises:
        ValidationError: If validation fails
    """
    validate_image_prompt(prompt)
    
    if style is not None:
        validate_image_style(style)
    
    if aspect_ratio is not None:
        validate_image_aspect_ratio(aspect_ratio)
    
    if count is not None:
        validate_image_count(count)


def validate_memory_params(key: str, namespace: str = None) -> None:
    """
    Validate memory parameters.
    
    Args:
        key: Memory key
        namespace: Memory namespace
        
    Raises:
        ValidationError: If validation fails
    """
    validate_memory_key(key)
    
    if namespace is not None:
        validate_memory_namespace(namespace)


# -----------------------------------------------------------------------------
# Schema Validation Functions
# -----------------------------------------------------------------------------

def validate_search_params(query: str, engine: str = None, count: int = None, offset: int = None) -> None:
    """
    Validate search parameters.
    
    Args:
        query: Search query
        engine: Search engine
        count: Number of results
        offset: Pagination offset
        
    Raises:
        ValidationError: If validation fails
    """
    validate_search_query(query)
    
    if engine is not None:
        validate_search_engine(engine)
    
    if count is not None:
        validate_search_count(count)
    
    if offset is not None:
        validate_search_offset(offset)


def validate_image_params(prompt: str, style: str = None, aspect_ratio: str = None, count: int = None) -> None:
    """
    Validate image generation parameters.
    
    Args:
        prompt: Image prompt
        style: Image style
        aspect_ratio: Image aspect ratio
        count: Number of images
        
    Raises:
        ValidationError: If validation fails
    """
    validate_image_prompt(prompt)
    
    if style is not None:
        validate_image_style(style)
    
    if aspect_ratio is not None:
        validate_image_aspect_ratio(aspect_ratio)
    
    if count is not None:
        validate_image_count(count)


def validate_memory_params(key: str, namespace: str = None) -> None:
    """
    Validate memory parameters.
    
    Args:
        key: Memory key
        namespace: Memory namespace
        
    Raises:
        ValidationError: If validation fails
    """
    validate_memory_key(key)
    
    if namespace is not None:
        validate_memory_namespace(namespace)


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any], field_name: str = "data") -> None:
    """
    Validate data against a JSON schema.

    Args:
        data: Data to validate
        schema: JSON schema
        field_name: Name of the field for error message

    Raises:
        ValidationError: If validation fails
    """
    try:
        import jsonschema
    except ImportError:
        raise ImportError("jsonschema package is required for schema validation")

    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        raise ValidationError(
            f"Field '{field_name}' schema validation failed: {e.message}",
            {"field": field_name, "path": list(e.path), "schema_path": list(e.schema_path)}
        )


def validate_model(data: Dict, model_class: Type) -> Any:
    """
    Validate data against a Pydantic model.
    
    Args:
        data: Data to validate
        model_class: Pydantic model class
        
    Returns:
        Validated model instance
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        from pydantic import ValidationError as PydanticValidationError
    except ImportError:
        raise ImportError("pydantic package is required for model validation")
    
    try:
        return model_class(**data)
    except PydanticValidationError as e:
        errors = e.errors()
        raise ValidationError(
            f"Model validation failed: {errors[0]['msg']}",
            {"errors": errors}
        )


def validate_api_key(api_key: str, field_name: str = "API Key") -> None:
    """
    Validate that a string is a non-empty API key.

    Args:
        api_key: The API key string to validate.
        field_name: The name of the field for error messages.

    Raises:
        ValidationError: If the API key is invalid.
    """
    validate_required(api_key, field_name)
    validate_type(api_key, str, field_name)
    validate_length(api_key, min_length=1, field_name=field_name)
    # Add more specific API key format validation if needed
    # e.g., validate_regex(api_key, r'^[a-f0-9]{32}$', field_name)


# Example usage
if __name__ == "__main__":
    # Example of basic validation
    try:
        validate_required("", "username")
    except ValidationError as e:
        print(f"Validation error: {e}")
    
    # Example of email validation
    try:
        validate_email("invalid-email")
    except ValidationError as e:
        print(f"Validation error: {e}")
    
    # Example of URL validation
    try:
        validate_url("example.com")  # Missing protocol
    except ValidationError as e:
        print(f"Validation error: {e}")
    
    # Example of search query validation
    try:
        validate_search_query("a" * 500)  # Too long
    except ValidationError as e:
        print(f"Validation error: {e}")
    
    # Example of composite validation
    try:
        validate_search_params(
            query="machine learning",
            engine="invalid_engine",
            count=30  # Too high
        )
    except ValidationError as e:
        print(f"Validation error: {e}")
    
    # Example of dictionary validation
    validator = validate_dict_keys(
        required_keys=["name", "email"],
        optional_keys=["phone"]
    )
    
    try:
        validator({"name": "John", "email": "john@example.com", "unknown": "value"})
    except ValidationError as e:
        print(f"Validation error: {e}")
    
    # Example of list validation
    validator = validate_list_items(lambda x: validate_email(x))
    
    try:
        validator(["john@example.com", "invalid-email"])
    except ValidationError as e:
        print(f"Validation error: {e}")
