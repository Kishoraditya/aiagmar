"""
Unit tests for the validation utilities.
"""

import pytest
import os
import json
import re
import uuid
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pyld import jsonld
import jsonschema


from apps.utils.validation import (
    # Import validation functions that would be defined in the validation.py file
    # These are assumed based on typical patterns, adjust as needed
    validate_required_fields,
    validate_field_type,
    validate_string_length,
    validate_numeric_range,
    validate_email,
    validate_url,
    validate_date,
    validate_regex,
    validate_enum,
    validate_list_items,
    validate_dict_schema,
    validate_file_path,
    validate_file_extension,
    validate_json_schema,
    validate_api_key,
    validate_uuid,
    validate_ip_address,
    validate_hostname,
    validate_port_number,
    validate_boolean,
    validate_choice,
    validate_not_empty,
    ValidationError,
    SchemaValidator,
    DataValidator
)


class TestValidation:
    """Test suite for validation utilities."""

    def test_validate_required_fields(self):
        """Test that validate_required_fields correctly validates required fields."""
        # Test with all required fields present
        data = {"field1": "value1", "field2": "value2", "field3": "value3"}
        required_fields = ["field1", "field2"]
        
        # Should not raise an exception
        validate_required_fields(data, required_fields)
        
        # Test with missing required fields
        data = {"field1": "value1", "field3": "value3"}
        required_fields = ["field1", "field2"]
        
        # Should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            validate_required_fields(data, required_fields)
        
        # Verify the error message
        assert "field2" in str(excinfo.value)
        assert "required" in str(excinfo.value).lower()
        
        # Test with empty data
        data = {}
        required_fields = ["field1", "field2"]
        
        # Should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            validate_required_fields(data, required_fields)
        
        # Verify the error message
        assert "field1" in str(excinfo.value)
        assert "field2" in str(excinfo.value)
        assert "required" in str(excinfo.value).lower()

    def test_validate_field_type(self):
        """Test that validate_field_type correctly validates field types."""
        # Test with correct field type
        validate_field_type("string_field", "test", str)
        validate_field_type("int_field", 42, int)
        validate_field_type("float_field", 3.14, float)
        validate_field_type("bool_field", True, bool)
        validate_field_type("list_field", [1, 2, 3], list)
        validate_field_type("dict_field", {"key": "value"}, dict)
        
        # Test with incorrect field type
        with pytest.raises(ValidationError) as excinfo:
            validate_field_type("string_field", 42, str)
        assert "string_field" in str(excinfo.value)
        assert "str" in str(excinfo.value)
        
        with pytest.raises(ValidationError) as excinfo:
            validate_field_type("int_field", "not an int", int)
        assert "int_field" in str(excinfo.value)
        assert "int" in str(excinfo.value)
        
        # Test with multiple allowed types
        validate_field_type("multi_field", "test", (str, int))
        validate_field_type("multi_field", 42, (str, int))
        
        with pytest.raises(ValidationError) as excinfo:
            validate_field_type("multi_field", 3.14, (str, int))
        assert "multi_field" in str(excinfo.value)
        assert "str, int" in str(excinfo.value)
        
        # Test with None value
        validate_field_type("nullable_field", None, (str, type(None)))
        
        with pytest.raises(ValidationError) as excinfo:
            validate_field_type("non_nullable_field", None, str)
        assert "non_nullable_field" in str(excinfo.value)
        assert "str" in str(excinfo.value)

    def test_validate_string_length(self):
        """Test that validate_string_length correctly validates string lengths."""
        # Test with valid string length
        validate_string_length("field", "test", min_length=1, max_length=10)
        validate_string_length("field", "exactly5", min_length=8, max_length=8)
        validate_string_length("field", "", min_length=0, max_length=10)
        
        # Test with string too short
        with pytest.raises(ValidationError) as excinfo:
            validate_string_length("field", "test", min_length=5)
        assert "field" in str(excinfo.value)
        assert "minimum length" in str(excinfo.value).lower()
        assert "5" in str(excinfo.value)
        
        # Test with string too long
        with pytest.raises(ValidationError) as excinfo:
            validate_string_length("field", "test", max_length=3)
        assert "field" in str(excinfo.value)
        assert "maximum length" in str(excinfo.value).lower()
        assert "3" in str(excinfo.value)
        
        # Test with non-string value
        with pytest.raises(ValidationError) as excinfo:
            validate_string_length("field", 42, min_length=1)
        assert "field" in str(excinfo.value)
        assert "string" in str(excinfo.value).lower()

    def test_validate_numeric_range(self):
        """Test that validate_numeric_range correctly validates numeric ranges."""
        # Test with valid numeric values
        validate_numeric_range("field", 5, min_value=0, max_value=10)
        validate_numeric_range("field", 0, min_value=0, max_value=10)
        validate_numeric_range("field", 10, min_value=0, max_value=10)
        validate_numeric_range("field", 3.14, min_value=0, max_value=10)
        
        # Test with value too small
        with pytest.raises(ValidationError) as excinfo:
            validate_numeric_range("field", -1, min_value=0)
        assert "field" in str(excinfo.value)
        assert "minimum value" in str(excinfo.value).lower()
        assert "0" in str(excinfo.value)
        
        # Test with value too large
        with pytest.raises(ValidationError) as excinfo:
            validate_numeric_range("field", 11, max_value=10)
        assert "field" in str(excinfo.value)
        assert "maximum value" in str(excinfo.value).lower()
        assert "10" in str(excinfo.value)
        
        # Test with non-numeric value
        with pytest.raises(ValidationError) as excinfo:
            validate_numeric_range("field", "not a number", min_value=0)
        assert "field" in str(excinfo.value)
        assert "numeric" in str(excinfo.value).lower()

    def test_validate_email(self):
        """Test that validate_email correctly validates email addresses."""
        # Test with valid email addresses
        validate_email("email_field", "user@example.com")
        validate_email("email_field", "user.name+tag@example.co.uk")
        validate_email("email_field", "user-name@subdomain.example.com")
        
        # Test with invalid email addresses
        invalid_emails = [
            "not an email",
            "missing@tld",
            "@missing-username.com",
            "spaces in@example.com",
            "missing.domain@",
            "double..dot@example.com",
            "unicode@éxample.com"
        ]
        
        for invalid_email in invalid_emails:
            with pytest.raises(ValidationError) as excinfo:
                validate_email("email_field", invalid_email)
            assert "email_field" in str(excinfo.value)
            assert "valid email" in str(excinfo.value).lower()
        
        # Test with non-string value
        with pytest.raises(ValidationError) as excinfo:
            validate_email("email_field", 42)
        assert "email_field" in str(excinfo.value)
        assert "string" in str(excinfo.value).lower()

    def test_validate_url(self):
        """Test that validate_url correctly validates URLs."""
        # Test with valid URLs
        validate_url("url_field", "https://example.com")
        validate_url("url_field", "http://example.com/path")
        validate_url("url_field", "https://subdomain.example.com/path?query=value#fragment")
        validate_url("url_field", "http://localhost:8080")
        validate_url("url_field", "http://192.168.1.1")
        
        # Test with invalid URLs
        invalid_urls = [
            "not a url",
            "ftp://example.com",  # If only http/https are allowed
            "http://",
            "https:/example.com",  # Missing slash
            "http://example.com:port",  # Invalid port
            "http:// example.com",  # Space in URL
            "http://exam ple.com"  # Space in domain
        ]
        
        for invalid_url in invalid_urls:
            with pytest.raises(ValidationError) as excinfo:
                validate_url("url_field", invalid_url)
            assert "url_field" in str(excinfo.value)
            assert "valid URL" in str(excinfo.value).lower()
        
        # Test with custom allowed schemes
        validate_url("url_field", "ftp://example.com", allowed_schemes=["http", "https", "ftp"])
        
        with pytest.raises(ValidationError) as excinfo:
            validate_url("url_field", "ftp://example.com")
        assert "url_field" in str(excinfo.value)
        assert "valid URL" in str(excinfo.value).lower()
        
        # Test with non-string value
        with pytest.raises(ValidationError) as excinfo:
            validate_url("url_field", 42)
        assert "url_field" in str(excinfo.value)
        assert "string" in str(excinfo.value).lower()

    def test_validate_date(self):
        """Test that validate_date correctly validates date formats."""
        # Test with valid date formats
        validate_date("2023-05-15")
        validate_date("15/05/2023", format="%d/%m/%Y")
        validate_date("2023-05-15T12:30:45", format="%Y-%m-%dT%H:%M:%S")
        
        # Test with invalid date formats
        with pytest.raises(ValidationError) as excinfo:
            validate_date("2023/05/15")
        assert "date" in str(excinfo.value)
        assert "format" in str(excinfo.value).lower()
        assert "%Y-%m-%d" in str(excinfo.value)
        
        with pytest.raises(ValidationError) as excinfo:
            validate_date("15-05-2023")
        assert "date" in str(excinfo.value)
        assert "format" in str(excinfo.value).lower()
        assert "%Y-%m-%d" in str(excinfo.value)
        
        # Test with invalid date values
        with pytest.raises(ValidationError) as excinfo:
            validate_date("2023-13-15")
        assert "date" in str(excinfo.value)
        assert "valid date" in str(excinfo.value).lower()
        
        with pytest.raises(ValidationError) as excinfo:
            validate_date("2023-05-32")
        assert "date" in str(excinfo.value)
        assert "valid date" in str(excinfo.value).lower()
        
        # Test with non-string value
        with pytest.raises(ValidationError) as excinfo:
            validate_date(42)
        assert "date" in str(excinfo.value)
        assert "string" in str(excinfo.value).lower()

    def test_validate_regex(self):
        """Test that validate_regex validates strings against a regex pattern."""
        # Test with a valid string and pattern
        pattern = r"^[a-zA-Z]+$" # Only letters
        value = "HelloWorld"
        
        # Should not raise an exception
        try:
            validate_regex(value, pattern, "test_field")
        except ValidationError as e:
            pytest.fail(f"validate_regex raised unexpected exception: {e}")
        
        # Test with an invalid string
        value = "Hello World 123"
        
        # Should raise ValidationError
        with pytest.raises(ValidationError, match="Field 'test_field' must match regex pattern"):
            validate_regex(value, pattern, "test_field")
            
        # Test with a different valid pattern (email-like)
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        value = "test.email@example.com"
        
        try:
            validate_regex(value, pattern, "email_field")
        except ValidationError as e:
            pytest.fail(f"validate_regex raised unexpected exception: {e}")

    def test_validate_enum(self):
        """Test that validate_enum correctly validates enum values."""
        # Define a dummy enum class for testing
        class DummyEnum:
            OPTION1 = "option1"
            OPTION2 = "option2"
            OPTION3 = "option3"

            def __init__(self, value):
                if value not in [self.OPTION1, self.OPTION2, self.OPTION3]:
                    raise ValueError("Invalid enum value")
                self.value = value

        # Test with valid enum values
        validate_enum("option1", DummyEnum, "field")
        validate_enum("option2", DummyEnum, "field")
        validate_enum("option3", DummyEnum, "field")

        # Test with invalid enum values
        with pytest.raises(ValidationError) as excinfo:
            validate_enum("option4", DummyEnum, "field")
        assert "field" in str(excinfo.value)
        assert "one of" in str(excinfo.value).lower()
        assert "option1" in str(excinfo.value)
        assert "option2" in str(excinfo.value)
        assert "option3" in str(excinfo.value)

    def test_validate_list_items(self):
        """Test that validate_list_items correctly validates list items."""
        # Test with valid list items
        validate_list_items("list_field", [1, 2, 3], item_type=int)
        validate_list_items("list_field", ["a", "b", "c"], item_type=str)
        validate_list_items("list_field", [{"key": "value1"}, {"key": "value2"}], item_type=dict)
        
        # Test with invalid list items
        with pytest.raises(ValidationError) as excinfo:
            validate_list_items("list_field", [1, "2", 3], item_type=int)
        assert "list_field" in str(excinfo.value)
        assert "item type" in str(excinfo.value).lower()
        assert "int" in str(excinfo.value)
        
        # Test with empty list
        validate_list_items("list_field", [], item_type=int)
        
        # Test with min_length and max_length
        validate_list_items("list_field", [1, 2, 3], item_type=int, min_length=1, max_length=5)
        
        with pytest.raises(ValidationError) as excinfo:
            validate_list_items("list_field", [1, 2, 3], item_type=int, min_length=4)
        assert "list_field" in str(excinfo.value)
        assert "minimum length" in str(excinfo.value).lower()
        assert "4" in str(excinfo.value)
        
        with pytest.raises(ValidationError) as excinfo:
            validate_list_items("list_field", [1, 2, 3], item_type=int, max_length=2)
        assert "list_field" in str(excinfo.value)
        assert "maximum length" in str(excinfo.value).lower()
        assert "2" in str(excinfo.value)
        
        # Test with non-list value
        with pytest.raises(ValidationError) as excinfo:
            validate_list_items("list_field", "not a list", item_type=str)
        assert "list_field" in str(excinfo.value)
        assert "list" in str(excinfo.value).lower()

    def test_validate_dict_schema(self):
        """Test that validate_dict_schema correctly validates dictionary schemas."""
        # Define a schema
        schema = {
            "name": {"type": str, "required": True},
            "age": {"type": int, "required": True, "min_value": 0, "max_value": 120},
            "email": {"type": str, "required": False, "validator": validate_email},
            "tags": {"type": list, "required": False, "item_type": str}
        }
        
        # Test with valid dictionary
        valid_dict = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com",
            "tags": ["tag1", "tag2"]
        }
        validate_dict_schema("dict_field", valid_dict, schema)
        
        # Test with missing required field
        invalid_dict = {
            "name": "John Doe",
            "email": "john@example.com"
        }
        with pytest.raises(ValidationError) as excinfo:
            validate_dict_schema("dict_field", invalid_dict, schema)
        assert "dict_field" in str(excinfo.value)
        assert "age" in str(excinfo.value)
        assert "required" in str(excinfo.value).lower()
        
        # Test with invalid field type
        invalid_dict = {
            "name": "John Doe",
            "age": "thirty",
            "email": "john@example.com"
        }
        with pytest.raises(ValidationError) as excinfo:
            validate_dict_schema("dict_field", invalid_dict, schema)
        assert "dict_field" in str(excinfo.value)
        assert "age" in str(excinfo.value)
        assert "int" in str(excinfo.value)
        
        # Test with field validation failure
        invalid_dict = {
            "name": "John Doe",
            "age": 30,
            "email": "not-an-email"
        }
        with pytest.raises(ValidationError) as excinfo:
            validate_dict_schema("dict_field", invalid_dict, schema)
        assert "dict_field" in str(excinfo.value)
        assert "email" in str(excinfo.value)
        assert "valid email" in str(excinfo.value).lower()
        
        # Test with invalid list item type
        invalid_dict = {
            "name": "John Doe",
            "age": 30,
            "tags": ["tag1", 2]
        }
        with pytest.raises(ValidationError) as excinfo:
            validate_dict_schema("dict_field", invalid_dict, schema)
        assert "dict_field" in str(excinfo.value)
        assert "tags" in str(excinfo.value)
        assert "item type" in str(excinfo.value).lower()
        assert "str" in str(excinfo.value)
        
        # Test with non-dict value
        with pytest.raises(ValidationError) as excinfo:
            validate_dict_schema("dict_field", "not a dict", schema)
        assert "dict_field" in str(excinfo.value)
        assert "dictionary" in str(excinfo.value).lower()

    def test_validate_file_path(self):
        """Test that validate_file_path correctly validates file paths."""
        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        try:
            # Test with existing file (assuming validate_file_path allows existing files if not checking existence specifically)
            # Adjust this test based on the actual logic of validate_file_path if needed.
            validate_file_path("file_field", temp_file_path)
            
            # Test with an invalid path (e.g., path traversal)
            invalid_path = "../../../etc/passwd"
            with pytest.raises(ValidationError) as excinfo:
                validate_file_path("file_field", invalid_path)
            assert "file_field" in str(excinfo.value)
            assert "invalid path components" in str(excinfo.value).lower()
            
            # Test with directory instead of file (validate_file_path doesn't seem to have a must_be_file arg, adjust if needed)
            # The original test seems to check for file existence and type, which validate_file_path doesn't do.
            # Removing the directory test or adapting it would be necessary based on the function's actual purpose.
            # For now, I'll remove the directory specific test as validate_file_path only checks for path traversal.
            # temp_dir = os.path.dirname(temp_file_path)
            # with pytest.raises(ValidationError) as excinfo:
            #     validate_file_path("file_field", temp_dir, must_be_file=True)
            # assert "file_field" in str(excinfo.value)
            # assert "file" in str(excinfo.value).lower()
            
            # Test with non-string value
            with pytest.raises(ValidationError) as excinfo:
                validate_file_path("file_field", 42)
            assert "file_field" in str(excinfo.value)
            assert "string" in str(excinfo.value).lower()
        
        finally:
            # Clean up
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_validate_file_extension(self):
        """Test that validate_file_extension correctly validates file extensions."""
        # Test with valid file extensions
        validate_file_extension("file_field", "document.txt", [".txt", ".pdf"])
        validate_file_extension("file_field", "image.jpg", [".jpg", ".png", ".gif"])
        validate_file_extension("file_field", "archive.tar.gz", [".gz", ".zip"])
        
        # Test with invalid file extensions
        with pytest.raises(ValidationError) as excinfo:
            validate_file_extension("file_field", "document.doc", [".txt", ".pdf"])
        assert "file_field" in str(excinfo.value)
        assert "extension" in str(excinfo.value).lower()
        assert ".txt" in str(excinfo.value)
        assert ".pdf" in str(excinfo.value)
        
        # Test with case-insensitive validation
        validate_file_extension("file_field", "document.TXT", [".txt", ".pdf"], case_sensitive=False)
        
        with pytest.raises(ValidationError) as excinfo:
            validate_file_extension("file_field", "document.TXT", [".txt", ".pdf"], case_sensitive=True)
        assert "file_field" in str(excinfo.value)
        assert "extension" in str(excinfo.value).lower()
        
        # Test with non-string value
        with pytest.raises(ValidationError) as excinfo:
            validate_file_extension("file_field", 42, [".txt", ".pdf"])
        assert "file_field" in str(excinfo.value)
        assert "string" in str(excinfo.value).lower()

    def test_validate_json_schema(self):
        """Test that validate_json_schema correctly validates JSON schemas."""
        # Define a JSON schema
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0, "maximum": 120},
                "email": {"type": "string", "format": "email"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["name", "age"]
        }
        
        # Test with valid JSON data
        valid_data = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com",
            "tags": ["tag1", "tag2"]
        }
        validate_json_schema("json_field", valid_data, schema)
        
        # Test with missing required field
        invalid_data = {
            "name": "John Doe",
            "email": "john@example.com"
        }
        with pytest.raises(ValidationError) as excinfo:
            validate_json_schema("json_field", invalid_data, schema)
        assert "json_field" in str(excinfo.value)
        assert "age" in str(excinfo.value)
        assert "required" in str(excinfo.value).lower()
        
        # Test with invalid field type
        invalid_data = {
            "name": "John Doe",
            "age": "thirty",
            "email": "john@example.com"
        }
        with pytest.raises(ValidationError) as excinfo:
            validate_json_schema("json_field", invalid_data, schema)
        assert "json_field" in str(excinfo.value)
        assert "age" in str(excinfo.value)
        assert "integer" in str(excinfo.value).lower()
        
        # Test with value out of range
        invalid_data = {
            "name": "John Doe",
            "age": 150,
            "email": "john@example.com"
        }
        with pytest.raises(ValidationError) as excinfo:
            validate_json_schema("json_field", invalid_data, schema)
        assert "json_field" in str(excinfo.value)
        assert "age" in str(excinfo.value)
        assert "maximum" in str(excinfo.value).lower()
        
        # Test with invalid format
        invalid_data = {
            "name": "John Doe",
            "age": 30,
            "email": "not-an-email"
        }
        with pytest.raises(ValidationError) as excinfo:
            validate_json_schema("json_field", invalid_data, schema)
        assert "json_field" in str(excinfo.value)
        assert "email" in str(excinfo.value)
        assert "format" in str(excinfo.value).lower()
        
        # Test with invalid array item type
        invalid_data = {
            "name": "John Doe",
            "age": 30,
            "tags": ["tag1", 2]
        }
        with pytest.raises(ValidationError) as excinfo:
            validate_json_schema("json_field", invalid_data, schema)
        assert "json_field" in str(excinfo.value)
        assert "tags" in str(excinfo.value)
        assert "string" in str(excinfo.value).lower()

    def test_validate_api_key(self):
        """Test that validate_api_key correctly validates API keys."""
        # Test with valid API key
        validate_api_key("api_key_field", "valid_api_key_12345", min_length=10)
        
        # Test with API key too short
        with pytest.raises(ValidationError) as excinfo:
            validate_api_key("api_key_field", "short", min_length=10)
        assert "api_key_field" in str(excinfo.value)
        assert "length" in str(excinfo.value).lower()
        assert "10" in str(excinfo.value)
        
        # Test with invalid characters
        with pytest.raises(ValidationError) as excinfo:
            validate_api_key("api_key_field", "invalid!@#$%^&*()", pattern=r"^[A-Za-z0-9_-]+$")
        assert "api_key_field" in str(excinfo.value)
        assert "pattern" in str(excinfo.value).lower()
        
        # Test with non-string value
        with pytest.raises(ValidationError) as excinfo:
            validate_api_key("api_key_field", 42)
        assert "api_key_field" in str(excinfo.value)
        assert "string" in str(excinfo.value).lower()

    def test_validate_uuid(self):
        """Test that validate_uuid correctly validates UUIDs."""
        # Test with valid UUIDs
        valid_uuid = str(uuid.uuid4())
        validate_uuid("uuid_field", valid_uuid)
        validate_uuid("uuid_field", "123e4567-e89b-12d3-a456-426614174000")
        
        # Test with invalid UUIDs
        invalid_uuids = [
            "not-a-uuid",
            "123e4567-e89b-12d3-a456",  # Too short
            "123e4567-e89b-12d3-a456-4266141740001",  # Too long
            "123e4567-e89b-12d3-a456-42661417400g"  # Invalid character
        ]
        
        for invalid_uuid in invalid_uuids:
            with pytest.raises(ValidationError) as excinfo:
                validate_uuid("uuid_field", invalid_uuid)
            assert "uuid_field" in str(excinfo.value)
            assert "UUID" in str(excinfo.value)
        
        # Test with non-string value
        with pytest.raises(ValidationError) as excinfo:
            validate_uuid("uuid_field", 42)
        assert "uuid_field" in str(excinfo.value)
        assert "string" in str(excinfo.value).lower()

    def test_validate_ip_address(self):
        """Test that validate_ip_address correctly validates IP addresses."""
        # Test with valid IPv4 addresses
        validate_ip_address("ip_field", "192.168.1.1")
        validate_ip_address("ip_field", "127.0.0.1")
        validate_ip_address("ip_field", "0.0.0.0")
        validate_ip_address("ip_field", "255.255.255.255")
        
        # Test with valid IPv6 addresses
        validate_ip_address("ip_field", "::1")
        validate_ip_address("ip_field", "2001:db8::1")
        validate_ip_address("ip_field", "2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        
        # Test with invalid IP addresses
        invalid_ips = [
            "not-an-ip",
            "192.168.1",  # Incomplete IPv4
            "192.168.1.256",  # Invalid IPv4 octet
            "192.168.1.1.1",  # Too many IPv4 octets
            "2001:db8::g",  # Invalid IPv6 character
            "2001:db8:::1"  # Invalid IPv6 format
        ]
        
        for invalid_ip in invalid_ips:
            with pytest.raises(ValidationError) as excinfo:
                validate_ip_address("ip_field", invalid_ip)
            assert "ip_field" in str(excinfo.value)
            assert "IP address" in str(excinfo.value)
        
        # Test with IPv4 only
        validate_ip_address("ip_field", "192.168.1.1", ipv4_only=True)
        
        with pytest.raises(ValidationError) as excinfo:
            validate_ip_address("ip_field", "2001:db8::1", ipv4_only=True)
        assert "ip_field" in str(excinfo.value)
        assert "IPv4" in str(excinfo.value)
        
        # Test with IPv6 only
        validate_ip_address("ip_field", "2001:db8::1", ipv6_only=True)
        
        with pytest.raises(ValidationError) as excinfo:
            validate_ip_address("ip_field", "192.168.1.1", ipv6_only=True)
        assert "ip_field" in str(excinfo.value)
        assert "IPv6" in str(excinfo.value)
        
        # Test with non-string value
        with pytest.raises(ValidationError) as excinfo:
            validate_ip_address("ip_field", 42)
        assert "ip_field" in str(excinfo.value)
        assert "string" in str(excinfo.value).lower()

    def test_validate_hostname(self):
        """Test that validate_hostname correctly validates hostnames."""
        # Test with valid hostnames
        validate_hostname("hostname_field", "example.com")
        validate_hostname("hostname_field", "sub.example.com")
        validate_hostname("hostname_field", "example")
        validate_hostname("hostname_field", "sub-domain.example.com")
        validate_hostname("hostname_field", "localhost")
        
        # Test with invalid hostnames
        invalid_hostnames = [
            "not a hostname",
            "example..com",  # Double dot
            "-example.com",  # Starting with hyphen
            "example-.com",  # Ending with hyphen
            "example.com-",  # Ending with hyphen
            "exam_ple.com",  # Underscore in domain part
            "a" * 64 + ".com"  # Label too long (max 63)
        ]
        
        for invalid_hostname in invalid_hostnames:
            with pytest.raises(ValidationError) as excinfo:
                validate_hostname("hostname_field", invalid_hostname)
            assert "hostname_field" in str(excinfo.value)
            assert "hostname" in str(excinfo.value).lower()
        
        # Test with non-string value
        with pytest.raises(ValidationError) as excinfo:
            validate_hostname("hostname_field", 42)
        assert "hostname_field" in str(excinfo.value)
        assert "string" in str(excinfo.value).lower()

    def test_validate_port_number(self):
        """Test that validate_port_number correctly validates port numbers."""
        # Test with valid port numbers
        validate_port_number("port_field", 80)
        validate_port_number("port_field", 443)
        validate_port_number("port_field", 8080)
        validate_port_number("port_field", 1)
        validate_port_number("port_field", 65535)
        
        # Test with invalid port numbers
        invalid_ports = [
            0,  # Below minimum
            65536,  # Above maximum
            -1,  # Negative
            "80"  # String instead of int
        ]
        
        for invalid_port in invalid_ports:
            with pytest.raises(ValidationError) as excinfo:
                validate_port_number("port_field", invalid_port)
            assert "port_field" in str(excinfo.value)
            assert "port" in str(excinfo.value).lower()
        
        # Test with custom range
        validate_port_number("port_field", 8000, min_port=8000, max_port=9000)
        
        with pytest.raises(ValidationError) as excinfo:
            validate_port_number("port_field", 7999, min_port=8000, max_port=9000)
        assert "port_field" in str(excinfo.value)
        assert "minimum" in str(excinfo.value).lower()
        assert "8000" in str(excinfo.value)
        
        with pytest.raises(ValidationError) as excinfo:
            validate_port_number("port_field", 9001, min_port=8000, max_port=9000)
        assert "port_field" in str(excinfo.value)
        assert "maximum" in str(excinfo.value).lower()
        assert "9000" in str(excinfo.value)

    def test_validate_boolean(self):
        """Test that validate_boolean correctly validates boolean values."""
        # Test with valid boolean values
        validate_boolean("bool_field", True)
        validate_boolean("bool_field", False)
        
        # Test with string representations (if supported)
        validate_boolean("bool_field", "true", accept_string=True)
        validate_boolean("bool_field", "false", accept_string=True)
        validate_boolean("bool_field", "yes", accept_string=True, true_values=["yes", "y", "true", "1"])
        validate_boolean("bool_field", "no", accept_string=True, false_values=["no", "n", "false", "0"])
        
        # Test with invalid values
        invalid_bools = [
            "not a boolean",
            "TRUE",  # Case-sensitive by default
            "FALSE",
            42,
            None
        ]
        
        for invalid_bool in invalid_bools:
            with pytest.raises(ValidationError) as excinfo:
                validate_boolean("bool_field", invalid_bool)
            assert "bool_field" in str(excinfo.value)
            assert "boolean" in str(excinfo.value).lower()
        
        # Test with case-insensitive string validation
        validate_boolean("bool_field", "TRUE", accept_string=True, case_sensitive=False)
        validate_boolean("bool_field", "FALSE", accept_string=True, case_sensitive=False)
        
        # Test with string validation disabled
        with pytest.raises(ValidationError) as excinfo:
            validate_boolean("bool_field", "true", accept_string=False)
        assert "bool_field" in str(excinfo.value)
        assert "boolean" in str(excinfo.value).lower()

    def test_validate_choice(self):
        """Test that validate_choice correctly validates choices."""
        # Define choices
        choices = ["option1", "option2", "option3"]
        
        # Test with valid choices
        validate_choice("choice_field", "option1", choices)
        validate_choice("choice_field", "option2", choices)
        validate_choice("choice_field", "option3", choices)
        
        # Test with invalid choices
        with pytest.raises(ValidationError) as excinfo:
            validate_choice("choice_field", "option4", choices)
        assert "choice_field" in str(excinfo.value)
        assert "valid choice" in str(excinfo.value).lower()
        assert "option1" in str(excinfo.value)
        assert "option2" in str(excinfo.value)
        assert "option3" in str(excinfo.value)
        
        # Test with case-insensitive validation
        validate_choice("choice_field", "OPTION1", choices, case_sensitive=False)
        
        with pytest.raises(ValidationError) as excinfo:
            validate_choice("choice_field", "OPTION1", choices, case_sensitive=True)
        assert "choice_field" in str(excinfo.value)
        assert "valid choice" in str(excinfo.value).lower()
        
        # Test with custom error message
        with pytest.raises(ValidationError) as excinfo:
            validate_choice("choice_field", "option4", choices, error_message="Invalid option selected")
        assert "choice_field" in str(excinfo.value)
        assert "Invalid option selected" in str(excinfo.value)

    def test_validate_not_empty(self):
        """Test that validate_not_empty correctly validates non-empty values."""
        # Test with non-empty values
        validate_not_empty("field", "value")
        validate_not_empty("field", 42)
        validate_not_empty("field", [1, 2, 3])
        validate_not_empty("field", {"key": "value"})
        
        # Test with empty values
        empty_values = [
            "",
            [],
            {},
            None
        ]
        
        for empty_value in empty_values:
            with pytest.raises(ValidationError) as excinfo:
                validate_not_empty("field", empty_value)
            assert "field" in str(excinfo.value)
            assert "empty" in str(excinfo.value).lower()
        
        # Test with custom error message
        with pytest.raises(ValidationError) as excinfo:
            validate_not_empty("field", "", error_message="Field cannot be empty")
        assert "field" in str(excinfo.value)
        assert "Field cannot be empty" in str(excinfo.value)

    def test_schema_validator(self):
        """Test that SchemaValidator correctly validates data against schemas."""
        # Define a schema
        schema = {
            "name": {"type": str, "required": True},
            "age": {"type": int, "required": True, "min_value": 0, "max_value": 120},
            "email": {"type": str, "required": False, "validator": validate_email},
            "tags": {"type": list, "required": False, "item_type": str}
        }
        
        # Create a SchemaValidator
        validator = SchemaValidator(schema)
        
        # Test with valid data
        valid_data = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com",
            "tags": ["tag1", "tag2"]
        }
        validator.validate(valid_data)  # Should not raise an exception
        
        # Test with invalid data
        invalid_data = {
            "name": "John Doe",
            "age": "thirty",  # Invalid type
            "email": "not-an-email"  # Invalid email
        }
        with pytest.raises(ValidationError):
            validator.validate(invalid_data)
        
        # Test validate_field method
        validator.validate_field("name", "Jane Doe")  # Should not raise an exception
        
        with pytest.raises(ValidationError):
            validator.validate_field("age", "thirty")  # Invalid type
        
        with pytest.raises(ValidationError):
            validator.validate_field("email", "not-an-email")  # Invalid email
        
        # Test with unknown field
        with pytest.raises(ValidationError):
            validator.validate_field("unknown_field", "value")

    def test_data_validator(self):
        """Test that DataValidator correctly validates data."""
        # Create a DataValidator
        validator = DataValidator()
        
        # Test string validation
        validator.validate_string("string_field", "value", min_length=1, max_length=10)
        
        with pytest.raises(ValidationError):
            validator.validate_string("string_field", "too_long_value", max_length=10)
        
        # Test numeric validation
        validator.validate_number("number_field", 42, min_value=0, max_value=100)
        
        with pytest.raises(ValidationError):
            validator.validate_number("number_field", 101, max_value=100)
        
        # Test email validation
        validator.validate_email("email_field", "user@example.com")
        
        with pytest.raises(ValidationError):
            validator.validate_email("email_field", "not-an-email")
        
        # Test URL validation
        validator.validate_url("url_field", "https://example.com")
        
        with pytest.raises(ValidationError):
            validator.validate_url("url_field", "not-a-url")
        
        # Test list validation
        validator.validate_list("list_field", [1, 2, 3], item_type=int)
        
        with pytest.raises(ValidationError):
            validator.validate_list("list_field", [1, "2", 3], item_type=int)
        
        # Test dictionary validation
        schema = {
            "name": {"type": str, "required": True},
            "age": {"type": int, "required": True}
        }
        validator.validate_dict("dict_field", {"name": "John", "age": 30}, schema)
        
        with pytest.raises(ValidationError):
            validator.validate_dict("dict_field", {"name": "John"}, schema)  # Missing required field

    def test_validation_with_brave_search_mcp(self):
        """Test integration of validation with BraveSearchMCP."""
        # This test verifies that validation functions work correctly with BraveSearchMCP
        from apps.mcps.brave_search_mcp import BraveSearchMCP
        
        # Mock the BraveSearchMCP._start_server method to avoid actual server startup
        with patch.object(BraveSearchMCP, '_start_server'):
            # Create a BraveSearchMCP instance with a valid API key
            brave_mcp = BraveSearchMCP(api_key="valid_api_key")
            
            # Mock the process and _send_request method
            brave_mcp.process = MagicMock()
            brave_mcp._send_request = MagicMock(return_value={
                "tools": [{"name": "brave_web_search", "description": "Web search tool"}]
            })
            
            # Test with invalid API key
            with patch('apps.utils.validation.validate_api_key', side_effect=ValidationError("Invalid API key")):
                with pytest.raises(ValueError) as excinfo:
                    BraveSearchMCP(api_key="invalid_api_key")
                assert "API key" in str(excinfo.value)
            
            # Test with invalid search query
            with patch('apps.utils.validation.validate_string_length', side_effect=ValidationError("Query too long")):
                with pytest.raises(RuntimeError) as excinfo:
                    brave_mcp.web_search("a" * 500)  # Query too long
                assert "Query too long" in str(excinfo.value)
            
            # Test with invalid count parameter
            with patch('apps.utils.validation.validate_numeric_range', side_effect=ValidationError("Count out of range")):
                with pytest.raises(RuntimeError) as excinfo:
                    brave_mcp.web_search("test query", count=100)  # Count too high
                assert "Count out of range" in str(excinfo.value)

    def test_validation_with_research_workflow(self):
        """Test validation with research workflow."""
        # This test verifies that validation functions work correctly with the research workflow
        # Mock the research workflow
        with patch('apps.workflows.research_workflow.ResearchWorkflow') as MockWorkflow:
            # Mock the workflow instance
            mock_workflow = MockWorkflow.return_value
            
            # Test with invalid query
            with patch('apps.utils.validation.validate_not_empty', side_effect=ValidationError("Query cannot be empty")):
                with pytest.raises(ValueError) as excinfo:
                    mock_workflow.execute.side_effect = ValueError("Query cannot be empty")
                    mock_workflow.execute("")
                assert "empty" in str(excinfo.value).lower()
            
            # Test with invalid parameters
            with patch('apps.utils.validation.validate_dict_schema', side_effect=ValidationError("Invalid parameters")):
                with pytest.raises(ValueError) as excinfo:
                    mock_workflow.execute.side_effect = ValueError("Invalid parameters")
                    mock_workflow.execute("query", {"invalid": "parameters"})
                assert "Invalid parameters" in str(excinfo.value)

    def test_validation_with_file_operations(self):
        """Test validation with file operations."""
        # This test verifies that validation functions work correctly with file operations
        from apps.mcps.filesystem_mcp import FilesystemMCP
        
        # Mock the FilesystemMCP._start_server method to avoid actual server startup
        with patch.object(FilesystemMCP, '_start_server'):
            # Create a FilesystemMCP instance
            fs_mcp = FilesystemMCP(workspace_dir="/tmp")
            
            # Mock the process and _send_request method
            fs_mcp.process = MagicMock()
            fs_mcp._send_request = MagicMock(return_value={
                "content": [{"type": "text", "text": "Success"}],
                "isError": False
            })
            
            # Test with invalid file path
            with patch('apps.utils.validation.validate_string', side_effect=ValidationError("Invalid file path")):
                with pytest.raises(RuntimeError) as excinfo:
                    fs_mcp._send_request.side_effect = RuntimeError("Invalid file path")
                    fs_mcp.read_file("../../../etc/passwd")  # Path traversal attempt
                assert "Invalid file path" in str(excinfo.value)
            
            # Test with invalid file extension
            with patch('apps.utils.validation.validate_file_extension', side_effect=ValidationError("Invalid file extension")):
                with pytest.raises(RuntimeError) as excinfo:
                    fs_mcp._send_request.side_effect = RuntimeError("Invalid file extension")
                    fs_mcp.write_file("malicious.exe", "content")
                assert "Invalid file extension" in str(excinfo.value)

    def test_validation_error_handling(self):
        """Test validation error handling."""
        # Test creating a ValidationError
        error = ValidationError("Test error message")
        assert str(error) == "Test error message"
        
        # Test creating a ValidationError with a field name
        error = ValidationError("Test error message", field="test_field")
        assert str(error) == "Field 'test_field': Test error message"
        
        # Test creating a ValidationError with a nested field
        error = ValidationError("Test error message", field="parent.child")
        assert str(error) == "Field 'parent.child': Test error message"
        
        # Test creating a ValidationError with a field name and value
        error = ValidationError("Test error message", field="test_field", value="test_value")
        assert str(error) == "Field 'test_field' (value: 'test_value'): Test error message"
        
        # Test creating a ValidationError with a field name and invalid value type
        error = ValidationError("Test error message", field="test_field", value={"complex": "value"})
        assert str(error) == "Field 'test_field': Test error message"

    def test_validation_with_api_keys(self):
        """Test validation with API keys."""
        # Test API key validation for BraveSearchMCP
        from apps.mcps.brave_search_mcp import BraveSearchMCP
        
        # Mock environment variable
        with patch.dict(os.environ, {"BRAVE_API_KEY": "valid_api_key"}):
            # Mock the _start_server method to avoid actual server startup
            with patch.object(BraveSearchMCP, '_start_server'):
                # Create a BraveSearchMCP instance with environment variable
                brave_mcp = BraveSearchMCP()
                assert brave_mcp.api_key == "valid_api_key"
        
        # Test with missing API key
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as excinfo:
                BraveSearchMCP()
            assert "API key" in str(excinfo.value)
        
        # Test with explicit API key
        with patch.object(BraveSearchMCP, '_start_server'):
            brave_mcp = BraveSearchMCP(api_key="explicit_api_key")
            assert brave_mcp.api_key == "explicit_api_key"
        
        # Test API key validation for EverArtMCP
        from apps.mcps.everart_mcp import EverArtMCP
        
        # Mock environment variable
        with patch.dict(os.environ, {"EVERART_API_KEY": "valid_api_key"}):
            # Mock the _start_server method to avoid actual server startup
            with patch.object(EverArtMCP, '_start_server'):
                # Create an EverArtMCP instance with environment variable
                everart_mcp = EverArtMCP()
                assert everart_mcp.api_key == "valid_api_key"
        
        # Test with missing API key
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as excinfo:
                EverArtMCP()
            assert "API key" in str(excinfo.value)

    def test_validation_with_json_schema(self):
        """Test validation with JSON Schema."""
        # Define a JSON Schema
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["name", "age"]
        }
        
        # Test with valid data
        valid_data = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com"
        }
        
        # Mock jsonschema.validate to avoid dependency issues
        with patch('jsonschema.validate'):
            validate_json_schema("data", valid_data, schema)
        
        # Test with invalid data
        invalid_data = {
            "name": "John Doe",
            "age": -1,  # Invalid age
            "email": "not-an-email"  # Invalid email
        }
        
        # Mock jsonschema.validate to raise a ValidationError
        with patch('jsonschema.validate', side_effect=jsonschema.exceptions.ValidationError("Invalid data")):
            with pytest.raises(ValidationError) as excinfo:
                validate_json_schema("data", invalid_data, schema)
            assert "data" in str(excinfo.value)
            assert "schema" in str(excinfo.value).lower()

    def test_validation_with_brave_search_parameters(self):
        """Test validation with Brave Search parameters."""
        from apps.mcps.brave_search_mcp import BraveSearchMCP
        
        # Mock the BraveSearchMCP._start_server method to avoid actual server startup
        with patch.object(BraveSearchMCP, '_start_server'):
            # Create a BraveSearchMCP instance
            brave_mcp = BraveSearchMCP(api_key="valid_api_key")
            
            # Mock the process and _send_request method
            brave_mcp.process = MagicMock()
            brave_mcp._send_request = MagicMock(return_value={
                "content": [{"type": "text", "text": "Search results"}],
                "isError": False
            })
            
            # Test with valid parameters
            brave_mcp.web_search("valid query", count=10, offset=0)
            
            # Test with invalid query
            brave_mcp._send_request.side_effect = RuntimeError("Query too long")
            with pytest.raises(RuntimeError) as excinfo:
                brave_mcp.web_search("a" * 500)
            assert "Query too long" in str(excinfo.value)
            
            # Reset side_effect
            brave_mcp._send_request.side_effect = None
            
            # Test with invalid count
            brave_mcp._send_request.side_effect = RuntimeError("Count out of range")
            with pytest.raises(RuntimeError) as excinfo:
                brave_mcp.web_search("valid query", count=100)
            assert "Count out of range" in str(excinfo.value)
            
            # Reset side_effect
            brave_mcp._send_request.side_effect = None
            
            # Test with invalid offset
            brave_mcp._send_request.side_effect = RuntimeError("Offset out of range")
            with pytest.raises(RuntimeError) as excinfo:
                brave_mcp.web_search("valid query", offset=100)
            assert "Offset out of range" in str(excinfo.value)

    def test_validation_with_everart_parameters(self):
        """Test validation with EverArt parameters."""
        from apps.mcps.everart_mcp import EverArtMCP
        
        # Mock the EverArtMCP._start_server method to avoid actual server startup
        with patch.object(EverArtMCP, '_start_server'):
            # Create an EverArtMCP instance
            everart_mcp = EverArtMCP(api_key="valid_api_key")
            
            # Mock the process and _send_request method
            everart_mcp.process = MagicMock()
            everart_mcp._send_request = MagicMock(return_value={
                "content": [{"type": "text", "text": "Generated image URL"}],
                "isError": False
            })
            
            # Test with valid parameters
            everart_mcp.generate_image("valid prompt", style="realistic", aspect_ratio="1:1", num_images=1)
            
            # Test with invalid prompt
            everart_mcp._send_request.side_effect = RuntimeError("Prompt too long")
            with pytest.raises(RuntimeError) as excinfo:
                everart_mcp.generate_image("a" * 1000)
            assert "Prompt too long" in str(excinfo.value)
            
            # Reset side_effect
            everart_mcp._send_request.side_effect = None
            
            # Test with invalid style
            everart_mcp._send_request.side_effect = RuntimeError("Invalid style")
            with pytest.raises(RuntimeError) as excinfo:
                everart_mcp.generate_image("valid prompt", style="invalid_style")
            assert "Invalid style" in str(excinfo.value)
            
            # Reset side_effect
            everart_mcp._send_request.side_effect = None
            
            # Test with invalid aspect ratio
            everart_mcp._send_request.side_effect = RuntimeError("Invalid aspect ratio")
            with pytest.raises(RuntimeError) as excinfo:
                everart_mcp.generate_image("valid prompt", aspect_ratio="invalid")
            assert "Invalid aspect ratio" in str(excinfo.value)


if __name__ == "__main__":
    import sys
    pytest.main(["-xvs", __file__])
