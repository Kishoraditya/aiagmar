"""
Unit tests for the helper utilities.
"""

import pytest
import os
import json
import uuid
import time
import asyncio
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

from apps.utils.helpers import (
    # Import helper functions that would be defined in the helpers.py file
    # These are assumed based on typical patterns, adjust as needed
    generate_uuid,
    format_timestamp,
    sanitize_filename,
    read_json_file,
    write_json_file,
    merge_dicts,
    flatten_dict,
    chunk_list,
    retry_operation,
    async_retry_with_backoff,
    is_valid_url,
    extract_domain,
    truncate_text,
    extract_keywords,
    calculate_similarity,
    safe_request,
    async_safe_request,
    create_directory_if_not_exists,
    remove_file_if_exists,
    get_file_extension,
    is_image_file,
    is_text_file,
    get_mime_type,
    encode_base64,
    decode_base64,
    compress_data,
    decompress_data,
    encrypt_data,
    decrypt_data,
    parse_args,
    setup_logging,
    timer,
    async_timer
)


class TestHelpers:
    """Test suite for helper utilities."""

    def test_generate_uuid(self):
        """Test that generate_uuid returns a valid UUID."""
        # Generate a UUID
        uuid_str = generate_uuid()
        
        # Verify it's a valid UUID string
        assert isinstance(uuid_str, str)
        assert len(uuid_str) == 36  # Standard UUID length
        
        # Verify it can be parsed as a UUID
        uuid_obj = uuid.UUID(uuid_str)
        assert str(uuid_obj) == uuid_str
        
        # Generate another UUID and verify it's different
        uuid_str2 = generate_uuid()
        assert uuid_str != uuid_str2

    def test_format_timestamp(self):
        """Test that format_timestamp formats datetime objects correctly."""
        # Create a datetime object
        dt = datetime(2023, 5, 15, 12, 30, 45)
        
        # Format with default format
        formatted = format_timestamp(dt)
        assert formatted == "2023-05-15T12:30:45"
        
        # Format with custom format
        custom_format = "%Y/%m/%d %H:%M:%S"
        formatted = format_timestamp(dt, format_str=custom_format)
        assert formatted == "2023/05/15 12:30:45"
        
        # Test with current time
        now = datetime.now()
        formatted = format_timestamp(now)
        # Can't assert exact value, but can check format
        assert len(formatted) == 19  # YYYY-MM-DDTHH:MM:SS
        assert "T" in formatted
        assert formatted[:4] == str(now.year)

    def test_sanitize_filename(self):
        """Test that sanitize_filename removes invalid characters from filenames."""
        # Test with invalid characters
        filename = "file/with\\invalid:characters?*|\"<>"
        sanitized = sanitize_filename(filename)
        
        # Verify invalid characters are removed or replaced
        assert "/" not in sanitized
        assert "\\" not in sanitized
        assert ":" not in sanitized
        assert "?" not in sanitized
        assert "*" not in sanitized
        assert "|" not in sanitized
        assert "\"" not in sanitized
        assert "<" not in sanitized
        assert ">" not in sanitized
        
        # Test with spaces
        filename = "file with spaces"
        sanitized = sanitize_filename(filename, replace_spaces=True)
        assert " " not in sanitized
        assert "_" in sanitized
        
        # Test with maximum length
        long_filename = "a" * 300
        sanitized = sanitize_filename(long_filename, max_length=255)
        assert len(sanitized) <= 255

    def test_read_json_file(self):
        """Test that read_json_file loads JSON data from a file."""
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
            json_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
            json.dump(json_data, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Load the JSON file
            loaded_data = read_json_file(temp_file_path)
            
            # Verify the loaded data
            assert loaded_data == json_data
            assert loaded_data["key"] == "value"
            assert loaded_data["number"] == 42
            assert loaded_data["list"] == [1, 2, 3]
            
            # Test with non-existent file
            with pytest.raises(FileNotFoundError):
                read_json_file("non_existent_file.json")
            
            # Test with invalid JSON
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as invalid_file:
                invalid_file.write("invalid json")
                invalid_file_path = invalid_file.name
            
            with pytest.raises(json.JSONDecodeError):
                read_json_file(invalid_file_path)
            
            # Clean up the invalid file
            os.remove(invalid_file_path)
            
        finally:
            # Clean up the temporary file
            os.remove(temp_file_path)

    def test_write_json_file(self):
        """Test that write_json_file saves JSON data to a file."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define the JSON file path
            json_file_path = os.path.join(temp_dir, "test.json")
            
            # Define the data to save
            json_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
            
            # Save the JSON file
            write_json_file(json_file_path, json_data)
            
            # Verify the file exists
            assert os.path.exists(json_file_path)
            
            # Load the file and verify the content
            with open(json_file_path, "r") as f:
                loaded_data = json.load(f)
                assert loaded_data == json_data
            
            # Test with pretty print
            write_json_file(json_file_path, json_data, indent=2)
            
            # Verify the file exists and is formatted
            with open(json_file_path, "r") as f:
                content = f.read()
                assert "{\n  \"key\": \"value\",\n" in content  # Check for pretty print formatting
                loaded_data = json.loads(content)
                assert loaded_data == json_data

    def test_merge_dictionaries(self):
        """Test that merge_dicts correctly merges dictionaries."""
        # Define dictionaries to merge
        dict1 = {
            "a": 1,
            "b": {
                "c": 2,
                "d": 3
            },
            "e": [1, 2, 3]
        }

        dict2 = {
            "a": 10,  # Overwrite
            "b": {
                "c": 20,  # Overwrite
                "f": 30   # Add
            },
            "g": 40,      # Add
            "e": [4, 5]   # Overwrite
        }

        # Merge the dictionaries
        merged = merge_dicts(dict1, dict2)

        # Verify the merged dictionary
        assert merged["a"] == 10  # Overwritten
        assert merged["b"]["c"] == 20  # Overwritten
        assert merged["b"]["d"] == 3  # Preserved
        assert merged["b"]["f"] == 30  # Added
        assert merged["e"] == [4, 5]  # Overwritten
        assert merged["g"] == 40  # Added

        # Test with empty dictionaries
        assert merge_dicts({}, {}) == {}
        assert merge_dicts(dict1, {}) == dict1
        assert merge_dicts({}, dict2) == dict2

    def test_flatten_dict(self):
        """Test that flatten_dict correctly flattens nested dictionaries."""
        # Define a nested dictionary
        nested_dict = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            },
            "f": [1, 2, 3]
        }
        
        # Flatten the dictionary
        flattened = flatten_dict(nested_dict)
        
        # Verify the flattened dictionary
        assert flattened["a"] == 1
        assert flattened["b.c"] == 2
        assert flattened["b.d.e"] == 3
        assert flattened["f"] == [1, 2, 3]
        
        # Test with custom separator
        flattened = flatten_dict(nested_dict, separator="/")
        assert flattened["a"] == 1
        assert flattened["b/c"] == 2
        assert flattened["b/d/e"] == 3
        
        # Test with empty dictionary
        assert flatten_dict({}) == {}

    def test_chunk_list(self):
        """Test that chunk_list correctly splits lists into chunks."""
        # Define a list to chunk
        items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Chunk the list
        chunks = chunk_list(items, chunk_size=3)
        
        # Verify the chunks
        assert len(chunks) == 4
        assert chunks[0] == [1, 2, 3]
        assert chunks[1] == [4, 5, 6]
        assert chunks[2] == [7, 8, 9]
        assert chunks[3] == [10]
        
        # Test with empty list
        assert chunk_list([], chunk_size=3) == []
        
        # Test with chunk size larger than list
        chunks = chunk_list(items, chunk_size=20)
        assert len(chunks) == 1
        assert chunks[0] == items
        
        # Test with chunk size equal to list length
        chunks = chunk_list(items, chunk_size=10)
        assert len(chunks) == 1
        assert chunks[0] == items

    def test_retry_with_backoff(self):
        """Test that retry_with_backoff retries failed operations with exponential backoff."""
        # Create a function that succeeds on the third attempt
        mock_func = MagicMock(side_effect=[
            ValueError("First attempt failed"),
            ValueError("Second attempt failed"),
            "success"
        ])
        
        # Wrap the function with retry_with_backoff
        result = retry_with_backoff(
            mock_func,
            max_retries=3,
            initial_delay=0.01,
            max_delay=0.1,
            backoff_factor=2,
            exceptions=(ValueError,)
        )
        
        # Verify the function was called three times and returned the successful result
        assert mock_func.call_count == 3
        assert result == "success"
        
        # Create a function that always fails
        mock_func = MagicMock(side_effect=ValueError("Always fails"))
        
        # Wrap the function with retry_with_backoff
        with pytest.raises(ValueError, match="Always fails"):
            retry_with_backoff(
                mock_func,
                max_retries=2,
                initial_delay=0.01,
                max_delay=0.1,
                backoff_factor=2,
                exceptions=(ValueError,)
            )
        
        # Verify the function was called the maximum number of times
        assert mock_func.call_count == 3  # Initial attempt + 2 retries

    @pytest.mark.asyncio
    async def test_async_retry_with_backoff(self):
        """Test that async_retry_with_backoff retries failed async operations with exponential backoff."""
        # Create an async function that succeeds on the third attempt
        mock_func = AsyncMock(side_effect=[
            ValueError("First attempt failed"),
            ValueError("Second attempt failed"),
            "success"
        ])
        
        # Wrap the function with async_retry_with_backoff
        result = await async_retry_with_backoff(
            mock_func,
            max_retries=3,
            initial_delay=0.01,
            max_delay=0.1,
            backoff_factor=2,
            exceptions=(ValueError,)
        )
        
        # Verify the function was called three times and returned the successful result
        assert mock_func.call_count == 3
        assert result == "success"
        
        # Create an async function that always fails
        mock_func = AsyncMock(side_effect=ValueError("Always fails"))
        
        # Wrap the function with async_retry_with_backoff
        with pytest.raises(ValueError, match="Always fails"):
            await async_retry_with_backoff(
                mock_func,
                max_retries=2,
                initial_delay=0.01,
                max_delay=0.1,
                backoff_factor=2,
                exceptions=(ValueError,)
            )
        
        # Verify the function was called the maximum number of times
        assert mock_func.call_count == 3  # Initial attempt + 2 retries

    def test_is_valid_url(self):
        """Test that is_valid_url correctly validates URLs."""
        # Test valid URLs
        valid_urls = [
            "https://example.com",
            "http://example.com",
            "https://example.com/path",
            "https://example.com/path?query=value",
            "https://example.com/path#fragment",
            "https://sub.example.com",
            "https://example.com:8080",
            "http://192.168.1.1",
            "http://localhost",
            "http://localhost:8080"
        ]
        
        for url in valid_urls:
            assert is_valid_url(url), f"URL should be valid: {url}"
        # Test invalid URLs
        invalid_urls = [
            "example.com",  # Missing scheme
            "https://",  # Missing host
            "http:/example.com",  # Missing slash
            "https://example.com:port",  # Invalid port
            "ftp://example.com",  # Unsupported scheme (if only http/https are supported)
            "not a url at all",
            "http:// example.com",  # Space in URL
            "http://exam ple.com",  # Space in domain
            "file:///path/to/file"  # File scheme (if not supported)
        ]
        
        for url in invalid_urls:
            assert not is_valid_url(url), f"URL should be invalid: {url}"
        
        # Test with custom schemes
        assert is_valid_url("ftp://example.com", allowed_schemes=["ftp", "http", "https"])
        assert not is_valid_url("ftp://example.com", allowed_schemes=["http", "https"])

    def test_extract_domain(self):
        """Test that extract_domain correctly extracts domains from URLs."""
        # Test domain extraction
        test_cases = [
            ("https://example.com", "example.com"),
            ("http://example.com/path", "example.com"),
            ("https://sub.example.com", "sub.example.com"),
            ("https://sub.sub.example.com", "sub.sub.example.com"),
            ("https://example.com:8080", "example.com"),
            ("http://192.168.1.1", "192.168.1.1"),
            ("http://localhost", "localhost"),
            ("http://localhost:8080", "localhost")
        ]
        
        for url, expected_domain in test_cases:
            assert extract_domain(url) == expected_domain
        
        # Test with invalid URLs
        with pytest.raises(ValueError):
            extract_domain("not a url")
        
        # Test with include_subdomain option
        assert extract_domain("https://sub.example.com", include_subdomain=True) == "sub.example.com"
        assert extract_domain("https://sub.example.com", include_subdomain=False) == "example.com"
        assert extract_domain("https://sub.sub.example.com", include_subdomain=False) == "example.com"

    def test_truncate_text(self):
        """Test that truncate_text correctly truncates text to a specified length."""
        # Test text truncation
        text = "This is a long text that needs to be truncated."
        
        # Truncate to 10 characters
        truncated = truncate_text(text, max_length=10)
        assert len(truncated) <= 10
        assert truncated == "This is..."
        
        # Truncate to 20 characters
        truncated = truncate_text(text, max_length=20)
        assert len(truncated) <= 20
        assert truncated == "This is a long..."
        
        # Test with custom suffix
        truncated = truncate_text(text, max_length=10, suffix="[...]")
        assert truncated == "This is[...]"
        
        # Test with text shorter than max_length
        short_text = "Short text"
        truncated = truncate_text(short_text, max_length=20)
        assert truncated == short_text
        
        # Test with empty text
        assert truncate_text("", max_length=10) == ""

    def test_extract_keywords(self):
        """Test that extract_keywords correctly extracts keywords from text."""
        # Test keyword extraction
        text = "This is a sample text about artificial intelligence and machine learning."
        
        # Extract keywords
        keywords = extract_keywords(text)
        
        # Verify keywords are extracted
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        
        # Common keywords that should be extracted
        expected_keywords = ["artificial", "intelligence", "machine", "learning"]
        for keyword in expected_keywords:
            assert keyword in keywords
        
        # Test with max_keywords
        limited_keywords = extract_keywords(text, max_keywords=2)
        assert len(limited_keywords) <= 2
        
        # Test with min_word_length
        long_keywords = extract_keywords(text, min_word_length=8)
        for keyword in long_keywords:
            assert len(keyword) >= 8
        
        # Test with empty text
        assert extract_keywords("") == []

    def test_calculate_similarity(self):
        """Test that calculate_similarity correctly calculates similarity between texts."""
        # Test text similarity calculation
        text1 = "This is a sample text about artificial intelligence."
        text2 = "This is a text about AI and machine learning."
        text3 = "Completely unrelated text about cooking recipes."
        
        # Calculate similarities
        similarity_1_2 = calculate_similarity(text1, text2)
        similarity_1_3 = calculate_similarity(text1, text3)
        
        # Verify similarities
        assert 0 <= similarity_1_2 <= 1
        assert 0 <= similarity_1_3 <= 1
        
        # text1 and text2 should be more similar than text1 and text3
        assert similarity_1_2 > similarity_1_3
        
        # Same text should have similarity 1
        assert calculate_similarity(text1, text1) == 1.0
        
        # Test with empty texts
        assert calculate_similarity("", "") == 1.0
        assert calculate_similarity(text1, "") == 0.0

    def test_safe_request(self):
        """Test that safe_request safely makes HTTP requests with retries and error handling."""
        # Mock the requests.request function
        with patch("requests.request") as mock_request:
            # Mock a successful response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "test"}
            mock_request.return_value = mock_response
            
            # Make a request
            response = safe_request("GET", "https://example.com")
            
            # Verify the request was made correctly
            mock_request.assert_called_once()
            assert response.status_code == 200
            assert response.json() == {"data": "test"}
            
            # Mock a failed response that succeeds on retry
            mock_request.reset_mock()
            mock_request.side_effect = [
                Exception("Connection error"),
                mock_response
            ]
            
            # Make a request with retries
            response = safe_request("GET", "https://example.com", max_retries=2)
            
            # Verify the request was retried
            assert mock_request.call_count == 2
            assert response.status_code == 200
            
            # Mock a response with an error status code
            mock_request.reset_mock()
            error_response = MagicMock()
            error_response.status_code = 404
            error_response.text = "Not Found"
            mock_request.return_value = error_response
            
            # Make a request that returns an error
            response = safe_request("GET", "https://example.com")
            
            # Verify the error response is returned
            assert response.status_code == 404
            
            # Test with raise_for_status=True
            mock_request.reset_mock()
            error_response.raise_for_status.side_effect = Exception("404 Client Error")
            
            # Make a request that raises an exception
            with pytest.raises(Exception, match="404 Client Error"):
                safe_request("GET", "https://example.com", raise_for_status=True)

    @pytest.mark.asyncio
    async def test_async_safe_request(self):
        """Test that async_safe_request safely makes async HTTP requests with retries and error handling."""
        # Mock the aiohttp.ClientSession.request method
        with patch("aiohttp.ClientSession.request") as mock_request:
            # Mock a successful response
            mock_response = AsyncMock()
            mock_response.__aenter__.return_value = mock_response
            mock_response.status = 200
            mock_response.json.return_value = {"data": "test"}
            mock_request.return_value = mock_response
            
            # Make a request
            response = await async_safe_request("GET", "https://example.com")
            
            # Verify the request was made correctly
            mock_request.assert_called_once()
            assert response.status == 200
            assert await response.json() == {"data": "test"}
            
            # Mock a failed response that succeeds on retry
            mock_request.reset_mock()
            mock_request.side_effect = [
                Exception("Connection error"),
                mock_response
            ]
            
            # Make a request with retries
            response = await async_safe_request("GET", "https://example.com", max_retries=2)
            
            # Verify the request was retried
            assert mock_request.call_count == 2
            assert response.status == 200
            
            # Mock a response with an error status code
            mock_request.reset_mock()
            error_response = AsyncMock()
            error_response.__aenter__.return_value = error_response
            error_response.status = 404
            error_response.text.return_value = "Not Found"
            mock_request.return_value = error_response
            
            # Make a request that returns an error
            response = await async_safe_request("GET", "https://example.com")
            
            # Verify the error response is returned
            assert response.status == 404
            
            # Test with raise_for_status=True
            mock_request.reset_mock()
            error_response.raise_for_status.side_effect = Exception("404 Client Error")
            
            # Make a request that raises an exception
            with pytest.raises(Exception, match="404 Client Error"):
                await async_safe_request("GET", "https://example.com", raise_for_status=True)

    def test_create_directory_if_not_exists(self):
        """Test that create_directory_if_not_exists creates directories if they don't exist."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define a subdirectory path
            subdir_path = os.path.join(temp_dir, "subdir")
            
            # Verify the subdirectory doesn't exist
            assert not os.path.exists(subdir_path)
            
            # Create the subdirectory
            create_directory_if_not_exists(subdir_path)
            
            # Verify the subdirectory exists
            assert os.path.exists(subdir_path)
            assert os.path.isdir(subdir_path)
            
            # Call the function again (should not raise an error)
            create_directory_if_not_exists(subdir_path)
            
            # Test with nested directories
            nested_path = os.path.join(temp_dir, "nested1", "nested2", "nested3")
            create_directory_if_not_exists(nested_path)
            assert os.path.exists(nested_path)
            assert os.path.isdir(nested_path)

    def test_remove_file_if_exists(self):
        """Test that remove_file_if_exists removes files if they exist."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        # Verify the file exists
        assert os.path.exists(temp_file_path)
        
        # Remove the file
        remove_file_if_exists(temp_file_path)
        
        # Verify the file doesn't exist
        assert not os.path.exists(temp_file_path)
        
        # Call the function again (should not raise an error)
        remove_file_if_exists(temp_file_path)
        
        # Test with a directory (should not remove)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Verify the directory exists
            assert os.path.exists(temp_dir)
            
            # Try to remove the directory
            remove_file_if_exists(temp_dir)
            
            # Verify the directory still exists
            assert os.path.exists(temp_dir)

    def test_get_file_extension(self):
        """Test that get_file_extension correctly extracts file extensions."""
        # Test file extension extraction
        test_cases = [
            ("file.txt", ".txt"),
            ("file.tar.gz", ".gz"),
            ("file", ""),
            ("/path/to/file.jpg", ".jpg"),
            ("file.name.with.dots.pdf", ".pdf"),
            (".hidden", ".hidden"),
            ("", "")
        ]
        
        for filename, expected_extension in test_cases:
            assert get_file_extension(filename) == expected_extension
        
        # Test with include_dot=False
        assert get_file_extension("file.txt", include_dot=False) == "txt"
        assert get_file_extension("file.tar.gz", include_dot=False) == "gz"
        assert get_file_extension("file", include_dot=False) == ""

    def test_is_image_file(self):
        """Test that is_image_file correctly identifies image files."""
        # Test image file identification
        image_files = [
            "image.jpg",
            "image.jpeg",
            "image.png",
            "image.gif",
            "image.bmp",
            "image.tiff",
            "image.webp",
            "/path/to/image.JPG",  # Case insensitive
            "image.PNG"
        ]
        
        for filename in image_files:
            assert is_image_file(filename), f"Should be identified as an image: {filename}"
        
        # Test non-image files
        non_image_files = [
            "document.pdf",
            "document.txt",
            "document.doc",
            "document.html",
            "archive.zip",
            "file.unknown",
            "image.jpg.txt",  # Misleading extension
            ""
        ]
        
        for filename in non_image_files:
            assert not is_image_file(filename), f"Should not be identified as an image: {filename}"
        
        # Test with custom extensions
        assert is_image_file("image.svg", additional_extensions=[".svg"])
        assert not is_image_file("image.svg")  # Not in default list

    def test_is_text_file(self):
        """Test that is_text_file correctly identifies text files."""
        # Test text file identification
        text_files = [
            "document.txt",
            "document.md",
            "document.json",
            "document.xml",
            "document.html",
            "document.css",
            "document.js",
            "document.py",
            "/path/to/document.TXT",  # Case insensitive
            "document.MD"
        ]
        
        for filename in text_files:
            assert is_text_file(filename), f"Should be identified as a text file: {filename}"
        
        # Test non-text files
        non_text_files = [
            "image.jpg",
            "document.pdf",
            "document.doc",
            "archive.zip",
            "file.unknown",
            "document.txt.jpg",  # Misleading extension
            ""
        ]
        
        for filename in non_text_files:
            assert not is_text_file(filename), f"Should not be identified as a text file: {filename}"
        
        # Test with custom extensions
        assert is_text_file("document.rst", additional_extensions=[".rst"])
        assert not is_text_file("document.rst")  # Not in default list

    def test_get_mime_type(self):
        """Test that get_mime_type correctly determines MIME types."""
        # Test MIME type determination
        test_cases = [
            ("file.txt", "text/plain"),
            ("file.html", "text/html"),
            ("file.jpg", "image/jpeg"),
            ("file.png", "image/png"),
            ("file.pdf", "application/pdf"),
            ("file.json", "application/json"),
            ("file.unknown", "application/octet-stream")  # Default for unknown
        ]
        
        for filename, expected_mime_type in test_cases:
            assert get_mime_type(filename) == expected_mime_type
        
        # Test with a file that actually exists
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            mime_type = get_mime_type(temp_file.name)
            assert mime_type == "text/plain"

    def test_encode_decode_base64(self):
        """Test that encode_base64 and decode_base64 correctly encode and decode data."""
        # Test string encoding and decoding
        original_str = "Hello, world!"
        encoded = encode_base64(original_str)
        
        # Verify the encoded string is base64
        assert isinstance(encoded, str)
        assert encoded != original_str
        
        # Decode and verify
        decoded = decode_base64(encoded)
        assert decoded == original_str
        
        # Test bytes encoding and decoding
        original_bytes = b"Hello, world!"
        encoded = encode_base64(original_bytes)
        
        # Verify the encoded string is base64
        assert isinstance(encoded, str)
        
        # Decode and verify
        decoded = decode_base64(encoded, return_bytes=True)
        assert decoded == original_bytes
        
        # Test with URL-safe encoding
        encoded_url_safe = encode_base64(original_str, url_safe=True)
        decoded_url_safe = decode_base64(encoded_url_safe)
        assert decoded_url_safe == original_str
        
        # Test with invalid base64
        with pytest.raises(Exception):
            decode_base64("not-valid-base64")

    def test_compress_decompress_data(self):
        """Test that compress_data and decompress_data correctly compress and decompress data."""
        # Test string compression and decompression
        original_str = "Hello, world!" * 100  # Make it long enough to benefit from compression
        compressed = compress_data(original_str)
        
        # Verify the compressed data is smaller
        assert len(compressed) < len(original_str.encode())
        
        # Decompress and verify
        decompressed = decompress_data(compressed)
        assert decompressed == original_str
        
        # Test bytes compression and decompression
        original_bytes = b"Hello, world!" * 100
        compressed = compress_data(original_bytes)
        
        # Verify the compressed data is smaller
        assert len(compressed) < len(original_bytes)
        
        # Decompress and verify
        decompressed = decompress_data(compressed, return_bytes=True)
        assert decompressed == original_bytes
        
        # Test with different compression levels
        compressed_high = compress_data(original_str, level=9)
        compressed_low = compress_data(original_str, level=1)
        
        # Higher compression level should result in smaller data (or at least not larger)
        assert len(compressed_high) <= len(compressed_low)
        
        # Both should decompress to the original
        assert decompress_data(compressed_high) == original_str
        assert decompress_data(compressed_low) == original_str

    def test_encrypt_decrypt_data(self):
        """Test that encrypt_data and decrypt_data correctly encrypt and decrypt data."""
        # Generate a key for encryption
        key = "mysecretkey123456"  # 16 bytes for AES-128
        
        # Test string encryption and decryption
        original_str = "Hello, world!"
        encrypted = encrypt_data(original_str, key)
        
        # Verify the encrypted data is different
        assert encrypted != original_str.encode()
        
        # Decrypt and verify
        decrypted = decrypt_data(encrypted, key)
        assert decrypted == original_str
        
        # Test bytes encryption and decryption
        original_bytes = b"Hello, world!"
        encrypted = encrypt_data(original_bytes, key)
        
        # Verify the encrypted data is different
        assert encrypted != original_bytes
        
        # Decrypt and verify
        decrypted = decrypt_data(encrypted, key, return_bytes=True)
        assert decrypted == original_bytes
        
        # Test with wrong key
        wrong_key = "wrongkey12345678"
        with pytest.raises(Exception):
            decrypt_data(encrypted, wrong_key)

    def test_parse_args(self):
        """Test that parse_args correctly parses command line arguments."""
        # Define argument specifications
        arg_specs = [
            {"name": "input", "help": "Input file", "required": True},
            {"name": "output", "help": "Output file", "default": "output.txt"},
            {"name": "verbose", "help": "Verbose mode", "action": "store_true"}
        ]
        
        # Test with required arguments
        with patch("sys.argv", ["script.py", "--input", "input.txt"]):
            args = parse_args(arg_specs)
            assert args.input == "input.txt"
            assert args.output == "output.txt"  # Default value
            assert not args.verbose  # Default is False
        
        # Test with all arguments
        with patch("sys.argv", ["script.py", "--input", "input.txt", "--output", "custom.txt", "--verbose"]):
            args = parse_args(arg_specs)
            assert args.input == "input.txt"
            assert args.output == "custom.txt"
            assert args.verbose
        
        # Test with missing required argument
        with patch("sys.argv", ["script.py"]):
            with pytest.raises(SystemExit):
                parse_args(arg_specs)

    def test_setup_logging(self):
        """Test that setup_logging correctly configures logging."""
        # Test with default configuration
        logger = setup_logging()
        
        # Verify the logger is configured
        assert logger.level == 20  # INFO level
        
        # Test with custom level
        logger = setup_logging(level="DEBUG")
        assert logger.level == 10  # DEBUG level
        
        # Test with custom format
        with patch("logging.Formatter") as mock_formatter:
            custom_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            logger = setup_logging(format_str=custom_format)
            
            # Verify the formatter was created with the custom format
            mock_formatter.assert_called_with(custom_format)
        
        # Test with log file
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            logger = setup_logging(log_file=log_file)
            
            # Verify the log file was created
            assert os.path.exists(log_file)
            
            # Log a message
            logger.info("Test message")
            
            # Verify the message was written to the log file
            with open(log_file, "r") as f:
                log_content = f.read()
                assert "Test message" in log_content

    def test_timer_decorator(self):
        """Test that timer decorator correctly measures function execution time."""
        # Create a mock logger
        mock_logger = MagicMock()
        
        # Define a function with the timer decorator
        @timer(logger=mock_logger)
        def slow_function():
            time.sleep(0.1)
            return "result"
        
        # Call the function
        result = slow_function()
        
        # Verify the function returned the correct result
        assert result == "result"
        
        # Verify the logger was called with timing information
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "slow_function" in log_message
        assert "took" in log_message
        assert "seconds" in log_message

    @pytest.mark.asyncio
    async def test_async_timer_decorator(self):
        """Test that async_timer decorator correctly measures async function execution time."""
        # Create a mock logger
        mock_logger = MagicMock()
        
        # Define an async function with the async_timer decorator
        @async_timer(logger=mock_logger)
        async def slow_async_function():
            await asyncio.sleep(0.1)
            return "async result"
        
        # Call the function
        result = await slow_async_function()
        
        # Verify the function returned the correct result
        assert result == "async result"
        
        # Verify the logger was called with timing information
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "slow_async_function" in log_message
        assert "took" in log_message
        assert "seconds" in log_message

    def test_helpers_with_brave_search_mcp(self):
        """Test integration of helpers with BraveSearchMCP."""
        # This test verifies that helper functions work correctly with BraveSearchMCP
        from apps.mcps.brave_search_mcp import BraveSearchMCP
        
        # Mock the BraveSearchMCP._start_server method to avoid actual server startup
        with patch.object(BraveSearchMCP, '_start_server'):
            # Create a BraveSearchMCP instance
            brave_mcp = BraveSearchMCP(api_key="test_key")
            
            # Mock the process and _send_request method
            brave_mcp.process = MagicMock()
            brave_mcp._send_request = MagicMock(return_value={
                "tools": [{"name": "brave_web_search", "description": "Web search tool"}]
            })
            
            # Test list_tools with retry_with_backoff
            with patch('apps.utils.helpers.retry_with_backoff', side_effect=lambda f, *args, **kwargs: f()) as mock_retry:
                tools = brave_mcp.list_tools()
                
                # Verify retry_with_backoff was used
                mock_retry.assert_called_once()
                
                # Verify the result
                assert tools == [{"name": "brave_web_search", "description": "Web search tool"}]

    def test_helpers_with_filesystem_operations(self):
        """Test integration of helpers with filesystem operations."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test create_directory_if_not_exists
            subdir_path = os.path.join(temp_dir, "subdir")
            create_directory_if_not_exists(subdir_path)
            assert os.path.exists(subdir_path)
            
            # Test save_json_file and read_json_file
            json_path = os.path.join(subdir_path, "test.json")
            test_data = {"key": "value", "number": 42}
            write_json_file(json_path, test_data)
            loaded_data = read_json_file(json_path)
            assert loaded_data == test_data
            
            # Test sanitize_filename
            unsafe_filename = "file/with\\invalid:characters?*|\"<>"
            safe_filename = sanitize_filename(unsafe_filename)
            safe_path = os.path.join(subdir_path, safe_filename)
            
            # Create a file with the safe filename
            with open(safe_path, "w") as f:
                f.write("test content")
            
            # Verify the file exists
            assert os.path.exists(safe_path)
            
            # Test remove_file_if_exists
            remove_file_if_exists(safe_path)
            assert not os.path.exists(safe_path)

    def test_helpers_with_http_requests(self):
        """Test integration of helpers with HTTP requests."""
        # Mock the requests.request function
        with patch("requests.request") as mock_request:
            # Mock a successful response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "test"}
            mock_request.return_value = mock_response
            
            # Test safe_request with retry_with_backoff
            with patch('apps.utils.helpers.retry_with_backoff', side_effect=lambda f, *args, **kwargs: f()) as mock_retry:
                response = safe_request("GET", "https://example.com")
                
                # Verify retry_with_backoff was used
                mock_retry.assert_called_once()
                
                # Verify the response
                assert response.status_code == 200
                assert response.json() == {"data": "test"}
            
            # Test is_valid_url and extract_domain together
            url = "https://example.com/path?query=value"
            assert is_valid_url(url)
            assert extract_domain(url) == "example.com"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
