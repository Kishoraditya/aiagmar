"""
Unit tests for the Agent Communication Protocol (ACP) implementation.
"""

import json
import pytest
import asyncio
import io
from unittest.mock import patch, MagicMock, AsyncMock
from apps.protocols.acp import EnvelopeMetadata, ACPClient, ACPServer
from apps.utils.exceptions import ValidationError

class TestEnvelopeMetadata:
    """Test suite for EnvelopeMetadata class."""

    def test_init_valid(self):
        """Test initialization with valid parameters."""
        metadata = EnvelopeMetadata(
            message_id="msg-123",
            timestamp="2023-06-15T12:00:00Z",
            sender="agent-1",
            recipient="agent-2",
            content_type="text/plain"
        )
        
        assert metadata.message_id == "msg-123"
        assert metadata.timestamp == "2023-06-15T12:00:00Z"
        assert metadata.sender == "agent-1"
        assert metadata.recipient == "agent-2"
        assert metadata.content_type == "text/plain"

    def test_init_with_optional_fields(self):
        """Test initialization with optional fields."""
        metadata = EnvelopeMetadata(
            message_id="msg-123",
            timestamp="2023-06-15T12:00:00Z",
            sender="agent-1",
            recipient="agent-2",
            content_type="text/plain",
            subject="Test Subject",
            priority="high",
            correlation_id="corr-456"
        )
        
        assert metadata.subject == "Test Subject"
        assert metadata.priority == "high"
        assert metadata.correlation_id == "corr-456"

    def test_init_invalid_content_type(self):
        """Test initialization with invalid content type."""
        with pytest.raises(ValueError, match="Invalid content type"):
            EnvelopeMetadata(
                message_id="msg-123",
                timestamp="2023-06-15T12:00:00Z",
                sender="agent-1",
                recipient="agent-2",
                content_type="invalid/type"
            )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = EnvelopeMetadata(
            message_id="msg-123",
            timestamp="2023-06-15T12:00:00Z",
            sender="agent-1",
            recipient="agent-2",
            content_type="text/plain",
            subject="Test Subject"
        )
        
        metadata_dict = metadata.to_dict()
        
        assert metadata_dict["message_id"] == "msg-123"
        assert metadata_dict["timestamp"] == "2023-06-15T12:00:00Z"
        assert metadata_dict["sender"] == "agent-1"
        assert metadata_dict["recipient"] == "agent-2"
        assert metadata_dict["content_type"] == "text/plain"
        assert metadata_dict["subject"] == "Test Subject"

    def test_from_dict(self):
        """Test creation from dictionary."""
        metadata_dict = {
            "message_id": "msg-123",
            "timestamp": "2023-06-15T12:00:00Z",
            "sender": "agent-1",
            "recipient": "agent-2",
            "content_type": "text/plain",
            "subject": "Test Subject"
        }
        
        metadata = EnvelopeMetadata.from_dict(metadata_dict)
        
        assert metadata.message_id == "msg-123"
        assert metadata.timestamp == "2023-06-15T12:00:00Z"
        assert metadata.sender == "agent-1"
        assert metadata.recipient == "agent-2"
        assert metadata.content_type == "text/plain"
        assert metadata.subject == "Test Subject"

    def test_from_dict_missing_fields(self):
        """Test creation from dictionary with missing required fields."""
        metadata_dict = {
            "message_id": "msg-123",
            "sender": "agent-1",
            # Missing timestamp, recipient, content_type
        }
        
        with pytest.raises(ValueError, match="Missing required fields"):
            EnvelopeMetadata.from_dict(metadata_dict)

    def test_to_json(self):
        """Test conversion to JSON string."""
        metadata = EnvelopeMetadata(
            message_id="msg-123",
            timestamp="2023-06-15T12:00:00Z",
            sender="agent-1",
            recipient="agent-2",
            content_type="text/plain"
        )
        
        json_str = metadata.to_json()
        
        # Parse back to verify
        parsed = json.loads(json_str)
        assert parsed["message_id"] == "msg-123"
        assert parsed["timestamp"] == "2023-06-15T12:00:00Z"
        assert parsed["sender"] == "agent-1"
        assert parsed["recipient"] == "agent-2"
        assert parsed["content_type"] == "text/plain"

    def test_from_json(self):
        """Test creation from JSON string."""
        json_str = json.dumps({
            "message_id": "msg-123",
            "timestamp": "2023-06-15T12:00:00Z",
            "sender": "agent-1",
            "recipient": "agent-2",
            "content_type": "text/plain"
        })
        
        metadata = EnvelopeMetadata.from_json(json_str)
        
        assert metadata.message_id == "msg-123"
        assert metadata.timestamp == "2023-06-15T12:00:00Z"
        assert metadata.sender == "agent-1"
        assert metadata.recipient == "agent-2"
        assert metadata.content_type == "text/plain"

    def test_from_json_invalid(self):
        """Test creation from invalid JSON string."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            EnvelopeMetadata.from_json("not-json")

    def test_validate(self):
        """Test validation of metadata."""
        # Valid metadata
        metadata = EnvelopeMetadata(
            message_id="msg-123",
            timestamp="2023-06-15T12:00:00Z",
            sender="agent-1",
            recipient="agent-2",
            content_type="text/plain"
        )
        
        # Should not raise exception
        metadata.validate()
        
        # Invalid timestamp format
        metadata.timestamp = "invalid-timestamp"
        with pytest.raises(ValidationError, match="Invalid timestamp format"):
            metadata.validate()


class TestACPClient:
    """Test suite for ACPClient class."""

    @pytest.fixture
    def acp_client(self):
        """Fixture to create an ACPClient instance."""
        return ACPClient(endpoint="https://example.com/acp")

    @pytest.fixture
    def metadata(self):
        """Fixture to create sample metadata."""
        return EnvelopeMetadata(
            message_id="msg-123",
            timestamp="2023-06-15T12:00:00Z",
            sender="agent-1",
            recipient="agent-2",
            content_type="text/plain"
        )

    def test_init(self):
        """Test initialization."""
        client = ACPClient(
            endpoint="https://example.com/acp",
            callback_url="https://example.com/callback",
            auth_token="test-token"
        )
        
        assert client.endpoint == "https://example.com/acp"
        assert client.callback_url == "https://example.com/callback"
        assert client.auth_token == "test-token"

    def test_init_invalid_endpoint(self):
        """Test initialization with invalid endpoint URL."""
        with pytest.raises(ValueError, match="Invalid endpoint URL"):
            ACPClient(endpoint="not-a-url")

    def test_init_invalid_callback_url(self):
        """Test initialization with invalid callback URL."""
        with pytest.raises(ValueError, match="Invalid callback URL"):
            ACPClient(
                endpoint="https://example.com/acp",
                callback_url="not-a-url"
            )

    @pytest.mark.asyncio
    async def test_send_text_success(self, acp_client, metadata):
        """Test sending text message successfully."""
        # Mock the requests.post
        with patch("apps.protocols.acp.requests.post") as mock_post:
            # Create mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.iter_content.return_value = [b"response chunk"]
            mock_post.return_value = mock_response
            
            # Send message
            response_chunks = list(acp_client.send(metadata, "Hello, world!"))
            
            # Verify response
            assert response_chunks == [b"response chunk"]
            
            # Verify request was sent correctly
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert args[0] == "https://example.com/acp"
            
            # Check files parameter
            files = kwargs["files"]
            assert "metadata" in files
            assert files["metadata"][0] == "metadata.json"
            assert files["metadata"][2] == "application/json"
            
            assert "text" in files
            assert files["text"][0] == "message.txt"
            assert files["text"][1] == "Hello, world!"
            assert files["text"][2] == "text/plain"
            
            # Check headers
            assert "X-Callback-URL" not in kwargs["headers"]
            assert "Authorization" not in kwargs["headers"]
            
            # Check streaming
            assert kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_send_json_success(self, acp_client, metadata):
        """Test sending JSON message successfully."""
        # Update metadata for JSON
        metadata.content_type = "application/json"
        
        # Mock the requests.post
        with patch("apps.protocols.acp.requests.post") as mock_post:
            # Create mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.iter_content.return_value = [b"response chunk"]
            mock_post.return_value = mock_response
            
            # Send message
            payload = {"key": "value"}
            response_chunks = list(acp_client.send(metadata, payload))
            
            # Verify response
            assert response_chunks == [b"response chunk"]
            
            # Verify request was sent correctly
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            
            # Check files parameter
            files = kwargs["files"]
            assert "metadata" in files
            
            assert "json" in files
            assert files["json"][0] == "message.json"
            assert "key" in json.loads(files["json"][1])
            assert files["json"][2] == "application/json"

    @pytest.mark.asyncio
    async def test_send_with_attachments(self, acp_client, metadata):
        """Test sending message with attachments."""
        # Mock the requests.post
        with patch("apps.protocols.acp.requests.post") as mock_post:
            # Create mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.iter_content.return_value = [b"response chunk"]
            mock_post.return_value = mock_response
            
            # Create attachments
            attachments = {
                "image": (b"image data", "image/png"),
                "document": (b"document data", "application/pdf")
            }
            
            # Send message with attachments
            response_chunks = list(acp_client.send(metadata, "Hello with attachments", attachments))
            
            # Verify response
            assert response_chunks == [b"response chunk"]
            
            # Verify request was sent correctly
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            
            # Check files parameter
            files = kwargs["files"]
            assert "metadata" in files
            assert "text" in files
            
            assert "image" in files
            assert files["image"][0] == "image"
            assert files["image"][1] == b"image data"
            assert files["image"][2] == "image/png"
            
            assert "document" in files
            assert files["document"][0] == "document"
            assert files["document"][1] == b"document data"
            assert files["document"][2] == "application/pdf"

    @pytest.mark.asyncio
    async def test_send_with_callback_url(self, metadata):
        """Test sending message with callback URL."""
        # Create client with callback URL
        client = ACPClient(
            endpoint="https://example.com/acp",
            callback_url="https://example.com/callback"
        )
        
        # Mock the requests.post
        with patch("apps.protocols.acp.requests.post") as mock_post:
            # Create mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.iter_content.return_value = [b"response chunk"]
            mock_post.return_value = mock_response
            
            # Send message
            list(client.send(metadata, "Hello, world!"))
            
            # Verify callback URL was included in headers
            args, kwargs = mock_post.call_args
            assert kwargs["headers"]["X-Callback-URL"] == "https://example.com/callback"

    @pytest.mark.asyncio
    async def test_send_with_auth_token(self, metadata):
        """Test sending message with authentication token."""
        # Create client with auth token
        client = ACPClient(
            endpoint="https://example.com/acp",
            auth_token="test-token"
        )
        
        # Mock the requests.post
        with patch("apps.protocols.acp.requests.post") as mock_post:
            # Create mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.iter_content.return_value = [b"response chunk"]
            mock_post.return_value = mock_response
            
            # Send message
            list(client.send(metadata, "Hello, world!"))
            
            # Verify auth token was included in headers
            args, kwargs = mock_post.call_args
            assert kwargs["headers"]["Authorization"] == "Bearer test-token"

    @pytest.mark.asyncio
    async def test_send_http_error(self, acp_client, metadata):
        """Test sending message with HTTP error."""
        # Mock the requests.post
        with patch("apps.protocols.acp.requests.post") as mock_post:
            # Create mock response with error
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = Exception("404 Not Found")
            mock_post.return_value = mock_response
            
            # Send message and expect error
            with pytest.raises(Exception, match="404 Not Found"):
                list(acp_client.send(metadata, "Hello, world!"))

    @pytest.mark.asyncio
    async def test_send_connection_error(self, acp_client, metadata):
        """Test sending message with connection error."""
        # Mock the requests.post to raise exception
        with patch("apps.protocols.acp.requests.post") as mock_post:
            mock_post.side_effect = Exception("Connection error")
            
            # Send message and expect error
            with pytest.raises(Exception, match="Connection error"):
                list(acp_client.send(metadata, "Hello, world!"))

    @pytest.mark.asyncio
    async def test_send_invalid_payload_type(self, acp_client, metadata):
        """Test sending message with invalid payload type."""
        # Set content type to JSON but provide text
        metadata.content_type = "application/json"
        
        # Send message with incompatible payload and expect error
        with pytest.raises(ValueError, match="Payload must be a dictionary for JSON content type"):
            list(acp_client.send(metadata, "Not a JSON object"))

    @pytest.mark.asyncio
    async def test_send_invalid_attachment_type(self, acp_client, metadata):
        """Test sending message with invalid attachment type."""
        # Create invalid attachment (not bytes)
        attachments = {
            "invalid": ("not bytes", "text/plain")
        }
        
        # Send message with invalid attachment and expect error
        with pytest.raises(ValueError, match="Attachment data must be bytes"):
            list(acp_client.send(metadata, "Hello, world!", attachments))

    @pytest.mark.asyncio
    async def test_send_async(self, acp_client, metadata):
        """Test sending message asynchronously."""
        # Mock the aiohttp.ClientSession
        with patch("apps.protocols.acp.aiohttp.ClientSession") as mock_session:
            # Create mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.content.read.side_effect = [b"chunk1", b"chunk2", None]
            
            # Setup mock session
            session_instance = AsyncMock()
            session_instance.post.return_value.__aenter__.return_value = mock_response
            mock_session.return_value.__aenter__.return_value = session_instance
            
            # Send message asynchronously
            chunks = []
            async for chunk in acp_client.send_async(metadata, "Hello, world!"):
                chunks.append(chunk)
            
            # Verify response chunks
            assert chunks == [b"chunk1", b"chunk2"]
            
            # Verify request was sent correctly
            session_instance.post.assert_called_once()
            args, kwargs = session_instance.post.call_args
            assert args[0] == "https://example.com/acp"
            
            # Check that FormData was used
            assert "data" in kwargs
            assert isinstance(kwargs["data"], MagicMock)  # Mocked FormData

    @pytest.mark.asyncio
    async def test_send_async_http_error(self, acp_client, metadata):
        """Test sending message asynchronously with HTTP error."""
        # Mock the aiohttp.ClientSession
        with patch("apps.protocols.acp.aiohttp.ClientSession") as mock_session:
            # Create mock response with error
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.raise_for_status.side_effect = Exception("404 Not Found")
            
            # Setup mock session
            session_instance = AsyncMock()
            session_instance.post.return_value.__aenter__.return_value = mock_response
            mock_session.return_value.__aenter__.return_value = session_instance
            
            # Send message asynchronously and expect error
            with pytest.raises(Exception, match="404 Not Found"):
                async for _ in acp_client.send_async(metadata, "Hello, world!"):
                    pass

    @pytest.mark.asyncio
    async def test_send_async_with_retry(self, metadata):
        """Test asynchronous sending with automatic retry on transient errors."""
        # Create client with retry settings
        client = ACPClient(
            endpoint="https://example.com/acp",
            max_retries=2,
            retry_delay=0.1
        )
        
        # Mock the aiohttp.ClientSession
        with patch("apps.protocols.acp.aiohttp.ClientSession") as mock_session:
            # Create mock responses - first fails, second succeeds
            mock_error_response = AsyncMock()
            mock_error_response.status = 503
            mock_error_response.raise_for_status.side_effect = Exception("Service Unavailable")
            
            mock_success_response = AsyncMock()
            mock_success_response.status = 200
            mock_success_response.content.read.side_effect = [b"success", None]
            
            # Setup mock session to return error then success
            session_instance = AsyncMock()
            session_instance.post.side_effect = [
                AsyncMock(__aenter__=AsyncMock(return_value=mock_error_response)),
                AsyncMock(__aenter__=AsyncMock(return_value=mock_success_response))
            ]
            mock_session.return_value.__aenter__.return_value = session_instance
            
            # Send message asynchronously
            chunks = []
            async for chunk in client.send_async(metadata, "Hello, world!"):
                chunks.append(chunk)
            
            # Verify response from successful retry
            assert chunks == [b"success"]
            
            # Verify post was called twice (initial + 1 retry)
            assert session_instance.post.call_count == 2


class TestACPServer:
    """Test suite for ACP server implementation."""

    @pytest.fixture
    def acp_server(self):
        """Fixture to create an ACPServer instance."""
        server = ACPServer(
            name="test_server",
            handlers={
                "text/plain": AsyncMock(return_value="Processed text"),
                "application/json": AsyncMock(return_value={"status": "success"})
            }
        )
        return server

    @pytest.fixture
    def metadata(self):
        """Fixture to create sample metadata."""
        return EnvelopeMetadata(
            message_id="msg-123",
            timestamp="2023-06-15T12:00:00Z",
            sender="agent-1",
            recipient="test_server",
            content_type="text/plain"
        )

    def test_init(self):
        """Test initialization."""
        server = ACPServer(
            name="test_server",
            handlers={
                "text/plain": lambda metadata, payload: "Processed text"
            },
            auth_required=True
        )
        
        assert server.name == "test_server"
        assert "text/plain" in server.handlers
        assert server.auth_required is True

    @pytest.mark.asyncio
    async def test_handle_request_text(self, acp_server, metadata):
        """Test handling a text request."""
        # Create mock request with text payload
        mock_request = MagicMock()
        mock_request.files = {
            "metadata": (io.BytesIO(metadata.to_json().encode()), "metadata.json"),
            "text": (io.BytesIO(b"Hello, world!"), "message.txt")
        }
        
        # Handle the request
        response = await acp_server.handle_request(mock_request)
        
        # Verify response
        assert response == "Processed text"
        
        # Verify handler was called with correct parameters
        text_handler = acp_server.handlers["text/plain"]
        text_handler.assert_called_once()
        call_args = text_handler.call_args[0]
        assert isinstance(call_args[0], EnvelopeMetadata)
        assert call_args[0].message_id == metadata.message_id
        assert call_args[1] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_handle_request_json(self, acp_server):
        """Test handling a JSON request."""
        # Create metadata with JSON content type
        metadata = EnvelopeMetadata(
            message_id="msg-123",
            timestamp="2023-06-15T12:00:00Z",
            sender="agent-1",
            recipient="test_server",
            content_type="application/json"
        )
        
        # Create mock request with JSON payload
        mock_request = MagicMock()
        mock_request.files = {
            "metadata": (io.BytesIO(metadata.to_json().encode()), "metadata.json"),
            "json": (io.BytesIO(b'{"key": "value"}'), "message.json")
        }
        
        # Handle the request
        response = await acp_server.handle_request(mock_request)
        
        # Verify response
        assert response == {"status": "success"}
        
        # Verify handler was called with correct parameters
        json_handler = acp_server.handlers["application/json"]
        json_handler.assert_called_once()
        call_args = json_handler.call_args[0]
        assert isinstance(call_args[0], EnvelopeMetadata)
        assert call_args[1] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_handle_request_with_attachments(self, acp_server, metadata):
        """Test handling a request with attachments."""
        # Create mock request with text payload and attachments
        mock_request = MagicMock()
        mock_request.files = {
            "metadata": (io.BytesIO(metadata.to_json().encode()), "metadata.json"),
            "text": (io.BytesIO(b"Hello, world!"), "message.txt"),
            "image": (io.BytesIO(b"image data"), "image.png", "image/png"),
            "document": (io.BytesIO(b"document data"), "doc.pdf", "application/pdf")
        }
        
        # Handle the request
        response = await acp_server.handle_request(mock_request)
        
        # Verify response
        assert response == "Processed text"
        
        # Verify handler was called with correct parameters
        text_handler = acp_server.handlers["text/plain"]
        text_handler.assert_called_once()
        call_args = text_handler.call_args[0]
        
        # Check attachments
        attachments = call_args[2]
        assert "image" in attachments
        assert attachments["image"][0] == b"image data"
        assert attachments["image"][1] == "image/png"
        
        assert "document" in attachments
        assert attachments["document"][0] == b"document data"
        assert attachments["document"][1] == "application/pdf"

    @pytest.mark.asyncio
    async def test_handle_request_missing_metadata(self, acp_server):
        """Test handling a request with missing metadata."""
        # Create mock request without metadata
        mock_request = MagicMock()
        mock_request.files = {
            "text": (io.BytesIO(b"Hello, world!"), "message.txt")
        }
        
        # Handle the request and expect error
        with pytest.raises(ValueError, match="Missing metadata"):
            await acp_server.handle_request(mock_request)

    @pytest.mark.asyncio
    async def test_handle_request_invalid_metadata(self, acp_server):
        """Test handling a request with invalid metadata."""
        # Create mock request with invalid metadata
        mock_request = MagicMock()
        mock_request.files = {
            "metadata": (io.BytesIO(b"not valid json"), "metadata.json"),
            "text": (io.BytesIO(b"Hello, world!"), "message.txt")
        }
        
        # Handle the request and expect error
        with pytest.raises(ValueError, match="Invalid metadata"):
            await acp_server.handle_request(mock_request)

    @pytest.mark.asyncio
    async def test_handle_request_missing_payload(self, acp_server, metadata):
        """Test handling a request with missing payload."""
        # Create mock request without payload
        mock_request = MagicMock()
        mock_request.files = {
            "metadata": (io.BytesIO(metadata.to_json().encode()), "metadata.json")
        }
        
        # Handle the request and expect error
        with pytest.raises(ValueError, match="Missing payload"):
            await acp_server.handle_request(mock_request)

    @pytest.mark.asyncio
    async def test_handle_request_unsupported_content_type(self, acp_server):
        """Test handling a request with unsupported content type."""
        # Create metadata with unsupported content type
        metadata = EnvelopeMetadata(
            message_id="msg-123",
            timestamp="2023-06-15T12:00:00Z",
            sender="agent-1",
            recipient="test_server",
            content_type="application/xml"  # Not supported
        )
        
        # Create mock request
        mock_request = MagicMock()
        mock_request.files = {
            "metadata": (io.BytesIO(metadata.to_json().encode()), "metadata.json"),
            "xml": (io.BytesIO(b"<root>data</root>"), "message.xml")
        }
        
        # Handle the request and expect error
        with pytest.raises(ValueError, match="Unsupported content type"):
            await acp_server.handle_request(mock_request)

    @pytest.mark.asyncio
    async def test_handle_request_wrong_recipient(self, acp_server):
        """Test handling a request with wrong recipient."""
        # Create metadata with wrong recipient
        metadata = EnvelopeMetadata(
            message_id="msg-123",
            timestamp="2023-06-15T12:00:00Z",
            sender="agent-1",
            recipient="wrong_server",  # Not matching server name
            content_type="text/plain"
        )
        
        # Create mock request
        mock_request = MagicMock()
        mock_request.files = {
            "metadata": (io.BytesIO(metadata.to_json().encode()), "metadata.json"),
            "text": (io.BytesIO(b"Hello, world!"), "message.txt")
        }
        
        # Handle the request and expect error
        with pytest.raises(ValueError, match="Wrong recipient"):
            await acp_server.handle_request(mock_request)

    @pytest.mark.asyncio
    async def test_handle_request_auth_required(self):
        """Test handling a request when authentication is required."""
        # Create server requiring authentication
        server = ACPServer(
            name="test_server",
            handlers={"text/plain": AsyncMock()},
            auth_required=True
        )
        
        # Create metadata
        metadata = EnvelopeMetadata(
            message_id="msg-123",
            timestamp="2023-06-15T12:00:00Z",
            sender="agent-1",
            recipient="test_server",
            content_type="text/plain"
        )
        
        # Create mock request without auth header
        mock_request = MagicMock()
        mock_request.files = {
            "metadata": (io.BytesIO(metadata.to_json().encode()), "metadata.json"),
            "text": (io.BytesIO(b"Hello, world!"), "message.txt")
        }
        mock_request.headers = {}
        # Handle the request and expect error
        with pytest.raises(ValueError, match="Authentication required"):
            await server.handle_request(mock_request)

    @pytest.mark.asyncio
    async def test_handle_request_with_auth(self):
        """Test handling a request with valid authentication."""
        # Create server requiring authentication
        server = ACPServer(
            name="test_server",
            handlers={"text/plain": AsyncMock(return_value="Authenticated response")},
            auth_required=True,
            auth_validator=lambda token: token == "valid-token"
        )
        
        # Create metadata
        metadata = EnvelopeMetadata(
            message_id="msg-123",
            timestamp="2023-06-15T12:00:00Z",
            sender="agent-1",
            recipient="test_server",
            content_type="text/plain"
        )
        
        # Create mock request with valid auth header
        mock_request = MagicMock()
        mock_request.files = {
            "metadata": (io.BytesIO(metadata.to_json().encode()), "metadata.json"),
            "text": (io.BytesIO(b"Hello, world!"), "message.txt")
        }
        mock_request.headers = {"Authorization": "Bearer valid-token"}
        
        # Handle the request
        response = await server.handle_request(mock_request)
        
        # Verify response
        assert response == "Authenticated response"

    @pytest.mark.asyncio
    async def test_handle_request_invalid_auth(self):
        """Test handling a request with invalid authentication."""
        # Create server requiring authentication
        server = ACPServer(
            name="test_server",
            handlers={"text/plain": AsyncMock()},
            auth_required=True,
            auth_validator=lambda token: token == "valid-token"
        )
        
        # Create metadata
        metadata = EnvelopeMetadata(
            message_id="msg-123",
            timestamp="2023-06-15T12:00:00Z",
            sender="agent-1",
            recipient="test_server",
            content_type="text/plain"
        )
        
        # Create mock request with invalid auth header
        mock_request = MagicMock()
        mock_request.files = {
            "metadata": (io.BytesIO(metadata.to_json().encode()), "metadata.json"),
            "text": (io.BytesIO(b"Hello, world!"), "message.txt")
        }
        mock_request.headers = {"Authorization": "Bearer invalid-token"}
        
        # Handle the request and expect error
        with pytest.raises(ValueError, match="Invalid authentication token"):
            await server.handle_request(mock_request)

    @pytest.mark.asyncio
    async def test_handle_request_handler_error(self, acp_server, metadata):
        """Test handling a request where handler raises an exception."""
        # Make handler raise an exception
        acp_server.handlers["text/plain"].side_effect = Exception("Handler error")
        
        # Create mock request
        mock_request = MagicMock()
        mock_request.files = {
            "metadata": (io.BytesIO(metadata.to_json().encode()), "metadata.json"),
            "text": (io.BytesIO(b"Hello, world!"), "message.txt")
        }
        
        # Handle the request and expect error
        with pytest.raises(Exception, match="Handler error"):
            await acp_server.handle_request(mock_request)

    @pytest.mark.asyncio
    async def test_start_server(self, acp_server):
        """Test starting the server."""
        # Mock the web server
        with patch("apps.protocols.acp.web.Application") as mock_app, \
             patch("apps.protocols.acp.web.run_app") as mock_run_app:
            
            # Start the server
            await acp_server.start(host="localhost", port=8080)
            
            # Verify web application was created
            mock_app.assert_called_once()
            
            # Verify routes were added
            app_instance = mock_app.return_value
            app_instance.router.add_post.assert_called_once()
            
            # Verify server was started
            mock_run_app.assert_called_once_with(
                app_instance,
                host="localhost",
                port=8080
            )

    @pytest.mark.asyncio
    async def test_handle_post_request(self, acp_server):
        """Test handling HTTP POST request."""
        # Mock the web request
        mock_request = MagicMock()
        
        # Mock the handle_request method
        acp_server.handle_request = AsyncMock(return_value="Processed response")
        
        # Mock the web response
        with patch("apps.protocols.acp.web.Response") as mock_response:
            # Handle POST request
            await acp_server._handle_post(mock_request)
            
            # Verify handle_request was called
            acp_server.handle_request.assert_called_once_with(mock_request)
            
            # Verify response
            mock_response.assert_called_once()
            args, kwargs = mock_response.call_args
            assert args[0] == "Processed response"

    @pytest.mark.asyncio
    async def test_handle_post_request_error(self, acp_server):
        """Test handling HTTP POST request with error."""
        # Mock the web request
        mock_request = MagicMock()
        
        # Mock the handle_request method to raise exception
        acp_server.handle_request = AsyncMock(side_effect=ValueError("Test error"))
        
        # Mock the web response
        with patch("apps.protocols.acp.web.Response") as mock_response:
            # Handle POST request
            await acp_server._handle_post(mock_request)
            
            # Verify error response
            mock_response.assert_called_once()
            args, kwargs = mock_response.call_args
            assert kwargs["status"] == 400
            assert "Test error" in args[0]

    @pytest.mark.asyncio
    async def test_handle_post_request_server_error(self, acp_server):
        """Test handling HTTP POST request with server error."""
        # Mock the web request
        mock_request = MagicMock()
        
        # Mock the handle_request method to raise unexpected exception
        acp_server.handle_request = AsyncMock(side_effect=Exception("Server error"))
        
        # Mock the web response
        with patch("apps.protocols.acp.web.Response") as mock_response:
            # Handle POST request
            await acp_server._handle_post(mock_request)
            
            # Verify error response
            mock_response.assert_called_once()
            args, kwargs = mock_response.call_args
            assert kwargs["status"] == 500
            assert "Server error" in args[0]


class TestIntegration:
    """Integration tests for ACP protocol."""

    @pytest.mark.asyncio
    async def test_client_server_interaction(self):
        """Test interaction between ACP client and server."""
        # Create a server with a test handler
        server = ACPServer(
            name="test_server",
            handlers={
                "text/plain": AsyncMock(return_value="Processed text"),
                "application/json": AsyncMock(return_value={"status": "success"})
            }
        )
        
        # Create a client
        client = ACPClient(endpoint="https://example.com/acp")
        
        # Create metadata
        metadata = EnvelopeMetadata(
            message_id="msg-123",
            timestamp="2023-06-15T12:00:00Z",
            sender="client",
            recipient="test_server",
            content_type="text/plain"
        )
        
        # Mock the client's send method to directly use server's handle_request
        original_send = client.send
        
        def mocked_send(metadata, payload, attachments=None):
            # Create mock request
            mock_request = MagicMock()
            
            # Prepare files
            files = {}
            files["metadata"] = (io.BytesIO(metadata.to_json().encode()), "metadata.json")
            
            if metadata.content_type == "text/plain":
                files["text"] = (io.BytesIO(payload.encode()), "message.txt")
            elif metadata.content_type == "application/json":
                files["json"] = (io.BytesIO(json.dumps(payload).encode()), "message.json")
            
            if attachments:
                for name, (data, mime_type) in attachments.items():
                    files[name] = (io.BytesIO(data), f"{name}", mime_type)
            
            mock_request.files = files
            mock_request.headers = {}
            
            # Call server's handle_request
            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(server.handle_request(mock_request))
            
            # Return response as a single chunk
            if isinstance(response, str):
                yield response.encode()
            else:
                yield json.dumps(response).encode()
        
        # Replace client's send method
        client.send = mocked_send
        
        # Send text message
        response_chunks = list(client.send(metadata, "Hello, world!"))
        
        # Verify response
        assert len(response_chunks) == 1
        assert response_chunks[0] == b"Processed text"
        
        # Verify handler was called with correct parameters
        text_handler = server.handlers["text/plain"]
        text_handler.assert_called_once()
        call_args = text_handler.call_args[0]
        assert isinstance(call_args[0], EnvelopeMetadata)
        assert call_args[1] == "Hello, world!"
        
        # Reset mock
        text_handler.reset_mock()
        
        # Send JSON message
        metadata.content_type = "application/json"
        response_chunks = list(client.send(metadata, {"key": "value"}))
        
        # Verify response
        assert len(response_chunks) == 1
        assert b'"status": "success"' in response_chunks[0]
        
        # Verify handler was called with correct parameters
        json_handler = server.handlers["application/json"]
        json_handler.assert_called_once()
        call_args = json_handler.call_args[0]
        assert isinstance(call_args[0], EnvelopeMetadata)
        assert call_args[1] == {"key": "value"}
        
        # Restore original send method
        client.send = original_send

    @pytest.mark.asyncio
    async def test_client_server_with_attachments(self):
        """Test interaction with attachments between ACP client and server."""
        # Create a server with a test handler that processes attachments
        async def attachment_handler(metadata, payload, attachments):
            # Return information about received attachments
            return {
                "message": payload,
                "attachments": {
                    name: {"size": len(data), "type": mime_type}
                    for name, (data, mime_type) in attachments.items()
                }
            }
        
        server = ACPServer(
            name="test_server",
            handlers={
                "text/plain": attachment_handler
            }
        )
        
        # Create a client
        client = ACPClient(endpoint="https://example.com/acp")
        
        # Create metadata
        metadata = EnvelopeMetadata(
            message_id="msg-123",
            timestamp="2023-06-15T12:00:00Z",
            sender="client",
            recipient="test_server",
            content_type="text/plain"
        )
        
        # Create attachments
        attachments = {
            "image": (b"image data", "image/png"),
            "document": (b"document data", "application/pdf")
        }
        
        # Mock the client's send method to directly use server's handle_request
        original_send = client.send
        
        def mocked_send(metadata, payload, attachments=None):
            # Create mock request
            mock_request = MagicMock()
            
            # Prepare files
            files = {}
            files["metadata"] = (io.BytesIO(metadata.to_json().encode()), "metadata.json")
            
            if metadata.content_type == "text/plain":
                files["text"] = (io.BytesIO(payload.encode()), "message.txt")
            elif metadata.content_type == "application/json":
                files["json"] = (io.BytesIO(json.dumps(payload).encode()), "message.json")
            
            if attachments:
                for name, (data, mime_type) in attachments.items():
                    files[name] = (io.BytesIO(data), f"{name}", mime_type)
            
            mock_request.files = files
            mock_request.headers = {}
            
            # Call server's handle_request
            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(server.handle_request(mock_request))
            
            # Return response as a single chunk
            yield json.dumps(response).encode()
        
        # Replace client's send method
        client.send = mocked_send
        
        # Send message with attachments
        response_chunks = list(client.send(metadata, "Message with attachments", attachments))
        
        # Verify response
        assert len(response_chunks) == 1
        response = json.loads(response_chunks[0])
        
        assert response["message"] == "Message with attachments"
        assert "attachments" in response
        assert "image" in response["attachments"]
        assert response["attachments"]["image"]["size"] == len(b"image data")
        assert response["attachments"]["image"]["type"] == "image/png"
        assert "document" in response["attachments"]
        assert response["attachments"]["document"]["size"] == len(b"document data")
        assert response["attachments"]["document"]["type"] == "application/pdf"
        
        # Restore original send method
        client.send = original_send
