"""
Unit tests for the Agent-to-Agent (A2A) protocol implementation.
"""

import json
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from apps.protocols.a2a import AgentCard, TaskRequest, TaskResponseEvent, A2AClient
from apps.utils.exceptions import ValidationError

class TestAgentCard:
    """Test suite for AgentCard class."""

    def test_init_valid(self):
        """Test initialization with valid parameters."""
        agent_card = AgentCard(
            name="test_agent",
            version="1.0.0",
            endpoint="https://example.com/agent",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        )
        
        assert agent_card.name == "test_agent"
        assert agent_card.version == "1.0.0"
        assert agent_card.endpoint == "https://example.com/agent"
        assert "query" in agent_card.schema["properties"]

    def test_init_invalid_endpoint(self):
        """Test initialization with invalid endpoint URL."""
        with pytest.raises(ValueError, match="Invalid endpoint URL"):
            AgentCard(
                name="test_agent",
                version="1.0.0",
                endpoint="not-a-url",
                schema={}
            )

    def test_init_invalid_schema(self):
        """Test initialization with invalid JSON schema."""
        with pytest.raises(ValueError, match="Invalid schema"):
            AgentCard(
                name="test_agent",
                version="1.0.0",
                endpoint="https://example.com/agent",
                schema="not-a-schema"
            )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        agent_card = AgentCard(
            name="test_agent",
            version="1.0.0",
            endpoint="https://example.com/agent",
            schema={"type": "object"}
        )
        
        card_dict = agent_card.to_dict()
        
        assert card_dict["name"] == "test_agent"
        assert card_dict["version"] == "1.0.0"
        assert card_dict["endpoint"] == "https://example.com/agent"
        assert card_dict["schema"] == {"type": "object"}

    def test_from_dict(self):
        """Test creation from dictionary."""
        card_dict = {
            "name": "test_agent",
            "version": "1.0.0",
            "endpoint": "https://example.com/agent",
            "schema": {"type": "object"}
        }
        
        agent_card = AgentCard.from_dict(card_dict)
        
        assert agent_card.name == "test_agent"
        assert agent_card.version == "1.0.0"
        assert agent_card.endpoint == "https://example.com/agent"
        assert agent_card.schema == {"type": "object"}

    def test_from_dict_missing_fields(self):
        """Test creation from dictionary with missing required fields."""
        card_dict = {
            "name": "test_agent",
            "version": "1.0.0"
            # Missing endpoint and schema
        }
        
        with pytest.raises(ValueError, match="Missing required fields"):
            AgentCard.from_dict(card_dict)

    def test_from_json(self):
        """Test creation from JSON string."""
        card_json = json.dumps({
            "name": "test_agent",
            "version": "1.0.0",
            "endpoint": "https://example.com/agent",
            "schema": {"type": "object"}
        })
        
        agent_card = AgentCard.from_json(card_json)
        
        assert agent_card.name == "test_agent"
        assert agent_card.version == "1.0.0"
        assert agent_card.endpoint == "https://example.com/agent"
        assert agent_card.schema == {"type": "object"}

    def test_from_json_invalid(self):
        """Test creation from invalid JSON string."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            AgentCard.from_json("not-json")


class TestTaskRequest:
    """Test suite for TaskRequest class."""

    def test_init_valid(self):
        """Test initialization with valid parameters."""
        task_request = TaskRequest(
            capability="search",
            version="1.0.0",
            payload={"query": "test query"}
        )
        
        assert task_request.capability == "search"
        assert task_request.version == "1.0.0"
        assert task_request.payload == {"query": "test query"}

    def test_init_invalid_payload(self):
        """Test initialization with invalid payload type."""
        with pytest.raises(ValueError, match="Payload must be a dictionary"):
            TaskRequest(
                capability="search",
                version="1.0.0",
                payload="not-a-dict"
            )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        task_request = TaskRequest(
            capability="search",
            version="1.0.0",
            payload={"query": "test query"}
        )
        
        req_dict = task_request.to_dict()
        
        assert req_dict["capability"] == "search"
        assert req_dict["version"] == "1.0.0"
        assert req_dict["payload"] == {"query": "test query"}

    def test_from_dict(self):
        """Test creation from dictionary."""
        req_dict = {
            "capability": "search",
            "version": "1.0.0",
            "payload": {"query": "test query"}
        }
        
        task_request = TaskRequest.from_dict(req_dict)
        
        assert task_request.capability == "search"
        assert task_request.version == "1.0.0"
        assert task_request.payload == {"query": "test query"}

    def test_from_dict_missing_fields(self):
        """Test creation from dictionary with missing required fields."""
        req_dict = {
            "capability": "search",
            # Missing version and payload
        }
        
        with pytest.raises(ValueError, match="Missing required fields"):
            TaskRequest.from_dict(req_dict)

    def test_from_json(self):
        """Test creation from JSON string."""
        req_json = json.dumps({
            "capability": "search",
            "version": "1.0.0",
            "payload": {"query": "test query"}
        })
        
        task_request = TaskRequest.from_json(req_json)
        
        assert task_request.capability == "search"
        assert task_request.version == "1.0.0"
        assert task_request.payload == {"query": "test query"}

    def test_from_json_invalid(self):
        """Test creation from invalid JSON string."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            TaskRequest.from_json("not-json")

    def test_validate_against_schema_valid(self):
        """Test validation against a schema with valid payload."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
        
        task_request = TaskRequest(
            capability="search",
            version="1.0.0",
            payload={"query": "test query"}
        )
        
        # Should not raise an exception
        task_request.validate_against_schema(schema)

    def test_validate_against_schema_invalid(self):
        """Test validation against a schema with invalid payload."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
        
        task_request = TaskRequest(
            capability="search",
            version="1.0.0",
            payload={"not_query": "test"}  # Missing required 'query' field
        )
        
        with pytest.raises(ValidationError, match="Payload does not match schema"):
            task_request.validate_against_schema(schema)


class TestTaskResponseEvent:
    """Test suite for TaskResponseEvent class."""

    def test_init_valid(self):
        """Test initialization with valid parameters."""
        event = TaskResponseEvent(
            event="data",
            data={"result": "test result"}
        )
        
        assert event.event == "data"
        assert event.data == {"result": "test result"}

    def test_init_invalid_event_type(self):
        """Test initialization with invalid event type."""
        with pytest.raises(ValueError, match="Invalid event type"):
            TaskResponseEvent(
                event="invalid",  # Not one of "data", "error", "end"
                data={"result": "test result"}
            )

    def test_data_event(self):
        """Test creating a data event."""
        event = TaskResponseEvent.data({"result": "test result"})
        
        assert event.event == "data"
        assert event.data == {"result": "test result"}

    def test_error_event(self):
        """Test creating an error event."""
        event = TaskResponseEvent.error("Test error", code=400)
        
        assert event.event == "error"
        assert event.data["message"] == "Test error"
        assert event.data["code"] == 400

    def test_end_event(self):
        """Test creating an end event."""
        event = TaskResponseEvent.end()
        
        assert event.event == "end"
        assert event.data is None

    def test_to_sse(self):
        """Test conversion to Server-Sent Event format."""
        event = TaskResponseEvent(
            event="data",
            data={"result": "test result"}
        )
        
        sse = event.to_sse()
        
        assert sse.startswith("event: data\ndata: ")
        assert "result" in sse
        assert "test result" in sse

    def test_from_sse(self):
        """Test creation from Server-Sent Event format."""
        sse = 'event: data\ndata: {"result": "test result"}\n\n'
        
        event = TaskResponseEvent.from_sse(sse)
        
        assert event.event == "data"
        assert event.data["result"] == "test result"

    def test_from_sse_invalid(self):
        """Test creation from invalid Server-Sent Event format."""
        with pytest.raises(ValueError, match="Invalid SSE format"):
            TaskResponseEvent.from_sse("not-sse")


class TestA2AClient:
    """Test suite for A2AClient class."""

    @pytest.fixture
    def agent_card(self):
        """Fixture to create a sample AgentCard."""
        return AgentCard(
            name="test_agent",
            version="1.0.0",
            endpoint="https://example.com/agent",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        )

    @pytest.fixture
    def a2a_client(self, agent_card):
        """Fixture to create an A2AClient instance."""
        return A2AClient(agent_card)

    def test_init(self, agent_card):
        """Test initialization."""
        client = A2AClient(agent_card)
        
        assert client.agent_card == agent_card

    @pytest.mark.asyncio
    async def test_invoke_success(self, a2a_client):
        """Test successful invocation of an agent capability."""
        # Create a task request
        task_request = TaskRequest(
            capability="search",
            version="1.0.0",
            payload={"query": "test query"}
        )
        
        # Mock the httpx.AsyncClient
        with patch("apps.protocols.a2a.httpx.AsyncClient") as mock_client:
            # Create mock response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = AsyncMock()
            
            # Setup mock for aiter_lines to return SSE events
            mock_response.aiter_lines.return_value = [
                'data: {"event": "data", "data": {"partial": "first part"}}',
                'data: {"event": "data", "data": {"partial": "second part"}}',
                'data: {"event": "end", "data": null}'
            ].__aiter__()
            
            # Setup mock client to return mock response
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # Invoke the agent
            events = []
            async for event in a2a_client.invoke(task_request):
                events.append(event)
            
            # Verify the correct request was sent
            mock_client_instance.post.assert_called_once()
            args, kwargs = mock_client_instance.post.call_args
            assert args[0] == "https://example.com/agent"
            assert kwargs["json"] == task_request.to_dict()
            assert kwargs["headers"] == {"Accept": "text/event-stream"}
            
            # Verify events were received correctly
            assert len(events) == 3
            assert events[0].event == "data"
            assert events[0].data["partial"] == "first part"
            assert events[1].event == "data"
            assert events[1].data["partial"] == "second part"
            assert events[2].event == "end"

    @pytest.mark.asyncio
    async def test_invoke_http_error(self, a2a_client):
        """Test invocation with HTTP error."""
        # Create a task request
        task_request = TaskRequest(
            capability="search",
            version="1.0.0",
            payload={"query": "test query"}
        )
        
        # Mock the httpx.AsyncClient
        with patch("apps.protocols.a2a.httpx.AsyncClient") as mock_client:
            # Create mock response with error
            mock_response = AsyncMock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = Exception("404 Not Found")
            
            # Setup mock client to return mock response
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # Invoke the agent and expect error
            with pytest.raises(Exception, match="404 Not Found"):
                async for _ in a2a_client.invoke(task_request):
                    pass

    @pytest.mark.asyncio
    async def test_invoke_invalid_sse(self, a2a_client):
        """Test invocation with invalid SSE response."""
        # Create a task request
        task_request = TaskRequest(
            capability="search",
            version="1.0.0",
            payload={"query": "test query"}
        )
        
        # Mock the httpx.AsyncClient
        with patch("apps.protocols.a2a.httpx.AsyncClient") as mock_client:
            # Create mock response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = AsyncMock()
            # Setup mock for aiter_lines to return invalid SSE events
            mock_response.aiter_lines.return_value = [
                'not a valid SSE line',
                'data: {"event": "end", "data": null}'
            ].__aiter__()
            
            # Setup mock client to return mock response
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # Invoke the agent and expect error
            with pytest.raises(ValueError, match="Invalid SSE format"):
                async for _ in a2a_client.invoke(task_request):
                    pass

    @pytest.mark.asyncio
    async def test_invoke_error_event(self, a2a_client):
        """Test invocation with error event in response."""
        # Create a task request
        task_request = TaskRequest(
            capability="search",
            version="1.0.0",
            payload={"query": "test query"}
        )
        
        # Mock the httpx.AsyncClient
        with patch("apps.protocols.a2a.httpx.AsyncClient") as mock_client:
            # Create mock response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = AsyncMock()
            
            # Setup mock for aiter_lines to return an error event
            mock_response.aiter_lines.return_value = [
                'data: {"event": "error", "data": {"message": "Test error", "code": 400}}',
                'data: {"event": "end", "data": null}'
            ].__aiter__()
            
            # Setup mock client to return mock response
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # Invoke the agent and collect events
            events = []
            async for event in a2a_client.invoke(task_request):
                events.append(event)
            
            # Verify error event was received
            assert len(events) == 2
            assert events[0].event == "error"
            assert events[0].data["message"] == "Test error"
            assert events[0].data["code"] == 400
            assert events[1].event == "end"

    @pytest.mark.asyncio
    async def test_invoke_validate_payload(self, a2a_client):
        """Test payload validation before invocation."""
        # Create a task request with invalid payload
        task_request = TaskRequest(
            capability="search",
            version="1.0.0",
            payload={"not_query": "test"}  # Missing required 'query' field
        )
        
        # Invoke with validate=True and expect validation error
        with pytest.raises(ValidationError, match="Payload does not match schema"):
            async for _ in a2a_client.invoke(task_request, validate=True):
                pass

    @pytest.mark.asyncio
    async def test_invoke_skip_validation(self, a2a_client):
        """Test skipping payload validation."""
        # Create a task request with invalid payload
        task_request = TaskRequest(
            capability="search",
            version="1.0.0",
            payload={"not_query": "test"}  # Missing required 'query' field
        )
        
        # Mock the httpx.AsyncClient
        with patch("apps.protocols.a2a.httpx.AsyncClient") as mock_client:
            # Create mock response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = AsyncMock()
            
            # Setup mock for aiter_lines to return SSE events
            mock_response.aiter_lines.return_value = [
                'data: {"event": "data", "data": {"result": "test result"}}',
                'data: {"event": "end", "data": null}'
            ].__aiter__()
            
            # Setup mock client to return mock response
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # Invoke with validate=False to skip validation
            events = []
            async for event in a2a_client.invoke(task_request, validate=False):
                events.append(event)
            
            # Verify events were received
            assert len(events) == 2
            assert events[0].event == "data"
            assert events[0].data["result"] == "test result"
            assert events[1].event == "end"

    @pytest.mark.asyncio
    async def test_invoke_with_auth_token(self, agent_card):
        """Test invocation with authentication token."""
        # Create client with auth token
        client = A2AClient(agent_card, auth_token="test_token")
        
        # Create a task request
        task_request = TaskRequest(
            capability="search",
            version="1.0.0",
            payload={"query": "test query"}
        )
        
        # Mock the httpx.AsyncClient
        with patch("apps.protocols.a2a.httpx.AsyncClient") as mock_client:
            # Create mock response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = AsyncMock()
            
            # Setup mock for aiter_lines to return SSE events
            mock_response.aiter_lines.return_value = [
                'data: {"event": "end", "data": null}'
            ].__aiter__()
            
            # Setup mock client to return mock response
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # Invoke the agent
            async for _ in client.invoke(task_request):
                pass
            
            # Verify auth token was included in headers
            args, kwargs = mock_client_instance.post.call_args
            assert kwargs["headers"]["Authorization"] == "Bearer test_token"

    @pytest.mark.asyncio
    async def test_invoke_with_timeout(self, a2a_client):
        """Test invocation with custom timeout."""
        # Create a task request
        task_request = TaskRequest(
            capability="search",
            version="1.0.0",
            payload={"query": "test query"}
        )
        
        # Mock the httpx.AsyncClient
        with patch("apps.protocols.a2a.httpx.AsyncClient") as mock_client:
            # Create mock response
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = AsyncMock()
            
            # Setup mock for aiter_lines to return SSE events
            mock_response.aiter_lines.return_value = [
                'data: {"event": "end", "data": null}'
            ].__aiter__()
            
            # Setup mock client to return mock response
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # Invoke the agent with custom timeout
            async for _ in a2a_client.invoke(task_request, timeout=60.0):
                pass
            
            # Verify timeout was set
            args, kwargs = mock_client_instance.post.call_args
            assert kwargs["timeout"] == 60.0

    @pytest.mark.asyncio
    async def test_invoke_connection_error(self, a2a_client):
        """Test invocation with connection error."""
        # Create a task request
        task_request = TaskRequest(
            capability="search",
            version="1.0.0",
            payload={"query": "test query"}
        )
        
        # Mock the httpx.AsyncClient
        with patch("apps.protocols.a2a.httpx.AsyncClient") as mock_client:
            # Setup mock client to raise connection error
            mock_client_instance = AsyncMock()
            mock_client_instance.post.side_effect = Exception("Connection error")
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # Invoke the agent and expect error
            with pytest.raises(Exception, match="Connection error"):
                async for _ in a2a_client.invoke(task_request):
                    pass

    @pytest.mark.asyncio
    async def test_invoke_retry_on_error(self, agent_card):
        """Test automatic retry on transient errors."""
        # Create client with retry settings
        client = A2AClient(
            agent_card,
            max_retries=2,
            retry_delay=0.1
        )
        
        # Create a task request
        task_request = TaskRequest(
            capability="search",
            version="1.0.0",
            payload={"query": "test query"}
        )
        
        # Mock the httpx.AsyncClient
        with patch("apps.protocols.a2a.httpx.AsyncClient") as mock_client:
            # Create mock responses - first fails, second succeeds
            mock_error_response = AsyncMock()
            mock_error_response.status_code = 503
            mock_error_response.raise_for_status.side_effect = Exception("Service Unavailable")
            
            mock_success_response = AsyncMock()
            mock_success_response.status_code = 200
            mock_success_response.raise_for_status = AsyncMock()
            mock_success_response.aiter_lines.return_value = [
                'data: {"event": "data", "data": {"result": "test result"}}',
                'data: {"event": "end", "data": null}'
            ].__aiter__()
            
            # Setup mock client to return error then success
            mock_client_instance = AsyncMock()
            mock_client_instance.post.side_effect = [
                mock_error_response,
                mock_success_response
            ]
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # Invoke the agent
            events = []
            async for event in client.invoke(task_request):
                events.append(event)
            
            # Verify post was called twice (initial + 1 retry)
            assert mock_client_instance.post.call_count == 2
            
            # Verify events from successful retry were received
            assert len(events) == 2
            assert events[0].event == "data"
            assert events[0].data["result"] == "test result"
            assert events[1].event == "end"

    @pytest.mark.asyncio
    async def test_invoke_max_retries_exceeded(self, agent_card):
        """Test when max retries are exceeded."""
        # Create client with retry settings
        client = A2AClient(
            agent_card,
            max_retries=2,
            retry_delay=0.1
        )
        
        # Create a task request
        task_request = TaskRequest(
            capability="search",
            version="1.0.0",
            payload={"query": "test query"}
        )
        
        # Mock the httpx.AsyncClient
        with patch("apps.protocols.a2a.httpx.AsyncClient") as mock_client:
            # Create mock response that always fails
            mock_error_response = AsyncMock()
            mock_error_response.status_code = 503
            mock_error_response.raise_for_status.side_effect = Exception("Service Unavailable")
            
            # Setup mock client to always return error
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_error_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            # Invoke the agent and expect error after retries
            with pytest.raises(Exception, match="Service Unavailable"):
                async for _ in client.invoke(task_request):
                    pass
            
            # Verify post was called 3 times (initial + 2 retries)
            assert mock_client_instance.post.call_count == 3


class TestA2AServer:
    """Test suite for A2A server implementation."""

    @pytest.fixture
    def a2a_server(self):
        """Fixture to create an A2A server instance."""
        from apps.protocols.a2a import A2AServer
        
        server = A2AServer(
            name="test_server",
            version="1.0.0",
            capabilities={
                "search": {
                    "version": "1.0.0",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    },
                    "handler": AsyncMock(return_value={"result": "test result"})
                }
            }
        )
        
        return server

    @pytest.mark.asyncio
    async def test_get_agent_card(self, a2a_server):
        """Test getting the agent card."""
        card = a2a_server.get_agent_card()
        
        assert isinstance(card, AgentCard)
        assert card.name == "test_server"
        assert card.version == "1.0.0"
        assert "search" in card.schema["properties"]["capability"]["enum"]

    @pytest.mark.asyncio
    async def test_handle_request_valid(self, a2a_server):
        """Test handling a valid request."""
        # Create a valid request
        request_data = {
            "capability": "search",
            "version": "1.0.0",
            "payload": {"query": "test query"}
        }
        
        # Handle the request
        response_stream = a2a_server.handle_request(request_data)
        
        # Collect response events
        events = []
        async for event in response_stream:
            events.append(event)
        
        # Verify response events
        assert len(events) == 2
        assert events[0].event == "data"
        assert events[0].data["result"] == "test result"
        assert events[1].event == "end"
        
        # Verify handler was called with correct payload
        search_handler = a2a_server.capabilities["search"]["handler"]
        search_handler.assert_called_once_with({"query": "test query"})

    @pytest.mark.asyncio
    async def test_handle_request_unknown_capability(self, a2a_server):
        """Test handling a request for unknown capability."""
        # Create a request with unknown capability
        request_data = {
            "capability": "unknown",
            "version": "1.0.0",
            "payload": {}
        }
        
        # Handle the request
        response_stream = a2a_server.handle_request(request_data)
        
        # Collect response events
        events = []
        async for event in response_stream:
            events.append(event)
        
        # Verify error response
        assert len(events) == 2
        assert events[0].event == "error"
        assert "Unknown capability" in events[0].data["message"]
        assert events[1].event == "end"

    @pytest.mark.asyncio
    async def test_handle_request_version_mismatch(self, a2a_server):
        """Test handling a request with version mismatch."""
        # Create a request with wrong version
        # Create a request with wrong version
        request_data = {
            "capability": "search",
            "version": "2.0.0",  # Different from server's version
            "payload": {"query": "test query"}
        }
        
        # Handle the request
        response_stream = a2a_server.handle_request(request_data)
        
        # Collect response events
        events = []
        async for event in response_stream:
            events.append(event)
        
        # Verify error response
        assert len(events) == 2
        assert events[0].event == "error"
        assert "Version mismatch" in events[0].data["message"]
        assert events[1].event == "end"

    @pytest.mark.asyncio
    async def test_handle_request_invalid_payload(self, a2a_server):
        """Test handling a request with invalid payload."""
        # Create a request with invalid payload
        request_data = {
            "capability": "search",
            "version": "1.0.0",
            "payload": {}  # Missing required 'query' field
        }
        
        # Handle the request
        response_stream = a2a_server.handle_request(request_data)
        
        # Collect response events
        events = []
        async for event in response_stream:
            events.append(event)
        
        # Verify error response
        assert len(events) == 2
        assert events[0].event == "error"
        assert "Invalid payload" in events[0].data["message"]
        assert events[1].event == "end"

    @pytest.mark.asyncio
    async def test_handle_request_handler_error(self, a2a_server):
        """Test handling a request where handler raises an exception."""
        # Create a valid request
        request_data = {
            "capability": "search",
            "version": "1.0.0",
            "payload": {"query": "test query"}
        }
        
        # Make handler raise an exception
        a2a_server.capabilities["search"]["handler"].side_effect = Exception("Handler error")
        
        # Handle the request
        response_stream = a2a_server.handle_request(request_data)
        
        # Collect response events
        events = []
        async for event in response_stream:
            events.append(event)
        
        # Verify error response
        assert len(events) == 2
        assert events[0].event == "error"
        assert "Handler error" in events[0].data["message"]
        assert events[1].event == "end"

    @pytest.mark.asyncio
    async def test_start_server(self, a2a_server):
        """Test starting the server."""
        # Mock the web server
        with patch("apps.protocols.a2a.web.Application") as mock_app, \
             patch("apps.protocols.a2a.web.run_app") as mock_run_app:
            
            # Start the server
            await a2a_server.start(host="localhost", port=8080)
            
            # Verify web application was created
            mock_app.assert_called_once()
            
            # Verify routes were added
            app_instance = mock_app.return_value
            app_instance.router.add_get.assert_called_once()
            app_instance.router.add_post.assert_called_once()
            
            # Verify server was started
            mock_run_app.assert_called_once_with(
                app_instance,
                host="localhost",
                port=8080
            )

    @pytest.mark.asyncio
    async def test_handle_get_agent_card(self, a2a_server):
        """Test handling GET request for agent card."""
        # Mock the web request
        mock_request = MagicMock()
        
        # Mock the web response
        with patch("apps.protocols.a2a.web.json_response") as mock_json_response:
            # Handle GET request
            await a2a_server._handle_get_agent_card(mock_request)
            
            # Verify response
            mock_json_response.assert_called_once()
            response_data = mock_json_response.call_args[0][0]
            assert response_data["name"] == "test_server"
            assert response_data["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_handle_post_task(self, a2a_server):
        """Test handling POST request for task execution."""
        # Create a valid request body
        request_body = {
            "capability": "search",
            "version": "1.0.0",
            "payload": {"query": "test query"}
        }
        
        # Mock the web request
        mock_request = MagicMock()
        mock_request.json.return_value = asyncio.Future()
        mock_request.json.return_value.set_result(request_body)
        
        # Mock the web response
        with patch("apps.protocols.a2a.web.StreamResponse") as mock_stream_response:
            # Create mock response object
            mock_response = MagicMock()
            mock_stream_response.return_value = mock_response
            
            # Handle POST request
            await a2a_server._handle_post_task(mock_request)
            
            # Verify response setup
            mock_response.headers.__setitem__.assert_called_with(
                "Content-Type", "text/event-stream"
            )
            mock_response.enable_chunked_encoding.assert_called_once()
            mock_response.prepare.assert_called_once_with(mock_request)
            
            # Verify events were written to response
            assert mock_response.write.call_count >= 2  # At least data and end events

    @pytest.mark.asyncio
    async def test_handle_post_task_invalid_json(self, a2a_server):
        """Test handling POST request with invalid JSON."""
        # Mock the web request with invalid JSON
        mock_request = MagicMock()
        mock_request.json.side_effect = ValueError("Invalid JSON")
        
        # Mock the web response
        with patch("apps.protocols.a2a.web.StreamResponse") as mock_stream_response:
            # Create mock response object
            mock_response = MagicMock()
            mock_stream_response.return_value = mock_response
            
            # Handle POST request
            await a2a_server._handle_post_task(mock_request)
            
            # Verify error response
            assert mock_response.write.call_count >= 2  # Error and end events
            
            # Check that error event was written
            error_event_call = mock_response.write.call_args_list[0]
            error_event_data = error_event_call[0][0].decode('utf-8')
            assert "error" in error_event_data
            assert "Invalid JSON" in error_event_data


class TestIntegration:
    """Integration tests for A2A protocol."""

    @pytest.mark.asyncio
    async def test_client_server_interaction(self):
        """Test interaction between A2A client and server."""
        from apps.protocols.a2a import A2AServer
        
        # Create a server with a test capability
        server = A2AServer(
            name="test_server",
            version="1.0.0",
            capabilities={
                "echo": {
                    "version": "1.0.0",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"}
                        },
                        "required": ["message"]
                    },
                    "handler": AsyncMock(side_effect=lambda payload: {"echo": payload["message"]})
                }
            }
        )
        
        # Get the agent card
        agent_card = server.get_agent_card()
        
        # Create a client
        client = A2AClient(agent_card)
        
        # Mock the client's invoke method to directly use server's handle_request
        original_invoke = client.invoke
        
        async def mocked_invoke(task_request, validate=True, timeout=30.0):
            if validate:
                schema = agent_card.schema
                task_request.validate_against_schema(schema)
            
            return server.handle_request(task_request.to_dict())
        
        # Replace client's invoke method
        client.invoke = mocked_invoke
        
        # Create a task request
        task_request = TaskRequest(
            capability="echo",
            version="1.0.0",
            payload={"message": "Hello, world!"}
        )
        
        # Invoke the capability
        events = []
        async for event in client.invoke(task_request):
            events.append(event)
        
        # Verify response events
        assert len(events) == 2
        assert events[0].event == "data"
        assert events[0].data["echo"] == "Hello, world!"
        assert events[1].event == "end"
        
        # Verify handler was called with correct payload
        echo_handler = server.capabilities["echo"]["handler"]
        echo_handler.assert_called_once_with({"message": "Hello, world!"})
        
        # Restore original invoke method
        client.invoke = original_invoke
