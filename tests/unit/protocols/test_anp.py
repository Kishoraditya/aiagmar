"""
Unit tests for the Agent Network Protocol (ANP) implementation.
"""

import json
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from apps.protocols.anp import DIDDocument, ServiceDescriptor, NetworkMessage, ANPRegistry, ANPMessenger
from apps.utils.exceptions import ValidationError
from unittest.mock import ANY


class TestDIDDocument:
    """Test suite for DIDDocument class."""

    def test_init_valid(self):
        """Test initialization with valid parameters."""
        service = [
            {
                "id": "did:example:123#service-1",
                "type": "A2A",
                "serviceEndpoint": "https://example.com/a2a"
            }
        ]
        
        doc = DIDDocument(
            id="did:example:123",
            service=service
        )
        
        assert doc.id == "did:example:123"
        assert len(doc.service) == 1
        assert doc.service[0]["id"] == "did:example:123#service-1"
        assert doc.service[0]["type"] == "A2A"
        assert doc.service[0]["serviceEndpoint"] == "https://example.com/a2a"

    def test_init_with_optional_fields(self):
        """Test initialization with optional fields."""
        service = [
            {
                "id": "did:example:123#service-1",
                "type": "A2A",
                "serviceEndpoint": "https://example.com/a2a"
            }
        ]
        
        doc = DIDDocument(
            id="did:example:123",
            service=service,
            controller="did:example:456",
            authentication=["did:example:123#keys-1"],
            verificationMethod=[
                {
                    "id": "did:example:123#keys-1",
                    "type": "Ed25519VerificationKey2020",
                    "controller": "did:example:123",
                    "publicKeyMultibase": "z6MkszZtxCmA2Ce4vUV132PCuLQmwnaDD"
                }
            ]
        )
        
        assert doc.id == "did:example:123"
        assert doc.controller == "did:example:456"
        assert doc.authentication == ["did:example:123#keys-1"]
        assert len(doc.verificationMethod) == 1
        assert doc.verificationMethod[0]["id"] == "did:example:123#keys-1"

    def test_init_invalid_id(self):
        """Test initialization with invalid DID format."""
        service = [
            {
                "id": "service-1",
                "type": "A2A",
                "serviceEndpoint": "https://example.com/a2a"
            }
        ]
        
        with pytest.raises(ValueError, match="Invalid DID format"):
            DIDDocument(
                id="invalid-did",
                service=service
            )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        service = [
            {
                "id": "did:example:123#service-1",
                "type": "A2A",
                "serviceEndpoint": "https://example.com/a2a"
            }
        ]
        
        doc = DIDDocument(
            id="did:example:123",
            service=service,
            controller="did:example:456"
        )
        
        doc_dict = doc.to_dict()
        
        assert doc_dict["id"] == "did:example:123"
        assert doc_dict["controller"] == "did:example:456"
        assert len(doc_dict["service"]) == 1
        assert doc_dict["service"][0]["id"] == "did:example:123#service-1"

    def test_from_dict(self):
        """Test creation from dictionary."""
        doc_dict = {
            "id": "did:example:123",
            "controller": "did:example:456",
            "service": [
                {
                    "id": "did:example:123#service-1",
                    "type": "A2A",
                    "serviceEndpoint": "https://example.com/a2a"
                }
            ]
        }
        
        doc = DIDDocument.from_dict(doc_dict)
        
        assert doc.id == "did:example:123"
        assert doc.controller == "did:example:456"
        assert len(doc.service) == 1
        assert doc.service[0]["id"] == "did:example:123#service-1"

    def test_from_dict_missing_fields(self):
        """Test creation from dictionary with missing required fields."""
        doc_dict = {
            "id": "did:example:123"
            # Missing service
        }
        
        with pytest.raises(ValueError, match="Missing required fields"):
            DIDDocument.from_dict(doc_dict)

    def test_to_json(self):
        """Test conversion to JSON string."""
        service = [
            {
                "id": "did:example:123#service-1",
                "type": "A2A",
                "serviceEndpoint": "https://example.com/a2a"
            }
        ]
        
        doc = DIDDocument(
            id="did:example:123",
            service=service
        )
        
        json_str = doc.to_json()
        
        # Parse back to verify
        parsed = json.loads(json_str)
        assert parsed["id"] == "did:example:123"
        assert len(parsed["service"]) == 1
        assert parsed["service"][0]["id"] == "did:example:123#service-1"

    def test_from_json(self):
        """Test creation from JSON string."""
        json_str = json.dumps({
            "id": "did:example:123",
            "service": [
                {
                    "id": "did:example:123#service-1",
                    "type": "A2A",
                    "serviceEndpoint": "https://example.com/a2a"
                }
            ]
        })
        
        doc = DIDDocument.from_json(json_str)
        
        assert doc.id == "did:example:123"
        assert len(doc.service) == 1
        assert doc.service[0]["id"] == "did:example:123#service-1"

    def test_from_json_invalid(self):
        """Test creation from invalid JSON string."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            DIDDocument.from_json("not-json")

    def test_validate(self):
        """Test validation of DID document."""
        # Valid document
        service = [
            {
                "id": "did:example:123#service-1",
                "type": "A2A",
                "serviceEndpoint": "https://example.com/a2a"
            }
        ]
        
        doc = DIDDocument(
            id="did:example:123",
            service=service
        )
        
        # Should not raise exception
        doc.validate()
        
        # Invalid service endpoint
        doc.service[0]["serviceEndpoint"] = "invalid-url"
        with pytest.raises(ValidationError, match="Invalid service endpoint URL"):
            doc.validate()

    def test_get_service_by_type(self):
        """Test getting service by type."""
        service = [
            {
                "id": "did:example:123#service-1",
                "type": "A2A",
                "serviceEndpoint": "https://example.com/a2a"
            },
            {
                "id": "did:example:123#service-2",
                "type": "ACP",
                "serviceEndpoint": "https://example.com/acp"
            }
        ]
        
        doc = DIDDocument(
            id="did:example:123",
            service=service
        )
        
        a2a_service = doc.get_service_by_type("A2A")
        assert a2a_service["id"] == "did:example:123#service-1"
        assert a2a_service["serviceEndpoint"] == "https://example.com/a2a"
        
        acp_service = doc.get_service_by_type("ACP")
        assert acp_service["id"] == "did:example:123#service-2"
        assert acp_service["serviceEndpoint"] == "https://example.com/acp"
        
        # Non-existent service type
        with pytest.raises(ValueError, match="No service found with type"):
            doc.get_service_by_type("NonExistent")


class TestServiceDescriptor:
    """Test suite for ServiceDescriptor class."""

    def test_init_valid(self):
        """Test initialization with valid parameters."""
        service = ServiceDescriptor(
            id="did:example:123#service-1",
            type="A2A",
            serviceEndpoint="https://example.com/a2a"
        )
        
        assert service.id == "did:example:123#service-1"
        assert service.type == "A2A"
        assert service.serviceEndpoint == "https://example.com/a2a"

    def test_init_invalid_endpoint(self):
        """Test initialization with invalid endpoint URL."""
        with pytest.raises(ValueError, match="Invalid service endpoint URL"):
            ServiceDescriptor(
                id="did:example:123#service-1",
                type="A2A",
                serviceEndpoint="invalid-url"
            )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        service = ServiceDescriptor(
            id="did:example:123#service-1",
            type="A2A",
            serviceEndpoint="https://example.com/a2a"
        )
        
        service_dict = service.to_dict()
        
        assert service_dict["id"] == "did:example:123#service-1"
        assert service_dict["type"] == "A2A"
        assert service_dict["serviceEndpoint"] == "https://example.com/a2a"

    def test_from_dict(self):
        """Test creation from dictionary."""
        service_dict = {
            "id": "did:example:123#service-1",
            "type": "A2A",
            "serviceEndpoint": "https://example.com/a2a"
        }
        
        service = ServiceDescriptor.from_dict(service_dict)
        
        assert service.id == "did:example:123#service-1"
        assert service.type == "A2A"
        assert service.serviceEndpoint == "https://example.com/a2a"

    def test_from_dict_missing_fields(self):
        """Test creation from dictionary with missing required fields."""
        service_dict = {
            "id": "did:example:123#service-1",
            "type": "A2A"
            # Missing serviceEndpoint
        }
        
        with pytest.raises(ValueError, match="Missing required fields"):
            ServiceDescriptor.from_dict(service_dict)


class TestNetworkMessage:
    """Test suite for NetworkMessage class."""

    def test_init_valid(self):
        """Test initialization with valid parameters."""
        message = NetworkMessage(
            context=["https://w3id.org/did/v1"],
            id="msg-123",
            type="Request",
            sender="did:example:123",
            recipient="did:example:456",
            body={"action": "query", "data": "test"}
        )
        
        assert message.context == ["https://w3id.org/did/v1"]
        assert message.id == "msg-123"
        assert message.type == "Request"
        assert message.sender == "did:example:123"
        assert message.recipient == "did:example:456"
        assert message.body["action"] == "query"
        assert message.body["data"] == "test"

    def test_init_invalid_did(self):
        """Test initialization with invalid DID format."""
        with pytest.raises(ValueError, match="Invalid DID format"):
            NetworkMessage(
                context=["https://w3id.org/did/v1"],
                id="msg-123",
                type="Request",
                sender="invalid-did",
                recipient="did:example:456",
                body={}
            )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        message = NetworkMessage(
            context=["https://w3id.org/did/v1"],
            id="msg-123",
            type="Request",
            sender="did:example:123",
            recipient="did:example:456",
            body={"action": "query"}
        )
        
        message_dict = message.to_dict()
        
        assert message_dict["@context"] == ["https://w3id.org/did/v1"]
        assert message_dict["@id"] == "msg-123"
        assert message_dict["@type"] == "Request"
        assert message_dict["sender"] == "did:example:123"
        assert message_dict["recipient"] == "did:example:456"
        assert message_dict["body"]["action"] == "query"

    def test_from_dict(self):
        """Test creation from dictionary."""
        message_dict = {
            "@context": ["https://w3id.org/did/v1"],
            "@id": "msg-123",
            "@type": "Request",
            "sender": "did:example:123",
            "recipient": "did:example:456",
            "body": {"action": "query"}
        }
        
        message = NetworkMessage.from_dict(message_dict)
        
        assert message.context == ["https://w3id.org/did/v1"]
        assert message.id == "msg-123"
        assert message.type == "Request"
        assert message.sender == "did:example:123"
        assert message.recipient == "did:example:456"
        assert message.body["action"] == "query"

    def test_from_dict_missing_fields(self):
        """Test creation from dictionary with missing required fields."""
        message_dict = {
            "@context": ["https://w3id.org/did/v1"],
            "@id": "msg-123",
            "@type": "Request",
            "sender": "did:example:123"
            # Missing recipient and body
        }
        
        with pytest.raises(ValueError, match="Missing required fields"):
            NetworkMessage.from_dict(message_dict)

    def test_to_json(self):
        """Test conversion to JSON string."""
        message = NetworkMessage(
            context=["https://w3id.org/did/v1"],
            id="msg-123",
            type="Request",
            sender="did:example:123",
            recipient="did:example:456",
            body={"action": "query"}
        )
        
        json_str = message.to_json()
        
        # Parse back to verify
        parsed = json.loads(json_str)
        assert parsed["@context"] == ["https://w3id.org/did/v1"]
        assert parsed["@id"] == "msg-123"
        assert parsed["@type"] == "Request"
        assert parsed["sender"] == "did:example:123"
        assert parsed["recipient"] == "did:example:456"
        assert parsed["body"]["action"] == "query"

    def test_from_json(self):
        """Test creation from JSON string."""
        json_str = json.dumps({
            "@context": ["https://w3id.org/did/v1"],
            "@id": "msg-123",
            "@type": "Request",
            "sender": "did:example:123",
            "recipient": "did:example:456",
            "body": {"action": "query"}
        })
        
        message = NetworkMessage.from_json(json_str)
        
        assert message.context == ["https://w3id.org/did/v1"]
        assert message.id == "msg-123"
        assert message.type == "Request"
        assert message.sender == "did:example:123"
        assert message.recipient == "did:example:456"
        assert message.body["action"] == "query"

    def test_from_json_invalid(self):
        """Test creation from invalid JSON string."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            NetworkMessage.from_json("not-json")

    def test_validate(self):
        """Test validation of network message."""
        # Valid message
        message = NetworkMessage(
            context=["https://w3id.org/did/v1"],
            id="msg-123",
            type="Request",
            sender="did:example:123",
            recipient="did:example:456",
            body={"action": "query"}
        )
        
        # Should not raise exception
        message.validate()
        
        # Invalid timestamp
        message.timestamp = "invalid-timestamp"
        with pytest.raises(ValidationError, match="Invalid timestamp format"):
            message.validate()
        
        # Reset timestamp and set invalid nonce
        message.timestamp = None
        message.nonce = "short"
        with pytest.raises(ValidationError, match="Nonce must be at least 8 characters"):
            message.validate()


class TestANPRegistry:
    """Test suite for ANPRegistry class."""

    @pytest.fixture
    def registry(self):
        """Fixture to create an ANPRegistry instance."""
        return ANPRegistry(registry_url="https://example.com/registry")

    @pytest.fixture
    def did_document(self):
        """Fixture to create a sample DID document."""
        service = [
            {
                "id": "did:example:123#service-1",
                "type": "A2A",
                "serviceEndpoint": "https://example.com/a2a"
            }
        ]
        
        return DIDDocument(
            id="did:example:123",
            service=service
        )

    def test_init(self):
        """Test initialization."""
        registry = ANPRegistry(
            registry_url="https://example.com/registry",
            auth_token="test-token"
        )
        
        assert registry.registry_url == "https://example.com/registry"
        assert registry.auth_token == "test-token"

    def test_init_invalid_url(self):
        """Test initialization with invalid registry URL."""
        with pytest.raises(ValueError, match="Invalid registry URL"):
            ANPRegistry(registry_url="invalid-url")

    @pytest.mark.asyncio
    async def test_publish_success(self, registry, did_document):
        """Test publishing a DID document successfully."""
        # Mock the requests.put
        with patch("apps.protocols.anp.requests.put") as mock_put:
            # Create mock response
            mock_response = MagicMock()
            mock_response.status_code = 201
            mock_response.raise_for_status = MagicMock()
            mock_put.return_value = mock_response
            
            # Publish document
            result = await registry.publish(did_document)
            
            # Verify result
            assert result is True
            
            # Verify request was sent correctly
            mock_put.assert_called_once()
            args, kwargs = mock_put.call_args
            assert args[0] == "https://example.com/registry/did:example:123"
            
            # Check JSON payload
            payload = json.loads(kwargs["data"])
            assert payload["id"] == "did:example:123"
            assert len(payload["service"]) == 1
            
            # Check headers
            assert "Content-Type" in kwargs["headers"]
            assert kwargs["headers"]["Content-Type"] == "application/json"
            assert "Authorization" not in kwargs["headers"]

    @pytest.mark.asyncio
    async def test_publish_with_auth(self, did_document):
        """Test publishing with authentication token."""
        # Create registry with auth token
        registry = ANPRegistry(
            registry_url="https://example.com/registry",
            auth_token="test-token"
        )
        
        # Mock the requests.put
        with patch("apps.protocols.anp.requests.put") as mock_put:
            # Create mock response
            mock_response = MagicMock()
            mock_response.status_code = 201
            mock_response.raise_for_status = MagicMock()
            mock_put.return_value = mock_response
            
            # Publish document
            await registry.publish(did_document)
            
            # Verify auth token was included in headers
            args, kwargs = mock_put.call_args
            assert kwargs["headers"]["Authorization"] == "Bearer test-token"

    @pytest.mark.asyncio
    async def test_publish_http_error(self, registry, did_document):
        """Test publishing with HTTP error."""
        # Mock the requests.put
        with patch("apps.protocols.anp.requests.put") as mock_put:
            # Create mock response with error
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = Exception("404 Not Found")
            mock_put.return_value = mock_response
            
            # Publish document and expect error
            with pytest.raises(Exception, match="404 Not Found"):
                await registry.publish(did_document)

    @pytest.mark.asyncio
    async def test_publish_connection_error(self, registry, did_document):
        """Test publishing with connection error."""
        # Mock the requests.put to raise exception
        with patch("apps.protocols.anp.requests.put") as mock_put:
            mock_put.side_effect = Exception("Connection error")
            
            # Publish document and expect error
            with pytest.raises(Exception, match="Connection error"):
                await registry.publish(did_document)

    @pytest.mark.asyncio
    async def test_resolve_success(self, registry):
        """Test resolving a DID document successfully."""
        # Mock the requests.get
        with patch("apps.protocols.anp.requests.get") as mock_get:
            # Create mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "did:example:123",
                "service": [
                    {
                        "id": "did:example:123#service-1",
                        "type": "A2A",
                        "serviceEndpoint": "https://example.com/a2a"
                    }
                ]
            }
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            # Resolve document
            doc = await registry.resolve("did:example:123")
            
            # Verify document
            assert isinstance(doc, DIDDocument)
            assert doc.id == "did:example:123"
            assert len(doc.service) == 1
            assert doc.service[0]["id"] == "did:example:123#service-1"
            
            # Verify request was sent correctly
            mock_get.assert_called_once()
            args, kwargs = mock_get.call_args
            assert args[0] == "https://example.com/registry/did:example:123"
            
            # Check headers
            assert "Authorization" not in kwargs["headers"]

    @pytest.mark.asyncio
    async def test_resolve_with_auth(self):
        """Test resolving with authentication token."""
        # Create registry with auth token
        registry = ANPRegistry(
            registry_url="https://example.com/registry",
            auth_token="test-token"
        )
        
        # Mock the requests.get
        with patch("apps.protocols.anp.requests.get") as mock_get:
            # Create mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "did:example:123",
                "service": [
                    {
                        "id": "did:example:123#service-1",
                        "type": "A2A",
                        "serviceEndpoint": "https://example.com/a2a"
                    }
                ]
            }
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            # Resolve document
            await registry.resolve("did:example:123")
            
            # Verify auth token was included in headers
            args, kwargs = mock_get.call_args
            assert kwargs["headers"]["Authorization"] == "Bearer test-token"

    @pytest.mark.asyncio
    async def test_resolve_http_error(self, registry):
        """Test resolving with HTTP error."""
        # Mock the requests.get
        with patch("apps.protocols.anp.requests.get") as mock_get:
            # Create mock response with error
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = Exception("404 Not Found")
            mock_get.return_value = mock_response
            
            # Resolve document and expect error
            with pytest.raises(Exception, match="404 Not Found"):
                await registry.resolve("did:example:123")

    @pytest.mark.asyncio
    async def test_resolve_connection_error(self, registry):
        """Test resolving with connection error."""
        # Mock the requests.get to raise exception
        with patch("apps.protocols.anp.requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection error")
            
            # Resolve document and expect error
            with pytest.raises(Exception, match="Connection error"):
                await registry.resolve("did:example:123")

    @pytest.mark.asyncio
    async def test_resolve_invalid_response(self, registry):
        """Test resolving with invalid response format."""
        # Mock the requests.get
        with patch("apps.protocols.anp.requests.get") as mock_get:
            # Create mock response with invalid data
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "did:example:123"
                # Missing service field
            }
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            # Resolve document and expect error
            with pytest.raises(ValueError, match="Invalid DID document"):
                await registry.resolve("did:example:123")

    @pytest.mark.asyncio
    async def test_list_documents(self, registry):
        """Test listing all DID documents."""
        # Mock the requests.get
        with patch("apps.protocols.anp.requests.get") as mock_get:
            # Create mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                "did:example:123",
                "did:example:456"
            ]
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            # List documents
            dids = await registry.list_documents()
            
            # Verify result
            assert len(dids) == 2
            assert "did:example:123" in dids
            assert "did:example:456" in dids
            
            # Verify request was sent correctly
            mock_get.assert_called_once()
            args, kwargs = mock_get.call_args
            assert args[0] == "https://example.com/registry"


class TestANPMessenger:
    """Test suite for ANPMessenger class."""

    @pytest.fixture
    def registry(self):
        """Fixture to create a mock ANPRegistry."""
        registry = MagicMock()
        registry.resolve = AsyncMock()
        return registry

    @pytest.fixture
    def messenger(self, registry):
        """Fixture to create an ANPMessenger instance."""
        return ANPMessenger(registry=registry)

    @pytest.fixture
    def network_message(self):
        """Fixture to create a sample network message."""
        return NetworkMessage(
            context=["https://w3id.org/did/v1"],
            id="msg-123",
            type="Request",
            sender="did:example:123",
            recipient="did:example:456",
            body={"action": "query", "data": "test"}
        )

    @pytest.fixture
    def did_document(self):
        """Fixture to create a sample DID document."""
        return DIDDocument(
            id="did:example:456",
            service=[
                {
                    "id": "did:example:456#service-1",
                    "type": "A2A",
                    "serviceEndpoint": "https://example.com/a2a"
                },
                {
                    "id": "did:example:456#service-2",
                    "type": "ACP",
                    "serviceEndpoint": "https://example.com/acp"
                }
            ]
        )

    def test_init(self, registry):
        """Test initialization."""
        messenger = ANPMessenger(
            registry=registry,
            auth_token="test-token"
        )
        
        assert messenger.registry == registry
        assert messenger.auth_token == "test-token"

    @pytest.mark.asyncio
    async def test_send_a2a_success(self, messenger, network_message, did_document):
        """Test sending a message via A2A protocol successfully."""
        # Set message type to A2A
        network_message.type = "A2A"
        
        # Mock registry.resolve to return the DID document
        messenger.registry.resolve.return_value = did_document
        
        # Mock the A2A client
        with patch("apps.protocols.anp.A2AClient") as mock_a2a_client:
            # Create mock client instance
            client_instance = MagicMock()
            mock_a2a_client.return_value = client_instance
            
            # Mock invoke method
            client_instance.invoke = AsyncMock()
            client_instance.invoke.return_value = AsyncMock(__aiter__=AsyncMock(
                return_value=AsyncMock(__anext__=AsyncMock(side_effect=[
                    MagicMock(event="data", data={"result": "success"}),
                    MagicMock(event="end"),
                    StopAsyncIteration
                ]))
            ))
            
            # Send message
            result = await messenger.send(network_message)
            
            # Verify result
            assert result == {"result": "success"}
            
            # Verify registry.resolve was called
            messenger.registry.resolve.assert_called_once_with("did:example:456")
            
            # Verify A2AClient was created with correct endpoint
            mock_a2a_client.assert_called_once()
            args, kwargs = mock_a2a_client.call_args
            assert args[0].endpoint == "https://example.com/a2a"
            
            # Verify invoke was called with correct parameters
            client_instance.invoke.assert_called_once()
            task_request = client_instance.invoke.call_args[0][0]
            assert task_request.payload == {"action": "query", "data": "test"}

    @pytest.mark.asyncio
    async def test_send_acp_success(self, messenger, network_message, did_document):
        """Test sending a message via ACP protocol successfully."""
        # Set message type to ACP
        network_message.type = "ACP"
        
        # Mock registry.resolve to return the DID document
        messenger.registry.resolve.return_value = did_document
        
        # Mock the ACP client
        with patch("apps.protocols.anp.ACPClient") as mock_acp_client:
            # Create mock client instance
            client_instance = MagicMock()
            mock_acp_client.return_value = client_instance
            
            # Mock send method
            client_instance.send = MagicMock(return_value=iter([b'{"status": "success"}']))
            
            # Send message
            # Send message
            result = await messenger.send(network_message)
            
            # Verify result
            assert result == {"status": "success"}
            
            # Verify registry.resolve was called
            messenger.registry.resolve.assert_called_once_with("did:example:456")
            
            # Verify ACPClient was created with correct endpoint
            mock_acp_client.assert_called_once()
            args, kwargs = mock_acp_client.call_args
            assert args[0] == "https://example.com/acp"
            
            # Verify send was called with correct parameters
            client_instance.send.assert_called_once()
            metadata, payload = client_instance.send.call_args[0]
            assert metadata.sender == "did:example:123"
            assert metadata.recipient == "did:example:456"
            assert payload == {"action": "query", "data": "test"}

    @pytest.mark.asyncio
    async def test_send_unsupported_protocol(self, messenger, network_message, did_document):
        """Test sending a message with unsupported protocol."""
        # Set message type to unsupported protocol
        network_message.type = "UnsupportedProtocol"
        
        # Mock registry.resolve to return the DID document
        messenger.registry.resolve.return_value = did_document
        
        # Send message and expect error
        with pytest.raises(ValueError, match="Unsupported message type"):
            await messenger.send(network_message)

    @pytest.mark.asyncio
    async def test_send_missing_service(self, messenger, network_message, did_document):
        """Test sending a message when recipient has no matching service."""
        # Set message type to A2A
        network_message.type = "A2A"
        
        # Remove A2A service from DID document
        did_document.service = [s for s in did_document.service if s["type"] != "A2A"]
        
        # Mock registry.resolve to return the modified DID document
        messenger.registry.resolve.return_value = did_document
        
        # Send message and expect error
        with pytest.raises(ValueError, match="No A2A service found for recipient"):
            await messenger.send(network_message)

    @pytest.mark.asyncio
    async def test_send_resolve_error(self, messenger, network_message):
        """Test sending a message when recipient DID cannot be resolved."""
        # Mock registry.resolve to raise exception
        messenger.registry.resolve.side_effect = Exception("DID not found")
        
        # Send message and expect error
        with pytest.raises(Exception, match="DID not found"):
            await messenger.send(network_message)

    @pytest.mark.asyncio
    async def test_send_with_auth(self, network_message, did_document, registry):
        """Test sending a message with authentication token."""
        # Create messenger with auth token
        messenger = ANPMessenger(
            registry=registry,
            auth_token="test-token"
        )
        
        # Set message type to A2A
        network_message.type = "A2A"
        
        # Mock registry.resolve to return the DID document
        messenger.registry.resolve.return_value = did_document
        
        # Mock the A2A client
        with patch("apps.protocols.anp.A2AClient") as mock_a2a_client:
            # Create mock client instance
            client_instance = MagicMock()
            mock_a2a_client.return_value = client_instance
            
            # Mock invoke method
            client_instance.invoke = AsyncMock()
            client_instance.invoke.return_value = AsyncMock(__aiter__=AsyncMock(
                return_value=AsyncMock(__anext__=AsyncMock(side_effect=[
                    MagicMock(event="data", data={"result": "success"}),
                    MagicMock(event="end"),
                    StopAsyncIteration
                ]))
            ))
            
            # Send message
            await messenger.send(network_message)
            
            # Verify A2AClient was created with auth token
            mock_a2a_client.assert_called_once()
            args, kwargs = mock_a2a_client.call_args
            assert kwargs.get("auth_token") == "test-token"

    @pytest.mark.asyncio
    async def test_receive_a2a(self, messenger):
        """Test receiving a message via A2A protocol."""
        # Mock the A2A server
        with patch("apps.protocols.anp.A2AServer") as mock_a2a_server:
            # Create mock server instance
            server_instance = MagicMock()
            mock_a2a_server.return_value = server_instance
            
            # Create handler function
            async def handler(request):
                return {"result": "processed"}
            
            # Register handler
            await messenger.register_handler("A2A", handler)
            
            # Verify A2AServer was created
            mock_a2a_server.assert_called_once()
            
            # Verify handler was registered
            server_instance.register_handler.assert_called_once()
            registered_handler = server_instance.register_handler.call_args[0][0]
            
            # Test the registered handler
            request = MagicMock()
            result = await registered_handler(request)
            assert result == {"result": "processed"}

    @pytest.mark.asyncio
    async def test_receive_acp(self, messenger):
        """Test receiving a message via ACP protocol."""
        # Mock the ACP server
        with patch("apps.protocols.anp.ACPServer") as mock_acp_server:
            # Create mock server instance
            server_instance = MagicMock()
            mock_acp_server.return_value = server_instance
            
            # Create handler function
            async def handler(metadata, payload, attachments=None):
                return {"result": "processed"}
            
            # Register handler
            await messenger.register_handler("ACP", handler)
            
            # Verify ACPServer was created
            mock_acp_server.assert_called_once()
            
            # Verify handler was registered
            server_instance.register_handler.assert_called_once()
            registered_handler = server_instance.register_handler.call_args[0][1]
            
            # Test the registered handler
            metadata = MagicMock()
            payload = {"action": "query"}
            result = await registered_handler(metadata, payload)
            assert result == {"result": "processed"}

    @pytest.mark.asyncio
    async def test_receive_unsupported_protocol(self, messenger):
        """Test receiving a message with unsupported protocol."""
        # Create handler function
        async def handler(request):
            return {"result": "processed"}
        
        # Register handler for unsupported protocol and expect error
        with pytest.raises(ValueError, match="Unsupported protocol type"):
            await messenger.register_handler("UnsupportedProtocol", handler)

    @pytest.mark.asyncio
    async def test_start_servers(self, messenger):
        """Test starting all registered servers."""
        # Mock the A2A and ACP servers
        with patch("apps.protocols.anp.A2AServer") as mock_a2a_server, \
             patch("apps.protocols.anp.ACPServer") as mock_acp_server:
            
            # Create mock server instances
            a2a_instance = MagicMock()
            acp_instance = MagicMock()
            mock_a2a_server.return_value = a2a_instance
            mock_acp_server.return_value = acp_instance
            
            # Register handlers
            await messenger.register_handler("A2A", AsyncMock())
            await messenger.register_handler("ACP", AsyncMock())
            
            # Start servers
            await messenger.start_servers(host="localhost", port_a2a=8080, port_acp=8081)
            
            # Verify servers were started
            a2a_instance.start.assert_called_once_with(host="localhost", port=8080)
            acp_instance.start.assert_called_once_with(host="localhost", port=8081)

    @pytest.mark.asyncio
    async def test_stop_servers(self, messenger):
        """Test stopping all registered servers."""
        # Mock the A2A and ACP servers
        with patch("apps.protocols.anp.A2AServer") as mock_a2a_server, \
             patch("apps.protocols.anp.ACPServer") as mock_acp_server:
            
            # Create mock server instances
            a2a_instance = MagicMock()
            acp_instance = MagicMock()
            mock_a2a_server.return_value = a2a_instance
            mock_acp_server.return_value = acp_instance
            
            # Register handlers
            await messenger.register_handler("A2A", AsyncMock())
            await messenger.register_handler("ACP", AsyncMock())
            
            # Stop servers
            await messenger.stop_servers()
            
            # Verify servers were stopped
            a2a_instance.stop.assert_called_once()
            acp_instance.stop.assert_called_once()


class TestIntegration:
    """Integration tests for ANP protocol."""

    @pytest.mark.asyncio
    async def test_registry_and_messenger(self):
        """Test integration between ANPRegistry and ANPMessenger."""
        # Create a mock HTTP server for registry
        with patch("apps.protocols.anp.requests.get") as mock_get, \
             patch("apps.protocols.anp.requests.put") as mock_put:
            
            # Mock registry responses
            mock_get_response = MagicMock()
            mock_get_response.status_code = 200
            mock_get_response.json.return_value = {
                "id": "did:example:456",
                "service": [
                    {
                        "id": "did:example:456#service-1",
                        "type": "A2A",
                        "serviceEndpoint": "https://example.com/a2a"
                    }
                ]
            }
            mock_get_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_get_response
            
            mock_put_response = MagicMock()
            mock_put_response.status_code = 201
            mock_put_response.raise_for_status = MagicMock()
            mock_put.return_value = mock_put_response
            
            # Create registry and messenger
            registry = ANPRegistry(registry_url="https://example.com/registry")
            messenger = ANPMessenger(registry=registry)
            
            # Mock the A2A client
            with patch("apps.protocols.anp.A2AClient") as mock_a2a_client:
                # Create mock client instance
                client_instance = MagicMock()
                mock_a2a_client.return_value = client_instance
                
                # Mock invoke method
                client_instance.invoke = AsyncMock()
                client_instance.invoke.return_value = AsyncMock(__aiter__=AsyncMock(
                    return_value=AsyncMock(__anext__=AsyncMock(side_effect=[
                        MagicMock(event="data", data={"result": "success"}),
                        MagicMock(event="end"),
                        StopAsyncIteration
                    ]))
                ))
                
                # Create and publish DID document for sender
                sender_doc = DIDDocument(
                    id="did:example:123",
                    service=[
                        {
                            "id": "did:example:123#service-1",
                            "type": "A2A",
                            "serviceEndpoint": "https://example.com/sender/a2a"
                        }
                    ]
                )
                
                await registry.publish(sender_doc)
                
                # Create and send message
                message = NetworkMessage(
                    context=["https://w3id.org/did/v1"],
                    id="msg-123",
                    type="A2A",
                    sender="did:example:123",
                    recipient="did:example:456",
                    body={"action": "query", "data": "test"}
                )
                
                result = await messenger.send(message)
                
                # Verify result
                assert result == {"result": "success"}
                
                # Verify registry.resolve was called
                mock_get.assert_called_with(
                    "https://example.com/registry/did:example:456",
                    headers={"Accept": "application/json"}
                )
                
                # Verify registry.publish was called
                mock_put.assert_called_with(
                    "https://example.com/registry/did:example:123",
                    data=ANY,
                    headers={"Content-Type": "application/json", "Accept": "application/json"}
                )
                
                # Verify A2AClient was created with correct endpoint
                mock_a2a_client.assert_called_once()
                args, kwargs = mock_a2a_client.call_args
                assert args[0].endpoint == "https://example.com/a2a"
                
                # Verify invoke was called with correct parameters
                client_instance.invoke.assert_called_once()
                task_request = client_instance.invoke.call_args[0][0]
                assert task_request.payload == {"action": "query", "data": "test"}

    @pytest.mark.asyncio
    async def test_message_routing(self):
        """Test message routing between multiple agents."""
        # Create mock registry
        registry = MagicMock()
        
        # Create DID documents for agents
        agent1_doc = DIDDocument(
            id="did:example:agent1",
            service=[
                {
                    "id": "did:example:agent1#service-1",
                    "type": "A2A",
                    "serviceEndpoint": "https://example.com/agent1/a2a"
                }
            ]
        )
        
        agent2_doc = DIDDocument(
            id="did:example:agent2",
            service=[
                {
                    "id": "did:example:agent2#service-1",
                    "type": "A2A",
                    "serviceEndpoint": "https://example.com/agent2/a2a"
                }
            ]
        )
        
        agent3_doc = DIDDocument(
            id="did:example:agent3",
            service=[
                {
                    "id": "did:example:agent3#service-1",
                    "type": "A2A",
                    "serviceEndpoint": "https://example.com/agent3/a2a"
                }
            ]
        )
        
        # Setup registry mock to return appropriate DID documents
        async def mock_resolve(did):
            if did == "did:example:agent1":
                return agent1_doc
            elif did == "did:example:agent2":
                return agent2_doc
            elif did == "did:example:agent3":
                return agent3_doc
            else:
                raise ValueError(f"Unknown DID: {did}")
        
        registry.resolve = mock_resolve
        
        # Create messengers for each agent
        agent1_messenger = ANPMessenger(registry=registry)
        agent2_messenger = ANPMessenger(registry=registry)
        agent3_messenger = ANPMessenger(registry=registry)
        
        # Mock A2A clients for each agent
        with patch("apps.protocols.anp.A2AClient") as mock_a2a_client:
            # Create different mock responses for each agent
            client_instances = {}
            
            for agent_id in ["agent1", "agent2", "agent3"]:
                client_instance = MagicMock()
                client_instance.invoke = AsyncMock()
                client_instance.invoke.return_value = AsyncMock(__aiter__=AsyncMock(
                    return_value=AsyncMock(__anext__=AsyncMock(side_effect=[
                        MagicMock(event="data", data={"from": f"did:example:{agent_id}", "status": "processed"}),
                        MagicMock(event="end"),
                        StopAsyncIteration
                    ]))
                ))
                client_instances[agent_id] = client_instance
            
            # Setup mock to return appropriate client instance based on endpoint
            def get_client_instance(agent_card):
                if "agent1" in agent_card.endpoint:
                    return client_instances["agent1"]
                elif "agent2" in agent_card.endpoint:
                    return client_instances["agent2"]
                elif "agent3" in agent_card.endpoint:
                    return client_instances["agent3"]
            
            mock_a2a_client.side_effect = get_client_instance
            
            # Test message routing: agent1 -> agent2 -> agent3
            
            # Step 1: agent1 sends message to agent2
            message1 = NetworkMessage(
                context=["https://w3id.org/did/v1"],
                id="msg-1",
                type="A2A",
                sender="did:example:agent1",
                recipient="did:example:agent2",
                body={"action": "process", "data": "step1"}
            )
            
            result1 = await agent1_messenger.send(message1)
            assert result1["from"] == "did:example:agent2"
            assert result1["status"] == "processed"
            
            # Verify agent2's client was invoked
            client_instances["agent2"].invoke.assert_called_once()
            task_request = client_instances["agent2"].invoke.call_args[0][0]
            assert task_request.payload["action"] == "process"
            assert task_request.payload["data"] == "step1"
            
            # Reset mock for next test
            client_instances["agent2"].invoke.reset_mock()
            
            # Step 2: agent2 sends message to agent3
            message2 = NetworkMessage(
                context=["https://w3id.org/did/v1"],
                id="msg-2",
                type="A2A",
                sender="did:example:agent2",
                recipient="did:example:agent3",
                body={"action": "process", "data": "step2"}
            )
            
            result2 = await agent2_messenger.send(message2)
            assert result2["from"] == "did:example:agent3"
            assert result2["status"] == "processed"
            
            # Verify agent3's client was invoked
            client_instances["agent3"].invoke.assert_called_once()
            task_request = client_instances["agent3"].invoke.call_args[0][0]
            assert task_request.payload["action"] == "process"
            assert task_request.payload["data"] == "step2"
            
            # Step 3: agent3 sends message back to agent1
            message3 = NetworkMessage(
                context=["https://w3id.org/did/v1"],
                id="msg-3",
                type="A2A",
                sender="did:example:agent3",
                recipient="did:example:agent1",
                body={"action": "process", "data": "step3"}
            )
            
            result3 = await agent3_messenger.send(message3)
            assert result3["from"] == "did:example:agent1"
            assert result3["status"] == "processed"
            
            # Verify agent1's client was invoked
            client_instances["agent1"].invoke.assert_called_once()
            task_request = client_instances["agent1"].invoke.call_args[0][0]
            assert task_request.payload["action"] == "process"
            assert task_request.payload["data"] == "step3"

    @pytest.mark.asyncio
    async def test_did_verification(self):
        """Test DID document verification during message exchange."""
        # Create registry with verification enabled
        registry = MagicMock()
        registry.verify_signatures = True
        
        # Create DID document with verification key
        did_doc = DIDDocument(
            id="did:example:123",
            service=[
                {
                    "id": "did:example:123#service-1",
                    "type": "A2A",
                    "serviceEndpoint": "https://example.com/a2a"
                }
            ],
            verificationMethod=[
                {
                    "id": "did:example:123#keys-1",
                    "type": "Ed25519VerificationKey2020",
                    "controller": "did:example:123",
                    "publicKeyMultibase": "z6MkszZtxCmA2Ce4vUV132PCuLQmwnaDD"
                }
            ],
            authentication=["did:example:123#keys-1"]
        )
        
        # Mock registry.resolve to return the DID document
        registry.resolve = AsyncMock(return_value=did_doc)
        
        # Create messenger
        messenger = ANPMessenger(registry=registry)
        
        # Create signed message
        message = NetworkMessage(
            context=["https://w3id.org/did/v1"],
            id="msg-123",
            type="A2A",
            sender="did:example:123",
            recipient="did:example:456",
            body={"action": "query"},
            proof={
                "type": "Ed25519Signature2020",
                "created": "2023-06-15T12:00:00Z",
                "verificationMethod": "did:example:123#keys-1",
                "proofPurpose": "authentication",
                "proofValue": "z3FXQhSvnJMQyX2KMX7PeQsYQyPpCGy1v1VL8Z2XNP3fNzWJ"
            }
        )
        
        # Mock the signature verification
        with patch("apps.protocols.anp.verify_signature") as mock_verify:
            # Set up verification to succeed
            mock_verify.return_value = True
            
            # Mock the A2A client
            with patch("apps.protocols.anp.A2AClient") as mock_a2a_client:
                # Create mock client instance
                client_instance = MagicMock()
                mock_a2a_client.return_value = client_instance
                
                # Mock invoke method
                client_instance.invoke = AsyncMock()
                client_instance.invoke.return_value = AsyncMock(__aiter__=AsyncMock(
                    return_value=AsyncMock(__anext__=AsyncMock(side_effect=[
                        MagicMock(event="data", data={"result": "success"}),
                        MagicMock(event="end"),
                        StopAsyncIteration
                    ]))
                ))
                
                # Send message
                result = await messenger.send(message)
                
                # Verify result
                assert result == {"result": "success"}
                
                # Verify signature was verified
                mock_verify.assert_called_once()
                args, kwargs = mock_verify.call_args
                assert args[0] == message
                assert args[1] == did_doc
            
            # Reset mocks
            mock_verify.reset_mock()
            
            # Set up verification to fail
            mock_verify.return_value = False
            
            # Send message and expect error
            with pytest.raises(ValueError, match="Invalid message signature"):
                await messenger.send(message)
                
                # Verify signature verification was attempted
                mock_verify.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
