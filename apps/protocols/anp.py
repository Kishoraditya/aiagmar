"""
Agent Network Protocol (ANP) Implementation

Enables open-network discovery and collaboration among agents via decentralized 
identifiers (DIDs) and JSON-LD graphs, suitable for large, multi-cloud settings.
"""

from pydantic import BaseModel, Field, HttpUrl, validator, root_validator, model_validator
from typing import List, Dict, Any, Optional, Union, Callable, Awaitable
import httpx
import json
import asyncio
import uuid
import time
import logging
import base64
import hashlib
import os
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidSignature
from pyld import jsonld
from fastapi import FastAPI, Request


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("anp")

# -----------------------------------------------------------------------------
# Schema Definitions
# -----------------------------------------------------------------------------

class ServiceDescriptor(BaseModel):
    """
    Service endpoint descriptor embedded in a DID document.
    """
    id: str = Field(..., description="Service identifier")
    type: str = Field(..., description="Service type (e.g., 'A2A', 'ACP')")
    serviceEndpoint: HttpUrl = Field(..., description="Service endpoint URL")
    description: Optional[str] = Field(None, description="Human-readable description")
    properties: Optional[Dict[str, Any]] = Field(None, description="Additional properties")


class VerificationMethod(BaseModel):
    """
    Verification method for a DID document (e.g., public key).
    """
    id: str = Field(..., description="Verification method identifier")
    type: str = Field(..., description="Verification method type")
    controller: str = Field(..., description="DID of the controller")
    publicKeyJwk: Optional[Dict[str, Any]] = Field(None, description="Public key in JWK format")
    publicKeyMultibase: Optional[str] = Field(None, description="Public key in multibase format")
    
    @model_validator(mode='after')
    def validate_key_format(self) -> 'VerificationMethod':
        """Ensure at least one key format is provided"""
        if not self.publicKeyJwk and not self.publicKeyMultibase:
            raise ValueError("At least one of publicKeyJwk or publicKeyMultibase must be provided")
        return self


class DIDDocument(BaseModel):
    """
    DID document describing an agent's identity and services.
    """
    id: str = Field(..., description="DID URI (e.g., 'did:example:123')")
    controller: Optional[str] = Field(None, description="DID of the controller")
    verificationMethod: List[VerificationMethod] = Field(default_factory=list, 
                                                        description="Verification methods")
    authentication: List[str] = Field(default_factory=list, 
                                     description="Authentication method references")
    assertionMethod: Optional[List[str]] = Field(None, 
                                                description="Assertion method references")
    service: List[ServiceDescriptor] = Field(default_factory=list, 
                                            description="Service endpoints")
    created: Optional[str] = Field(None, description="Creation timestamp")
    updated: Optional[str] = Field(None, description="Last update timestamp")
    
    @validator('id')
    def validate_did(cls, v):
        """Ensure the ID is a valid DID URI"""
        if not v.startswith('did:'):
            raise ValueError("DID must start with 'did:'")
        parts = v.split(':')
        if len(parts) < 3:
            raise ValueError("DID must have at least 3 parts: did:method:specific-id")
        return v


class NetworkMessage(BaseModel):
    """
    Message sent between agents in the network.
    """
    context: List[str] = Field(..., alias="@context", description="JSON-LD context URIs")
    id: str = Field(..., alias="@id", description="Message ID")
    type: str = Field(..., alias="@type", description="Message type")
    sender: str = Field(..., description="DID URI of the sender")
    recipient: str = Field(..., description="DID URI of the recipient")
    created: str = Field(..., description="Creation timestamp (ISO format)")
    expires: Optional[str] = Field(None, description="Expiration timestamp (ISO format)")
    nonce: str = Field(..., description="Unique nonce to prevent replay attacks")
    body: Dict[str, Any] = Field(..., description="Message payload")
    signature: Optional[Dict[str, Any]] = Field(None, description="Message signature")
    
    class Config:
        allow_population_by_field_name = True
        
    @validator('created', 'expires')
    def validate_timestamp(cls, v, values, **kwargs):
        """Validate timestamp format"""
        if v:
            try:
                import datetime
                datetime.datetime.fromisoformat(v.replace('Z', '+00:00'))
                return v
            except ValueError:
                raise ValueError(f"Invalid timestamp format: {v}")
        return v


# -----------------------------------------------------------------------------
# Registry Implementation
# -----------------------------------------------------------------------------

class ANPRegistry:
    """
    Registry for DID documents, supporting both centralized and decentralized storage.
    """
    
    def __init__(self, registry_url: str, api_key: Optional[str] = None):
        """
        Initialize the ANP registry.
        
        Args:
            registry_url: URL of the registry service
            api_key: Optional API key for authentication
        """
        self.registry_url = registry_url
        self.api_key = api_key
        self.local_cache = {}  # Cache of DID documents
    
    async def publish(self, doc: DIDDocument) -> bool:
        """
        Publish a DID document to the registry.
        
        Args:
            doc: The DID document to publish
            
        Returns:
            True if successful, False otherwise
        """
        # Validate JSON-LD compliance
        try:
            # Add JSON-LD context if not present
            doc_dict = doc.dict(by_alias=True)
            if '@context' not in doc_dict:
                doc_dict['@context'] = [
                    "https://www.w3.org/ns/did/v1",
                    "https://w3id.org/security/suites/jws-2020/v1"
                ]
            
            # Validate with JSON-LD processor
            expanded = jsonld.expand(doc_dict)
            
            # Prepare request
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Send to registry
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.registry_url}/did/{doc.id}",
                    json=doc_dict,
                    headers=headers,
                    timeout=30.0
                )
                
                if response.status_code in (200, 201, 204):
                    # Update local cache
                    self.local_cache[doc.id] = doc
                    return True
                else:
                    logger.error(f"Failed to publish DID document: {response.status_code} {response.text}")
                    return False
                
        except Exception as e:
            logger.exception(f"Error publishing DID document: {e}")
            return False
    
    async def resolve(self, did: str) -> Optional[DIDDocument]:
        """
        Resolve a DID to its DID document.
        
        Args:
            did: The DID to resolve
            
        Returns:
            The DID document if found, None otherwise
        """
        # Check local cache first
        if did in self.local_cache:
            return self.local_cache[did]
        
        try:
            # Prepare request
            headers = {"Accept": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Fetch from registry
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.registry_url}/did/{did}",
                    headers=headers,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    doc_dict = response.json()
                    
                    # Parse JSON-LD
                    expanded = jsonld.expand(doc_dict)
                    
                    # Convert to DIDDocument
                    doc = DIDDocument.parse_obj(doc_dict)
                    
                    # Update local cache
                    self.local_cache[did] = doc
                    
                    return doc
                else:
                    logger.error(f"Failed to resolve DID: {response.status_code} {response.text}")
                    return None
                
        except Exception as e:
            logger.exception(f"Error resolving DID: {e}")
            return None
    
    async def list_dids(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        List DIDs in the registry, optionally filtered.
        
        Args:
            filter_criteria: Optional criteria to filter DIDs
            
        Returns:
            List of DIDs matching the criteria
        """
        try:
            # Prepare request
            headers = {"Accept": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Build query parameters
            params = {}
            if filter_criteria:
                for key, value in filter_criteria.items():
                    params[key] = value
            
            # Fetch from registry
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.registry_url}/dids",
                    params=params,
                    headers=headers,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json().get("dids", [])
                else:
                    logger.error(f"Failed to list DIDs: {response.status_code} {response.text}")
                    return []
                
        except Exception as e:
            logger.exception(f"Error listing DIDs: {e}")
            return []


# -----------------------------------------------------------------------------
# Cryptographic Utilities
# -----------------------------------------------------------------------------

class ANPCrypto:
    """
    Cryptographic utilities for ANP.
    """
    
    @staticmethod
    def generate_key_pair() -> Dict[str, Any]:
        """
        Generate a new RSA key pair.
        
        Returns:
            Dictionary containing private and public keys
        """
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Get public key
        public_key = private_key.public_key()
        
        # Serialize private key to PEM
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Serialize public key to PEM
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Convert to JWK format
        public_numbers = public_key.public_numbers()
        e_bytes = public_numbers.e.to_bytes((public_numbers.e.bit_length() + 7) // 8, byteorder='big')
        n_bytes = public_numbers.n.to_bytes((public_numbers.n.bit_length() + 7) // 8, byteorder='big')
        
        jwk = {
            "kty": "RSA",
            "alg": "RS256",
            "use": "sig",
            "e": base64.urlsafe_b64encode(e_bytes).decode('ascii').rstrip('='),
            "n": base64.urlsafe_b64encode(n_bytes).decode('ascii').rstrip('='),
        }
        
        return {
            "privateKeyPem": private_pem.decode('utf-8'),
            "publicKeyPem": public_pem.decode('utf-8'),
            "publicKeyJwk": jwk
        }
    
    @staticmethod
    def sign_message(message: Dict[str, Any], private_key_pem: str) -> Dict[str, Any]:
        """
        Sign a message using a private key.
        
        Args:
            message: The message to sign
            private_key_pem: Private key in PEM format
            
        Returns:
            The message with signature added
        """
        # Create a copy of the message without the signature field
        message_copy = message.copy()
        if 'signature' in message_copy:
            del message_copy['signature']
        
        # Canonicalize the message
        message_json = json.dumps(message_copy, sort_keys=True, separators=(',', ':'))
        message_bytes = message_json.encode('utf-8')
        
        # Load the private key
        private_key = serialization.load_pem_private_key(
            private_key_pem.encode('utf-8'),
            password=None
        )
        
        # Sign the message
        signature = private_key.sign(
            message_bytes,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        
        # Create signature object
        signature_obj = {
            "type": "RsaSignature2018",
            "created": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "value": base64.b64encode(signature).decode('ascii')
        }
        
        # Add signature to message
        message_copy['signature'] = signature_obj
        
        return message_copy
    
    @staticmethod
    def verify_signature(message: Dict[str, Any], public_key_pem: str) -> bool:
        """
        Verify a message signature.
        
        Args:
            message: The signed message
            public_key_pem: Public key in PEM format
            
        Returns:
            True if signature is valid, False otherwise
        """
        if 'signature' not in message:
            return False
        
        # Extract signature
        signature_obj = message['signature']
        if 'value' not in signature_obj:
            return False
        
        signature = base64.b64decode(signature_obj['value'])
        
        # Create a copy of the message without the signature field
        message_copy = message.copy()
        del message_copy['signature']
        
        # Canonicalize the message
        message_json = json.dumps(message_copy, sort_keys=True, separators=(',', ':'))
        message_bytes = message_json.encode('utf-8')
        
        # Load the public key
        public_key = serialization.load_pem_public_key(
            public_key_pem.encode('utf-8')
        )
        
        # Verify the signature
        try:
            public_key.verify(
                signature,
                message_bytes,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            logger.exception(f"Error verifying signature: {e}")
            return False


# -----------------------------------------------------------------------------
# Agent Identity Management
# -----------------------------------------------------------------------------

class ANPIdentity:
    """
    Manages agent identity, including DID document and keys.
    """
    
    def __init__(self, method: str = "key", registry: Optional[ANPRegistry] = None):
        """
        Initialize agent identity.
        
        Args:
            method: DID method to use (e.g., 'key', 'web', 'peer')
            registry: Optional ANP registry for publishing DIDs
        """
        self.method = method
        self.registry = registry
        self.did_document = None
        self.private_key = None
        self.public_key = None
    
    async def create_identity(self, services: List[ServiceDescriptor] = None) -> DIDDocument:
        """
        Create a new agent identity.
        
        Args:
            services: List of service descriptors to include in the DID document
            
        Returns:
            The created DID document
        """
        # Generate key pair
        keys = ANPCrypto.generate_key_pair()
        self.private_key = keys["privateKeyPem"]
        self.public_key = keys["publicKeyPem"]
        
        # Generate a DID based on the method
        if self.method == "key":
            # Create a DID from the public key
            key_hash = hashlib.sha256(self.public_key.encode()).digest()
            key_id = base64.urlsafe_b64encode(key_hash).decode('ascii').rstrip('=')
            did = f"did:key:{key_id}"
        elif self.method == "web":
            # Use a domain-based DID (would need a domain)
            domain = os.environ.get("ANP_DOMAIN", "example.com")
            did = f"did:web:{domain}"
        else:
            # Generate a random DID
            random_id = str(uuid.uuid4())
            did = f"did:{self.method}:{random_id}"
        
        # Create verification method
        vm_id = f"{did}#keys-1"
        verification_method = VerificationMethod(
            id=vm_id,
            type="RsaVerificationKey2018",
            controller=did,
            publicKeyJwk=keys["publicKeyJwk"]
        )
        
        # Create DID document
        now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        self.did_document = DIDDocument(
            id=did,
            controller=did,
            verificationMethod=[verification_method],
            authentication=[vm_id],
            service=services or [],
            created=now,
            updated=now
        )
        
        # Publish to registry if available
        if self.registry:
            success = await self.registry.publish(self.did_document)
            if not success:
                logger.warning(f"Failed to publish DID document to registry")
        
        return self.did_document
    
    async def update_services(self, services: List[ServiceDescriptor]) -> bool:
        """
        Update the services in the DID document.
        
        Args:
            services: New list of service descriptors
            
        Returns:
            True if successful, False otherwise
        """
        if not self.did_document:
            logger.error("No DID document available")
            return False
        
        # Update services
        self.did_document.service = services
        self.did_document.updated = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        
        # Publish to registry if available
        if self.registry:
            return await self.registry.publish(self.did_document)
        
        return True
    
    def create_message(self, recipient_did: str, message_type: str, body: Dict[str, Any]) -> NetworkMessage:
        """
        Create a signed network message.
        
        Args:
            recipient_did: DID of the recipient
            message_type: Type of message
            body: Message payload
            
        Returns:
            Signed network message
        """
        if not self.did_document:
            raise ValueError("No DID document available")
        
        # Create message
        message_id = f"urn:uuid:{uuid.uuid4()}"
        now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        expires = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() + 3600))  # 1 hour expiry
        
        message_dict = {
            "@context": [
                "https://www.w3.org/ns/did/v1",
                "https://w3id.org/security/suites/jws-2020/v1"
            ],
            "@id": message_id,
            "@type": message_type,
            "sender": self.did_document.id,
            "recipient": recipient_did,
            "created": now,
            "expires": expires,
            "nonce": str(uuid.uuid4()),
            "body": body
        }
        
        # Sign the message
        if self.private_key:
            message_dict = ANPCrypto.sign_message(message_dict, self.private_key)
        
        # Create NetworkMessage object
        return NetworkMessage.parse_obj(message_dict)


# -----------------------------------------------------------------------------
# Messenger Implementation
# -----------------------------------------------------------------------------

class ANPMessenger:
    """
    Handles sending and receiving messages between agents.
    """
    
    def __init__(self, registry: ANPRegistry, identity: Optional[ANPIdentity] = None):
        """
        Initialize the messenger.
        
        Args:
            registry: ANP registry for resolving DIDs
            identity: Optional agent identity
        """
        self.registry = registry
        self.identity = identity
        self.message_handlers = {}  # Type -> handler mapping
        self.received_nonces = set()  # For replay protection
    
    def register_handler(self, message_type: str, handler: Callable[[NetworkMessage], Awaitable[Any]]):
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Async function to handle the message
        """
        self.message_handlers[message_type] = handler
    
    async def send(self, message: NetworkMessage) -> Optional[Dict[str, Any]]:
        """
        Send a message to its recipient.
        
        Args:
            message: The message to send
            
        Returns:
            Response from the recipient, if any
        """
        try:
            # Resolve recipient DID
            recipient_doc = await self.registry.resolve(message.recipient)
            if not recipient_doc:
                logger.error(f"Could not resolve recipient DID: {message.recipient}")
                return None
            
            # Find appropriate service endpoint
            service_endpoint = None
            for service in recipient_doc.service:
                # Choose based on message type or other criteria
                if service.type == "ANP" or service.type == "A2A":
                    service_endpoint = service.serviceEndpoint
                    break
            
            if not service_endpoint:
                logger.error(f"No suitable service endpoint found for {message.recipient}")
                return None
            
            # Send message
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    str(service_endpoint),
                    json=message.dict(by_alias=True),
                    headers={"Content-Type": "application/json"},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to send message: {response.status_code} {response.text}")
                    return None
                
        except Exception as e:
            logger.exception(f"Error sending message: {e}")
            return None
    
    async def receive(self, message_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Receive and process an incoming message.
        
        Args:
            message_dict: The received message as a dictionary
            
        Returns:
            Response to the message, if any
        """
        try:
            # Parse message
            message = NetworkMessage.parse_obj(message_dict)
            
            # Check expiry
            now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            if message.expires and message.expires < now:
                logger.warning(f"Received expired message: {message.id}")
                return {"error": "Message expired"}
            
            # Check for replay
            if message.nonce in self.received_nonces:
                logger.warning(f"Received duplicate message: {message.id}")
                return {"error": "Duplicate message"}
            
            # Add nonce to received set (with cleanup for old nonces)
            self.received_nonces.add(message.nonce)
            if len(self.received_nonces) > 1000:  # Limit size
                self.received_nonces = set(list(self.received_nonces)[-1000:])
            
            # Verify sender
            sender_doc = await self.registry.resolve(message.sender)
            if not sender_doc:
                logger.error(f"Could not resolve sender DID: {message.sender}")
                return {"error": "Unknown sender"}
            
            # Verify signature if present
            if message.signature:
                # Find verification method
                vm = None
                for method in sender_doc.verificationMethod:
                    if method.type == "RsaVerificationKey2018":
                        vm = method
                        break
                
                if not vm or not vm.publicKeyJwk:
                    logger.error(f"No suitable verification method found for {message.sender}")
                    return {"error": "Cannot verify signature"}
                
                # Convert JWK to PEM
                # This is a simplified version - in practice you'd use a library
                # to properly convert JWK to PEM
                public_key_pem = "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----"
                
                # Verify signature
                if not ANPCrypto.verify_signature(message_dict, public_key_pem):
                    logger.error(f"Invalid signature for message: {message.id}")
                    return {"error": "Invalid signature"}
            
            # Handle message based on type
            if message.type in self.message_handlers:
                handler = self.message_handlers[message.type]
                return await handler(message)
            else:
                logger.warning(f"No handler for message type: {message.type}")
                return {"error": f"Unsupported message type: {message.type}"}
            
        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            return {"error": f"Error processing message: {str(e)}"}


# -----------------------------------------------------------------------------
# Discovery Implementation
# -----------------------------------------------------------------------------

class ANPDiscovery:
    """
    Handles discovery of agents and their capabilities.
    """
    
    def __init__(self, registry: ANPRegistry):
        """
        Initialize the discovery service.
        
        Args:
            registry: ANP registry for querying DIDs
        """
        self.registry = registry
    
    async def find_agents_by_service_type(self, service_type: str) -> List[DIDDocument]:
        """
        Find agents that provide a specific service type.
        
        Args:
            service_type: Type of service to look for
            
        Returns:
            List of DID documents for agents providing the service
        """
        # Get all DIDs (in a real implementation, you'd use a more efficient query)
        dids = await self.registry.list_dids()
        
        # Resolve each DID and filter by service type
        results = []
        for did in dids:
            doc = await self.registry.resolve(did)
            if doc:
                for service in doc.service:
                    if service.type == service_type:
                        results.append(doc)
                        break
        
        return results
    
    async def find_agent_by_did(self, did: str) -> Optional[DIDDocument]:
        """
        Find an agent by its DID.
        
        Args:
            did: DID to look for
            
        Returns:
            DID document if found, None otherwise
        """
        return await self.registry.resolve(did)
    
    async def find_agents_by_property(self, property_name: str, property_value: Any) -> List[DIDDocument]:
        """
        Find agents that have a specific property in their service descriptors.
        
        Args:
            property_name: Name of the property to look for
            property_value: Value of the property to match
            
        Returns:
            List of DID documents for agents with the specified property
        """
        # Get all DIDs (in a real implementation, you'd use a more efficient query)
        dids = await self.registry.list_dids()
        
        # Resolve each DID and filter by property
        results = []
        for did in dids:
            doc = await self.registry.resolve(did)
            if doc:
                for service in doc.service:
                    if service.properties and property_name in service.properties:
                        if service.properties[property_name] == property_value:
                            results.append(doc)
                            break
        
        return results


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

async def example_usage():
    """Example usage of the ANP protocol."""
    
    # Create a registry
    registry = ANPRegistry("https://example.com/registry")
    
    # Create agent identities
    agent1 = ANPIdentity(method="example", registry=registry)
    agent2 = ANPIdentity(method="example", registry=registry)
    
    # Define services
    services1 = [
        ServiceDescriptor(
            id="service-1",
            type="A2A",
            serviceEndpoint="https://agent1.example.com/a2a",
            description="Agent 1 A2A endpoint",
            properties={"capabilities": ["text-processing", "translation"]}
        )
    ]
    
    services2 = [
        ServiceDescriptor(
            id="service-1",
            type="A2A",
            serviceEndpoint="https://agent2.example.com/a2a",
            description="Agent 2 A2A endpoint",
            properties={"capabilities": ["image-processing", "ocr"]}
        )
    ]
    
    # Create identities
    doc1 = await agent1.create_identity(services1)
    doc2 = await agent2.create_identity(services2)
    
    print(f"Agent 1 DID: {doc1.id}")
    print(f"Agent 2 DID: {doc2.id}")
    
    # Create messengers
    messenger1 = ANPMessenger(registry, agent1)
    messenger2 = ANPMessenger(registry, agent2)
    
    # Register message handlers
    async def handle_greeting(message: NetworkMessage):
        print(f"Agent 2 received greeting: {message.body.get('greeting')}")
        return {"status": "ok", "message": "Greeting received"}
    
    messenger2.register_handler("greeting", handle_greeting)
    
    # Send a message
    message = agent1.create_message(
        recipient_did=doc2.id,
        message_type="greeting",
        body={"greeting": "Hello from Agent 1!"}
    )
    
    response = await messenger1.send(message)
    print(f"Response from Agent 2: {response}")
    
    # Use discovery to find agents
    discovery = ANPDiscovery(registry)
    
    # Find agents by service type
    a2a_agents = await discovery.find_agents_by_service_type("A2A")
    print(f"Found {len(a2a_agents)} agents supporting A2A")
    
    # Find agents by property
    translation_agents = await discovery.find_agents_by_property("capabilities", "translation")
    print(f"Found {len(translation_agents)} agents supporting translation")


# -----------------------------------------------------------------------------
# FastAPI Server Implementation
# -----------------------------------------------------------------------------

class ANPServer:
    """
    Server implementation for ANP protocol.
    """
    
    def __init__(self, identity: ANPIdentity, messenger: ANPMessenger):
        """
        Initialize the ANP server.
        
        Args:
            identity: Agent identity
            messenger: Message handler
        """
        self.identity = identity
        self.messenger = messenger
        self.app = FastAPI(title=f"ANP Server - {identity.did_document.id}", 
                          version="1.0.0")
        
        # Register routes
        self.setup_routes()
    
    def setup_routes(self):
        """Set up the FastAPI routes"""
        
        @self.app.get("/did.json")
        async def get_did_document():
            """Return this agent's DID document"""
            return self.identity.did_document.dict(by_alias=True)
        
        @self.app.post("/message")
        async def handle_message(request: Request):
            """Handle incoming ANP messages"""
            try:
                message_dict = await request.json()
                response = await self.messenger.receive(message_dict)
                return response or {"status": "processed"}
            except Exception as e:
                logger.exception(f"Error handling message: {e}")
                return {"error": f"Error handling message: {str(e)}"}
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Run the server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    
    async def main():
        # This is a simplified example - in a real application,
        # you would use a proper DID registry and configure your agent
        
        # Create a registry (using a mock URL for demonstration)
        registry = ANPRegistry("https://example.com/registry")
        
        # Create an identity
        identity = ANPIdentity(method="example", registry=registry)
        
        # Define services
        services = [
            ServiceDescriptor(
                id="anp-service",
                type="ANP",
                serviceEndpoint="http://localhost:8000/message",
                description="ANP message endpoint",
                properties={"capabilities": ["text-processing"]}
            )
        ]
        
        # Create identity with services
        did_doc = await identity.create_identity(services)
        print(f"Created DID: {did_doc.id}")
        
        # Create messenger
        messenger = ANPMessenger(registry, identity)
        
        # Register message handlers
        async def handle_text_request(message: NetworkMessage):
            print(f"Received text processing request: {message.body}")
            # Process the text (simplified example)
            text = message.body.get("text", "")
            processed = text.upper()  # Just convert to uppercase as an example
            return {
                "status": "success",
                "result": processed
            }
        
        messenger.register_handler("text-request", handle_text_request)
        
        # Create and run server
        server = ANPServer(identity, messenger)
        server.run()
    
    # Run the main function
    asyncio.run(main())
