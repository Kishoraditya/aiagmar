"""
Integration tests for protocol integration.

These tests verify that the different agent communication protocols (A2A, ANP, and ACP)
can work together and integrate with MCP services.
"""

import os
import pytest
import json
import tempfile
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Import protocols
from apps.protocols.a2a import AgentCard, TaskRequest, TaskResponseEvent, A2AClient
from apps.protocols.anp import DIDDocument, ServiceDescriptor, NetworkMessage, ANPRegistry, ANPMessenger
from apps.protocols.acp import EnvelopeMetadata, ACPClient

# Import MCPs
from apps.mcps.brave_search_mcp import BraveSearchMCP
from apps.mcps.everart_mcp import EverArtMCP
from apps.mcps.fetch_mcp import FetchMCP
from apps.mcps.filesystem_mcp import FilesystemMCP
from apps.mcps.memory_mcp import MemoryMCP

# Import mocks
from tests.mocks.mock_mcps import (
    MockBraveSearchMCP,
    MockEverArtMCP,
    MockFetchMCP,
    MockFilesystemMCP,
    MockMemoryMCP,
    patch_mcps
)


class TestProtocolIntegration:
    """Test integration between different agent communication protocols."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up test environment before each test."""
        # Create a temporary workspace directory
        self.workspace_dir = str(tmp_path / "workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Set up mock MCPs
        self.brave_search = MockBraveSearchMCP(api_key="mock_brave_api_key")
        self.everart = MockEverArtMCP(api_key="mock_everart_api_key")
        self.fetch = MockFetchMCP()
        self.filesystem = MockFilesystemMCP(workspace_dir=self.workspace_dir)
        self.memory = MockMemoryMCP()
        
        # Configure mock responses
        self._configure_mock_responses()
        
        # Apply patches
        self.patches = patch_mcps()
        
        # Set up mock protocol servers
        self._setup_mock_protocol_servers()
        
        yield
        
        # Clean up
        for p in self.patches:
            p.stop()
    
    def _configure_mock_responses(self):
        """Configure mock responses for the MCPs."""
        # Configure BraveSearchMCP
        self.brave_search.set_web_search_result("""
        Title: Python Programming Language
        Description: Python is a high-level, interpreted programming language known for its readability and versatility.
        URL: https://www.python.org/
        
        Title: Python Tutorial - W3Schools
        Description: Python is a popular programming language. Learn Python with our step-by-step tutorial.
        URL: https://www.w3schools.com/python/
        
        Title: Python (programming language) - Wikipedia
        Description: Python is an interpreted high-level general-purpose programming language.
        URL: https://en.wikipedia.org/wiki/Python_(programming_language)
        """)
        
        # Configure EverArtMCP
        self.everart.set_generate_image_result("""
        Image generated successfully!
        
        URL: https://example.com/images/mock-image-12345.jpg
        
        The image shows a Python logo with code snippets in the background.
        """)
        
        # Configure FetchMCP
        self.fetch.set_fetch_url_result("""
        <html>
        <head><title>Python Programming</title></head>
        <body>
        <h1>Python Programming Language</h1>
        <p>Python is a high-level, interpreted programming language known for its readability and versatility.</p>
        <p>Key features include:</p>
        <ul>
            <li>Easy to learn syntax</li>
            <li>Interpreted nature</li>
            <li>Dynamic typing</li>
            <li>High-level data structures</li>
        </ul>
        </body>
        </html>
        """)
    
    def _setup_mock_protocol_servers(self):
        """Set up mock protocol servers for testing."""
        # Mock A2A server
        self.a2a_server_responses = {
            "research": [
                TaskResponseEvent(event="data", data="Searching for Python programming language..."),
                TaskResponseEvent(event="data", data="Found information about Python at python.org"),
                TaskResponseEvent(event="end", data={"status": "success", "results": "Python is a high-level programming language."})
            ],
            "generate_image": [
                TaskResponseEvent(event="data", data="Generating image for Python..."),
                TaskResponseEvent(event="end", data={"status": "success", "image_url": "https://example.com/images/python.jpg"})
            ]
        }
        
        # Mock ANP registry
        self.anp_registry = {
            "did:example:research": DIDDocument(
                id="did:example:research",
                service=[{
                    "id": "research-service",
                    "type": "A2A",
                    "serviceEndpoint": "https://example.com/research"
                }]
            ),
            "did:example:image": DIDDocument(
                id="did:example:image",
                service=[{
                    "id": "image-service",
                    "type": "A2A",
                    "serviceEndpoint": "https://example.com/image"
                }]
            ),
            "did:example:summary": DIDDocument(
                id="did:example:summary",
                service=[{
                    "id": "summary-service",
                    "type": "ACP",
                    "serviceEndpoint": "https://example.com/summary"
                }]
            )
        }
        
        # Mock ACP server responses
        self.acp_server_responses = {
            "summary": {
                "status": "success",
                "summary": "Python is a high-level programming language known for its readability and versatility."
            },
            "verification": {
                "status": "success",
                "verified": True,
                "facts": ["Python was created by Guido van Rossum", "Python was first released in 1991"]
            }
        }
    
    @pytest.mark.asyncio
    async def test_a2a_with_mcp_integration(self):
        """Test integration between A2A protocol and MCP services."""
        # Create a mock A2A client that uses BraveSearchMCP
        agent_card = AgentCard(
            name="ResearchAgent",
            version="1.0",
            endpoint="https://example.com/research",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        )
        
        # Mock the A2A client's invoke method
        async def mock_invoke(req):
            # Use BraveSearchMCP to perform the actual search
            search_results = self.brave_search.web_search(req.payload["query"])
            
            # Return the results as A2A events
            yield TaskResponseEvent(event="data", data="Searching...")
            yield TaskResponseEvent(event="data", data=f"Found results for {req.payload['query']}")
            yield TaskResponseEvent(event="end", data={"status": "success", "results": search_results})
        
        with patch.object(A2AClient, 'invoke', side_effect=mock_invoke):
            client = A2AClient(agent_card)
            
            # Create a task request
            task_request = TaskRequest(
                capability="search",
                version="1.0",
                payload={"query": "Python programming language"}
            )
            
            # Invoke the task
            results = []
            async for event in client.invoke(task_request):
                results.append(event)
            
            # Verify BraveSearchMCP was called
            assert self.brave_search.web_search_called
            assert "Python programming language" in self.brave_search.web_search_args[0]
            
            # Verify A2A events were returned correctly
            assert len(results) >= 3
            assert results[0].event == "data"
            assert results[-1].event == "end"
            assert "success" in results[-1].data["status"]
            assert "python" in results[-1].data["results"].lower()
    
    @pytest.mark.asyncio
    async def test_anp_with_a2a_integration(self):
        """Test integration between ANP and A2A protocols."""
        # Create a mock ANP registry
        registry = ANPRegistry("https://example.com/registry")
        
        # Mock the resolve method
        def mock_resolve(did):
            return self.anp_registry.get(did)
        
        with patch.object(ANPRegistry, 'resolve', side_effect=mock_resolve):
            # Create a mock ANP messenger
            messenger = ANPMessenger(registry)
            
            # Mock the send method to use A2A
            async def mock_send(msg):
                # Get the DID document for the recipient
                doc = registry.resolve(msg.recipient)
                
                # Find the A2A service endpoint
                service = next((s for s in doc.service if s["type"] == "A2A"), None)
                assert service is not None, f"No A2A service found for {msg.recipient}"
                
                # Create an A2A client for the service
                agent_card = AgentCard(
                    name=doc.id,
                    version="1.0",
                    endpoint=service["serviceEndpoint"],
                    schema={}
                )
                client = A2AClient(agent_card)
                
                # Create a task request from the message body
                task_request = TaskRequest(
                    capability=msg.body.get("capability", "default"),
                    version=msg.body.get("version", "1.0"),
                    payload=msg.body.get("payload", {})
                )
                
                # Mock the invoke method to return predefined responses
                async def mock_invoke(req):
                    capability = req.capability
                    if capability == "research":
                        for event in self.a2a_server_responses["research"]:
                            yield event
                    elif capability == "generate_image":
                        for event in self.a2a_server_responses["generate_image"]:
                            yield event
                    else:
                        yield TaskResponseEvent(event="error", data={"message": f"Unknown capability: {capability}"})
                
                with patch.object(A2AClient, 'invoke', side_effect=mock_invoke):
                    # Invoke the task and collect results
                    results = []
                    async for event in client.invoke(task_request):
                        results.append(event)
                    
                    return results
            
            with patch.object(ANPMessenger, 'send', side_effect=mock_send):
                # Create a network message
                network_message = NetworkMessage(
                    **{
                        "@context": ["https://example.com/context"],
                        "@id": "msg-123",
                        "@type": "ResearchRequest",
                        "sender": "did:example:sender",
                        "recipient": "did:example:research",
                        "body": {
                            "capability": "research",
                            "version": "1.0",
                            "payload": {"query": "Python programming language"}
                        }
                    }
                )
                
                # Send the message
                results = await messenger.send(network_message)
                
                # Verify results
                assert len(results) >= 3
                assert results[0].event == "data"
                assert results[-1].event == "end"
                assert "success" in results[-1].data["status"]
                assert "python" in results[-1].data["results"].lower()
    
    def test_acp_with_mcp_integration(self):
        """Test integration between ACP protocol and MCP services."""
        # Create a mock ACP client
        client = ACPClient("https://example.com/summary")
        
        # Create envelope metadata
        metadata = EnvelopeMetadata(
            message_id="msg-123",
            timestamp="2023-06-01T12:00:00Z",
            sender="agent-1",
            recipient="summary-agent",
            content_type="application/json"
        )
        
        # Mock the send method to use MemoryMCP
        def mock_send(metadata, payload, attachments={}):
            # Store the request in memory
            self.memory.store_memory(
                f"request_{metadata.message_id}",
                json.dumps({
                    "metadata": metadata.dict(),
                    "payload": payload,
                    "attachments": {k: "binary_data" for k in attachments}
                })
            )
            
            # Process the request based on recipient
            if metadata.recipient == "summary-agent":
                # Use the payload to generate a summary
                if isinstance(payload, dict) and "text" in payload:
                    summary = f"Summary of: {payload['text'][:50]}..."
                    return self.acp_server_responses["summary"]
                else:
                    return {"status": "error", "message": "Invalid payload"}
            
            elif metadata.recipient == "verification-agent":
                # Use BraveSearchMCP to verify facts
                if isinstance(payload, dict) and "facts" in payload:
                    for fact in payload["facts"]:
                        self.brave_search.web_search(fact)
                    return self.acp_server_responses["verification"]
                else:
                    return {"status": "error", "message": "Invalid payload"}
            
            else:
                return {"status": "error", "message": f"Unknown recipient: {metadata.recipient}"}
        
        with patch.object(ACPClient, 'send', side_effect=mock_send):
            # Send a request to the summary agent
            payload = {
                "text": "Python is a high-level, interpreted programming language known for its readability and versatility."
            }
            
            response = client.send(metadata, payload)
            
            # Verify the request was stored in memory
            assert self.memory.store_memory_called
            stored_request = json.loads(self.memory.retrieve_memory(f"request_{metadata.message_id}"))
            assert stored_request["metadata"]["message_id"] == metadata.message_id
            assert stored_request["payload"]["text"] == payload["text"]
            
            # Verify response
            assert response["status"] == "success"
            assert "summary" in response
    
    def test_acp_with_brave_search_mcp_integration(self):
        """Test integration between ACP protocol and BraveSearchMCP."""
        # Create a mock ACP client
        client = ACPClient("https://example.com/verification")
        
        # Create envelope metadata
        metadata = EnvelopeMetadata(
            message_id="msg-456",
            timestamp="2023-06-01T12:30:00Z",
            sender="agent-1",
            recipient="verification-agent",
            content_type="application/json"
        )
        
        # Mock the send method to use BraveSearchMCP
        def mock_send(metadata, payload, attachments={}):
            # Process the request based on recipient
            if metadata.recipient == "verification-agent":
                # Use BraveSearchMCP to verify facts
                if isinstance(payload, dict) and "facts" in payload:
                    verified_facts = []
                    for fact in payload["facts"]:
                        # Search for evidence of the fact
                        search_results = self.brave_search.web_search(fact)
                        verified = "python" in search_results.lower() and any(
                            keyword in search_results.lower() 
                            for keyword in fact.lower().split()
                        )
                        verified_facts.append({
                            "fact": fact,
                            "verified": verified,
                            "evidence": search_results[:200] + "..." if len(search_results) > 200 else search_results
                        })
                    
                    return {
                        "status": "success",
                        "verified_facts": verified_facts
                    }
                else:
                    return {"status": "error", "message": "Invalid payload"}
            else:
                return {"status": "error", "message": f"Unknown recipient: {metadata.recipient}"}
        
        with patch.object(ACPClient, 'send', side_effect=mock_send):
            # Send a request to the verification agent
            payload = {
                "facts": [
                    "Python was created by Guido van Rossum",
                    "Python is a high-level programming language"
                ]
            }
            
            response = client.send(metadata, payload)
            
            # Verify BraveSearchMCP was called for each fact
            assert self.brave_search.web_search_called
            assert len(self.brave_search.web_search_args) >= len(payload["facts"])
            
            # Verify response
            assert response["status"] == "success"
            assert "verified_facts" in response
            assert len(response["verified_facts"]) == len(payload["facts"])
            
            # Verify each fact has verification information
            for fact_result in response["verified_facts"]:
                assert "fact" in fact_result
                assert "verified" in fact_result
                assert "evidence" in fact_result
    
    @pytest.mark.asyncio
    async def test_a2a_anp_acp_chain(self):
        """Test a chain of A2A, ANP, and ACP protocols working together."""
        # Set up mock clients and services
        registry = ANPRegistry("https://example.com/registry")
        
        # Mock the resolve method for ANP
        def mock_resolve(did):
            return self.anp_registry.get(did)
        
        # Mock A2A invoke method
        async def mock_a2a_invoke(req):
            capability = req.capability
            if capability == "research":
                # Use BraveSearchMCP for research
                search_results = self.brave_search.web_search(req.payload["query"])
                
                yield TaskResponseEvent(event="data", data="Searching...")
                yield TaskResponseEvent(event="data", data=f"Found results for {req.payload['query']}")
                yield TaskResponseEvent(event="end", data={"status": "success", "results": search_results})
                
            elif capability == "generate_image":
                # Use EverArtMCP for image generation
                image_result = self.everart.generate_image(req.payload["prompt"])
                
                yield TaskResponseEvent(event="data", data="Generating image...")
                yield TaskResponseEvent(event="end", data={"status": "success", "image_url": image_result})
                
            else:
                yield TaskResponseEvent(event="error", data={"message": f"Unknown capability: {capability}"})
        
        # Mock ANP send method
        async def mock_anp_send(msg):
            # Get the DID document for the recipient
            doc = registry.resolve(msg.recipient)
            
            # Find the appropriate service endpoint
            service_type = msg.body.get("service_type", "A2A")
            service = next((s for s in doc.service if s["type"] == service_type), None)
            
            if not service:
                return [TaskResponseEvent(event="error", data={"message": f"No {service_type} service found for {msg.recipient}"})]
            
            if service_type == "A2A":
                # Create an A2A client
                agent_card = AgentCard(
                    name=doc.id,
                    version="1.0",
                    endpoint=service["serviceEndpoint"],
                    schema={}
                )
                client = A2AClient(agent_card)
                
                # Create a task request from the message body
                task_request = TaskRequest(
                    capability=msg.body.get("capability", "default"),
                    version=msg.body.get("version", "1.0"),
                    payload=msg.body.get("payload", {})
                )
                
                # Invoke the A2A task
                with patch.object(A2AClient, 'invoke', side_effect=mock_a2a_invoke):
                    results = []
                    async for event in client.invoke(task_request):
                        results.append(event)
                    
                    return results
                    
            elif service_type == "ACP":
                # Create an ACP client
                client = ACPClient(service["serviceEndpoint"])
                
                # Create envelope metadata
                metadata = EnvelopeMetadata(
                    message_id=msg["@id"],
                    timestamp="2023-06-01T12:00:00Z",
                    sender=msg.sender,
                    recipient=msg.recipient,
                    content_type="application/json"
                )
                
                # Mock ACP send
                def mock_acp_send(metadata, payload, attachments={}):
                    if metadata.recipient == "did:example:summary":
                        return self.acp_server_responses["summary"]
                    else:
                        return {"status": "error", "message": f"Unknown recipient: {metadata.recipient}"}
                
                with patch.object(ACPClient, 'send', side_effect=mock_acp_send):
                    response = client.send(metadata, msg.body.get("payload", {}))
                    
                    # Convert ACP response to A2A format for consistency
                    return [TaskResponseEvent(event="end", data=response)]
            
            else:
                return [TaskResponseEvent(event="error", data={"message": f"Unsupported service type: {service_type}"})]
        
        # Apply patches
        with patch.object(ANPRegistry, 'resolve', side_effect=mock_resolve):
            with patch.object(ANPMessenger, 'send', side_effect=mock_anp_send):
                # Create an ANP messenger
                messenger = ANPMessenger(registry)
                
                # Step 1: Send a research request via ANP to an A2A service
                research_message = NetworkMessage(
                    **{
                        "@context": ["https://example.com/context"],
                        "@id": "msg-research-123",
                        "@type": "ResearchRequest",
                        "sender": "did:example:sender",
                        "recipient": "did:example:research",
                        "body": {
                            "service_type": "A2A",
                            "capability": "research",
                            "version": "1.0",
                            "payload": {"query": "Python programming language"}
                        }
                    }
                )
                
                research_results = await messenger.send(research_message)
                
                # Verify research results
                assert len(research_results) >= 1
                assert research_results[-1].event == "end"
                assert "success" in research_results[-1].data["status"]
                research_content = research_results[-1].data["results"]
                
                # Step 2: Extract key information for image generation
                image_prompt = "Python programming language logo and code"
                
                # Step 3: Send an image generation request via ANP to another A2A service
                image_message = NetworkMessage(
                    **{
                        "@context": ["https://example.com/context"],
                        "@id": "msg-image-123",
                        "@type": "ImageRequest",
                        "sender": "did:example:sender",
                        "recipient": "did:example:image",
                        "body": {
                            "service_type": "A2A",
                            "capability": "generate_image",
                            "version": "1.0",
                            "payload": {"prompt": image_prompt}
                        }
                    }
                )
                
                image_results = await messenger.send(image_message)
                
                # Verify image results
                assert len(image_results) >= 1
                assert image_results[-1].event == "end"
                assert "success" in image_results[-1].data["status"]
                assert "image_url" in image_results[-1].data
                
                # Step 4: Combine research and image results
                combined_data = {
                    "research": research_content,
                    "image_url": image_results[-1].data["image_url"]
                }
                
                # Step 5: Send a summary request via ANP to an ACP service
                summary_message = NetworkMessage(
                    **{
                        "@context": ["https://example.com/context"],
                        "@id": "msg-summary-123",
                        "@type": "SummaryRequest",
                        "sender": "did:example:sender",
                        "recipient": "did:example:summary",
                        "body": {
                            "service_type": "ACP",
                            "payload": {"text": combined_data["research"]}
                        }
                    }
                )
                
                summary_results = await messenger.send(summary_message)
                
                # Verify summary results
                assert len(summary_results) >= 1
                assert summary_results[-1].event == "end"
                assert "success" in summary_results[-1].data["status"]
                assert "summary" in summary_results[-1].data
                
                # Verify the complete chain worked correctly
                assert self.brave_search.web_search_called
                assert self.everart.generate_image_called
                
                # Final result should contain all components
                final_result = {
                    "research": research_content,
                    "image_url": image_results[-1].data["image_url"],
                    "summary": summary_results[-1].data["summary"]
                }
                
                assert "python" in final_result["research"].lower()
                assert "image" in final_result["image_url"].lower()
                assert "python" in final_result["summary"].lower()
    
    def test_protocol_with_filesystem_mcp(self):
        """Test integration between protocols and FilesystemMCP."""
        # Create a mock ACP client
        client = ACPClient("https://example.com/file-manager")
        
        # Create envelope metadata
        metadata = EnvelopeMetadata(
            message_id="msg-file-123",
            timestamp="2023-06-01T13:00:00Z",
            sender="agent-1",
            recipient="file-manager-agent",
            content_type="application/json"
        )
        
        # Mock the send method to use FilesystemMCP
        def mock_send(metadata, payload, attachments={}):
            # Process the request based on action
            if isinstance(payload, dict) and "action" in payload:
                action = payload["action"]
                
                if action == "write":
                    # Write content to a file
                    file_path = payload.get("path", "default.txt")
                    content = payload.get("content", "")
                    
                    self.filesystem.write_file(file_path, content)
                    
                    return {
                        "status": "success",
                        "message": f"File {file_path} written successfully",
                        "path": file_path
                    }
                
                elif action == "read":
                    # Read content from a file
                    file_path = payload.get("path", "default.txt")
                    
                    try:
                        content = self.filesystem.read_file(file_path)
                        return {
                            "status": "success",
                            "content": content,
                            "path": file_path
                        }
                    except Exception as e:
                        return {
                            "status": "error",
                            "message": f"Error reading file: {str(e)}",
                            "path": file_path
                        }
                
                elif action == "list":
                    # List directory contents
                    dir_path = payload.get("path", ".")
                    recursive = payload.get("recursive", False)
                    
                    listing = self.filesystem.list_directory(dir_path, recursive)
                    
                    return {
                        "status": "success",
                        "listing": listing,
                        "path": dir_path
                    }
                
                else:
                    return {"status": "error", "message": f"Unknown action: {action}"}
            else:
                return {"status": "error", "message": "Invalid payload, missing action"}
        
        with patch.object(ACPClient, 'send', side_effect=mock_send):
            # Step 1: Write a file
            write_payload = {
                "action": "write",
                "path": "python_info.txt",
                "content": "Python is a high-level programming language."
            }
            
            write_response = client.send(metadata, write_payload)
            
            # Verify write was successful
            assert write_response["status"] == "success"
            assert self.filesystem.write_file_called
            assert write_payload["path"] in self.filesystem.write_file_args[0]
            assert write_payload["content"] in self.filesystem.write_file_args[1]
            
            # Step 2: Read the file back
            read_payload = {
                "action": "read",
                "path": "python_info.txt"
            }
            
            read_response = client.send(metadata, read_payload)
            
            # Verify read was successful
            assert read_response["status"] == "success"
            assert self.filesystem.read_file_called
            assert read_payload["path"] in self.filesystem.read_file_args[0]
            assert "content" in read_response
            assert "python" in read_response["content"].lower()
            
            # Step 3: List directory contents
            list_payload = {
                "action": "list",
                "path": ".",
                "recursive": False
            }
            
            list_response = client.send(metadata, list_payload)
            
            # Verify list was successful
            assert list_response["status"] == "success"
            assert self.filesystem.list_directory_called
            assert list_payload["path"] in self.filesystem.list_directory_args[0]
            assert "listing" in list_response
            assert "python_info.txt" in list_response["listing"]
    
    def test_protocol_with_memory_mcp(self):
        """Test integration between protocols and MemoryMCP."""
        # Create a mock ACP client
        client = ACPClient("https://example.com/memory-manager")
        
        # Create envelope metadata
        metadata = EnvelopeMetadata(
            message_id="msg-memory-123",
            timestamp="2023-06-01T14:00:00Z",
            sender="agent-1",
            recipient="memory-manager-agent",
            content_type="application/json"
        )
        
        # Mock the send method to use MemoryMCP
        def mock_send(metadata, payload, attachments={}):
            # Process the request based on action
            if isinstance(payload, dict) and "action" in payload:
                action = payload["action"]
                
                if action == "store":
                    # Store a memory item
                    key = payload.get("key", "default_key")
                    value = payload.get("value", "")
                    namespace = payload.get("namespace", "default")
                    
                    self.memory.store_memory(key, value, namespace)
                    
                    return {
                        "status": "success",
                        "message": f"Memory {key} stored successfully",
                        "key": key,
                        "namespace": namespace
                    }
                
                elif action == "retrieve":
                    # Retrieve a memory item
                    key = payload.get("key", "default_key")
                    namespace = payload.get("namespace", "default")
                    
                    try:
                        value = self.memory.retrieve_memory(key, namespace)
                        return {
                            "status": "success",
                            "value": value,
                            "key": key,
                            "namespace": namespace
                        }
                    except Exception as e:
                        return {
                            "status": "error",
                            "message": f"Error retrieving memory: {str(e)}",
                            "key": key,
                            "namespace": namespace
                        }
                
                elif action == "list":
                    # List memories in a namespace
                    namespace = payload.get("namespace", "default")
                    
                    listing = self.memory.list_memories(namespace)
                    
                    return {
                        "status": "success",
                        "listing": listing,
                        "namespace": namespace
                    }
                
                elif action == "search":
                    # Search memories
                    query = payload.get("query", "")
                    namespace = payload.get("namespace", "default")
                    
                    results = self.memory.search_memories(query, namespace)
                    
                    return {
                        "status": "success",
                        "results": results,
                        "query": query,
                        "namespace": namespace
                    }
                
                else:
                    return {"status": "error", "message": f"Unknown action: {action}"}
            else:
                return {"status": "error", "message": "Invalid payload, missing action"}
        
        with patch.object(ACPClient, 'send', side_effect=mock_send):
            # Step 1: Store a memory
            store_payload = {
                "action": "store",
                "key": "python_info",
                "value": "Python is a high-level programming language.",
                "namespace": "research"
            }
            
            store_response = client.send(metadata, store_payload)
            
            # Verify store was successful
            assert store_response["status"] == "success"
            assert self.memory.store_memory_called
            assert store_payload["key"] in self.memory.store_memory_args[0]
            assert store_payload["value"] in self.memory.store_memory_args[1]
            assert store_payload["namespace"] in self.memory.store_memory_args[2]
            
            # Step 2: Retrieve the memory
            retrieve_payload = {
                "action": "retrieve",
                "key": "python_info",
                "namespace": "research"
            }
            
            retrieve_response = client.send(metadata, retrieve_payload)
            
            # Verify retrieve was successful
            assert retrieve_response["status"] == "success"
            assert self.memory.retrieve_memory_called
            assert retrieve_payload["key"] in self.memory.retrieve_memory_args[0]
            assert retrieve_payload["namespace"] in self.memory.retrieve_memory_args[1]
            assert "value" in retrieve_response
            assert "python" in retrieve_response["value"].lower()
            
            # Step 3: List memories in the namespace
            list_payload = {
                "action": "list",
                "namespace": "research"
            }
            
            list_response = client.send(metadata, list_payload)
            
            # Verify list was successful
            assert list_response["status"] == "success"
            assert self.memory.list_memories_called
            assert list_payload["namespace"] in self.memory.list_memories_args[0]
            assert "listing" in list_response
            
            # Step 4: Search memories
            search_payload = {
                "action": "search",
                "query": "python",
                "namespace": "research"
            }
            
            search_response = client.send(metadata, search_payload)
            
            # Verify search was successful
            assert search_response["status"] == "success"
            assert self.memory.search_memories_called
            assert search_payload["query"] in self.memory.search_memories_args[0]
            assert search_payload["namespace"] in self.memory.search_memories_args[1]
            assert "results" in search_response
    
    @pytest.mark.asyncio
    async def test_multi_protocol_research_workflow(self):
        """Test a complete research workflow using multiple protocols and MCPs."""
        # Set up mock clients and services
        registry = ANPRegistry("https://example.com/registry")
        
        # Mock the resolve method for ANP
        def mock_resolve(did):
            return self.anp_registry.get(did)
        
        # Mock A2A invoke method
        async def mock_a2a_invoke(req):
            capability = req.capability
            if capability == "research":
                # Use BraveSearchMCP for research
                search_results = self.brave_search.web_search(req.payload["query"])
                
                yield TaskResponseEvent(event="data", data="Searching...")
                yield TaskResponseEvent(event="data", data=f"Found results for {req.payload['query']}")
                yield TaskResponseEvent(event="end", data={"status": "success", "results": search_results})
                
            elif capability == "generate_image":
                # Use EverArtMCP for image generation
                image_result = self.everart.generate_image(req.payload["prompt"])
                
                yield TaskResponseEvent(event="data", data="Generating image...")
                yield TaskResponseEvent(event="end", data={"status": "success", "image_url": image_result})
                
            else:
                yield TaskResponseEvent(event="error", data={"message": f"Unknown capability: {capability}"})
        
        # Mock ANP send method
        async def mock_anp_send(msg):
            # Get the DID document for the recipient
            doc = registry.resolve(msg.recipient)
            
            # Find the appropriate service endpoint
            service_type = msg.body.get("service_type", "A2A")
            service = next((s for s in doc.service if s["type"] == service_type), None)
            
            if not service:
                return [TaskResponseEvent(event="error", data={"message": f"No {service_type} service found for {msg.recipient}"})]
            
            if service_type == "A2A":
                # Create an A2A client
                agent_card = AgentCard(
                    name=doc.id,
                    version="1.0",
                    endpoint=service["serviceEndpoint"],
                    schema={}
                )
                client = A2AClient(agent_card)
                
                # Create a task request from the message body
                task_request = TaskRequest(
                    capability=msg.body.get("capability", "default"),
                    version=msg.body.get("version", "1.0"),
                    payload=msg.body.get("payload", {})
                )
                
                # Invoke the A2A task
                with patch.object(A2AClient, 'invoke', side_effect=mock_a2a_invoke):
                    results = []
                    async for event in client.invoke(task_request):
                        results.append(event)
                    
                    return results
                    
            elif service_type == "ACP":
                # Create an ACP client
                client = ACPClient(service["serviceEndpoint"])
                
                # Create envelope metadata
                metadata = EnvelopeMetadata(
                    message_id=msg["@id"],
                    timestamp="2023-06-01T12:00:00Z",
                    sender=msg.sender,
                    recipient=msg.recipient,
                    content_type="application/json"
                )
                
                # Mock ACP send
                def mock_acp_send(metadata, payload, attachments={}):
                    if metadata.recipient == "did:example:summary":
                        return self.acp_server_responses["summary"]
                    else:
                        return {"status": "error", "message": f"Unknown recipient: {metadata.recipient}"}
                
                with patch.object(ACPClient, 'send', side_effect=mock_acp_send):
                    response = client.send(metadata, msg.body.get("payload", {}))
                    
                    # Convert ACP response to A2A format for consistency
                    return [TaskResponseEvent(event="end", data=response)]
            
            else:
                return [TaskResponseEvent(event="error", data={"message": f"Unsupported service type: {service_type}"})]
        
        # Mock ACP client for file operations
        def mock_file_acp_send(metadata, payload, attachments={}):
            # Process file operations
            if isinstance(payload, dict) and "action" in payload:
                action = payload["action"]
                
                if action == "write":
                    file_path = payload.get("path", "default.txt")
                    content = payload.get("content", "")
                    
                    self.filesystem.write_file(file_path, content)
                    
                    return {
                        "status": "success",
                        "message": f"File {file_path} written successfully",
                        "path": file_path
                    }
                else:
                    return {"status": "error", "message": f"Unknown action: {action}"}
            else:
                return {"status": "error", "message": "Invalid payload"}
        
        # Apply patches
        with patch.object(ANPRegistry, 'resolve', side_effect=mock_resolve):
            with patch.object(ANPMessenger, 'send', side_effect=mock_anp_send):
                with patch.object(ACPClient, 'send', side_effect=mock_file_acp_send):
                    # Create clients
                    messenger = ANPMessenger(registry)
                    file_client = ACPClient("https://example.com/file-manager")
                    
                    # Step 1: Research using A2A via ANP
                    research_message = NetworkMessage(
                        **{
                            "@context": ["https://example.com/context"],
                            "@id": "msg-research-123",
                            "@type": "ResearchRequest",
                            "sender": "did:example:sender",
                            "recipient": "did:example:research",
                            "body": {
                                "service_type": "A2A",
                                "capability": "research",
                                "version": "1.0",
                                "payload": {"query": "Python programming language"}
                            }
                        }
                    )
                    
                    research_results = await messenger.send(research_message)
                    research_content = research_results[-1].data["results"]
                    
                    # Step 2: Generate image using A2A via ANP
                    image_message = NetworkMessage(
                        **{
                            "@context": ["https://example.com/context"],
                            "@id": "msg-image-123",
                            "@type": "ImageRequest",
                            "sender": "did:example:sender",
                            "recipient": "did:example:image",
                            "body": {
                                "service_type": "A2A",
                                "capability": "generate_image",
                                "version": "1.0",
                                "payload": {"prompt": "Python programming language logo and code"}
                            }
                        }
                    )
                    
                    image_results = await messenger.send(image_message)
                    image_url = image_results[-1].data["image_url"]
                    
                    # Step 3: Get summary using ACP via ANP
                    summary_message = NetworkMessage(
                        **{
                            "@context": ["https://example.com/context"],
                            "@id": "msg-summary-123",
                            "@type": "SummaryRequest",
                            "sender": "did:example:sender",
                            "recipient": "did:example:summary",
                            "body": {
                                "service_type": "ACP",
                                "payload": {"text": research_content}
                            }
                        }
                    )
                    
                    summary_results = await messenger.send(summary_message)
                    summary = summary_results[-1].data["summary"]
                    
                    # Step 4: Save results to file using ACP directly
                    file_metadata = EnvelopeMetadata(
                        message_id="msg-file-123",
                        timestamp="2023-06-01T15:00:00Z",
                        sender="agent-1",
                        recipient="file-manager-agent",
                        content_type="application/json"
                    )
                    
                    # Create final report
                    final_report = f"""
                    # Python Programming Language Research
                    
                    ## Summary
                    {summary}
                    
                    ## Image
                    ![Python Image]({image_url})
                    
                    ## Detailed Information
                    {research_content}
                    """
                    
                    file_payload = {
                        "action": "write",
                        "path": "python_research_report.md",
                        "content": final_report
                    }
                    
                    file_response = file_client.send(file_metadata, file_payload)
                    
                    # Verify the complete workflow worked correctly
                    assert self.brave_search.web_search_called
                    assert self.everart.generate_image_called
                    assert self.filesystem.write_file_called
                    
                    # Verify file was written successfully
                    assert file_response["status"] == "success"
                    assert file_payload["path"] in self.filesystem.write_file_args[0]
                    
                    # Verify file content contains all components
                    file_content = self.filesystem.read_file("python_research_report.md")
                    assert "Python" in file_content
                    assert "Summary" in file_content
                    assert "Image" in file_content
                    assert image_url in file_content
                    assert "Detailed Information" in file_content


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
