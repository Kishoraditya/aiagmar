"""
Agent Communication Protocol (ACP) Implementation

A REST-native protocol for multimodal, asynchronous agent messaging—ideal for 
sending text, images, attachments, and streaming results without tight coupling.
"""

from pydantic import BaseModel, Field, HttpUrl, validator
from typing import Dict, List, Any, Optional, Union, BinaryIO, AsyncGenerator, Callable
import httpx
import json
import asyncio
import uuid
import time
import logging
import os
import mimetypes
from datetime import datetime
from fastapi import FastAPI, Request, Response, File, Form, UploadFile, BackgroundTasks, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
import aiofiles
import aiofiles.os
from io import BytesIO
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("acp")

# -----------------------------------------------------------------------------
# Schema Definitions
# -----------------------------------------------------------------------------

class EnvelopeMetadata(BaseModel):
    """
    Metadata for a message envelope.
    """
    message_id: str = Field(..., description="Unique message identifier")
    timestamp: str = Field(..., description="ISO 8601 timestamp when the message was created")
    sender: str = Field(..., description="Identifier of the sender")
    recipient: str = Field(..., description="Identifier of the recipient")
    content_type: str = Field(..., description="MIME type of the primary content")
    reply_to: Optional[str] = Field(None, description="Message ID this is replying to")
    correlation_id: Optional[str] = Field(None, description="ID for correlating related messages")
    ttl: Optional[int] = Field(None, description="Time-to-live in seconds")
    callback_url: Optional[HttpUrl] = Field(None, description="URL for asynchronous callbacks")
    parts: Optional[Dict[str, str]] = Field(None, description="Map of part names to MIME types")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate ISO 8601 timestamp format"""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError(f"Invalid timestamp format: {v}")


class MessagePart(BaseModel):
    """
    A part of a multipart message.
    """
    name: str = Field(..., description="Name of the part")
    content_type: str = Field(..., description="MIME type of the content")
    content: Union[str, bytes] = Field(..., description="Content of the part")
    filename: Optional[str] = Field(None, description="Original filename if an attachment")


class Message(BaseModel):
    """
    Complete message with metadata and parts.
    """
    metadata: EnvelopeMetadata
    parts: Dict[str, MessagePart] = Field(default_factory=dict)


class CallbackStatus(BaseModel):
    """
    Status update for asynchronous processing.
    """
    message_id: str = Field(..., description="ID of the original message")
    timestamp: str = Field(..., description="ISO 8601 timestamp of the status update")
    status: str = Field(..., description="Status code (e.g., 'processing', 'completed', 'failed')")
    progress: Optional[float] = Field(None, description="Progress as a percentage (0-100)")
    details: Optional[str] = Field(None, description="Additional status details")
    result_url: Optional[HttpUrl] = Field(None, description="URL where results can be retrieved")


# -----------------------------------------------------------------------------
# Rate Limiting and Validation
# -----------------------------------------------------------------------------

class RateLimiter:
    """Simple rate limiter implementation"""
    
    def __init__(self, rate_limit_per_minute: int = 60):
        self.rate_limit = rate_limit_per_minute
        self.clients = {}  # Dict mapping client IDs to their request counts
        self.reset_time = time.time() + 60
    
    def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if the request should be rate limited.
        
        Args:
            client_id: Identifier for the client
            
        Returns:
            True if request is allowed, False if rate limited
        """
        current_time = time.time()
        
        # Reset counters if a minute has passed
        if current_time > self.reset_time:
            self.clients = {}
            self.reset_time = current_time + 60
        
        # Check client-based rate limit
        if client_id not in self.clients:
            self.clients[client_id] = 0
        self.clients[client_id] += 1
        
        return self.clients[client_id] <= self.rate_limit


class ACPValidator:
    """Validator for ACP messages"""
    
    @staticmethod
    def validate_envelope(metadata: EnvelopeMetadata, parts: Dict[str, Any]) -> Optional[str]:
        """
        Validate an ACP envelope.
        
        Args:
            metadata: Envelope metadata
            parts: Message parts
            
        Returns:
            Error message if validation fails, None otherwise
        """
        # Check if all declared parts are present
        if metadata.parts:
            for part_name in metadata.parts:
                if part_name not in parts:
                    return f"Missing declared part: {part_name}"
        
        # Check content type consistency
        if metadata.content_type.startswith("multipart/") and not metadata.parts:
            return "Multipart content type declared but no parts specified"
        
        # Check TTL
        if metadata.ttl is not None:
            try:
                timestamp = datetime.fromisoformat(metadata.timestamp.replace('Z', '+00:00'))
                expiry = timestamp.timestamp() + metadata.ttl
                if time.time() > expiry:
                    return "Message has expired (TTL exceeded)"
            except ValueError:
                return "Invalid timestamp format"
        
        return None


# -----------------------------------------------------------------------------
# Client Implementation
# -----------------------------------------------------------------------------

class ACPClient:
    """
    Client for sending messages using the ACP protocol.
    """
    
    def __init__(self, endpoint: str, sender_id: str, callback_url: Optional[str] = None):
        """
        Initialize the ACP client.
        
        Args:
            endpoint: URL of the ACP endpoint
            sender_id: Identifier for this sender
            callback_url: Optional URL for receiving callbacks
        """
        self.endpoint = endpoint
        self.sender_id = sender_id
        self.callback_url = callback_url
    
    async def send(self, recipient_id: str, content: Union[str, bytes], 
                  content_type: str, attachments: Dict[str, BinaryIO] = None,
                  reply_to: Optional[str] = None, ttl: Optional[int] = None,
                  stream_response: bool = False) -> Union[Dict[str, Any], AsyncGenerator[bytes, None]]:
        """
        Send a message to the specified recipient.
        
        Args:
            recipient_id: Identifier for the recipient
            content: Primary content of the message
            content_type: MIME type of the primary content
            attachments: Optional dict of attachment name -> file-like object
            reply_to: Optional message ID this is replying to
            ttl: Optional time-to-live in seconds
            stream_response: Whether to stream the response
            
        Returns:
            Response data or streaming response
        """
        # Create message ID and timestamp
        message_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        # Prepare metadata
        metadata = EnvelopeMetadata(
            message_id=message_id,
            timestamp=timestamp,
            sender=self.sender_id,
            recipient=recipient_id,
            content_type=content_type,
            reply_to=reply_to,
            ttl=ttl,
            callback_url=self.callback_url
        )
        
        # Prepare parts dictionary if there are attachments
        if attachments:
            metadata.parts = {}
            for name, file_obj in attachments.items():
                # Try to guess MIME type from filename
                filename = getattr(file_obj, 'name', name)
                mime_type, _ = mimetypes.guess_type(filename)
                metadata.parts[name] = mime_type or 'application/octet-stream'
        
        # Prepare multipart request
        files = {
            'metadata': ('metadata.json', metadata.json(), 'application/json')
        }
        
        # Add primary content
        if isinstance(content, str):
            files['content'] = ('content', content, content_type)
        else:
            files['content'] = ('content', content, content_type)
        
        # Add attachments
        if attachments:
            for name, file_obj in attachments.items():
                filename = getattr(file_obj, 'name', name)
                mime_type = metadata.parts[name]
                files[name] = (filename, file_obj, mime_type)
        
        # Prepare headers
        headers = {}
        if self.callback_url:
            headers['X-Callback-URL'] = str(self.callback_url)
        
        # Send request
        async with httpx.AsyncClient() as client:
            if stream_response:
                # Stream the response
                async with client.stream('POST', self.endpoint, files=files, headers=headers) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk
            else:
                # Get the full response
                response = await client.post(self.endpoint, files=files, headers=headers)
                response.raise_for_status()
                
                # Try to parse as JSON, fall back to text
                try:
                    yield response.json()
                except json.JSONDecodeError:
                    yield {'content': response.text}
    
    async def send_text(self, recipient_id: str, text: str, 
                       attachments: Dict[str, BinaryIO] = None,
                       reply_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a text message.
        
        Args:
            recipient_id: Identifier for the recipient
            text: Text content
            attachments: Optional attachments
            reply_to: Optional message ID this is replying to
            
        Returns:
            Response data
        """
        response = await self.send(
            recipient_id=recipient_id,
            content=text,
            content_type='text/plain',
            attachments=attachments,
            reply_to=reply_to
        )
        return response
    
    async def send_json(self, recipient_id: str, data: Dict[str, Any],
                       attachments: Dict[str, BinaryIO] = None,
                       reply_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a JSON message.
        
        Args:
            recipient_id: Identifier for the recipient
            data: JSON data
            attachments: Optional attachments
            reply_to: Optional message ID this is replying to
            
        Returns:
            Response data
        """
        response = await self.send(
            recipient_id=recipient_id,
            content=json.dumps(data),
            content_type='application/json',
            attachments=attachments,
            reply_to=reply_to
        )
        return response
    
    async def send_image(self, recipient_id: str, image_data: bytes, 
                        image_type: str = 'image/jpeg',
                        attachments: Dict[str, BinaryIO] = None,
                        reply_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Send an image message.
        
        Args:
            recipient_id: Identifier for the recipient
            image_data: Binary image data
            image_type: MIME type of the image
            attachments: Optional attachments
            reply_to: Optional message ID this is replying to
            
        Returns:
            Response data
        """
        response = await self.send(
            recipient_id=recipient_id,
            content=image_data,
            content_type=image_type,
            attachments=attachments,
            reply_to=reply_to
        )
        return response
    
    async def send_callback(self, original_message_id: str, status: str,
                           progress: Optional[float] = None, 
                           details: Optional[str] = None,
                           result_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a callback status update.
        
        Args:
            original_message_id: ID of the original message
            status: Status code (e.g., 'processing', 'completed', 'failed')
            progress: Optional progress percentage (0-100)
            details: Optional status details
            result_url: Optional URL where results can be retrieved
            
        Returns:
            Response data
        """
        # Create callback status
        callback = CallbackStatus(
            message_id=original_message_id,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            status=status,
            progress=progress,
            details=details,
            result_url=result_url
        )
        
        # Send as JSON
        async with httpx.AsyncClient() as client:
            response = await client.post(
                str(self.callback_url),
                json=callback.dict(),
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            # Try to parse as JSON, fall back to text
            try:
                return response.json()
            except json.JSONDecodeError:
                return {'content': response.text}


# -----------------------------------------------------------------------------
# Server Implementation
# -----------------------------------------------------------------------------

class ACPServer:
    """
    Server implementation for the ACP protocol.
    """
    
    def __init__(self, agent_id: str, storage_dir: Optional[str] = None, 
                rate_limit_per_minute: int = 60):
        """
        Initialize the ACP server.
        
        Args:
            agent_id: Identifier for this agent
            storage_dir: Optional directory for storing message attachments
            rate_limit_per_minute: Maximum requests per minute per client
        """
        self.agent_id = agent_id
        self.storage_dir = storage_dir or os.path.join(os.getcwd(), 'acp_storage')
        self.app = FastAPI(title=f"ACP Server - {agent_id}", version="1.0.0")
        self.rate_limiter = RateLimiter(rate_limit_per_minute)
        self.message_handlers = {}
        self.callback_handlers = {}
        
        # Ensure storage directory exists
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Register routes
        self.setup_routes()
    
    def setup_routes(self):
        """Set up the FastAPI routes"""
        
        @self.app.post("/")
        async def handle_message(
            request: Request,
            background_tasks: BackgroundTasks,
            x_callback_url: Optional[str] = Header(None)
        ):
            """Handle incoming ACP messages"""
            # Get client ID for rate limiting (use sender or IP)
            form_data = await request.form()
            metadata_file = form_data.get('metadata')
            
            if not metadata_file:
                raise HTTPException(status_code=400, detail="Missing metadata part")
            
            try:
                # Parse metadata
                metadata_content = await metadata_file.read()
                metadata = EnvelopeMetadata.parse_raw(metadata_content)
                
                # Check rate limit
                client_id = metadata.sender
                if not self.rate_limiter.check_rate_limit(client_id):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
                # Extract parts
                parts = {}
                for name, upload_file in form_data.items():
                    if name != 'metadata':
                        # Read content
                        content = await upload_file.read()
                        
                        # Create MessagePart
                        parts[name] = MessagePart(
                            name=name,
                            content_type=upload_file.content_type,
                            content=content,
                            filename=upload_file.filename
                        )
                
                # Validate envelope
                error = ACPValidator.validate_envelope(metadata, parts)
                if error:
                    raise HTTPException(status_code=400, detail=error)
                
                # Create complete message
                message = Message(metadata=metadata, parts=parts)
                
                # Store attachments if needed
                if self.storage_dir:
                    message_dir = os.path.join(self.storage_dir, metadata.message_id)
                    os.makedirs(message_dir, exist_ok=True)
                    
                    for name, part in parts.items():
                        if isinstance(part.content, bytes) and part.filename:
                            file_path = os.path.join(message_dir, part.filename)
                            async with aiofiles.open(file_path, 'wb') as f:
                                await f.write(part.content)
                
                # Get callback URL (from header or metadata)
                callback_url = x_callback_url or (
                    str(metadata.callback_url) if metadata.callback_url else None
                )
                
                # Handle message based on content type
                if metadata.content_type in self.message_handlers:
                    # Get handler for this content type
                    handler = self.message_handlers[metadata.content_type]
                    
                    if asyncio.iscoroutinefunction(handler):
                        # For async handlers, process in background if callback URL is provided
                        if callback_url:
                            # Send immediate acknowledgement
                            background_tasks.add_task(
                                self._process_async, 
                                handler, 
                                message, 
                                callback_url
                            )
                            return {
                                "status": "accepted",
                                "message_id": metadata.message_id
                            }
                        else:
                            # Process synchronously and return result
                            result = await handler(message)
                            return result
                    else:
                        # For sync handlers, always process immediately
                        result = handler(message)
                        return result
                else:
                    # No handler for this content type
                    raise HTTPException(
                        status_code=415, 
                        detail=f"Unsupported content type: {metadata.content_type}"
                    )
                
            except Exception as e:
                logger.exception(f"Error processing message: {e}")
                raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")
        
        @self.app.post("/callback")
        async def handle_callback(callback: CallbackStatus):
            """Handle callback status updates"""
            try:
                # Check if we have a handler for this message ID
                if callback.message_id in self.callback_handlers:
                    handler = self.callback_handlers[callback.message_id]
                    if asyncio.iscoroutinefunction(handler):
                        await handler(callback)
                    else:
                        handler(callback)
                    return {"status": "processed"}
                else:
                    # No handler for this message ID
                    raise HTTPException(
                        status_code=404, 
                        detail=f"No handler for message ID: {callback.message_id}"
                    )
            except Exception as e:
                logger.exception(f"Error processing callback: {e}")
                raise HTTPException(status_code=500, detail=f"Error processing callback: {str(e)}")
    
    async def _process_async(self, handler, message, callback_url):
        """
        Process a message asynchronously and send callbacks.
        
        Args:
            handler: The message handler function
            message: The message to process
            callback_url: URL to send callbacks to
        """
        try:
            # Create a client for sending callbacks
            client = ACPClient(
                endpoint="",  # Not used for callbacks
                sender_id=self.agent_id,
                callback_url=None  # No nested callbacks
            )
            
            # Send 'processing' status
            await client.send_callback(
                original_message_id=message.metadata.message_id,
                status="processing",
                progress=0,
                details="Starting processing"
            )
            
            # Process the message
            result = await handler(message)
            
            # Send 'completed' status
            await client.send_callback(
                original_message_id=message.metadata.message_id,
                status="completed",
                progress=100,
                details="Processing completed",
                result_url=result.get("result_url")
            )
            
        except Exception as e:
            logger.exception(f"Error in async processing: {e}")
            
            # Send 'failed' status
            await client.send_callback(
                original_message_id=message.metadata.message_id,
                status="failed",
                details=f"Processing failed: {str(e)}"
            )
    
    def register_handler(self, content_type: str, handler: Callable[[Message], Any]):
        """
        Register a handler for a specific content type.
        
        Args:
            content_type: MIME type to handle
            handler: Function to handle messages of this type
        """
        self.message_handlers[content_type] = handler
    
    def register_callback_handler(self, message_id: str, handler: Callable[[CallbackStatus], Any]):
        """
        Register a handler for callbacks related to a specific message.
        
        Args:
            message_id: ID of the message to handle callbacks for
            handler: Function to handle the callbacks
        """
        self.callback_handlers[message_id] = handler
    
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
# Streaming Implementation
# -----------------------------------------------------------------------------

class ACPStream:
    """
    Utilities for streaming content with ACP.
    """
    
    @staticmethod
    async def create_streaming_response(generator, content_type: str = "text/plain"):
        """
        Create a FastAPI StreamingResponse from an async generator.
        
        Args:
            generator: Async generator yielding content chunks
            content_type: MIME type of the content
            
        Returns:
            FastAPI StreamingResponse
        """
        async def stream_generator():
            async for chunk in generator:
                yield chunk
        
        return StreamingResponse(
            stream_generator(),
            media_type=content_type,
            headers={"X-Content-Type-Options": "nosniff"}
        )
    
    @staticmethod
    async def stream_file(file_path: str, chunk_size: int = 8192):
        """
        Stream a file from disk.
        
        Args:
            file_path: Path to the file
            chunk_size: Size of chunks to read
            
        Yields:
            File chunks
        """
        async with aiofiles.open(file_path, 'rb') as f:
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    
    @staticmethod
    async def stream_text_generator(text_generator, encoding: str = 'utf-8'):
        """
        Stream text from a generator.
        
        Args:
            text_generator: Generator yielding text strings
            encoding: Text encoding
            
        Yields:
            Encoded text chunks
        """
        async for text in text_generator:
            yield text.encode(encoding)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

async def save_message_parts(message: Message, directory: str):
    """
    Save all parts of a message to disk.
    
    Args:
        message: The message to save
        directory: Directory to save parts to
    """
    os.makedirs(directory, exist_ok=True)
    
    # Save metadata
    metadata_path = os.path.join(directory, 'metadata.json')
    async with aiofiles.open(metadata_path, 'w') as f:
        await f.write(message.metadata.json(indent=2))
    
    # Save parts
    for name, part in message.parts.items():
        if isinstance(part.content, bytes):
            # Binary content
            filename = part.filename or name
            part_path = os.path.join(directory, filename)
            async with aiofiles.open(part_path, 'wb') as f:
                await f.write(part.content)
        else:
            # Text content
            filename = part.filename or f"{name}.txt"
            part_path = os.path.join(directory, filename)
            async with aiofiles.open(part_path, 'w') as f:
                await f.write(part.content)


async def load_message(directory: str) -> Message:
    """
    Load a message from disk.
    
    Args:
        directory: Directory containing the message parts
        
    Returns:
        Reconstructed message
    """
    # Load metadata
    metadata_path = os.path.join(directory, 'metadata.json')
    async with aiofiles.open(metadata_path, 'r') as f:
        metadata_json = await f.read()
    
    metadata = EnvelopeMetadata.parse_raw(metadata_json)
    
    # Load parts
    parts = {}
    for filename in await aiofiles.os.listdir(directory):
        if filename != 'metadata.json':
            part_path = os.path.join(directory, filename)
            
            # Determine content type
            mime_type, _ = mimetypes.guess_type(filename)
            content_type = mime_type or 'application/octet-stream'
            
            # Determine if binary or text
            if content_type.startswith(('text/', 'application/json')):
                async with aiofiles.open(part_path, 'r') as f:
                    content = await f.read()
            else:
                async with aiofiles.open(part_path, 'rb') as f:
                    content = await f.read()
            
            # Create part
            part_name = os.path.splitext(filename)[0]
            parts[part_name] = MessagePart(
                name=part_name,
                content_type=content_type,
                content=content,
                filename=filename
            )
    
    return Message(metadata=metadata, parts=parts)


def compute_checksum(data: Union[str, bytes]) -> str:
    """
    Compute SHA-256 checksum of data.
    
    Args:
        data: Data to compute checksum for
        
    Returns:
        Hex-encoded checksum
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return hashlib.sha256(data).hexdigest()


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

async def example_text_handler(message: Message):
    """Example handler for text messages"""
    # Get the text content
    content_part = message.parts.get('content')
    if not content_part:
        return {"error": "No content part found"}
    
    text = content_part.content
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    
    # Process the text (just echo it back with some stats)
    word_count = len(text.split())
    char_count = len(text)
    
    return {
        "message_id": message.metadata.message_id,
        "word_count": word_count,
        "character_count": char_count,
        "text": text
    }


async def example_json_handler(message: Message):
    """Example handler for JSON messages"""
    # Get the JSON content
    content_part = message.parts.get('content')
    if not content_part:
        return {"error": "No content part found"}
    
    # Parse JSON
    content = content_part.content
    if isinstance(content, bytes):
        content = content.decode('utf-8')
    
    try:
        data = json.loads(content)
        
        # Process the JSON (just echo it back with a timestamp)
        return {
            "message_id": message.metadata.message_id,
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "data": data
        }
    except json.JSONDecodeError:
        return {"error": "Invalid JSON content"}


async def example_image_handler(message: Message):
    """Example handler for image messages"""
    # Get the image content
    content_part = message.parts.get('content')
    if not content_part:
        return {"error": "No content part found"}
    
    # Ensure it's binary data
    if not isinstance(content_part.content, bytes):
        return {"error": "Image content must be binary"}
    
    # Process the image (just return its size and checksum)
    image_data = content_part.content
    size = len(image_data)
    checksum = compute_checksum(image_data)
    
    return {
        "message_id": message.metadata.message_id,
        "content_type": content_part.content_type,
        "size_bytes": size,
        "checksum": checksum
    }


async def example_streaming_handler(message: Message):
    """Example handler that streams a response"""
    # This would be used with ACPStream.create_streaming_response
    
    # Get the content
    content_part = message.parts.get('content')
    if not content_part:
        yield json.dumps({"error": "No content part found"}).encode('utf-8')
        return
    
    # Process in chunks (simulated)
    text = content_part.content
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    
    # Split into words and yield each with a delay
    words = text.split()
    for i, word in enumerate(words):
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Create a progress update
        progress = {
            "word": word,
            "position": i + 1,
            "total": len(words),
            "progress_percent": round((i + 1) / len(words) * 100, 1)
        }
        
        yield json.dumps(progress).encode('utf-8') + b'\n'


if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Create and configure server
        server = ACPServer(
            agent_id="example-agent",
            storage_dir="./acp_storage"
        )
        
        # Register handlers
        server.register_handler("text/plain", example_text_handler)
        server.register_handler("application/json", example_json_handler)
        server.register_handler("image/jpeg", example_image_handler)
        server.register_handler("image/png", example_image_handler)
        
        # Example of registering a streaming handler
        # This would need special handling in the server implementation
        # server.register_streaming_handler("text/plain", example_streaming_handler)
        
        print(f"Starting ACP server for agent {server.agent_id}")
        print(f"Storing messages in {server.storage_dir}")
        
        # Run the server
        server.run()
    
    # Run the main function
    asyncio.run(main())


# -----------------------------------------------------------------------------
# Client Example
# -----------------------------------------------------------------------------

async def client_example():
    """Example usage of the ACP client"""
    
    # Create a client
    client = ACPClient(
        endpoint="http://localhost:8000",
        sender_id="example-client",
        callback_url="http://localhost:8001/callback"
    )
    
    # Send a text message
    print("Sending text message...")
    text_response = await client.send_text(
        recipient_id="example-agent",
        text="Hello, world! This is a test message from the ACP client."
    )
    print(f"Response: {text_response}")
    
    # Send a JSON message
    print("\nSending JSON message...")
    json_data = {
        "command": "process",
        "parameters": {
            "option1": "value1",
            "option2": 42,
            "nested": {
                "key": "value"
            }
        }
    }
    json_response = await client.send_json(
        recipient_id="example-agent",
        data=json_data
    )
    print(f"Response: {json_response}")
    
    # Send an image (if available)
    try:
        print("\nSending image...")
        # Create a simple test image (a colored rectangle)
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        image_response = await client.send_image(
            recipient_id="example-agent",
            image_data=img_bytes.getvalue(),
            image_type="image/png"
        )
        print(f"Response: {image_response}")
    except ImportError:
        print("PIL not available, skipping image example")
    
    # Send a message with attachments
    print("\nSending message with attachments...")
    attachments = {}
    
    # Create a text file attachment
    text_file = BytesIO(b"This is the content of the text file attachment.")
    text_file.name = "example.txt"
    attachments["text_file"] = text_file
    
    # Create a JSON file attachment
    json_file = BytesIO(json.dumps({"key": "value"}).encode('utf-8'))
    json_file.name = "data.json"
    attachments["json_file"] = json_file
    
    attachment_response = await client.send_text(
        recipient_id="example-agent",
        text="This message has attachments.",
        attachments=attachments
    )
    print(f"Response: {attachment_response}")


# -----------------------------------------------------------------------------
# Advanced Features
# -----------------------------------------------------------------------------

class ACPBackpressureClient(ACPClient):
    """
    ACP client with backpressure handling.
    """
    
    async def send_with_backpressure(self, recipient_id: str, content: Union[str, bytes],
                                    content_type: str, max_retries: int = 3,
                                    initial_delay: float = 1.0, max_delay: float = 30.0):
        """
        Send a message with backpressure handling.
        
        Args:
            recipient_id: Identifier for the recipient
            content: Primary content of the message
            content_type: MIME type of the primary content
            max_retries: Maximum number of retries
            initial_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            
        Returns:
            Response data
        """
        delay = initial_delay
        retries = 0
        
        while True:
            try:
                response = await self.send(
                    recipient_id=recipient_id,
                    content=content,
                    content_type=content_type
                )
                return response
            
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Too Many Requests
                    # Get retry delay from header if available
                    retry_after = e.response.headers.get('Retry-After')
                    if retry_after and retry_after.isdigit():
                        delay = float(retry_after)
                    else:
                        # Exponential backoff
                        delay = min(delay * 2, max_delay)
                    
                    retries += 1
                    if retries > max_retries:
                        raise Exception(f"Max retries exceeded: {max_retries}")
                    
                    logger.info(f"Rate limited, retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    # Other HTTP errors
                    raise
            
            except Exception as e:
                # Other exceptions
                raise


class ACPBatchClient(ACPClient):
    """
    ACP client with batching capabilities.
    """
    
    async def send_batch(self, recipient_id: str, messages: List[Dict[str, Any]],
                        batch_size: int = 5, delay_between_batches: float = 1.0):
        """
        Send a batch of messages with rate control.
        
        Args:
            recipient_id: Identifier for the recipient
            messages: List of message data (each with 'content' and 'content_type')
            batch_size: Number of messages to send in each batch
            delay_between_batches: Delay between batches (seconds)
            
        Returns:
            List of responses
        """
        responses = []
        
        # Process in batches
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i+batch_size]
            
            # Send messages in batch concurrently
            batch_tasks = []
            for msg in batch:
                task = asyncio.create_task(self.send(
                    recipient_id=recipient_id,
                    content=msg['content'],
                    content_type=msg['content_type'],
                    attachments=msg.get('attachments')
                ))
                batch_tasks.append(task)
            
            # Wait for all messages in batch
            batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)
            responses.extend(batch_responses)
            
            # Delay before next batch
            if i + batch_size < len(messages):
                await asyncio.sleep(delay_between_batches)
        
        return responses


class ACPSecureClient(ACPClient):
    """
    ACP client with additional security features.
    """
    
    def __init__(self, endpoint: str, sender_id: str, 
                api_key: Optional[str] = None, 
                callback_url: Optional[str] = None):
        """
        Initialize the secure ACP client.
        
        Args:
            endpoint: URL of the ACP endpoint
            sender_id: Identifier for this sender
            api_key: API key for authentication
            callback_url: Optional URL for receiving callbacks
        """
        super().__init__(endpoint, sender_id, callback_url)
        self.api_key = api_key
    
    async def send(self, recipient_id: str, content: Union[str, bytes], 
                  content_type: str, attachments: Dict[str, BinaryIO] = None,
                  reply_to: Optional[str] = None, ttl: Optional[int] = None,
                  stream_response: bool = False) -> Union[Dict[str, Any], AsyncGenerator[bytes, None]]:
        """
        Send a message with security headers.
        
        Args:
            recipient_id: Identifier for the recipient
            content: Primary content of the message
            content_type: MIME type of the primary content
            attachments: Optional dict of attachment name -> file-like object
            reply_to: Optional message ID this is replying to
            ttl: Optional time-to-live in seconds
            stream_response: Whether to stream the response
            
        Returns:
            Response data or streaming response
        """
        # Create message ID and timestamp
        message_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        # Prepare metadata
        metadata = EnvelopeMetadata(
            message_id=message_id,
            timestamp=timestamp,
            sender=self.sender_id,
            recipient=recipient_id,
            content_type=content_type,
            reply_to=reply_to,
            ttl=ttl,
            callback_url=self.callback_url
        )
        
        # Prepare parts dictionary if there are attachments
        if attachments:
            metadata.parts = {}
            for name, file_obj in attachments.items():
                # Try to guess MIME type from filename
                filename = getattr(file_obj, 'name', name)
                mime_type, _ = mimetypes.guess_type(filename)
                metadata.parts[name] = mime_type or 'application/octet-stream'
        
        # Prepare multipart request
        files = {
            'metadata': ('metadata.json', metadata.json(), 'application/json')
        }
        
        # Add primary content
        if isinstance(content, str):
            files['content'] = ('content', content, content_type)
        else:
            files['content'] = ('content', content, content_type)
        
        # Add attachments
        if attachments:
            for name, file_obj in attachments.items():
                filename = getattr(file_obj, 'name', name)
                mime_type = metadata.parts[name]
                files[name] = (filename, file_obj, mime_type)
        
        # Prepare headers with security
        headers = {}
        if self.callback_url:
            headers['X-Callback-URL'] = str(self.callback_url)
        
        # Add API key if available
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"
        
        # Add content checksum for integrity
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        else:
            content_bytes = content
        
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        headers['X-Content-SHA256'] = content_hash
        
        # Add timestamp for replay protection
        headers['X-Timestamp'] = timestamp
        
        # Send request
        async with httpx.AsyncClient() as client:
            if stream_response:
                # Stream the response
                async with client.stream('POST', self.endpoint, files=files, headers=headers) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk
            else:
                # Get the full response
                response = await client.post(self.endpoint, files=files, headers=headers)
                response.raise_for_status()
                
                # Try to parse as JSON, fall back to text
                try:
                    yield response.json()
                except json.JSONDecodeError:
                    yield {'content': response.text}
