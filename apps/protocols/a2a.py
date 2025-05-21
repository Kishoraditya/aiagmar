"""
Agent-to-Agent (A2A) Protocol Implementation

A peer-to-peer protocol enabling agents to offload sub-tasks to specialized agents,
using HTTP for requests and Server-Sent Events (SSE) for streaming responses.
"""

from pydantic import BaseModel, Field, ValidationError, HttpUrl, validator
from typing import Dict, List, Any, Optional, AsyncGenerator, Union
import httpx
import json
import asyncio
import jsonschema
import time
from fastapi import FastAPI, Request, Response, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sse_starlette.sse import EventSourceResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("a2a")

# -----------------------------------------------------------------------------
# Schema Definitions
# -----------------------------------------------------------------------------

class AgentCard(BaseModel):
    """
    Capability descriptor that advertises an agent's functionality, version,
    input/output schema, and endpoint URL.
    """
    name: str = Field(..., description="Name of the agent")
    version: str = Field(..., description="Version of the agent's API")
    endpoint: HttpUrl = Field(..., description="HTTP endpoint where the agent can be reached")
    schema: Dict[str, Any] = Field(..., description="JSON Schema describing the agent's capabilities")
    description: Optional[str] = Field(None, description="Human-readable description of the agent")
    auth_required: bool = Field(False, description="Whether authentication is required")

    @validator('schema')
    def validate_schema(cls, v):
        """Ensure the schema is a valid JSON Schema"""
        try:
            jsonschema.Draft7Validator.check_schema(v)
            return v
        except jsonschema.exceptions.SchemaError as e:
            raise ValueError(f"Invalid JSON Schema: {e}")


class TaskRequest(BaseModel):
    """
    JSON object invoking a specific capability with parameters.
    """
    capability: str = Field(..., description="Name of the capability to invoke")
    version: str = Field(..., description="Version of the capability to invoke")
    payload: Dict[str, Any] = Field(..., description="Parameters for the capability")
    auth_token: Optional[str] = Field(None, description="Authentication token if required")


class TaskResponseEvent(BaseModel):
    """
    Event carrying partial outputs or status updates.
    """
    event: str = Field(..., description="Event type: 'data', 'error', or 'end'")
    data: Any = Field(..., description="Event payload")
    id: Optional[str] = Field(None, description="Optional event ID for correlation")
    retry: Optional[int] = Field(None, description="Retry delay in milliseconds if applicable")


class ErrorResponse(BaseModel):
    """
    Standard error response format.
    """
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    retry_after: Optional[int] = Field(None, description="Suggested retry delay in seconds")


# -----------------------------------------------------------------------------
# Rate Limiting
# -----------------------------------------------------------------------------

class RateLimiter:
    """Simple rate limiter implementation"""
    
    def __init__(self, rate_limit_per_minute: int = 60):
        self.rate_limit = rate_limit_per_minute
        self.tokens = {}  # Dict mapping tokens to their request counts
        self.ips = {}     # Dict mapping IPs to their request counts
        self.reset_time = time.time() + 60
    
    def check_rate_limit(self, token: Optional[str], ip: str) -> bool:
        """
        Check if the request should be rate limited.
        
        Args:
            token: Authentication token if available
            ip: IP address of the requester
            
        Returns:
            True if request is allowed, False if rate limited
        """
        current_time = time.time()
        
        # Reset counters if a minute has passed
        if current_time > self.reset_time:
            self.tokens = {}
            self.ips = {}
            self.reset_time = current_time + 60
        
        # Check token-based rate limit if token is provided
        if token:
            if token not in self.tokens:
                self.tokens[token] = 0
            self.tokens[token] += 1
            if self.tokens[token] > self.rate_limit:
                return False
        
        # Check IP-based rate limit
        if ip not in self.ips:
            self.ips[ip] = 0
        self.ips[ip] += 1
        if self.ips[ip] > self.rate_limit:
            return False
        
        return True


# -----------------------------------------------------------------------------
# Client Implementation
# -----------------------------------------------------------------------------

class A2AClient:
    """
    Client for invoking capabilities on remote agents via the A2A protocol.
    """
    
    def __init__(self, agent_card: AgentCard, auth_token: Optional[str] = None):
        """
        Initialize the A2A client.
        
        Args:
            agent_card: The AgentCard describing the remote agent
            auth_token: Optional authentication token
        """
        self.agent_card = agent_card
        self.auth_token = auth_token
    
    async def invoke(self, capability: str, payload: Dict[str, Any], 
                    version: Optional[str] = None) -> AsyncGenerator[TaskResponseEvent, None]:
        """
        Invoke a capability on the remote agent.
        
        Args:
            capability: Name of the capability to invoke
            payload: Parameters for the capability
            version: Optional version of the capability (defaults to agent's version)
            
        Yields:
            TaskResponseEvent objects containing streaming responses
            
        Raises:
            ValidationError: If the payload doesn't match the capability's schema
            httpx.HTTPStatusError: If the HTTP request fails
        """
        # Validate payload against schema if available
        if 'properties' in self.agent_card.schema:
            try:
                jsonschema.validate(instance=payload, schema=self.agent_card.schema)
            except jsonschema.exceptions.ValidationError as e:
                error_event = TaskResponseEvent(
                    event="error",
                    data={
                        "code": "validation_error",
                        "message": f"Payload validation failed: {str(e)}",
                        "details": {"validation_error": str(e)}
                    }
                )
                yield error_event
                return
        
        # Prepare the request
        req = TaskRequest(
            capability=capability,
            version=version or self.agent_card.version,
            payload=payload,
            auth_token=self.auth_token
        )
        
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json"
        }
        
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        # Send the request
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", 
                    str(self.agent_card.endpoint), 
                    json=req.dict(exclude_none=True), 
                    headers=headers,
                    timeout=30.0
                ) as response:
                    response.raise_for_status()
                    
                    # Process SSE events
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        
                        while "\n\n" in buffer:
                            event_data, buffer = buffer.split("\n\n", 1)
                            lines = event_data.strip().split("\n")
                            
                            event_type = "message"
                            data = None
                            event_id = None
                            retry = None
                            
                            for line in lines:
                                if line.startswith("event:"):
                                    event_type = line[6:].strip()
                                elif line.startswith("data:"):
                                    data = line[5:].strip()
                                elif line.startswith("id:"):
                                    event_id = line[3:].strip()
                                elif line.startswith("retry:"):
                                    retry = int(line[6:].strip())
                            
                            if data:
                                try:
                                    parsed_data = json.loads(data)
                                    event = TaskResponseEvent(
                                        event=event_type,
                                        data=parsed_data,
                                        id=event_id,
                                        retry=retry
                                    )
                                    yield event
                                    
                                    # If this is an end event, we're done
                                    if event_type == "end":
                                        return
                                except json.JSONDecodeError:
                                    # If data isn't JSON, yield it as a string
                                    event = TaskResponseEvent(
                                        event=event_type,
                                        data=data,
                                        id=event_id,
                                        retry=retry
                                    )
                                    yield event
        
        except httpx.HTTPStatusError as e:
            # Handle HTTP errors
            error_data = {
                "code": f"http_error_{e.response.status_code}",
                "message": f"HTTP error: {e.response.status_code} {e.response.reason_phrase}"
            }
            
            # Try to parse error response
            try:
                error_body = e.response.json()
                if isinstance(error_body, dict):
                    error_data.update(error_body)
            except:
                pass
            
            yield TaskResponseEvent(event="error", data=error_data)
        
        except httpx.RequestError as e:
            # Handle request errors (connection issues, timeouts, etc.)
            yield TaskResponseEvent(
                event="error",
                data={
                    "code": "request_error",
                    "message": f"Request error: {str(e)}"
                }
            )


# -----------------------------------------------------------------------------
# Server Implementation
# -----------------------------------------------------------------------------

class A2AServer:
    """
    Server implementation for the A2A protocol.
    """
    
    def __init__(self, agent_card: AgentCard, rate_limit_per_minute: int = 60):
        """
        Initialize the A2A server.
        
        Args:
            agent_card: The AgentCard describing this agent
            rate_limit_per_minute: Maximum requests per minute per client
        """
        self.agent_card = agent_card
        self.app = FastAPI(title=f"A2A Server - {agent_card.name}", version=agent_card.version)
        self.rate_limiter = RateLimiter(rate_limit_per_minute)
        self.security = HTTPBearer(auto_error=False)
        
        # Register routes
        self.setup_routes()
    
    def setup_routes(self):
        """Set up the FastAPI routes"""
        
        @self.app.get("/agent-card")
        async def get_agent_card():
            """Return this agent's capability card"""
            return self.agent_card.dict(exclude_none=True)
        
        @self.app.post("/")
        async def handle_task(
            request: Request,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Handle incoming task requests"""
            # Get client IP for rate limiting
            client_ip = request.client.host if request.client else "unknown"
            
            # Check rate limit
            token = credentials.credentials if credentials else None
            if not self.rate_limiter.check_rate_limit(token, client_ip):
                return Response(
                    content=json.dumps({
                        "code": "rate_limit_exceeded",
                        "message": "Rate limit exceeded",
                        "retry_after": 60
                    }),
                    media_type="application/json",
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    headers={"Retry-After": "60"}
                )
            
            # Parse and validate request
            try:
                body = await request.json()
                task_request = TaskRequest(**body)
            except ValidationError as e:
                return Response(
                    content=json.dumps({
                        "code": "invalid_request",
                        "message": "Invalid request format",
                        "details": {"validation_errors": e.errors()}
                    }),
                    media_type="application/json",
                    status_code=status.HTTP_400_BAD_REQUEST
                )
            
            # Check authentication if required
            if self.agent_card.auth_required:
                if not credentials:
                    return Response(
                        content=json.dumps({
                            "code": "authentication_required",
                            "message": "Authentication is required"
                        }),
                        media_type="application/json",
                        status_code=status.HTTP_401_UNAUTHORIZED
                    )
                
                # Here you would validate the token
                # This is a placeholder for your actual auth logic
                if not self._validate_token(credentials.credentials):
                    return Response(
                        content=json.dumps({
                            "code": "invalid_token",
                            "message": "Invalid authentication token"
                        }),
                        media_type="application/json",
                        status_code=status.HTTP_401_UNAUTHORIZED
                    )
            
            # Validate payload against schema
            try:
                jsonschema.validate(instance=task_request.payload, schema=self.agent_card.schema)
            except jsonschema.exceptions.ValidationError as e:
                return Response(
                    content=json.dumps({
                        "code": "validation_error",
                        "message": f"Payload validation failed: {str(e)}",
                        "details": {"validation_error": str(e)}
                    }),
                    media_type="application/json",
                    status_code=status.HTTP_400_BAD_REQUEST
                )
            
            # Process the task and return SSE response
            return EventSourceResponse(
                self._process_task(task_request),
                media_type="text/event-stream"
            )
    
    def _validate_token(self, token: str) -> bool:
        """
        Validate an authentication token.
        
        Args:
            token: The token to validate
            
        Returns:
            True if the token is valid, False otherwise
        """
        # This is a placeholder for your actual token validation logic
        # You should implement this based on your authentication system
        return True
    
    async def _process_task(self, task_request: TaskRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a task request and generate SSE events.
        
        Args:
            task_request: The task request to process
            
        Yields:
            Dictionaries representing SSE events
        """
        # This is where you would implement your task processing logic
        # For now, we'll just yield some example events
        
        try:
            # Initial acknowledgment
            yield {
                "event": "data",
                "data": json.dumps({
                    "message": f"Processing task: {task_request.capability}",
                    "status": "started"
                })
            }
            
            # Simulate some processing time
            await asyncio.sleep(1)
            
            # Example: dispatch to different handlers based on capability
            if task_request.capability == "echo":
                # Simple echo capability
                yield {
                    "event": "data",
                    "data": json.dumps({
                        "message": "Echo capability invoked",
                        "payload": task_request.payload
                    })
                }
            elif task_request.capability == "process_text":
                # Example text processing capability with multiple updates
                text = task_request.payload.get("text", "")
                
                # Send word count
                yield {
                    "event": "data",
                    "data": json.dumps({
                        "step": "word_count",
                        "result": len(text.split())
                    })
                }
                
                await asyncio.sleep(0.5)
                
                # Send character count
                yield {
                    "event": "data",
                    "data": json.dumps({
                        "step": "character_count",
                        "result": len(text)
                    })
                }
                
                await asyncio.sleep(0.5)
                
                # Send uppercase version
                yield {
                    "event": "data",
                    "data": json.dumps({
                        "step": "uppercase",
                        "result": text.upper()
                    })
                }
            else:
                # Unknown capability
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "code": "unknown_capability",
                        "message": f"Unknown capability: {task_request.capability}"
                    })
                }
                return
            
            # Final success event
            yield {
                "event": "end",
                "data": json.dumps({
                    "status": "completed",
                    "message": f"Task {task_request.capability} completed successfully"
                })
            }
            
        except Exception as e:
            # Handle any unexpected errors
            logger.exception(f"Error processing task: {e}")
            yield {
                "event": "error",
                "data": json.dumps({
                    "code": "internal_error",
                    "message": f"Internal server error: {str(e)}"
                })
            }
    
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
# Utility Functions
# -----------------------------------------------------------------------------

async def stream_processor(events_generator):
    """
    Process a stream of events from an A2A client.
    
    Args:
        events_generator: AsyncGenerator yielding TaskResponseEvent objects
        
    Returns:
        Aggregated results from the stream
    """
    results = []
    errors = []
    
    async for event in events_generator:
        if event.event == "error":
            errors.append(event.data)
            logger.error(f"Error event received: {event.data}")
        elif event.event == "data":
            results.append(event.data)
            logger.info(f"Data event received: {event.data}")
        elif event.event == "end":
            logger.info(f"End event received: {event.data}")
            break
    
    return {
        "results": results,
        "errors": errors,
        "success": len(errors) == 0
    }


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    
    # Example schema for a simple agent
    example_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to process"
            },
            "options": {
                "type": "object",
                "properties": {
                    "uppercase": {"type": "boolean"},
                    "count_words": {"type": "boolean"}
                }
            }
        },
        "required": ["text"]
    }
    
    # Create an agent card
    agent_card = AgentCard(
        name="ExampleAgent",
        version="1.0.0",
        endpoint="http://localhost:8000",
        schema=example_schema,
        description="An example agent that processes text"
    )
    
    # Example server
    async def run_server():
        server = A2AServer(agent_card)
        server.run()
    
    # Example client usage
    async def run_client():
        # Wait for server to start
        await asyncio.sleep(2)
        
        client = A2AClient(agent_card)
        
        # Invoke the process_text capability
        events = client.invoke(
            capability="process_text",
            payload={"text": "Hello, Agent-to-Agent Protocol!"}
        )
        
        # Process the events
        results = await stream_processor(events)
        print("Results:", results)
    
    # Run server and client
    loop = asyncio.get_event_loop()
    
    # In a real application, you would run the server and client separately
    # For this example, we'll just run the client
    loop.run_until_complete(run_client())
