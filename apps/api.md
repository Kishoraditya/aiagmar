# Project Analysis and API Endpoint Implementation Plan

## Current Project Analysis

Based on the codebase context, I can see a well-structured AI research assistant system with the following components:

### Core Components

1. **MCP Servers/Clients**: Wrappers for various Model Context Protocol servers (Brave Search, EverArt, Fetch, Filesystem, Memory)
2. **Agent System**: Specialized agents (Manager, Research, Summary, etc.) that use MCPs to perform tasks
3. **Workflow System**: Orchestrates agents to complete complex research tasks
4. **Protocol Implementations**: A2A, ANP, ACP for agent interoperability
5. **Testing Framework**: Comprehensive test suite for all components

### Project Structure

- `apps/` - Core application logic
  - `agents/` - Agent implementations
  - `mcps/` - MCP client wrappers
  - `protocols/` - Protocol implementations
  - `utils/` - Utility functions
  - `workflows/` - Workflow orchestration
- `services/` - MCP server implementations
- `tests/` - Test suite

### Current State

The project has a solid foundation with well-defined components, but lacks API endpoints to expose functionality to external clients or a web interface. The Wagtail app is mentioned but not clearly integrated with the core functionality.

## API Endpoint Implementation Plan

### Approach

1. **API Framework Selection**:
   - FastAPI would be ideal due to its modern features, automatic OpenAPI documentation, and async support
   - Flask could be an alternative if simplicity is preferred

2. **API Structure**:
   - RESTful endpoints for core functionality
   - WebSocket endpoints for real-time updates during research
   - Authentication and rate limiting middleware

3. **Integration Strategy**:
   - Create a new module `apps/api/` for API-specific code
   - Use dependency injection to connect API endpoints with existing components
   - Implement adapters between Wagtail and core functionality

### Prioritization

1. **First Priority**: Core research functionality endpoints
2. **Second Priority**: Agent management and configuration endpoints
3. **Third Priority**: Wagtail integration
4. **Fourth Priority**: Advanced features (streaming, webhooks, etc.)

## Implementation Plan

### 1. Create API Module Structure

```bash
apps/api/
├── __init__.py
├── main.py             # FastAPI app initialization
├── middleware/         # Auth, logging, error handling
├── models/             # Pydantic models for requests/responses
├── routers/            # API route definitions
│   ├── research.py     # Research endpoints
│   ├── agents.py       # Agent management endpoints
│   ├── mcps.py         # MCP configuration endpoints
│   └── admin.py        # Admin operations
├── services/           # API-specific business logic
└── utils/              # API-specific utilities
```

### 2. Core API Endpoints Implementation

#### Research Endpoints

```python
# apps/api/routers/research.py
from fastapi import APIRouter, Depends, BackgroundTasks, WebSocket
from apps.workflows.research_workflow import ResearchWorkflow
from apps.api.models.research import ResearchRequest, ResearchResponse
from apps.api.services.research_service import ResearchService

router = APIRouter(prefix="/research", tags=["research"])

@router.post("/", response_model=ResearchResponse)
async def create_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    research_service: ResearchService = Depends()
):
    """Start a new research task"""
    task_id = research_service.start_research(request.query, request.preferences)
    background_tasks.add_task(research_service.process_research, task_id)
    return {"task_id": task_id, "status": "processing"}

@router.get("/{task_id}", response_model=ResearchResponse)
async def get_research(task_id: str, research_service: ResearchService = Depends()):
    """Get research results by task ID"""
    return research_service.get_research(task_id)

@router.websocket("/ws/{task_id}")
async def research_updates(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time research updates"""
    await websocket.accept()
    # Subscribe to updates for this task_id
    # Send updates as they occur
```

#### Agent Management Endpoints

```python
# apps/api/routers/agents.py
from fastapi import APIRouter, Depends
from apps.api.models.agents import AgentConfig, AgentStatus
from apps.api.services.agent_service import AgentService

router = APIRouter(prefix="/agents", tags=["agents"])

@router.get("/", response_model=list[AgentStatus])
async def list_agents(agent_service: AgentService = Depends()):
    """List all available agents and their status"""
    return agent_service.list_agents()

@router.get("/{agent_id}", response_model=AgentStatus)
async def get_agent(agent_id: str, agent_service: AgentService = Depends()):
    """Get agent status by ID"""
    return agent_service.get_agent(agent_id)

@router.post("/{agent_id}/configure", response_model=AgentStatus)
async def configure_agent(
    agent_id: str, 
    config: AgentConfig, 
    agent_service: AgentService = Depends()
):
    """Configure an agent"""
    return agent_service.configure_agent(agent_id, config)
```

### 3. Service Layer Implementation

```python
# apps/api/services/research_service.py
import uuid
from typing import Dict, Any, Optional
from apps.workflows.research_workflow import ResearchWorkflow
from apps.agents.manager_agent import ManagerAgent
# Import other agents

class ResearchService:
    def __init__(self):
        # Initialize agents and workflow
        self.manager_agent = ManagerAgent()
        # Initialize other agents
        self.workflow = ResearchWorkflow(
            manager_agent=self.manager_agent,
            # Other agents
        )
        self.tasks = {}  # In-memory store of tasks (replace with database in production)
        
    def start_research(self, query: str, preferences: Optional[Dict[str, Any]] = None) -> str:
        """Start a new research task"""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "query": query,
            "preferences": preferences or {},
            "status": "queued",
            "result": None
        }
        return task_id
        
    def process_research(self, task_id: str):
        """Process a research task (runs in background)"""
        task = self.tasks[task_id]
        self.tasks[task_id]["status"] = "processing"
        try:
            result = self.workflow.execute(task["query"], **task["preferences"])
            self.tasks[task_id]["result"] = result
            self.tasks[task_id]["status"] = "completed"
        except Exception as e:
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["error"] = str(e)
            
    def get_research(self, task_id: str) -> Dict[str, Any]:
        """Get research results by task ID"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        return self.tasks[task_id]
```

### 4. Wagtail Integration

Create a bridge between the Wagtail app and the core functionality:

```python
# apps/api/integrations/wagtail_bridge.py
from typing import Dict, Any, Optional
from apps.api.services.research_service import ResearchService

class WagtailBridge:
    """Bridge between Wagtail and core research functionality"""
    
    def __init__(self):
        self.research_service = ResearchService()
    
    def start_research_from_wagtail(self, query: str, user_id: str, preferences: Optional[Dict[str, Any]] = None) -> str:
        """Start research from a Wagtail page or component"""
        task_id = self.research_service.start_research(query, preferences)
        # Store user_id association for permissions
        return task_id
    
    def get_research_for_wagtail(self, task_id: str, user_id: str) -> Dict[str, Any]:
        """Get research results formatted for Wagtail display"""
        result = self.research_service.get_research(task_id)
        # Format result for Wagtail display
        return result
```

In the Wagtail app:

```python
# wagtail_app/views.py
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from apps.api.integrations.wagtail_bridge import WagtailBridge

bridge = WagtailBridge()

@require_POST
def start_research(request):
    """Django view to start research from Wagtail"""
    query = request.POST.get('query')
    user_id = request.user.id
    preferences = request.POST.get('preferences', {})
    
    task_id = bridge.start_research_from_wagtail(query, user_id, preferences)
    return JsonResponse({"task_id": task_id})

def get_research(request, task_id):
    """Django view to get research results"""
    user_id = request.user.id
    result = bridge.get_research_for_wagtail(task_id, user_id)
    return JsonResponse(result)
```

### 5. Main FastAPI Application

```python
# apps/api/main.py
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from apps.api.middleware.auth import get_current_user
from apps.api.routers import research, agents, mcps, admin

app = FastAPI(title="Research Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(research.router)
app.include_router(agents.router)
app.include_router(mcps.router)
app.include_router(admin.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Research Assistant API"}
```

## Database Integration

For production, replace in-memory storage with a database:

```python
# apps/api/database.py
from sqlalchemy import create_engine, Column, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./research_api.db"
# For production: use PostgreSQL or other production DB

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ResearchTask(Base):
    __tablename__ = "research_tasks"
    
    id = Column(String, primary_key=True)
    query = Column(String)
    preferences = Column(JSON)
    status = Column(String)
    result = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    user_id = Column(String, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)
```

## Implementation Steps and Timeline

### Week 1: Core API Setup

1. Set up FastAPI application structure
2. Implement research endpoints
3. Create service layer for research workflow
4. Add basic authentication

### Week 2: Agent and MCP Management

1. Implement agent management endpoints
2. Add MCP configuration endpoints
3. Create admin endpoints
4. Implement database integration

### Week 3: Wagtail Integration

1. Create Wagtail bridge
2. Implement Wagtail views and templates
3. Add JavaScript for frontend interaction
4. Implement WebSocket for real-time updates

### Week 4: Testing and Refinement

1. Write API tests
2. Implement rate limiting and security features
3. Add documentation
4. Performance optimization

## Conclusion and Recommendations

1. **Start Small**: Begin with core research endpoints to get the API working quickly
2. **Use Dependency Injection**: Keep the API loosely coupled from core components
3. **Database First**: Implement database storage early to avoid data loss
4. **Test Thoroughly**: Extend the existing test suite to cover API endpoints
5. **Document Well**: Use FastAPI's automatic documentation and add usage examples
6. **Consider Scaling**: Design with horizontal scaling in mind from the beginning

This plan provides a structured approach to implementing API endpoints that integrate with both the core functionality and the Wagtail app, while maintaining the existing architecture and leveraging the comprehensive test suite.
