"""
Unit tests for the Manager Agent.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from apps.agents.manager_agent import ManagerAgent
from apps.agents.research_agent import ResearchAgent
from apps.agents.summary_agent import SummaryAgent
from apps.agents.verification_agent import VerificationAgent
from apps.agents.pre_response_agent import PreResponseAgent
from apps.agents.image_generation_agent import ImageGenerationAgent
from apps.agents.file_manager_agent import FileManagerAgent
from apps.mcps.memory_mcp import MemoryMCP
from apps.utils.exceptions import WorkflowError, AgentCommunicationError

class TestManagerAgent:
    """Test suite for ManagerAgent class."""

    @pytest.fixture
    def memory_mcp(self):
        """Fixture to create a mock MemoryMCP."""
        mock_memory = MagicMock(spec=MemoryMCP)
        mock_memory.store_memory = MagicMock()
        mock_memory.retrieve_memory = MagicMock()
        mock_memory.list_memories = MagicMock()
        mock_memory.search_memories = MagicMock()
        return mock_memory

    @pytest.fixture
    def research_agent(self):
        """Fixture to create a mock ResearchAgent."""
        mock_agent = MagicMock(spec=ResearchAgent)
        mock_agent.search_web = AsyncMock()
        mock_agent.fetch_content = AsyncMock()
        mock_agent.analyze_search_results = AsyncMock()
        return mock_agent

    @pytest.fixture
    def summary_agent(self):
        """Fixture to create a mock SummaryAgent."""
        mock_agent = MagicMock(spec=SummaryAgent)
        mock_agent.summarize_content = AsyncMock()
        mock_agent.generate_key_points = AsyncMock()
        return mock_agent

    @pytest.fixture
    def verification_agent(self):
        """Fixture to create a mock VerificationAgent."""
        mock_agent = MagicMock(spec=VerificationAgent)
        mock_agent.verify_facts = AsyncMock()
        mock_agent.check_sources = AsyncMock()
        return mock_agent

    @pytest.fixture
    def pre_response_agent(self):
        """Fixture to create a mock PreResponseAgent."""
        mock_agent = MagicMock(spec=PreResponseAgent)
        mock_agent.clarify_query = AsyncMock()
        mock_agent.present_plan = AsyncMock()
        return mock_agent

    @pytest.fixture
    def image_generation_agent(self):
        """Fixture to create a mock ImageGenerationAgent."""
        mock_agent = MagicMock(spec=ImageGenerationAgent)
        mock_agent.generate_image_from_research = AsyncMock()
        mock_agent.generate_diagram = AsyncMock()
        return mock_agent

    @pytest.fixture
    def file_manager_agent(self):
        """Fixture to create a mock FileManagerAgent."""
        mock_agent = MagicMock(spec=FileManagerAgent)
        mock_agent.save_research_content = AsyncMock()
        mock_agent.save_summary = AsyncMock()
        mock_agent.save_image = AsyncMock()
        mock_agent.create_research_package = AsyncMock()
        mock_agent.export_research_to_markdown = AsyncMock()
        return mock_agent

    @pytest.fixture
    def manager_agent(self, memory_mcp, research_agent, summary_agent, verification_agent, 
                     pre_response_agent, image_generation_agent, file_manager_agent):
        """Fixture to create a ManagerAgent instance with mock dependencies."""
        agent = ManagerAgent(
            name="manager",
            memory_mcp=memory_mcp,
            research_agent=research_agent,
            summary_agent=summary_agent,
            verification_agent=verification_agent,
            pre_response_agent=pre_response_agent,
            image_generation_agent=image_generation_agent,
            file_manager_agent=file_manager_agent
        )
        return agent

    def test_init(self, memory_mcp, research_agent, summary_agent, verification_agent, 
                 pre_response_agent, image_generation_agent, file_manager_agent):
        """Test initialization of ManagerAgent."""
        agent = ManagerAgent(
            name="manager",
            memory_mcp=memory_mcp,
            research_agent=research_agent,
            summary_agent=summary_agent,
            verification_agent=verification_agent,
            pre_response_agent=pre_response_agent,
            image_generation_agent=image_generation_agent,
            file_manager_agent=file_manager_agent
        )
        
        assert agent.name == "manager"
        assert agent.memory_mcp == memory_mcp
        assert agent.research_agent == research_agent
        assert agent.summary_agent == summary_agent
        assert agent.verification_agent == verification_agent
        assert agent.pre_response_agent == pre_response_agent
        assert agent.image_generation_agent == image_generation_agent
        assert agent.file_manager_agent == file_manager_agent

    @pytest.mark.asyncio
    async def test_initialize_workflow(self, manager_agent):
        """Test initializing the workflow."""
        # Call the method
        await manager_agent.initialize_workflow()
        
        # Verify memory was updated
        manager_agent.memory_mcp.store_memory.assert_called_with(
            "workflow_initialized", "true", namespace="manager"
        )
        
        # Verify file manager was initialized
        manager_agent.file_manager_agent.initialize_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_workflow_error(self, manager_agent):
        """Test handling errors during workflow initialization."""
        # Setup mock to raise exception
        manager_agent.file_manager_agent.initialize_workspace.side_effect = Exception("Initialization failed")
        
        # Call the method and expect error
        with pytest.raises(WorkflowError, match="Failed to initialize workflow"):
            await manager_agent.initialize_workflow()
        
        # Verify memory was not updated
        manager_agent.memory_mcp.store_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_user_query(self, manager_agent):
        """Test processing a user query."""
        # Setup mock responses
        manager_agent.pre_response_agent.clarify_query.return_value = {
            "query": "climate change impacts",
            "clarified": True,
            "additional_context": "Focus on coastal regions"
        }
        
        # Call the method
        query = "Tell me about climate change"
        result = await manager_agent.process_user_query(query)
        
        # Verify pre-response agent was called
        manager_agent.pre_response_agent.clarify_query.assert_called_once_with(query)
        
        # Verify memory was updated
        manager_agent.memory_mcp.store_memory.assert_called_with(
            "current_query", "climate change impacts", namespace="manager"
        )
        manager_agent.memory_mcp.store_memory.assert_any_call(
            "query_context", "Focus on coastal regions", namespace="manager"
        )
        
        # Verify result
        assert result["query"] == "climate change impacts"
        assert result["clarified"] is True
        assert result["additional_context"] == "Focus on coastal regions"

    @pytest.mark.asyncio
    async def test_process_user_query_error(self, manager_agent):
        """Test handling errors during query processing."""
        # Setup mock to raise exception
        manager_agent.pre_response_agent.clarify_query.side_effect = Exception("Clarification failed")
        
        # Call the method and expect error
        with pytest.raises(WorkflowError, match="Failed to process user query"):
            await manager_agent.process_user_query("test query")
        
        # Verify memory was not updated
        manager_agent.memory_mcp.store_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_research_plan(self, manager_agent):
        """Test creating a research plan."""
        # Setup mock responses
        manager_agent.pre_response_agent.present_plan.return_value = {
            "plan": [
                "Search for recent articles on climate change impacts",
                "Focus on coastal regions as specified",
                "Analyze and summarize findings",
                "Generate visual representation of key impacts"
            ],
            "approved": True
        }
        
        # Call the method
        query = "climate change impacts"
        context = "Focus on coastal regions"
        
        plan = await manager_agent.create_research_plan(query, context)
        
        # Verify pre-response agent was called
        manager_agent.pre_response_agent.present_plan.assert_called_once_with(query, context)
        
        # Verify memory was updated
        manager_agent.memory_mcp.store_memory.assert_called_with(
            "research_plan", str(plan["plan"]), namespace="manager"
        )
        
        # Verify result
        assert len(plan["plan"]) == 4
        assert plan["approved"] is True
        assert "coastal regions" in str(plan["plan"])

    @pytest.mark.asyncio
    async def test_create_research_plan_not_approved(self, manager_agent):
        """Test creating a research plan that is not approved."""
        # Setup mock responses
        manager_agent.pre_response_agent.present_plan.return_value = {
            "plan": ["Search for climate change articles"],
            "approved": False,
            "feedback": "Need more focus on economic impacts"
        }
        
        # Call the method
        plan = await manager_agent.create_research_plan("climate change", "")
        
        # Verify result
        assert plan["approved"] is False
        assert "feedback" in plan
        assert "economic impacts" in plan["feedback"]
        
        # Verify memory was not updated
        manager_agent.memory_mcp.store_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_research(self, manager_agent):
        """Test executing research."""
        # Setup mock responses
        manager_agent.research_agent.search_web.return_value = [
            {"title": "Climate Change Impact on Coastal Regions", "url": "https://example.com/article1"},
            {"title": "Rising Sea Levels and Coastal Cities", "url": "https://example.com/article2"}
        ]
        manager_agent.research_agent.fetch_content.return_value = "Article content about climate change impacts..."
        manager_agent.research_agent.analyze_search_results.return_value = {
            "relevant_results": [{"title": "Climate Change Impact on Coastal Regions", "url": "https://example.com/article1"}],
            "analysis": "The most relevant article discusses coastal erosion and flooding."
        }
        
        # Call the method
        query = "climate change impacts"
        context = "Focus on coastal regions"
        
        results = await manager_agent.execute_research(query, context)
        
        # Verify research agent methods were called
        manager_agent.research_agent.search_web.assert_called_once_with(query, context=context)
        manager_agent.research_agent.analyze_search_results.assert_called_once()
        
        # Verify file manager was called to save content
        manager_agent.file_manager_agent.save_research_content.assert_called_once()
        
        # Verify memory was updated
        manager_agent.memory_mcp.store_memory.assert_called_with(
            "research_results", str(results["relevant_results"]), namespace="manager"
        )
        
        # Verify result structure
        assert "relevant_results" in results
        assert "analysis" in results
        assert "content" in results
        assert "coastal erosion" in results["analysis"]

    @pytest.mark.asyncio
    async def test_execute_research_error(self, manager_agent):
        """Test handling errors during research execution."""
        # Setup mock to raise exception
        manager_agent.research_agent.search_web.side_effect = Exception("Search failed")
        
        # Call the method and expect error
        with pytest.raises(WorkflowError, match="Failed to execute research"):
            await manager_agent.execute_research("test query", "")
        
        # Verify memory was not updated
        manager_agent.memory_mcp.store_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_research(self, manager_agent):
        """Test verifying research findings."""
        # Setup mock responses
        research_content = "Climate change is causing sea levels to rise by 3mm per year."
        manager_agent.verification_agent.verify_facts.return_value = {
            "verified_facts": [
                {"fact": "Sea levels rising by 3mm per year", "verified": True, "source": "https://example.com/source"}
            ],
            "unverified_facts": [],
            "confidence": 0.9
        }
        
        # Call the method
        verification_results = await manager_agent.verify_research(research_content)
        
        # Verify verification agent was called
        manager_agent.verification_agent.verify_facts.assert_called_once_with(research_content)
        
        # Verify memory was updated
        manager_agent.memory_mcp.store_memory.assert_called_with(
            "verification_results", str(verification_results), namespace="manager"
        )
        
        # Verify result structure
        assert "verified_facts" in verification_results
        assert "unverified_facts" in verification_results
        assert "confidence" in verification_results
        assert verification_results["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_verify_research_with_unverified_facts(self, manager_agent):
        """Test verifying research with some unverified facts."""
        # Setup mock responses
        research_content = "Climate change is causing sea levels to rise by 30mm per year."
        manager_agent.verification_agent.verify_facts.return_value = {
            "verified_facts": [],
            "unverified_facts": [
                {"fact": "Sea levels rising by 30mm per year", "issue": "Exaggerated rate", "correction": "~3mm per year"}
            ],
            "confidence": 0.5
        }
        
        # Call the method
        verification_results = await manager_agent.verify_research(research_content)
        
        # Verify result contains unverified facts
        assert len(verification_results["unverified_facts"]) == 1
        assert "Exaggerated rate" in verification_results["unverified_facts"][0]["issue"]
        assert verification_results["confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_generate_summary(self, manager_agent):
        """Test generating a summary of research."""
        # Setup mock responses
        research_content = "Detailed content about climate change impacts on coastal regions..."
        verification_results = {
            "verified_facts": [{"fact": "Sea levels rising", "verified": True}],
            "unverified_facts": [],
            "confidence": 0.9
        }
        
        manager_agent.summary_agent.summarize_content.return_value = "Climate change is causing significant impacts on coastal regions through rising sea levels and increased flooding."
        manager_agent.summary_agent.generate_key_points.return_value = [
            "Sea levels are rising due to climate change",
            "Coastal regions are experiencing increased flooding",
            "Infrastructure damage is a major concern"
        ]
        
        # Call the method
        summary = await manager_agent.generate_summary(research_content, verification_results)
        
        # Verify summary agent was called
        manager_agent.summary_agent.summarize_content.assert_called_once_with(research_content)
        manager_agent.summary_agent.generate_key_points.assert_called_once_with(research_content)
        
        # Verify file manager was called to save summary
        manager_agent.file_manager_agent.save_summary.assert_called_once()
        
        # Verify memory was updated
        manager_agent.memory_mcp.store_memory.assert_called_with(
            "summary", summary["summary"], namespace="manager"
        )
        
        # Verify result structure
        assert "summary" in summary
        assert "key_points" in summary
        assert "coastal regions" in summary["summary"]
        assert len(summary["key_points"]) == 3

    @pytest.mark.asyncio
    async def test_generate_summary_error(self, manager_agent):
        """Test handling errors during summary generation."""
        # Setup mock to raise exception
        manager_agent.summary_agent.summarize_content.side_effect = Exception("Summarization failed")
        
        # Call the method and expect error
        with pytest.raises(WorkflowError, match="Failed to generate summary"):
            await manager_agent.generate_summary("research content", {})
        
        # Verify memory was not updated
        manager_agent.memory_mcp.store_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_visuals(self, manager_agent):
        """Test generating visuals for research."""
        # Setup mock responses
        research_content = "Climate change is causing sea levels to rise, leading to coastal flooding."
        summary = {
            "summary": "Climate change impacts on coastal regions include rising sea levels and flooding.",
            "key_points": ["Sea levels rising", "Coastal flooding", "Infrastructure damage"]
        }
        
        manager_agent.image_generation_agent.generate_image_from_research.return_value = "https://example.com/climate-image.jpg"
        manager_agent.image_generation_agent.generate_diagram.return_value = "https://example.com/climate-diagram.jpg"
        
        # Call the method
        visuals = await manager_agent.generate_visuals(research_content, summary, "climate change impacts")
        
        # Verify image generation agent was called
        manager_agent.image_generation_agent.generate_image_from_research.assert_called_once_with(
            research_content, query="climate change impacts"
        )
        
        # Verify file manager was called to save image
        manager_agent.file_manager_agent.save_image.assert_called()
        
        # Verify memory was updated
        manager_agent.memory_mcp.store_memory.assert_called_with(
            "visuals", str(visuals), namespace="manager"
        )
        
        # Verify result structure
        assert "main_image_url" in visuals
        assert "diagram_url" in visuals
        assert visuals["main_image_url"] == "https://example.com/climate-image.jpg"

    @pytest.mark.asyncio
    async def test_generate_visuals_with_diagram_data(self, manager_agent):
        """Test generating visuals with structured diagram data."""
        # Setup mock responses
        research_content = "Climate change causes rising temperatures, which lead to melting ice caps and rising sea levels."
        summary = {
            "summary": "Climate change impacts include rising temperatures, melting ice, and sea level rise.",
            "key_points": ["Rising temperatures", "Melting ice caps", "Rising sea levels"]
        }
        
        # Create diagram data based on key points
        diagram_data = {
            "title": "Climate Change Effects",
            "nodes": ["Climate Change", "Rising Temperatures", "Melting Ice Caps", "Rising Sea Levels", "Coastal Flooding"],
            "connections": [
                ["Climate Change", "Rising Temperatures"],
                ["Rising Temperatures", "Melting Ice Caps"],
                ["Melting Ice Caps", "Rising Sea Levels"],
                ["Rising Sea Levels", "Coastal Flooding"]
            ]
        }
        
        manager_agent.image_generation_agent.generate_image_from_research.return_value = "https://example.com/climate-image.jpg"
        manager_agent.image_generation_agent.generate_diagram.return_value = "https://example.com/climate-diagram.jpg"
        
        # Call the method with diagram data
        visuals = await manager_agent.generate_visuals(
            research_content, summary, "climate change impacts", diagram_data=diagram_data
        )
        
        # Verify diagram generation was called with the data
        manager_agent.image_generation_agent.generate_diagram.assert_called_once_with(diagram_data)
        
        # Verify result includes diagram URL
        assert visuals["diagram_url"] == "https://example.com/climate-diagram.jpg"

    @pytest.mark.asyncio
    async def test_generate_visuals_error(self, manager_agent):
        """Test handling errors during visual generation."""
        # Setup mock to raise exception
        manager_agent.image_generation_agent.generate_image_from_research.side_effect = Exception("Image generation failed")
        
        # Call the method and expect error
        with pytest.raises(WorkflowError, match="Failed to generate visuals"):
            await manager_agent.generate_visuals("research content", {}, "test query")
        
        # Verify memory was not updated
        manager_agent.memory_mcp.store_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_compile_research_package(self, manager_agent):
        """Test compiling a complete research package."""
        # Setup mock responses
        query = "climate change impacts"
        research_results = {
            "relevant_results": [{"title": "Climate Impact Study", "url": "https://example.com/study"}],
            "analysis": "Analysis of climate impacts",
            "content": "Detailed research content"
        }
        verification_results = {
            "verified_facts": [{"fact": "Sea levels rising", "verified": True}],
            "confidence": 0.9
        }
        summary = {
            "summary": "Summary of climate impacts",
            "key_points": ["Point 1", "Point 2"]
        }
        visuals = {
            "main_image_url": "https://example.com/image.jpg",
            "diagram_url": "https://example.com/diagram.jpg"
        }
        
        manager_agent.file_manager_agent.create_research_package.return_value = {
            "query": query,
            "research_content": "Detailed research content",
            "summary": "Summary of climate impacts",
            "image_url": "https://example.com/image.jpg",
            "image_description": "Climate impact visualization"
        }
        
        manager_agent.file_manager_agent.export_research_to_markdown.return_value = "# Research: Climate Change Impacts\n\n## Summary\n..."
        
        # Call the method
        package = await manager_agent.compile_research_package(
            query, research_results, verification_results, summary, visuals
        )
        
        # Verify file manager was called
        manager_agent.file_manager_agent.create_research_package.assert_called_once_with(query)
        manager_agent.file_manager_agent.export_research_to_markdown.assert_called_once()
        manager_agent.file_manager_agent.save_markdown_export.assert_called_once()
        
        # Verify memory was updated
        manager_agent.memory_mcp.store_memory.assert_called_with(
            "research_package", str(package), namespace="manager"
        )
        
        # Verify result structure
        assert "query" in package
        assert "research_content" in package
        assert "summary" in package
        assert "image_url" in package
        assert "markdown_export" in package
        assert "export_file" in package

    @pytest.mark.asyncio
    async def test_compile_research_package_error(self, manager_agent):
        """Test handling errors during research package compilation."""
        # Setup mock to raise exception
        manager_agent.file_manager_agent.create_research_package.side_effect = Exception("Package creation failed")
        
        # Call the method and expect error
        with pytest.raises(WorkflowError, match="Failed to compile research package"):
            await manager_agent.compile_research_package("query", {}, {}, {}, {})
        
        # Verify memory was not updated
        manager_agent.memory_mcp.store_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_full_workflow(self, manager_agent):
        """Test executing the full research workflow."""
        # Setup mock responses for each step
        # Initialize workflow
        manager_agent.initialize_workflow = AsyncMock()
        
        # Process query
        manager_agent.process_user_query = AsyncMock(return_value={
            "query": "climate change impacts",
            "clarified": True,
            "additional_context": "Focus on coastal regions"
        })
        
        # Create plan
        manager_agent.create_research_plan = AsyncMock(return_value={
            "plan": ["Step 1", "Step 2"],
            "approved": True
        })
        
        # Execute research
        manager_agent.execute_research = AsyncMock(return_value={
            "relevant_results": [{"title": "Study", "url": "https://example.com"}],
            "analysis": "Analysis",
            "content": "Research content"
        })
        
        # Verify research
        manager_agent.verify_research = AsyncMock(return_value={
            "verified_facts": [{"fact": "Fact 1", "verified": True}],
            "confidence": 0.9
        })
        
        # Generate summary
        manager_agent.generate_summary = AsyncMock(return_value={
            "summary": "Summary",
            "key_points": ["Point 1", "Point 2"]
        })
        
        # Generate visuals
        manager_agent.generate_visuals = AsyncMock(return_value={
            "main_image_url": "https://example.com/image.jpg",
            "diagram_url": "https://example.com/diagram.jpg"
        })
        
        # Compile package
        manager_agent.compile_research_package = AsyncMock(return_value={
            "query": "climate change impacts",
            "research_content": "Content",
            "summary": "Summary",
            "image_url": "https://example.com/image.jpg",
            "markdown_export": "# Research",
            "export_file": "export_123"
        })
        
        # Call the method
        result = await manager_agent.execute_full_workflow("Tell me about climate change")
        
        # Verify all steps were called in sequence
        manager_agent.initialize_workflow.assert_called_once()
        manager_agent.process_user_query.assert_called_once_with("Tell me about climate change")
        manager_agent.create_research_plan.assert_called_once()
        manager_agent.execute_research.assert_called_once()
        manager_agent.verify_research.assert_called_once()
        manager_agent.generate_summary.assert_called_once()
        manager_agent.generate_visuals.assert_called_once()
        manager_agent.compile_research_package.assert_called_once()
        
        # Verify result structure
        assert "query" in result
        assert "research_package" in result
        assert "markdown_export" in result
        assert "export_file" in result

    @pytest.mark.asyncio
    async def test_execute_full_workflow_plan_not_approved(self, manager_agent):
        """Test workflow when research plan is not approved."""
        # Setup mocks
        manager_agent.initialize_workflow = AsyncMock()
        manager_agent.process_user_query = AsyncMock(return_value={"query": "test query"})
        manager_agent.create_research_plan = AsyncMock(return_value={
            "plan": ["Step 1"],
            "approved": False,
            "feedback": "Need more focus"
        })
        
        # Call the method
        result = await manager_agent.execute_full_workflow("test query")
        
        # Verify early steps were called
        manager_agent.initialize_workflow.assert_called_once()
        manager_agent.process_user_query.assert_called_once()
        manager_agent.create_research_plan.assert_called_once()
        
        # Verify later steps were not called
        manager_agent.execute_research.assert_not_called()
        manager_agent.verify_research.assert_not_called()
        
        # Verify result contains feedback
        assert "status" in result
        assert result["status"] == "plan_not_approved"
        assert "feedback" in result
        assert result["feedback"] == "Need more focus"

    @pytest.mark.asyncio
    async def test_execute_full_workflow_error(self, manager_agent):
        """Test handling errors during full workflow execution."""
        # Setup mock to raise exception at research step
        manager_agent.initialize_workflow = AsyncMock()
        manager_agent.process_user_query = AsyncMock(return_value={"query": "test query"})
        manager_agent.create_research_plan = AsyncMock(return_value={"plan": ["Step 1"], "approved": True})
        manager_agent.execute_research = AsyncMock(side_effect=Exception("Research failed"))
        
        # Call the method
        result = await manager_agent.execute_full_workflow("test query")
        
        # Verify result contains error
        assert "status" in result
        assert result["status"] == "error"
        assert "error" in result
        assert "Research failed" in result["error"]
        
        # Verify steps were called until the error
        manager_agent.initialize_workflow.assert_called_once()
        manager_agent.process_user_query.assert_called_once()
        manager_agent.create_research_plan.assert_called_once()
        manager_agent.execute_research.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_workflow_status(self, manager_agent):
        """Test getting the current workflow status."""
        # Setup mock responses
        manager_agent.memory_mcp.retrieve_memory.side_effect = [
            "climate change impacts",  # current_query
            "true",                    # workflow_initialized
            "Summary of research"      # summary
        ]
        
        # Call the method
        status = await manager_agent.get_workflow_status()
        
        # Verify memory was queried
        manager_agent.memory_mcp.retrieve_memory.assert_any_call(
            "current_query", namespace="manager"
        )
        manager_agent.memory_mcp.retrieve_memory.assert_any_call(
            "workflow_initialized", namespace="manager"
        )
        
        # Verify result structure
        assert "current_query" in status
        assert "initialized" in status
        assert "completed_steps" in status
        assert status["current_query"] == "climate change impacts"
        assert status["initialized"] is True

    @pytest.mark.asyncio
    async def test_get_workflow_status_not_initialized(self, manager_agent):
        """Test getting workflow status when not initialized."""
        # Setup mock to raise exception for workflow_initialized
        def mock_retrieve(key, namespace=None):
            if key == "workflow_initialized":
                raise Exception("Memory not found")
            return None
        
        manager_agent.memory_mcp.retrieve_memory.side_effect = mock_retrieve
        
        # Call the method
        status = await manager_agent.get_workflow_status()
        
        # Verify result shows not initialized
        assert "initialized" in status
        assert status["initialized"] is False
        assert status["current_query"] is None
        assert len(status["completed_steps"]) == 0

    @pytest.mark.asyncio
    async def test_reset_workflow(self, manager_agent):
        """Test resetting the workflow state."""
        # Call the method
        await manager_agent.reset_workflow()
        
        # Verify memory namespace was cleared
        manager_agent.memory_mcp.clear_namespace.assert_called_once_with("manager")
        
        # Verify file manager was called to clean workspace
        manager_agent.file_manager_agent.clean_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_workflow_error(self, manager_agent):
        """Test handling errors during workflow reset."""
        # Setup mock to raise exception
        manager_agent.memory_mcp.clear_namespace.side_effect = Exception("Clear failed")
        
        # Call the method and expect error
        with pytest.raises(WorkflowError, match="Failed to reset workflow"):
            await manager_agent.reset_workflow()

    @pytest.mark.asyncio
    async def test_backup_workflow_state(self, manager_agent):
        """Test backing up the workflow state."""
        # Setup mock responses
        manager_agent.file_manager_agent.backup_workspace.return_value = "backup_20230615"
        
        # Call the method
        backup_id = await manager_agent.backup_workflow_state()
        
        # Verify file manager was called
        manager_agent.file_manager_agent.backup_workspace.assert_called_once()
        
        # Verify memory was updated
        manager_agent.memory_mcp.store_memory.assert_called_with(
            "last_backup", "backup_20230615", namespace="manager"
        )
        
        # Verify result
        assert backup_id == "backup_20230615"

    @pytest.mark.asyncio
    async def test_restore_workflow_state(self, manager_agent):
        """Test restoring the workflow state from backup."""
        # Setup mock responses
        manager_agent.file_manager_agent.restore_from_backup.return_value = True
        
        # Call the method
        success = await manager_agent.restore_workflow_state("backup_20230615")
        
        # Verify file manager was called
        manager_agent.file_manager_agent.restore_from_backup.assert_called_once_with("backup_20230615")
        
        # Verify result
        assert success is True

    @pytest.mark.asyncio
    async def test_restore_workflow_state_failure(self, manager_agent):
        """Test restoring workflow state when backup doesn't exist."""
        # Setup mock responses
        manager_agent.file_manager_agent.restore_from_backup.return_value = False
        
        # Call the method
        success = await manager_agent.restore_workflow_state("nonexistent_backup")
        
        # Verify result
        assert success is False

    @pytest.mark.asyncio
    async def test_get_research_history(self, manager_agent):
        """Test retrieving research history."""
        # Setup mock responses
        manager_agent.memory_mcp.list_memories.return_value = """
        current_query
        research_plan
        research_results
        summary
        """
        
        manager_agent.memory_mcp.retrieve_memory.side_effect = [
            "climate change impacts",                                # current_query
            "['Search for articles', 'Analyze findings']",           # research_plan
            "[{'title': 'Study', 'url': 'https://example.com'}]",    # research_results
            "Summary of climate impacts"                             # summary
        ]
        
        # Call the method
        history = await manager_agent.get_research_history()
        
        # Verify memory was queried
        manager_agent.memory_mcp.list_memories.assert_called_once_with(namespace="manager")
        
        # Verify result structure
        assert "queries" in history
        assert "climate change impacts" in history["queries"]
        assert "plans" in history
        assert "results" in history
        assert "summaries" in history
        assert "Summary of climate impacts" in history["summaries"]["climate change impacts"]

    @pytest.mark.asyncio
    async def test_get_research_history_empty(self, manager_agent):
        """Test retrieving research history when empty."""
        # Setup mock with empty response
        manager_agent.memory_mcp.list_memories.return_value = ""
        
        # Call the method
        history = await manager_agent.get_research_history()
        
        # Verify result is empty
        assert history["queries"] == []
        assert history["plans"] == {}
        assert history["results"] == {}
        assert history["summaries"] == {}

    @pytest.mark.asyncio
    async def test_get_research_package(self, manager_agent):
        """Test retrieving a specific research package."""
        # Setup mock responses
        manager_agent.memory_mcp.retrieve_memory.return_value = """
        {
            "query": "climate change impacts",
            "research_content": "Detailed content",
            "summary": "Summary",
            "image_url": "https://example.com/image.jpg",
            "export_file": "export_123"
        }
        """
        
        manager_agent.file_manager_agent.get_export_file.return_value = "# Research: Climate Change Impacts\n\n## Summary\n..."
        
        # Call the method
        query = "climate change impacts"
        package = await manager_agent.get_research_package(query)
        
        # Verify memory was queried
        manager_agent.memory_mcp.retrieve_memory.assert_called_once_with(
            "research_package", namespace="manager"
        )
        
        # Verify file manager was called
        manager_agent.file_manager_agent.get_export_file.assert_called_once_with(query)
        
        # Verify result structure
        assert "query" in package
        assert "research_content" in package
        assert "summary" in package
        assert "image_url" in package
        assert "markdown_export" in package
        assert package["query"] == "climate change impacts"

    @pytest.mark.asyncio
    async def test_get_research_package_not_found(self, manager_agent):
        """Test retrieving a research package that doesn't exist."""
        # Setup mock to raise exception
        manager_agent.memory_mcp.retrieve_memory.side_effect = Exception("Memory not found")
        
        # Call the method
        package = await manager_agent.get_research_package("nonexistent query")
        
        # Verify result is None
        assert package is None

    @pytest.mark.asyncio
    async def test_extract_diagram_data_from_research(self, manager_agent):
        """Test extracting diagram data from research content."""
        # Setup research content with causal relationships
        research_content = """
        Climate change is primarily caused by greenhouse gas emissions.
        These emissions lead to rising global temperatures.
        Rising temperatures cause ice caps to melt.
        Melting ice caps contribute to rising sea levels.
        Rising sea levels result in coastal flooding and erosion.
        """
        
        # Call the method
        diagram_data = await manager_agent.extract_diagram_data_from_research(
            research_content, "Climate Change Effects"
        )
        
        # Verify result structure
        assert "title" in diagram_data
        assert "nodes" in diagram_data
        assert "connections" in diagram_data
        assert diagram_data["title"] == "Climate Change Effects"
        
        # Verify nodes contain key concepts
        assert "Climate Change" in diagram_data["nodes"]
        assert "Greenhouse Gas Emissions" in diagram_data["nodes"]
        assert "Rising Temperatures" in diagram_data["nodes"]
        assert "Melting Ice Caps" in diagram_data["nodes"]
        assert "Rising Sea Levels" in diagram_data["nodes"]
        
        # Verify connections show causal relationships
        assert len(diagram_data["connections"]) >= 4
        assert ["Greenhouse Gas Emissions", "Rising Temperatures"] in diagram_data["connections"] or \
               ["Climate Change", "Rising Temperatures"] in diagram_data["connections"]
        assert ["Rising Temperatures", "Melting Ice Caps"] in diagram_data["connections"]
        assert ["Melting Ice Caps", "Rising Sea Levels"] in diagram_data["connections"]
        assert ["Rising Sea Levels", "Coastal Flooding"] in diagram_data["connections"] or \
               ["Rising Sea Levels", "Coastal Erosion"] in diagram_data["connections"]

    @pytest.mark.asyncio
    async def test_extract_diagram_data_insufficient_content(self, manager_agent):
        """Test extracting diagram data from insufficient content."""
        # Setup minimal research content
        research_content = "Climate change is a global issue."
        
        # Call the method
        diagram_data = await manager_agent.extract_diagram_data_from_research(
            research_content, "Climate Change"
        )
        
        # Verify basic structure is still returned
        assert "title" in diagram_data
        assert "nodes" in diagram_data
        assert "connections" in diagram_data
        assert diagram_data["title"] == "Climate Change"
        assert "Climate Change" in diagram_data["nodes"]
        assert len(diagram_data["nodes"]) >= 1
        
        # Connections might be minimal or empty
        assert isinstance(diagram_data["connections"], list)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
