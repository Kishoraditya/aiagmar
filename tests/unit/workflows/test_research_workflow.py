"""
Unit tests for the Research Workflow.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import os
import json
from pathlib import Path

from apps.workflows.research_workflow import ResearchWorkflow
from apps.agents.manager_agent import ManagerAgent
from apps.agents.pre_response_agent import PreResponseAgent
from apps.agents.research_agent import ResearchAgent
from apps.agents.summary_agent import SummaryAgent
from apps.agents.verification_agent import VerificationAgent
from apps.agents.image_generation_agent import ImageGenerationAgent
from apps.agents.file_manager_agent import FileManagerAgent

from apps.mcps.brave_search_mcp import BraveSearchMCP
from apps.mcps.fetch_mcp import FetchMCP
from apps.mcps.memory_mcp import MemoryMCP
from apps.mcps.everart_mcp import EverArtMCP
from apps.mcps.filesystem_mcp import FilesystemMCP

from apps.utils.exceptions import WorkflowError, ValidationError


class TestResearchWorkflow:
    """Test suite for ResearchWorkflow class."""

    @pytest.fixture
    def mock_mcps(self):
        """Fixture to create mock MCP instances."""
        mock_brave = MagicMock(spec=BraveSearchMCP)
        mock_brave.web_search = AsyncMock()
        mock_brave.local_search = AsyncMock()

        mock_fetch = MagicMock(spec=FetchMCP)
        mock_fetch.fetch_url = AsyncMock()
        mock_fetch.fetch_text = AsyncMock()

        mock_memory = MagicMock(spec=MemoryMCP)
        mock_memory.store_memory = AsyncMock()
        mock_memory.retrieve_memory = AsyncMock()
        mock_memory.search_memories = AsyncMock()
        mock_memory.list_memories = AsyncMock()

        mock_everart = MagicMock(spec=EverArtMCP)
        mock_everart.generate_image = AsyncMock()
        mock_everart.describe_image = AsyncMock()

        mock_filesystem = MagicMock(spec=FilesystemMCP)
        mock_filesystem.write_file = AsyncMock()
        mock_filesystem.read_file = AsyncMock()
        mock_filesystem.list_directory = AsyncMock()
        mock_filesystem.create_directory = AsyncMock()

        return {
            "brave_search_mcp": mock_brave,
            "fetch_mcp": mock_fetch,
            "memory_mcp": mock_memory,
            "everart_mcp": mock_everart,
            "filesystem_mcp": mock_filesystem
        }

    @pytest.fixture
    def mock_agents(self, mock_mcps):
        """Fixture to create mock agent instances."""
        mock_manager = MagicMock(spec=ManagerAgent)
        mock_manager.plan_research = AsyncMock()
        mock_manager.coordinate_agents = AsyncMock()
        mock_manager.track_progress = AsyncMock()
        mock_manager.handle_errors = AsyncMock()

        mock_pre_response = MagicMock(spec=PreResponseAgent)
        mock_pre_response.clarify_query = AsyncMock()
        mock_pre_response.present_plan = AsyncMock()
        mock_pre_response.get_user_feedback = AsyncMock()

        mock_research = MagicMock(spec=ResearchAgent)
        mock_research.search_web = AsyncMock()
        mock_research.fetch_content = AsyncMock()
        mock_research.analyze_sources = AsyncMock()

        mock_summary = MagicMock(spec=SummaryAgent)
        mock_summary.summarize_text = AsyncMock()
        mock_summary.extract_key_points = AsyncMock()
        mock_summary.compare_documents = AsyncMock()

        mock_verification = MagicMock(spec=VerificationAgent)
        mock_verification.verify_fact = AsyncMock()
        mock_verification.cross_check_information = AsyncMock()
        mock_verification.evaluate_source_credibility = AsyncMock()

        mock_image_generation = MagicMock(spec=ImageGenerationAgent)
        mock_image_generation.generate_image = AsyncMock()
        mock_image_generation.create_diagram = AsyncMock()

        mock_file_manager = MagicMock(spec=FileManagerAgent)
        mock_file_manager.save_research = AsyncMock()
        mock_file_manager.organize_files = AsyncMock()
        mock_file_manager.create_research_directory = AsyncMock()

        return {
            "manager_agent": mock_manager,
            "pre_response_agent": mock_pre_response,
            "research_agent": mock_research,
            "summary_agent": mock_summary,
            "verification_agent": mock_verification,
            "image_generation_agent": mock_image_generation,
            "file_manager_agent": mock_file_manager
        }

    @pytest.fixture
    def research_workflow(self, mock_mcps, mock_agents):
        """Fixture to create a ResearchWorkflow instance with mock dependencies."""
        workflow = ResearchWorkflow(
            brave_search_mcp=mock_mcps["brave_search_mcp"],
            fetch_mcp=mock_mcps["fetch_mcp"],
            memory_mcp=mock_mcps["memory_mcp"],
            everart_mcp=mock_mcps["everart_mcp"],
            filesystem_mcp=mock_mcps["filesystem_mcp"]
        )
        
        # Replace the agents with mocks
        workflow.manager_agent = mock_agents["manager_agent"]
        workflow.pre_response_agent = mock_agents["pre_response_agent"]
        workflow.research_agent = mock_agents["research_agent"]
        workflow.summary_agent = mock_agents["summary_agent"]
        workflow.verification_agent = mock_agents["verification_agent"]
        workflow.image_generation_agent = mock_agents["image_generation_agent"]
        workflow.file_manager_agent = mock_agents["file_manager_agent"]
        
        return workflow

    def test_init(self, mock_mcps):
        """Test initialization of ResearchWorkflow."""
        workflow = ResearchWorkflow(
            brave_search_mcp=mock_mcps["brave_search_mcp"],
            fetch_mcp=mock_mcps["fetch_mcp"],
            memory_mcp=mock_mcps["memory_mcp"],
            everart_mcp=mock_mcps["everart_mcp"],
            filesystem_mcp=mock_mcps["filesystem_mcp"]
        )
        
        # Verify MCPs are set correctly
        assert workflow.brave_search_mcp == mock_mcps["brave_search_mcp"]
        assert workflow.fetch_mcp == mock_mcps["fetch_mcp"]
        assert workflow.memory_mcp == mock_mcps["memory_mcp"]
        assert workflow.everart_mcp == mock_mcps["everart_mcp"]
        assert workflow.filesystem_mcp == mock_mcps["filesystem_mcp"]
        
        # Verify agents are initialized
        assert isinstance(workflow.manager_agent, ManagerAgent)
        assert isinstance(workflow.pre_response_agent, PreResponseAgent)
        assert isinstance(workflow.research_agent, ResearchAgent)
        assert isinstance(workflow.summary_agent, SummaryAgent)
        assert isinstance(workflow.verification_agent, VerificationAgent)
        assert isinstance(workflow.image_generation_agent, ImageGenerationAgent)
        assert isinstance(workflow.file_manager_agent, FileManagerAgent)

    @pytest.mark.asyncio
    async def test_run_research_workflow_success(self, research_workflow):
        """Test running the research workflow successfully."""
        # Setup mock returns
        research_workflow.pre_response_agent.clarify_query.return_value = {
            "clarified_query": "Impact of climate change on global agriculture",
            "focus_areas": ["crop yields", "adaptation strategies", "food security"]
        }
        
        research_workflow.manager_agent.plan_research.return_value = {
            "plan": [
                {"step": "Search for recent studies on climate change and agriculture"},
                {"step": "Analyze impacts on different regions"},
                {"step": "Summarize adaptation strategies"}
            ],
            "estimated_time": "15 minutes"
        }
        
        research_workflow.pre_response_agent.present_plan.return_value = True  # User approves plan
        
        research_workflow.research_agent.search_web.return_value = {
            "search1": {"title": "Climate Change and Agriculture", "url": "https://example.com/1"},
            "search2": {"title": "Adapting Agriculture to Climate Change", "url": "https://example.com/2"}
        }
        
        research_workflow.research_agent.fetch_content.return_value = {
            "content1": "Climate change is affecting crop yields globally...",
            "content2": "Adaptation strategies include drought-resistant crops..."
        }
        
        research_workflow.summary_agent.summarize_text.return_value = "Climate change is reducing crop yields but adaptation strategies show promise..."
        
        research_workflow.verification_agent.cross_check_information.return_value = {
            "consensus": True,
            "agreement_level": 0.9,
            "sources": ["IPCC", "FAO"]
        }
        
        research_workflow.image_generation_agent.generate_image.return_value = "https://example.com/image.jpg"
        
        research_workflow.file_manager_agent.save_research.return_value = "/research/climate_agriculture/summary.md"
        
        # Call the method
        query = "How is climate change affecting agriculture?"
        result = await research_workflow.run_research(query)
        
        # Verify all expected agent methods were called
        research_workflow.pre_response_agent.clarify_query.assert_called_once_with(query)
        research_workflow.manager_agent.plan_research.assert_called_once()
        research_workflow.pre_response_agent.present_plan.assert_called_once()
        research_workflow.research_agent.search_web.assert_called()
        research_workflow.research_agent.fetch_content.assert_called()
        research_workflow.summary_agent.summarize_text.assert_called()
        research_workflow.verification_agent.cross_check_information.assert_called()
        research_workflow.file_manager_agent.save_research.assert_called()
        
        # Verify result structure
        assert "summary" in result
        assert "sources" in result
        assert "verified_facts" in result
        assert "visualizations" in result
        assert "file_path" in result
        
        # Verify memory was updated with final results
        research_workflow.memory_mcp.store_memory.assert_called_with(
            "research_result", str(result), namespace="research"
        )

    @pytest.mark.asyncio
    async def test_run_research_user_rejects_plan(self, research_workflow):
        """Test workflow when user rejects the research plan."""
        # Setup mock returns
        research_workflow.pre_response_agent.clarify_query.return_value = {
            "clarified_query": "Impact of climate change on global agriculture",
            "focus_areas": ["crop yields", "adaptation strategies"]
        }
        
        research_workflow.manager_agent.plan_research.return_value = {
            "plan": [
                {"step": "Search for recent studies on climate change and agriculture"},
                {"step": "Analyze impacts on different regions"}
            ],
            "estimated_time": "10 minutes"
        }
        
        # User rejects the plan
        research_workflow.pre_response_agent.present_plan.return_value = False
        
        # Call the method
        query = "How is climate change affecting agriculture?"
        result = await research_workflow.run_research(query)
        
        # Verify early methods were called
        research_workflow.pre_response_agent.clarify_query.assert_called_once_with(query)
        research_workflow.manager_agent.plan_research.assert_called_once()
        research_workflow.pre_response_agent.present_plan.assert_called_once()
        
        # Verify later methods were not called
        research_workflow.research_agent.search_web.assert_not_called()
        research_workflow.research_agent.fetch_content.assert_not_called()
        
        # Verify result indicates plan rejection
        assert "status" in result
        assert result["status"] == "cancelled"
        assert "reason" in result
        assert "user rejected" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_run_research_with_error_handling(self, research_workflow):
        """Test workflow with error handling during research."""
        # Setup mock returns
        research_workflow.pre_response_agent.clarify_query.return_value = {
            "clarified_query": "Impact of climate change on global agriculture",
            "focus_areas": ["crop yields", "adaptation strategies"]
        }
        
        research_workflow.manager_agent.plan_research.return_value = {
            "plan": [
                {"step": "Search for recent studies on climate change and agriculture"},
                {"step": "Analyze impacts on different regions"}
            ],
            "estimated_time": "10 minutes"
        }
        
        research_workflow.pre_response_agent.present_plan.return_value = True
        
        # Simulate error during search
        research_workflow.research_agent.search_web.side_effect = Exception("Search API error")
        
        # Setup error handling response
        research_workflow.manager_agent.handle_errors.return_value = {
            "action": "continue",
            "alternative_plan": [
                {"step": "Use cached data instead of live search"}
            ]
        }
        
        # Setup remaining successful steps
        research_workflow.research_agent.fetch_content.return_value = {
            "content1": "Using cached data on climate change and agriculture..."
        }
        
        research_workflow.summary_agent.summarize_text.return_value = "Based on available data, climate change impacts include..."
        
        # Call the method
        query = "How is climate change affecting agriculture?"
        result = await research_workflow.run_research(query)
        
        # Verify error handling was called
        research_workflow.manager_agent.handle_errors.assert_called_once()
        
        # Verify workflow continued with alternative plan
        research_workflow.research_agent.fetch_content.assert_called()
        research_workflow.summary_agent.summarize_text.assert_called()
        
        # Verify result includes error information
        assert "errors" in result
        assert "search" in str(result["errors"]).lower()
        assert "summary" in result  # Still produced a summary despite errors

    @pytest.mark.asyncio
    async def test_run_research_with_fatal_error(self, research_workflow):
        """Test workflow with a fatal error that cannot be recovered from."""
        # Setup mock returns
        research_workflow.pre_response_agent.clarify_query.return_value = {
            "clarified_query": "Impact of climate change on global agriculture",
            "focus_areas": ["crop yields", "adaptation strategies"]
        }
        
        research_workflow.manager_agent.plan_research.return_value = {
            "plan": [
                {"step": "Search for recent studies on climate change and agriculture"},
                {"step": "Analyze impacts on different regions"}
            ],
            "estimated_time": "10 minutes"
        }
        
        research_workflow.pre_response_agent.present_plan.return_value = True
        
        # Simulate error during search
        research_workflow.research_agent.search_web.side_effect = Exception("Search API error")
        
        # Setup error handling response - fatal error
        research_workflow.manager_agent.handle_errors.return_value = {
            "action": "abort",
            "reason": "Critical API failure, cannot proceed with research"
        }
        
        # Call the method
        query = "How is climate change affecting agriculture?"
        result = await research_workflow.run_research(query)
        
        # Verify error handling was called
        research_workflow.manager_agent.handle_errors.assert_called_once()
        
        # Verify workflow was aborted
        research_workflow.research_agent.fetch_content.assert_not_called()
        research_workflow.summary_agent.summarize_text.assert_not_called()
        
        # Verify result indicates workflow abortion
        assert "status" in result
        assert result["status"] == "failed"
        assert "error" in result
        assert "critical api failure" in result["error"].lower()
        assert "partial_results" in result  # Should include any partial results obtained before failure

    @pytest.mark.asyncio
    async def test_run_research_with_verification_focus(self, research_workflow):
        """Test workflow with emphasis on fact verification."""
        # Setup mock returns
        research_workflow.pre_response_agent.clarify_query.return_value = {
            "clarified_query": "Effectiveness of renewable energy in reducing carbon emissions",
            "focus_areas": ["solar energy", "wind power", "emission reduction"],
            "verification_focus": True  # User wants emphasis on fact verification
        }
        
        research_workflow.manager_agent.plan_research.return_value = {
            "plan": [
                {"step": "Search for studies on renewable energy effectiveness"},
                {"step": "Verify emission reduction claims"},
                {"step": "Cross-check data from multiple sources"}
            ],
            "estimated_time": "20 minutes"
        }
        
        research_workflow.pre_response_agent.present_plan.return_value = True
        
        research_workflow.research_agent.search_web.return_value = {
            "search1": {"title": "Renewable Energy Impact Study", "url": "https://example.com/1"},
            "search2": {"title": "Carbon Reduction Analysis", "url": "https://example.com/2"}
        }
        
        research_workflow.research_agent.fetch_content.return_value = {
            "content1": "Solar energy can reduce carbon emissions by 70%...",
            "content2": "Wind power has prevented 1.1 billion tons of CO2 emissions..."
        }
        
        # Setup verification results
        research_workflow.verification_agent.verify_fact.side_effect = [
            {"verified": True, "confidence": 0.9, "sources": ["IPCC", "IEA"]},  # First fact
            {"verified": False, "confidence": 0.8, "sources": ["Nature Energy", "Science"]}  # Second fact
        ]
        
        research_workflow.verification_agent.cross_check_information.return_value = {
            "consensus": True,
            "agreement_level": 0.85,
            "sources": ["NREL", "IEA", "IRENA"]
        }
        
        research_workflow.summary_agent.summarize_text.return_value = "Renewable energy significantly reduces carbon emissions, though some claims are overstated..."
        
        # Call the method
        query = "How effective is renewable energy at reducing carbon emissions?"
        result = await research_workflow.run_research(query)
        
        # Verify verification methods were called
        assert research_workflow.verification_agent.verify_fact.call_count >= 2
        research_workflow.verification_agent.cross_check_information.assert_called()
        
        # Verify result includes detailed verification information
        assert "verified_facts" in result
        assert "disputed_claims" in result
        assert len(result["verified_facts"]) >= 1
        assert len(result["disputed_claims"]) >= 1
        assert "verification_summary" in result

    @pytest.mark.asyncio
    async def test_run_research_with_visualization_focus(self, research_workflow):
        """Test workflow with emphasis on visual content generation."""
        # Setup mock returns
        research_workflow.pre_response_agent.clarify_query.return_value = {
            "clarified_query": "Ocean plastic pollution distribution and impacts",
            "focus_areas": ["distribution patterns", "marine life impact", "cleanup efforts"],
            "visualization_focus": True  # User wants emphasis on visual content
        }
        
        research_workflow.manager_agent.plan_research.return_value = {
            "plan": [
                {"step": "Research ocean plastic distribution"},
                {"step": "Generate visualization of pollution hotspots"},
                {"step": "Create diagram of impacts on marine ecosystems"}
            ],
            "estimated_time": "25 minutes"
        }
        
        research_workflow.pre_response_agent.present_plan.return_value = True
        
        research_workflow.research_agent.search_web.return_value = {
            "search1": {"title": "Ocean Plastic Pollution Study", "url": "https://example.com/1"},
            "search2": {"title": "Marine Ecosystem Impacts", "url": "https://example.com/2"}
        }
        
        research_workflow.research_agent.fetch_content.return_value = {
            "content1": "Plastic pollution is concentrated in five major ocean gyres...",
            "content2": "Marine species are ingesting microplastics at alarming rates..."
        }
        
        # Setup image generation results
        research_workflow.image_generation_agent.generate_image.return_value = "https://example.com/plastic_pollution_map.jpg"
        research_workflow.image_generation_agent.create_diagram.return_value = "https://example.com/marine_impact_diagram.jpg"
        
        research_workflow.summary_agent.summarize_text.return_value = "Ocean plastic pollution is concentrated in specific regions and severely impacts marine life..."
        
        # Call the method
        query = "How is plastic pollution distributed in oceans and what are its impacts?"
        result = await research_workflow.run_research(query)
        
        # Verify image generation methods were called
        assert research_workflow.image_generation_agent.generate_image.call_count >= 1
        assert research_workflow.image_generation_agent.create_diagram.call_count >= 1
        
        # Verify result includes visual content
        assert "visualizations" in result
        assert len(result["visualizations"]) >= 2
        assert "map" in str(result["visualizations"]).lower()
        assert "diagram" in str(result["visualizations"]).lower()
        assert "visual_descriptions" in result

    @pytest.mark.asyncio
    async def test_run_research_with_file_organization(self, research_workflow):
        """Test workflow with emphasis on file organization and output structure."""
        # Setup mock returns
        research_workflow.pre_response_agent.clarify_query.return_value = {
            "clarified_query": "History and evolution of artificial intelligence",
            "focus_areas": ["early development", "key milestones", "current trends"],
            "structured_output": True  # User wants well-organized output files
        }
        
        research_workflow.manager_agent.plan_research.return_value = {
            "plan": [
                {"step": "Research AI history by decade"},
                {"step": "Identify key breakthroughs"},
                {"step": "Organize findings chronologically"}
            ],
            "estimated_time": "30 minutes"
        }
        
        research_workflow.pre_response_agent.present_plan.return_value = True
        
        research_workflow.research_agent.search_web.return_value = {
            "search1": {"title": "History of AI", "url": "https://example.com/1"},
            "search2": {"title": "AI Timeline", "url": "https://example.com/2"}
        }
        
        research_workflow.research_agent.fetch_content.return_value = {
            "content1": "AI research began in the 1950s with the Dartmouth Workshop...",
            "content2": "Key milestones include the development of expert systems in the 1970s..."
        }
        
        research_workflow.summary_agent.summarize_text.return_value = "AI has evolved through several distinct phases..."
        
        # Setup file organization results
        research_workflow.file_manager_agent.create_research_directory.return_value = "/research/ai_history/"
        research_workflow.file_manager_agent.save_research.side_effect = [
            "/research/ai_history/summary.md",
            "/research/ai_history/timeline.md",
            "/research/ai_history/key_figures.md"
        ]
        research_workflow.file_manager_agent.organize_files.return_value = {
            "directory": "/research/ai_history/",
            "files": [
                {"name": "summary.md", "type": "overview"},
                {"name": "timeline.md", "type": "chronology"},
                {"name": "key_figures.md", "type": "biography"}
            ],
            "structure": "Chronological by decade with separate sections for key innovations"
        }
        
        # Call the method
        query = "What is the history and evolution of artificial intelligence?"
        result = await research_workflow.run_research(query)
        
        # Verify file organization methods were called
        research_workflow.file_manager_agent.create_research_directory.assert_called_once()
        assert research_workflow.file_manager_agent.save_research.call_count >= 3
        research_workflow.file_manager_agent.organize_files.assert_called_once()
        
        # Verify result includes file organization information
        assert "file_structure" in result
        assert "directory" in result["file_structure"]
        assert "files" in result["file_structure"]
        assert len(result["file_structure"]["files"]) >= 3
        assert "main_file" in result
        assert result["main_file"].endswith("summary.md")

    @pytest.mark.asyncio
    async def test_run_research_with_incremental_updates(self, research_workflow):
        """Test workflow with incremental updates during long-running research."""
        # Setup mock returns
        research_workflow.pre_response_agent.clarify_query.return_value = {
            "clarified_query": "Quantum computing advancements and applications",
            "focus_areas": ["quantum supremacy", "error correction", "practical applications"],
            "provide_updates": True  # User wants incremental updates
        }
        
        research_workflow.manager_agent.plan_research.return_value = {
            "plan": [
                {"step": "Research quantum computing fundamentals"},
                {"step": "Analyze recent breakthroughs"},
                {"step": "Explore practical applications"}
            ],
            "estimated_time": "45 minutes"
        }
        
        research_workflow.pre_response_agent.present_plan.return_value = True
        
        # Setup progress tracking
        research_workflow.manager_agent.track_progress.side_effect = [
            {"step": 1, "progress": 0.3, "status": "Researching fundamentals..."},
            {"step": 2, "progress": 0.6, "status": "Analyzing breakthroughs..."},
            {"step": 3, "progress": 0.9, "status": "Exploring applications..."}
        ]
        
        # Setup research results
        research_workflow.research_agent.search_web.return_value = {
            "search1": {"title": "Quantum Computing Advances", "url": "https://example.com/1"},
            "search2": {"title": "Quantum Applications", "url": "https://example.com/2"}
        }
        
        research_workflow.research_agent.fetch_content.return_value = {
            "content1": "Quantum computers have achieved significant milestones in recent years...",
            "content2": "Practical applications include cryptography, drug discovery, and optimization..."
        }
        
        research_workflow.summary_agent.summarize_text.return_value = "Quantum computing has seen rapid advancement with several key breakthroughs..."
        
        # Call the method
        query = "What are the latest advancements in quantum computing and their applications?"
        result = await research_workflow.run_research(query)
        
        # Verify progress tracking was called multiple times
        assert research_workflow.manager_agent.track_progress.call_count >= 3
        
        # Verify memory was updated with progress updates
        assert research_workflow.memory_mcp.store_memory.call_count >= 3
        
        # Verify result includes progress information
        assert "progress_log" in result
        assert len(result["progress_log"]) >= 3
        assert "summary" in result
        assert "completion_time" in result

    @pytest.mark.asyncio
    async def test_run_research_with_source_analysis(self, research_workflow):
        """Test workflow with detailed source analysis and credibility assessment."""
        # Setup mock returns
        research_workflow.pre_response_agent.clarify_query.return_value = {
            "clarified_query": "Effectiveness of various COVID-19 vaccines",
            "focus_areas": ["efficacy rates", "side effects", "variants"],
            "source_analysis": True  # User wants detailed source analysis
        }
        
        research_workflow.manager_agent.plan_research.return_value = {
            "plan": [
                {"step": "Research vaccine clinical trials"},
                {"step": "Analyze source credibility"},
                {"step": "Compare findings across sources"}
            ],
            "estimated_time": "35 minutes"
        }
        
        research_workflow.pre_response_agent.present_plan.return_value = True
        
        research_workflow.research_agent.search_web.return_value = {
            "search1": {"title": "COVID-19 Vaccine Efficacy", "url": "https://example.com/1"},
            "search2": {"title": "Vaccine Comparison Study", "url": "https://example.com/2"}
        }
        
        research_workflow.research_agent.fetch_content.return_value = {
            "content1": "mRNA vaccines have shown efficacy rates of 94-95% in clinical trials...",
            "content2": "Side effect profiles vary between vaccine types..."
        }
        
        # Setup source analysis
        research_workflow.research_agent.analyze_sources.return_value = {
            "source1": {
                "url": "https://example.com/1",
                "type": "peer-reviewed journal",
                "credibility": "high",
                "potential_bias": "none detected",
                "publication_date": "2023-01-15"
            },
            "source2": {
                "url": "https://example.com/2",
                "type": "preprint",
                "credibility": "medium",
                "potential_bias": "industry funding",
                "publication_date": "2023-03-22"
            }
        }
        
        research_workflow.verification_agent.evaluate_source_credibility.side_effect = [
            {"credibility_score": 9.2, "strengths": ["peer-reviewed", "large sample size"]},
            {"credibility_score": 6.5, "limitations": ["not peer-reviewed", "potential conflict of interest"]}
        ]
        
        research_workflow.summary_agent.summarize_text.return_value = "COVID-19 vaccines show varying efficacy rates with mRNA vaccines demonstrating the highest rates..."
        
        # Call the method
        query = "How effective are different COVID-19 vaccines?"
        result = await research_workflow.run_research(query)
        
        # Verify source analysis methods were called
        research_workflow.research_agent.analyze_sources.assert_called_once()
        assert research_workflow.verification_agent.evaluate_source_credibility.call_count >= 2
        
        # Verify result includes source analysis
        assert "source_analysis" in result
        assert len(result["source_analysis"]) >= 2
        assert "credibility_ratings" in result
        assert "high_credibility_sources" in result
        assert "questionable_sources" in result
        assert any(source["credibility"] == "high" for source in result["source_analysis"].values())
        assert any(source["credibility"] == "medium" for source in result["source_analysis"].values())

    @pytest.mark.asyncio
    async def test_run_research_with_custom_output_format(self, research_workflow):
        """Test workflow with custom output format specification."""
        # Setup mock returns
        research_workflow.pre_response_agent.clarify_query.return_value = {
            "clarified_query": "Space exploration missions to Mars",
            "focus_areas": ["rover missions", "human mission plans", "scientific discoveries"],
            "output_format": "timeline"  # User wants a timeline format
        }
        
        research_workflow.manager_agent.plan_research.return_value = {
            "plan": [
                {"step": "Research Mars mission history"},
                {"step": "Compile mission timeline"},
                {"step": "Organize discoveries chronologically"}
            ],
            "estimated_time": "30 minutes"
        }
        
        research_workflow.pre_response_agent.present_plan.return_value = True
        
        research_workflow.research_agent.search_web.return_value = {
            "search1": {"title": "Mars Exploration History", "url": "https://example.com/1"},
            "search2": {"title": "Mars Rover Missions", "url": "https://example.com/2"}
        }
        
        research_workflow.research_agent.fetch_content.return_value = {
            "content1": "Mars exploration began with the Mariner 4 flyby in 1965...",
            "content2": "The Perseverance rover landed in Jezero Crater in 2021..."
        }
        
        # Setup timeline format results
        timeline_data = {
            "1965": "Mariner 4 performs first successful Mars flyby",
            "1976": "Viking 1 and 2 become first spacecraft to land on Mars",
            "1997": "Mars Pathfinder delivers first rover (Sojourner)",
            "2004": "Spirit and Opportunity rovers land",
            "2012": "Curiosity rover lands in Gale Crater",
            "2021": "Perseverance rover lands with Ingenuity helicopter"
        }
        
        research_workflow.summary_agent.extract_key_points.return_value = timeline_data
        
        research_workflow.file_manager_agent.save_research.return_value = "/research/mars_exploration/timeline.md"
        
        # Call the method
        query = "What are the major Mars exploration missions throughout history?"
        result = await research_workflow.run_research(query)
        
        # Verify custom format methods were called
        research_workflow.summary_agent.extract_key_points.assert_called_once()
        
        # Verify result has timeline format
        assert "timeline" in result
        assert len(result["timeline"]) >= 5
        assert "1965" in result["timeline"]
        assert "2021" in result["timeline"]
        assert "format" in result
        assert result["format"] == "timeline"

    @pytest.mark.asyncio
    async def test_run_research_with_query_refinement(self, research_workflow):
        """Test workflow with significant query refinement."""
        # Setup mock returns - simulate vague initial query that needs refinement
        research_workflow.pre_response_agent.clarify_query.return_value = {
            "original_query": "Tell me about black holes",
            "clarified_query": "The formation, structure, and recent discoveries about black holes",
            "focus_areas": ["formation mechanisms", "event horizon", "Hawking radiation", "recent observations"],
            "refinement_needed": True
        }
        
        # Mock the feedback interaction
        research_workflow.pre_response_agent.get_user_feedback.return_value = {
            "approved": True,
            "additional_focus": "black hole imaging"
        }
        
        research_workflow.manager_agent.plan_research.return_value = {
            "plan": [
                {"step": "Research black hole formation"},
                {"step": "Analyze structure and properties"},
                {"step": "Compile recent observations including imaging"}
            ],
            "estimated_time": "25 minutes"
        }
        
        research_workflow.pre_response_agent.present_plan.return_value = True
        
        research_workflow.research_agent.search_web.return_value = {
            "search1": {"title": "Black Hole Formation", "url": "https://example.com/1"},
            "search2": {"title": "Event Horizon Telescope Results", "url": "https://example.com/2"}
        }
        
        research_workflow.research_agent.fetch_content.return_value = {
            "content1": "Black holes form when massive stars collapse under their own gravity...",
            "content2": "The Event Horizon Telescope captured the first image of a black hole in 2019..."
        }
        
        research_workflow.summary_agent.summarize_text.return_value = "Black holes are regions of spacetime where gravity is so strong that nothing can escape..."
        
        # Call the method
        query = "Tell me about black holes"
        result = await research_workflow.run_research(query)
        
        # Verify query refinement methods were called
        research_workflow.pre_response_agent.clarify_query.assert_called_once_with(query)
        research_workflow.pre_response_agent.get_user_feedback.assert_called_once()
        
        # Verify result includes refinement information
        assert "original_query" in result
        assert "refined_query" in result
        assert result["original_query"] == "Tell me about black holes"
        assert "formation" in result["refined_query"].lower()
        assert "black hole imaging" in str(result["focus_areas"]).lower()

    @pytest.mark.asyncio
    async def test_run_research_with_insufficient_information(self, research_workflow):
        """Test workflow when insufficient information is found."""
        # Setup mock returns
        research_workflow.pre_response_agent.clarify_query.return_value = {
            "clarified_query": "Recent breakthroughs in fusion energy containment",
            "focus_areas": ["tokamak advances", "inertial confinement", "recent milestones"]
        }
        
        research_workflow.manager_agent.plan_research.return_value = {
            "plan": [
                {"step": "Search for recent fusion containment research"},
                {"step": "Analyze breakthrough claims"},
                {"step": "Compile technical advancements"}
            ],
            "estimated_time": "20 minutes"
        }
        
        research_workflow.pre_response_agent.present_plan.return_value = True
        
        # Simulate limited search results
        research_workflow.research_agent.search_web.return_value = {
            "search1": {"title": "Fusion Energy Overview", "url": "https://example.com/1"}
        }
        
        research_workflow.research_agent.fetch_content.return_value = {
            "content1": "Fusion energy research continues to face challenges in plasma containment..."
        }
        
        # Indicate insufficient information
        research_workflow.manager_agent.coordinate_agents.return_value = {
            "status": "insufficient_information",
            "reason": "Limited recent data on fusion containment breakthroughs",
            "recommendation": "Broaden search or consult specialized databases"
        }
        
        research_workflow.summary_agent.summarize_text.return_value = "Limited information is available on very recent fusion containment breakthroughs..."
        
        # Call the method
        query = "What are the latest breakthroughs in fusion energy containment technology?"
        result = await research_workflow.run_research(query)
        
        # Verify coordination was called
        research_workflow.manager_agent.coordinate_agents.assert_called_once()
        
        # Verify result indicates information limitations
        assert "information_status" in result
        assert result["information_status"] == "insufficient"
        assert "limitations" in result
        assert "recommendations" in result
        assert "summary" in result  # Still provides what limited info was found
        assert "alternative_queries" in result

    @pytest.mark.asyncio
    async def test_run_research_with_conflicting_information(self, research_workflow):
        """Test workflow when conflicting information is found."""
        # Setup mock returns
        research_workflow.pre_response_agent.clarify_query.return_value = {
            "clarified_query": "Health effects of moderate coffee consumption",
            "focus_areas": ["cardiovascular effects", "cognitive effects", "long-term health"]
        }
        
        research_workflow.manager_agent.plan_research.return_value = {
            "plan": [
                {"step": "Research coffee health studies"},
                {"step": "Analyze conflicting findings"},
                {"step": "Evaluate study methodologies"}
            ],
            "estimated_time": "25 minutes"
        }
        
        research_workflow.pre_response_agent.present_plan.return_value = True
        
        research_workflow.research_agent.search_web.return_value = {
            "search1": {"title": "Coffee Health Benefits", "url": "https://example.com/1"},
            "search2": {"title": "Coffee Health Risks", "url": "https://example.com/2"}
        }
        
        research_workflow.research_agent.fetch_content.return_value = {
            "content1": "Studies suggest coffee may reduce risk of heart disease and type 2 diabetes...",
            "content2": "Some research indicates coffee may increase blood pressure and anxiety..."
        }
        
        # Indicate conflicting information
        research_workflow.verification_agent.cross_check_information.return_value = {
            "consensus": False,
            "conflicts": [
                {
                    "topic": "cardiovascular health",
                    "conflict": "Some studies show reduced heart disease risk, others show increased blood pressure"
                },
                {
                    "topic": "anxiety and sleep",
                    "conflict": "Conflicting findings on coffee's effects on anxiety and sleep quality"
                }
            ],
            "agreement_areas": [
                "Moderate consumption (3-4 cups daily) appears generally safe for most adults"
            ]
        }
        
        research_workflow.summary_agent.summarize_text.return_value = "Research on coffee's health effects shows both potential benefits and risks..."
        
        # Call the method
        query = "Is coffee good or bad for your health?"
        result = await research_workflow.run_research(query)
        
        # Verify cross-check was called
        research_workflow.verification_agent.cross_check_information.assert_called_once()
        
        # Verify result indicates conflicting information
        assert "information_status" in result
        assert result["information_status"] == "conflicting"
        assert "conflicts" in result
        assert len(result["conflicts"]) >= 2
        assert "cardiovascular" in str(result["conflicts"]).lower()
        assert "agreement_areas" in result
        assert "summary" in result
        assert "methodological_differences" in result

    @pytest.mark.asyncio
    async def test_run_research_with_technical_topic(self, research_workflow):
        """Test workflow with a highly technical research topic."""
        # Setup mock returns
        research_workflow.pre_response_agent.clarify_query.return_value = {
            "clarified_query": "CRISPR-Cas9 gene editing mechanisms and applications",
            "focus_areas": ["molecular mechanism", "off-target effects", "therapeutic applications"],
            "technical_level": "advanced"
        }
        
        research_workflow.manager_agent.plan_research.return_value = {
            "plan": [
                {"step": "Research CRISPR molecular biology"},
                {"step": "Analyze technical challenges"},
                {"step": "Compile clinical applications"}
            ],
            "estimated_time": "40 minutes"
        }
        
        research_workflow.pre_response_agent.present_plan.return_value = True
        
        research_workflow.research_agent.search_web.return_value = {
            "search1": {"title": "CRISPR-Cas9 Mechanism", "url": "https://example.com/1"},
            "search2": {"title": "CRISPR Clinical Trials", "url": "https://example.com/2"}
        }
        
        research_workflow.research_agent.fetch_content.return_value = {
            "content1": "CRISPR-Cas9 functions by using a guide RNA (gRNA) to target specific DNA sequences...",
            "content2": "Clinical applications include treatments for sickle cell disease, beta-thalassemia..."
        }
        
        research_workflow.summary_agent.summarize_text.return_value = "CRISPR-Cas9 is a revolutionary gene editing technology that uses RNA-guided nucleases..."
        
        # Call the method
        query = "How does CRISPR-Cas9 work and what are its applications?"
        result = await research_workflow.run_research(query)
        
        # Verify result includes technical information
        assert "technical_level" in result
        assert result["technical_level"] == "advanced"
        assert "technical_terms" in result
        assert "simplified_explanation" in result
        assert "detailed_explanation" in result
        assert len(result["detailed_explanation"]) > len(result["simplified_explanation"])
        assert "crispr" in result["technical_terms"].lower() or "cas9" in result["technical_terms"].lower()

    @pytest.mark.asyncio
    async def test_research_workflow_integration(self, mock_mcps):
        """Test the full integration of the research workflow with actual agent instances."""
        # Create a workflow with real agent instances but mock MCPs
        workflow = ResearchWorkflow(
            brave_search_mcp=mock_mcps["brave_search_mcp"],
            fetch_mcp=mock_mcps["fetch_mcp"],
            memory_mcp=mock_mcps["memory_mcp"],
            everart_mcp=mock_mcps["everart_mcp"],
            filesystem_mcp=mock_mcps["filesystem_mcp"]
        )
        
        # Setup mock returns for MCPs
        mock_mcps["brave_search_mcp"].web_search.return_value = """
        Title: Climate Change Overview
        Description: Comprehensive overview of climate change causes and effects.
        URL: https://example.com/climate-change

        Title: IPCC Sixth Assessment Report
        Description: Latest findings from the Intergovernmental Panel on Climate Change.
        URL: https://example.com/ipcc-report
        """
        
        mock_mcps["fetch_mcp"].fetch_url.return_value = """
        Climate change refers to long-term shifts in temperatures and weather patterns.
        Human activities have been the main driver of climate change since the 1800s,
        primarily due to burning fossil fuels like coal, oil and gas.
        """
        
        mock_mcps["memory_mcp"].store_memory.return_value = "Memory stored successfully"
        mock_mcps["memory_mcp"].retrieve_memory.return_value = "No previous research found"
        
        mock_mcps["everart_mcp"].generate_image.return_value = "https://example.com/climate-change-diagram.jpg"
        
        mock_mcps["filesystem_mcp"].write_file.return_value = "File written successfully"
        mock_mcps["filesystem_mcp"].create_directory.return_value = "Directory created successfully"
        
        # Patch the LLM calls that would be made by real agents
        with patch('apps.agents.base_agent.BaseAgent._call_llm', new_callable=AsyncMock) as mock_llm:
            # Configure mock LLM responses for different agent calls
            mock_llm.side_effect = [
                # Pre-response agent clarification
                {
                    "clarified_query": "Causes and effects of climate change",
                    "focus_areas": ["greenhouse gases", "global warming", "environmental impacts"]
                },
                # Manager agent planning
                {
                    "plan": [
                        {"step": "Research climate change causes"},
                        {"step": "Analyze environmental effects"},
                        {"step": "Compile scientific consensus"}
                    ],
                    "estimated_time": "20 minutes"
                },
                # Pre-response agent plan presentation (returns True)
                True,
                # Research agent search analysis
                {
                    "relevant_sources": ["IPCC report", "Climate Change Overview"],
                    "key_topics": ["greenhouse gas emissions", "temperature rise", "sea level rise"]
                },
                # Summary agent summarization
                "Climate change is primarily caused by human activities, especially burning fossil fuels. Effects include rising temperatures, sea level rise, and more extreme weather events.",
                # Verification agent cross-checking
                {
                    "consensus": True,
                    "agreement_level": 0.95,
                    "sources": ["IPCC", "NASA", "NOAA"]
                },
                # Image generation agent planning
                {
                    "image_concepts": ["climate change causes and effects diagram", "temperature rise graph"],
                    "visualization_approach": "informational diagram"
                },
                # File manager agent organization
                {
                    "file_structure": {
                        "main_summary": "climate_change_summary.md",
                        "supporting_files": ["causes.md", "effects.md", "images/diagram.jpg"]
                    }
                }
            ]
            
            # Call the method
            query = "What causes climate change and what are its effects?"
            result = await workflow.run_research(query)
            
            # Verify LLM was called multiple times
            assert mock_llm.call_count >= 5
            
            # Verify MCPs were called
            mock_mcps["brave_search_mcp"].web_search.assert_called()
            mock_mcps["fetch_mcp"].fetch_url.assert_called()
            mock_mcps["memory_mcp"].store_memory.assert_called()
            mock_mcps["everart_mcp"].generate_image.assert_called()
            mock_mcps["filesystem_mcp"].write_file.assert_called()
            
            # Verify result structure
            assert "summary" in result
            assert "sources" in result
            assert "visualizations" in result
            assert "file_path" in result

    def test_workflow_initialization_with_missing_mcps(self):
        """Test that workflow initialization fails with missing required MCPs."""
        # Missing brave_search_mcp
        with pytest.raises(ValueError, match="brave_search_mcp is required"):
            ResearchWorkflow(
                fetch_mcp=MagicMock(),
                memory_mcp=MagicMock(),
                everart_mcp=MagicMock(),
                filesystem_mcp=MagicMock()
            )
        
        # Missing fetch_mcp
        with pytest.raises(ValueError, match="fetch_mcp is required"):
            ResearchWorkflow(
                brave_search_mcp=MagicMock(),
                memory_mcp=MagicMock(),
                everart_mcp=MagicMock(),
                filesystem_mcp=MagicMock()
            )
        
        # Missing memory_mcp
        with pytest.raises(ValueError, match="memory_mcp is required"):
            ResearchWorkflow(
                brave_search_mcp=MagicMock(),
                fetch_mcp=MagicMock(),
                everart_mcp=MagicMock(),
                filesystem_mcp=MagicMock()
            )

    @pytest.mark.asyncio
    async def test_workflow_validation_methods(self, research_workflow):
        """Test the workflow's validation methods."""
        # Test valid query
        valid_query = "What are the environmental impacts of plastic pollution?"
        research_workflow._validate_query(valid_query)  # Should not raise exception
        
        # Test empty query
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            research_workflow._validate_query("")
        
        # Test too short query
        with pytest.raises(ValidationError, match="Query is too short"):
            research_workflow._validate_query("Plastic?")
        
        # Test valid research plan
        valid_plan = {
            "plan": [{"step": "Research plastic pollution sources"}],
            "estimated_time": "15 minutes"
        }
        research_workflow._validate_research_plan(valid_plan)  # Should not raise exception
        
        # Test invalid research plan (missing plan)
        invalid_plan = {"estimated_time": "15 minutes"}
        with pytest.raises(ValidationError, match="Research plan must include 'plan'"):
            research_workflow._validate_research_plan(invalid_plan)
        
        # Test invalid research plan (empty steps)
        invalid_plan = {"plan": [], "estimated_time": "15 minutes"}
        with pytest.raises(ValidationError, match="Research plan must include at least one step"):
            research_workflow._validate_research_plan(invalid_plan)

    @pytest.mark.asyncio
    async def test_workflow_error_handling_methods(self, research_workflow):
        """Test the workflow's error handling methods."""
        # Test handling of search error
        search_error = Exception("API rate limit exceeded")
        error_context = {"query": "climate change", "step": "web_search"}
        
        # Configure mock response for error handling
        research_workflow.manager_agent.handle_errors.return_value = {
            "action": "retry",
            "wait_time": 60,
            "message": "Waiting for rate limit to reset"
        }
        
        # Call error handling method
        result = await research_workflow._handle_workflow_error(search_error, error_context)
        
        # Verify manager agent was called
        research_workflow.manager_agent.handle_errors.assert_called_once_with(
            error=search_error,
            context=error_context
        )
        
        # Verify result
        assert result["action"] == "retry"
        assert result["wait_time"] == 60
        
        # Test handling of fatal error
        fatal_error = Exception("Authentication failed")
        error_context = {"query": "climate change", "step": "api_authentication"}
        
        # Configure mock response for fatal error
        research_workflow.manager_agent.handle_errors.return_value = {
            "action": "abort",
            "reason": "API authentication failed, cannot proceed"
        }
        
        # Call error handling method
        result = await research_workflow._handle_workflow_error(fatal_error, error_context)
        
        # Verify result indicates workflow should abort
        assert result["action"] == "abort"
        assert "reason" in result

    @pytest.mark.asyncio
    async def test_workflow_cleanup_and_resource_management(self, research_workflow):
        """Test the workflow's cleanup and resource management."""
        # Setup mock for cleanup operations
        research_workflow.memory_mcp.clear_namespace = AsyncMock()
        
        # Call cleanup method
        await research_workflow._cleanup_resources("test_research_id")
        
        # Verify temporary resources were cleaned up
        research_workflow.memory_mcp.clear_namespace.assert_called_with("temp_test_research_id")
        
        # Test with error during cleanup
        research_workflow.memory_mcp.clear_namespace.side_effect = Exception("Cleanup error")
        
        # Should not raise exception, just log the error
        await research_workflow._cleanup_resources("test_research_id")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
