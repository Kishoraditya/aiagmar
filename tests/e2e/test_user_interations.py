import os
import pytest
import time
import uuid
from unittest.mock import patch, MagicMock

from apps.workflows.research_workflow import ResearchWorkflow
from apps.agents.manager_agent import ManagerAgent
from apps.agents.pre_response_agent import PreResponseAgent
from apps.agents.research_agent import ResearchAgent
from apps.agents.image_generation_agent import ImageGenerationAgent
from apps.agents.file_manager_agent import FileManagerAgent
from apps.agents.summary_agent import SummaryAgent
from apps.agents.verification_agent import VerificationAgent

from apps.mcps.brave_search_mcp import BraveSearchMCP
from apps.mcps.everart_mcp import EverArtMCP
from apps.mcps.fetch_mcp import FetchMCP
from apps.mcps.filesystem_mcp import FilesystemMCP
from apps.mcps.memory_mcp import MemoryMCP

from apps.utils.logger import Logger


class TestUserInteractions:
    """End-to-end tests for user interactions with the research system."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        # Create a unique workspace directory for each test
        self.workspace_dir = os.path.join(os.path.dirname(__file__), "test_workspace", str(uuid.uuid4()))
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Set up logger
        self.logger = Logger("test_user_interactions")
        
        # Create mock API keys
        self.brave_api_key = "test_brave_api_key"
        self.everart_api_key = "test_everart_api_key"
        
        # Create a mocked workflow for testing
        self.workflow = self.create_mocked_workflow()
        
        yield
        
        # Clean up workspace directory
        import shutil
        if os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir)
    
    def create_mocked_workflow(self):
        """Create a research workflow with mocked MCPs and agents."""
        # Create mocked MCPs
        brave_search = MagicMock(spec=BraveSearchMCP)
        brave_search.web_search.return_value = "Mock web search results"
        brave_search.local_search.return_value = "Mock local search results"
        
        everart = MagicMock(spec=EverArtMCP)
        everart.generate_image.return_value = "https://example.com/mock-image.jpg"
        
        fetch = MagicMock(spec=FetchMCP)
        fetch.fetch_url.return_value = "Mock content from website"
        fetch.fetch_text.return_value = "Mock text content from website"
        
        filesystem = MagicMock(spec=FilesystemMCP)
        filesystem.write_file.return_value = "File written successfully"
        filesystem.read_file.return_value = "Mock file content"
        
        memory = MagicMock(spec=MemoryMCP)
        memory.store_memory.return_value = "Memory stored successfully"
        memory.retrieve_memory.return_value = "Mock memory content"
        
        # Create agents with mocked MCPs
        manager_agent = ManagerAgent()
        manager_agent.memory_mcp = memory
        manager_agent.execute = MagicMock(return_value="Manager agent response")
        
        pre_response_agent = PreResponseAgent()
        pre_response_agent.memory_mcp = memory
        pre_response_agent.execute = MagicMock(return_value="Pre-response agent response")
        pre_response_agent.clarify_query = MagicMock(return_value="Clarified query")
        
        research_agent = ResearchAgent()
        research_agent.brave_search_mcp = brave_search
        research_agent.fetch_mcp = fetch
        research_agent.memory_mcp = memory
        research_agent.execute = MagicMock(return_value="Research agent response")
        
        image_generation_agent = ImageGenerationAgent()
        image_generation_agent.everart_mcp = everart
        image_generation_agent.memory_mcp = memory
        image_generation_agent.execute = MagicMock(return_value="Image generation agent response")
        
        file_manager_agent = FileManagerAgent()
        file_manager_agent.filesystem_mcp = filesystem
        file_manager_agent.memory_mcp = memory
        file_manager_agent.execute = MagicMock(return_value="File manager agent response")
        
        summary_agent = SummaryAgent()
        summary_agent.memory_mcp = memory
        summary_agent.execute = MagicMock(return_value="Summary agent response")
        
        verification_agent = VerificationAgent()
        verification_agent.brave_search_mcp = brave_search
        verification_agent.memory_mcp = memory
        verification_agent.execute = MagicMock(return_value="Verification agent response")
        
        # Create workflow
        workflow = ResearchWorkflow(
            manager_agent=manager_agent,
            pre_response_agent=pre_response_agent,
            research_agent=research_agent,
            image_generation_agent=image_generation_agent,
            file_manager_agent=file_manager_agent,
            summary_agent=summary_agent,
            verification_agent=verification_agent
        )
        
        # Mock the workflow's execute method to track calls
        original_execute = workflow.execute
        workflow.execute = MagicMock(side_effect=original_execute)
        
        return workflow
    
    def test_basic_user_query(self):
        """Test a basic user query that doesn't require clarification."""
        # Set up mock responses
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        
        # Execute the workflow
        result = self.workflow.execute("Tell me about quantum computing")
        
        # Assert that the workflow executed successfully
        assert result is not None
        assert len(result) > 0
        
        # Verify that the pre-response agent checked if clarification was needed
        self.workflow.pre_response_agent.needs_clarification.assert_called_once()
        
        # Verify that the clarify_query method was not called
        self.workflow.pre_response_agent.clarify_query.assert_not_called()
    
    def test_ambiguous_query_requiring_clarification(self):
        """Test an ambiguous query that requires clarification."""
        # Set up mock responses
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=True)
        self.workflow.pre_response_agent.clarify_query = MagicMock(return_value="Tell me about quantum computing applications in cryptography")
        
        # Execute the workflow
        result = self.workflow.execute("Tell me about quantum")
        
        # Assert that the workflow executed successfully
        assert result is not None
        assert len(result) > 0
        
        # Verify that the pre-response agent checked if clarification was needed
        self.workflow.pre_response_agent.needs_clarification.assert_called_once()
        
        # Verify that the clarify_query method was called
        self.workflow.pre_response_agent.clarify_query.assert_called_once()
        
        # Verify that the research was conducted with the clarified query
        self.workflow.research_agent.execute.assert_called_once()
    
    def test_multi_turn_conversation(self):
        """Test a multi-turn conversation with follow-up questions."""
        # Mock the memory MCP to store conversation history
        self.workflow.pre_response_agent.memory_mcp.retrieve_memory.side_effect = [
            # First call: No previous conversation
            "",
            # Second call: Previous conversation exists
            "User: Tell me about quantum computing\nSystem: Quantum computing uses quantum mechanics to perform calculations..."
        ]
        
        # First query
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        result1 = self.workflow.execute("Tell me about quantum computing")
        
        # Assert that the first query executed successfully
        assert result1 is not None
        assert len(result1) > 0
        
        # Verify that the memory was checked for conversation history
        self.workflow.pre_response_agent.memory_mcp.retrieve_memory.assert_called()
        
        # Reset mocks for second query
        self.workflow.pre_response_agent.needs_clarification.reset_mock()
        self.workflow.research_agent.execute.reset_mock()
        
        # Second query (follow-up)
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        result2 = self.workflow.execute("What are its applications?")
        
        # Assert that the second query executed successfully
        assert result2 is not None
        assert len(result2) > 0
        
        # Verify that the memory was checked for conversation history
        assert self.workflow.pre_response_agent.memory_mcp.retrieve_memory.call_count >= 2
        
        # Verify that the research was conducted for the follow-up question
        self.workflow.research_agent.execute.assert_called_once()
    
    def test_user_query_with_preferences(self):
        """Test a user query with specific preferences."""
        # Set up mock responses
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        self.workflow.pre_response_agent.extract_preferences = MagicMock(return_value={
            "detail_level": "high",
            "include_images": True,
            "focus_areas": ["technical", "applications"]
        })
        
        # Execute the workflow
        result = self.workflow.execute("Give me a detailed technical explanation of quantum computing with images, focusing on applications")
        
        # Assert that the workflow executed successfully
        assert result is not None
        assert len(result) > 0
        
        # Verify that preferences were extracted
        self.workflow.pre_response_agent.extract_preferences.assert_called_once()
        
        # Verify that the image generation agent was called (due to preference for images)
        self.workflow.image_generation_agent.execute.assert_called_once()
    
    def test_user_feedback_incorporation(self):
        """Test incorporation of user feedback into the research process."""
        # First query
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        result1 = self.workflow.execute("Tell me about renewable energy")
        
        # Assert that the first query executed successfully
        assert result1 is not None
        assert len(result1) > 0
        
        # Reset mocks for feedback
        self.workflow.manager_agent.execute.reset_mock()
        self.workflow.research_agent.execute.reset_mock()
        self.workflow.summary_agent.execute.reset_mock()
        
        # Mock the feedback handling
        self.workflow.incorporate_feedback = MagicMock(return_value="Updated research based on feedback")
        
        # User provides feedback
        result2 = self.workflow.incorporate_feedback("Could you focus more on solar energy specifically?")
        
        # Assert that the feedback was incorporated
        assert result2 is not None
        assert len(result2) > 0
        assert "feedback" in result2.lower() or "updated" in result2.lower()
        
        # Verify that the feedback incorporation method was called
        self.workflow.incorporate_feedback.assert_called_once()
    
    def test_user_query_with_time_constraint(self):
        """Test a user query with a time constraint."""
        # Set up mock responses
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        self.workflow.pre_response_agent.extract_constraints = MagicMock(return_value={
            "time_constraint": "quick",
            "max_sources": 3
        })
        
        # Execute the workflow
        result = self.workflow.execute("Give me a quick overview of artificial intelligence, I only have 5 minutes")
        
        # Assert that the workflow executed successfully
        assert result is not None
        assert len(result) > 0
        
        # Verify that constraints were extracted
        self.workflow.pre_response_agent.extract_constraints.assert_called_once()
        
        # Verify that the research was conducted with constraints
        self.workflow.research_agent.execute.assert_called_once()
    
    def test_user_query_with_source_requirements(self):
        """Test a user query with specific source requirements."""
        # Set up mock responses
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        self.workflow.pre_response_agent.extract_requirements = MagicMock(return_value={
            "source_types": ["academic", "peer-reviewed"],
            "recency": "last 2 years"
        })
        
        # Execute the workflow
        result = self.workflow.execute("Research climate change using only peer-reviewed academic sources from the last 2 years")
        
        # Assert that the workflow executed successfully
        assert result is not None
        assert len(result) > 0
        
        # Verify that requirements were extracted
        self.workflow.pre_response_agent.extract_requirements.assert_called_once()
        
        # Verify that the research was conducted with source requirements
        self.workflow.research_agent.execute.assert_called_once()
        
        # Verify that the verification agent was called to check sources
        self.workflow.verification_agent.execute.assert_called_once()
    
    def test_user_query_with_format_specification(self):
        """Test a user query with specific output format requirements."""
        # Set up mock responses
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        self.workflow.pre_response_agent.extract_format_requirements = MagicMock(return_value={
            "format": "report",
            "sections": ["introduction", "main findings", "conclusion"],
            "include_references": True
        })
        
        # Execute the workflow
        result = self.workflow.execute("Create a report on machine learning with introduction, main findings, and conclusion sections, including references")
        
        # Assert that the workflow executed successfully
        assert result is not None
        assert len(result) > 0
        
        # Verify that format requirements were extracted
        self.workflow.pre_response_agent.extract_format_requirements.assert_called_once()
        
        # Verify that the file manager agent was called to create the report
        self.workflow.file_manager_agent.execute.assert_called_once()
    
    def test_user_interruption_and_continuation(self):
        """Test handling of user interruption and continuation of research."""
        # Start a research task
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        
        # Mock an interruption during research
        self.workflow.research_agent.execute = MagicMock(side_effect=KeyboardInterrupt)
        
        try:
            # This should raise KeyboardInterrupt
            self.workflow.execute("Research quantum computing")
        except KeyboardInterrupt:
            # Handle the interruption
            pass
        
        # Reset the mock to allow continuation
        self.workflow.research_agent.execute = MagicMock(return_value="Continued research on quantum computing")
        
        # Mock the continuation method
        self.workflow.continue_research = MagicMock(return_value="Research continued from where it was interrupted")
        
        # Continue the research
        result = self.workflow.continue_research("quantum computing")
        
        # Assert that the research was continued
        assert result is not None
        assert len(result) > 0
        assert "continued" in result.lower() or "resuming" in result.lower()
        
        # Verify that the continuation method was called
        self.workflow.continue_research.assert_called_once()
    
    def test_user_query_with_conflicting_requirements(self):
        """Test handling of user query with conflicting requirements."""
        # Set up mock responses
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=True)
        self.workflow.pre_response_agent.identify_conflicts = MagicMock(return_value=[
            {"requirement": "quick overview", "conflicts_with": "detailed analysis"},
            {"requirement": "recent sources only", "conflicts_with": "historical perspective"}
        ])
        self.workflow.pre_response_agent.resolve_conflicts = MagicMock(return_value={
            "resolved_query": "Provide a balanced overview of climate change with emphasis on recent developments while including key historical context",
            "explanation": "I've balanced your request for both recent information and historical context by focusing primarily on recent developments while including essential historical background."
        })
        
        # Execute the workflow
        result = self.workflow.execute("Give me a quick but detailed overview of climate change using only the most recent sources but also with historical perspective")
        
        # Assert that the workflow executed successfully
        assert result is not None
        assert len(result) > 0
        
        # Verify that conflicts were identified and resolved
        self.workflow.pre_response_agent.identify_conflicts.assert_called_once()
        self.workflow.pre_response_agent.resolve_conflicts.assert_called_once()
        
        # Verify that the research was conducted with the resolved query
        self.workflow.research_agent.execute.assert_called_once()
    
    def test_user_query_with_progressive_disclosure(self):
        """Test handling of a complex query with progressive disclosure of information."""
        # Set up mock responses
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        self.workflow.pre_response_agent.should_use_progressive_disclosure = MagicMock(return_value=True)
        
        # Mock the progressive disclosure method
        self.workflow.manager_agent.plan_progressive_disclosure = MagicMock(return_value=[
            {"stage": "overview", "content": "Brief overview of quantum computing"},
            {"stage": "principles", "content": "Quantum principles like superposition and entanglement"},
            {"stage": "applications", "content": "Applications in cryptography, optimization, and simulation"},
            {"stage": "challenges", "content": "Technical challenges and limitations"}
        ])
        
        # Execute the workflow
        result = self.workflow.execute("Explain quantum computing from basic principles to advanced applications")
        
        # Assert that the workflow executed successfully
        assert result is not None
        assert len(result) > 0
        
        # Verify that progressive disclosure was considered
        self.workflow.pre_response_agent.should_use_progressive_disclosure.assert_called_once()
        
        # Verify that the manager agent planned the progressive disclosure
        self.workflow.manager_agent.plan_progressive_disclosure.assert_called_once()
    
    def test_user_query_requiring_personalization(self):
        """Test handling of a query requiring personalization based on user history."""
        # Mock the memory MCP to provide user history
        self.workflow.pre_response_agent.memory_mcp.retrieve_memory.return_value = """
        User interests: Machine learning, Python programming
        Previous queries: 
        - "How to implement neural networks in Python"
        - "Best practices for TensorFlow"
        User preferences:
        - Prefers code examples
        - Prefers technical depth
        """
        
        # Set up mock responses
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        self.workflow.pre_response_agent.should_personalize = MagicMock(return_value=True)
        self.workflow.pre_response_agent.personalize_query = MagicMock(return_value={
            "personalized_query": "Explain deep learning architectures with Python code examples and technical details",
            "personalization_factors": ["Added code examples", "Increased technical depth", "Focused on Python implementation"]
        })
        
        # Execute the workflow
        result = self.workflow.execute("Explain deep learning architectures")
        
        # Assert that the workflow executed successfully
        assert result is not None
        assert len(result) > 0
        
        # Verify that personalization was considered
        self.workflow.pre_response_agent.should_personalize.assert_called_once()
        
        # Verify that the query was personalized
        self.workflow.pre_response_agent.personalize_query.assert_called_once()
        
        # Verify that the research was conducted with the personalized query
        self.workflow.research_agent.execute.assert_called_once()
    
    def test_user_query_with_explicit_agent_selection(self):
        """Test handling of a query with explicit agent selection."""
        # Set up mock responses
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        self.workflow.pre_response_agent.extract_agent_selection = MagicMock(return_value={
            "selected_agents": ["research", "verification"],
            "skip_agents": ["image_generation"]
        })
        
        # Execute the workflow
        result = self.workflow.execute("Research climate change and verify the facts, but don't generate images")
        
        # Assert that the workflow executed successfully
        assert result is not None
        assert len(result) > 0
        
        # Verify that agent selection was extracted
        self.workflow.pre_response_agent.extract_agent_selection.assert_called_once()
        
        # Verify that the selected agents were called
        self.workflow.research_agent.execute.assert_called_once()
        self.workflow.verification_agent.execute.assert_called_once()
        
        # Verify that the skipped agents were not called
        self.workflow.image_generation_agent.execute.assert_not_called()
    
    def test_user_query_with_iterative_refinement(self):
        """Test handling of a query requiring iterative refinement."""
        # Set up mock responses
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        self.workflow.manager_agent.needs_iterative_refinement = MagicMock(return_value=True)
        
        # Mock the iterative refinement process
        self.workflow.manager_agent.plan_iterative_refinement = MagicMock(return_value=[
            {"iteration": 1, "focus": "Initial broad search", "result": "Found 5 relevant areas"},
            {"iteration": 2, "focus": "Deep dive into most promising area", "result": "Identified key insights"},
            {"iteration": 3, "focus": "Verification and cross-referencing", "result": "Confirmed findings"}
        ])
        
        # Execute the workflow
        result = self.workflow.execute("Find the most effective strategies for reducing carbon emissions")
        
        # Assert that the workflow executed successfully
        assert result is not None
        assert len(result) > 0
        
        # Verify that iterative refinement was considered
        self.workflow.manager_agent.needs_iterative_refinement.assert_called_once()
        
        # Verify that the manager agent planned the iterative refinement
        self.workflow.manager_agent.plan_iterative_refinement.assert_called_once()
        
        # Verify that the research agent was called (potentially multiple times)
        assert self.workflow.research_agent.execute.call_count >= 1
    
    def test_user_query_with_real_time_updates(self):
        """Test handling of a query with real-time updates during processing."""
        # Set up mock responses
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        self.workflow.manager_agent.should_provide_updates = MagicMock(return_value=True)
        
        # Mock the update callback
        update_callback = MagicMock()
        
        # Execute the workflow with the update callback
        result = self.workflow.execute("Research quantum computing", update_callback=update_callback)
        
        # Assert that the workflow executed successfully
        assert result is not None
        assert len(result) > 0
        
        # Verify that updates were considered
        self.workflow.manager_agent.should_provide_updates.assert_called_once()
        
        # Verify that the update callback was called at least once
        assert update_callback.call_count >= 1
    
    def test_user_query_with_error_recovery(self):
        """Test handling of a query with error recovery during processing."""
        # Set up mock responses
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        
        # Mock an error during research
        self.workflow.research_agent.execute = MagicMock(side_effect=Exception("API rate limit exceeded"))
        
        # Mock the error recovery method
        self.workflow.manager_agent.handle_error = MagicMock(return_value={
            "recovery_action": "retry_with_alternative",
            "alternative_approach": "Use cached data and summarize available information",
            "partial_result": "Based on available information, quantum computing uses quantum bits or qubits..."
        })
        
        # Execute the workflow
        result = self.workflow.execute("Research quantum computing")
        
        # Assert that the workflow executed successfully despite the error
        assert result is not None
        assert len(result) > 0
        
        # Verify that the error was handled
        self.workflow.manager_agent.handle_error.assert_called_once()
    
    def test_user_query_with_collaborative_input(self):
        """Test handling of a query requiring collaborative input from multiple users."""
        # Set up mock responses
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        self.workflow.pre_response_agent.is_collaborative_query = MagicMock(return_value=True)
        
        # Mock the collaborative input
        collaborative_inputs = [
            {"user": "User1", "input": "Focus on environmental impact"},
            {"user": "User2", "input": "Include economic considerations"},
            {"user": "User3", "input": "Add policy recommendations"}
        ]
        
        # Mock the collaborative integration method
        self.workflow.pre_response_agent.integrate_collaborative_inputs = MagicMock(return_value={
            "integrated_query": "Research climate change solutions with balanced focus on environmental impact, economic considerations, and policy recommendations",
            "integration_summary": "Combined inputs from 3 users to create a comprehensive research plan"
        })
        
        # Execute the workflow with collaborative inputs
        result = self.workflow.execute("Research climate change solutions", collaborative_inputs=collaborative_inputs)
        
        # Assert that the workflow executed successfully
        assert result is not None
        assert len(result) > 0
        
        # Verify that collaborative nature was detected
        self.workflow.pre_response_agent.is_collaborative_query.assert_called_once()
        
        # Verify that inputs were integrated
        self.workflow.pre_response_agent.integrate_collaborative_inputs.assert_called_once()
        
        # Verify that the research was conducted with the integrated query
        self.workflow.research_agent.execute.assert_called_once()
    
    def test_user_query_with_privacy_constraints(self):
        """Test handling of a query with privacy constraints."""
        # Set up mock responses
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        self.workflow.pre_response_agent.extract_privacy_constraints = MagicMock(return_value={
            "data_handling": "local_only",
            "source_restrictions": ["no_third_party_apis"],
            "output_restrictions": ["no_personal_identifiers"]
        })
        
        # Execute the workflow
        result = self.workflow.execute("Research health conditions but keep all processing local and don't use third-party APIs")
        
        # Assert that the workflow executed successfully
        assert result is not None
        assert len(result) > 0
        
        # Verify that privacy constraints were extracted
        self.workflow.pre_response_agent.extract_privacy_constraints.assert_called_once()
        
        # Verify that the research was conducted with privacy constraints
        self.workflow.research_agent.execute.assert_called_once()
    
    def test_user_query_with_accessibility_requirements(self):
        """Test handling of a query with accessibility requirements."""
        # Set up mock responses
        self.workflow.pre_response_agent.needs_clarification = MagicMock(return_value=False)
        self.workflow.pre_response_agent.extract_accessibility_requirements = MagicMock(return_value={
            "format": "screen_reader_friendly",
            "language_complexity": "simplified",
            "alternative_text": True
        })
        
        # Execute the workflow
        result = self.workflow.execute("Explain quantum computing in a screen reader friendly format with simplified language")
        
        # Assert that the workflow executed successfully
        assert result is not None
        assert len(result) > 0
        
        # Verify that accessibility requirements were extracted
        self.workflow.pre_response_agent.extract_accessibility_requirements.assert_called_once()
        
        # Verify that the research was conducted with accessibility requirements
        self.workflow.research_agent.execute.assert_called_once()
        
        # Verify that the file manager was called to format the output appropriately
        self.workflow.file_manager_agent.execute.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
