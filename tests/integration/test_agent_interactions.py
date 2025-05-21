"""
Integration tests for agent interactions in the research workflow.

These tests verify that agents can communicate and collaborate effectively,
using mock MCPs to simulate external services.
"""

import os
import pytest
import uuid
from unittest.mock import patch, MagicMock
import json

# Import agents
from apps.agents.manager_agent import ManagerAgent
from apps.agents.pre_response_agent import PreResponseAgent
from apps.agents.research_agent import ResearchAgent
from apps.agents.summary_agent import SummaryAgent
from apps.agents.verification_agent import VerificationAgent
from apps.agents.image_generation_agent import ImageGenerationAgent
from apps.agents.file_manager_agent import FileManagerAgent

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

# Import workflow
from apps.workflows.research_workflow import ResearchWorkflow


class TestAgentInteractions:
    """Test interactions between different agents in the research workflow."""
    
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
        
        self.brave_search.set_local_search_result("""
        Name: Python Software Foundation
        Address: 9450 SW Gemini Dr, Beaverton, OR 97008, USA
        Phone: +1-555-123-4567
        Rating: 4.8 (120 reviews)
        Price Range: N/A
        Hours: Mon-Fri 9:00-17:00
        Description: The Python Software Foundation (PSF) is a non-profit organization devoted to the Python programming language.
        """)
        
        # Configure EverArtMCP
        self.everart.set_generate_image_result("""
        Image generated successfully!
        
        URL: https://example.com/images/mock-image-12345.jpg
        
        The image shows a Python logo with code snippets in the background, 
        created in the requested oil painting style.
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
        
        self.fetch.set_fetch_text_result("""
        Python Programming Language
        
        Python is a high-level, interpreted programming language known for its readability and versatility.
        
        Key features include:
        - Easy to learn syntax
        - Interpreted nature
        - Dynamic typing
        - High-level data structures
        """)
        
        # Configure FilesystemMCP
        self.filesystem.files = {
            "research_results.txt": "Initial research results placeholder",
            "images/python_logo.jpg": b"Mock image data"
        }
        
        # Configure MemoryMCP
        self.memory.memories = {
            "default": {
                "user_query": "Tell me about Python programming language",
                "research_plan": "1. Search for Python programming language\n2. Gather key information\n3. Generate summary",
                "search_results": "Found information about Python programming language from python.org, w3schools, and wikipedia."
            },
            "research": {
                "topic": "Python programming language",
                "sources": "python.org, w3schools.com, wikipedia.org",
                "key_points": "High-level, interpreted, dynamic typing, readability"
            }
        }
    
    def test_manager_agent_delegates_to_research_agent(self):
        """Test that the Manager Agent correctly delegates research tasks to the Research Agent."""
        # Create agents
        manager = ManagerAgent(memory_mcp=self.memory)
        research = ResearchAgent(brave_search_mcp=self.brave_search, fetch_mcp=self.fetch, memory_mcp=self.memory)
        
        # Mock the research agent's research method
        original_research = research.research
        research.research = MagicMock()
        research.research.return_value = {
            "status": "success",
            "results": "Research results about Python programming language",
            "sources": ["python.org", "w3schools.com", "wikipedia.org"]
        }
        
        try:
            # Manager delegates to research agent
            query = "Tell me about Python programming language"
            result = manager.delegate_research(query, research)
            
            # Verify research agent was called
            research.research.assert_called_once()
            call_args = research.research.call_args[0][0]
            assert query in call_args, f"Research query '{call_args}' should contain '{query}'"
            
            # Verify result was stored in memory
            assert self.memory.store_memory_called
            assert "research_results" in [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
            
        finally:
            # Restore original method
            research.research = original_research
    
    def test_research_agent_uses_brave_search_and_fetch(self):
        """Test that the Research Agent uses BraveSearchMCP and FetchMCP correctly."""
        # Create agent
        research = ResearchAgent(brave_search_mcp=self.brave_search, fetch_mcp=self.fetch, memory_mcp=self.memory)
        
        # Perform research
        query = "Python programming language"
        result = research.research(query)
        
        # Verify BraveSearchMCP was used
        assert self.brave_search.web_search_called
        assert query in self.brave_search.web_search_args[0]
        
        # Verify FetchMCP was used (should be called to fetch content from search results)
        assert self.fetch.fetch_url_called or self.fetch.fetch_text_called
        
        # Verify results were stored in memory
        assert self.memory.store_memory_called
    
    def test_summary_agent_creates_summary_from_research(self):
        """Test that the Summary Agent creates a summary from research results."""
        # Create agents
        summary_agent = SummaryAgent(memory_mcp=self.memory)
        
        # Store research results in memory
        research_results = """
        Python is a high-level, interpreted programming language known for its readability and versatility.
        Key features include easy syntax, dynamic typing, and high-level data structures.
        It was created by Guido van Rossum and released in 1991.
        Python is widely used in web development, data science, AI, and automation.
        """
        self.memory.store_memory("research_results", research_results, namespace="research")
        
        # Create summary
        result = summary_agent.create_summary("Python programming language")
        
        # Verify memory was accessed
        assert self.memory.retrieve_memory_called
        
        # Verify summary was stored
        assert self.memory.store_memory_called
        assert "summary" in [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
    
    def test_verification_agent_verifies_facts(self):
        """Test that the Verification Agent verifies facts from research."""
        # Create agent
        verification = VerificationAgent(brave_search_mcp=self.brave_search, memory_mcp=self.memory)
        
        # Store facts to verify in memory
        facts = [
            "Python was created by Guido van Rossum",
            "Python was first released in 1991",
            "Python is a high-level programming language"
        ]
        self.memory.store_memory("facts_to_verify", "\n".join(facts), namespace="research")
        
        # Verify facts
        result = verification.verify_facts("Python programming language")
        
        # Verify BraveSearchMCP was used for verification
        assert self.brave_search.web_search_called
        
        # Verify verification results were stored
        assert self.memory.store_memory_called
        assert "verified_facts" in [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
    
    def test_image_generation_agent_creates_images(self):
        """Test that the Image Generation Agent creates images based on research."""
        # Create agent
        image_gen = ImageGenerationAgent(everart_mcp=self.everart, memory_mcp=self.memory)
        
        # Store research summary in memory
        summary = "Python is a high-level programming language known for its readability and versatility."
        self.memory.store_memory("summary", summary, namespace="research")
        
        # Generate image
        result = image_gen.generate_image("Python programming language")
        
        # Verify EverArtMCP was used
        assert self.everart.generate_image_called
        assert "Python" in self.everart.generate_image_args[0]
        
        # Verify image URL was stored in memory
        assert self.memory.store_memory_called
        assert "image_url" in [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
    
    def test_file_manager_agent_saves_research_outputs(self):
        """Test that the File Manager Agent saves research outputs to files."""
        # Create agent
        file_manager = FileManagerAgent(filesystem_mcp=self.filesystem, memory_mcp=self.memory)
        
        # Store research outputs in memory
        self.memory.store_memory("summary", "Python summary text", namespace="research")
        self.memory.store_memory("image_url", "https://example.com/images/python.jpg", namespace="research")
        
        # Save research outputs
        result = file_manager.save_research_outputs("Python programming language")
        
        # Verify FilesystemMCP was used
        assert self.filesystem.write_file_called
        
        # Verify file paths were stored in memory
        assert self.memory.store_memory_called
        assert "output_files" in [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
    
    def test_pre_response_agent_creates_research_plan(self):
        """Test that the Pre-response Agent creates a research plan based on user query."""
        # Create agent
        pre_response = PreResponseAgent(memory_mcp=self.memory)
        
        # Create research plan
        query = "Tell me about Python programming language"
        result = pre_response.create_research_plan(query)
        
        # Verify plan was stored in memory
        assert self.memory.store_memory_called
        assert "research_plan" in [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
    
    def test_full_research_workflow_integration(self):
        """Test the full research workflow integration with all agents."""
        # Create workflow with all agents
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(memory_mcp=self.memory),
            pre_response_agent=PreResponseAgent(memory_mcp=self.memory),
            research_agent=ResearchAgent(brave_search_mcp=self.brave_search, fetch_mcp=self.fetch, memory_mcp=self.memory),
            summary_agent=SummaryAgent(memory_mcp=self.memory),
            verification_agent=VerificationAgent(brave_search_mcp=self.brave_search, memory_mcp=self.memory),
            image_generation_agent=ImageGenerationAgent(everart_mcp=self.everart, memory_mcp=self.memory),
            file_manager_agent=FileManagerAgent(filesystem_mcp=self.filesystem, memory_mcp=self.memory)
        )
        
        # Execute workflow
        query = "Tell me about Python programming language"
        result = workflow.execute(query)
        
        # Verify all agents were involved
        assert self.memory.retrieve_memory_called
        assert self.brave_search.web_search_called
        assert self.fetch.fetch_url_called or self.fetch.fetch_text_called
        assert self.everart.generate_image_called
        assert self.filesystem.write_file_called
        
        # Verify final results were stored
        assert "final_summary" in [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
        assert "output_files" in [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
        
        # Verify workflow returned success
        assert result["status"] == "success"
        assert "summary" in result
        assert "files" in result
        assert "images" in result
    
    def test_workflow_handles_errors_gracefully(self):
        """Test that the research workflow handles errors gracefully."""
        # Create workflow with all agents
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(memory_mcp=self.memory),
            pre_response_agent=PreResponseAgent(memory_mcp=self.memory),
            research_agent=ResearchAgent(brave_search_mcp=self.brave_search, fetch_mcp=self.fetch, memory_mcp=self.memory),
            summary_agent=SummaryAgent(memory_mcp=self.memory),
            verification_agent=VerificationAgent(brave_search_mcp=self.brave_search, memory_mcp=self.memory),
            image_generation_agent=ImageGenerationAgent(everart_mcp=self.everart, memory_mcp=self.memory),
            file_manager_agent=FileManagerAgent(filesystem_mcp=self.filesystem, memory_mcp=self.memory)
        )
        
        # Make BraveSearchMCP raise an exception
        self.brave_search.web_search = MagicMock(side_effect=RuntimeError("API rate limit exceeded"))
        
        # Execute workflow
        query = "Tell me about Python programming language"
        result = workflow.execute(query)
        
        # Verify workflow handled the error
        assert result["status"] == "error"
        assert "error" in result
        assert "rate limit" in result["error"].lower()
        
        # Verify error was logged in memory
        assert self.memory.store_memory_called
        assert "error_log" in [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
    
    def test_agents_communicate_through_memory(self):
        """Test that agents can communicate with each other through shared memory."""
        # Create agents
        manager = ManagerAgent(memory_mcp=self.memory)
        research = ResearchAgent(brave_search_mcp=self.brave_search, fetch_mcp=self.fetch, memory_mcp=self.memory)
        summary = SummaryAgent(memory_mcp=self.memory)
        
        # Manager stores a task in memory
        task_id = str(uuid.uuid4())
        manager.assign_task(task_id, "research", "Tell me about Python programming language")
        
        # Research agent retrieves the task and stores results
        task = research.get_assigned_task(task_id)
        assert task["type"] == "research"
        assert "Python" in task["query"]
        
        research.store_research_results(task_id, {
            "topic": "Python programming language",
            "sources": ["python.org", "w3schools.com"],
            "content": "Python is a high-level programming language."
        })
        
        # Summary agent retrieves research results and creates summary
        research_data = summary.get_research_data(task_id)
        assert "Python" in research_data["topic"]
        assert "high-level" in research_data["content"]
        
        summary.store_summary(task_id, "Python is a versatile high-level programming language.")
        
        # Manager retrieves the summary
        result = manager.get_task_result(task_id)
        assert "Python" in result["summary"]
        assert result["status"] == "completed"
    
    def test_parallel_agent_execution(self):
        """Test that multiple agents can execute tasks in parallel."""
        # This test simulates parallel execution by having multiple agents work on different aspects
        # of the same research topic simultaneously
        
        # Create agents
        research = ResearchAgent(brave_search_mcp=self.brave_search, fetch_mcp=self.fetch, memory_mcp=self.memory)
        verification = VerificationAgent(brave_search_mcp=self.brave_search, memory_mcp=self.memory)
        image_gen = ImageGenerationAgent(everart_mcp=self.everart, memory_mcp=self.memory)
        
        # Set up different mock responses for parallel searches
        self.brave_search.web_search = MagicMock(side_effect=[
            "Python was created by Guido van Rossum in 1991.",  # For research agent
            "Python is widely used in data science, web development, and AI.",  # For verification agent
        ])
        
        # Execute tasks "in parallel" (simulated)
        research_result = research.research("Python history")
        verification_result = verification.verify_facts("Python applications")
        image_result = image_gen.generate_image("Python logo")
        
        # Verify all tasks completed successfully
        assert "success" in research_result["status"]
        assert "success" in verification_result["status"]
        assert "success" in image_result["status"]
        
        # Verify memory contains results from all agents
        memory_keys = []
        for args in self.memory.store_memory_args:
            if isinstance(args, tuple) and len(args) > 0:
                memory_keys.append(args[0])
        
        assert any("research" in key for key in memory_keys)
        assert any("verif" in key for key in memory_keys)
        assert any("image" in key for key in memory_keys)
    
    def test_workflow_with_user_feedback(self):
        """Test the research workflow with simulated user feedback."""
        # Create workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(memory_mcp=self.memory),
            pre_response_agent=PreResponseAgent(memory_mcp=self.memory),
            research_agent=ResearchAgent(brave_search_mcp=self.brave_search, fetch_mcp=self.fetch, memory_mcp=self.memory),
            summary_agent=SummaryAgent(memory_mcp=self.memory),
            verification_agent=VerificationAgent(brave_search_mcp=self.brave_search, memory_mcp=self.memory),
            image_generation_agent=ImageGenerationAgent(everart_mcp=self.everart, memory_mcp=self.memory),
            file_manager_agent=FileManagerAgent(filesystem_mcp=self.filesystem, memory_mcp=self.memory)
        )
        
        # Start workflow
        query = "Tell me about Python programming language"
        initial_plan = workflow.create_research_plan(query)
        
        # Simulate user feedback
        user_feedback = "Please focus more on Python's applications in data science"
        workflow.incorporate_user_feedback(user_feedback)
        
        # Continue workflow
        result = workflow.execute(query)
        
        # Verify user feedback was incorporated
        assert self.memory.store_memory_called
        assert "user_feedback" in [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
        
        # Verify search queries reflect the feedback
        search_queries = [args[0] for args in self.brave_search.web_search_args]
        assert any("data science" in query.lower() for query in search_queries)
        
        # Verify final results reflect the feedback
        assert "data science" in result["summary"].lower()
    
    def test_multi_step_research_process(self):
        """Test a multi-step research process with progressive refinement."""
        # Create agents
        manager = ManagerAgent(memory_mcp=self.memory)
        research = ResearchAgent(brave_search_mcp=self.brave_search, fetch_mcp=self.fetch, memory_mcp=self.memory)
        summary = SummaryAgent(memory_mcp=self.memory)
        
        # Step 1: Initial broad research
        step1_query = "Python programming language overview"
        step1_result = research.research(step1_query)
        
        # Store initial findings
        self.memory.store_memory("initial_research", json.dumps(step1_result), namespace="multi_step")
        
        # Step 2: Identify specific areas to explore further
        areas_to_explore = ["Python in data science", "Python web frameworks", "Python for AI"]
        self.memory.store_memory("areas_to_explore", json.dumps(areas_to_explore), namespace="multi_step")
        
        # Step 3: Detailed research on each area
        detailed_results = {}
        for area in areas_to_explore:
            result = research.research(area)
            detailed_results[area] = result
        
        # Store detailed findings
        self.memory.store_memory("detailed_research", json.dumps(detailed_results), namespace="multi_step")
        
        # Step 4: Create comprehensive summary
        all_research = {
            "initial": json.loads(self.memory.retrieve_memory("initial_research", namespace="multi_step")),
            "detailed": json.loads(self.memory.retrieve_memory("detailed_research", namespace="multi_step"))
        }
        
        final_summary = summary.create_comprehensive_summary(all_research)
        self.memory.store_memory("final_summary", final_summary, namespace="multi_step")
        
        # Verify the multi-step process
        assert self.brave_search.web_search_called
        assert len(self.brave_search.web_search_args) >= len(areas_to_explore) + 1
        
        # Verify memory contains all the intermediate and final results
        assert self.memory.retrieve_memory_called
        assert "final_summary" in [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
        
        # Verify the final summary contains information from all research areas
        final_summary = self.memory.retrieve_memory("final_summary", namespace="multi_step")
        for area in areas_to_explore:
            assert area.lower().replace(" ", "") in final_summary.lower().replace(" ", "")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
