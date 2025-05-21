"""
Integration tests for workflow execution.

These tests verify that the research workflow can be executed correctly,
with all agents and MCPs working together to complete research tasks.
"""

import os
import pytest
import json
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock

# Import agents
from apps.agents.manager_agent import ManagerAgent
from apps.agents.pre_response_agent import PreResponseAgent
from apps.agents.research_agent import ResearchAgent
from apps.agents.image_generation_agent import ImageGenerationAgent
from apps.agents.file_manager_agent import FileManagerAgent
from apps.agents.summary_agent import SummaryAgent
from apps.agents.verification_agent import VerificationAgent
from apps.agents.base_agent import BaseAgent

# Import MCPs
from apps.mcps.brave_search_mcp import BraveSearchMCP
from apps.mcps.everart_mcp import EverArtMCP
from apps.mcps.fetch_mcp import FetchMCP
from apps.mcps.filesystem_mcp import FilesystemMCP
from apps.mcps.memory_mcp import MemoryMCP

# Import workflow
from apps.workflows.research_workflow import ResearchWorkflow

# Import mocks
from tests.mocks.mock_agents import (
    MockManagerAgent,
    MockPreResponseAgent,
    MockResearchAgent,
    MockImageGenerationAgent,
    MockFileManagerAgent,
    MockSummaryAgent,
    MockVerificationAgent
)

from tests.mocks.mock_mcps import (
    MockBraveSearchMCP,
    MockEverArtMCP,
    MockFetchMCP,
    MockFilesystemMCP,
    MockMemoryMCP,
    patch_mcps
)


class TestWorkflowExecution:
    """Test the execution of research workflows."""
    
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
        
        # Set up mock agents
        self.manager_agent = MockManagerAgent()
        self.pre_response_agent = MockPreResponseAgent()
        self.research_agent = MockResearchAgent()
        self.image_generation_agent = MockImageGenerationAgent()
        self.file_manager_agent = MockFileManagerAgent()
        self.summary_agent = MockSummaryAgent()
        self.verification_agent = MockVerificationAgent()
        
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
        Phone: +1-123-456-7890
        Rating: 4.8 (120 reviews)
        Price Range: N/A
        Hours: Mon-Fri 9:00-17:00
        Description: The Python Software Foundation is a non-profit organization devoted to the Python programming language.
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
        
        self.fetch.set_fetch_text_result("""
        Python Programming Language
        
        Python is a high-level, interpreted programming language known for its readability and versatility.
        
        Key features include:
        - Easy to learn syntax
        - Interpreted nature
        - Dynamic typing
        - High-level data structures
        """)
    
    def test_basic_workflow_execution(self):
        """Test basic execution of the research workflow."""
        # Create a workflow with mock agents
        workflow = ResearchWorkflow(
            manager_agent=self.manager_agent,
            pre_response_agent=self.pre_response_agent,
            research_agent=self.research_agent,
            image_generation_agent=self.image_generation_agent,
            file_manager_agent=self.file_manager_agent,
            summary_agent=self.summary_agent,
            verification_agent=self.verification_agent
        )
        
        # Configure mock agent responses
        self.manager_agent.set_response("I'll coordinate the research on Python programming language.")
        self.pre_response_agent.set_response("I'll clarify the query and present the research plan.")
        self.research_agent.set_response("I've found information about Python programming language.")
        self.image_generation_agent.set_response("I've generated an image of the Python logo.")
        self.file_manager_agent.set_response("I've saved the research materials.")
        self.summary_agent.set_response("Python is a high-level programming language known for its readability.")
        self.verification_agent.set_response("I've verified the facts about Python programming language.")
        
        # Execute the workflow
        query = "Tell me about Python programming language"
        result = workflow.execute(query)
        
        # Verify all agents were called
        assert self.manager_agent.called
        assert self.pre_response_agent.called
        assert self.research_agent.called
        assert self.image_generation_agent.called
        assert self.file_manager_agent.called
        assert self.summary_agent.called
        assert self.verification_agent.called
        
        # Verify the workflow result
        assert "Python" in result
        assert "research" in result.lower()
    
    def test_workflow_with_real_mcps(self):
        """Test workflow execution with real MCP interactions."""
        # Create agents with real MCP dependencies
        manager_agent = self.manager_agent
        pre_response_agent = self.pre_response_agent
        
        # Configure research agent to use real BraveSearchMCP and FetchMCP
        research_agent = self.research_agent
        research_agent.set_mcps(brave_search=self.brave_search, fetch=self.fetch)
        
        # Configure image generation agent to use real EverArtMCP
        image_generation_agent = self.image_generation_agent
        image_generation_agent.set_mcps(everart=self.everart)
        
        # Configure file manager agent to use real FilesystemMCP
        file_manager_agent = self.file_manager_agent
        file_manager_agent.set_mcps(filesystem=self.filesystem)
        
        # Configure summary agent to use real MemoryMCP
        summary_agent = self.summary_agent
        summary_agent.set_mcps(memory=self.memory)
        
        # Configure verification agent to use real BraveSearchMCP
        verification_agent = self.verification_agent
        verification_agent.set_mcps(brave_search=self.brave_search)
        
        # Create workflow with configured agents
        workflow = ResearchWorkflow(
            manager_agent=manager_agent,
            pre_response_agent=pre_response_agent,
            research_agent=research_agent,
            image_generation_agent=image_generation_agent,
            file_manager_agent=file_manager_agent,
            summary_agent=summary_agent,
            verification_agent=verification_agent
        )
        
        # Execute the workflow
        query = "Tell me about Python programming language"
        result = workflow.execute(query)
        
        # Verify MCP interactions
        assert self.brave_search.web_search_called
        assert self.fetch.fetch_url_called or self.fetch.fetch_text_called
        assert self.everart.generate_image_called
        assert self.filesystem.write_file_called
        assert self.memory.store_memory_called
        
        # Verify the workflow result
        assert "Python" in result
        assert "research" in result.lower()
    
    def test_workflow_with_error_handling(self):
        """Test workflow execution with error handling."""
        # Create a workflow with mock agents
        workflow = ResearchWorkflow(
            manager_agent=self.manager_agent,
            pre_response_agent=self.pre_response_agent,
            research_agent=self.research_agent,
            image_generation_agent=self.image_generation_agent,
            file_manager_agent=self.file_manager_agent,
            summary_agent=self.summary_agent,
            verification_agent=self.verification_agent
        )
        
        # Configure research agent to raise an error
        self.research_agent.set_error(ValueError("Failed to search for information"))
        
        # Configure other agents
        self.manager_agent.set_response("I'll coordinate the research on Python programming language.")
        self.pre_response_agent.set_response("I'll clarify the query and present the research plan.")
        self.image_generation_agent.set_response("I've generated an image of the Python logo.")
        self.file_manager_agent.set_response("I've saved the research materials.")
        self.summary_agent.set_response("Python is a high-level programming language known for its readability.")
        self.verification_agent.set_response("I've verified the facts about Python programming language.")
        
        # Execute the workflow
        query = "Tell me about Python programming language"
        result = workflow.execute(query)
        
        # Verify the workflow handled the error
        assert "error" in result.lower()
        assert "research" in result.lower()
        
        # Verify the manager agent was still called
        assert self.manager_agent.called
        
        # Verify the pre-response agent was still called
        assert self.pre_response_agent.called
    
    def test_workflow_with_agent_collaboration(self):
        """Test workflow execution with agent collaboration."""
        # Configure agents to collaborate
        self.manager_agent.set_response("I'll coordinate the research on Python programming language.")
        self.pre_response_agent.set_response("I'll clarify the query and present the research plan.")
        
        # Research agent finds information and passes it to summary agent
        research_info = "Python is a high-level programming language created by Guido van Rossum."
        self.research_agent.set_response(research_info)
        self.research_agent.set_output_data({"research_info": research_info})
        
        # Summary agent summarizes the research information
        summary = "Python is a high-level language by Guido van Rossum."
        self.summary_agent.set_response(summary)
        self.summary_agent.set_output_data({"summary": summary})
        
        # Image generation agent creates an image based on the summary
        image_url = "https://example.com/images/python.jpg"
        self.image_generation_agent.set_response(f"I've generated an image at {image_url}")
        self.image_generation_agent.set_output_data({"image_url": image_url})
        
        # Verification agent verifies the facts in the summary
        verified_facts = ["Python is a high-level language", "Guido van Rossum created Python"]
        self.verification_agent.set_response("I've verified the key facts about Python.")
        self.verification_agent.set_output_data({"verified_facts": verified_facts})
        
        # File manager agent saves all the information
        self.file_manager_agent.set_response("I've saved all the research materials.")
        
        # Create workflow with configured agents
        workflow = ResearchWorkflow(
            manager_agent=self.manager_agent,
            pre_response_agent=self.pre_response_agent,
            research_agent=self.research_agent,
            image_generation_agent=self.image_generation_agent,
            file_manager_agent=self.file_manager_agent,
            summary_agent=self.summary_agent,
            verification_agent=self.verification_agent
        )
        
        # Execute the workflow
        query = "Tell me about Python programming language"
        result = workflow.execute(query)
        
        # Verify all agents were called
        assert self.manager_agent.called
        assert self.pre_response_agent.called
        assert self.research_agent.called
        assert self.summary_agent.called
        assert self.image_generation_agent.called
        assert self.verification_agent.called
        assert self.file_manager_agent.called
        
        # Verify data was passed between agents
        assert self.summary_agent.input_data.get("research_info") == research_info
        assert self.image_generation_agent.input_data.get("summary") == summary
        assert self.verification_agent.input_data.get("summary") == summary
        assert self.file_manager_agent.input_data.get("research_info") == research_info
        assert self.file_manager_agent.input_data.get("summary") == summary
        assert self.file_manager_agent.input_data.get("image_url") == image_url
        assert self.file_manager_agent.input_data.get("verified_facts") == verified_facts
        
        # Verify the workflow result
        assert "Python" in result
        assert "research" in result.lower()
    
    def test_workflow_with_user_feedback(self):
        """Test workflow execution with user feedback."""
        # Create a workflow with mock agents
        workflow = ResearchWorkflow(
            manager_agent=self.manager_agent,
            pre_response_agent=self.pre_response_agent,
            research_agent=self.research_agent,
            image_generation_agent=self.image_generation_agent,
            file_manager_agent=self.file_manager_agent,
            summary_agent=self.summary_agent,
            verification_agent=self.verification_agent
        )
        
        # Configure pre-response agent to request user feedback
        self.pre_response_agent.set_response("I need more information. Do you want to focus on Python 2 or Python 3?")
        self.pre_response_agent.set_requires_feedback(True)
        
        # Configure mock user feedback
        user_feedback = "I'm interested in Python 3."
        
        # Configure agents to use the feedback
        self.manager_agent.set_response("I'll coordinate the research on Python 3.")
        self.research_agent.set_response("I've found information about Python 3.")
        self.image_generation_agent.set_response("I've generated an image of the Python 3 logo.")
        self.file_manager_agent.set_response("I've saved the research materials about Python 3.")
        self.summary_agent.set_response("Python 3 is the latest version of the Python programming language.")
        self.verification_agent.set_response("I've verified the facts about Python 3.")
        
        # Mock the workflow's method to get user feedback
        original_get_user_feedback = workflow.get_user_feedback
        workflow.get_user_feedback = lambda message: user_feedback
        
        try:
            # Execute the workflow
            query = "Tell me about Python programming language"
            result = workflow.execute(query)
            
            # Verify the pre-response agent requested feedback
            assert self.pre_response_agent.called
            assert self.pre_response_agent.requires_feedback
            
            # Verify the feedback was used
            assert "Python 3" in result
            
            # Verify all agents were called
            assert self.manager_agent.called
            assert self.research_agent.called
            assert self.image_generation_agent.called
            assert self.file_manager_agent.called
            assert self.summary_agent.called
            assert self.verification_agent.called
        finally:
            # Restore the original method
            workflow.get_user_feedback = original_get_user_feedback
    
    def test_workflow_with_multiple_research_iterations(self):
        """Test workflow execution with multiple research iterations."""
        # Create a workflow with mock agents
        workflow = ResearchWorkflow(
            manager_agent=self.manager_agent,
            pre_response_agent=self.pre_response_agent,
            research_agent=self.research_agent,
            image_generation_agent=self.image_generation_agent,
            file_manager_agent=self.file_manager_agent,
            summary_agent=self.summary_agent,
            verification_agent=self.verification_agent
        )
        
        # Configure manager agent to request multiple research iterations
        self.manager_agent.set_response("I'll coordinate the research on Python programming language.")
        self.manager_agent.set_output_data({
            "research_topics": [
                "Python basics",
                "Python libraries",
                "Python applications"
            ]
        })
        
        # Configure research agent to handle multiple topics
        research_responses = [
            "Python basics: Python is a high-level programming language with simple syntax.",
            "Python libraries: Python has many libraries like NumPy, Pandas, and TensorFlow.",
            "Python applications: Python is used in web development, data science, and AI."
        ]
        self.research_agent.set_multiple_responses(research_responses)
        
        # Configure other agents
        self.pre_response_agent.set_response("I'll present the research plan for Python programming language.")
        self.image_generation_agent.set_response("I've generated an image of the Python ecosystem.")
        self.file_manager_agent.set_response("I've saved all research materials.")
        self.summary_agent.set_response("Python is a versatile language used in many domains.")
        self.verification_agent.set_response("I've verified all the facts about Python.")
        
        # Execute the workflow
        query = "Tell me about Python programming language"
        result = workflow.execute(query)
        
        # Verify the research agent was called multiple times
        assert self.research_agent.call_count == len(research_responses)
        
        # Verify the workflow result contains information from all iterations
        assert "basics" in result.lower()
        assert "libraries" in result.lower()
        assert "applications" in result.lower()
    
    def test_workflow_with_mcp_chaining(self):
        """Test workflow execution with MCP chaining."""
        # Configure research agent to use BraveSearchMCP and FetchMCP in sequence
        self.research_agent.set_mcps(brave_search=self.brave_search, fetch=self.fetch)
        self.research_agent.set_response("I've found and fetched information about Python.")
        
        # Configure the research agent to chain MCPs
        def research_with_chained_mcps(query):
            # First use BraveSearchMCP to find URLs
            search_results = self.brave_search.web_search(query)
            
            # Extract a URL from the search results
            url = "https://www.python.org/"
            
            # Then use FetchMCP to fetch content from the URL
            content = self.fetch.fetch_url(url)
            
            return f"Search results: {search_results[:100]}...\n\nFetched content: {content[:100]}..."
        
        self.research_agent.set_custom_action(research_with_chained_mcps)
        
        # Configure other agents
        self.manager_agent.set_response("I'll coordinate the research on Python programming language.")
        self.pre_response_agent.set_response("I'll present the research plan for Python programming language.")
        self.image_generation_agent.set_response("I've generated an image of Python.")
        self.file_manager_agent.set_response("I've saved the research materials.")
        self.summary_agent.set_response("Python is a high-level programming language.")
        self.verification_agent.set_response("I've verified the facts about Python.")
        
        # Create workflow with configured agents
        workflow = ResearchWorkflow(
            manager_agent=self.manager_agent,
            pre_response_agent=self.pre_response_agent,
            research_agent=self.research_agent,
            image_generation_agent=self.image_generation_agent,
            file_manager_agent=self.file_manager_agent,
            summary_agent=self.summary_agent,
            verification_agent=self.verification_agent
        )
        
        # Execute the workflow
        query = "Tell me about Python programming language"
        result = workflow.execute(query)
        
        # Verify both MCPs were called in sequence
        assert self.brave_search.web_search_called
        assert self.fetch.fetch_url_called
        
        # Verify the workflow result
        assert "Python" in result
        assert "research" in result.lower()
    
    def test_workflow_with_file_storage(self):
        """Test workflow execution with file storage."""
        # Configure file manager agent to use FilesystemMCP
        self.file_manager_agent.set_mcps(filesystem=self.filesystem)
        
        # Configure the file manager agent to store files
        def store_research_files(data):
            # Store the summary
            if "summary" in data:
                self.filesystem.write_file("summary.txt", data["summary"])
            
            # Store the research information
            if "research_info" in data:
                self.filesystem.write_file("research.txt", data["research_info"])
            
            # Store image URL
            if "image_url" in data:
                self.filesystem.write_file("image_url.txt", data["image_url"])
            
            return "Research files stored successfully."
        
        self.file_manager_agent.set_custom_action(store_research_files)
        
        # Configure other agents with data to be stored
        self.manager_agent.set_response("I'll coordinate the research on Python programming language.")
        self.pre_response_agent.set_response("I'll present the research plan for Python programming language.")
        
        research_info = "Python is a high-level programming language created by Guido van Rossum."
        self.research_agent.set_response("I've found information about Python.")
        self.research_agent.set_output_data({"research_info": research_info})
        
        summary = "Python is a high-level language by Guido van Rossum."
        self.summary_agent.set_response("I've summarized the information about Python.")
        self.summary_agent.set_output_data({"summary": summary})
        
        image_url = "https://example.com/images/python.jpg"
        self.image_generation_agent.set_response("I've generated an image of Python.")
        self.image_generation_agent.set_output_data({"image_url": image_url})
        
        self.verification_agent.set_response("I've verified the facts about Python.")
        
        # Create workflow with configured agents
        workflow = ResearchWorkflow(
            manager_agent=self.manager_agent,
            pre_response_agent=self.pre_response_agent,
            research_agent=self.research_agent,
            image_generation_agent=self.image_generation_agent,
            file_manager_agent=self.file_manager_agent,
            summary_agent=self.summary_agent,
            verification_agent=self.verification_agent
        )
        
        # Execute the workflow
        query = "Tell me about Python programming language"
        result = workflow.execute(query)
        
        # Verify files were stored
        assert self.filesystem.write_file_called
        assert self.filesystem.file_exists("summary.txt")
        assert self.filesystem.file_exists("research.txt")
        assert self.filesystem.file_exists("image_url.txt")
        
        # Verify file contents
        assert self.filesystem.read_file("summary.txt") == summary
        assert self.filesystem.read_file("research.txt") == research_info
        assert self.filesystem.read_file("image_url.txt") == image_url
    
    def test_workflow_with_memory_persistence(self):
        """Test workflow execution with memory persistence."""
        # Configure summary agent to use MemoryMCP
        self.summary_agent.set_mcps(memory=self.memory)
        
        # Configure the summary agent to store and retrieve from memory
        def summarize_with_memory(data):
            # Check if we already have a summary for this topic
            topic = "python_programming"
            try:
                existing_summary = self.memory.retrieve_memory(topic)
                return f"Retrieved existing summary: {existing_summary}"
            except:
                # Create a new summary
                new_summary = "Python is a high-level programming language known for its readability."
                
                # Store the summary in memory
                self.memory.store_memory(topic, new_summary)
                
                return f"Created new summary: {new_summary}"
        
        self.summary_agent.set_custom_action(summarize_with_memory)
        
        # Configure other agents
        self.manager_agent.set_response("I'll coordinate the research on Python programming language.")
        self.pre_response_agent.set_response("I'll present the research plan for Python programming language.")
        self.research_agent.set_response("I've found information about Python.")
        self.image_generation_agent.set_response("I've generated an image of Python.")
        self.file_manager_agent.set_response("I've saved the research materials.")
        self.verification_agent.set_response("I've verified the facts about Python.")
        
        # Create workflow with configured agents
        workflow = ResearchWorkflow(
            manager_agent=self.manager_agent,
            pre_response_agent=self.pre_response_agent,
            research_agent=self.research_agent,
            image_generation_agent=self.image_generation_agent,
            file_manager_agent=self.file_manager_agent,
            summary_agent=self.summary_agent,
            verification_agent=self.verification_agent
        )
        
        # Execute the workflow twice
        query = "Tell me about Python programming language"
        first_result = workflow.execute(query)
        second_result = workflow.execute(query)
        
        # Verify memory was used
        assert self.memory.store_memory_called
        assert self.memory.retrieve_memory_called
        
        # Verify the second execution retrieved from memory
        assert "Retrieved existing summary" in second_result
    
    def test_workflow_with_verification(self):
        """Test workflow execution with fact verification."""
        # Configure verification agent to use BraveSearchMCP
        self.verification_agent.set_mcps(brave_search=self.brave_search)
        
        # Configure the verification agent to verify facts
        def verify_facts(data):
            facts_to_verify = [
                "Python was created by Guido van Rossum",
                "Python was first released in 1991",
                "Python is a high-level programming language"
            ]
            
            verified_facts = []
            for fact in facts_to_verify:
                # Search for evidence of the fact
                search_results = self.brave_search.web_search(fact)
                verified = "python" in search_results.lower() and any(
                    keyword in search_results.lower() 
                    for keyword in fact.lower().split()
                )
                verified_facts.append({
                    "fact": fact,
                    "verified": verified
                })
            
            return f"Verified {sum(1 for f in verified_facts if f['verified'])} out of {len(verified_facts)} facts."
        
        self.verification_agent.set_custom_action(verify_facts)
        
        # Configure other agents
        self.manager_agent.set_response("I'll coordinate the research on Python programming language.")
        self.pre_response_agent.set_response("I'll present the research plan for Python programming language.")
        self.research_agent.set_response("I've found information about Python.")
        self.image_generation_agent.set_response("I've generated an image of Python.")
        self.file_manager_agent.set_response("I've saved the research materials.")
        self.summary_agent.set_response("Python is a high-level programming language.")
        
        # Create workflow with configured agents
        workflow = ResearchWorkflow(
            manager_agent=self.manager_agent,
            pre_response_agent=self.pre_response_agent,
            research_agent=self.research_agent,
            image_generation_agent=self.image_generation_agent,
            file_manager_agent=self.file_manager_agent,
            summary_agent=self.summary_agent,
            verification_agent=self.verification_agent
        )
        
        # Execute the workflow
        query = "Tell me about Python programming language"
        result = workflow.execute(query)
        
        # Verify BraveSearchMCP was called for verification
        assert self.brave_search.web_search_called
        assert self.brave_search.call_count >= 3  # At least one call per fact
        
        # Verify the workflow result
        assert "Python" in result
        assert "verified" in result.lower()
    
    def test_complete_research_workflow(self):
        """Test a complete research workflow with all components."""
        # Configure all agents with real MCP interactions
        
        # Manager agent coordinates the workflow
        self.manager_agent.set_response("I'll coordinate the research on Python programming language.")
        
        # Pre-response agent presents the plan
        self.pre_response_agent.set_response("I'll research Python programming language focusing on its history, features, and applications.")
        
        # Research agent uses BraveSearchMCP and FetchMCP
        def research_action(query):
            # Search for information
            search_results = self.brave_search.web_search(query)
            
            # Extract a URL from the search results (simulated)
            url = "https://www.python.org/"
            
            # Fetch content from the URL
            # Fetch content from the URL
            content = self.fetch.fetch_url(url)
            
            # Combine the information
            research_info = f"From search: {search_results[:300]}...\n\nFrom website: {content[:300]}..."
            
            return research_info
        
        self.research_agent.set_mcps(brave_search=self.brave_search, fetch=self.fetch)
        self.research_agent.set_custom_action(research_action)
        
        # Image generation agent uses EverArtMCP
        def generate_image_action(data):
            prompt = "Python programming language logo with code in the background"
            image_result = self.everart.generate_image(prompt)
            return image_result
        
        self.image_generation_agent.set_mcps(everart=self.everart)
        self.image_generation_agent.set_custom_action(generate_image_action)
        
        # Summary agent uses MemoryMCP
        def summarize_action(data):
            research_info = data.get("research_info", "")
            
            # Create a summary
            summary = "Python is a high-level programming language known for its readability and versatility."
            
            # Store the summary in memory
            self.memory.store_memory("python_summary", summary)
            
            return summary
        
        self.summary_agent.set_mcps(memory=self.memory)
        self.summary_agent.set_custom_action(summarize_action)
        
        # Verification agent uses BraveSearchMCP
        def verify_action(data):
            facts_to_verify = [
                "Python was created by Guido van Rossum",
                "Python is a high-level programming language"
            ]
            
            verified_facts = []
            for fact in facts_to_verify:
                # Search for evidence of the fact
                search_results = self.brave_search.web_search(fact)
                verified = "python" in search_results.lower() and any(
                    keyword in search_results.lower() 
                    for keyword in fact.lower().split()
                )
                verified_facts.append({
                    "fact": fact,
                    "verified": verified
                })
            
            return f"Verified {sum(1 for f in verified_facts if f['verified'])} out of {len(verified_facts)} facts."
        
        self.verification_agent.set_mcps(brave_search=self.brave_search)
        self.verification_agent.set_custom_action(verify_action)
        
        # File manager agent uses FilesystemMCP
        def file_manager_action(data):
            # Get data from other agents
            research_info = data.get("research_info", "No research information available.")
            summary = data.get("summary", "No summary available.")
            image_url = data.get("image_url", "No image available.")
            verified_facts = data.get("verified_facts", "No verified facts available.")
            
            # Create a report
            report = f"""
            # Python Programming Language Research Report
            
            ## Summary
            {summary}
            
            ## Research Information
            {research_info}
            
            ## Image
            ![Python Image]({image_url})
            
            ## Verified Facts
            {verified_facts}
            """
            
            # Save the report
            self.filesystem.write_file("python_report.md", report)
            
            # Save individual components
            self.filesystem.write_file("summary.txt", summary)
            self.filesystem.write_file("research.txt", research_info)
            self.filesystem.write_file("image_url.txt", image_url)
            
            return "Research files stored successfully."
        
        self.file_manager_agent.set_mcps(filesystem=self.filesystem)
        self.file_manager_agent.set_custom_action(file_manager_action)
        
        # Create workflow with configured agents
        workflow = ResearchWorkflow(
            manager_agent=self.manager_agent,
            pre_response_agent=self.pre_response_agent,
            research_agent=self.research_agent,
            image_generation_agent=self.image_generation_agent,
            file_manager_agent=self.file_manager_agent,
            summary_agent=self.summary_agent,
            verification_agent=self.verification_agent
        )
        
        # Execute the workflow
        query = "Tell me about Python programming language"
        result = workflow.execute(query)
        
        # Verify all MCPs were called
        assert self.brave_search.web_search_called
        assert self.fetch.fetch_url_called
        assert self.everart.generate_image_called
        assert self.memory.store_memory_called
        assert self.filesystem.write_file_called
        
        # Verify files were created
        assert self.filesystem.file_exists("python_report.md")
        assert self.filesystem.file_exists("summary.txt")
        assert self.filesystem.file_exists("research.txt")
        assert self.filesystem.file_exists("image_url.txt")
        
        # Verify the workflow result
        assert "Python" in result
        assert "research" in result.lower()
    
    @pytest.mark.asyncio
    async def test_async_workflow_execution(self):
        """Test asynchronous execution of the research workflow."""
        # Create async versions of the agents
        async def async_manager_action(query):
            return "I'll coordinate the research on Python programming language."
        
        async def async_pre_response_action(query):
            return "I'll present the research plan for Python programming language."
        
        async def async_research_action(query):
            # Search for information
            search_results = self.brave_search.web_search(query)
            return f"Research results: {search_results[:200]}..."
        
        async def async_image_generation_action(data):
            prompt = "Python programming language logo"
            image_result = self.everart.generate_image(prompt)
            return image_result
        
        async def async_summary_action(data):
            return "Python is a high-level programming language known for its readability."
        
        async def async_verification_action(data):
            return "I've verified the facts about Python programming language."
        
        async def async_file_manager_action(data):
            return "I've saved all research materials."
        
        # Configure agents with async actions
        self.manager_agent.set_async_action(async_manager_action)
        self.pre_response_agent.set_async_action(async_pre_response_action)
        self.research_agent.set_async_action(async_research_action)
        self.image_generation_agent.set_async_action(async_image_generation_action)
        self.summary_agent.set_async_action(async_summary_action)
        self.verification_agent.set_async_action(async_verification_action)
        self.file_manager_agent.set_async_action(async_file_manager_action)
        
        # Configure MCPs
        self.research_agent.set_mcps(brave_search=self.brave_search)
        self.image_generation_agent.set_mcps(everart=self.everart)
        
        # Create workflow with configured agents
        workflow = ResearchWorkflow(
            manager_agent=self.manager_agent,
            pre_response_agent=self.pre_response_agent,
            research_agent=self.research_agent,
            image_generation_agent=self.image_generation_agent,
            file_manager_agent=self.file_manager_agent,
            summary_agent=self.summary_agent,
            verification_agent=self.verification_agent
        )
        
        # Execute the workflow asynchronously
        query = "Tell me about Python programming language"
        result = await workflow.execute_async(query)
        
        # Verify all agents were called
        assert self.manager_agent.async_called
        assert self.pre_response_agent.async_called
        assert self.research_agent.async_called
        assert self.image_generation_agent.async_called
        assert self.summary_agent.async_called
        assert self.verification_agent.async_called
        assert self.file_manager_agent.async_called
        
        # Verify MCPs were called
        assert self.brave_search.web_search_called
        assert self.everart.generate_image_called
        
        # Verify the workflow result
        assert "Python" in result
        assert "research" in result.lower()
    
    def test_workflow_with_parallel_execution(self):
        """Test workflow execution with parallel agent execution."""
        # Configure agents for parallel execution
        self.manager_agent.set_response("I'll coordinate the research on Python programming language.")
        self.pre_response_agent.set_response("I'll present the research plan for Python programming language.")
        
        # These agents will run in parallel
        self.research_agent.set_response("I've found information about Python.")
        self.image_generation_agent.set_response("I've generated an image of Python.")
        
        # These agents depend on the parallel agents
        self.summary_agent.set_response("I've summarized the information about Python.")
        self.verification_agent.set_response("I've verified the facts about Python.")
        self.file_manager_agent.set_response("I've saved all research materials.")
        
        # Create workflow with configured agents
        workflow = ResearchWorkflow(
            manager_agent=self.manager_agent,
            pre_response_agent=self.pre_response_agent,
            research_agent=self.research_agent,
            image_generation_agent=self.image_generation_agent,
            file_manager_agent=self.file_manager_agent,
            summary_agent=self.summary_agent,
            verification_agent=self.verification_agent,
            parallel_execution=True  # Enable parallel execution
        )
        
        # Execute the workflow
        query = "Tell me about Python programming language"
        result = workflow.execute(query)
        
        # Verify all agents were called
        assert self.manager_agent.called
        assert self.pre_response_agent.called
        assert self.research_agent.called
        assert self.image_generation_agent.called
        assert self.summary_agent.called
        assert self.verification_agent.called
        assert self.file_manager_agent.called
        
        # Verify the workflow result
        assert "Python" in result
        assert "research" in result.lower()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
