import os
import pytest
import time
import uuid
from unittest.mock import patch, MagicMock
import json

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


class TestResearchScenarios:
    """End-to-end tests for research scenarios."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        # Create a unique workspace directory for each test
        self.workspace_dir = os.path.join(os.path.dirname(__file__), "test_workspace", str(uuid.uuid4()))
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Set up logger
        self.logger = Logger("test_research_scenarios")
        
        # Create mock API keys
        self.brave_api_key = "test_brave_api_key"
        self.everart_api_key = "test_everart_api_key"
        
        yield
        
        # Clean up workspace directory
        import shutil
        if os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir)
    
    def create_mocked_workflow(self):
        """Create a research workflow with mocked MCPs and agents."""
        # Create mocked MCPs
        brave_search = MagicMock(spec=BraveSearchMCP)
        brave_search.web_search.return_value = "Mock web search results for Python programming language"
        brave_search.local_search.return_value = "Mock local search results for Python programming language"
        
        everart = MagicMock(spec=EverArtMCP)
        everart.generate_image.return_value = "https://example.com/mock-image.jpg"
        
        fetch = MagicMock(spec=FetchMCP)
        fetch.fetch_url.return_value = "Mock content from Python.org"
        fetch.fetch_text.return_value = "Python is a programming language that lets you work quickly and integrate systems more effectively."
        
        filesystem = MagicMock(spec=FilesystemMCP)
        filesystem.write_file.return_value = "File written successfully"
        filesystem.read_file.return_value = "Mock file content"
        
        memory = MagicMock(spec=MemoryMCP)
        memory.store_memory.return_value = "Memory stored successfully"
        memory.retrieve_memory.return_value = "Mock memory content"
        
        # Create agents with mocked MCPs
        manager_agent = ManagerAgent()
        manager_agent.memory_mcp = memory
        
        pre_response_agent = PreResponseAgent()
        pre_response_agent.memory_mcp = memory
        
        research_agent = ResearchAgent()
        research_agent.brave_search_mcp = brave_search
        research_agent.fetch_mcp = fetch
        research_agent.memory_mcp = memory
        
        image_generation_agent = ImageGenerationAgent()
        image_generation_agent.everart_mcp = everart
        image_generation_agent.memory_mcp = memory
        
        file_manager_agent = FileManagerAgent()
        file_manager_agent.filesystem_mcp = filesystem
        file_manager_agent.memory_mcp = memory
        
        summary_agent = SummaryAgent()
        summary_agent.memory_mcp = memory
        
        verification_agent = VerificationAgent()
        verification_agent.brave_search_mcp = brave_search
        verification_agent.memory_mcp = memory
        
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
        
        return workflow
    
    def test_basic_research_query(self):
        """Test a basic research query about a programming language."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate the research on Python programming language.")
        workflow.pre_response_agent.set_response("I'll present the research plan for Python programming language.")
        workflow.research_agent.set_response("I've found information about Python programming language.")
        workflow.image_generation_agent.set_response("I've generated an image of Python programming language.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("Python is a high-level programming language.")
        workflow.verification_agent.set_response("I've verified the facts about Python programming language.")
        
        # Execute the workflow
        result = workflow.execute("Tell me about Python programming language")
        
        # Assert that the result contains expected information
        assert "Python" in result
        assert "programming language" in result
        
        # Verify that all agents were called
        workflow.manager_agent.execute.assert_called_once()
        workflow.pre_response_agent.execute.assert_called_once()
        workflow.research_agent.execute.assert_called_once()
        workflow.image_generation_agent.execute.assert_called_once()
        workflow.file_manager_agent.execute.assert_called_once()
        workflow.summary_agent.execute.assert_called_once()
        workflow.verification_agent.execute.assert_called_once()
    
    def test_historical_research_query(self):
        """Test a historical research query."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate the research on World War II.")
        workflow.pre_response_agent.set_response("I'll present the research plan for World War II.")
        workflow.research_agent.set_response("I've found information about World War II (1939-1945).")
        workflow.image_generation_agent.set_response("I've generated an image related to World War II.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("World War II was a global conflict from 1939 to 1945.")
        workflow.verification_agent.set_response("I've verified the historical facts about World War II.")
        
        # Execute the workflow
        result = workflow.execute("Research the causes and impact of World War II")
        
        # Assert that the result contains expected information
        assert "World War II" in result
        assert "1939" in result or "1945" in result
        
        # Verify that all agents were called
        workflow.manager_agent.execute.assert_called_once()
        workflow.pre_response_agent.execute.assert_called_once()
        workflow.research_agent.execute.assert_called_once()
        workflow.image_generation_agent.execute.assert_called_once()
        workflow.file_manager_agent.execute.assert_called_once()
        workflow.summary_agent.execute.assert_called_once()
        workflow.verification_agent.execute.assert_called_once()
    
    def test_scientific_research_query(self):
        """Test a scientific research query."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate the research on quantum computing.")
        workflow.pre_response_agent.set_response("I'll present the research plan for quantum computing.")
        workflow.research_agent.set_response("I've found information about quantum computing principles.")
        workflow.image_generation_agent.set_response("I've generated an image illustrating quantum computing concepts.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("Quantum computing uses quantum mechanics to perform calculations.")
        workflow.verification_agent.set_response("I've verified the scientific facts about quantum computing.")
        
        # Execute the workflow
        result = workflow.execute("Explain the principles of quantum computing")
        
        # Assert that the result contains expected information
        assert "quantum" in result.lower()
        assert "computing" in result.lower()
        
        # Verify that all agents were called
        workflow.manager_agent.execute.assert_called_once()
        workflow.pre_response_agent.execute.assert_called_once()
        workflow.research_agent.execute.assert_called_once()
        workflow.image_generation_agent.execute.assert_called_once()
        workflow.file_manager_agent.execute.assert_called_once()
        workflow.summary_agent.execute.assert_called_once()
        workflow.verification_agent.execute.assert_called_once()
    
    def test_current_events_query(self):
        """Test a query about current events."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate the research on recent climate change policies.")
        workflow.pre_response_agent.set_response("I'll present the research plan for recent climate change policies.")
        workflow.research_agent.set_response("I've found information about recent climate change policies and agreements.")
        workflow.image_generation_agent.set_response("I've generated an image related to climate change impacts.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("Recent climate change policies include the Paris Agreement and various national initiatives.")
        workflow.verification_agent.set_response("I've verified the facts about recent climate change policies.")
        
        # Execute the workflow
        result = workflow.execute("What are the most recent international climate change policies?")
        
        # Assert that the result contains expected information
        assert "climate" in result.lower()
        assert "policy" in result.lower() or "policies" in result.lower() or "agreement" in result.lower()
        
        # Verify that all agents were called
        workflow.manager_agent.execute.assert_called_once()
        workflow.pre_response_agent.execute.assert_called_once()
        workflow.research_agent.execute.assert_called_once()
        workflow.image_generation_agent.execute.assert_called_once()
        workflow.file_manager_agent.execute.assert_called_once()
        workflow.summary_agent.execute.assert_called_once()
        workflow.verification_agent.execute.assert_called_once()
    
    def test_comparative_research_query(self):
        """Test a comparative research query."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate the research comparing renewable energy sources.")
        workflow.pre_response_agent.set_response("I'll present the research plan for comparing renewable energy sources.")
        workflow.research_agent.set_response("I've found information about solar, wind, hydro, and geothermal energy.")
        workflow.image_generation_agent.set_response("I've generated a comparative chart of renewable energy sources.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("Solar energy is most efficient in sunny regions, while wind energy works best in coastal areas.")
        workflow.verification_agent.set_response("I've verified the facts about different renewable energy sources.")
        
        # Execute the workflow
        result = workflow.execute("Compare the efficiency and cost of different renewable energy sources")
        
        # Assert that the result contains expected information
        assert "renewable" in result.lower()
        assert "energy" in result.lower()
        assert any(source in result.lower() for source in ["solar", "wind", "hydro", "geothermal"])
        
        # Verify that all agents were called
        workflow.manager_agent.execute.assert_called_once()
        workflow.pre_response_agent.execute.assert_called_once()
        workflow.research_agent.execute.assert_called_once()
        workflow.image_generation_agent.execute.assert_called_once()
        workflow.file_manager_agent.execute.assert_called_once()
        workflow.summary_agent.execute.assert_called_once()
        workflow.verification_agent.execute.assert_called_once()
    
    def test_technical_research_query(self):
        """Test a technical research query."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate the research on blockchain technology.")
        workflow.pre_response_agent.set_response("I'll present the research plan for blockchain technology.")
        workflow.research_agent.set_response("I've found information about blockchain architecture and applications.")
        workflow.image_generation_agent.set_response("I've generated an image illustrating blockchain concepts.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("Blockchain is a distributed ledger technology that ensures secure and transparent transactions.")
        workflow.verification_agent.set_response("I've verified the technical facts about blockchain technology.")
        
        # Execute the workflow
        result = workflow.execute("Explain how blockchain technology works and its applications")
        
        # Assert that the result contains expected information
        assert "blockchain" in result.lower()
        assert any(term in result.lower() for term in ["ledger", "distributed", "block", "transaction", "crypto"])
        
        # Verify that all agents were called
        workflow.manager_agent.execute.assert_called_once()
        workflow.pre_response_agent.execute.assert_called_once()
        workflow.research_agent.execute.assert_called_once()
        workflow.image_generation_agent.execute.assert_called_once()
        workflow.file_manager_agent.execute.assert_called_once()
        workflow.summary_agent.execute.assert_called_once()
        workflow.verification_agent.execute.assert_called_once()
    
    def test_cultural_research_query(self):
        """Test a cultural research query."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate the research on Japanese tea ceremony.")
        workflow.pre_response_agent.set_response("I'll present the research plan for Japanese tea ceremony.")
        workflow.research_agent.set_response("I've found information about the history and rituals of Japanese tea ceremony.")
        workflow.image_generation_agent.set_response("I've generated an image of a traditional Japanese tea ceremony.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("The Japanese tea ceremony (chado) is a cultural ritual with deep philosophical roots.")
        workflow.verification_agent.set_response("I've verified the cultural facts about Japanese tea ceremony.")
        
        # Execute the workflow
        result = workflow.execute("Research the history and significance of the Japanese tea ceremony")
        
        # Assert that the result contains expected information
        assert "japanese" in result.lower()
        assert "tea" in result.lower()
        assert "ceremony" in result.lower()
        
        # Verify that all agents were called
        workflow.manager_agent.execute.assert_called_once()
        workflow.pre_response_agent.execute.assert_called_once()
        workflow.research_agent.execute.assert_called_once()
        workflow.image_generation_agent.execute.assert_called_once()
        workflow.file_manager_agent.execute.assert_called_once()
        workflow.summary_agent.execute.assert_called_once()
        workflow.verification_agent.execute.assert_called_once()
    
    def test_business_research_query(self):
        """Test a business research query."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate the research on startup funding strategies.")
        workflow.pre_response_agent.set_response("I'll present the research plan for startup funding strategies.")
        workflow.research_agent.set_response("I've found information about venture capital, angel investors, and crowdfunding.")
        workflow.image_generation_agent.set_response("I've generated an image illustrating different funding sources.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("Startups can seek funding through venture capital, angel investors, crowdfunding, or bootstrapping.")
        workflow.verification_agent.set_response("I've verified the facts about startup funding strategies.")
        
        # Execute the workflow
        result = workflow.execute("What are the most effective funding strategies for tech startups?")
        
        # Assert that the result contains expected information
        assert "startup" in result.lower() or "startups" in result.lower()
        assert "funding" in result.lower()
        assert any(term in result.lower() for term in ["venture", "angel", "investor", "crowdfunding", "bootstrap"])
        
        # Verify that all agents were called
        workflow.manager_agent.execute.assert_called_once()
        workflow.pre_response_agent.execute.assert_called_once()
        workflow.research_agent.execute.assert_called_once()
        workflow.image_generation_agent.execute.assert_called_once()
        workflow.file_manager_agent.execute.assert_called_once()
        workflow.summary_agent.execute.assert_called_once()
        workflow.verification_agent.execute.assert_called_once()
    
    def test_medical_research_query(self):
        """Test a medical research query."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate the research on diabetes management.")
        workflow.pre_response_agent.set_response("I'll present the research plan for diabetes management.")
        workflow.research_agent.set_response("I've found information about Type 1 and Type 2 diabetes and treatment approaches.")
        workflow.image_generation_agent.set_response("I've generated an image illustrating diabetes management strategies.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("Diabetes management involves monitoring blood glucose, medication, diet, and exercise.")
        workflow.verification_agent.set_response("I've verified the medical facts about diabetes management.")
        
        # Execute the workflow
        result = workflow.execute("What are the latest approaches to managing diabetes?")
        
        # Assert that the result contains expected information
        assert "diabetes" in result.lower()
        assert any(term in result.lower() for term in ["glucose", "insulin", "blood sugar", "treatment", "management"])
        
        # Verify that all agents were called
        workflow.manager_agent.execute.assert_called_once()
        workflow.pre_response_agent.execute.assert_called_once()
        workflow.research_agent.execute.assert_called_once()
        workflow.image_generation_agent.execute.assert_called_once()
        workflow.file_manager_agent.execute.assert_called_once()
        workflow.summary_agent.execute.assert_called_once()
        workflow.verification_agent.execute.assert_called_once()
    
    def test_environmental_research_query(self):
        """Test an environmental research query."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate the research on ocean plastic pollution.")
        workflow.pre_response_agent.set_response("I'll present the research plan for ocean plastic pollution.")
        workflow.research_agent.set_response("I've found information about sources, impacts, and solutions for ocean plastic pollution.")
        workflow.image_generation_agent.set_response("I've generated an image showing the impact of plastic pollution on marine life.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("Ocean plastic pollution threatens marine ecosystems and requires global action.")
        workflow.verification_agent.set_response("I've verified the environmental facts about ocean plastic pollution.")
        
        # Execute the workflow
        result = workflow.execute("Research the impact of plastic pollution on ocean ecosystems")
        
        # Assert that the result contains expected information
        assert "plastic" in result.lower()
        assert "pollution" in result.lower()
        assert "ocean" in result.lower() or "marine" in result.lower()
        
        # Verify that all agents were called
        workflow.manager_agent.execute.assert_called_once()
        workflow.pre_response_agent.execute.assert_called_once()
        workflow.research_agent.execute.assert_called_once()
        workflow.image_generation_agent.execute.assert_called_once()
        workflow.file_manager_agent.execute.assert_called_once()
        workflow.summary_agent.execute.assert_called_once()
        workflow.verification_agent.execute.assert_called_once()
    
    def test_multifaceted_research_query(self):
        """Test a multifaceted research query requiring diverse sources."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate the comprehensive research on artificial intelligence ethics.")
        workflow.pre_response_agent.set_response("I'll present the research plan for AI ethics across technical, philosophical, and policy dimensions.")
        workflow.research_agent.set_response("I've found information about AI ethics from technical, philosophical, and policy perspectives.")
        workflow.image_generation_agent.set_response("I've generated an image illustrating the ethical dimensions of AI.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("AI ethics encompasses technical safety, fairness, transparency, privacy, and long-term governance.")
        workflow.verification_agent.set_response("I've verified the facts about AI ethics from multiple sources.")
        
        # Execute the workflow
        result = workflow.execute("Analyze the ethical implications of artificial intelligence from technical, philosophical, and policy perspectives")
        
        # Assert that the result contains expected information
        assert "artificial intelligence" in result.lower() or "ai" in result.lower()
        assert "ethics" in result.lower() or "ethical" in result.lower()
        assert any(term in result.lower() for term in ["technical", "philosophical", "policy", "governance"])
        
        # Verify that all agents were called
        workflow.manager_agent.execute.assert_called_once()
        workflow.pre_response_agent.execute.assert_called_once()
        workflow.research_agent.execute.assert_called_once()
        workflow.image_generation_agent.execute.assert_called_once()
        workflow.file_manager_agent.execute.assert_called_once()
        workflow.summary_agent.execute.assert_called_once()
        workflow.verification_agent.execute.assert_called_once()
    
    def test_local_research_query(self):
        """Test a local research query using local search."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate the research on restaurants in San Francisco.")
        workflow.pre_response_agent.set_response("I'll present the research plan for finding top restaurants in San Francisco.")
        workflow.research_agent.set_response("I've found information about highly-rated restaurants in San Francisco.")
        workflow.image_generation_agent.set_response("I've generated an image of San Francisco's culinary scene.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("San Francisco offers diverse dining options, from Michelin-starred restaurants to local favorites.")
        workflow.verification_agent.set_response("I've verified the facts about San Francisco restaurants.")
        
        # Mock the local search method specifically
        workflow.research_agent.brave_search_mcp.local_search.return_value = """
        Name: Golden Gate Grill
        Address: 123 Market St, San Francisco, CA 94105
        Phone: (415) 555-1234
        Rating: 4.8 (1250 reviews)
        Price Range: $$$
        Hours: Mon-Sun 11:00 AM - 10:00 PM
        
        Name: Fog City Diner
        Address: 456 Embarcadero, San Francisco, CA 94111
        Phone: (415) 555-5678
        Rating: 4.6 (980 reviews)
        Price Range: $$
        Hours: Tue-Sun 8:00 AM - 9:00 PM
        """
        
        # Execute the workflow
        result = workflow.execute("What are the best restaurants in San Francisco?")
        
        # Assert that the result contains expected information
        assert "san francisco" in result.lower()
        assert "restaurant" in result.lower() or "restaurants" in result.lower()
        
        # Verify that the local search was used
        workflow.research_agent.brave_search_mcp.local_search.assert_called()
    
    def test_image_focused_research_query(self):
        """Test a research query with emphasis on image generation."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate the research on architectural styles.")
        workflow.pre_response_agent.set_response("I'll present the research plan for architectural styles with visual examples.")
        workflow.research_agent.set_response("I've found information about Gothic, Baroque, Modern, and Art Deco architectural styles.")
        workflow.image_generation_agent.set_response("I've generated images illustrating different architectural styles.")
        workflow.file_manager_agent.set_response("I've saved all research materials including the architectural style images.")
        workflow.summary_agent.set_response("Architectural styles evolved from Gothic to Renaissance, Baroque, Neoclassical, and Modern.")
        workflow.verification_agent.set_response("I've verified the facts about architectural styles.")
        
        # Mock the image generation method specifically
        workflow.image_generation_agent.everart_mcp.generate_image.return_value = """
        Generated 4 images:
        1. Gothic Architecture: https://example.com/gothic.jpg
        2. Baroque Architecture: https://example.com/baroque.jpg
        3. Modern Architecture: https://example.com/modern.jpg
        4. Art Deco Architecture: https://example.com/artdeco.jpg
        """
        
        # Execute the workflow
        result = workflow.execute("Show me examples of different architectural styles throughout history")
        
        # Assert that the result contains expected information
        assert "architecture" in result.lower() or "architectural" in result.lower()
        assert any(style in result.lower() for style in ["gothic", "baroque", "modern", "art deco"])
        
        # Verify that image generation was emphasized
        workflow.image_generation_agent.everart_mcp.generate_image.assert_called()
        assert workflow.image_generation_agent.everart_mcp.generate_image.call_count >= 1
    
    def test_fact_verification_research_query(self):
        """Test a research query requiring extensive fact verification."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate the research on common misconceptions.")
        workflow.pre_response_agent.set_response("I'll present the research plan for debunking common misconceptions.")
        workflow.research_agent.set_response("I've found information about common misconceptions in science, history, and health.")
        workflow.image_generation_agent.set_response("I've generated images illustrating the truth behind common misconceptions.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("Common misconceptions include: humans only use 10% of their brains (false), Napoleon was short (false), and bats are blind (false).")
        workflow.verification_agent.set_response("I've thoroughly verified the facts about these misconceptions using multiple sources.")
        
        # Mock the verification method specifically
        workflow.verification_agent.verify_facts = MagicMock(return_value={
            "verified_facts": [
                {"claim": "Humans only use 10% of their brains", "status": "False", "evidence": "Neurological research shows activity throughout the brain"},
                {"claim": "Napoleon was unusually short", "status": "False", "evidence": "He was 5'7\", average height for his time"},
                {"claim": "Bats are blind", "status": "False", "evidence": "Bats can see, though many species rely more on echolocation"}
            ]
        })
        
        # Execute the workflow
        result = workflow.execute("Debunk common misconceptions in science and history")
        
        # Assert that the result contains expected information
        assert "misconception" in result.lower()
        assert any(term in result.lower() for term in ["debunk", "false", "myth"])
        
        # Verify that fact verification was emphasized
        workflow.verification_agent.execute.assert_called_once()
    
    def test_time_sensitive_research_query(self):
        """Test a time-sensitive research query requiring recent information."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate the research on recent technological innovations.")
        workflow.pre_response_agent.set_response("I'll present the research plan for recent technological innovations.")
        workflow.research_agent.set_response("I've found information about recent innovations in AI, quantum computing, and renewable energy.")
        workflow.image_generation_agent.set_response("I've generated images illustrating recent technological breakthroughs.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("Recent technological innovations include advances in large language models, quantum supremacy demonstrations, and perovskite solar cells.")
        workflow.verification_agent.set_response("I've verified the facts about recent technological innovations.")
        
        # Mock the web search method to include date information
        workflow.research_agent.brave_search_mcp.web_search.return_value = """
        Title: Breakthrough in Quantum Computing Achieved
        Description: Researchers demonstrate quantum advantage with 100-qubit processor
        URL: https://example.com/quantum-news
        Published: 2023-05-15
        
        Title: New AI Model Sets Benchmark Record
        Description: Latest large language model achieves human-level performance on complex tasks
        URL: https://example.com/ai-news
        Published: 2023-06-02
        """
        
        # Execute the workflow
        result = workflow.execute("What are the most significant technological breakthroughs in the past year?")
        
        # Assert that the result contains expected information
        # Assert that the result contains expected information
        assert any(tech in result.lower() for tech in ["ai", "quantum", "renewable", "innovation", "breakthrough"])
        assert "recent" in result.lower() or "new" in result.lower() or "latest" in result.lower()
        
        # Verify that recent information was prioritized
        workflow.research_agent.brave_search_mcp.web_search.assert_called()
    
    def test_controversial_research_query(self):
        """Test a research query on a controversial topic requiring balanced perspective."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate balanced research on genetic modification in agriculture.")
        workflow.pre_response_agent.set_response("I'll present a research plan covering multiple perspectives on GMOs.")
        workflow.research_agent.set_response("I've found information about GMOs from scientific, economic, environmental, and ethical perspectives.")
        workflow.image_generation_agent.set_response("I've generated images illustrating genetic modification in agriculture.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("Genetic modification in agriculture has potential benefits like increased yield and pest resistance, but also raises concerns about biodiversity, corporate control, and unknown long-term effects.")
        workflow.verification_agent.set_response("I've verified facts from multiple perspectives on GMOs.")
        
        # Execute the workflow
        result = workflow.execute("Research the pros and cons of genetic modification in agriculture")
        
        # Assert that the result contains balanced information
        assert "genetic" in result.lower() and "modification" in result.lower() or "gmo" in result.lower()
        assert any(pro in result.lower() for pro in ["benefit", "advantage", "pro", "yield", "resistance"])
        assert any(con in result.lower() for con in ["concern", "risk", "con", "biodiversity", "unknown"])
        
        # Verify that multiple sources were consulted
        workflow.research_agent.execute.assert_called_once()
        workflow.verification_agent.execute.assert_called_once()
    
    def test_interdisciplinary_research_query(self):
        """Test an interdisciplinary research query spanning multiple domains."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate interdisciplinary research on the intersection of neuroscience and artificial intelligence.")
        workflow.pre_response_agent.set_response("I'll present a research plan covering the neuroscience-AI intersection.")
        workflow.research_agent.set_response("I've found information about how neuroscience inspires AI and how AI helps understand the brain.")
        workflow.image_generation_agent.set_response("I've generated images illustrating the parallels between neural networks and brain structures.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("The intersection of neuroscience and AI has led to neural networks inspired by brain architecture, while AI helps analyze brain data and model neural processes.")
        workflow.verification_agent.set_response("I've verified the interdisciplinary facts about neuroscience and AI.")
        
        # Execute the workflow
        result = workflow.execute("Explore the intersection of neuroscience and artificial intelligence")
        
        # Assert that the result contains interdisciplinary information
        assert "neuroscience" in result.lower() or "brain" in result.lower() or "neural" in result.lower()
        assert "artificial intelligence" in result.lower() or "ai" in result.lower()
        assert "intersection" in result.lower() or "connection" in result.lower() or "relationship" in result.lower()
        
        # Verify that interdisciplinary research was conducted
        workflow.research_agent.execute.assert_called_once()
    
    def test_practical_how_to_research_query(self):
        """Test a practical 'how-to' research query."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate research on how to grow vegetables in a small space.")
        workflow.pre_response_agent.set_response("I'll present a research plan for small-space vegetable gardening.")
        workflow.research_agent.set_response("I've found information about container gardening, vertical gardening, and suitable vegetables for small spaces.")
        workflow.image_generation_agent.set_response("I've generated images illustrating small-space gardening techniques.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("Small-space vegetable gardening can be achieved through containers, vertical systems, and choosing compact varieties. Key techniques include proper soil preparation, adequate lighting, and regular watering.")
        workflow.verification_agent.set_response("I've verified the gardening techniques and recommendations.")
        
        # Execute the workflow
        result = workflow.execute("How to grow vegetables in a small apartment")
        
        # Assert that the result contains practical information
        assert "vegetable" in result.lower() or "garden" in result.lower() or "grow" in result.lower()
        assert "small" in result.lower() or "apartment" in result.lower() or "space" in result.lower()
        assert any(technique in result.lower() for technique in ["container", "vertical", "compact", "soil", "water", "light"])
        
        # Verify that practical information was provided
        workflow.research_agent.execute.assert_called_once()
    
    def test_data_intensive_research_query(self):
        """Test a data-intensive research query requiring statistical information."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate research on global renewable energy adoption trends.")
        workflow.pre_response_agent.set_response("I'll present a research plan for analyzing renewable energy statistics.")
        workflow.research_agent.set_response("I've found statistical data on renewable energy adoption across different countries and sectors.")
        workflow.image_generation_agent.set_response("I've generated charts visualizing renewable energy growth trends.")
        workflow.file_manager_agent.set_response("I've saved all research materials including statistical data.")
        workflow.summary_agent.set_response("Global renewable energy capacity has grown by an average of 8-10% annually over the past decade, with solar growing at 25% and wind at 15%. Leading countries include China (1,200 GW), the US (380 GW), and Germany (150 GW).")
        workflow.verification_agent.set_response("I've verified the statistical data on renewable energy adoption.")
        
        # Execute the workflow
        result = workflow.execute("Analyze global trends in renewable energy adoption with statistics")
        
        # Assert that the result contains statistical information
        assert "renewable" in result.lower() and "energy" in result.lower()
        assert any(country in result.lower() for country in ["china", "us", "germany", "india", "japan"])
        assert any(str(num) in result for num in range(10, 100))  # Contains some numbers
        assert any(unit in result.lower() for unit in ["gw", "mw", "twh", "percent", "%"])
        
        # Verify that statistical analysis was performed
        workflow.research_agent.execute.assert_called_once()
    
    def test_future_prediction_research_query(self):
        """Test a research query about future predictions and forecasts."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate research on future transportation technologies.")
        workflow.pre_response_agent.set_response("I'll present a research plan for analyzing future transportation predictions.")
        workflow.research_agent.set_response("I've found expert forecasts and research on future transportation technologies.")
        workflow.image_generation_agent.set_response("I've generated images illustrating future transportation concepts.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("Future transportation is likely to include autonomous vehicles, hyperloop systems, electric vertical takeoff aircraft, and advanced high-speed rail. Experts predict widespread adoption of autonomous vehicles by 2035 and commercial hyperloop systems by 2040.")
        workflow.verification_agent.set_response("I've verified the credibility of future transportation forecasts from multiple sources.")
        
        # Execute the workflow
        result = workflow.execute("What will transportation look like in 2050?")
        
        # Assert that the result contains future-oriented information
        assert "transportation" in result.lower() or "transport" in result.lower()
        assert "future" in result.lower() or "2050" in result.lower() or "will" in result.lower()
        assert any(tech in result.lower() for tech in ["autonomous", "electric", "hyperloop", "flying", "sustainable"])
        
        # Verify that future forecasts were researched
        workflow.research_agent.execute.assert_called_once()
        workflow.verification_agent.execute.assert_called_once()
    
    def test_research_query_with_file_output(self):
        """Test a research query that generates file outputs."""
        workflow = self.create_mocked_workflow()
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate comprehensive research on renewable energy.")
        workflow.pre_response_agent.set_response("I'll present a research plan for renewable energy with file outputs.")
        workflow.research_agent.set_response("I've found detailed information on solar, wind, hydro, and geothermal energy.")
        workflow.image_generation_agent.set_response("I've generated images for each renewable energy type.")
        workflow.file_manager_agent.set_response("I've created a detailed report with sections on each energy type.")
        workflow.summary_agent.set_response("Renewable energy sources are growing rapidly worldwide, with solar and wind seeing the fastest adoption.")
        workflow.verification_agent.set_response("I've verified all facts in the renewable energy report.")
        
        # Mock the file operations specifically
        workflow.file_manager_agent.filesystem_mcp.write_file.return_value = "File written successfully"
        workflow.file_manager_agent.create_report = MagicMock(return_value="renewable_energy_report.md")
        
        # Execute the workflow
        result = workflow.execute("Create a comprehensive report on renewable energy sources")
        
        # Assert that the result mentions file output
        assert "report" in result.lower() or "file" in result.lower() or "document" in result.lower()
        assert "renewable" in result.lower() and "energy" in result.lower()
        
        # Verify that file operations were performed
        workflow.file_manager_agent.execute.assert_called_once()
        assert workflow.file_manager_agent.filesystem_mcp.write_file.call_count >= 1
    
    def test_research_query_with_real_mcps(self):
        """Test a research query with real (but mocked) MCP interactions."""
        # Create real MCPs with mocked subprocess
        with patch('subprocess.Popen') as mock_popen:
            # Mock the process
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_process.stdin = MagicMock()
            mock_process.stdout = MagicMock()
            mock_process.stdout.readline.return_value = json.dumps({
                "jsonrpc": "2.0",
                "id": "test-id",
                "result": {
                    "content": [{"type": "text", "text": "Mock MCP response"}],
                    "isError": False
                }
            })
            mock_popen.return_value = mock_process
            
            # Create real MCPs
            brave_search = BraveSearchMCP(api_key="test_brave_api_key")
            everart = EverArtMCP(api_key="test_everart_api_key")
            fetch = FetchMCP()
            filesystem = FilesystemMCP(workspace_dir=self.workspace_dir)
            memory = MemoryMCP()
            
            # Create agents with real MCPs
            manager_agent = ManagerAgent()
            manager_agent.memory_mcp = memory
            
            pre_response_agent = PreResponseAgent()
            pre_response_agent.memory_mcp = memory
            
            research_agent = ResearchAgent()
            research_agent.brave_search_mcp = brave_search
            research_agent.fetch_mcp = fetch
            research_agent.memory_mcp = memory
            
            image_generation_agent = ImageGenerationAgent()
            image_generation_agent.everart_mcp = everart
            image_generation_agent.memory_mcp = memory
            
            file_manager_agent = FileManagerAgent()
            file_manager_agent.filesystem_mcp = filesystem
            file_manager_agent.memory_mcp = memory
            
            summary_agent = SummaryAgent()
            summary_agent.memory_mcp = memory
            
            verification_agent = VerificationAgent()
            verification_agent.brave_search_mcp = brave_search
            verification_agent.memory_mcp = memory
            
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
            
            # Mock the execute methods of all agents
            for agent in [
                workflow.manager_agent,
                workflow.pre_response_agent,
                workflow.research_agent,
                workflow.image_generation_agent,
                workflow.file_manager_agent,
                workflow.summary_agent,
                workflow.verification_agent
            ]:
                agent.execute = MagicMock(return_value=f"Mock response from {agent.__class__.__name__}")
            
            # Execute the workflow
            result = workflow.execute("Research quantum computing applications")
            
            # Assert that the workflow executed successfully
            assert result is not None
            assert len(result) > 0
            
            # Verify that all agents were called
            workflow.manager_agent.execute.assert_called_once()
            workflow.pre_response_agent.execute.assert_called_once()
            workflow.research_agent.execute.assert_called_once()
            workflow.image_generation_agent.execute.assert_called_once()
            workflow.file_manager_agent.execute.assert_called_once()
            workflow.summary_agent.execute.assert_called_once()
            workflow.verification_agent.execute.assert_called_once()
            
            # Clean up
            brave_search.close()
            everart.close()
            fetch.close()
            filesystem.close()
            memory.close()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
