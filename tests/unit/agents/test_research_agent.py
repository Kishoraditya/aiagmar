"""
Unit tests for the Research Agent.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from apps.agents.research_agent import ResearchAgent
from apps.mcps.brave_search_mcp import BraveSearchMCP
from apps.mcps.fetch_mcp import FetchMCP
from apps.mcps.memory_mcp import MemoryMCP
from apps.utils.exceptions import AgentError

class TestResearchAgent:
    """Test suite for ResearchAgent class."""

    @pytest.fixture
    def brave_search_mcp(self):
        """Fixture to create a mock BraveSearchMCP."""
        mock_search = MagicMock(spec=BraveSearchMCP)
        mock_search.web_search = MagicMock()
        mock_search.local_search = MagicMock()
        return mock_search

    @pytest.fixture
    def fetch_mcp(self):
        """Fixture to create a mock FetchMCP."""
        mock_fetch = MagicMock(spec=FetchMCP)
        mock_fetch.fetch_url = MagicMock()
        mock_fetch.fetch_text = MagicMock()
        mock_fetch.fetch_html = MagicMock()
        return mock_fetch

    @pytest.fixture
    def memory_mcp(self):
        """Fixture to create a mock MemoryMCP."""
        mock_memory = MagicMock(spec=MemoryMCP)
        mock_memory.store_memory = MagicMock()
        mock_memory.retrieve_memory = MagicMock()
        mock_memory.search_memories = MagicMock()
        return mock_memory

    @pytest.fixture
    def research_agent(self, brave_search_mcp, fetch_mcp, memory_mcp):
        """Fixture to create a ResearchAgent instance with mock dependencies."""
        agent = ResearchAgent(
            name="research",
            brave_search_mcp=brave_search_mcp,
            fetch_mcp=fetch_mcp,
            memory_mcp=memory_mcp
        )
        return agent

    def test_init(self, brave_search_mcp, fetch_mcp, memory_mcp):
        """Test initialization of ResearchAgent."""
        agent = ResearchAgent(
            name="research",
            brave_search_mcp=brave_search_mcp,
            fetch_mcp=fetch_mcp,
            memory_mcp=memory_mcp
        )
        
        assert agent.name == "research"
        assert agent.brave_search_mcp == brave_search_mcp
        assert agent.fetch_mcp == fetch_mcp
        assert agent.memory_mcp == memory_mcp

    @pytest.mark.asyncio
    async def test_search_web_basic(self, research_agent):
        """Test basic web search functionality."""
        # Setup mock response
        search_results = """
        Title: Climate Change Impact on Coastal Regions
        Description: A comprehensive study of climate change effects on coastal areas.
        URL: https://example.com/climate-coastal

        Title: Rising Sea Levels and Coastal Cities
        Description: How coastal cities are adapting to rising sea levels.
        URL: https://example.com/sea-levels
        """
        research_agent.brave_search_mcp.web_search.return_value = search_results
        
        # Call the method
        query = "climate change coastal impacts"
        results = await research_agent.search_web(query)
        
        # Verify brave_search_mcp was called correctly
        research_agent.brave_search_mcp.web_search.assert_called_once_with(
            query, count=10, offset=0
        )
        
        # Verify memory was updated
        research_agent.memory_mcp.store_memory.assert_called_with(
            "search_query", query, namespace="research"
        )
        research_agent.memory_mcp.store_memory.assert_any_call(
            "search_results", str(results), namespace="research"
        )
        
        # Verify result structure
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["title"] == "Climate Change Impact on Coastal Regions"
        assert results[0]["url"] == "https://example.com/climate-coastal"
        assert results[1]["title"] == "Rising Sea Levels and Coastal Cities"

    @pytest.mark.asyncio
    async def test_search_web_with_context(self, research_agent):
        """Test web search with additional context."""
        # Setup mock response
        search_results = "Title: Climate Change Impact on Agriculture\nURL: https://example.com/climate-agriculture"
        research_agent.brave_search_mcp.web_search.return_value = search_results
        
        # Call the method with context
        query = "climate change impacts"
        context = "Focus on agricultural effects"
        results = await research_agent.search_web(query, context=context)
        
        # Verify brave_search_mcp was called with enhanced query
        research_agent.brave_search_mcp.web_search.assert_called_once()
        call_args = research_agent.brave_search_mcp.web_search.call_args[0][0]
        assert "climate change impacts" in call_args
        assert "agricultural" in call_args
        
        # Verify memory was updated with context
        research_agent.memory_mcp.store_memory.assert_any_call(
            "search_context", context, namespace="research"
        )

    @pytest.mark.asyncio
    async def test_search_web_with_count_offset(self, research_agent):
        """Test web search with custom count and offset parameters."""
        # Setup mock response
        search_results = "Title: Result 1\nURL: https://example.com/1"
        research_agent.brave_search_mcp.web_search.return_value = search_results
        
        # Call the method with custom parameters
        query = "test query"
        results = await research_agent.search_web(query, count=5, offset=2)
        
        # Verify brave_search_mcp was called with correct parameters
        research_agent.brave_search_mcp.web_search.assert_called_once_with(
            query, count=5, offset=2
        )

    @pytest.mark.asyncio
    async def test_search_web_error(self, research_agent):
        """Test handling errors during web search."""
        # Setup mock to raise exception
        research_agent.brave_search_mcp.web_search.side_effect = Exception("Search API error")
        
        # Call the method and expect error
        with pytest.raises(AgentError, match="Failed to perform web search"):
            await research_agent.search_web("test query")
        
        # Verify memory was not updated with results
        for call_args in research_agent.memory_mcp.store_memory.call_args_list:
            args, kwargs = call_args
            assert args[0] != "search_results"

    @pytest.mark.asyncio
    async def test_search_local(self, research_agent):
        """Test local search functionality."""
        # Setup mock response
        local_results = """
        Name: Climate Research Center
        Address: 123 Science Ave, Research City, CA 94043
        Phone: (555) 123-4567
        Rating: 4.8 (42 reviews)
        
        Name: Environmental Studies Institute
        Address: 456 Green St, Eco Town, CA 94044
        Phone: (555) 987-6543
        Rating: 4.5 (38 reviews)
        """
        research_agent.brave_search_mcp.local_search.return_value = local_results
        
        # Call the method
        query = "climate research centers in California"
        results = await research_agent.search_local(query)
        
        # Verify brave_search_mcp was called correctly
        research_agent.brave_search_mcp.local_search.assert_called_once_with(
            query, count=5
        )
        
        # Verify memory was updated
        research_agent.memory_mcp.store_memory.assert_called_with(
            "local_search_query", query, namespace="research"
        )
        research_agent.memory_mcp.store_memory.assert_any_call(
            "local_search_results", str(results), namespace="research"
        )
        
        # Verify result structure
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["name"] == "Climate Research Center"
        assert results[0]["address"] == "123 Science Ave, Research City, CA 94043"
        assert results[0]["phone"] == "(555) 123-4567"
        assert results[1]["name"] == "Environmental Studies Institute"

    @pytest.mark.asyncio
    async def test_search_local_with_count(self, research_agent):
        """Test local search with custom count parameter."""
        # Setup mock response
        local_results = "Name: Result 1\nAddress: Address 1"
        research_agent.brave_search_mcp.local_search.return_value = local_results
        
        # Call the method with custom count
        query = "test local query"
        results = await research_agent.search_local(query, count=3)
        
        # Verify brave_search_mcp was called with correct parameters
        research_agent.brave_search_mcp.local_search.assert_called_once_with(
            query, count=3
        )

    @pytest.mark.asyncio
    async def test_search_local_error(self, research_agent):
        """Test handling errors during local search."""
        # Setup mock to raise exception
        research_agent.brave_search_mcp.local_search.side_effect = Exception("Local search API error")
        
        # Call the method and expect error
        with pytest.raises(AgentError, match="Failed to perform local search"):
            await research_agent.search_local("test local query")
        
        # Verify memory was not updated with results
        for call_args in research_agent.memory_mcp.store_memory.call_args_list:
            args, kwargs = call_args
            assert args[0] != "local_search_results"

    @pytest.mark.asyncio
    async def test_fetch_content_url(self, research_agent):
        """Test fetching content from a URL."""
        # Setup mock response
        content = "This is the content of the webpage about climate change impacts."
        research_agent.fetch_mcp.fetch_url.return_value = content
        
        # Call the method
        url = "https://example.com/climate-article"
        result = await research_agent.fetch_content(url)
        
        # Verify fetch_mcp was called correctly
        research_agent.fetch_mcp.fetch_url.assert_called_once_with(url)
        
        # Verify memory was updated
        research_agent.memory_mcp.store_memory.assert_called_with(
            "fetched_url", url, namespace="research"
        )
        research_agent.memory_mcp.store_memory.assert_any_call(
            "fetched_content", content, namespace="research"
        )
        
        # Verify result
        assert result == content

    @pytest.mark.asyncio
    async def test_fetch_content_with_selector(self, research_agent):
        """Test fetching content with a specific CSS selector."""
        # Setup mock response
        content = "This is the main article content."
        research_agent.fetch_mcp.fetch_url.return_value = content
        
        # Call the method with selector
        url = "https://example.com/article"
        selector = "article.main-content"
        result = await research_agent.fetch_content(url, selector=selector)
        
        # Verify fetch_mcp was called with selector
        research_agent.fetch_mcp.fetch_url.assert_called_once_with(
            url, selector=selector
        )
        
        # Verify result
        assert result == content

    @pytest.mark.asyncio
    async def test_fetch_content_text_only(self, research_agent):
        """Test fetching text-only content from a URL."""
        # Setup mock response
        text_content = "Plain text content without HTML tags."
        research_agent.fetch_mcp.fetch_text.return_value = text_content
        
        # Call the method with text_only=True
        url = "https://example.com/article"
        result = await research_agent.fetch_content(url, text_only=True)
        
        # Verify fetch_mcp.fetch_text was called
        research_agent.fetch_mcp.fetch_text.assert_called_once_with(url)
        
        # Verify result
        assert result == text_content

    @pytest.mark.asyncio
    async def test_fetch_content_html(self, research_agent):
        """Test fetching raw HTML content from a URL."""
        # Setup mock response
        html_content = "<html><body><h1>Article Title</h1><p>Content</p></body></html>"
        research_agent.fetch_mcp.fetch_html.return_value = html_content
        
        # Call the method with raw_html=True
        url = "https://example.com/article"
        result = await research_agent.fetch_content(url, raw_html=True)
        
        # Verify fetch_mcp.fetch_html was called
        research_agent.fetch_mcp.fetch_html.assert_called_once_with(url)
        
        # Verify result
        assert result == html_content

    @pytest.mark.asyncio
    async def test_fetch_content_error(self, research_agent):
        """Test handling errors during content fetching."""
        # Setup mock to raise exception
        research_agent.fetch_mcp.fetch_url.side_effect = Exception("Fetch error")
        
        # Call the method and expect error
        with pytest.raises(AgentError, match="Failed to fetch content"):
            await research_agent.fetch_content("https://example.com/article")
        
        # Verify memory was not updated with content
        for call_args in research_agent.memory_mcp.store_memory.call_args_list:
            args, kwargs = call_args
            assert args[0] != "fetched_content"

    @pytest.mark.asyncio
    async def test_fetch_multiple_urls(self, research_agent):
        """Test fetching content from multiple URLs."""
        # Setup mock responses
        research_agent.fetch_mcp.fetch_url.side_effect = [
            "Content from first URL",
            "Content from second URL",
            "Content from third URL"
        ]
        
        # Call the method
        urls = [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/article3"
        ]
        results = await research_agent.fetch_multiple_urls(urls)
        
        # Verify fetch_mcp was called for each URL
        assert research_agent.fetch_mcp.fetch_url.call_count == 3
        
        # Verify memory was updated
        research_agent.memory_mcp.store_memory.assert_any_call(
            "fetched_urls", str(urls), namespace="research"
        )
        
        # Verify result structure
        assert isinstance(results, dict)
        assert len(results) == 3
        assert results[urls[0]] == "Content from first URL"
        assert results[urls[1]] == "Content from second URL"
        assert results[urls[2]] == "Content from third URL"

    @pytest.mark.asyncio
    async def test_fetch_multiple_urls_with_failures(self, research_agent):
        """Test fetching multiple URLs with some failures."""
        # Setup mock responses with one failure
        def mock_fetch(url):
            if "article2" in url:
                raise Exception("Failed to fetch article2")
            return f"Content from {url}"
        
        research_agent.fetch_mcp.fetch_url.side_effect = mock_fetch
        
        # Call the method
        urls = [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/article3"
        ]
        results = await research_agent.fetch_multiple_urls(urls)
        
        # Verify fetch_mcp was called for each URL
        assert research_agent.fetch_mcp.fetch_url.call_count == 3
        
        # Verify result structure
        assert isinstance(results, dict)
        assert len(results) == 2  # Only successful fetches
        assert urls[0] in results
        assert urls[1] not in results  # Failed URL
        assert urls[2] in results
        assert "Error fetching" not in results[urls[0]]
        assert "Error fetching" not in results[urls[2]]

    @pytest.mark.asyncio
    async def test_fetch_multiple_urls_all_fail(self, research_agent):
        """Test behavior when all URL fetches fail."""
        # Setup mock to always raise exception
        research_agent.fetch_mcp.fetch_url.side_effect = Exception("Fetch error")
        
        # Call the method
        urls = [
            "https://example.com/article1",
            "https://example.com/article2"
        ]
        
        # Should raise error when all URLs fail
        with pytest.raises(AgentError, match="Failed to fetch any content"):
            await research_agent.fetch_multiple_urls(urls)

    @pytest.mark.asyncio
    async def test_analyze_search_results(self, research_agent):
        """Test analyzing search results to find most relevant ones."""
        # Setup search results
        search_results = [
            {
                "title": "Climate Change Impact on Coastal Regions",
                "description": "A comprehensive study of climate change effects on coastal areas.",
                "url": "https://example.com/climate-coastal"
            },
            {
                "title": "Rising Sea Levels and Coastal Cities",
                "description": "How coastal cities are adapting to rising sea levels.",
                "url": "https://example.com/sea-levels"
            },
            {
                "title": "Unrelated Article About Technology",
                "description": "Latest technology trends and innovations.",
                "url": "https://example.com/tech-trends"
            }
        ]
        
        # Call the method
        query = "climate change coastal impacts"
        analysis = await research_agent.analyze_search_results(search_results, query)
        
        # Verify memory was updated
        research_agent.memory_mcp.store_memory.assert_called_with(
            "search_analysis", str(analysis), namespace="research"
        )
        
        # Verify result structure
        assert "relevant_results" in analysis
        assert "relevance_scores" in analysis
        assert "top_keywords" in analysis
        assert "analysis_summary" in analysis
        
        # Verify relevant results are correctly identified
        assert len(analysis["relevant_results"]) == 2
        assert analysis["relevant_results"][0]["title"] == "Climate Change Impact on Coastal Regions"
        assert analysis["relevant_results"][1]["title"] == "Rising Sea Levels and Coastal Cities"
        assert all("Unrelated Article" not in result["title"] for result in analysis["relevant_results"])
        
        # Verify keywords are extracted
        assert len(analysis["top_keywords"]) > 0
        assert any("climate" in keyword.lower() for keyword in analysis["top_keywords"])
        assert any("coastal" in keyword.lower() for keyword in analysis["top_keywords"])

    @pytest.mark.asyncio
    async def test_analyze_search_results_empty(self, research_agent):
        """Test analyzing empty search results."""
        # Call the method with empty results
        query = "climate change"
        analysis = await research_agent.analyze_search_results([], query)
        
        # Verify result structure
        assert "relevant_results" in analysis
        assert "relevance_scores" in analysis
        assert "top_keywords" in analysis
        assert "analysis_summary" in analysis
        
        # Verify empty results are handled
        assert len(analysis["relevant_results"]) == 0
        assert len(analysis["relevance_scores"]) == 0
        assert "No search results found" in analysis["analysis_summary"]

    @pytest.mark.asyncio
    async def test_analyze_search_results_error(self, research_agent):
        """Test handling errors during search results analysis."""
        # Setup mock to simulate internal error during analysis
        with patch.object(research_agent, '_calculate_relevance_scores', side_effect=Exception("Analysis error")):
            # Call the method and expect error
            with pytest.raises(AgentError, match="Failed to analyze search results"):
                await research_agent.analyze_search_results([{"title": "Test"}], "query")

    @pytest.mark.asyncio
    async def test_extract_key_information(self, research_agent):
        """Test extracting key information from content."""
        # Setup content
        content = """
        Climate change is causing sea levels to rise at an accelerating rate.
        According to recent studies, global sea levels rose by about 8-9 inches (21-24 cm) since 1880.
        The rate of sea level rise has doubled from 1.4 mm per year throughout most of the 20th century to 3.6 mm per year from 2006-2015.
        Coastal cities like Miami, New York, and New Orleans are particularly vulnerable.
        Adaptation strategies include building sea walls, elevating structures, and managed retreat from high-risk areas.
        """
        
        # Call the method
        topic = "sea level rise impacts"
        key_info = await research_agent.extract_key_information(content, topic)
        
        # Verify memory was updated
        research_agent.memory_mcp.store_memory.assert_called_with(
            "extracted_information", str(key_info), namespace="research"
        )
        
        # Verify result structure
        assert "facts" in key_info
        assert "statistics" in key_info
        assert "entities" in key_info
        assert "summary" in key_info
        
        # Verify extracted information
        assert len(key_info["facts"]) >= 2
        assert any("sea levels" in fact.lower() for fact in key_info["facts"])
        assert any("coastal cities" in fact.lower() for fact in key_info["facts"])
        
        assert len(key_info["statistics"]) >= 1
        assert any("8-9 inches" in stat or "21-24 cm" in stat for stat in key_info["statistics"])
        
        assert len(key_info["entities"]) >= 3
        assert "Miami" in key_info["entities"] or "New York" in key_info["entities"] or "New Orleans" in key_info["entities"]
        
        assert len(key_info["summary"]) > 50

    @pytest.mark.asyncio
    async def test_extract_key_information_minimal_content(self, research_agent):
        """Test extracting key information from minimal content."""
        # Setup minimal content
        content = "Climate change is a global issue."
        
        # Call the method
        key_info = await research_agent.extract_key_information(content, "climate change")
        
        # Verify result structure is maintained even with minimal content
        assert "facts" in key_info
        assert "statistics" in key_info
        assert "entities" in key_info
        assert "summary" in key_info
        
        # Verify minimal extraction
        assert len(key_info["facts"]) >= 1
        assert key_info["facts"][0] == "Climate change is a global issue."
        assert len(key_info["statistics"]) == 0
        assert len(key_info["entities"]) >= 1
        assert "Climate change" in key_info["entities"]
        assert key_info["summary"] == "Climate change is a global issue."

    @pytest.mark.asyncio
    async def test_extract_key_information_error(self, research_agent):
        """Test handling errors during key information extraction."""
        # Call the method with empty content and expect error
        with pytest.raises(AgentError, match="Failed to extract key information"):
            await research_agent.extract_key_information("", "topic")

    @pytest.mark.asyncio
    async def test_research_topic(self, research_agent):
        """Test the complete research topic workflow."""
        # Setup mock responses
        # 1. Search results
        search_results = [
            {
                "title": "Climate Change Impact on Coastal Regions",
                "description": "A comprehensive study of climate change effects on coastal areas.",
                "url": "https://example.com/climate-coastal"
            },
            {
                "title": "Rising Sea Levels and Coastal Cities",
                "description": "How coastal cities are adapting to rising sea levels.",
                "url": "https://example.com/sea-levels"
            }
        ]
        research_agent.search_web = AsyncMock(return_value=search_results)
        
        # 2. Content fetching
        content = "Climate change is causing sea levels to rise, affecting coastal regions worldwide."
        research_agent.fetch_content = AsyncMock(return_value=content)
        
        # 3. Analysis
        analysis_result = {
            "relevant_results": search_results,
            "relevance_scores": {"https://example.com/climate-coastal": 0.9, "https://example.com/sea-levels": 0.8},
            "top_keywords": ["climate change", "coastal", "sea levels"],
            "analysis_summary": "Found relevant articles about climate change impacts on coastal regions."
        }
        research_agent.analyze_search_results = AsyncMock(return_value=analysis_result)
        
        # 4. Key information extraction
        key_info = {
            "facts": ["Sea levels are rising due to climate change", "Coastal cities are at risk"],
            "statistics": ["Sea levels rose 8-9 inches since 1880"],
            "entities": ["Miami", "New York", "sea walls"],
            "summary": "Climate change is causing sea level rise that threatens coastal regions."
        }
        research_agent.extract_key_information = AsyncMock(return_value=key_info)
        
        # Call the method
        query = "climate change coastal impacts"
        context = "Focus on sea level rise"
        
        result = await research_agent.research_topic(query, context)
        
        # Verify all steps were called
        research_agent.search_web.assert_called_once_with(query, context=context)
        research_agent.fetch_content.assert_called()
        research_agent.analyze_search_results.assert_called_once_with(search_results, query)
        research_agent.extract_key_information.assert_called()
        
        # Verify memory was updated with final results
        research_agent.memory_mcp.store_memory.assert_called_with(
            "research_results", str(result), namespace="research"
        )
        
        # Verify result structure
        assert "query" in result
        assert "context" in result
        assert "search_results" in result
        assert "content" in result
        assert "analysis" in result
        assert "key_information" in result
        
        assert result["query"] == query
        assert result["context"] == context
        assert result["search_results"] == search_results
        assert result["content"] == content
        assert result["analysis"] == analysis_result
        assert result["key_information"] == key_info

    @pytest.mark.asyncio
    async def test_research_topic_search_error(self, research_agent):
        """Test research topic workflow when search fails."""
        # Setup search to fail
        research_agent.search_web = AsyncMock(side_effect=AgentError("Search failed"))
        
        # Call the method and expect error
        with pytest.raises(AgentError, match="Search failed"):
            await research_agent.research_topic("query", "context")
        
        # Verify no further steps were called
        research_agent.fetch_content.assert_not_called()
        research_agent.analyze_search_results.assert_not_called()
        research_agent.extract_key_information.assert_not_called()

    @pytest.mark.asyncio
    async def test_research_topic_fetch_error(self, research_agent):
        """Test research topic workflow when content fetching fails but continues with analysis."""
        # Setup mocks
        search_results = [{"title": "Article", "url": "https://example.com/article"}]
        research_agent.search_web = AsyncMock(return_value=search_results)
        
        # Setup fetch to fail
        research_agent.fetch_content = AsyncMock(side_effect=AgentError("Fetch failed"))
        
        # Setup analysis
        analysis_result = {"relevant_results": search_results, "analysis_summary": "Analysis"}
        research_agent.analyze_search_results = AsyncMock(return_value=analysis_result)
        
        # Call the method - should continue despite fetch error
        result = await research_agent.research_topic("query", "context")
        
        # Verify steps were called
        research_agent.search_web.assert_called_once()
        research_agent.fetch_content.assert_called()
        research_agent.analyze_search_results.assert_called_once()
        
        # Verify result structure
        assert "query" in result
        assert "search_results" in result
        assert "analysis" in result
        assert "content" in result
        assert result["content"] == ""  # Empty content due to fetch error
        assert "error" in result
        assert "Fetch failed" in result["error"]

    @pytest.mark.asyncio
    async def test_find_related_topics(self, research_agent):
        """Test finding related topics based on research results."""
        # Setup research results
        research_results = {
            "query": "climate change coastal impacts",
            "search_results": [
                {"title": "Sea Level Rise and Coastal Flooding", "url": "https://example.com/1"},
                {"title": "Climate Change Effects on Marine Ecosystems", "url": "https://example.com/2"}
            ],
            "key_information": {
                "facts": ["Sea levels are rising", "Coastal erosion is increasing"],
                "entities": ["sea walls", "mangroves", "coral reefs"]
            }
        }
        
        # Call the method
        related_topics = await research_agent.find_related_topics(research_results)
        
        # Verify result structure
        assert isinstance(related_topics, list)
        assert len(related_topics) >= 3
        
        # Verify related topics are relevant
        relevant_terms = ["coastal", "sea level", "erosion", "marine", "climate", "ecosystem", "coral"]
        for topic in related_topics:
            assert any(term in topic.lower() for term in relevant_terms)

    @pytest.mark.asyncio
    async def test_evaluate_source_credibility(self, research_agent):
        """Test evaluating the credibility of a source."""
        # Call the method
        # Call the method
        url = "https://climate.nasa.gov/evidence/"
        title = "Climate Change Evidence - NASA"
        description = "Scientific evidence for warming of the climate system is unequivocal."
        
        credibility = await research_agent.evaluate_source_credibility(url, title, description)
        
        # Verify result structure
        assert "score" in credibility
        assert "factors" in credibility
        assert "explanation" in credibility
        
        # Verify NASA is evaluated as highly credible
        assert credibility["score"] >= 8  # On a 10-point scale
        assert len(credibility["factors"]) >= 3
        assert any("government" in factor.lower() or "official" in factor.lower() for factor in credibility["factors"])
        assert len(credibility["explanation"]) > 50

    @pytest.mark.asyncio
    async def test_evaluate_source_credibility_low(self, research_agent):
        """Test evaluating a less credible source."""
        # Call the method with a less credible source
        url = "https://climate-skeptic-blog.example.com/opinion"
        title = "Why Climate Change is a Hoax"
        description = "Personal blog revealing the truth about the climate change conspiracy."
        
        credibility = await research_agent.evaluate_source_credibility(url, title, description)
        
        # Verify result indicates lower credibility
        assert credibility["score"] < 5  # On a 10-point scale
        assert any("blog" in factor.lower() or "opinion" in factor.lower() for factor in credibility["factors"])
        assert "conspiracy" in credibility["explanation"].lower() or "bias" in credibility["explanation"].lower()

    @pytest.mark.asyncio
    async def test_compare_multiple_sources(self, research_agent):
        """Test comparing information from multiple sources."""
        # Setup sources with content
        sources = {
            "https://example.com/source1": "Climate change is causing sea levels to rise at a rate of 3.3 mm per year.",
            "https://example.com/source2": "Sea levels are rising at approximately 3.4 mm per year due to global warming.",
            "https://example.com/source3": "Some scientists dispute climate change, but sea level rise is measured at 1 mm per year."
        }
        
        # Call the method
        topic = "sea level rise rate"
        comparison = await research_agent.compare_multiple_sources(sources, topic)
        
        # Verify result structure
        assert "agreements" in comparison
        assert "disagreements" in comparison
        assert "consensus" in comparison
        assert "outliers" in comparison
        
        # Verify content analysis
        assert len(comparison["agreements"]) >= 1
        assert "sea level" in str(comparison["agreements"]).lower()
        
        assert len(comparison["disagreements"]) >= 1
        assert "rate" in str(comparison["disagreements"]).lower() or "mm per year" in str(comparison["disagreements"]).lower()
        
        assert "3.3" in comparison["consensus"] or "3.4" in comparison["consensus"]
        assert "source3" in str(comparison["outliers"]).lower() or "1 mm" in str(comparison["outliers"]).lower()

    @pytest.mark.asyncio
    async def test_compare_multiple_sources_identical(self, research_agent):
        """Test comparing sources with identical information."""
        # Setup sources with identical content
        sources = {
            "https://example.com/source1": "Climate change is real and caused by human activities.",
            "https://example.com/source2": "Climate change is real and caused by human activities.",
            "https://example.com/source3": "Climate change is real and caused by human activities."
        }
        
        # Call the method
        comparison = await research_agent.compare_multiple_sources(sources, "climate change causes")
        
        # Verify result shows complete agreement
        assert len(comparison["agreements"]) >= 1
        assert "climate change is real" in str(comparison["agreements"]).lower()
        assert len(comparison["disagreements"]) == 0
        assert "complete agreement" in comparison["consensus"].lower()
        assert len(comparison["outliers"]) == 0

    @pytest.mark.asyncio
    async def test_find_primary_sources(self, research_agent):
        """Test finding primary sources from search results."""
        # Setup search results
        search_results = [
            {
                "title": "NASA Study on Climate Change",
                "description": "Original research by NASA scientists on climate patterns.",
                "url": "https://nasa.gov/research/climate"
            },
            {
                "title": "News Article About Climate",
                "description": "News coverage of recent climate events.",
                "url": "https://news.example.com/climate"
            },
            {
                "title": "IPCC Assessment Report",
                "description": "Official IPCC scientific assessment of climate change.",
                "url": "https://ipcc.ch/report"
            },
            {
                "title": "Blog Post on Climate Politics",
                "description": "A blogger's opinion on climate policy.",
                "url": "https://climate-blog.example.com"
            }
        ]
        
        # Call the method
        primary_sources = await research_agent.find_primary_sources(search_results)
        
        # Verify result structure
        assert "primary_sources" in primary_sources
        assert "secondary_sources" in primary_sources
        assert "explanation" in primary_sources
        
        # Verify classification
        assert len(primary_sources["primary_sources"]) == 2
        assert any("NASA" in source["title"] for source in primary_sources["primary_sources"])
        assert any("IPCC" in source["title"] for source in primary_sources["primary_sources"])
        
        assert len(primary_sources["secondary_sources"]) == 2
        assert any("News" in source["title"] for source in primary_sources["secondary_sources"])
        assert any("Blog" in source["title"] for source in primary_sources["secondary_sources"])
        
        assert len(primary_sources["explanation"]) > 50

    @pytest.mark.asyncio
    async def test_extract_citations(self, research_agent):
        """Test extracting citations from content."""
        # Setup content with citations
        content = """
        According to Smith et al. (2020), climate change is accelerating.
        The IPCC (2021) report states that "human influence has warmed the climate at a rate that is unprecedented."
        A recent study (Johnson, 2019) found that sea levels could rise by 2 meters by 2100.
        For more information, see Brown and Davis (2018).
        """
        
        # Call the method
        citations = await research_agent.extract_citations(content)
        
        # Verify result structure
        assert "citations" in citations
        assert "count" in citations
        assert "formatted_citations" in citations
        
        # Verify extracted citations
        assert citations["count"] == 4
        assert "Smith et al. (2020)" in citations["citations"]
        assert "IPCC (2021)" in citations["citations"]
        assert "Johnson, 2019" in citations["citations"]
        assert "Brown and Davis (2018)" in citations["citations"]
        
        # Verify formatted citations
        assert len(citations["formatted_citations"]) == 4
        for citation in citations["formatted_citations"]:
            assert "title" in citation or "author" in citation
            assert "year" in citation

    @pytest.mark.asyncio
    async def test_extract_citations_no_citations(self, research_agent):
        """Test extracting citations from content with no citations."""
        # Setup content without citations
        content = "Climate change is a global issue that affects everyone."
        
        # Call the method
        citations = await research_agent.extract_citations(content)
        
        # Verify result
        assert citations["count"] == 0
        assert len(citations["citations"]) == 0
        assert len(citations["formatted_citations"]) == 0

    @pytest.mark.asyncio
    async def test_generate_research_questions(self, research_agent):
        """Test generating research questions based on a topic."""
        # Call the method
        topic = "climate change impacts on agriculture"
        
        questions = await research_agent.generate_research_questions(topic)
        
        # Verify result structure
        assert isinstance(questions, list)
        assert len(questions) >= 5
        
        # Verify questions are relevant and properly formatted
        for question in questions:
            assert question.endswith("?")
            assert any(term in question.lower() for term in ["climate", "agriculture", "impact", "crop", "farm"])

    @pytest.mark.asyncio
    async def test_identify_research_gaps(self, research_agent):
        """Test identifying gaps in research content."""
        # Setup research content
        content = """
        Climate change is causing sea levels to rise.
        Coastal cities are at risk of flooding.
        Some adaptation strategies include building sea walls.
        """
        
        # Call the method
        topic = "climate change coastal impacts"
        gaps = await research_agent.identify_research_gaps(content, topic)
        
        # Verify result structure
        assert "missing_aspects" in gaps
        assert "suggested_searches" in gaps
        assert "explanation" in gaps
        
        # Verify identified gaps
        assert len(gaps["missing_aspects"]) >= 3
        assert any("economic" in aspect.lower() for aspect in gaps["missing_aspects"]) or \
               any("cost" in aspect.lower() for aspect in gaps["missing_aspects"])
        
        assert len(gaps["suggested_searches"]) >= 2
        assert all(isinstance(search, str) for search in gaps["suggested_searches"])
        
        assert len(gaps["explanation"]) > 50

    @pytest.mark.asyncio
    async def test_create_research_summary(self, research_agent):
        """Test creating a comprehensive research summary."""
        # Setup research results
        research_results = {
            "query": "climate change coastal impacts",
            "search_results": [
                {"title": "Sea Level Rise Study", "url": "https://example.com/1"},
                {"title": "Coastal Erosion Research", "url": "https://example.com/2"}
            ],
            "content": "Climate change is causing sea levels to rise, leading to coastal erosion and flooding.",
            "key_information": {
                "facts": ["Sea levels are rising", "Coastal erosion is increasing"],
                "statistics": ["Sea levels rose 8-9 inches since 1880"],
                "entities": ["coastal cities", "sea walls", "erosion"],
                "summary": "Climate change impacts on coastal regions include rising sea levels and erosion."
            }
        }
        
        # Call the method
        summary = await research_agent.create_research_summary(research_results)
        
        # Verify result structure
        assert "title" in summary
        assert "executive_summary" in summary
        assert "key_findings" in summary
        assert "detailed_analysis" in summary
        assert "sources" in summary
        assert "recommendations" in summary
        
        # Verify content
        assert "climate change" in summary["title"].lower()
        assert "coastal" in summary["title"].lower()
        
        assert len(summary["executive_summary"]) >= 100
        assert "sea level" in summary["executive_summary"].lower()
        
        assert len(summary["key_findings"]) >= 2
        assert any("sea level" in finding.lower() for finding in summary["key_findings"])
        
        assert len(summary["detailed_analysis"]) >= 200
        
        assert len(summary["sources"]) >= 2
        assert all(source["title"] for source in summary["sources"])
        assert all(source["url"] for source in summary["sources"])
        
        assert len(summary["recommendations"]) >= 2

    @pytest.mark.asyncio
    async def test_get_research_history(self, research_agent):
        """Test retrieving research history."""
        # Setup mock responses
        research_agent.memory_mcp.list_memories.return_value = """
        search_query_1
        search_results_1
        fetched_url_1
        search_query_2
        search_results_2
        research_results_1
        """
        
        research_agent.memory_mcp.retrieve_memory.side_effect = [
            "climate change impacts",                                # search_query_1
            "[{'title': 'Article 1', 'url': 'https://example.com'}]", # search_results_1
            "https://example.com/article",                           # fetched_url_1
            "renewable energy trends",                               # search_query_2
            "[{'title': 'Article 2', 'url': 'https://example.com'}]", # search_results_2
            "{'query': 'climate change impacts', 'content': 'Research content'}" # research_results_1
        ]
        
        # Call the method
        history = await research_agent.get_research_history()
        
        # Verify memory was queried
        research_agent.memory_mcp.list_memories.assert_called_once_with(namespace="research")
        
        # Verify result structure
        assert "queries" in history
        assert "results" in history
        assert "urls" in history
        
        assert len(history["queries"]) == 2
        assert "climate change impacts" in history["queries"]
        assert "renewable energy trends" in history["queries"]
        
        assert len(history["results"]) >= 1
        assert "climate change impacts" in history["results"]
        
        assert len(history["urls"]) >= 1
        assert "https://example.com/article" in history["urls"]

    @pytest.mark.asyncio
    async def test_get_research_history_empty(self, research_agent):
        """Test retrieving research history when empty."""
        # Setup mock with empty response
        research_agent.memory_mcp.list_memories.return_value = ""
        
        # Call the method
        history = await research_agent.get_research_history()
        
        # Verify result is empty
        assert history["queries"] == []
        assert history["results"] == {}
        assert history["urls"] == []


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
