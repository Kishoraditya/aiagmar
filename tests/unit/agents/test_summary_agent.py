"""
Unit tests for the Summary Agent.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from apps.agents.summary_agent import SummaryAgent
from apps.mcps.memory_mcp import MemoryMCP
from apps.utils.exceptions import AgentError, ValidationError

class TestSummaryAgent:
    """Test suite for SummaryAgent class."""

    @pytest.fixture
    def memory_mcp(self):
        """Fixture to create a mock MemoryMCP."""
        mock_memory = MagicMock(spec=MemoryMCP)
        mock_memory.store_memory = MagicMock()
        mock_memory.retrieve_memory = MagicMock()
        mock_memory.search_memories = MagicMock()
        mock_memory.list_memories = MagicMock()
        return mock_memory

    @pytest.fixture
    def summary_agent(self, memory_mcp):
        """Fixture to create a SummaryAgent instance with mock dependencies."""
        agent = SummaryAgent(
            name="summary",
            memory_mcp=memory_mcp
        )
        return agent

    def test_init(self, memory_mcp):
        """Test initialization of SummaryAgent."""
        agent = SummaryAgent(
            name="summary",
            memory_mcp=memory_mcp
        )
        
        assert agent.name == "summary"
        assert agent.memory_mcp == memory_mcp

    @pytest.mark.asyncio
    async def test_summarize_text_basic(self, summary_agent):
        """Test basic text summarization functionality."""
        # Setup test content
        content = """
        Climate change is the long-term alteration of temperature and typical weather patterns in a place. 
        Climate change could refer to a particular location or the planet as a whole. 
        Climate change may cause weather patterns to be less predictable. 
        These unexpected weather patterns can make it difficult to maintain and grow crops in regions that rely on farming.
        Climate change has also been connected with other damaging weather events such as more frequent and more intense hurricanes, floods, downpours, and winter storms.
        In polar regions, the warming global temperatures associated with climate change have meant ice sheets and glaciers are melting at an accelerated rate.
        This contributes to sea levels rising in different regions of the planet.
        Together with expanding ocean waters due to rising temperatures, the resulting rise in sea level has begun to damage coastlines as a result of increased flooding and erosion.
        """
        
        # Call the method
        summary = await summary_agent.summarize_text(content)
        
        # Verify memory was updated
        summary_agent.memory_mcp.store_memory.assert_called_with(
            "original_content", content, namespace="summary"
        )
        summary_agent.memory_mcp.store_memory.assert_any_call(
            "text_summary", summary, namespace="summary"
        )
        
        # Verify summary structure and content
        assert isinstance(summary, str)
        assert len(summary) < len(content)  # Summary should be shorter than original
        assert len(summary) > 50  # But still substantial
        assert "climate change" in summary.lower()

    @pytest.mark.asyncio
    async def test_summarize_text_with_length(self, summary_agent):
        """Test text summarization with specified length."""
        # Setup test content
        content = "Climate change is causing global temperatures to rise, leading to melting ice caps, rising sea levels, and more extreme weather events. Scientists have observed these changes over decades and attribute them primarily to human activities, especially the burning of fossil fuels."
        
        # Call the method with different lengths
        short_summary = await summary_agent.summarize_text(content, max_length=50)
        long_summary = await summary_agent.summarize_text(content, max_length=150)
        
        # Verify summaries have appropriate lengths
        assert len(short_summary) <= 60  # Allow some flexibility
        assert len(long_summary) > len(short_summary)
        assert "climate change" in short_summary.lower()
        assert "climate change" in long_summary.lower()

    @pytest.mark.asyncio
    async def test_summarize_text_with_focus(self, summary_agent):
        """Test text summarization with specific focus."""
        # Setup test content
        content = """
        Climate change affects various aspects of our planet. Rising temperatures lead to melting ice caps and glaciers.
        This contributes to sea level rise, threatening coastal communities and ecosystems.
        Climate change also impacts agriculture by altering growing seasons and increasing extreme weather events.
        Additionally, it affects human health through increased heat-related illnesses and changing disease patterns.
        Economic impacts include damage to infrastructure and changes in tourism patterns.
        """
        
        # Call the method with different focuses
        sea_level_summary = await summary_agent.summarize_text(content, focus="sea level rise")
        agriculture_summary = await summary_agent.summarize_text(content, focus="agricultural impacts")
        
        # Verify summaries focus on the specified aspects
        assert "sea level" in sea_level_summary.lower() or "coastal" in sea_level_summary.lower()
        assert "agriculture" in agriculture_summary.lower() or "growing" in agriculture_summary.lower()
        
        # Verify the focused summaries are different from each other
        assert sea_level_summary != agriculture_summary

    @pytest.mark.asyncio
    async def test_summarize_text_empty_content(self, summary_agent):
        """Test summarization with empty content."""
        # Call the method with empty content
        with pytest.raises(ValueError, match="Content cannot be empty"):
            await summary_agent.summarize_text("")

    @pytest.mark.asyncio
    async def test_summarize_text_very_short_content(self, summary_agent):
        """Test summarization with very short content."""
        # Setup very short content
        content = "Climate change is real."
        
        # Call the method
        summary = await summary_agent.summarize_text(content)
        
        # Verify summary is the same as the original for very short content
        assert summary == content

    @pytest.mark.asyncio
    async def test_generate_key_points(self, summary_agent):
        """Test generating key points from content."""
        # Setup test content
        content = """
        Climate change is primarily caused by human activities, especially the burning of fossil fuels.
        The main greenhouse gases are carbon dioxide, methane, and nitrous oxide.
        Global temperatures have increased by about 1°C since pre-industrial times.
        The Paris Agreement aims to limit global warming to well below 2°C.
        Renewable energy sources like solar and wind are crucial for reducing emissions.
        Climate change leads to more frequent and intense extreme weather events.
        Sea levels are rising due to melting ice and thermal expansion of oceans.
        """
        
        # Call the method
        key_points = await summary_agent.generate_key_points(content)
        
        # Verify memory was updated
        summary_agent.memory_mcp.store_memory.assert_called_with(
            "key_points", str(key_points), namespace="summary"
        )
        
        # Verify key points structure
        assert isinstance(key_points, list)
        assert len(key_points) >= 3
        
        # Verify key points content
        assert any("fossil fuels" in point.lower() for point in key_points)
        assert any("temperature" in point.lower() for point in key_points)
        assert any("sea level" in point.lower() for point in key_points)

    @pytest.mark.asyncio
    async def test_generate_key_points_with_count(self, summary_agent):
        """Test generating key points with specified count."""
        # Setup test content
        content = """
        Climate change is primarily caused by human activities, especially the burning of fossil fuels.
        The main greenhouse gases are carbon dioxide, methane, and nitrous oxide.
        Global temperatures have increased by about 1°C since pre-industrial times.
        The Paris Agreement aims to limit global warming to well below 2°C.
        Renewable energy sources like solar and wind are crucial for reducing emissions.
        Climate change leads to more frequent and intense extreme weather events.
        Sea levels are rising due to melting ice and thermal expansion of oceans.
        """
        
        # Call the method with different counts
        few_points = await summary_agent.generate_key_points(content, count=3)
        many_points = await summary_agent.generate_key_points(content, count=7)
        
        # Verify point counts
        assert len(few_points) == 3
        assert len(many_points) == 7

    @pytest.mark.asyncio
    async def test_generate_key_points_with_focus(self, summary_agent):
        """Test generating key points with specific focus."""
        # Setup test content
        content = """
        Climate change has various impacts. In agriculture, changing weather patterns affect crop yields.
        Rising temperatures lead to more frequent droughts in some regions.
        Extreme weather events like hurricanes and floods are becoming more common.
        Sea levels are rising due to melting ice caps and thermal expansion.
        Many species are at risk of extinction due to habitat loss.
        Human health is affected through heat-related illnesses and changing disease patterns.
        Economic impacts include damage to infrastructure and reduced productivity.
        """
        
        # Call the method with different focuses
        environmental_points = await summary_agent.generate_key_points(content, focus="environmental impacts")
        economic_points = await summary_agent.generate_key_points(content, focus="economic impacts")
        
        # Verify points focus on the specified aspects
        assert any("species" in point.lower() or "habitat" in point.lower() for point in environmental_points)
        assert any("economic" in point.lower() or "infrastructure" in point.lower() for point in economic_points)

    @pytest.mark.asyncio
    async def test_extract_insights(self, summary_agent):
        """Test extracting insights from content."""
        # Setup test content
        content = """
        Recent studies show that climate change is accelerating faster than previously predicted.
        While global emissions temporarily decreased during the COVID-19 pandemic, they quickly rebounded.
        Developing countries are disproportionately affected by climate change despite contributing less to emissions.
        Technological solutions alone are insufficient; policy changes and behavioral shifts are also necessary.
        Climate adaptation strategies are becoming as important as mitigation efforts.
        """
        
        # Call the method
        insights = await summary_agent.extract_insights(content)
        
        # Verify memory was updated
        summary_agent.memory_mcp.store_memory.assert_called_with(
            "insights", str(insights), namespace="summary"
        )
        
        # Verify insights structure
        assert "main_insights" in insights
        assert "implications" in insights
        assert "unexpected_findings" in insights
        
        # Verify insights content
        assert len(insights["main_insights"]) >= 2
        assert len(insights["implications"]) >= 2
        assert len(insights["unexpected_findings"]) >= 1
        
        # Check for specific content
        assert any("accelerating" in insight.lower() for insight in insights["main_insights"])
        assert any("developing countries" in insight.lower() for insight in insights["main_insights"])
        assert any("policy" in implication.lower() or "behavioral" in implication.lower() for implication in insights["implications"])

    @pytest.mark.asyncio
    async def test_extract_insights_minimal_content(self, summary_agent):
        """Test extracting insights from minimal content."""
        # Setup minimal content
        content = "Climate change is happening faster than expected."
        
        # Call the method
        insights = await summary_agent.extract_insights(content)
        
        # Verify insights structure is maintained even with minimal content
        assert "main_insights" in insights
        assert "implications" in insights
        assert "unexpected_findings" in insights
        
        # Verify minimal insights
        assert len(insights["main_insights"]) >= 1
        assert insights["main_insights"][0] == "Climate change is happening faster than expected."
        assert len(insights["implications"]) >= 1
        assert len(insights["unexpected_findings"]) >= 0

    @pytest.mark.asyncio
    async def test_create_executive_summary(self, summary_agent):
        """Test creating an executive summary from research results."""
        # Setup research results
        research_results = {
            "query": "climate change impacts",
            "search_results": [
                {"title": "Sea Level Rise Study", "url": "https://example.com/1"},
                {"title": "Agricultural Impacts of Climate Change", "url": "https://example.com/2"}
            ],
            "content": "Climate change is causing sea levels to rise and affecting agricultural productivity worldwide.",
            "key_information": {
                "facts": ["Sea levels are rising", "Crop yields are decreasing in many regions"],
                "statistics": ["Sea levels rose 8-9 inches since 1880"],
                "entities": ["coastal cities", "agriculture", "food security"],
                "summary": "Climate change has significant impacts on coastal regions and agriculture."
            }
        }
        
        # Call the method
        executive_summary = await summary_agent.create_executive_summary(research_results)
        
        # Verify memory was updated
        summary_agent.memory_mcp.store_memory.assert_called_with(
            "executive_summary", executive_summary, namespace="summary"
        )
        
        # Verify executive summary structure and content
        assert isinstance(executive_summary, str)
        assert len(executive_summary) >= 100
        assert "climate change" in executive_summary.lower()
        assert "sea level" in executive_summary.lower() or "coastal" in executive_summary.lower()
        assert "agriculture" in executive_summary.lower() or "crop" in executive_summary.lower()

    @pytest.mark.asyncio
    async def test_create_executive_summary_minimal(self, summary_agent):
        """Test creating an executive summary with minimal research results."""
        # Setup minimal research results
        research_results = {
            "query": "climate change",
            "content": "Climate change is a global issue.",
            "key_information": {
                "summary": "Climate change affects the entire planet."
            }
        }
        
        # Call the method
        executive_summary = await summary_agent.create_executive_summary(research_results)
        
        # Verify executive summary is created even with minimal input
        assert isinstance(executive_summary, str)
        assert len(executive_summary) >= 20
        assert "climate change" in executive_summary.lower()

    @pytest.mark.asyncio
    async def test_create_executive_summary_invalid_input(self, summary_agent):
        """Test creating an executive summary with invalid input."""
        # Call the method with invalid input
        with pytest.raises(ValueError, match="Invalid research results"):
            await summary_agent.create_executive_summary({})

    @pytest.mark.asyncio
    async def test_summarize_multiple_documents(self, summary_agent):
        """Test summarizing multiple documents."""
        # Setup documents
        documents = {
            "doc1": "Climate change is causing sea levels to rise, threatening coastal communities.",
            "doc2": "Global temperatures have increased by about 1°C since pre-industrial times.",
            "doc3": "Extreme weather events like hurricanes and floods are becoming more frequent and intense."
        }
        
        # Call the method
        summary = await summary_agent.summarize_multiple_documents(documents)
        
        # Verify memory was updated
        summary_agent.memory_mcp.store_memory.assert_called_with(
            "multi_document_summary", summary, namespace="summary"
        )
        
        # Verify summary structure and content
        assert isinstance(summary, str)
        assert len(summary) >= 50
        assert "climate change" in summary.lower()
        assert "sea level" in summary.lower() or "coastal" in summary.lower()
        assert "temperature" in summary.lower()
        assert "extreme weather" in summary.lower() or "hurricanes" in summary.lower() or "floods" in summary.lower()

    @pytest.mark.asyncio
    async def test_summarize_multiple_documents_with_focus(self, summary_agent):
        """Test summarizing multiple documents with a specific focus."""
        # Setup documents
        documents = {
            "doc1": "Climate change is causing sea levels to rise, threatening coastal communities.",
            "doc2": "Global temperatures have increased by about 1°C since pre-industrial times.",
            "doc3": "Extreme weather events like hurricanes and floods are becoming more frequent and intense."
        }
        
        # Call the method with focus
        summary = await summary_agent.summarize_multiple_documents(documents, focus="coastal impacts")
        
        # Verify summary focuses on coastal impacts
        assert "sea level" in summary.lower() or "coastal" in summary.lower()
        assert len(summary) >= 30

    @pytest.mark.asyncio
    async def test_summarize_multiple_documents_empty(self, summary_agent):
        """Test summarizing empty documents."""
        # Call the method with empty documents
        with pytest.raises(ValueError, match="No documents provided"):
            await summary_agent.summarize_multiple_documents({})

    @pytest.mark.asyncio
    async def test_create_abstract(self, summary_agent):
        """Test creating an abstract from content."""
        # Setup content
        content = """
        Climate change is the long-term alteration of temperature and typical weather patterns in a place.
        The primary cause is the burning of fossil fuels, which releases carbon dioxide and other greenhouse gases into the atmosphere.
        These gases trap heat from the sun's rays inside the atmosphere, causing Earth's average temperature to rise.
        Rising global temperatures have been accompanied by changes in weather and climate.
        Many places have seen changes in rainfall, resulting in more floods, droughts, or intense rain, as well as more frequent and severe heat waves.
        The planet's oceans and glaciers have also experienced changes—oceans are warming and becoming more acidic, ice caps are melting, and sea levels are rising.
        These changes can harm animals and their habitats, as well as affect people's health and way of life.
        """
        
        # Call the method
        abstract = await summary_agent.create_abstract(content)
        
        # Verify memory was updated
        summary_agent.memory_mcp.store_memory.assert_called_with(
            "abstract", abstract, namespace="summary"
        )
        
        # Verify abstract structure and content
        assert isinstance(abstract, str)
        assert len(abstract) >= 100
        assert len(abstract) < len(content)  # Abstract should be shorter than original
        assert "climate change" in abstract.lower()
        assert "temperature" in abstract.lower()
        assert "greenhouse gases" in abstract.lower() or "carbon dioxide" in abstract.lower()

    @pytest.mark.asyncio
    async def test_create_abstract_with_length(self, summary_agent):
        """Test creating an abstract with specified length."""
        # Setup content
        content = """
        Climate change refers to significant changes in global temperature, precipitation, wind patterns, and other measures of climate that occur over several decades or longer.
        The Earth's climate has changed throughout history. Just in the last 650,000 years, there have been seven cycles of glacial advance and retreat.
        However, the current warming trend is of particular significance because it is primarily the result of human activities since the mid-20th century.
        The industrial activities that our modern civilization depends upon have raised atmospheric carbon dioxide levels from 280 parts per million to about 420 parts per million in the last 150 years.
        The consequences of changing the natural atmospheric greenhouse are hard to predict, but certain effects seem likely.
        On average, Earth will become warmer. Some regions may welcome warmer temperatures, but others may not.
        """
        
        # Call the method with different lengths
        short_abstract = await summary_agent.create_abstract(content, max_length=100)
        long_abstract = await summary_agent.create_abstract(content, max_length=300)
        
        # Verify abstracts have appropriate lengths
        assert len(short_abstract) <= 120  # Allow some flexibility
        assert len(long_abstract) > len(short_abstract)
        assert len(long_abstract) <= 320  # Allow some flexibility

    @pytest.mark.asyncio
    async def test_create_abstract_with_style(self, summary_agent):
        """Test creating an abstract with different styles."""
        # Setup content
        content = """
        Climate change is a significant and lasting change in the statistical distribution of weather patterns over periods ranging from decades to millions of years.
        It may be a change in average weather conditions, or in the distribution of weather around the average conditions.
        Climate change is caused by factors such as biotic processes, variations in solar radiation received by Earth, plate tectonics, and volcanic eruptions.
        Certain human activities have been identified as primary causes of ongoing climate change, often referred to as global warming.
        """
        
        # Call the method with different styles
        academic_abstract = await summary_agent.create_abstract(content, style="academic")
        general_abstract = await summary_agent.create_abstract(content, style="general")
        
        # Verify abstracts have different styles
        assert academic_abstract != general_abstract
        
        # Academic style should use more formal language
        assert any(term in academic_abstract.lower() for term in ["significant", "statistical", "distribution", "biotic processes"])
        
        # General style should be more accessible
        assert "global warming" in general_abstract.lower() or "human activities" in general_abstract.lower()

    @pytest.mark.asyncio
    async def test_generate_summary_bullets(self, summary_agent):
        """Test generating summary bullets from content."""
        # Setup content
        content = """
        Climate change is primarily caused by human activities, especially the burning of fossil fuels.
        The main greenhouse gases are carbon dioxide, methane, and nitrous oxide.
        Global temperatures have increased by about 1°C since pre-industrial times.
        The Paris Agreement aims to limit global warming to well below 2°C.
        Renewable energy sources like solar and wind are crucial for reducing emissions.
        Climate change leads to more frequent and intense extreme weather events.
        Sea levels are rising due to melting ice and thermal expansion of oceans.
        """
        
        # Call the method
        bullets = await summary_agent.generate_summary_bullets(content)
        
        # Verify memory was updated
        summary_agent.memory_mcp.store_memory.assert_called_with(
            "summary_bullets", str(bullets), namespace="summary"
        )
        
        # Verify bullets structure
        assert isinstance(bullets, list)
        assert len(bullets) >= 5
        
        # Verify bullets content
        assert all(isinstance(bullet, str) for bullet in bullets)
        assert any("fossil fuels" in bullet.lower() for bullet in bullets)
        assert any("greenhouse gases" in bullet.lower() for bullet in bullets)
        assert any("temperature" in bullet.lower() for bullet in bullets)
        assert any("paris agreement" in bullet.lower() for bullet in bullets)
        assert any("renewable energy" in bullet.lower() for bullet in bullets)

    @pytest.mark.asyncio
    async def test_generate_summary_bullets_with_count(self, summary_agent):
        """Test generating summary bullets with specified count."""
        # Setup content
        content = """
        Climate change is primarily caused by human activities, especially the burning of fossil fuels.
        The main greenhouse gases are carbon dioxide, methane, and nitrous oxide.
        Global temperatures have increased by about 1°C since pre-industrial times.
        The Paris Agreement aims to limit global warming to well below 2°C.
        Renewable energy sources like solar and wind are crucial for reducing emissions.
        Climate change leads to more frequent and intense extreme weather events.
        Sea levels are rising due to melting ice and thermal expansion of oceans.
        """
        
        # Call the method with different counts
        few_bullets = await summary_agent.generate_summary_bullets(content, count=3)
        many_bullets = await summary_agent.generate_summary_bullets(content, count=7)
        
        # Verify bullet counts
        assert len(few_bullets) == 3
        assert len(many_bullets) == 7

    @pytest.mark.asyncio
    async def test_generate_summary_bullets_with_style(self, summary_agent):
        """Test generating summary bullets with different styles."""
        # Setup content
        content = """
        Climate change is primarily caused by human activities, especially the burning of fossil fuels.
        The main greenhouse gases are carbon dioxide, methane, and nitrous oxide.
        Global temperatures have increased by about 1°C since pre-industrial times.
        The Paris Agreement aims to limit global warming to well below 2°C.
        """
        
        # Call the method with different styles
        concise_bullets = await summary_agent.generate_summary_bullets(content, style="concise")
        detailed_bullets = await summary_agent.generate_summary_bullets(content, style="detailed")
        
        # Verify bullets have different styles
        assert len(concise_bullets[0]) < len(detailed_bullets[0])
        
        # Concise style should be shorter
        assert all(len(bullet) < 100 for bullet in concise_bullets)
        
        # Detailed style should provide more information
        assert any(len(bullet) > 50 for bullet in detailed_bullets)

    @pytest.mark.asyncio
    async def test_identify_themes(self, summary_agent):
        """Test identifying themes in content."""
        # Setup content
        content = """
        Climate change is causing sea levels to rise, threatening coastal communities.
        Many island nations are at risk of being submerged in the coming decades.
        Coastal erosion is accelerating due to stronger storms and higher tides.
        Agricultural productivity is declining in many regions due to changing weather patterns.
        Droughts are becoming more common in some areas, while others experience increased flooding.
        Food security is a growing concern as crop yields become less predictable.
        Human health is affected through heat-related illnesses and changing disease patterns.
        Air pollution, often from the same sources as greenhouse gases, causes respiratory problems.
        Mental health impacts include anxiety about climate change and displacement stress.
        """
        
        # Call the method
        themes = await summary_agent.identify_themes(content)
        
        # Verify memory was updated
        summary_agent.memory_mcp.store_memory.assert_called_with(
            "content_themes", str(themes), namespace="summary"
        )
        
        # Verify themes structure
        assert "main_themes" in themes
        assert "sub_themes" in themes
        assert "relationships" in themes
        
        # Verify themes content
        assert len(themes["main_themes"]) >= 3
        assert len(themes["sub_themes"]) >= 3
        assert len(themes["relationships"]) >= 2
        
        # Check for specific themes
        assert any("coastal" in theme.lower() for theme in themes["main_themes"]) or \
               any("sea level" in theme.lower() for theme in themes["main_themes"])
        
        assert any("agriculture" in theme.lower() for theme in themes["main_themes"]) or \
               any("food" in theme.lower() for theme in themes["main_themes"])
        
        assert any("health" in theme.lower() for theme in themes["main_themes"])

    @pytest.mark.asyncio
    async def test_identify_themes_minimal_content(self, summary_agent):
        """Test identifying themes with minimal content."""
        # Setup minimal content
        content = "Climate change affects coastal regions and agriculture."
        
        # Call the method
        themes = await summary_agent.identify_themes(content)
        
        # Verify themes structure is maintained even with minimal content
        assert "main_themes" in themes
        assert "sub_themes" in themes
        assert "relationships" in themes
        
        # Verify minimal themes
        assert len(themes["main_themes"]) >= 2
        assert "coastal" in str(themes["main_themes"]).lower() or "agriculture" in str(themes["main_themes"]).lower()
        assert len(themes["sub_themes"]) >= 0
        assert len(themes["relationships"]) >= 0

    @pytest.mark.asyncio
    async def test_compare_documents(self, summary_agent):
        """Test comparing multiple documents."""
        # Setup documents
        documents = {
            "doc1": "Climate change is causing sea levels to rise, threatening coastal communities.",
            "doc2": "Global warming leads to sea level rise due to melting ice caps and thermal expansion.",
            "doc3": "Climate change affects agriculture through changing rainfall patterns and temperature increases."
        }
        
        # Call the method
        comparison = await summary_agent.compare_documents(documents)
        
        # Verify memory was updated
        summary_agent.memory_mcp.store_memory.assert_called_with(
            "document_comparison", str(comparison), namespace="summary"
        )
        
        # Verify comparison structure
        assert "common_themes" in comparison
        assert "unique_points" in comparison
        assert "contradictions" in comparison
        assert "synthesis" in comparison
        
        # Verify comparison content
        assert len(comparison["common_themes"]) >= 1
        assert "climate change" in str(comparison["common_themes"]).lower() or \
               "sea level" in str(comparison["common_themes"]).lower()
        
        assert len(comparison["unique_points"]) >= 2
        assert "agriculture" in str(comparison["unique_points"]).lower() or \
               "rainfall" in str(comparison["unique_points"]).lower()
        
        assert isinstance(comparison["synthesis"], str)
        assert len(comparison["synthesis"]) >= 50

    @pytest.mark.asyncio
    async def test_compare_documents_with_focus(self, summary_agent):
        """Test comparing documents with a specific focus."""
        # Setup documents
        documents = {
            "doc1": "Climate change is causing sea levels to rise, threatening coastal communities.",
            "doc2": "Global warming leads to sea level rise due to melting ice caps and thermal expansion.",
            "doc3": "Climate change affects agriculture through changing rainfall patterns and temperature increases."
        }
        
        # Call the method with focus
        comparison = await summary_agent.compare_documents(documents, focus="sea level rise")
        
        # Verify comparison focuses on sea level rise
        assert "sea level" in str(comparison["common_themes"]).lower() or \
               "coastal" in str(comparison["common_themes"]).lower()
        
        assert "melting ice" in str(comparison["unique_points"]).lower() or \
               "thermal expansion" in str(comparison["unique_points"]).lower()
        
        assert "sea level" in comparison["synthesis"].lower()

    @pytest.mark.asyncio
    async def test_compare_documents_similar(self, summary_agent):
        """Test comparing very similar documents."""
        # Setup similar documents
        documents = {
            "doc1": "Climate change is causing sea levels to rise.",
            "doc2": "Climate change is leading to rising sea levels.",
            "doc3": "Sea levels are increasing due to climate change."
        }
        
        # Call the method
        comparison = await summary_agent.compare_documents(documents)
        
        # Verify comparison identifies similarity
        assert len(comparison["common_themes"]) >= 1
        assert "climate change" in str(comparison["common_themes"]).lower() and \
               "sea level" in str(comparison["common_themes"]).lower()
        
        assert len(comparison["unique_points"]) == 0 or len(comparison["unique_points"]) <= 1
        assert len(comparison["contradictions"]) == 0
        assert "similar" in comparison["synthesis"].lower() or "same" in comparison["synthesis"].lower()

    @pytest.mark.asyncio
    async def test_compare_documents_contradictory(self, summary_agent):
        """Test comparing contradictory documents."""
        # Setup contradictory documents
        documents = {
            "doc1": "Climate change is primarily caused by human activities and requires immediate action.",
            "doc2": "Climate change is a natural cycle and human impact is minimal.",
            "doc3": "Climate change is real but the extent of human contribution is debated."
        }
        
        # Call the method
        comparison = await summary_agent.compare_documents(documents)
        
        # Verify comparison identifies contradictions
        assert len(comparison["contradictions"]) >= 1
        assert "human" in str(comparison["contradictions"]).lower() or \
               "natural" in str(comparison["contradictions"]).lower()
        
        assert "debate" in comparison["synthesis"].lower() or \
               "disagree" in comparison["synthesis"].lower() or \
               "contrast" in comparison["synthesis"].lower()

    @pytest.mark.asyncio
    async def test_create_summary_with_sections(self, summary_agent):
        """Test creating a structured summary with sections."""
        # Setup content
        content = """
        Climate change is the long-term alteration of temperature and typical weather patterns in a place.
        The primary cause is the burning of fossil fuels, which releases carbon dioxide and other greenhouse gases.
        These gases trap heat from the sun's rays inside the atmosphere, causing Earth's average temperature to rise.
        Rising global temperatures have been accompanied by changes in weather and climate.
        Many places have seen changes in rainfall, resulting in more floods, droughts, or intense rain.
        The planet's oceans and glaciers have also experienced changes—oceans are warming and becoming more acidic.
        Ice caps are melting, and sea levels are rising, threatening coastal communities.
        These changes can harm animals and their habitats, as well as affect people's health and way of life.
        Mitigation strategies include transitioning to renewable energy, improving energy efficiency, and reforestation.
        Adaptation measures involve building sea walls, developing drought-resistant crops, and improving early warning systems.
        """
        
        # Call the method
        sections = ["Introduction", "Causes", "Effects", "Solutions"]
        summary = await summary_agent.create_summary_with_sections(content, sections)
        
        # Verify memory was updated
        summary_agent.memory_mcp.store_memory.assert_called_with(
            "sectioned_summary", str(summary), namespace="summary"
        )
        
        # Verify summary structure
        assert isinstance(summary, dict)
        assert all(section in summary for section in sections)
        
        # Verify section content
        assert "climate change" in summary["Introduction"].lower()
        assert "fossil fuels" in summary["Causes"].lower() or "greenhouse gases" in summary["Causes"].lower()
        assert "temperature" in summary["Effects"].lower() or "sea level" in summary["Effects"].lower()
        assert "renewable" in summary["Solutions"].lower() or "adaptation" in summary["Solutions"].lower()

    @pytest.mark.asyncio
    async def test_create_summary_with_custom_sections(self, summary_agent):
        """Test creating a summary with custom sections."""
        # Setup content
        content = """
        Climate change is affecting global food security through multiple pathways.
        Rising temperatures reduce crop yields in many regions, especially in tropical areas.
        Changing precipitation patterns lead to more frequent droughts and floods, damaging crops.
        Extreme weather events like hurricanes and heatwaves can destroy entire harvests.
        Carbon dioxide fertilization may benefit some crops but reduces nutritional quality.
        Pests and diseases are expanding their ranges due to warming temperatures.
        Food prices are becoming more volatile due to climate-related supply disruptions.
        Smallholder farmers in developing countries are particularly vulnerable.
        Adaptation strategies include developing heat-resistant crop varieties and improving irrigation.
        Climate-smart agriculture practices can help mitigate emissions while improving resilience.
        """
        
        # Call the method with custom sections
        custom_sections = ["Climate Impacts on Agriculture", "Food Security Implications", "Adaptation Strategies"]
        summary = await summary_agent.create_summary_with_sections(content, custom_sections)
        
        # Verify summary structure
        assert isinstance(summary, dict)
        assert all(section in summary for section in custom_sections)
        
        # Verify section content
        assert "temperature" in summary["Climate Impacts on Agriculture"].lower() or \
               "crop yields" in summary["Climate Impacts on Agriculture"].lower()
        
        assert "prices" in summary["Food Security Implications"].lower() or \
               "vulnerable" in summary["Food Security Implications"].lower()
        
        assert "heat-resistant" in summary["Adaptation Strategies"].lower() or \
               "irrigation" in summary["Adaptation Strategies"].lower() or \
               "climate-smart" in summary["Adaptation Strategies"].lower()

    @pytest.mark.asyncio
    async def test_create_summary_with_sections_invalid_input(self, summary_agent):
        """Test creating a sectioned summary with invalid input."""
        # Call the method with empty content
        with pytest.raises(ValueError, match="Content cannot be empty"):
            await summary_agent.create_summary_with_sections("", ["Section"])
        
        # Call the method with empty sections
        with pytest.raises(ValueError, match="At least one section must be provided"):
            await summary_agent.create_summary_with_sections("Content", [])

    @pytest.mark.asyncio
    async def test_extract_statistics(self, summary_agent):
        """Test extracting statistics from content."""
        # Setup content with statistics
        content = """
        Global temperatures have increased by about 1.1°C since pre-industrial times.
        Sea levels have risen by 8-9 inches (21-24 cm) since 1880.
        The rate of sea level rise has doubled from 1.4 mm per year throughout most of the 20th century to 3.6 mm per year from 2006-2015.
        Arctic sea ice has declined by about 13% per decade since 1979.
        Carbon dioxide levels in the atmosphere have increased by 48% since pre-industrial times, from 280 ppm to 415 ppm.
        About 75% of greenhouse gas emissions come from the energy and transportation sectors.
        The world's oceans have absorbed more than 90% of the excess heat from global warming.
        """
        
        # Call the method
        statistics = await summary_agent.extract_statistics(content)
        
        # Verify memory was updated
        summary_agent.memory_mcp.store_memory.assert_called_with(
            "extracted_statistics", str(statistics), namespace="summary"
        )
        
        # Verify statistics structure
        assert isinstance(statistics, list)
        assert len(statistics) >= 5
        
        # Verify statistics content
        assert any("1.1°C" in stat for stat in statistics)
        assert any("8-9 inches" in stat or "21-24 cm" in stat for stat in statistics)
        assert any("13%" in stat and "Arctic" in stat for stat in statistics)
        assert any("48%" in stat and "carbon dioxide" in stat for stat in statistics)
        assert any("90%" in stat and "oceans" in stat for stat in statistics)

    @pytest.mark.asyncio
    async def test_extract_statistics_no_stats(self, summary_agent):
        """Test extracting statistics from content with no statistics."""
        # Setup content without statistics
        content = "Climate change is a global issue that affects everyone. It causes various environmental problems."
        
        # Call the method
        statistics = await summary_agent.extract_statistics(content)
        
        # Verify result
        assert isinstance(statistics, list)
        assert len(statistics) == 0

    @pytest.mark.asyncio
    async def test_extract_statistics_with_focus(self, summary_agent):
        """Test extracting statistics with a specific focus."""
        # Setup content with various statistics
        content = """
        Global temperatures have increased by about 1.1°C since pre-industrial times.
        Sea levels have risen by 8-9 inches (21-24 cm) since 1880.
        The rate of sea level rise has doubled from 1.4 mm per year throughout most of the 20th century to 3.6 mm per year from 2006-2015.
        Arctic sea ice has declined by about 13% per decade since 1979.
        Carbon dioxide levels in the atmosphere have increased by 48% since pre-industrial times, from 280 ppm to 415 ppm.
        About 75% of greenhouse gas emissions come from the energy and transportation sectors.
        The world's oceans have absorbed more than 90% of the excess heat from global warming.
        """
        
        # Call the method with focus
        sea_level_stats = await summary_agent.extract_statistics(content, focus="sea level")
        emissions_stats = await summary_agent.extract_statistics(content, focus="emissions")
        
        # Verify focused statistics
        assert any("8-9 inches" in stat or "21-24 cm" in stat for stat in sea_level_stats)
        assert any("1.4 mm" in stat or "3.6 mm" in stat for stat in sea_level_stats)
        assert not any("13%" in stat and "Arctic" in stat for stat in sea_level_stats)
        
        assert any("75%" in stat and "emissions" in stat for stat in emissions_stats)
        assert any("48%" in stat and "carbon dioxide" in stat for stat in emissions_stats)
        assert not any("8-9 inches" in stat for stat in emissions_stats)

    @pytest.mark.asyncio
    async def test_generate_summary_timeline(self, summary_agent):
        """Test generating a timeline summary from content."""
        # Setup content with timeline information
        content = """
        In the 1820s, Joseph Fourier first described the greenhouse effect.
        In 1896, Svante Arrhenius calculated how changes in CO2 levels could affect global temperatures.
        By the 1950s, scientists began regular measurements of CO2 in the atmosphere.
        In 1988, the Intergovernmental Panel on Climate Change (IPCC) was established.
        The Kyoto Protocol was adopted in 1997 as the first international agreement to reduce greenhouse gas emissions.
        In 2015, the Paris Agreement set the goal of limiting global warming to well below 2°C.
        The IPCC's 2018 special report warned that limiting warming to 1.5°C would require rapid, far-reaching changes.
        In 2021, the IPCC's Sixth Assessment Report stated that human influence on the climate system is now "unequivocal."
        """
        
        # Call the method
        timeline = await summary_agent.generate_summary_timeline(content)
        
        # Verify memory was updated
        summary_agent.memory_mcp.store_memory.assert_called_with(
            "summary_timeline", str(timeline), namespace="summary"
        )
        
        # Verify timeline structure
        assert isinstance(timeline, list)
        assert len(timeline) >= 5
        assert all(isinstance(event, dict) for event in timeline)
        assert all("year" in event and "event" in event for event in timeline)
        
        # Verify timeline content
        assert any(event["year"] == "1820s" and "Fourier" in event["event"] for event in timeline)
        assert any(event["year"] == "1988" and "IPCC" in event["event"] for event in timeline)
        assert any(event["year"] == "2015" and "Paris Agreement" in event["event"] for event in timeline)
        
        # Verify chronological order
        years = [int(event["year"].replace("s", "")) if "s" in event["year"] else int(event["year"]) for event in timeline]
        assert years == sorted(years)

    @pytest.mark.asyncio
    async def test_generate_summary_timeline_no_dates(self, summary_agent):
        """Test generating a timeline from content with no clear dates."""
        # Setup content without clear dates
        content = "Climate change is a global issue that affects everyone. It causes various environmental problems."
        
        # Call the method
        timeline = await summary_agent.generate_summary_timeline(content)
        
        # Verify result
        assert isinstance(timeline, list)
        assert len(timeline) == 0 or (len(timeline) == 1 and "No clear timeline" in timeline[0]["event"])

    @pytest.mark.asyncio
    async def test_summarize_research_findings(self, summary_agent):
        """Test summarizing research findings from multiple sources."""
        # Setup research findings
        findings = {
            "source1": {
                "title": "Sea Level Rise Study",
                "content": "Sea levels are rising at an accelerating rate due to climate change."
            },
            "source2": {
                "title": "Agricultural Impacts Research",
                "content": "Climate change is reducing crop yields in many regions."
            },
            "source3": {
                "title": "Health Effects Analysis",
                "content": "Rising temperatures are increasing heat-related illnesses."
            }
        }
        
        # Call the method
        summary = await summary_agent.summarize_research_findings(findings)
        
        # Verify memory was updated
        summary_agent.memory_mcp.store_memory.assert_called_with(
            "research_findings_summary", str(summary), namespace="summary"
        )
        
        # Verify summary structure
        assert "overview" in summary
        assert "key_findings" in summary
        assert "source_analysis" in summary
        assert "implications" in summary
        
        # Verify summary content
        assert len(summary["overview"]) >= 50
        assert len(summary["key_findings"]) >= 3
        assert len(summary["source_analysis"]) >= 3
        assert len(summary["implications"]) >= 2
        
        # Check for specific content
        assert "sea level" in summary["overview"].lower() or "sea level" in str(summary["key_findings"]).lower()
        assert "agriculture" in summary["overview"].lower() or "crop" in str(summary["key_findings"]).lower()
        assert "health" in summary["overview"].lower() or "health" in str(summary["key_findings"]).lower()

    @pytest.mark.asyncio
    async def test_summarize_research_findings_minimal(self, summary_agent):
        """Test summarizing research findings with minimal input."""
        # Setup minimal findings
        findings = {
            "source1": {
                "title": "Climate Study",
                "content": "Climate change is a global issue."
            }
        }
        
        # Call the method
        summary = await summary_agent.summarize_research_findings(findings)
        
        # Verify summary structure is maintained even with minimal input
        assert "overview" in summary
        assert "key_findings" in summary
        assert "source_analysis" in summary
        assert "implications" in summary
        
        # Verify minimal content
        assert len(summary["overview"]) >= 10
        assert len(summary["key_findings"]) >= 1
        assert len(summary["source_analysis"]) >= 1
        assert len(summary["implications"]) >= 1
        assert "climate change" in summary["overview"].lower()

    @pytest.mark.asyncio
    async def test_summarize_research_findings_invalid_input(self, summary_agent):
        """Test summarizing research findings with invalid input."""
        # Call the method with empty findings
        with pytest.raises(ValueError, match="No research findings provided"):
            await summary_agent.summarize_research_findings({})


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
