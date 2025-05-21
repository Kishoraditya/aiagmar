"""
Unit tests for the Image Generation Agent.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from apps.agents.image_generation_agent import ImageGenerationAgent
from apps.mcps.everart_mcp import EverArtMCP
from apps.mcps.memory_mcp import MemoryMCP
from apps.utils.exceptions import ImageGenerationError

class TestImageGenerationAgent:
    """Test suite for ImageGenerationAgent class."""

    @pytest.fixture
    def everart_mcp(self):
        """Fixture to create a mock EverArtMCP."""
        mock_everart = MagicMock(spec=EverArtMCP)
        mock_everart.generate_image = MagicMock()
        mock_everart.enhance_image = MagicMock()
        mock_everart.describe_image = MagicMock()
        return mock_everart

    @pytest.fixture
    def memory_mcp(self):
        """Fixture to create a mock MemoryMCP."""
        mock_memory = MagicMock(spec=MemoryMCP)
        mock_memory.store_memory = MagicMock()
        mock_memory.retrieve_memory = MagicMock()
        return mock_memory

    @pytest.fixture
    def image_generation_agent(self, everart_mcp, memory_mcp):
        """Fixture to create an ImageGenerationAgent instance with mock dependencies."""
        agent = ImageGenerationAgent(
            name="image_generator",
            everart_mcp=everart_mcp,
            memory_mcp=memory_mcp
        )
        return agent

    def test_init(self, everart_mcp, memory_mcp):
        """Test initialization of ImageGenerationAgent."""
        agent = ImageGenerationAgent(
            name="image_generator",
            everart_mcp=everart_mcp,
            memory_mcp=memory_mcp
        )
        
        assert agent.name == "image_generator"
        assert agent.everart_mcp == everart_mcp
        assert agent.memory_mcp == memory_mcp

    @pytest.mark.asyncio
    async def test_generate_image_from_text(self, image_generation_agent):
        """Test generating an image from text description."""
        # Setup mock response
        image_generation_agent.everart_mcp.generate_image.return_value = """
        Generated image URL: https://example.com/image1.jpg
        """
        
        # Call the method
        prompt = "A beautiful mountain landscape with a lake"
        style = "realistic"
        aspect_ratio = "16:9"
        
        result = await image_generation_agent.generate_image_from_text(prompt, style, aspect_ratio)
        
        # Verify EverArt MCP was called correctly
        image_generation_agent.everart_mcp.generate_image.assert_called_once_with(
            prompt=prompt,
            style=style,
            aspect_ratio=aspect_ratio,
            num_images=1
        )
        
        # Verify result
        assert "https://example.com/image1.jpg" in result
        
        # Verify memory was updated
        image_generation_agent.memory_mcp.store_memory.assert_called_once()
        args, kwargs = image_generation_agent.memory_mcp.store_memory.call_args
        assert args[0].startswith("image_")
        assert "https://example.com/image1.jpg" in args[1]
        assert kwargs["namespace"] == "image_generation"

    @pytest.mark.asyncio
    async def test_generate_image_from_text_error(self, image_generation_agent):
        """Test handling errors when generating an image from text."""
        # Setup mock to raise exception
        image_generation_agent.everart_mcp.generate_image.side_effect = Exception("Generation failed")
        
        # Call the method and expect error
        with pytest.raises(ImageGenerationError, match="Failed to generate image from text"):
            await image_generation_agent.generate_image_from_text("A mountain landscape")
        
        # Verify memory was not updated
        image_generation_agent.memory_mcp.store_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_image_from_research(self, image_generation_agent):
        """Test generating an image based on research content."""
        # Setup mock response
        image_generation_agent.everart_mcp.generate_image.return_value = """
        Generated image URL: https://example.com/research-image.jpg
        """
        
        # Call the method
        research_content = """
        Climate change is causing global temperatures to rise.
        This is leading to melting ice caps and rising sea levels.
        Many coastal cities are at risk of flooding in the coming decades.
        """
        
        result = await image_generation_agent.generate_image_from_research(
            research_content,
            query="climate change"
        )
        
        # Verify EverArt MCP was called with appropriate prompt
        image_generation_agent.everart_mcp.generate_image.assert_called_once()
        args, kwargs = image_generation_agent.everart_mcp.generate_image.call_args
        
        # The prompt should contain elements from the research content
        prompt = kwargs["prompt"]
        assert "climate change" in prompt.lower()
        assert "temperature" in prompt.lower() or "sea levels" in prompt.lower()
        
        # Verify result
        assert "https://example.com/research-image.jpg" in result
        
        # Verify memory was updated
        image_generation_agent.memory_mcp.store_memory.assert_called_once()
        args, kwargs = image_generation_agent.memory_mcp.store_memory.call_args
        assert args[0] == "image_climate change"
        assert "https://example.com/research-image.jpg" in args[1]
        assert kwargs["namespace"] == "image_generation"

    @pytest.mark.asyncio
    async def test_generate_image_from_research_with_style(self, image_generation_agent):
        """Test generating an image from research with specific style."""
        # Setup mock response
        image_generation_agent.everart_mcp.generate_image.return_value = """
        Generated image URL: https://example.com/styled-image.jpg
        """
        
        # Call the method
        research_content = "AI is transforming many industries through automation."
        
        result = await image_generation_agent.generate_image_from_research(
            research_content,
            query="AI impact",
            style="digital art",
            aspect_ratio="1:1"
        )
        
        # Verify EverArt MCP was called with correct parameters
        image_generation_agent.everart_mcp.generate_image.assert_called_once()
        args, kwargs = image_generation_agent.everart_mcp.generate_image.call_args
        assert kwargs["style"] == "digital art"
        assert kwargs["aspect_ratio"] == "1:1"
        
        # Verify result
        assert "https://example.com/styled-image.jpg" in result

    @pytest.mark.asyncio
    async def test_enhance_existing_image(self, image_generation_agent):
        """Test enhancing an existing image."""
        # Setup mock response
        image_generation_agent.everart_mcp.enhance_image.return_value = """
        Enhanced image URL: https://example.com/enhanced-image.jpg
        """
        
        # Call the method
        image_url = "https://example.com/original-image.jpg"
        enhancement_prompt = "Make the colors more vibrant and add a sunset glow"
        
        result = await image_generation_agent.enhance_existing_image(
            image_url,
            enhancement_prompt,
            strength=0.7
        )
        
        # Verify EverArt MCP was called correctly
        image_generation_agent.everart_mcp.enhance_image.assert_called_once_with(
            image_url=image_url,
            prompt=enhancement_prompt,
            strength=0.7
        )
        
        # Verify result
        assert "https://example.com/enhanced-image.jpg" in result
        
        # Verify memory was updated
        image_generation_agent.memory_mcp.store_memory.assert_called_once()
        args, kwargs = image_generation_agent.memory_mcp.store_memory.call_args
        assert args[0].startswith("enhanced_image_")
        assert "https://example.com/enhanced-image.jpg" in args[1]
        assert kwargs["namespace"] == "image_generation"

    @pytest.mark.asyncio
    async def test_enhance_existing_image_error(self, image_generation_agent):
        """Test handling errors when enhancing an image."""
        # Setup mock to raise exception
        image_generation_agent.everart_mcp.enhance_image.side_effect = Exception("Enhancement failed")
        
        # Call the method and expect error
        with pytest.raises(ImageGenerationError, match="Failed to enhance image"):
            await image_generation_agent.enhance_existing_image(
                "https://example.com/image.jpg",
                "Make it better"
            )
        
        # Verify memory was not updated
        image_generation_agent.memory_mcp.store_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_describe_image(self, image_generation_agent):
        """Test describing an image."""
        # Setup mock response
        image_generation_agent.everart_mcp.describe_image.return_value = """
        The image shows a mountain landscape with a clear blue lake in the foreground.
        Snow-capped peaks are visible in the background, and there are pine trees along the shore.
        The sky is clear with a few scattered clouds.
        """
        
        # Call the method
        image_url = "https://example.com/landscape.jpg"
        
        result = await image_generation_agent.describe_image(
            image_url,
            detail_level="high"
        )
        
        # Verify EverArt MCP was called correctly
        image_generation_agent.everart_mcp.describe_image.assert_called_once_with(
            image_url=image_url,
            detail_level="high"
        )
        
        # Verify result
        assert "mountain landscape" in result
        assert "blue lake" in result
        
        # Verify memory was updated
        image_generation_agent.memory_mcp.store_memory.assert_called_once()
        args, kwargs = image_generation_agent.memory_mcp.store_memory.call_args
        assert args[0].startswith("image_description_")
        assert "mountain landscape" in args[1]
        assert kwargs["namespace"] == "image_generation"

    @pytest.mark.asyncio
    async def test_describe_image_error(self, image_generation_agent):
        """Test handling errors when describing an image."""
        # Setup mock to raise exception
        image_generation_agent.everart_mcp.describe_image.side_effect = Exception("Description failed")
        
        # Call the method and expect error
        with pytest.raises(ImageGenerationError, match="Failed to describe image"):
            await image_generation_agent.describe_image("https://example.com/image.jpg")
        
        # Verify memory was not updated
        image_generation_agent.memory_mcp.store_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_diagram(self, image_generation_agent):
        """Test generating a diagram from structured data."""
        # Setup mock response
        image_generation_agent.everart_mcp.generate_image.return_value = """
        Generated diagram URL: https://example.com/diagram.jpg
        """
        
        # Call the method
        data = {
            "title": "Climate Change Effects",
            "nodes": [
                "Rising Temperatures",
                "Melting Ice Caps",
                "Rising Sea Levels",
                "Coastal Flooding"
            ],
            "connections": [
                ["Rising Temperatures", "Melting Ice Caps"],
                ["Melting Ice Caps", "Rising Sea Levels"],
                ["Rising Sea Levels", "Coastal Flooding"]
            ]
        }
        
        result = await image_generation_agent.generate_diagram(data)
        
        # Verify EverArt MCP was called with appropriate prompt
        image_generation_agent.everart_mcp.generate_image.assert_called_once()
        args, kwargs = image_generation_agent.everart_mcp.generate_image.call_args
        
        # The prompt should describe the diagram structure
        prompt = kwargs["prompt"]
        assert "diagram" in prompt.lower()
        assert "climate change" in prompt.lower()
        assert "rising temperatures" in prompt.lower()
        assert "connections" in prompt.lower() or "arrows" in prompt.lower()
        
        # Verify style is set to "diagram" or similar
        assert kwargs["style"] in ["diagram", "infographic", "technical illustration"]
        
        # Verify result
        assert "https://example.com/diagram.jpg" in result
        
        # Verify memory was updated
        image_generation_agent.memory_mcp.store_memory.assert_called_once()
        args, kwargs = image_generation_agent.memory_mcp.store_memory.call_args
        assert args[0].startswith("diagram_")
        assert "https://example.com/diagram.jpg" in args[1]
        assert kwargs["namespace"] == "image_generation"

    @pytest.mark.asyncio
    async def test_generate_diagram_error(self, image_generation_agent):
        """Test handling errors when generating a diagram."""
        # Setup mock to raise exception
        image_generation_agent.everart_mcp.generate_image.side_effect = Exception("Diagram generation failed")
        
        # Call the method and expect error
        with pytest.raises(ImageGenerationError, match="Failed to generate diagram"):
            await image_generation_agent.generate_diagram({
                "title": "Test Diagram",
                "nodes": ["A", "B"],
                "connections": [["A", "B"]]
            })
        
        # Verify memory was not updated
        image_generation_agent.memory_mcp.store_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_comparison_image(self, image_generation_agent):
        """Test generating a comparison image."""
        # Setup mock response
        image_generation_agent.everart_mcp.generate_image.return_value = """
        Generated comparison image URL: https://example.com/comparison.jpg
        """
        
        # Call the method
        items = ["Solar Energy", "Wind Energy"]
        attributes = ["Cost", "Efficiency", "Environmental Impact"]
        comparisons = {
            "Solar Energy": {
                "Cost": "High initial investment",
                "Efficiency": "Varies with sunlight",
                "Environmental Impact": "Very low"
            },
            "Wind Energy": {
                "Cost": "Medium initial investment",
                "Efficiency": "Varies with wind conditions",
                "Environmental Impact": "Low"
            }
        }
        
        result = await image_generation_agent.generate_comparison_image(
            items, attributes, comparisons, title="Renewable Energy Comparison"
        )
        
        # Verify EverArt MCP was called with appropriate prompt
        image_generation_agent.everart_mcp.generate_image.assert_called_once()
        args, kwargs = image_generation_agent.everart_mcp.generate_image.call_args
        
        # The prompt should describe the comparison structure
        prompt = kwargs["prompt"]
        assert "comparison" in prompt.lower()
        assert "renewable energy" in prompt.lower()
        assert "solar energy" in prompt.lower()
        assert "wind energy" in prompt.lower()
        
        # Verify style is appropriate for comparison
        assert kwargs["style"] in ["infographic", "comparison chart", "educational"]
        
        # Verify result
        assert "https://example.com/comparison.jpg" in result
        
        # Verify memory was updated
        image_generation_agent.memory_mcp.store_memory.assert_called_once()
        args, kwargs = image_generation_agent.memory_mcp.store_memory.call_args
        assert args[0].startswith("comparison_")
        assert "https://example.com/comparison.jpg" in args[1]
        assert kwargs["namespace"] == "image_generation"

    @pytest.mark.asyncio
    async def test_generate_timeline_image(self, image_generation_agent):
        """Test generating a timeline image."""
        # Setup mock response
        image_generation_agent.everart_mcp.generate_image.return_value = """
        Generated timeline image URL: https://example.com/timeline.jpg
        """
        
        # Call the method
        events = [
            {"year": 1990, "event": "IPCC First Assessment Report"},
            {"year": 1997, "event": "Kyoto Protocol"},
            {"year": 2015, "event": "Paris Agreement"},
            {"year": 2021, "event": "COP26 in Glasgow"}
        ]
        
        result = await image_generation_agent.generate_timeline_image(
            events, title="Climate Change Policy Timeline"
        )
        
        # Verify EverArt MCP was called with appropriate prompt
        image_generation_agent.everart_mcp.generate_image.assert_called_once()
        args, kwargs = image_generation_agent.everart_mcp.generate_image.call_args
        
        # The prompt should describe the timeline structure
        prompt = kwargs["prompt"]
        assert "timeline" in prompt.lower()
        assert "climate change" in prompt.lower()
        assert "1990" in prompt
        assert "paris agreement" in prompt.lower()
        
        # Verify aspect ratio is appropriate for timeline (typically horizontal)
        assert kwargs["aspect_ratio"] in ["16:9", "3:2", "2:1"]
        
        # Verify result
        assert "https://example.com/timeline.jpg" in result
        
        # Verify memory was updated
        image_generation_agent.memory_mcp.store_memory.assert_called_once()
        args, kwargs = image_generation_agent.memory_mcp.store_memory.call_args
        assert args[0].startswith("timeline_")
        assert "https://example.com/timeline.jpg" in args[1]
        assert kwargs["namespace"] == "image_generation"

    @pytest.mark.asyncio
    async def test_generate_concept_visualization(self, image_generation_agent):
        """Test generating a concept visualization."""
        # Setup mock response
        image_generation_agent.everart_mcp.generate_image.return_value = """
        Generated concept image URL: https://example.com/concept.jpg
        """
        
        # Call the method
        concept = "Quantum Computing"
        description = "Visualization of quantum bits (qubits) in superposition state"
        
        result = await image_generation_agent.generate_concept_visualization(
            concept, description
        )
        
        # Verify EverArt MCP was called with appropriate prompt
        image_generation_agent.everart_mcp.generate_image.assert_called_once()
        args, kwargs = image_generation_agent.everart_mcp.generate_image.call_args
        
        # The prompt should describe the concept
        prompt = kwargs["prompt"]
        assert "quantum computing" in prompt.lower()
        assert "qubits" in prompt.lower()
        assert "superposition" in prompt.lower()
        
        # Verify style is appropriate for concept visualization
        assert kwargs["style"] in ["conceptual", "scientific", "educational", "digital art"]
        
        # Verify result
        assert "https://example.com/concept.jpg" in result
        
        # Verify memory was updated
        image_generation_agent.memory_mcp.store_memory.assert_called_once()
        args, kwargs = image_generation_agent.memory_mcp.store_memory.call_args
        assert args[0].startswith("concept_")
        assert "https://example.com/concept.jpg" in args[1]
        assert kwargs["namespace"] == "image_generation"

    @pytest.mark.asyncio
    async def test_get_stored_image(self, image_generation_agent):
        """Test retrieving a stored image by query."""
        # Setup mock response
        image_generation_agent.memory_mcp.retrieve_memory.return_value = """
        https://example.com/stored-image.jpg
        Description: A visualization of climate change impacts
        """
        
        # Call the method
        query = "climate change"
        
        url, description = await image_generation_agent.get_stored_image(query)
        
        # Verify memory was queried
        image_generation_agent.memory_mcp.retrieve_memory.assert_called_once_with(
            "image_climate change", namespace="image_generation"
        )
        
        # Verify correct values were extracted
        assert url == "https://example.com/stored-image.jpg"
        assert description == "A visualization of climate change impacts"

    @pytest.mark.asyncio
    async def test_get_stored_image_not_found(self, image_generation_agent):
        """Test retrieving a stored image that doesn't exist."""
        # Setup mock to raise exception
        image_generation_agent.memory_mcp.retrieve_memory.side_effect = Exception("Memory not found")
        
        # Call the method
        url, description = await image_generation_agent.get_stored_image("nonexistent query")
        
        # Verify result is None for both values
        assert url is None
        assert description is None

    @pytest.mark.asyncio
    async def test_get_stored_image_invalid_format(self, image_generation_agent):
        """Test retrieving a stored image with invalid format."""
        # Setup mock with invalid format
        image_generation_agent.memory_mcp.retrieve_memory.return_value = "Invalid format without URL"
        
        # Call the method
        url, description = await image_generation_agent.get_stored_image("test query")
        
        # Verify result is None for both values
        assert url is None
        assert description is None

    @pytest.mark.asyncio
    async def test_list_stored_images(self, image_generation_agent):
        """Test listing all stored images."""
        # Setup mock response
        image_generation_agent.memory_mcp.list_memories.return_value = """
        image_climate_change
        image_ai_impact
        image_renewable_energy
        """
        
        # Call the method
        images = await image_generation_agent.list_stored_images()
        
        # Verify memory was queried
        image_generation_agent.memory_mcp.list_memories.assert_called_once_with(
            namespace="image_generation"
        )
        
        # Verify images were filtered and processed correctly
        assert "climate_change" in images
        assert "ai_impact" in images
        assert "renewable_energy" in images

    @pytest.mark.asyncio
    async def test_list_stored_images_empty(self, image_generation_agent):
        """Test listing stored images when none exist."""
        # Setup mock with empty response
        image_generation_agent.memory_mcp.list_memories.return_value = ""
        
        # Call the method
        images = await image_generation_agent.list_stored_images()
        
        # Verify result is an empty list
        assert images == []

    @pytest.mark.asyncio
    async def test_delete_stored_image(self, image_generation_agent):
        """Test deleting a stored image."""
        # Setup mock response
        image_generation_agent.memory_mcp.retrieve_memory.return_value = "https://example.com/image.jpg"
        
        # Call the method
        success = await image_generation_agent.delete_stored_image("test query")
        
        # Verify memory was queried and deleted
        image_generation_agent.memory_mcp.retrieve_memory.assert_called_once_with(
            "image_test query", namespace="image_generation"
        )
        image_generation_agent.memory_mcp.delete_memory.assert_called_once_with(
            "image_test query", namespace="image_generation"
        )
        
        # Verify result
        assert success is True

    @pytest.mark.asyncio
    async def test_delete_stored_image_not_found(self, image_generation_agent):
        """Test deleting a stored image that doesn't exist."""
        # Setup mock to raise exception
        image_generation_agent.memory_mcp.retrieve_memory.side_effect = Exception("Memory not found")
        
        # Call the method
        success = await image_generation_agent.delete_stored_image("nonexistent query")
        
        # Verify delete was not called
        image_generation_agent.memory_mcp.delete_memory.assert_not_called()
        
        # Verify result
        assert success is False

    @pytest.mark.asyncio
    async def test_generate_image_variations(self, image_generation_agent):
        """Test generating variations of an image."""
        # Setup mock responses
        image_generation_agent.everart_mcp.generate_image.return_value = """
        Generated image URLs:
        https://example.com/variation1.jpg
        https://example.com/variation2.jpg
        https://example.com/variation3.jpg
        """
        
        # Call the method
        prompt = "A futuristic city with flying cars"
        
        results = await image_generation_agent.generate_image_variations(
            prompt, num_variations=3
        )
        
        # Verify EverArt MCP was called correctly
        image_generation_agent.everart_mcp.generate_image.assert_called_once_with(
            prompt=prompt,
            style=None,
            aspect_ratio="16:9",
            num_images=3
        )
        
        # Verify results contain all variations
        assert len(results) == 3
        assert "https://example.com/variation1.jpg" in results
        assert "https://example.com/variation2.jpg" in results
        assert "https://example.com/variation3.jpg" in results
        
        # Verify memory was updated
        assert image_generation_agent.memory_mcp.store_memory.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_image_with_style_transfer(self, image_generation_agent):
        """Test generating an image with a specific artistic style."""
        # Setup mock response
        image_generation_agent.everart_mcp.generate_image.return_value = """
        Generated styled image URL: https://example.com/styled-image.jpg
        """
        
        # Call the method
        content = "A cityscape of New York"
        style = "Van Gogh's Starry Night"
        
        result = await image_generation_agent.generate_image_with_style_transfer(
            content, style
        )
        
        # Verify EverArt MCP was called with appropriate prompt
        image_generation_agent.everart_mcp.generate_image.assert_called_once()
        args, kwargs = image_generation_agent.everart_mcp.generate_image.call_args
        
        # The prompt should describe both content and style
        prompt = kwargs["prompt"]
        assert "cityscape" in prompt.lower()
        assert "new york" in prompt.lower()
        assert "van gogh" in prompt.lower() or "starry night" in prompt.lower()
        
        # Verify result
        assert "https://example.com/styled-image.jpg" in result
        
        # Verify memory was updated
        image_generation_agent.memory_mcp.store_memory.assert_called_once()
        args, kwargs = image_generation_agent.memory_mcp.store_memory.call_args
        assert args[0].startswith("styled_image_")
        assert "https://example.com/styled-image.jpg" in args[1]
        assert kwargs["namespace"] == "image_generation"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
