"""
Unit tests for the File Manager Agent.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from apps.agents.file_manager_agent import FileManagerAgent
from apps.mcps.filesystem_mcp import FilesystemMCP
from apps.mcps.memory_mcp import MemoryMCP
from apps.utils.exceptions import FileOperationError

class TestFileManagerAgent:
    """Test suite for FileManagerAgent class."""

    @pytest.fixture
    def filesystem_mcp(self):
        """Fixture to create a mock FilesystemMCP."""
        mock_fs = MagicMock(spec=FilesystemMCP)
        mock_fs.read_file = MagicMock()
        mock_fs.write_file = MagicMock()
        mock_fs.list_directory = MagicMock()
        mock_fs.create_directory = MagicMock()
        mock_fs.delete_file = MagicMock()
        mock_fs.file_exists = MagicMock()
        mock_fs.search_files = MagicMock()
        return mock_fs

    @pytest.fixture
    def memory_mcp(self):
        """Fixture to create a mock MemoryMCP."""
        mock_memory = MagicMock(spec=MemoryMCP)
        mock_memory.store_memory = MagicMock()
        mock_memory.retrieve_memory = MagicMock()
        mock_memory.list_memories = MagicMock()
        return mock_memory

    @pytest.fixture
    def file_manager_agent(self, filesystem_mcp, memory_mcp):
        """Fixture to create a FileManagerAgent instance with mock dependencies."""
        agent = FileManagerAgent(
            name="file_manager",
            filesystem_mcp=filesystem_mcp,
            memory_mcp=memory_mcp,
            workspace_dir="/test/workspace"
        )
        return agent

    def test_init(self, filesystem_mcp, memory_mcp):
        """Test initialization of FileManagerAgent."""
        agent = FileManagerAgent(
            name="file_manager",
            filesystem_mcp=filesystem_mcp,
            memory_mcp=memory_mcp,
            workspace_dir="/test/workspace"
        )
        
        assert agent.name == "file_manager"
        assert agent.filesystem_mcp == filesystem_mcp
        assert agent.memory_mcp == memory_mcp
        assert agent.workspace_dir == "/test/workspace"
        assert agent.research_dir == "/test/workspace/research"
        assert agent.images_dir == "/test/workspace/images"
        assert agent.summaries_dir == "/test/workspace/summaries"

    def test_init_with_defaults(self, filesystem_mcp, memory_mcp):
        """Test initialization with default values."""
        agent = FileManagerAgent(
            name="file_manager",
            filesystem_mcp=filesystem_mcp,
            memory_mcp=memory_mcp
        )
        
        assert agent.name == "file_manager"
        assert agent.filesystem_mcp == filesystem_mcp
        assert agent.memory_mcp == memory_mcp
        assert agent.workspace_dir == os.getcwd()
        assert agent.research_dir == os.path.join(os.getcwd(), "research")
        assert agent.images_dir == os.path.join(os.getcwd(), "images")
        assert agent.summaries_dir == os.path.join(os.getcwd(), "summaries")

    @pytest.mark.asyncio
    async def test_initialize_workspace(self, file_manager_agent):
        """Test initializing the workspace directories."""
        # Setup mock responses
        file_manager_agent.filesystem_mcp.file_exists.return_value = False
        
        # Call the method
        await file_manager_agent.initialize_workspace()
        
        # Verify directories were created
        file_manager_agent.filesystem_mcp.create_directory.assert_any_call(file_manager_agent.research_dir)
        file_manager_agent.filesystem_mcp.create_directory.assert_any_call(file_manager_agent.images_dir)
        file_manager_agent.filesystem_mcp.create_directory.assert_any_call(file_manager_agent.summaries_dir)
        
        # Verify memory was updated
        file_manager_agent.memory_mcp.store_memory.assert_called_with(
            "workspace_initialized", "true", namespace="file_manager"
        )

    @pytest.mark.asyncio
    async def test_initialize_workspace_already_exists(self, file_manager_agent):
        """Test initializing when workspace already exists."""
        # Setup mock responses
        file_manager_agent.filesystem_mcp.file_exists.return_value = True
        
        # Call the method
        await file_manager_agent.initialize_workspace()
        
        # Verify no directories were created
        file_manager_agent.filesystem_mcp.create_directory.assert_not_called()
        
        # Verify memory was updated
        file_manager_agent.memory_mcp.store_memory.assert_called_with(
            "workspace_initialized", "true", namespace="file_manager"
        )

    @pytest.mark.asyncio
    async def test_initialize_workspace_error(self, file_manager_agent):
        """Test handling errors during workspace initialization."""
        # Setup mock to raise exception
        file_manager_agent.filesystem_mcp.create_directory.side_effect = Exception("Directory creation failed")
        
        # Call the method and expect error
        with pytest.raises(FileOperationError, match="Failed to initialize workspace"):
            await file_manager_agent.initialize_workspace()
        
        # Verify memory was not updated
        file_manager_agent.memory_mcp.store_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_research_content(self, file_manager_agent):
        """Test saving research content to a file."""
        # Setup test data
        query = "test query"
        content = "This is test research content"
        
        # Call the method
        filename = await file_manager_agent.save_research_content(query, content)
        
        # Verify file was written
        expected_path = os.path.join(file_manager_agent.research_dir, f"{filename}.txt")
        file_manager_agent.filesystem_mcp.write_file.assert_called_once_with(
            expected_path, content
        )
        
        # Verify memory was updated
        file_manager_agent.memory_mcp.store_memory.assert_called_with(
            f"research_{query}", filename, namespace="file_manager"
        )

    @pytest.mark.asyncio
    async def test_save_research_content_error(self, file_manager_agent):
        """Test handling errors when saving research content."""
        # Setup mock to raise exception
        file_manager_agent.filesystem_mcp.write_file.side_effect = Exception("Write failed")
        
        # Call the method and expect error
        with pytest.raises(FileOperationError, match="Failed to save research content"):
            await file_manager_agent.save_research_content("test query", "content")
        
        # Verify memory was not updated
        file_manager_agent.memory_mcp.store_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_summary(self, file_manager_agent):
        """Test saving a summary to a file."""
        # Setup test data
        query = "test query"
        summary = "This is a test summary"
        
        # Call the method
        filename = await file_manager_agent.save_summary(query, summary)
        
        # Verify file was written
        expected_path = os.path.join(file_manager_agent.summaries_dir, f"{filename}.txt")
        file_manager_agent.filesystem_mcp.write_file.assert_called_once_with(
            expected_path, summary
        )
        
        # Verify memory was updated
        file_manager_agent.memory_mcp.store_memory.assert_called_with(
            f"summary_{query}", filename, namespace="file_manager"
        )

    @pytest.mark.asyncio
    async def test_save_summary_error(self, file_manager_agent):
        """Test handling errors when saving a summary."""
        # Setup mock to raise exception
        file_manager_agent.filesystem_mcp.write_file.side_effect = Exception("Write failed")
        
        # Call the method and expect error
        with pytest.raises(FileOperationError, match="Failed to save summary"):
            await file_manager_agent.save_summary("test query", "summary")
        
        # Verify memory was not updated
        file_manager_agent.memory_mcp.store_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_image(self, file_manager_agent):
        """Test saving an image URL to a file."""
        # Setup test data
        query = "test query"
        image_url = "https://example.com/image.jpg"
        description = "Test image description"
        
        # Call the method
        filename = await file_manager_agent.save_image(query, image_url, description)
        
        # Verify file was written
        expected_path = os.path.join(file_manager_agent.images_dir, f"{filename}.txt")
        expected_content = f"URL: {image_url}\nDescription: {description}"
        file_manager_agent.filesystem_mcp.write_file.assert_called_once_with(
            expected_path, expected_content
        )
        
        # Verify memory was updated
        file_manager_agent.memory_mcp.store_memory.assert_called_with(
            f"image_{query}", filename, namespace="file_manager"
        )

    @pytest.mark.asyncio
    async def test_save_image_error(self, file_manager_agent):
        """Test handling errors when saving an image."""
        # Setup mock to raise exception
        file_manager_agent.filesystem_mcp.write_file.side_effect = Exception("Write failed")
        
        # Call the method and expect error
        with pytest.raises(FileOperationError, match="Failed to save image"):
            await file_manager_agent.save_image("test query", "https://example.com/image.jpg", "description")
        
        # Verify memory was not updated
        file_manager_agent.memory_mcp.store_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_research_content(self, file_manager_agent):
        """Test retrieving research content."""
        # Setup mock responses
        file_manager_agent.memory_mcp.retrieve_memory.return_value = "research_file_123"
        file_manager_agent.filesystem_mcp.read_file.return_value = "Research content"
        
        # Call the method
        content = await file_manager_agent.get_research_content("test query")
        
        # Verify memory was queried
        file_manager_agent.memory_mcp.retrieve_memory.assert_called_once_with(
            "research_test query", namespace="file_manager"
        )
        
        # Verify file was read
        expected_path = os.path.join(file_manager_agent.research_dir, "research_file_123.txt")
        file_manager_agent.filesystem_mcp.read_file.assert_called_once_with(expected_path)
        
        # Verify correct content was returned
        assert content == "Research content"

    @pytest.mark.asyncio
    async def test_get_research_content_not_found(self, file_manager_agent):
        """Test retrieving research content when not found."""
        # Setup mock to raise exception for memory retrieval
        file_manager_agent.memory_mcp.retrieve_memory.side_effect = Exception("Memory not found")
        
        # Call the method
        content = await file_manager_agent.get_research_content("test query")
        
        # Verify result is None
        assert content is None
        
        # Verify file was not read
        file_manager_agent.filesystem_mcp.read_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_research_content_file_error(self, file_manager_agent):
        """Test handling file errors when retrieving research content."""
        # Setup mock responses
        file_manager_agent.memory_mcp.retrieve_memory.return_value = "research_file_123"
        file_manager_agent.filesystem_mcp.read_file.side_effect = Exception("File read error")
        
        # Call the method
        content = await file_manager_agent.get_research_content("test query")
        
        # Verify result is None
        assert content is None

    @pytest.mark.asyncio
    async def test_get_summary(self, file_manager_agent):
        """Test retrieving a summary."""
        # Setup mock responses
        file_manager_agent.memory_mcp.retrieve_memory.return_value = "summary_file_123"
        file_manager_agent.filesystem_mcp.read_file.return_value = "Summary content"
        
        # Call the method
        content = await file_manager_agent.get_summary("test query")
        
        # Verify memory was queried
        file_manager_agent.memory_mcp.retrieve_memory.assert_called_once_with(
            "summary_test query", namespace="file_manager"
        )
        
        # Verify file was read
        expected_path = os.path.join(file_manager_agent.summaries_dir, "summary_file_123.txt")
        file_manager_agent.filesystem_mcp.read_file.assert_called_once_with(expected_path)
        
        # Verify correct content was returned
        assert content == "Summary content"

    @pytest.mark.asyncio
    async def test_get_image_info(self, file_manager_agent):
        """Test retrieving image information."""
        # Setup mock responses
        file_manager_agent.memory_mcp.retrieve_memory.return_value = "image_file_123"
        file_manager_agent.filesystem_mcp.read_file.return_value = "URL: https://example.com/image.jpg\nDescription: Test image"
        
        # Call the method
        url, description = await file_manager_agent.get_image_info("test query")
        
        # Verify memory was queried
        file_manager_agent.memory_mcp.retrieve_memory.assert_called_once_with(
            "image_test query", namespace="file_manager"
        )
        
        # Verify file was read
        expected_path = os.path.join(file_manager_agent.images_dir, "image_file_123.txt")
        file_manager_agent.filesystem_mcp.read_file.assert_called_once_with(expected_path)
        
        # Verify correct information was returned
        assert url == "https://example.com/image.jpg"
        assert description == "Test image"

    @pytest.mark.asyncio
    async def test_get_image_info_invalid_format(self, file_manager_agent):
        """Test retrieving image information with invalid file format."""
        # Setup mock responses
        file_manager_agent.memory_mcp.retrieve_memory.return_value = "image_file_123"
        file_manager_agent.filesystem_mcp.read_file.return_value = "Invalid format"
        
        # Call the method
        url, description = await file_manager_agent.get_image_info("test query")
        
        # Verify result is None for both values
        assert url is None
        assert description is None
    @pytest.mark.asyncio
    async def test_list_research_files(self, file_manager_agent):
        """Test listing all research files."""
        # Setup mock response
        file_manager_agent.filesystem_mcp.list_directory.return_value = """
        research_file_1.txt
        research_file_2.txt
        research_file_3.txt
        """
        
        # Call the method
        files = await file_manager_agent.list_research_files()
        
        # Verify directory was listed
        file_manager_agent.filesystem_mcp.list_directory.assert_called_once_with(
            file_manager_agent.research_dir
        )
        
        # Verify correct files were returned
        assert "research_file_1.txt" in files
        assert "research_file_2.txt" in files
        assert "research_file_3.txt" in files

    @pytest.mark.asyncio
    async def test_list_research_files_error(self, file_manager_agent):
        """Test handling errors when listing research files."""
        # Setup mock to raise exception
        file_manager_agent.filesystem_mcp.list_directory.side_effect = Exception("Directory listing failed")
        
        # Call the method and expect error
        with pytest.raises(FileOperationError, match="Failed to list research files"):
            await file_manager_agent.list_research_files()

    @pytest.mark.asyncio
    async def test_list_summary_files(self, file_manager_agent):
        """Test listing all summary files."""
        # Setup mock response
        file_manager_agent.filesystem_mcp.list_directory.return_value = """
        summary_file_1.txt
        summary_file_2.txt
        """
        
        # Call the method
        files = await file_manager_agent.list_summary_files()
        
        # Verify directory was listed
        file_manager_agent.filesystem_mcp.list_directory.assert_called_once_with(
            file_manager_agent.summaries_dir
        )
        
        # Verify correct files were returned
        assert "summary_file_1.txt" in files
        assert "summary_file_2.txt" in files

    @pytest.mark.asyncio
    async def test_list_image_files(self, file_manager_agent):
        """Test listing all image files."""
        # Setup mock response
        file_manager_agent.filesystem_mcp.list_directory.return_value = """
        image_file_1.txt
        image_file_2.txt
        """
        
        # Call the method
        files = await file_manager_agent.list_image_files()
        
        # Verify directory was listed
        file_manager_agent.filesystem_mcp.list_directory.assert_called_once_with(
            file_manager_agent.images_dir
        )
        
        # Verify correct files were returned
        assert "image_file_1.txt" in files
        assert "image_file_2.txt" in files

    @pytest.mark.asyncio
    async def test_search_files(self, file_manager_agent):
        """Test searching for files by pattern."""
        # Setup mock response
        file_manager_agent.filesystem_mcp.search_files.return_value = """
        research/climate_change_123.txt
        summaries/climate_summary_456.txt
        """
        
        # Call the method
        results = await file_manager_agent.search_files("climate")
        
        # Verify search was performed
        file_manager_agent.filesystem_mcp.search_files.assert_called_once_with(
            pattern="*climate*", path=file_manager_agent.workspace_dir, recursive=True
        )
        
        # Verify correct results were returned
        assert "research/climate_change_123.txt" in results
        assert "summaries/climate_summary_456.txt" in results

    @pytest.mark.asyncio
    async def test_search_files_error(self, file_manager_agent):
        """Test handling errors when searching for files."""
        # Setup mock to raise exception
        file_manager_agent.filesystem_mcp.search_files.side_effect = Exception("Search failed")
        
        # Call the method and expect error
        with pytest.raises(FileOperationError, match="Failed to search files"):
            await file_manager_agent.search_files("climate")

    @pytest.mark.asyncio
    async def test_delete_research_file(self, file_manager_agent):
        """Test deleting a research file."""
        # Setup mock responses
        file_manager_agent.memory_mcp.retrieve_memory.return_value = "research_file_123"
        
        # Call the method
        success = await file_manager_agent.delete_research_file("test query")
        
        # Verify memory was queried
        file_manager_agent.memory_mcp.retrieve_memory.assert_called_once_with(
            "research_test query", namespace="file_manager"
        )
        
        # Verify file was deleted
        expected_path = os.path.join(file_manager_agent.research_dir, "research_file_123.txt")
        file_manager_agent.filesystem_mcp.delete_file.assert_called_once_with(expected_path)
        
        # Verify result
        assert success is True

    @pytest.mark.asyncio
    async def test_delete_research_file_not_found(self, file_manager_agent):
        """Test deleting a research file that doesn't exist."""
        # Setup mock to raise exception for memory retrieval
        file_manager_agent.memory_mcp.retrieve_memory.side_effect = Exception("Memory not found")
        
        # Call the method
        success = await file_manager_agent.delete_research_file("test query")
        
        # Verify result is False
        assert success is False
        
        # Verify file was not deleted
        file_manager_agent.filesystem_mcp.delete_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_research_file_error(self, file_manager_agent):
        """Test handling errors when deleting a research file."""
        # Setup mock responses
        file_manager_agent.memory_mcp.retrieve_memory.return_value = "research_file_123"
        file_manager_agent.filesystem_mcp.delete_file.side_effect = Exception("Delete failed")
        
        # Call the method
        success = await file_manager_agent.delete_research_file("test query")
        
        # Verify result is False
        assert success is False

    @pytest.mark.asyncio
    async def test_delete_summary_file(self, file_manager_agent):
        """Test deleting a summary file."""
        # Setup mock responses
        file_manager_agent.memory_mcp.retrieve_memory.return_value = "summary_file_123"
        
        # Call the method
        success = await file_manager_agent.delete_summary_file("test query")
        
        # Verify memory was queried
        file_manager_agent.memory_mcp.retrieve_memory.assert_called_once_with(
            "summary_test query", namespace="file_manager"
        )
        
        # Verify file was deleted
        expected_path = os.path.join(file_manager_agent.summaries_dir, "summary_file_123.txt")
        file_manager_agent.filesystem_mcp.delete_file.assert_called_once_with(expected_path)
        
        # Verify result
        assert success is True

    @pytest.mark.asyncio
    async def test_delete_image_file(self, file_manager_agent):
        """Test deleting an image file."""
        # Setup mock responses
        file_manager_agent.memory_mcp.retrieve_memory.return_value = "image_file_123"
        
        # Call the method
        success = await file_manager_agent.delete_image_file("test query")
        
        # Verify memory was queried
        file_manager_agent.memory_mcp.retrieve_memory.assert_called_once_with(
            "image_test query", namespace="file_manager"
        )
        
        # Verify file was deleted
        expected_path = os.path.join(file_manager_agent.images_dir, "image_file_123.txt")
        file_manager_agent.filesystem_mcp.delete_file.assert_called_once_with(expected_path)
        
        # Verify result
        assert success is True

    @pytest.mark.asyncio
    async def test_create_research_package(self, file_manager_agent):
        """Test creating a research package with all related files."""
        # Setup mock responses
        file_manager_agent.memory_mcp.retrieve_memory.side_effect = [
            "research_file_123",  # For research content
            "summary_file_456",   # For summary
            "image_file_789"      # For image
        ]
        file_manager_agent.filesystem_mcp.read_file.side_effect = [
            "Research content",
            "Summary content",
            "URL: https://example.com/image.jpg\nDescription: Test image"
        ]
        
        # Call the method
        package = await file_manager_agent.create_research_package("test query")
        
        # Verify memory was queried for each file type
        file_manager_agent.memory_mcp.retrieve_memory.assert_any_call(
            "research_test query", namespace="file_manager"
        )
        file_manager_agent.memory_mcp.retrieve_memory.assert_any_call(
            "summary_test query", namespace="file_manager"
        )
        file_manager_agent.memory_mcp.retrieve_memory.assert_any_call(
            "image_test query", namespace="file_manager"
        )
        
        # Verify files were read
        file_manager_agent.filesystem_mcp.read_file.assert_any_call(
            os.path.join(file_manager_agent.research_dir, "research_file_123.txt")
        )
        file_manager_agent.filesystem_mcp.read_file.assert_any_call(
            os.path.join(file_manager_agent.summaries_dir, "summary_file_456.txt")
        )
        file_manager_agent.filesystem_mcp.read_file.assert_any_call(
            os.path.join(file_manager_agent.images_dir, "image_file_789.txt")
        )
        
        # Verify package structure
        assert package["query"] == "test query"
        assert package["research_content"] == "Research content"
        assert package["summary"] == "Summary content"
        assert package["image_url"] == "https://example.com/image.jpg"
        assert package["image_description"] == "Test image"

    @pytest.mark.asyncio
    async def test_create_research_package_partial(self, file_manager_agent):
        """Test creating a research package with only some files available."""
        # Setup mock responses
        # Research content available
        file_manager_agent.memory_mcp.retrieve_memory.side_effect = [
            "research_file_123",  # For research content
            Exception("Memory not found"),  # For summary
            Exception("Memory not found")   # For image
        ]
        file_manager_agent.filesystem_mcp.read_file.return_value = "Research content"
        
        # Call the method
        package = await file_manager_agent.create_research_package("test query")
        
        # Verify memory was queried for each file type
        file_manager_agent.memory_mcp.retrieve_memory.assert_any_call(
            "research_test query", namespace="file_manager"
        )
        file_manager_agent.memory_mcp.retrieve_memory.assert_any_call(
            "summary_test query", namespace="file_manager"
        )
        file_manager_agent.memory_mcp.retrieve_memory.assert_any_call(
            "image_test query", namespace="file_manager"
        )
        
        # Verify only research file was read
        file_manager_agent.filesystem_mcp.read_file.assert_called_once_with(
            os.path.join(file_manager_agent.research_dir, "research_file_123.txt")
        )
        
        # Verify package structure
        assert package["query"] == "test query"
        assert package["research_content"] == "Research content"
        assert package["summary"] is None
        assert package["image_url"] is None
        assert package["image_description"] is None

    @pytest.mark.asyncio
    async def test_create_research_package_empty(self, file_manager_agent):
        """Test creating a research package when no files are available."""
        # Setup mock responses - all retrievals fail
        file_manager_agent.memory_mcp.retrieve_memory.side_effect = Exception("Memory not found")
        
        # Call the method
        package = await file_manager_agent.create_research_package("test query")
        
        # Verify memory was queried
        assert file_manager_agent.memory_mcp.retrieve_memory.call_count == 3
        
        # Verify no files were read
        file_manager_agent.filesystem_mcp.read_file.assert_not_called()
        
        # Verify package structure
        assert package["query"] == "test query"
        assert package["research_content"] is None
        assert package["summary"] is None
        assert package["image_url"] is None
        assert package["image_description"] is None

    @pytest.mark.asyncio
    async def test_export_research_to_markdown(self, file_manager_agent):
        """Test exporting research package to markdown format."""
        # Setup test data
        package = {
            "query": "climate change",
            "research_content": "Research about climate change...",
            "summary": "Climate change is a significant issue...",
            "image_url": "https://example.com/climate.jpg",
            "image_description": "Graph showing temperature rise"
        }
        
        # Call the method
        markdown = await file_manager_agent.export_research_to_markdown(package)
        
        # Verify markdown structure
        assert "# Research: climate change" in markdown
        assert "## Summary" in markdown
        assert "Climate change is a significant issue..." in markdown
        assert "## Research Content" in markdown
        assert "Research about climate change..." in markdown
        assert "## Images" in markdown
        assert "![Graph showing temperature rise](https://example.com/climate.jpg)" in markdown

    @pytest.mark.asyncio
    async def test_export_research_to_markdown_partial(self, file_manager_agent):
        """Test exporting partial research package to markdown."""
        # Setup test data with missing elements
        package = {
            "query": "climate change",
            "research_content": "Research about climate change...",
            "summary": None,
            "image_url": None,
            "image_description": None
        }
        
        # Call the method
        markdown = await file_manager_agent.export_research_to_markdown(package)
        
        # Verify markdown structure
        assert "# Research: climate change" in markdown
        assert "## Research Content" in markdown
        assert "Research about climate change..." in markdown
        assert "## Summary" not in markdown
        assert "## Images" not in markdown

    @pytest.mark.asyncio
    async def test_save_markdown_export(self, file_manager_agent):
        """Test saving markdown export to file."""
        # Setup test data
        query = "climate change"
        markdown = "# Research: climate change\n\nContent here..."
        
        # Call the method
        filename = await file_manager_agent.save_markdown_export(query, markdown)
        
        # Verify file was written
        expected_path = os.path.join(file_manager_agent.workspace_dir, f"{filename}.md")
        file_manager_agent.filesystem_mcp.write_file.assert_called_once_with(
            expected_path, markdown
        )
        
        # Verify memory was updated
        file_manager_agent.memory_mcp.store_memory.assert_called_with(
            f"export_{query}", filename, namespace="file_manager"
        )

    @pytest.mark.asyncio
    async def test_save_markdown_export_error(self, file_manager_agent):
        """Test handling errors when saving markdown export."""
        # Setup mock to raise exception
        file_manager_agent.filesystem_mcp.write_file.side_effect = Exception("Write failed")
        
        # Call the method and expect error
        with pytest.raises(FileOperationError, match="Failed to save markdown export"):
            await file_manager_agent.save_markdown_export("test query", "markdown content")
        
        # Verify memory was not updated
        file_manager_agent.memory_mcp.store_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_export_file(self, file_manager_agent):
        """Test retrieving an export file."""
        # Setup mock responses
        file_manager_agent.memory_mcp.retrieve_memory.return_value = "export_file_123"
        file_manager_agent.filesystem_mcp.read_file.return_value = "# Markdown content"
        
        # Call the method
        content = await file_manager_agent.get_export_file("test query")
        
        # Verify memory was queried
        file_manager_agent.memory_mcp.retrieve_memory.assert_called_once_with(
            "export_test query", namespace="file_manager"
        )
        
        # Verify file was read
        expected_path = os.path.join(file_manager_agent.workspace_dir, "export_file_123.md")
        file_manager_agent.filesystem_mcp.read_file.assert_called_once_with(expected_path)
        
        # Verify correct content was returned
        assert content == "# Markdown content"

    @pytest.mark.asyncio
    async def test_get_export_file_not_found(self, file_manager_agent):
        """Test retrieving an export file that doesn't exist."""
        # Setup mock to raise exception for memory retrieval
        file_manager_agent.memory_mcp.retrieve_memory.side_effect = Exception("Memory not found")
        
        # Call the method
        content = await file_manager_agent.get_export_file("test query")
        
        # Verify result is None
        assert content is None
        
        # Verify file was not read
        file_manager_agent.filesystem_mcp.read_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_clean_workspace(self, file_manager_agent):
        """Test cleaning the workspace by removing all files."""
        # Setup mock responses
        file_manager_agent.filesystem_mcp.list_directory.side_effect = [
            "file1.txt\nfile2.txt",  # research dir
            "file3.txt",             # summaries dir
            "file4.txt\nfile5.txt"   # images dir
        ]
        
        # Call the method
        await file_manager_agent.clean_workspace()
        
        # Verify directories were listed
        file_manager_agent.filesystem_mcp.list_directory.assert_any_call(file_manager_agent.research_dir)
        file_manager_agent.filesystem_mcp.list_directory.assert_any_call(file_manager_agent.summaries_dir)
        file_manager_agent.filesystem_mcp.list_directory.assert_any_call(file_manager_agent.images_dir)
        
        # Verify files were deleted
        file_manager_agent.filesystem_mcp.delete_file.assert_any_call(
            os.path.join(file_manager_agent.research_dir, "file1.txt")
        )
        file_manager_agent.filesystem_mcp.delete_file.assert_any_call(
            os.path.join(file_manager_agent.research_dir, "file2.txt")
        )
        file_manager_agent.filesystem_mcp.delete_file.assert_any_call(
            os.path.join(file_manager_agent.summaries_dir, "file3.txt")
        )
        file_manager_agent.filesystem_mcp.delete_file.assert_any_call(
            os.path.join(file_manager_agent.images_dir, "file4.txt")
        )
        file_manager_agent.filesystem_mcp.delete_file.assert_any_call(
            os.path.join(file_manager_agent.images_dir, "file5.txt")
        )
        
        # Verify memory was cleared
        file_manager_agent.memory_mcp.clear_namespace.assert_called_once_with("file_manager")

    @pytest.mark.asyncio
    async def test_clean_workspace_listing_error(self, file_manager_agent):
        """Test handling errors when listing files during workspace cleaning."""
        # Setup mock to raise exception for directory listing
        file_manager_agent.filesystem_mcp.list_directory.side_effect = Exception("Listing failed")
        
        # Call the method and expect error
        with pytest.raises(FileOperationError, match="Failed to clean workspace"):
            await file_manager_agent.clean_workspace()
        
        # Verify memory namespace was not cleared
        file_manager_agent.memory_mcp.clear_namespace.assert_not_called()

    @pytest.mark.asyncio
    async def test_clean_workspace_deletion_error(self, file_manager_agent):
        """Test handling errors when deleting files during workspace cleaning."""
        # Setup mock responses
        file_manager_agent.filesystem_mcp.list_directory.return_value = "file1.txt\nfile2.txt"
        file_manager_agent.filesystem_mcp.delete_file.side_effect = Exception("Deletion failed")
        
        # Call the method and expect error
        with pytest.raises(FileOperationError, match="Failed to clean workspace"):
            await file_manager_agent.clean_workspace()
        
        # Verify memory namespace was not cleared
        file_manager_agent.memory_mcp.clear_namespace.assert_not_called()

    @pytest.mark.asyncio
    async def test_organize_files_by_topic(self, file_manager_agent):
        """Test organizing files by topic."""
        # Setup mock responses
        file_manager_agent.filesystem_mcp.list_directory.side_effect = [
            "climate_research_1.txt\nai_research_1.txt\nclimate_research_2.txt",  # research dir
            "climate_summary_1.txt\nai_summary_1.txt",                           # summaries dir
            "climate_image_1.txt\nai_image_1.txt"                                # images dir
        ]
        file_manager_agent.filesystem_mcp.file_exists.return_value = False
        
        # Call the method
        topics = await file_manager_agent.organize_files_by_topic()
        
        # Verify directories were listed
        file_manager_agent.filesystem_mcp.list_directory.assert_any_call(file_manager_agent.research_dir)
        file_manager_agent.filesystem_mcp.list_directory.assert_any_call(file_manager_agent.summaries_dir)
        file_manager_agent.filesystem_mcp.list_directory.assert_any_call(file_manager_agent.images_dir)
        
        # Verify topic directories were created
        file_manager_agent.filesystem_mcp.create_directory.assert_any_call(
            os.path.join(file_manager_agent.workspace_dir, "topics", "climate")
        )
        file_manager_agent.filesystem_mcp.create_directory.assert_any_call(
            os.path.join(file_manager_agent.workspace_dir, "topics", "ai")
        )
        
        # Verify files were organized correctly
        assert "climate" in topics
        assert "ai" in topics
        assert len(topics["climate"]["research"]) == 2
        assert len(topics["climate"]["summaries"]) == 1
        assert len(topics["climate"]["images"]) == 1
        assert len(topics["ai"]["research"]) == 1
        assert len(topics["ai"]["summaries"]) == 1
        assert len(topics["ai"]["images"]) == 1

    @pytest.mark.asyncio
    async def test_organize_files_by_topic_error(self, file_manager_agent):
        """Test handling errors when organizing files by topic."""
        # Setup mock to raise exception
        file_manager_agent.filesystem_mcp.list_directory.side_effect = Exception("Listing failed")
        
        # Call the method and expect error
        with pytest.raises(FileOperationError, match="Failed to organize files by topic"):
            await file_manager_agent.organize_files_by_topic()

    @pytest.mark.asyncio
    async def test_backup_workspace(self, file_manager_agent):
        """Test backing up the workspace."""
        # Setup mock responses
        file_manager_agent.filesystem_mcp.file_exists.return_value = False
        
        # Call the method
        backup_dir = await file_manager_agent.backup_workspace()
        
        # Verify backup directory was created
        assert "backup_" in backup_dir
        file_manager_agent.filesystem_mcp.create_directory.assert_called_with(
            os.path.join(file_manager_agent.workspace_dir, backup_dir)
        )
        
        # Verify directories were copied
        file_manager_agent.filesystem_mcp.list_directory.assert_any_call(file_manager_agent.research_dir)
        file_manager_agent.filesystem_mcp.list_directory.assert_any_call(file_manager_agent.summaries_dir)
        file_manager_agent.filesystem_mcp.list_directory.assert_any_call(file_manager_agent.images_dir)

    @pytest.mark.asyncio
    async def test_backup_workspace_error(self, file_manager_agent):
        """Test handling errors when backing up the workspace."""
        # Setup mock to raise exception
        file_manager_agent.filesystem_mcp.create_directory.side_effect = Exception("Directory creation failed")
        
        # Call the method and expect error
        with pytest.raises(FileOperationError, match="Failed to backup workspace"):
            await file_manager_agent.backup_workspace()

    @pytest.mark.asyncio
    async def test_restore_from_backup(self, file_manager_agent):
        """Test restoring workspace from backup."""
        # Setup mock responses
        file_manager_agent.filesystem_mcp.file_exists.return_value = True
        
        # Call the method
        success = await file_manager_agent.restore_from_backup("backup_20230615")
        
        # Verify backup was checked
        file_manager_agent.filesystem_mcp.file_exists.assert_called_with(
            os.path.join(file_manager_agent.workspace_dir, "backup_20230615")
        )
        
        # Verify workspace was cleaned before restore
        file_manager_agent.filesystem_mcp.list_directory.assert_any_call(file_manager_agent.research_dir)
        file_manager_agent.filesystem_mcp.list_directory.assert_any_call(file_manager_agent.summaries_dir)
        file_manager_agent.filesystem_mcp.list_directory.assert_any_call(file_manager_agent.images_dir)
        
        # Verify result
        assert success is True

    @pytest.mark.asyncio
    async def test_restore_from_backup_not_found(self, file_manager_agent):
        """Test restoring from a backup that doesn't exist."""
        # Setup mock responses
        file_manager_agent.filesystem_mcp.file_exists.return_value = False
        
        # Call the method
        success = await file_manager_agent.restore_from_backup("backup_20230615")
        
        # Verify result
        assert success is False
        
        # Verify workspace was not cleaned
        file_manager_agent.filesystem_mcp.list_directory.assert_not_called()

    @pytest.mark.asyncio
    async def test_restore_from_backup_error(self, file_manager_agent):
        """Test handling errors when restoring from backup."""
        # Setup mock responses
        file_manager_agent.filesystem_mcp.file_exists.return_value = True
        file_manager_agent.filesystem_mcp.list_directory.side_effect = Exception("Listing failed")
        
        # Call the method and expect error
        with pytest.raises(FileOperationError, match="Failed to restore from backup"):
            await file_manager_agent.restore_from_backup("backup_20230615")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
