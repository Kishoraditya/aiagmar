"""
Integration tests for MCP chaining in complex workflows.

These tests verify that multiple MCP services can be chained together
in complex workflows, with the output of one MCP feeding into another.
"""

import os
import pytest
import json
import tempfile
from unittest.mock import patch, MagicMock

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


class TestMCPChaining:
    """Test chaining of multiple MCP services in integrated workflows."""
    
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
    
    def test_search_fetch_chain(self):
        """Test chaining BraveSearchMCP and FetchMCP."""
        # Step 1: Use BraveSearchMCP to search for information
        search_query = "Python programming language"
        search_results = self.brave_search.web_search(search_query)
        
        # Verify search was successful
        assert "python.org" in search_results.lower()
        
        # Extract URL from search results
        import re
        url_match = re.search(r"URL: (https?://[^\s]+)", search_results)
        assert url_match, "URL not found in search results"
        url = url_match.group(1)
        
        # Step 2: Use FetchMCP to fetch content from the URL
        fetched_content = self.fetch.fetch_url(url)
        
        # Verify fetch was successful
        assert "python" in fetched_content.lower()
        assert "programming language" in fetched_content.lower()
        
        # Verify the chain worked correctly
        assert self.brave_search.web_search_called
        assert self.fetch.fetch_url_called
        assert search_query in self.brave_search.web_search_args[0]
        assert url in self.fetch.fetch_url_args[0]
    
    def test_search_fetch_memory_chain(self):
        """Test chaining BraveSearchMCP, FetchMCP, and MemoryMCP."""
        # Step 1: Use BraveSearchMCP to search for information
        search_query = "Python programming language"
        search_results = self.brave_search.web_search(search_query)
        
        # Extract URL from search results
        import re
        url_match = re.search(r"URL: (https?://[^\s]+)", search_results)
        assert url_match, "URL not found in search results"
        url = url_match.group(1)
        
        # Step 2: Use FetchMCP to fetch content from the URL
        fetched_content = self.fetch.fetch_url(url)
        
        # Step 3: Store the fetched content in MemoryMCP
        memory_key = "python_info"
        self.memory.store_memory(memory_key, fetched_content)
        
        # Step 4: Retrieve the content from MemoryMCP
        retrieved_content = self.memory.retrieve_memory(memory_key)
        
        # Verify the chain worked correctly
        assert self.brave_search.web_search_called
        assert self.fetch.fetch_url_called
        assert self.memory.store_memory_called
        assert self.memory.retrieve_memory_called
        
        assert search_query in self.brave_search.web_search_args[0]
        assert url in self.fetch.fetch_url_args[0]
        assert memory_key in self.memory.retrieve_memory_args[0]
        
        # Verify content was preserved through the chain
        assert "python" in retrieved_content.lower()
        assert "programming language" in retrieved_content.lower()
    
    def test_search_fetch_generate_image_chain(self):
        """Test chaining BraveSearchMCP, FetchMCP, and EverArtMCP."""
        # Step 1: Use BraveSearchMCP to search for information
        search_query = "Python programming language"
        search_results = self.brave_search.web_search(search_query)
        
        # Extract URL from search results
        import re
        url_match = re.search(r"URL: (https?://[^\s]+)", search_results)
        assert url_match, "URL not found in search results"
        url = url_match.group(1)
        
        # Step 2: Use FetchMCP to fetch content from the URL
        fetched_content = self.fetch.fetch_url(url)
        
        # Step 3: Extract key information for image generation
        key_info = "Python programming language with its key features: easy syntax, dynamic typing, and high-level data structures"
        
        # Step 4: Generate an image based on the information
        image_result = self.everart.generate_image(key_info)
        
        # Verify the chain worked correctly
        assert self.brave_search.web_search_called
        assert self.fetch.fetch_url_called
        assert self.everart.generate_image_called
        
        assert search_query in self.brave_search.web_search_args[0]
        assert url in self.fetch.fetch_url_args[0]
        assert "python" in self.everart.generate_image_args[0].lower()
        
        # Verify image generation was successful
        assert "image generated successfully" in image_result.lower()
        assert "url" in image_result.lower()
    
    def test_search_fetch_filesystem_chain(self):
        """Test chaining BraveSearchMCP, FetchMCP, and FilesystemMCP."""
        # Step 1: Use BraveSearchMCP to search for information
        search_query = "Python programming language"
        search_results = self.brave_search.web_search(search_query)
        
        # Extract URL from search results
        import re
        url_match = re.search(r"URL: (https?://[^\s]+)", search_results)
        assert url_match, "URL not found in search results"
        url = url_match.group(1)
        
        # Step 2: Use FetchMCP to fetch content from the URL
        fetched_content = self.fetch.fetch_url(url)
        
        # Step 3: Save the fetched content to a file using FilesystemMCP
        file_path = "python_info.txt"
        self.filesystem.write_file(file_path, fetched_content)
        
        # Step 4: Read the file back using FilesystemMCP
        read_content = self.filesystem.read_file(file_path)
        
        # Verify the chain worked correctly
        assert self.brave_search.web_search_called
        assert self.fetch.fetch_url_called
        assert self.filesystem.write_file_called
        assert self.filesystem.read_file_called
        
        assert search_query in self.brave_search.web_search_args[0]
        assert url in self.fetch.fetch_url_args[0]
        assert file_path in self.filesystem.write_file_args[0]
        assert file_path in self.filesystem.read_file_args[0]
        
        # Verify content was preserved through the chain
        assert "python" in read_content.lower()
        assert "programming language" in read_content.lower()
    
    def test_complete_research_workflow_chain(self):
        """Test a complete research workflow chain using all MCPs."""
        # Step 1: Use BraveSearchMCP to search for information
        search_query = "Python programming language"
        search_results = self.brave_search.web_search(search_query)
        
        # Extract URL from search results
        import re
        url_match = re.search(r"URL: (https?://[^\s]+)", search_results)
        assert url_match, "URL not found in search results"
        url = url_match.group(1)
        
        # Step 2: Use FetchMCP to fetch content from the URL
        fetched_content = self.fetch.fetch_url(url)
        
        # Step 3: Store the fetched content in MemoryMCP
        self.memory.store_memory("research_data", fetched_content)
        
        # Step 4: Generate a summary based on the fetched content
        summary = "Python is a high-level programming language known for its readability and versatility."
        self.memory.store_memory("summary", summary)
        
        # Step 5: Generate an image based on the summary
        image_result = self.everart.generate_image(summary)
        
        # Extract image URL
        image_url_match = re.search(r"URL: (https?://[^\s]+)", image_result)
        assert image_url_match, "Image URL not found in result"
        image_url = image_url_match.group(1)
        
        # Step 6: Save the summary and image URL to files
        self.filesystem.write_file("python_summary.txt", summary)
        self.filesystem.write_file("python_image_url.txt", image_url)
        
        # Step 7: Create a final report combining all information
        final_report = f"""
        # Python Programming Language Research
        
        ## Summary
        {summary}
        
        ## Image
        ![Python Image]({image_url})
        
        ## Sources
        - {url}
        
        ## Raw Data
        ```
        {fetched_content[:200]}...
        ```
        """
        
        # Step 8: Save the final report
        report_path = "python_report.md"
        self.filesystem.write_file(report_path, final_report)
        
        # Step 9: Read back the report
        read_report = self.filesystem.read_file(report_path)
        
        # Verify the complete chain worked correctly
        assert self.brave_search.web_search_called
        assert self.fetch.fetch_url_called
        assert self.memory.store_memory_called
        assert self.everart.generate_image_called
        assert self.filesystem.write_file_called
        assert self.filesystem.read_file_called
        
        # Verify content was preserved through the chain
        assert "python" in read_report.lower()
        assert "summary" in read_report.lower()
        assert image_url in read_report
        assert url in read_report
    
    def test_error_recovery_in_chain(self):
        """Test error recovery in an MCP chain."""
        # Step 1: Use BraveSearchMCP to search for information
        search_query = "Python programming language"
        
        # Make BraveSearchMCP fail
        original_web_search = self.brave_search.web_search
        self.brave_search.web_search = MagicMock(side_effect=RuntimeError("API rate limit exceeded"))
        
        try:
            # Attempt to search (will fail)
            try:
                search_results = self.brave_search.web_search(search_query)
                assert False, "BraveSearchMCP should have failed"
            except RuntimeError:
                # Expected error
                pass
            
            # Step 2: Recover by using a fallback search
            fallback_search_results = """
            Title: Python Fallback Result
            Description: This is a fallback result for Python.
            URL: https://fallback.example.com/python
            """
            
            # Step 3: Continue the chain with the fallback results
            url = "https://fallback.example.com/python"
            
            # Step 4: Use FetchMCP to fetch content from the URL
            fetched_content = self.fetch.fetch_url(url)
            
            # Step 5: Store the fetched content in MemoryMCP
            self.memory.store_memory("fallback_data", fetched_content)
            
            # Verify the chain recovered and continued
            assert self.brave_search.web_search_called
            assert self.fetch.fetch_url_called
            assert self.memory.store_memory_called
            
            assert url in self.fetch.fetch_url_args[0]
            assert "fallback_data" in [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
            
            # Verify content was preserved despite the error
            retrieved_content = self.memory.retrieve_memory("fallback_data")
            assert "python" in retrieved_content.lower()
        
        finally:
            # Restore original method
            self.brave_search.web_search = original_web_search
    
    def test_parallel_mcp_chains(self):
        """Test parallel MCP chains executing simultaneously."""
        # Chain 1: Search -> Fetch -> Memory
        def chain_1():
            search_results = self.brave_search.web_search("Python programming")
            url = "https://www.python.org/"
            fetched_content = self.fetch.fetch_url(url)
            self.memory.store_memory("chain_1_data", fetched_content)
            return fetched_content
        
        # Chain 2: Search -> Generate Image -> Filesystem
        def chain_2():
            search_results = self.brave_search.web_search("Python logo")
            image_prompt = "Python programming language logo with snake"
            image_result = self.everart.generate_image(image_prompt)
            self.filesystem.write_file("python_image.txt", image_result)
            return image_result
        
        # Execute both chains
        content_1 = chain_1()
        content_2 = chain_2()
        
        # Verify both chains executed successfully
        assert "python" in content_1.lower()
        assert "image" in content_2.lower()
        
        # Verify all MCPs were called
        assert self.brave_search.web_search_called
        assert self.fetch.fetch_url_called
        assert self.memory.store_memory_called
        assert self.everart.generate_image_called
        assert self.filesystem.write_file_called
        
        # Verify chain 1 results
        retrieved_content = self.memory.retrieve_memory("chain_1_data")
        assert "python" in retrieved_content.lower()
        
        # Verify chain 2 results
        image_content = self.filesystem.read_file("python_image.txt")
        assert "image" in image_content.lower()
    
    def test_branching_mcp_chain(self):
        """Test a branching MCP chain where one output feeds multiple downstream MCPs."""
        # Step 1: Use BraveSearchMCP to search for information
        search_query = "Python programming language"
        search_results = self.brave_search.web_search(search_query)
        
        # Extract URL from search results
        import re
        url_match = re.search(r"URL: (https?://[^\s]+)", search_results)
        assert url_match, "URL not found in search results"
        url = url_match.group(1)
        
        # Step 2: Use FetchMCP to fetch content from the URL
        fetched_content = self.fetch.fetch_url(url)
        
        # Branch 1: Store in Memory
        self.memory.store_memory("python_info", fetched_content)
        
        # Branch 2: Generate image based on content
        image_prompt = "Python programming language logo and code"
        image_result = self.everart.generate_image(image_prompt)
        
        # Branch 3: Save to filesystem
        self.filesystem.write_file("python_info.txt", fetched_content)
        
        # Verify all branches executed successfully
        assert self.memory.store_memory_called
        assert self.everart.generate_image_called
        assert self.filesystem.write_file_called
        
        # Verify branch 1 results
        retrieved_content = self.memory.retrieve_memory("python_info")
        assert "python" in retrieved_content.lower()
        
        # Verify branch 2 results
        assert "image" in image_result.lower()
        
        # Verify branch 3 results
        file_content = self.filesystem.read_file("python_info.txt")
        assert "python" in file_content.lower()
    
    def test_iterative_mcp_chain(self):
        """Test an iterative MCP chain with multiple rounds of processing."""
        # Initial search
        topics = ["Python basics", "Python libraries", "Python applications"]
        all_results = {}
        
        for topic in topics:
            # Step 1: Search for the topic
            search_results = self.brave_search.web_search(topic)
            
            # Extract URL from search results
            import re
            url_match = re.search(r"URL: (https?://[^\s]+)", search_results)
            assert url_match, "URL not found in search results"
            url = url_match.group(1)
            
            # Step 2: Fetch content for the topic
            fetched_content = self.fetch.fetch_url(url)
            
            # Step 3: Store in memory
            memory_key = f"{topic.lower().replace(' ', '_')}_info"
            self.memory.store_memory(memory_key, fetched_content)
            
            # Save results for this topic
            all_results[topic] = {
                "search_results": search_results,
                "url": url,
                "content": fetched_content,
                "memory_key": memory_key
            }
        
        # Step 4: Combine all results into a comprehensive report
        combined_report = "# Python Research Report\n\n"
        
        for topic, data in all_results.items():
            combined_report += f"## {topic}\n\n"
            combined_report += f"Source: {data['url']}\n\n"
            combined_report += f"Content excerpt: {data['content'][:200]}...\n\n"
        
        # Step 5: Save the combined report
        report_path = "python_comprehensive_report.md"
        self.filesystem.write_file(report_path, combined_report)
        
        # Step 6: Generate an image for the report
        image_prompt = "Python programming language comprehensive visual guide"
        image_result = self.everart.generate_image(image_prompt)
        
        # Extract image URL
        image_url_match = re.search(r"URL: (https?://[^\s]+)", image_result)
        assert image_url_match, "Image URL not found in result"
        image_url = image_url_match.group(1)
        
        # Step 7: Add the image to the report
        updated_report = combined_report + f"\n\n## Visual Representation\n\n![Python Visual Guide]({image_url})\n"
        self.filesystem.write_file(report_path, updated_report)
        
        # Verify the iterative chain worked correctly
        assert self.brave_search.web_search_called
        assert len(self.brave_search.web_search_args) >= len(topics)
        
        assert self.fetch.fetch_url_called
        assert len(self.fetch.fetch_url_args) >= len(topics)
        
        assert self.memory.store_memory_called
        assert len(self.memory.store_memory_args) >= len(topics)
        
        assert self.filesystem.write_file_called
        assert self.everart.generate_image_called
        
        # Verify final report contains information from all topics
        final_report = self.filesystem.read_file(report_path)
        for topic in topics:
            assert topic in final_report
    
    def test_conditional_mcp_chain(self):
        """Test a conditional MCP chain where execution path depends on intermediate results."""
        # Step 1: Use BraveSearchMCP to search for information
        search_query = "Python programming language"
        search_results = self.brave_search.web_search(search_query)
        
        # Step 2: Check if search results contain specific information
        if "high-level" in search_results.lower():
            # Path A: Focus on Python's high-level features
            fetched_content = self.fetch.fetch_text("https://www.python.org/about/")
            self.memory.store_memory("python_features", fetched_content)
            
            # Generate an image highlighting Python's high-level nature
            image_result = self.everart.generate_image("Python high-level programming language features")
            
        else:
            # Path B: Focus on Python's general information
            fetched_content = self.fetch.fetch_text("https://www.python.org/")
            self.memory.store_memory("python_general", fetched_content)
            
            # Generate a general Python image
            image_result = self.everart.generate_image("Python programming language general overview")
        
        # Verify the conditional chain executed correctly
        assert self.brave_search.web_search_called
        assert self.fetch.fetch_text_called
        assert self.memory.store_memory_called
        assert self.everart.generate_image_called
        
        # Verify the correct path was taken
        if "high-level" in search_results.lower():
            assert "python_features" in [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
            assert "high-level" in self.everart.generate_image_args[0].lower()
        else:
            assert "python_general" in [args[0] for args in self.memory.store_memory_args if isinstance(args, tuple) and len(args) > 0]
            assert "general" in self.everart.generate_image_args[0].lower()
    
    def test_mcp_chain_with_feedback_loop(self):
        """Test an MCP chain with a feedback loop for refinement."""
        # Step 1: Initial search
        search_query = "Python programming language"
        search_results = self.brave_search.web_search(search_query)
        
        # Extract URL from search results
        import re
        url_match = re.search(r"URL: (https?://[^\s]+)", search_results)
        assert url_match, "URL not found in search results"
        url = url_match.group(1)
        
        # Step 2: Fetch initial content
        initial_content = self.fetch.fetch_url(url)
        
        # Step 3: Store initial content
        self.memory.store_memory("initial_research", initial_content)
        
        # Step 4: Analyze content and identify areas needing more information
        missing_topics = ["Python libraries", "Python applications"]
        
        # Feedback loop - iterate through missing topics
        for topic in missing_topics:
            # Step 5: Perform additional searches
            additional_results = self.brave_search.web_search(topic)
            
            # Extract URL from additional results
            url_match = re.search(r"URL: (https?://[^\s]+)", additional_results)
            if url_match:
                additional_url = url_match.group(1)
                
                # Step 6: Fetch additional content
                additional_content = self.fetch.fetch_url(additional_url)
                
                # Step 7: Store additional content
                self.memory.store_memory(f"{topic.lower().replace(' ', '_')}_research", additional_content)
        
        # Step 8: Combine all research into a comprehensive report
        initial_research = self.memory.retrieve_memory("initial_research")
        
        combined_report = "# Python Comprehensive Research\n\n"
        combined_report += "## General Information\n\n"
        combined_report += initial_research[:200] + "...\n\n"
        
        for topic in missing_topics:
            memory_key = f"{topic.lower().replace(' ', '_')}_research"
            topic_research = self.memory.retrieve_memory(memory_key)
            
            combined_report += f"## {topic}\n\n"
            combined_report += topic_research[:200] + "...\n\n"
        
        # Step 9: Save the final report
        report_path = "python_comprehensive_report.md"
        self.filesystem.write_file(report_path, combined_report)
        
        # Verify the feedback loop chain worked correctly
        assert self.brave_search.web_search_called
        assert len(self.brave_search.web_search_args) >= len(missing_topics) + 1
        
        assert self.fetch.fetch_url_called
        assert self.memory.store_memory_called
        assert self.memory.retrieve_memory_called
        assert self.filesystem.write_file_called
        
        # Verify final report contains information from all topics
        final_report = self.filesystem.read_file(report_path)
        assert "General Information" in final_report
        for topic in missing_topics:
            assert topic in final_report
    
    def test_mcp_chain_with_error_handling_and_retry(self):
        """Test an MCP chain with error handling and retry logic."""
        # Step 1: Use BraveSearchMCP to search for information
        search_query = "Python programming language"
        
        # Make BraveSearchMCP fail on first attempt but succeed on retry
        original_web_search = self.brave_search.web_search
        fail_once_mock = MagicMock()
        fail_once_mock.side_effect = [
            RuntimeError("API rate limit exceeded"),
            """
            Title: Python Programming Language
            Description: Python is a high-level programming language.
            URL: https://www.python.org/
            """
        ]
        self.brave_search.web_search = fail_once_mock
        
        try:
            # Attempt to search with retry logic
            max_retries = 3
            retry_count = 0
            search_results = None
            
            while retry_count < max_retries:
                try:
                    search_results = self.brave_search.web_search(search_query)
                    break  # Success, exit the loop
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise  # Re-raise if max retries reached
                    # In a real implementation, you might add a delay here
            
            # Verify retry was successful
            assert search_results is not None
            assert "python" in search_results.lower()
            
            # Continue the chain
            url = "https://www.python.org/"
            
            # Step 2: Use FetchMCP to fetch content from the URL
            fetched_content = self.fetch.fetch_url(url)
            
            # Step 3: Store the fetched content in MemoryMCP
            self.memory.store_memory("python_info", fetched_content)
            
            # Verify the chain completed successfully despite the initial error
            assert fail_once_mock.call_count == 2  # Initial failure + successful retry
            assert self.fetch.fetch_url_called
            assert self.memory.store_memory_called
            
            # Verify content was preserved
            retrieved_content = self.memory.retrieve_memory("python_info")
            assert "python" in retrieved_content.lower()
            
        finally:
            # Restore original method
            self.brave_search.web_search = original_web_search


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
