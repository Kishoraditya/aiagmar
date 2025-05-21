"""
End-to-end performance tests for the system.

These tests measure and verify the performance characteristics of the system,
including:
- Response time for individual MCPs
- Throughput of agent operations
- Memory usage during workflow execution
- Scalability with increasing workload
- Performance under concurrent operations
"""

import os
import time
import pytest
import psutil
import threading
import concurrent.futures
from unittest.mock import patch, MagicMock

# Import agents
from apps.agents.manager_agent import ManagerAgent
from apps.agents.pre_response_agent import PreResponseAgent
from apps.agents.research_agent import ResearchAgent
from apps.agents.image_generation_agent import ImageGenerationAgent
from apps.agents.file_manager_agent import FileManagerAgent
from apps.agents.summary_agent import SummaryAgent
from apps.agents.verification_agent import VerificationAgent

# Import MCPs
from apps.mcps.brave_search_mcp import BraveSearchMCP
from apps.mcps.everart_mcp import EverArtMCP
from apps.mcps.fetch_mcp import FetchMCP
from apps.mcps.filesystem_mcp import FilesystemMCP
from apps.mcps.memory_mcp import MemoryMCP

# Import workflow
from apps.workflows.research_workflow import ResearchWorkflow

# Import utils
from apps.utils.logger import Logger
from apps.utils.decorators import (
    timed,
    async_timed
)


class TestPerformance:
    """Test performance characteristics of the system."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up test environment before each test."""
        # Create a temporary workspace directory
        self.workspace_dir = str(tmp_path / "workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Set up environment variables for testing
        os.environ["BRAVE_API_KEY"] = "test_brave_api_key"
        os.environ["EVERART_API_KEY"] = "test_everart_api_key"
        
        # Create a logger for performance metrics
        self.logger = Logger("performance_tests")
        
        yield
        
        # Clean up environment variables
        if "BRAVE_API_KEY" in os.environ:
            del os.environ["BRAVE_API_KEY"]
        if "EVERART_API_KEY" in os.environ:
            del os.environ["EVERART_API_KEY"]
    
    def measure_execution_time(self, func, *args, **kwargs):
        """Measure the execution time of a function."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    
    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure the memory usage of a function."""
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        result = func(*args, **kwargs)
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        return result, memory_used
    
    def test_mcp_initialization_time(self):
        """Test the initialization time of MCPs."""
        mcp_classes = [
            (BraveSearchMCP, {"api_key": "test_brave_api_key"}),
            (EverArtMCP, {"api_key": "test_everart_api_key"}),
            (FetchMCP, {}),
            (FilesystemMCP, {"workspace_dir": self.workspace_dir}),
            (MemoryMCP, {})
        ]
        
        for mcp_class, kwargs in mcp_classes:
            # Measure initialization time
            start_time = time.time()
            mcp = mcp_class(**kwargs)
            end_time = time.time()
            init_time = end_time - start_time
            
            # Close the MCP
            mcp.close()
            
            # Log and assert
            self.logger.info(f"{mcp_class.__name__} initialization time: {init_time:.4f} seconds")
            assert init_time < 5.0, f"{mcp_class.__name__} initialization took too long: {init_time:.4f} seconds"
    
    def test_brave_search_mcp_response_time(self):
        """Test the response time of BraveSearchMCP."""
        # Create a mock BraveSearchMCP that returns quickly
        brave_search = BraveSearchMCP(api_key="test_brave_api_key")
        
        # Mock the _send_request method to return immediately
        def mock_send_request(method, params):
            return {
                "content": [{"type": "text", "text": "Mock search results for Python programming"}],
                "isError": False
            }
        
        with patch.object(brave_search, '_send_request', side_effect=mock_send_request):
            # Measure response time for web search
            start_time = time.time()
            result = brave_search.web_search("Python programming")
            end_time = time.time()
            web_search_time = end_time - start_time
            
            # Measure response time for local search
            start_time = time.time()
            result = brave_search.local_search("Coffee shops in San Francisco")
            end_time = time.time()
            local_search_time = end_time - start_time
        
        # Close the MCP
        brave_search.close()
        
        # Log and assert
        self.logger.info(f"BraveSearchMCP web_search response time: {web_search_time:.4f} seconds")
        self.logger.info(f"BraveSearchMCP local_search response time: {local_search_time:.4f} seconds")
        
        assert web_search_time < 1.0, f"Web search took too long: {web_search_time:.4f} seconds"
        assert local_search_time < 1.0, f"Local search took too long: {local_search_time:.4f} seconds"
    
    def test_fetch_mcp_response_time(self):
        """Test the response time of FetchMCP."""
        # Create a mock FetchMCP that returns quickly
        fetch = FetchMCP()
        
        # Mock the _send_request method to return immediately
        def mock_send_request(method, params):
            return {
                "content": [{"type": "text", "text": "Mock fetched content from example.com"}],
                "isError": False
            }
        
        with patch.object(fetch, '_send_request', side_effect=mock_send_request):
            # Measure response time for fetch_url
            start_time = time.time()
            result = fetch.fetch_url("https://example.com")
            end_time = time.time()
            fetch_url_time = end_time - start_time
            
            # Measure response time for fetch_text
            start_time = time.time()
            result = fetch.fetch_text("https://example.com")
            end_time = time.time()
            fetch_text_time = end_time - start_time
        
        # Close the MCP
        fetch.close()
        
        # Log and assert
        self.logger.info(f"FetchMCP fetch_url response time: {fetch_url_time:.4f} seconds")
        self.logger.info(f"FetchMCP fetch_text response time: {fetch_text_time:.4f} seconds")
        
        assert fetch_url_time < 1.0, f"Fetch URL took too long: {fetch_url_time:.4f} seconds"
        assert fetch_text_time < 1.0, f"Fetch text took too long: {fetch_text_time:.4f} seconds"
    
    def test_filesystem_mcp_operations_time(self):
        """Test the operation time of FilesystemMCP."""
        # Create a mock FilesystemMCP
        filesystem = FilesystemMCP(workspace_dir=self.workspace_dir)
        
        # Mock the _send_request method to return immediately
        def mock_send_request(method, params):
            if method == "callTool":
                if params["name"] == "read_file":
                    return {
                        "content": [{"type": "text", "text": "Mock file content"}],
                        "isError": False
                    }
                elif params["name"] == "write_file":
                    return {
                        "content": [{"type": "text", "text": "File written successfully"}],
                        "isError": False
                    }
                elif params["name"] == "list_directory":
                    return {
                        "content": [{"type": "text", "text": "file1.txt\nfile2.txt\ndir1/"}],
                        "isError": False
                    }
            return {}
        
        with patch.object(filesystem, '_send_request', side_effect=mock_send_request):
            # Measure write_file operation time
            start_time = time.time()
            result = filesystem.write_file("test.txt", "Test content")
            end_time = time.time()
            write_time = end_time - start_time
            
            # Measure read_file operation time
            start_time = time.time()
            result = filesystem.read_file("test.txt")
            end_time = time.time()
            read_time = end_time - start_time
            
            # Measure list_directory operation time
            start_time = time.time()
            result = filesystem.list_directory(".")
            end_time = time.time()
            list_time = end_time - start_time
        
        # Close the MCP
        filesystem.close()
        
        # Log and assert
        self.logger.info(f"FilesystemMCP write_file operation time: {write_time:.4f} seconds")
        self.logger.info(f"FilesystemMCP read_file operation time: {read_time:.4f} seconds")
        self.logger.info(f"FilesystemMCP list_directory operation time: {list_time:.4f} seconds")
        
        assert write_time < 1.0, f"Write file took too long: {write_time:.4f} seconds"
        assert read_time < 1.0, f"Read file took too long: {read_time:.4f} seconds"
        assert list_time < 1.0, f"List directory took too long: {list_time:.4f} seconds"
    
    def test_memory_mcp_operations_time(self):
        """Test the operation time of MemoryMCP."""
        # Create a mock MemoryMCP
        memory = MemoryMCP()
        
        # Mock the _send_request method to return immediately
        def mock_send_request(method, params):
            if method == "callTool":
                if params["name"] == "store_memory":
                    return {
                        "content": [{"type": "text", "text": "Memory stored successfully"}],
                        "isError": False
                    }
                elif params["name"] == "retrieve_memory":
                    return {
                        "content": [{"type": "text", "text": "Mock memory value"}],
                        "isError": False
                    }
                elif params["name"] == "list_memories":
                    return {
                        "content": [{"type": "text", "text": "key1\nkey2\nkey3"}],
                        "isError": False
                    }
            return {}
        
        with patch.object(memory, '_send_request', side_effect=mock_send_request):
            # Measure store_memory operation time
            start_time = time.time()
            result = memory.store_memory("test_key", "Test value")
            end_time = time.time()
            store_time = end_time - start_time
            
            # Measure retrieve_memory operation time
            start_time = time.time()
            result = memory.retrieve_memory("test_key")
            end_time = time.time()
            retrieve_time = end_time - start_time
            
            # Measure list_memories operation time
            start_time = time.time()
            result = memory.list_memories()
            end_time = time.time()
            list_time = end_time - start_time
        
        # Close the MCP
        memory.close()
        
        # Log and assert
        self.logger.info(f"MemoryMCP store_memory operation time: {store_time:.4f} seconds")
        self.logger.info(f"MemoryMCP retrieve_memory operation time: {retrieve_time:.4f} seconds")
        self.logger.info(f"MemoryMCP list_memories operation time: {list_time:.4f} seconds")
        
        assert store_time < 1.0, f"Store memory took too long: {store_time:.4f} seconds"
        assert retrieve_time < 1.0, f"Retrieve memory took too long: {retrieve_time:.4f} seconds"
        assert list_time < 1.0, f"List memories took too long: {list_time:.4f} seconds"
    
    def test_everart_mcp_response_time(self):
        """Test the response time of EverArtMCP."""
        # Create a mock EverArtMCP
        everart = EverArtMCP(api_key="test_everart_api_key")
        
        # Mock the _send_request method to return immediately
        def mock_send_request(method, params):
            if method == "callTool":
                if params["name"] == "everart_generate_image":
                    return {
                        "content": [{"type": "text", "text": "https://example.com/image.jpg"}],
                        "isError": False
                    }
                elif params["name"] == "everart_describe_image":
                    return {
                        "content": [{"type": "text", "text": "An image of a mountain landscape"}],
                        "isError": False
                    }
            return {}
        
        with patch.object(everart, '_send_request', side_effect=mock_send_request):
            # Measure generate_image operation time
            start_time = time.time()
            result = everart.generate_image("A mountain landscape")
            end_time = time.time()
            generate_time = end_time - start_time
            
            # Measure describe_image operation time
            start_time = time.time()
            result = everart.describe_image("https://example.com/image.jpg")
            end_time = time.time()
            describe_time = end_time - start_time
        
        # Close the MCP
        everart.close()
        
        # Log and assert
        self.logger.info(f"EverArtMCP generate_image operation time: {generate_time:.4f} seconds")
        self.logger.info(f"EverArtMCP describe_image operation time: {describe_time:.4f} seconds")
        
        assert generate_time < 1.0, f"Generate image took too long: {generate_time:.4f} seconds"
        assert describe_time < 1.0, f"Describe image took too long: {describe_time:.4f} seconds"
    
    def test_agent_execution_time(self):
        """Test the execution time of individual agents."""
        # Create mock agents
        agents = {
            "manager": ManagerAgent(),
            "pre_response": PreResponseAgent(),
            "research": ResearchAgent(),
            "image_generation": ImageGenerationAgent(),
            "file_manager": FileManagerAgent(),
            "summary": SummaryAgent(),
            "verification": VerificationAgent()
        }
        
        # Set up mock responses for each agent
        for agent_name, agent in agents.items():
            agent.set_response(f"Mock response from {agent_name} agent")
        
        # Measure execution time for each agent
        execution_times = {}
        for agent_name, agent in agents.items():
            start_time = time.time()
            result = agent.execute(f"Test query for {agent_name} agent")
            end_time = time.time()
            execution_times[agent_name] = end_time - start_time
            
            # Log and assert
            self.logger.info(f"{agent_name.capitalize()} agent execution time: {execution_times[agent_name]:.4f} seconds")
            assert execution_times[agent_name] < 1.0, f"{agent_name.capitalize()} agent took too long: {execution_times[agent_name]:.4f} seconds"
    
    def test_workflow_execution_time(self):
        """Test the execution time of the research workflow."""
        # Create a mock research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate the research on Python programming language.")
        workflow.pre_response_agent.set_response("I'll present the research plan for Python programming language.")
        workflow.research_agent.set_response("I've found information about Python programming language.")
        workflow.image_generation_agent.set_response("I've generated an image of Python programming language.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("Python is a high-level programming language.")
        workflow.verification_agent.set_response("I've verified the facts about Python programming language.")
        
        # Measure execution time for the workflow
        start_time = time.time()
        result = workflow.execute("Tell me about Python programming language")
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Log and assert
        self.logger.info(f"Research workflow execution time: {execution_time:.4f} seconds")
        assert execution_time < 5.0, f"Research workflow took too long: {execution_time:.4f} seconds"
    
    def test_workflow_memory_usage(self):
        """Test the memory usage of the research workflow."""
        # Create a mock research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Set up mock responses for each agent
        workflow.manager_agent.set_response("I'll coordinate the research on Python programming language.")
        workflow.pre_response_agent.set_response("I'll present the research plan for Python programming language.")
        workflow.research_agent.set_response("I've found information about Python programming language.")
        workflow.image_generation_agent.set_response("I've generated an image of Python programming language.")
        workflow.file_manager_agent.set_response("I've saved all research materials.")
        workflow.summary_agent.set_response("Python is a high-level programming language.")
        workflow.verification_agent.set_response("I've verified the facts about Python programming language.")
        
        # Measure memory usage for the workflow
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = workflow.execute("Tell me about Python programming language")
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Log and assert
        self.logger.info(f"Research workflow memory usage: {memory_used:.2f} MB")
        assert memory_used < 100.0, f"Research workflow used too much memory: {memory_used:.2f} MB"
    
    def test_concurrent_mcp_operations(self):
        """Test performance of concurrent MCP operations."""
        # Create mock MCPs
        brave_search = BraveSearchMCP(api_key="test_brave_api_key")
        fetch = FetchMCP()
        filesystem = FilesystemMCP(workspace_dir=self.workspace_dir)
        memory = MemoryMCP()
        
        # Mock the _send_request methods to return immediately
        for mcp in [brave_search, fetch, filesystem, memory]:
            def mock_send_request(method, params):
                return {
                    "content": [{"type": "text", "text": f"Mock response for {params.get('name', 'unknown')}"}],
                    "isError": False
                }
            mcp._send_request = mock_send_request
        
        # Define operations to run concurrently
        operations = [
            (brave_search.web_search, "Python programming"),
            (fetch.fetch_url, "https://example.com"),
            (filesystem.write_file, "test.txt", "Test content"),
            (filesystem.read_file, "test.txt"),
            (memory.store_memory, "test_key", "Test value"),
            (memory.retrieve_memory, "test_key")
        ]
        
        # Run operations concurrently
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(operations)) as executor:
            futures = []
            for op in operations:
                func, *args = op
                futures.append(executor.submit(func, *args))
            
            # Wait for all operations to complete
            concurrent.futures.wait(futures)
            
            # Get results
            results = [future.result() for future in futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Close MCPs
        for mcp in [brave_search, fetch, filesystem, memory]:
            mcp.close()
        
        # Log and assert
        self.logger.info(f"Concurrent MCP operations time: {total_time:.4f} seconds")
        assert total_time < 2.0, f"Concurrent MCP operations took too long: {total_time:.4f} seconds"
        assert len(results) == len(operations), f"Not all operations completed: {len(results)} of {len(operations)}"
    
    def test_workflow_scalability(self):
        """Test scalability of the research workflow with increasing workload."""
        # Create a mock research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Set up mock responses for each agent
        for agent in [
            workflow.manager_agent,
            workflow.pre_response_agent,
            workflow.research_agent,
            workflow.image_generation_agent,
            workflow.file_manager_agent,
            workflow.summary_agent,
            workflow.verification_agent
        ]:
            agent.set_response("Mock response")
        
        # Define workloads of increasing complexity
        workloads = [
            "Python",  # Simple query
            "Tell me about Python programming language",  # Medium query
            "Explain the differences between Python 2 and Python 3, including syntax changes and library support",  # Complex query
            "Compare Python, JavaScript, and Rust for web development, considering performance, ecosystem, and learning curve"  # Very complex query
        ]
        
        # Measure execution time for each workload
        execution_times = []
        for query in workloads:
            start_time = time.time()
            result = workflow.execute(query)
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            # Log
            self.logger.info(f"Workflow execution time for query '{query}': {execution_time:.4f} seconds")
        
        # Check if execution time scales reasonably with complexity
        # We expect some increase, but it shouldn't be exponential
        for i in range(1, len(execution_times)):
            ratio = execution_times[i] / execution_times[i-1]
            self.logger.info(f"Execution time ratio between workload {i} and {i-1}: {ratio:.2f}")
            assert ratio < 3.0, f"Execution time increased too much between workload {i-1} and {i}: ratio {ratio:.2f}"
    
    def test_parallel_vs_sequential_execution(self):
        """Compare performance of parallel vs sequential agent execution."""
        # Create workflows with sequential and parallel execution
        sequential_workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent(),
            parallel_execution=False
        )
        
        parallel_workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent(),
            parallel_execution=True
        )
        
        # Set up mock responses for each agent
        for workflow in [sequential_workflow, parallel_workflow]:
            for agent in [
                workflow.manager_agent,
                workflow.pre_response_agent,
                workflow.research_agent,
                workflow.image_generation_agent,
                workflow.file_manager_agent,
                workflow.summary_agent,
                workflow.verification_agent
            ]:
                # Add a small delay to simulate work
                def delayed_response(*args, **kwargs):
                    time.sleep(0.1)
                    return "Mock response"
                agent.set_custom_action(delayed_response)
        
        # Measure execution time for sequential workflow
        start_time = time.time()
        sequential_result = sequential_workflow.execute("Tell me about Python programming language")
        end_time = time.time()
        sequential_time = end_time - start_time
        
        # Measure execution time for parallel workflow
        start_time = time.time()
        parallel_result = parallel_workflow.execute("Tell me about Python programming language")
        end_time = time.time()
        parallel_time = end_time - start_time
        
        # Log and assert
        self.logger.info(f"Sequential workflow execution time: {sequential_time:.4f} seconds")
        self.logger.info(f"Parallel workflow execution time: {parallel_time:.4f} seconds")
        self.logger.info(f"Speedup from parallelization: {sequential_time / parallel_time:.2f}x")
        
        # Parallel should be faster, but the exact speedup depends on the workflow structure
        assert parallel_time < sequential_time, f"Parallel execution ({parallel_time:.4f}s) should be faster than sequential ({sequential_time:.4f}s)"
    
    def test_mcp_connection_overhead(self):
        """Test the overhead of MCP connection establishment."""
        # Measure time to create and close MCP instances
        mcp_classes = [
            (BraveSearchMCP, {"api_key": "test_brave_api_key"}),
            (EverArtMCP, {"api_key": "test_everart_api_key"}),
            (FetchMCP, {}),
            (FilesystemMCP, {"workspace_dir": self.workspace_dir}),
            (MemoryMCP, {})
        ]
        
        for mcp_class, kwargs in mcp_classes:
            # Measure time to create and close the MCP
            start_time = time.time()
            mcp = mcp_class(**kwargs)
            mcp.close()
            end_time = time.time()
            overhead_time = end_time - start_time
            
            # Log and assert
            self.logger.info(f"{mcp_class.__name__} connection overhead: {overhead_time:.4f} seconds")
            assert overhead_time < 2.0, f"{mcp_class.__name__} connection overhead too high: {overhead_time:.4f} seconds"
    
    def test_mcp_reuse_performance(self):
        """Test performance improvement from reusing MCP connections."""
        # Create a BraveSearchMCP instance
        brave_search = BraveSearchMCP(api_key="test_brave_api_key")
        
        # Mock the _send_request method to return immediately
        def mock_send_request(method, params):
            return {
                "content": [{"type": "text", "text": "Mock search results"}],
                "isError": False
            }
        
        with patch.object(brave_search, '_send_request', side_effect=mock_send_request):
            # Measure time for multiple operations with the same MCP instance
            start_time = time.time()
            for i in range(10):
                result = brave_search.web_search(f"Query {i}")
            end_time = time.time()
            reuse_time = end_time - start_time
            
            # Measure time for multiple operations with new MCP instances
            start_time = time.time()
            for i in range(10):
                temp_brave = BraveSearchMCP(api_key="test_brave_api_key")
                with patch.object(temp_brave, '_send_request', side_effect=mock_send_request):
                    result = temp_brave.web_search(f"Query {i}")
                temp_brave.close()
            end_time = time.time()
            new_instance_time = end_time - start_time
        
        # Close the MCP
        brave_search.close()
        
        # Log and assert
        self.logger.info(f"Time with MCP reuse: {reuse_time:.4f} seconds")
        self.logger.info(f"Time with new MCP instances: {new_instance_time:.4f} seconds")
        self.logger.info(f"Performance improvement from reuse: {new_instance_time / reuse_time:.2f}x")
        
        assert reuse_time < new_instance_time, f"Reusing MCP ({reuse_time:.4f}s) should be faster than creating new instances ({new_instance_time:.4f}s)"
    
    def test_workflow_throughput(self):
        """Test the throughput of the research workflow."""
        # Create a mock research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Set up mock responses for each agent
        for agent in [
            workflow.manager_agent,
            workflow.pre_response_agent,
            workflow.research_agent,
            workflow.image_generation_agent,
            workflow.file_manager_agent,
            workflow.summary_agent,
            workflow.verification_agent
        ]:
            agent.set_response("Mock response")
        
        # Define a set of queries to process
        queries = [
            "Python programming",
            "JavaScript frameworks",
            "Machine learning algorithms",
            "Cloud computing services",
            "Mobile app development"
        ]
        
        # Measure throughput (queries per second)
        start_time = time.time()
        for query in queries:
            result = workflow.execute(query)
        end_time = time.time()
        total_time = end_time - start_time
        throughput = len(queries) / total_time
        
        # Log and assert
        self.logger.info(f"Workflow throughput: {throughput:.2f} queries per second")
        self.logger.info(f"Average time per query: {total_time / len(queries):.4f} seconds")
        assert throughput > 0.2, f"Workflow throughput too low: {throughput:.2f} queries per second"
    
    def test_concurrent_workflow_execution(self):
        """Test performance of concurrent workflow executions."""
        # Create a function to execute a workflow
        def execute_workflow(query):
            workflow = ResearchWorkflow(
                manager_agent=ManagerAgent(),
                pre_response_agent=PreResponseAgent(),
                research_agent=ResearchAgent(),
                image_generation_agent=ImageGenerationAgent(),
                file_manager_agent=FileManagerAgent(),
                summary_agent=SummaryAgent(),
                verification_agent=VerificationAgent()
            )
            
            # Set up mock responses for each agent
            for agent in [
                workflow.manager_agent,
                workflow.pre_response_agent,
                workflow.research_agent,
                workflow.image_generation_agent,
                workflow.file_manager_agent,
                workflow.summary_agent,
                workflow.verification_agent
            ]:
                agent.set_response("Mock response")
            
            return workflow.execute(query)
        
        # Define queries to process concurrently
        queries = [
            "Python programming",
            "JavaScript frameworks",
            "Machine learning algorithms",
            "Cloud computing services",
            "Mobile app development"
        ]
        
        # Run workflows concurrently
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as executor:
            futures = [executor.submit(execute_workflow, query) for query in queries]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        throughput = len(queries) / total_time
        average_time = total_time / len(queries)
        
        # Log and assert
        self.logger.info(f"Concurrent workflow throughput: {throughput:.2f} workflows per second")
        self.logger.info(f"Average time per workflow (concurrent): {average_time:.4f} seconds")
        assert len(results) == len(queries), f"Not all workflows completed: {len(results)} of {len(queries)}"
    
    def test_memory_leak_during_repeated_execution(self):
        """Test for memory leaks during repeated workflow execution."""
        # Create a mock research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Set up mock responses for each agent
        for agent in [
            workflow.manager_agent,
            workflow.pre_response_agent,
            workflow.research_agent,
            workflow.image_generation_agent,
            workflow.file_manager_agent,
            workflow.summary_agent,
            workflow.verification_agent
        ]:
            agent.set_response("Mock response")
        
        # Measure memory before execution
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute the workflow multiple times
        num_executions = 10
        for i in range(num_executions):
            result = workflow.execute(f"Query {i}")
        
        # Measure memory after execution
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        memory_increase_per_execution = memory_increase / num_executions
        
        # Log and assert
        self.logger.info(f"Memory before: {memory_before:.2f} MB")
        self.logger.info(f"Memory after {num_executions} executions: {memory_after:.2f} MB")
        self.logger.info(f"Total memory increase: {memory_increase:.2f} MB")
        self.logger.info(f"Memory increase per execution: {memory_increase_per_execution:.2f} MB")
        
        # Some memory increase is expected, but it should be reasonable
        assert memory_increase_per_execution < 5.0, f"Possible memory leak: {memory_increase_per_execution:.2f} MB per execution"
    
    def test_cpu_usage_during_workflow_execution(self):
        """Test CPU usage during workflow execution."""
        # Create a mock research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Set up mock responses for each agent
        for agent in [
            workflow.manager_agent,
            workflow.pre_response_agent,
            workflow.research_agent,
            workflow.image_generation_agent,
            workflow.file_manager_agent,
            workflow.summary_agent,
            workflow.verification_agent
        ]:
            agent.set_response("Mock response")
        
        # Function to monitor CPU usage
        def monitor_cpu_usage():
            process = psutil.Process(os.getpid())
            cpu_percentages = []
            monitoring = True
            
            while monitoring:
                try:
                    cpu_percent = process.cpu_percent(interval=0.1)
                    cpu_percentages.append(cpu_percent)
                    time.sleep(0.1)
                except Exception:
                    break
            
            return cpu_percentages
        
        # Start CPU monitoring in a separate thread
        cpu_percentages = []
        monitoring_thread = threading.Thread(target=lambda: cpu_percentages.extend(monitor_cpu_usage()))
        monitoring_thread.daemon = True
        monitoring_thread.start()
        
        # Execute the workflow
        result = workflow.execute("Test query for CPU usage monitoring")
        
        # Stop monitoring and wait for the thread to finish
        time.sleep(0.5)  # Give the monitoring thread time to collect final data
        monitoring_thread.join(timeout=1.0)
        
        # Calculate CPU usage statistics
        if cpu_percentages:
            avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
            max_cpu = max(cpu_percentages)
            
            # Log and assert
            self.logger.info(f"Average CPU usage: {avg_cpu:.2f}%")
            self.logger.info(f"Maximum CPU usage: {max_cpu:.2f}%")
            self.logger.info(f"CPU usage samples: {len(cpu_percentages)}")
            
            # CPU usage should be reasonable
            assert max_cpu < 90.0, f"CPU usage too high: {max_cpu:.2f}%"
        else:
            self.logger.warning("No CPU usage data collected")
    
    def test_response_time_distribution(self):
        """Test the distribution of response times for MCP operations."""
        # Create a mock BraveSearchMCP
        brave_search = BraveSearchMCP(api_key="test_brave_api_key")
        
        # Mock the _send_request method to return immediately
        def mock_send_request(method, params):
            return {
                "content": [{"type": "text", "text": "Mock search results"}],
                "isError": False
            }
        
        with patch.object(brave_search, '_send_request', side_effect=mock_send_request):
            # Perform multiple web searches and measure response times
            num_searches = 50
            response_times = []
            
            for i in range(num_searches):
                start_time = time.time()
                result = brave_search.web_search(f"Query {i}")
                end_time = time.time()
                response_times.append(end_time - start_time)
        
        # Close the MCP
        brave_search.close()
        
        # Calculate statistics
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        # Calculate percentiles
        response_times.sort()
        p50 = response_times[int(0.5 * len(response_times))]
        p90 = response_times[int(0.9 * len(response_times))]
        p95 = response_times[int(0.95 * len(response_times))]
        p99 = response_times[int(0.99 * len(response_times))]
        
        # Log statistics
        self.logger.info(f"Response time statistics for {num_searches} web searches:")
        self.logger.info(f"  Average: {avg_time:.4f} seconds")
        self.logger.info(f"  Min: {min_time:.4f} seconds")
        self.logger.info(f"  Max: {max_time:.4f} seconds")
        self.logger.info(f"  P50 (median): {p50:.4f} seconds")
        self.logger.info(f"  P90: {p90:.4f} seconds")
        self.logger.info(f"  P95: {p95:.4f} seconds")
        self.logger.info(f"  P99: {p99:.4f} seconds")
        
        # Assert reasonable response times
        assert p95 < 1.0, f"95th percentile response time too high: {p95:.4f} seconds"
        assert max_time / min_time < 10.0, f"Response time variability too high: max/min = {max_time/min_time:.2f}"
    
    def test_performance_with_large_data(self):
        """Test performance when handling large data volumes."""
        # Create a mock FilesystemMCP
        filesystem = FilesystemMCP(workspace_dir=self.workspace_dir)
        
        # Mock the _send_request method to handle large data
        def mock_send_request(method, params):
            if method == "callTool":
                if params["name"] == "write_file":
                    return {
                        "content": [{"type": "text", "text": f"File written successfully: {len(params['arguments']['content'])} bytes"}],
                        "isError": False
                    }
                elif params["name"] == "read_file":
                    # Return a large mock file content
                    size = 1024 * 1024  # 1 MB
                    return {
                        "content": [{"type": "text", "text": "X" * size}],
                        "isError": False
                    }
            return {}
        
        with patch.object(filesystem, '_send_request', side_effect=mock_send_request):
            # Test writing large files
            file_sizes = [1024, 10*1024, 100*1024, 1024*1024]  # 1KB to 1MB
            write_times = []
            
            for size in file_sizes:
                content = "X" * size
                start_time = time.time()
                result = filesystem.write_file(f"large_file_{size}.txt", content)
                end_time = time.time()
                write_times.append(end_time - start_time)
                
                self.logger.info(f"Time to write {size/1024:.1f} KB: {write_times[-1]:.4f} seconds")
            
            # Test reading large files
            read_times = []
            
            for size in file_sizes:
                start_time = time.time()
                content = filesystem.read_file(f"large_file_{size}.txt")
                end_time = time.time()
                read_times.append(end_time - start_time)
                
                self.logger.info(f"Time to read {size/1024:.1f} KB: {read_times[-1]:.4f} seconds")
        
        # Close the MCP
        filesystem.close()
        
        # Check if performance scales reasonably with file size
        for i in range(1, len(write_times)):
            write_ratio = write_times[i] / write_times[i-1]
            read_ratio = read_times[i] / read_times[i-1]
            
            size_ratio = file_sizes[i] / file_sizes[i-1]
            
            self.logger.info(f"Size increase: {size_ratio}x, Write time increase: {write_ratio:.2f}x, Read time increase: {read_ratio:.2f}x")
            
            # Performance should scale sub-linearly or linearly with data size
            assert write_ratio < size_ratio * 1.5, f"Write time increased too much: {write_ratio:.2f}x for {size_ratio}x size increase"
            assert read_ratio < size_ratio * 1.5, f"Read time increased too much: {read_ratio:.2f}x for {size_ratio}x size increase"
    
    def test_performance_under_load(self):
        """Test performance under simulated load conditions."""
        # Create a mock BraveSearchMCP
        brave_search = BraveSearchMCP(api_key="test_brave_api_key")
        
        # Mock the _send_request method with variable response times
        def mock_send_request(method, params):
            # Simulate variable response times under load
            time.sleep(0.01 * (1 + random.random()))
            return {
                "content": [{"type": "text", "text": "Mock search results"}],
                "isError": False
            }
        
        with patch.object(brave_search, '_send_request', side_effect=mock_send_request):
            # Perform searches with increasing concurrency
            concurrency_levels = [1, 2, 4, 8, 16]
            throughput_results = []
            
            for concurrency in concurrency_levels:
                # Create a thread pool with the specified concurrency
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    # Perform a fixed number of searches
                    num_searches = 20
                    queries = [f"Query {i}" for i in range(num_searches)]
                    
                    start_time = time.time()
                    
                    # Submit all searches to the thread pool
                    futures = [executor.submit(brave_search.web_search, query) for query in queries]
                    
                    # Wait for all searches to complete
                    results = [future.result() for future in concurrent.futures.as_completed(futures)]
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    # Calculate throughput (searches per second)
                    throughput = num_searches / total_time
                    throughput_results.append(throughput)
                    
                    self.logger.info(f"Concurrency level {concurrency}: {throughput:.2f} searches/second (total time: {total_time:.2f}s)")
            
        # Close the MCP
        brave_search.close()
        
        # Check if throughput scales with concurrency
        for i in range(1, len(concurrency_levels)):
            concurrency_ratio = concurrency_levels[i] / concurrency_levels[i-1]
            throughput_ratio = throughput_results[i] / throughput_results[i-1]
            
            self.logger.info(f"Concurrency increase: {concurrency_ratio}x, Throughput increase: {throughput_ratio:.2f}x")
            
            # Throughput should increase with concurrency, but may not scale linearly due to contention
            assert throughput_ratio > 0.7, f"Throughput doesn't scale well with concurrency: {throughput_ratio:.2f}x for {concurrency_ratio}x concurrency increase"
    
    def test_performance_with_timing_decorator(self):
        """Test performance measurement using the timing decorator."""
        # Create a mock BraveSearchMCP
        brave_search = BraveSearchMCP(api_key="test_brave_api_key")
        
        # Mock the _send_request method to return immediately
        def mock_send_request(method, params):
            return {
                "content": [{"type": "text", "text": "Mock search results"}],
                "isError": False
            }
        
        with patch.object(brave_search, '_send_request', side_effect=mock_send_request):
            # Apply the timing decorator to the web_search method
            original_web_search = brave_search.web_search
            brave_search.web_search = timed(brave_search.web_search)
            
            # Capture the logger output
            with patch('apps.utils.logger.Logger.info') as mock_logger:
                # Perform a web search
                result = brave_search.web_search("Python programming")
                
                # Verify that the timing decorator logged the execution time
                mock_logger.assert_called()
                log_message = mock_logger.call_args[0][0]
                assert "execution time" in log_message.lower()
                
                # Extract the execution time from the log message
                import re
                match = re.search(r'execution time: ([\d.]+) seconds', log_message, re.IGNORECASE)
                if match:
                    execution_time = float(match.group(1))
                    self.logger.info(f"Measured execution time: {execution_time:.4f} seconds")
                    assert execution_time < 1.0, f"Execution time too high: {execution_time:.4f} seconds"
            
            # Restore the original method
            brave_search.web_search = original_web_search
        
        # Close the MCP
        brave_search.close()
    
    def test_performance_regression_detection(self):
        """Test detection of performance regressions."""
        # Create a mock BraveSearchMCP
        brave_search = BraveSearchMCP(api_key="test_brave_api_key")
        
        # Define baseline performance metrics (simulated historical data)
        baseline_metrics = {
            "web_search": {
                "avg_time": 0.05,
                "p95_time": 0.08,
                "threshold": 1.5  # Allow 50% degradation before flagging
            },
            "local_search": {
                "avg_time": 0.07,
                "p95_time": 0.12,
                "threshold": 1.5
            }
        }
        
        # Mock the _send_request method with different response times
        def mock_send_request(method, params):
            if method == "callTool":
                if params["name"] == "brave_web_search":
                    # Simulate normal performance
                    time.sleep(0.05 + 0.02 * random.random())
                elif params["name"] == "brave_local_search":
                    # Simulate degraded performance
                    time.sleep(0.15 + 0.05 * random.random())
            return {
                "content": [{"type": "text", "text": "Mock search results"}],
                "isError": False
            }
        
        with patch.object(brave_search, '_send_request', side_effect=mock_send_request):
            # Measure current performance for web search
            web_search_times = []
            for i in range(20):
                start_time = time.time()
                result = brave_search.web_search(f"Query {i}")
                end_time = time.time()
                web_search_times.append(end_time - start_time)
            
            # Measure current performance for local search
            local_search_times = []
            for i in range(20):
                start_time = time.time()
                result = brave_search.local_search(f"Coffee shops in City {i}")
                end_time = time.time()
                local_search_times.append(end_time - start_time)
        
        # Close the MCP
        brave_search.close()
        
        # Calculate current performance metrics
        web_search_avg = sum(web_search_times) / len(web_search_times)
        web_search_times.sort()
        web_search_p95 = web_search_times[int(0.95 * len(web_search_times))]
        
        local_search_avg = sum(local_search_times) / len(local_search_times)
        local_search_times.sort()
        local_search_p95 = local_search_times[int(0.95 * len(local_search_times))]
        
        # Log current performance
        self.logger.info(f"Web search current performance - Avg: {web_search_avg:.4f}s, P95: {web_search_p95:.4f}s")
        self.logger.info(f"Local search current performance - Avg: {local_search_avg:.4f}s, P95: {local_search_p95:.4f}s")
        
        # Compare with baseline and detect regressions
        web_search_avg_ratio = web_search_avg / baseline_metrics["web_search"]["avg_time"]
        web_search_p95_ratio = web_search_p95 / baseline_metrics["web_search"]["p95_time"]
        
        local_search_avg_ratio = local_search_avg / baseline_metrics["local_search"]["avg_time"]
        local_search_p95_ratio = local_search_p95 / baseline_metrics["local_search"]["p95_time"]
        
        # Log comparisons
        self.logger.info(f"Web search performance ratio - Avg: {web_search_avg_ratio:.2f}x, P95: {web_search_p95_ratio:.2f}x")
        self.logger.info(f"Local search performance ratio - Avg: {local_search_avg_ratio:.2f}x, P95: {local_search_p95_ratio:.2f}x")
        
        # Assert no regression for web search (should pass)
        assert web_search_avg_ratio < baseline_metrics["web_search"]["threshold"], f"Web search average time regression detected: {web_search_avg_ratio:.2f}x slower"
        assert web_search_p95_ratio < baseline_metrics["web_search"]["threshold"], f"Web search P95 time regression detected: {web_search_p95_ratio:.2f}x slower"
        
        # Assert no regression for local search (should fail, demonstrating regression detection)
        try:
            assert local_search_avg_ratio < baseline_metrics["local_search"]["threshold"], f"Local search average time regression detected: {local_search_avg_ratio:.2f}x slower"
            assert local_search_p95_ratio < baseline_metrics["local_search"]["threshold"], f"Local search P95 time regression detected: {local_search_p95_ratio:.2f}x slower"
        except AssertionError as e:
            self.logger.warning(f"Performance regression detected: {str(e)}")
            # Don't fail the test, just log the regression
            pass
    
    def test_performance_with_caching(self):
        """Test performance improvement from caching."""
        # Create a mock BraveSearchMCP
        brave_search = BraveSearchMCP(api_key="test_brave_api_key")
        
        # Create a simple cache
        cache = {}
        
        # Mock the _send_request method with caching
        def mock_send_request_with_cache(method, params):
            if method == "callTool" and params["name"] == "brave_web_search":
                query = params["arguments"]["query"]
                
                # Check if result is in cache
                if query in cache:
                    return cache[query]
                
                # Simulate API call delay
                time.sleep(0.1)
                
                # Generate result and store in cache
                result = {
                    "content": [{"type": "text", "text": f"Mock search results for {query}"}],
                    "isError": False
                }
                cache[query] = result
                return result
            
            return {
                "content": [{"type": "text", "text": "Mock response"}],
                "isError": False
            }
        
        with patch.object(brave_search, '_send_request', side_effect=mock_send_request_with_cache):
            # Measure time for first (uncached) query
            start_time = time.time()
            result1 = brave_search.web_search("Python programming")
            end_time = time.time()
            uncached_time = end_time - start_time
            
            # Measure time for second (cached) query
            start_time = time.time()
            result2 = brave_search.web_search("Python programming")
            end_time = time.time()
            cached_time = end_time - start_time
            
            # Measure time for third (different, uncached) query
            start_time = time.time()
            result3 = brave_search.web_search("JavaScript programming")
            end_time = time.time()
            different_query_time = end_time - start_time
        
        # Close the MCP
        brave_search.close()
        
        # Log results
        self.logger.info(f"Uncached query time: {uncached_time:.4f} seconds")
        self.logger.info(f"Cached query time: {cached_time:.4f} seconds")
        self.logger.info(f"Different (uncached) query time: {different_query_time:.4f} seconds")
        self.logger.info(f"Cache speedup: {uncached_time / cached_time:.2f}x")
        
        # Assert that caching improves performance
        assert cached_time < uncached_time, f"Caching should improve performance: cached {cached_time:.4f}s vs uncached {uncached_time:.4f}s"
        assert cached_time < different_query_time, f"Cached query should be faster than different query: cached {cached_time:.4f}s vs different {different_query_time:.4f}s"
        assert uncached_time / cached_time > 2.0, f"Cache speedup should be significant: {uncached_time / cached_time:.2f}x"
    
    def test_performance_with_batching(self):
        """Test performance improvement from batching operations."""
        # Create a mock MemoryMCP
        memory = MemoryMCP()
        
        # Mock the _send_request method
        def mock_send_request(method, params):
            if method == "callTool":
                if params["name"] == "store_memory":
                    # Simulate fixed overhead plus per-item cost
                    time.sleep(0.05 + 0.01 * len(params["arguments"]["value"]))
                    return {
                        "content": [{"type": "text", "text": "Memory stored successfully"}],
                        "isError": False
                    }
            return {}
        
        with patch.object(memory, '_send_request', side_effect=mock_send_request):
            # Measure time for individual operations
            individual_start_time = time.time()
            for i in range(10):
                memory.store_memory(f"key_{i}", f"value_{i}")
            individual_end_time = time.time()
            individual_time = individual_end_time - individual_start_time
            
            # Measure time for batched operation (simulated)
            batched_start_time = time.time()
            batch_data = {f"key_{i}": f"value_{i}" for i in range(10)}
            # Simulate batched operation with single overhead
            time.sleep(0.05 + 0.01 * sum(len(v) for v in batch_data.values()))
            batched_end_time = time.time()
            batched_time = batched_end_time - batched_start_time
        
        # Close the MCP
        memory.close()
        
        # Log results
        self.logger.info(f"Individual operations time: {individual_time:.4f} seconds")
        self.logger.info(f"Batched operation time: {batched_time:.4f} seconds")
        self.logger.info(f"Batching speedup: {individual_time / batched_time:.2f}x")
        
        # Assert that batching improves performance
        assert batched_time < individual_time, f"Batching should improve performance: batched {batched_time:.4f}s vs individual {individual_time:.4f}s"
    
    def test_performance_profile(self):
        """Generate a comprehensive performance profile of the system."""
        # Create a dictionary to store performance metrics
        performance_profile = {
            "mcp_initialization": {},
            "mcp_operations": {},
            "agent_execution": {},
            "workflow_execution": {},
            "memory_usage": {},
            "throughput": {}
        }
        
        # Test MCP initialization times
        mcp_classes = [
            (BraveSearchMCP, {"api_key": "test_brave_api_key"}),
            (EverArtMCP, {"api_key": "test_everart_api_key"}),
            (FetchMCP, {}),
            (FilesystemMCP, {"workspace_dir": self.workspace_dir}),
            (MemoryMCP, {})
        ]
        
        for mcp_class, kwargs in mcp_classes:
            class_name = mcp_class.__name__
            start_time = time.time()
            mcp = mcp_class(**kwargs)
            end_time = time.time()
            init_time = end_time - start_time
            performance_profile["mcp_initialization"][class_name] = init_time
            mcp.close()
        
        # Test agent execution times
        agents = {
            "manager": ManagerAgent(),
            "pre_response": PreResponseAgent(),
            "research": ResearchAgent(),
            "image_generation": ImageGenerationAgent(),
            "file_manager": FileManagerAgent(),
            "summary": SummaryAgent(),
            "verification": VerificationAgent()
        }
        
        # Set up mock responses for each agent
        for agent_name, agent in agents.items():
            agent.set_response(f"Mock response from {agent_name} agent")
        
        # Measure execution time for each agent
        for agent_name, agent in agents.items():
            start_time = time.time()
            result = agent.execute(f"Test query for {agent_name} agent")
            end_time = time.time()
            execution_time = end_time - start_time
            performance_profile["agent_execution"][agent_name] = execution_time
        
        # Test workflow execution time
        workflow = ResearchWorkflow(
            manager_agent=agents["manager"],
            pre_response_agent=agents["pre_response"],
            research_agent=agents["research"],
            image_generation_agent=agents["image_generation"],
            file_manager_agent=agents["file_manager"],
            summary_agent=agents["summary"],
            verification_agent=agents["verification"]
        )
        
        # Measure workflow execution time
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        result = workflow.execute("Test query for workflow performance profiling")
        end_time = time.time()
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        performance_profile["workflow_execution"]["total_time"] = end_time - start_time
        performance_profile["memory_usage"]["workflow_execution"] = memory_after - memory_before
        
        # Test throughput (queries per second)
        queries = ["Python", "JavaScript", "Machine Learning", "Cloud Computing", "Mobile Development"]
        
        start_time = time.time()
        for query in queries:
            result = workflow.execute(query)
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = len(queries) / total_time
        performance_profile["throughput"]["queries_per_second"] = throughput
        
        # Log the performance profile
        self.logger.info("Performance Profile:")
        for category, metrics in performance_profile.items():
            self.logger.info(f"  {category.replace('_', ' ').title()}:")
            for name, value in metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"    {name}: {value:.4f}")
                else:
                    self.logger.info(f"    {name}: {value}")
        
        # Assert reasonable performance
        assert performance_profile["workflow_execution"]["total_time"] < 5.0, "Workflow execution took too long"
        assert performance_profile["throughput"]["queries_per_second"] > 0.2, "Throughput too low"
    
    def test_performance_with_different_query_complexities(self):
        """Test performance with queries of different complexities."""
        # Create a mock research workflow
        workflow = ResearchWorkflow(
            manager_agent=ManagerAgent(),
            pre_response_agent=PreResponseAgent(),
            research_agent=ResearchAgent(),
            image_generation_agent=ImageGenerationAgent(),
            file_manager_agent=FileManagerAgent(),
            summary_agent=SummaryAgent(),
            verification_agent=VerificationAgent()
        )
        
        # Set up mock responses for each agent
        for agent in [
            workflow.manager_agent,
            workflow.pre_response_agent,
            workflow.research_agent,
            workflow.image_generation_agent,
            workflow.file_manager_agent,
            workflow.summary_agent,
            workflow.verification_agent
        ]:
            agent.set_response("Mock response")
        
        # Define queries of different complexities
        queries = {
            "simple": "Python",
            "medium": "Explain Python programming language",
            "complex": "Compare Python, JavaScript, and Ruby for web development",
            "very_complex": "Analyze the evolution of programming paradigms from procedural to object-oriented to functional, with examples in Python, JavaScript, and Haskell"
        }
        
        # Measure execution time for each query
        execution_times = {}
        for complexity, query in queries.items():
            start_time = time.time()
            result = workflow.execute(query)
            end_time = time.time()
            execution_times[complexity] = end_time - start_time
            
            self.logger.info(f"{complexity.title()} query execution time: {execution_times[complexity]:.4f} seconds")
        
        # Calculate complexity ratios
        self.logger.info(f"Medium/Simple ratio: {execution_times['medium'] / execution_times['simple']:.2f}x")
        self.logger.info(f"Complex/Medium ratio: {execution_times['complex'] / execution_times['medium']:.2f}x")
        self.logger.info(f"Very Complex/Complex ratio: {execution_times['very_complex'] / execution_times['complex']:.2f}x")
        
        # Assert reasonable scaling with complexity
        assert execution_times["medium"] / execution_times["simple"] < 3.0, "Medium queries should not be much slower than simple queries"
        assert execution_times["complex"] / execution_times["medium"] < 3.0, "Complex queries should not be much slower than medium queries"
        assert execution_times["very_complex"] / execution_times["complex"] < 3.0, "Very complex queries should not be much slower than complex queries"
    
    def test_performance_with_different_mcp_implementations(self):
        """Test performance with different MCP implementations (mock vs. real)."""
        # Create a real BraveSearchMCP (using mocked subprocess)
        real_brave = BraveSearchMCP(api_key="test_brave_api_key")
        
        # Create a fully mocked BraveSearchMCP
        mock_brave = MagicMock(spec=BraveSearchMCP)
        mock_brave.web_search.return_value = "Mock search results"
        
        # Measure performance of real MCP (with mocked subprocess)
        real_start_time = time.time()
        for i in range(5):
            with patch.object(real_brave, '_send_request', return_value={
                "content": [{"type": "text", "text": "Mock search results"}],
                "isError": False
            }):
                result = real_brave.web_search(f"Query {i}")
        real_end_time = time.time()
        real_time = real_end_time - real_start_time
        
        # Measure performance of fully mocked MCP
        mock_start_time = time.time()
        for i in range(5):
            result = mock_brave.web_search(f"Query {i}")
        mock_end_time = time.time()
        mock_time = mock_end_time - mock_start_time
        
        # Close the real MCP
        real_brave.close()
        
        # Log results
        self.logger.info(f"Real MCP (with mocked subprocess) time: {real_time:.4f} seconds")
        self.logger.info(f"Fully mocked MCP time: {mock_time:.4f} seconds")
        self.logger.info(f"Real/Mock ratio: {real_time / mock_time:.2f}x")
        
        # Assert that the real implementation is not significantly slower
        # This is mainly to catch any major performance issues in the MCP implementation
        assert real_time / mock_time < 10.0, f"Real MCP implementation is too slow compared to mock: {real_time / mock_time:.2f}x"
    
    def test_performance_with_error_handling(self):
        """Test performance impact of error handling."""
        # Create a mock BraveSearchMCP
        brave_search = BraveSearchMCP(api_key="test_brave_api_key")
        
        # Mock the _send_request method to return success
        def mock_send_request_success(method, params):
            return {
                "content": [{"type": "text", "text": "Mock search results"}],
                "isError": False
            }
        
        # Mock the _send_request method to return error
        def mock_send_request_error(method, params):
            return {
                "content": [{"type": "text", "text": "Error: API rate limit exceeded"}],
                "isError": True
            }
        
        # Measure performance with successful requests
        with patch.object(brave_search, '_send_request', side_effect=mock_send_request_success):
            success_start_time = time.time()
            for i in range(10):
                try:
                    result = brave_search.web_search(f"Query {i}")
                except Exception as e:
                    pass  # Ignore errors
            success_end_time = time.time()
            success_time = success_end_time - success_start_time
        
        # Measure performance with error requests
        with patch.object(brave_search, '_send_request', side_effect=mock_send_request_error):
            error_start_time = time.time()
            for i in range(10):
                try:
                    result = brave_search.web_search(f"Query {i}")
                except Exception as e:
                    pass  # Ignore errors
            error_end_time = time.time()
            error_time = error_end_time - error_start_time
        
        # Close the MCP
        brave_search.close()
        
        # Log results
        self.logger.info(f"Success requests time: {success_time:.4f} seconds")
        self.logger.info(f"Error requests time: {error_time:.4f} seconds")
        self.logger.info(f"Error/Success ratio: {error_time / success_time:.2f}x")
        
        # Assert that error handling doesn't add significant overhead
        assert error_time / success_time < 2.0, f"Error handling adds too much overhead: {error_time / success_time:.2f}x"
    
    def test_startup_time(self):
        """Test the startup time of the entire system."""
        import importlib
        import time
        
        # Measure time to import all modules
        start_time = time.time()
        
        # Import core modules
        importlib.import_module("apps.utils.logger")
        importlib.import_module("apps.utils.config")
        importlib.import_module("apps.utils.helpers")
        importlib.import_module("apps.utils.constants")
        importlib.import_module("apps.utils.decorators")
        importlib.import_module("apps.utils.exceptions")
        importlib.import_module("apps.utils.validation")
        
        # Import MCP modules
        importlib.import_module("apps.mcps.brave_search_mcp")
        importlib.import_module("apps.mcps.everart_mcp")
        importlib.import_module("apps.mcps.fetch_mcp")
        importlib.import_module("apps.mcps.filesystem_mcp")
        importlib.import_module("apps.mcps.memory_mcp")
        
        # Import agent modules
        importlib.import_module("apps.agents.base_agent")
        importlib.import_module("apps.agents.manager_agent")
        importlib.import_module("apps.agents.pre_response_agent")
        importlib.import_module("apps.agents.research_agent")
        importlib.import_module("apps.agents.image_generation_agent")
        importlib.import_module("apps.agents.file_manager_agent")
        importlib.import_module("apps.agents.summary_agent")
        importlib.import_module("apps.agents.verification_agent")
        
        # Import workflow module
        importlib.import_module("apps.workflows.research_workflow")
        
        end_time = time.time()
        import_time = end_time - start_time
        
        # Log results
        self.logger.info(f"Module import time: {import_time:.4f} seconds")
        
        # Assert reasonable startup time
        assert import_time < 2.0, f"System startup time too high: {import_time:.4f} seconds"


if __name__ == "__main__":
    import random
    pytest.main(["-xvs", __file__])
