"""
Research Workflow

This module implements a comprehensive research workflow that coordinates multiple specialized
agents to perform research tasks, from query clarification to final output generation.
"""

import os
import time
import uuid
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, TypedDict
from datetime import datetime

# Import agents
from apps.agents.manager_agent import ManagerAgent
from apps.agents.pre_response_agent import PreResponseAgent
from apps.agents.research_agent import ResearchAgent
from apps.agents.summary_agent import SummaryAgent
from apps.agents.verification_agent import VerificationAgent
from apps.agents.image_generation_agent import ImageGenerationAgent
from apps.agents.file_manager_agent import FileManagerAgent

# Import MCP clients
from apps.mcps.brave_search_mcp import BraveSearchMCP
from apps.mcps.fetch_mcp import FetchMCP
from apps.mcps.everart_mcp import EverArtMCP
from apps.mcps.filesystem_mcp import FilesystemMCP
from apps.mcps.memory_mcp import MemoryMCP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("research_workflow")


# -----------------------------------------------------------------------------
# Type Definitions
# -----------------------------------------------------------------------------

class ResearchRequest(TypedDict):
    """Research request structure."""
    query: str
    session_id: Optional[str]
    user_id: Optional[str]
    parameters: Optional[Dict[str, Any]]


class ResearchResponse(TypedDict):
    """Research response structure."""
    success: bool
    query: str
    summary: str
    sources: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    files: List[Dict[str, Any]]
    execution_stats: Dict[str, Any]
    errors: List[Dict[str, Any]]


class ResearchParameters(TypedDict, total=False):
    """Parameters for the research workflow."""
    max_sources: int
    include_images: bool
    verify_facts: bool
    save_results: bool
    research_depth: str  # "basic", "standard", "deep"
    output_format: str  # "markdown", "json", "text"
    workspace_dir: Optional[str]
    namespace: str


# -----------------------------------------------------------------------------
# Research Workflow Class
# -----------------------------------------------------------------------------

class ResearchWorkflow:
    """
    Orchestrates the research workflow using multiple specialized agents.
    
    The workflow consists of the following steps:
    1. Query clarification and research planning
    2. Web search and content retrieval
    3. Content summarization and fact verification
    4. Image generation for visual aids
    5. File organization and storage
    6. Final output generation
    """
    
    def __init__(self, workspace_dir: Optional[str] = None):
        """
        Initialize the research workflow.
        
        Args:
            workspace_dir: Optional directory to use as the workspace root.
                           If not provided, a temporary directory will be created.
        """
        self.workspace_dir = workspace_dir or os.path.join(os.getcwd(), "research_workspace")
        self.session_id = str(uuid.uuid4())
        
        # Ensure workspace directory exists
        os.makedirs(self.workspace_dir, exist_ok=True)
        
        # Initialize MCP clients
        self.brave_search = None
        self.fetch = None
        self.everart = None
        self.filesystem = None
        self.memory = None
        
        # Initialize agents
        self.manager_agent = None
        self.pre_response_agent = None
        self.research_agent = None
        self.summary_agent = None
        self.verification_agent = None
        self.image_generation_agent = None
        self.file_manager_agent = None
    
    def _initialize_mcps(self):
        """Initialize MCP clients if not already initialized."""
        # Initialize Brave Search MCP
        if not self.brave_search:
            brave_api_key = os.environ.get("BRAVE_API_KEY")
            if not brave_api_key:
                logger.warning("BRAVE_API_KEY not found in environment variables")
            else:
                self.brave_search = BraveSearchMCP(api_key=brave_api_key)
        
        # Initialize Fetch MCP
        if not self.fetch:
            self.fetch = FetchMCP()
        
        # Initialize EverArt MCP
        if not self.everart:
            everart_api_key = os.environ.get("EVERART_API_KEY")
            if not everart_api_key:
                logger.warning("EVERART_API_KEY not found in environment variables")
            else:
                self.everart = EverArtMCP(api_key=everart_api_key)
        
        # Initialize Filesystem MCP
        if not self.filesystem:
            self.filesystem = FilesystemMCP(workspace_dir=self.workspace_dir)
        
        # Initialize Memory MCP
        if not self.memory:
            memory_storage_path = os.path.join(self.workspace_dir, "memory_storage")
            os.makedirs(memory_storage_path, exist_ok=True)
            self.memory = MemoryMCP(storage_path=memory_storage_path)
    
    def _initialize_agents(self):
        """Initialize agent instances if not already initialized."""
        # Initialize Manager Agent
        if not self.manager_agent:
            self.manager_agent = ManagerAgent()
        
        # Initialize Pre-response Agent
        if not self.pre_response_agent:
            self.pre_response_agent = PreResponseAgent()
        
        # Initialize Research Agent
        if not self.research_agent:
            self.research_agent = ResearchAgent()
        
        # Initialize Summary Agent
        if not self.summary_agent:
            self.summary_agent = SummaryAgent()
        
        # Initialize Verification Agent
        if not self.verification_agent:
            self.verification_agent = VerificationAgent()
        
        # Initialize Image Generation Agent
        if not self.image_generation_agent:
            self.image_generation_agent = ImageGenerationAgent()
        
        # Initialize File Manager Agent
        if not self.file_manager_agent:
            self.file_manager_agent = FileManagerAgent()
    
    def _store_research_plan(self, query: str, plan: Dict[str, Any], namespace: str = "research"):
        """
        Store the research plan in memory.
        
        Args:
            query: Research query
            plan: Research plan
            namespace: Memory namespace
        """
        if not self.memory:
            logger.warning("Memory MCP not initialized, cannot store research plan")
            return
        
        plan_key = f"research_plan_{self.session_id}"
        plan_value = json.dumps({
            "query": query,
            "plan": plan,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id
        })
        
        self.memory.store_memory(plan_key, plan_value, namespace=namespace)
        logger.info(f"Stored research plan with key: {plan_key}")
    
    def _store_research_results(self, query: str, results: Dict[str, Any], namespace: str = "research"):
        """
        Store the research results in memory.
        
        Args:
            query: Research query
            results: Research results
            namespace: Memory namespace
        """
        if not self.memory:
            logger.warning("Memory MCP not initialized, cannot store research results")
            return
        
        results_key = f"research_results_{self.session_id}"
        results_value = json.dumps({
            "query": query,
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id
        })
        
        self.memory.store_memory(results_key, results_value, namespace=namespace)
        logger.info(f"Stored research results with key: {results_key}")
    
    def _create_research_directory(self, query: str):
        """
        Create a structured directory for the research.
        
        Args:
            query: Research query
            
        Returns:
            Path to the research directory
        """
        if not self.file_manager_agent:
            logger.warning("File Manager Agent not initialized, cannot create research directory")
            return None
        
        response = self.file_manager_agent.create_research_directory(
            topic=query,
            session_id=self.session_id
        )
        
        if not response["success"]:
            logger.error(f"Failed to create research directory: {response['message']}")
            return None
        
        research_dir = response["directory_info"]["path"]
        logger.info(f"Created research directory: {research_dir}")
        return research_dir
    
    def _save_research_summary(self, summary: str, query: str, research_dir: str):
        """
        Save the research summary to a file.
        
        Args:
            summary: Research summary
            query: Research query
            research_dir: Research directory
            
        Returns:
            File info for the saved summary
        """
        if not self.file_manager_agent:
            logger.warning("File Manager Agent not initialized, cannot save research summary")
            return None
        
        response = self.file_manager_agent.save_research_summary(
            content=summary,
            topic=query,
            research_dir=research_dir,
            namespace="research",
            session_id=self.session_id
        )
        
        if not response["success"]:
            logger.error(f"Failed to save research summary: {response['message']}")
            return None
        
        logger.info(f"Saved research summary to {response['file_info']['path']}")
        return response["file_info"]
    
    def _save_research_image(self, image_url: str, description: str, query: str, research_dir: str):
        """
        Save a research image to a file.
        
        Args:
            image_url: URL of the image
            description: Description of the image
            query: Research query
            research_dir: Research directory
            
        Returns:
            File info for the saved image
        """
        if not self.file_manager_agent:
            logger.warning("File Manager Agent not initialized, cannot save research image")
            return None
        
        response = self.file_manager_agent.save_research_image(
            image_url=image_url,
            description=description,
            topic=query,
            research_dir=research_dir,
            namespace="research",
            session_id=self.session_id
        )
        
        if not response["success"]:
            logger.error(f"Failed to save research image: {response['message']}")
            return None
        
        logger.info(f"Saved research image to {response['file_info']['path']}")
        return response["file_info"]
    
    def _save_source_content(self, url: str, content: str, query: str, research_dir: str):
        """
        Save source content to a file.
        
        Args:
            url: Source URL
            content: Source content
            query: Research query
            research_dir: Research directory
            
        Returns:
            File info for the saved source
        """
        if not self.file_manager_agent:
            logger.warning("File Manager Agent not initialized, cannot save source content")
            return None
        
        # Create a sanitized filename from the URL
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace("www.", "")
        path = parsed_url.path.replace("/", "_").strip("_")
        if not path:
            path = "index"
        filename = f"{domain}_{path}.txt"
        
        file_path = os.path.join(research_dir, "sources", filename)
        
        metadata = {
            "source_url": url,
            "query": query,
            "retrieved_at": datetime.now().isoformat(),
            "session_id": self.session_id
        }
        
        response = self.file_manager_agent.save_file({
            "content": content,
            "path": file_path,
            "metadata": metadata,
            "overwrite": True,
            "namespace": "research",
            "session_id": self.session_id
        })
        
        if not response["success"]:
            logger.error(f"Failed to save source content: {response['message']}")
            return None
        
        logger.info(f"Saved source content to {response['file_info']['path']}")
        return response["file_info"]
    
    def _generate_final_output(self, query: str, summary: str, sources: List[Dict[str, Any]], 
                              images: List[Dict[str, Any]], research_dir: str, 
                              output_format: str = "markdown"):
        """
        Generate the final research output.
        
        Args:
            query: Research query
            summary: Research summary
            sources: List of sources
            images: List of images
            research_dir: Research directory
            output_format: Output format (markdown, json, text)
            
        Returns:
            Path to the final output file
        """
        if not self.file_manager_agent:
            logger.warning("File Manager Agent not initialized, cannot generate final output")
            return None
        
        # Create the final output content based on format
        if output_format == "json":
            content = json.dumps({
                "query": query,
                "summary": summary,
                "sources": sources,
                "images": images,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id
            }, indent=2)
            file_extension = "json"
        
        elif output_format == "text":
            content = f"Research: {query}\n\n"
            content += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            content += f"Summary:\n{summary}\n\n"
            
            if sources:
                content += "Sources:\n"
                for i, source in enumerate(sources, 1):
                    content += f"{i}. {source.get('title', 'Untitled')} - {source.get('url', 'No URL')}\n"
            
            if images:
                content += "\nImages:\n"
                for i, image in enumerate(images, 1):
                    content += f"{i}. {image.get('description', 'No description')} - {image.get('url', 'No URL')}\n"
            
            file_extension = "txt"
        
        else:  # markdown (default)
            content = f"# Research: {query}\n\n"
            content += f"*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            content += f"## Summary\n\n{summary}\n\n"
            
            if sources:
                content += "## Sources\n\n"
                for i, source in enumerate(sources, 1):
                    content += f"{i}. [{source.get('title', 'Untitled')}]({source.get('url', '#')})\n"
            
            if images:
                content += "\n## Images\n\n"
                for image in images:
                    content += f"![{image.get('description', 'Research image')}]({image.get('path', '#')})\n\n"
                    content += f"*{image.get('description', 'No description')}*\n\n"
            
            file_extension = "md"
        
        # Save the final output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_query = query.lower().replace(" ", "_")[:30]
        file_name = f"research_{sanitized_query}_{timestamp}.{file_extension}"
        file_path = os.path.join(research_dir, "final_output", file_name)
        
        response = self.file_manager_agent.save_file({
            "content": content,
            "path": file_path,
            "metadata": {
                "query": query,
                "format": output_format,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id
            },
            "overwrite": True,
            "namespace": "research",
            "session_id": self.session_id
        })
        
        if not response["success"]:
            logger.error(f"Failed to save final output: {response['message']}")
            return None
        
        logger.info(f"Saved final output to {response['file_info']['path']}")
        return response["file_info"]["path"]
    
    def execute(self, request: ResearchRequest) -> ResearchResponse:
        """
        Execute the research workflow.
        
        Args:
            request: Research request
            
        Returns:
            Research response
        """
        start_time = time.time()
        
        # Initialize response
        response: ResearchResponse = {
            "success": False,
            "query": request["query"],
            "summary": "",
            "sources": [],
            "images": [],
            "files": [],
            "execution_stats": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 0,
                "session_id": self.session_id
            },
            "errors": []
        }
        
        # Set session ID if provided
        if request.get("session_id"):
            self.session_id = request["session_id"]
        
        # Extract parameters
        parameters: ResearchParameters = request.get("parameters", {})
        max_sources = parameters.get("max_sources", 5)
        include_images = parameters.get("include_images", True)
        verify_facts = parameters.get("verify_facts", True)
        save_results = parameters.get("save_results", True)
        research_depth = parameters.get("research_depth", "standard")
        output_format = parameters.get("output_format", "markdown")
        namespace = parameters.get("namespace", "research")
        
        # Set workspace directory if provided
        if parameters.get("workspace_dir"):
            self.workspace_dir = parameters["workspace_dir"]
            os.makedirs(self.workspace_dir, exist_ok=True)
        
        try:
            # Initialize MCPs and agents
            self._initialize_mcps()
            self._initialize_agents()
            
            # Step 1: Query clarification and research planning
            logger.info(f"Step 1: Query clarification and research planning for '{request['query']}'")
            
            pre_response_request = {
                "task": "clarify_query",
                "parameters": {
                    "query": request["query"],
                    "research_depth": research_depth
                },
                "session_id": self.session_id,
                "namespace": namespace
            }
            
            pre_response_result = self.pre_response_agent.process_request(pre_response_request)
            
            if not pre_response_result["success"]:
                error_msg = f"Query clarification failed: {pre_response_result['message']}"
                logger.error(error_msg)
                response["errors"].append({
                    "type": "query_clarification_error",
                    "error": error_msg
                })
                return response
            
            # Extract clarified query and research plan
            clarified_query = pre_response_result["data"].get("clarified_query", request["query"])
            research_plan = pre_response_result["data"].get("research_plan", {})
            
            # Store the research plan
            self._store_research_plan(clarified_query, research_plan, namespace=namespace)
            
            # Step 2: Create research directory structure
            logger.info("Step 2: Creating research directory structure")
            research_dir = self._create_research_directory(clarified_query)
            
            if not research_dir:
                error_msg = "Failed to create research directory"
                logger.error(error_msg)
                response["errors"].append({
                    "type": "directory_creation_error",
                    "error": error_msg
                })
                return response
            
            # Step 3: Web search and content retrieval
            logger.info("Step 3: Web search and content retrieval")
            
            research_request = {
                "task": "research_topic",
                "parameters": {
                    "query": clarified_query,
                    "max_sources": max_sources,
                    "research_depth": research_depth
                },
                "session_id": self.session_id,
                "namespace": namespace
            }
            
            research_result = self.research_agent.process_request(research_request)
            
            if not research_result["success"]:
                error_msg = f"Research failed: {research_result['message']}"
                logger.error(error_msg)
                response["errors"].append({
                    "type": "research_error",
                    "error": error_msg
                })
                return response
            
            # Extract sources and content
            sources = research_result["data"].get("sources", [])
            
            # Save source content
            saved_sources = []
            for source in sources:
                if source.get("content"):
                    file_info = self._save_source_content(
                        url=source["url"],
                        content=source["content"],
                        query=clarified_query,
                        research_dir=research_dir
                    )
                    
                    if file_info:
                        source["file_path"] = file_info["path"]
                        saved_sources.append(source)
            
            # Step 4: Content summarization
            logger.info("Step 4: Content summarization")
            
            summary_request = {
                "task": "summarize_research",
                "parameters": {
                    "query": clarified_query,
                    "sources": sources,
                    "research_depth": research_depth
                },
                "session_id": self.session_id,
                "namespace": namespace
            }
            
            summary_result = self.summary_agent.process_request(summary_request)
            
            if not summary_result["success"]:
                error_msg = f"Summarization failed: {summary_result['message']}"
                logger.error(error_msg)
                response["errors"].append({
                    "type": "summarization_error",
                    "error": error_msg
                })
                return response
            
            # Extract summary
            summary = summary_result["data"].get("summary", "")
            
            # Step 5: Fact verification (if enabled)
            if verify_facts:
                logger.info("Step 5: Fact verification")
                
                verification_request = {
                    "task": "verify_facts",
                    "parameters": {
                        "query": clarified_query,
                        "summary": summary,
                        "sources": sources
                    },
                    "session_id": self.session_id,
                    "namespace": namespace
                }
                
                verification_result = self.verification_agent.process_request(verification_request)
                
                if verification_result["success"]:
                    # Update summary with verified facts
                    verified_summary = verification_result["data"].get("verified_summary")
                    if verified_summary:
                        summary = verified_summary
                    
                    # Add verification notes
                    verification_notes = verification_result["data"].get("verification_notes", [])
                    if verification_notes:
                        summary += "\n\n## Verification Notes\n\n"
                        for note in verification_notes:
                            summary += f"- {note}\n"
                else:
                    logger.warning(f"Fact verification had issues: {verification_result['message']}")
                    response["errors"].append({
                        "type": "verification_warning",
                        "error": verification_result["message"]
                    })
            
            # Step 6: Image generation (if enabled)
            images = []
            if include_images:
                logger.info("Step 6: Image generation")
                
                image_request = {
                    "task": "generate_images",
                    "parameters": {
                        "query": clarified_query,
                        "summary": summary,
                        "num_images": 2
                    },
                    "session_id": self.session_id,
                    "namespace": namespace
                }
                
                image_result = self.image_generation_agent.process_request(image_request)
                
                if image_result["success"]:
                    generated_images = image_result["data"].get("images", [])
                    
                    # Save images
                    for image in generated_images:
                        if image.get("url"):
                            file_info = self._save_research_image(
                                image_url=image["url"],
                                description=image.get("description", "Research image"),
                                query=clarified_query,
                                research_dir=research_dir
                            )
                            
                            if file_info:
                                image["file_path"] = file_info["path"]
                                images.append(image)
                else:
                    logger.warning(f"Image generation had issues: {image_result['message']}")
                    response["errors"].append({
                        "type": "image_generation_warning",
                        "error": image_result["message"]
                    })
            
            # Step 7: Save research summary
            logger.info("Step 7: Saving research summary")
            summary_file = self._save_research_summary(
                summary=summary,
                query=clarified_query,
                research_dir=research_dir
            )
            
            # Step 8: Generate final output
            logger.info("Step 8: Generating final output")
            final_output_path = self._generate_final_output(
                query=clarified_query,
                summary=summary,
                sources=sources,
                images=images,
                research_dir=research_dir,
                output_format=output_format
            )
            
            # Store research results
            self._store_research_results(
                query=clarified_query,
                results={
                    "summary": summary,
                    "sources": sources,
                    "images": images,
                    "final_output_path": final_output_path
                },
                namespace=namespace
            )
            
            # Prepare files list for response
            files = []
            if summary_file:
                files.append(summary_file)
            
            if final_output_path:
                files.append({
                    "path": final_output_path,
                    "type": "final_output",
                    "format": output_format
                })
            
            # Update response
            response["success"] = True
            response["query"] = clarified_query
            response["summary"] = summary
            response["sources"] = sources
            response["images"] = images
            response["files"] = files
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            
            return response
        
        except Exception as e:
            logger.error(f"Error in research workflow: {str(e)}", exc_info=True)
            response["success"] = False
            response["errors"].append({
                "type": "workflow_error",
                "error": str(e)
            })
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            return response
        
        finally:
            # Clean up resources
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources used by the workflow."""
        # Clean up agents
        if self.manager_agent:
            self.manager_agent.cleanup()
        
        if self.pre_response_agent:
            self.pre_response_agent.cleanup()
        
        if self.research_agent:
            self.research_agent.cleanup()
        
        if self.summary_agent:
            self.summary_agent.cleanup()
        
        if self.verification_agent:
            self.verification_agent.cleanup()
        
        if self.image_generation_agent:
            self.image_generation_agent.cleanup()
        
        if self.file_manager_agent:
            self.file_manager_agent.cleanup()
        
        # Close MCP clients
        if self.brave_search:
            self.brave_search.close()
        
        if self.fetch:
            self.fetch.close()
        
        if self.everart:
            self.everart.close()
        
        if self.filesystem:
            self.filesystem.close()
        
        if self.memory:
            self.memory.close()


# -----------------------------------------------------------------------------
# Command-line Interface
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Execute a research workflow")
    parser.add_argument("query", type=str, help="Research query")
    parser.add_argument("--workspace", type=str, help="Workspace directory")
    parser.add_argument("--max-sources", type=int, default=5, help="Maximum number of sources")
    parser.add_argument("--no-images", action="store_true", help="Disable image generation")
    parser.add_argument("--no-verify", action="store_true", help="Disable fact verification")
    parser.add_argument("--depth", type=str, choices=["basic", "standard", "deep"], default="standard", 
                      help="Research depth")
    parser.add_argument("--format", type=str, choices=["markdown", "json", "text"], default="markdown", 
                      help="Output format")
    parser.add_argument("--session-id", type=str, help="Session ID for continuity")
    args = parser.parse_args()
    
    # Create the workflow
    workflow = ResearchWorkflow(workspace_dir=args.workspace)
    
    # Prepare the request
    request: ResearchRequest = {
        "query": args.query,
        "session_id": args.session_id,
        "user_id": None,
        "parameters": {
            "max_sources": args.max_sources,
            "include_images": not args.no_images,
            "verify_facts": not args.no_verify,
            "save_results": True,
            "research_depth": args.depth,
            "output_format": args.format,
            "workspace_dir": args.workspace
        }
    }
    
    try:
        # Execute the workflow
        response = workflow.execute(request)
        
        # Print the results
        if response["success"]:
            print(f"\nResearch: {response['query']}")
            print(f"Duration: {response['execution_stats']['duration_seconds']} seconds")
            
            print("\nSummary:")
            print(response["summary"])
            
            print("\nSources:")
            for i, source in enumerate(response["sources"], 1):
                print(f"{i}. {source.get('title', 'Untitled')} - {source.get('url', 'No URL')}")
            
            if response["images"]:
                print("\nImages:")
                for i, image in enumerate(response["images"], 1):
                    print(f"{i}. {image.get('description', 'No description')} - {image.get('url', 'No URL')}")
            
            if response["files"]:
                print("\nFiles:")
                for i, file in enumerate(response["files"], 1):
                    print(f"{i}. {file.get('path', 'Unknown path')}")
            
            print(f"\nFull results saved to: {response['files'][-1]['path'] if response['files'] else 'N/A'}")
        else:
            print(f"Research failed: {response['errors'][0]['error'] if response['errors'] else 'Unknown error'}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nResearch workflow interrupted.")
        workflow.cleanup()
        sys.exit(1)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


def run_interactive_mode():
    """Run the research workflow in interactive mode."""
    print("Research Workflow - Interactive Mode")
    print("===================================")
    
    # Get workspace directory
    workspace_dir = input("Enter workspace directory (leave empty for default): ")
    if not workspace_dir:
        workspace_dir = os.path.join(os.getcwd(), "research_workspace")
    
    # Create the workflow
    workflow = ResearchWorkflow(workspace_dir=workspace_dir)
    
    try:
        while True:
            # Get research query
            query = input("\nEnter research query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            
            # Get research parameters
            print("\nResearch Parameters:")
            max_sources = int(input("Maximum number of sources (default: 5): ") or "5")
            include_images = input("Include images? (y/n, default: y): ").lower() != 'n'
            verify_facts = input("Verify facts? (y/n, default: y): ").lower() != 'n'
            
            depth_options = {
                "1": "basic",
                "2": "standard",
                "3": "deep"
            }
            print("Research depth:")
            print("1. Basic - Quick overview")
            print("2. Standard - Comprehensive research")
            print("3. Deep - In-depth analysis")
            depth_choice = input("Choose depth (1-3, default: 2): ") or "2"
            research_depth = depth_options.get(depth_choice, "standard")
            
            format_options = {
                "1": "markdown",
                "2": "json",
                "3": "text"
            }
            print("Output format:")
            print("1. Markdown - Formatted with headings and links")
            print("2. JSON - Structured data format")
            print("3. Text - Plain text format")
            format_choice = input("Choose format (1-3, default: 1): ") or "1"
            output_format = format_options.get(format_choice, "markdown")
            
            # Prepare the request
            request: ResearchRequest = {
                "query": query,
                "session_id": str(uuid.uuid4()),
                "user_id": None,
                "parameters": {
                    "max_sources": max_sources,
                    "include_images": include_images,
                    "verify_facts": verify_facts,
                    "save_results": True,
                    "research_depth": research_depth,
                    "output_format": output_format,
                    "workspace_dir": workspace_dir
                }
            }
            
            print(f"\nStarting research on: {query}")
            print("This may take a few minutes...")
            
            # Execute the workflow
            response = workflow.execute(request)
            
            # Print the results
            if response["success"]:
                print(f"\nResearch: {response['query']}")
                print(f"Duration: {response['execution_stats']['duration_seconds']} seconds")
                
                print("\nSummary:")
                print(response["summary"])
                
                print("\nSources:")
                for i, source in enumerate(response["sources"], 1):
                    print(f"{i}. {source.get('title', 'Untitled')} - {source.get('url', 'No URL')}")
                
                if response["images"]:
                    print("\nImages:")
                    for i, image in enumerate(response["images"], 1):
                        print(f"{i}. {image.get('description', 'No description')} - {image.get('url', 'No URL')}")
                
                if response["files"]:
                    print("\nFiles:")
                    for i, file in enumerate(response["files"], 1):
                        print(f"{i}. {file.get('path', 'Unknown path')}")
                
                print(f"\nFull results saved to: {response['files'][-1]['path'] if response['files'] else 'N/A'}")
            else:
                print(f"Research failed: {response['errors'][0]['error'] if response['errors'] else 'Unknown error'}")
    
    except KeyboardInterrupt:
        print("\nResearch workflow interrupted.")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Clean up
        workflow.cleanup()
        print("\nThank you for using the Research Workflow!")


if __name__ == "__main__" and len(sys.argv) == 1:
    # If no arguments provided, run in interactive mode
    run_interactive_mode()
