"""
Manager Agent

This agent coordinates the workflow between all other agents, deciding which agents to call
based on the user's query and managing the overall research process.

It uses LangGraph for workflow orchestration and Memory MCP for state persistence.
"""

import os
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Literal, TypedDict, cast

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Import MCP clients
from apps.mcps.memory_mcp import MemoryMCP
from apps.mcps.brave_search_mcp import BraveSearchMCP
from apps.mcps.fetch_mcp import FetchMCP
from apps.mcps.everart_mcp import EverArtMCP
from apps.mcps.filesystem_mcp import FilesystemMCP

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("manager_agent")

# -----------------------------------------------------------------------------
# State Definitions
# -----------------------------------------------------------------------------

class ResearchState(TypedDict):
    """State maintained throughout the research workflow."""
    query: str  # Original user query
    conversation_history: List[Dict[str, Any]]  # Full conversation history
    research_plan: Optional[Dict[str, Any]]  # Structured research plan
    search_results: List[Dict[str, Any]]  # Results from web searches
    fetched_content: List[Dict[str, Any]]  # Content fetched from URLs
    summaries: List[Dict[str, Any]]  # Summaries of fetched content
    verified_facts: List[Dict[str, Any]]  # Facts that have been verified
    generated_images: List[Dict[str, Any]]  # Images generated for the research
    file_locations: Dict[str, str]  # Map of file types to their locations
    current_agent: str  # Currently active agent
    status: str  # Current status of the research process
    errors: List[Dict[str, Any]]  # Errors encountered during processing


# -----------------------------------------------------------------------------
# MCP Client Management
# -----------------------------------------------------------------------------

class MCPClientManager:
    """Manages connections to various MCP services."""
    
    def __init__(self):
        """Initialize MCP client manager."""
        self.memory_mcp = None
        self.brave_search_mcp = None
        self.fetch_mcp = None
        self.everart_mcp = None
        self.filesystem_mcp = None
    
    def get_memory_mcp(self) -> MemoryMCP:
        """Get or create Memory MCP client."""
        if self.memory_mcp is None:
            storage_path = os.environ.get("MEMORY_STORAGE_PATH", "./memory_storage")
            self.memory_mcp = MemoryMCP(storage_path=storage_path)
        return self.memory_mcp
    
    def get_brave_search_mcp(self) -> BraveSearchMCP:
        """Get or create Brave Search MCP client."""
        if self.brave_search_mcp is None:
            api_key = os.environ.get("BRAVE_API_KEY")
            if not api_key:
                raise ValueError("BRAVE_API_KEY environment variable is required")
            self.brave_search_mcp = BraveSearchMCP(api_key=api_key)
        return self.brave_search_mcp
    
    def get_fetch_mcp(self) -> FetchMCP:
        """Get or create Fetch MCP client."""
        if self.fetch_mcp is None:
            self.fetch_mcp = FetchMCP()
        return self.fetch_mcp
    
    def get_everart_mcp(self) -> EverArtMCP:
        """Get or create EverArt MCP client."""
        if self.everart_mcp is None:
            api_key = os.environ.get("EVERART_API_KEY")
            if not api_key:
                raise ValueError("EVERART_API_KEY environment variable is required")
            self.everart_mcp = EverArtMCP(api_key=api_key)
        return self.everart_mcp
    
    def get_filesystem_mcp(self) -> FilesystemMCP:
        """Get or create Filesystem MCP client."""
        if self.filesystem_mcp is None:
            workspace_dir = os.environ.get("WORKSPACE_DIR", "./research_workspace")
            self.filesystem_mcp = FilesystemMCP(workspace_dir=workspace_dir)
        return self.filesystem_mcp
    
    def close_all(self):
        """Close all MCP clients."""
        if self.memory_mcp:
            self.memory_mcp.close()
        if self.brave_search_mcp:
            self.brave_search_mcp.close()
        if self.fetch_mcp:
            self.fetch_mcp.close()
        if self.everart_mcp:
            self.everart_mcp.close()
        if self.filesystem_mcp:
            self.filesystem_mcp.close()


# Create a singleton instance
mcp_manager = MCPClientManager()


# -----------------------------------------------------------------------------
# Agent Nodes
# -----------------------------------------------------------------------------

def create_llm(model: str = "gpt-4o", temperature: float = 0.2):
    """Create a language model instance."""
    return ChatOpenAI(model=model, temperature=temperature)


def analyze_query(state: ResearchState) -> ResearchState:
    """
    Analyze the user query and create a research plan.
    
    Args:
        state: Current research state
        
    Returns:
        Updated state with research plan
    """
    llm = create_llm()
    
    # Create prompt for analyzing the query
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a research planning assistant. Your task is to analyze the user's query 
        and create a detailed research plan. The plan should include:
        
        1. Main research questions to answer
        2. Subtopics to explore
        3. Types of information needed (facts, statistics, opinions, etc.)
        4. Potential search queries to use
        5. Visual elements that might enhance the research
        
        Structure your response as a JSON object with these fields."""),
        HumanMessage(content="{query}")
    ])
    
    # Execute the prompt
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"query": state["query"]})
    
    try:
        # Parse the result as JSON
        research_plan = json.loads(result)
        
        # Update the state
        new_state = state.copy()
        new_state["research_plan"] = research_plan
        new_state["status"] = "plan_created"
        new_state["current_agent"] = "manager_agent"
        
        # Store the plan in memory MCP
        memory_mcp = mcp_manager.get_memory_mcp()
        memory_mcp.store_memory(
            key=f"research_plan_{uuid.uuid4()}",
            value=json.dumps(research_plan),
            namespace=state["query"]
        )
        
        return new_state
    
    except json.JSONDecodeError as e:
        # Handle parsing error
        logger.error(f"Error parsing research plan: {e}")
        new_state = state.copy()
        new_state["errors"].append({
            "agent": "manager_agent",
            "function": "analyze_query",
            "error": str(e),
            "raw_output": result
        })
        new_state["status"] = "plan_creation_failed"
        return new_state


def determine_next_agent(state: ResearchState) -> Dict[str, Any]:
    """
    Determine which agent should be called next based on the current state.
    
    Args:
        state: Current research state
        
    Returns:
        Decision about which agent to call next
    """
    # Define the decision logic based on the current state
    if state["status"] == "plan_created":
        # After creating a plan, consult with the user via pre-response agent
        return {"next": "pre_response_agent"}
    
    elif state["status"] == "plan_approved":
        # If plan is approved, start research
        return {"next": "research_agent"}
    
    elif state["status"] == "research_completed" and not state["summaries"]:
        # If research is done but no summaries yet, create summaries
        return {"next": "summary_agent"}
    
    elif state["status"] == "summaries_completed" and not state["verified_facts"]:
        # If summaries are done but facts not verified, verify facts
        return {"next": "verification_agent"}
    
    elif state["status"] == "verification_completed" and not state["generated_images"]:
        # If verification is done but no images, generate images
        return {"next": "image_generation_agent"}
    
    elif state["status"] == "images_generated" and not state["file_locations"]:
        # If images are generated but files not organized, organize files
        return {"next": "file_manager_agent"}
    
    elif state["status"] == "files_organized":
        # If everything is done, end the workflow
        return {"next": END}
    
    else:
        # If status doesn't match expected flow or there's an error
        if state["errors"]:
            # Handle errors by determining recovery path
            return handle_errors(state)
        else:
            # Default to research agent if unsure
            return {"next": "research_agent"}


def handle_errors(state: ResearchState) -> Dict[str, Any]:
    """
    Handle errors in the workflow and determine recovery path.
    
    Args:
        state: Current research state with errors
        
    Returns:
        Decision about how to recover
    """
    # Get the most recent error
    latest_error = state["errors"][-1]
    error_agent = latest_error["agent"]
    
    # Define recovery strategies based on which agent had an error
    if error_agent == "research_agent":
        # If research failed, try with different search terms
        return {"next": "research_agent", "retry": True}
    
    elif error_agent == "summary_agent":
        # If summarization failed, try with simpler content
        return {"next": "summary_agent", "retry": True}
    
    elif error_agent == "verification_agent":
        # If verification failed, proceed without verification
        new_state = state.copy()
        new_state["status"] = "verification_skipped"
        return {"next": "image_generation_agent"}
    
    elif error_agent == "image_generation_agent":
        # If image generation failed, proceed without images
        new_state = state.copy()
        new_state["status"] = "images_skipped"
        return {"next": "file_manager_agent"}
    
    else:
        # For other errors, end the workflow with what we have
        return {"next": END}


# -----------------------------------------------------------------------------
# Pre-Response Agent Node
# -----------------------------------------------------------------------------

def pre_response_agent(state: ResearchState) -> ResearchState:
    """
    Present the research plan to the user and get approval.
    
    Args:
        state: Current research state
        
    Returns:
        Updated state with user approval
    """
    llm = create_llm()
    
    # Create prompt for presenting the plan
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a research assistant communicating with the user. 
        Present the research plan in a clear, organized way and ask for their approval or suggestions.
        Be concise but informative."""),
        HumanMessage(content="Original query: {query}"),
        MessagesPlaceholder(variable_name="conversation_history"),
        SystemMessage(content="Research plan: {research_plan}")
    ])
    
    # Execute the prompt
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "query": state["query"],
        "conversation_history": state["conversation_history"],
        "research_plan": json.dumps(state["research_plan"], indent=2)
    })
    
    # In a real implementation, we would wait for user feedback here
    # For this example, we'll simulate user approval
    
    # Update the state
    new_state = state.copy()
    new_state["conversation_history"].append({
        "role": "assistant",
        "content": result
    })
    new_state["conversation_history"].append({
        "role": "user",
        "content": "The plan looks good. Please proceed with the research."
    })
    new_state["status"] = "plan_approved"
    new_state["current_agent"] = "pre_response_agent"
    
    return new_state


# -----------------------------------------------------------------------------
# Research Agent Node
# -----------------------------------------------------------------------------

def research_agent(state: ResearchState) -> ResearchState:
    """
    Perform web searches and fetch content based on the research plan.
    
    Args:
        state: Current research state
        
    Returns:
        Updated state with search results and fetched content
    """
    # Get the research plan
    research_plan = state["research_plan"]
    
    # Get MCP clients
    brave_search = mcp_manager.get_brave_search_mcp()
    fetch = mcp_manager.get_fetch_mcp()
    
    # Extract search queries from the research plan
    search_queries = research_plan.get("potential_search_queries", [])
    if not search_queries and "main_questions" in research_plan:
        # If no explicit search queries, use main questions
        search_queries = research_plan["main_questions"]
    
    # Ensure we have at least one query
    if not search_queries:
        search_queries = [state["query"]]
    
    # Perform searches
    search_results = []
    for query in search_queries[:3]:  # Limit to first 3 queries for efficiency
        try:
            # Perform web search
            results = brave_search.web_search(query, count=5)
            
            # Extract URLs from results
            urls = []
            for line in results.split('\n\n'):
                if line.startswith('URL:'):
                    url = line.replace('URL:', '').strip()
                    urls.append(url)
            
            search_results.append({
                "query": query,
                "raw_results": results,
                "urls": urls
            })
            
        except Exception as e:
            logger.error(f"Error performing search for query '{query}': {e}")
            # Continue with other queries despite error
    
    # Fetch content from top URLs
    fetched_content = []
    all_urls = [url for result in search_results for url in result["urls"]]
    unique_urls = list(dict.fromkeys(all_urls))  # Remove duplicates while preserving order
    
    for url in unique_urls[:5]:  # Limit to first 5 URLs for efficiency
        try:
            # Fetch text content
            content = fetch.fetch_text(url)
            
            fetched_content.append({
                "url": url,
                "content": content[:10000]  # Limit content size
            })
            
        except Exception as e:
            logger.error(f"Error fetching content from URL '{url}': {e}")
            # Continue with other URLs despite error
    
    # Update the state
    new_state = state.copy()
    new_state["search_results"] = search_results
    new_state["fetched_content"] = fetched_content
    new_state["status"] = "research_completed"
    new_state["current_agent"] = "research_agent"
    
    # Store results in memory MCP
    memory_mcp = mcp_manager.get_memory_mcp()
    memory_mcp.store_memory(
        key=f"search_results_{uuid.uuid4()}",
        value=json.dumps(search_results),
        namespace=state["query"]
    )
    memory_mcp.store_memory(
        key=f"fetched_content_{uuid.uuid4()}",
        value=json.dumps(fetched_content),
        namespace=state["query"]
    )
    
    return new_state


# -----------------------------------------------------------------------------
# Summary Agent Node
# -----------------------------------------------------------------------------

def summary_agent(state: ResearchState) -> ResearchState:
    """
    Summarize the fetched content using the language model.
    
    Args:
        state: Current research state
        
    Returns:
        Updated state with content summaries
    """
    llm = create_llm()
    
    # Create prompt for summarization
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a research summarization assistant. Your task is to create a 
        concise but comprehensive summary of the provided content. Focus on extracting key facts, 
        insights, and information relevant to the original query. Structure your summary with clear 
        sections and bullet points where appropriate."""),
        HumanMessage(content="Original query: {query}"),
        HumanMessage(content="Research plan: {research_plan}"),
        HumanMessage(content="Content to summarize: {content}"),
    ])
    
    # Summarize each piece of fetched content
    summaries = []
    for item in state["fetched_content"]:
        try:
            # Execute the prompt
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({
                "query": state["query"],
                "research_plan": json.dumps(state["research_plan"]),
                "content": item["content"][:4000]  # Limit content size for the prompt
            })
            
            summaries.append({
                "url": item["url"],
                "summary": result
            })
            
        except Exception as e:
            logger.error(f"Error summarizing content from URL '{item['url']}': {e}")
            # Continue with other content despite error
    
    # Create an overall summary
    if summaries:
        try:
            # Combine individual summaries
            combined_summaries = "\n\n".join([f"Source: {s['url']}\n{s['summary']}" for s in summaries])
            
            # Create overall summary prompt
            overall_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a research synthesis assistant. Your task is to create a 
                cohesive overall summary from multiple source summaries. Identify common themes, 
                reconcile any contradictions, and present a unified view of the research topic.
                Structure your response with clear sections addressing the main research questions."""),
                HumanMessage(content="Original query: {query}"),
                HumanMessage(content="Research plan: {research_plan}"),
                HumanMessage(content="Individual summaries:\n{summaries}"),
            ])
            
            # Execute the prompt
            chain = overall_prompt | llm | StrOutputParser()
            overall_result = chain.invoke({
                "query": state["query"],
                "research_plan": json.dumps(state["research_plan"]),
                "summaries": combined_summaries
            })
            
            # Add overall summary
            summaries.append({
                "url": "overall_summary",
                "summary": overall_result
            })
            
        except Exception as e:
            logger.error(f"Error creating overall summary: {e}")
    
    # Update the state
    new_state = state.copy()
    new_state["summaries"] = summaries
    new_state["status"] = "summaries_completed"
    new_state["current_agent"] = "summary_agent"
    
    # Store summaries in memory MCP
    memory_mcp = mcp_manager.get_memory_mcp()
    memory_mcp.store_memory(
        key=f"summaries_{uuid.uuid4()}",
        value=json.dumps(summaries),
        namespace=state["query"]
    )
    
    return new_state


# -----------------------------------------------------------------------------
# Verification Agent Node
# -----------------------------------------------------------------------------

def verification_agent(state: ResearchState) -> ResearchState:
    """
    Verify facts by searching for additional sources.
    
    Args:
        state: Current research state
        
    Returns:
        Updated state with verified facts
    """
    llm = create_llm()
    
    # First, extract key facts from summaries
    facts_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a fact extraction assistant. Your task is to identify 
        the key factual claims from the provided summaries. Focus on specific, verifiable 
        statements rather than general observations or opinions. List each fact as a separate 
        item in a JSON array."""),
        HumanMessage(content="Summaries to analyze: {summaries}")
    ])
    
    # Combine all summaries
    combined_summaries = "\n\n".join([f"Source: {s['url']}\n{s['summary']}" for s in state["summaries"]])
    
    # Extract facts
    chain = facts_prompt | llm | StrOutputParser()
    facts_result = chain.invoke({"summaries": combined_summaries})
    
    try:
        # Parse the facts
        facts = json.loads(facts_result)
        if not isinstance(facts, list):
            facts = [facts]  # Ensure we have a list
    except json.JSONDecodeError:
        # If parsing fails, extract facts manually
        facts = []
        for line in facts_result.split('\n'):
            line = line.strip()
            if line and not line.startswith(('#', '-', '*', '{')):
                facts.append(line)
    
    # Get Brave Search MCP
    brave_search = mcp_manager.get_brave_search_mcp()
    
    # Verify each fact
    verified_facts = []
    for fact in facts[:10]:  # Limit to 10 facts for efficiency
        fact_text = fact if isinstance(fact, str) else json.dumps(fact)
        
        try:
            # Search for verification
            verification_query = f"verify {fact_text}"
            results = brave_search.web_search(verification_query, count=3)
            
            # Analyze verification results
            verification_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a fact verification assistant. Your task is to 
                determine whether the search results support, contradict, or are neutral about 
                the given fact. Provide your assessment and reasoning."""),
                HumanMessage(content="Fact to verify: {fact}"),
                HumanMessage(content="Search results: {results}")
            ])
            
            # Execute verification
            chain = verification_prompt | llm | StrOutputParser()
            verification_result = chain.invoke({
                "fact": fact_text,
                "results": results
            })
            
            verified_facts.append({
                "fact": fact_text,
                "verification_results": results,
                "verification_analysis": verification_result
            })
            
        except Exception as e:
            logger.error(f"Error verifying fact '{fact_text}': {e}")
            # Continue with other facts despite error
    
    # Update the state
    new_state = state.copy()
    new_state["verified_facts"] = verified_facts
    new_state["status"] = "verification_completed"
    new_state["current_agent"] = "verification_agent"
    
    # Store verified facts in memory MCP
    memory_mcp = mcp_manager.get_memory_mcp()
    memory_mcp.store_memory(
        key=f"verified_facts_{uuid.uuid4()}",
        value=json.dumps(verified_facts),
        namespace=state["query"]
    )
    
    return new_state


# -----------------------------------------------------------------------------
# Image Generation Agent Node
# -----------------------------------------------------------------------------

def image_generation_agent(state: ResearchState) -> ResearchState:
    """
    Generate images based on the research findings.
    
    Args:
        state: Current research state
        
    Returns:
        Updated state with generated images
    """
    llm = create_llm()
    
    # Create prompt for image generation ideas
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an image concept designer. Your task is to create 
        detailed prompts for generating images that would enhance the research findings. 
        Focus on visualizing key concepts, data, or relationships identified in the research.
        For each image, provide a detailed description that could be used as a prompt for 
        an AI image generator. Return a JSON array of image prompts."""),
        HumanMessage(content="Original query: {query}"),
        HumanMessage(content="Research findings: {findings}")
    ])
    
    # Combine research findings
    findings = ""
    if state["summaries"]:
        overall_summary = next((s["summary"] for s in state["summaries"] if s["url"] == "overall_summary"), "")
        findings += f"Overall Summary:\n{overall_summary}\n\n"
    
    if state["verified_facts"]:
        facts_text = "\n".join([f"- {f['fact']}" for f in state["verified_facts"]])
        findings += f"Verified Facts:\n{facts_text}"
    
    # Generate image prompts
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "query": state["query"],
        "findings": findings
    })
    
    try:
        # Parse image prompts
        image_prompts = json.loads(result)
        if not isinstance(image_prompts, list):
            image_prompts = [image_prompts]  # Ensure we have a list
    except json.JSONDecodeError:
        # If parsing fails, extract prompts manually
        image_prompts = []
        current_prompt = ""
        for line in result.split('\n'):
            line = line.strip()
            if line.startswith(('Image', 'Prompt', 'Visualization', 'Figure', 'Diagram')):
                if current_prompt:
                    image_prompts.append(current_prompt)
                current_prompt = line
            elif line and current_prompt:
                current_prompt += " " + line
        if current_prompt:
            image_prompts.append(current_prompt)
    
    # Get EverArt MCP
    everart = mcp_manager.get_everart_mcp()
    
    # Generate images
    generated_images = []
    for prompt in image_prompts[:3]:  # Limit to 3 images for efficiency
        prompt_text = prompt if isinstance(prompt, str) else json.dumps(prompt)
        
        try:
            # Generate image
            result = everart.generate_image(
                prompt=prompt_text,
                style="realistic",
                aspect_ratio="16:9",
                num_images=1
            )
            
            # Extract image URL from result
            image_url = None
            for line in result.split('\n'):
                if line.startswith(('http://', 'https://')):
                    image_url = line.strip()
                    break
            
            if image_url:
                generated_images.append({
                    "prompt": prompt_text,
                    "image_url": image_url
                })
            
        except Exception as e:
            logger.error(f"Error generating image for prompt '{prompt_text}': {e}")
            # Continue with other prompts despite error
    
    # Update the state
    new_state = state.copy()
    new_state["generated_images"] = generated_images
    new_state["status"] = "images_generated"
    new_state["current_agent"] = "image_generation_agent"
    
    # Store generated images in memory MCP
    memory_mcp = mcp_manager.get_memory_mcp()
    memory_mcp.store_memory(
        key=f"generated_images_{uuid.uuid4()}",
        value=json.dumps(generated_images),
        namespace=state["query"]
    )
    
    return new_state


# -----------------------------------------------------------------------------
# File Manager Agent Node
# -----------------------------------------------------------------------------

def file_manager_agent(state: ResearchState) -> ResearchState:
    """
    Organize and store research materials in a structured file system.
    
    Args:
        state: Current research state
        
    Returns:
        Updated state with file locations
    """
    # Get Filesystem MCP
    filesystem = mcp_manager.get_filesystem_mcp()
    
    # Create directory structure
    research_id = str(uuid.uuid4())[:8]
    base_dir = f"research_{research_id}"
    
    try:
        # Create main directory
        filesystem.create_directory(base_dir)
        
        # Create subdirectories
        subdirs = ["summaries", "images", "raw_data", "verified_facts"]
        for subdir in subdirs:
            filesystem.create_directory(f"{base_dir}/{subdir}")
        
        # Store file locations
        file_locations = {
            "base_directory": base_dir
        }
        
        # Save summaries
        if state["summaries"]:
            summary_files = []
            for i, summary in enumerate(state["summaries"]):
                filename = f"{base_dir}/summaries/summary_{i}.md"
                content = f"# Summary for {summary['url']}\n\n{summary['summary']}"
                filesystem.write_file(filename, content)
                summary_files.append(filename)
            
            # Save overall summary separately
            overall_summary = next((s for s in state["summaries"] if s["url"] == "overall_summary"), None)
            if overall_summary:
                filename = f"{base_dir}/overall_summary.md"
                content = f"# Overall Research Summary\n\nQuery: {state['query']}\n\n{overall_summary['summary']}"
                filesystem.write_file(filename, content)
                file_locations["overall_summary"] = filename
            
            file_locations["summaries"] = summary_files
        
        # Save verified facts
        if state["verified_facts"]:
            facts_filename = f"{base_dir}/verified_facts/facts.md"
            facts_content = "# Verified Facts\n\n"
            for fact in state["verified_facts"]:
                facts_content += f"## Fact\n{fact['fact']}\n\n"
                facts_content += f"### Verification\n{fact['verification_analysis']}\n\n"
            
            filesystem.write_file(facts_filename, facts_content)
            file_locations["verified_facts"] = facts_filename
        
        # Save image information
        if state["generated_images"]:
            images_filename = f"{base_dir}/images/image_catalog.md"
            images_content = "# Generated Images\n\n"
            for i, image in enumerate(state["generated_images"]):
                images_content += f"## Image {i+1}\n\n"
                images_content += f"Prompt: {image['prompt']}\n\n"
                images_content += f"URL: {image['image_url']}\n\n"
                images_content += f"![Image {i+1}]({image['image_url']})\n\n"
            
            filesystem.write_file(images_filename, images_content)
            file_locations["images"] = images_filename
        
        # Save raw search results
        if state["search_results"]:
            raw_data_filename = f"{base_dir}/raw_data/search_results.json"
            filesystem.write_file(raw_data_filename, json.dumps(state["search_results"], indent=2))
            file_locations["search_results"] = raw_data_filename
        
        # Save fetched content
        if state["fetched_content"]:
            for i, content_item in enumerate(state["fetched_content"]):
                content_filename = f"{base_dir}/raw_data/content_{i}.txt"
                filesystem.write_file(content_filename, content_item["content"])
            
            # Create an index file
            index_filename = f"{base_dir}/raw_data/content_index.md"
            index_content = "# Content Index\n\n"
            for i, content_item in enumerate(state["fetched_content"]):
                index_content += f"{i+1}. [{content_item['url']}](content_{i}.txt)\n"
            
            filesystem.write_file(index_filename, index_content)
            file_locations["content_index"] = index_filename
        
        # Create a README file
        readme_filename = f"{base_dir}/README.md"
        readme_content = f"""# Research: {state['query']}

## Overview
This directory contains research materials related to the query: "{state['query']}"

## Contents
- `overall_summary.md`: Comprehensive summary of all findings
- `summaries/`: Individual summaries of each source
- `verified_facts/`: Facts that have been verified with additional sources
- `images/`: Generated images related to the research
- `raw_data/`: Raw search results and fetched content

## Research ID
{research_id}
"""
        filesystem.write_file(readme_filename, readme_content)
        file_locations["readme"] = readme_filename
        
    except Exception as e:
        logger.error(f"Error organizing files: {e}")
        # Add error to state but continue
        new_state = state.copy()
        new_state["errors"].append({
            "agent": "file_manager_agent",
            "function": "file_manager_agent",
            "error": str(e)
        })
        return new_state
    
    # Update the state
    new_state = state.copy()
    new_state["file_locations"] = file_locations
    new_state["status"] = "files_organized"
    new_state["current_agent"] = "file_manager_agent"
    
    # Store file locations in memory MCP
    memory_mcp = mcp_manager.get_memory_mcp()
    memory_mcp.store_memory(
        key=f"file_locations_{uuid.uuid4()}",
        value=json.dumps(file_locations),
        namespace=state["query"]
    )
    
    return new_state


# -----------------------------------------------------------------------------
# LangGraph Workflow Definition
# -----------------------------------------------------------------------------

def create_research_workflow() -> StateGraph:
    """
    Create the research workflow graph.
    
    Returns:
        StateGraph: The workflow graph
    """
    # Define the workflow
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("pre_response_agent", pre_response_agent)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("summary_agent", summary_agent)
    workflow.add_node("verification_agent", verification_agent)
    workflow.add_node("image_generation_agent", image_generation_agent)
    workflow.add_node("file_manager_agent", file_manager_agent)
    
    # Add conditional routing node
    workflow.add_router("determine_next_agent", determine_next_agent)
    
    # Define the edges
    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "determine_next_agent")
    workflow.add_edge("pre_response_agent", "determine_next_agent")
    workflow.add_edge("research_agent", "determine_next_agent")
    workflow.add_edge("summary_agent", "determine_next_agent")
    workflow.add_edge("verification_agent", "determine_next_agent")
    workflow.add_edge("image_generation_agent", "determine_next_agent")
    workflow.add_edge("file_manager_agent", "determine_next_agent")
    
    # Connect router to agent nodes
    workflow.add_conditional_edges(
        "determine_next_agent",
        lambda x: x["next"],
        {
            "pre_response_agent": "pre_response_agent",
            "research_agent": "research_agent",
            "summary_agent": "summary_agent",
            "verification_agent": "verification_agent",
            "image_generation_agent": "image_generation_agent",
            "file_manager_agent": "file_manager_agent",
            END: END
        }
    )
    
    # Compile the workflow
    return workflow.compile()


# -----------------------------------------------------------------------------
# Manager Agent Class
# -----------------------------------------------------------------------------

class ManagerAgent:
    """
    Manager Agent that coordinates the research workflow.
    """
    
    def __init__(self, memory_storage_path: Optional[str] = None):
        """
        Initialize the Manager Agent.
        
        Args:
            memory_storage_path: Optional path for storing memory data
        """
        # Set up memory storage path
        if memory_storage_path:
            os.environ["MEMORY_STORAGE_PATH"] = memory_storage_path
        
        # Create the workflow
        self.workflow = create_research_workflow()
        
        # Set up memory saver for checkpointing
        self.memory_saver = MemorySaver()
    
    def start_research(self, query: str) -> Dict[str, Any]:
        """
        Start a new research workflow.
        
        Args:
            query: The user's research query
            
        Returns:
            Final state of the research workflow
        """
        # Initialize the state
        initial_state = {
            "query": query,
            "conversation_history": [
                {"role": "user", "content": query}
            ],
            "research_plan": None,
            "search_results": [],
            "fetched_content": [],
            "summaries": [],
            "verified_facts": [],
            "generated_images": [],
            "file_locations": {},
            "current_agent": "manager_agent",
            "status": "initialized",
            "errors": []
        }
        
        # Create a unique thread ID for this research
        thread_id = f"research_{uuid.uuid4()}"
        
        try:
            # Run the workflow with checkpointing
            config = {"configurable": {"thread_id": thread_id}}
            final_state = self.workflow.invoke(
                initial_state,
                config=config,
                checkpoint_saver=self.memory_saver
            )
            
            return final_state
        
        except Exception as e:
            logger.error(f"Error in research workflow: {e}")
            # Return the state with error
            return {
                **initial_state,
                "status": "failed",
                "errors": [
                    {
                        "agent": "manager_agent",
                        "function": "start_research",
                        "error": str(e)
                    }
                ]
            }
    
    def resume_research(self, thread_id: str) -> Dict[str, Any]:
        """
        Resume a research workflow from a checkpoint.
        
        Args:
            thread_id: The thread ID of the research to resume
            
        Returns:
            Final state of the research workflow
        """
        try:
            # Run the workflow from the checkpoint
            config = {"configurable": {"thread_id": thread_id}}
            final_state = self.workflow.invoke(
                None,  # State will be loaded from checkpoint
                config=config,
                checkpoint_saver=self.memory_saver
            )
            
            return final_state
        
        except Exception as e:
            logger.error(f"Error resuming research workflow: {e}")
            return {
                "status": "resume_failed",
                "errors": [
                    {
                        "agent": "manager_agent",
                        "function": "resume_research",
                        "error": str(e)
                    }
                ]
            }
    
    def get_research_summary(self, thread_id: str) -> Dict[str, Any]:
        """
        Get a summary of a research workflow.
        
        Args:
            thread_id: The thread ID of the research
            
        Returns:
            Summary of the research
        """
        try:
            # Load the checkpoint
            checkpoint = self.memory_saver.get(thread_id)
            if not checkpoint:
                return {"error": "Research not found"}
            
            # Extract relevant information
            state = checkpoint["state"]
            
            # Create a summary
            summary = {
                "query": state.get("query", ""),
                "status": state.get("status", "unknown"),
                "current_agent": state.get("current_agent", ""),
                "has_research_plan": bool(state.get("research_plan")),
                "search_results_count": len(state.get("search_results", [])),
                "fetched_content_count": len(state.get("fetched_content", [])),
                "summaries_count": len(state.get("summaries", [])),
                "verified_facts_count": len(state.get("verified_facts", [])),
                "generated_images_count": len(state.get("generated_images", [])),
                "file_locations": state.get("file_locations", {}),
                "errors_count": len(state.get("errors", []))
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error getting research summary: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Clean up resources used by the Manager Agent."""
        # Close all MCP clients
        mcp_manager.close_all()


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Manager Agent for research")
    parser.add_argument("--query", type=str, help="Research query to process")
    parser.add_argument("--resume", type=str, help="Thread ID to resume")
    parser.add_argument("--summary", type=str, help="Thread ID to summarize")
    parser.add_argument("--memory-path", type=str, help="Path for memory storage")
    args = parser.parse_args()
    
    # Create the Manager Agent
    manager = ManagerAgent(memory_storage_path=args.memory_path)
    
    try:
        if args.query:
            # Start new research
            print(f"Starting research for query: {args.query}")
            result = manager.start_research(args.query)
            print(f"Research completed with status: {result['status']}")
            
            if result.get("file_locations"):
                print("\nResearch materials saved to:")
                for key, location in result["file_locations"].items():
                    print(f"- {key}: {location}")
            
            if result.get("errors"):
                print("\nErrors encountered:")
                for error in result["errors"]:
                    print(f"- {error['agent']}: {error['error']}")
        
        elif args.resume:
            # Resume research
            print(f"Resuming research with thread ID: {args.resume}")
            result = manager.resume_research(args.resume)
            print(f"Research completed with status: {result['status']}")
        
        elif args.summary:
            # Get research summary
            print(f"Getting summary for thread ID: {args.summary}")
            summary = manager.get_research_summary(args.summary)
            print(json.dumps(summary, indent=2))
        
        else:
            # No action specified
            print("Please specify --query, --resume, or --summary")
            parser.print_help()
    
    finally:
        # Clean up
        manager.cleanup()
