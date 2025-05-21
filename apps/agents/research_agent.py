"""
Research Agent

This agent performs web searches to find relevant articles and fetches content for further processing.
It relies on Brave Search MCP for searching and Fetch MCP for retrieving content.
"""

import os
import uuid
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Literal, TypedDict, cast

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

# Import MCP clients
from apps.mcps.brave_search_mcp import BraveSearchMCP
from apps.mcps.fetch_mcp import FetchMCP
from apps.mcps.memory_mcp import MemoryMCP

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("research_agent")


# -----------------------------------------------------------------------------
# Type Definitions
# -----------------------------------------------------------------------------

class SearchResult(TypedDict):
    """Structure for search results."""
    query: str
    raw_results: str
    urls: List[str]
    title: Optional[str]
    snippet: Optional[str]


class FetchedContent(TypedDict):
    """Structure for fetched content."""
    url: str
    content: str
    title: Optional[str]
    content_type: str
    fetch_time: str
    error: Optional[str]


class ResearchRequest(TypedDict):
    """Structure for research requests."""
    query: str
    research_plan: Optional[Dict[str, Any]]
    search_queries: Optional[List[str]]
    max_results: Optional[int]
    max_urls_per_query: Optional[int]
    content_fetch_strategy: Optional[Literal["all", "top_n", "selective"]]
    namespace: Optional[str]


class ResearchResponse(TypedDict):
    """Structure for research responses."""
    original_query: str
    search_results: List[SearchResult]
    fetched_content: List[FetchedContent]
    suggested_follow_up_queries: List[str]
    execution_stats: Dict[str, Any]
    errors: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# MCP Client Management
# -----------------------------------------------------------------------------

class MCPClientManager:
    """Manages connections to MCP services used by the Research Agent."""
    
    def __init__(self):
        """Initialize MCP client manager."""
        self.brave_search_mcp = None
        self.fetch_mcp = None
        self.memory_mcp = None
    
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
    
    def get_memory_mcp(self) -> MemoryMCP:
        """Get or create Memory MCP client."""
        if self.memory_mcp is None:
            storage_path = os.environ.get("MEMORY_STORAGE_PATH", "./memory_storage")
            self.memory_mcp = MemoryMCP(storage_path=storage_path)
        return self.memory_mcp
    
    def close_all(self):
        """Close all MCP clients."""
        if self.brave_search_mcp:
            self.brave_search_mcp.close()
        if self.fetch_mcp:
            self.fetch_mcp.close()
        if self.memory_mcp:
            self.memory_mcp.close()


# Create a singleton instance
mcp_manager = MCPClientManager()


# -----------------------------------------------------------------------------
# Research Agent Core Functions
# -----------------------------------------------------------------------------

def create_llm(model: str = "gpt-4o", temperature: float = 0.2):
    """Create a language model instance."""
    return ChatOpenAI(model=model, temperature=temperature)


def generate_search_queries(query: str, research_plan: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Generate effective search queries based on the original query and research plan.
    
    Args:
        query: Original user query
        research_plan: Optional structured research plan
        
    Returns:
        List of search queries
    """
    llm = create_llm()
    
    # If research plan already has search queries, use those
    if research_plan and "potential_search_queries" in research_plan:
        return research_plan["potential_search_queries"]
    
    # Create prompt for generating search queries
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a search query optimization assistant. Your task is to generate 
        effective search queries based on the user's research question. Create 3-5 different search queries 
        that will help find comprehensive information on the topic. Include:
        
        1. Broad queries to get general information
        2. Specific queries targeting key aspects
        3. Queries using alternative terminology or perspectives
        
        Format your response as a JSON array of strings."""),
        HumanMessage(content="Research question: {query}")
    ])
    
    # Add research plan context if available
    if research_plan:
        prompt_messages = prompt.messages.copy()
        prompt_messages.append(SystemMessage(content="Research plan: {research_plan}"))
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
    
    # Execute the prompt
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "query": query,
        "research_plan": json.dumps(research_plan) if research_plan else None
    })
    
    try:
        # Parse the result as JSON
        search_queries = json.loads(result)
        if not isinstance(search_queries, list):
            search_queries = [search_queries]  # Ensure we have a list
        
        # Ensure we have at least one query
        if not search_queries:
            search_queries = [query]
        
        return search_queries
    
    except json.JSONDecodeError:
        # If parsing fails, extract queries manually
        search_queries = []
        for line in result.split('\n'):
            line = line.strip()
            if line and not line.startswith(('#', '-', '*', '{')):
                search_queries.append(line)
        
        # Ensure we have at least one query
        if not search_queries:
            search_queries = [query]
        
        return search_queries


def perform_web_search(query: str, max_results: int = 10) -> SearchResult:
    """
    Perform a web search using Brave Search MCP.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        Search results
    """
    brave_search = mcp_manager.get_brave_search_mcp()
    
    try:
        # Perform web search
        results = brave_search.web_search(query, count=max_results)
        
        # Extract URLs, title, and snippet from results
        urls = []
        title = None
        snippet = None
        
        # Parse the results
        lines = results.split('\n\n')
        for i, line in enumerate(lines):
            if line.startswith('Title:'):
                if i == 0:  # First result title
                    title = line.replace('Title:', '').strip()
            elif line.startswith('Description:'):
                if i == 0:  # First result snippet
                    snippet = line.replace('Description:', '').strip()
            elif line.startswith('URL:'):
                url = line.replace('URL:', '').strip()
                urls.append(url)
        
        return {
            "query": query,
            "raw_results": results,
            "urls": urls,
            "title": title,
            "snippet": snippet
        }
        
    except Exception as e:
        logger.error(f"Error performing search for query '{query}': {e}")
        return {
            "query": query,
            "raw_results": f"Error: {str(e)}",
            "urls": [],
            "title": None,
            "snippet": None
        }


def fetch_content(url: str) -> FetchedContent:
    """
    Fetch content from a URL using Fetch MCP.
    
    Args:
        url: URL to fetch content from
        
    Returns:
        Fetched content
    """
    fetch = mcp_manager.get_fetch_mcp()
    
    try:
        # Fetch text content
        content = fetch.fetch_text(url)
        
        # Try to extract title
        title = None
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and len(line) < 200:  # Reasonable title length
                title = line
                break
        
        return {
            "url": url,
            "content": content,
            "title": title,
            "content_type": "text/plain",
            "fetch_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Error fetching content from URL '{url}': {e}")
        return {
            "url": url,
            "content": f"Error fetching content: {str(e)}",
            "title": None,
            "content_type": "text/plain",
            "fetch_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e)
        }


def analyze_search_results(search_results: List[SearchResult], original_query: str) -> List[str]:
    """
    Analyze search results and suggest follow-up queries.
    
    Args:
        search_results: List of search results
        original_query: Original user query
        
    Returns:
        List of suggested follow-up queries
    """
    llm = create_llm()
    
    # If no search results, return empty list
    if not search_results:
        return []
    
    # Combine search results
    combined_results = ""
    for result in search_results:
        combined_results += f"Query: {result['query']}\n"
        combined_results += f"Results: {result['raw_results'][:500]}...\n\n"
    
    # Create prompt for analyzing results
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a research assistant analyzing search results. Based on the 
        search results provided, suggest 2-3 follow-up queries that would help gather additional 
        information on the topic. Focus on:
        
        1. Aspects not covered in the current results
        2. Deeper exploration of interesting findings
        3. Alternative perspectives or approaches
        
        Format your response as a JSON array of strings."""),
        HumanMessage(content="Original query: {original_query}"),
        HumanMessage(content="Search results: {search_results}")
    ])
    
    # Execute the prompt
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "original_query": original_query,
        "search_results": combined_results
    })
    
    try:
        # Parse the result as JSON
        follow_up_queries = json.loads(result)
        if not isinstance(follow_up_queries, list):
            follow_up_queries = [follow_up_queries]  # Ensure we have a list
        
        return follow_up_queries
    
    except json.JSONDecodeError:
        # If parsing fails, extract queries manually
        follow_up_queries = []
        for line in result.split('\n'):
            line = line.strip()
            if line and not line.startswith(('#', '-', '*', '{')):
                follow_up_queries.append(line)
        
        return follow_up_queries


def prioritize_urls(urls: List[str], query: str) -> List[str]:
    """
    Prioritize URLs based on relevance to the query.
    
    Args:
        urls: List of URLs to prioritize
        query: Search query
        
    Returns:
        Prioritized list of URLs
    """
    llm = create_llm()
    
    # If few URLs, return as is
    if len(urls) <= 3:
        return urls
    
    # Create prompt for prioritizing URLs
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a URL prioritization assistant. Your task is to analyze a list 
        of URLs and prioritize them based on their likely relevance to the search query. Consider:
        
        1. Domain authority and credibility
        2. Relevance of the URL to the query
        3. Diversity of sources
        
        Return a JSON array of the URLs in priority order (most relevant first)."""),
        HumanMessage(content="Search query: {query}"),
        HumanMessage(content="URLs to prioritize: {urls}")
    ])
    
    # Execute the prompt
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "query": query,
        "urls": "\n".join(urls)
    })
    
    try:
        # Parse the result as JSON
        prioritized_urls = json.loads(result)
        if not isinstance(prioritized_urls, list):
            prioritized_urls = [prioritized_urls]  # Ensure we have a list
        
        # Ensure all original URLs are included
        original_set = set(urls)
        prioritized_set = set(prioritized_urls)
        
        # Add any missing URLs
        for url in original_set - prioritized_set:
            prioritized_urls.append(url)
        
        # Remove any URLs not in the original list
        prioritized_urls = [url for url in prioritized_urls if url in original_set]
        
        return prioritized_urls
    
    except json.JSONDecodeError:
        # If parsing fails, return original URLs
        return urls


# -----------------------------------------------------------------------------
# Research Agent Class
# -----------------------------------------------------------------------------

class ResearchAgent:
    """
    Research Agent that performs web searches and fetches content.
    """
    
    def __init__(self):
        """Initialize the Research Agent."""
        pass
    
    def research(self, request: ResearchRequest) -> ResearchResponse:
        """
        Perform research based on the request.
        
        Args:
            request: Research request
            
        Returns:
            Research response
        """
        start_time = time.time()
        
        # Initialize response
        response: ResearchResponse = {
            "original_query": request["query"],
            "search_results": [],
            "fetched_content": [],
            "suggested_follow_up_queries": [],
            "execution_stats": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 0,
                "search_queries_count": 0,
                "urls_fetched_count": 0,
                "successful_fetches_count": 0
            },
            "errors": []
        }
        
        try:
            # Get search queries
            search_queries = request.get("search_queries")
            if not search_queries:
                search_queries = generate_search_queries(
                    request["query"], 
                    request.get("research_plan")
                )
            
            # Set limits
            max_results = request.get("max_results", 10)
            max_urls_per_query = request.get("max_urls_per_query", 3)
            content_fetch_strategy = request.get("content_fetch_strategy", "selective")
            
            # Perform searches
            all_search_results = []
            for query in search_queries:
                try:
                    result = perform_web_search(query, max_results=max_results)
                    all_search_results.append(result)
                except Exception as e:
                    logger.error(f"Error searching for query '{query}': {e}")
                    response["errors"].append({
                        "type": "search_error",
                        "query": query,
                        "error": str(e)
                    })
            
            # Update response with search results
            response["search_results"] = all_search_results
            
            # Collect all URLs
            all_urls = []
            for result in all_search_results:
                all_urls.extend(result["urls"])
            
            # Remove duplicates while preserving order
            unique_urls = []
            seen_urls = set()
            for url in all_urls:
                if url not in seen_urls:
                    unique_urls.append(url)
                    seen_urls.add(url)
            
            # Prioritize URLs if using selective strategy
            if content_fetch_strategy == "selective" and unique_urls:
                unique_urls = prioritize_urls(unique_urls, request["query"])
            
            # Determine how many URLs to fetch
            urls_to_fetch = []
            if content_fetch_strategy == "all":
                urls_to_fetch = unique_urls
            elif content_fetch_strategy == "top_n":
                urls_to_fetch = unique_urls[:max_urls_per_query]
            else:  # selective
                # Take top N from each search result
                for result in all_search_results:
                    urls_to_fetch.extend(result["urls"][:max_urls_per_query])
                # Remove duplicates while preserving order
                urls_to_fetch = list(dict.fromkeys(urls_to_fetch))
            
            # Fetch content
            fetched_content = []
            for url in urls_to_fetch:
                try:
                    content = fetch_content(url)
                    fetched_content.append(content)
                    if content["error"] is None:
                        response["execution_stats"]["successful_fetches_count"] += 1
                except Exception as e:
                    logger.error(f"Error fetching content from URL '{url}': {e}")
                    response["errors"].append({
                        "type": "fetch_error",
                        "url": url,
                        "error": str(e)
                    })
            
            # Update response with fetched content
            response["fetched_content"] = fetched_content
            
            # Generate follow-up queries
            try:
                follow_up_queries = analyze_search_results(all_search_results, request["query"])
                response["suggested_follow_up_queries"] = follow_up_queries
            except Exception as e:
                logger.error(f"Error generating follow-up queries: {e}")
                response["errors"].append({
                    "type": "analysis_error",
                    "error": str(e)
                })
            
            # Store results in memory if namespace provided
            if request.get("namespace"):
                try:
                    memory_mcp = mcp_manager.get_memory_mcp()
                    
                    # Store search results
                    memory_mcp.store_memory(
                        key=f"search_results_{uuid.uuid4()}",
                        value=json.dumps(response["search_results"]),
                        namespace=request["namespace"]
                    )
                    
                    # Store fetched content
                    memory_mcp.store_memory(
                        key=f"fetched_content_{uuid.uuid4()}",
                        value=json.dumps(response["fetched_content"]),
                        namespace=request["namespace"]
                    )
                    
                    # Store follow-up queries
                    memory_mcp.store_memory(
                        key=f"follow_up_queries_{uuid.uuid4()}",
                        value=json.dumps(response["suggested_follow_up_queries"]),
                        namespace=request["namespace"]
                    )
                except Exception as e:
                    logger.error(f"Error storing results in memory: {e}")
                    response["errors"].append({
                        "type": "storage_error",
                        "error": str(e)
                    })
            
            # Update execution stats
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            response["execution_stats"]["search_queries_count"] = len(search_queries)
            response["execution_stats"]["urls_fetched_count"] = len(urls_to_fetch)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in research process: {e}")
            response["errors"].append({
                "type": "general_error",
                "error": str(e)
            })
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            return response
    
    def research_with_follow_up(self, request: ResearchRequest, max_iterations: int = 2) -> ResearchResponse:
        """
        Perform research with automatic follow-up queries.
        
        Args:
            request: Initial research request
            max_iterations: Maximum number of follow-up iterations
            
        Returns:
            Combined research response
        """
        # Perform initial research
        response = self.research(request)
        
        # Initialize combined response
        combined_response: ResearchResponse = {
            "original_query": request["query"],
            "search_results": response["search_results"].copy(),
            "fetched_content": response["fetched_content"].copy(),
            "suggested_follow_up_queries": response["suggested_follow_up_queries"].copy(),
            "execution_stats": {
                "start_time": response["execution_stats"]["start_time"],
                "duration_seconds": response["execution_stats"]["duration_seconds"],
                "search_queries_count": response["execution_stats"]["search_queries_count"],
                "urls_fetched_count": response["execution_stats"]["urls_fetched_count"],
                "successful_fetches_count": response["execution_stats"]["successful_fetches_count"],
                "follow_up_iterations": 0
            },
            "errors": response["errors"].copy()
        }
        
        # Perform follow-up iterations
        for i in range(max_iterations):
            # Check if we have follow-up queries
            if not response["suggested_follow_up_queries"]:
                break
            
            # Select the top follow-up query
            follow_up_query = response["suggested_follow_up_queries"][0]
            
            # Create follow-up request
            follow_up_request = {
                "query": follow_up_query,
                "research_plan": request.get("research_plan"),
                "max_results": request.get("max_results", 5),  # Reduce results for follow-ups
                "max_urls_per_query": request.get("max_urls_per_query", 2),  # Reduce URLs for follow-ups
                "content_fetch_strategy": "top_n",  # Simplify strategy for follow-ups
                "namespace": request.get("namespace")
            }
            
            # Perform follow-up research
            follow_up_response = self.research(follow_up_request)
            
            # Update combined response
            combined_response["search_results"].extend(follow_up_response["search_results"])
            combined_response["fetched_content"].extend(follow_up_response["fetched_content"])
            combined_response["suggested_follow_up_queries"] = follow_up_response["suggested_follow_up_queries"]
            combined_response["execution_stats"]["duration_seconds"] += follow_up_response["execution_stats"]["duration_seconds"]
            combined_response["execution_stats"]["search_queries_count"] += follow_up_response["execution_stats"]["search_queries_count"]
            combined_response["execution_stats"]["urls_fetched_count"] += follow_up_response["execution_stats"]["urls_fetched_count"]
            combined_response["execution_stats"]["successful_fetches_count"] += follow_up_response["execution_stats"]["successful_fetches_count"]
            combined_response["execution_stats"]["follow_up_iterations"] += 1
            combined_response["errors"].extend(follow_up_response["errors"])
            
            # Update response for next iteration
            response = follow_up_response
        
        return combined_response
    
    def evaluate_sources(self, urls: List[str]) -> Dict[str, Any]:
        """
        Evaluate the credibility and relevance of sources.
        
        Args:
            urls: List of URLs to evaluate
            
        Returns:
            Evaluation results
        """
        llm = create_llm()
        
        # Create prompt for evaluating sources
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a source evaluation assistant. Your task is to analyze a list 
            of URLs and evaluate their credibility and relevance. Consider:
            
            1. Domain authority and reputation
            2. Type of source (academic, news, blog, etc.)
            3. Potential biases or conflicts of interest
            
            For each URL, provide a brief assessment and a credibility score (1-10).
            Format your response as a JSON object with URLs as keys and objects with 'assessment' and 'score' as values."""),
            HumanMessage(content="URLs to evaluate: {urls}")
        ])
        
        # Execute the prompt
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({
            "urls": "\n".join(urls)
        })
        
        try:
            # Parse the result as JSON
            evaluation = json.loads(result)
            return evaluation
        
        except json.JSONDecodeError:
            # If parsing fails, return simple evaluation
            evaluation = {}
            for url in urls:
                evaluation[url] = {
                    "assessment": "Unable to automatically evaluate this source.",
                    "score": 5  # Neutral score
                }
            return evaluation
    
    def cleanup(self):
        """Clean up resources used by the Research Agent."""
        # Close all MCP clients
        mcp_manager.close_all()


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def extract_key_information(content: str, query: str) -> Dict[str, Any]:
    """
    Extract key information from content relevant to the query.
    
    Args:
        content: Content to analyze
        query: Query to focus extraction on
        
    Returns:
        Extracted key information
    """
    llm = create_llm()
    
    # Truncate content if too long
    max_content_length = 8000  # Adjust based on model context limits
    if len(content) > max_content_length:
        content = content[:max_content_length] + "..."
    
    # Create prompt for extracting key information
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an information extraction assistant. Your task is to extract 
        key information from the provided content that is relevant to the query. Focus on:
        
        1. Main facts and data points
        2. Key concepts and definitions
        3. Important quotes or statements
        4. Relevant statistics or numbers
        
        Format your response as a JSON object with these categories as keys and arrays of extracted information as values."""),
        HumanMessage(content="Query: {query}"),
        HumanMessage(content="Content: {content}")
    ])
    
    # Execute the prompt
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "query": query,
        "content": content
    })
    
    try:
        # Parse the result as JSON
        extracted_info = json.loads(result)
        return extracted_info
    
    except json.JSONDecodeError:
        # If parsing fails, return simple extraction
        return {
            "main_facts": [result],
            "key_concepts": [],
            "important_quotes": [],
            "relevant_statistics": []
        }


def format_research_results(response: ResearchResponse) -> str:
    """
    Format research results into a readable report.
    
    Args:
        response: Research response
        
    Returns:
        Formatted research report
    """
    report = f"# Research Report: {response['original_query']}\n\n"
    
    # Add summary stats
    report += "## Summary\n"
    report += f"- Searched {response['execution_stats']['search_queries_count']} queries\n"
    report += f"- Found {len(response['search_results'])} search results\n"
    report += f"- Fetched content from {response['execution_stats']['urls_fetched_count']} URLs\n"
    report += f"- Research took {response['execution_stats']['duration_seconds']} seconds\n\n"
    
    # Add search results
    report += "## Search Results\n\n"
    for i, result in enumerate(response['search_results']):
        report += f"### Query {i+1}: {result['query']}\n"
        if result.get('title') and result.get('snippet'):
            report += f"**Top Result:** {result['title']}\n"
            report += f"**Snippet:** {result['snippet']}\n"
        report += f"**Found URLs:**\n"
        for url in result['urls'][:5]:  # Limit to first 5 URLs
            report += f"- {url}\n"
        report += "\n"
    
    # Add fetched content summaries
    report += "## Content Summaries\n\n"
    for i, content in enumerate(response['fetched_content']):
        if content.get('error'):
            continue  # Skip failed fetches
        
        report += f"### Source {i+1}: {content.get('title', 'Untitled')}\n"
        report += f"**URL:** {content['url']}\n"
        
        # Add a brief excerpt
        excerpt = content['content'][:300].replace('\n', ' ')
        report += f"**Excerpt:** {excerpt}...\n\n"
    
    # Add suggested follow-up queries
    if response['suggested_follow_up_queries']:
        report += "## Suggested Follow-up Queries\n"
        for query in response['suggested_follow_up_queries']:
            report += f"- {query}\n"
        report += "\n"
    
    # Add errors if any
    if response['errors']:
        report += "## Errors\n"
        for error in response['errors']:
            report += f"- {error['type']}: {error['error']}\n"
        report += "\n"
    
    return report


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Research Agent")
    parser.add_argument("--query", type=str, required=True, help="Research query to process")
    parser.add_argument("--max-results", type=int, default=10, help="Maximum search results per query")
    parser.add_argument("--max-urls", type=int, default=3, help="Maximum URLs to fetch per query")
    parser.add_argument("--strategy", type=str, choices=["all", "top_n", "selective"], default="selective", 
                        help="Content fetch strategy")
    parser.add_argument("--follow-up", action="store_true", help="Enable automatic follow-up queries")
    parser.add_argument("--max-iterations", type=int, default=2, help="Maximum follow-up iterations")
    parser.add_argument("--memory-path", type=str, help="Path for memory storage")
    parser.add_argument("--output", type=str, help="Output file for research report")
    args = parser.parse_args()
    
    # Set memory storage path if provided
    if args.memory_path:
        os.environ["MEMORY_STORAGE_PATH"] = args.memory_path
    
    # Create the Research Agent
    agent = ResearchAgent()
    
    try:
        # Create research request
        request: ResearchRequest = {
            "query": args.query,
            "research_plan": None,  # Will be generated automatically
            "search_queries": None,  # Will be generated automatically
            "max_results": args.max_results,
            "max_urls_per_query": args.max_urls,
            "content_fetch_strategy": args.strategy,
            "namespace": args.query  # Use query as namespace
        }
        
        print(f"Starting research for query: {args.query}")
        print(f"Strategy: {args.strategy}, Max results: {args.max_results}, Max URLs: {args.max_urls}")
        
        # Perform research
        if args.follow_up:
            print(f"Performing research with automatic follow-up (max {args.max_iterations} iterations)...")
            response = agent.research_with_follow_up(request, max_iterations=args.max_iterations)
        else:
            print("Performing research...")
            response = agent.research(request)
        
        # Format results
        report = format_research_results(response)
        
        # Output results
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Research report saved to: {args.output}")
        else:
            print("\n" + "="*80 + "\n")
            print(report)
            print("\n" + "="*80 + "\n")
        
        # Print stats
        print("Research completed!")
        print(f"- Searched {response['execution_stats']['search_queries_count']} queries")
        print(f"- Found {len(response['search_results'])} search results")
        print(f"- Fetched content from {response['execution_stats']['urls_fetched_count']} URLs")
        print(f"- Research took {response['execution_stats']['duration_seconds']} seconds")
        
        if response['errors']:
            print(f"- Encountered {len(response['errors'])} errors")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up
        agent.cleanup()
