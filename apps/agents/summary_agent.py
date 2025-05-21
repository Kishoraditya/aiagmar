"""
Summary Agent

This agent summarizes the fetched content using the language model's capabilities,
providing concise insights and storing summaries for later use.
"""

import os
import uuid
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Literal, TypedDict, cast
import sys

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

# Import MCP clients
from apps.mcps.memory_mcp import MemoryMCP

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("summary_agent")


# -----------------------------------------------------------------------------
# Type Definitions
# -----------------------------------------------------------------------------

class FetchedContent(TypedDict):
    """Structure for fetched content."""
    url: str
    content: str
    title: Optional[str]
    content_type: str
    fetch_time: str
    error: Optional[str]


class ContentSummary(TypedDict):
    """Structure for content summaries."""
    url: str
    title: Optional[str]
    summary: str
    key_points: List[str]
    entities: Dict[str, List[str]]
    summary_type: str  # "detailed", "concise", "bullet_points", etc.
    word_count: int
    original_content_length: int


class SummaryRequest(TypedDict):
    """Structure for summary requests."""
    content_items: List[FetchedContent]
    query: str
    summary_type: Optional[str]  # "detailed", "concise", "bullet_points", etc.
    max_length: Optional[int]
    focus_aspects: Optional[List[str]]
    namespace: Optional[str]


class SummaryResponse(TypedDict):
    """Structure for summary responses."""
    content_summaries: List[ContentSummary]
    overall_summary: Optional[ContentSummary]
    execution_stats: Dict[str, Any]
    errors: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# MCP Client Management
# -----------------------------------------------------------------------------

class MCPClientManager:
    """Manages connections to MCP services used by the Summary Agent."""
    
    def __init__(self):
        """Initialize MCP client manager."""
        self.memory_mcp = None
    
    def get_memory_mcp(self) -> MemoryMCP:
        """Get or create Memory MCP client."""
        if self.memory_mcp is None:
            storage_path = os.environ.get("MEMORY_STORAGE_PATH", "./memory_storage")
            self.memory_mcp = MemoryMCP(storage_path=storage_path)
        return self.memory_mcp
    
    def close_all(self):
        """Close all MCP clients."""
        if self.memory_mcp:
            self.memory_mcp.close()


# Create a singleton instance
mcp_manager = MCPClientManager()


# -----------------------------------------------------------------------------
# Summary Agent Core Functions
# -----------------------------------------------------------------------------

def create_llm(model: str = "gpt-4o", temperature: float = 0.2):
    """Create a language model instance."""
    return ChatOpenAI(model=model, temperature=temperature)


def count_words(text: str) -> int:
    """Count the number of words in a text."""
    return len(text.split())


def summarize_content(content_item: FetchedContent, query: str, summary_type: str = "detailed", 
                     max_length: int = 500, focus_aspects: Optional[List[str]] = None) -> ContentSummary:
    """
    Summarize a single content item.
    
    Args:
        content_item: Content to summarize
        query: Original query for context
        summary_type: Type of summary to generate
        max_length: Maximum length of summary in words
        focus_aspects: Specific aspects to focus on
        
    Returns:
        Content summary
    """
    llm = create_llm()
    
    # Truncate content if too long
    max_content_length = 8000  # Adjust based on model context limits
    content = content_item["content"]
    original_length = len(content)
    if len(content) > max_content_length:
        content = content[:max_content_length] + "..."
    
    # Determine system prompt based on summary type
    system_prompts = {
        "detailed": """You are a comprehensive summarization assistant. Create a detailed summary that captures 
        the main arguments, key evidence, and important context from the content. Maintain the nuance and 
        complexity of the original while making it more accessible. Include relevant quotes where appropriate.""",
        
        "concise": """You are a concise summarization assistant. Create a brief, clear summary that captures 
        the essential points of the content in a compact form. Focus on the core message and most critical 
        details. Aim for clarity and brevity while preserving accuracy.""",
        
        "bullet_points": """You are a structured summarization assistant. Create a summary in bullet point 
        format that organizes the key information into clear, scannable points. Group related information 
        under appropriate headings. Each bullet should be self-contained and informative.""",
        
        "analytical": """You are an analytical summarization assistant. Create a summary that not only 
        condenses the content but also analyzes its arguments, evidence quality, potential biases, and 
        relationship to the broader topic. Evaluate the strength of the content's position."""
    }
    
    system_content = system_prompts.get(summary_type, system_prompts["detailed"])
    
    # Add focus aspects if provided
    if focus_aspects:
        aspects_str = ", ".join(focus_aspects)
        system_content += f"\n\nFocus particularly on these aspects: {aspects_str}."
    
    # Create prompt for summarization
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        HumanMessage(content="""Please summarize the following content in relation to this query: "{query}"
        
        Content from: {url}
        
        {content}
        
        Provide your summary in this format:
        1. A coherent summary text (maximum {max_length} words)
        2. A list of 3-5 key points
        3. Important entities mentioned (people, organizations, concepts, etc.)
        
        Format your response as JSON with these keys: "summary", "key_points", "entities".""")
    ])
    
    # Execute the prompt
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "query": query,
        "url": content_item["url"],
        "content": content,
        "max_length": max_length
    })
    
    try:
        # Parse the result as JSON
        parsed_result = json.loads(result)
        
        # Ensure expected keys exist
        if not all(k in parsed_result for k in ["summary", "key_points", "entities"]):
            raise ValueError("Missing required keys in summary result")
        
        # Create content summary
        summary = {
            "url": content_item["url"],
            "title": content_item.get("title"),
            "summary": parsed_result["summary"],
            "key_points": parsed_result["key_points"],
            "entities": parsed_result["entities"],
            "summary_type": summary_type,
            "word_count": count_words(parsed_result["summary"]),
            "original_content_length": original_length
        }
        
        return summary
    
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing summary result: {e}")
        
        # Extract summary manually
        lines = result.split('\n')
        summary_text = ""
        key_points = []
        entities = {}
        
        # Simple parsing logic
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if "summary" in line.lower() and ":" in line:
                current_section = "summary"
                summary_text = line.split(":", 1)[1].strip()
            elif "key point" in line.lower() or "- " in line:
                if current_section != "key_points":
                    current_section = "key_points"
                key_points.append(line.replace("- ", "").strip())
            elif "entit" in line.lower() and ":" in line:
                current_section = "entities"
                entity_type = line.split(":", 1)[0].strip()
                entity_value = line.split(":", 1)[1].strip()
                entities[entity_type] = [entity_value]
        
        # Create content summary with extracted data
        summary = {
            "url": content_item["url"],
            "title": content_item.get("title"),
            "summary": summary_text or result[:500],  # Use first 500 chars if no summary found
            "key_points": key_points or ["No key points extracted"],
            "entities": entities or {"general": ["No entities extracted"]},
            "summary_type": summary_type,
            "word_count": count_words(summary_text or result[:500]),
            "original_content_length": original_length
        }
        
        return summary


def create_overall_summary(content_summaries: List[ContentSummary], query: str, 
                          summary_type: str = "detailed") -> ContentSummary:
    """
    Create an overall summary from multiple content summaries.
    
    Args:
        content_summaries: List of content summaries
        query: Original query for context
        summary_type: Type of summary to generate
        
    Returns:
        Overall content summary
    """
    llm = create_llm()
    
    # Combine summaries
    combined_summaries = ""
    for i, summary in enumerate(content_summaries):
        combined_summaries += f"Source {i+1}: {summary['url']}\n"
        if summary.get("title"):
            combined_summaries += f"Title: {summary['title']}\n"
        combined_summaries += f"Summary: {summary['summary']}\n"
        combined_summaries += f"Key Points: {', '.join(summary['key_points'])}\n\n"
    
    # Determine system prompt based on summary type
    system_prompts = {
        "detailed": """You are a research synthesis assistant. Create a comprehensive overall summary 
        that integrates information from multiple sources. Identify common themes, reconcile contradictions, 
        and present a unified view of the topic. Note areas of consensus and disagreement between sources.""",
        
        "concise": """You are a research synthesis assistant. Create a concise overall summary that 
        captures the essential information from multiple sources. Focus on the most important findings 
        and insights that appear across sources. Be brief but comprehensive.""",
        
        "bullet_points": """You are a research synthesis assistant. Create a structured overall summary 
        in bullet point format that organizes key information from multiple sources. Group related points 
        under clear headings. Highlight areas of consensus and important differences.""",
        
        "analytical": """You are a research synthesis assistant. Create an analytical overall summary 
        that not only integrates information from multiple sources but also evaluates the quality of 
        evidence, identifies gaps in knowledge, and assesses the overall state of understanding on the topic."""
    }
    
    system_content = system_prompts.get(summary_type, system_prompts["detailed"])
    
    # Create prompt for overall summarization
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        HumanMessage(content="""Please create an overall summary synthesizing information from these sources 
        in relation to the query: "{query}"
        
        {summaries}
        
        Provide your synthesis in this format:
        1. A coherent overall summary that integrates information from all sources
        2. A list of 5-7 key findings or insights
        3. Important concepts, entities, or terminology relevant to the topic
        4. Areas where sources agree or disagree (if applicable)
        
        Format your response as JSON with these keys: "summary", "key_points", "entities", "consensus_disagreement".""")
    ])
    
    # Execute the prompt
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "query": query,
        "summaries": combined_summaries
    })
    
    try:
        # Parse the result as JSON
        parsed_result = json.loads(result)
        
        # Extract entities from consensus_disagreement if present
        entities = parsed_result.get("entities", {})
        if "consensus_disagreement" in parsed_result:
            entities["consensus_disagreement"] = parsed_result["consensus_disagreement"]
        
        # Create overall summary
        overall_summary = {
            "url": "overall_summary",
            "title": f"Overall Summary: {query}",
            "summary": parsed_result["summary"],
            "key_points": parsed_result["key_points"],
            "entities": entities,
            "summary_type": summary_type,
            "word_count": count_words(parsed_result["summary"]),
            "original_content_length": sum(s["original_content_length"] for s in content_summaries)
        }
        
        return overall_summary
    
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing overall summary result: {e}")
        
        # Create a simple overall summary
        summary_text = f"Overall summary for query: {query}\n\n" + result[:1000]
        
        overall_summary = {
            "url": "overall_summary",
            "title": f"Overall Summary: {query}",
            "summary": summary_text,
            "key_points": ["Unable to extract structured key points from the overall summary"],
            "entities": {"general": ["Unable to extract structured entities from the overall summary"]},
            "summary_type": summary_type,
            "word_count": count_words(summary_text),
            "original_content_length": sum(s["original_content_length"] for s in content_summaries)
        }
        
        return overall_summary


def extract_key_insights(content_summaries: List[ContentSummary], query: str) -> Dict[str, Any]:
    """
    Extract key insights across all content summaries.
    
    Args:
        content_summaries: List of content summaries
        query: Original query for context
        
    Returns:
        Key insights
    """
    llm = create_llm()
    
    # Combine summaries
    combined_summaries = ""
    for i, summary in enumerate(content_summaries):
        combined_summaries += f"Source {i+1}: {summary['url']}\n"
        combined_summaries += f"Summary: {summary['summary']}\n"
        combined_summaries += f"Key Points: {', '.join(summary['key_points'])}\n\n"
    
    # Create prompt for extracting insights
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an insights extraction assistant. Your task is to identify the most 
        significant insights, patterns, and implications across multiple content summaries. Focus on:
        
        1. Recurring themes or findings
        2. Surprising or counterintuitive discoveries
        3. Practical implications or applications
        4. Gaps in knowledge or areas for further research
        
        Organize your insights by category and provide specific evidence from the sources."""),
        HumanMessage(content="Query: {query}"),
        HumanMessage(content="Content summaries: {summaries}")
    ])
    
    # Execute the prompt
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "query": query,
        "summaries": combined_summaries
    })
    
    try:
        # Parse the result as JSON
        insights = json.loads(result)
        return insights
    
    except json.JSONDecodeError:
        # If parsing fails, return simple insights
        return {
            "recurring_themes": [result],
            "surprising_discoveries": [],
            "practical_implications": [],
            "knowledge_gaps": []
        }


# -----------------------------------------------------------------------------
# Summary Agent Class
# -----------------------------------------------------------------------------

class SummaryAgent:
    """
    Summary Agent that summarizes fetched content.
    """
    
    def __init__(self):
        """Initialize the Summary Agent."""
        pass
    
    def summarize(self, request: SummaryRequest) -> SummaryResponse:
        """
        Summarize content based on the request.
        
        Args:
            request: Summary request
            
        Returns:
            Summary response
        """
        start_time = time.time()
        
        # Initialize response
        response: SummaryResponse = {
            "content_summaries": [],
            "overall_summary": None,
            "execution_stats": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 0,
                "content_items_count": len(request["content_items"]),
                "total_words_summarized": 0,
                "average_compression_ratio": 0
            },
            "errors": []
        }
        
        try:
            # Set defaults
            summary_type = request.get("summary_type", "detailed")
            max_length = request.get("max_length", 500)
            focus_aspects = request.get("focus_aspects")
            
            # Summarize each content item
            content_summaries = []
            total_original_words = 0
            total_summary_words = 0
            
            for item in request["content_items"]:
                try:
                    # Skip items with errors
                    if item.get("error"):
                        response["errors"].append({
                            "type": "content_error",
                            "url": item["url"],
                            "error": item["error"]
                        })
                        continue
                    
                    # Summarize content
                    summary = summarize_content(
                        item,
                        request["query"],
                        summary_type=summary_type,
                        max_length=max_length,
                        focus_aspects=focus_aspects
                    )
                    
                    content_summaries.append(summary)
                    
                    # Update word counts
                    original_words = count_words(item["content"])
                    total_original_words += original_words
                    total_summary_words += summary["word_count"]
                    
                except Exception as e:
                    logger.error(f"Error summarizing content from URL '{item['url']}': {e}")
                    response["errors"].append({
                        "type": "summarization_error",
                        "url": item["url"],
                        "error": str(e)
                    })
            
            # Update response with content summaries
            response["content_summaries"] = content_summaries
            
            # Create overall summary if we have multiple content items
            if len(content_summaries) > 1:
                try:
                    overall_summary = create_overall_summary(
                        content_summaries,
                        request["query"],
                        summary_type=summary_type
                    )
                    response["overall_summary"] = overall_summary
                    total_summary_words += overall_summary["word_count"]
                except Exception as e:
                    logger.error(f"Error creating overall summary: {e}")
                    response["errors"].append({
                        "type": "overall_summary_error",
                        "error": str(e)
                    })
            elif len(content_summaries) == 1:
                # If only one content item, use its summary as the overall summary
                response["overall_summary"] = content_summaries[0]
            
            # Calculate compression ratio
            if total_original_words > 0:
                compression_ratio = total_summary_words / total_original_words
            else:
                compression_ratio = 0
            
            # Store summaries in memory if namespace provided
            if request.get("namespace"):
                try:
                    memory_mcp = mcp_manager.get_memory_mcp()
                    
                    # Store content summaries
                    memory_mcp.store_memory(
                        key=f"content_summaries_{uuid.uuid4()}",
                        value=json.dumps(response["content_summaries"]),
                        namespace=request["namespace"]
                    )
                    
                    # Store overall summary
                    if response["overall_summary"]:
                        memory_mcp.store_memory(
                            key=f"overall_summary_{uuid.uuid4()}",
                            value=json.dumps(response["overall_summary"]),
                            namespace=request["namespace"]
                        )
                    
                    # Extract and store key insights
                    if len(content_summaries) > 0:
                        insights = extract_key_insights(content_summaries, request["query"])
                        memory_mcp.store_memory(
                            key=f"key_insights_{uuid.uuid4()}",
                            value=json.dumps(insights),
                            namespace=request["namespace"]
                        )
                except Exception as e:
                    logger.error(f"Error storing summaries in memory: {e}")
                    response["errors"].append({
                        "type": "storage_error",
                        "error": str(e)
                    })
            
            # Update execution stats
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            response["execution_stats"]["total_words_summarized"] = total_original_words
            response["execution_stats"]["average_compression_ratio"] = round(compression_ratio, 4)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in summarization process: {e}")
            response["errors"].append({
                "type": "general_error",
                "error": str(e)
            })
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            return response
    
    def summarize_with_variations(self, request: SummaryRequest, 
                                 variations: List[str] = ["detailed", "concise", "bullet_points"]) -> Dict[str, SummaryResponse]:
        """
        Generate multiple summary variations.
        
        Args:
            request: Summary request
            variations: List of summary types to generate
            
        Returns:
            Dictionary of summary responses by type
        """
        results = {}
        
        for summary_type in variations:
            # Create a copy of the request with the current summary type
            variation_request = request.copy()
            variation_request["summary_type"] = summary_type
            
            # Generate summary for this variation
            response = self.summarize(variation_request)
            
            # Store in results
            results[summary_type] = response
        
        return results
    
    def create_comparative_analysis(self, content_summaries: List[ContentSummary], query: str) -> Dict[str, Any]:
        """
        Create a comparative analysis of multiple content sources.
        
        Args:
            content_summaries: List of content summaries to compare
            query: Original query for context
            
        Returns:
            Comparative analysis
        """
        llm = create_llm()
        
        # Need at least 2 summaries for comparison
        if len(content_summaries) < 2:
            return {
                "error": "Need at least 2 content summaries for comparative analysis",
                "comparison": {}
            }
        
        # Combine summaries
        combined_summaries = ""
        for i, summary in enumerate(content_summaries):
            combined_summaries += f"Source {i+1}: {summary['url']}\n"
            if summary.get("title"):
                combined_summaries += f"Title: {summary['title']}\n"
            combined_summaries += f"Summary: {summary['summary']}\n"
            combined_summaries += f"Key Points: {', '.join(summary['key_points'])}\n\n"
        
        # Create prompt for comparative analysis
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a comparative analysis assistant. Your task is to compare and 
            contrast information from multiple sources on the same topic. Focus on:
            
            1. Areas of agreement across sources
            2. Significant disagreements or contradictions
            3. Unique perspectives or information provided by each source
            4. Potential biases or limitations in each source
            5. Overall assessment of the information landscape
            
            Provide a structured analysis that helps understand the full picture across all sources."""),
            HumanMessage(content="Query: {query}"),
            HumanMessage(content="Content summaries: {summaries}")
        ])
        
        # Execute the prompt
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({
            "query": query,
            "summaries": combined_summaries
        })
        
        try:
            # Parse the result as JSON
            analysis = json.loads(result)
            return {"comparison": analysis}
        
        except json.JSONDecodeError:
            # If parsing fails, return simple analysis
            return {
                "comparison": {
                    "agreements": [],
                    "disagreements": [],
                    "unique_perspectives": {},
                    "potential_biases": {},
                    "overall_assessment": result
                }
            }
    
    def cleanup(self):
        """Clean up resources used by the Summary Agent."""
        # Close all MCP clients
        mcp_manager.close_all()


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def format_summary_as_markdown(summary: ContentSummary) -> str:
    """
    Format a content summary as Markdown.
    
    Args:
        summary: Content summary to format
        
    Returns:
        Markdown-formatted summary
    """
    markdown = f"# {summary.get('title', 'Summary')}\n\n"
    
    # Add source
    if summary["url"] != "overall_summary":
        markdown += f"**Source:** [{summary['url']}]({summary['url']})\n\n"
    
    # Add summary
    markdown += f"## Summary\n\n{summary['summary']}\n\n"
    
    # Add key points
    markdown += "## Key Points\n\n"
    for point in summary["key_points"]:
        markdown += f"- {point}\n"
    markdown += "\n"
    
    # Add entities
    markdown += "## Key Entities\n\n"
    for entity_type, entities in summary["entities"].items():
        markdown += f"### {entity_type.title()}\n"
        for entity in entities:
            markdown += f"- {entity}\n"
        markdown += "\n"
    
    # Add metadata
    markdown += "## Metadata\n\n"
    markdown += f"- Summary Type: {summary['summary_type']}\n"
    markdown += f"- Word Count: {summary['word_count']}\n"
    markdown += f"- Original Content Length: {summary['original_content_length']} characters\n"
    
    return markdown


def format_summaries_as_report(response: SummaryResponse, query: str) -> str:
    """
    Format summary results into a readable report.
    
    Args:
        response: Summary response
        query: Original query
        
    Returns:
        Formatted summary report
    """
    report = f"# Summary Report: {query}\n\n"
    
    # Add overall summary if available
    if response["overall_summary"]:
        report += "## Overall Summary\n\n"
        report += response["overall_summary"]["summary"] + "\n\n"
        
        report += "### Key Findings\n\n"
        for point in response["overall_summary"]["key_points"]:
            report += f"- {point}\n"
        report += "\n"
    
    # Add individual summaries
    report += "## Individual Source Summaries\n\n"
    for i, summary in enumerate(response["content_summaries"]):
        report += f"### Source {i+1}: {summary.get('title', 'Untitled')}\n"
        report += f"**URL:** {summary['url']}\n\n"
        report += f"{summary['summary']}\n\n"
        
        report += "**Key Points:**\n"
        for point in summary["key_points"]:
            report += f"- {point}\n"
        report += "\n"
    
    # Add stats
    report += "## Summary Statistics\n\n"
    report += f"- Summarized {response['execution_stats']['content_items_count']} content items\n"
    report += f"- Total words summarized: {response['execution_stats']['total_words_summarized']}\n"
    report += f"- Average compression ratio: {response['execution_stats']['average_compression_ratio']}\n"
    report += f"- Processing time: {response['execution_stats']['duration_seconds']} seconds\n\n"
    
    # Add errors if any
    if response["errors"]:
        report += "## Errors\n\n"
        for error in response["errors"]:
            report += f"- {error['type']}: {error.get('error', 'Unknown error')}\n"
        report += "\n"
    
    return report


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Summary Agent")
    parser.add_argument("--input", type=str, required=True, help="JSON file with content items to summarize")
    parser.add_argument("--query", type=str, required=True, help="Research query for context")
    parser.add_argument("--type", type=str, choices=["detailed", "concise", "bullet_points", "analytical"], 
                        default="detailed", help="Summary type")
    parser.add_argument("--max-length", type=int, default=500, help="Maximum summary length in words")
    parser.add_argument("--variations", action="store_true", help="Generate multiple summary variations")
    parser.add_argument("--memory-path", type=str, help="Path for memory storage")
    parser.add_argument("--output", type=str, help="Output file for summary report")
    parser.add_argument("--namespace", type=str, help="Namespace for storing in memory")
    args = parser.parse_args()
    
    # Set memory storage path if provided
    if args.memory_path:
        os.environ["MEMORY_STORAGE_PATH"] = args.memory_path
    
    # Load content items from input file
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            content_items = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)
    
    # Create the Summary Agent
    agent = SummaryAgent()
    
    try:
        # Create summary request
        request: SummaryRequest = {
            "content_items": content_items,
            "query": args.query,
            "summary_type": args.type,
            "max_length": args.max_length,
            "focus_aspects": None,  # Could be added as a command-line argument
            "namespace": args.namespace or args.query  # Use query as namespace if not specified
        }
        
        print(f"Starting summarization for query: {args.query}")
        print(f"Summary type: {args.type}, Max length: {args.max_length}")
        print(f"Processing {len(content_items)} content items...")
        
        # Perform summarization
        if args.variations:
            print("Generating multiple summary variations...")
            variations = ["detailed", "concise", "bullet_points"]
            responses = agent.summarize_with_variations(request, variations=variations)
            
            # Use the requested type for the main report
            response = responses[args.type]
            
            # Create additional reports for each variation
            for variation_type, variation_response in responses.items():
                variation_report = format_summaries_as_report(variation_response, args.query)
                
                # Save variation report
                if args.output:
                    variation_output = args.output.replace(".md", f"_{variation_type}.md")
                    with open(variation_output, "w", encoding="utf-8") as f:
                        f.write(variation_report)
                    print(f"{variation_type.title()} summary report saved to: {variation_output}")
        else:
            print("Performing summarization...")
            response = agent.summarize(request)
        
        # Format results
        report = format_summaries_as_report(response, args.query)
        
        # Output results
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Summary report saved to: {args.output}")
        else:
            print("\n" + "="*80 + "\n")
            print(report)
            print("\n" + "="*80 + "\n")
        
        # Print stats
        print("Summarization completed!")
        print(f"- Summarized {response['execution_stats']['content_items_count']} content items")
        print(f"- Total words processed: {response['execution_stats']['total_words_summarized']}")
        print(f"- Average compression ratio: {response['execution_stats']['average_compression_ratio']}")
        print(f"- Processing took {response['execution_stats']['duration_seconds']} seconds")
        
        if response['errors']:
            print(f"- Encountered {len(response['errors'])} errors")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up
        agent.cleanup()
