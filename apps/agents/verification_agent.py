"""
Verification Agent

This agent verifies facts by searching for additional sources, ensuring accuracy and reliability of information.
It relies on Brave Search MCP to find corroborating sources.
"""

import os
import uuid
import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Union, Literal, TypedDict, cast

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

# Import MCP clients
from apps.mcps.brave_search_mcp import BraveSearchMCP
from apps.mcps.memory_mcp import MemoryMCP

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("verification_agent")


# -----------------------------------------------------------------------------
# Type Definitions
# -----------------------------------------------------------------------------

class Claim(TypedDict):
    """Structure for claims to verify."""
    id: str
    text: str
    source: Optional[str]
    context: Optional[str]
    importance: Optional[str]  # "high", "medium", "low"


class VerificationResult(TypedDict):
    """Structure for verification results."""
    claim_id: str
    claim_text: str
    verification_status: str  # "verified", "refuted", "uncertain"
    confidence: float  # 0.0 to 1.0
    supporting_evidence: List[Dict[str, str]]
    contradicting_evidence: List[Dict[str, str]]
    search_queries: List[str]
    suggested_correction: Optional[str]
    verification_summary: str


class VerificationRequest(TypedDict):
    """Structure for verification requests."""
    claims: List[Claim]
    context: Optional[str]
    search_depth: Optional[int]  # Number of sources to check per claim
    confidence_threshold: Optional[float]  # Minimum confidence to consider verified
    namespace: Optional[str]


class VerificationResponse(TypedDict):
    """Structure for verification responses."""
    results: List[VerificationResult]
    overall_assessment: Dict[str, Any]
    execution_stats: Dict[str, Any]
    errors: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# MCP Client Management
# -----------------------------------------------------------------------------

class MCPClientManager:
    """Manages connections to MCP services used by the Verification Agent."""
    
    def __init__(self):
        """Initialize MCP client manager."""
        self.brave_search_mcp = None
        self.memory_mcp = None
    
    def get_brave_search_mcp(self) -> BraveSearchMCP:
        """Get or create Brave Search MCP client."""
        if self.brave_search_mcp is None:
            api_key = os.environ.get("BRAVE_API_KEY")
            if not api_key:
                raise ValueError("BRAVE_API_KEY environment variable is required")
            self.brave_search_mcp = BraveSearchMCP(api_key=api_key)
        return self.brave_search_mcp
    
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
        if self.memory_mcp:
            self.memory_mcp.close()


# Create a singleton instance
mcp_manager = MCPClientManager()


# -----------------------------------------------------------------------------
# Verification Agent Core Functions
# -----------------------------------------------------------------------------

def create_llm(model: str = "gpt-4o", temperature: float = 0.1):
    """Create a language model instance with low temperature for factual tasks."""
    return ChatOpenAI(model=model, temperature=temperature)


def generate_verification_queries(claim: Claim, context: Optional[str] = None) -> List[str]:
    """
    Generate effective search queries to verify a claim.
    
    Args:
        claim: Claim to verify
        context: Optional context for the claim
        
    Returns:
        List of search queries
    """
    llm = create_llm()
    
    # Create prompt for generating verification queries
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a fact-checking assistant. Your task is to generate effective search 
        queries to verify the accuracy of a claim. Create 2-3 different search queries that will help find 
        reliable sources to confirm or refute the claim. Consider:
        
        1. Include key entities and specific facts from the claim
        2. Use neutral phrasing that doesn't assume the claim is true or false
        3. Include terms like "fact check" or "evidence" where appropriate
        4. For numerical claims, include specific numbers or ranges
        
        Format your response as a JSON array of strings."""),
        HumanMessage(content="Claim to verify: {claim}")
    ])
    
    # Add context if available
    if context:
        prompt_messages = prompt.messages.copy()
        prompt_messages.append(SystemMessage(content="Additional context: {context}"))
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
    
    # Execute the prompt
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "claim": claim["text"],
        "context": context
    })
    
    try:
        # Parse the result as JSON
        search_queries = json.loads(result)
        if not isinstance(search_queries, list):
            search_queries = [search_queries]  # Ensure we have a list
        
        # Ensure we have at least one query
        if not search_queries:
            search_queries = [claim["text"]]
        
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
            search_queries = [claim["text"]]
        
        return search_queries


def perform_verification_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Perform a search to verify a claim using Brave Search MCP.
    
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
        
        # Extract URLs, titles, and snippets from results
        structured_results = []
        
        # Parse the results
        entries = results.split('\n\n')
        current_entry = {}
        
        for entry in entries:
            lines = entry.strip().split('\n')
            if len(lines) >= 3:  # Typical entry has Title, Description, URL
                title = lines[0].replace('Title:', '').strip() if lines[0].startswith('Title:') else ''
                description = lines[1].replace('Description:', '').strip() if lines[1].startswith('Description:') else ''
                url = lines[2].replace('URL:', '').strip() if lines[2].startswith('URL:') else ''
                
                if url:  # Only add if we have a URL
                    structured_results.append({
                        "title": title,
                        "snippet": description,
                        "url": url
                    })
        
        return {
            "query": query,
            "results": structured_results,
            "raw_results": results
        }
        
    except Exception as e:
        logger.error(f"Error performing verification search for query '{query}': {e}")
        return {
            "query": query,
            "results": [],
            "raw_results": f"Error: {str(e)}",
            "error": str(e)
        }


def extract_evidence_from_search_results(search_results: Dict[str, Any], claim: Claim) -> Dict[str, Any]:
    """
    Extract relevant evidence from search results for a claim.
    
    Args:
        search_results: Search results
        claim: Claim being verified
        
    Returns:
        Extracted evidence
    """
    # Initialize evidence containers
    supporting_evidence = []
    contradicting_evidence = []
    neutral_evidence = []
    
    # Extract evidence from search results
    for result in search_results.get("results", []):
        # Combine title and snippet for analysis
        content = f"{result.get('title', '')} {result.get('snippet', '')}"
        
        # Simple heuristic to categorize evidence
        # This is a basic approach - the LLM-based analysis will be more sophisticated
        if content:
            evidence = {
                "text": content,
                "source": result.get("url", "Unknown source"),
                "title": result.get("title", "Untitled")
            }
            
            # For now, collect all as neutral evidence
            # The actual categorization will be done by the LLM
            neutral_evidence.append(evidence)
    
    return {
        "supporting": supporting_evidence,
        "contradicting": contradicting_evidence,
        "neutral": neutral_evidence
    }


def analyze_claim_with_evidence(claim: Claim, evidence: Dict[str, Any], 
                               context: Optional[str] = None) -> VerificationResult:
    """
    Analyze a claim with collected evidence to determine verification status.
    
    Args:
        claim: Claim to verify
        evidence: Evidence collected for the claim
        context: Optional context for the claim
        
    Returns:
        Verification result
    """
    llm = create_llm()
    
    # Combine all evidence
    all_evidence = []
    all_evidence.extend(evidence.get("supporting", []))
    all_evidence.extend(evidence.get("contradicting", []))
    all_evidence.extend(evidence.get("neutral", []))
    
    # Format evidence for the prompt
    formatted_evidence = ""
    for i, item in enumerate(all_evidence):
        formatted_evidence += f"Source {i+1}: {item.get('title', 'Untitled')}\n"
        formatted_evidence += f"URL: {item.get('source', 'Unknown')}\n"
        formatted_evidence += f"Content: {item.get('text', '')}\n\n"
    
    # Create prompt for analyzing claim with evidence
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a fact-checking assistant. Your task is to analyze a claim against 
        the provided evidence and determine if the claim is verified, refuted, or uncertain. Follow these steps:
        
        1. Carefully analyze each piece of evidence in relation to the claim
        2. Identify evidence that supports or contradicts the claim
        3. Assess the reliability and relevance of each source
        4. Determine an overall verification status and confidence level
        5. Provide a suggested correction if the claim is refuted
        
        Format your response as JSON with these keys:
        - verification_status: "verified", "refuted", or "uncertain"
        - confidence: a number between 0.0 and 1.0
        - supporting_evidence: array of objects with "text" and "source" keys
        - contradicting_evidence: array of objects with "text" and "source" keys
        - suggested_correction: string (null if not applicable)
        - verification_summary: string explaining your reasoning"""),
        HumanMessage(content="Claim to verify: {claim}"),
        HumanMessage(content="Evidence: {evidence}")
    ])
    
    # Add context if available
    if context:
        prompt_messages = prompt.messages.copy()
        prompt_messages.append(SystemMessage(content="Additional context: {context}"))
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
    
    # Execute the prompt
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "claim": claim["text"],
        "evidence": formatted_evidence,
        "context": context
    })
    
    try:
        # Parse the result as JSON
        analysis = json.loads(result)
        
        # Create verification result
        verification_result: VerificationResult = {
            "claim_id": claim["id"],
            "claim_text": claim["text"],
            "verification_status": analysis.get("verification_status", "uncertain"),
            "confidence": analysis.get("confidence", 0.0),
            "supporting_evidence": analysis.get("supporting_evidence", []),
            "contradicting_evidence": analysis.get("contradicting_evidence", []),
            "search_queries": [],  # Will be filled later
            "suggested_correction": analysis.get("suggested_correction"),
            "verification_summary": analysis.get("verification_summary", "No summary provided")
        }
        
        return verification_result
    
    except json.JSONDecodeError:
        # If parsing fails, create a basic verification result
        logger.error(f"Error parsing analysis result: {result[:100]}...")
        
        # Extract verification status using regex
        status_match = re.search(r"status:?\s*[\"']?(verified|refuted|uncertain)[\"']?", result, re.IGNORECASE)
        status = status_match.group(1).lower() if status_match else "uncertain"
        
        # Extract confidence using regex
        confidence_match = re.search(r"confidence:?\s*(\d+\.?\d*)", result)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        
        # Create basic verification result
        verification_result: VerificationResult = {
            "claim_id": claim["id"],
            "claim_text": claim["text"],
            "verification_status": status,
            "confidence": confidence,
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "search_queries": [],  # Will be filled later
            "suggested_correction": None,
            "verification_summary": result[:500]  # Use first 500 chars as summary
        }
        
        return verification_result


def create_overall_assessment(verification_results: List[VerificationResult]) -> Dict[str, Any]:
    """
    Create an overall assessment of all verification results.
    
    Args:
        verification_results: List of verification results
        
    Returns:
        Overall assessment
    """
    # Count verification statuses
    total_claims = len(verification_results)
    verified_count = sum(1 for r in verification_results if r["verification_status"] == "verified")
    refuted_count = sum(1 for r in verification_results if r["verification_status"] == "refuted")
    uncertain_count = sum(1 for r in verification_results if r["verification_status"] == "uncertain")
    
    # Calculate average confidence
    avg_confidence = sum(r["confidence"] for r in verification_results) / total_claims if total_claims > 0 else 0
    
    # Identify most concerning claims (refuted with high confidence)
    concerning_claims = [
        {
            "claim_id": r["claim_id"],
            "claim_text": r["claim_text"],
            "confidence": r["confidence"],
            "suggested_correction": r["suggested_correction"]
        }
        for r in verification_results
        if r["verification_status"] == "refuted" and r["confidence"] >= 0.7
    ]
    
    # Identify most reliable claims (verified with high confidence)
    reliable_claims = [
        {
            "claim_id": r["claim_id"],
            "claim_text": r["claim_text"],
            "confidence": r["confidence"]
        }
        for r in verification_results
        if r["verification_status"] == "verified" and r["confidence"] >= 0.8
    ]
    
    # Create overall assessment
    assessment = {
        "total_claims": total_claims,
        "verified_count": verified_count,
        "refuted_count": refuted_count,
        "uncertain_count": uncertain_count,
        "verification_rate": verified_count / total_claims if total_claims > 0 else 0,
        "refutation_rate": refuted_count / total_claims if total_claims > 0 else 0,
        "average_confidence": avg_confidence,
        "concerning_claims": concerning_claims,
        "reliable_claims": reliable_claims,
        "overall_reliability": "high" if verified_count / total_claims >= 0.8 and total_claims > 0 else
                              "medium" if verified_count / total_claims >= 0.5 and total_claims > 0 else
                              "low"
    }
    
    return assessment


# -----------------------------------------------------------------------------
# Verification Agent Class
# -----------------------------------------------------------------------------

class VerificationAgent:
    """
    Verification Agent that verifies facts by searching for additional sources.
    """
    
    def __init__(self):
        """Initialize the Verification Agent."""
        pass
    
    def verify_claim(self, claim: Claim, context: Optional[str] = None, 
                    search_depth: int = 3) -> VerificationResult:
        """
        Verify a single claim.
        
        Args:
            claim: Claim to verify
            context: Optional context for the claim
            search_depth: Number of sources to check
            
        Returns:
            Verification result
        """
        # Generate search queries
        search_queries = generate_verification_queries(claim, context)
        
        # Collect evidence from search results
        all_evidence = {
            "supporting": [],
            "contradicting": [],
            "neutral": []
        }
        
        # Perform searches and collect evidence
        for query in search_queries:
            search_results = perform_verification_search(query, max_results=search_depth)
            evidence = extract_evidence_from_search_results(search_results, claim)
            
            # Merge evidence
            all_evidence["supporting"].extend(evidence["supporting"])
            all_evidence["contradicting"].extend(evidence["contradicting"])
            all_evidence["neutral"].extend(evidence["neutral"])
        
        # Analyze claim with collected evidence
        verification_result = analyze_claim_with_evidence(claim, all_evidence, context)
        
        # Add search queries to result
        verification_result["search_queries"] = search_queries
        
        return verification_result
    
    def verify(self, request: VerificationRequest) -> VerificationResponse:
        """
        Verify multiple claims based on the request.
        
        Args:
            request: Verification request
            
        Returns:
            Verification response
        """
        start_time = time.time()
        
        # Initialize response
        response: VerificationResponse = {
            "results": [],
            "overall_assessment": {},
            "execution_stats": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 0,
                "claims_count": len(request["claims"]),
                "verified_count": 0,
                "refuted_count": 0,
                "uncertain_count": 0,
                "searches_performed": 0
            },
            "errors": []
        }
        
        try:
            # Set defaults
            context = request.get("context")
            search_depth = request.get("search_depth", 3)
            confidence_threshold = request.get("confidence_threshold", 0.7)
            
            # Verify each claim
            verification_results = []
            searches_performed = 0
            
            for claim in request["claims"]:
                try:
                    # Verify claim
                    result = self.verify_claim(
                        claim,
                        context=context,
                        search_depth=search_depth
                    )
                    
                    verification_results.append(result)
                    searches_performed += len(result["search_queries"])
                    
                    # Update counts based on verification status
                    if result["verification_status"] == "verified":
                        response["execution_stats"]["verified_count"] += 1
                    elif result["verification_status"] == "refuted":
                        response["execution_stats"]["refuted_count"] += 1
                    else:  # uncertain
                        response["execution_stats"]["uncertain_count"] += 1
                    
                except Exception as e:
                    logger.error(f"Error verifying claim '{claim['text']}': {e}")
                    response["errors"].append({
                        "type": "verification_error",
                        "claim_id": claim["id"],
                        "claim_text": claim["text"],
                        "error": str(e)
                    })
            
            # Update response with verification results
            response["results"] = verification_results
            
            # Create overall assessment
            response["overall_assessment"] = create_overall_assessment(verification_results)
            
            # Store results in memory if namespace provided
            if request.get("namespace"):
                try:
                    memory_mcp = mcp_manager.get_memory_mcp()
                    
                    # Store verification results
                    memory_mcp.store_memory(
                        key=f"verification_results_{uuid.uuid4()}",
                        value=json.dumps(response["results"]),
                        namespace=request["namespace"]
                    )
                    
                    # Store overall assessment
                    memory_mcp.store_memory(
                        key=f"verification_assessment_{uuid.uuid4()}",
                        value=json.dumps(response["overall_assessment"]),
                        namespace=request["namespace"]
                    )
                except Exception as e:
                    logger.error(f"Error storing verification results in memory: {e}")
                    response["errors"].append({
                        "type": "storage_error",
                        "error": str(e)
                    })
            
            # Update execution stats
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            response["execution_stats"]["searches_performed"] = searches_performed
            
            return response
            
        except Exception as e:
            logger.error(f"Error in verification process: {e}")
            response["errors"].append({
                "type": "general_error",
                "error": str(e)
            })
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            return response
    
    def verify_text(self, text: str, context: Optional[str] = None, 
                   search_depth: int = 3) -> VerificationResponse:
        """
        Extract and verify claims from a text.
        
        Args:
            text: Text containing claims to verify
            context: Optional context for the claims
            search_depth: Number of sources to check per claim
            
        Returns:
            Verification response
        """
        # Extract claims from text
        claims = self.extract_claims_from_text(text, context)
        
        # Create verification request
        request: VerificationRequest = {
            "claims": claims,
            "context": context,
            "search_depth": search_depth,
            "confidence_threshold": 0.7,
            "namespace": None
        }
        
        # Verify claims
        return self.verify(request)
    
    def extract_claims_from_text(self, text: str, context: Optional[str] = None) -> List[Claim]:
        """
        Extract verifiable claims from a text.
        
        Args:
            text: Text to extract claims from
            context: Optional context for the claims
            
        Returns:
            List of extracted claims
        """
        llm = create_llm()
        
        # Create prompt for extracting claims
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a claim extraction assistant. Your task is to identify specific, 
            verifiable factual claims from the provided text. Focus on:
            
            1. Statements presented as facts that can be verified
            2. Specific assertions about events, statistics, or relationships
            3. Claims that are concrete enough to be checked against other sources
            
            Ignore opinions, subjective statements, or vague generalizations.
            
            Format your response as a JSON array of objects, each with:
            - id: a unique identifier (e.g., "claim1", "claim2")
            - text: the exact claim text
            - importance: "high", "medium", or "low" based on how central the claim is
            
            Extract up to 5 of the most significant claims."""),
            HumanMessage(content="Text to analyze: {text}")
        ])
        
        # Add context if available
        if context:
            prompt_messages = prompt.messages.copy()
            prompt_messages.append(SystemMessage(content="Additional context: {context}"))
            prompt = ChatPromptTemplate.from_messages(prompt_messages)
        
        # Execute the prompt
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({
            "text": text,
            "context": context
        })
        
        try:
            # Parse the result as JSON
            extracted_claims = json.loads(result)
            
            # Ensure each claim has required fields
            claims = []
            for claim in extracted_claims:
                if "text" in claim and "id" in claim:
                    claims.append({
                        "id": claim["id"],
                        "text": claim["text"],
                        "source": None,
                        "context": context,
                        "importance": claim.get("importance", "medium")
                    })
            
            return claims
        
        except json.JSONDecodeError:
            # If parsing fails, extract claims manually
            claims = []
            claim_id = 1
            
            # Simple regex-based extraction
            claim_patterns = [
                r"(?:^|\n)(?:\d+\.\s*|\*\s*|-\s*|•\s*)?([A-Z][^.!?]*(?:[.!?])['\"]?)",  # Numbered or bulleted statements
                r"([A-Z][^.!?]*(?:is|are|was|were|has|have|had|will)[^.!?]*(?:[.!?])['\"]?)"  # Statements with common verbs
            ]
            
            extracted_texts = []
            for pattern in claim_patterns:
                matches = re.findall(pattern, text)
                extracted_texts.extend(matches)
            
            # Deduplicate and create claims
            seen_texts = set()
            for text in extracted_texts:
                text = text.strip()
                if text and text not in seen_texts and len(text) > 20:  # Minimum length to filter noise
                    seen_texts.add(text)
                    claims.append({
                        "id": f"claim{claim_id}",
                        "text": text,
                        "source": None,
                        "context": context,
                        "importance": "medium"
                    })
                    claim_id += 1
                    
                    # Limit to 5 claims
                    if len(claims) >= 5:
                        break
            
            return claims
    
    def cleanup(self):
        """Clean up resources used by the Verification Agent."""
        # Close all MCP clients
        mcp_manager.close_all()


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def format_verification_result_as_markdown(result: VerificationResult) -> str:
    """
    Format a verification result as Markdown.
    
    Args:
        result: Verification result to format
        
    Returns:
        Markdown-formatted verification result
    """
    # Determine emoji based on verification status
    status_emoji = {
        "verified": "✅",
        "refuted": "❌",
        "uncertain": "❓"
    }
    emoji = status_emoji.get(result["verification_status"], "❓")
    
    markdown = f"## Claim: {result['claim_text']}\n\n"
    markdown += f"**Verification Status:** {emoji} {result['verification_status'].title()} (Confidence: {result['confidence']:.2f})\n\n"
    
    # Add verification summary
    markdown += f"### Summary\n\n{result['verification_summary']}\n\n"
    
    # Add suggested correction if available
    if result["suggested_correction"]:
        markdown += f"### Suggested Correction\n\n{result['suggested_correction']}\n\n"
    
    # Add supporting evidence
    if result["supporting_evidence"]:
        markdown += "### Supporting Evidence\n\n"
        for evidence in result["supporting_evidence"]:
            markdown += f"- **Source:** [{evidence.get('source', 'Unknown')}]({evidence.get('source', '#')})\n"
            markdown += f"  {evidence.get('text', 'No text provided')}\n\n"
    
    # Add contradicting evidence
    if result["contradicting_evidence"]:
        markdown += "### Contradicting Evidence\n\n"
        for evidence in result["contradicting_evidence"]:
            markdown += f"- **Source:** [{evidence.get('source', 'Unknown')}]({evidence.get('source', '#')})\n"
            markdown += f"  {evidence.get('text', 'No text provided')}\n\n"
    
    # Add search queries
    markdown += "### Search Queries Used\n\n"
    for query in result["search_queries"]:
        markdown += f"- {query}\n"
    
    return markdown


def format_verification_as_report(response: VerificationResponse) -> str:
    """
    Format verification results into a readable report.
    
    Args:
        response: Verification response
        
    Returns:
        Formatted verification report
    """
    report = "# Fact Verification Report\n\n"
    
    # Add overall assessment
    assessment = response["overall_assessment"]
    report += "## Overall Assessment\n\n"
    report += f"- **Total Claims:** {assessment['total_claims']}\n"
    report += f"- **Verified:** {assessment['verified_count']} ({assessment['verification_rate']:.0%})\n"
    report += f"- **Refuted:** {assessment['refuted_count']} ({assessment['refutation_rate']:.0%})\n"
    report += f"- **Uncertain:** {assessment['uncertain_count']} ({1 - assessment['verification_rate'] - assessment['refutation_rate']:.0%})\n"
    report += f"- **Average Confidence:** {assessment['average_confidence']:.2f}\n"
    report += f"- **Overall Reliability:** {assessment['overall_reliability'].title()}\n\n"
    
    # Add concerning claims
    if assessment.get("concerning_claims"):
        report += "### Concerning Claims (Refuted with High Confidence)\n\n"
        for claim in assessment["concerning_claims"]:
            report += f"- **Claim:** {claim['claim_text']}\n"
            if claim.get("suggested_correction"):
                report += f"  **Correction:** {claim['suggested_correction']}\n"
            report += "\n"
    
    # Add individual verification results
    report += "## Detailed Verification Results\n\n"
    for result in response["results"]:
        # Determine emoji based on verification status
        status_emoji = {
            "verified": "✅",
            "refuted": "❌",
            "uncertain": "❓"
        }
        emoji = status_emoji.get(result["verification_status"], "❓")
        
        report += f"### {emoji} Claim: {result['claim_text']}\n\n"
        report += f"**Status:** {result['verification_status'].title()} (Confidence: {result['confidence']:.2f})\n\n"
        report += f"{result['verification_summary']}\n\n"
        
        # Add suggested correction if available
        if result.get("suggested_correction"):
            report += f"**Suggested Correction:** {result['suggested_correction']}\n\n"
        
        # Add evidence counts
        supporting_count = len(result.get("supporting_evidence", []))
        contradicting_count = len(result.get("contradicting_evidence", []))
        report += f"**Evidence:** {supporting_count} supporting, {contradicting_count} contradicting\n\n"
        
        # Add divider
        report += "---\n\n"
    
    # Add stats
    report += "## Verification Statistics\n\n"
    report += f"- Verified {response['execution_stats']['claims_count']} claims\n"
    report += f"- Performed {response['execution_stats']['searches_performed']} searches\n"
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
    import sys
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Verification Agent")
    parser.add_argument("--input", type=str, help="JSON file with claims to verify or text file to extract claims from")
    parser.add_argument("--text", type=str, help="Text containing claims to verify")
    parser.add_argument("--claim", type=str, help="Single claim to verify")
    parser.add_argument("--context", type=str, help="Context for the claims")
    parser.add_argument("--depth", type=int, default=3, help="Search depth per claim")
    parser.add_argument("--memory-path", type=str, help="Path for memory storage")
    parser.add_argument("--output", type=str, help="Output file for verification report")
    parser.add_argument("--namespace", type=str, help="Namespace for storing in memory")
    args = parser.parse_args()
    
    # Set memory storage path if provided
    if args.memory_path:
        os.environ["MEMORY_STORAGE_PATH"] = args.memory_path
    
    # Ensure we have input
    if not args.input and not args.text and not args.claim:
        print("Error: Please provide input claims using --input, --text, or --claim")
        sys.exit(1)
    
    # Create the Verification Agent
    agent = VerificationAgent()
    
    try:
        # Process input based on provided arguments
        if args.claim:
            # Verify a single claim
            print(f"Verifying claim: {args.claim}")
            
            claim: Claim = {
                "id": "claim1",
                "text": args.claim,
                "source": None,
                "context": args.context,
                "importance": "high"
            }
            
            request: VerificationRequest = {
                "claims": [claim],
                "context": args.context,
                "search_depth": args.depth,
                "confidence_threshold": 0.7,
                "namespace": args.namespace
            }
            
            response = agent.verify(request)
            
        elif args.text:
            # Extract and verify claims from text
            print(f"Extracting and verifying claims from text...")
            response = agent.verify_text(args.text, context=args.context, search_depth=args.depth)
            
        else:  # args.input
            # Load claims from input file
            print(f"Loading claims from: {args.input}")
            
            try:
                with open(args.input, "r", encoding="utf-8") as f:
                    # Try to parse as JSON
                    try:
                        claims = json.load(f)
                        
                        # Handle different input formats
                        if isinstance(claims, list):
                            # List of claims
                            request: VerificationRequest = {
                                "claims": claims,
                                "context": args.context,
                                "search_depth": args.depth,
                                "confidence_threshold": 0.7,
                                "namespace": args.namespace
                            }
                            
                            response = agent.verify(request)
                        elif isinstance(claims, dict) and "claims" in claims:
                            # Full request object
                            request = claims
                            
                            # Override with command line arguments if provided
                            if args.context:
                                request["context"] = args.context
                            if args.depth:
                                request["search_depth"] = args.depth
                            if args.namespace:
                                request["namespace"] = args.namespace
                            
                            response = agent.verify(request)
                        else:
                            raise ValueError("Invalid JSON format: expected list of claims or request object")
                    
                    except json.JSONDecodeError:
                        # Not JSON, treat as text file
                        f.seek(0)  # Reset file pointer
                        text = f.read()
                        
                        print(f"Extracting and verifying claims from text file...")
                        response = agent.verify_text(text, context=args.context, search_depth=args.depth)
            
            except Exception as e:
                print(f"Error loading input file: {e}")
                sys.exit(1)
        
        # Format results
        report = format_verification_as_report(response)
        
        # Output results
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Verification report saved to: {args.output}")
        else:
            print("\n" + "="*80 + "\n")
            print(report)
            print("\n" + "="*80 + "\n")
        
        # Print stats
        print("Verification completed!")
        print(f"- Verified {response['execution_stats']['claims_count']} claims")
        print(f"- Results: {response['execution_stats']['verified_count']} verified, " +
              f"{response['execution_stats']['refuted_count']} refuted, " +
              f"{response['execution_stats']['uncertain_count']} uncertain")
        print(f"- Performed {response['execution_stats']['searches_performed']} searches")
        print(f"- Processing took {response['execution_stats']['duration_seconds']} seconds")
        
        if response['errors']:
            print(f"- Encountered {len(response['errors'])} errors")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up
        agent.cleanup()
