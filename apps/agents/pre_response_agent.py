"""
Pre-response Agent

This agent interacts with the user, clarifies queries if needed, and presents the research plan 
before execution, collaborating with the Manager Agent.
It relies on Memory MCP to access conversation history and stored plans.
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
from apps.mcps.memory_mcp import MemoryMCP
from apps.mcps.brave_search_mcp import BraveSearchMCP

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pre_response_agent")


# -----------------------------------------------------------------------------
# Type Definitions
# -----------------------------------------------------------------------------

class QueryClarification(TypedDict):
    """Structure for query clarification."""
    original_query: str
    clarified_query: Optional[str]
    clarification_questions: List[str]
    ambiguities: List[str]
    suggested_refinements: List[str]
    needs_clarification: bool


class ResearchPlan(TypedDict):
    """Structure for research plan."""
    query: str
    objective: str
    key_aspects: List[str]
    search_queries: List[str]
    information_sources: List[str]
    expected_outputs: List[str]
    timeline: Dict[str, str]
    agents_involved: List[Dict[str, str]]


class PreResponseRequest(TypedDict):
    """Structure for pre-response requests."""
    query: str
    conversation_history: Optional[List[Dict[str, str]]]
    user_preferences: Optional[Dict[str, Any]]
    namespace: Optional[str]
    session_id: Optional[str]


class PreResponseResult(TypedDict):
    """Structure for pre-response results."""
    original_query: str
    clarified_query: Optional[str]
    research_plan: ResearchPlan
    clarification_needed: bool
    clarification_questions: List[str]
    execution_ready: bool


class PreResponseResponse(TypedDict):
    """Structure for pre-response responses."""
    result: PreResponseResult
    execution_stats: Dict[str, Any]
    errors: List[Dict[str, Any]]


# -----------------------------------------------------------------------------
# MCP Client Management
# -----------------------------------------------------------------------------

class MCPClientManager:
    """Manages connections to MCP services used by the Pre-response Agent."""
    
    def __init__(self):
        """Initialize MCP client manager."""
        self.memory_mcp = None
        self.brave_search_mcp = None
    
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
    
    def close_all(self):
        """Close all MCP clients."""
        if self.memory_mcp:
            self.memory_mcp.close()
        if self.brave_search_mcp:
            self.brave_search_mcp.close()


# Create a singleton instance
mcp_manager = MCPClientManager()


# -----------------------------------------------------------------------------
# Pre-response Agent Core Functions
# -----------------------------------------------------------------------------

def create_llm(model: str = "gpt-4o", temperature: float = 0.7):
    """Create a language model instance."""
    return ChatOpenAI(model=model, temperature=temperature)


def analyze_query(query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> QueryClarification:
    """
    Analyze a query to identify ambiguities and potential clarification needs.
    
    Args:
        query: User query to analyze
        conversation_history: Optional conversation history for context
        
    Returns:
        Query clarification analysis
    """
    llm = create_llm(temperature=0.3)  # Lower temperature for analytical tasks
    
    # Format conversation history if available
    formatted_history = ""
    if conversation_history:
        for message in conversation_history:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            formatted_history += f"{role.capitalize()}: {content}\n\n"
    
    # Create prompt for query analysis
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a query analysis assistant. Your task is to analyze the user's research query 
        to identify any ambiguities, vague terms, or areas that need clarification. Consider:
        
        1. Ambiguous terms or phrases that could have multiple interpretations
        2. Vague concepts that would benefit from more specificity
        3. Missing parameters (time period, geographic scope, etc.)
        4. Potential clarification questions to improve the query
        
        Format your response as JSON with these keys:
        - ambiguities: array of specific ambiguous terms or concepts
        - suggested_refinements: array of specific suggestions to improve the query
        - clarification_questions: array of specific questions to ask the user
        - needs_clarification: boolean indicating if clarification is needed
        - clarified_query: improved version of the query (null if clarification needed from user)"""),
        HumanMessage(content="Query: {query}")
    ])
    
    # Add conversation history if available
    if conversation_history:
        prompt_messages = prompt.messages.copy()
        prompt_messages.append(SystemMessage(content="Conversation history: {history}"))
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
    
    # Execute the prompt
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "query": query,
        "history": formatted_history
    })
    
    try:
        # Parse the result as JSON
        analysis = json.loads(result)
        
        # Create query clarification
        clarification: QueryClarification = {
            "original_query": query,
            "clarified_query": analysis.get("clarified_query"),
            "clarification_questions": analysis.get("clarification_questions", []),
            "ambiguities": analysis.get("ambiguities", []),
            "suggested_refinements": analysis.get("suggested_refinements", []),
            "needs_clarification": analysis.get("needs_clarification", False)
        }
        
        return clarification
    
    except json.JSONDecodeError:
        # If parsing fails, create a basic clarification
        logger.error(f"Error parsing analysis result: {result[:100]}...")
        
        # Extract key information using regex
        ambiguities = re.findall(r"(?:ambiguit(?:y|ies):|unclear:)([^\n]+)", result, re.IGNORECASE)
        questions = re.findall(r"(?:question:|clarification:)([^\n]+)", result, re.IGNORECASE)
        
        # Create basic clarification
        clarification: QueryClarification = {
            "original_query": query,
            "clarified_query": None,
            "clarification_questions": [q.strip() for q in questions if q.strip()],
            "ambiguities": [a.strip() for a in ambiguities if a.strip()],
            "suggested_refinements": [],
            "needs_clarification": len(questions) > 0 or len(ambiguities) > 0
        }
        
        return clarification


def perform_quick_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Perform a quick search to gather context for research planning.
    
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
        logger.error(f"Error performing quick search for query '{query}': {e}")
        return {
            "query": query,
            "results": [],
            "raw_results": f"Error: {str(e)}",
            "error": str(e)
        }


def create_research_plan(query: str, search_results: Optional[Dict[str, Any]] = None, 
                        user_preferences: Optional[Dict[str, Any]] = None) -> ResearchPlan:
    """
    Create a comprehensive research plan based on the query.
    
    Args:
        query: User query
        search_results: Optional quick search results for context
        user_preferences: Optional user preferences
        
    Returns:
        Research plan
    """
    llm = create_llm()
    
    # Format search results if available
    formatted_results = ""
    if search_results and search_results.get("results"):
        formatted_results = "Preliminary search results:\n\n"
        for i, result in enumerate(search_results["results"]):
            formatted_results += f"Result {i+1}: {result.get('title', 'Untitled')}\n"
            formatted_results += f"URL: {result.get('url', 'No URL')}\n"
            formatted_results += f"Snippet: {result.get('snippet', 'No description')}\n\n"
    
    # Format user preferences if available
    formatted_preferences = ""
    if user_preferences:
        formatted_preferences = "User preferences:\n"
        for key, value in user_preferences.items():
            formatted_preferences += f"- {key}: {value}\n"
    
    # Create prompt for research plan
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a research planning assistant. Your task is to create a comprehensive 
        research plan based on the user's query. The plan should include:
        
        1. Clear research objective
        2. Key aspects to investigate
        3. Specific search queries to use
        4. Information sources to consult
        5. Expected outputs
        6. Timeline for research phases
        7. Agents involved and their roles
        
        Format your response as JSON with these keys:
        - objective: string describing the main research goal
        - key_aspects: array of specific aspects to investigate
        - search_queries: array of effective search queries
        - information_sources: array of source types to consult
        - expected_outputs: array of deliverables
        - timeline: object mapping phases to timeframes
        - agents_involved: array of objects with "agent" and "role" keys"""),
        HumanMessage(content="Research query: {query}")
    ])
    
    # Add search results if available
    if search_results and search_results.get("results"):
        prompt_messages = prompt.messages.copy()
        prompt_messages.append(SystemMessage(content="{search_results}"))
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
    
    # Add user preferences if available
    if user_preferences:
        prompt_messages = prompt.messages.copy()
        prompt_messages.append(SystemMessage(content="{preferences}"))
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
    
    # Execute the prompt
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "query": query,
        "search_results": formatted_results,
        "preferences": formatted_preferences
    })
    
    try:
        # Parse the result as JSON
        plan_data = json.loads(result)
        
        # Create research plan
        plan: ResearchPlan = {
            "query": query,
            "objective": plan_data.get("objective", ""),
            "key_aspects": plan_data.get("key_aspects", []),
            "search_queries": plan_data.get("search_queries", []),
            "information_sources": plan_data.get("information_sources", []),
            "expected_outputs": plan_data.get("expected_outputs", []),
            "timeline": plan_data.get("timeline", {}),
            "agents_involved": plan_data.get("agents_involved", [])
        }
        
        return plan
    
    except json.JSONDecodeError:
        # If parsing fails, create a basic plan
        logger.error(f"Error parsing plan result: {result[:100]}...")
        
        # Extract key information using regex
        objective_match = re.search(r"objective:?\s*([^\n]+)", result, re.IGNORECASE)
        objective = objective_match.group(1).strip() if objective_match else "Research the query"
        
        aspects = re.findall(r"(?:aspect|investigate|explore):?\s*([^\n]+)", result, re.IGNORECASE)
        queries = re.findall(r"(?:query|search for):?\s*([^\n]+)", result, re.IGNORECASE)
        sources = re.findall(r"(?:source|consult):?\s*([^\n]+)", result, re.IGNORECASE)
        outputs = re.findall(r"(?:output|deliverable):?\s*([^\n]+)", result, re.IGNORECASE)
        
        # Create basic plan
        plan: ResearchPlan = {
            "query": query,
            "objective": objective,
            "key_aspects": [a.strip() for a in aspects if a.strip()][:5],
            "search_queries": [q.strip() for q in queries if q.strip()][:5] or [query],
            "information_sources": [s.strip() for s in sources if s.strip()][:5] or ["Web search", "Academic databases"],
            "expected_outputs": [o.strip() for o in outputs if o.strip()][:3] or ["Summary report", "Key findings"],
            "timeline": {"research": "1-2 hours", "analysis": "1 hour", "synthesis": "30 minutes"},
            "agents_involved": [
                {"agent": "Research Agent", "role": "Gather information"},
                {"agent": "Summary Agent", "role": "Synthesize findings"},
                {"agent": "Verification Agent", "role": "Verify facts"}
            ]
        }
        
        return plan


def format_plan_for_user(plan: ResearchPlan) -> str:
    """
    Format a research plan for user presentation.
    
    Args:
        plan: Research plan to format
        
    Returns:
        User-friendly formatted plan
    """
    formatted_plan = f"# Research Plan: {plan['query']}\n\n"
    
    # Add objective
    formatted_plan += f"## Objective\n\n{plan['objective']}\n\n"
    
    # Add key aspects
    formatted_plan += "## Key Aspects to Investigate\n\n"
    for i, aspect in enumerate(plan['key_aspects']):
        formatted_plan += f"{i+1}. {aspect}\n"
    formatted_plan += "\n"
    
    # Add search strategy
    formatted_plan += "## Search Strategy\n\n"
    formatted_plan += "I'll use these search queries:\n\n"
    for i, query in enumerate(plan['search_queries']):
        formatted_plan += f"- {query}\n"
    formatted_plan += "\n"
    
    # Add information sources
    formatted_plan += "## Information Sources\n\n"
    for source in plan['information_sources']:
        formatted_plan += f"- {source}\n"
    formatted_plan += "\n"
    
    # Add expected outputs
    formatted_plan += "## Expected Outputs\n\n"
    for output in plan['expected_outputs']:
        formatted_plan += f"- {output}\n"
    formatted_plan += "\n"
    
    # Add timeline
    formatted_plan += "## Timeline\n\n"
    for phase, time in plan['timeline'].items():
        formatted_plan += f"- {phase.capitalize()}: {time}\n"
    formatted_plan += "\n"
    
    # Add agents involved
    formatted_plan += "## Agents Involved\n\n"
    for agent_info in plan['agents_involved']:
        formatted_plan += f"- **{agent_info.get('agent', 'Unknown')}**: {agent_info.get('role', 'No role specified')}\n"
    
    return formatted_plan


def store_plan_in_memory(plan: ResearchPlan, namespace: str, session_id: Optional[str] = None) -> str:
    """
    Store a research plan in memory for later use.
    
    Args:
        plan: Research plan to store
        namespace: Namespace for storage
        session_id: Optional session ID
        
    Returns:
        Memory key where plan is stored
    """
    memory_mcp = mcp_manager.get_memory_mcp()
    
    # Generate a key for the plan
    plan_id = session_id or str(uuid.uuid4())
    key = f"research_plan_{plan_id}"
    
    # Store the plan
    memory_mcp.store_memory(
        key=key,
        value=json.dumps(plan),
        namespace=namespace
    )
    
    return key


# -----------------------------------------------------------------------------
# Pre-response Agent Class
# -----------------------------------------------------------------------------

class PreResponseAgent:
    """
    Pre-response Agent that interacts with the user, clarifies queries, and presents research plans.
    """
    
    def __init__(self):
        """Initialize the Pre-response Agent."""
        pass
    
    def process_query(self, request: PreResponseRequest) -> PreResponseResponse:
        """
        Process a user query, analyze it, and create a research plan.
        
        Args:
            request: Pre-response request
            
        Returns:
            Pre-response response
        """
        start_time = time.time()
        
        # Initialize response
        response: PreResponseResponse = {
            "result": {
                "original_query": request["query"],
                "clarified_query": None,
                "research_plan": {
                    "query": request["query"],
                    "objective": "",
                    "key_aspects": [],
                    "search_queries": [],
                    "information_sources": [],
                    "expected_outputs": [],
                    "timeline": {},
                    "agents_involved": []
                },
                "clarification_needed": False,
                "clarification_questions": [],
                "execution_ready": False
            },
            "execution_stats": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": 0,
                "quick_search_performed": False,
                "clarification_analysis_performed": True,
                "plan_created": False
            },
            "errors": []
        }
        
        try:
            # Analyze query for clarification needs
            query_analysis = analyze_query(
                query=request["query"],
                conversation_history=request.get("conversation_history")
            )
            
            # Update response with analysis results
            response["result"]["clarification_needed"] = query_analysis["needs_clarification"]
            response["result"]["clarification_questions"] = query_analysis["clarification_questions"]
            
            # If clarification is needed, return early
            if query_analysis["needs_clarification"]:
                response["result"]["execution_ready"] = False
                response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
                return response
            
            # Use clarified query if available
            query_to_use = query_analysis["clarified_query"] or request["query"]
            response["result"]["clarified_query"] = query_analysis["clarified_query"]
            
            # Perform quick search for context
            search_results = None
            try:
                search_results = perform_quick_search(query_to_use)
                response["execution_stats"]["quick_search_performed"] = True
            except Exception as e:
                logger.error(f"Error performing quick search: {e}")
                response["errors"].append({
                    "type": "search_error",
                    "error": str(e)
                })
            
            # Create research plan
            plan = create_research_plan(
                query=query_to_use,
                search_results=search_results,
                user_preferences=request.get("user_preferences")
            )
            
            # Update response with plan
            response["result"]["research_plan"] = plan
            response["result"]["execution_ready"] = True
            response["execution_stats"]["plan_created"] = True
            
            # Store plan in memory if namespace provided
            if request.get("namespace"):
                try:
                    plan_key = store_plan_in_memory(
                        plan=plan,
                        namespace=request["namespace"],
                        session_id=request.get("session_id")
                    )
                    
                    # Also store the clarified query if available
                    if query_analysis["clarified_query"]:
                        memory_mcp = mcp_manager.get_memory_mcp()
                        memory_mcp.store_memory(
                            key=f"clarified_query_{request.get('session_id', uuid.uuid4())}",
                            value=query_analysis["clarified_query"],
                            namespace=request["namespace"]
                        )
                except Exception as e:
                    logger.error(f"Error storing plan in memory: {e}")
                    response["errors"].append({
                        "type": "storage_error",
                        "error": str(e)
                    })
            
            # Update execution stats
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in pre-response process: {e}")
            response["errors"].append({
                "type": "general_error",
                "error": str(e)
            })
            response["execution_stats"]["duration_seconds"] = round(time.time() - start_time, 2)
            return response
    
    def get_clarification(self, query: str, ambiguities: List[str], 
                         conversation_history: Optional[List[Dict[str, str]]] = None) -> List[str]:
        """
        Generate clarification questions for ambiguous queries.
        
        Args:
            query: Original query
            ambiguities: List of identified ambiguities
            conversation_history: Optional conversation history
            
        Returns:
            List of clarification questions
        """
        llm = create_llm()
        
        # Format conversation history if available
        formatted_history = ""
        if conversation_history:
            for message in conversation_history:
                role = message.get("role", "unknown")
                content = message.get("content", "")
                formatted_history += f"{role.capitalize()}: {content}\n\n"
        
        # Format ambiguities
        formatted_ambiguities = "\n".join([f"- {a}" for a in ambiguities])
        
        # Create prompt for generating clarification questions
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a query clarification assistant. Your task is to generate 
            clear, specific questions to help clarify ambiguities in the user's query. Create questions that:
            
            1. Address each identified ambiguity
            2. Are specific and focused on one aspect at a time
            3. Can be answered concisely by the user
            4. Will meaningfully improve the research plan
            
            Format your response as a JSON array of strings, with each string being a clarification question."""),
            HumanMessage(content="Query: {query}"),
            SystemMessage(content="Identified ambiguities:\n{ambiguities}")
        ])
        
        # Add conversation history if available
        if conversation_history:
            prompt_messages = prompt.messages.copy()
            prompt_messages.append(SystemMessage(content="Conversation history: {history}"))
            prompt = ChatPromptTemplate.from_messages(prompt_messages)
        
        # Execute the prompt
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({
            "query": query,
            "ambiguities": formatted_ambiguities,
            "history": formatted_history
        })
        
        try:
            # Parse the result as JSON
            questions = json.loads(result)
            return questions if isinstance(questions, list) else [questions]
        
        except json.JSONDecodeError:
            # If parsing fails, extract questions manually
            questions = []
            for line in result.split('\n'):
                line = line.strip()
                if line and (line.endswith('?') or '?' in line):
                    questions.append(line)
            
            return questions if questions else [f"Could you clarify what you mean by {ambiguities[0]}?"] if ambiguities else ["Could you provide more details about your query?"]
    
    def refine_plan_with_feedback(self, plan: ResearchPlan, feedback: str) -> ResearchPlan:
        """
        Refine a research plan based on user feedback.
        
        Args:
            plan: Original research plan
            feedback: User feedback
            
        Returns:
            Refined research plan
        """
        llm = create_llm()
        
        # Format original plan
        formatted_plan = json.dumps(plan, indent=2)
        
        # Create prompt for refining the plan
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a research planning assistant. Your task is to refine 
            a research plan based on user feedback. Modify the plan to incorporate the feedback while 
            maintaining its structure and comprehensiveness.
            
            Format your response as JSON with the same structure as the original plan."""),
            SystemMessage(content="Original plan: {plan}"),
            HumanMessage(content="User feedback: {feedback}")
        ])
        
        # Execute the prompt
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({
            "plan": formatted_plan,
            "feedback": feedback
        })
        
        try:
            # Parse the result as JSON
            refined_plan = json.loads(result)
            
            # Ensure all required fields are present
            for key in plan.keys():
                if key not in refined_plan:
                    refined_plan[key] = plan[key]
            
            return refined_plan
        
        except json.JSONDecodeError:
            # If parsing fails, return the original plan
            logger.error(f"Error parsing refined plan: {result[:100]}...")
            return plan
    
    def cleanup(self):
        """Clean up resources used by the Pre-response Agent."""
        # Close all MCP clients
        mcp_manager.close_all()


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Pre-response Agent")
    parser.add_argument("--query", type=str, required=True, help="User query to process")
    parser.add_argument("--history", type=str, help="JSON file with conversation history")
    parser.add_argument("--preferences", type=str, help="JSON file with user preferences")
    parser.add_argument("--memory-path", type=str, help="Path for memory storage")
    parser.add_argument("--output", type=str, help="Output file for research plan")
    parser.add_argument("--namespace", type=str, default="default", help="Namespace for storing in memory")
    parser.add_argument("--session-id", type=str, help="Session ID for continuity")
    args = parser.parse_args()
    
    # Set memory storage path if provided
    if args.memory_path:
        os.environ["MEMORY_STORAGE_PATH"] = args.memory_path
    
    # Load conversation history if provided
    conversation_history = None
    if args.history:
        try:
            with open(args.history, "r", encoding="utf-8") as f:
                conversation_history = json.load(f)
        except Exception as e:
            print(f"Error loading conversation history: {e}")
            sys.exit(1)
    
    # Load user preferences if provided
    user_preferences = None
    if args.preferences:
        try:
            with open(args.preferences, "r", encoding="utf-8") as f:
                user_preferences = json.load(f)
        except Exception as e:
            print(f"Error loading user preferences: {e}")
            sys.exit(1)
    
    # Create the Pre-response Agent
    agent = PreResponseAgent()
    
    try:
        # Create request
        request: PreResponseRequest = {
            "query": args.query,
            "conversation_history": conversation_history,
            "user_preferences": user_preferences,
            "namespace": args.namespace,
            "session_id": args.session_id
        }
        
        # Process query
        print(f"Processing query: {args.query}")
        response = agent.process_query(request)
        
        # Check if clarification is needed
        if response["result"]["clarification_needed"]:
            print("\nClarification needed:")
            for i, question in enumerate(response["result"]["clarification_questions"]):
                print(f"{i+1}. {question}")
            
            # In a real application, you would wait for user input here
            print("\nIn a real application, you would wait for user input here.")
            print("For this example, we'll proceed with the original query.")
        else:
            # Format the research plan for user presentation
            plan = response["result"]["research_plan"]
            formatted_plan = format_plan_for_user(plan)
            
            # Output the plan
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(formatted_plan)
                print(f"Research plan saved to: {args.output}")
            else:
                print("\n" + "="*80 + "\n")
                print(formatted_plan)
                print("\n" + "="*80 + "\n")
            
            # Print stats
            print("Pre-response processing completed!")
            print(f"- Processing took {response['execution_stats']['duration_seconds']} seconds")
            
            if response["result"]["clarified_query"]:
                print(f"- Original query: {args.query}")
                print(f"- Clarified query: {response['result']['clarified_query']}")
            
            if response["errors"]:
                print(f"- Encountered {len(response['errors'])} errors")
        
        # In a real application, you would ask for user approval here
        print("\nIn a real application, you would ask for user approval of the plan.")
        print("For this example, we'll assume the plan is approved.")
        
        # Example of how to handle user feedback in a real application
        """
        user_feedback = input("Do you approve this research plan? (yes/no): ")
        if user_feedback.lower() in ["yes", "y"]:
            print("Plan approved! Proceeding with research...")
            # Here you would trigger the Manager Agent to execute the plan
        elif user_feedback.lower() in ["no", "n"]:
            print("Please provide feedback to improve the plan:")
            feedback = input("> ")
            refined_plan = agent.refine_plan_with_feedback(plan, feedback)
            print("Plan refined based on your feedback.")
            # Here you would present the refined plan
        else:
            print("Invalid response. Assuming plan is approved.")
        """
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up
        agent.cleanup()


# -----------------------------------------------------------------------------
# Interactive Mode
# -----------------------------------------------------------------------------

def run_interactive_mode():
    """Run the Pre-response Agent in interactive mode."""
    print("Pre-response Agent - Interactive Mode")
    print("====================================")
    print("This agent will help clarify your research query and create a research plan.")
    print("Type 'exit' to quit.\n")
    
    # Create the Pre-response Agent
    agent = PreResponseAgent()
    
    # Initialize conversation history
    conversation_history = []
    session_id = str(uuid.uuid4())
    namespace = "interactive_session"
    
    try:
        while True:
            # Get user query
            query = input("\nEnter your research query: ")
            
            # Check for exit command
            if query.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            
            # Create request
            request: PreResponseRequest = {
                "query": query,
                "conversation_history": conversation_history,
                "user_preferences": None,
                "namespace": namespace,
                "session_id": session_id
            }
            
            # Process query
            print("\nAnalyzing your query...")
            response = agent.process_query(request)
            
            # Check if clarification is needed
            if response["result"]["clarification_needed"]:
                print("\nI need some clarification to better understand your query:")
                for i, question in enumerate(response["result"]["clarification_questions"]):
                    print(f"{i+1}. {question}")
                
                # Get user clarification
                print("\nPlease provide clarification:")
                clarification = input("> ")
                
                # Add to conversation history
                conversation_history.append({"role": "user", "content": query})
                conversation_history.append({"role": "assistant", "content": "I need some clarification: " + 
                                           ", ".join(response["result"]["clarification_questions"])})
                conversation_history.append({"role": "user", "content": clarification})
                
                # Create new request with clarification
                request = {
                    "query": query + " " + clarification,
                    "conversation_history": conversation_history,
                    "user_preferences": None,
                    "namespace": namespace,
                    "session_id": session_id
                }
                
                # Process updated query
                print("\nCreating research plan with your clarification...")
                response = agent.process_query(request)
            else:
                # Add to conversation history
                conversation_history.append({"role": "user", "content": query})
            
            # Format the research plan for user presentation
            plan = response["result"]["research_plan"]
            formatted_plan = format_plan_for_user(plan)
            
            # Present the plan
            print("\n" + "="*80 + "\n")
            print(formatted_plan)
            print("\n" + "="*80 + "\n")
            
            # Add to conversation history
            conversation_history.append({"role": "assistant", "content": "I've created a research plan for your query."})
            
            # Ask for user approval
            approval = input("Do you approve this research plan? (yes/no): ")
            
            if approval.lower() in ["yes", "y"]:
                print("Plan approved! In a full implementation, I would now execute this plan.")
                # Add to conversation history
                conversation_history.append({"role": "user", "content": "I approve the research plan."})
                conversation_history.append({"role": "assistant", "content": "Thank you! I'll proceed with the research."})
            else:
                print("Please provide feedback to improve the plan:")
                feedback = input("> ")
                
                # Add to conversation history
                conversation_history.append({"role": "user", "content": "I don't approve the plan. " + feedback})
                
                # Refine the plan
                print("\nRefining the plan based on your feedback...")
                refined_plan = agent.refine_plan_with_feedback(plan, feedback)
                
                # Format and present the refined plan
                formatted_refined_plan = format_plan_for_user(refined_plan)
                print("\n" + "="*80 + "\n")
                print(formatted_refined_plan)
                print("\n" + "="*80 + "\n")
                
                # Add to conversation history
                conversation_history.append({"role": "assistant", "content": "I've refined the research plan based on your feedback."})
                
                # Store the refined plan
                try:
                    store_plan_in_memory(
                        plan=refined_plan,
                        namespace=namespace,
                        session_id=session_id
                    )
                except Exception as e:
                    print(f"Error storing refined plan: {e}")
    
    except KeyboardInterrupt:
        print("\nExiting interactive mode...")
    
    except Exception as e:
        print(f"Error in interactive mode: {e}")
    
    finally:
        # Clean up
        agent.cleanup()


if __name__ == "__main__" and len(sys.argv) == 1:
    # If no arguments provided, run in interactive mode
    run_interactive_mode()
