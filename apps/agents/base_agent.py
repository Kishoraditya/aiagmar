"""
Base Agent

This module provides the BaseAgent class that serves as a foundation for all specialized agents
in the system. It implements common functionality such as LLM interaction, memory management,
and standardized response formatting.
"""

import os
import time
import uuid
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, TypedDict
from datetime import datetime
import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("base_agent")


# -----------------------------------------------------------------------------
# Type Definitions
# -----------------------------------------------------------------------------

class AgentResponse(TypedDict):
    """Standard structure for agent responses."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]]
    execution_stats: Dict[str, Any]
    errors: List[Dict[str, Any]]


class AgentRequest(TypedDict):
    """Standard structure for agent requests."""
    task: str
    parameters: Dict[str, Any]
    session_id: Optional[str]
    namespace: Optional[str]


# -----------------------------------------------------------------------------
# Base Agent Class
# -----------------------------------------------------------------------------

class BaseAgent:
    """
    Base class for all agents in the system.
    
    Provides common functionality such as:
    - LLM interaction
    - Memory management
    - Standardized response formatting
    - Error handling
    - Execution statistics
    """
    
    def __init__(self, name: str, description: str = "", model: str = "gpt-4o", temperature: float = 0.2):
        """
        Initialize the base agent.
        
        Args:
            name: Name of the agent
            description: Description of the agent's role and capabilities
            model: LLM model to use
            temperature: Temperature for LLM generation
        """
        self.name = name
        self.description = description
        self.model_name = model
        self.temperature = temperature
        self.llm = self._create_llm()
        self.session_id = str(uuid.uuid4())
    
    def _create_llm(self) -> ChatOpenAI:
        """Create a language model instance."""
        return ChatOpenAI(model=self.model_name, temperature=self.temperature)
    
    def create_response(self, success: bool, message: str, data: Optional[Dict[str, Any]] = None, 
                       errors: Optional[List[Dict[str, Any]]] = None, 
                       start_time: Optional[float] = None) -> AgentResponse:
        """
        Create a standardized agent response.
        
        Args:
            success: Whether the operation was successful
            message: Response message
            data: Optional data to include in the response
            errors: Optional list of errors
            start_time: Optional start time for calculating execution duration
            
        Returns:
            Standardized agent response
        """
        end_time = time.time()
        duration = round(end_time - (start_time or end_time), 2)
        
        return {
            "success": success,
            "message": message,
            "data": data or {},
            "execution_stats": {
                "agent": self.name,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)) if start_time else None,
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": duration,
                "model": self.model_name,
                "session_id": self.session_id
            },
            "errors": errors or []
        }
    
    def execute_prompt(self, prompt_template: Union[str, ChatPromptTemplate], 
                      variables: Dict[str, Any], 
                      output_parser: Optional[Any] = None,
                      config: Optional[RunnableConfig] = None) -> Any:
        """
        Execute a prompt with the language model.
        
        Args:
            prompt_template: Prompt template to use
            variables: Variables to fill in the prompt template
            output_parser: Optional output parser
            config: Optional runnable config
            
        Returns:
            LLM response
        """
        # Create prompt if string is provided
        if isinstance(prompt_template, str):
            prompt_template = ChatPromptTemplate.from_messages([
                SystemMessage(content=prompt_template)
            ])
        
        # Create chain
        if output_parser:
            chain = prompt_template | self.llm | output_parser
        else:
            chain = prompt_template | self.llm | StrOutputParser()
        
        # Execute chain
        return chain.invoke(variables, config=config)
    
    def execute_with_error_handling(self, func: Callable, *args, **kwargs) -> AgentResponse:
        """
        Execute a function with standardized error handling.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Standardized agent response
        """
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # If result is already an AgentResponse, return it
            if isinstance(result, dict) and "success" in result and "message" in result:
                if "execution_stats" not in result:
                    result["execution_stats"] = {
                        "agent": self.name,
                        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
                        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "duration_seconds": round(time.time() - start_time, 2),
                        "model": self.model_name,
                        "session_id": self.session_id
                    }
                return result
            
            # Otherwise, wrap the result in a standard response
            return self.create_response(
                success=True,
                message="Operation completed successfully",
                data={"result": result} if result is not None else {},
                start_time=start_time
            )
        
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}", exc_info=True)
            return self.create_response(
                success=False,
                message=f"Error: {str(e)}",
                errors=[{
                    "type": "execution_error",
                    "error": str(e)
                }],
                start_time=start_time
            )
    
    def process_request(self, request: AgentRequest) -> AgentResponse:
        """
        Process a standardized agent request.
        
        This method should be implemented by subclasses.
        
        Args:
            request: Agent request
            
        Returns:
            Agent response
        """
        return self.create_response(
            success=False,
            message="Method not implemented",
            errors=[{
                "type": "not_implemented",
                "error": "The process_request method must be implemented by subclasses"
            }]
        )
    
    def cleanup(self):
        """
        Clean up resources used by the agent.
        
        This method should be implemented by subclasses if they need to clean up resources.
        """
        pass
    
    def __enter__(self):
        """Support for context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting context."""
        self.cleanup()
