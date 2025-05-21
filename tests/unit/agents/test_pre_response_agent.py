"""
Unit tests for the Pre-response Agent.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from apps.agents.pre_response_agent import PreResponseAgent
from apps.mcps.memory_mcp import MemoryMCP
from apps.utils.exceptions import QueryClarificationError, PlanPresentationError

class TestPreResponseAgent:
    """Test suite for PreResponseAgent class."""

    @pytest.fixture
    def memory_mcp(self):
        """Fixture to create a mock MemoryMCP."""
        mock_memory = MagicMock(spec=MemoryMCP)
        mock_memory.store_memory = MagicMock()
        mock_memory.retrieve_memory = MagicMock()
        mock_memory.search_memories = MagicMock()
        return mock_memory

    @pytest.fixture
    def pre_response_agent(self, memory_mcp):
        """Fixture to create a PreResponseAgent instance with mock dependencies."""
        agent = PreResponseAgent(
            name="pre_response",
            memory_mcp=memory_mcp
        )
        return agent

    def test_init(self, memory_mcp):
        """Test initialization of PreResponseAgent."""
        agent = PreResponseAgent(
            name="pre_response",
            memory_mcp=memory_mcp
        )
        
        assert agent.name == "pre_response"
        assert agent.memory_mcp == memory_mcp

    @pytest.mark.asyncio
    async def test_clarify_query_simple(self, pre_response_agent):
        """Test clarifying a simple, clear query."""
        # Call the method with a clear query
        query = "What are the environmental impacts of climate change?"
        
        result = await pre_response_agent.clarify_query(query)
        
        # Verify memory was updated
        pre_response_agent.memory_mcp.store_memory.assert_called_with(
            "original_query", query, namespace="pre_response"
        )
        pre_response_agent.memory_mcp.store_memory.assert_any_call(
            "clarified_query", result["query"], namespace="pre_response"
        )
        
        # Verify result structure
        assert "query" in result
        assert "clarified" in result
        assert "additional_context" in result
        
        # For a clear query, it should not need clarification
        assert result["query"] == query
        assert result["clarified"] is False
        assert result["additional_context"] == ""

    @pytest.mark.asyncio
    async def test_clarify_query_ambiguous(self, pre_response_agent):
        """Test clarifying an ambiguous query that needs clarification."""
        # Mock the _get_user_clarification method
        pre_response_agent._get_user_clarification = AsyncMock(return_value={
            "clarified_query": "What are the environmental impacts of plastic pollution in oceans?",
            "additional_context": "Focus on marine life and ecosystems"
        })
        
        # Call the method with an ambiguous query
        query = "Tell me about plastic pollution"
        
        result = await pre_response_agent.clarify_query(query)
        
        # Verify _get_user_clarification was called
        pre_response_agent._get_user_clarification.assert_called_once_with(query)
        
        # Verify memory was updated
        pre_response_agent.memory_mcp.store_memory.assert_called_with(
            "original_query", query, namespace="pre_response"
        )
        pre_response_agent.memory_mcp.store_memory.assert_any_call(
            "clarified_query", result["query"], namespace="pre_response"
        )
        pre_response_agent.memory_mcp.store_memory.assert_any_call(
            "query_context", result["additional_context"], namespace="pre_response"
        )
        
        # Verify result structure
        assert result["query"] == "What are the environmental impacts of plastic pollution in oceans?"
        assert result["clarified"] is True
        assert result["additional_context"] == "Focus on marine life and ecosystems"

    @pytest.mark.asyncio
    async def test_clarify_query_with_context_history(self, pre_response_agent):
        """Test clarifying a query with context from conversation history."""
        # Mock memory to return conversation history
        pre_response_agent.memory_mcp.search_memories.return_value = """
        Previous query: What are the causes of climate change?
        Previous response: The main causes of climate change include greenhouse gas emissions...
        """
        
        # Call the method with a follow-up query
        query = "What about its effects on agriculture?"
        
        result = await pre_response_agent.clarify_query(query)
        
        # Verify memory was searched for context
        pre_response_agent.memory_mcp.search_memories.assert_called_with(
            "climate change", namespace="pre_response"
        )
        
        # Verify result structure
        assert "query" in result
        assert "clarified" in result
        assert "additional_context" in result
        
        # The query should be clarified based on conversation history
        assert "climate change" in result["query"].lower()
        assert "agriculture" in result["query"].lower()
        assert result["clarified"] is True
        assert "previous query" in result["additional_context"].lower()

    @pytest.mark.asyncio
    async def test_clarify_query_error(self, pre_response_agent):
        """Test handling errors during query clarification."""
        # Mock _get_user_clarification to raise exception
        pre_response_agent._get_user_clarification = AsyncMock(side_effect=Exception("Clarification failed"))
        
        # Call the method and expect error
        with pytest.raises(QueryClarificationError, match="Failed to clarify user query"):
            await pre_response_agent.clarify_query("ambiguous query")
        
        # Verify memory was not updated with clarified query
        for call_args in pre_response_agent.memory_mcp.store_memory.call_args_list:
            args, kwargs = call_args
            assert args[0] != "clarified_query"

    @pytest.mark.asyncio
    async def test_get_user_clarification(self, pre_response_agent):
        """Test getting clarification from user."""
        # This is an internal method that would typically interact with the user
        # For testing, we'll patch any external interaction
        
        with patch('builtins.input', return_value="y"):
            # Mock any LLM interaction
            with patch.object(pre_response_agent, '_generate_clarification_options', return_value=[
                "What are the environmental impacts of plastic pollution in oceans?",
                "What are the economic impacts of plastic pollution on industries?"
            ]):
                # Call the method
                query = "Tell me about plastic pollution"
                result = await pre_response_agent._get_user_clarification(query)
                
                # Verify result structure
                assert "clarified_query" in result
                assert "additional_context" in result
                assert result["clarified_query"] in [
                    "What are the environmental impacts of plastic pollution in oceans?",
                    "What are the economic impacts of plastic pollution on industries?"
                ]

    @pytest.mark.asyncio
    async def test_present_plan_simple(self, pre_response_agent):
        """Test presenting a simple research plan."""
        # Mock the _generate_research_plan method
        pre_response_agent._generate_research_plan = AsyncMock(return_value=[
            "Search for recent articles on climate change impacts",
            "Focus on peer-reviewed scientific studies",
            "Analyze and summarize findings",
            "Generate visual representation of key impacts"
        ])
        
        # Mock the _get_user_approval method
        pre_response_agent._get_user_approval = AsyncMock(return_value={
            "approved": True,
            "feedback": ""
        })
        
        # Call the method
        query = "What are the impacts of climate change?"
        context = "Focus on scientific evidence"
        
        result = await pre_response_agent.present_plan(query, context)
        
        # Verify _generate_research_plan was called
        pre_response_agent._generate_research_plan.assert_called_once_with(query, context)
        
        # Verify _get_user_approval was called
        pre_response_agent._get_user_approval.assert_called_once()
        
        # Verify memory was updated
        pre_response_agent.memory_mcp.store_memory.assert_called_with(
            "research_plan", str(result["plan"]), namespace="pre_response"
        )
        pre_response_agent.memory_mcp.store_memory.assert_any_call(
            "plan_approved", "true", namespace="pre_response"
        )
        
        # Verify result structure
        assert "plan" in result
        assert "approved" in result
        assert len(result["plan"]) == 4
        assert result["approved"] is True
        assert "climate change impacts" in str(result["plan"])

    @pytest.mark.asyncio
    async def test_present_plan_not_approved(self, pre_response_agent):
        """Test presenting a plan that is not approved by the user."""
        # Mock the _generate_research_plan method
        pre_response_agent._generate_research_plan = AsyncMock(return_value=[
            "Search for articles on AI",
            "Summarize findings"
        ])
        
        # Mock the _get_user_approval method with rejection
        pre_response_agent._get_user_approval = AsyncMock(return_value={
            "approved": False,
            "feedback": "Need more focus on ethical considerations"
        })
        
        # Call the method
        query = "Tell me about AI"
        
        result = await pre_response_agent.present_plan(query, "")
        
        # Verify result structure
        assert "plan" in result
        assert "approved" in result
        assert "feedback" in result
        assert result["approved"] is False
        assert "ethical considerations" in result["feedback"]
        
        # Verify memory was updated with the plan but not approval
        pre_response_agent.memory_mcp.store_memory.assert_called_with(
            "research_plan", str(result["plan"]), namespace="pre_response"
        )
        
        # Verify plan_approved was not stored
        for call_args in pre_response_agent.memory_mcp.store_memory.call_args_list:
            args, kwargs = call_args
            assert args[0] != "plan_approved" or args[1] != "true"

    @pytest.mark.asyncio
    async def test_present_plan_revised(self, pre_response_agent):
        """Test presenting a plan, getting feedback, and presenting a revised plan."""
        # Mock the _generate_research_plan method for initial and revised plans
        pre_response_agent._generate_research_plan = AsyncMock(side_effect=[
            # Initial plan
            [
                "Search for articles on AI",
                "Summarize findings"
            ],
            # Revised plan
            [
                "Search for articles on AI ethics",
                "Focus on bias and fairness issues",
                "Analyze ethical frameworks",
                "Summarize findings with ethical considerations"
            ]
        ])
        
        # Mock the _get_user_approval method with rejection then approval
        pre_response_agent._get_user_approval = AsyncMock(side_effect=[
            # First call - reject
            {
                "approved": False,
                "feedback": "Need more focus on ethical considerations"
            },
            # Second call - approve
            {
                "approved": True,
                "feedback": ""
            }
        ])
        
        # Call the method
        query = "Tell me about AI"
        
        # First attempt - should be rejected
        result1 = await pre_response_agent.present_plan(query, "")
        assert result1["approved"] is False
        
        # Second attempt with feedback - should be approved
        result2 = await pre_response_agent.present_plan(query, result1["feedback"])
        
        # Verify the revised plan was generated with the feedback
        pre_response_agent._generate_research_plan.assert_called_with(query, result1["feedback"])
        
        # Verify result structure
        assert result2["approved"] is True
        assert len(result2["plan"]) > len(result1["plan"])
        assert "ethical considerations" in str(result2["plan"])
        
        # Verify memory was updated with the approved plan
        pre_response_agent.memory_mcp.store_memory.assert_any_call(
            "plan_approved", "true", namespace="pre_response"
        )

    @pytest.mark.asyncio
    async def test_present_plan_error(self, pre_response_agent):
        """Test handling errors during plan presentation."""
        # Mock _generate_research_plan to raise exception
        pre_response_agent._generate_research_plan = AsyncMock(side_effect=Exception("Plan generation failed"))
        
        # Call the method and expect error
        with pytest.raises(PlanPresentationError, match="Failed to present research plan"):
            await pre_response_agent.present_plan("test query", "")
        
        # Verify memory was not updated with plan or approval
        for call_args in pre_response_agent.memory_mcp.store_memory.call_args_list:
            args, kwargs = call_args
            assert args[0] != "research_plan"
            assert args[0] != "plan_approved"

    @pytest.mark.asyncio
    async def test_generate_research_plan(self, pre_response_agent):
        """Test generating a research plan."""
        # Call the method
        query = "What are the impacts of climate change on agriculture?"
        context = "Focus on developing countries"
        
        plan = await pre_response_agent._generate_research_plan(query, context)
        
        # Verify result structure
        assert isinstance(plan, list)
        assert len(plan) > 0
        
        # Verify plan content reflects query and context
        plan_str = str(plan)
        assert "climate change" in plan_str.lower()
        assert "agriculture" in plan_str.lower()
        assert "developing countries" in plan_str.lower()

    @pytest.mark.asyncio
    async def test_get_user_approval(self, pre_response_agent):
        """Test getting user approval for a plan."""
        # This is an internal method that would typically interact with the user
        # For testing, we'll patch any external interaction
        
        with patch('builtins.input', return_value="y"):
            # Call the method
            plan = [
                "Search for recent articles on climate change impacts",
                "Focus on peer-reviewed scientific studies",
                "Analyze and summarize findings"
            ]
            
            result = await pre_response_agent._get_user_approval(plan)
            
            # Verify result structure
            assert "approved" in result
            assert "feedback" in result
            assert result["approved"] is True
            assert result["feedback"] == ""

    @pytest.mark.asyncio
    async def test_get_user_approval_rejected(self, pre_response_agent):
        """Test getting user rejection and feedback for a plan."""
        # Mock user input to reject and provide feedback
        with patch('builtins.input', side_effect=["n", "Need more focus on economic impacts"]):
            # Call the method
            plan = ["Search for climate change articles", "Summarize findings"]
            
            result = await pre_response_agent._get_user_approval(plan)
            
            # Verify result structure
            assert "approved" in result
            assert "feedback" in result
            assert result["approved"] is False
            assert "economic impacts" in result["feedback"]

    @pytest.mark.asyncio
    async def test_get_conversation_history(self, pre_response_agent):
        """Test retrieving conversation history."""
        # Setup mock responses
        pre_response_agent.memory_mcp.list_memories.return_value = """
        original_query_1
        clarified_query_1
        query_context_1
        research_plan_1
        original_query_2
        clarified_query_2
        """
        
        pre_response_agent.memory_mcp.retrieve_memory.side_effect = [
            "What are the causes of climate change?",                                # original_query_1
            "What are the primary human and natural causes of climate change?",      # clarified_query_1
            "Focus on industrial emissions and deforestation",                       # query_context_1
            "['Search for scientific papers', 'Analyze findings']",                  # research_plan_1
            "How does climate change affect agriculture?",                           # original_query_2
            "What are the impacts of climate change on agricultural productivity?"   # clarified_query_2
        ]
        
        # Call the method
        history = await pre_response_agent.get_conversation_history()
        
        # Verify memory was queried
        pre_response_agent.memory_mcp.list_memories.assert_called_once_with(namespace="pre_response")
        
        # Verify result structure
        assert "queries" in history
        assert len(history["queries"]) == 2
        assert history["queries"][0]["original"] == "What are the causes of climate change?"
        assert history["queries"][0]["clarified"] == "What are the primary human and natural causes of climate change?"
        assert history["queries"][0]["context"] == "Focus on industrial emissions and deforestation"
        assert history["queries"][1]["original"] == "How does climate change affect agriculture?"
        assert "plans" in history
        assert len(history["plans"]) == 1
        assert "Search for scientific papers" in str(history["plans"][0])

    @pytest.mark.asyncio
    async def test_get_conversation_history_empty(self, pre_response_agent):
        """Test retrieving conversation history when empty."""
        # Setup mock with empty response
        pre_response_agent.memory_mcp.list_memories.return_value = ""
        
        # Call the method
        history = await pre_response_agent.get_conversation_history()
        
        # Verify result is empty
        assert history["queries"] == []
        assert history["plans"] == []

    @pytest.mark.asyncio
    async def test_analyze_query_complexity(self, pre_response_agent):
        """Test analyzing query complexity."""
        # Call the method with different queries
        simple_result = await pre_response_agent.analyze_query_complexity(
            "What is climate change?"
        )
        
        complex_result = await pre_response_agent.analyze_query_complexity(
            "How do socioeconomic factors influence the adoption of renewable energy technologies in developing countries, and what policy interventions might be most effective?"
        )
        
        ambiguous_result = await pre_response_agent.analyze_query_complexity(
            "Tell me about it"
        )
        
        # Verify result structure
        assert "complexity" in simple_result
        assert "ambiguity" in simple_result
        assert "needs_clarification" in simple_result
        
        # Verify simple query analysis
        assert simple_result["complexity"] == "low"
        assert simple_result["ambiguity"] == "low"
        assert simple_result["needs_clarification"] is False
        
        # Verify complex query analysis
        assert complex_result["complexity"] == "high"
        assert complex_result["ambiguity"] == "low"
        assert complex_result["needs_clarification"] is True
        
        # Verify ambiguous query analysis
        assert ambiguous_result["complexity"] == "low"
        assert ambiguous_result["ambiguity"] == "high"
        assert ambiguous_result["needs_clarification"] is True

    @pytest.mark.asyncio
    async def test_generate_clarification_options(self, pre_response_agent):
        """Test generating clarification options for ambiguous queries."""
        # Call the method
        query = "Tell me about AI"
        
        options = await pre_response_agent._generate_clarification_options(query)
        
        # Verify result structure
        assert isinstance(options, list)
        assert len(options) >= 2
        
        # Verify options are more specific than the original query
        for option in options:
            assert len(option) > len(query)
            assert "AI" in option

    @pytest.mark.asyncio
    async def test_generate_clarification_options_with_history(self, pre_response_agent):
        """Test generating clarification options with conversation history."""
        # Mock memory to return conversation history
        pre_response_agent.memory_mcp.search_memories.return_value = """
        Previous query: What are the ethical concerns in AI development?
        Previous response: The main ethical concerns include bias, privacy, and job displacement...
        """
        
        # Call the method
        query = "Tell me more about that"
        
        options = await pre_response_agent._generate_clarification_options(query)
        
        # Verify memory was searched for context
        pre_response_agent.memory_mcp.search_memories.assert_called_with(
            "ethical concerns AI", namespace="pre_response"
        )
        
        # Verify options incorporate history context
        for option in options:
            assert any(term in option.lower() for term in ["ethical", "bias", "privacy", "ai"])

    @pytest.mark.asyncio
    async def test_format_plan_for_presentation(self, pre_response_agent):
        """Test formatting a research plan for presentation."""
        # Call the method
        plan = [
            "Search for recent articles on climate change impacts",
            "Focus on peer-reviewed scientific studies",
            "Analyze and summarize findings",
            "Generate visual representation of key impacts"
        ]
        
        formatted_plan = await pre_response_agent._format_plan_for_presentation(plan)
        
        # Verify result structure
        assert isinstance(formatted_plan, str)
        
        # Verify formatting includes numbering and clear structure
        assert "1." in formatted_plan
        assert "2." in formatted_plan
        assert "3." in formatted_plan
        assert "4." in formatted_plan
        assert "climate change impacts" in formatted_plan

    @pytest.mark.asyncio
    async def test_store_interaction_history(self, pre_response_agent):
        """Test storing interaction history."""
        # Call the method
        await pre_response_agent.store_interaction_history(
            original_query="What is climate change?",
            clarified_query="What are the scientific definitions and causes of climate change?",
            context="Focus on IPCC reports",
            plan=["Search for IPCC reports", "Analyze definitions"]
        )
        
        # Verify memory was updated
        pre_response_agent.memory_mcp.store_memory.assert_any_call(
            "original_query_1", "What is climate change?", namespace="pre_response"
        )
        pre_response_agent.memory_mcp.store_memory.assert_any_call(
            "clarified_query_1", "What are the scientific definitions and causes of climate change?", namespace="pre_response"
        )
        pre_response_agent.memory_mcp.store_memory.assert_any_call(
            "query_context_1", "Focus on IPCC reports", namespace="pre_response"
        )
        pre_response_agent.memory_mcp.store_memory.assert_any_call(
            "research_plan_1", str(["Search for IPCC reports", "Analyze definitions"]), namespace="pre_response"
        )

    @pytest.mark.asyncio
    async def test_store_interaction_history_with_existing(self, pre_response_agent):
        """Test storing interaction history when previous interactions exist."""
        # Mock memory to return existing interaction count
        pre_response_agent.memory_mcp.list_memories.return_value = """
        original_query_1
        clarified_query_1
        original_query_2
        clarified_query_2
        """
        
        # Call the method
        await pre_response_agent.store_interaction_history(
            original_query="What is renewable energy?",
            clarified_query="What are the main types and benefits of renewable energy sources?",
            context="Focus on solar and wind",
            plan=["Research solar energy", "Research wind energy"]
        )
        
        # Verify memory was updated with correct index (3)
        pre_response_agent.memory_mcp.store_memory.assert_any_call(
            "original_query_3", "What is renewable energy?", namespace="pre_response"
        )
        pre_response_agent.memory_mcp.store_memory.assert_any_call(
            "clarified_query_3", "What are the main types and benefits of renewable energy sources?", namespace="pre_response"
        )

    @pytest.mark.asyncio
    async def test_get_query_suggestions(self, pre_response_agent):
        """Test getting query suggestions based on a topic."""
        # Call the method
        topic = "climate change"
        
        suggestions = await pre_response_agent.get_query_suggestions(topic)
        
        # Verify result structure
        assert isinstance(suggestions, list)
        assert len(suggestions) >= 3
        
        # Verify suggestions are related to the topic
        for suggestion in suggestions:
            assert "climate" in suggestion.lower() or "environment" in suggestion.lower() or "warming" in suggestion.lower()

    @pytest.mark.asyncio
    async def test_refine_query_with_feedback(self, pre_response_agent):
        """Test refining a query with user feedback."""
        # Call the method
        original_query = "Tell me about renewable energy"
        feedback = "I'm specifically interested in solar power efficiency improvements"
        
        refined_query = await pre_response_agent.refine_query_with_feedback(original_query, feedback)
        
        # Verify result is more specific
        assert len(refined_query) > len(original_query)
        assert "renewable energy" in refined_query.lower()
        assert "solar power" in refined_query.lower()
        assert "efficiency" in refined_query.lower()

    @pytest.mark.asyncio
    async def test_explain_research_approach(self, pre_response_agent):
        """Test explaining the research approach for a query."""
        # Call the method
        query = "What are the impacts of climate change on agriculture?"
        plan = [
            "Search for scientific papers on climate change and agriculture",
            "Focus on crop yields and growing seasons",
            "Analyze regional differences",
            "Summarize findings with data visualization"
        ]
        
        explanation = await pre_response_agent.explain_research_approach(query, plan)
        
        # Verify result structure
        assert isinstance(explanation, str)
        assert len(explanation) > 100  # Should be a substantial explanation
        
        # Verify explanation covers the plan elements
        assert "scientific papers" in explanation.lower()
        assert "crop yields" in explanation.lower()
        assert "regional differences" in explanation.lower()
        assert "visualization" in explanation.lower()

    @pytest.mark.asyncio
    async def test_generate_follow_up_questions(self, pre_response_agent):
        """Test generating follow-up questions based on research results."""
        # Call the method
        query = "What are the impacts of climate change on agriculture?"
        summary = "Climate change affects agriculture through changing precipitation patterns, temperature increases, and extreme weather events. Crop yields are projected to decrease in many regions, while some northern areas may see increased productivity."
        
        follow_ups = await pre_response_agent.generate_follow_up_questions(query, summary)
        
        # Verify result structure
        assert isinstance(follow_ups, list)
        assert len(follow_ups) >= 3
        
        # Verify questions are related to the topic and summary
        relevant_terms = ["climate", "agriculture", "crop", "weather", "temperature", "precipitation", "regions"]
        for question in follow_ups:
            assert any(term in question.lower() for term in relevant_terms)
            assert question.endswith("?")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
