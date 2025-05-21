"""
Unit tests for the Verification Agent.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from apps.agents.verification_agent import VerificationAgent
from apps.mcps.brave_search_mcp import BraveSearchMCP
from apps.mcps.memory_mcp import MemoryMCP
from apps.utils.exceptions import AgentError, ValidationError

class TestVerificationAgent:
    """Test suite for VerificationAgent class."""

    @pytest.fixture
    def brave_search_mcp(self):
        """Fixture to create a mock BraveSearchMCP."""
        mock_brave = MagicMock(spec=BraveSearchMCP)
        mock_brave.web_search = AsyncMock()
        return mock_brave

    @pytest.fixture
    def memory_mcp(self):
        """Fixture to create a mock MemoryMCP."""
        mock_memory = MagicMock(spec=MemoryMCP)
        mock_memory.store_memory = AsyncMock()
        mock_memory.retrieve_memory = AsyncMock()
        mock_memory.search_memories = AsyncMock()
        return mock_memory

    @pytest.fixture
    def verification_agent(self, brave_search_mcp, memory_mcp):
        """Fixture to create a VerificationAgent instance with mock dependencies."""
        agent = VerificationAgent(
            name="verification",
            brave_search_mcp=brave_search_mcp,
            memory_mcp=memory_mcp
        )
        return agent

    def test_init(self, brave_search_mcp, memory_mcp):
        """Test initialization of VerificationAgent."""
        agent = VerificationAgent(
            name="verification",
            brave_search_mcp=brave_search_mcp,
            memory_mcp=memory_mcp
        )
        
        assert agent.name == "verification"
        assert agent.brave_search_mcp == brave_search_mcp
        assert agent.memory_mcp == memory_mcp

    @pytest.mark.asyncio
    async def test_verify_fact_true(self, verification_agent):
        """Test verifying a fact that is true."""
        # Setup mock response
        verification_agent.brave_search_mcp.web_search.return_value = """
        Title: NASA - Climate Change Facts
        Description: Scientific evidence for warming of the climate system is unequivocal.
        URL: https://climate.nasa.gov/evidence/

        Title: NOAA Climate.gov
        Description: Earth's climate is changing, with global temperature rising at an unprecedented rate.
        URL: https://www.climate.gov/news-features/understanding-climate/climate-change-global-temperature

        Title: UN Climate Action
        Description: Global temperatures have already risen 1.1°C above pre-industrial levels.
        URL: https://www.un.org/en/climatechange/science/key-findings
        """
        
        # Call the method
        fact = "Global temperatures have risen about 1.1°C since pre-industrial times."
        result = await verification_agent.verify_fact(fact)
        
        # Verify search was called with appropriate query
        verification_agent.brave_search_mcp.web_search.assert_called_once()
        search_query = verification_agent.brave_search_mcp.web_search.call_args[0][0]
        assert "global temperature" in search_query.lower() or "1.1" in search_query
        
        # Verify memory was updated
        verification_agent.memory_mcp.store_memory.assert_called_with(
            f"verification_{fact[:50]}", str(result), namespace="verification"
        )
        
        # Verify result structure
        assert "verified" in result
        assert "confidence" in result
        assert "sources" in result
        assert "explanation" in result
        
        # Verify the fact was confirmed
        assert result["verified"] is True
        assert result["confidence"] >= 0.8  # High confidence
        assert len(result["sources"]) >= 2
        assert any("nasa" in source.lower() for source in result["sources"])
        assert len(result["explanation"]) >= 50

    @pytest.mark.asyncio
    async def test_verify_fact_false(self, verification_agent):
        """Test verifying a fact that is false."""
        # Setup mock response
        verification_agent.brave_search_mcp.web_search.return_value = """
        Title: Climate Myths Debunked
        Description: The claim that global temperatures have risen by 5°C is false. Actual warming is about 1.1°C.
        URL: https://climate-science.org/myths-debunked

        Title: Fact Check: Climate Change
        Description: We rate the claim that global temperatures have risen 5°C as FALSE. Scientific consensus puts warming at 1.1°C.
        URL: https://factcheck.org/climate-claims

        Title: IPCC Sixth Assessment Report
        Description: Global surface temperature has increased by 1.09°C compared to pre-industrial levels.
        URL: https://www.ipcc.ch/report/ar6/wg1/
        """
        
        # Call the method
        fact = "Global temperatures have risen by 5°C since pre-industrial times."
        result = await verification_agent.verify_fact(fact)
        
        # Verify search was called
        verification_agent.brave_search_mcp.web_search.assert_called_once()
        
        # Verify result
        assert result["verified"] is False
        assert result["confidence"] >= 0.7  # Reasonably high confidence in the falsification
        assert len(result["sources"]) >= 2
        assert any("ipcc" in source.lower() for source in result["sources"])
        assert "1.1" in result["explanation"] or "1.09" in result["explanation"]

    @pytest.mark.asyncio
    async def test_verify_fact_uncertain(self, verification_agent):
        """Test verifying a fact with uncertain results."""
        # Setup mock response with mixed or ambiguous information
        verification_agent.brave_search_mcp.web_search.return_value = """
        Title: Climate Perspectives
        Description: Some researchers suggest Antarctic ice loss has accelerated, while others point to regional variations.
        URL: https://climate-research.org/perspectives

        Title: Antarctic Ice Studies
        Description: Recent studies show complex patterns of ice gain and loss across different regions of Antarctica.
        URL: https://polar-science.org/antarctic-studies

        Title: Climate Monitoring
        Description: Satellite data shows varying rates of ice loss in Antarctica, with some regions gaining ice and others losing it.
        URL: https://earth-monitor.org/polar-regions
        """
        
        # Call the method
        fact = "Antarctica is losing ice at an accelerating rate across the entire continent."
        result = await verification_agent.verify_fact(fact)
        
        # Verify result indicates uncertainty
        assert result["verified"] is None  # Uncertain
        assert 0.3 <= result["confidence"] <= 0.7  # Medium confidence
        assert "complex" in result["explanation"].lower() or "varying" in result["explanation"].lower()
        assert "regional" in result["explanation"].lower() or "different regions" in result["explanation"].lower()

    @pytest.mark.asyncio
    async def test_verify_fact_no_information(self, verification_agent):
        """Test verifying a fact when no relevant information is found."""
        # Setup mock response with irrelevant information
        verification_agent.brave_search_mcp.web_search.return_value = """
        Title: Climate Change News
        Description: Latest updates on climate policy and international agreements.
        URL: https://climate-news.org/policy

        Title: Environmental Protection
        Description: Strategies for conservation and sustainable development.
        URL: https://environmental-protection.org/strategies

        Title: Green Energy Transition
        Description: Progress in renewable energy adoption worldwide.
        URL: https://green-energy.org/transition
        """
        
        # Call the method
        fact = "The moon's gravitational pull affects climate change patterns on Earth."
        result = await verification_agent.verify_fact(fact)
        
        # Verify result indicates insufficient information
        assert result["verified"] is None
        assert result["confidence"] <= 0.3  # Low confidence
        assert "insufficient" in result["explanation"].lower() or "not enough" in result["explanation"].lower()
        assert len(result["sources"]) <= 1

    @pytest.mark.asyncio
    async def test_verify_multiple_facts(self, verification_agent):
        """Test verifying multiple facts at once."""
        # Setup mock responses for different searches
        verification_agent.brave_search_mcp.web_search.side_effect = [
            # Response for first fact
            """
            Title: NASA - Climate Change Facts
            Description: Scientific evidence for warming of the climate system is unequivocal.
            URL: https://climate.nasa.gov/evidence/
            """,
            # Response for second fact
            """
            Title: Fact Check: Renewable Energy
            Description: Solar energy is indeed the fastest growing renewable energy source globally.
            URL: https://energy-facts.org/renewables
            """
        ]
        
        # Call the method
        facts = [
            "Climate change is primarily caused by human activities.",
            "Solar energy is the fastest growing renewable energy source."
        ]
        results = await verification_agent.verify_multiple_facts(facts)
        
        # Verify searches were called for each fact
        assert verification_agent.brave_search_mcp.web_search.call_count == 2
        
        # Verify results structure
        assert len(results) == 2
        assert all("verified" in result for result in results)
        assert all("confidence" in result for result in results)
        assert all("sources" in result for result in results)
        
        # Verify memory was updated
        assert verification_agent.memory_mcp.store_memory.call_count >= 2

    @pytest.mark.asyncio
    async def test_verify_multiple_facts_empty(self, verification_agent):
        """Test verifying an empty list of facts."""
        # Call the method with empty list
        results = await verification_agent.verify_multiple_facts([])
        
        # Verify no searches were performed
        verification_agent.brave_search_mcp.web_search.assert_not_called()
        
        # Verify empty results
        assert results == []

    @pytest.mark.asyncio
    async def test_cross_check_information(self, verification_agent):
        """Test cross-checking information across multiple sources."""
        # Setup mock response
        verification_agent.brave_search_mcp.web_search.return_value = """
        Title: NASA Climate
        Description: Sea levels have risen about 8-9 inches (21-24 cm) since 1880.
        URL: https://climate.nasa.gov/vital-signs/sea-level/

        Title: NOAA Sea Level Rise
        Description: Global sea level has risen by 8 to 9 inches since 1880.
        URL: https://oceanservice.noaa.gov/facts/sealevel.html

        Title: Climate.gov
        Description: Global mean sea level has risen about 8–9 inches (21–24 centimeters) since 1880.
        URL: https://www.climate.gov/news-features/understanding-climate/climate-change-global-sea-level

        Title: EPA Climate Indicators
        Description: Sea level has risen 8 to 9 inches since 1880, with about a third of that coming in just the last two and a half decades.
        URL: https://www.epa.gov/climate-indicators/climate-change-indicators-sea-level
        """
        
        # Call the method
        information = "Sea levels have risen about 8-9 inches since 1880."
        result = await verification_agent.cross_check_information(information)
        
        # Verify search was called
        verification_agent.brave_search_mcp.web_search.assert_called_once()
        
        # Verify memory was updated
        verification_agent.memory_mcp.store_memory.assert_called_with(
            f"cross_check_{information[:50]}", str(result), namespace="verification"
        )
        
        # Verify result structure
        assert "consensus" in result
        assert "agreement_level" in result
        assert "sources" in result
        assert "variations" in result
        
        # Verify the cross-check results
        assert result["consensus"] is True
        assert result["agreement_level"] >= 0.8  # High agreement
        assert len(result["sources"]) >= 3
        assert any("nasa" in source.lower() for source in result["sources"])
        assert any("noaa" in source.lower() for source in result["sources"])
        assert len(result["variations"]) <= 1  # Little to no variation in the information

    @pytest.mark.asyncio
    async def test_cross_check_information_disagreement(self, verification_agent):
        """Test cross-checking information with significant disagreement."""
        # Setup mock response with disagreeing information
        verification_agent.brave_search_mcp.web_search.return_value = """
        Title: Climate Research Institute
        Description: Some studies suggest Arctic sea ice could disappear in summer by 2035.
        URL: https://climate-research.org/arctic-predictions

        Title: Polar Science Center
        Description: Models project that Arctic summers could be ice-free by 2050-2060.
        URL: https://polar-science.org/projections

        Title: Arctic Monitoring
        Description: Conservative estimates suggest Arctic summer sea ice will remain until at least 2070.
        URL: https://arctic-monitor.org/sea-ice-trends

        Title: Climate Modeling Group
        Description: The exact timing of ice-free Arctic summers remains uncertain, with estimates ranging from 2035 to beyond 2100.
        URL: https://climate-models.org/arctic-future
        """
        
        # Call the method
        information = "The Arctic will be ice-free in summer by 2035."
        result = await verification_agent.cross_check_information(information)
        
        # Verify result indicates disagreement
        assert result["consensus"] is False
        assert result["agreement_level"] <= 0.5  # Low to medium agreement
        assert len(result["sources"]) >= 3
        assert len(result["variations"]) >= 2
        assert "2035" in str(result["variations"]) and "2050" in str(result["variations"]) and "2070" in str(result["variations"])

    @pytest.mark.asyncio
    async def test_evaluate_source_credibility(self, verification_agent):
        """Test evaluating the credibility of a source."""
        # Call the method
        source = {
            "name": "NASA Climate Change",
            "url": "https://climate.nasa.gov/",
            "type": "government agency",
            "description": "Official NASA website providing climate data and research."
        }
        
        result = await verification_agent.evaluate_source_credibility(source)
        
        # Verify memory was updated
        verification_agent.memory_mcp.store_memory.assert_called_with(
            f"source_credibility_{source['name']}", str(result), namespace="verification"
        )
        
        # Verify result structure
        assert "credibility_score" in result
        assert "factors" in result
        assert "strengths" in result
        assert "limitations" in result
        assert "recommendation" in result
        
        # Verify the credibility evaluation
        assert result["credibility_score"] >= 8  # High credibility for NASA
        assert len(result["factors"]) >= 3
        assert any("government" in factor.lower() or "official" in factor.lower() for factor in result["factors"])
        assert len(result["strengths"]) >= 2
        assert "scientific" in str(result["strengths"]).lower() or "research" in str(result["strengths"]).lower()
        assert result["recommendation"] in ["highly reliable", "reliable", "generally reliable"]

    @pytest.mark.asyncio
    async def test_evaluate_source_credibility_low(self, verification_agent):
        """Test evaluating the credibility of a low-credibility source."""
        # Call the method with a questionable source
        source = {
            "name": "Climate Truth Revealed",
            "url": "https://climate-truth-revealed.com/",
            "type": "blog",
            "description": "Alternative perspectives on climate science from independent researchers."
        }
        
        result = await verification_agent.evaluate_source_credibility(source)
        
        # Verify the credibility evaluation
        assert result["credibility_score"] <= 5  # Lower credibility
        assert len(result["limitations"]) >= 2
        assert "bias" in str(result["limitations"]).lower() or "alternative" in str(result["limitations"]).lower()
        assert result["recommendation"] in ["use with caution", "verify with other sources", "not recommended"]

    @pytest.mark.asyncio
    async def test_evaluate_multiple_sources(self, verification_agent):
        """Test evaluating multiple sources at once."""
        # Call the method
        sources = [
            {
                "name": "IPCC",
                "url": "https://www.ipcc.ch/",
                "type": "international organization",
                "description": "Intergovernmental Panel on Climate Change, the UN body for assessing climate science."
            },
            {
                "name": "Climate Research Blog",
                "url": "https://climate-blog.com/",
                "type": "blog",
                "description": "Personal blog discussing climate research and policy."
            }
        ]
        
        results = await verification_agent.evaluate_multiple_sources(sources)
        
        # Verify results structure
        assert len(results) == 2
        assert all("credibility_score" in result for result in results)
        assert all("recommendation" in result for result in results)
        
        # Verify IPCC has higher credibility than the blog
        ipcc_result = results[0]
        blog_result = results[1]
        assert ipcc_result["credibility_score"] > blog_result["credibility_score"]
        assert "reliable" in ipcc_result["recommendation"].lower()

    @pytest.mark.asyncio
    async def test_fact_check_claim(self, verification_agent):
        """Test fact-checking a specific claim."""
        # Setup mock response
        verification_agent.brave_search_mcp.web_search.return_value = """
        Title: Fact Check: Hurricane Frequency
        Description: Studies show no clear trend in global hurricane frequency, though intensity may be increasing.
        URL: https://factcheck.org/hurricane-trends

        Title: NOAA Hurricane Research
        Description: Long-term data shows fluctuations in hurricane frequency but no significant global trend.
        URL: https://www.nhc.noaa.gov/research/

        Title: Climate Science: Hurricanes
        Description: While warming oceans may intensify hurricanes, data doesn't support claims of increasing frequency.
        URL: https://climate-science.org/hurricanes
        """
        
        # Call the method
        claim = "Hurricanes are becoming more frequent globally due to climate change."
        result = await verification_agent.fact_check_claim(claim)
        
        # Verify search was called
        verification_agent.brave_search_mcp.web_search.assert_called_once()
        
        # Verify memory was updated
        verification_agent.memory_mcp.store_memory.assert_called_with(
            f"fact_check_{claim[:50]}", str(result), namespace="verification"
        )
        
        # Verify result structure
        assert "rating" in result
        assert "explanation" in result
        assert "evidence" in result
        assert "sources" in result
        
        # Verify the fact check results
        assert result["rating"] in ["false", "mostly false", "mixed", "partly true"]
        assert "no clear trend" in result["explanation"].lower() or "no significant global trend" in result["explanation"].lower()
        assert len(result["evidence"]) >= 2
        assert "intensity" in str(result["evidence"]).lower() and "frequency" in str(result["evidence"]).lower()
        assert len(result["sources"]) >= 2
        assert any("noaa" in source.lower() for source in result["sources"])

    @pytest.mark.asyncio
    async def test_fact_check_claim_true(self, verification_agent):
        """Test fact-checking a claim that is true."""
        # Setup mock response
        verification_agent.brave_search_mcp.web_search.return_value = """
        Title: NASA: Carbon Dioxide
        Description: Carbon dioxide levels have increased by 48% since pre-industrial times, from 280 ppm to 415 ppm.
        URL: https://climate.nasa.gov/vital-signs/carbon-dioxide/

        Title: NOAA Climate.gov
        Description: Atmospheric CO2 has risen from pre-industrial levels of 280 ppm to over 410 ppm today.
        URL: https://www.climate.gov/news-features/understanding-climate/climate-change-atmospheric-carbon-dioxide

        Title: Global Carbon Project
        Description: CO2 concentration in the atmosphere has increased from approximately 280 ppm in 1750 to 415 ppm in 2021.
        URL: https://www.globalcarbonproject.org/carbonbudget/
        """
        
        # Call the method
        claim = "Carbon dioxide levels in the atmosphere have increased by about 48% since pre-industrial times."
        result = await verification_agent.fact_check_claim(claim)
        
        # Verify the fact check results
        assert result["rating"] in ["true", "mostly true"]
        assert "48%" in result["explanation"] or "280 ppm to 415 ppm" in result["explanation"]
        assert len(result["sources"]) >= 2

    @pytest.mark.asyncio
    async def test_verify_consistency(self, verification_agent):
        """Test verifying consistency between multiple statements."""
        # Call the method
        statements = [
            "Global temperatures have risen by about 1.1°C since pre-industrial times.",
            "Earth has warmed by approximately 1.1 degrees Celsius compared to pre-industrial levels.",
            "The planet has experienced about 1.1°C of warming since the late 19th century."
        ]
        
        result = await verification_agent.verify_consistency(statements)
        
        # Verify memory was updated
        verification_agent.memory_mcp.store_memory.assert_called_with(
            "consistency_check", str(result), namespace="verification"
        )
        
        # Verify result structure
        assert "consistent" in result
        assert "confidence" in result
        assert "explanation" in result
        assert "discrepancies" in result
        
        # Verify the consistency check
        assert result["consistent"] is True
        assert result["confidence"] >= 0.8  # High confidence
        assert "1.1" in result["explanation"]
        assert len(result["discrepancies"]) == 0

    @pytest.mark.asyncio
    async def test_verify_consistency_inconsistent(self, verification_agent):
        """Test verifying consistency between inconsistent statements."""
        # Call the method
        statements = [
            "Global temperatures have risen by about 1.1°C since pre-industrial times.",
            "Earth has warmed by approximately 2.5 degrees Celsius compared to pre-industrial levels.",
            "The planet has experienced about 0.5°C of warming since the late 19th century."
        ]
        
        result = await verification_agent.verify_consistency(statements)
        
        # Verify the consistency check
        assert result["consistent"] is False
        assert result["confidence"] >= 0.7  # Reasonably high confidence
        assert len(result["discrepancies"]) >= 2
        assert "1.1" in str(result["discrepancies"]) and "2.5" in str(result["discrepancies"]) and "0.5" in str(result["discrepancies"])

    @pytest.mark.asyncio
    async def test_find_corroborating_sources(self, verification_agent):
        """Test finding corroborating sources for a statement."""
        # Setup mock response
        verification_agent.brave_search_mcp.web_search.return_value = """
        Title: IPCC Sixth Assessment Report
        Description: Human influence has warmed the climate at a rate that is unprecedented in at least the last 2000 years.
        URL: https://www.ipcc.ch/report/ar6/wg1/

        Title: NASA Climate
        Description: The current warming trend is of particular significance because it is unequivocally the result of human activity.
        URL: https://climate.nasa.gov/evidence/

        Title: National Academy of Sciences
        Description: Scientific evidence is clear: global climate change is real and human activities are the main driver.
        URL: https://www.nationalacademies.org/based-on-science/climate-change-is-caused-by-human-activities
        """
        
        # Call the method
        statement = "Human activities are the primary driver of current climate change."
        result = await verification_agent.find_corroborating_sources(statement)
        
        # Verify search was called
        verification_agent.brave_search_mcp.web_search.assert_called_once()
        
        # Verify memory was updated
        verification_agent.memory_mcp.store_memory.assert_called_with(
            f"corroborating_sources_{statement[:50]}", str(result), namespace="verification"
        )
        
        # Verify result structure
        assert "sources" in result
        assert "strength" in result
        assert "consensus" in result
        
        # Verify the corroborating sources
        assert len(result["sources"]) >= 3
        assert any("ipcc" in source.lower() for source in result["sources"])
        assert any("nasa" in source.lower() for source in result["sources"])
        assert result["strength"] in ["strong", "very strong"]
        assert result["consensus"] is True

    @pytest.mark.asyncio
    async def test_find_corroborating_sources_weak(self, verification_agent):
        """Test finding corroborating sources for a statement with weak support."""
        # Setup mock response with limited or mixed support
        verification_agent.brave_search_mcp.web_search.return_value = """
        Title: Climate Research
        Description: Some studies suggest a potential link between solar activity and regional climate patterns.
        URL: https://climate-research.org/solar-influence

        Title: Solar Physics
        Description: The relationship between solar cycles and Earth's climate remains an area of ongoing research.
        URL: https://solar-physics.org/climate-connection

        Title: Climate Science Review
        Description: While solar activity does affect Earth's climate, its contribution to recent warming is minimal compared to greenhouse gases.
        URL: https://climate-science.org/solar-vs-anthropogenic
        """
        
        # Call the method
        statement = "Solar activity is the main driver of current climate change."
        result = await verification_agent.find_corroborating_sources(statement)
        
        # Verify the corroborating sources
        assert len(result["sources"]) <= 2
        assert result["strength"] in ["weak", "limited", "mixed"]
        assert result["consensus"] is False

    @pytest.mark.asyncio
    async def test_verify_numerical_claim(self, verification_agent):
        """Test verifying a numerical claim."""
        # Setup mock response
        verification_agent.brave_search_mcp.web_search.return_value = """
        Title: Global Carbon Project
        Description: Global CO2 emissions were approximately 36.3 billion tonnes in 2021.
        URL: https://www.globalcarbonproject.org/carbonbudget/

        Title: International Energy Agency
        Description: Global energy-related CO2 emissions reached 36.3 gigatonnes in 2021.
        URL: https://www.iea.org/reports/global-energy-review-co2-emissions-in-2021

        Title: Climate Analytics
        Description: In 2021, global carbon dioxide emissions rebounded to 36.3 Gt CO2.
        URL: https://climateanalytics.org/latest/global-emissions-2021/
        """
        
        # Call the method
        claim = "Global CO2 emissions were approximately 36 billion tonnes in 2021."
        result = await verification_agent.verify_numerical_claim(claim)
        
        # Verify search was called
        verification_agent.brave_search_mcp.web_search.assert_called_once()
        
        # Verify memory was updated
        verification_agent.memory_mcp.store_memory.assert_called_with(
            f"numerical_verification_{claim[:50]}", str(result), namespace="verification"
        )
        
        # Verify result structure
        assert "verified" in result
        assert "actual_value" in result
        assert "margin_of_error" in result
        assert "sources" in result
        assert "explanation" in result
        
        # Verify the numerical verification
        assert result["verified"] is True
        assert result["actual_value"] == "36.3 billion tonnes"
        assert result["margin_of_error"] <= 1.0  # Small margin of error
        assert len(result["sources"]) >= 2
        assert any("global carbon project" in source.lower() for source in result["sources"])

    @pytest.mark.asyncio
    async def test_verify_numerical_claim_inaccurate(self, verification_agent):
        """Test verifying a numerical claim that is significantly inaccurate."""
        # Setup mock response
        verification_agent.brave_search_mcp.web_search.return_value = """
        Title: Global Carbon Project
        Description: Global CO2 emissions were approximately 36.3 billion tonnes in 2021.
        URL: https://www.globalcarbonproject.org/carbonbudget/

        Title: International Energy Agency
        Description: Global energy-related CO2 emissions reached 36.3 gigatonnes in 2021.
        URL: https://www.iea.org/reports/global-energy-review-co2-emissions-in-2021
        """
        
        # Call the method
        claim = "Global CO2 emissions were approximately 50 billion tonnes in 2021."
        result = await verification_agent.verify_numerical_claim(claim)
        
        # Verify the numerical verification
        assert result["verified"] is False
        assert result["actual_value"] == "36.3 billion tonnes"
        assert result["margin_of_error"] >= 10.0  # Large margin of error
        assert "significantly higher" in result["explanation"].lower() or "overestimated" in result["explanation"].lower()

    @pytest.mark.asyncio
    async def test_verify_date_claim(self, verification_agent):
        """Test verifying a claim about a date or time period."""
        # Setup mock response
        verification_agent.brave_search_mcp.web_search.return_value = """
        Title: Paris Agreement - UNFCCC
        Description: The Paris Agreement was adopted on 12 December 2015 at COP21 in Paris, France.
        URL: https://unfccc.int/process-and-meetings/the-paris-agreement/the-paris-agreement

        Title: United Nations Climate Change
        Description: The Paris Agreement entered into force on 4 November 2016.
        URL: https://unfccc.int/process/the-paris-agreement/status-of-ratification

        Title: Paris Climate Agreement History
        Description: The Paris Agreement was adopted in December 2015 and entered into force in November 2016.
        URL: https://climate-history.org/paris-agreement
        """
        
        # Call the method
        claim = "The Paris Agreement was adopted in December 2015."
        result = await verification_agent.verify_date_claim(claim)
        
        # Verify search was called
        verification_agent.brave_search_mcp.web_search.assert_called_once()
        
        # Verify memory was updated
        verification_agent.memory_mcp.store_memory.assert_called_with(
            f"date_verification_{claim[:50]}", str(result), namespace="verification"
        )
        
        # Verify result structure
        assert "verified" in result
        assert "correct_date" in result
        assert "precision" in result
        assert "sources" in result
        assert "additional_context" in result
        
        # Verify the date verification
        assert result["verified"] is True
        assert "december 2015" in result["correct_date"].lower() or "12 december 2015" in result["correct_date"].lower()
        assert result["precision"] in ["exact", "high"]
        assert len(result["sources"]) >= 2
        assert "entered into force" in result["additional_context"].lower() and "november 2016" in result["additional_context"].lower()

    @pytest.mark.asyncio
    async def test_verify_date_claim_incorrect(self, verification_agent):
        """Test verifying a date claim that is incorrect."""
        # Setup mock response
        verification_agent.brave_search_mcp.web_search.return_value = """
        Title: Paris Agreement - UNFCCC
        Description: The Paris Agreement was adopted on 12 December 2015 at COP21 in Paris, France.
        URL: https://unfccc.int/process-and-meetings/the-paris-agreement/the-paris-agreement

        Title: United Nations Climate Change
        Description: The Paris Agreement entered into force on 4 November 2016.
        URL: https://unfccc.int/process/the-paris-agreement/status-of-ratification
        """
        
        # Call the method
        claim = "The Paris Agreement was adopted in December 2016."
        result = await verification_agent.verify_date_claim(claim)
        
        # Verify the date verification
        assert result["verified"] is False
        assert "december 2015" in result["correct_date"].lower() or "12 december 2015" in result["correct_date"].lower()
        assert "incorrect year" in result["additional_context"].lower() or "2015, not 2016" in result["additional_context"].lower()

    @pytest.mark.asyncio
    async def test_verify_attribution_claim(self, verification_agent):
        """Test verifying a claim attributing a statement to a person or organization."""
        # Setup mock response
        verification_agent.brave_search_mcp.web_search.return_value = """
        Title: NASA Climate - Scientific Consensus
        Description: "97 percent or more of actively publishing climate scientists agree: Climate-warming trends over the past century are extremely likely due to human activities."
        URL: https://climate.nasa.gov/scientific-consensus/

        Title: NASA on Climate Change
        Description: NASA states that "97 percent of climate scientists agree that climate-warming trends over the past century are very likely due to human activities."
        URL: https://climate.nasa.gov/evidence/

        Title: Scientific Consensus on Climate Change
        Description: According to NASA, 97% of climate scientists agree that human-caused climate change is occurring.
        URL: https://science.org/consensus
        """
        
        # Call the method
        claim = "NASA states that 97% of climate scientists agree that human-caused climate change is occurring."
        result = await verification_agent.verify_attribution_claim(claim)
        
        # Verify search was called
        verification_agent.brave_search_mcp.web_search.assert_called_once()
        
        # Verify memory was updated
        verification_agent.memory_mcp.store_memory.assert_called_with(
            f"attribution_verification_{claim[:50]}", str(result), namespace="verification"
        )
        
        # Verify result structure
        assert "verified" in result
        assert "accuracy" in result
        assert "exact_quote" in result
        assert "sources" in result
        assert "context" in result
        
        # Verify the attribution verification
        assert result["verified"] is True
        assert result["accuracy"] in ["high", "exact", "very high"]
        assert "97 percent" in result["exact_quote"] or "97%" in result["exact_quote"]
        assert len(result["sources"]) >= 2
        assert all("nasa" in source.lower() for source in result["sources"])

    @pytest.mark.asyncio
    async def test_verify_attribution_claim_misattributed(self, verification_agent):
        """Test verifying a claim that misattributes a statement."""
        # Setup mock response
        verification_agent.brave_search_mcp.web_search.return_value = """
        Title: IPCC Sixth Assessment Report
        Description: The IPCC states that "it is unequivocal that human influence has warmed the atmosphere, ocean and land."
        URL: https://www.ipcc.ch/report/ar6/wg1/

        Title: UN Climate Reports
        Description: The statement "it is unequivocal that human influence has warmed the atmosphere, ocean and land" comes from the IPCC, not NASA.
        URL: https://un.org/climate-reports

        Title: Climate Science Attribution
        Description: The IPCC's Sixth Assessment Report states that human influence on the climate system is now "unequivocal."
        URL: https://climate-science.org/attribution
        """
        
        # Call the method
        claim = "NASA states that it is unequivocal that human influence has warmed the atmosphere, ocean and land."
        result = await verification_agent.verify_attribution_claim(claim)
        
        # Verify the attribution verification
        assert result["verified"] is False
        assert "misattributed" in result["context"].lower() or "incorrect attribution" in result["context"].lower()
        assert "ipcc" in result["exact_quote"].lower() or "ipcc" in result["context"].lower()

    @pytest.mark.asyncio
    async def test_verify_research_findings(self, verification_agent):
        """Test verifying a collection of research findings."""
        # Setup research findings
        findings = {
            "finding1": {
                "claim": "Global temperatures have risen by about 1.1°C since pre-industrial times.",
                "source": "IPCC Sixth Assessment Report"
            },
            "finding2": {
                "claim": "Sea levels have risen by 8-9 inches (21-24 cm) since 1880.",
                "source": "NASA Climate"
            },
            "finding3": {
                "claim": "The Greenland ice sheet is losing mass at an accelerating rate.",
                "source": "Nature Climate Change study"
            }
        }
        
        # Setup mock responses for different searches
        verification_agent.brave_search_mcp.web_search.side_effect = [
            # Response for first finding
            """
            Title: IPCC Sixth Assessment Report
            Description: Global surface temperature has increased by 1.09°C compared to pre-industrial levels.
            URL: https://www.ipcc.ch/report/ar6/wg1/
            """,
            # Response for second finding
            """
            Title: NASA Sea Level
            Description: Global sea level has risen by 8 to 9 inches since 1880.
            URL: https://climate.nasa.gov/vital-signs/sea-level/
            """,
            # Response for third finding
            """
            Title: Nature Climate Change
            Description: Recent studies show Greenland's ice sheet is losing mass at an accelerating rate.
            URL: https://nature.com/articles/climate-change-greenland
            """
        ]
        
        # Call the method
        result = await verification_agent.verify_research_findings(findings)
        
        # Verify searches were called for each finding
        assert verification_agent.brave_search_mcp.web_search.call_count == 3
        
        # Verify memory was updated
        verification_agent.memory_mcp.store_memory.assert_called_with(
            "research_findings_verification", str(result), namespace="verification"
        )
        
        # Verify result structure
        assert "overall_accuracy" in result
        assert "verified_findings" in result
        assert "questionable_findings" in result
        assert "recommendations" in result
        
        # Verify the research findings verification
        assert result["overall_accuracy"] >= 0.8  # High overall accuracy
        assert len(result["verified_findings"]) >= 2
        assert len(result["questionable_findings"]) <= 1
        assert len(result["recommendations"]) >= 1

    @pytest.mark.asyncio
    async def test_verify_research_findings_mixed(self, verification_agent):
        """Test verifying research findings with mixed accuracy."""
        # Setup research findings with some inaccuracies
        findings = {
            "finding1": {
                "claim": "Global temperatures have risen by about 1.1°C since pre-industrial times.",
                "source": "IPCC Sixth Assessment Report"
            },
            "finding2": {
                "claim": "Sea levels have risen by 20 inches since 1880.",  # Inaccurate
                "source": "Climate Research Blog"
            },
            "finding3": {
                "claim": "The Amazon rainforest produces 50% of the world's oxygen.",  # Commonly cited but incorrect
                "source": "Environmental Newsletter"
            }
        }
        
        # Setup mock responses
        verification_agent.brave_search_mcp.web_search.side_effect = [
            # Response for first finding
            """
            Title: IPCC Sixth Assessment Report
            Description: Global surface temperature has increased by 1.09°C compared to pre-industrial levels.
            URL: https://www.ipcc.ch/report/ar6/wg1/
            """,
            # Response for second finding
            """
            Title: NASA Sea Level
            Description: Global sea level has risen by 8 to 9 inches since 1880, not 20 inches.
            URL: https://climate.nasa.gov/vital-signs/sea-level/
            """,
            # Response for third finding
            """
            Title: Scientific American
            Description: The Amazon rainforest produces approximately 6-9% of the world's oxygen, not 50% as commonly claimed.
            URL: https://scientificamerican.com/article/amazon-oxygen-myth/
            """
        ]
        
        # Call the method
        result = await verification_agent.verify_research_findings(findings)
        
        # Verify the research findings verification
        assert 0.3 <= result["overall_accuracy"] <= 0.7  # Medium accuracy
        assert len(result["verified_findings"]) == 1
        assert len(result["questionable_findings"]) == 2
        assert "sea level" in str(result["questionable_findings"]).lower()
        assert "amazon" in str(result["questionable_findings"]).lower() or "oxygen" in str(result["questionable_findings"]).lower()
        assert "verify sources" in str(result["recommendations"]).lower() or "check claims" in str(result["recommendations"]).lower()

    @pytest.mark.asyncio
    async def test_verify_research_findings_empty(self, verification_agent):
        """Test verifying empty research findings."""
        # Call the method with empty findings
        with pytest.raises(ValueError, match="No research findings provided"):
            await verification_agent.verify_research_findings({})

    @pytest.mark.asyncio
    async def test_generate_verification_report(self, verification_agent):
        """Test generating a comprehensive verification report."""
        # Setup verification results
        verification_results = {
            "fact1": {
                "claim": "Global temperatures have risen by about 1.1°C since pre-industrial times.",
                "verified": True,
                "confidence": 0.95,
                "sources": ["IPCC", "NASA"]
            },
            "fact2": {
                "claim": "Sea levels have risen by 8-9 inches since 1880.",
                "verified": True,
                "confidence": 0.9,
                "sources": ["NOAA", "NASA"]
            },
            "fact3": {
                "claim": "The Amazon rainforest produces 50% of the world's oxygen.",
                "verified": False,
                "confidence": 0.85,
                "sources": ["Scientific American", "Oxford University"]
            }
        }
        
        # Call the method
        report = await verification_agent.generate_verification_report(verification_results)
        
        # Verify memory was updated
        verification_agent.memory_mcp.store_memory.assert_called_with(
            "verification_report", str(report), namespace="verification"
        )
        
        # Verify report structure
        assert "summary" in report
        assert "verified_claims" in report
        assert "refuted_claims" in report
        assert "uncertain_claims" in report
        assert "methodology" in report
        assert "recommendations" in report
        
        # Verify report content
        assert len(report["summary"]) >= 100
        assert len(report["verified_claims"]) == 2
        assert len(report["refuted_claims"]) == 1
        assert "temperature" in str(report["verified_claims"]).lower()
        assert "sea level" in str(report["verified_claims"]).lower()
        assert "amazon" in str(report["refuted_claims"]).lower() or "oxygen" in str(report["refuted_claims"]).lower()
        assert len(report["methodology"]) >= 50
        assert len(report["recommendations"]) >= 2

    @pytest.mark.asyncio
    async def test_generate_verification_report_empty(self, verification_agent):
        """Test generating a verification report with no results."""
        # Call the method with empty results
        with pytest.raises(ValueError, match="No verification results provided"):
            await verification_agent.generate_verification_report({})


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
