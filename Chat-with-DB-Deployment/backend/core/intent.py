"""
Intent analysis for natural language queries with improved accuracy
"""

import logging
import re
import time
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from backend.core.entity_loader import get_entity_loader
from backend.core.types import IntentType, QueryAnalysis, QueryType

logger = logging.getLogger(__name__)


class IntentAnalyzer:
    """
    Analyzes natural language queries to extract intent, entities, and metadata.
    """

    def __init__(self):
        # Initialize entity loader
        self.entity_loader = get_entity_loader()

        # Query type keywords
        self.query_type_keywords = {
            QueryType.STATE: ["state", "states", "maharashtra", "delhi", "karnataka"],
            QueryType.REGION: [
                "region",
                "regions",
                "northern",
                "southern",
                "eastern",
                "western",
            ],
            QueryType.GENERATION: ["generation", "source", "solar", "thermal", "hydro"],
            QueryType.TRANSMISSION: [
                "transmission",
                "line",
                "flow",
                "import",
                "export",
                "loading",
                "link",
                "corridor",
                "domestic",
            ],
            QueryType.EXCHANGE: ["exchange", "country", "international", "cross border", "cross-border"],
            QueryType.TIME_BLOCK: ["time block", "time-block", "hourly", "15 min", "15-minute", "intraday"],
        }

        # Intent type keywords
        self.intent_keywords = {
            IntentType.DATA_RETRIEVAL: ["show", "get", "find", "what", "which"],
            IntentType.COMPARISON: ["compare", "versus", "vs", "against", "difference"],
            IntentType.TREND_ANALYSIS: [
                "trend",
                "growth",
                "increase",
                "decrease",
                "over time",
            ],
            IntentType.AGGREGATION: ["total", "sum", "average", "maximum", "minimum"],
        }

    async def analyze_intent(self, query: str, llm_provider=None) -> QueryAnalysis:
        """
        Analyze the intent of a natural language query.
        """
        try:
            # Detect query type
            query_type = self._detect_query_type(query)

            # Analyze intent
            intent_type = self._analyze_intent_patterns(query)

            # Extract entities
            entities = self._extract_entities(query)

            # Extract time period
            time_period = await self._analyze_time_period(query, llm_provider)
            logger.info(f"Time period analysis result: {time_period}")

            # Extract metrics
            metrics = self._extract_metrics(query)

            # Calculate confidence
            confidence = self._calculate_confidence(query, query_type, intent_type)

            return QueryAnalysis(
                query_type=query_type,
                intent=intent_type,
                entities=entities,
                time_period=time_period,
                metrics=metrics,
                confidence=confidence,
                main_table=self._get_main_table(query_type),
                dimension_table=self._get_dimension_table(query_type),
                join_key=self._get_join_key(query_type),
                name_column=self._get_name_column(query_type),
                detected_keywords=self._get_detected_keywords(query),
            )

        except Exception as e:
            logger.error(f"Error analyzing intent: {e}")
            return self._fallback_analysis(query)

    def _detect_query_type(self, query: str) -> QueryType:
        """
        Detect the query type based on keywords and entities.
        """
        query_lower = query.lower()

        # Force time-block tables for sub-daily granularity
        timeblock_terms = ["time block", "time-block", "hourly", "15 min", "15-minute", "intraday"]
        if any(t in query_lower for t in timeblock_terms):
            # Choose generation variant only when explicitly asking by source or mentioning source categories
            if any(k in query_lower for k in ["by source", "source", "sources"]):
                return QueryType.TIME_BLOCK_GENERATION
            return QueryType.TIME_BLOCK

        # Force transmission/international link flow
        if ("flow" in query_lower or "link flow" in query_lower or "linkflow" in query_lower or "energy exchanged" in query_lower):
            if ("international" in query_lower) or any(c in query_lower for c in ["bangladesh","nepal","bhutan","myanmar"]):
                return QueryType.INTERNATIONAL_TRANSMISSION
            # Heuristic: 'energy exchanged' and 'line-wise' implies international link flow table
            if "energy exchanged" in query_lower and ("line-wise" in query_lower or "line wise" in query_lower or "by line" in query_lower):
                return QueryType.INTERNATIONAL_TRANSMISSION
            if "transmission" in query_lower or "link" in query_lower:
                return QueryType.TRANSMISSION

        # Check for outage keywords - these should use region-level data
        outage_keywords = [
            "outage",
            "central sector outage",
            "state sector outage",
            "private sector outage",
            "total outage",
        ]
        if any(keyword in query_lower for keyword in outage_keywords):
            logger.info(
                f"Outage keywords detected in query, returning REGION query type for FactAllIndiaDailySummary"
            )
            return QueryType.REGION

        # Check for "states in region" patterns FIRST - these should be STATE queries not REGION
        # This needs to come before demand keywords to override them
        states_in_region_patterns = [
            "states in",
            "all states in", 
            "all the states in",
            "states of",
            "all states of",
            "states within",
            "states from"
        ]
        if any(pattern in query_lower for pattern in states_in_region_patterns):
            logger.info(
                f"'States in region' pattern detected in query, returning STATE query type"
            )
            return QueryType.STATE

        # Check for demand keywords - these should also use region-level data for peak demand
        demand_keywords = [
            "peak demand",
            "maximum demand",
            "evening peak demand",
            "max demand",
        ]
        if any(keyword in query_lower for keyword in demand_keywords):
            logger.info(
                f"Demand keywords detected in query, returning REGION query type for FactAllIndiaDailySummary"
            )
            return QueryType.REGION

        # Check for state names - if a state is mentioned, prioritize STATE type
        indian_states = self.entity_loader.get_indian_states()
        for state in indian_states:
            if state in query_lower:
                logger.info(
                    f"State '{state}' detected in query, returning STATE query type"
                )
                return QueryType.STATE

        # Count keyword matches for each type
        type_scores = {}
        for query_type, keywords in self.query_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                type_scores[query_type] = score

        if not type_scores:
            return QueryType.GENERATION  # Default to generation

        # Return the type with the highest score
        return max(type_scores.items(), key=lambda x: x[1])[0]

    def _analyze_intent_patterns(self, query: str) -> IntentType:
        """
        Analyze intent patterns in the query.
        """
        query_lower = query.lower()

        # Check for comparison patterns
        if any(word in query_lower for word in ["compare", "versus", "vs", "against"]):
            return IntentType.COMPARISON

        # Check for trend patterns
        if any(
            word in query_lower
            for word in ["trend", "over time", "growth", "increase", "decrease"]
        ):
            return IntentType.TREND_ANALYSIS

        # Check for summary patterns
        if any(
            word in query_lower
            for word in ["summary", "total", "sum", "average", "mean"]
        ):
            return IntentType.AGGREGATION

        # Default to query
        return IntentType.DATA_RETRIEVAL

    async def _analyze_time_period(
        self, query: str, llm_provider=None
    ) -> Optional[Dict[str, Any]]:
        """Extract time period information from query using robust parsing with LLM fallback"""
        import re  # Ensure re is imported at the top

        query_lower = query.lower()

        # First, try the regex-based approach (which we know works)
        try:
            # Check for specific month/year combinations using keyword matching
            month_keywords = self.entity_loader.get_month_keywords()

            # Look for month + year pattern with optional day (prioritize this over standalone year)
            for month_name, month_num in month_keywords.items():
                if month_name in query_lower:
                    # Look for day + month + year pattern (e.g., "1st July 2025", "1st jul 2025")
                    day_month_year_patterns = [
                        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{month_name}\s+(\d{{4}})\b",  # "1st July 2025"
                        rf"\b{month_name}\s+(\d{{1,2}})(?:st|nd|rd|th)?\s+(\d{{4}})\b",  # "July 1st 2025"
                        rf"\b(\d{{1,2}})\s+{month_name}\s+(\d{{4}})\b",  # "1 July 2025"
                        rf"\b{month_name}\s+(\d{{1,2}})\s+(\d{{4}})\b",  # "July 1 2025"
                    ]

                    for pattern in day_month_year_patterns:
                        match = re.search(pattern, query_lower)
                        if match:
                            day = int(match.group(1))
                            year = int(match.group(2))
                            logger.info(
                                f"Found day+month+year pattern: day={day}, month={month_num}, year={year}"
                            )
                            result = {
                                "type": "specific_date",
                                "year": year,
                                "month": month_num,
                                "day": day,
                                "confidence": 0.9,
                            }
                            logger.info(f"Returning time period result: {result}")
                            return result

                    # Look for month + year pattern without day
                    year_match = re.search(
                        rf"\b{month_name}\s+(\d{{4}})\b", query_lower
                    )
                    if year_match:
                        year = int(year_match.group(1))
                        logger.info(
                            f"Found month+year pattern: month={month_num}, year={year}"
                        )
                        return {
                            "type": "specific_date",
                            "year": year,
                            "month": month_num,
                            "day": None,
                            "confidence": 0.9,
                        }

            # Check for standalone year (e.g., "in 2025") - only if no month was found
            year_match = re.search(r"\b(\d{4})\b", query_lower)
            if year_match:
                year = int(year_match.group(1))
                # Check if this is a reasonable year (2000-2030)
                if 2000 <= year <= 2030:
                    return {
                        "type": "specific_date",
                        "year": year,
                        "month": None,
                        "day": None,
                        "confidence": 0.8,
                    }
        except Exception as e:
            logger.warning(f"Regex-based time period extraction failed: {e}")

        # If regex approach fails, try dateutil (but don't let it block the method)
        try:
            # Use dateutil as the primary method for robust date parsing
            from dateutil import parser
            from dateutil.relativedelta import relativedelta

            # First, try to extract date expressions using dateutil
            # Look for common date patterns in the query
            date_expressions = []

            # Common date patterns to look for
            date_patterns = [
                # "June 2025", "Jan 2024" - Month Year format
                r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})\b",
                # "1st July 2025", "July 1st 2025"
                r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})\b",
                r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?\s+(\d{4})\b",
                # "2025-06-01", "01/06/2025"
                r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",
                r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b",
                # Date ranges
                r"\bfrom\s+(.+?)\s+to\s+(.+?)\b",
                r"\b(.+?)\s+to\s+(.+?)\b",
            ]

            for pattern in date_patterns:
                matches = re.finditer(pattern, query_lower)
                for match in matches:
                    date_expressions.append(match.group())

            # If we found date expressions, try to parse them
            if date_expressions:
                parsed_dates: List[date] = []

                for expr in date_expressions:
                    try:
                        # Handle special cases
                        if "to" in expr or "from" in expr:
                            # This is a date range, handle separately
                            continue

                        # Try to parse with dateutil
                        parsed_date: Any = parser.parse(expr, fuzzy=True)
                        # Convert datetime to date if needed
                        if hasattr(parsed_date, "date"):
                            parsed_date = parsed_date.date()
                        parsed_dates.append(parsed_date)

                    except Exception as e:
                        # If dateutil fails, try manual parsing
                        try:
                            manual_parsed_date: Optional[date] = (
                                self._manual_date_parse(expr)
                            )
                            if manual_parsed_date:
                                parsed_dates.append(manual_parsed_date)
                        except:
                            continue

                if parsed_dates:
                    if len(parsed_dates) == 1:
                        # Single date
                        date_obj = parsed_dates[0]

                        # Check if this was a month-only date by looking at the original expression
                        original_expr = date_expressions[0].lower()
                        # If the expression matches "month year" pattern without day, set day to None
                        month_year_pattern = r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})\b"
                        is_month_only = bool(
                            re.match(month_year_pattern, original_expr)
                        )

                        return {
                            "type": "specific_date",
                            "year": date_obj.year,
                            "month": date_obj.month,
                            "day": None if is_month_only else date_obj.day,
                            "confidence": 0.9,
                        }
                    elif len(parsed_dates) == 2:
                        # Date range
                        start_date, end_date = parsed_dates[0], parsed_dates[1]
                        return {
                            "type": "date_range",
                            "start_year": start_date.year,
                            "start_month": start_date.month,
                            "end_year": end_date.year,
                            "end_month": end_date.month,
                            "confidence": 0.9,
                        }

            # Check for relative time periods using keyword matching
            relative_patterns = {
                "last_week": ["last week", "past week", "previous week"],
                "last_month": ["last month", "past month", "previous month"],
                "last_year": ["last year", "past year", "previous year"],
                "this_week": ["this week", "current week"],
                "this_month": ["this month", "current month"],
                "this_year": ["this year", "current year"],
            }

            for period, patterns in relative_patterns.items():
                if any(pattern in query_lower for pattern in patterns):
                    return {"period": period, "type": "relative", "confidence": 0.8}

        except ImportError:
            logger.warning("dateutil not available, using regex-only approach")
        except Exception as e:
            logger.warning(f"dateutil-based time period extraction failed: {e}")

        # === LLM Hook #8 – "Time Period Identification" ===
        # If regex patterns didn't find anything, use LLM to extract time information
        if llm_provider:
            prompt = f"""Extract time period information from this query: "{query}"

Return ONLY a JSON object with this exact format:
- For specific dates: {{"type": "specific_date", "year": 2024, "month": 6, "day": null, "confidence": 0.9}}
- For date ranges: {{"type": "date_range", "start_year": 2024, "start_month": 1, "end_year": 2025, "end_month": 6, "confidence": 0.9}}
- For relative periods: {{"type": "relative", "period": "this_month", "confidence": 0.8}}

Examples:
- "in June 2025" → {{"type": "specific_date", "year": 2025, "month": 6, "day": null, "confidence": 0.9}}
- "from Jan 2024 to Jun 2025" → {{"type": "date_range", "start_year": 2024, "start_month": 1, "end_year": 2025, "end_month": 6, "confidence": 0.9}}
- "this month" → {{"type": "relative", "period": "this_month", "confidence": 0.8}}

Return ONLY the JSON object, no explanations or code:"""

            try:
                response = await llm_provider.generate(prompt)

                # Try to parse the JSON response
                import json

                try:
                    # Clean the response to extract just the JSON
                    response_content = (
                        response.content.strip()
                        if hasattr(response, "content")
                        else str(response).strip()
                    )
                    if response_content.startswith("```json"):
                        response_content = response_content[7:]
                    if response_content.endswith("```"):
                        response_content = response_content[:-3]
                    response_content = response_content.strip()

                    time_period = json.loads(response_content)
                    if time_period and isinstance(time_period, dict):
                        return time_period
                except json.JSONDecodeError:
                    logger.warning(
                        f"LLM Hook #8: Failed to parse JSON response: {response_content}"
                    )
                    return None

            except Exception as e:
                logger.warning(f"LLM Hook #8: Error in time period extraction: {e}")
                return None

        return None

    def _manual_date_parse(self, date_str: str) -> Optional[date]:
        """Manual date parsing for common formats"""
        try:
            import re

            # Handle "June 2025" format
            month_names = {
                "january": 1,
                "february": 2,
                "march": 3,
                "april": 4,
                "may": 5,
                "june": 6,
                "july": 7,
                "august": 8,
                "september": 9,
                "october": 10,
                "november": 11,
                "december": 12,
                "jan": 1,
                "feb": 2,
                "mar": 3,
                "apr": 4,
                "may": 5,
                "jun": 6,
                "jul": 7,
                "aug": 8,
                "sep": 9,
                "oct": 10,
                "nov": 11,
                "dec": 12,
            }

            match = re.match(r"(\w+)\s+(\d{4})", date_str.strip())
            if match:
                month_name, year = match.groups()
                month = month_names.get(month_name.lower())
                if month:
                    return date(int(year), month, 1)

            return None
        except:
            return None

    def _fallback_date_parsing(self, query_lower: str) -> Optional[Dict[str, Any]]:
        """Fallback to simple regex parsing if advanced parsing fails"""
        import re

        month_names = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }

        # Check for standalone year first (e.g., "in 2025")
        year_match = re.search(r"\b(\d{4})\b", query_lower)
        if year_match:
            year = int(year_match.group(1))
            # Check if this is a reasonable year (2000-2030)
            if 2000 <= year <= 2030:
                return {
                    "type": "specific_date",
                    "year": year,
                    "month": None,
                    "day": None,
                    "confidence": 0.8,
                }

        # Simple month + year pattern
        for month_name, month_num in month_names.items():
            if month_name in query_lower:
                year_match = re.search(rf"\b{month_name}\s+(\d{{4}})\b", query_lower)
                if year_match:
                    year = int(year_match.group(1))
                    return {
                        "type": "specific_date",
                        "year": year,
                        "month": month_num,
                        "day": None,
                        "confidence": 0.7,
                    }

        return None

    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract entities from the query.
        """
        entities = []
        query_lower = query.lower()

        # Extract energy-related entities
        if "energy" in query_lower:
            entities.append("energy")
        elif "power" in query_lower:
            entities.append("power")
        elif "generation" in query_lower:
            entities.append("generation")
        elif "consumption" in query_lower:
            entities.append("consumption")
        elif "demand" in query_lower:
            entities.append("demand")
        elif "load" in query_lower:
            entities.append("load")
        elif "shortage" in query_lower:
            entities.append("shortage")
        elif "outage" in query_lower:
            entities.append("outage")
        elif "drawal" in query_lower:
            entities.append("drawal")
        elif "import" in query_lower:
            entities.append("import")
        elif "export" in query_lower:
            entities.append("export")

        # Transmission-specific and time-block entities
        transmission_terms = ["transmission", "line", "flow", "import", "export", "loading", "corridor", "link"]
        timeblock_terms = ["time block", "time-block", "hourly", "15 min", "15-minute", "intraday"]
        for t in transmission_terms:
            if t in query_lower and t not in entities:
                entities.append(t)
        for t in timeblock_terms:
            if t in query_lower and t not in entities:
                entities.append(t)

        # Extract regions using centralized entity loader with alias support (e.g., "north")
        # First try direct matches from configured regions
        indian_regions = self.entity_loader.get_indian_regions()
        for region in indian_regions:
            if region.lower() in query_lower:
                entities.append(region)
        # Then scan tokens/phrases for region aliases using mappings
        region_mappings = self.entity_loader.get_region_name_mappings()
        if region_mappings:
            # Sort by descending key length to prefer longer matches (e.g., "north eastern region" before "north")
            for alias in sorted(region_mappings.keys(), key=len, reverse=True):
                alias_lower = alias.lower()
                if alias_lower:
                    # Use word boundary matching to avoid false positives like "er" in "andhra pradesh"
                    # For single-letter or very short aliases, only match if they are standalone words
                    if len(alias_lower) <= 3:
                        # For short aliases, use word boundaries to ensure they're standalone
                        import re
                        pattern = r'\b' + re.escape(alias_lower) + r'\b'
                        if re.search(pattern, query_lower):
                            canonical = region_mappings[alias]
                            if canonical not in entities:
                                entities.append(canonical)
                    else:
                        # For longer aliases, use substring matching as before
                        if alias_lower in query_lower:
                            canonical = region_mappings[alias]
                            if canonical not in entities:
                                entities.append(canonical)

        # Extract states using centralized entity loader
        indian_states = self.entity_loader.get_indian_states()
        for state in indian_states:
            if state.lower() in query_lower:
                entities.append(state)

        return entities

    def _extract_metrics(self, query: str) -> List[str]:
        """
        Extract metrics/measures from the query.
        """
        metrics = []
        query_lower = query.lower()

        # Use centralized entity loader for energy metrics
        energy_metrics = self.entity_loader.get_energy_metrics()

        for metric in energy_metrics:
            if metric in query_lower:
                metrics.append(metric)

        return metrics

    def _calculate_confidence(
        self, query: str, query_type: QueryType, intent_type: IntentType
    ) -> float:
        """
        Calculate confidence score for the analysis.
        """
        confidence = 0.5  # Base confidence

        # Boost confidence based on keyword matches
        query_lower = query.lower()

        # Query type confidence
        type_keywords = self.query_type_keywords.get(query_type, [])
        type_matches = sum(1 for keyword in type_keywords if keyword in query_lower)
        if type_matches > 0:
            confidence += 0.2

        # Intent type confidence
        intent_keywords = self.intent_keywords.get(intent_type, [])
        intent_matches = sum(1 for keyword in intent_keywords if keyword in query_lower)
        if intent_matches > 0:
            confidence += 0.2

        # Entity confidence
        entities = self._extract_entities(query)
        if entities:
            confidence += 0.1

        return min(confidence, 1.0)

    def _get_detected_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from the query"""
        query_lower = query.lower()
        detected_keywords = []

        # Get keywords from entity loader
        growth_keywords = self.entity_loader.get_growth_keywords()
        aggregation_keywords = self.entity_loader.get_aggregation_keywords()
        comparison_keywords = self.entity_loader.get_comparison_keywords()

        # Check for growth keywords
        for keyword in growth_keywords:
            if keyword in query_lower:
                detected_keywords.append(keyword)

        # Check for aggregation keywords
        for keyword in aggregation_keywords:
            if keyword in query_lower:
                detected_keywords.append(keyword)

        # Check for comparison keywords
        for keyword in comparison_keywords:
            if keyword in query_lower:
                detected_keywords.append(keyword)

        # Additional growth pattern detection
        growth_patterns = [
            "monthly growth",
            "quarterly growth",
            "yearly growth",
            "annual growth",
            "month over month",
            "quarter over quarter",
            "year over year",
            "growth rate",
            "growth percentage",
            "growth trend",
            "increase over time",
            "decrease over time",
            "trend analysis",
        ]

        for pattern in growth_patterns:
            if pattern in query_lower:
                detected_keywords.append(pattern)

        # Time period keywords
        time_keywords = [
            "monthly",
            "quarterly",
            "yearly",
            "annual",
            "month",
            "quarter",
            "year",
        ]
        for keyword in time_keywords:
            if keyword in query_lower:
                detected_keywords.append(keyword)

        # Add domain-specific routing keywords for later stages
        for k in ["time block","time-block","hourly","15 min","15-minute","intraday","link flow","linkflow","flow","transmission","international","energy flow"]:
            if k in query_lower:
                detected_keywords.append(k)

        return detected_keywords

    def _get_main_table(self, query_type: QueryType) -> str:
        """Get the main fact table for the query type"""
        table_mapping = {
            QueryType.STATE: "FactStateDailyEnergy",
            QueryType.REGION: "FactAllIndiaDailySummary",
            QueryType.GENERATION: "FactDailyGenerationBreakdown",
            QueryType.TRANSMISSION: "FactTransmissionLinkFlow",
            QueryType.INTERNATIONAL_TRANSMISSION: "FactInternationalTransmissionLinkFlow",
            QueryType.EXCHANGE: "FactCountryDailyExchange",
            QueryType.EXCHANGE_DETAIL: "FactTransnationalExchangeDetail",
            QueryType.TIME_BLOCK: "FactTimeBlockPowerData",
            QueryType.TIME_BLOCK_GENERATION: "FactTimeBlockGeneration",
        }
        return table_mapping.get(query_type, "FactAllIndiaDailySummary")

    def _get_international_transmission_table(self, query: str) -> str:
        """Determine if international transmission table should be used"""
        query_lower = query.lower()
        international_keywords = [
            "international",
            "india-nepal",
            "india-bhutan",
            "india-bangladesh",
            "nepal",
            "bhutan",
            "bangladesh",
        ]
        if any(keyword in query_lower for keyword in international_keywords):
            return "FactInternationalTransmissionLinkFlow"
        return "FactTransmissionLinkFlow"

    def _get_dimension_table(self, query_type: QueryType) -> str:
        """Get dimension table for query type"""
        table_mapping = {
            QueryType.REGION: "DimRegions",
            QueryType.STATE: "DimStates",
            QueryType.GENERATION: "DimGenerationSources",
            QueryType.TRANSMISSION: "DimTransmissionLines",
            QueryType.INTERNATIONAL_TRANSMISSION: "DimTransmissionLines",
            QueryType.EXCHANGE: "DimCountries",
            QueryType.TIME_BLOCK: "DimTimeBlocks",
            QueryType.TIME_BLOCK_GENERATION: "DimGenerationSources",
        }
        return table_mapping.get(query_type, "DimRegions")

    def _get_join_key(self, query_type: QueryType) -> str:
        """Get join key for query type"""
        key_mapping = {
            QueryType.REGION: "RegionID",
            QueryType.STATE: "StateID",
            QueryType.GENERATION: "GenerationSourceID",
            QueryType.TRANSMISSION: "LineID",  # Fixed: actual column name
            QueryType.INTERNATIONAL_TRANSMISSION: "LineID",
            QueryType.EXCHANGE: "CountryID",
            QueryType.TIME_BLOCK: "TimeBlockID",
            QueryType.TIME_BLOCK_GENERATION: "GenerationSourceID",
        }
        return key_mapping.get(query_type, "RegionID")

    def _get_name_column(self, query_type: QueryType) -> str:
        """Get name column for query type"""
        column_mapping = {
            QueryType.REGION: "RegionName",
            QueryType.STATE: "StateName",
            QueryType.GENERATION: "SourceName",
            QueryType.TRANSMISSION: "LineIdentifier",  # Fixed: actual column name
            QueryType.INTERNATIONAL_TRANSMISSION: "LineIdentifier",
            QueryType.EXCHANGE: "CountryName",
            QueryType.TIME_BLOCK: "TimeBlockName",
            QueryType.TIME_BLOCK_GENERATION: "SourceName",
        }
        return column_mapping.get(query_type, "RegionName")

    def _fallback_analysis(self, query: str) -> QueryAnalysis:
        """
        Fallback analysis when the main analysis fails.
        """
        # Determine the appropriate query type based on the query content
        query_lower = query.lower()

        # Check for region-related keywords
        region_keywords = [
            "region",
            "regions",
            "northern",
            "southern",
            "eastern",
            "western",
            "north eastern",
        ]
        state_keywords = [
            "state",
            "states",
            "maharashtra",
            "karnataka",
            "tamil nadu",
            "gujarat",
            "andhra pradesh",
        ]
        generation_keywords = [
            "generation",
            "coal",
            "thermal",
            "solar",
            "wind",
            "nuclear",
            "hydro",
            "gas",
            "biomass",
        ]

        if any(keyword in query_lower for keyword in region_keywords):
            query_type = QueryType.REGION
            main_table = "FactAllIndiaDailySummary"
            dimension_table = "DimRegions"
            join_key = "RegionID"
            name_column = "RegionName"
        elif any(keyword in query_lower for keyword in state_keywords):
            query_type = QueryType.STATE
            main_table = "FactStateDailyEnergy"
            dimension_table = "DimStates"
            join_key = "StateID"
            name_column = "StateName"
        elif any(keyword in query_lower for keyword in generation_keywords):
            query_type = QueryType.GENERATION
            main_table = "FactDailyGenerationBreakdown"
            dimension_table = "DimGenerationSources"
            join_key = "GenerationSourceID"
            name_column = "SourceName"
        else:
            # Default to region for Energy Met queries
            query_type = QueryType.REGION
            main_table = "FactAllIndiaDailySummary"
            dimension_table = "DimRegions"
            join_key = "RegionID"
            name_column = "RegionName"

        return QueryAnalysis(
            query_type=query_type,
            intent=IntentType.DATA_RETRIEVAL,
            entities=[],
            time_period=None,
            metrics=[],
            confidence=0.3,
            main_table=main_table,
            dimension_table=dimension_table,
            join_key=join_key,
            name_column=name_column,
            detected_keywords=[],
        )
