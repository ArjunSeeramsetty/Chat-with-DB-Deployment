"""
SQL Assembler Module
Provides intelligent SQL generation from natural language queries
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..core.entity_loader import get_entity_loader
from ..core.schema_linker import SchemaLinker
from ..core.types import ContextInfo, QueryAnalysis, SQLGenerationResult, UserMapping

logger = logging.getLogger(__name__)


class SQLAssembler:
    """Enhanced SQL assembler with improved template handling and dynamic table/column selection"""

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        logger.info("SQLAssembler initialized - Monthly detection enabled")
        self.general_geographic_keywords = [
            "all states",
            "all regions",
            "states",
            "regions",
            "country",
            "india",
        ]
        self.entity_loader = get_entity_loader()
        self.sql_templates = {
            "region_query": """
                SELECT d.RegionName, ROUND({aggregation_function}(f.{energy_column}), 2) as {column_alias}
                FROM FactAllIndiaDailySummary f
                JOIN DimRegions d ON f.RegionID = d.RegionID
                JOIN DimDates dt ON f.DateID = dt.DateID
                {where_clause}
                GROUP BY d.RegionName
                ORDER BY {column_alias} DESC
            """,
            "state_query": """
                SELECT ds.StateName, ROUND({aggregation_function}(fs.{energy_column}), 2) as {column_alias}
                FROM FactStateDailyEnergy fs
                JOIN DimStates ds ON fs.StateID = ds.StateID
                JOIN DimDates dt ON fs.DateID = dt.DateID
                {where_clause}
                GROUP BY ds.StateName
                ORDER BY {column_alias} DESC
            """,
            "generation_query": """
                SELECT dgs.SourceName, ROUND(SUM(fdgb.GenerationAmount), 2) as TotalGeneration
                FROM FactDailyGenerationBreakdown fdgb
                JOIN DimGenerationSources dgs ON fdgb.GenerationSourceID = dgs.GenerationSourceID
                JOIN DimDates dt ON fdgb.DateID = dt.DateID
                {where_clause}
                GROUP BY dgs.SourceName
                ORDER BY TotalGeneration DESC
            """,
            "region_generation_query": """
                SELECT d.RegionName, ROUND(SUM(fdgb.GenerationAmount), 2) as TotalGeneration
                FROM FactDailyGenerationBreakdown fdgb
                JOIN DimRegions d ON fdgb.RegionID = d.RegionID
                JOIN DimGenerationSources gs ON fdgb.GenerationSourceID = gs.GenerationSourceID
                JOIN DimDates dt ON fdgb.DateID = dt.DateID
                {where_clause}
                GROUP BY d.RegionName
                ORDER BY TotalGeneration DESC
            """,
            "transmission_query": """
                SELECT dtl.LineIdentifier, ROUND({aggregation_function}(ftl.{energy_column}), 2) as {column_alias}
                FROM FactTransmissionLinkFlow ftl
                JOIN DimTransmissionLinks dtl ON ftl.LineID = dtl.LineID
                JOIN DimDates dt ON ftl.DateID = dt.DateID
                {where_clause}
                GROUP BY dtl.LineIdentifier
                ORDER BY {column_alias} DESC
            """,
            "international_transmission_query": """
                SELECT dtl.LineIdentifier, ROUND({aggregation_function}(fitl.{energy_column}), 2) as {column_alias}
                FROM FactInternationalTransmissionLinkFlow fitl
                JOIN DimTransmissionLines dtl ON fitl.LineID = dtl.LineID
                JOIN DimDates dt ON fitl.DateID = dt.DateID
                {where_clause}
                GROUP BY dtl.LineIdentifier
                ORDER BY {column_alias} DESC
            """,
            "exchange_query": """
                SELECT 
                    dc.CountryName,
                    dem.MechanismName,
                    fted.ExchangeValue,
                    fted.ExchangeDirection,
                    dt.Year,
                    dt.Month,
                    dt.DayOfMonth
                FROM FactTransnationalExchangeDetail fted
                JOIN DimCountries dc ON fted.CountryID = dc.CountryID
                JOIN DimExchangeMechanisms dem ON fted.MechanismID = dem.MechanismID
                JOIN DimDates dt ON fted.DateID = dt.DateID
                {where_clause}
                ORDER BY fted.ExchangeValue DESC
            """,
            "country_daily_exchange_query": """
                SELECT 
                    dc.CountryName,
                    ROUND({aggregation_function}(fce.{energy_column}), 2) as {column_alias},
                    dt.Year,
                    dt.Month,
                    dt.DayOfMonth
                FROM FactCountryDailyExchange fce
                JOIN DimCountries dc ON fce.CountryID = dc.CountryID
                JOIN DimDates dt ON fce.DateID = dt.DateID
                {where_clause}
                GROUP BY dc.CountryName, dt.Year, dt.Month, dt.DayOfMonth
                ORDER BY {column_alias} DESC
            """,
            "time_block_query": """
                SELECT ROUND({aggregation_function}(ftb.{energy_column}), 2) as {column_alias}
                FROM FactTimeBlockPowerData ftb
                JOIN DimDates dt ON ftb.DateID = dt.DateID
                {where_clause}
                ORDER BY {column_alias} DESC
            """,
            "time_block_generation_query": """
                SELECT dgs.SourceName, ftbg.BlockNumber, ROUND(SUM(ftbg.GenerationOutput), 2) as TotalGeneration
                FROM FactTimeBlockGeneration ftbg
                JOIN DimGenerationSources dgs ON ftbg.GenerationSourceID = dgs.GenerationSourceID
                JOIN DimDates dt ON ftbg.DateID = dt.DateID
                {where_clause}
                GROUP BY dgs.SourceName, ftbg.BlockNumber
                ORDER BY TotalGeneration DESC
            """,
            "monthly_dynamic_growth_query": """
                SELECT 
                    current.{name_column},
                    current.Month,
                    current.MonthlyValue as {current_alias},
                    previous.MonthlyValue as {previous_alias},
                    CASE 
                        WHEN previous.MonthlyValue IS NULL THEN NULL
                        WHEN previous.MonthlyValue = 0 THEN NULL
                        ELSE ROUND(((current.MonthlyValue - previous.MonthlyValue) / previous.MonthlyValue) * 100, 2)
                    END as GrowthPercentage,
                    CASE 
                        WHEN previous.MonthlyValue IS NULL THEN 'No Previous Data'
                        WHEN ((current.MonthlyValue - previous.MonthlyValue) / previous.MonthlyValue) * 100 > 0 THEN 'Growth'
                        WHEN ((current.MonthlyValue - previous.MonthlyValue) / previous.MonthlyValue) * 100 < 0 THEN 'Decline'
                        ELSE 'No Change'
                    END as Trend
                FROM (
                    SELECT 
                        d.{name_column},
                        dt.Month,
                        ROUND(SUM(f.{energy_column}), 2) as MonthlyValue
                    FROM {main_table} f
                    {join_clause}
                    JOIN DimDates dt ON f.DateID = dt.DateID
                    WHERE {where_clause}
                    GROUP BY d.{name_column}, dt.Year, dt.Month
                ) current
                LEFT JOIN (
                    SELECT 
                        d.{name_column},
                        dt.Month,
                        ROUND(SUM(f.{energy_column}), 2) as MonthlyValue
                    FROM {main_table} f
                    {join_clause}
                    JOIN DimDates dt ON f.DateID = dt.DateID
                    WHERE {previous_where_clause}
                    GROUP BY d.{name_column}, dt.Year, dt.Month
                ) previous ON 
                    current.{name_column} = previous.{name_column} AND
                    (current.Month = previous.Month + 1 OR (current.Month = 1 AND previous.Month = 12))
                ORDER BY current.{name_column}, current.Month
            """,
            "weekly_dynamic_growth_query": """
                SELECT 
                    current.{name_column},
                    current.Week,
                    current.WeeklyValue as {current_alias},
                    previous.WeeklyValue as {previous_alias},
                    CASE 
                        WHEN previous.WeeklyValue IS NULL THEN NULL
                        WHEN previous.WeeklyValue = 0 THEN NULL
                        ELSE ROUND(((current.WeeklyValue - previous.WeeklyValue) / previous.WeeklyValue) * 100, 2)
                    END as GrowthPercentage,
                    CASE 
                        WHEN previous.WeeklyValue IS NULL THEN 'No Previous Data'
                        WHEN ((current.WeeklyValue - previous.WeeklyValue) / previous.WeeklyValue) * 100 > 0 THEN 'Growth'
                        WHEN ((current.WeeklyValue - previous.WeeklyValue) / previous.WeeklyValue) * 100 < 0 THEN 'Decline'
                        ELSE 'No Change'
                    END as Trend
                FROM (
                    SELECT 
                        d.{name_column},
                        dt.Week,
                        ROUND(SUM(f.{energy_column}), 2) as WeeklyValue
                    FROM {main_table} f
                    {join_clause}
                    JOIN DimDates dt ON f.DateID = dt.DateID
                    WHERE {where_clause}
                    GROUP BY d.{name_column}, dt.Year, dt.Week
                ) current
                LEFT JOIN (
                    SELECT 
                        d.{name_column},
                        dt.Week,
                        ROUND(SUM(f.{energy_column}), 2) as WeeklyValue
                    FROM {main_table} f
                    {join_clause}
                    JOIN DimDates dt ON f.DateID = dt.DateID
                    WHERE {previous_where_clause}
                    GROUP BY d.{name_column}, dt.Year, dt.Week
                ) previous ON 
                    current.{name_column} = previous.{name_column} AND
                    current.Week = previous.Week + 1
                ORDER BY current.{name_column}, current.Week
            """,
            "daily_dynamic_growth_query": """
                SELECT 
                    current.{name_column},
                    current.DayOfMonth,
                    current.DailyValue as {current_alias},
                    previous.DailyValue as {previous_alias},
                    CASE 
                        WHEN previous.DailyValue IS NULL THEN NULL
                        WHEN previous.DailyValue = 0 THEN NULL
                        ELSE ROUND(((current.DailyValue - previous.DailyValue) / previous.DailyValue) * 100, 2)
                    END as GrowthPercentage,
                    CASE 
                        WHEN previous.DailyValue IS NULL THEN 'No Previous Data'
                        WHEN ((current.DailyValue - previous.DailyValue) / previous.DailyValue) * 100 > 0 THEN 'Growth'
                        WHEN ((current.DailyValue - previous.DailyValue) / previous.DailyValue) * 100 < 0 THEN 'Decline'
                        ELSE 'No Change'
                    END as Trend
                FROM (
                    SELECT 
                        d.{name_column},
                        dt.DayOfMonth,
                        ROUND(SUM(f.{energy_column}), 2) as DailyValue
                    FROM {main_table} f
                    {join_clause}
                    JOIN DimDates dt ON f.DateID = dt.DateID
                    WHERE {where_clause}
                    GROUP BY d.{name_column}, dt.Year, dt.Month, dt.DayOfMonth
                ) current
                LEFT JOIN (
                    SELECT 
                        d.{name_column},
                        dt.DayOfMonth,
                        ROUND(SUM(f.{energy_column}), 2) as DailyValue
                    FROM {main_table} f
                    {join_clause}
                    JOIN DimDates dt ON f.DateID = dt.DateID
                    WHERE {previous_where_clause}
                    GROUP BY d.{name_column}, dt.Year, dt.Month, dt.DayOfMonth
                ) previous ON 
                    current.{name_column} = previous.{name_column} AND
                    current.DayOfMonth = previous.DayOfMonth + 1
                ORDER BY current.{name_column}, current.DayOfMonth
            """,
            "yearly_dynamic_growth_query": """
                SELECT 
                    current.{name_column},
                    current.Year,
                    current.YearlyValue as {current_alias},
                    previous.YearlyValue as {previous_alias},
                    CASE 
                        WHEN previous.YearlyValue IS NULL THEN NULL
                        WHEN previous.YearlyValue = 0 THEN NULL
                        ELSE ROUND(((current.YearlyValue - previous.YearlyValue) / previous.YearlyValue) * 100, 2)
                    END as GrowthPercentage,
                    CASE 
                        WHEN previous.YearlyValue IS NULL THEN 'No Previous Data'
                        WHEN ((current.YearlyValue - previous.YearlyValue) / previous.YearlyValue) * 100 > 0 THEN 'Growth'
                        WHEN ((current.YearlyValue - previous.YearlyValue) / previous.YearlyValue) * 100 < 0 THEN 'Decline'
                        ELSE 'No Change'
                    END as Trend
                FROM (
                    SELECT 
                        d.{name_column},
                        dt.Year,
                        ROUND(SUM(f.{energy_column}), 2) as YearlyValue
                    FROM {main_table} f
                    {join_clause}
                    JOIN DimDates dt ON f.DateID = dt.DateID
                    WHERE {where_clause}
                    GROUP BY d.{name_column}, dt.Year
                ) current
                LEFT JOIN (
                    SELECT 
                        d.{name_column},
                        dt.Year,
                        ROUND(SUM(f.{energy_column}), 2) as YearlyValue
                    FROM {main_table} f
                    {join_clause}
                    JOIN DimDates dt ON f.DateID = dt.DateID
                    WHERE {previous_where_clause}
                    GROUP BY d.{name_column}, dt.Year
                ) previous ON 
                    current.{name_column} = previous.{name_column} AND
                    current.Year = previous.Year + 1
                ORDER BY current.{name_column}, current.Year
            """,
            "quarterly_dynamic_growth_query": """
                WITH QuarterlyData AS (
                    SELECT 
                        dt.Year,
                        CASE 
                            WHEN dt.Month IN (1,2,3) THEN 'Q1'
                            WHEN dt.Month IN (4,5,6) THEN 'Q2'
                            WHEN dt.Month IN (7,8,9) THEN 'Q3'
                            WHEN dt.Month IN (10,11,12) THEN 'Q4'
                        END as Quarter,
                        d.{name_column},
                        ROUND(SUM(f.{energy_column}), 2) as QuarterlyValue
                    FROM {main_table} f
                    {join_clause}
                    JOIN DimDates dt ON f.DateID = dt.DateID
                    WHERE {where_clause}
                    GROUP BY dt.Year, 
                        CASE 
                            WHEN dt.Month IN (1,2,3) THEN 'Q1'
                            WHEN dt.Month IN (4,5,6) THEN 'Q2'
                            WHEN dt.Month IN (7,8,9) THEN 'Q3'
                            WHEN dt.Month IN (10,11,12) THEN 'Q4'
                        END,
                        d.{name_column}
                    ORDER BY dt.Year, Quarter, d.{name_column}
                ),
                GrowthData AS (
                    SELECT 
                        q1.Year,
                        q1.Quarter,
                        q1.{name_column},
                        q1.QuarterlyValue,
                        q2.QuarterlyValue as PreviousQuarterValue,
                        CASE 
                            WHEN q2.QuarterlyValue IS NULL THEN NULL
                            WHEN q2.QuarterlyValue = 0 THEN NULL
                            ELSE ROUND(((q1.QuarterlyValue - q2.QuarterlyValue) / q2.QuarterlyValue) * 100, 2)
                        END as GrowthPercentage
                    FROM QuarterlyData q1
                    LEFT JOIN QuarterlyData q2 ON 
                        q1.{name_column} = q2.{name_column} AND
                        (q1.Year = q2.Year AND 
                         ((q1.Quarter = 'Q2' AND q2.Quarter = 'Q1') OR
                          (q1.Quarter = 'Q3' AND q2.Quarter = 'Q2') OR
                          (q1.Quarter = 'Q4' AND q2.Quarter = 'Q3')) OR
                         (q1.Year = q2.Year + 1 AND q1.Quarter = 'Q1' AND q2.Quarter = 'Q4'))
                )
                SELECT 
                    {name_column},
                    Quarter,
                    QuarterlyValue,
                    PreviousQuarterValue,
                    GrowthPercentage,
                    CASE 
                        WHEN GrowthPercentage IS NULL THEN 'No Previous Data'
                        WHEN GrowthPercentage > 0 THEN 'Growth'
                        WHEN GrowthPercentage < 0 THEN 'Decline'
                        ELSE 'No Change'
                    END as Trend
                FROM GrowthData
                ORDER BY {name_column}, Year, Quarter
            """,
        }

    def generate_sql(
        self, query: str, analysis: QueryAnalysis, context: ContextInfo
    ) -> SQLGenerationResult:
        """
        Generate SQL based on query analysis and context.
        """
        print(
            f"🔍 GENERATE_SQL CALLED - Query: '{query}', Query Type: {analysis.query_type.value}"
        )
        try:
            # Build generation prompt
            prompt = self._build_generation_prompt(query, analysis, context)

            # Generate SQL using LLM or templates
            print(f"🔍 CALLING _generate_with_templates")
            sql = self._generate_with_templates(analysis, context, query)

            if not sql:
                print(f"🔍 TEMPLATE GENERATION FAILED, falling back to LLM")
                logger.warning(
                    f"Template generation failed for query: {query}, falling back to LLM"
                )
                logger.warning(
                    f"Context has schema_linker: {hasattr(context, 'schema_linker')}"
                )
                if hasattr(context, "schema_linker"):
                    logger.warning(
                        f"Schema linker is None: {context.schema_linker is None}"
                    )
                # Fallback to LLM if available
                sql = self._generate_with_llm(prompt)
            else:
                print(f"🔍 TEMPLATE GENERATION SUCCEEDED")
                logger.info(f"Template generation succeeded for query: {query}")

            if not sql:
                return SQLGenerationResult(
                    success=False,
                    sql="",  # Provide required field
                    error="Failed to generate SQL",
                    confidence=0.0,
                )

            # Post-process SQL
            processed_sql = self._post_process_sql(sql)

            # Calculate confidence based on query clarity and completeness
            confidence = self._calculate_query_confidence(query, analysis)
            logger.info(f"Final confidence score for query '{query}': {confidence}")

            return SQLGenerationResult(
                success=True, sql=processed_sql, confidence=confidence
            )

        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return SQLGenerationResult(
                success=False,
                sql="",  # Provide required field
                error=f"SQL generation error: {str(e)}",
                confidence=0.0,
            )

    def _calculate_query_confidence(self, query: str, analysis: QueryAnalysis) -> float:
        """
        Calculate confidence score for the query based on various factors.
        Higher confidence means less likely to need clarification.
        """
        query_lower = query.lower()
        confidence = 0.0

        # Base confidence from having a query type
        if analysis.query_type:
            confidence += 0.3

        # Geographic information
        has_geographic_info = (
            any("state" in entity.lower() for entity in analysis.entities)
            or any("region" in entity.lower() for entity in analysis.entities)
            or any(
                keyword in query_lower for keyword in self.general_geographic_keywords
            )
        )
        if has_geographic_info:
            confidence += 0.2

        # Time information
        has_time_info = (
            analysis.time_period
            or (analysis.time_period and analysis.time_period.get("start_year"))
            or (analysis.time_period and analysis.time_period.get("end_year"))
            or any(
                keyword in query_lower
                for keyword in ["2024", "2025", "2023", "this year", "last year"]
            )
        )
        if has_time_info:
            confidence += 0.2

        # Metric information - check for specific metrics mentioned
        metric_keywords = [
            "energymet",
            "energy met",
            "energy supplied",
            "energy consumed",
            "energy consumption",
            "energyshortage",
            "energy shortage",
            "energy not supplied",
            "demandmet",
            "demand met",
            "maximum demand",
            "peak demand",
            "outage",
            "central sector outage",
            "state sector outage",
            "private sector outage",
            "generation",
            "thermal generation",
            "total generation",
            "shortage",
            "power shortage",
        ]

        has_metric_info = any(keyword in query_lower for keyword in metric_keywords)
        if has_metric_info:
            confidence += 0.4  # Higher confidence when specific metrics are mentioned

        # Aggregation information
        aggregation_keywords = [
            "maximum",
            "max",
            "highest",
            "top",
            "most",
            "minimum",
            "min",
            "lowest",
            "least",
            "average",
            "avg",
            "mean",
            "total",
            "sum",
            "sum of",
            "count",
            "number of",
        ]

        has_aggregation_info = any(
            keyword in query_lower for keyword in aggregation_keywords
        )
        if has_aggregation_info:
            confidence += 0.3  # Higher confidence for aggregation queries

        # Bonus for very specific queries with clear intent
        if (
            has_metric_info
            and has_aggregation_info
            and has_geographic_info
            and has_time_info
        ):
            confidence += 0.2  # Extra bonus for complete queries

        # Penalize very vague queries
        vague_indicators = [
            "what is",
            "show me",
            "give me",
            "tell me about",
            "data",
            "information",
            "details",
        ]

        if (
            any(indicator in query_lower for indicator in vague_indicators)
            and not has_metric_info
        ):
            confidence -= 0.3  # Stronger penalty for vague queries

        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))

        # Log detailed confidence breakdown
        logger.info(f"Confidence calculation for '{query}':")
        logger.info(f"  - Query type: {analysis.query_type} (+0.3)")
        logger.info(f"  - Geographic info: {has_geographic_info} (+0.2)")
        logger.info(f"  - Time info: {has_time_info} (+0.2)")
        logger.info(f"  - Metric info: {has_metric_info} (+0.4)")
        logger.info(f"  - Aggregation info: {has_aggregation_info} (+0.3)")
        logger.info(
            f"  - Complete query bonus: {has_metric_info and has_aggregation_info and has_geographic_info and has_time_info} (+0.2)"
        )
        logger.info(f"  - Final confidence: {confidence:.3f}")

        return confidence

    def _build_generation_prompt(
        self, query: str, analysis: QueryAnalysis, context: ContextInfo
    ) -> str:
        """
        Build a prompt for SQL generation.
        """
        prompt = f"""
        Generate SQL for the following query: "{query}"
        
        Query Analysis:
        - Type: {analysis.query_type}
        - Intent: {analysis.intent}
        - Entities: {analysis.entities}
        - Metrics: {analysis.metrics}
        
        Available Tables and Columns:
        {self._format_schema_info(context.schema_info)}
        
        Available Dimension Values:
        {self._format_dimension_values({k: [v.name for v in values] for k, values in context.dimension_values.items()})}
        
        User Mappings:
        {self._format_user_mappings([{"user_term": m.mapping_type, "dimension_table": m.entity.table, "dimension_value": m.entity.name} for m in context.user_mappings])}
        
        Generate a valid SQLite SQL query that:
        1. Uses the appropriate fact and dimension tables
        2. Includes proper JOINs
        3. Applies relevant filters based on user mappings
        4. Groups and orders results appropriately
        5. Returns only the SQL query, no explanations
        """

        return prompt

    def _generate_with_templates(
        self, analysis: QueryAnalysis, context: ContextInfo, original_query: str = ""
    ) -> Optional[str]:
        """
        Generate SQL using predefined templates.
        """
        print(f"🔍 TEMPLATE GENERATION CALLED - Query: '{original_query}'")
        logger.info(f"🔍 TEMPLATE GENERATION STARTED - Query: '{original_query}'")
        logger.info(f"Template generation - Query: '{original_query}'")
        logger.info(f"Detected keywords: {analysis.detected_keywords}")
        logger.info(f"Query type: {analysis.query_type.value}")
        logger.info(f"Intent: {analysis.intent.value}")

        # Growth queries are now handled by the dynamic template selection logic below
        # This ensures all growth queries use the new dynamic templates that adapt to different tables and columns

        # Look for template based on query type and intent
        fallback_template_key: Optional[str] = None
        template_key: Optional[str] = None  # Initialize template_key
        if analysis.query_type.value == "exchange_detail":
            fallback_template_key = "exchange_query"
        elif analysis.query_type.value == "exchange":
            fallback_template_key = "country_daily_exchange_query"
        elif analysis.query_type.value == "transmission":
            original_query_lower = original_query.lower() if original_query else ""
            # Check if this is an international transmission query
            international_keywords = [
                "international",
                "india-nepal",
                "india-bhutan",
                "india-bangladesh",
                "nepal",
                "bhutan",
                "bangladesh",
            ]
            if any(
                keyword in original_query_lower for keyword in international_keywords
            ):
                fallback_template_key = "international_transmission_query"
            else:
                fallback_template_key = "transmission_query"
        elif analysis.query_type.value == "generation":
            # Check if this is a region-level generation query
            original_query_lower = original_query.lower() if original_query else ""
            if any(
                word in original_query_lower
                for word in ["region", "regions", "all regions"]
            ):
                fallback_template_key = "region_generation_query"
            else:
                fallback_template_key = "generation_query"
        elif analysis.query_type.value == "state":
            fallback_template_key = "state_query"
            template_key = fallback_template_key
        elif analysis.intent.value == "trend_analysis":
            # Dynamic trend analysis - determine template based on query content
            original_query_lower = original_query.lower() if original_query else ""

            # STEP 1: Determine if it's a growth query or aggregation query
            is_growth_query = any(
                word in original_query_lower
                for word in ["growth", "increase", "decrease", "change", "trend"]
            )
            is_aggregation_query = any(
                word in original_query_lower
                for word in [
                    "total",
                    "sum",
                    "average",
                    "avg",
                    "maximum",
                    "max",
                    "minimum",
                    "min",
                    "aggregate",
                ]
            )

            # If both are detected, prioritize growth (growth queries are more specific)
            if is_growth_query and is_aggregation_query:
                is_aggregation_query = False
                logger.info(
                    f"Both growth and aggregation detected, prioritizing growth query"
                )

            # STEP 2: Determine time period
            time_period = "none"  # Default - no time-based grouping unless explicitly mentioned
            if any(
                word in original_query_lower
                for word in ["quarterly", "quarter", "q1", "q2", "q3", "q4"]
            ):
                time_period = "quarterly"
            elif any(
                word in original_query_lower
                for word in ["yearly", "annual", "year over year"]
            ):
                time_period = "yearly"
            elif any(word in original_query_lower for word in ["weekly", "week"]):
                time_period = "weekly"
            elif any(word in original_query_lower for word in ["daily", "day"]):
                time_period = "daily"
            elif any(
                word in original_query_lower
                for word in ["monthly", "month over month", "per month", "by month"]
            ):
                time_period = "monthly"

            logger.info(
                f"Trend analysis - Growth: {is_growth_query}, Aggregation: {is_aggregation_query}, Time period: {time_period}"
            )

            # STEP 3: Select appropriate template
            if is_growth_query:
                # Use dynamic growth template that adapts based on query analysis
                template_key = f"{time_period}_dynamic_growth_query"
                logger.info(f"Selected dynamic growth template: {template_key}")
            elif is_aggregation_query:
                # Use aggregation template based on query type
                if analysis.query_type.value == "region":
                    template_key = "region_query"
                elif analysis.query_type.value == "state":
                    template_key = "state_query"
                elif analysis.query_type.value == "generation":
                    template_key = "region_generation_query"
                else:
                    template_key = "region_query"  # Default
                logger.info(f"Selected aggregation template: {template_key}")
            else:
                # Default to region query if unclear
                template_key = "region_query"
                logger.info(f"Default template selected: {template_key}")
        elif analysis.query_type.value == "region":
            fallback_template_key = "region_query"
        elif analysis.query_type.value == "time_block_generation":
            fallback_template_key = "time_block_generation_query"
        elif analysis.query_type.value == "time_block":
            fallback_template_key = "time_block_query"
        else:
            # For non-trend analysis queries, use fallback template selection
            if fallback_template_key:
                template_key = fallback_template_key
            else:
                template_key = "region_query"  # Default

        logger.info(
            f"Looking for template: {template_key}, Intent: {analysis.intent.value}, Keywords: {analysis.detected_keywords}"
        )
        logger.info(
            f"Query type: {analysis.query_type.value}, Template key: {template_key}"
        )
        logger.info(f"Available templates: {list(self.sql_templates.keys())}")
        logger.info(
            f"🔍 TEMPLATE SELECTION: query_type={analysis.query_type.value}, template_key={template_key}"
        )


        if template_key in self.sql_templates:
            logger.info(f"✅ Template found: {template_key}")
            template = self.sql_templates[template_key]

            # Special handling for exchange queries - they don't need energy_column processing
            if template_key == "exchange_query":
                # Build where clause based on extracted entities and user mappings
                where_clause = self._build_where_clause_from_entities(
                    analysis, context.user_mappings
                )

                sql = template.format(where_clause=where_clause)
                logger.info(f"Generated exchange SQL: {sql[:100]}...")
                return sql

            # Special handling for generation queries - they don't need energy_column processing
            if template_key == "generation_query":
                # Build where clause based on extracted entities and user mappings
                where_clause = self._build_where_clause_from_entities(
                    analysis, context.user_mappings
                )

                # Detect and add source filtering dynamically
                generation_source = self._detect_generation_source(original_query)
                if generation_source:
                    logger.info(
                        f"Detected generation source for generation query: {generation_source}"
                    )
                    where_clause = self._add_source_filtering(
                        where_clause, generation_source
                    )

                sql = template.format(where_clause=where_clause)
                logger.info(f"Generated generation SQL: {sql[:100]}...")
                return sql

            # Special handling for region generation queries - they need source filtering
            if template_key == "region_generation_query":
                # Build where clause based on extracted entities and user mappings
                where_clause = self._build_where_clause_from_entities(
                    analysis, context.user_mappings
                )

                # Detect and add source filtering dynamically
                generation_source = self._detect_generation_source(original_query)
                if generation_source:
                    logger.info(
                        f"Detected generation source for region query: {generation_source}"
                    )
                    where_clause = self._add_source_filtering(
                        where_clause, generation_source
                    )

                sql = template.format(where_clause=where_clause)
                logger.info(f"Generated region generation SQL: {sql[:100]}...")
                return sql

            # Special handling for international transmission queries - they don't need energy_column processing
            if template_key == "international_transmission_query":
                # Build where clause based on extracted entities and user mappings
                where_clause = self._build_where_clause_from_entities(
                    analysis, context.user_mappings
                )

                # Determine aggregation function and energy column for international transmission
                aggregation_function = "SUM"  # Default
                if (
                    "maximum" in analysis.detected_keywords
                    or "max" in analysis.detected_keywords
                ):
                    aggregation_function = "MAX"
                elif (
                    "minimum" in analysis.detected_keywords
                    or "min" in analysis.detected_keywords
                ):
                    aggregation_function = "MIN"
                elif (
                    "average" in analysis.detected_keywords
                    or "avg" in analysis.detected_keywords
                ):
                    aggregation_function = "AVG"

                energy_column = (
                    "MaxLoading"
                    if "loading" in analysis.detected_keywords
                    else "MaxLoading"
                )
                column_alias = (
                    f"Maximum{energy_column}"
                    if aggregation_function == "MAX"
                    else f"Total{energy_column}"
                )

                sql = template.format(
                    aggregation_function=aggregation_function,
                    energy_column=energy_column,
                    column_alias=column_alias,
                    where_clause=where_clause,
                )
                logger.info(f"Generated international transmission SQL: {sql[:100]}...")
                return sql

            # Special handling for regular transmission queries - they don't need energy_column processing
            if template_key == "transmission_query":
                # Build where clause based on extracted entities and user mappings
                where_clause = self._build_where_clause_from_entities(
                    analysis, context.user_mappings
                )

                # Determine aggregation function and energy column for transmission
                aggregation_function = "SUM"  # Default
                if (
                    "maximum" in analysis.detected_keywords
                    or "max" in analysis.detected_keywords
                ):
                    aggregation_function = "MAX"
                elif (
                    "minimum" in analysis.detected_keywords
                    or "min" in analysis.detected_keywords
                ):
                    aggregation_function = "MIN"
                elif (
                    "average" in analysis.detected_keywords
                    or "avg" in analysis.detected_keywords
                ):
                    aggregation_function = "AVG"

                energy_column = (
                    "MaxImport"
                    if "import" in analysis.detected_keywords
                    else (
                        "MaxExport"
                        if "export" in analysis.detected_keywords
                        else "MaxImport"
                    )
                )
                column_alias = (
                    f"Maximum{energy_column}"
                    if aggregation_function == "MAX"
                    else f"Total{energy_column}"
                )

                sql = template.format(
                    aggregation_function=aggregation_function,
                    energy_column=energy_column,
                    column_alias=column_alias,
                    where_clause=where_clause,
                )
                logger.info(f"Generated transmission SQL: {sql[:100]}...")
                return sql

            # Special handling for time block queries - they don't need energy_column processing
            if template_key == "time_block_query":
                # Build where clause based on extracted entities and user mappings
                where_clause = self._build_where_clause_from_entities(
                    analysis, context.user_mappings
                )

                # Determine aggregation function and energy column for time block
                aggregation_function = (
                    "MAX"
                    if "maximum" in analysis.detected_keywords
                    or "max" in analysis.detected_keywords
                    else "SUM"
                )
                energy_column = (
                    "TotalGeneration"
                    if "generation" in analysis.detected_keywords
                    else "DemandMet"
                )
                column_alias = (
                    f"Maximum{energy_column}"
                    if aggregation_function == "MAX"
                    else f"Total{energy_column}"
                )

                sql = template.format(
                    aggregation_function=aggregation_function,
                    energy_column=energy_column,
                    column_alias=column_alias,
                    where_clause=where_clause,
                )
                logger.info(f"Generated time block SQL: {sql[:100]}...")
                return sql

            # Special handling for country daily exchange queries - they don't need energy_column processing
            if template_key == "country_daily_exchange_query":
                # Build where clause based on extracted entities and user mappings
                where_clause = self._build_where_clause_from_entities(
                    analysis, context.user_mappings
                )

                # Determine aggregation function and energy column for country daily exchange
                aggregation_function = (
                    "SUM"
                    if "total" in analysis.detected_keywords
                    else (
                        "MAX"
                        if "maximum" in analysis.detected_keywords
                        or "peak" in analysis.detected_keywords
                        else "SUM"
                    )
                )
                energy_column = (
                    "TotalEnergyExchanged"
                    if "energy" in analysis.detected_keywords
                    else (
                        "PeakExchange"
                        if "peak" in analysis.detected_keywords
                        else "TotalEnergyExchanged"
                    )
                )
                column_alias = (
                    f"Total{energy_column}"
                    if aggregation_function == "SUM"
                    else f"Maximum{energy_column}"
                )

                sql = template.format(
                    aggregation_function=aggregation_function,
                    energy_column=energy_column,
                    column_alias=column_alias,
                    where_clause=where_clause,
                )
                logger.info(f"Generated country daily exchange SQL: {sql[:100]}...")
                return sql

            # Dynamic template processing for trend analysis and other queries
            if template_key in ["region_query", "state_query"]:
                # Build where clause based on extracted entities and user mappings
                where_clause = self._build_where_clause_from_entities(
                    analysis, context.user_mappings
                )

                # Determine aggregation function dynamically
                aggregation_function = self._determine_aggregation_function(
                    original_query, analysis
                )

                # Get energy column from schema linker
                if hasattr(context, "schema_linker") and context.schema_linker:
                    table_name = (
                        "FactAllIndiaDailySummary"
                        if analysis.query_type.value == "region"
                        else "FactStateDailyEnergy"
                    )
                    energy_column = (
                        context.schema_linker.get_best_column_match(
                            user_query=original_query,
                            table_name=table_name,
                            query_type="energy",
                        )
                        or "EnergyMet"
                    )
                else:
                    energy_column = "EnergyMet"

                # Generate column alias
                column_alias = self._generate_column_alias(
                    original_query, energy_column, aggregation_function
                )

                # Check if this is a trend analysis query that needs time-based grouping
                if analysis.intent.value == "trend_analysis":
                    # Determine if it's growth or aggregation
                    original_query_lower = (
                        original_query.lower() if original_query else ""
                    )
                    is_growth_query = any(
                        word in original_query_lower
                        for word in [
                            "growth",
                            "increase",
                            "decrease",
                            "change",
                            "trend",
                        ]
                    )
                    is_aggregation_query = any(
                        word in original_query_lower
                        for word in [
                            "total",
                            "sum",
                            "average",
                            "avg",
                            "maximum",
                            "max",
                            "minimum",
                            "min",
                            "aggregate",
                        ]
                    )

                    time_period = self._determine_time_period(original_query)
                    logger.info(
                        f"Trend analysis detected - Growth: {is_growth_query}, Aggregation: {is_aggregation_query}, Time period: {time_period}, Aggregation: {aggregation_function}"
                    )

                    if is_growth_query:
                        # For growth queries, use the growth template (handled by template selection above)
                        # This should have been selected as {time_period}_growth_query
                        logger.info(f"Growth query - using growth template")
                        return None  # Let the growth template handle it
                    elif is_aggregation_query:
                        # For aggregation queries, use time-based grouping
                        logger.info(f"Aggregation query - using time-based grouping")

                        # Get table configuration from schema linker
                        if hasattr(context, "schema_linker") and context.schema_linker:
                            table_name = (
                                "FactAllIndiaDailySummary"
                                if analysis.query_type.value == "region"
                                else "FactStateDailyEnergy"
                            )
                            table_config = context.schema_linker.get_table_config(
                                table_name, original_query
                            )
                        else:
                            # Fallback table configuration
                            table_config = {
                                "name_column": (
                                    "RegionName"
                                    if analysis.query_type.value == "region"
                                    else "StateName"
                                ),
                                "join_clause": (
                                    "JOIN DimRegions d ON f.RegionID = d.RegionID"
                                    if analysis.query_type.value == "region"
                                    else "JOIN DimStates d ON f.StateID = d.StateID"
                                ),
                            }

                        # Use time-based grouping for aggregation trend analysis
                        if time_period == "monthly":
                            sql = f"""
                                SELECT d.{table_config['name_column']}, dt.Month, ROUND({aggregation_function}(f.{energy_column}), 2) as {column_alias}
                                FROM {table_name} f
                                {table_config['join_clause']}
                                JOIN DimDates dt ON f.DateID = dt.DateID
                                {where_clause}
                                GROUP BY d.{table_config['name_column']}, dt.Month
                                ORDER BY d.{table_config['name_column']}, dt.Month
                            """
                        elif time_period == "quarterly":
                            sql = f"""
                                SELECT d.{table_config['name_column']}, dt.Quarter, ROUND({aggregation_function}(f.{energy_column}), 2) as {column_alias}
                                FROM {table_name} f
                                {table_config['join_clause']}
                                JOIN DimDates dt ON f.DateID = dt.DateID
                                {where_clause}
                                GROUP BY d.{table_config['name_column']}, dt.Quarter
                                ORDER BY d.{table_config['name_column']}, dt.Quarter
                            """
                        elif time_period == "weekly":
                            sql = f"""
                                SELECT d.{table_config['name_column']}, dt.Week, ROUND({aggregation_function}(f.{energy_column}), 2) as {column_alias}
                                FROM {table_name} f
                                {table_config['join_clause']}
                                JOIN DimDates dt ON f.DateID = dt.DateID
                                {where_clause}
                                GROUP BY d.{table_config['name_column']}, dt.Week
                                ORDER BY d.{table_config['name_column']}, dt.Week
                            """
                        elif time_period == "daily":
                            sql = f"""
                                SELECT d.{table_config['name_column']}, dt.DayOfMonth, ROUND({aggregation_function}(f.{energy_column}), 2) as {column_alias}
                                FROM {table_name} f
                                {table_config['join_clause']}
                                JOIN DimDates dt ON f.DateID = dt.DateID
                                {where_clause}
                                GROUP BY d.{table_config['name_column']}, dt.DayOfMonth
                                ORDER BY d.{table_config['name_column']}, dt.DayOfMonth
                            """
                        elif time_period == "yearly":
                            sql = f"""
                                SELECT d.{table_config['name_column']}, dt.Year, ROUND({aggregation_function}(f.{energy_column}), 2) as {column_alias}
                                FROM {table_name} f
                                {table_config['join_clause']}
                                JOIN DimDates dt ON f.DateID = dt.DateID
                                {where_clause}
                                GROUP BY d.{table_config['name_column']}, dt.Year
                                ORDER BY d.{table_config['name_column']}, dt.Year
                            """
                        else:  # time_period == "none" - no time-based grouping
                            sql = f"""
                                SELECT d.{table_config['name_column']}, ROUND({aggregation_function}(f.{energy_column}), 2) as {column_alias}
                                FROM {table_name} f
                                {table_config['join_clause']}
                                JOIN DimDates dt ON f.DateID = dt.DateID
                                {where_clause}
                                GROUP BY d.{table_config['name_column']}
                                ORDER BY {column_alias} DESC
                            """

                        logger.info(
                            f"Generated aggregation trend analysis SQL: {sql[:100]}..."
                        )
                        return sql
                    else:
                        # Default to regular template processing
                        logger.info(f"Default trend analysis - using regular template")
                        sql = template.format(
                            aggregation_function=aggregation_function,
                            energy_column=energy_column,
                            column_alias=column_alias,
                            where_clause=where_clause,
                        )
                        logger.info(f"Generated default SQL: {sql[:100]}...")
                        return sql
                else:
                    # Regular template processing
                    sql = template.format(
                        aggregation_function=aggregation_function,
                        energy_column=energy_column,
                        column_alias=column_alias,
                        where_clause=where_clause,
                    )
                    logger.info(f"Generated regular SQL: {sql[:100]}...")
                    return sql

            # Special handling for time block generation queries - they don't need energy_column processing
            if template_key == "time_block_generation_query":
                where_clause = self._build_where_clause_from_entities(
                    analysis, context.user_mappings
                )
                sql = template.format(where_clause=where_clause)
                logger.info(f"Generated time block generation SQL: {sql[:100]}...")
                return sql

            # Special handling for dynamic growth queries - they need dynamic table and column resolution
            if template_key and template_key.endswith("_dynamic_growth_query"):
                sql = self._process_dynamic_growth_template(
                    template, analysis, context, original_query
                )
                if sql:
                    logger.info(f"Generated dynamic growth SQL: {sql[:100]}...")
                    return sql
                else:
                    logger.error("Failed to generate dynamic growth SQL")
                    return None

            # Special handling for state queries - they need energy_column processing
            if template_key == "state_query":
                # Build where clause based on extracted entities and user mappings
                where_clause = self._build_where_clause_from_entities(
                    analysis, context.user_mappings
                )

                # Determine aggregation function based on query intent
                aggregation_function = "SUM"
                if any(
                    word in analysis.detected_keywords
                    for word in ["maximum", "max", "highest"]
                ):
                    aggregation_function = "MAX"
                elif any(
                    word in analysis.detected_keywords
                    for word in ["minimum", "min", "lowest"]
                ):
                    aggregation_function = "MIN"
                elif any(
                    word in analysis.detected_keywords
                    for word in ["average", "avg", "mean"]
                ):
                    aggregation_function = "AVG"

                # Additional check for aggregation keywords in the original query
                original_query_lower = original_query.lower() if original_query else ""
                if any(
                    word in original_query_lower
                    for word in ["maximum", "max", "highest"]
                ):
                    aggregation_function = "MAX"
                elif any(
                    word in original_query_lower
                    for word in ["minimum", "min", "lowest"]
                ):
                    aggregation_function = "MIN"
                elif any(
                    word in original_query_lower for word in ["average", "avg", "mean"]
                ):
                    aggregation_function = "AVG"
                elif any(word in original_query_lower for word in ["total", "sum"]):
                    aggregation_function = "SUM"

                # Check if monthly grouping is needed
                original_query_lower = original_query.lower() if original_query else ""
                is_monthly_query = any(
                    word in original_query_lower
                    for word in ["monthly", "month", "by month"]
                )

                # Use schema linker to get the correct column
                if hasattr(context, "schema_linker") and context.schema_linker:
                    table_name = "FactStateDailyEnergy"
                    energy_column = context.schema_linker.get_best_column_match(
                        user_query=original_query,
                        table_name=table_name,
                        query_type="energy",  # Default for state queries
                    )

                    logger.info(
                        f"🔍 SCHEMA LINKER RESULT - Table: {table_name}, Energy Column: {energy_column}"
                    )

                    # CRITICAL CHANGE: No more fallbacks. If no column is found, fail explicitly.
                    if not energy_column:
                        logger.error(
                            f"No column match found for state query: {original_query}. Clarification required."
                        )
                        return None

                    # Generate informative column alias
                    column_alias = self._generate_column_alias(
                        original_query, energy_column, aggregation_function
                    )

                    # Modify template based on whether monthly grouping is needed
                    if is_monthly_query:
                        # Use monthly grouping template
                        sql = f"""
                            SELECT ds.StateName, dt.Month, ROUND({aggregation_function}(fs.{energy_column}), 2) as {column_alias}
                            FROM FactStateDailyEnergy fs
                            JOIN DimStates ds ON fs.StateID = ds.StateID
                            JOIN DimDates dt ON fs.DateID = dt.DateID
                            {where_clause}
                            GROUP BY ds.StateName, dt.Month
                            ORDER BY ds.StateName, dt.Month
                        """
                        logger.info(f"🔍 GENERATED MONTHLY SQL: {sql[:200]}...")
                    else:
                        # Use regular template
                        sql = template.format(
                            aggregation_function=aggregation_function,
                            energy_column=energy_column,
                            column_alias=column_alias,
                            where_clause=where_clause,
                        )
                        logger.info(f"🔍 GENERATED REGULAR SQL: {sql[:200]}...")

                    logger.info(
                        f"Generated state SQL (monthly={is_monthly_query}): {sql[:100]}..."
                    )
                    return sql
                else:
                    logger.error(
                        f"No schema linker available for state query. Clarification required."
                    )
                    return None

            # Special handling for region queries - they need energy_column processing
            if template_key == "region_query":
                logger.info(
                    f"🔍 REGION QUERY HANDLING STARTED - Query: '{original_query}'"
                )
                logger.info(
                    f"REGION QUERY HANDLING: Processing region query with template_key={template_key}"
                )
                # Build where clause based on extracted entities and user mappings
                where_clause = self._build_where_clause_from_entities(
                    analysis, context.user_mappings
                )

                # Determine aggregation function based on query intent
                aggregation_function = "SUM"
                if any(
                    word in analysis.detected_keywords
                    for word in ["maximum", "max", "highest"]
                ):
                    aggregation_function = "MAX"
                elif any(
                    word in analysis.detected_keywords
                    for word in ["minimum", "min", "lowest"]
                ):
                    aggregation_function = "MIN"
                elif any(
                    word in analysis.detected_keywords
                    for word in ["average", "avg", "mean"]
                ):
                    aggregation_function = "AVG"

                # Additional check for aggregation keywords in the original query
                original_query_lower = original_query.lower() if original_query else ""
                if any(
                    word in original_query_lower
                    for word in ["maximum", "max", "highest"]
                ):
                    aggregation_function = "MAX"
                elif any(
                    word in original_query_lower
                    for word in ["minimum", "min", "lowest"]
                ):
                    aggregation_function = "MIN"
                elif any(
                    word in original_query_lower for word in ["average", "avg", "mean"]
                ):
                    aggregation_function = "AVG"
                elif any(word in original_query_lower for word in ["total", "sum"]):
                    aggregation_function = "SUM"

                # Check if monthly grouping is needed
                original_query_lower = original_query.lower() if original_query else ""
                is_monthly_query = any(
                    word in original_query_lower
                    for word in ["monthly", "month", "by month"]
                )
                logger.info(
                    f"🔍 REGION MONTHLY DETECTION - Query: '{original_query}', Lower: '{original_query_lower}', Is Monthly: {is_monthly_query}"
                )

                # Use schema linker to get the correct column
                if hasattr(context, "schema_linker") and context.schema_linker:
                    table_name = "FactAllIndiaDailySummary"
                    energy_column = context.schema_linker.get_best_column_match(
                        user_query=original_query,
                        table_name=table_name,
                        query_type="energy",  # Default for region queries
                    )

                    logger.info(
                        f"🔍 REGION SCHEMA LINKER RESULT - Table: {table_name}, Energy Column: {energy_column}"
                    )

                    # CRITICAL CHANGE: No more fallbacks. If no column is found, fail explicitly.
                    if not energy_column:
                        logger.error(
                            f"No column match found for region query: {original_query}. Clarification required."
                        )
                        return None

                    # Generate informative column alias
                    column_alias = self._generate_column_alias(
                        original_query, energy_column, aggregation_function
                    )

                    # Modify template based on whether monthly grouping is needed
                    if is_monthly_query:
                        # Use monthly grouping template
                        sql = f"""
                            SELECT dr.RegionName, dt.Month, ROUND({aggregation_function}(f.{energy_column}), 2) as {column_alias}
                            FROM FactAllIndiaDailySummary f
                            JOIN DimRegions dr ON f.RegionID = dr.RegionID
                            JOIN DimDates dt ON f.DateID = dt.DateID
                            {where_clause}
                            GROUP BY dr.RegionName, dt.Month
                            ORDER BY dr.RegionName, dt.Month
                        """
                        logger.info(f"🔍 GENERATED REGION MONTHLY SQL: {sql[:200]}...")
                    else:
                        # Use regular template
                        sql = template.format(
                            aggregation_function=aggregation_function,
                            energy_column=energy_column,
                            column_alias=column_alias,
                            where_clause=where_clause,
                        )
                        logger.info(f"🔍 GENERATED REGION REGULAR SQL: {sql[:200]}...")

                    logger.info(
                        f"Generated region SQL (monthly={is_monthly_query}): {sql[:100]}..."
                    )
                    return sql
                else:
                    logger.error(
                        f"No schema linker available for region query. Clarification required."
                    )
                    return None

            # Determine aggregation function based on query intent
            aggregation_function = "SUM"
            if any(
                word in analysis.detected_keywords
                for word in ["maximum", "max", "highest"]
            ):
                aggregation_function = "MAX"
            elif any(
                word in analysis.detected_keywords
                for word in ["minimum", "min", "lowest"]
            ):
                aggregation_function = "MIN"
            elif any(
                word in analysis.detected_keywords
                for word in ["average", "avg", "mean"]
            ):
                aggregation_function = "AVG"

            # Determine energy column based on query keywords and business rules
            energy_column = ""  # No default - will be set by similarity matching
            query_type = "energy"  # Default

            if "evening" in analysis.detected_keywords:
                query_type = "evening_demand"
            elif (
                "drawal" in analysis.detected_keywords
                or "schedule" in analysis.detected_keywords
            ):
                query_type = "drawal_schedule"
            elif (
                "actual" in analysis.detected_keywords
                and "drawal" in analysis.detected_keywords
            ):
                query_type = "actual_drawal"
            elif "shortage" in analysis.detected_keywords:
                query_type = "shortage"
            elif (
                "transmission" in analysis.detected_keywords
                or "flow" in analysis.detected_keywords
            ):
                query_type = "transmission"
            elif (
                "import" in analysis.detected_keywords
                or "export" in analysis.detected_keywords
            ):
                query_type = "transmission"  # Import/export are transmission operations
            elif (
                "exchange" in analysis.detected_keywords
                or "trade" in analysis.detected_keywords
            ):
                query_type = "exchange"
            elif "outage" in analysis.detected_keywords:
                query_type = "outage"
            elif (
                "ratio" in analysis.detected_keywords
                or "share" in analysis.detected_keywords
            ):
                query_type = "ratio"
            elif "frequency" in analysis.detected_keywords:
                query_type = "percentage"  # Frequency durations are percentages
            elif (
                "percentage" in analysis.detected_keywords
                or "%" in analysis.detected_keywords
            ):
                query_type = "percentage"
            elif (
                "time" in analysis.detected_keywords
                or "duration" in analysis.detected_keywords
            ):
                query_type = "time"
            elif (
                "power" in analysis.detected_keywords
                or "mw" in analysis.detected_keywords
            ):
                query_type = "power"
            elif "demand" in analysis.detected_keywords:
                query_type = "demand"
            elif "energy" in analysis.detected_keywords:
                query_type = "energy"

            # Use business rules to get correct column
            if hasattr(context, "schema_linker") and context.schema_linker:
                logger.info(
                    f"Using similarity-based column matching for query: '{original_query}'"
                )
                # Determine the correct table based on query type
                table_name = "FactAllIndiaDailySummary"  # Default for region queries
                if analysis.query_type.value == "state":
                    table_name = "FactStateDailyEnergy"
                elif analysis.query_type.value == "generation":
                    table_name = "FactDailyGenerationBreakdown"
                elif analysis.query_type.value == "transmission":
                    table_name = "FactTransmissionLinkFlow"
                elif analysis.query_type.value == "exchange":
                    table_name = "FactTransnationalExchangeDetail"
                elif analysis.query_type.value == "country_daily_exchange":
                    table_name = "FactCountryDailyExchange"

                logger.info(f"Selected table: {table_name}, query_type: {query_type}")

                # Use similarity-based column matching with fallback to business rules
                energy_column = context.schema_linker.get_best_column_match(
                    user_query=original_query,  # Pass the original query for similarity matching
                    table_name=table_name,
                    query_type=query_type,
                    llm_provider=getattr(
                        context, "llm_provider", None
                    ),  # Pass LLM provider for Hook #2
                )

                logger.info(f"Similarity-based column match result: {energy_column}")

                # If no column match found, return None to trigger LLM fallback
                if not energy_column:
                    # === LLM Hook #3 – "Template Slot Filler" ===
                    if self.llm_provider:
                        try:
                            table_name = analysis.main_table
                            candidate_columns = (
                                context.schema_info.tables.get(table_name, [])
                                if context.schema_info and context.schema_info.tables
                                else []
                            )
                            prompt = f"Given the natural language query '{original_query}', choose the correct column for the {{energy_column}} slot in the SQL template for the table '{table_name}' from this list: {candidate_columns}."

                            # For now, we'll use a simple approach without async to avoid complexity
                            # In a production system, this would be properly async
                            logger.info(
                                f"LLM Hook #3: Attempting slot filling for query: {original_query}"
                            )
                            # Note: This is a simplified version - in production, you'd want proper async handling
                            return None  # Skip for now to avoid async issues
                        except Exception as e:
                            logger.warning(f"LLM Hook #3 (Slot Filler) failed: {e}")
                            return None  # Fail template generation if LLM fails
                    else:
                        logger.warning(
                            "Template slot 'energy_column' could not be resolved."
                        )
                        return None

                # Generate informative column alias
                column_alias = self._generate_column_alias(
                    original_query, energy_column, aggregation_function
                )

                # Get data type and appropriate aggregation functions
                data_type = context.schema_linker._get_column_data_type(
                    table_name, energy_column
                )
                appropriate_functions = (
                    context.schema_linker._get_appropriate_aggregation_functions(
                        data_type
                    )
                )

                # If no appropriate functions found, return None
                if not appropriate_functions:
                    logger.warning(
                        f"No appropriate aggregation functions found for column {energy_column}"
                    )
                    return None

                # Use the first appropriate function or the detected one
                if aggregation_function not in appropriate_functions:
                    aggregation_function = appropriate_functions[0]

                logger.info(
                    f"Final column: {energy_column}, aggregation: {aggregation_function}, alias: {column_alias}"
                )

                # Build where clause based on extracted entities and user mappings
                where_clause = self._build_where_clause_from_entities(
                    analysis, context.user_mappings
                )

                # Generate informative column alias
                column_alias = self._generate_column_alias(
                    original_query, energy_column, aggregation_function
                )

                sql = template.format(
                    where_clause=where_clause,
                    aggregation_function=aggregation_function,
                    energy_column=energy_column,
                    column_alias=column_alias,
                )
                logger.info(
                    f"Generated regular SQL with {aggregation_function} aggregation and {energy_column} column: {sql[:100]}..."
                )
                return sql
            else:
                logger.info(f"Using fallback logic - schema_linker not available")
                # No fallback logic - return None if no schema linker available
                logger.warning(f"No schema linker available, cannot determine column")
                return None

        logger.warning(f"No template found for key: {fallback_template_key}")
        return None

    def _generate_with_llm(self, prompt: str) -> Optional[str]:
        """
        Generate SQL using LLM (placeholder for future implementation).
        """
        # This would integrate with OpenAI or other LLM providers
        # For now, return None to use templates
        return None

    def _build_where_clause(self, user_mappings: List[Dict[str, Any]]) -> str:
        """
        Build WHERE clause based on user mappings.
        """
        if not user_mappings:
            return ""

        conditions = []
        for mapping in user_mappings:
            table_name = mapping["dimension_table"]
            column_name = self._get_name_column(table_name)
            value = mapping["dimension_value"]

            condition = f"{table_name}.{column_name} = '{value}'"
            conditions.append(condition)

        if conditions:
            return f"WHERE {' OR '.join(conditions)}"

        return ""

    def _build_where_clause_from_entities(
        self, analysis: QueryAnalysis, user_mappings: List[UserMapping]
    ) -> str:
        """
        Build WHERE clause based on extracted entities and user mappings.
        """
        conditions = []

        # Check if this is an "all states" or "all regions" query
        original_query = analysis.original_query or ""
        query_lower = original_query.lower() if original_query else ""
        is_all_states_query = any(
            phrase in query_lower
            for phrase in ["all states", "all state", "every state", "each state"]
        )
        is_all_regions_query = any(
            phrase in query_lower
            for phrase in ["all regions", "all region", "every region", "each region"]
        )

        logger.info(
            f"🔍 ALL QUERY DETECTION - Query: '{original_query}', All States: {is_all_states_query}, All Regions: {is_all_regions_query}"
        )

        # Add conditions from extracted entities (but skip if it's an "all" query)
        if analysis.entities and not (is_all_states_query or is_all_regions_query):
            logger.info(f"Extracted entities: {analysis.entities}")
            for entity in analysis.entities:
                if analysis.query_type.value == "region":
                    # Map region entity via centralized entity loader (supports aliases like "north")
                    entity_lower = entity.lower()
                    mapped_region = self.entity_loader.get_proper_region_name(entity_lower)

                    if mapped_region:
                        conditions.append(f"d.RegionName = '{mapped_region}'")
                        logger.info(
                            f"Added region condition with mapping: d.RegionName = '{mapped_region}' (mapped from '{entity}')"
                        )
                    else:
                        # Use the entity as-is if no mapping found
                        conditions.append(f"d.RegionName = '{entity}'")
                        logger.info(
                            f"Added region condition as-is: d.RegionName = '{entity}' (no mapping found)"
                        )
                elif analysis.query_type.value == "state":
                    # Only add WHERE conditions for entities that are actually valid state names
                    entity_lower = entity.lower()
                    
                    # Check if this entity is actually a state name
                    if self.entity_loader.is_indian_state(entity):
                        mapped_state = self.entity_loader.get_proper_state_name(entity_lower)
                        conditions.append(f"d.StateName = '{mapped_state}'")
                        logger.info(
                            f"Added state condition: d.StateName = '{mapped_state}' (mapped from '{entity}')"
                        )
                    else:
                        # Skip non-state entities (like 'energy', regions, etc.)
                        logger.info(
                            f"Skipping non-state entity: '{entity}' in state query"
                        )
                elif analysis.query_type.value == "exchange":
                    # Handle exchange entities with proper mapping
                    if entity.lower() in ["nepal", "bhutan", "bangladesh", "myanmar"]:
                        # Map country names to correct database values
                        country_mapping = {
                            "nepal": "Nepal",
                            "bhutan": "Bhutan",
                            "bangladesh": "Bangladesh",
                            "myanmar": "Myanmar",
                        }
                        mapped_country = country_mapping.get(
                            entity.lower(), entity.title()
                        )
                        conditions.append(f"dc.CountryName = '{mapped_country}'")
                        logger.info(
                            f"Added country condition with mapping: dc.CountryName = '{mapped_country}' (mapped from '{entity}')"
                        )
                    elif entity.upper() in [
                        "DAM IEX",
                        "DAM PXIL",
                        "DAM HPX",
                        "RTM IEX",
                        "RTM PXIL",
                        "RTM HPX",
                        "BILATERAL",
                        "PPA",
                    ]:
                        # Exchange mechanisms are already properly mapped in entity extraction
                        conditions.append(f"dem.MechanismName = '{entity}'")
                        logger.info(
                            f"Added mechanism condition: dem.MechanismName = '{entity}'"
                        )
                    elif entity.lower() in ["import", "export"]:
                        # Map exchange directions to correct database values
                        direction_mapping = {"import": "Import", "export": "Export"}
                        mapped_direction = direction_mapping.get(
                            entity.lower(), entity.title()
                        )
                        conditions.append(
                            f"fted.ExchangeDirection = '{mapped_direction}'"
                        )
                        logger.info(
                            f"Added direction condition with mapping: fted.ExchangeDirection = '{mapped_direction}' (mapped from '{entity}')"
                        )
                elif analysis.query_type.value == "country_daily_exchange":
                    # Handle country daily exchange entities with proper mapping
                    if entity.lower() in ["nepal", "bhutan", "bangladesh", "myanmar"]:
                        # Map country names to correct database values
                        country_mapping = {
                            "nepal": "Nepal",
                            "bhutan": "Bhutan",
                            "bangladesh": "Bangladesh",
                            "myanmar": "Myanmar",
                        }
                        mapped_country = country_mapping.get(
                            entity.lower(), entity.title()
                        )
                        conditions.append(f"dc.CountryName = '{mapped_country}'")
                        logger.info(
                            f"Added country condition with mapping: dc.CountryName = '{mapped_country}' (mapped from '{entity}')"
                        )
        else:
            if is_all_states_query or is_all_regions_query:
                logger.info(f"Skipping entity conditions for 'all' query")
            else:
                logger.warning(f"No entities extracted from query analysis")
            # Manual fallback: check for region names in the original query
            # This is a temporary fix until entity extraction is fully working
            if analysis.original_query:
                query_lower = analysis.original_query.lower()
                if "northern region" in query_lower:
                    conditions.append("d.RegionName = 'Northern Region'")
                    logger.info(
                        "Added manual region condition: d.RegionName = 'Northern Region'"
                    )
                elif "southern region" in query_lower:
                    conditions.append("d.RegionName = 'Southern Region'")
                    logger.info(
                        "Added manual region condition: d.RegionName = 'Southern Region'"
                    )
                elif "western region" in query_lower:
                    conditions.append("d.RegionName = 'Western Region'")
                    logger.info(
                        "Added manual region condition: d.RegionName = 'Western Region'"
                    )
                elif "eastern region" in query_lower:
                    conditions.append("d.RegionName = 'Eastern Region'")
                    logger.info(
                        "Added manual region condition: d.RegionName = 'Eastern Region'"
                    )
                elif "north eastern region" in query_lower:
                    conditions.append("d.RegionName = 'North Eastern Region'")
                    logger.info(
                        "Added manual region condition: d.RegionName = 'North Eastern Region'"
                    )
                # === SPRINT 3: REMOVED HARDCODED MANUAL STATE CONDITIONS ===
                # Manual fallback for state names - now using centralized entity loader
                # The entity extraction should handle state names properly

        # Add date conditions from time period analysis
        if analysis.time_period:
            logger.info(f"Time period analysis: {analysis.time_period}")
            time_period = analysis.time_period

            if time_period.get("type") == "specific_date":
                year = time_period.get("year")
                month = time_period.get("month")
                day = time_period.get("day")

                if year:
                    conditions.append(f"dt.Year = {year}")
                    logger.info(f"Added year condition: dt.Year = {year}")

                if month:
                    conditions.append(f"dt.Month = {month}")
                    logger.info(f"Added month condition: dt.Month = {month}")

                if day:
                    conditions.append(f"dt.DayOfMonth = {day}")
                    logger.info(f"Added day condition: dt.DayOfMonth = {day}")

            elif time_period.get("type") == "date_range":
                start_year = time_period.get("start_year")
                start_month = time_period.get("start_month")
                end_year = time_period.get("end_year")
                end_month = time_period.get("end_month")

                if start_year and end_year:
                    if start_year == end_year:
                        conditions.append(f"dt.Year = {start_year}")
                        logger.info(f"Added year condition: dt.Year = {start_year}")

                        if start_month and end_month:
                            conditions.append(
                                f"dt.Month BETWEEN {start_month} AND {end_month}"
                            )
                            logger.info(
                                f"Added month range condition: dt.Month BETWEEN {start_month} AND {end_month}"
                            )
                    else:
                        conditions.append(
                            f"(dt.Year > {start_year} OR (dt.Year = {start_year} AND dt.Month >= {start_month or 1}))"
                        )
                        conditions.append(
                            f"(dt.Year < {end_year} OR (dt.Year = {end_year} AND dt.Month <= {end_month or 12}))"
                        )
                        logger.info(
                            f"Added date range conditions for {start_year}-{start_month} to {end_year}-{end_month}"
                        )

        # Add conditions from user mappings
        for mapping in user_mappings:
            table_name = mapping.entity.table
            column_name = self._get_name_column(table_name)
            value = mapping.entity.name

            condition = f"{table_name}.{column_name} = '{value}'"
            conditions.append(condition)
            logger.info(f"Added user mapping condition: {condition}")

        if conditions:
            where_clause = f"WHERE {' AND '.join(conditions)}"
            logger.info(f"Final WHERE clause: {where_clause}")
            return where_clause

        logger.info("No conditions found, returning empty WHERE clause")
        return ""

    def _get_name_column(self, table_name: str) -> str:
        """
        Get the name column for a dimension table.
        """
        name_columns = {
            "DimRegions": "RegionName",
            "DimStates": "StateName",
            "DimCountries": "CountryName",
            "DimGenerationSources": "SourceName",
            "DimTransmissionLinks": "LinkName",
            "DimTimeBlocks": "TimeBlockName",
        }

        return name_columns.get(table_name, "Name")

    def _format_schema_info(self, schema_info) -> str:
        """
        Format schema information for the prompt.
        """
        if not schema_info:
            return "Schema information not available"

        # Handle both SchemaInfo objects and raw dictionaries
        if hasattr(schema_info, "tables"):
            # It's a SchemaInfo object
            tables_dict = schema_info.tables
        else:
            # It's a raw dictionary
            tables_dict = schema_info

        if not tables_dict:
            return "Schema information not available"

        formatted = []
        for table, columns in tables_dict.items():
            formatted.append(f"{table}: {', '.join(columns)}")

        return "\n".join(formatted)

    def _format_dimension_values(self, dimension_values: Dict[str, List[str]]) -> str:
        """
        Format dimension values for the prompt.
        """
        if not dimension_values:
            return "Dimension values not available"

        formatted = []
        for table, values in dimension_values.items():
            formatted.append(
                f"{table}: {', '.join(values[:10])}"
            )  # Limit to first 10 values

        return "\n".join(formatted)

    def _format_user_mappings(self, user_mappings: List[Dict[str, Any]]) -> str:
        """
        Format user mappings for the prompt.
        """
        if not user_mappings:
            return "No user mappings found"

        formatted = []
        for mapping in user_mappings:
            formatted.append(
                f"'{mapping['user_term']}' -> {mapping['dimension_table']}.{mapping['dimension_value']}"
            )

        return "\n".join(formatted)

    def _post_process_sql(self, sql: str) -> str:
        """
        Post-process generated SQL.
        """
        # Remove extra whitespace
        sql = " ".join(sql.split())

        # Ensure proper termination
        if not sql.rstrip().endswith(";"):
            sql += ";"

        return sql

    def _generate_column_alias(
        self, query: str, column_name: str, aggregation_function: str
    ) -> str:
        """
        Generate an informative column alias based on the query and column being used.

        Args:
            query: The original user query
            column_name: The database column name
            aggregation_function: The aggregation function being used (SUM, MAX, MIN, AVG)

        Returns:
            Informative column alias
        """
        query_lower = query.lower()

        # Map column names to user-friendly terms
        column_mapping = {
            "EnergyMet": "EnergyMet",
            "EnergyShortage": "EnergyShortage",
            "DrawalSchedule": "DrawalSchedule",
            "ActualDrawal": "ActualDrawal",
            "MaximumDemand": "MaximumDemand",
            "EveningPeakDemand": "EveningPeakDemand",
            "MaxDemandSCADA": "MaximumDemand",
            "EveningPeakDemandMet": "EveningPeakDemand",
            "ScheduleDrawal": "ScheduleDrawal",
            "GenerationAmount": "GenerationAmount",
        }

        # Get the user-friendly column name
        friendly_column = column_mapping.get(column_name, column_name)

        # Map aggregation functions to user-friendly terms
        agg_mapping = {
            "SUM": "Total",
            "MAX": "Maximum",
            "MIN": "Minimum",
            "AVG": "Average",
        }

        # Get the user-friendly aggregation term
        agg_term = agg_mapping.get(aggregation_function, aggregation_function)

        # Combine aggregation term with column name
        alias = f"{agg_term}{friendly_column}"

        # Special cases for common query patterns
        if "total" in query_lower and aggregation_function == "SUM":
            alias = f"Total{friendly_column}"
        elif "maximum" in query_lower or "max" in query_lower:
            alias = f"Maximum{friendly_column}"
        elif "minimum" in query_lower or "min" in query_lower:
            alias = f"Minimum{friendly_column}"
        elif "average" in query_lower or "avg" in query_lower:
            alias = f"Average{friendly_column}"

        return alias

    def _determine_aggregation_function(
        self, query: str, analysis: QueryAnalysis
    ) -> str:
        """
        Dynamically determine the aggregation function based on query keywords.
        """
        query_lower = query.lower()
        detected_keywords = (
            analysis.detected_keywords if hasattr(analysis, "detected_keywords") else []
        )

        # Check for aggregation keywords in the query
        if any(word in query_lower for word in ["maximum", "max", "highest", "peak"]):
            return "MAX"
        elif any(word in query_lower for word in ["minimum", "min", "lowest"]):
            return "MIN"
        elif any(word in query_lower for word in ["average", "avg", "mean"]):
            return "AVG"
        elif any(word in query_lower for word in ["total", "sum", "aggregate"]):
            return "SUM"
        else:
            # Check detected keywords
            if any(
                word in detected_keywords
                for word in ["maximum", "max", "highest", "peak"]
            ):
                return "MAX"
            elif any(
                word in detected_keywords for word in ["minimum", "min", "lowest"]
            ):
                return "MIN"
            elif any(word in detected_keywords for word in ["average", "avg", "mean"]):
                return "AVG"
            else:
                return "SUM"  # Default to SUM

    def _determine_time_period(self, query: str) -> str:
        """
        Dynamically determine the time period for aggregation.
        """
        query_lower = query.lower()

        # Check for specific month names first (e.g., "June", "January")
        month_names = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ]
        if any(month in query_lower for month in month_names):
            return "monthly"

        if any(
            word in query_lower
            for word in ["quarterly", "quarter", "q1", "q2", "q3", "q4"]
        ):
            return "quarterly"
        elif any(
            word in query_lower for word in ["yearly", "annual", "year over year"]
        ):
            return "yearly"
        elif any(word in query_lower for word in ["weekly", "week", "per week"]):
            return "weekly"
        elif any(
            word in query_lower for word in ["monthly", "month over month", "per month", "by month"]
        ):
            return "monthly"
        elif any(word in query_lower for word in ["daily", "day", "per day"]):
            return "daily"
        else:
            return "none"  # No time period specified - don't add time-based grouping

    def _generate_growth_column_aliases(
        self, energy_column: str, time_period: str
    ) -> tuple:
        """
        Generate context-aware column aliases for growth queries.
        """
        # Map energy columns to readable names
        column_mapping = {
            "EnergyMet": "EnergyMet",
            "EnergyShortage": "EnergyShortage",
            "EnergyConsumption": "EnergyConsumption",
            "GenerationAmount": "Generation",
            "ShareRESInTotalGeneration": "RESGeneration",
            "CoalGeneration": "CoalGeneration",
            "GasGeneration": "GasGeneration",
            "NuclearGeneration": "NuclearGeneration",
            "SolarGeneration": "SolarGeneration",
            "WindGeneration": "WindGeneration",
            "HydroGeneration": "HydroGeneration",
            "GenerationSourceID": "Generation",  # Handle the ID column case
        }

        # Get the readable name for the energy column
        metric_name = column_mapping.get(energy_column, energy_column)

        # Generate time-specific aliases
        if time_period == "monthly":
            current_alias = f"CurrentMonth{metric_name}"
            previous_alias = f"PreviousMonth{metric_name}"
        elif time_period == "weekly":
            current_alias = f"CurrentWeek{metric_name}"
            previous_alias = f"PreviousWeek{metric_name}"
        elif time_period == "daily":
            current_alias = f"CurrentDay{metric_name}"
            previous_alias = f"PreviousDay{metric_name}"
        elif time_period == "quarterly":
            current_alias = f"CurrentQuarter{metric_name}"
            previous_alias = f"PreviousQuarter{metric_name}"
        elif time_period == "yearly":
            current_alias = f"CurrentYear{metric_name}"
            previous_alias = f"PreviousYear{metric_name}"
        else:
            current_alias = f"Current{metric_name}"
            previous_alias = f"Previous{metric_name}"

        return current_alias, previous_alias

    def _detect_generation_source(self, query: str) -> Optional[str]:
        """
        Detect generation source from user query.
        Returns the source name to filter for, or None if no specific source detected.
        """
        query_lower = query.lower()

        # Define source mappings
        source_mappings = {
            "coal": ["coal", "thermal", "fossil"],
            "solar": ["solar", "photovoltaic", "pv"],
            "wind": ["wind", "wind power"],
            "nuclear": ["nuclear", "atomic"],
            "hydro": ["hydro", "hydropower", "water"],
            "gas": ["gas", "natural gas", "lng"],
            "biomass": ["biomass", "bio"],
            "geothermal": ["geothermal", "geo"],
            "renewable": ["renewable", "clean energy", "green energy"],
            "thermal": ["thermal", "coal", "gas", "fossil"],
        }

        # Check for specific source mentions
        for source_name, keywords in source_mappings.items():
            if any(keyword in query_lower for keyword in keywords):
                return source_name

        # Special handling for thermal vs coal
        if "thermal" in query_lower and "coal" not in query_lower:
            return "thermal"
        elif "coal" in query_lower:
            return "coal"

        return None

    def _process_dynamic_growth_template(
        self,
        template: str,
        analysis: QueryAnalysis,
        context: ContextInfo,
        original_query: str,
    ) -> Optional[str]:
        """
        Process dynamic growth templates by resolving table and column information dynamically.
        """
        try:
            # Get table configuration based on query analysis
            main_table = analysis.main_table
            dimension_table = analysis.dimension_table
            join_key = analysis.join_key
            name_column = analysis.name_column

            logger.info(
                f"Dynamic growth template - Main table: {main_table}, Dimension table: {dimension_table}"
            )

            # Build join clause dynamically
            join_clause = f"JOIN {dimension_table} d ON f.{join_key} = d.{join_key}"

            # Get energy column from schema linker or use default
            energy_column = "EnergyMet"  # Default
            if hasattr(context, "schema_linker") and context.schema_linker:
                energy_column = (
                    context.schema_linker.get_best_column_match(
                        user_query=original_query,
                        table_name=main_table,
                        query_type="energy",
                    )
                    or "EnergyMet"
                )

            # Build where clause
            where_clause = self._build_where_clause_from_entities(
                analysis, context.user_mappings
            )

            # Strip "WHERE" prefix if present (template expects just conditions)
            if where_clause.startswith("WHERE "):
                where_clause = where_clause[6:]  # Remove "WHERE "

            # Build previous where clause for growth comparison
            previous_where_clause = self._build_previous_period_where_clause(
                analysis, context.user_mappings
            )

            # Strip "WHERE" prefix if present
            if previous_where_clause.startswith("WHERE "):
                previous_where_clause = previous_where_clause[6:]  # Remove "WHERE "

            # Generate column aliases
            current_alias, previous_alias = self._generate_growth_column_aliases(
                energy_column, "none"  # Default to no time-based grouping unless explicitly mentioned
            )

            # Format the template with dynamic values
            sql = template.format(
                main_table=main_table,
                join_clause=join_clause,
                name_column=name_column,
                energy_column=energy_column,
                where_clause=where_clause,
                previous_where_clause=previous_where_clause,
                current_alias=current_alias,
                previous_alias=previous_alias,
            )

            logger.info(
                f"Generated dynamic growth SQL with table: {main_table}, column: {energy_column}"
            )
            return sql

        except Exception as e:
            logger.error(f"Error processing dynamic growth template: {str(e)}")
            return None

    def _build_previous_period_where_clause(
        self, analysis: QueryAnalysis, user_mappings: List[UserMapping]
    ) -> str:
        """
        Build WHERE clause for the previous period in growth queries.
        """
        conditions = []

        # Add entity conditions (same as current period)
        for mapping in user_mappings:
            table_name = mapping.entity.table
            column_name = self._get_name_column(table_name)
            value = mapping.entity.name
            conditions.append(f"{table_name}.{column_name} = '{value}'")

        # Add time period conditions for previous period
        if analysis.time_period:
            time_period = analysis.time_period
            if time_period.get("type") == "specific_date":
                year = time_period.get("year")
                if year:
                    # Use same year for monthly growth (previous month within same year)
                    conditions.append(f"dt.Year = {year}")

        if conditions:
            return f"WHERE {' AND '.join(conditions)}"

        return ""

    def _add_source_filtering(self, where_clause: str, generation_source: str) -> str:
        """
        Add source filtering to WHERE clause for generation queries.
        """
        if not generation_source:
            return where_clause

        # Define source name mappings for database filtering
        source_filters = {
            "coal": "gs.SourceName LIKE '%coal%' OR gs.SourceName LIKE '%thermal%'",
            "solar": "gs.SourceName LIKE '%solar%' OR gs.SourceName LIKE '%photovoltaic%'",
            "wind": "gs.SourceName LIKE '%wind%'",
            "nuclear": "gs.SourceName LIKE '%nuclear%' OR gs.SourceName LIKE '%atomic%'",
            "hydro": "gs.SourceName LIKE '%hydro%' OR gs.SourceName LIKE '%water%'",
            "gas": "gs.SourceName LIKE '%gas%' OR gs.SourceName LIKE '%lng%'",
            "biomass": "gs.SourceName LIKE '%biomass%' OR gs.SourceName LIKE '%bio%'",
            "geothermal": "gs.SourceName LIKE '%geothermal%'",
            "renewable": "gs.SourceName NOT LIKE '%coal%' AND gs.SourceName NOT LIKE '%thermal%' AND gs.SourceName NOT LIKE '%gas%'",
            "thermal": "gs.SourceName LIKE '%coal%' OR gs.SourceName LIKE '%thermal%' OR gs.SourceName LIKE '%gas%'",
        }

        source_filter = source_filters.get(generation_source.lower())
        if source_filter:
            if where_clause:
                return f"{where_clause} AND ({source_filter})"
            else:
                return f"WHERE {source_filter}"

        return where_clause

    def _generate_fallback_sql(self, query: str) -> str:
        """Generate basic SQL using keywords when templates don't match"""
        query_lower = query.lower()
        
        # Determine energy column based on query keywords
        energy_column = self._determine_energy_column(query_lower)
        
        # Check for growth queries
        if any(word in query_lower for word in ['growth', 'monthly growth', 'increase']):
            return self._generate_growth_fallback_sql(query_lower, energy_column)
        
        # Check for state queries
        if any(word in query_lower for word in ['state', 'states']):
            return self._generate_state_fallback_sql(query_lower, energy_column)
        
        # Check for region queries
        if any(word in query_lower for word in ['region', 'regions']):
            return self._generate_region_fallback_sql(query_lower, energy_column)
        
        # Default query
        return f"""
        SELECT 
            SUM({energy_column}) as TotalEnergy
        FROM FactAllIndiaDailySummary f
        JOIN DimDates d ON f.DateID = d.DateID
        WHERE strftime('%Y', d.Date) = '2024'
        """
    
    def _generate_growth_fallback_sql(self, query_lower: str, energy_column: str) -> str:
        """Generate SQLite-friendly growth SQL using self-joins"""
        return f"""
        SELECT 
            r.RegionName,
            strftime('%Y-%m', d.Date) as Month,
            SUM(fs.{energy_column}) as TotalEnergy,
            prev.PreviousMonthEnergy,
            CASE 
                WHEN prev.PreviousMonthEnergy > 0 
                THEN ((SUM(fs.{energy_column}) - prev.PreviousMonthEnergy) / prev.PreviousMonthEnergy) * 100 
                ELSE 0 
            END as GrowthRate
        FROM FactAllIndiaDailySummary fs
        JOIN DimRegions r ON fs.RegionID = r.RegionID
        JOIN DimDates d ON fs.DateID = d.DateID
        LEFT JOIN (
            SELECT 
                r2.RegionName,
                strftime('%Y-%m', d2.Date) as Month,
                SUM(fs2.{energy_column}) as PreviousMonthEnergy
            FROM FactAllIndiaDailySummary fs2
            JOIN DimRegions r2 ON fs2.RegionID = r2.RegionID
            JOIN DimDates d2 ON fs2.DateID = d2.DateID
            WHERE strftime('%Y', d2.Date) = '2024'
            GROUP BY r2.RegionName, strftime('%Y-%m', d2.Date)
        ) prev ON r.RegionName = prev.RegionName 
            AND strftime('%Y-%m', d.Date) = date(prev.Month || '-01', '+1 month')
        WHERE strftime('%Y', d.Date) = '2024'
        GROUP BY r.RegionName, strftime('%Y-%m', d.Date)
        ORDER BY r.RegionName, Month
        """
