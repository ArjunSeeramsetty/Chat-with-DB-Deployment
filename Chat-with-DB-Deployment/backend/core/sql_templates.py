"""
Constrained SQL Generation with Templates

This module implements query-type specific templates with strict validation rules
to improve SQL generation accuracy and enforce business rules.

Phase 4.1: Constrained SQL Generation with Templates
Expected Impact: +15-20% accuracy improvement
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query types for template selection"""
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    TIME_SERIES = "time_series"
    COMPARISON = "comparison"
    TREND_ANALYSIS = "trend_analysis"
    REGIONAL_ANALYSIS = "regional_analysis"
    GENERATION_ANALYSIS = "generation_analysis"
    SHORTAGE_ANALYSIS = "shortage_analysis"
    DEMAND_ANALYSIS = "demand_analysis"
    EXCHANGE_ANALYSIS = "exchange_analysis"


class AggregationType(Enum):
    """Aggregation types for SQL generation"""
    SUM = "SUM"
    AVG = "AVG"
    MAX = "MAX"
    MIN = "MIN"
    COUNT = "COUNT"


@dataclass
class TemplateContext:
    """Context for template generation"""
    query_type: QueryType
    aggregation_type: AggregationType
    target_column: str
    tables: List[str]
    filters: Dict[str, Any]
    grouping: List[str]
    ordering: List[str]
    time_period: Optional[str] = None
    region: Optional[str] = None
    source: Optional[str] = None


@dataclass
class TemplateValidation:
    """Validation result for template"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    confidence: float


class SQLTemplateEngine:
    """Engine for constrained SQL generation using templates"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.validation_rules = self._initialize_validation_rules()
        self.business_rules = self._initialize_business_rules()
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize SQL templates for different query types"""
        return {
            # Aggregation templates
            "avg_shortage": """
SELECT AVG(fs.EnergyShortage) AS AverageShortage
FROM FactAllIndiaDailySummary fs
JOIN DimRegions r ON fs.RegionID = r.RegionID
JOIN DimDates d ON fs.DateID = d.DateID
WHERE {where_clause}
GROUP BY {group_by}
ORDER BY {order_by}
""",
            
            "avg_energy": """
SELECT AVG(fs.EnergyMet) AS AverageEnergy
FROM FactAllIndiaDailySummary fs
JOIN DimRegions r ON fs.RegionID = r.RegionID
JOIN DimDates d ON fs.DateID = d.DateID
WHERE {where_clause}
GROUP BY {group_by}
ORDER BY {order_by}
""",
            
            "sum_shortage": """
SELECT SUM(fs.EnergyShortage) AS TotalShortage
FROM FactAllIndiaDailySummary fs
JOIN DimRegions r ON fs.RegionID = r.RegionID
JOIN DimDates d ON fs.DateID = d.DateID
WHERE {where_clause}
GROUP BY {group_by}
ORDER BY {order_by}
""",
            
            "sum_energy": """
SELECT SUM(fs.EnergyMet) AS TotalEnergy
FROM FactAllIndiaDailySummary fs
JOIN DimRegions r ON fs.RegionID = r.RegionID
JOIN DimDates d ON fs.DateID = d.DateID
WHERE {where_clause}
GROUP BY {group_by}
ORDER BY {order_by}
""",
            
            "max_demand": """
SELECT MAX(fs.MaxDemandSCADA) AS MaxDemand
FROM FactAllIndiaDailySummary fs
JOIN DimRegions r ON fs.RegionID = r.RegionID
JOIN DimDates d ON fs.DateID = d.DateID
WHERE {where_clause}
GROUP BY {group_by}
ORDER BY {order_by}
""",
            
            "min_demand": """
SELECT MIN(fs.MaxDemandSCADA) AS MinDemand
FROM FactAllIndiaDailySummary fs
JOIN DimRegions r ON fs.RegionID = r.RegionID
JOIN DimDates d ON fs.DateID = d.DateID
WHERE {where_clause}
GROUP BY {group_by}
ORDER BY {order_by}
""",
            
            # Time series templates
            "time_series_energy": """
SELECT 
    FORMAT(d.ActualDate, 'yyyy-MM') AS Month,
    {aggregation}(fs.{column}) AS {metric_name}
FROM FactAllIndiaDailySummary fs
JOIN DimRegions r ON fs.RegionID = r.RegionID
JOIN DimDates d ON fs.DateID = d.DateID
WHERE {where_clause}
GROUP BY FORMAT(d.ActualDate, 'yyyy-MM')
ORDER BY FORMAT(d.ActualDate, 'yyyy-MM')
""",
            
            "time_series_shortage": """
SELECT 
    FORMAT(d.ActualDate, 'yyyy-MM') AS Month,
    {aggregation}(fs.EnergyShortage) AS {metric_name}
FROM FactAllIndiaDailySummary fs
JOIN DimRegions r ON fs.RegionID = r.RegionID
JOIN DimDates d ON fs.DateID = d.DateID
WHERE {where_clause}
GROUP BY FORMAT(d.ActualDate, 'yyyy-MM')
ORDER BY FORMAT(d.ActualDate, 'yyyy-MM')
""",
            
            # Regional analysis templates
            "regional_energy": """
SELECT 
    r.RegionName,
    {aggregation}(fs.{column}) AS {metric_name}
FROM FactAllIndiaDailySummary fs
JOIN DimRegions r ON fs.RegionID = r.RegionID
JOIN DimDates d ON fs.DateID = d.DateID
WHERE {where_clause}
GROUP BY r.RegionName
ORDER BY {metric_name} DESC
""",
            
            "regional_shortage": """
SELECT 
    r.RegionName,
    {aggregation}(fs.EnergyShortage) AS {metric_name}
FROM FactAllIndiaDailySummary fs
JOIN DimRegions r ON fs.RegionID = r.RegionID
JOIN DimDates d ON fs.DateID = d.DateID
WHERE {where_clause}
GROUP BY r.RegionName
ORDER BY {metric_name} DESC
""",
            
            # Generation analysis templates
            "generation_analysis": """
SELECT 
    fg.SourceName,
    {aggregation}(fg.GenerationOutput) AS {metric_name}
FROM FactTimeBlockGeneration fg
JOIN DimDates d ON fg.DateID = d.DateID
WHERE {where_clause}
GROUP BY fg.SourceName
ORDER BY {metric_name} DESC
""",
            
            # State-level energy templates
            "state_energy": """
SELECT 
    ds.StateName,
    {aggregation}(fs.{column}) AS {metric_name}
FROM FactStateDailyEnergy fs
JOIN DimStates ds ON fs.StateID = ds.StateID
JOIN DimDates d ON fs.DateID = d.DateID
WHERE {where_clause}
GROUP BY ds.StateName
ORDER BY {metric_name} DESC
""",
            
            "state_shortage": """
SELECT 
    ds.StateName,
    {aggregation}(fs.EnergyShortage) AS {metric_name}
FROM FactStateDailyEnergy fs
JOIN DimStates ds ON fs.StateID = ds.StateID
JOIN DimDates d ON fs.DateID = d.DateID
WHERE {where_clause}
GROUP BY ds.StateName
ORDER BY {metric_name} DESC
""",

            "state_demand_shortage": """
SELECT 
    ds.StateName,
    {aggregation}(fs.Shortage) AS {metric_name}
FROM FactStateDailyEnergy fs
JOIN DimStates ds ON fs.StateID = ds.StateID
JOIN DimDates d ON fs.DateID = d.DateID
WHERE {where_clause}
GROUP BY ds.StateName
ORDER BY {metric_name} DESC
""",

            
            "state_demand": """
SELECT 
    ds.StateName,
    {aggregation}(fs.MaximumDemand) AS {metric_name}
FROM FactStateDailyEnergy fs
JOIN DimStates ds ON fs.StateID = ds.StateID
JOIN DimDates d ON fs.DateID = d.DateID
WHERE {where_clause}
GROUP BY ds.StateName
ORDER BY {metric_name} DESC
""",
            
            # Generation breakdown templates
            "generation_breakdown": """
SELECT 
    dgs.SourceName,
    dr.RegionName,
    {aggregation}(fdgb.GenerationAmount) AS {metric_name}
FROM FactDailyGenerationBreakdown fdgb
JOIN DimGenerationSources dgs ON fdgb.GenerationSourceID = dgs.GenerationSourceID
JOIN DimRegions dr ON fdgb.RegionID = dr.RegionID
JOIN DimDates d ON fdgb.DateID = d.DateID
WHERE {where_clause}
GROUP BY dgs.SourceName, dr.RegionName
ORDER BY {metric_name} DESC
""",
            
            "fuel_source_generation": """
SELECT 
    dgs.SourceName,
    {aggregation}(fdgb.GenerationAmount) AS {metric_name}
FROM FactDailyGenerationBreakdown fdgb
JOIN DimGenerationSources dgs ON fdgb.GenerationSourceID = dgs.GenerationSourceID
JOIN DimDates d ON fdgb.DateID = d.DateID
WHERE {where_clause}
GROUP BY dgs.SourceName
ORDER BY {metric_name} DESC
""",
            
            # Time series templates for state data
            "time_series_state_energy": """
SELECT 
    FORMAT(d.ActualDate, 'yyyy-MM') AS Month,
    ds.StateName,
    {aggregation}(fs.{column}) AS {metric_name}
FROM FactStateDailyEnergy fs
JOIN DimStates ds ON fs.StateID = ds.StateID
JOIN DimDates d ON fs.DateID = d.DateID
WHERE {where_clause}
GROUP BY FORMAT(d.ActualDate, 'yyyy-MM'), ds.StateName
ORDER BY FORMAT(d.ActualDate, 'yyyy-MM'), {metric_name} DESC
""",
            
            # Exchange analysis templates
            "exchange_analysis": """
SELECT 
    dc.CountryName,
    dem.MechanismName,
    {aggregation}(fted.ExchangeValue) AS {metric_name}
FROM FactTransnationalExchangeDetail fted
JOIN DimCountries dc ON fted.CountryID = dc.CountryID
JOIN DimExchangeMechanisms dem ON fted.MechanismID = dem.MechanismID
JOIN DimDates d ON fted.DateID = d.DateID
WHERE {where_clause}
GROUP BY dc.CountryName, dem.MechanismName
ORDER BY {metric_name} DESC
"""
        }
    
    def _initialize_validation_rules(self) -> Dict[str, List[str]]:
        """Initialize validation rules for different query types"""
        return {
            "avg_shortage": [
                "MUST use AVG(fs.EnergyShortage)",
                "MUST NOT use SUM()",
                "MUST NOT use EnergyMet column",
                "MUST use FactAllIndiaDailySummary table"
            ],
            "avg_energy": [
                "MUST use AVG(fs.EnergyMet)",
                "MUST NOT use SUM()",
                "MUST NOT use EnergyShortage column unless specifically requested",
                "MUST use FactAllIndiaDailySummary table"
            ],
            "sum_shortage": [
                "MUST use SUM(fs.EnergyShortage)",
                "MUST NOT use AVG()",
                "MUST NOT use EnergyMet column",
                "MUST use FactAllIndiaDailySummary table"
            ],
            "sum_energy": [
                "MUST use SUM(fs.EnergyMet)",
                "MUST NOT use AVG()",
                "MUST NOT use EnergyShortage column unless specifically requested",
                "MUST use FactAllIndiaDailySummary table"
            ],
            "max_demand": [
                "MUST use MAX(fs.MaxDemandSCADA)",
                "MUST NOT use AVG() or SUM()",
                "MUST use FactAllIndiaDailySummary table"
            ],
            "min_demand": [
                "MUST use MIN(fs.MaxDemandSCADA)",
                "MUST NOT use AVG() or SUM()",
                "MUST use FactAllIndiaDailySummary table"
            ]
        }
    
    def _initialize_business_rules(self) -> Dict[str, Any]:
        """Initialize business rules for validation"""
        return {
            "column_mappings": {
                "energy": "EnergyMet",
                "shortage": "EnergyShortage",
                "demand": "MaxDemandSCADA",
                "generation": "GenerationOutput",
                "exchange": "ExchangeValue"
            },
            "aggregation_mappings": {
                "average": "AVG",
                "avg": "AVG",
                "mean": "AVG",
                "total": "SUM",
                "sum": "SUM",
                "maximum": "MAX",
                "max": "MAX",
                "highest": "MAX",
                "minimum": "MIN",
                "min": "MIN",
                "lowest": "MIN"
            },
            "table_mappings": {
                "energy": "FactAllIndiaDailySummary",
                "shortage": "FactAllIndiaDailySummary",
                "demand": "FactAllIndiaDailySummary",
                "generation": "FactTimeBlockGeneration",
                "exchange": "FactTransnationalExchangeDetail"
            }
        }
    
    def analyze_query(self, query: str) -> TemplateContext:
        """Analyze query to determine template context"""
        query_lower = query.lower()
        
        # Determine query type
        query_type = self._determine_query_type(query_lower)
        
        # Determine aggregation type
        aggregation_type = self._determine_aggregation_type(query_lower)
        
        # Determine target column
        target_column = self._determine_target_column(query_lower)
        
        # Determine tables
        tables = self._determine_tables(query_lower, target_column)
        
        # Determine filters
        filters = self._extract_filters(query_lower)
        
        # Determine grouping
        grouping = self._extract_grouping(query_lower)
        
        # Determine ordering
        ordering = self._extract_ordering(query_lower)
        
        # Determine time period
        time_period = self._extract_time_period(query_lower)
        
        # Determine region
        region = self._extract_region(query_lower)
        
        # Determine source
        source = self._extract_source(query_lower)
        
        return TemplateContext(
            query_type=query_type,
            aggregation_type=aggregation_type,
            target_column=target_column,
            tables=tables,
            filters=filters,
            grouping=grouping,
            ordering=ordering,
            time_period=time_period,
            region=region,
            source=source
        )
    
    def _determine_query_type(self, query_lower: str) -> QueryType:
        """Determine the type of query"""
        # Check for time-series queries first (including time-based grouping)
        if any(keyword in query_lower for keyword in ["trend", "over time", "by month", "by year", "monthly", "yearly", "daily", "weekly"]):
            return QueryType.TIME_SERIES
        
        # Check for shortage analysis (prioritize over regional analysis)
        if "shortage" in query_lower:
            return QueryType.SHORTAGE_ANALYSIS
        
        # Check for regional analysis (including "by region")
        if "region" in query_lower or "by region" in query_lower:
            return QueryType.REGIONAL_ANALYSIS
        
        # Check for state-specific queries
        if "state" in query_lower and ("energy" in query_lower or "shortage" in query_lower or "demand" in query_lower):
            return QueryType.REGIONAL_ANALYSIS
        
        # Check for generation analysis
        if "generation" in query_lower or "fuel" in query_lower or "source" in query_lower:
            return QueryType.GENERATION_ANALYSIS
        
        # Check for exchange analysis
        if "exchange" in query_lower:
            return QueryType.EXCHANGE_ANALYSIS
        
        # Default to aggregation
        return QueryType.AGGREGATION
    
    def _determine_aggregation_type(self, query_lower: str) -> AggregationType:
        """Determine the aggregation type"""
        if any(word in query_lower for word in ["average", "avg", "mean"]):
            return AggregationType.AVG
        elif any(word in query_lower for word in ["total", "sum"]):
            return AggregationType.SUM
        elif any(word in query_lower for word in ["maximum", "max", "highest"]):
            return AggregationType.MAX
        elif any(word in query_lower for word in ["minimum", "min", "lowest"]):
            return AggregationType.MIN
        else:
            # Check for demand-related queries that might imply MAX/MIN
            if "demand" in query_lower:
                if any(word in query_lower for word in ["maximum", "max", "highest"]):
                    return AggregationType.MAX
                elif any(word in query_lower for word in ["minimum", "min", "lowest"]):
                    return AggregationType.MIN
                else:
                    # Default for demand queries is MAX
                    return AggregationType.MAX
            return AggregationType.SUM  # Default
    
    def _determine_target_column(self, query_lower: str) -> str:
        """Determine the target column"""
        if "shortage" in query_lower:
            # Check if it's state-level shortage during maximum demand
            if ("demand" in query_lower or "peak" in query_lower) and "state" in query_lower:
                return "Shortage"  # State-level shortage during maximum demand
            else:
                return "EnergyShortage"  # Regular energy shortage
        elif "consumption" in query_lower or ("energy" in query_lower and "demand" in query_lower):
            # Energy consumption can also be termed as energy demand
            return "EnergyConsumption"
        elif "energy" in query_lower and "met" in query_lower:
            return "EnergyMet"
        elif any(term in query_lower for term in ["maximum demand", "peak demand", "maximum power demand", "power demand"]) or ("demand" in query_lower and ("maximum" in query_lower or "peak" in query_lower)):
            # Maximum Demand can also be termed as Maximum Power Demand, Peak Demand
            # Check context: if asking about regions, use regional column; if states, use state column
            if "region" in query_lower:
                return "MaxDemandSCADA"  # Regional maximum demand from national summary
            elif "state" in query_lower:
                return "MaximumDemand"  # State-level maximum demand
            else:
                # Default to regional if no specific context
                return "MaxDemandSCADA"
        elif "generation" in query_lower:
            return "GenerationAmount"
        elif "exchange" in query_lower:
            return "ExchangeValue"
        elif "energy" in query_lower:
            # Generic energy queries - try to be more specific based on context
            if "demand" in query_lower:
                return "EnergyConsumption"  # Energy demand typically means consumption
            elif "met" in query_lower:
                return "EnergyMet"
            else:
                return "EnergyMet"  # Default for generic energy queries
        else:
            # No specific energy-related keywords found
            # This could indicate a very generic query - let the template system handle it
            return "EnergyMet"  # Default fallback
    
    def _determine_tables(self, query_lower: str, target_column: str) -> List[str]:
        """Determine the required tables"""
        # Check for state-specific queries
        if "state" in query_lower:
            if "energy" in query_lower or "shortage" in query_lower or "demand" in query_lower:
                return ["FactStateDailyEnergy", "DimStates", "DimDates"]
            else:
                return ["FactStateDailyEnergy", "DimStates", "DimDates"]
        
        # Check for generation queries
        if "generation" in query_lower or "fuel" in query_lower or "source" in query_lower:
            return ["FactDailyGenerationBreakdown", "DimGenerationSources", "DimRegions", "DimDates"]
        
        # Check for specific columns
        if target_column == "GenerationOutput":
            return ["FactTimeBlockGeneration", "DimDates"]
        elif target_column == "ExchangeValue":
            return ["FactTransnationalExchangeDetail", "DimCountries", "DimExchangeMechanisms", "DimDates"]
        
        # Default to national summary
        return ["FactAllIndiaDailySummary", "DimRegions", "DimDates"]
    
    def _extract_filters(self, query_lower: str) -> Dict[str, Any]:
        """Extract filters from query"""
        filters = {}
        
        # Extract time filters
        if "2023" in query_lower:
            filters["year"] = "2023"
        elif "2024" in query_lower:
            filters["year"] = "2024"
        
        # Extract region filters
        if "north" in query_lower:
            filters["region"] = "North"
        elif "south" in query_lower:
            filters["region"] = "South"
        elif "east" in query_lower:
            filters["region"] = "East"
        elif "west" in query_lower:
            filters["region"] = "West"
        
        return filters
    
    def _extract_grouping(self, query_lower: str) -> List[str]:
        """Extract grouping from query"""
        grouping = []
        
        if "by region" in query_lower:
            grouping.append("dr.RegionName")
        elif "by state" in query_lower:
            grouping.append("ds.StateName")
        elif "by month" in query_lower or "monthly" in query_lower:
            grouping.append("strftime('%Y-%m', d.ActualDate)")
        
        return grouping
    
    def _extract_ordering(self, query_lower: str) -> List[str]:
        """Extract ordering from query"""
        ordering = []
        
        if "descending" in query_lower or "desc" in query_lower:
            ordering.append("DESC")
        elif "ascending" in query_lower or "asc" in query_lower:
            ordering.append("ASC")
        else:
            ordering.append("DESC")  # Default
        
        return ordering
    
    def _extract_time_period(self, query_lower: str) -> Optional[str]:
        """Extract time period from query"""
        if "month" in query_lower:
            return "month"
        elif "year" in query_lower:
            return "year"
        elif "week" in query_lower:
            return "week"
        return None
    
    def _extract_region(self, query_lower: str) -> Optional[str]:
        """Extract region from query"""
        regions = ["north", "south", "east", "west", "central"]
        for region in regions:
            if region in query_lower:
                return region.title()
        return None
    
    def _extract_source(self, query_lower: str) -> Optional[str]:
        """Extract source from query"""
        sources = ["solar", "wind", "thermal", "hydro", "nuclear"]
        for source in sources:
            if source in query_lower:
                return source.title()
        return None
    
    def select_template(self, context: TemplateContext) -> Optional[str]:
        """Select the appropriate template based on context"""
        template_key = self._generate_template_key(context)
        return self.templates.get(template_key)
    
    def _generate_template_key(self, context: TemplateContext) -> str:
        """Generate template key based on context"""
        # Check for state-specific queries first
        if "FactStateDailyEnergy" in context.tables:
            if context.query_type == QueryType.TIME_SERIES:
                return "time_series_state_energy"
            elif context.target_column == "EnergyShortage":
                return "state_shortage"
            elif context.target_column == "Shortage":
                return "state_demand_shortage"
            elif context.target_column == "MaximumDemand":
                return "state_demand"
            else:
                return "state_energy"
        
        # Check for generation breakdown queries
        if "FactDailyGenerationBreakdown" in context.tables:
            if "RegionName" in context.grouping:
                return "generation_breakdown"
            else:
                return "fuel_source_generation"
        
        # Check for shortage analysis
        if context.query_type == QueryType.SHORTAGE_ANALYSIS:
            if context.aggregation_type == AggregationType.AVG:
                return "avg_shortage"
            elif context.aggregation_type == AggregationType.SUM:
                return "sum_shortage"
        
        # Check for aggregation queries
        if context.query_type == QueryType.AGGREGATION:
            if context.target_column == "EnergyMet":
                if context.aggregation_type == AggregationType.AVG:
                    return "avg_energy"
                elif context.aggregation_type == AggregationType.SUM:
                    return "sum_energy"
            elif context.target_column == "EnergyConsumption":
                if context.aggregation_type == AggregationType.AVG:
                    return "avg_energy"  # Use energy template for consumption
                elif context.aggregation_type == AggregationType.SUM:
                    return "sum_energy"
            elif context.target_column == "MaxDemandSCADA":
                if context.aggregation_type == AggregationType.MAX:
                    return "max_demand"
                elif context.aggregation_type == AggregationType.MIN:
                    return "min_demand"
            elif context.target_column == "MaximumDemand":
                # State-level maximum demand
                if context.aggregation_type == AggregationType.MAX:
                    return "state_demand"  # Use state demand template
                elif context.aggregation_type == AggregationType.MIN:
                    return "state_demand"  # Use state demand template
        
        # Check for time series queries
        if context.query_type == QueryType.TIME_SERIES:
            if context.target_column == "EnergyShortage":
                return "time_series_shortage"
            else:
                return "time_series_energy"
        
        # Check for regional analysis
        if context.query_type == QueryType.REGIONAL_ANALYSIS:
            if context.target_column == "EnergyShortage":
                return "regional_shortage"
            elif context.target_column == "MaxDemandSCADA":
                if context.aggregation_type == AggregationType.MAX:
                    return "max_demand"
                elif context.aggregation_type == AggregationType.MIN:
                    return "min_demand"
            else:
                return "regional_energy"
        
        # Check for generation analysis
        if context.query_type == QueryType.GENERATION_ANALYSIS:
            return "generation_analysis"
        
        # Check for exchange analysis
        if context.query_type == QueryType.EXCHANGE_ANALYSIS:
            return "exchange_analysis"
        
        # Default template
        return "sum_energy"
    
    def generate_sql(self, query: str) -> Tuple[str, TemplateValidation]:
        """Generate SQL using templates with validation"""
        try:
            # Analyze query
            context = self.analyze_query(query)
            
            # Select template
            template = self.select_template(context)
            if not template:
                return "", TemplateValidation(
                    is_valid=False,
                    errors=["No suitable template found"],
                    warnings=[],
                    confidence=0.0
                )
            
            # Generate SQL
            sql = self._fill_template(template, context)
            
            # Validate SQL
            validation = self._validate_sql(sql, context)
            
            return sql, validation
            
        except Exception as e:
            logger.error(f"Error generating SQL with templates: {e}")
            return "", TemplateValidation(
                is_valid=False,
                errors=[f"Template generation failed: {str(e)}"],
                warnings=[],
                confidence=0.0
            )
    
    def _fill_template(self, template: str, context: TemplateContext) -> str:
        """Fill template with context values"""
        # Build where clause
        where_clause = self._build_where_clause(context.filters)
        
        # Build group by clause
        group_by = ", ".join(context.grouping) if context.grouping else "1"
        
        # Build order by clause
        order_by = ", ".join(context.ordering) if context.ordering else "1"
        
        # Fill template
        sql = template.format(
            where_clause=where_clause,
            group_by=group_by,
            order_by=order_by,
            aggregation=context.aggregation_type.value,
            column=context.target_column,
            metric_name=f"{context.aggregation_type.value}{context.target_column}"
        )
        
        return sql.strip()
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> str:
        """Build WHERE clause from filters"""
        conditions = []
        
        if "year" in filters:
            conditions.append(f"strftime('%Y', d.ActualDate) = '{filters['year']}'")
        
        if "region" in filters:
            conditions.append(f"r.RegionName = '{filters['region']}'")
        
        if conditions:
            return " AND ".join(conditions)
        else:
            return "1=1"  # Default condition
    
    def _validate_sql(self, sql: str, context: TemplateContext) -> TemplateValidation:
        """Validate generated SQL against rules"""
        errors = []
        warnings = []
        confidence = 1.0
        
        # Get validation rules for template
        template_key = self._generate_template_key(context)
        rules = self.validation_rules.get(template_key, [])
        
        # Check each rule
        for rule in rules:
            if not self._check_rule(sql, rule):
                errors.append(f"Validation failed: {rule}")
                confidence -= 0.2
        
        # Check for forbidden patterns
        forbidden_patterns = [
            (r"\bDROP\b", "DROP operations are not allowed"),
            (r"\bDELETE\b", "DELETE operations are not allowed"),
            (r"\bUPDATE\b", "UPDATE operations are not allowed"),
            (r"\bINSERT\b", "INSERT operations are not allowed"),
            (r"\bALTER\b", "ALTER operations are not allowed"),
            (r"\bCREATE\b", "CREATE operations are not allowed"),
        ]
        
        for pattern, message in forbidden_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                errors.append(message)
                confidence -= 0.3
        
        # Check for required patterns
        required_patterns = [
            (r"\bSELECT\b", "Query must start with SELECT"),
            (r"\bFROM\b", "Query must have FROM clause"),
        ]
        
        for pattern, message in required_patterns:
            if not re.search(pattern, sql, re.IGNORECASE):
                errors.append(message)
                confidence -= 0.2
        
        is_valid = len(errors) == 0
        confidence = max(0.0, min(1.0, confidence))
        
        return TemplateValidation(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            confidence=confidence
        )
    
    def _check_rule(self, sql: str, rule: str) -> bool:
        """Check if SQL follows a specific rule"""
        sql_lower = sql.lower()
        
        if "MUST use" in rule:
            # Extract required pattern
            pattern = rule.split("MUST use ")[1].split()[0]
            return pattern.lower() in sql_lower
        elif "MUST NOT use" in rule:
            # Extract forbidden pattern
            pattern = rule.split("MUST NOT use ")[1].split()[0]
            return pattern.lower() not in sql_lower
        elif "MUST use" in rule and "table" in rule:
            # Extract required table
            table = rule.split("MUST use ")[1].split(" table")[0]
            return table.lower() in sql_lower
        
        return True
