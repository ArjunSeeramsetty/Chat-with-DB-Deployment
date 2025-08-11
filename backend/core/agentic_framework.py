"""
Agentic Framework for Phase 2: Agentic Workflow Implementation
Inspired by Motia framework for step-based architecture and event-driven processing
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Types of specialized agents"""
    QUERY_ANALYSIS = "query_analysis"
    SQL_GENERATION = "sql_generation"
    VALIDATION = "validation"
    EXECUTION = "execution"
    VISUALIZATION = "visualization"
    FEEDBACK = "feedback"


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventType(str, Enum):
    """Event types for event-driven processing"""
    QUERY_RECEIVED = "query_received"
    ANALYSIS_COMPLETE = "analysis_complete"
    SQL_GENERATED = "sql_generated"
    VALIDATION_COMPLETE = "validation_complete"
    EXECUTION_COMPLETE = "execution_complete"
    VISUALIZATION_COMPLETE = "visualization_complete"
    ERROR_OCCURRED = "error_occurred"
    WORKFLOW_COMPLETE = "workflow_complete"


@dataclass
class WorkflowContext:
    """Context for workflow execution"""
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default_user"
    query: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from agent execution"""
    success: bool
    data: Dict[str, Any]
    confidence: float = 0.0
    execution_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStep:
    """Individual step in a workflow"""
    step_id: str
    agent_type: AgentType
    name: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0
    required: bool = True
    handler: Optional[Callable] = None


@dataclass
class WorkflowDefinition:
    """Definition of a complete workflow"""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    max_execution_time: float = 300.0
    parallel_execution: bool = False
    error_handling: str = "continue"  # continue, stop, retry


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, agent_type: AgentType, name: str):
        self.agent_type = agent_type
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")
        
    @abstractmethod
    async def execute(self, context: WorkflowContext, **kwargs) -> AgentResult:
        """Execute the agent's main logic"""
        pass
    
    async def pre_execute(self, context: WorkflowContext) -> bool:
        """Pre-execution checks and setup"""
        return True
    
    async def post_execute(self, context: WorkflowContext, result: AgentResult) -> None:
        """Post-execution cleanup and logging"""
        self.logger.info(f"Agent {self.name} completed with confidence: {result.confidence:.2f}")
    
    def get_required_context(self) -> List[str]:
        """Get required context keys for this agent"""
        return []


class QueryAnalysisAgent(BaseAgent):
    """Agent for analyzing natural language queries"""
    
    def __init__(self):
        super().__init__(AgentType.QUERY_ANALYSIS, "QueryAnalysisAgent")
        
    async def execute(self, context: WorkflowContext, **kwargs) -> AgentResult:
        start_time = time.time()
        
        try:
            # Extract query from context
            query = context.query
            if not query:
                return AgentResult(
                    success=False,
                    data={},
                    error="No query provided",
                    execution_time=time.time() - start_time
                )
            
            # Perform query analysis
            analysis_result = await self._analyze_query(query)
            
            # Update context with analysis results
            context.state["query_analysis"] = analysis_result
            
            return AgentResult(
                success=True,
                data=analysis_result,
                confidence=analysis_result.get("confidence", 0.0),
                execution_time=time.time() - start_time,
                metadata={"query_length": len(query)}
            )
            
        except Exception as e:
            self.logger.error(f"Query analysis failed: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query for intent, entities, and context"""
        # This would integrate with the existing intent analyzer
        from backend.core.intent import IntentAnalyzer
        
        try:
            analyzer = IntentAnalyzer()
            analysis = await analyzer.analyze_intent(query)
            
            return {
                "intent": analysis.intent.value if hasattr(analysis.intent, 'value') else str(analysis.intent),
                "entities": analysis.entities if hasattr(analysis, 'entities') else [],
                "query_type": analysis.query_type.value if hasattr(analysis.query_type, 'value') else str(analysis.query_type),
                "confidence": analysis.confidence if hasattr(analysis, 'confidence') else 0.0,
                "time_period": analysis.time_period if hasattr(analysis, 'time_period') else "unknown",
                "metrics": analysis.metrics if hasattr(analysis, 'metrics') else [],
                "main_table": analysis.main_table if hasattr(analysis, 'main_table') else "",
                "dimension_table": analysis.dimension_table if hasattr(analysis, 'dimension_table') else "",
                "join_key": analysis.join_key if hasattr(analysis, 'join_key') else "",
                "name_column": analysis.name_column if hasattr(analysis, 'name_column') else "",
                "detected_keywords": analysis.detected_keywords if hasattr(analysis, 'detected_keywords') else []
            }
        except Exception as e:
            self.logger.error(f"Query analysis error: {e}")
            return {
                "intent": "unknown",
                "entities": [],
                "query_type": "unknown",
                "confidence": 0.0,
                "time_period": "unknown",
                "metrics": [],
                "main_table": "",
                "dimension_table": "",
                "join_key": "",
                "name_column": "",
                "detected_keywords": []
            }


class SQLGenerationAgent(BaseAgent):
    """Agent for generating SQL queries"""
    
    def __init__(self):
        super().__init__(AgentType.SQL_GENERATION, "SQLGenerationAgent")
        
    async def execute(self, context: WorkflowContext, **kwargs) -> AgentResult:
        start_time = time.time()
        
        try:
            # Get query analysis from context
            query_analysis = context.state.get("query_analysis")
            if not query_analysis:
                return AgentResult(
                    success=False,
                    data={},
                    error="Query analysis not available",
                    execution_time=time.time() - start_time
                )
            
            # Generate SQL using existing assembler
            sql_result = await self._generate_sql(context.query, query_analysis)
            
            # Update context with SQL result
            context.state["sql_generation"] = sql_result
            
            return AgentResult(
                success=True,
                data=sql_result,
                confidence=sql_result.get("confidence", 0.0),
                execution_time=time.time() - start_time,
                metadata={"sql_length": len(sql_result.get("sql", ""))}
            )
            
        except Exception as e:
            self.logger.error(f"SQL generation failed: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def _generate_sql(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL using existing assembler"""
        from backend.core.assembler import SQLAssembler
        from backend.core.types import QueryAnalysis, ContextInfo, QueryType, IntentType
        
        try:
            # Create SQLAssembler instance
            assembler = SQLAssembler()
            
            # Convert analysis dict to QueryAnalysis object if needed
            if isinstance(analysis, dict):
                # Create a basic QueryAnalysis object from the dict with required fields
                query_analysis = QueryAnalysis(
                    original_query=query,
                    query_type=QueryType(analysis.get("query_type", "unknown")),
                    intent=IntentType(analysis.get("intent", "unknown")),
                    entities=analysis.get("entities", []),
                    time_period=analysis.get("time_period", "unknown"),
                    confidence=analysis.get("confidence", 0.0),
                    main_table=analysis.get("main_table", ""),
                    dimension_table=analysis.get("dimension_table", ""),
                    join_key=analysis.get("join_key", ""),
                    name_column=analysis.get("name_column", ""),
                    detected_keywords=analysis.get("detected_keywords", [])
                )
            else:
                query_analysis = analysis
            
            # Create a basic context
            context = ContextInfo(
                query_analysis=query_analysis,
                user_mappings=[],
                dimension_values={},
                schema_linker=None,
                llm_provider=None
            )
            
            # Generate SQL using the correct method
            sql_result = assembler.generate_sql(query, query_analysis, context)
            
            # Extract SQL from the result
            if hasattr(sql_result, 'sql'):
                sql = sql_result.sql
                success = sql_result.success
            elif isinstance(sql_result, dict):
                sql = sql_result.get('sql', '')
                success = sql_result.get('success', False)
            else:
                sql = str(sql_result)
                success = bool(sql and sql.strip() and not sql.startswith('--'))
            
            # If SQL generation failed, try to generate a basic fallback SQL
            if not sql or not success:
                sql = self._generate_fallback_sql(query, analysis)
                generation_method = "fallback"
                confidence = min(analysis.get("confidence", 0.0) * 0.8, 0.8)  # Reduce confidence for fallback
            else:
                generation_method = "assembler"
                confidence = analysis.get("confidence", 0.0)
            
            return {
                "sql": sql,
                "confidence": confidence,
                "query_type": analysis.get("query_type", "unknown"),
                "tables_used": analysis.get("main_table", ""),
                "generation_method": generation_method,
                "warnings": ["Fallback SQL used - results may be approximate"] if generation_method == "fallback" else []
            }
            
        except Exception as e:
            self.logger.error(f"SQL generation error: {e}")
            # Generate fallback SQL even on error
            fallback_sql = self._generate_fallback_sql(query, analysis)
            return {
                "sql": fallback_sql,
                "confidence": 0.5,  # Low confidence for error fallback
                "query_type": "unknown",
                "tables_used": "",
                "generation_method": "error_fallback",
                "warnings": [
                    "SQL generation failed - using fallback SQL",
                    "Results may be approximate due to generation error",
                    f"Error: {str(e)}"
                ]
            }
    
    def _generate_fallback_sql(self, query: str, analysis: Dict[str, Any]) -> str:
        """Generate intelligent fallback SQL when template matching fails"""
        query_lower = query.lower()
        
        # Extract context from analysis
        query_type = analysis.get("query_type", "unknown")
        intent = analysis.get("intent", "unknown")
        entities = analysis.get("entities", [])
        time_period = analysis.get("time_period", "unknown")
        metrics = analysis.get("metrics", [])
        
        # Determine time filter based on query content and analysis
        time_filter = self._extract_time_filter(query_lower, time_period)
        
        # Determine aggregation function based on query intent
        aggregation_function = self._determine_aggregation_function(query_lower, intent)
        
        # Determine energy column based on metrics
        energy_column = self._determine_energy_column(metrics, query_lower)
        
        # Generate contextually appropriate SQL
        if "state" in query_lower or "states" in query_lower or query_type == "state":
            return self._generate_state_fallback_sql(query_lower, entities, time_filter, aggregation_function, energy_column)
        elif "region" in query_lower or "regions" in query_lower or query_type == "region":
            return self._generate_region_fallback_sql(query_lower, entities, time_filter, aggregation_function, energy_column)
        elif "growth" in query_lower or "monthly" in query_lower or "trend" in query_lower:
            return self._generate_growth_fallback_sql(query_lower, entities, time_filter, energy_column)
        elif "generation" in query_lower or "source" in query_lower:
            return self._generate_generation_fallback_sql(query_lower, entities, time_filter, aggregation_function)
        else:
            # Default fallback with warning
            return self._generate_default_fallback_sql(query, time_filter)
    
    def _extract_time_filter(self, query_lower: str, time_period: str) -> str:
        """Extract time filter from query using SQLite-compatible date functions"""
        import re
        # Look for specific 4-digit year
        year_match = re.search(r'20\d{2}', query_lower)
        if year_match:
            year = year_match.group()
            return f"WHERE strftime('%Y', dt.ActualDate) = '{year}'"

        # Look for time period indicators
        if "this year" in query_lower or "current year" in query_lower:
            return "WHERE strftime('%Y', dt.ActualDate) = '2024'"
        elif "last year" in query_lower or "previous year" in query_lower:
            return "WHERE strftime('%Y', dt.ActualDate) = '2023'"
        elif "2023" in query_lower:
            return "WHERE strftime('%Y', dt.ActualDate) = '2023'"
        elif "2024" in query_lower:
            return "WHERE strftime('%Y', dt.ActualDate) = '2024'"
        elif "2025" in query_lower:
            return "WHERE strftime('%Y', dt.ActualDate) = '2025'"
        else:
            # Default to current year if no specific time mentioned
            return "WHERE strftime('%Y', dt.ActualDate) = '2024'"
    
    def _determine_aggregation_function(self, query_lower: str, intent: str) -> str:
        """Determine appropriate aggregation function"""
        if any(word in query_lower for word in ["total", "sum", "sum of"]):
            return "SUM"
        elif any(word in query_lower for word in ["average", "avg", "mean"]):
            return "AVG"
        elif any(word in query_lower for word in ["maximum", "max", "highest"]):
            return "MAX"
        elif any(word in query_lower for word in ["minimum", "min", "lowest"]):
            return "MIN"
        elif any(word in query_lower for word in ["count", "number of"]):
            return "COUNT"
        else:
            # Default to SUM for most energy queries
            return "SUM"
    
    def _determine_energy_column(self, metrics: list, query_lower: str) -> str:
        """Determine appropriate energy column"""
        if metrics:
            # Use first metric if available
            return metrics[0]
        
        # Default based on query content and correct column names
        if "demand" in query_lower:
            return "MaximumDemand"  # Correct column name for state table
        elif "consumption" in query_lower:
            return "EnergyMet"  # Correct column name for energy consumption
        elif "generation" in query_lower:
            return "GenerationAmount"  # Correct column name for generation
        elif "shortage" in query_lower:
            return "EnergyShortage"  # Correct column name for shortage
        else:
            return "EnergyMet"  # Default energy column - correct name
    
    def _generate_state_fallback_sql(self, query_lower: str, entities: list, time_filter: str, aggregation_function: str, energy_column: str) -> str:
        """Generate state-specific fallback SQL"""
        # Check for specific state mentions
        state_filter = ""
        if entities:
            state_entities = [e for e in entities if "state" in e.lower() or e.lower() in ["northern", "southern", "eastern", "western"]]
            if state_entities:
                state_names = "', '".join(state_entities)
                state_filter = f"AND ds.StateName IN ('{state_names}')"
        
        return f"""
            SELECT ds.StateName, ROUND({aggregation_function}(fs.{energy_column}), 2) as TotalEnergy
            FROM FactStateDailyEnergy fs
            JOIN DimStates ds ON fs.StateID = ds.StateID
            JOIN DimDates dt ON fs.DateID = dt.DateID
            {time_filter}
            {state_filter}
            GROUP BY ds.StateName
            ORDER BY TotalEnergy DESC
        """
    
    def _generate_region_fallback_sql(self, query_lower: str, entities: list, time_filter: str, aggregation_function: str, energy_column: str) -> str:
        """Generate region-specific fallback SQL"""
        # Check for specific region mentions
        region_filter = ""
        if entities:
            region_entities = [e for e in entities if "region" in e.lower() or e.lower() in ["northern", "southern", "eastern", "western"]]
            if region_entities:
                region_names = "', '".join(region_entities)
                region_filter = f"AND d.RegionName IN ('{region_names}')"
        
        return f"""
            SELECT d.RegionName, ROUND({aggregation_function}(f.{energy_column}), 2) as TotalEnergy
            FROM FactAllIndiaDailySummary f
            JOIN DimRegions d ON f.RegionID = d.RegionID
            JOIN DimDates dt ON f.DateID = dt.DateID
            {time_filter}
            {region_filter}
            GROUP BY d.RegionName
            ORDER BY TotalEnergy DESC
        """
    
    def _generate_growth_fallback_sql(self, query_lower: str, entities: list, time_filter: str, energy_column: str) -> str:
        """Generate SQLite-friendly growth SQL using self-joins"""
        return f"""
        SELECT 
            r.RegionName,
            strftime('%Y-%m', d.ActualDate) as Month,
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
                strftime('%Y-%m', d2.ActualDate) as Month,
                SUM(fs2.{energy_column}) as PreviousMonthEnergy
            FROM FactAllIndiaDailySummary fs2
            JOIN DimRegions r2 ON fs2.RegionID = r2.RegionID
            JOIN DimDates d2 ON fs2.DateID = d2.DateID
            WHERE strftime('%Y', d2.ActualDate) = '2024'
            GROUP BY r2.RegionName, strftime('%Y-%m', d2.ActualDate)
        ) prev ON r.RegionName = prev.RegionName 
            AND strftime('%Y-%m', d.ActualDate) = date(prev.Month || '-01', '+1 month')
        WHERE strftime('%Y', d.ActualDate) = '2024'
        GROUP BY r.RegionName, strftime('%Y-%m', d.ActualDate)
        ORDER BY r.RegionName, Month
        """
    
    def _generate_generation_fallback_sql(self, query_lower: str, entities: list, time_filter: str, aggregation_function: str) -> str:
        """Generate generation-specific fallback SQL"""
        return f"""
            SELECT dgs.SourceName, ROUND({aggregation_function}(fdgb.GenerationAmount), 2) as TotalGeneration
            FROM FactDailyGenerationBreakdown fdgb
            JOIN DimGenerationSources dgs ON fdgb.GenerationSourceID = dgs.GenerationSourceID
            JOIN DimDates dt ON fdgb.DateID = dt.DateID
            {time_filter}
            GROUP BY dgs.SourceName
            ORDER BY TotalGeneration DESC
        """
    
    def _generate_default_fallback_sql(self, query: str, time_filter: str) -> str:
        """Generate a valid, conservative default SQL (All-India EnergyMet)"""
        return f"""
            SELECT 
                d.RegionName,
                ROUND(SUM(f.EnergyMet), 2) as TotalEnergyConsumption
            FROM FactAllIndiaDailySummary f
            JOIN DimRegions d ON f.RegionID = d.RegionID
            JOIN DimDates dt ON f.DateID = dt.DateID
            {time_filter}
            AND d.RegionName = 'India'
        """


class ValidationAgent(BaseAgent):
    """Agent for validating SQL queries"""
    
    def __init__(self):
        super().__init__(AgentType.VALIDATION, "ValidationAgent")
        
    async def execute(self, context: WorkflowContext, **kwargs) -> AgentResult:
        start_time = time.time()
        
        try:
            # Get SQL from context
            sql_generation = context.state.get("sql_generation")
            if not sql_generation:
                return AgentResult(
                    success=False,
                    data={},
                    error="SQL generation not available",
                    execution_time=time.time() - start_time
                )
            
            sql = sql_generation.get("sql", "")
            if not sql or sql.strip() == "":
                return AgentResult(
                    success=False,
                    data={},
                    error="No SQL to validate",
                    execution_time=time.time() - start_time
                )
            
            # Skip validation for fallback SQL (basic validation only)
            if sql_generation.get("generation_method") in ["fallback", "error_fallback"]:
                return AgentResult(
                    success=True,
                    data={
                        "is_valid": True,
                        "validation_method": "fallback_skip",
                        "warnings": [
                            "Using fallback SQL - results may be approximate",
                            "Template matching failed, using intelligent fallback",
                            "Consider rephrasing query for more accurate results"
                        ],
                        "confidence_reason": "Fallback SQL generated due to template mismatch"
                    },
                    confidence=0.6,  # Lower confidence for fallback SQL
                    execution_time=time.time() - start_time
                )
            
            # Validate SQL using existing validator
            validation_result = await self._validate_sql(sql)
            
            # Update context with validation result
            context.state["validation"] = validation_result
            
            return AgentResult(
                success=validation_result.get("is_valid", False),
                data=validation_result,
                confidence=validation_result.get("confidence", 0.0),
                execution_time=time.time() - start_time,
                metadata={"sql_length": len(sql)}
            )
            
        except Exception as e:
            self.logger.error(f"SQL validation failed: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def _validate_sql(self, sql: str) -> Dict[str, Any]:
        """Validate SQL using existing validator"""
        from backend.core.validator import EnhancedSQLValidator
        
        validator = EnhancedSQLValidator()
        result = validator.validate_sql(sql)
        
        return {
            "is_valid": result.is_valid,
            "confidence": result.confidence if hasattr(result, 'confidence') else 0.0,
            "errors": result.errors if hasattr(result, 'errors') else [],
            "warnings": result.warnings if hasattr(result, 'warnings') else [],
            "checks": ["syntax", "schema", "security"]
        }


class ExecutionAgent(BaseAgent):
    """Agent for executing SQL queries"""
    
    def __init__(self):
        super().__init__(AgentType.EXECUTION, "ExecutionAgent")
        
    async def execute(self, context: WorkflowContext, **kwargs) -> AgentResult:
        start_time = time.time()
        
        try:
            # Get SQL and validation from context
            sql_generation = context.state.get("sql_generation")
            validation = context.state.get("validation")
            
            if not sql_generation:
                return AgentResult(
                    success=False,
                    data={},
                    error="SQL generation not available",
                    execution_time=time.time() - start_time
                )
            
            # Check validation only if it's not fallback SQL
            if validation and not validation.get("is_valid", False) and sql_generation.get("generation_method") not in ["fallback", "error_fallback"]:
                return AgentResult(
                    success=False,
                    data={},
                    error="SQL validation failed",
                    execution_time=time.time() - start_time
                )
            
            sql = sql_generation.get("sql", "")
            if not sql or sql.strip() == "":
                return AgentResult(
                    success=False,
                    data={},
                    error="No SQL to execute",
                    execution_time=time.time() - start_time
                )
            
            # Execute SQL using existing executor
            execution_result = await self._execute_sql(sql)
            
            # Update context with execution result
            context.state["execution"] = execution_result
            
            return AgentResult(
                success=execution_result.get("success", False),
                data=execution_result,
                confidence=1.0 if execution_result.get("success") else 0.0,
                execution_time=time.time() - start_time,
                metadata={"row_count": execution_result.get("row_count", 0)}
            )
            
        except Exception as e:
            self.logger.error(f"SQL execution failed: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def _execute_sql(self, sql: str) -> Dict[str, Any]:
        """Execute SQL using existing executor"""
        from backend.core.executor import AsyncSQLExecutor
        from backend.config import get_settings
        
        try:
            settings = get_settings()
            executor = AsyncSQLExecutor(settings.database_path)
            result = await executor.execute_sql_async(sql)
            
            return {
                "success": result.success,
                "data": result.data,
                "row_count": result.row_count,
                "execution_time": result.execution_time,
                "error": result.error
            }
        except Exception as e:
            self.logger.error(f"SQL execution error: {e}")
            return {
                "success": False,
                "data": [],
                "row_count": 0,
                "execution_time": 0.0,
                "error": str(e)
            }


class VisualizationAgent(BaseAgent):
    """Agent for generating visualizations"""
    
    def __init__(self):
        super().__init__(AgentType.VISUALIZATION, "VisualizationAgent")
        
    async def execute(self, context: WorkflowContext, **kwargs) -> AgentResult:
        start_time = time.time()
        
        try:
            # Get execution result from context
            execution = context.state.get("execution")
            if not execution or not execution.get("success"):
                return AgentResult(
                    success=False,
                    data={},
                    error="No execution data available",
                    execution_time=time.time() - start_time
                )
            
            data = execution.get("data", [])
            if not data:
                return AgentResult(
                    success=False,
                    data={},
                    error="No data to visualize",
                    execution_time=time.time() - start_time
                )
            
            # Generate visualization
            viz_result = await self._generate_visualization(data, context.query)
            
            # Update context with visualization result
            context.state["visualization"] = viz_result
            
            return AgentResult(
                success=True,
                data=viz_result,
                confidence=viz_result.get("confidence", 0.0),
                execution_time=time.time() - start_time,
                metadata={"chart_type": viz_result.get("chart_type")}
            )
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def _generate_visualization(self, data: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Generate visualization recommendation"""
        if not data:
            return {"chart_type": "none", "confidence": 0.0}

        # Try WrenAI-style chart intelligence (vega-lite), fallback to heuristics
        try:
            from backend.core.visualization_wren import WrenAIVisualizationService
            viz = WrenAIVisualizationService()
            spec = viz.generate_visualization(data, query, sql="")
            if spec:
                if spec.get("chartType") == "vega_lite":
                    # UI may not support vega_lite yet. Provide a safe fallback payload
                    raw_opts = spec.get("options", {}) or {}
                    x_field = raw_opts.get("xField")
                    y_field = raw_opts.get("yField")
                    series_field = raw_opts.get("seriesField")
                    mapped_opts = {
                        "xAxis": x_field or "Month",
                        "yAxis": [y_field] if y_field else [self._infer_first_numeric_key(data)],
                        "groupBy": series_field or "",
                        "data": data,
                        "valueLabel": y_field or self._infer_first_numeric_key(data),
                    }
                    # Preserve Vega-Lite spec for future UI support
                    mapped_opts["vegaSpec"] = spec.get("vegaSpec", {})
                    fallback_type = "line" if (mapped_opts.get("xAxis") or "").lower() in {"month", "date", "actualdate", "day", "year"} else "bar"
                    return {"chart_type": fallback_type, "config": mapped_opts, "confidence": 0.9}
                # Legacy compatibility for non-vega outputs
                raw_opts = spec.get("options", {}) or {}
                if "data" not in raw_opts:
                    raw_opts["data"] = data
                # Also normalize to xAxis/yAxis to match UI reducer
                if raw_opts.get("xField") or raw_opts.get("yField"):
                    raw_opts = {
                        **raw_opts,
                        "xAxis": raw_opts.get("xField", "Month"),
                        "yAxis": [raw_opts.get("yField")] if raw_opts.get("yField") else [self._infer_first_numeric_key(data)],
                        "groupBy": raw_opts.get("seriesField", ""),
                        "valueLabel": raw_opts.get("yField") or self._infer_first_numeric_key(data),
                    }
                return {"chart_type": spec.get("chartType", "bar"), "config": raw_opts, "confidence": 0.85}
        except Exception:
            pass

        # Heuristic fallback
        headers = list(data[0].keys()) if data else []
        if any("month" in h.lower() for h in headers):
            chart_type = "line"
        elif any("growth" in h.lower() or "percentage" in h.lower() for h in headers):
            chart_type = "dualAxisLine"
        else:
            chart_type = "bar"
        return {
            "chart_type": chart_type,
            "confidence": 0.8,
            "config": {"title": f"Data Visualization for: {query[:50]}...", "xAxis": headers[0] if headers else "", "yAxis": headers[1:] if len(headers) > 1 else []}
        }

    def _infer_first_numeric_key(self, data: List[dict]) -> str:
        if not data:
            return "Value"
        first = data[0]
        for k, v in first.items():
            if isinstance(v, (int, float)):
                return k
        # default
        return "Value"


class WorkflowEngine:
    """Main workflow engine for orchestrating agentic workflows"""
    
    def __init__(self):
        self.agents: Dict[AgentType, BaseAgent] = {}
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.logger = logging.getLogger("workflow_engine")
        
        # Register default agents
        self._register_default_agents()
        
        # Register default workflows
        self._register_default_workflows()
        
        # Register default event handlers
        self._register_default_event_handlers()
    
    def _register_default_agents(self):
        """Register default agents"""
        self.register_agent(QueryAnalysisAgent())
        self.register_agent(SQLGenerationAgent())
        self.register_agent(ValidationAgent())
        self.register_agent(ExecutionAgent())
        self.register_agent(VisualizationAgent())
    
    def _register_default_workflows(self):
        """Register default workflows"""
        # Standard query processing workflow
        standard_workflow = WorkflowDefinition(
            workflow_id="standard_query_processing",
            name="Standard Query Processing",
            description="Standard workflow for processing natural language queries",
            steps=[
                WorkflowStep(
                    step_id="query_analysis",
                    agent_type=AgentType.QUERY_ANALYSIS,
                    name="Query Analysis",
                    description="Analyze natural language query for intent and entities"
                ),
                WorkflowStep(
                    step_id="sql_generation",
                    agent_type=AgentType.SQL_GENERATION,
                    name="SQL Generation",
                    description="Generate SQL query based on analysis",
                    dependencies=["query_analysis"]
                ),
                WorkflowStep(
                    step_id="validation",
                    agent_type=AgentType.VALIDATION,
                    name="SQL Validation",
                    description="Validate generated SQL",
                    dependencies=["sql_generation"],
                    required=False  # Make validation optional
                ),
                WorkflowStep(
                    step_id="execution",
                    agent_type=AgentType.EXECUTION,
                    name="SQL Execution",
                    description="Execute validated SQL",
                    dependencies=["sql_generation"]  # Can execute even if validation fails
                ),
                WorkflowStep(
                    step_id="visualization",
                    agent_type=AgentType.VISUALIZATION,
                    name="Visualization",
                    description="Generate visualization recommendations",
                    dependencies=["execution"]
                )
            ]
        )
        
        self.register_workflow(standard_workflow)
    
    def _register_default_event_handlers(self):
        """Register default event handlers"""
        self.register_event_handler(EventType.QUERY_RECEIVED, self._log_query_received)
        self.register_event_handler(EventType.WORKFLOW_COMPLETE, self._log_workflow_complete)
        self.register_event_handler(EventType.ERROR_OCCURRED, self._log_error)
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent"""
        self.agents[agent.agent_type] = agent
        self.logger.info(f"Registered agent: {agent.name} ({agent.agent_type})")
    
    def register_workflow(self, workflow: WorkflowDefinition):
        """Register a workflow"""
        self.workflows[workflow.workflow_id] = workflow
        self.logger.info(f"Registered workflow: {workflow.name} ({workflow.workflow_id})")
    
    def register_event_handler(self, event_type: EventType, handler: Callable):
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def execute_workflow(self, workflow_id: str, context: WorkflowContext) -> Dict[str, Any]:
        """Execute a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        self.logger.info(f"Starting workflow: {workflow.name} ({workflow.workflow_id})")
        
        # Emit workflow start event
        await self._emit_event(EventType.QUERY_RECEIVED, context)
        
        try:
            # Execute steps
            results = {}
            for step in workflow.steps:
                step_result = await self._execute_step(step, context)
                results[step.step_id] = step_result
                
                if not step_result.success and step.required:
                    self.logger.error(f"Required step {step.name} failed")
                    await self._emit_event(EventType.ERROR_OCCURRED, context, error=step_result.error)
                    break
            
            # Emit workflow complete event
            await self._emit_event(EventType.WORKFLOW_COMPLETE, context)
            
            return {
                "workflow_id": workflow_id,
                "success": all(r.success for r in results.values()),
                "results": results,
                "execution_time": time.time() - context.start_time.timestamp(),
                "context": context
            }
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            await self._emit_event(EventType.ERROR_OCCURRED, context, error=str(e))
            raise
    
    async def _execute_step(self, step: WorkflowStep, context: WorkflowContext) -> AgentResult:
        """Execute a single workflow step"""
        self.logger.info(f"Executing step: {step.name}")
        
        # Check dependencies
        for dep in step.dependencies:
            if dep not in context.state:
                return AgentResult(
                    success=False,
                    data={},
                    error=f"Dependency {dep} not satisfied"
                )
        
        # Get agent
        agent = self.agents.get(step.agent_type)
        if not agent:
            return AgentResult(
                success=False,
                data={},
                error=f"Agent {step.agent_type} not found"
            )
        
        # Execute agent with retries
        for attempt in range(step.retry_count):
            try:
                result = await agent.execute(context)
                if result.success:
                    return result
                elif attempt < step.retry_count - 1:
                    await asyncio.sleep(step.retry_delay)
                    continue
                else:
                    return result
            except Exception as e:
                if attempt < step.retry_count - 1:
                    await asyncio.sleep(step.retry_delay)
                    continue
                else:
                    return AgentResult(
                        success=False,
                        data={},
                        error=str(e)
                    )
        
        return AgentResult(success=False, data={}, error="Max retries exceeded")
    
    async def _emit_event(self, event_type: EventType, context: WorkflowContext, **kwargs):
        """Emit an event"""
        event = {
            "event_type": event_type,
            "workflow_id": context.workflow_id,
            "session_id": context.session_id,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        context.events.append(event)
        
        # Call event handlers
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event, context)
                else:
                    handler(event, context)
            except Exception as e:
                self.logger.error(f"Event handler failed: {e}")
    
    async def _log_query_received(self, event: Dict[str, Any], context: WorkflowContext):
        """Log query received event"""
        self.logger.info(f"Query received: {context.query[:100]}...")
    
    async def _log_workflow_complete(self, event: Dict[str, Any], context: WorkflowContext):
        """Log workflow complete event"""
        execution_time = time.time() - context.start_time.timestamp()
        self.logger.info(f"Workflow completed in {execution_time:.2f}s")
    
    async def _log_error(self, event: Dict[str, Any], context: WorkflowContext):
        """Log error event"""
        error = event.get("error", "Unknown error")
        self.logger.error(f"Workflow error: {error}")


# Global workflow engine instance
workflow_engine = WorkflowEngine() 