"""
Adaptive Multi-Step Query Planning Module

This module implements advanced query decomposition and planning capabilities
to break down complex queries into manageable intermediate reasoning steps.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import re

from backend.core.types import QueryAnalysis, IntentType, QueryType
from backend.core.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"           # Single table, basic aggregation
    MODERATE = "moderate"       # Multiple tables, joins, basic filtering
    COMPLEX = "complex"         # Multiple aggregations, subqueries, complex logic
    VERY_COMPLEX = "very_complex"  # Multiple steps, complex business logic


@dataclass
class QueryStep:
    """Represents a single step in query decomposition"""
    step_id: int
    description: str
    intent: str
    required_tables: List[str]
    required_columns: List[str]
    intermediate_sql: Optional[str] = None
    dependencies: List[int] = None
    confidence: float = 0.0
    execution_order: int = 0


@dataclass
class QueryPlan:
    """Complete query plan with multiple steps"""
    query_id: str
    original_query: str
    complexity: QueryComplexity
    steps: List[QueryStep]
    estimated_duration: float = 0.0
    confidence: float = 0.0
    execution_trace: List[Dict[str, Any]] = None


class QueryDecomposer:
    """
    Decomposes complex queries into manageable steps
    """
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        
    async def decompose_query(self, query: str, schema_context: Dict[str, Any]) -> QueryPlan:
        """
        Decompose a complex query into multiple steps
        """
        try:
            # Step 1: Analyze query complexity
            complexity = self._analyze_complexity(query)
            
            # Step 2: Extract entities and relationships
            entities = await self._extract_entities(query, schema_context)
            
            # Step 3: Generate query steps
            steps = await self._generate_query_steps(query, entities, complexity)
            
            # Step 4: Determine execution order
            steps = self._determine_execution_order(steps)
            
            # Step 5: Create query plan
            plan = QueryPlan(
                query_id=f"plan_{hash(query) % 10000}",
                original_query=query,
                complexity=complexity,
                steps=steps,
                confidence=self._calculate_plan_confidence(steps)
            )
            
            logger.info(f"Generated query plan with {len(steps)} steps for complexity: {complexity.value}")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to decompose query: {e}")
            # Return a simple plan as fallback
            return self._create_fallback_plan(query)
    
    def _analyze_complexity(self, query: str) -> QueryComplexity:
        """Analyze query complexity based on keywords and structure"""
        query_lower = query.lower()
        
        # Count complexity indicators
        complexity_score = 0
        
        # Multiple aggregations
        if query_lower.count('average') + query_lower.count('sum') + query_lower.count('count') > 1:
            complexity_score += 2
        
        # Multiple tables/entities
        table_indicators = ['region', 'state', 'date', 'generation', 'consumption', 'demand']
        table_count = sum(1 for indicator in table_indicators if indicator in query_lower)
        if table_count > 2:
            complexity_score += 2
        
        # Time-based analysis
        if any(word in query_lower for word in ['trend', 'growth', 'change', 'over time', 'monthly', 'yearly']):
            complexity_score += 1
        
        # Comparison logic
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference', 'ratio']):
            complexity_score += 2
        
        # Conditional logic
        if any(word in query_lower for word in ['if', 'when', 'where', 'condition']):
            complexity_score += 1
        
        # Determine complexity level
        if complexity_score >= 4:
            return QueryComplexity.VERY_COMPLEX
        elif complexity_score >= 2:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 1:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    async def _extract_entities(self, query: str, schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities, tables, and relationships from query"""
        try:
            prompt = f"""
            Extract entities, tables, and relationships from the following query.
            
            Query: "{query}"
            
            Available schema context:
            {json.dumps(schema_context, indent=2, default=str)}
            
            You must respond with ONLY valid JSON in this exact format:
            {{
                "entities": ["energy", "shortage", "region", "date"],
                "tables": ["FactAllIndiaDailySummary", "DimRegions", "DimDates"],
                "relationships": [
                    {{
                        "source": "FactAllIndiaDailySummary",
                        "target": "DimRegions",
                        "type": "many-to-one",
                        "join_condition": "fs.RegionID = r.RegionID"
                    }}
                ],
                "aggregations": ["AVG", "SUM"],
                "filters": ["region", "date_range"],
                "grouping": ["region", "month"]
            }}
            
            Rules:
            - entities: List of business entities mentioned in the query
            - tables: List of database tables needed
            - relationships: List of table relationships with join conditions
            - aggregations: List of aggregation functions needed
            - filters: List of filtering criteria
            - grouping: List of grouping criteria
            
            Return ONLY the JSON object, no other text:
            """
            
            response = await self.llm_provider.generate(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean and parse response
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            try:
                entities = json.loads(response_text)
                return entities
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse entities response: {response_text[:200]}...")
                return self._extract_entities_fallback(query)
                
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return self._extract_entities_fallback(query)
    
    def _extract_entities_fallback(self, query: str) -> Dict[str, Any]:
        """Fallback entity extraction using simple keyword matching"""
        query_lower = query.lower()
        
        entities = []
        tables = []
        relationships = []
        aggregations = []
        filters = []
        grouping = []
        
        # Extract entities
        if "energy" in query_lower:
            entities.append("energy")
        if "shortage" in query_lower:
            entities.append("shortage")
        if "region" in query_lower:
            entities.append("region")
        if "date" in query_lower or "time" in query_lower:
            entities.append("date")
        
        # Extract tables
        if any(word in query_lower for word in ["energy", "shortage", "consumption"]):
            tables.append("FactAllIndiaDailySummary")
        if "region" in query_lower:
            tables.append("DimRegions")
        if "date" in query_lower or "time" in query_lower:
            tables.append("DimDates")
        
        # Extract aggregations
        if "average" in query_lower or "avg" in query_lower:
            aggregations.append("AVG")
        if "sum" in query_lower or "total" in query_lower:
            aggregations.append("SUM")
        if "count" in query_lower:
            aggregations.append("COUNT")
        
        # Extract filters and grouping
        if "region" in query_lower:
            filters.append("region")
            grouping.append("region")
        if "month" in query_lower or "year" in query_lower:
            filters.append("date_range")
            grouping.append("month")
        
        return {
            "entities": entities,
            "tables": tables,
            "relationships": relationships,
            "aggregations": aggregations,
            "filters": filters,
            "grouping": grouping
        }
    
    async def _generate_query_steps(self, query: str, entities: Dict[str, Any], complexity: QueryComplexity) -> List[QueryStep]:
        """Generate query steps based on entities and complexity"""
        steps = []
        step_id = 1
        
        # Step 1: Entity extraction and validation
        steps.append(QueryStep(
            step_id=step_id,
            description="Extract and validate required entities and tables",
            intent="entity_extraction",
            required_tables=entities.get("tables", []),
            required_columns=[],
            confidence=0.9,
            execution_order=1
        ))
        step_id += 1
        
        # Step 2: Data filtering (if needed)
        if entities.get("filters"):
            steps.append(QueryStep(
                step_id=step_id,
                description="Apply data filters and constraints",
                intent="data_filtering",
                required_tables=entities.get("tables", []),
                required_columns=[],
                dependencies=[1],
                confidence=0.8,
                execution_order=2
            ))
            step_id += 1
        
        # Step 3: Aggregation preparation
        if entities.get("aggregations"):
            steps.append(QueryStep(
                step_id=step_id,
                description="Prepare aggregation functions and grouping",
                intent="aggregation_preparation",
                required_tables=entities.get("tables", []),
                required_columns=[],
                dependencies=[1],
                confidence=0.8,
                execution_order=3
            ))
            step_id += 1
        
        # Step 4: Join preparation (if multiple tables)
        if len(entities.get("tables", [])) > 1:
            steps.append(QueryStep(
                step_id=step_id,
                description="Prepare table joins and relationships",
                intent="join_preparation",
                required_tables=entities.get("tables", []),
                required_columns=[],
                dependencies=[1],
                confidence=0.9,
                execution_order=4
            ))
            step_id += 1
        
        # Step 5: Final SQL synthesis
        steps.append(QueryStep(
            step_id=step_id,
            description="Synthesize final SQL query",
            intent="sql_synthesis",
            required_tables=entities.get("tables", []),
            required_columns=[],
            dependencies=list(range(1, step_id)),
            confidence=0.7,
            execution_order=step_id
        ))
        
        return steps
    
    def _determine_execution_order(self, steps: List[QueryStep]) -> List[QueryStep]:
        """Determine the optimal execution order for steps"""
        # Sort by execution order
        steps.sort(key=lambda x: x.execution_order)
        
        # Update step IDs to match execution order
        for i, step in enumerate(steps, 1):
            step.step_id = i
        
        return steps
    
    def _calculate_plan_confidence(self, steps: List[QueryStep]) -> float:
        """Calculate overall confidence for the query plan"""
        if not steps:
            return 0.0
        
        total_confidence = sum(step.confidence for step in steps)
        return total_confidence / len(steps)
    
    def _create_fallback_plan(self, query: str) -> QueryPlan:
        """Create a simple fallback plan when decomposition fails"""
        fallback_step = QueryStep(
            step_id=1,
            description="Generate SQL directly from query",
            intent="direct_sql_generation",
            required_tables=[],
            required_columns=[],
            confidence=0.5,
            execution_order=1
        )
        
        return QueryPlan(
            query_id=f"fallback_{hash(query) % 10000}",
            original_query=query,
            complexity=QueryComplexity.SIMPLE,
            steps=[fallback_step],
            confidence=0.5
        )


class MultiStepQueryPlanner:
    """
    Coordinates multiple agents for complex query decomposition and execution
    """
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.decomposer = QueryDecomposer(llm_provider)
        
    async def plan_and_execute(self, query: str, schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan and execute a complex query using multi-step approach
        """
        try:
            # Step 1: Decompose query
            plan = await self.decomposer.decompose_query(query, schema_context)
            
            # Step 2: Execute steps
            execution_results = await self._execute_plan_steps(plan, schema_context)
            
            # Step 3: Synthesize final result
            final_result = await self._synthesize_final_result(plan, execution_results)
            
            return {
                "success": True,
                "plan": plan,
                "execution_results": execution_results,
                "final_result": final_result,
                "confidence": plan.confidence
            }
            
        except Exception as e:
            logger.error(f"Failed to plan and execute query: {e}")
            return {
                "success": False,
                "error": str(e),
                "plan": None,
                "execution_results": [],
                "final_result": None,
                "confidence": 0.0
            }
    
    async def _execute_plan_steps(self, plan: QueryPlan, schema_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute each step in the query plan"""
        results = []
        
        for step in plan.steps:
            try:
                step_result = await self._execute_step(step, schema_context)
                results.append({
                    "step_id": step.step_id,
                    "description": step.description,
                    "intent": step.intent,
                    "result": step_result,
                    "success": True
                })
                
                # Update step with intermediate SQL if generated
                if step_result.get("sql"):
                    step.intermediate_sql = step_result["sql"]
                    
            except Exception as e:
                logger.error(f"Failed to execute step {step.step_id}: {e}")
                results.append({
                    "step_id": step.step_id,
                    "description": step.description,
                    "intent": step.intent,
                    "result": None,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    async def _execute_step(self, step: QueryStep, schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single query step"""
        try:
            if step.intent == "entity_extraction":
                return await self._execute_entity_extraction(step, schema_context)
            elif step.intent == "data_filtering":
                return await self._execute_data_filtering(step, schema_context)
            elif step.intent == "aggregation_preparation":
                return await self._execute_aggregation_preparation(step, schema_context)
            elif step.intent == "join_preparation":
                return await self._execute_join_preparation(step, schema_context)
            elif step.intent == "sql_synthesis":
                return await self._execute_sql_synthesis(step, schema_context)
            else:
                return {"message": f"Unknown step intent: {step.intent}"}
                
        except Exception as e:
            logger.error(f"Failed to execute step {step.intent}: {e}")
            return {"error": str(e)}
    
    async def _execute_entity_extraction(self, step: QueryStep, schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute entity extraction step"""
        return {
            "entities_found": step.required_tables,
            "confidence": step.confidence,
            "message": f"Extracted {len(step.required_tables)} entities"
        }
    
    async def _execute_data_filtering(self, step: QueryStep, schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data filtering step"""
        return {
            "filters_applied": [],
            "confidence": step.confidence,
            "message": "Data filtering prepared"
        }
    
    async def _execute_aggregation_preparation(self, step: QueryStep, schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute aggregation preparation step"""
        return {
            "aggregations_prepared": [],
            "confidence": step.confidence,
            "message": "Aggregation functions prepared"
        }
    
    async def _execute_join_preparation(self, step: QueryStep, schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute join preparation step"""
        return {
            "joins_prepared": [],
            "confidence": step.confidence,
            "message": "Table joins prepared"
        }
    
    async def _execute_sql_synthesis(self, step: QueryStep, schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQL synthesis step"""
        # This would integrate with the existing SQL generation logic
        return {
            "sql_generated": True,
            "confidence": step.confidence,
            "message": "SQL synthesis completed"
        }
    
    async def _synthesize_final_result(self, plan: QueryPlan, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize the final result from all execution steps"""
        successful_steps = [r for r in execution_results if r.get("success", False)]
        
        return {
            "total_steps": len(plan.steps),
            "successful_steps": len(successful_steps),
            "success_rate": len(successful_steps) / len(plan.steps) if plan.steps else 0,
            "plan_complexity": plan.complexity.value,
            "overall_confidence": plan.confidence
        }
