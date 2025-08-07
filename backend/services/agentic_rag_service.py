"""
Agentic RAG Service for Phase 2: Agentic Workflow Implementation
Integrates the Motia-inspired workflow engine with enhanced semantic processing
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from backend.config import get_settings
from backend.core.agentic_framework import (
    WorkflowEngine, WorkflowContext, AgentResult, 
    AgentType, EventType, workflow_engine
)
from backend.core.semantic_processor import SemanticQueryProcessor, EnhancedQueryResult
from backend.core.llm_provider import create_llm_provider
from backend.core.assembler import SQLAssembler
from backend.core.schema_linker import SchemaLinker
from backend.core.validator import EnhancedSQLValidator
from backend.core.intent import IntentAnalyzer
from backend.core.executor import AsyncSQLExecutor
from backend.core.types import (
    QueryRequest, QueryResponse, ExecutionResult,
    VisualizationRecommendation, ProcessingMode
)

logger = logging.getLogger(__name__)


@dataclass
class AgenticProcessingResult:
    """Result of agentic query processing"""
    query_response: QueryResponse
    workflow_results: Dict[str, AgentResult]
    workflow_context: WorkflowContext
    processing_metrics: Dict[str, Any]
    agent_insights: Dict[str, Any]
    recommendations: List[str]


class AgenticRAGService:
    """
    Agentic RAG service that integrates workflow engine with semantic processing
    Provides step-based, event-driven query processing with specialized agents
    """
    
    def __init__(self, db_path: str, llm_provider=None):
        self.db_path = db_path
        self.settings = get_settings()
        
        # Initialize LLM provider
        if llm_provider:
            self.llm_provider = llm_provider
        else:
            self.llm_provider = create_llm_provider(
                provider_type=self.settings.llm_provider_type,
                api_key=self.settings.llm_api_key,
                model=self.settings.llm_model,
                base_url=self.settings.llm_base_url,
                enable_gpu=getattr(self.settings, 'enable_gpu_acceleration', False)
            )
        
        # Initialize core components
        schema_info = self._get_schema_info()
        self.schema_linker = SchemaLinker(schema_info, db_path, self.llm_provider)
        self.intent_analyzer = IntentAnalyzer()
        self.sql_assembler = SQLAssembler()
        self.query_validator = EnhancedSQLValidator()
        self.sql_executor = AsyncSQLExecutor(db_path)
        
        # Initialize semantic processor
        self.semantic_processor = SemanticQueryProcessor(
            llm_provider=self.llm_provider,
            schema_linker=self.schema_linker,
            sql_assembler=self.sql_assembler,
            query_validator=self.query_validator,
            intent_analyzer=self.intent_analyzer
        )
        
        # Initialize workflow engine
        self.workflow_engine = workflow_engine
        
        # Processing statistics
        self.service_stats = {
            "total_requests": 0,
            "agentic_workflows": 0,
            "workflow_success_rate": 0.0,
            "average_workflow_time": 0.0,
            "agent_performance": {},
            "event_counts": {}
        }
        
        # Register custom event handlers
        self._register_custom_event_handlers()
    
    def _get_schema_info(self) -> Dict[str, List[str]]:
        """Get schema information for the database"""
        try:
            import sqlite3
            schema_info = {}
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    # Get columns for each table
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = [row[1] for row in cursor.fetchall()]
                    schema_info[table] = columns
                    
            return schema_info
        except Exception as e:
            logger.warning(f"Could not load schema info: {e}")
            return {}
    
    def _register_custom_event_handlers(self):
        """Register custom event handlers for monitoring and analytics"""
        self.workflow_engine.register_event_handler(
            EventType.QUERY_RECEIVED, 
            self._handle_query_received
        )
        self.workflow_engine.register_event_handler(
            EventType.WORKFLOW_COMPLETE, 
            self._handle_workflow_complete
        )
        self.workflow_engine.register_event_handler(
            EventType.ERROR_OCCURRED, 
            self._handle_error_occurred
        )
    
    async def _handle_query_received(self, event: Dict[str, Any], context: WorkflowContext):
        """Handle query received event"""
        self.service_stats["total_requests"] += 1
        self.service_stats["event_counts"]["query_received"] = \
            self.service_stats["event_counts"].get("query_received", 0) + 1
        
        logger.info(f"Agentic query received: {context.query[:100]}...")
    
    async def _handle_workflow_complete(self, event: Dict[str, Any], context: WorkflowContext):
        """Handle workflow complete event"""
        self.service_stats["agentic_workflows"] += 1
        self.service_stats["event_counts"]["workflow_complete"] = \
            self.service_stats["event_counts"].get("workflow_complete", 0) + 1
        
        execution_time = time.time() - context.start_time.timestamp()
        self.service_stats["average_workflow_time"] = (
            (self.service_stats["average_workflow_time"] * (self.service_stats["agentic_workflows"] - 1) + execution_time) 
            / self.service_stats["agentic_workflows"]
        )
        
        logger.info(f"Agentic workflow completed in {execution_time:.2f}s")
    
    async def _handle_error_occurred(self, event: Dict[str, Any], context: WorkflowContext):
        """Handle error occurred event"""
        self.service_stats["event_counts"]["error_occurred"] = \
            self.service_stats["event_counts"].get("error_occurred", 0) + 1
        
        error = event.get("error", "Unknown error")
        logger.error(f"Agentic workflow error: {error}")
    
    async def process_query_agentic(
        self, 
        request: QueryRequest,
        workflow_id: str = "standard_query_processing"
    ) -> AgenticProcessingResult:
        """
        Process query using agentic workflow approach
        
        Args:
            request: Query request with natural language query
            workflow_id: ID of the workflow to execute
            
        Returns:
            AgenticProcessingResult with workflow results and insights
        """
        start_time = time.time()
        
        # Create workflow context
        context = WorkflowContext(
            user_id=request.user_id,
            query=request.question,
            session_id=request.session_id or str(uuid.uuid4()),
            metadata={
                "processing_mode": request.processing_mode.value,
                "workflow_id": workflow_id,
                "request_id": request.correlation_id
            }
        )
        
        logger.info(f"🚀 Starting agentic workflow: {workflow_id}")
        
        try:
            # Execute workflow
            workflow_result = await self.workflow_engine.execute_workflow(workflow_id, context)
            
            # Extract results from workflow
            query_response = await self._build_query_response(workflow_result, context)
            
            # Calculate processing metrics
            processing_metrics = self._calculate_processing_metrics(workflow_result, context)
            
            # Generate agent insights
            agent_insights = self._extract_agent_insights(workflow_result, context)
            
            # Generate recommendations
            recommendations = await self._generate_agentic_recommendations(workflow_result, context)
            
            # Update service statistics
            self._update_service_stats(processing_metrics, workflow_result)
            
            result = AgenticProcessingResult(
                query_response=query_response,
                workflow_results=workflow_result["results"],
                workflow_context=context,
                processing_metrics=processing_metrics,
                agent_insights=agent_insights,
                recommendations=recommendations
            )
            
            logger.info(f"✅ Agentic workflow completed successfully in {processing_metrics['total_time']:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Agentic workflow failed: {e}", exc_info=True)
            
            # Generate fallback response
            fallback_response = await self._generate_fallback_response(request, context, str(e))
            
            return AgenticProcessingResult(
                query_response=fallback_response,
                workflow_results={},
                workflow_context=context,
                processing_metrics={
                    "total_time": time.time() - start_time,
                    "error": True,
                    "error_message": str(e)
                },
                agent_insights={"error": str(e)},
                recommendations=["Please try rephrasing your query for better results"]
            )
    
    async def _build_query_response(self, workflow_result: Dict[str, Any], context: WorkflowContext) -> QueryResponse:
        """Build query response from workflow results"""
        
        # Extract data from workflow results
        results = workflow_result.get("results", {})
        
        # Get SQL from SQL generation step
        sql_generation = results.get("sql_generation", {})
        sql = sql_generation.get("data", {}).get("sql", "") if sql_generation.get("success") else ""
        
        # Get execution data
        execution = results.get("execution", {})
        data = execution.get("data", {}).get("data", []) if execution.get("success") else []
        row_count = execution.get("data", {}).get("row_count", 0) if execution.get("success") else 0
        
        # Get visualization
        visualization = results.get("visualization", {})
        viz_data = visualization.get("data", {}) if visualization.get("success") else {}
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(results)
        
        # Build visualization recommendation
        viz_recommendation = None
        if viz_data:
            viz_recommendation = VisualizationRecommendation(
                chart_type=viz_data.get("chart_type", "bar"),
                config=viz_data.get("config", {}),
                confidence=viz_data.get("confidence", 0.0),
                reasoning=f"AI-powered recommendation from {visualization.get('metadata', {}).get('chart_type', 'unknown')} agent"
            )
        
        return QueryResponse(
            sql=sql,
            data=data,
            visualization=viz_recommendation,
            explanation=self._generate_explanation(results),
            confidence=confidence,
            execution_time=workflow_result.get("execution_time", 0.0),
            session_id=context.session_id,
            processing_mode=ProcessingMode.AGENTIC_WORKFLOW,
            context_info=self._build_context_info(results, context),
            row_count=row_count
        )
    
    def _calculate_overall_confidence(self, results: Dict[str, AgentResult]) -> float:
        """Calculate overall confidence from agent results"""
        if not results:
            return 0.0
        
        confidences = [result.confidence for result in results.values() if result.success]
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _generate_explanation(self, results: Dict[str, AgentResult]) -> str:
        """Generate explanation from workflow results"""
        explanations = []
        
        for step_id, result in results.items():
            if result.success:
                step_name = step_id.replace("_", " ").title()
                confidence = result.confidence
                explanations.append(f"{step_name}: {confidence:.1%} confidence")
        
        if explanations:
            return f"Query processed successfully through agentic workflow: {'; '.join(explanations)}"
        else:
            return "Query processing completed with some issues"
    
    def _build_context_info(self, results: Dict[str, AgentResult], context: WorkflowContext) -> Dict[str, Any]:
        """Build context information from workflow results"""
        return {
            "workflow_id": context.workflow_id,
            "session_id": context.session_id,
            "processing_mode": "agentic_workflow",
            "agent_results": {
                step_id: {
                    "success": result.success,
                    "confidence": result.confidence,
                    "execution_time": result.execution_time
                }
                for step_id, result in results.items()
            },
            "workflow_events": len(context.events),
            "workflow_errors": len(context.errors)
        }
    
    def _calculate_processing_metrics(self, workflow_result: Dict[str, Any], context: WorkflowContext) -> Dict[str, Any]:
        """Calculate comprehensive processing metrics"""
        results = workflow_result.get("results", {})
        
        metrics = {
            "total_time": workflow_result.get("execution_time", 0.0),
            "workflow_success": workflow_result.get("success", False),
            "steps_completed": len([r for r in results.values() if r.success]),
            "total_steps": len(results),
            "agent_performance": {}
        }
        
        # Calculate agent-specific metrics
        for step_id, result in results.items():
            metrics["agent_performance"][step_id] = {
                "success": result.success,
                "confidence": result.confidence,
                "execution_time": result.execution_time,
                "error": result.error
            }
        
        return metrics
    
    def _extract_agent_insights(self, workflow_result: Dict[str, Any], context: WorkflowContext) -> Dict[str, Any]:
        """Extract insights from agent execution"""
        results = workflow_result.get("results", {})
        
        insights = {
            "workflow_id": context.workflow_id,
            "total_agents": len(results),
            "successful_agents": len([r for r in results.values() if r.success]),
            "agent_breakdown": {},
            "performance_insights": []
        }
        
        # Analyze each agent's performance
        for step_id, result in results.items():
            agent_name = step_id.replace("_", " ").title()
            insights["agent_breakdown"][agent_name] = {
                "success": result.success,
                "confidence": result.confidence,
                "execution_time": result.execution_time,
                "error": result.error
            }
            
            if result.success:
                insights["performance_insights"].append(
                    f"{agent_name} completed successfully with {result.confidence:.1%} confidence"
                )
            else:
                insights["performance_insights"].append(
                    f"{agent_name} failed: {result.error}"
                )
        
        return insights
    
    async def _generate_agentic_recommendations(
        self, 
        workflow_result: Dict[str, Any], 
        context: WorkflowContext
    ) -> List[str]:
        """Generate recommendations based on agentic workflow results"""
        recommendations = []
        results = workflow_result.get("results", {})
        
        # Check overall workflow success
        if not workflow_result.get("success", False):
            recommendations.append(
                "Workflow execution encountered issues. Consider rephrasing your query."
            )
        
        # Check individual agent performance
        for step_id, result in results.items():
            if not result.success:
                step_name = step_id.replace("_", " ").title()
                recommendations.append(
                    f"{step_name} step failed. This may indicate issues with query complexity or data availability."
                )
            elif result.confidence < 0.6:
                step_name = step_id.replace("_", " ").title()
                recommendations.append(
                    f"{step_name} had low confidence ({result.confidence:.1%}). Consider being more specific."
                )
        
        # Add general recommendations
        if not recommendations:
            recommendations.append(
                "Query processed successfully through agentic workflow. All agents performed well."
            )
        
        return recommendations[:3]  # Limit to top 3 recommendations
    
    def _update_service_stats(self, metrics: Dict[str, Any], workflow_result: Dict[str, Any]):
        """Update service statistics"""
        self.service_stats["total_requests"] += 1
        
        if workflow_result.get("success", False):
            self.service_stats["agentic_workflows"] += 1
        
        # Update success rate
        total_workflows = self.service_stats["agentic_workflows"]
        if total_workflows > 0:
            self.service_stats["workflow_success_rate"] = (
                len([w for w in range(total_workflows) if workflow_result.get("success", False)]) / total_workflows
            )
        
        # Update agent performance
        for step_id, result in workflow_result.get("results", {}).items():
            if step_id not in self.service_stats["agent_performance"]:
                self.service_stats["agent_performance"][step_id] = {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "average_confidence": 0.0,
                    "average_execution_time": 0.0
                }
            
            agent_stats = self.service_stats["agent_performance"][step_id]
            agent_stats["total_executions"] += 1
            
            if result.success:
                agent_stats["successful_executions"] += 1
            
            # Update averages
            current_avg_conf = agent_stats["average_confidence"]
            current_avg_time = agent_stats["average_execution_time"]
            total_execs = agent_stats["total_executions"]
            
            agent_stats["average_confidence"] = (
                (current_avg_conf * (total_execs - 1) + result.confidence) / total_execs
            )
            agent_stats["average_execution_time"] = (
                (current_avg_time * (total_execs - 1) + result.execution_time) / total_execs
            )
    
    async def _generate_fallback_response(
        self, 
        request: QueryRequest, 
        context: WorkflowContext, 
        error: str
    ) -> QueryResponse:
        """Generate fallback response when workflow fails"""
        return QueryResponse(
            sql="-- Query processing failed",
            data=[],
            visualization=None,
            explanation=f"Agentic workflow failed: {error}",
            confidence=0.0,
            execution_time=0.0,
            session_id=context.session_id,
            processing_mode=ProcessingMode.FALLBACK,
            context_info={"error": error, "workflow_id": context.workflow_id},
            error=error
        )
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        total = self.service_stats["total_requests"]
        if total == 0:
            return {"message": "No agentic workflows processed yet"}
        
        return {
            "total_requests": total,
            "agentic_workflows": self.service_stats["agentic_workflows"],
            "workflow_success_rate": self.service_stats["workflow_success_rate"],
            "average_workflow_time": self.service_stats["average_workflow_time"],
            "agent_performance": self.service_stats["agent_performance"],
            "event_counts": self.service_stats["event_counts"],
            "workflow_definitions": list(self.workflow_engine.workflows.keys())
        }
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a specific workflow"""
        if workflow_id not in self.workflow_engine.workflows:
            return {"error": f"Workflow {workflow_id} not found"}
        
        workflow = self.workflow_engine.workflows[workflow_id]
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "steps": [
                {
                    "step_id": step.step_id,
                    "name": step.name,
                    "description": step.description,
                    "dependencies": step.dependencies,
                    "required": step.required
                }
                for step in workflow.steps
            ],
            "max_execution_time": workflow.max_execution_time,
            "parallel_execution": workflow.parallel_execution
        } 