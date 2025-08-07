"""
Enhanced RAG Service with Wren AI Integration
Implements advanced semantic processing with MDL support and vector search
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

from backend.core.types import QueryAnalysis, IntentType, QueryType, ProcessingMode
from backend.core.llm_provider import LLMProvider, create_llm_provider
from backend.core.semantic_engine import SemanticEngine
from backend.core.wren_ai_integration import WrenAIIntegration
from backend.core.assembler import SQLAssembler
from backend.core.validator import EnhancedSQLValidator
from backend.core.executor import AsyncSQLExecutor
from backend.config import get_settings

logger = logging.getLogger(__name__)


class EnhancedRAGService:
    """
    Enhanced RAG Service with Wren AI Integration
    Combines semantic processing, MDL support, and advanced vector search
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.settings = get_settings()
        
        # Initialize LLM provider
        self.llm_provider = create_llm_provider(
            provider_type=self.settings.llm_provider_type,
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
            base_url=self.settings.llm_base_url,
            enable_gpu=self.settings.enable_gpu_acceleration
        )
        
        # Initialize semantic components
        self.semantic_engine = SemanticEngine(self.llm_provider, db_path)
        self.wren_ai_integration = WrenAIIntegration(self.llm_provider)
        
        # Initialize core components
        self.sql_assembler = SQLAssembler(self.llm_provider)
        self.sql_validator = EnhancedSQLValidator()
        self.sql_executor = AsyncSQLExecutor(db_path)
        
        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "semantic_enhancement_rate": 0.0,
            "average_response_time": 0.0,
            "mdl_usage_rate": 0.0,
            "vector_search_success_rate": 0.0
        }
        
        self._initialized = False
        
    async def initialize(self):
        """Initialize all components"""
        if self._initialized:
            return
            
        try:
            # Initialize semantic engine
            await self.semantic_engine.initialize()
            
            # Initialize Wren AI integration
            await self.wren_ai_integration.initialize()
            
            self._initialized = True
            logger.info("Enhanced RAG Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced RAG Service: {e}")
            raise
            
    async def process_query_enhanced(self, query: str, processing_mode: str = "adaptive") -> Dict[str, Any]:
        """
        Process query with enhanced semantic understanding and Wren AI integration
        """
        start_time = datetime.now()
        
        try:
            # Ensure initialization
            if not self._initialized:
                await self.initialize()
                
            # Step 1: Extract semantic context using Wren AI integration
            semantic_context = await self.wren_ai_integration.extract_semantic_context(query)
            
            # Step 2: Determine processing mode based on confidence
            confidence = semantic_context.get("confidence", 0.0)
            actual_mode = self._determine_processing_mode(confidence, processing_mode)
            
            # Step 3: Process query based on mode
            if actual_mode == ProcessingMode.SEMANTIC_FIRST:
                result = await self._process_semantic_first(query, semantic_context)
            elif actual_mode == ProcessingMode.HYBRID:
                result = await self._process_hybrid(query, semantic_context)
            elif actual_mode == ProcessingMode.AGENTIC_WORKFLOW:
                result = await self._process_agentic(query, semantic_context)
            else:
                result = await self._process_traditional(query, semantic_context)
                
            # Step 4: Update statistics
            self._update_statistics(start_time, semantic_context, result)
            
            # Step 5: Add semantic insights
            result["semantic_insights"] = {
                "processing_mode": actual_mode.value,
                "mdl_context": semantic_context.get("mdl_context", {}),
                "business_entities": semantic_context.get("business_entities", []),
                "confidence_breakdown": {
                    "overall": confidence,
                    "mdl_understanding": semantic_context.get("mdl_context", {}).get("confidence", 0.0),
                    "vector_similarity": semantic_context.get("search_results", {}).get("similarity_score", 0.0)
                },
                "wren_ai_features": {
                    "mdl_support": True,
                    "vector_search": True,
                    "business_context": True
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced query processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_mode": processing_mode,
                "semantic_insights": {
                    "processing_mode": "fallback",
                    "error": str(e)
                }
            }
            
    async def _process_semantic_first(self, query: str, semantic_context: Dict) -> Dict[str, Any]:
        """Process query using semantic-first approach with Wren AI integration"""
        try:
            # Use Wren AI integration for MDL-aware SQL generation
            wren_result = await self.wren_ai_integration.generate_mdl_aware_sql(query, semantic_context)
            
            if wren_result.get("sql") and wren_result.get("confidence", 0.0) > 0.7:
                # Execute the generated SQL
                execution_result = await self.sql_executor.execute_sql_async(wren_result["sql"])
                
                return {
                    "success": True,
                    "sql": wren_result["sql"],
                    "data": execution_result.data,
                    "confidence": wren_result["confidence"],
                    "processing_method": "wren_ai_mdl",
                    "mdl_context": wren_result.get("mdl_context", {}),
                    "business_entities": wren_result.get("business_entities", [])
                }
            else:
                # Fallback to semantic engine
                return await self._process_with_semantic_engine(query, semantic_context)
                
        except Exception as e:
            logger.error(f"Semantic-first processing failed: {e}")
            return await self._process_with_semantic_engine(query, semantic_context)
            
    async def _process_hybrid(self, query: str, semantic_context: Dict) -> Dict[str, Any]:
        """Process query using hybrid approach combining multiple methods"""
        try:
            # Try Wren AI first
            wren_result = await self.wren_ai_integration.generate_mdl_aware_sql(query, semantic_context)
            
            # Try semantic engine
            semantic_result = await self._process_with_semantic_engine(query, semantic_context)
            
            # Choose the best result based on confidence
            if wren_result.get("confidence", 0.0) > semantic_result.get("confidence", 0.0):
                return {
                    "success": True,
                    "sql": wren_result["sql"],
                    "data": semantic_result.get("data", []),
                    "confidence": wren_result["confidence"],
                    "processing_method": "hybrid_wren_ai",
                    "mdl_context": wren_result.get("mdl_context", {}),
                    "business_entities": wren_result.get("business_entities", [])
                }
            else:
                return semantic_result
                
        except Exception as e:
            logger.error(f"Hybrid processing failed: {e}")
            return await self._process_traditional(query, semantic_context)
            
    async def _process_with_semantic_engine(self, query: str, semantic_context: Dict) -> Dict[str, Any]:
        """Process query using semantic engine"""
        try:
            # Use semantic engine for context-aware SQL generation
            semantic_result = await self.semantic_engine.generate_contextual_sql(
                query, semantic_context, {}
            )
            
            if semantic_result.get("sql"):
                # Execute the generated SQL
                execution_result = await self.sql_executor.execute_sql_async(semantic_result["sql"])
                
                return {
                    "success": True,
                    "sql": semantic_result["sql"],
                    "data": execution_result.data,
                    "confidence": semantic_result.get("confidence", 0.0),
                    "processing_method": "semantic_engine",
                    "semantic_context": semantic_context
                }
            else:
                return await self._process_traditional(query, semantic_context)
                
        except Exception as e:
            logger.error(f"Semantic engine processing failed: {e}")
            return await self._process_traditional(query, semantic_context)
            
    async def _process_traditional(self, query: str, semantic_context: Dict) -> Dict[str, Any]:
        """Process query using traditional approach"""
        try:
            # Use traditional SQL assembler
            sql_result = self.sql_assembler.generate_sql(query, None, None)
            
            if sql_result and hasattr(sql_result, 'sql') and sql_result.sql:
                # Execute the generated SQL
                execution_result = await self.sql_executor.execute_sql_async(sql_result.sql)
                
                return {
                    "success": True,
                    "sql": sql_result.sql,
                    "data": execution_result.data,
                    "confidence": 0.6,  # Lower confidence for traditional approach
                    "processing_method": "traditional",
                    "semantic_context": semantic_context
                }
            else:
                return {
                    "success": False,
                    "error": "SQL generation failed",
                    "processing_method": "traditional"
                }
                
        except Exception as e:
            logger.error(f"Traditional processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_method": "traditional"
            }
            
    async def _process_agentic(self, query: str, semantic_context: Dict) -> Dict[str, Any]:
        """Process query using agentic workflow"""
        try:
            # Import agentic service
            from backend.services.agentic_rag_service import AgenticRAGService
            
            agentic_service = AgenticRAGService(self.db_path)
            result = await agentic_service.process_query_agentic(query)
            
            # Add semantic context to result
            result["semantic_context"] = semantic_context
            result["processing_method"] = "agentic_workflow"
            
            return result
            
        except Exception as e:
            logger.error(f"Agentic processing failed: {e}")
            return await self._process_hybrid(query, semantic_context)
            
    def _determine_processing_mode(self, confidence: float, requested_mode: str) -> ProcessingMode:
        """Determine the actual processing mode based on confidence and request"""
        if requested_mode == "adaptive":
            if confidence >= 0.8:
                return ProcessingMode.SEMANTIC_FIRST
            elif confidence >= 0.6:
                return ProcessingMode.HYBRID
            elif confidence >= 0.4:
                return ProcessingMode.AGENTIC_WORKFLOW
            else:
                return ProcessingMode.TRADITIONAL
        else:
            return ProcessingMode(requested_mode)
            
    def _update_statistics(self, start_time: datetime, semantic_context: Dict, result: Dict):
        """Update service statistics"""
        self.stats["total_requests"] += 1
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        self.stats["average_response_time"] = (
            (self.stats["average_response_time"] * (self.stats["total_requests"] - 1) + response_time) 
            / self.stats["total_requests"]
        )
        
        # Update semantic enhancement rate
        if result.get("processing_method") in ["wren_ai_mdl", "semantic_engine", "hybrid_wren_ai"]:
            self.stats["semantic_enhancement_rate"] = (
                (self.stats["semantic_enhancement_rate"] * (self.stats["total_requests"] - 1) + 1) 
                / self.stats["total_requests"]
            )
            
        # Update MDL usage rate
        if semantic_context.get("mdl_context", {}).get("relevant_models"):
            self.stats["mdl_usage_rate"] = (
                (self.stats["mdl_usage_rate"] * (self.stats["total_requests"] - 1) + 1) 
                / self.stats["total_requests"]
            )
            
        # Update vector search success rate
        if semantic_context.get("search_results"):
            self.stats["vector_search_success_rate"] = (
                (self.stats["vector_search_success_rate"] * (self.stats["total_requests"] - 1) + 1) 
                / self.stats["total_requests"]
            )
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "success": True,
            "statistics": self.stats,
            "system_status": {
                "semantic_engine": "operational" if self._initialized else "initializing",
                "wren_ai_integration": "operational" if self._initialized else "initializing",
                "vector_database": "operational" if self._initialized else "initializing",
                "mdl_support": "enabled" if self._initialized else "disabled",
                "accuracy_target": "85-90%"
            }
        }