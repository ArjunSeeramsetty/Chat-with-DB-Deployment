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
from backend.core.feedback_storage import FeedbackStorage, FeedbackRecord, ExecutionTrace
from backend.core.advanced_retrieval import AdvancedRetrieval, ContextualRetrieval, RetrievalResult
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
        
        # Initialize feedback storage
        self.feedback_storage = FeedbackStorage(db_path)
        
        # Initialize advanced retrieval system
        self.advanced_retrieval = AdvancedRetrieval()
        self.contextual_retrieval = ContextualRetrieval(self.advanced_retrieval)
        
        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "semantic_enhancement_rate": 0.0,
            "average_response_time": 0.0,
            "mdl_usage_rate": 0.0,
            "vector_search_success_rate": 0.0,
            "feedback_learning_rate": 0.0,
            "hybrid_retrieval_rate": 0.0
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
            
            # Initialize advanced retrieval with documents
            await self._initialize_advanced_retrieval()
            
            self._initialized = True
            logger.info("Enhanced RAG Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced RAG Service: {e}")
            raise
            
    async def _initialize_advanced_retrieval(self):
        """Initialize advanced retrieval system with documents"""
        try:
            # Get documents from various sources
            documents = await self._collect_documents()
            
            # Add documents to advanced retrieval system
            self.advanced_retrieval.add_documents(documents)
            
            logger.info(f"Initialized advanced retrieval with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced retrieval: {e}")
            
    async def _collect_documents(self) -> List[Dict[str, Any]]:
        """Collect documents from various sources for advanced retrieval"""
        documents = []
        
        try:
            # Add schema information
            schema_docs = await self._get_schema_documents()
            documents.extend(schema_docs)
            
            # Add business rules
            business_docs = await self._get_business_documents()
            documents.extend(business_docs)
            
            # Add historical queries
            historical_docs = await self._get_historical_documents()
            documents.extend(historical_docs)
            
            # Add feedback examples
            feedback_docs = await self._get_feedback_documents()
            documents.extend(feedback_docs)
            
            logger.info(f"Collected {len(documents)} documents for advanced retrieval")
            
        except Exception as e:
            logger.error(f"Failed to collect documents: {e}")
            
        return documents
        
    async def _get_schema_documents(self) -> List[Dict[str, Any]]:
        """Get schema-related documents"""
        documents = []
        
        try:
            # Get schema metadata
            schema_metadata = await self.semantic_engine.get_schema_metadata()
            
            for table_name, table_info in schema_metadata.items():
                content = f"Table: {table_name}\n"
                content += f"Columns: {', '.join(table_info.get('columns', []))}\n"
                content += f"Description: {table_info.get('description', '')}"
                
                documents.append({
                    "content": content,
                    "source": "schema",
                    "metadata": {
                        "table_name": table_name,
                        "type": "schema",
                        "columns": table_info.get('columns', [])
                    }
                })
                
        except Exception as e:
            logger.error(f"Failed to get schema documents: {e}")
            
        return documents
        
    async def _get_business_documents(self) -> List[Dict[str, Any]]:
        """Get business rule documents"""
        documents = []
        
        try:
            # Add business rules and domain knowledge
            business_rules = [
                {
                    "content": "Energy Met represents actual energy consumption in megawatt hours",
                    "source": "business_rules",
                    "metadata": {"type": "business_rule", "domain": "energy"}
                },
                {
                    "content": "Energy Shortage represents unmet energy demand in megawatt hours",
                    "source": "business_rules", 
                    "metadata": {"type": "business_rule", "domain": "energy"}
                },
                {
                    "content": "States are grouped into regions: Northern, Southern, Eastern, Western, North Eastern, and Central",
                    "source": "business_rules",
                    "metadata": {"type": "business_rule", "domain": "geography"}
                }
            ]
            
            documents.extend(business_rules)
            
        except Exception as e:
            logger.error(f"Failed to get business documents: {e}")
            
        return documents
        
    async def _get_historical_documents(self) -> List[Dict[str, Any]]:
        """Get historical query documents"""
        documents = []
        
        try:
            # Get historical queries from feedback storage
            # This would be populated as the system is used
            pass
            
        except Exception as e:
            logger.error(f"Failed to get historical documents: {e}")
            
        return documents
        
    async def _get_feedback_documents(self) -> List[Dict[str, Any]]:
        """Get feedback-based documents"""
        documents = []
        
        try:
            # Get successful feedback examples
            # This would be populated as the system is used
            pass
            
        except Exception as e:
            logger.error(f"Failed to get feedback documents: {e}")
            
        return documents
        
    async def advanced_retrieve(self, query: str, context: Dict[str, Any] = None, 
                              top_k: int = 5) -> List[RetrievalResult]:
        """
        Perform advanced retrieval using hybrid search
        """
        try:
            if context:
                # Use contextual retrieval
                results = await self.contextual_retrieval.retrieve_with_context(
                    query, context, top_k
                )
            else:
                # Use basic hybrid search
                results = self.advanced_retrieval.hybrid_search(query, top_k)
                
            # Update statistics
            if results:
                self.stats["hybrid_retrieval_rate"] = (
                    (self.stats["hybrid_retrieval_rate"] * (self.stats["total_requests"] - 1) + 1) 
                    / self.stats["total_requests"]
                ) if self.stats["total_requests"] > 0 else 1.0
                
            return results
            
        except Exception as e:
            logger.error(f"Advanced retrieval failed: {e}")
            return []
            
    async def process_query_enhanced(self, query: str, processing_mode: str = "adaptive", 
                                   session_id: str = None, user_id: str = None) -> Dict[str, Any]:
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
                
            # Step 4: Store feedback and execution traces
            if session_id and user_id:
                await self._store_query_feedback(
                    query, result, semantic_context, session_id, user_id, start_time
                )
                
            # Step 5: Update statistics
            self._update_statistics(start_time, semantic_context, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced query processing failed: {e}")
            # Store error feedback if session_id is provided
            if session_id and user_id:
                await self._store_error_feedback(
                    query, str(e), session_id, user_id, start_time
                )
            return {
                "success": False,
                "error": str(e),
                "processing_method": "enhanced_rag"
            }
            
    async def process_feedback(self, session_id: str, feedback_data: Dict) -> Dict[str, Any]:
        """
        Process feedback for continuous learning and improvement
        """
        try:
            logger.info(f"Processing feedback for session {session_id}")
            
            # Extract feedback information
            feedback_record = FeedbackRecord(
                session_id=session_id,
                user_id=feedback_data.get("user_id", "unknown"),
                original_query=feedback_data.get("original_query", ""),
                generated_sql=feedback_data.get("generated_sql", ""),
                executed_sql=feedback_data.get("executed_sql", ""),
                feedback_text=feedback_data.get("feedback_text", ""),
                is_correct=feedback_data.get("is_correct", True),
                accuracy_rating=feedback_data.get("accuracy_rating", 0.0),
                usefulness_rating=feedback_data.get("usefulness_rating", 0.0),
                execution_time=feedback_data.get("execution_time", 0.0),
                row_count=feedback_data.get("row_count", 0),
                error_message=feedback_data.get("error_message"),
                processing_mode=feedback_data.get("processing_mode", "adaptive"),
                confidence_score=feedback_data.get("confidence_score", 0.0),
                query_complexity=feedback_data.get("query_complexity", "medium"),
                query_type=feedback_data.get("query_type", "aggregation"),
                tags=feedback_data.get("tags", []),
                created_at=datetime.now()
            )
            
            # Store feedback
            feedback_id = self.feedback_storage.store_feedback(feedback_record)
            
            # Analyze feedback for learning insights
            insights = await self._analyze_feedback_for_learning(feedback_record)
            
            # Update learning statistics
            if self.stats["total_requests"] > 0:
                self.stats["feedback_learning_rate"] = (
                    (self.stats["feedback_learning_rate"] * (self.stats["total_requests"] - 1) + 1) 
                    / self.stats["total_requests"]
                )
            else:
                self.stats["feedback_learning_rate"] = 1.0
            
            return {
                "success": True,
                "feedback_id": feedback_id,
                "insights": insights,
                "message": "Feedback processed successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to process feedback: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    async def _store_query_feedback(self, query: str, result: Dict, semantic_context: Dict,
                                  session_id: str, user_id: str, start_time: datetime):
        """Store feedback for a processed query"""
        try:
            # Create feedback record
            feedback_record = FeedbackRecord(
                session_id=session_id,
                user_id=user_id,
                original_query=query,
                generated_sql=result.get("sql", ""),
                executed_sql=result.get("executed_sql", ""),
                feedback_text="",
                is_correct=result.get("success", False),
                accuracy_rating=result.get("confidence", 0.0),
                usefulness_rating=0.0,  # Will be updated when user provides feedback
                execution_time=(datetime.now() - start_time).total_seconds(),
                row_count=len(result.get("data", [])),
                error_message=result.get("error"),
                processing_mode=result.get("processing_method", "adaptive"),
                confidence_score=result.get("confidence", 0.0),
                query_complexity=semantic_context.get("complexity", "medium"),
                query_type=semantic_context.get("query_type", "aggregation"),
                tags=semantic_context.get("tags", []),
                created_at=datetime.now()
            )
            
            # Store feedback
            self.feedback_storage.store_feedback(feedback_record)
            
            # Store execution trace
            trace = ExecutionTrace(
                session_id=session_id,
                query_id=str(feedback_record.id) if feedback_record.id else "unknown",
                step_name="query_processing",
                step_data={
                    "processing_mode": result.get("processing_method"),
                    "confidence": result.get("confidence"),
                    "semantic_context": semantic_context
                },
                execution_time=(datetime.now() - start_time).total_seconds(),
                success=result.get("success", False),
                error_message=result.get("error"),
                created_at=datetime.now()
            )
            
            self.feedback_storage.store_execution_trace(trace)
            
        except Exception as e:
            logger.error(f"Failed to store query feedback: {e}")
            
    async def _store_error_feedback(self, query: str, error_message: str,
                                  session_id: str, user_id: str, start_time: datetime):
        """Store feedback for a failed query"""
        try:
            feedback_record = FeedbackRecord(
                session_id=session_id,
                user_id=user_id,
                original_query=query,
                generated_sql="",
                executed_sql="",
                feedback_text="",
                is_correct=False,
                accuracy_rating=0.0,
                usefulness_rating=0.0,
                execution_time=(datetime.now() - start_time).total_seconds(),
                row_count=0,
                error_message=error_message,
                processing_mode="enhanced_rag",
                confidence_score=0.0,
                query_complexity="unknown",
                query_type="unknown",
                tags=[],
                created_at=datetime.now()
            )
            
            self.feedback_storage.store_feedback(feedback_record)
            
        except Exception as e:
            logger.error(f"Failed to store error feedback: {e}")
            
    async def _analyze_feedback_for_learning(self, feedback_record: FeedbackRecord) -> Dict[str, Any]:
        """Analyze feedback for learning insights"""
        try:
            insights = {
                "feedback_type": "positive" if feedback_record.is_correct else "negative",
                "accuracy_impact": feedback_record.accuracy_rating,
                "usefulness_impact": feedback_record.usefulness_rating,
                "processing_mode_effectiveness": {},
                "error_patterns": [],
                "improvement_suggestions": []
            }
            
            # Get similar feedback for pattern analysis
            similar_feedback = self.feedback_storage.get_similar_feedback(
                feedback_record.original_query, limit=3
            )
            
            if similar_feedback:
                insights["similar_queries"] = [
                    {
                        "query": f.original_query,
                        "success": f.is_correct,
                        "accuracy": f.accuracy_rating
                    }
                    for f in similar_feedback
                ]
                
            # Get performance analytics
            analytics = self.feedback_storage.get_performance_analytics(days=7)
            insights["recent_performance"] = analytics
            
            # Get learning insights
            learning_insights = self.feedback_storage.get_learning_insights()
            insights.update(learning_insights)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to analyze feedback for learning: {e}")
            return {}
            
    async def get_feedback_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive feedback analytics"""
        try:
            analytics = self.feedback_storage.get_performance_analytics(days)
            insights = self.feedback_storage.get_learning_insights()
            
            return {
                "success": True,
                "analytics": analytics,
                "insights": insights,
                "feedback_learning_rate": self.stats["feedback_learning_rate"]
            }
            
        except Exception as e:
            logger.error(f"Failed to get feedback analytics: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    async def get_similar_feedback(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Get similar feedback for a query"""
        try:
            similar_feedback = self.feedback_storage.get_similar_feedback(query, limit)
            
            return {
                "success": True,
                "similar_feedback": [
                    {
                        "query": f.original_query,
                        "sql": f.generated_sql,
                        "success": f.is_correct,
                        "accuracy": f.accuracy_rating,
                        "created_at": f.created_at.isoformat() if f.created_at else None
                    }
                    for f in similar_feedback
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get similar feedback: {e}")
            return {
                "success": False,
                "error": str(e)
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
        if self.stats["total_requests"] > 0:
            self.stats["average_response_time"] = (
                (self.stats["average_response_time"] * (self.stats["total_requests"] - 1) + response_time) 
                / self.stats["total_requests"]
            )
        
        # Update semantic enhancement rate
        if result.get("processing_method") in ["wren_ai_mdl", "semantic_engine", "hybrid_wren_ai"]:
            if self.stats["total_requests"] > 0:
                self.stats["semantic_enhancement_rate"] = (
                    (self.stats["semantic_enhancement_rate"] * (self.stats["total_requests"] - 1) + 1) 
                    / self.stats["total_requests"]
                )
            
        # Update MDL usage rate
        if semantic_context.get("mdl_context", {}).get("relevant_models"):
            if self.stats["total_requests"] > 0:
                self.stats["mdl_usage_rate"] = (
                    (self.stats["mdl_usage_rate"] * (self.stats["total_requests"] - 1) + 1) 
                    / self.stats["total_requests"]
                )
            
        # Update vector search success rate
        if semantic_context.get("search_results"):
            if self.stats["total_requests"] > 0:
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