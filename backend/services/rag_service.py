"""
Enhanced RAG service with improved SQL validation and candidate ranking
"""

import logging
import re
import time
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional

from backend.config import get_settings
from backend.core.assembler import SQLAssembler
from backend.core.candidate_ranker import CandidateRanker, RankedCandidate
from backend.core.executor import AsyncSQLExecutor, SQLExecutor
from backend.core.intent import IntentAnalyzer
from backend.core.llm_provider import create_llm_provider
from backend.core.schema_linker import SchemaContext, SchemaLinker
from backend.core.sql_validator import SQLValidator as LegacySQLValidator
from backend.core.types import (
    ContextInfo,
    ExecutionResult,
    IntentType,
    ProcessingMode,
    QueryAnalysis,
    QueryRequest,
    QueryResponse,
    QueryType,
    SQLGenerationResult,
    VisualizationRecommendation,
)
from backend.core.validator import EnhancedSQLValidator, SQLSandbox

logger = logging.getLogger(__name__)


class EnhancedRAGService:
    """
    Enhanced RAG service with LLM integration hooks for improved SQL generation
    """

    def __init__(self, db_path: str, llm_provider=None, memory_service=None):
        self.db_path = db_path
        self.memory_service = memory_service

        # Load settings
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
            )

        # Initialize components
        self.intent_analyzer = IntentAnalyzer()
        self.sql_assembler = SQLAssembler(llm_provider=self.llm_provider)
        # Sprint 4: Use async executor for better performance
        self.sql_executor = SQLExecutor(db_path)
        self.async_sql_executor = AsyncSQLExecutor(db_path)
        self.sql_sandbox = SQLSandbox(db_path)

        # Initialize cache attributes first
        self._schema_cache: Optional[Dict[str, List[str]]] = None
        self._schema_cache_time: float = 0
        self._cache_ttl = self.settings.memory_cache_ttl  # Use configurable TTL

        # Initialize enhanced components
        self.schema_info = self._get_schema_info()
        # Sprint 5: Use enhanced SQL validator with auto-repair and guard-rails
        self.enhanced_validator = EnhancedSQLValidator(
            self.schema_info, llm_provider=self.llm_provider
        )
        self.sql_validator = LegacySQLValidator(
            self.schema_info
        )  # Legacy validator for backward compatibility
        # Ensure schema_info is properly typed
        schema_info_dict: Dict[str, List[str]] = self.schema_info or {}
        self.schema_linker = SchemaLinker(
            schema_info_dict, db_path, llm_provider=self.llm_provider
        )
        self.candidate_ranker = CandidateRanker(schema_info_dict)

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a natural language query through the complete pipeline

        Args:
            request: QueryRequest with question and metadata

        Returns:
            QueryResponse with results and metadata
        """
        start_time = time.time()
        correlation_id = (
            request.correlation_id if hasattr(request, "correlation_id") else None
        )

        try:
            logger.info(
                f"Starting query processing - Question: {request.question[:100]} - User: {request.user_id} - Mode: {request.processing_mode.value} - Clarification attempts: {request.clarification_attempt_count}"
            )

            # Check if we've exceeded the maximum clarification attempts
            if request.clarification_attempt_count >= 3:
                response = QueryResponse(
                    success=False,
                    error="Maximum clarification attempts reached. Proper context was not provided.",
                    clarification_question="",
                    clarification_needed=False,
                    sql_query="",
                    correlation_id=correlation_id or str(uuid.uuid4()),
                    processing_time=time.time() - start_time,
                    processing_mode=request.processing_mode,
                    query_type=QueryType.UNKNOWN,
                    api_calls=0,
                    clarification_attempt_count=request.clarification_attempt_count,
                )
                return response

            # Check if clarification answers are provided
            if request.clarification_answers and len(request.clarification_answers) > 0:
                logger.info(
                    f"Processing query with clarification answers: {request.clarification_answers}"
                )
                # Enhance the question with clarification context
                enhanced_question = self._enhance_question_with_clarification(
                    request.question, request.clarification_answers
                )
                logger.info(f"Enhanced question: {enhanced_question}")
            else:
                enhanced_question = request.question

            # === LLM Hook #1 – "Paraphrase & Disambiguate" ===
            # Temporarily disabled to test date parsing
            rewritten_question = enhanced_question
            # if self.llm_provider:
            #     try:
            #         paraphrase_prompt = "Rewrite the following question in clear, unambiguous English while preserving the original intent and entities. Only clarify if the question is truly ambiguous. Keep the same entities, metrics, and time periods mentioned in the original question."
            #         llm_response = await self.llm_provider.generate(f"{paraphrase_prompt}\n\nUser Question: {request.question}")
            #         if llm_response and llm_response.content:
            #             # Only use the rewritten question if it's significantly different and better
            #             original_words = set(request.question.lower().split())
            #             rewritten_words = set(llm_response.content.lower().split())
            #             similarity = len(original_words.intersection(rewritten_words)) / len(original_words.union(rewritten_words))
            #
            #             if similarity < 0.7:  # Only use if significantly different
            #                 rewritten_question = llm_response.content
            #                 logger.info(f"LLM Hook #1 Rewritten Question: {rewritten_question}")
            #             else:
            #                 logger.info(f"LLM Hook #1: Keeping original question (similarity: {similarity:.2f})")
            #     except Exception as e:
            #         logger.warning(f"LLM Hook #1 (Paraphrase) failed: {e}")
            #         rewritten_question = request.question  # Fallback to original

            # Use the rewritten question for intent analysis
            query_analysis = await self.intent_analyzer.analyze_intent(
                rewritten_question, self.llm_provider
            )

            # Re-extract entities from the original question to preserve accuracy
            original_entities = self.intent_analyzer._extract_entities(request.question)
            query_analysis.entities = original_entities
            logger.info(
                f"Re-extracted entities from original query: {original_entities}"
            )
            # Add original query for manual entity extraction
            query_analysis.original_query = request.question
            logger.info(
                f"Intent analysis completed - Type: {query_analysis.query_type.value}, Intent: {query_analysis.intent.value}, Confidence: {query_analysis.confidence}"
            )

            # Step 2: Build Context
            context_info = self._build_context(
                rewritten_question, query_analysis, request.user_id
            )
            logger.info(
                f"Context built - User mappings: {len(context_info.user_mappings)}, Dimension values: {len(context_info.dimension_values)}"
            )

            # Step 3: Generate SQL
            sql_result = await self._generate_sql_or_request_clarification(
                rewritten_question, query_analysis, request
            )

            # Check if clarification is needed
            if (
                hasattr(sql_result, "clarification_question")
                and sql_result.clarification_question
            ):
                attempt_count = request.clarification_attempt_count + 1
                logger.info(
                    f"Generating clarification response. Current attempt: {request.clarification_attempt_count}, New attempt: {attempt_count}"
                )
                response = QueryResponse(
                    success=False,
                    error="Query needs clarification",
                    clarification_question=sql_result.clarification_question,
                    clarification_needed=True,
                    sql_query="",
                    correlation_id=correlation_id or str(uuid.uuid4()),
                    processing_time=time.time() - start_time,
                    processing_mode=request.processing_mode,
                    query_type=query_analysis.query_type,
                    api_calls=self._count_api_calls(request.processing_mode, None),
                    clarification_attempt_count=attempt_count,
                )
                logger.info(
                    f"Generated clarification response with attempt count: {response.clarification_attempt_count}"
                )
                logger.info(f"Response object: {response}")
                return response

            # === LLM Hook #6 – "Contextual Clarifier" ===
            if not sql_result.success or (
                sql_result.confidence is not None and sql_result.confidence < 0.6
            ):
                if self.llm_provider:
                    try:
                        # Build contextual clarification prompt
                        detected_info = []
                        if query_analysis.entities:
                            detected_info.append(
                                f"Detected entities: {', '.join(query_analysis.entities)}"
                            )
                        if query_analysis.time_period:
                            detected_info.append(
                                f"Detected time period: {query_analysis.time_period}"
                            )
                        if query_analysis.query_type:
                            detected_info.append(
                                f"Query type: {query_analysis.query_type.value}"
                            )

                        detected_context = (
                            "\n".join(detected_info)
                            if detected_info
                            else "No specific entities or time periods detected."
                        )

                        clarification_prompt = f"""You are helping clarify a database query about energy data. 

ORIGINAL QUERY: "{request.question}"

WHAT WE DETECTED:
{detected_context}

Generate ONE specific clarification question that:
1. Maintains the context of the original query
2. Asks for the most critical missing information
3. Is specific and actionable
4. Focuses on ONE aspect that would resolve the ambiguity
5. Is concise and direct (no explanations or additional text)

EXAMPLES OF GOOD CLARIFICATION QUESTIONS:
- "Do you want EnergyShortage (energy not supplied) or EnergyMet (energy actually supplied) for all states in 2025?"
- "For Maximum Demand Met in June 2025, do you want the maximum value across all regions, or the maximum value for each individual region?"
- "When you say 'all states in 2025', do you want data for each month of 2025, or the total for the entire year?"

EXAMPLES OF BAD CLARIFICATION QUESTIONS:
- "Could you please provide more context or specificity?" (too vague)
- "Are we talking about energy shortages, water shortages, or something else?" (ignores context)
- "What type of data are you looking for?" (too generic)
- "Here's a clarification question: Do you want..." (includes explanatory text)

Generate ONLY the clarification question, no explanations or additional text:"""

                        llm_response = await self.llm_provider.generate(
                            clarification_prompt
                        )
                        if (
                            llm_response
                            and hasattr(llm_response, "content")
                            and llm_response.content
                        ):
                            clarification_question = llm_response.content.strip()
                            # Clean up the clarification question
                            clarification_question = self._clean_clarification_question(
                                clarification_question
                            )
                            logger.info(
                                f"LLM Hook #6: Generated clarification question: {clarification_question}"
                            )

                            response = QueryResponse(
                                success=False,
                                error="Query needs clarification",
                                clarification_question=clarification_question,
                                clarification_needed=True,
                                sql_query="",
                                correlation_id=correlation_id or str(uuid.uuid4()),
                                processing_time=time.time() - start_time,
                                processing_mode=request.processing_mode,
                                query_type=query_analysis.query_type,
                                api_calls=self._count_api_calls(
                                    request.processing_mode, None
                                ),
                                clarification_attempt_count=request.clarification_attempt_count
                                + 1,
                            )
                            logger.info(
                                f"Generated clarification response with attempt count: {response.clarification_attempt_count}"
                            )
                            return response
                    except Exception as e:
                        logger.warning(f"LLM Hook #6 (Clarification) failed: {e}")

                # If LLM clarification fails, return generic clarification request
                response = QueryResponse(
                    success=False,
                    error="Query needs clarification",
                    clarification_question="I need more specific information about your query. Could you please clarify the date, metric, or geographical area you're interested in?",
                    clarification_needed=True,
                    sql_query="",
                    correlation_id=correlation_id or str(uuid.uuid4()),
                    processing_time=time.time() - start_time,
                    processing_mode=request.processing_mode,
                    query_type=query_analysis.query_type,
                    api_calls=self._count_api_calls(request.processing_mode, None),
                    clarification_attempt_count=request.clarification_attempt_count + 1,
                )
                logger.info(
                    f"Generated clarification response with attempt count: {response.clarification_attempt_count}"
                )
                return response

            if not sql_result.success:
                # No fallback SQL generation - ask for clarification instead
                # Build contextual clarification prompt
                detected_info = []
                if query_analysis.entities:
                    detected_info.append(
                        f"Detected entities: {', '.join(query_analysis.entities)}"
                    )
                if query_analysis.time_period:
                    detected_info.append(
                        f"Detected time period: {query_analysis.time_period}"
                    )
                if query_analysis.query_type:
                    detected_info.append(
                        f"Query type: {query_analysis.query_type.value}"
                    )

                detected_context = (
                    "\n".join(detected_info)
                    if detected_info
                    else "No specific entities or time periods detected."
                )

                clarification_prompt = f"""You are helping clarify a database query about energy data. 

ORIGINAL QUERY: "{request.question}"

WHAT WE DETECTED:
{detected_context}

Generate ONE specific clarification question that:
1. Maintains the context of the original query
2. Asks for the most critical missing information
3. Is specific and actionable
4. Focuses on ONE aspect that would resolve the ambiguity
5. Is concise and direct (no explanations or additional text)

EXAMPLES OF GOOD CLARIFICATION QUESTIONS:
- "Do you want EnergyShortage (energy not supplied) or EnergyMet (energy actually supplied) for all states in 2025?"
- "For Maximum Demand Met in June 2025, do you want the maximum value across all regions, or the maximum value for each individual region?"
- "When you say 'all states in 2025', do you want data for each month of 2025, or the total for the entire year?"

EXAMPLES OF BAD CLARIFICATION QUESTIONS:
- "Could you please provide more context or specificity?" (too vague)
- "Are we talking about energy shortages, water shortages, or something else?" (ignores context)
- "What type of data are you looking for?" (too generic)
- "Here's a clarification question: Do you want..." (includes explanatory text)

Generate ONLY the clarification question, no explanations or additional text:"""

                if self.llm_provider:
                    try:
                        llm_response = await self.llm_provider.generate(
                            clarification_prompt
                        )
                        if (
                            llm_response
                            and hasattr(llm_response, "content")
                            and llm_response.content
                        ):
                            clarification_question = llm_response.content.strip()
                        else:
                            clarification_question = "I need more specific information about your query. Could you please clarify the date, metric, or geographical area you're interested in?"
                    except Exception as e:
                        logger.warning(f"LLM clarification generation failed: {e}")
                        clarification_question = "I need more specific information about your query. Could you please clarify the date, metric, or geographical area you're interested in?"
                else:
                    clarification_question = "I need more specific information about your query. Could you please clarify the date, metric, or geographical area you're interested in?"

                response = QueryResponse(
                    success=False,
                    error="Clarification required",
                    clarification_question=clarification_question,
                    sql_query="",
                    correlation_id=correlation_id or str(uuid.uuid4()),
                    processing_time=time.time() - start_time,
                    processing_mode=request.processing_mode,
                    query_type=query_analysis.query_type,
                    api_calls=self._count_api_calls(request.processing_mode, None),
                    clarification_attempt_count=request.clarification_attempt_count + 1,
                )
                logger.info(
                    f"Generated clarification response with attempt count: {response.clarification_attempt_count}"
                )
                return response

            # Step 4: Execute SQL
            logger.info(f"Executing SQL: {sql_result.sql[:200]}...")

            # Temporarily use synchronous execution to debug the issue
            try:
                execution_result = self.sql_executor.execute_sql(sql_result.sql)
                logger.info(
                    f"Sync SQL execution result: success={execution_result.success}, row_count={execution_result.row_count}, error={execution_result.error}"
                )
            except Exception as e:
                logger.error(f"Sync SQL execution failed: {e}")
                execution_result = ExecutionResult(
                    success=False,
                    error=f"Sync execution error: {str(e)}",
                    execution_time=0.0,
                    sql=sql_result.sql,
                )

            logger.info(
                f"SQL execution result: success={execution_result.success}, row_count={execution_result.row_count}, error={execution_result.error}"
            )

            if not execution_result.success:
                logger.error(f"SQL execution failed: {execution_result.error}")
                response = QueryResponse(
                    success=False,
                    error=f"SQL execution failed: {execution_result.error}",
                    sql_query=sql_result.sql,
                    correlation_id=correlation_id
                    or str(uuid.uuid4()),  # Provide default if None
                    processing_time=time.time() - start_time,
                    processing_mode=request.processing_mode,
                    query_type=query_analysis.query_type,
                    api_calls=self._count_api_calls(request.processing_mode, None),
                )
                return response

            # Step 5: Generate Visualization
            visualization = None
            if execution_result.row_count > 0:
                visualization = self._generate_visualization(
                    execution_result.data, request.question
                )

            # Step 6: Generate Suggestions
            suggestions = self._generate_suggestions(
                request.question, execution_result.data, query_analysis
            )

            # === LLM Hook #7 – "NL Result Narration" ===
            summary = "Query executed successfully."  # Default summary
            if (
                execution_result.success
                and execution_result.row_count > 0
                and self.llm_provider
            ):
                try:
                    narration_prompt = f"The query '{request.question}' returned {execution_result.row_count} rows. The first 5 are: {execution_result.data[:5]}. Provide a one-paragraph narrative summarizing the key insight."
                    llm_response = await self.llm_provider.generate(narration_prompt)
                    if llm_response and llm_response.content:
                        summary = llm_response.content
                        logger.info(f"LLM Hook #7: Generated result narration.")
                except Exception as e:
                    logger.warning(f"LLM Hook #7 (Narration) failed: {e}")

            # Step 7: Build Response
            response = QueryResponse(
                success=True,
                data=execution_result.data,
                sql_query=sql_result.sql,
                row_count=execution_result.row_count,
                correlation_id=correlation_id
                or str(uuid.uuid4()),  # Provide default if None
                processing_time=time.time() - start_time,
                processing_mode=request.processing_mode,
                query_type=query_analysis.query_type,
                intent_analysis=query_analysis.dict(),
                follow_up_suggestions=suggestions,
                visualization=visualization if visualization else None,
                plot=(
                    {
                        "chartType": visualization.chart_type,
                        "options": visualization.config,
                    }
                    if visualization
                    else None
                ),
                table=self._format_table_data(execution_result),
                api_calls=self._count_api_calls(request.processing_mode, visualization),
                confidence=sql_result.confidence,  # Include confidence score
                clarification_attempt_count=request.clarification_attempt_count,
            )

            logger.info(
                f"Query processing completed successfully - Rows: {execution_result.row_count}, Time: {response.processing_time:.3f}s"
            )
            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            response = QueryResponse(
                success=False,
                error=f"Internal server error: {str(e)}",
                sql_query="",  # Provide required field
                correlation_id=correlation_id
                or str(uuid.uuid4()),  # Provide default if None
                processing_time=time.time() - start_time,
                processing_mode=request.processing_mode,
                api_calls=0,
            )
            return response

    def _build_context(
        self, query: str, query_analysis: QueryAnalysis, user_id: str
    ) -> ContextInfo:
        """Build context information for SQL generation"""

        # Get dimension values
        dimension_values = self._get_dimension_values()

        # Map user references to dimension values
        user_mappings = self._map_user_references(query, dimension_values)

        # Get memory context
        memory_context = None
        if self.memory_service:
            try:
                memory_context = self.memory_service.get_relevant_context(
                    query, user_id
                )
            except Exception as e:
                logger.warning(f"Failed to get memory context: {str(e)}")

        # Get schema info
        schema_info = self._get_schema_info()

        # Create context with LLM provider for hooks
        context = ContextInfo(
            query_analysis=query_analysis,
            user_mappings=user_mappings,
            dimension_values=dimension_values,
            memory_context=memory_context,
            schema_info=schema_info,
            relevant_examples=[],
            schema_linker=self.schema_linker,  # Pass schema linker for business rules access
        )

        # Add LLM provider to context for hooks
        context.llm_provider = self.llm_provider

        return context

    async def _generate_sql_or_request_clarification(
        self, query: str, analysis: QueryAnalysis, request: QueryRequest
    ) -> SQLGenerationResult:
        """
        Generate SQL or request clarification based on query analysis.
        """
        try:
            query_lower = query.lower()

            # Enhanced trend analysis handling - covers both growth and aggregation queries
            is_trend_analysis = analysis.intent.value == "trend_analysis"

            if is_trend_analysis:
                logger.info(
                    f"Trend analysis query detected, using dynamic template-based generation"
                )
                print(
                    f"🔍 TREND ANALYSIS DETECTED, using dynamic template-based generation"
                )

                # Determine the type of trend analysis
                trend_type = self._determine_trend_analysis_type(query, analysis)
                logger.info(f"Trend analysis type: {trend_type}")

                return await self._generate_sql_with_templates(query, analysis)

            # For very clear queries, try direct SQL generation immediately
            clear_patterns = [
                "total energy consumption",
                "total energy met",
                "energy consumption of all states",
                "energy met of all states",
                "total energy consumption of all states",
                "energy consumption in 2024",
                "energy consumption in 2025",
                "energy met in 2024",
                "energy met in 2025",
            ]

            if any(pattern in query_lower for pattern in clear_patterns):
                logger.info(
                    f"Clear query pattern detected, trying direct SQL generation"
                )
                direct_sql = self._generate_direct_sql(query, analysis)
                if direct_sql:
                    logger.info(f"Direct SQL generation successful for clear query")
                    return SQLGenerationResult(
                        success=True,
                        sql=direct_sql,
                        error=None,
                        confidence=0.9,
                        clarification_question=None,
                    )

            # Calculate confidence
            confidence = self.sql_assembler._calculate_query_confidence(query, analysis)
            print(f"🔍 CONFIDENCE CALCULATION: {confidence:.3f} for query: '{query}'")
            logger.info(f"Query confidence: {confidence:.3f}")

            # For high confidence queries, try direct SQL generation
            if confidence >= 0.4:
                print(
                    f"🔍 HIGH CONFIDENCE ({confidence:.3f}), trying direct SQL generation"
                )
                logger.info(
                    f"High confidence ({confidence:.3f}), trying direct SQL generation"
                )
                direct_sql = self._generate_direct_sql(query, analysis)
                if direct_sql:
                    print(f"🔍 DIRECT SQL GENERATION SUCCESSFUL")
                    logger.info(f"Direct SQL generation successful")
                    return SQLGenerationResult(
                        success=True,
                        sql=direct_sql,
                        error=None,
                        confidence=confidence,
                        clarification_question=None,
                    )

            # If we have clarification answers, try direct SQL generation
            if (
                hasattr(request, "clarification_answers")
                and request.clarification_answers
            ):
                enhanced_query = self._enhance_question_with_clarification(
                    query, request.clarification_answers
                )
                logger.info(f"Enhanced query with clarifications: {enhanced_query}")

                # Try direct SQL generation with enhanced query
                direct_sql = self._generate_direct_sql(enhanced_query, analysis)
                if direct_sql:
                    logger.info(f"Direct SQL generation with clarifications successful")
                    return SQLGenerationResult(
                        success=True,
                        sql=direct_sql,
                        error=None,
                        confidence=0.8,
                        clarification_question=None,
                    )

            # Fallback to template-based generation
            sql_result = await self._generate_sql_with_templates(query, analysis)

            # === LLM Hook #6 – "Contextual Clarifier" ===
            # Check if we need clarification based on confidence or SQL generation failure
            if not sql_result.success or (
                sql_result.confidence is not None and sql_result.confidence < 0.6
            ):
                if self.llm_provider and self.settings.enable_llm_clarification:
                    try:
                        # Build contextual clarification prompt
                        detected_info = []
                        if analysis.entities:
                            detected_info.append(
                                f"Detected entities: {', '.join(analysis.entities)}"
                            )
                        if analysis.time_period:
                            detected_info.append(
                                f"Detected time period: {analysis.time_period}"
                            )
                        if analysis.query_type:
                            detected_info.append(
                                f"Query type: {analysis.query_type.value}"
                            )

                        detected_context = (
                            "\n".join(detected_info)
                            if detected_info
                            else "No specific entities or time periods detected."
                        )

                        clarification_prompt = f"""You are helping clarify a database query about energy data. 

ORIGINAL QUERY: "{query}"

WHAT WE DETECTED:
{detected_context}

Generate ONE specific clarification question that:
1. Maintains the context of the original query
2. Asks for the most critical missing information
3. Is specific and actionable
4. Focuses on ONE aspect that would resolve the ambiguity
5. Is concise and direct (no explanations or additional text)

EXAMPLES OF GOOD CLARIFICATION QUESTIONS:
- "Do you want EnergyShortage (energy not supplied) or EnergyMet (energy actually supplied) for all states in 2025?"
- "For Maximum Demand Met in June 2025, do you want the maximum value across all regions, or the maximum value for each individual region?"
- "When you say 'all states in 2025', do you want data for each month of 2025, or the total for the entire year?"

EXAMPLES OF BAD CLARIFICATION QUESTIONS:
- "Could you please provide more context or specificity?" (too vague)
- "Are we talking about energy shortages, water shortages, or something else?" (ignores context)
- "What type of data are you looking for?" (too generic)
- "Here's a clarification question: Do you want..." (includes explanatory text)

Generate ONLY the clarification question, no explanations or additional text:"""

                        llm_response = await self.llm_provider.generate(
                            clarification_prompt
                        )
                        if (
                            llm_response
                            and hasattr(llm_response, "content")
                            and llm_response.content
                        ):
                            clarification_question = llm_response.content.strip()
                            # Clean up the clarification question
                            clarification_question = self._clean_clarification_question(
                                clarification_question
                            )
                            logger.info(
                                f"LLM Hook #6: Generated clarification question: {clarification_question}"
                            )

                            return SQLGenerationResult(
                                success=False,
                                sql="",
                                error="Query needs clarification",
                                confidence=confidence,
                                clarification_question=clarification_question,
                            )
                    except Exception as e:
                        logger.warning(f"LLM Hook #6 (Clarification) failed: {e}")

                # If LLM clarification fails, return generic clarification request
                return SQLGenerationResult(
                    success=False,
                    sql="",
                    error="Query needs clarification",
                    confidence=confidence,
                    clarification_question="I need more specific information about your query. Could you please clarify the date, metric, or geographical area you're interested in?",
                )

            # If we have a valid SQL but low confidence, return it anyway
            logger.info(f"Returning SQL despite low confidence ({confidence:.3f})")
            return SQLGenerationResult(
                success=True,
                sql=sql_result.sql,
                error=sql_result.error,
                confidence=confidence,
                clarification_question=None,
            )

        except Exception as e:
            logger.error(f"Error in SQL generation: {str(e)}")
            return SQLGenerationResult(
                success=False,
                sql="",
                error=f"Error generating SQL: {str(e)}",
                confidence=0.0,
                clarification_question=None,
            )

    # === SPRINT 2: REMOVED MULTI-ATTEMPT CANDIDATE GENERATION ===
    # The following methods have been removed to implement ask-first policy:
    # - _generate_sql_candidates (removed)
    # - _generate_llm_candidates (removed)
    # - _select_best_candidate (removed)
    # - _create_validation_result (removed)

    async def _generate_llm_sql(
        self, query: str, query_analysis: QueryAnalysis, enhanced_context: str
    ) -> Optional[str]:
        """Generate SQL using LLM with enhanced context"""
        if not self.llm_provider:
            return None

        prompt = f"""
Generate a SQLite SQL query for the following natural language question:

Question: {query}

Query Analysis:
- Type: {query_analysis.query_type.value}
- Intent: {query_analysis.intent.value}
- Entities: {query_analysis.entities}
- Time Period: {query_analysis.time_period}

{enhanced_context}

Generate only the SQL query, no explanations. Ensure the query is:
1. Syntactically correct for SQLite
2. Uses the correct tables and columns from the schema
3. Handles the specific intent and entities mentioned
4. Is safe to execute (read-only operations only)

SQL Query:
"""

        try:
            response = await self.llm_provider.generate(prompt)
            # Extract SQL from response (remove markdown if present)
            sql = response.strip()
            if sql.startswith("```sql"):
                sql = sql[6:]
            if sql.endswith("```"):
                sql = sql[:-3]
            return sql.strip()
        except Exception as e:
            logger.error(f"LLM SQL generation failed: {str(e)}")
            return None

    def _get_energy_column_for_table(self, table_name: str) -> str:
        """Get the correct energy demand column name based on the table"""
        # Use schema linker instead of hard-coded mappings
        schema_info = self._get_schema_info()
        if not schema_info or table_name not in schema_info:
            return ""

        # Look for energy-related columns in the table
        energy_columns = []
        for col in schema_info[table_name].get("columns", []):
            col_lower = col.lower()
            if any(
                term in col_lower for term in ["energy", "generation", "demand", "met"]
            ):
                energy_columns.append(col)

        # Return the first energy column found, or empty string
        return energy_columns[0] if energy_columns else ""

    def _generate_visualization(
        self, data: List[Dict[str, Any]], query: str
    ) -> Optional[VisualizationRecommendation]:
        """Generate visualization recommendation using AI analysis"""
        try:
            if not data:
                return None

            # Get headers from the first row
            headers = list(data[0].keys())

            # Skip if no meaningful data
            if len(headers) < 2:
                return None

            # Debug logging
            logger.info(f"🔍 VISUALIZATION DEBUG - Headers: {headers}")
            logger.info(
                f"🔍 VISUALIZATION DEBUG - Sample data: {data[0] if data else 'No data'}"
            )
            logger.info(f"🔍 VISUALIZATION DEBUG - Query: {query}")

            # Use AI-powered chart recommendation
            ai_recommendation = self._analyze_data_and_suggest_visualization(
                headers, data, query
            )
            if ai_recommendation:
                logger.info(f"AI chart recommendation: {ai_recommendation}")
                return ai_recommendation

            # Fallback to rule-based logic if AI fails
            return self._generate_fallback_visualization(headers, data, query)

        except Exception as e:
            logger.warning(f"Visualization generation failed: {str(e)}")
            return self._generate_fallback_visualization(headers, data, query)

    def _analyze_data_and_suggest_visualization(
        self, headers: List[str], data: List[Dict[str, Any]], query: str
    ) -> Optional[VisualizationRecommendation]:
        """Use AI agent to analyze data and suggest optimal visualization"""
        try:
            # Prepare data summary for the agent
            data_summary = self._prepare_data_summary(headers, data)

            # Create a more detailed and structured prompt for the AI
            prompt = f"""You are an expert data visualization analyst. Your task is to analyze the provided query and data summary to recommend the best possible chart configuration.

QUERY: "{query}"

DATA SUMMARY:
{data_summary}

Analyze the data and determine the best visualization by following these steps:
1. **Identify Data Type**: Is this a time-series, a categorical comparison, a distribution, or a mix?
2. **Select Chart Type**: Based on the data type, choose the most appropriate chart from this list: `dualAxisBarLine`, `dualAxisLine`, `multiLine`, `line`, `bar`, `pie`.
3. **Assign Axes**:
   * Identify the best column for the X-axis (usually a date or a category).
   * Identify the primary numerical column for the Y-axis.
4. **Check for Dual-Axis Potential**:
   * Look at the 'Column Analysis' in the summary. Are there two numerical columns with vastly different scales (e.g., one in thousands and one in percentages)?
   * If yes, choose `dualAxisBarLine` or `dualAxisLine`. Assign the column with larger values to `yAxis` and the column with smaller values (like growth/percentage) to `yAxisSecondary`.
5. **Check for Grouping Potential**: Is there a second categorical column that can be used to group the data (e.g., 'SourceName')? If so, assign it to `groupBy`. This is essential for `multiLine` and `groupedBar` charts.

Provide your final recommendation in this exact JSON format. Do not include any other text or explanations.

{{
    "chartType": "your_chosen_chart_type",
    "options": {{
        "title": "A concise and descriptive title for the chart",
        "xAxis": "the_best_column_for_the_x_axis",
        "yAxis": ["the_primary_numerical_column_for_the_y_axis"],
        "yAxisSecondary": "the_secondary_numerical_column_or_null",
        "groupBy": "the_column_for_grouping_or_null",
        "description": "A brief explanation of what the chart shows."
    }}
}}
"""

            # Get AI recommendation
            response = self.llm_provider.generate_text(prompt)

            # Parse JSON response
            import json

            try:
                recommendation = json.loads(response.strip())

                # Validate the recommendation structure
                if not isinstance(recommendation, dict):
                    return None

                chart_type = recommendation.get("chartType")
                options = recommendation.get("options", {})

                if not chart_type or not options:
                    return None

                # Create visualization recommendation
                return VisualizationRecommendation(
                    chart_type=chart_type,
                    config=options,
                    confidence=0.9,
                    reasoning="AI-powered chart recommendation",
                )

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse AI response as JSON: {response}")
                return None

        except Exception as e:
            logger.error(f"Error in AI visualization analysis: {e}")
            return None

    def _prepare_data_summary(
        self, headers: List[str], data: List[Dict[str, Any]]
    ) -> str:
        """Prepare a comprehensive data summary for AI analysis"""
        if not data:
            return "No data available"

        # Analyze data characteristics
        num_rows = len(data)
        num_cols = len(headers)

        # Identify column types
        numeric_columns = []
        categorical_columns = []
        time_columns = []

        for header in headers:
            header_lower = header.lower()

            # Check for time-related columns
            if any(
                time_word in header_lower
                for time_word in ["date", "month", "year", "time", "day", "quarter"]
            ):
                time_columns.append(header)
            # Check for numeric columns
            elif any(
                num_word in header_lower
                for num_word in [
                    "value",
                    "amount",
                    "total",
                    "sum",
                    "count",
                    "percentage",
                    "growth",
                    "current",
                    "previous",
                    "generation",
                    "consumption",
                ]
            ):
                numeric_columns.append(header)
            else:
                categorical_columns.append(header)

        # Analyze data ranges for dual-axis detection
        column_analysis = []
        for col in numeric_columns:
            values = [row.get(col, 0) for row in data if row.get(col) is not None]
            if values:
                min_val = min(values)
                max_val = max(values)
                avg_val = sum(values) / len(values)
                column_analysis.append(
                    f"  - {col}: min={min_val}, max={max_val}, avg={avg_val:.2f}"
                )

        # Check for growth/percentage patterns
        growth_columns = [
            col
            for col in numeric_columns
            if any(
                word in col.lower()
                for word in ["growth", "percentage", "ratio", "rate"]
            )
        ]
        total_columns = [
            col
            for col in numeric_columns
            if not any(
                word in col.lower()
                for word in ["growth", "percentage", "ratio", "rate"]
            )
        ]

        summary = f"""
Data Summary:
- Rows: {num_rows}
- Columns: {num_cols}
- Headers: {headers}

Column Analysis:
- Time columns: {time_columns}
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}

Column Details:
{chr(10).join(column_analysis)}

Dual-Axis Analysis:
- Growth/Percentage columns: {growth_columns}
- Total/Value columns: {total_columns}
- Potential dual-axis: {'Yes' if growth_columns and total_columns else 'No'}

Sample Data (first 3 rows):
{chr(10).join([str(row) for row in data[:3]])}
"""
        return summary

    def _generate_fallback_visualization(
        self, headers: List[str], data: List[Dict[str, Any]], query: str
    ) -> Optional[VisualizationRecommendation]:
        """Fallback rule-based visualization when AI fails"""
        try:
            # Identify numeric and non-numeric columns
            numeric_columns = []
            non_numeric_columns = []

            for header in headers:
                # Check if column is likely numeric
                numeric_keywords = [
                    "value",
                    "amount",
                    "total",
                    "sum",
                    "count",
                    "percentage",
                    "growth",
                    "current",
                    "previous",
                    "generation",
                    "consumption",
                    "shortage",
                    "met",
                    "maximum",
                    "minimum",
                    "average",
                    "avg",
                    "max",
                    "min",
                ]
                non_numeric_keywords = [
                    "name",
                    "region",
                    "state",
                    "source",
                    "trend",
                    "month",
                    "year",
                    "quarter",
                    "week",
                    "day",
                    "id",
                    "type",
                    "category",
                    "description",
                    "status",
                ]

                header_lower = header.lower()
                has_numeric_keyword = any(
                    keyword in header_lower for keyword in numeric_keywords
                )
                has_non_numeric_keyword = any(
                    keyword in header_lower for keyword in non_numeric_keywords
                )

                if has_numeric_keyword or not has_non_numeric_keyword:
                    numeric_columns.append(header)
                else:
                    non_numeric_columns.append(header)

            # Determine X-axis (first non-numeric column or first column)
            x_axis = non_numeric_columns[0] if non_numeric_columns else headers[0]

            # Distribute numeric columns between primary and secondary axes
            primary_y_axis = []
            secondary_y_axis = []

            if len(numeric_columns) == 1:
                # Single numeric column - put on primary axis
                primary_y_axis = numeric_columns
            elif len(numeric_columns) == 2:
                # Two numeric columns - put both on primary axis for comparison
                primary_y_axis = numeric_columns
            elif len(numeric_columns) > 2:
                # Multiple numeric columns - distribute intelligently
                for col in numeric_columns:
                    col_lower = col.lower()
                    # Check if it's specifically a growth percentage column
                    if any(
                        keyword in col_lower
                        for keyword in [
                            "growthpercentage",
                            "growth_percentage",
                            "percentage",
                        ]
                    ):
                        secondary_y_axis.append(col)
                    # Check if it's a generation value (current or previous)
                    elif any(
                        keyword in col_lower
                        for keyword in ["generation", "current", "previous"]
                    ):
                        primary_y_axis.append(col)
                    # For other cases, use the general logic
                    elif any(
                        keyword in col_lower
                        for keyword in ["percentage", "growth", "ratio", "rate"]
                    ):
                        secondary_y_axis.append(col)
                    else:
                        primary_y_axis.append(col)

                # If secondary is empty, put one column there
                if not secondary_y_axis and len(primary_y_axis) > 1:
                    secondary_y_axis.append(primary_y_axis.pop())

            # Check for monthly multi-state data first (before row count checks)
            has_month_column = any("month" in header.lower() for header in headers)
            has_state_column = any("state" in header.lower() for header in headers)
            has_region_column = any("region" in header.lower() for header in headers)
            is_monthly_multi_state = has_month_column and (
                has_state_column or has_region_column
            )

            # Check for growth data (has current/previous/growth columns)
            has_growth_data = any(
                any(
                    growth_word in header.lower()
                    for growth_word in ["growth", "current", "previous", "percentage"]
                )
                for header in headers
            )

            if is_monthly_multi_state or has_growth_data:
                # For monthly data with multiple states/regions or growth data, use multiLine chart
                # Determine the group by column (first non-numeric, non-month column)
                group_by_column = None
                for header in headers:
                    header_lower = header.lower()
                    if not any(
                        num_word in header_lower
                        for num_word in [
                            "month",
                            "growth",
                            "current",
                            "previous",
                            "percentage",
                        ]
                    ) and not any(
                        numeric_word in header_lower
                        for numeric_word in numeric_keywords
                    ):
                        group_by_column = header
                        break

                # If no group by found, use the first column that's not the x-axis
                if not group_by_column:
                    group_by_column = headers[0] if headers[0] != x_axis else headers[1]

                config = {
                    "title": f"Monthly {numeric_columns[0] if numeric_columns else 'Data'} by {group_by_column}",
                    "xAxis": x_axis,
                    "yAxis": primary_y_axis,
                    "yAxisSecondary": secondary_y_axis,
                    "groupBy": group_by_column,
                    "description": "Multi-line chart showing monthly trends",
                    "dataType": (
                        "monthly_time_series"
                        if is_monthly_multi_state
                        else "growth_time_series"
                    ),
                }
                logger.info(f"Generated multi-line chart config: {config}")
                return VisualizationRecommendation(
                    chart_type="multiLine",
                    config=config,
                    confidence=0.9,
                    reasoning="Multi-line chart for monthly data with multiple entities",
                )

            # Determine chart type based on data characteristics
            if len(data) == 1:
                # Single row - use bar chart for single value display
                config = {
                    "title": f"{primary_y_axis[0] if primary_y_axis else 'Data'} by {x_axis}",
                    "xAxis": x_axis,
                    "yAxis": primary_y_axis,
                    "yAxisSecondary": secondary_y_axis,
                    "description": "Single data point displayed as bar chart",
                    "dataType": "single_value",
                }
                logger.info(f"Generated single-row chart config: {config}")
                return VisualizationRecommendation(
                    chart_type="bar",
                    config=config,
                    confidence=0.8,
                    reasoning="Single data point displayed as bar chart",
                )
            elif len(data) <= 10:
                # Few rows - use bar chart
                config = {
                    "title": f"{primary_y_axis[0] if primary_y_axis else 'Data'} by {x_axis}",
                    "xAxis": x_axis,
                    "yAxis": primary_y_axis,
                    "yAxisSecondary": secondary_y_axis,
                    "description": "Bar chart for categorical comparison",
                    "dataType": "categorical",
                }
                logger.info(f"Generated multi-row chart config: {config}")
                return VisualizationRecommendation(
                    chart_type="bar",
                    config=config,
                    confidence=0.8,
                    reasoning="Bar chart for categorical comparison",
                )
            elif len(data) > 10 and len(data) <= 50:
                # Medium rows - use bar chart for better readability
                config = {
                    "title": f"{primary_y_axis[0] if primary_y_axis else 'Data'} by {x_axis}",
                    "xAxis": x_axis,
                    "yAxis": primary_y_axis,
                    "yAxisSecondary": secondary_y_axis,
                    "description": "Bar chart for medium dataset comparison",
                    "dataType": "categorical",
                }
                logger.info(f"Generated medium dataset chart config: {config}")
                return VisualizationRecommendation(
                    chart_type="bar",
                    config=config,
                    confidence=0.7,
                    reasoning="Bar chart for medium dataset",
                )
            elif len(data) > 50:
                # Many rows - use line chart for trends, but only if it's time series data
                # Check all headers for time-related words, not just the first one
                is_time_series = any(
                    any(
                        time_word in header.lower()
                        for time_word in [
                            "date",
                            "month",
                            "year",
                            "time",
                            "day",
                            "quarter",
                            "block",
                        ]
                    )
                    for header in headers
                )

                if is_time_series:
                    # Regular time series
                    config = {
                        "title": f"{primary_y_axis[0] if primary_y_axis else 'Data'} Trend by {x_axis}",
                        "xAxis": x_axis,
                        "yAxis": primary_y_axis,
                        "yAxisSecondary": secondary_y_axis,
                        "description": "Line chart for trend visualization",
                        "dataType": "time_series",
                    }
                    chart_type = "line"
                else:
                    # For large non-time-series datasets, use bar chart but limit display
                    config = {
                        "title": f"{primary_y_axis[0] if primary_y_axis else 'Data'} by {x_axis} (Top 20)",
                        "xAxis": x_axis,
                        "yAxis": primary_y_axis,
                        "yAxisSecondary": secondary_y_axis,
                        "description": "Bar chart for large dataset (showing top 20)",
                        "dataType": "categorical",
                    }
                    chart_type = "bar"

                logger.info(f"Generated large dataset chart config: {config}")
                return VisualizationRecommendation(
                    chart_type=chart_type,
                    config=config,
                    confidence=0.6,
                    reasoning=f"{chart_type.title()} chart for large dataset",
                )

            return None

        except Exception as e:
            logger.warning(f"Fallback visualization generation failed: {str(e)}")
            return None

    def _generate_suggestions(
        self, query: str, data: List[Dict[str, Any]], query_analysis: QueryAnalysis
    ) -> List[str]:
        """Generate follow-up suggestions"""
        suggestions = []

        try:
            # Add suggestions based on query type
            if query_analysis.query_type == QueryType.STATE:
                suggestions.append("Compare with other states")
                suggestions.append("Show trend over time")
            elif query_analysis.query_type == QueryType.REGION:
                suggestions.append("Show regional breakdown")
                suggestions.append("Compare regions")
            elif query_analysis.query_type == QueryType.GENERATION:
                suggestions.append("Show generation mix")
                suggestions.append("Compare generation sources")

            # Add general suggestions
            suggestions.append("Show maximum values")
            suggestions.append("Show average values")

        except Exception as e:
            logger.warning(f"Suggestion generation failed: {str(e)}")

        return suggestions[:5]  # Limit to 5 suggestions

    def _format_table_data(self, execution_result: ExecutionResult) -> Dict[str, Any]:
        """Format data for frontend table display"""
        if not execution_result.data:
            return {"headers": [], "rows": [], "chartData": []}

        headers = list(execution_result.data[0].keys())
        rows = [
            [row.get(header, "") for header in headers] for row in execution_result.data
        ]

        # Include chartData for frontend charts
        chartData = execution_result.data

        return {"headers": headers, "rows": rows, "chartData": chartData}

    def _count_api_calls(
        self,
        processing_mode: ProcessingMode,
        visualization: Optional[VisualizationRecommendation],
    ) -> int:
        """Count API calls based on processing mode"""
        base_calls = 1  # SQL generation

        if processing_mode == ProcessingMode.COMPREHENSIVE:
            base_calls += 1  # Intent analysis

        if visualization:
            base_calls += 1  # Visualization generation

        return base_calls

    def _get_dimension_values(self) -> Dict[str, List[Any]]:
        """Get dimension table values"""
        # This would be implemented to fetch from database
        # For now, return empty dict
        return {}

    def _map_user_references(
        self, query: str, dimension_values: Dict[str, List[Any]]
    ) -> List[Any]:
        """Map user references to dimension values"""
        # This would be implemented to map user terms to database entities
        # For now, return empty list
        return []

    async def _get_schema_info_async(self) -> Optional[Any]:
        """Get cached schema information using async execution"""
        current_time = time.time()

        if (
            self._schema_cache is None
            or current_time - self._schema_cache_time > self._cache_ttl
        ):

            try:
                # Sprint 4: Use async schema info retrieval for better performance
                self._schema_cache = (
                    await self.async_sql_executor.get_schema_info_async()
                )
                self._schema_cache_time = current_time
            except Exception as e:
                logger.warning(f"Failed to get schema info: {str(e)}")
                return None

        return self._schema_cache

    def _get_schema_info(self) -> Optional[Any]:
        """Get cached schema information (synchronous fallback)"""
        current_time = time.time()

        if (
            self._schema_cache is None
            or current_time - self._schema_cache_time > self._cache_ttl
        ):

            try:
                self._schema_cache = self.sql_executor.get_schema_info()
                self._schema_cache_time = current_time
            except Exception as e:
                logger.warning(f"Failed to get schema info: {str(e)}")
                return None

        return self._schema_cache

    def _get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU status information"""
        if self.llm_provider and hasattr(self.llm_provider, "get_gpu_status"):
            return self.llm_provider.get_gpu_status()
        return {
            "gpu_enabled": False,
            "gpu_used": False,
            "model": (
                getattr(self.llm_provider, "model", "unknown")
                if self.llm_provider
                else "unknown"
            ),
        }

    def _clean_clarification_question(self, question: str) -> str:
        """
        Clean up clarification questions by removing quotes and extra formatting.

        Args:
            question: Raw clarification question from LLM

        Returns:
            Cleaned clarification question
        """
        if not question:
            return question

        # Remove surrounding quotes
        question = question.strip()
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1]
        elif question.startswith("'") and question.endswith("'"):
            question = question[1:-1]

        # Remove any explanatory text that might come after the question
        if "\n" in question:
            question = question.split("\n")[0]

        # Remove any trailing punctuation that might be part of the explanation
        question = question.rstrip(".")

        return question.strip()

    def _generate_schema_based_clarification(
        self, query: str, analysis: QueryAnalysis
    ) -> str:
        """
        Generate a clarification question based on available schema columns.
        This is used as a fallback when the LLM generates invalid questions.

        Args:
            query: The original user query
            analysis: The query analysis

        Returns:
            A schema-based clarification question
        """
        schema_columns = self._get_available_schema_columns()

        # Extract key terms from the query
        query_lower = query.lower()
        geographic_level = self._get_geographic_level_for_query(query_lower)

        # Determine what type of clarification is needed based on the query
        if "maximum" in query_lower and (
            "energy" in query_lower or "consumption" in query_lower
        ):
            return f"Do you want EnergyMet (energy actually supplied) or MaximumDemand (peak demand capacity) for all {geographic_level} in 2025?"

        elif "demand" in query_lower:
            # Check for demand-related columns
            demand_columns = []
            for table, columns in schema_columns.items():
                for col in columns:
                    if "demand" in col.lower():
                        demand_columns.append(col)

            if len(demand_columns) >= 2:
                return f"Do you want {demand_columns[0]} or {demand_columns[1]} for all {geographic_level} in 2025?"
            else:
                return f"Do you want DemandMet (actual demand met) or MaximumDemand (peak demand capacity) for all {geographic_level} in 2025?"

        elif "outage" in query_lower:
            # Check for outage-related columns
            outage_columns = []
            for table, columns in schema_columns.items():
                for col in columns:
                    if "outage" in col.lower():
                        outage_columns.append(col)

            if len(outage_columns) >= 3:
                return f"Do you want {outage_columns[0]}, {outage_columns[1]}, or {outage_columns[2]} for all {geographic_level} in 2025?"
            elif len(outage_columns) >= 2:
                return f"Do you want {outage_columns[0]} or {outage_columns[1]} for all {geographic_level} in 2025?"
            else:
                return f"Do you want CentralSectorOutage, StateSectorOutage, or PrivateSectorOutage for all {geographic_level} in 2025?"

        elif "shortage" in query_lower:
            # Check for shortage-related columns
            shortage_columns = []
            for table, columns in schema_columns.items():
                for col in columns:
                    if "shortage" in col.lower():
                        shortage_columns.append(col)

            if len(shortage_columns) >= 2:
                return f"Do you want {shortage_columns[0]} or {shortage_columns[1]} for all {geographic_level} in 2025?"
            else:
                return f"Do you want EnergyShortage (energy not supplied) or PowerShortage (power not supplied) for all {geographic_level} in 2025?"

        elif "generation" in query_lower:
            # Check for generation-related columns
            generation_columns = []
            for table, columns in schema_columns.items():
                for col in columns:
                    if "generation" in col.lower():
                        generation_columns.append(col)

            if len(generation_columns) >= 2:
                return f"Do you want {generation_columns[0]} or {generation_columns[1]} for all {geographic_level} in 2025?"
            else:
                return f"Do you want TotalGeneration or ThermalGeneration for all {geographic_level} in 2025?"

        elif "energy" in query_lower:
            return f"Do you want EnergyShortage (energy not supplied) or EnergyMet (energy actually supplied) for all {geographic_level} in 2025?"

        else:
            # Generic fallback
            return f"Do you want EnergyShortage (energy not supplied) or EnergyMet (energy actually supplied) for all {geographic_level} in 2025?"

    def _validate_clarification_question(self, question: str) -> bool:
        """
        Validate that a clarification question only refers to valid schema terms.

        Args:
            question: The clarification question to validate

        Returns:
            True if the question is valid, False otherwise
        """
        if not question:
            return False

        # Get available schema columns
        schema_columns = self._get_available_schema_columns()
        all_valid_terms = []
        for table, columns in schema_columns.items():
            all_valid_terms.extend(columns)

        # Check if the question contains any valid schema terms
        question_lower = question.lower()
        for term in all_valid_terms:
            if term.lower() in question_lower:
                return True

        # If no schema terms found, the question is invalid
        return False

    def _enhance_question_with_clarification(
        self, original_question: str, clarification_answers: Dict[str, str]
    ) -> str:
        """
        Enhance the original question with clarification answers to make it more specific.
        This method preserves the original context while integrating clarification information.

        Args:
            original_question: The original user question
            clarification_answers: Dictionary of clarification questions and their answers

        Returns:
            Enhanced question with clarification context
        """
        if not clarification_answers:
            return original_question

        # Build enhancement context from clarification answers
        enhancement_parts = []

        for question, answer in clarification_answers.items():
            # Extract key information from the clarification answer
            answer_lower = answer.lower()

            # Handle specific metric clarifications
            if (
                "energyshortage" in answer_lower
                or "energy not supplied" in answer_lower
                or "energy shortage" in answer_lower
            ):
                enhancement_parts.append("EnergyShortage (energy not supplied)")
            elif (
                "energymet" in answer_lower
                or "energy actually supplied" in answer_lower
                or "energy supplied" in answer_lower
                or "energy met" in answer_lower
            ):
                enhancement_parts.append("EnergyMet (energy actually supplied)")
            elif "demandmet" in answer_lower or "demand met" in answer_lower:
                enhancement_parts.append("DemandMet")
            elif "maximumdemand" in answer_lower or "maximum demand" in answer_lower:
                enhancement_parts.append("MaximumDemand")
            elif (
                "energyoutage" in answer_lower
                or "energy outage" in answer_lower
                or "total outage" in answer_lower
            ):
                enhancement_parts.append("EnergyOutage (total energy interrupted)")
            elif (
                "energyrestoration" in answer_lower
                or "energy restoration" in answer_lower
                or "restoration" in answer_lower
            ):
                enhancement_parts.append(
                    "EnergyRestoration (energy restoration completed)"
                )
            elif (
                "central sector outage" in answer_lower
                or "central sector" in answer_lower
            ):
                enhancement_parts.append("CentralSectorOutage")
            elif (
                "state sector outage" in answer_lower or "state sector" in answer_lower
            ):
                enhancement_parts.append("StateSectorOutage")
            elif (
                "private sector outage" in answer_lower
                or "private sector" in answer_lower
            ):
                enhancement_parts.append("PrivateSectorOutage")

            # Handle time granularity clarifications
            if "monthly" in answer_lower or "each month" in answer_lower:
                enhancement_parts.append("monthly data")
            elif "daily" in answer_lower or "each day" in answer_lower:
                enhancement_parts.append("daily data")
            elif (
                "yearly" in answer_lower
                or "annual" in answer_lower
                or "entire year" in answer_lower
            ):
                enhancement_parts.append("yearly data")

            # Handle aggregation clarifications
            if "maximum" in answer_lower or "max" in answer_lower:
                enhancement_parts.append("maximum values")
            elif "minimum" in answer_lower or "min" in answer_lower:
                enhancement_parts.append("minimum values")
            elif "average" in answer_lower or "avg" in answer_lower:
                enhancement_parts.append("average values")
            elif "total" in answer_lower or "sum" in answer_lower:
                enhancement_parts.append("total values")

            # Handle geographic clarifications
            if "all states" in answer_lower or "every state" in answer_lower:
                enhancement_parts.append("all states")
            elif "all regions" in answer_lower or "every region" in answer_lower:
                enhancement_parts.append("all regions")
            elif "individual" in answer_lower or "each" in answer_lower:
                enhancement_parts.append("individual values")

        if enhancement_parts:
            # Create a more natural enhanced question that preserves original context
            enhanced_question = (
                f"{original_question} [Clarified: {'; '.join(enhancement_parts)}]"
            )
            logger.info(f"Enhanced question with clarification: {enhanced_question}")
            return enhanced_question

        return original_question

    def _get_available_schema_columns(self) -> Dict[str, List[str]]:
        """
        Get available columns from the database schema.

        Returns:
            Dictionary mapping table names to lists of column names
        """
        schema_info = self._get_schema_info()
        if not schema_info:
            return {}

        schema_columns = {}
        for table_name, table_info in schema_info.items():
            if "columns" in table_info:
                schema_columns[table_name] = table_info["columns"]

        return schema_columns

    def _get_schema_constraints_for_clarification(self) -> str:
        """
        Get schema constraints for clarification questions.

        Returns:
            Formatted string of available schema columns with geographic context
        """
        schema_columns = self._get_available_schema_columns()

        constraints = []
        for table, columns in schema_columns.items():
            # Add geographic context based on table name
            if "state" in table.lower():
                constraints.append(f"{table} (STATE level data): {', '.join(columns)}")
            elif "region" in table.lower():
                constraints.append(f"{table} (REGION level data): {', '.join(columns)}")
            elif "outage" in table.lower():
                constraints.append(f"{table} (REGION level data): {', '.join(columns)}")
            else:
                constraints.append(f"{table}: {', '.join(columns)}")

        return "\n".join(constraints)

    def _build_contextual_clarification_prompt(
        self,
        query: str,
        analysis: QueryAnalysis,
        clarification_answers: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Build a contextual clarification prompt that includes previous clarification answers.

        Args:
            query: The original user query
            analysis: The query analysis
            clarification_answers: Previous clarification answers

        Returns:
            Formatted clarification prompt
        """
        # Build previous context from clarification answers
        previous_context = ""
        if clarification_answers:
            context_parts = []
            for question, answer in clarification_answers.items():
                context_parts.append(f"Q: {question}\nA: {answer}")
            previous_context = (
                f"\nPREVIOUS CLARIFICATIONS:\n" + "\n".join(context_parts) + "\n"
            )

        # Build detected context
        detected_info = []
        if analysis.entities:
            detected_info.append(f"Detected entities: {', '.join(analysis.entities)}")
        if analysis.time_period:
            detected_info.append(f"Detected time period: {analysis.time_period}")
        if analysis.query_type:
            detected_info.append(f"Query type: {analysis.query_type.value}")

        detected_context = (
            "\n".join(detected_info)
            if detected_info
            else "No specific entities or time periods detected."
        )

        # Get schema constraints
        schema_constraints = self._get_schema_constraints_for_clarification()

        # Generate context-specific examples based on query analysis
        query_lower = query.lower()
        context_specific_examples = self._get_context_specific_examples(
            query_lower, analysis
        )

        prompt = f"""You are helping clarify a database query about INDIAN energy data.

ORIGINAL QUERY: "{query}"{previous_context}

WHAT WE DETECTED:
{detected_context}

AVAILABLE SCHEMA COLUMNS (ONLY USE THESE):
{schema_constraints}

CONTEXT-SPECIFIC EXAMPLES FOR THIS QUERY:
{context_specific_examples}

CRITICAL INSTRUCTIONS:
1. Your clarification question MUST be specific to the query context
2. This is INDIAN energy data - do NOT ask about US states or regions
3. If the query mentions "maximum energy consumption", ask about EnergyMet (energy actually supplied) or MaximumDemand (peak demand capacity)
4. If the query mentions "demand", ask about demand-related columns (DemandMet, MaximumDemand, PeakDemand)
5. If the query mentions "outage", ask about outage-related columns (CentralSectorOutage, StateSectorOutage, PrivateSectorOutage) - NOT EnergyOutage vs EnergyRestoration
6. If the query mentions "shortage", ask about shortage-related columns (EnergyShortage, PowerShortage)
7. If the query mentions "generation", ask about generation-related columns (TotalGeneration, ThermalGeneration)
8. DO NOT default to EnergyMet vs EnergyShortage unless the query specifically mentions energy supply
9. Build upon previous clarifications if any exist
10. Ask for the most critical missing information
11. Focus on ONE aspect that would resolve the ambiguity
12. Be concise and direct (no explanations or additional text)
13. ONLY use terms from the available schema columns
14. NEVER ask about US states, regions, or non-Indian geography

Generate ONLY the clarification question, no explanations or additional text:"""

        return prompt

    def _is_similar_question(self, question1: str, question2: str) -> bool:
        """
        Check if two clarification questions are too similar.

        Args:
            question1: First clarification question
            question2: Second clarification question

        Returns:
            True if questions are similar, False otherwise
        """
        # Normalize questions for comparison
        q1_lower = question1.lower().strip()
        q2_lower = question2.lower().strip()

        # Remove common prefixes and suffixes
        q1_clean = (
            q1_lower.replace("do you want", "")
            .replace("for all states", "")
            .replace("for all regions", "")
            .replace("for 2025", "")
            .strip()
        )
        q2_clean = (
            q2_lower.replace("do you want", "")
            .replace("for all states", "")
            .replace("for all regions", "")
            .replace("for 2025", "")
            .strip()
        )

        # Check if the core question is the same
        if q1_clean == q2_clean:
            logger.info(f"Exact match found: {q1_clean}")
            return True

        # Check if they contain the same key terms
        q1_words = set(q1_clean.split())
        q2_words = set(q2_clean.split())

        # If more than 70% of words are the same, consider them similar
        intersection = q1_words.intersection(q2_words)
        union = q1_words.union(q2_words)

        if len(union) > 0:
            similarity = len(intersection) / len(union)
            logger.info(
                f"Similarity score: {similarity:.2f} between '{q1_clean}' and '{q2_clean}'"
            )
            return similarity > 0.7

        return False

    def _get_geographic_level_for_query(self, query_lower: str) -> str:
        """
        Determine the appropriate geographic level (states vs regions) based on query type.

        Args:
            query_lower: The query in lowercase

        Returns:
            "states" or "regions" based on data availability
        """
        if "outage" in query_lower:
            return "regions"  # Outage data is available at region level
        elif (
            "demand" in query_lower
            or "energy" in query_lower
            or "shortage" in query_lower
        ):
            return "states"  # Energy/demand data is available at state level
        elif "generation" in query_lower:
            return "states"  # Generation data is available at state level
        else:
            return "states"  # Default to states

    def _get_context_specific_examples(
        self, query_lower: str, analysis: QueryAnalysis
    ) -> str:
        """
        Generates context-specific examples for the clarification prompt.
        This helps the LLM generate more relevant clarification questions.
        """
        examples = []
        geographic_level = self._get_geographic_level_for_query(query_lower)

        if "outage" in query_lower:
            examples.append("Examples for OUTAGE queries:")
            examples.append(
                f"- 'Do you want CentralSectorOutage, StateSectorOutage, or PrivateSectorOutage for all {geographic_level} in 2025?'"
            )
            examples.append(
                f"- 'For outage data, do you want CentralSectorOutage, StateSectorOutage, or PrivateSectorOutage for all {geographic_level}?'"
            )
            examples.append(
                f"- 'Do you want the total outage duration or the number of outage incidents for all {geographic_level}?'"
            )

        elif "maximum" in query_lower and (
            "energy" in query_lower or "consumption" in query_lower
        ):
            examples.append("Examples for MAXIMUM ENERGY CONSUMPTION queries:")
            examples.append(
                f"- 'Do you want EnergyMet (energy actually supplied) or MaximumDemand (peak demand capacity) for all {geographic_level} in 2025?'"
            )
            examples.append(
                f"- 'For maximum energy consumption, do you want the maximum EnergyMet value or the maximum DemandMet value for all {geographic_level}?'"
            )
            examples.append(
                f"- 'Do you want the state with the highest EnergyMet or the state with the highest MaximumDemand for all {geographic_level}?'"
            )

        elif "demand" in query_lower:
            examples.append("Examples for DEMAND queries:")
            examples.append(
                f"- 'Do you want DemandMet (actual demand met) or MaximumDemand (peak demand capacity) for all {geographic_level} in 2025?'"
            )
            examples.append(
                f"- 'For demand data, do you want PeakDemand, AverageDemand, or TotalDemand for all {geographic_level}?'"
            )
            examples.append(
                f"- 'Do you want the maximum demand value or the average demand value for all {geographic_level}?'"
            )

        elif "shortage" in query_lower:
            examples.append("Examples for SHORTAGE queries:")
            examples.append(
                f"- 'Do you want EnergyShortage (energy not supplied) or PowerShortage (power not supplied) for all {geographic_level} in 2025?'"
            )
            examples.append(
                f"- 'For shortage data, do you want TotalShortage, PeakShortage, or AverageShortage for all {geographic_level}?'"
            )
            examples.append(
                f"- 'Do you want the total shortage amount or the shortage percentage for all {geographic_level}?'"
            )

        elif "generation" in query_lower:
            examples.append("Examples for GENERATION queries:")
            examples.append(
                f"- 'Do you want TotalGeneration, ThermalGeneration, or RenewableGeneration for all {geographic_level} in 2025?'"
            )
            examples.append(
                f"- 'For generation data, do you want CoalGeneration, GasGeneration, or HydroGeneration for all {geographic_level}?'"
            )
            examples.append(
                f"- 'Do you want the total generation capacity or the actual generation output for all {geographic_level}?'"
            )

        else:
            # Generic examples for other query types
            examples.append("Examples for general queries:")
            examples.append(
                f"- 'Do you want EnergyShortage (energy not supplied) or EnergyMet (energy actually supplied) for all {geographic_level} in 2025?'"
            )
            examples.append(
                f"- 'For Maximum Demand Met in June 2025, do you want the maximum value across all {geographic_level}, or the maximum value for each individual {geographic_level}?'"
            )
            examples.append(
                f"- 'When you say 'all {geographic_level} in 2025', do you want data for each month of 2025, or the total for the entire year?'"
            )

        return "\n".join(examples)

    async def _generate_clarification_question(
        self, query: str, analysis: QueryAnalysis, request: QueryRequest
    ) -> str:
        """
        Generate a clarification question using the LLM.
        """
        try:
            # Build contextual clarification prompt
            clarification_prompt = self._build_contextual_clarification_prompt(
                query, analysis, getattr(request, "clarification_answers", None)
            )

            if self.llm_provider:
                llm_response = await self.llm_provider.generate(clarification_prompt)
                if (
                    llm_response
                    and hasattr(llm_response, "content")
                    and llm_response.content
                ):
                    clarification_question = llm_response.content.strip()
                    # Clean up the clarification question
                    clarification_question = self._clean_clarification_question(
                        clarification_question
                    )

                    # Validate the question
                    if self._validate_clarification_question(clarification_question):
                        logger.info(
                            f"Generated clarification question: {clarification_question}"
                        )
                        return clarification_question
                    else:
                        logger.warning(
                            f"LLM generated invalid clarification question: {clarification_question}"
                        )
                        # Fallback to schema-based clarification
                        return self._generate_schema_based_clarification(
                            query, analysis
                        )
                else:
                    logger.warning("LLM failed to generate clarification question")
                    return self._generate_schema_based_clarification(query, analysis)
            else:
                logger.warning("No LLM provider available for clarification")
                return self._generate_schema_based_clarification(query, analysis)

        except Exception as e:
            logger.error(f"Error generating clarification question: {str(e)}")
            return self._generate_schema_based_clarification(query, analysis)

    async def _generate_sql_with_templates(
        self, query: str, analysis: QueryAnalysis
    ) -> SQLGenerationResult:
        """
        Generate SQL using template-based approach.
        """
        print(
            f"🔍 RAG TEMPLATE GENERATION CALLED - Query: '{query}', Query Type: {analysis.query_type.value}"
        )
        try:
            logger.info(
                f"Attempting template-based SQL generation for query: '{query}'"
            )
            logger.info(
                f"Analysis: type={analysis.query_type}, intent={analysis.intent}"
            )

            # Build context for SQL generation
            context = self._build_context(query, analysis, "default_user")
            logger.info(f"Context built successfully")

            # Use the assembler to generate SQL
            print(f"🔍 CALLING ASSEMBLER generate_sql")
            result = self.sql_assembler.generate_sql(query, analysis, context)
            print(f"🔍 ASSEMBLER RESULT: success={result.success}")
            logger.info(
                f"SQL generation result: success={result.success}, sql_length={len(result.sql) if result.sql else 0}"
            )

            # If template-based generation failed, try simple fallback for clear queries
            if not result.success or not result.sql:
                logger.info("Template-based generation failed, trying simple fallback")
                # No more guessing - only explicit failures that require clarification
                return SQLGenerationResult(
                    success=False,
                    sql="",
                    error="Template-based SQL generation failed and no simple fallback available.",
                    confidence=0.0,
                    clarification_question="I cannot generate a SQL query for this request. Please provide more specific information.",
                )

            return result

        except Exception as e:
            logger.error(f"Error in template-based SQL generation: {str(e)}")
            return SQLGenerationResult(
                success=False,
                sql="",
                error=f"Template-based SQL generation failed: {str(e)}",
                confidence=0.0,
                clarification_question=None,
            )

    # === SPRINT 2: REMOVED SIMPLE FALLBACK SQL GENERATION ===
    # _generate_simple_fallback_sql method removed to implement ask-first policy
    # No more guessing - only explicit failures that require clarification

    def _has_sufficient_clarification(
        self, clarification_answers: Dict[str, str]
    ) -> bool:
        """
        Check if the user has provided sufficient clarification information.
        """
        if not clarification_answers:
            return False

        # Check for key clarification patterns
        sufficient_patterns = [
            "energy met",
            "total energy",
            "maximum energy",
            "energy consumption",
            "demand met",
            "total demand",
            "maximum demand",
            "outage",
            "total outage",
            "central sector",
            "state sector",
            "shortage",
            "energy shortage",
            "power shortage",
            "generation",
            "total generation",
            "thermal generation",
        ]

        # Check if any clarification answer contains sufficient information
        for question, answer in clarification_answers.items():
            answer_lower = answer.lower()
            for pattern in sufficient_patterns:
                if pattern in answer_lower:
                    logger.info(
                        f"Found sufficient clarification: '{pattern}' in '{answer}'"
                    )
                    return True

        return False

    def _generate_direct_sql(
        self, query: str, analysis: QueryAnalysis
    ) -> Optional[str]:
        """
        Generate SQL directly for clear queries without using complex templates.
        This is a fallback for when the template system fails.
        """
        try:
            query_lower = query.lower()

            # Use improved trend analysis logic for better detection
            # STEP 1: Determine if it's a growth query or aggregation query
            is_growth_query = any(
                word in query_lower
                for word in ["growth", "increase", "decrease", "change", "trend"]
            )
            is_aggregation_query = any(
                word in query_lower
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
                    f"Both growth and aggregation detected in direct SQL, prioritizing growth query"
                )

            # STEP 2: Determine time period with improved detection
            time_period = "monthly"  # Default
            if any(
                word in query_lower
                for word in ["quarterly", "quarter", "q1", "q2", "q3", "q4"]
            ):
                time_period = "quarterly"
            elif any(
                word in query_lower for word in ["yearly", "annual", "year over year"]
            ):
                time_period = "yearly"
            elif any(word in query_lower for word in ["weekly", "week"]):
                time_period = "weekly"
            elif any(word in query_lower for word in ["daily", "day"]):
                time_period = "daily"
            elif any(
                word in query_lower
                for word in ["monthly", "month", "by month", "per month"]
            ):
                time_period = "monthly"

            logger.info(
                f"🔍 DIRECT SQL ENHANCED DETECTION - Query: '{query}', Growth: {is_growth_query}, Aggregation: {is_aggregation_query}, Time period: {time_period}"
            )

            # For growth queries, redirect to template system for better handling
            if is_growth_query:
                logger.info(
                    f"Growth query detected in direct SQL, redirecting to template system"
                )
                return None  # Let the template system handle it

            # Use query analysis to determine the correct table
            if analysis.query_type.value == "region":
                table = "FactAllIndiaDailySummary"
                name_col = "RegionName"
                join_table = "DimRegions"
                join_col = "RegionID"
            elif analysis.query_type.value == "state":
                table = "FactStateDailyEnergy"
                name_col = "StateName"
                join_table = "DimStates"
                join_col = "StateID"
            else:
                # Default to state table
                table = "FactStateDailyEnergy"
                name_col = "StateName"
                join_table = "DimStates"
                join_col = "StateID"

            # Use schema linker to get the correct column - NO FALLBACKS
            if hasattr(self, "schema_linker") and self.schema_linker:
                metric_col = self.schema_linker.get_best_column_match(
                    user_query=query,
                    table_name=table,
                    query_type="energy",  # Default query type
                )

                if not metric_col:
                    logger.error(
                        f"No column match found for query: {query} in table: {table}. Clarification required."
                    )
                    return None

                logger.info(
                    f"Schema linker found column: {metric_col} for table: {table}"
                )
            else:
                logger.error(
                    f"No schema linker available for query: {query}. Clarification required."
                )
                return None

            # Determine the aggregation function
            has_maximum = any(
                term in query_lower for term in ["maximum", "max", "highest", "top"]
            )
            has_total = any(term in query_lower for term in ["total", "sum"])
            has_average = any(
                term in query_lower for term in ["average", "avg", "mean"]
            )

            if has_maximum:
                agg_func = "MAX"
            elif has_total:
                agg_func = "SUM"
            elif has_average:
                agg_func = "AVG"
            else:
                agg_func = "SUM"  # Default to sum

            # Determine the year
            has_2024 = "2024" in query_lower
            has_2025 = "2025" in query_lower
            year = "2025" if has_2025 else "2024"

            # Generate dynamic alias based on column and aggregation
            if agg_func == "MAX":
                alias_prefix = "Maximum"
            elif agg_func == "SUM":
                alias_prefix = "Total"
            elif agg_func == "AVG":
                alias_prefix = "Average"
            else:
                alias_prefix = "Total"

            # Create descriptive alias
            dynamic_alias = f"{alias_prefix}{metric_col}"

            # Generate the SQL based on time period and aggregation type
            if is_aggregation_query and time_period != "monthly":
                # Use time-based grouping for aggregation queries
                if time_period == "quarterly":
                    sql = f"""
                    SELECT {name_col}, dt.Quarter, ROUND({agg_func}({metric_col}), 2) as {dynamic_alias}
                    FROM {table} f
                    JOIN {join_table} d ON f.{join_col} = d.{join_col}
                    JOIN DimDates dt ON f.DateID = dt.DateID
                    WHERE dt.Year = {year}
                    GROUP BY {name_col}, dt.Quarter
                    ORDER BY {name_col}, dt.Quarter
                    """
                elif time_period == "weekly":
                    sql = f"""
                    SELECT {name_col}, dt.Week, ROUND({agg_func}({metric_col}), 2) as {dynamic_alias}
                    FROM {table} f
                    JOIN {join_table} d ON f.{join_col} = d.{join_col}
                    JOIN DimDates dt ON f.DateID = dt.DateID
                    WHERE dt.Year = {year}
                    GROUP BY {name_col}, dt.Week
                    ORDER BY {name_col}, dt.Week
                    """
                elif time_period == "daily":
                    sql = f"""
                    SELECT {name_col}, dt.DayOfMonth, ROUND({agg_func}({metric_col}), 2) as {dynamic_alias}
                    FROM {table} f
                    JOIN {join_table} d ON f.{join_col} = d.{join_col}
                    JOIN DimDates dt ON f.DateID = dt.DateID
                    WHERE dt.Year = {year}
                    GROUP BY {name_col}, dt.DayOfMonth
                    ORDER BY {name_col}, dt.DayOfMonth
                    """
                elif time_period == "yearly":
                    sql = f"""
                    SELECT {name_col}, dt.Year, ROUND({agg_func}({metric_col}), 2) as {dynamic_alias}
                    FROM {table} f
                    JOIN {join_table} d ON f.{join_col} = d.{join_col}
                    JOIN DimDates dt ON f.DateID = dt.DateID
                    WHERE dt.Year = {year}
                    GROUP BY {name_col}, dt.Year
                    ORDER BY {name_col}, dt.Year
                    """
                else:
                    # Default to monthly
                    sql = f"""
                    SELECT {name_col}, dt.Month, ROUND({agg_func}({metric_col}), 2) as {dynamic_alias}
                    FROM {table} f
                    JOIN {join_table} d ON f.{join_col} = d.{join_col}
                    JOIN DimDates dt ON f.DateID = dt.DateID
                    WHERE dt.Year = {year}
                    GROUP BY {name_col}, dt.Month
                    ORDER BY {name_col}, dt.Month
                    """
                logger.info(f"🔍 GENERATED DIRECT TIME-BASED SQL: {sql[:200]}...")
            else:
                # Regular grouping SQL (monthly or no specific time period)
                if time_period == "monthly":
                    sql = f"""
                    SELECT {name_col}, dt.Month, ROUND({agg_func}({metric_col}), 2) as {dynamic_alias}
                    FROM {table} f
                    JOIN {join_table} d ON f.{join_col} = d.{join_col}
                    JOIN DimDates dt ON f.DateID = dt.DateID
                    WHERE dt.Year = {year}
                    GROUP BY {name_col}, dt.Month
                    ORDER BY {name_col}, dt.Month
                    """
                    logger.info(f"🔍 GENERATED DIRECT MONTHLY SQL: {sql[:200]}...")
                else:
                    # Regular grouping SQL
                    sql = f"""
                    SELECT {name_col}, ROUND({agg_func}({metric_col}), 2) as {dynamic_alias}
                    FROM {table} f
                    JOIN {join_table} d ON f.{join_col} = d.{join_col}
                    JOIN DimDates dt ON f.DateID = dt.DateID
                    WHERE dt.Year = {year}
                    GROUP BY {name_col}
                    ORDER BY {dynamic_alias} DESC
                    """
                    logger.info(f"🔍 GENERATED DIRECT REGULAR SQL: {sql[:200]}...")

            return sql

        except Exception as e:
            logger.error(f"Error in direct SQL generation: {str(e)}")
            return None

    def _determine_trend_analysis_type(
        self, query: str, analysis: QueryAnalysis
    ) -> str:
        """
        Determine the type of trend analysis (growth or aggregation) from the query.
        """
        query_lower = query.lower()
        if (
            "growth" in query_lower
            or "increase" in query_lower
            or "decrease" in query_lower
        ):
            return "growth"
        elif (
            "maximum" in query_lower
            or "min" in query_lower
            or "average" in query_lower
            or "total" in query_lower
        ):
            return "aggregation"
        else:
            return "unknown"  # Fallback to template-based generation


# Factory function for dependency injection
@lru_cache()
def get_rag_service(
    db_path: str, llm_provider=None, memory_service=None
) -> EnhancedRAGService:
    """Get RAG service instance with caching"""
    return EnhancedRAGService(db_path, llm_provider, memory_service)
