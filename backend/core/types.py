"""
Core types and Pydantic models for the modular Text-to-SQL system
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Enumeration of supported query types"""

    REGION = "region"
    STATE = "state"
    GENERATION = "generation"
    TRANSMISSION = "transmission"
    EXCHANGE = "exchange"
    EXCHANGE_DETAIL = "exchange_detail"
    TIME_BLOCK = "time_block"
    TIME_BLOCK_GENERATION = "time_block_generation"
    UNKNOWN = "unknown"


class ProcessingMode(str, Enum):
    """Processing modes for different speed/accuracy trade-offs"""

    FAST = "fast"
    BALANCED = "balanced"
    COMPREHENSIVE = "comprehensive"


class IntentType(str, Enum):
    """Types of user intent"""

    DATA_RETRIEVAL = "data_retrieval"
    COMPARISON = "comparison"
    TREND_ANALYSIS = "trend_analysis"
    MAXIMUM_ANALYSIS = "maximum_analysis"
    AGGREGATION = "aggregation"
    UNKNOWN = "unknown"


class ValidationResult(BaseModel):
    """Result of SQL validation"""

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    fixed_sql: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class QueryAnalysis(BaseModel):
    """Analysis of user query"""

    query_type: QueryType
    intent: IntentType
    entities: List[str] = Field(default_factory=list)
    time_period: Optional[Dict[str, Any]] = None
    metrics: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    main_table: str
    dimension_table: str
    join_key: str
    name_column: str
    detected_keywords: List[str] = Field(default_factory=list)
    original_query: Optional[str] = None


class SQLGenerationResult(BaseModel):
    """Result of SQL generation process"""

    success: bool
    sql: str = ""
    error: Optional[str] = None
    confidence: float = 0.0
    warnings: List[str] = Field(default_factory=list)
    clarification_question: Optional[str] = None  # Added for clarification workflow


class ExecutionResult(BaseModel):
    """Result of SQL execution"""

    success: bool
    data: List[Dict[str, Any]] = Field(default_factory=list)
    headers: List[str] = Field(default_factory=list)
    row_count: int = 0
    execution_time: float = 0.0
    error: Optional[str] = None
    sql: str


class VisualizationRecommendation(BaseModel):
    """Chart recommendation for data visualization"""

    chart_type: str
    config: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None


class QueryRequest(BaseModel):
    """Request model for query processing"""

    question: str = Field(..., description="Natural language query")
    user_id: str = Field(default="default_user", description="User identifier")
    processing_mode: ProcessingMode = Field(default=ProcessingMode.BALANCED)
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    clarification_answers: Optional[Dict[str, str]] = Field(default=None)
    clarification_attempt_count: int = Field(
        default=0, description="Number of clarification attempts made"
    )
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class QueryResponse(BaseModel):
    """Response model for query processing"""

    success: bool
    data: List[Dict[str, Any]] = Field(default_factory=list)
    sql_query: str
    row_count: int = 0
    intent_analysis: Optional[Dict[str, Any]] = None
    follow_up_suggestions: List[str] = Field(default_factory=list)
    visualization: Optional[VisualizationRecommendation] = None
    plot: Optional[Dict[str, Any]] = None  # Frontend-compatible plot format
    table: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    api_calls: int = 0
    processing_mode: ProcessingMode
    query_type: Optional[QueryType] = None
    error: Optional[str] = None
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # Clarification support
    clarification_needed: Optional[bool] = None
    clarification_question: Optional[str] = None
    clarification_attempt_count: Optional[int] = Field(
        default=0, description="Number of clarification attempts made"
    )
    # Confidence score
    confidence: float = 0.0


class SchemaInfo(BaseModel):
    """Database schema information"""

    tables: Dict[str, List[str]] = Field(default_factory=dict)
    relationships: Dict[str, List[str]] = Field(default_factory=dict)
    cached_at: datetime = Field(default_factory=datetime.now)


class DimensionValue(BaseModel):
    """Dimension table value"""

    id: int
    name: str
    table: str


class UserMapping(BaseModel):
    """Mapping of user references to database entities"""

    mapping_type: str
    entity: DimensionValue
    confidence: float = Field(ge=0.0, le=1.0)


class ContextInfo(BaseModel):
    """Context information for SQL generation"""

    query_analysis: QueryAnalysis
    user_mappings: List[UserMapping] = Field(default_factory=list)
    dimension_values: Dict[str, List[DimensionValue]] = Field(default_factory=dict)
    memory_context: Optional[str] = None
    schema_info: Optional[SchemaInfo] = None
    relevant_examples: List[Dict[str, Any]] = Field(default_factory=list)
    schema_linker: Optional[Any] = None  # Add schema linker for business rules access
    llm_provider: Optional[Any] = None  # Add LLM provider for hooks
