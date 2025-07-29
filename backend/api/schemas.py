"""
API schemas for request/response models
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Re-export core types for API use
from ..core.types import (
    IntentType,
    ProcessingMode,
    QueryType,
    ValidationResult,
    VisualizationRecommendation,
)


class QueryRequest(BaseModel):
    """API request model for query processing"""

    question: str = Field(..., description="Natural language query")
    user_id: str = Field(default="default_user", description="User identifier")
    processing_mode: ProcessingMode = Field(default=ProcessingMode.BALANCED)
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    clarification_answers: Optional[Dict[str, str]] = Field(default=None)
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class QueryResponse(BaseModel):
    """API response model for query processing"""

    success: bool
    data: List[Dict[str, Any]] = Field(default_factory=list)
    sql_query: str
    row_count: int = 0
    intent_analysis: Optional[Dict[str, Any]] = None
    follow_up_suggestions: List[str] = Field(default_factory=list)
    table: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    api_calls: int = 0
    processing_mode: ProcessingMode
    query_type: Optional[QueryType] = None
    error: Optional[str] = None
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    database: str
    schema_cache: str
    timestamp: float
    error: Optional[str] = None


class SchemaResponse(BaseModel):
    """Schema information response"""

    schema_data: Dict[str, List[str]] = Field(alias="schema")


class ChartRequest(BaseModel):
    """Chart generation request"""

    data: List[Dict[str, Any]] = Field(..., description="Data to visualize")
    query: str = Field(..., description="Original query")
    chart_type: Optional[str] = Field(None, description="Preferred chart type")


class ChartResponse(BaseModel):
    """Chart generation response"""

    visualization: Optional[VisualizationRecommendation] = None


class MetricsResponse(BaseModel):
    """Application metrics response"""

    version: str
    config: Dict[str, Any]
    features: Dict[str, bool]


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str
    detail: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
