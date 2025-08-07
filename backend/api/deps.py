"""
FastAPI dependencies for dependency injection
"""

import logging
from functools import lru_cache
from typing import Optional

import sys
sys.path.insert(0, '.')
from backend.config import Settings, get_settings
from ..core.llm_provider import create_llm_provider
from ..core.types import QueryRequest
from ..services.rag_service import EnhancedRAGService

logger = logging.getLogger(__name__)


class MemoryService:
    """Memory service abstraction"""

    def __init__(self, settings: Settings):
        self.settings = settings

    def get_relevant_context(self, query: str, user_id: str) -> Optional[str]:
        """Get relevant memory context"""
        # Implementation would fetch from memory system
        return None


@lru_cache()
def get_settings_dep() -> Settings:
    """Get settings dependency"""
    return get_settings()


def get_llm_provider():
    """Get LLM provider dependency"""
    settings = get_settings_dep()
    return create_llm_provider(
        provider_type=settings.llm_provider_type,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        enable_gpu=settings.enable_gpu_acceleration,
        gpu_device=settings.gpu_device,
    )


def get_memory_service() -> MemoryService:
    """Get memory service dependency"""
    settings = get_settings_dep()
    return MemoryService(settings)


# Updated to inject LLM and Memory services
@lru_cache()
def get_rag_service() -> EnhancedRAGService:
    """Get RAG service instance with caching and injected dependencies"""
    settings = get_settings_dep()
    llm_provider = get_llm_provider()
    memory_service = get_memory_service()
    return EnhancedRAGService(
        db_path=settings.database_path,
        llm_provider=llm_provider,
        memory_service=memory_service,
    )


def validate_query_request(request: QueryRequest) -> QueryRequest:
    """Validate query request"""
    settings = get_settings_dep()

    # Check query length
    if len(request.question) > settings.max_query_length:
        raise ValueError(f"Query too long. Maximum length: {settings.max_query_length}")

    # Check for dangerous content
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "CREATE", "ALTER"]
    query_upper = request.question.upper()
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            raise ValueError(f"Query contains dangerous keyword: {keyword}")

    return request
