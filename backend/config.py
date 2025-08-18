"""
Configuration management using pydantic-settings
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    app_name: str = "Chat-with-DB"
    app_version: str = "1.0.0"
    app_env: str = Field(default="development", env="APP_ENV")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Database Configuration
    database_type: str = Field(default="mssql", env="DATABASE_TYPE")
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # MSSQL Configuration
    mssql_server: str = Field(default="localhost", env="MSSQL_SERVER")
    mssql_database: str = Field(default="Powerflow", env="MSSQL_DATABASE")
    mssql_username: str = Field(default="sa", env="MSSQL_USERNAME")
    mssql_password: str = Field(default="", env="MSSQL_PASSWORD")
    mssql_port: int = Field(default=1433, env="MSSQL_PORT")
    mssql_encrypt: bool = Field(default=True, env="MSSQL_ENCRYPT")
    mssql_trust_server_certificate: bool = Field(default=False, env="MSSQL_TRUST_SERVER_CERTIFICATE")
    
    # LLM Configuration
    llm_provider_type: str = Field(default="gemini", env="LLM_PROVIDER_TYPE")
    
    # Gemini Configuration (Primary LLM)
    gemini_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")  # Alias for compatibility
    gemini_model: str = Field(default="gemini-2.5-flash-lite", env="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.1, env="GEMINI_TEMPERATURE")
    gemini_max_output_tokens: int = Field(default=8192, env="GEMINI_MAX_OUTPUT_TOKENS")
    gemini_top_p: float = Field(default=0.8, env="GEMINI_TOP_P")
    gemini_top_k: int = Field(default=40, env="GEMINI_TOP_K")
    
    # OpenAI Configuration (Fallback)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    openai_base_url: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")
    
    # Anthropic Configuration (Alternative)
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    
    # Vector Database Configuration
    vector_db_url: Optional[str] = Field(default=None, env="VECTOR_DB_URL")
    vector_db_api_key: Optional[str] = Field(default=None, env="VECTOR_DB_API_KEY")
    
    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    
    # ChromaDB Configuration
    chroma_db_path: str = Field(default="/app/vector_db", env="CHROMA_DB_PATH")
    
    # AI/ML Components Configuration
    wren_ai_enabled: bool = Field(default=True, env="WREN_AI_ENABLED")
    semantic_engine_enabled: bool = Field(default=True, env="SEMANTIC_ENGINE_ENABLED")
    agentic_framework_enabled: bool = Field(default=True, env="AGENTIC_FRAMEWORK_ENABLED")
    entity_recognition_enabled: bool = Field(default=True, env="ENTITY_RECOGNITION_ENABLED")
    
    # Performance Configuration
    max_connections: int = Field(default=100, env="MAX_CONNECTIONS")
    connection_timeout: int = Field(default=30, env="CONNECTION_TIMEOUT")
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")
    
    # Caching Configuration
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    cache_max_size: int = Field(default=1000, env="CACHE_MAX_SIZE")
    
    # Security Configuration
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS Configuration
    cors_origins: List[str] = Field(default=["http://localhost:3000"], env="CORS_ORIGINS")
    frontend_origin: str = Field(default="http://localhost:3000", env="FRONTEND_ORIGIN")
    
    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_structured_logging: bool = Field(default=True, env="ENABLE_STRUCTURED_LOGGING")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Feature Flags
    enable_gpu_acceleration: bool = Field(default=False, env="ENABLE_GPU_ACCELERATION")
    enable_vector_search: bool = Field(default=True, env="ENABLE_VECTOR_SEARCH")
    enable_semantic_processing: bool = Field(default=True, env="ENABLE_SEMANTIC_PROCESSING")
    enable_agentic_workflows: bool = Field(default=True, env="ENABLE_AGENTIC_WORKFLOWS")
    enable_feedback_learning: bool = Field(default=True, env="ENABLE_FEEDBACK_LEARNING")
    enable_temporal_processing: bool = Field(default=True, env="ENABLE_TEMPORAL_PROCESSING")
    
    # Development Configuration
    enable_swagger_ui: bool = Field(default=True, env="ENABLE_SWAGGER_UI")
    enable_reload: bool = Field(default=False, env="ENABLE_RELOAD")
    enable_debug_endpoints: bool = Field(default=False, env="ENABLE_DEBUG_ENDPOINTS")
    testing_mode: bool = Field(default=False, env="TESTING_MODE")
    
    # Additional Configuration Fields
    # database_path is now derived from MSSQL configuration
    database_path: Optional[str] = None  # Will be set dynamically from MSSQL config
    llm_api_key: Optional[str] = Field(default=None, env="LLM_API_KEY")
    llm_model: Optional[str] = Field(default=None, env="LLM_MODEL")
    llm_base_url: Optional[str] = Field(default=None, env="LLM_BASE_URL")
    gpu_device: Optional[str] = Field(default=None, env="GPU_DEVICE")
    max_query_length: int = Field(default=1000, env="MAX_QUERY_LENGTH")
    enable_llm_clarification: bool = Field(default=True, env="ENABLE_LLM_CLARIFICATION")
    memory_cache_ttl: int = Field(default=3600, env="MEMORY_CACHE_TTL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from environment

    def get_database_url(self) -> str:
        """Get database URL based on configuration"""
        if self.database_url:
            return self.database_url
        
        if self.database_type.lower() == "mssql":
            return (
                f"mssql+pyodbc://{self.mssql_username}:{self.mssql_password}"
                f"@{self.mssql_server}:{self.mssql_port}/{self.mssql_database}"
                f"?driver=ODBC+Driver+18+for+SQL+Server"
                f"&Encrypt={'yes' if self.mssql_encrypt else 'no'}"
                f"&TrustServerCertificate={'yes' if self.mssql_trust_server_certificate else 'no'}"
            )
        
        raise ValueError(f"Unsupported database type: {self.database_type}")
    
    def is_azure_sql(self) -> bool:
        """Check if using Azure SQL Database"""
        return (
            self.database_type.lower() == "mssql" and 
            "database.windows.net" in self.mssql_server.lower()
        )
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins with fallback"""
        if self.cors_origins:
            return self.cors_origins
        return [self.frontend_origin] if self.frontend_origin else ["http://localhost:3000"]

# Global settings instance
_settings = None

def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
        # Set database_path dynamically
        if _settings.database_type.lower() == "mssql":
            _settings.database_path = _settings.get_database_url()
    return _settings

def update_settings(**kwargs):
    """Update settings dynamically"""
    global _settings
    if _settings is None:
        _settings = Settings()
    
    for key, value in kwargs.items():
        if hasattr(_settings, key):
            setattr(_settings, key, value)

def get_database_url() -> str:
    """Get database URL based on configuration"""
    settings = get_settings()
    return settings.get_database_url()

def get_llm_provider_config() -> dict:
    """Get LLM provider configuration"""
    settings = get_settings()
    
    if settings.llm_provider_type.lower() == "gemini":
        return {
            "provider": "gemini",
            "api_key": settings.gemini_api_key,
            "model": settings.gemini_model,
            "temperature": settings.gemini_temperature,
            "max_output_tokens": settings.gemini_max_output_tokens,
            "top_p": settings.gemini_top_p,
            "top_k": settings.gemini_top_k
        }
    elif settings.llm_provider_type.lower() == "openai":
        return {
            "provider": "openai",
            "api_key": settings.openai_api_key,
            "model": settings.openai_model,
            "base_url": settings.openai_base_url
        }
    elif settings.llm_provider_type.lower() == "anthropic":
        return {
            "provider": "anthropic",
            "api_key": settings.anthropic_api_key,
            "model": settings.anthropic_model
        }
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider_type}")
