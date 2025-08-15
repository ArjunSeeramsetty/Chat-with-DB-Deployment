"""
Configuration management using pydantic-settings
"""

import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Database Configuration
    database_type: str = "sqlite"  # "sqlite", "mssql", "postgresql"
    database_path: str = "/app/data/power_data.db"  # For SQLite fallback, overridden by DATABASE_PATH env var
    
    # Azure SQL Server Configuration (when database_type = "mssql")
    mssql_server: Optional[str] = None
    mssql_database: Optional[str] = None
    mssql_username: Optional[str] = None
    mssql_password: Optional[str] = None
    mssql_port: int = 1433
    mssql_timeout: int = 30
    mssql_encrypt: bool = True
    mssql_trust_server_certificate: bool = False
    mssql_connection_timeout: int = 30
    mssql_command_timeout: int = 30
    
    # Database URL (constructed from above or provided directly)
    database_url: Optional[str] = None

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    # Frontend Configuration
    frontend_origin: str = "http://localhost:3000"
    cors_origins: list = ["http://localhost:3000", "http://localhost:3001"]

    # Environment Configuration
    app_env: str = "development"  # "development", "staging", "production"
    debug: bool = False

    # LLM Configuration
    llm_provider_type: Optional[str] = "ollama"  # "openai", "anthropic", "ollama", "vertex"
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = (
        "llama3.2:3b"  # e.g., "gpt-3.5-turbo", "claude-3-sonnet", "llama3.2:3b"
    )
    llm_base_url: Optional[str] = (
        "http://localhost:11434"  # For local models or custom endpoints
    )

    # GPU Acceleration Configuration
    enable_gpu_acceleration: bool = False  # Enable GPU acceleration for LLM
    gpu_device: Optional[str] = None  # GPU device to use (e.g., "cuda:0", "mps")
    gpu_memory_fraction: float = 0.8  # Fraction of GPU memory to use (0.0-1.0)

    # LLM Feature Flags
    enable_llm_paraphrase: bool = False  # Hook #1
    enable_llm_column_linking: bool = True  # Hook #2 (high priority)
    enable_llm_slot_filling: bool = False  # Hook #3
    enable_llm_candidate_synthesis: bool = True  # Hook #4 (high priority)
    enable_llm_auto_repair: bool = False  # Hook #5
    enable_llm_clarification: bool = True  # Hook #6
    enable_llm_summary: bool = False  # Hook #7

    # LLM Safety Settings
    max_llm_response_length: int = 300
    banned_sql_keywords: list = [
        "DROP",
        "UPDATE",
        "DELETE",
        "INSERT",
        "CREATE",
        "ALTER",
        "TRUNCATE",
    ]

    # Processing Configuration
    default_processing_mode: str = "balanced"
    max_sql_attempts: int = 3
    sql_timeout: int = 30
    max_rows: int = 1000

    # Confidence thresholds for different processing modes
    confidence_thresholds: dict = {"comprehensive": 0.8, "balanced": 0.6, "fast": 0.4}

    # Validation Configuration
    enable_sql_validation: bool = True
    enable_sandbox_testing: bool = True

    # Memory Configuration
    memory_enabled: bool = True
    memory_cache_ttl: int = 300  # 5 minutes

    # Vector Database
    vector_db_type: str = "qdrant"  # "qdrant", "pgvector", "none"
    vector_db_path: str = "power_data_vectors_enhanced"  # For local Qdrant
    vector_db_url: Optional[str] = None  # For remote Qdrant
    vector_db_api_key: Optional[str] = None
    enable_vector_search: bool = True

    # Logging
    log_level: str = "INFO"
    enable_structured_logging: bool = True

    # Security
    enable_query_safety_check: bool = True
    max_query_length: int = 1000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields from environment

    def get_database_url(self) -> str:
        """Get the appropriate database URL based on configuration"""
        if self.database_url:
            return self.database_url
        
        if self.database_type == "mssql":
            if not all([self.mssql_server, self.mssql_database, self.mssql_username, self.mssql_password]):
                raise ValueError("Azure SQL configuration incomplete. Set mssql_server, mssql_database, mssql_username, mssql_password")
            
            # URL encode the password to handle special characters
            import urllib.parse
            encoded_password = urllib.parse.quote_plus(self.mssql_password)
            
            # Construct the proper SQLAlchemy URL for Azure SQL
            # Format: mssql+pyodbc://username:password@server:port/database?driver=ODBC+Driver+18+for+SQL+Server&param1=value1&param2=value2
            
            base_url = f"mssql+pyodbc://{self.mssql_username}:{encoded_password}@{self.mssql_server}.database.windows.net:{self.mssql_port}/{self.mssql_database}"
            
            # Build query parameters
            query_params = [
                "driver=ODBC+Driver+18+for+SQL+Server",
                f"Server=tcp:{self.mssql_server}.database.windows.net,{self.mssql_port}",
                f"Database={self.mssql_database}",
                f"Encrypt={'yes' if self.mssql_encrypt else 'no'}",
                f"TrustServerCertificate={'yes' if self.mssql_trust_server_certificate else 'no'}",
                f"Connection+Timeout={self.mssql_connection_timeout}",
                f"Command+Timeout={self.mssql_command_timeout}",
                "MultipleActiveResultSets=true",
                "ApplicationIntent=ReadWrite"
            ]
            
            # Combine base URL with query parameters
            full_url = f"{base_url}?{'&'.join(query_params)}"
            
            return full_url
            
        elif self.database_type == "sqlite":
            return f"sqlite:///{self.database_path}"
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")

    def get_cors_origins(self) -> list:
        """Get CORS origins, with frontend_origin as fallback"""
        if self.cors_origins:
            return self.cors_origins
        return [self.frontend_origin] if self.frontend_origin else []

    def is_azure_sql(self) -> bool:
        """Check if using Azure SQL Server"""
        return self.database_type == "mssql" and self.mssql_server and ".database.windows.net" in self.get_database_url()


# Lazy settings instance
_settings_instance = None


def get_settings() -> Settings:
    """Get application settings (lazy initialization)"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
