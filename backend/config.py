"""
Configuration management using pydantic-settings
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    # Database Configuration
    database_path: str = "C:/Users/arjun/Desktop/PSPreport/power_data.db"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"
    
    # LLM Configuration
    llm_provider_type: Optional[str] = "ollama"  # "openai", "anthropic", "ollama", etc.
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = "llama3.2:3b"  # e.g., "gpt-3.5-turbo", "claude-3-sonnet", "llama3.2:3b"
    llm_base_url: Optional[str] = "http://localhost:11434"  # For local models or custom endpoints
    
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
    banned_sql_keywords: list = ["DROP", "UPDATE", "DELETE", "INSERT", "CREATE", "ALTER", "TRUNCATE"]
    
    # Processing Configuration
    default_processing_mode: str = "balanced"
    max_sql_attempts: int = 3
    sql_timeout: int = 30
    max_rows: int = 1000
    
    # Confidence thresholds for different processing modes
    confidence_thresholds: dict = {
        "comprehensive": 0.8,
        "balanced": 0.6,
        "fast": 0.4
    }
    
    # Validation Configuration
    enable_sql_validation: bool = True
    enable_sandbox_testing: bool = True
    
    # Memory Configuration
    memory_enabled: bool = True
    memory_cache_ttl: int = 300  # 5 minutes
    
    # Vector Database
    vector_db_path: str = "power_data_vectors_enhanced"
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

# Lazy settings instance
_settings_instance = None

def get_settings() -> Settings:
    """Get application settings (lazy initialization)"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance 