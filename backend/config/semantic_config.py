"""
Semantic Engine Configuration
Configuration settings for enhanced semantic processing capabilities
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseSettings, Field


class SemanticProcessingMode(Enum):
    """Processing modes for semantic engine"""
    SEMANTIC_FIRST = "semantic_first"      # Use semantic engine primarily (high confidence)
    HYBRID = "hybrid"                      # Combine semantic with traditional (medium confidence)
    TRADITIONAL_FALLBACK = "traditional"   # Use traditional methods (low confidence)
    ADAPTIVE = "adaptive"                  # Automatically choose based on query characteristics


class VectorDatabaseType(Enum):
    """Supported vector database types"""
    QDRANT_MEMORY = "qdrant_memory"       # In-memory Qdrant (development)
    QDRANT_DISK = "qdrant_disk"           # Persistent Qdrant (production)
    CHROMA = "chroma"                     # ChromaDB
    FAISS = "faiss"                       # Facebook AI Similarity Search


@dataclass
class ConfidenceThresholds:
    """Confidence thresholds for semantic processing decisions"""
    semantic_first_threshold: float = 0.8    # Use semantic-first approach
    hybrid_threshold: float = 0.6            # Use hybrid approach
    fallback_threshold: float = 0.4          # Use traditional fallback
    execution_threshold: float = 0.3         # Minimum confidence to execute SQL


@dataclass
class VectorSearchConfig:
    """Vector search configuration"""
    similarity_threshold: float = 0.7        # Minimum similarity for relevance
    max_results: int = 5                     # Maximum search results
    embedding_model: str = "all-MiniLM-L6-v2"  # Sentence transformer model
    vector_dimensions: int = 384             # Vector dimensions
    distance_metric: str = "cosine"          # Distance metric (cosine, euclidean, dot)


@dataclass
class DomainModelConfig:
    """Domain model configuration for energy sector"""
    enable_business_glossary: bool = True
    enable_relationship_inference: bool = True
    enable_metric_calculations: bool = True
    enable_temporal_reasoning: bool = True
    
    # Energy domain specific settings
    default_energy_unit: str = "MW"
    default_time_aggregation: str = "daily"
    enable_growth_calculations: bool = True
    enable_capacity_utilization: bool = True


class SemanticEngineSettings(BaseSettings):
    """
    Settings for semantic engine configuration
    Provides comprehensive configuration for enhanced SQL generation
    """
    
    # Core semantic processing settings
    processing_mode: SemanticProcessingMode = SemanticProcessingMode.ADAPTIVE
    confidence_thresholds: ConfidenceThresholds = Field(default_factory=ConfidenceThresholds)
    
    # Vector database configuration
    vector_db_type: VectorDatabaseType = VectorDatabaseType.QDRANT_MEMORY
    vector_db_host: str = "localhost"
    vector_db_port: int = 6333
    vector_db_path: Optional[str] = None  # For disk-based storage
    vector_search_config: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
    
    # Domain model configuration
    domain_model_config: DomainModelConfig = Field(default_factory=DomainModelConfig)
    
    # LLM integration settings
    enable_llm_semantic_analysis: bool = True
    llm_semantic_timeout: float = 10.0
    llm_max_tokens: int = 1000
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # Cache time-to-live in seconds
    max_concurrent_requests: int = 10
    request_timeout: float = 30.0
    
    # Observability settings
    enable_metrics: bool = True
    enable_detailed_logging: bool = True
    log_level: str = "INFO"
    
    # Feedback and learning settings
    enable_feedback_learning: bool = True
    feedback_batch_size: int = 100
    learning_update_frequency: int = 3600  # Update frequency in seconds
    
    # Development and testing settings
    enable_debug_mode: bool = False
    enable_sql_explanation: bool = True
    enable_confidence_breakdown: bool = True
    
    class Config:
        env_prefix = "SEMANTIC_"
        env_file = ".env"
        case_sensitive = False


@dataclass
class AccuracyTargets:
    """Accuracy targets for different query types"""
    overall_target: float = 0.85              # 85% overall accuracy
    semantic_first_target: float = 0.90       # 90% for high-confidence semantic queries
    hybrid_target: float = 0.80               # 80% for hybrid processing
    growth_queries_target: float = 0.85       # 85% for growth/trend queries
    comparison_queries_target: float = 0.82   # 82% for comparison queries
    aggregation_queries_target: float = 0.88  # 88% for aggregation queries


@dataclass
class PerformanceTargets:
    """Performance targets for semantic processing"""
    max_response_time: float = 5.0            # Maximum response time in seconds
    semantic_analysis_time: float = 1.0       # Target semantic analysis time
    sql_generation_time: float = 0.5          # Target SQL generation time
    vector_search_time: float = 0.2           # Target vector search time


class EnhancedSystemConfig:
    """
    Comprehensive configuration for the enhanced Chat-with-DB system
    Integrates all semantic processing capabilities
    """
    
    def __init__(self):
        self.semantic_settings = SemanticEngineSettings()
        self.accuracy_targets = AccuracyTargets()
        self.performance_targets = PerformanceTargets()
        
        # Feature flags for phased rollout
        self.feature_flags = {
            "semantic_processing": True,
            "vector_search": True,
            "business_context_mapping": True,
            "domain_specific_intelligence": True,
            "advanced_visualization": True,
            "feedback_learning": True,
            
            # Phase 2 features (not yet implemented)
            "agentic_workflows": False,
            "event_driven_processing": False,
            "multi_language_support": False,
            "real_time_learning": False,
            
            # Phase 3 features (future)
            "predictive_analytics": False,
            "automated_insights": False,
            "cross_domain_knowledge": False
        }
        
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled"""
        return self.feature_flags.get(feature_name, False)
        
    def get_processing_mode(self, confidence: float) -> SemanticProcessingMode:
        """Determine processing mode based on confidence"""
        thresholds = self.semantic_settings.confidence_thresholds
        
        if confidence >= thresholds.semantic_first_threshold:
            return SemanticProcessingMode.SEMANTIC_FIRST
        elif confidence >= thresholds.hybrid_threshold:
            return SemanticProcessingMode.HYBRID
        else:
            return SemanticProcessingMode.TRADITIONAL_FALLBACK
            
    def should_execute_sql(self, confidence: float) -> bool:
        """Determine if SQL should be executed based on confidence"""
        return confidence >= self.semantic_settings.confidence_thresholds.execution_threshold
        
    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get current system capabilities"""
        return {
            "semantic_understanding": self.is_feature_enabled("semantic_processing"),
            "vector_similarity_search": self.is_feature_enabled("vector_search"),
            "business_context_mapping": self.is_feature_enabled("business_context_mapping"),
            "domain_intelligence": self.is_feature_enabled("domain_specific_intelligence"),
            "enhanced_visualization": self.is_feature_enabled("advanced_visualization"),
            "feedback_learning": self.is_feature_enabled("feedback_learning"),
            
            # Accuracy and performance info
            "accuracy_target": f"{self.accuracy_targets.overall_target * 100:.0f}%",
            "max_response_time": f"{self.performance_targets.max_response_time}s",
            
            # Processing modes
            "processing_modes": [mode.value for mode in SemanticProcessingMode],
            "vector_database": self.semantic_settings.vector_db_type.value,
            "embedding_model": self.semantic_settings.vector_search_config.embedding_model
        }
        
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary for monitoring/debugging"""
        return {
            "semantic_engine": {
                "processing_mode": self.semantic_settings.processing_mode.value,
                "vector_db_type": self.semantic_settings.vector_db_type.value,
                "embedding_model": self.semantic_settings.vector_search_config.embedding_model,
                "confidence_thresholds": {
                    "semantic_first": self.semantic_settings.confidence_thresholds.semantic_first_threshold,
                    "hybrid": self.semantic_settings.confidence_thresholds.hybrid_threshold,
                    "fallback": self.semantic_settings.confidence_thresholds.fallback_threshold,
                    "execution": self.semantic_settings.confidence_thresholds.execution_threshold
                }
            },
            "performance_targets": {
                "max_response_time": self.performance_targets.max_response_time,
                "semantic_analysis_time": self.performance_targets.semantic_analysis_time,
                "sql_generation_time": self.performance_targets.sql_generation_time
            },
            "accuracy_targets": {
                "overall": f"{self.accuracy_targets.overall_target * 100:.0f}%",
                "semantic_first": f"{self.accuracy_targets.semantic_first_target * 100:.0f}%",
                "hybrid": f"{self.accuracy_targets.hybrid_target * 100:.0f}%"
            },
            "enabled_features": [
                feature for feature, enabled in self.feature_flags.items() if enabled
            ],
            "domain_model": {
                "energy_domain": True,
                "business_glossary": self.semantic_settings.domain_model_config.enable_business_glossary,
                "relationship_inference": self.semantic_settings.domain_model_config.enable_relationship_inference,
                "temporal_reasoning": self.semantic_settings.domain_model_config.enable_temporal_reasoning
            }
        }


# Global configuration instance
semantic_config = EnhancedSystemConfig()


def get_semantic_config() -> EnhancedSystemConfig:
    """Get the global semantic configuration instance"""
    return semantic_config


def update_semantic_config(**kwargs) -> None:
    """Update semantic configuration settings"""
    for key, value in kwargs.items():
        if hasattr(semantic_config.semantic_settings, key):
            setattr(semantic_config.semantic_settings, key, value)
        elif key in semantic_config.feature_flags:
            semantic_config.feature_flags[key] = value
        else:
            raise ValueError(f"Unknown configuration key: {key}")
            

# Configuration validation
def validate_semantic_config(config: EnhancedSystemConfig) -> List[str]:
    """Validate semantic configuration and return any issues"""
    issues = []
    
    # Validate confidence thresholds
    thresholds = config.semantic_settings.confidence_thresholds
    if thresholds.semantic_first_threshold <= thresholds.hybrid_threshold:
        issues.append("semantic_first_threshold must be greater than hybrid_threshold")
        
    if thresholds.hybrid_threshold <= thresholds.fallback_threshold:
        issues.append("hybrid_threshold must be greater than fallback_threshold")
        
    if thresholds.execution_threshold < 0.0 or thresholds.execution_threshold > 1.0:
        issues.append("execution_threshold must be between 0.0 and 1.0")
        
    # Validate vector search config
    vector_config = config.semantic_settings.vector_search_config
    if vector_config.similarity_threshold < 0.0 or vector_config.similarity_threshold > 1.0:
        issues.append("similarity_threshold must be between 0.0 and 1.0")
        
    if vector_config.max_results <= 0:
        issues.append("max_results must be positive")
        
    # Validate performance targets
    perf = config.performance_targets
    if perf.max_response_time <= 0:
        issues.append("max_response_time must be positive")
        
    return issues