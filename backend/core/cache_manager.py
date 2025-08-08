"""
Cache Manager for Performance Optimization

This module provides Redis-based caching functionality to improve response times
for frequently accessed data including query results, schema metadata, and LLM responses.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import pickle
import redis
from redis.exceptions import RedisError, ConnectionError

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types for different data types."""
    QUERY_RESULT = "query_result"
    SCHEMA_METADATA = "schema_metadata"
    LLM_RESPONSE = "llm_response"
    SEMANTIC_CONTEXT = "semantic_context"
    FEW_SHOT_EXAMPLES = "few_shot_examples"


@dataclass
class CacheConfig:
    """Configuration for cache settings."""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    default_ttl: int = 3600  # 1 hour
    max_memory: str = "100mb"
    compression_threshold: int = 1024  # Compress data larger than 1KB
    enable_compression: bool = True
    enable_stats: bool = True


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_requests: int = 0
    average_response_time: float = 0.0
    memory_usage: float = 0.0


class CacheManager:
    """Redis-based cache manager for performance optimization."""
    
    def __init__(self, config: CacheConfig):
        """Initialize cache manager with Redis connection."""
        self.config = config
        self.redis_client = None
        self.stats = CacheStats()
        self._connect_redis()
        self._setup_redis_config()
    
    def _connect_redis(self) -> None:
        """Establish Redis connection with error handling."""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=False,  # Keep as bytes for compression
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Redis cache connected to {self.config.redis_host}:{self.config.redis_port}")
        except (ConnectionError, RedisError) as e:
            logger.warning(f"Redis connection failed: {e}. Cache will be disabled.")
            self.redis_client = None
    
    def _setup_redis_config(self) -> None:
        """Configure Redis settings for optimal performance."""
        if not self.redis_client:
            return
        
        try:
            # Set max memory policy
            self.redis_client.config_set("maxmemory", self.config.max_memory)
            self.redis_client.config_set("maxmemory-policy", "allkeys-lru")
            
            # Enable compression if configured
            if self.config.enable_compression:
                self.redis_client.config_set("save", "900 1 300 10 60 10000")
            
            logger.info("Redis configuration applied successfully")
        except RedisError as e:
            logger.warning(f"Failed to configure Redis: {e}")
    
    def _generate_key(self, strategy: CacheStrategy, identifier: str) -> str:
        """Generate cache key with strategy prefix."""
        return f"cache:{strategy.value}:{identifier}"
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if it exceeds threshold."""
        if not self.config.enable_compression or len(data) < self.config.compression_threshold:
            return data
        
        try:
            import zlib
            return zlib.compress(data)
        except ImportError:
            logger.warning("zlib not available, compression disabled")
            return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data if it was compressed."""
        if not self.config.enable_compression:
            return data
        
        try:
            import zlib
            return zlib.decompress(data)
        except (ImportError, zlib.error):
            return data
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data to bytes for caching."""
        try:
            # Try JSON first for simple data types
            if isinstance(data, (dict, list, str, int, float, bool)) or data is None:
                return json.dumps(data, default=str).encode('utf-8')
            else:
                # Use pickle for complex objects
                return pickle.dumps(data)
        except (TypeError, ValueError) as e:
            logger.warning(f"JSON serialization failed, using pickle: {e}")
            return pickle.dumps(data)
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from bytes."""
        try:
            # Try JSON first
            decoded = data.decode('utf-8')
            return json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError):
            try:
                # Fall back to pickle
                return pickle.loads(data)
            except pickle.UnpicklingError as e:
                logger.error(f"Failed to deserialize cached data: {e}")
                return None
    
    def get(self, strategy: CacheStrategy, identifier: str, default: Any = None) -> Any:
        """Retrieve data from cache."""
        if not self.redis_client:
            return default
        
        start_time = time.time()
        key = self._generate_key(strategy, identifier)
        
        try:
            cached_data = self.redis_client.get(key)
            if cached_data is None:
                self.stats.misses += 1
                return default
            
            # Decompress if needed
            decompressed_data = self._decompress_data(cached_data)
            result = self._deserialize_data(decompressed_data)
            
            self.stats.hits += 1
            self.stats.average_response_time = (
                (self.stats.average_response_time * (self.stats.hits - 1) + 
                 (time.time() - start_time)) / self.stats.hits
            )
            
            logger.debug(f"Cache hit for key: {key}")
            return result
            
        except RedisError as e:
            logger.error(f"Redis error during get operation: {e}")
            self.stats.errors += 1
            return default
        finally:
            self.stats.total_requests += 1
    
    def set(self, strategy: CacheStrategy, identifier: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Store data in cache with optional TTL."""
        if not self.redis_client:
            return False
        
        key = self._generate_key(strategy, identifier)
        ttl = ttl or self.config.default_ttl
        
        try:
            # Serialize and compress data
            serialized_data = self._serialize_data(data)
            compressed_data = self._compress_data(serialized_data)
            
            # Store in Redis
            result = self.redis_client.setex(key, ttl, compressed_data)
            
            if result:
                self.stats.sets += 1
                logger.debug(f"Cache set for key: {key} with TTL: {ttl}s")
            
            return bool(result)
            
        except RedisError as e:
            logger.error(f"Redis error during set operation: {e}")
            self.stats.errors += 1
            return False
    
    def delete(self, strategy: CacheStrategy, identifier: str) -> bool:
        """Delete data from cache."""
        if not self.redis_client:
            return False
        
        key = self._generate_key(strategy, identifier)
        
        try:
            result = self.redis_client.delete(key)
            if result:
                self.stats.deletes += 1
                logger.debug(f"Cache delete for key: {key}")
            
            return bool(result)
            
        except RedisError as e:
            logger.error(f"Redis error during delete operation: {e}")
            self.stats.errors += 1
            return False
    
    def clear_strategy(self, strategy: CacheStrategy) -> int:
        """Clear all cached data for a specific strategy."""
        if not self.redis_client:
            return 0
        
        pattern = f"cache:{strategy.value}:*"
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries for strategy: {strategy.value}")
                return deleted
            return 0
            
        except RedisError as e:
            logger.error(f"Redis error during clear operation: {e}")
            self.stats.errors += 1
            return 0
    
    def clear_all(self) -> int:
        """Clear all cached data."""
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys("cache:*")
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared all cache entries: {deleted}")
                return deleted
            return 0
            
        except RedisError as e:
            logger.error(f"Redis error during clear all operation: {e}")
            self.stats.errors += 1
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        if not self.redis_client:
            return asdict(self.stats)
        
        try:
            # Get Redis memory info
            info = self.redis_client.info("memory")
            memory_usage = info.get("used_memory_human", "0B")
            
            # Calculate hit rate
            total_requests = self.stats.hits + self.stats.misses
            hit_rate = (self.stats.hits / total_requests * 100) if total_requests > 0 else 0
            
            stats = asdict(self.stats)
            stats.update({
                "hit_rate": round(hit_rate, 2),
                "memory_usage": memory_usage,
                "redis_connected": True
            })
            
            return stats
            
        except RedisError as e:
            logger.error(f"Redis error during stats collection: {e}")
            stats = asdict(self.stats)
            stats.update({
                "hit_rate": 0,
                "memory_usage": "0B",
                "redis_connected": False
            })
            return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Check cache health and connectivity."""
        if not self.redis_client:
            return {
                "status": "unhealthy",
                "error": "Redis not connected",
                "connected": False
            }
        
        try:
            # Test connection
            self.redis_client.ping()
            
            # Get basic info
            info = self.redis_client.info()
            
            return {
                "status": "healthy",
                "connected": True,
                "redis_version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0)
            }
            
        except RedisError as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False
            }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.redis_client:
            self.redis_client.close()


class QueryResultCache:
    """Specialized cache for query results with intelligent invalidation."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
    
    def _generate_query_key(self, query: str, params: Dict[str, Any] = None) -> str:
        """Generate cache key for query results."""
        # Create a hash of the query and parameters
        query_data = {
            "query": query,
            "params": params or {}
        }
        query_hash = hashlib.md5(json.dumps(query_data, sort_keys=True).encode()).hexdigest()
        return f"query:{query_hash}"
    
    def get_query_result(self, query: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Get cached query result."""
        key = self._generate_query_key(query, params)
        return self.cache_manager.get(CacheStrategy.QUERY_RESULT, key)
    
    def set_query_result(self, query: str, params: Dict[str, Any], result: Dict[str, Any], ttl: int = 1800) -> bool:
        """Cache query result with 30-minute default TTL."""
        key = self._generate_query_key(query, params)
        return self.cache_manager.set(CacheStrategy.QUERY_RESULT, key, result, ttl)
    
    def invalidate_schema_cache(self) -> int:
        """Invalidate all schema-related caches when schema changes."""
        return self.cache_manager.clear_strategy(CacheStrategy.SCHEMA_METADATA)


class SchemaCache:
    """Specialized cache for schema metadata."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
    
    def get_schema_metadata(self, db_path: str) -> Optional[Dict[str, Any]]:
        """Get cached schema metadata."""
        return self.cache_manager.get(CacheStrategy.SCHEMA_METADATA, db_path)
    
    def set_schema_metadata(self, db_path: str, metadata: Dict[str, Any], ttl: int = 7200) -> bool:
        """Cache schema metadata with 2-hour default TTL."""
        return self.cache_manager.set(CacheStrategy.SCHEMA_METADATA, db_path, metadata, ttl)
    
    def invalidate_schema(self, db_path: str) -> bool:
        """Invalidate schema cache for specific database."""
        return self.cache_manager.delete(CacheStrategy.SCHEMA_METADATA, db_path)


class LLMCache:
    """Specialized cache for LLM responses."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
    
    def _generate_llm_key(self, prompt: str, model: str = "default") -> str:
        """Generate cache key for LLM responses."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return f"llm:{model}:{prompt_hash}"
    
    def get_llm_response(self, prompt: str, model: str = "default") -> Optional[str]:
        """Get cached LLM response."""
        key = self._generate_llm_key(prompt, model)
        return self.cache_manager.get(CacheStrategy.LLM_RESPONSE, key)
    
    def set_llm_response(self, prompt: str, response: str, model: str = "default", ttl: int = 3600) -> bool:
        """Cache LLM response with 1-hour default TTL."""
        key = self._generate_llm_key(prompt, model)
        return self.cache_manager.set(CacheStrategy.LLM_RESPONSE, key, response, ttl)
    
    def clear_llm_cache(self) -> int:
        """Clear all LLM response caches."""
        return self.cache_manager.clear_strategy(CacheStrategy.LLM_RESPONSE)
