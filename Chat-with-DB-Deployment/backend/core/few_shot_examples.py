#!/usr/bin/env python3
"""
Few-Shot Example Retrieval Module
Stores and retrieves query-SQL pairs to improve SQL generation accuracy through example-based learning
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class QueryExample:
    """Represents a query-SQL example pair"""
    id: int
    natural_query: str
    generated_sql: str
    confidence: float
    success: bool
    execution_time: float
    user_feedback: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    tags: List[str] = None
    complexity: str = "medium"  # simple, medium, complex
    domain: str = "energy"
    query_type: str = "aggregation"  # aggregation, comparison, trend, etc.


class FewShotExampleRepository:
    """Repository for storing and retrieving query-SQL examples"""
    
    def __init__(self, db_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.embedding_model = SentenceTransformer(embedding_model)
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize the examples database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create examples table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    natural_query TEXT NOT NULL,
                    generated_sql TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    execution_time REAL,
                    user_feedback TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT,
                    complexity TEXT DEFAULT 'medium',
                    domain TEXT DEFAULT 'energy',
                    query_type TEXT DEFAULT 'aggregation',
                    query_embedding BLOB
                )
            """)
            
            # Create indexes for efficient retrieval
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_examples_success 
                ON query_examples(success)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_examples_complexity 
                ON query_examples(complexity)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_examples_query_type 
                ON query_examples(query_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_examples_confidence 
                ON query_examples(confidence)
            """)
            
            conn.commit()
            
    def add_example(self, example: QueryExample) -> int:
        """Add a new query example to the repository"""
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(example.natural_query)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO query_examples (
                        natural_query, generated_sql, confidence, success, 
                        execution_time, user_feedback, tags, complexity, 
                        domain, query_type, query_embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    example.natural_query,
                    example.generated_sql,
                    example.confidence,
                    example.success,
                    example.execution_time,
                    example.user_feedback,
                    json.dumps(example.tags or []),
                    example.complexity,
                    example.domain,
                    example.query_type,
                    query_embedding.tobytes()
                ))
                
                example_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Added query example {example_id}: {example.natural_query[:50]}...")
                return example_id
                
        except Exception as e:
            logger.error(f"Failed to add query example: {e}")
            raise
            
    def get_example(self, example_id: int) -> Optional[QueryExample]:
        """Retrieve a specific example by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, natural_query, generated_sql, confidence, success,
                           execution_time, user_feedback, created_at, updated_at,
                           tags, complexity, domain, query_type
                    FROM query_examples
                    WHERE id = ?
                """, (example_id,))
                
                row = cursor.fetchone()
                if row:
                    return QueryExample(
                        id=row[0],
                        natural_query=row[1],
                        generated_sql=row[2],
                        confidence=row[3],
                        success=row[4],
                        execution_time=row[5],
                        user_feedback=row[6],
                        created_at=datetime.fromisoformat(row[7]) if row[7] else None,
                        updated_at=datetime.fromisoformat(row[8]) if row[8] else None,
                        tags=json.loads(row[9]) if row[9] else [],
                        complexity=row[10],
                        domain=row[11],
                        query_type=row[12]
                    )
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve example {example_id}: {e}")
            return None
            
    def search_similar_examples(
        self, 
        query: str, 
        limit: int = 5, 
        min_confidence: float = 0.7,
        only_successful: bool = True
    ) -> List[Tuple[QueryExample, float]]:
        """Search for similar examples using semantic similarity"""
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(query)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build the query
                sql = """
                    SELECT id, natural_query, generated_sql, confidence, success,
                           execution_time, user_feedback, created_at, updated_at,
                           tags, complexity, domain, query_type, query_embedding
                    FROM query_examples
                    WHERE confidence >= ?
                """
                params = [min_confidence]
                
                if only_successful:
                    sql += " AND success = 1"
                    
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                # Calculate similarities
                similarities = []
                for row in rows:
                    example = QueryExample(
                        id=row[0],
                        natural_query=row[1],
                        generated_sql=row[2],
                        confidence=row[3],
                        success=row[4],
                        execution_time=row[5],
                        user_feedback=row[6],
                        created_at=datetime.fromisoformat(row[7]) if row[7] else None,
                        updated_at=datetime.fromisoformat(row[8]) if row[8] else None,
                        tags=json.loads(row[9]) if row[9] else [],
                        complexity=row[10],
                        domain=row[11],
                        query_type=row[12]
                    )
                    
                    # Calculate cosine similarity
                    stored_embedding = np.frombuffer(row[13], dtype=np.float32)
                    similarity = self._cosine_similarity(query_embedding, stored_embedding)
                    similarities.append((example, similarity))
                
                # Sort by similarity and return top results
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:limit]
                
        except Exception as e:
            logger.error(f"Failed to search similar examples: {e}")
            return []
            
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except:
            return 0.0
            
    def get_examples_by_type(self, query_type: str, limit: int = 10) -> List[QueryExample]:
        """Get examples by query type"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, natural_query, generated_sql, confidence, success,
                           execution_time, user_feedback, created_at, updated_at,
                           tags, complexity, domain, query_type
                    FROM query_examples
                    WHERE query_type = ? AND success = 1
                    ORDER BY confidence DESC
                    LIMIT ?
                """, (query_type, limit))
                
                examples = []
                for row in cursor.fetchall():
                    example = QueryExample(
                        id=row[0],
                        natural_query=row[1],
                        generated_sql=row[2],
                        confidence=row[3],
                        success=row[4],
                        execution_time=row[5],
                        user_feedback=row[6],
                        created_at=datetime.fromisoformat(row[7]) if row[7] else None,
                        updated_at=datetime.fromisoformat(row[8]) if row[8] else None,
                        tags=json.loads(row[9]) if row[9] else [],
                        complexity=row[10],
                        domain=row[11],
                        query_type=row[12]
                    )
                    examples.append(example)
                    
                return examples
                
        except Exception as e:
            logger.error(f"Failed to get examples by type {query_type}: {e}")
            return []
            
    def update_example(self, example_id: int, updates: Dict[str, Any]) -> bool:
        """Update an existing example"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build update query
                set_clauses = []
                params = []
                
                for key, value in updates.items():
                    if key in ['natural_query', 'generated_sql', 'confidence', 'success', 
                              'execution_time', 'user_feedback', 'tags', 'complexity', 
                              'domain', 'query_type']:
                        set_clauses.append(f"{key} = ?")
                        if key == 'tags':
                            params.append(json.dumps(value))
                        else:
                            params.append(value)
                
                if set_clauses:
                    set_clauses.append("updated_at = CURRENT_TIMESTAMP")
                    sql = f"UPDATE query_examples SET {', '.join(set_clauses)} WHERE id = ?"
                    params.append(example_id)
                    
                    cursor.execute(sql, params)
                    conn.commit()
                    
                    logger.info(f"Updated example {example_id}")
                    return True
                    
                return False
                
        except Exception as e:
            logger.error(f"Failed to update example {example_id}: {e}")
            return False
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total examples
                cursor.execute("SELECT COUNT(*) FROM query_examples")
                total_examples = cursor.fetchone()[0]
                
                # Successful examples
                cursor.execute("SELECT COUNT(*) FROM query_examples WHERE success = 1")
                successful_examples = cursor.fetchone()[0]
                
                # Average confidence
                cursor.execute("SELECT AVG(confidence) FROM query_examples")
                avg_confidence = cursor.fetchone()[0] or 0.0
                
                # Examples by type
                cursor.execute("""
                    SELECT query_type, COUNT(*) 
                    FROM query_examples 
                    GROUP BY query_type
                """)
                examples_by_type = dict(cursor.fetchall())
                
                # Examples by complexity
                cursor.execute("""
                    SELECT complexity, COUNT(*) 
                    FROM query_examples 
                    GROUP BY complexity
                """)
                examples_by_complexity = dict(cursor.fetchall())
                
                return {
                    "total_examples": total_examples,
                    "successful_examples": successful_examples,
                    "success_rate": (successful_examples / total_examples * 100) if total_examples > 0 else 0,
                    "average_confidence": avg_confidence,
                    "examples_by_type": examples_by_type,
                    "examples_by_complexity": examples_by_complexity
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


class FewShotExampleRetriever:
    """Retrieves and formats few-shot examples for SQL generation"""
    
    def __init__(self, repository: FewShotExampleRepository):
        self.repository = repository
        
    def retrieve_examples_for_query(
        self, 
        query: str, 
        max_examples: int = 3,
        min_similarity: float = 0.3
    ) -> List[QueryExample]:
        """Retrieve relevant examples for a given query"""
        try:
            # Search for similar examples
            similar_examples = self.repository.search_similar_examples(
                query, 
                limit=max_examples * 2,  # Get more to filter by similarity
                min_confidence=0.7,
                only_successful=True
            )
            
            # Filter by similarity threshold
            filtered_examples = [
                example for example, similarity in similar_examples 
                if similarity >= min_similarity
            ]
            
            # Return top examples
            return filtered_examples[:max_examples]
            
        except Exception as e:
            logger.error(f"Failed to retrieve examples for query: {e}")
            return []
            
    def format_examples_for_prompt(self, examples: List[QueryExample]) -> str:
        """Format examples for inclusion in LLM prompts"""
        if not examples:
            return ""
            
        formatted_examples = []
        for i, example in enumerate(examples, 1):
            formatted_example = f"""
Example {i}:
Query: {example.natural_query}
SQL: {example.generated_sql}
Confidence: {example.confidence:.2f}
Success: {'Yes' if example.success else 'No'}
"""
            formatted_examples.append(formatted_example)
            
        return "\n".join(formatted_examples)
        
    def get_examples_by_complexity(self, complexity: str, limit: int = 5) -> List[QueryExample]:
        """Get examples by complexity level"""
        try:
            with sqlite3.connect(self.repository.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, natural_query, generated_sql, confidence, success,
                           execution_time, user_feedback, created_at, updated_at,
                           tags, complexity, domain, query_type
                    FROM query_examples
                    WHERE complexity = ? AND success = 1
                    ORDER BY confidence DESC
                    LIMIT ?
                """, (complexity, limit))
                
                examples = []
                for row in cursor.fetchall():
                    example = QueryExample(
                        id=row[0],
                        natural_query=row[1],
                        generated_sql=row[2],
                        confidence=row[3],
                        success=row[4],
                        execution_time=row[5],
                        user_feedback=row[6],
                        created_at=datetime.fromisoformat(row[7]) if row[7] else None,
                        updated_at=datetime.fromisoformat(row[8]) if row[8] else None,
                        tags=json.loads(row[9]) if row[9] else [],
                        complexity=row[10],
                        domain=row[11],
                        query_type=row[12]
                    )
                    examples.append(example)
                    
                return examples
                
        except Exception as e:
            logger.error(f"Failed to get examples by complexity {complexity}: {e}")
            return []
