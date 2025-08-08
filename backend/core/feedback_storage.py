#!/usr/bin/env python3
"""
Feedback Storage and Processing Module
Handles user feedback, execution traces, and continuous learning for SQL generation improvement
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
class FeedbackRecord:
    """Represents a feedback record for a query"""
    id: Optional[int] = None
    session_id: str = ""
    user_id: str = ""
    original_query: str = ""
    generated_sql: str = ""
    executed_sql: str = ""
    feedback_text: str = ""
    is_correct: bool = True
    accuracy_rating: float = 0.0
    usefulness_rating: float = 0.0
    execution_time: float = 0.0
    row_count: int = 0
    error_message: Optional[str] = None
    processing_mode: str = "adaptive"
    confidence_score: float = 0.0
    query_complexity: str = "medium"
    query_type: str = "aggregation"
    tags: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class ExecutionTrace:
    """Represents an execution trace for a query"""
    id: Optional[int] = None
    session_id: str = ""
    query_id: str = ""
    step_name: str = ""
    step_data: Dict[str, Any] = None
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    created_at: datetime = None


class FeedbackStorage:
    """Storage system for feedback and execution traces"""
    
    def __init__(self, db_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.embedding_model = SentenceTransformer(embedding_model)
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize the feedback database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    original_query TEXT NOT NULL,
                    generated_sql TEXT NOT NULL,
                    executed_sql TEXT,
                    feedback_text TEXT,
                    is_correct BOOLEAN NOT NULL,
                    accuracy_rating REAL DEFAULT 0.0,
                    usefulness_rating REAL DEFAULT 0.0,
                    execution_time REAL DEFAULT 0.0,
                    row_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    processing_mode TEXT DEFAULT 'adaptive',
                    confidence_score REAL DEFAULT 0.0,
                    query_complexity TEXT DEFAULT 'medium',
                    query_type TEXT DEFAULT 'aggregation',
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    query_embedding BLOB
                )
            """)
            
            # Create execution traces table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    step_data TEXT,
                    execution_time REAL DEFAULT 0.0,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create performance analytics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    total_queries INTEGER DEFAULT 0,
                    successful_queries INTEGER DEFAULT 0,
                    failed_queries INTEGER DEFAULT 0,
                    average_accuracy REAL DEFAULT 0.0,
                    average_execution_time REAL DEFAULT 0.0,
                    processing_mode_distribution TEXT,
                    query_complexity_distribution TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for efficient retrieval
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_session_id 
                ON feedback_records(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_user_id 
                ON feedback_records(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_is_correct 
                ON feedback_records(is_correct)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_created_at 
                ON feedback_records(created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_session_id 
                ON execution_traces(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_query_id 
                ON execution_traces(query_id)
            """)
            
            conn.commit()
            
    def store_feedback(self, feedback: FeedbackRecord) -> int:
        """Store a feedback record"""
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(feedback.original_query)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO feedback_records (
                        session_id, user_id, original_query, generated_sql, executed_sql,
                        feedback_text, is_correct, accuracy_rating, usefulness_rating,
                        execution_time, row_count, error_message, processing_mode,
                        confidence_score, query_complexity, query_type, tags, query_embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.session_id,
                    feedback.user_id,
                    feedback.original_query,
                    feedback.generated_sql,
                    feedback.executed_sql,
                    feedback.feedback_text,
                    feedback.is_correct,
                    feedback.accuracy_rating,
                    feedback.usefulness_rating,
                    feedback.execution_time,
                    feedback.row_count,
                    feedback.error_message,
                    feedback.processing_mode,
                    feedback.confidence_score,
                    feedback.query_complexity,
                    feedback.query_type,
                    json.dumps(feedback.tags or []),
                    query_embedding.tobytes()
                ))
                
                feedback_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Stored feedback record {feedback_id} for session {feedback.session_id}")
                return feedback_id
                
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            raise
            
    def store_execution_trace(self, trace: ExecutionTrace) -> int:
        """Store an execution trace"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO execution_traces (
                        session_id, query_id, step_name, step_data, execution_time,
                        success, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    trace.session_id,
                    trace.query_id,
                    trace.step_name,
                    json.dumps(trace.step_data or {}),
                    trace.execution_time,
                    trace.success,
                    trace.error_message
                ))
                
                trace_id = cursor.lastrowid
                conn.commit()
                
                logger.debug(f"Stored execution trace {trace_id} for query {trace.query_id}")
                return trace_id
                
        except Exception as e:
            logger.error(f"Failed to store execution trace: {e}")
            raise
            
    def get_feedback_by_session(self, session_id: str) -> List[FeedbackRecord]:
        """Get all feedback records for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM feedback_records 
                    WHERE session_id = ? 
                    ORDER BY created_at DESC
                """, (session_id,))
                
                records = []
                for row in cursor.fetchall():
                    record = FeedbackRecord(
                        id=row[0],
                        session_id=row[1],
                        user_id=row[2],
                        original_query=row[3],
                        generated_sql=row[4],
                        executed_sql=row[5],
                        feedback_text=row[6],
                        is_correct=bool(row[7]),
                        accuracy_rating=row[8],
                        usefulness_rating=row[9],
                        execution_time=row[10],
                        row_count=row[11],
                        error_message=row[12],
                        processing_mode=row[13],
                        confidence_score=row[14],
                        query_complexity=row[15],
                        query_type=row[16],
                        tags=json.loads(row[17]) if row[17] else [],
                        created_at=datetime.fromisoformat(row[18]),
                        updated_at=datetime.fromisoformat(row[19])
                    )
                    records.append(record)
                    
                return records
                
        except Exception as e:
            logger.error(f"Failed to get feedback by session: {e}")
            return []
            
    def get_similar_feedback(self, query: str, limit: int = 5) -> List[FeedbackRecord]:
        """Get similar feedback records based on query similarity"""
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode(query)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM feedback_records 
                    WHERE is_correct = 1 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit * 2,))  # Get more records for similarity filtering
                
                records = []
                similarities = []
                
                for row in cursor.fetchall():
                    # Get stored embedding
                    stored_embedding_bytes = row[20]  # query_embedding column
                    if stored_embedding_bytes:
                        stored_embedding = np.frombuffer(stored_embedding_bytes, dtype=np.float32)
                        
                        # Calculate cosine similarity
                        similarity = np.dot(query_embedding, stored_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                        )
                        
                        record = FeedbackRecord(
                            id=row[0],
                            session_id=row[1],
                            user_id=row[2],
                            original_query=row[3],
                            generated_sql=row[4],
                            executed_sql=row[5],
                            feedback_text=row[6],
                            is_correct=bool(row[7]),
                            accuracy_rating=row[8],
                            usefulness_rating=row[9],
                            execution_time=row[10],
                            row_count=row[11],
                            error_message=row[12],
                            processing_mode=row[13],
                            confidence_score=row[14],
                            query_complexity=row[15],
                            query_type=row[16],
                            tags=json.loads(row[17]) if row[17] else [],
                            created_at=datetime.fromisoformat(row[18]),
                            updated_at=datetime.fromisoformat(row[19])
                        )
                        
                        records.append(record)
                        similarities.append(similarity)
                
                # Sort by similarity and return top results
                if records and similarities:
                    sorted_pairs = sorted(zip(records, similarities), key=lambda x: x[1], reverse=True)
                    return [record for record, _ in sorted_pairs[:limit]]
                    
                return records[:limit]
                
        except Exception as e:
            logger.error(f"Failed to get similar feedback: {e}")
            return []
            
    def get_performance_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get performance analytics for the specified period"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_queries,
                        SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as successful_queries,
                        SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) as failed_queries,
                        AVG(accuracy_rating) as average_accuracy,
                        AVG(execution_time) as average_execution_time,
                        AVG(confidence_score) as average_confidence
                    FROM feedback_records 
                    WHERE created_at >= datetime('now', '-{} days')
                """.format(days))
                
                row = cursor.fetchone()
                if row:
                    return {
                        "total_queries": row[0] or 0,
                        "successful_queries": row[1] or 0,
                        "failed_queries": row[2] or 0,
                        "success_rate": (row[1] / row[0] * 100) if row[0] > 0 else 0,
                        "average_accuracy": row[3] or 0.0,
                        "average_execution_time": row[4] or 0.0,
                        "average_confidence": row[5] or 0.0
                    }
                    
                return {
                    "total_queries": 0,
                    "successful_queries": 0,
                    "failed_queries": 0,
                    "success_rate": 0.0,
                    "average_accuracy": 0.0,
                    "average_execution_time": 0.0,
                    "average_confidence": 0.0
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance analytics: {e}")
            return {}
            
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights for model improvement"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get common error patterns
                cursor.execute("""
                    SELECT error_message, COUNT(*) as count
                    FROM feedback_records 
                    WHERE error_message IS NOT NULL AND error_message != ''
                    GROUP BY error_message 
                    ORDER BY count DESC 
                    LIMIT 5
                """)
                
                error_patterns = [{"error": row[0], "count": row[1]} for row in cursor.fetchall()]
                
                # Get query complexity distribution
                cursor.execute("""
                    SELECT query_complexity, COUNT(*) as count
                    FROM feedback_records 
                    GROUP BY query_complexity
                """)
                
                complexity_distribution = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Get processing mode effectiveness
                cursor.execute("""
                    SELECT processing_mode, 
                           COUNT(*) as total,
                           SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as successful
                    FROM feedback_records 
                    GROUP BY processing_mode
                """)
                
                mode_effectiveness = {}
                for row in cursor.fetchall():
                    mode_effectiveness[row[0]] = {
                        "total": row[1],
                        "successful": row[2],
                        "success_rate": (row[2] / row[1] * 100) if row[1] > 0 else 0
                    }
                
                return {
                    "error_patterns": error_patterns,
                    "complexity_distribution": complexity_distribution,
                    "mode_effectiveness": mode_effectiveness
                }
                
        except Exception as e:
            logger.error(f"Failed to get learning insights: {e}")
            return {}
