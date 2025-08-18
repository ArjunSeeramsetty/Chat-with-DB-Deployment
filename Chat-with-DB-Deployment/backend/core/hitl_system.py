#!/usr/bin/env python3
"""
Human-in-the-Loop (HITL) System
Implements approval workflows, correction interfaces, feedback learning, and trust building
Phase 4.3: Human-in-the-Loop (HITL) System
"""

import asyncio
import logging
import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from pathlib import Path
import uuid

from backend.core.feedback_storage import FeedbackRecord, FeedbackStorage
from backend.core.types import ValidationResult

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Approval status for queries"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"
    NEEDS_REVIEW = "needs_review"


class QueryRiskLevel(Enum):
    """Risk level for queries"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CorrectionType(Enum):
    """Type of correction made"""
    SQL_SYNTAX = "sql_syntax"
    COLUMN_NAME = "column_name"
    TABLE_NAME = "table_name"
    AGGREGATION_FUNCTION = "aggregation_function"
    JOIN_CONDITION = "join_condition"
    WHERE_CLAUSE = "where_clause"
    BUSINESS_LOGIC = "business_logic"
    OTHER = "other"


@dataclass
class ApprovalRequest:
    """Represents an approval request for a query"""
    id: str
    session_id: str
    user_id: str
    original_query: str
    generated_sql: str
    confidence_score: float
    risk_level: QueryRiskLevel
    validation_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    expires_at: datetime = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None


@dataclass
class CorrectionRecord:
    """Represents a correction made to a query"""
    id: str
    approval_request_id: str
    original_sql: str
    corrected_sql: str
    correction_type: CorrectionType
    correction_reason: str
    corrected_by: str
    corrected_at: datetime
    confidence_improvement: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class TrustScore:
    """Represents a trust score for a user or system component"""
    id: str
    entity_id: str  # user_id or component_id
    entity_type: str  # "user" or "system"
    score: float  # 0.0 to 1.0
    factors: Dict[str, float]  # contributing factors
    last_updated: datetime
    metadata: Dict[str, Any] = None


@dataclass
class AuditTrail:
    """Represents an audit trail entry"""
    id: str
    session_id: str
    user_id: str
    action: str
    resource_type: str  # "query", "approval", "correction", "feedback"
    resource_id: str
    details: Dict[str, Any]
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class HITLSystem:
    """
    Human-in-the-Loop (HITL) System
    Implements approval workflows, correction interfaces, feedback learning, and trust building
    """
    
    def __init__(self, db_path: str, feedback_storage: Optional[FeedbackStorage] = None):
        self.db_path = db_path
        self.feedback_storage = feedback_storage or FeedbackStorage(db_path)
        self._initialize_database()
        
        # Configuration
        self.auto_approval_threshold = 0.85  # Confidence threshold for auto-approval
        self.risk_thresholds = {
            QueryRiskLevel.LOW: 0.0,
            QueryRiskLevel.MEDIUM: 0.3,
            QueryRiskLevel.HIGH: 0.7,
            QueryRiskLevel.CRITICAL: 0.9
        }
        
        # Trust scoring weights
        self.trust_weights = {
            "accuracy": 0.4,
            "consistency": 0.2,
            "feedback_quality": 0.2,
            "correction_rate": 0.1,
            "approval_rate": 0.1
        }
        
    def _initialize_database(self):
        """Initialize the HITL database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create approval requests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS approval_requests (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    original_query TEXT NOT NULL,
                    generated_sql TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    validation_result TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    approved_by TEXT,
                    approved_at TIMESTAMP,
                    rejection_reason TEXT
                )
            """)
            
            # Create correction records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS correction_records (
                    id TEXT PRIMARY KEY,
                    approval_request_id TEXT NOT NULL,
                    original_sql TEXT NOT NULL,
                    corrected_sql TEXT NOT NULL,
                    correction_type TEXT NOT NULL,
                    correction_reason TEXT NOT NULL,
                    corrected_by TEXT NOT NULL,
                    corrected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence_improvement REAL DEFAULT 0.0,
                    metadata TEXT,
                    FOREIGN KEY (approval_request_id) REFERENCES approval_requests (id)
                )
            """)
            
            # Create trust scores table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trust_scores (
                    id TEXT PRIMARY KEY,
                    entity_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    score REAL NOT NULL,
                    factors TEXT NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Create audit trail table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_trail (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    details TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_approval_requests_session ON approval_requests(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_approval_requests_status ON approval_requests(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_approval_requests_user ON approval_requests(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_correction_records_approval ON correction_records(approval_request_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trust_scores_entity ON trust_scores(entity_id, entity_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_trail_session ON audit_trail(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_trail_user ON audit_trail(user_id)")
            
            conn.commit()
    
    async def create_approval_request(
        self,
        session_id: str,
        user_id: str,
        original_query: str,
        generated_sql: str,
        confidence_score: float,
        validation_result: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ApprovalRequest:
        """Create a new approval request"""
        
        # Determine risk level based on confidence and validation
        risk_level = self._determine_risk_level(confidence_score, validation_result)
        
        # Check if auto-approval is possible
        status = ApprovalStatus.AUTO_APPROVED if confidence_score >= self.auto_approval_threshold else ApprovalStatus.PENDING
        
        approval_request = ApprovalRequest(
            id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            original_query=original_query,
            generated_sql=generated_sql,
            confidence_score=confidence_score,
            risk_level=risk_level,
            validation_result=validation_result,
            metadata=metadata or {},
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24),  # 24-hour expiration
            status=status
        )
        
        # Store in database
        await self._store_approval_request(approval_request)
        
        # Log audit trail
        await self._log_audit_trail(
            session_id=session_id,
            user_id=user_id,
            action="create_approval_request",
            resource_type="approval",
            resource_id=approval_request.id,
            details={
                "risk_level": risk_level.value,
                "confidence_score": confidence_score,
                "status": status.value
            }
        )
        
        return approval_request
    
    async def approve_query(
        self,
        approval_request_id: str,
        approved_by: str,
        comments: Optional[str] = None
    ) -> bool:
        """Approve a query"""
        
        approval_request = await self._get_approval_request(approval_request_id)
        if not approval_request:
            raise ValueError(f"Approval request {approval_request_id} not found")
        
        if approval_request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Approval request {approval_request_id} is not pending")
        
        # Update approval request
        approval_request.status = ApprovalStatus.APPROVED
        approval_request.approved_by = approved_by
        approval_request.approved_at = datetime.now()
        
        await self._update_approval_request(approval_request)
        
        # Update trust score for the user
        await self._update_user_trust_score(approval_request.user_id, "approval", 1.0)
        
        # Log audit trail
        await self._log_audit_trail(
            session_id=approval_request.session_id,
            user_id=approved_by,
            action="approve_query",
            resource_type="approval",
            resource_id=approval_request_id,
            details={"comments": comments}
        )
        
        return True
    
    async def reject_query(
        self,
        approval_request_id: str,
        rejected_by: str,
        rejection_reason: str,
        suggested_corrections: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Reject a query"""
        
        approval_request = await self._get_approval_request(approval_request_id)
        if not approval_request:
            raise ValueError(f"Approval request {approval_request_id} not found")
        
        if approval_request.status != ApprovalStatus.PENDING:
            raise ValueError(f"Approval request {approval_request_id} is not pending")
        
        # Update approval request
        approval_request.status = ApprovalStatus.REJECTED
        approval_request.approved_by = rejected_by
        approval_request.approved_at = datetime.now()
        approval_request.rejection_reason = rejection_reason
        
        await self._update_approval_request(approval_request)
        
        # Update trust score for the user
        await self._update_user_trust_score(approval_request.user_id, "rejection", -0.1)
        
        # Log audit trail
        await self._log_audit_trail(
            session_id=approval_request.session_id,
            user_id=rejected_by,
            action="reject_query",
            resource_type="approval",
            resource_id=approval_request_id,
            details={
                "rejection_reason": rejection_reason,
                "suggested_corrections": suggested_corrections
            }
        )
        
        return True
    
    async def submit_correction(
        self,
        approval_request_id: str,
        corrected_sql: str,
        correction_type: CorrectionType,
        correction_reason: str,
        corrected_by: str,
        confidence_improvement: float = 0.0
    ) -> CorrectionRecord:
        """Submit a correction for a query"""
        
        approval_request = await self._get_approval_request(approval_request_id)
        if not approval_request:
            raise ValueError(f"Approval request {approval_request_id} not found")
        
        correction_record = CorrectionRecord(
            id=str(uuid.uuid4()),
            approval_request_id=approval_request_id,
            original_sql=approval_request.generated_sql,
            corrected_sql=corrected_sql,
            correction_type=correction_type,
            correction_reason=correction_reason,
            corrected_by=corrected_by,
            corrected_at=datetime.now(),
            confidence_improvement=confidence_improvement
        )
        
        # Store correction record
        await self._store_correction_record(correction_record)
        
        # Update trust score for the user
        await self._update_user_trust_score(corrected_by, "correction", 0.1)
        
        # Log audit trail
        await self._log_audit_trail(
            session_id=approval_request.session_id,
            user_id=corrected_by,
            action="submit_correction",
            resource_type="correction",
            resource_id=correction_record.id,
            details={
                "correction_type": correction_type.value,
                "correction_reason": correction_reason,
                "confidence_improvement": confidence_improvement
            }
        )
        
        return correction_record
    
    async def get_pending_approvals(self, user_id: Optional[str] = None) -> List[ApprovalRequest]:
        """Get pending approval requests"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute("""
                    SELECT * FROM approval_requests 
                    WHERE status = 'pending' AND user_id = ?
                    ORDER BY created_at DESC
                """, (user_id,))
            else:
                cursor.execute("""
                    SELECT * FROM approval_requests 
                    WHERE status = 'pending'
                    ORDER BY created_at DESC
                """)
            
            rows = cursor.fetchall()
            return [self._row_to_approval_request(row) for row in rows]
    
    async def get_user_trust_score(self, user_id: str) -> Optional[TrustScore]:
        """Get trust score for a user"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM trust_scores 
                WHERE entity_id = ? AND entity_type = 'user'
                ORDER BY last_updated DESC
                LIMIT 1
            """, (user_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_trust_score(row)
            return None
    
    async def get_audit_trail(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditTrail]:
        """Get audit trail entries"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM audit_trail WHERE 1=1"
            params = []
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_audit_trail(row) for row in rows]
    
    def _determine_risk_level(self, confidence_score: float, validation_result: Optional[Dict[str, Any]]) -> QueryRiskLevel:
        """Determine risk level based on confidence and validation"""
        
        # Base risk on confidence score
        if confidence_score >= 0.9:
            base_risk = QueryRiskLevel.LOW
        elif confidence_score >= 0.7:
            base_risk = QueryRiskLevel.MEDIUM
        elif confidence_score >= 0.5:
            base_risk = QueryRiskLevel.HIGH
        else:
            base_risk = QueryRiskLevel.CRITICAL
        
        # Adjust based on validation result
        if validation_result:
            layers = validation_result.get("layers", {})
            
            # Check for critical validation failures
            for layer_name, layer in layers.items():
                if not layer.get("is_valid", True):
                    if layer_name in ["syntax", "business_rules"]:
                        return QueryRiskLevel.CRITICAL
                    elif layer_name in ["dry_run", "reasonableness"]:
                        if base_risk == QueryRiskLevel.LOW:
                            base_risk = QueryRiskLevel.MEDIUM
                        elif base_risk == QueryRiskLevel.MEDIUM:
                            base_risk = QueryRiskLevel.HIGH
        
        return base_risk
    
    async def _store_approval_request(self, approval_request: ApprovalRequest):
        """Store approval request in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO approval_requests (
                    id, session_id, user_id, original_query, generated_sql,
                    confidence_score, risk_level, validation_result, metadata,
                    created_at, expires_at, status, approved_by, approved_at, rejection_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                approval_request.id,
                approval_request.session_id,
                approval_request.user_id,
                approval_request.original_query,
                approval_request.generated_sql,
                approval_request.confidence_score,
                approval_request.risk_level.value,
                json.dumps(approval_request.validation_result) if approval_request.validation_result else None,
                json.dumps(approval_request.metadata) if approval_request.metadata else None,
                approval_request.created_at,
                approval_request.expires_at,
                approval_request.status.value,
                approval_request.approved_by,
                approval_request.approved_at,
                approval_request.rejection_reason
            ))
            conn.commit()
    
    async def _get_approval_request(self, approval_request_id: str) -> Optional[ApprovalRequest]:
        """Get approval request by ID"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM approval_requests WHERE id = ?", (approval_request_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_approval_request(row)
            return None
    
    async def _update_approval_request(self, approval_request: ApprovalRequest):
        """Update approval request in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE approval_requests SET
                    status = ?, approved_by = ?, approved_at = ?, rejection_reason = ?
                WHERE id = ?
            """, (
                approval_request.status.value,
                approval_request.approved_by,
                approval_request.approved_at,
                approval_request.rejection_reason,
                approval_request.id
            ))
            conn.commit()
    
    async def _store_correction_record(self, correction_record: CorrectionRecord):
        """Store correction record in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO correction_records (
                    id, approval_request_id, original_sql, corrected_sql,
                    correction_type, correction_reason, corrected_by, corrected_at,
                    confidence_improvement, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                correction_record.id,
                correction_record.approval_request_id,
                correction_record.original_sql,
                correction_record.corrected_sql,
                correction_record.correction_type.value,
                correction_record.correction_reason,
                correction_record.corrected_by,
                correction_record.corrected_at,
                correction_record.confidence_improvement,
                json.dumps(correction_record.metadata) if correction_record.metadata else None
            ))
            conn.commit()
    
    async def _update_user_trust_score(self, user_id: str, action: str, score_change: float):
        """Update user trust score"""
        
        # Get current trust score
        current_score = await self.get_user_trust_score(user_id)
        
        if current_score:
            # Update existing score
            new_score = max(0.0, min(1.0, current_score.score + score_change))
            factors = current_score.factors.copy()
            factors[action] = factors.get(action, 0.0) + score_change
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE trust_scores SET
                        score = ?, factors = ?, last_updated = ?
                    WHERE id = ?
                """, (
                    new_score,
                    json.dumps(factors),
                    datetime.now(),
                    current_score.id
                ))
                conn.commit()
        else:
            # Create new trust score
            factors = {action: score_change}
            trust_score = TrustScore(
                id=str(uuid.uuid4()),
                entity_id=user_id,
                entity_type="user",
                score=max(0.0, min(1.0, 0.5 + score_change)),  # Start at 0.5
                factors=factors,
                last_updated=datetime.now()
            )
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO trust_scores (id, entity_id, entity_type, score, factors, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    trust_score.id,
                    trust_score.entity_id,
                    trust_score.entity_type,
                    trust_score.score,
                    json.dumps(trust_score.factors),
                    trust_score.last_updated
                ))
                conn.commit()
    
    async def _log_audit_trail(
        self,
        session_id: str,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Dict[str, Any],
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Log audit trail entry"""
        
        audit_entry = AuditTrail(
            id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            timestamp=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_trail (
                    id, session_id, user_id, action, resource_type, resource_id,
                    details, timestamp, ip_address, user_agent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_entry.id,
                audit_entry.session_id,
                audit_entry.user_id,
                audit_entry.action,
                audit_entry.resource_type,
                audit_entry.resource_id,
                json.dumps(audit_entry.details),
                audit_entry.timestamp,
                audit_entry.ip_address,
                audit_entry.user_agent
            ))
            conn.commit()
    
    def _row_to_approval_request(self, row) -> ApprovalRequest:
        """Convert database row to ApprovalRequest"""
        return ApprovalRequest(
            id=row[0],
            session_id=row[1],
            user_id=row[2],
            original_query=row[3],
            generated_sql=row[4],
            confidence_score=row[5],
            risk_level=QueryRiskLevel(row[6]),
            validation_result=json.loads(row[7]) if row[7] else None,
            metadata=json.loads(row[8]) if row[8] else None,
            created_at=datetime.fromisoformat(row[9]) if row[9] else None,
            expires_at=datetime.fromisoformat(row[10]) if row[10] else None,
            status=ApprovalStatus(row[11]),
            approved_by=row[12],
            approved_at=datetime.fromisoformat(row[13]) if row[13] else None,
            rejection_reason=row[14]
        )
    
    def _row_to_trust_score(self, row) -> TrustScore:
        """Convert database row to TrustScore"""
        return TrustScore(
            id=row[0],
            entity_id=row[1],
            entity_type=row[2],
            score=row[3],
            factors=json.loads(row[4]) if row[4] else {},
            last_updated=datetime.fromisoformat(row[5]) if row[5] else None,
            metadata=json.loads(row[6]) if row[6] else None
        )
    
    def _row_to_audit_trail(self, row) -> AuditTrail:
        """Convert database row to AuditTrail"""
        return AuditTrail(
            id=row[0],
            session_id=row[1],
            user_id=row[2],
            action=row[3],
            resource_type=row[4],
            resource_id=row[5],
            details=json.loads(row[6]) if row[6] else {},
            timestamp=datetime.fromisoformat(row[7]) if row[7] else None,
            ip_address=row[8],
            user_agent=row[9]
        )
