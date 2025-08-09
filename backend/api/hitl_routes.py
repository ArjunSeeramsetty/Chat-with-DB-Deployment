#!/usr/bin/env python3
"""
HITL API Routes
Provides endpoints for Human-in-the-Loop (HITL) system functionality
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from backend.core.hitl_system import (
    HITLSystem, ApprovalRequest, CorrectionRecord, TrustScore, AuditTrail,
    ApprovalStatus, QueryRiskLevel, CorrectionType
)
from backend.core.feedback_storage import FeedbackStorage

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models for API requests/responses
class ApprovalRequestCreate(BaseModel):
    session_id: str
    user_id: str
    original_query: str
    generated_sql: str
    confidence_score: float
    validation_result: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ApprovalRequestResponse(BaseModel):
    id: str
    session_id: str
    user_id: str
    original_query: str
    generated_sql: str
    confidence_score: float
    risk_level: str
    status: str
    created_at: datetime
    expires_at: datetime
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None


class ApprovalAction(BaseModel):
    approved_by: str
    comments: Optional[str] = None


class RejectionAction(BaseModel):
    rejected_by: str
    rejection_reason: str
    suggested_corrections: Optional[Dict[str, Any]] = None


class CorrectionSubmit(BaseModel):
    corrected_sql: str
    correction_type: str
    correction_reason: str
    corrected_by: str
    confidence_improvement: float = 0.0


class CorrectionResponse(BaseModel):
    id: str
    approval_request_id: str
    original_sql: str
    corrected_sql: str
    correction_type: str
    correction_reason: str
    corrected_by: str
    corrected_at: datetime
    confidence_improvement: float


class TrustScoreResponse(BaseModel):
    id: str
    entity_id: str
    entity_type: str
    score: float
    factors: Dict[str, float]
    last_updated: datetime


class AuditTrailResponse(BaseModel):
    id: str
    session_id: str
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


# Dependency injection
def get_hitl_system() -> HITLSystem:
    """Get HITL system instance"""
    # In a real implementation, this would be injected from the main app
    db_path = "backend/energy_data.db"
    feedback_storage = FeedbackStorage(db_path)
    return HITLSystem(db_path, feedback_storage)


# Approval workflow endpoints
@router.post("/api/v1/hitl/approval-requests", response_model=ApprovalRequestResponse)
async def create_approval_request(
    request: ApprovalRequestCreate,
    hitl_system: HITLSystem = Depends(get_hitl_system)
):
    """Create a new approval request"""
    try:
        approval_request = await hitl_system.create_approval_request(
            session_id=request.session_id,
            user_id=request.user_id,
            original_query=request.original_query,
            generated_sql=request.generated_sql,
            confidence_score=request.confidence_score,
            validation_result=request.validation_result,
            metadata=request.metadata
        )
        
        return ApprovalRequestResponse(
            id=approval_request.id,
            session_id=approval_request.session_id,
            user_id=approval_request.user_id,
            original_query=approval_request.original_query,
            generated_sql=approval_request.generated_sql,
            confidence_score=approval_request.confidence_score,
            risk_level=approval_request.risk_level.value,
            status=approval_request.status.value,
            created_at=approval_request.created_at,
            expires_at=approval_request.expires_at,
            approved_by=approval_request.approved_by,
            approved_at=approval_request.approved_at,
            rejection_reason=approval_request.rejection_reason
        )
    except Exception as e:
        logger.error(f"Failed to create approval request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/hitl/approval-requests", response_model=List[ApprovalRequestResponse])
async def get_pending_approvals(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    hitl_system: HITLSystem = Depends(get_hitl_system)
):
    """Get pending approval requests"""
    try:
        approval_requests = await hitl_system.get_pending_approvals(user_id=user_id)
        
        return [
            ApprovalRequestResponse(
                id=ar.id,
                session_id=ar.session_id,
                user_id=ar.user_id,
                original_query=ar.original_query,
                generated_sql=ar.generated_sql,
                confidence_score=ar.confidence_score,
                risk_level=ar.risk_level.value,
                status=ar.status.value,
                created_at=ar.created_at,
                expires_at=ar.expires_at,
                approved_by=ar.approved_by,
                approved_at=ar.approved_at,
                rejection_reason=ar.rejection_reason
            )
            for ar in approval_requests
        ]
    except Exception as e:
        logger.error(f"Failed to get pending approvals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/hitl/approval-requests/{approval_request_id}/approve")
async def approve_query(
    approval_request_id: str,
    action: ApprovalAction,
    hitl_system: HITLSystem = Depends(get_hitl_system)
):
    """Approve a query"""
    try:
        success = await hitl_system.approve_query(
            approval_request_id=approval_request_id,
            approved_by=action.approved_by,
            comments=action.comments
        )
        
        if success:
            return {"status": "approved", "message": "Query approved successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to approve query")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to approve query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/hitl/approval-requests/{approval_request_id}/reject")
async def reject_query(
    approval_request_id: str,
    action: RejectionAction,
    hitl_system: HITLSystem = Depends(get_hitl_system)
):
    """Reject a query"""
    try:
        success = await hitl_system.reject_query(
            approval_request_id=approval_request_id,
            rejected_by=action.rejected_by,
            rejection_reason=action.rejection_reason,
            suggested_corrections=action.suggested_corrections
        )
        
        if success:
            return {"status": "rejected", "message": "Query rejected successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to reject query")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to reject query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Correction endpoints
@router.post("/api/v1/hitl/approval-requests/{approval_request_id}/corrections", response_model=CorrectionResponse)
async def submit_correction(
    approval_request_id: str,
    correction: CorrectionSubmit,
    hitl_system: HITLSystem = Depends(get_hitl_system)
):
    """Submit a correction for a query"""
    try:
        correction_type = CorrectionType(correction.correction_type)
        
        correction_record = await hitl_system.submit_correction(
            approval_request_id=approval_request_id,
            corrected_sql=correction.corrected_sql,
            correction_type=correction_type,
            correction_reason=correction.correction_reason,
            corrected_by=correction.corrected_by,
            confidence_improvement=correction.confidence_improvement
        )
        
        return CorrectionResponse(
            id=correction_record.id,
            approval_request_id=correction_record.approval_request_id,
            original_sql=correction_record.original_sql,
            corrected_sql=correction_record.corrected_sql,
            correction_type=correction_record.correction_type.value,
            correction_reason=correction_record.correction_reason,
            corrected_by=correction_record.corrected_by,
            corrected_at=correction_record.corrected_at,
            confidence_improvement=correction_record.confidence_improvement
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to submit correction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Trust management endpoints
@router.get("/api/v1/hitl/trust-scores/{user_id}", response_model=TrustScoreResponse)
async def get_user_trust_score(
    user_id: str,
    hitl_system: HITLSystem = Depends(get_hitl_system)
):
    """Get trust score for a user"""
    try:
        trust_score = await hitl_system.get_user_trust_score(user_id)
        
        if trust_score:
            return TrustScoreResponse(
                id=trust_score.id,
                entity_id=trust_score.entity_id,
                entity_type=trust_score.entity_type,
                score=trust_score.score,
                factors=trust_score.factors,
                last_updated=trust_score.last_updated
            )
        else:
            raise HTTPException(status_code=404, detail="Trust score not found")
    except Exception as e:
        logger.error(f"Failed to get trust score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Audit trail endpoints
@router.get("/api/v1/hitl/audit-trail", response_model=List[AuditTrailResponse])
async def get_audit_trail(
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(100, description="Maximum number of entries to return"),
    hitl_system: HITLSystem = Depends(get_hitl_system)
):
    """Get audit trail entries"""
    try:
        audit_entries = await hitl_system.get_audit_trail(
            session_id=session_id,
            user_id=user_id,
            limit=limit
        )
        
        return [
            AuditTrailResponse(
                id=entry.id,
                session_id=entry.session_id,
                user_id=entry.user_id,
                action=entry.action,
                resource_type=entry.resource_type,
                resource_id=entry.resource_id,
                details=entry.details,
                timestamp=entry.timestamp,
                ip_address=entry.ip_address,
                user_agent=entry.user_agent
            )
            for entry in audit_entries
        ]
    except Exception as e:
        logger.error(f"Failed to get audit trail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@router.get("/api/v1/hitl/health")
async def hitl_health_check(hitl_system: HITLSystem = Depends(get_hitl_system)):
    """Health check for HITL system"""
    try:
        # Test database connection
        pending_count = len(await hitl_system.get_pending_approvals())
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "pending_approvals": pending_count,
            "system": "HITL"
        }
    except Exception as e:
        logger.error(f"HITL health check failed: {e}")
        raise HTTPException(status_code=503, detail="HITL system unhealthy")
