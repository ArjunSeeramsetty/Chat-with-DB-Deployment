#!/usr/bin/env python3
"""
Test Feedback-Driven Fine-Tuning System
Validates feedback storage, processing, and learning capabilities
"""

import asyncio
import json
import logging
import sqlite3
import tempfile
import time
from datetime import datetime
from typing import Dict, Any

from backend.core.feedback_storage import FeedbackStorage, FeedbackRecord, ExecutionTrace
from backend.services.enhanced_rag_service import EnhancedRAGService
from backend.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackSystemTester:
    """Test suite for feedback-driven fine-tuning system"""
    
    def __init__(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        self.feedback_storage = FeedbackStorage(self.db_path)
        self.enhanced_rag = EnhancedRAGService(self.db_path)
        
    async def run_all_tests(self):
        """Run all feedback system tests"""
        logger.info("🚀 Starting Feedback System Tests")
        
        test_results = {
            "feedback_storage": await self.test_feedback_storage(),
            "feedback_processing": await self.test_feedback_processing(),
            "similar_feedback": await self.test_similar_feedback(),
            "performance_analytics": await self.test_performance_analytics(),
            "learning_insights": await self.test_learning_insights(),
            "integration": await self.test_integration()
        }
        
        # Calculate overall success rate
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result["success"])
        success_rate = (passed_tests / total_tests) * 100
        
        logger.info(f"📊 Test Results: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)")
        
        return {
            "overall_success": success_rate >= 80,
            "success_rate": success_rate,
            "test_results": test_results
        }
        
    async def test_feedback_storage(self) -> Dict[str, Any]:
        """Test feedback storage functionality"""
        logger.info("Testing feedback storage...")
        
        try:
            # Test storing feedback record
            feedback_record = FeedbackRecord(
                session_id="test_session_1",
                user_id="test_user",
                original_query="Show me the average energy shortage by state",
                generated_sql="SELECT state, AVG(EnergyShortage) FROM FactStateDailyEnergy GROUP BY state",
                executed_sql="SELECT state, AVG(EnergyShortage) FROM FactStateDailyEnergy GROUP BY state",
                feedback_text="Good SQL generation",
                is_correct=True,
                accuracy_rating=0.9,
                usefulness_rating=0.8,
                execution_time=1.2,
                row_count=25,
                processing_mode="semantic_first",
                confidence_score=0.85,
                query_complexity="medium",
                query_type="aggregation",
                tags=["energy", "shortage", "state"],
                created_at=datetime.now()
            )
            
            feedback_id = self.feedback_storage.store_feedback(feedback_record)
            assert feedback_id > 0, "Feedback ID should be positive"
            
            # Test storing execution trace
            trace = ExecutionTrace(
                session_id="test_session_1",
                query_id=str(feedback_id),
                step_name="query_processing",
                step_data={"processing_mode": "semantic_first", "confidence": 0.85},
                execution_time=1.2,
                success=True,
                created_at=datetime.now()
            )
            
            trace_id = self.feedback_storage.store_execution_trace(trace)
            assert trace_id > 0, "Trace ID should be positive"
            
            # Test retrieving feedback by session
            session_feedback = self.feedback_storage.get_feedback_by_session("test_session_1")
            assert len(session_feedback) > 0, "Should retrieve feedback for session"
            assert session_feedback[0].original_query == "Show me the average energy shortage by state"
            
            logger.info("✅ Feedback storage tests passed")
            return {"success": True, "message": "Feedback storage working correctly"}
            
        except Exception as e:
            logger.error(f"❌ Feedback storage test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_feedback_processing(self) -> Dict[str, Any]:
        """Test feedback processing functionality"""
        logger.info("Testing feedback processing...")
        
        try:
            # Test feedback processing
            feedback_data = {
                "user_id": "test_user",
                "original_query": "What is the total energy consumption by region?",
                "generated_sql": "SELECT region, SUM(EnergyMet) FROM FactAllIndiaDailySummary GROUP BY region",
                "executed_sql": "SELECT region, SUM(EnergyMet) FROM FactAllIndiaDailySummary GROUP BY region",
                "feedback_text": "Accurate SQL generation",
                "is_correct": True,
                "accuracy_rating": 0.95,
                "usefulness_rating": 0.9,
                "execution_time": 0.8,
                "row_count": 5,
                "processing_mode": "hybrid",
                "confidence_score": 0.9,
                "query_complexity": "simple",
                "query_type": "aggregation",
                "tags": ["energy", "consumption", "region"]
            }
            
            result = await self.enhanced_rag.process_feedback("test_session_2", feedback_data)
            
            assert result["success"], "Feedback processing should succeed"
            assert "feedback_id" in result, "Should return feedback ID"
            assert "insights" in result, "Should return learning insights"
            
            logger.info("✅ Feedback processing tests passed")
            return {"success": True, "message": "Feedback processing working correctly"}
            
        except Exception as e:
            logger.error(f"❌ Feedback processing test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_similar_feedback(self) -> Dict[str, Any]:
        """Test similar feedback retrieval"""
        logger.info("Testing similar feedback retrieval...")
        
        try:
            # Add some test feedback first
            test_queries = [
                "Show me the average energy shortage by state",
                "What is the total energy consumption by region?",
                "Display energy shortage trends over time",
                "Compare energy consumption between states"
            ]
            
            for i, query in enumerate(test_queries):
                feedback_record = FeedbackRecord(
                    session_id=f"test_session_{i+3}",
                    user_id="test_user",
                    original_query=query,
                    generated_sql=f"SELECT * FROM test_table_{i}",
                    executed_sql=f"SELECT * FROM test_table_{i}",
                    feedback_text="Test feedback",
                    is_correct=True,
                    accuracy_rating=0.8 + (i * 0.05),
                    usefulness_rating=0.7 + (i * 0.05),
                    execution_time=1.0,
                    row_count=10,
                    processing_mode="adaptive",
                    confidence_score=0.8,
                    query_complexity="medium",
                    query_type="aggregation",
                    tags=["test"],
                    created_at=datetime.now()
                )
                self.feedback_storage.store_feedback(feedback_record)
            
            # Test similar feedback retrieval
            similar_feedback = self.feedback_storage.get_similar_feedback(
                "Show me the average energy shortage by state", limit=3
            )
            
            assert len(similar_feedback) > 0, "Should retrieve similar feedback"
            
            logger.info("✅ Similar feedback tests passed")
            return {"success": True, "message": "Similar feedback retrieval working correctly"}
            
        except Exception as e:
            logger.error(f"❌ Similar feedback test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_performance_analytics(self) -> Dict[str, Any]:
        """Test performance analytics"""
        logger.info("Testing performance analytics...")
        
        try:
            # Get performance analytics
            analytics = self.feedback_storage.get_performance_analytics(days=30)
            
            # Check that analytics structure is correct
            required_keys = ["total_queries", "successful_queries", "failed_queries", "success_rate"]
            for key in required_keys:
                assert key in analytics, f"Analytics should contain {key}"
            
            logger.info("✅ Performance analytics tests passed")
            return {"success": True, "message": "Performance analytics working correctly"}
            
        except Exception as e:
            logger.error(f"❌ Performance analytics test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_learning_insights(self) -> Dict[str, Any]:
        """Test learning insights generation"""
        logger.info("Testing learning insights...")
        
        try:
            # Get learning insights
            insights = self.feedback_storage.get_learning_insights()
            
            # Check that insights structure is correct
            required_keys = ["error_patterns", "complexity_distribution", "mode_effectiveness"]
            for key in required_keys:
                assert key in insights, f"Insights should contain {key}"
            
            logger.info("✅ Learning insights tests passed")
            return {"success": True, "message": "Learning insights working correctly"}
            
        except Exception as e:
            logger.error(f"❌ Learning insights test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_integration(self) -> Dict[str, Any]:
        """Test integration with enhanced RAG service"""
        logger.info("Testing integration...")
        
        try:
            # Test feedback analytics endpoint
            analytics = await self.enhanced_rag.get_feedback_analytics(days=7)
            assert analytics["success"], "Feedback analytics should succeed"
            
            # Test similar feedback endpoint
            similar_feedback = await self.enhanced_rag.get_similar_feedback(
                "test query", limit=3
            )
            assert similar_feedback["success"], "Similar feedback should succeed"
            
            logger.info("✅ Integration tests passed")
            return {"success": True, "message": "Integration working correctly"}
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            return {"success": False, "error": str(e)}
            
    def cleanup(self):
        """Clean up test resources"""
        try:
            self.temp_db.close()
            import os
            os.unlink(self.db_path)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


async def main():
    """Main test runner"""
    tester = FeedbackSystemTester()
    
    try:
        results = await tester.run_all_tests()
        
        if results["overall_success"]:
            logger.info("🎉 All feedback system tests passed!")
        else:
            logger.warning("⚠️ Some feedback system tests failed")
            
        # Print detailed results
        for test_name, result in results["test_results"].items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            logger.info(f"{status} {test_name}: {result.get('message', result.get('error', 'No details'))}")
            
    finally:
        tester.cleanup()
        
    return results


if __name__ == "__main__":
    asyncio.run(main())
