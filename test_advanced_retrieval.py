#!/usr/bin/env python3
"""
Test Advanced Retrieval System
Validates hybrid search (dense + sparse) capabilities
"""

import asyncio
import logging
import tempfile
import time
from typing import Dict, Any

from backend.core.advanced_retrieval import AdvancedRetrieval, ContextualRetrieval, RetrievalResult
from backend.services.enhanced_rag_service import EnhancedRAGService
from backend.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedRetrievalTester:
    """Test suite for advanced retrieval system"""
    
    def __init__(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        self.advanced_retrieval = AdvancedRetrieval()
        self.contextual_retrieval = ContextualRetrieval(self.advanced_retrieval)
        self.enhanced_rag = EnhancedRAGService(self.db_path)
        
    async def run_all_tests(self):
        """Run all advanced retrieval tests"""
        logger.info("🚀 Starting Advanced Retrieval Tests")
        
        test_results = {
            "document_loading": await self.test_document_loading(),
            "dense_search": await self.test_dense_search(),
            "sparse_search": await self.test_sparse_search(),
            "hybrid_search": await self.test_hybrid_search(),
            "contextual_retrieval": await self.test_contextual_retrieval(),
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
        
    async def test_document_loading(self) -> Dict[str, Any]:
        """Test document loading functionality"""
        logger.info("Testing document loading...")
        
        try:
            # Create test documents
            test_documents = [
                {
                    "content": "Energy Met represents actual energy consumption in megawatt hours",
                    "source": "business_rules",
                    "metadata": {"type": "business_rule", "domain": "energy"}
                },
                {
                    "content": "Energy Shortage represents unmet energy demand in megawatt hours",
                    "source": "business_rules",
                    "metadata": {"type": "business_rule", "domain": "energy"}
                },
                {
                    "content": "States are grouped into regions: Northern, Southern, Eastern, Western, North Eastern, and Central",
                    "source": "business_rules",
                    "metadata": {"type": "business_rule", "domain": "geography"}
                },
                {
                    "content": "Show me the average energy shortage by state",
                    "source": "historical_query",
                    "metadata": {"type": "query", "complexity": "medium"}
                },
                {
                    "content": "What is the total energy consumption by region?",
                    "source": "historical_query",
                    "metadata": {"type": "query", "complexity": "simple"}
                }
            ]
            
            # Add documents to retrieval system
            self.advanced_retrieval.add_documents(test_documents)
            
            # Check if documents were loaded
            stats = self.advanced_retrieval.get_retrieval_stats()
            assert stats["total_documents"] == 5, f"Expected 5 documents, got {stats['total_documents']}"
            assert stats["fitted"] == True, "Retrieval system should be fitted"
            
            logger.info("✅ Document loading tests passed")
            return {"success": True, "message": "Document loading working correctly"}
            
        except Exception as e:
            logger.error(f"❌ Document loading test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_dense_search(self) -> Dict[str, Any]:
        """Test dense (semantic) search"""
        logger.info("Testing dense search...")
        
        try:
            # Test semantic search
            results = self.advanced_retrieval.semantic_search("energy consumption", top_k=3)
            
            assert len(results) > 0, "Should return results for semantic search"
            assert all(isinstance(result, RetrievalResult) for result in results), "Should return RetrievalResult objects"
            assert all(result.retrieval_type == "dense" for result in results), "Should be dense retrieval type"
            
            # Check that results are relevant
            relevant_content = any("energy" in result.content.lower() for result in results)
            assert relevant_content, "Should return relevant content"
            
            logger.info("✅ Dense search tests passed")
            return {"success": True, "message": "Dense search working correctly"}
            
        except Exception as e:
            logger.error(f"❌ Dense search test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_sparse_search(self) -> Dict[str, Any]:
        """Test sparse (keyword-based) search"""
        logger.info("Testing sparse search...")
        
        try:
            # Test keyword search
            results = self.advanced_retrieval.keyword_search("energy shortage", top_k=3)
            
            assert len(results) > 0, "Should return results for keyword search"
            assert all(isinstance(result, RetrievalResult) for result in results), "Should return RetrievalResult objects"
            assert all(result.retrieval_type == "sparse" for result in results), "Should be sparse retrieval type"
            
            # Check that results are relevant
            relevant_content = any("shortage" in result.content.lower() for result in results)
            assert relevant_content, "Should return relevant content"
            
            logger.info("✅ Sparse search tests passed")
            return {"success": True, "message": "Sparse search working correctly"}
            
        except Exception as e:
            logger.error(f"❌ Sparse search test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_hybrid_search(self) -> Dict[str, Any]:
        """Test hybrid search combining dense and sparse"""
        logger.info("Testing hybrid search...")
        
        try:
            # Test hybrid search
            results = self.advanced_retrieval.hybrid_search("energy consumption by region", top_k=3)
            
            assert len(results) > 0, "Should return results for hybrid search"
            assert all(isinstance(result, RetrievalResult) for result in results), "Should return RetrievalResult objects"
            assert all(result.retrieval_type == "hybrid" for result in results), "Should be hybrid retrieval type"
            
            # Check that results are relevant
            relevant_content = any("energy" in result.content.lower() for result in results)
            assert relevant_content, "Should return relevant content"
            
            logger.info("✅ Hybrid search tests passed")
            return {"success": True, "message": "Hybrid search working correctly"}
            
        except Exception as e:
            logger.error(f"❌ Hybrid search test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_contextual_retrieval(self) -> Dict[str, Any]:
        """Test contextual retrieval with user preferences"""
        logger.info("Testing contextual retrieval...")
        
        try:
            # Test contextual retrieval
            context = {
                "user_preferences": {
                    "preferred_complexity": "medium",
                    "preferred_visualization": "chart"
                },
                "conversation_history": [
                    {"content": "Show me energy data", "role": "user"},
                    {"content": "Here is the energy consumption data", "role": "assistant"}
                ],
                "domain_expertise": {
                    "expertise_level": "intermediate"
                }
            }
            
            results = await self.contextual_retrieval.retrieve_with_context(
                "energy consumption", context, top_k=3
            )
            
            assert len(results) > 0, "Should return results for contextual retrieval"
            assert all(isinstance(result, RetrievalResult) for result in results), "Should return RetrievalResult objects"
            
            logger.info("✅ Contextual retrieval tests passed")
            return {"success": True, "message": "Contextual retrieval working correctly"}
            
        except Exception as e:
            logger.error(f"❌ Contextual retrieval test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_integration(self) -> Dict[str, Any]:
        """Test integration with enhanced RAG service"""
        logger.info("Testing integration...")
        
        try:
            # Test advanced retrieval through enhanced RAG service
            results = await self.enhanced_rag.advanced_retrieve("energy consumption", top_k=3)
            
            assert isinstance(results, list), "Should return a list of results"
            
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
    tester = AdvancedRetrievalTester()
    
    try:
        results = await tester.run_all_tests()
        
        if results["overall_success"]:
            logger.info("🎉 All advanced retrieval tests passed!")
        else:
            logger.warning("⚠️ Some advanced retrieval tests failed")
            
        # Print detailed results
        for test_name, result in results["test_results"].items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            logger.info(f"{status} {test_name}: {result.get('message', result.get('error', 'No details'))}")
            
    finally:
        tester.cleanup()
        
    return results


if __name__ == "__main__":
    asyncio.run(main())
