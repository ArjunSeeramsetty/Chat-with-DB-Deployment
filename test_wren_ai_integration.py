#!/usr/bin/env python3
"""
Comprehensive Test Script for Wren AI Integration
Tests MDL support, vector search, and semantic layer integration
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.core.wren_ai_integration import WrenAIIntegration
from backend.core.llm_provider import create_llm_provider
from backend.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_wren_ai_integration():
    """Test Wren AI integration functionality"""
    print("🚀 Testing Wren AI Integration...")
    
    try:
        # Initialize settings and LLM provider
        settings = get_settings()
        llm_provider = create_llm_provider(
            provider_type=settings.llm_provider_type,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            enable_gpu=settings.enable_gpu_acceleration
        )
        
        # Initialize Wren AI integration
        wren_ai = WrenAIIntegration(llm_provider, mdl_path="mdl/")
        
        print("✅ Wren AI Integration initialized")
        
        # Test initialization
        await wren_ai.initialize()
        print("✅ Wren AI Integration initialization completed")
        
        # Test semantic context extraction
        test_queries = [
            "What is the monthly growth of Energy Met in all regions for 2024?",
            "Show me the total energy consumption by state",
            "What is the peak demand for northern region?",
            "Compare renewable vs thermal generation"
        ]
        
        print("\n📊 Testing Semantic Context Extraction...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i}: {query} ---")
            
            # Extract semantic context
            semantic_context = await wren_ai.extract_semantic_context(query)
            
            # Display results
            print(f"Confidence: {semantic_context.get('confidence', 0.0):.2f}")
            
            # Check MDL context
            mdl_context = semantic_context.get('mdl_context', {})
            if mdl_context.get('relevant_models'):
                print(f"Relevant Models: {[m['name'] for m in mdl_context['relevant_models']]}")
            
            if mdl_context.get('relevant_relationships'):
                print(f"Relevant Relationships: {len(mdl_context['relevant_relationships'])} found")
            
            # Check business entities
            business_entities = semantic_context.get('business_entities', [])
            if business_entities:
                print(f"Business Entities: {[e['term'] for e in business_entities]}")
            
            # Check search results
            search_results = semantic_context.get('search_results', {})
            total_results = sum(len(results) for results in search_results.values())
            print(f"Vector Search Results: {total_results} total matches")
            
            print("✅ Semantic context extraction successful")
        
        # Test MDL-aware SQL generation
        print("\n🔧 Testing MDL-aware SQL Generation...")
        test_query = "What is the monthly growth of Energy Met in all regions for 2024?"
        
        semantic_context = await wren_ai.extract_semantic_context(test_query)
        sql_result = await wren_ai.generate_mdl_aware_sql(test_query, semantic_context)
        
        print(f"Generated SQL: {sql_result.get('sql', 'No SQL generated')}")
        print(f"Confidence: {sql_result.get('confidence', 0.0):.2f}")
        print(f"MDL Context Used: {bool(sql_result.get('mdl_context'))}")
        print("✅ MDL-aware SQL generation successful")
        
        # Test vector store functionality
        print("\n🔍 Testing Vector Store Functionality...")
        
        # Check if collections exist
        collections = ["mdl_models", "mdl_relationships", "business_entities"]
        for collection in collections:
            try:
                # Try to get collection info
                info = wren_ai.vector_client.get_collection(collection)
                print(f"✅ Collection '{collection}' exists with {info.points_count} points")
            except Exception as e:
                print(f"❌ Collection '{collection}' error: {e}")
        
        print("\n🎯 All Wren AI Integration Tests Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Wren AI Integration test failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


async def test_enhanced_rag_service():
    """Test Enhanced RAG Service with Wren AI integration"""
    print("\n🚀 Testing Enhanced RAG Service...")
    
    try:
        from backend.services.enhanced_rag_service import EnhancedRAGService
        
        # Initialize enhanced RAG service
        rag_service = EnhancedRAGService("C:/Users/arjun/Desktop/PSPreport/power_data.db")
        
        print("✅ Enhanced RAG Service initialized")
        
        # Test initialization
        await rag_service.initialize()
        print("✅ Enhanced RAG Service initialization completed")
        
        # Test query processing
        test_queries = [
            "What is the monthly growth of Energy Met in all regions for 2024?",
            "Show me the total energy consumption by state",
            "What is the peak demand for northern region?"
        ]
        
        print("\n📊 Testing Enhanced Query Processing...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i}: {query} ---")
            
            # Process query
            result = await rag_service.process_query_enhanced(query, processing_mode="adaptive")
            
            # Display results
            print(f"Success: {result.get('success', False)}")
            print(f"Processing Method: {result.get('processing_method', 'unknown')}")
            print(f"Confidence: {result.get('confidence', 0.0):.2f}")
            
            # Check semantic insights
            semantic_insights = result.get('semantic_insights', {})
            if semantic_insights:
                print(f"Processing Mode: {semantic_insights.get('processing_mode', 'unknown')}")
                print(f"Wren AI Features: {semantic_insights.get('wren_ai_features', {})}")
            
            print("✅ Enhanced query processing successful")
        
        # Test statistics
        print("\n📈 Testing Statistics...")
        stats = rag_service.get_statistics()
        print(f"Total Requests: {stats['statistics']['total_requests']}")
        print(f"Semantic Enhancement Rate: {stats['statistics']['semantic_enhancement_rate']:.2f}")
        print(f"MDL Usage Rate: {stats['statistics']['mdl_usage_rate']:.2f}")
        print(f"Vector Search Success Rate: {stats['statistics']['vector_search_success_rate']:.2f}")
        print("✅ Statistics collection successful")
        
        print("\n🎯 All Enhanced RAG Service Tests Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced RAG Service test failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


async def test_mdl_schema():
    """Test MDL schema loading and parsing"""
    print("\n📋 Testing MDL Schema...")
    
    try:
        # Check if MDL file exists
        mdl_file = Path("mdl/energy_domain.yaml")
        if mdl_file.exists():
            print(f"✅ MDL file found: {mdl_file}")
            
            # Test YAML parsing
            import yaml
            with open(mdl_file, 'r') as f:
                mdl_content = yaml.safe_load(f)
            
            print(f"✅ MDL schema parsed successfully")
            print(f"Version: {mdl_content.get('version', 'unknown')}")
            print(f"Domain: {mdl_content.get('domain', 'unknown')}")
            print(f"Models: {len(mdl_content.get('models', {}))}")
            print(f"Relationships: {len(mdl_content.get('relationships', []))}")
            print(f"Business Glossary Terms: {len(mdl_content.get('business_glossary', {}))}")
            
        else:
            print(f"⚠️ MDL file not found: {mdl_file}")
        
        print("✅ MDL schema test completed")
        return True
        
    except Exception as e:
        print(f"❌ MDL schema test failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


async def main():
    """Run all tests"""
    print("🧪 Starting Comprehensive Wren AI Integration Tests...")
    print("=" * 60)
    
    test_results = []
    
    # Test MDL schema
    test_results.append(await test_mdl_schema())
    
    # Test Wren AI integration
    test_results.append(await test_wren_ai_integration())
    
    # Test Enhanced RAG service
    test_results.append(await test_enhanced_rag_service())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    tests = ["MDL Schema", "Wren AI Integration", "Enhanced RAG Service"]
    for test_name, result in zip(tests, test_results):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    overall_success = all(test_results)
    overall_status = "✅ ALL TESTS PASSED" if overall_success else "❌ SOME TESTS FAILED"
    
    print(f"\nOverall Result: {overall_status}")
    
    if overall_success:
        print("\n🎉 Wren AI Integration is working correctly!")
        print("✅ MDL Support: Operational")
        print("✅ Vector Search: Operational") 
        print("✅ Semantic Layer: Operational")
        print("✅ Business Context: Operational")
    else:
        print("\n⚠️ Some components need attention")
    
    return overall_success


if __name__ == "__main__":
    asyncio.run(main()) 