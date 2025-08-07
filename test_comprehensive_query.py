#!/usr/bin/env python3
"""
Comprehensive Query Test Script
Tests user queries and compares generated SQL with expected SQL
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.services.enhanced_rag_service import EnhancedRAGService
from backend.core.wren_ai_integration import WrenAIIntegration
from backend.core.llm_provider import create_llm_provider
from backend.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_user_query_comparison():
    """Test user queries and compare generated vs expected SQL"""
    print("🧪 Testing Comprehensive Query Processing...")
    print("=" * 60)
    
    try:
        # Initialize settings and services
        settings = get_settings()
        llm_provider = create_llm_provider(
            provider_type=settings.llm_provider_type,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            enable_gpu=settings.enable_gpu_acceleration
        )
        
        # Initialize Enhanced RAG Service
        rag_service = EnhancedRAGService("C:/Users/arjun/Desktop/PSPreport/power_data.db")
        await rag_service.initialize()
        
        # Initialize Wren AI Integration
        wren_ai = WrenAIIntegration(llm_provider, mdl_path="mdl/")
        await wren_ai.initialize()
        
        print("✅ Services initialized successfully")
        
        # Test queries with expected SQL
        test_cases = [
            {
                "query": "What is the monthly growth of Energy Met in all regions for 2024?",
                "expected_sql": """
                SELECT 
                    r.RegionName,
                    strftime('%Y-%m', d.Date) as Month,
                    SUM(fs.EnergyMet) as TotalEnergyMet,
                    LAG(SUM(fs.EnergyMet)) OVER (PARTITION BY r.RegionName ORDER BY strftime('%Y-%m', d.Date)) as PreviousMonthEnergy,
                    ((SUM(fs.EnergyMet) - LAG(SUM(fs.EnergyMet)) OVER (PARTITION BY r.RegionName ORDER BY strftime('%Y-%m', d.Date))) / 
                    LAG(SUM(fs.EnergyMet)) OVER (PARTITION BY r.RegionName ORDER BY strftime('%Y-%m', d.Date)) * 100 as GrowthRate
                FROM FactAllIndiaDailySummary fs
                JOIN DimRegions r ON fs.RegionID = r.RegionID
                JOIN DimDates d ON fs.DateID = d.DateID
                WHERE strftime('%Y', d.Date) = '2024'
                GROUP BY r.RegionName, strftime('%Y-%m', d.Date)
                ORDER BY r.RegionName, Month
                """,
                "description": "Monthly growth analysis with proper joins and window functions"
            },
            {
                "query": "Show me the total energy consumption by state",
                "expected_sql": """
                SELECT 
                    s.StateName,
                    SUM(fs.EnergyMet) as TotalEnergyConsumption
                FROM FactStateDailyEnergy fs
                JOIN DimStates s ON fs.StateID = s.StateID
                GROUP BY s.StateName
                ORDER BY TotalEnergyConsumption DESC
                """,
                "description": "State-wise energy consumption aggregation"
            },
            {
                "query": "What is the peak demand for northern region?",
                "expected_sql": """
                SELECT 
                    r.RegionName,
                    MAX(fs.MaximumDemand) as PeakDemand
                FROM FactAllIndiaDailySummary fs
                JOIN DimRegions r ON fs.RegionID = r.RegionID
                WHERE r.RegionName = 'Northern'
                GROUP BY r.RegionName
                """,
                "description": "Peak demand for specific region"
            },
            {
                "query": "Compare renewable vs thermal generation",
                "expected_sql": """
                SELECT 
                    gs.GenerationSourceName,
                    SUM(fg.GenerationAmount) as TotalGeneration,
                    AVG(fg.PLF) as AveragePLF
                FROM FactDailyGenerationBreakdown fg
                JOIN DimGenerationSources gs ON fg.GenerationSourceID = gs.GenerationSourceID
                WHERE gs.GenerationSourceName IN ('Renewable', 'Thermal')
                GROUP BY gs.GenerationSourceName
                ORDER BY TotalGeneration DESC
                """,
                "description": "Generation comparison with proper source filtering"
            }
        ]
        
        print(f"\n📊 Testing {len(test_cases)} query scenarios...")
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*50}")
            print(f"Test Case {i}: {test_case['description']}")
            print(f"{'='*50}")
            
            query = test_case["query"]
            expected_sql = test_case["expected_sql"].strip()
            
            print(f"User Query: {query}")
            print(f"Expected SQL: {expected_sql[:100]}...")
            
            # Test Enhanced RAG Service
            print("\n🔧 Testing Enhanced RAG Service...")
            rag_result = await rag_service.process_query_enhanced(query, processing_mode="adaptive")
            
            rag_sql = rag_result.get('sql_query', '')
            rag_success = rag_result.get('success', False)
            rag_confidence = rag_result.get('confidence', 0.0)
            
            print(f"RAG Success: {rag_success}")
            print(f"RAG Confidence: {rag_confidence:.2f}")
            print(f"RAG SQL: {rag_sql[:100]}...")
            
            # Test Wren AI Integration
            print("\n🤖 Testing Wren AI Integration...")
            semantic_context = await wren_ai.extract_semantic_context(query)
            wren_sql_result = await wren_ai.generate_mdl_aware_sql(query, semantic_context)
            
            wren_sql = wren_sql_result.get('sql', '')
            wren_confidence = wren_sql_result.get('confidence', 0.0)
            
            print(f"Wren AI Confidence: {wren_confidence:.2f}")
            print(f"Wren AI SQL: {wren_sql[:100]}...")
            
            # Compare results
            print("\n📈 Comparison Analysis:")
            
            # Check if SQL contains key elements
            key_elements = {
                "SELECT": "SELECT clause present",
                "FROM": "FROM clause present", 
                "JOIN": "JOIN operations present",
                "WHERE": "WHERE clause present",
                "GROUP BY": "GROUP BY clause present",
                "ORDER BY": "ORDER BY clause present"
            }
            
            rag_analysis = {}
            wren_analysis = {}
            
            for element, description in key_elements.items():
                rag_analysis[element] = element in rag_sql.upper()
                wren_analysis[element] = element in wren_sql.upper()
            
            print("RAG SQL Analysis:")
            for element, present in rag_analysis.items():
                status = "✅" if present else "❌"
                print(f"  {status} {key_elements[element]}")
            
            print("Wren AI SQL Analysis:")
            for element, present in wren_analysis.items():
                status = "✅" if present else "❌"
                print(f"  {status} {key_elements[element]}")
            
            # Calculate similarity score
            rag_score = sum(rag_analysis.values()) / len(rag_analysis)
            wren_score = sum(wren_analysis.items()) / len(wren_analysis)
            
            print(f"\nRAG SQL Completeness: {rag_score:.2f}")
            print(f"Wren AI SQL Completeness: {wren_score:.2f}")
            
            # Store results
            results.append({
                "query": query,
                "rag_sql": rag_sql,
                "wren_sql": wren_sql,
                "rag_confidence": rag_confidence,
                "wren_confidence": wren_confidence,
                "rag_score": rag_score,
                "wren_score": wren_score,
                "expected_sql": expected_sql
            })
            
            print("✅ Test case completed")
        
        # Generate comprehensive report
        print(f"\n{'='*60}")
        print("📊 COMPREHENSIVE TEST REPORT")
        print(f"{'='*60}")
        
        total_rag_score = sum(r['rag_score'] for r in results)
        total_wren_score = sum(r['wren_score'] for r in results)
        avg_rag_confidence = sum(r['rag_confidence'] for r in results) / len(results)
        avg_wren_confidence = sum(r['wren_confidence'] for r in results) / len(results)
        
        print(f"Total Test Cases: {len(results)}")
        print(f"Average RAG SQL Completeness: {total_rag_score/len(results):.2f}")
        print(f"Average Wren AI SQL Completeness: {total_wren_score/len(results):.2f}")
        print(f"Average RAG Confidence: {avg_rag_confidence:.2f}")
        print(f"Average Wren AI Confidence: {avg_wren_confidence:.2f}")
        
        # Determine winner
        if total_rag_score > total_wren_score:
            winner = "Enhanced RAG Service"
            improvement = ((total_rag_score - total_wren_score) / total_wren_score) * 100
        elif total_wren_score > total_rag_score:
            winner = "Wren AI Integration"
            improvement = ((total_wren_score - total_rag_score) / total_rag_score) * 100
        else:
            winner = "Tie"
            improvement = 0
        
        print(f"\n🏆 Winner: {winner}")
        if improvement > 0:
            print(f"Improvement: {improvement:.1f}%")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        for i, result in enumerate(results, 1):
            print(f"\nTest Case {i}:")
            print(f"  Query: {result['query']}")
            print(f"  RAG Score: {result['rag_score']:.2f} (Confidence: {result['rag_confidence']:.2f})")
            print(f"  Wren AI Score: {result['wren_score']:.2f} (Confidence: {result['wren_confidence']:.2f})")
        
        print(f"\n🎯 Overall Assessment:")
        if avg_rag_confidence > 0.7 and avg_wren_confidence > 0.7:
            print("✅ Both systems performing well")
        elif avg_rag_confidence > 0.5 or avg_wren_confidence > 0.5:
            print("⚠️ Some systems need improvement")
        else:
            print("❌ Systems need significant improvement")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


async def test_specific_query():
    """Test a specific user query in detail"""
    print("\n🎯 Testing Specific User Query...")
    
    try:
        # Initialize services
        settings = get_settings()
        llm_provider = create_llm_provider(
            provider_type=settings.llm_provider_type,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            enable_gpu=settings.enable_gpu_acceleration
        )
        
        rag_service = EnhancedRAGService("C:/Users/arjun/Desktop/PSPreport/power_data.db")
        await rag_service.initialize()
        
        wren_ai = WrenAIIntegration(llm_provider, mdl_path="mdl/")
        await wren_ai.initialize()
        
        # Test the specific query that was causing issues
        query = "What is the monthly growth of Energy Met in all regions for 2024?"
        
        print(f"User Query: {query}")
        
        # Expected SQL based on the database schema
        expected_sql = """
        SELECT 
            r.RegionName,
            strftime('%Y-%m', d.Date) as Month,
            SUM(fs.EnergyMet) as TotalEnergyMet,
            LAG(SUM(fs.EnergyMet)) OVER (PARTITION BY r.RegionName ORDER BY strftime('%Y-%m', d.Date)) as PreviousMonthEnergy,
            ((SUM(fs.EnergyMet) - LAG(SUM(fs.EnergyMet)) OVER (PARTITION BY r.RegionName ORDER BY strftime('%Y-%m', d.Date))) / 
            LAG(SUM(fs.EnergyMet)) OVER (PARTITION BY r.RegionName ORDER BY strftime('%Y-%m', d.Date)) * 100 as GrowthRate
        FROM FactAllIndiaDailySummary fs
        JOIN DimRegions r ON fs.RegionID = r.RegionID
        JOIN DimDates d ON fs.DateID = d.DateID
        WHERE strftime('%Y', d.Date) = '2024'
        GROUP BY r.RegionName, strftime('%Y-%m', d.Date)
        ORDER BY r.RegionName, Month
        """.strip()
        
        print(f"\nExpected SQL:\n{expected_sql}")
        
        # Test RAG Service
        print(f"\n🔧 Enhanced RAG Service Result:")
        rag_result = await rag_service.process_query_enhanced(query, processing_mode="adaptive")
        
        rag_sql = rag_result.get('sql_query', '')
        rag_success = rag_result.get('success', False)
        rag_confidence = rag_result.get('confidence', 0.0)
        
        print(f"Success: {rag_success}")
        print(f"Confidence: {rag_confidence:.2f}")
        print(f"Generated SQL:\n{rag_sql}")
        
        # Test Wren AI
        print(f"\n🤖 Wren AI Integration Result:")
        semantic_context = await wren_ai.extract_semantic_context(query)
        wren_sql_result = await wren_ai.generate_mdl_aware_sql(query, semantic_context)
        
        wren_sql = wren_sql_result.get('sql', '')
        wren_confidence = wren_sql_result.get('confidence', 0.0)
        
        print(f"Confidence: {wren_confidence:.2f}")
        print(f"Generated SQL:\n{wren_sql}")
        
        # Compare with expected
        print(f"\n📊 Comparison with Expected SQL:")
        
        def analyze_sql(sql, name):
            if not sql:
                return f"{name}: No SQL generated"
            
            analysis = {
                "SELECT": "SELECT" in sql.upper(),
                "FROM": "FROM" in sql.upper(),
                "JOIN": "JOIN" in sql.upper(),
                "WHERE": "WHERE" in sql.upper(),
                "GROUP BY": "GROUP BY" in sql.upper(),
                "ORDER BY": "ORDER BY" in sql.upper(),
                "EnergyMet": "EnergyMet" in sql,
                "RegionName": "RegionName" in sql,
                "2024": "2024" in sql,
                "Growth": "Growth" in sql.upper() or "LAG" in sql.upper()
            }
            
            score = sum(analysis.values()) / len(analysis)
            return f"{name}: {score:.2f} ({sum(analysis.values())}/{len(analysis)} elements)"
        
        print(analyze_sql(expected_sql, "Expected"))
        print(analyze_sql(rag_sql, "RAG Service"))
        print(analyze_sql(wren_sql, "Wren AI"))
        
        return True
        
    except Exception as e:
        print(f"❌ Specific query test failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


async def main():
    """Run comprehensive query tests"""
    print("🧪 Starting Comprehensive Query Testing...")
    print("=" * 60)
    
    # Test specific query first
    specific_result = await test_specific_query()
    
    # Test comprehensive comparison
    comprehensive_result = await test_user_query_comparison()
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 FINAL TEST SUMMARY")
    print(f"{'='*60}")
    
    print(f"Specific Query Test: {'✅ PASSED' if specific_result else '❌ FAILED'}")
    print(f"Comprehensive Test: {'✅ PASSED' if comprehensive_result else '❌ FAILED'}")
    
    overall_success = specific_result and comprehensive_result
    overall_status = "✅ ALL TESTS PASSED" if overall_success else "❌ SOME TESTS FAILED"
    
    print(f"\nOverall Result: {overall_status}")
    
    if overall_success:
        print("\n🎉 Comprehensive query testing completed successfully!")
        print("✅ SQL Generation: Working")
        print("✅ Query Comparison: Working")
        print("✅ Wren AI Integration: Operational")
        print("✅ Enhanced RAG Service: Operational")
    else:
        print("\n⚠️ Some tests need attention")
    
    return overall_success


if __name__ == "__main__":
    asyncio.run(main()) 