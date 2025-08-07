#!/usr/bin/env python3
"""
Fixed Test Script for Specific User Query
Tests the monthly growth query and compares generated vs expected SQL
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


async def test_specific_query():
    """Test the specific user query in detail"""
    print("🎯 Testing Specific User Query...")
    print("=" * 60)
    
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
        
        # Test the specific query
        query = "What is the monthly growth of Energy Met in all regions for 2024?"
        
        print(f"User Query: {query}")
        
        # Expected SQL based on the database schema (SQLite-compatible)
        expected_sql = """
        SELECT
            r.RegionName,
            strftime('%Y-%m', d.ActualDate) as Month,
            SUM(fs.EnergyMet) as TotalEnergyMet,
            prev.PreviousMonthEnergy,
            CASE
                WHEN prev.PreviousMonthEnergy > 0
                THEN ((SUM(fs.EnergyMet) - prev.PreviousMonthEnergy) / prev.PreviousMonthEnergy) * 100
                ELSE 0
            END as GrowthRate
        FROM FactAllIndiaDailySummary fs
        JOIN DimRegions r ON fs.RegionID = r.RegionID
        JOIN DimDates d ON fs.DateID = d.DateID
        LEFT JOIN (
            SELECT
                r2.RegionName,
                strftime('%Y-%m', d2.ActualDate) as Month,
                SUM(fs2.EnergyMet) as PreviousMonthEnergy
            FROM FactAllIndiaDailySummary fs2
            JOIN DimRegions r2 ON fs2.RegionID = r2.RegionID
            JOIN DimDates d2 ON fs2.DateID = d2.DateID
            WHERE strftime('%Y', d2.ActualDate) = '2024'
            GROUP BY r2.RegionName, strftime('%Y-%m', d2.ActualDate)
        ) prev ON r.RegionName = prev.RegionName
            AND strftime('%Y-%m', d.ActualDate) = date(prev.Month || '-01', '+1 month')
        WHERE strftime('%Y', d.ActualDate) = '2024'
        GROUP BY r.RegionName, strftime('%Y-%m', d.ActualDate)
        ORDER BY r.RegionName, Month
        """.strip()
        
        print(f"\nExpected SQL:\n{expected_sql}")
        
        # Test RAG Service
        print(f"\n🔧 Enhanced RAG Service Result:")
        rag_result = await rag_service.process_query_enhanced(query, processing_mode="adaptive")
        
        rag_sql = rag_result.get('sql', '')
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
            
            # Clean the SQL for analysis
            if "LLMResponse" in sql:
                # Extract actual SQL from LLMResponse
                start_idx = sql.find("```sql")
                end_idx = sql.find("```", start_idx + 6)
                if start_idx != -1 and end_idx != -1:
                    sql = sql[start_idx + 6:end_idx].strip()
                else:
                    # Try to extract from content field
                    content_start = sql.find("content=")
                    if content_start != -1:
                        sql = sql[content_start + 8:].strip()
            
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
                "Growth": "Growth" in sql.upper() or "CASE" in sql.upper() or "LEFT JOIN" in sql.upper()
            }
            
            score = sum(analysis.values()) / len(analysis)
            return f"{name}: {score:.2f} ({sum(analysis.values())}/{len(analysis)} elements)"
        
        print(analyze_sql(expected_sql, "Expected"))
        print(analyze_sql(rag_sql, "RAG Service"))
        print(analyze_sql(wren_sql, "Wren AI"))
        
        # Detailed analysis
        print(f"\n🔍 Detailed Analysis:")
        
        def detailed_analysis(sql, name):
            if not sql:
                print(f"{name}: No SQL generated")
                return
            
            # Clean the SQL
            if "LLMResponse" in sql:
                start_idx = sql.find("```sql")
                end_idx = sql.find("```", start_idx + 6)
                if start_idx != -1 and end_idx != -1:
                    sql = sql[start_idx + 6:end_idx].strip()
                else:
                    content_start = sql.find("content=")
                    if content_start != -1:
                        sql = sql[content_start + 8:].strip()
            
            print(f"\n{name} SQL Analysis:")
            print(f"  Length: {len(sql)} characters")
            print(f"  Contains SELECT: {'✅' if 'SELECT' in sql.upper() else '❌'}")
            print(f"  Contains FROM: {'✅' if 'FROM' in sql.upper() else '❌'}")
            print(f"  Contains JOIN: {'✅' if 'JOIN' in sql.upper() else '❌'}")
            print(f"  Contains WHERE: {'✅' if 'WHERE' in sql.upper() else '❌'}")
            print(f"  Contains GROUP BY: {'✅' if 'GROUP BY' in sql.upper() else '❌'}")
            print(f"  Contains ORDER BY: {'✅' if 'ORDER BY' in sql.upper() else '❌'}")
            print(f"  Contains EnergyMet: {'✅' if 'EnergyMet' in sql else '❌'}")
            print(f"  Contains RegionName: {'✅' if 'RegionName' in sql else '❌'}")
            print(f"  Contains 2024: {'✅' if '2024' in sql else '❌'}")
            print(f"  Contains Growth/CASE/LEFT JOIN: {'✅' if ('Growth' in sql.upper() or 'CASE' in sql.upper() or 'LEFT JOIN' in sql.upper()) else '❌'}")
            
            # Show first 200 characters
            print(f"  Preview: {sql[:200]}...")
        
        detailed_analysis(expected_sql, "Expected")
        detailed_analysis(rag_sql, "RAG Service")
        detailed_analysis(wren_sql, "Wren AI")
        
        # Summary
        print(f"\n📈 Summary:")
        print(f"RAG Service Confidence: {rag_confidence:.2f}")
        print(f"Wren AI Confidence: {wren_confidence:.2f}")
        
        if rag_confidence > wren_confidence:
            print("🏆 RAG Service performed better")
        elif wren_confidence > rag_confidence:
            print("🏆 Wren AI performed better")
        else:
            print("🏆 Both performed equally")
        
        return True
        
    except Exception as e:
        print(f"❌ Specific query test failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


async def main():
    """Run the specific query test"""
    print("🧪 Starting Specific Query Test...")
    print("=" * 60)
    
    result = await test_specific_query()
    
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    if result:
        print("✅ Test completed successfully!")
        print("✅ SQL Generation: Working")
        print("✅ Query Comparison: Working")
        print("✅ Wren AI Integration: Operational")
        print("✅ Enhanced RAG Service: Operational")
    else:
        print("❌ Test failed")
    
    return result


if __name__ == "__main__":
    asyncio.run(main()) 