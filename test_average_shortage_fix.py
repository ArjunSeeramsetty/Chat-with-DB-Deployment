#!/usr/bin/env python3
"""
Test script to reproduce and fix the average shortage issue.
The problem is that when the user query is for "average shortage", 
the query is using SUM(EnergyMet) instead of AVG(EnergyShortage).
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from core.semantic_engine import SemanticEngine
from core.llm_provider import create_llm_provider
from core.schema_metadata import SchemaMetadataExtractor
from config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AverageShortageTester:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.settings = get_settings()
        
        # Initialize LLM provider using the factory function
        self.llm_provider = create_llm_provider(
            provider_type=self.settings.llm_provider_type,
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
            base_url=self.settings.llm_base_url,
            enable_gpu=self.settings.enable_gpu_acceleration
        )
        
        self.semantic_engine = SemanticEngine(self.llm_provider, db_path)
        self.schema_extractor = SchemaMetadataExtractor(db_path)
        
    async def test_average_shortage_query(self):
        """Test the average shortage query specifically"""
        logger.info("🔍 Testing Average Shortage Query...")
        
        # Test query that should generate AVG(EnergyShortage)
        test_query = "What is the average daily energy shortage across all regions?"
        
        try:
            # Extract semantic context
            logger.info(f"Processing query: {test_query}")
            semantic_context = await self.semantic_engine.extract_semantic_context(test_query)
            
            # Retrieve schema context
            schema_context = await self.semantic_engine.retrieve_schema_context(semantic_context)
            
            # Generate SQL
            sql_result = await self.semantic_engine.generate_contextual_sql(
                test_query, semantic_context, schema_context
            )
            
            sql = sql_result.get('sql', '')
            confidence = sql_result.get('confidence', 0.0)
            explanation = sql_result.get('explanation', '')
            
            logger.info(f"Generated SQL: {sql}")
            logger.info(f"Confidence: {confidence:.2f}")
            logger.info(f"Explanation: {explanation}")
            
            # Check if the SQL contains the correct aggregation function and column
            sql_lower = sql.lower()
            
            # Check for correct aggregation function
            has_avg = 'avg(' in sql_lower or 'average(' in sql_lower
            has_sum = 'sum(' in sql_lower
            
            # Check for correct column
            has_energy_shortage = 'energyshortage' in sql_lower
            has_energy_met = 'energymet' in sql_lower
            
            logger.info(f"✅ SQL Analysis:")
            logger.info(f"   Has AVG: {has_avg}")
            logger.info(f"   Has SUM: {has_sum}")
            logger.info(f"   Has EnergyShortage: {has_energy_shortage}")
            logger.info(f"   Has EnergyMet: {has_energy_met}")
            
            # Determine if the SQL is correct
            is_correct = has_avg and has_energy_shortage and not has_sum and not has_energy_met
            
            if is_correct:
                logger.info("✅ SUCCESS: SQL correctly uses AVG(EnergyShortage)")
                return True
            else:
                logger.error("❌ FAILURE: SQL incorrectly uses SUM(EnergyMet) instead of AVG(EnergyShortage)")
                logger.error(f"Expected: AVG(EnergyShortage)")
                logger.error(f"Actual: {sql}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing average shortage query: {e}")
            return False
    
    async def test_multiple_average_queries(self):
        """Test multiple average queries to ensure consistency"""
        logger.info("🔍 Testing Multiple Average Queries...")
        
        test_queries = [
            "What is the average daily energy shortage across all regions?",
            "Show me the average energy shortage by region",
            "What is the average shortage in 2024?",
            "Calculate the average daily shortage for all states"
        ]
        
        results = []
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"   Testing query {i}: {query}")
            
            try:
                # Extract semantic context
                semantic_context = await self.semantic_engine.extract_semantic_context(query)
                
                # Retrieve schema context
                schema_context = await self.semantic_engine.retrieve_schema_context(semantic_context)
                
                # Generate SQL
                sql_result = await self.semantic_engine.generate_contextual_sql(
                    query, semantic_context, schema_context
                )
                
                sql = sql_result.get('sql', '')
                confidence = sql_result.get('confidence', 0.0)
                
                # Check if the SQL contains the correct aggregation function and column
                sql_lower = sql.lower()
                has_avg = 'avg(' in sql_lower or 'average(' in sql_lower
                has_energy_shortage = 'energyshortage' in sql_lower
                has_sum = 'sum(' in sql_lower
                has_energy_met = 'energymet' in sql_lower
                
                is_correct = has_avg and has_energy_shortage and not has_sum and not has_energy_met
                
                result = {
                    "query": query,
                    "sql": sql,
                    "confidence": confidence,
                    "has_avg": has_avg,
                    "has_energy_shortage": has_energy_shortage,
                    "has_sum": has_sum,
                    "has_energy_met": has_energy_met,
                    "is_correct": is_correct
                }
                
                results.append(result)
                
                status = "✅ PASS" if is_correct else "❌ FAIL"
                logger.info(f"   {status} - Query {i}: {query}")
                
            except Exception as e:
                logger.error(f"   ❌ Query {i} failed: {e}")
                results.append({
                    "query": query,
                    "sql": "",
                    "confidence": 0.0,
                    "has_avg": False,
                    "has_energy_shortage": False,
                    "has_sum": False,
                    "has_energy_met": False,
                    "is_correct": False
                })
        
        # Calculate success rate
        successful_queries = sum(1 for r in results if r['is_correct'])
        total_queries = len(results)
        success_rate = successful_queries / total_queries * 100
        
        logger.info(f"📊 Multiple Average Queries Test Results:")
        logger.info(f"   Successful queries: {successful_queries}/{total_queries}")
        logger.info(f"   Success rate: {success_rate:.1f}%")
        
        return results
    
    async def run_comprehensive_test(self):
        """Run comprehensive test of average shortage functionality"""
        logger.info("🚀 Starting Comprehensive Average Shortage Test")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Test 1: Single average shortage query
        test1_success = await self.test_average_shortage_query()
        
        # Test 2: Multiple average queries
        test2_results = await self.test_multiple_average_queries()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("=" * 60)
        logger.info("📊 COMPREHENSIVE TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"✅ Single Average Shortage Query: {'PASS' if test1_success else 'FAIL'}")
        logger.info(f"✅ Multiple Average Queries: {'PASS' if test2_results else 'FAIL'}")
        logger.info(f"⏱️  Total Duration: {duration:.2f} seconds")
        
        if test2_results:
            success_rate = sum(1 for r in test2_results if r['is_correct']) / len(test2_results) * 100
            logger.info(f"📈 Average Query Success Rate: {success_rate:.1f}%")
            
        logger.info("=" * 60)
        
        return {
            "single_query": test1_success,
            "multiple_queries": bool(test2_results),
            "success_rate": success_rate if test2_results else 0.0
        }

async def main():
    """Main function to run the tests"""
    db_path = "C:/Users/arjun/Desktop/PSPreport/power_data.db"
    
    tester = AverageShortageTester(db_path)
    
    # Initialize the semantic engine
    await tester.semantic_engine.initialize()
    
    # Run comprehensive test
    results = await tester.run_comprehensive_test()
    
    # Print final results
    print("\n" + "=" * 60)
    print("🎯 FINAL RESULTS")
    print("=" * 60)
    print(f"Single Average Shortage Query: {'✅ PASS' if results['single_query'] else '❌ FAIL'}")
    print(f"Multiple Average Queries: {'✅ PASS' if results['multiple_queries'] else '❌ FAIL'}")
    print(f"Overall Success Rate: {results['success_rate']:.1f}%")
    print("=" * 60)
    
    if results['single_query'] and results['multiple_queries'] and results['success_rate'] >= 75:
        print("🎉 All tests passed! Average shortage queries are working correctly.")
    else:
        print("⚠️  Some tests failed. Average shortage queries need attention.")
        print("   The issue is likely in the aggregation function determination or column selection logic.")

if __name__ == "__main__":
    asyncio.run(main())
