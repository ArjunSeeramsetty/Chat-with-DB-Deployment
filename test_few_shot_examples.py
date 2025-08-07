#!/usr/bin/env python3
"""
Test Script for Few-Shot Example Retrieval
Validates the new few-shot example retrieval functionality and its impact on SQL generation accuracy
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.core.few_shot_examples import FewShotExampleRepository, FewShotExampleRetriever, QueryExample
from backend.core.semantic_engine import SemanticEngine
from backend.core.llm_provider import create_llm_provider
from backend.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FewShotExampleTester:
    """Test few-shot example retrieval functionality"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.settings = get_settings()
        
        # Initialize LLM provider
        self.llm_provider = create_llm_provider(
            provider_type=self.settings.llm_provider_type,
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
            base_url=self.settings.llm_base_url,
            enable_gpu=self.settings.enable_gpu_acceleration
        )
        
        # Initialize few-shot example repository and retriever
        self.repository = FewShotExampleRepository(db_path)
        self.retriever = FewShotExampleRetriever(self.repository)
        
        # Initialize semantic engine with few-shot examples
        self.semantic_engine = SemanticEngine(self.llm_provider, db_path)
        
    async def test_example_repository(self):
        """Test example repository functionality"""
        logger.info("🔍 Testing Example Repository...")
        
        try:
            # Create test examples
            test_examples = [
                QueryExample(
                    id=0,
                    natural_query="What is the monthly growth of Energy Met of all regions in 2024?",
                    generated_sql="""
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
                    """,
                    confidence=0.85,
                    success=True,
                    execution_time=0.5,
                    tags=["growth", "monthly", "energy", "regions"],
                    complexity="complex",
                    domain="energy",
                    query_type="trend_analysis"
                ),
                QueryExample(
                    id=0,
                    natural_query="Show me the total energy consumption by state for 2024",
                    generated_sql="""
                    SELECT 
                        s.StateName,
                        strftime('%Y-%m', d.ActualDate) as Month,
                        SUM(fs.EnergyMet) as TotalEnergyMet
                    FROM FactAllIndiaDailySummary fs
                    JOIN DimStates s ON fs.StateID = s.StateID
                    JOIN DimDates d ON fs.DateID = d.DateID
                    WHERE strftime('%Y', d.ActualDate) = '2024'
                    GROUP BY s.StateName, strftime('%Y-%m', d.ActualDate)
                    ORDER BY s.StateName, Month
                    """,
                    confidence=0.78,
                    success=True,
                    execution_time=0.3,
                    tags=["consumption", "state", "energy"],
                    complexity="medium",
                    domain="energy",
                    query_type="aggregation"
                ),
                QueryExample(
                    id=0,
                    natural_query="Compare energy generation between Eastern and Western regions",
                    generated_sql="""
                    SELECT 
                        r.RegionName,
                        strftime('%Y-%m', d.ActualDate) as Month,
                        SUM(fs.EnergyMet) as TotalEnergyMet
                    FROM FactAllIndiaDailySummary fs
                    JOIN DimRegions r ON fs.RegionID = r.RegionID
                    JOIN DimDates d ON fs.DateID = d.DateID
                    WHERE r.RegionName IN ('Eastern', 'Western')
                        AND strftime('%Y', d.ActualDate) = '2024'
                    GROUP BY r.RegionName, strftime('%Y-%m', d.ActualDate)
                    ORDER BY r.RegionName, Month
                    """,
                    confidence=0.82,
                    success=True,
                    execution_time=0.4,
                    tags=["comparison", "regions", "energy"],
                    complexity="medium",
                    domain="energy",
                    query_type="comparison"
                )
            ]
            
            # Add examples to repository
            example_ids = []
            for example in test_examples:
                example_id = self.repository.add_example(example)
                example_ids.append(example_id)
                logger.info(f"✅ Added example {example_id}: {example.natural_query[:50]}...")
            
            # Test retrieval by similarity
            test_query = "What is the monthly growth of Energy Met of all regions in 2024?"
            similar_examples = self.repository.search_similar_examples(
                test_query, 
                limit=3,
                min_confidence=0.7,
                only_successful=True
            )
            
            logger.info(f"✅ Retrieved {len(similar_examples)} similar examples")
            for example, similarity in similar_examples:
                logger.info(f"   Example {example.id}: {example.natural_query[:50]}... (similarity: {similarity:.3f})")
            
            # Test statistics
            stats = self.repository.get_statistics()
            logger.info(f"✅ Repository statistics:")
            logger.info(f"   Total examples: {stats['total_examples']}")
            logger.info(f"   Successful examples: {stats['successful_examples']}")
            logger.info(f"   Success rate: {stats['success_rate']:.1f}%")
            logger.info(f"   Average confidence: {stats['average_confidence']:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Example repository test failed: {e}")
            return False
            
    async def test_example_retrieval(self):
        """Test example retrieval functionality"""
        logger.info("🔍 Testing Example Retrieval...")
        
        try:
            # Test retrieval for different query types
            test_queries = [
                "What is the monthly growth of Energy Met of all regions in 2024?",
                "Show me the total energy consumption by state for 2024",
                "Compare energy generation between Eastern and Western regions"
            ]
            
            for query in test_queries:
                examples = self.retriever.retrieve_examples_for_query(
                    query, 
                    max_examples=2,
                    min_similarity=0.3
                )
                
                logger.info(f"✅ Retrieved {len(examples)} examples for: {query[:50]}...")
                for example in examples:
                    logger.info(f"   - {example.natural_query[:50]}... (confidence: {example.confidence:.2f})")
            
            # Test formatting for prompts
            examples = self.retriever.retrieve_examples_for_query(
                "What is the monthly growth of Energy Met of all regions in 2024?",
                max_examples=2
            )
            
            formatted_examples = self.retriever.format_examples_for_prompt(examples)
            logger.info(f"✅ Formatted examples for prompt:")
            logger.info(f"   Length: {len(formatted_examples)} characters")
            logger.info(f"   Examples: {len(examples)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Example retrieval test failed: {e}")
            return False
            
    async def test_semantic_engine_integration(self):
        """Test semantic engine integration with few-shot examples"""
        logger.info("🔍 Testing Semantic Engine Integration...")
        
        try:
            # Initialize semantic engine
            await self.semantic_engine.initialize()
            
            # Test query with few-shot examples
            test_query = "What is the monthly growth of Energy Met of all regions in 2024?"
            
            # Extract semantic context
            semantic_context = await self.semantic_engine.extract_semantic_context(test_query)
            
            logger.info(f"✅ Semantic context extracted successfully")
            logger.info(f"   Intent: {semantic_context.intent}")
            logger.info(f"   Confidence: {semantic_context.confidence:.2f}")
            logger.info(f"   Business entities: {len(semantic_context.business_entities)}")
            
            # Retrieve schema context
            schema_context = await self.semantic_engine.retrieve_schema_context(semantic_context)
            
            logger.info(f"✅ Schema context retrieved successfully")
            
            # Generate SQL with few-shot examples
            sql_result = await self.semantic_engine.generate_contextual_sql(
                test_query, semantic_context, schema_context
            )
            
            logger.info(f"✅ SQL generated with few-shot examples")
            logger.info(f"   SQL: {sql_result.get('sql', 'None')[:200]}...")
            logger.info(f"   Confidence: {sql_result.get('confidence', 0.0):.2f}")
            logger.info(f"   Explanation: {sql_result.get('explanation', 'None')}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Semantic engine integration failed: {e}")
            return False
            
    async def test_sql_accuracy_improvement(self):
        """Test SQL accuracy improvement with few-shot examples"""
        logger.info("🔍 Testing SQL Accuracy Improvement...")
        
        test_queries = [
            "What is the monthly growth of Energy Met of all regions in 2024?",
            "Show me the total energy consumption by state for 2024",
            "Compare energy generation between Eastern and Western regions",
            "What is the average daily energy shortage across all regions?"
        ]
        
        results = []
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"   Testing query {i}: {query}")
            
            try:
                # Extract semantic context
                semantic_context = await self.semantic_engine.extract_semantic_context(query)
                
                # Retrieve schema context
                schema_context = await self.semantic_engine.retrieve_schema_context(semantic_context)
                
                # Generate SQL with few-shot examples
                sql_result = await self.semantic_engine.generate_contextual_sql(
                    query, semantic_context, schema_context
                )
                
                result = {
                    "query": query,
                    "sql": sql_result.get('sql', ''),
                    "confidence": sql_result.get('confidence', 0.0),
                    "explanation": sql_result.get('explanation', ''),
                    "success": bool(sql_result.get('sql', ''))
                }
                
                results.append(result)
                
                logger.info(f"   ✅ Query {i} completed - Confidence: {result['confidence']:.2f}")
                
            except Exception as e:
                logger.error(f"   ❌ Query {i} failed: {e}")
                results.append({
                    "query": query,
                    "sql": "",
                    "confidence": 0.0,
                    "explanation": f"Error: {e}",
                    "success": False
                })
                
        # Calculate accuracy metrics
        successful_queries = sum(1 for r in results if r['success'])
        total_queries = len(results)
        average_confidence = sum(r['confidence'] for r in results) / total_queries
        
        logger.info(f"✅ SQL Accuracy Test Results:")
        logger.info(f"   Successful queries: {successful_queries}/{total_queries}")
        logger.info(f"   Success rate: {successful_queries/total_queries*100:.1f}%")
        logger.info(f"   Average confidence: {average_confidence:.2f}")
        
        return results
        
    async def run_comprehensive_test(self):
        """Run comprehensive test of few-shot example retrieval"""
        logger.info("🚀 Starting Comprehensive Few-Shot Example Retrieval Test")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Test 1: Example repository
        test1_success = await self.test_example_repository()
        
        # Test 2: Example retrieval
        test2_success = await self.test_example_retrieval()
        
        # Test 3: Semantic engine integration
        test3_success = await self.test_semantic_engine_integration()
        
        # Test 4: SQL accuracy improvement
        test4_results = await self.test_sql_accuracy_improvement()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("=" * 60)
        logger.info("📊 COMPREHENSIVE TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"✅ Example Repository: {'PASS' if test1_success else 'FAIL'}")
        logger.info(f"✅ Example Retrieval: {'PASS' if test2_success else 'FAIL'}")
        logger.info(f"✅ Semantic Engine Integration: {'PASS' if test3_success else 'FAIL'}")
        logger.info(f"✅ SQL Accuracy Improvement: {'PASS' if test4_results else 'FAIL'}")
        logger.info(f"⏱️  Total Duration: {duration:.2f} seconds")
        
        if test4_results:
            success_rate = sum(1 for r in test4_results if r['success']) / len(test4_results) * 100
            avg_confidence = sum(r['confidence'] for r in test4_results) / len(test4_results)
            logger.info(f"📈 SQL Generation Success Rate: {success_rate:.1f}%")
            logger.info(f"📈 Average Confidence: {avg_confidence:.2f}")
            
        logger.info("=" * 60)
        
        return {
            "example_repository": test1_success,
            "example_retrieval": test2_success,
            "semantic_integration": test3_success,
            "sql_accuracy": bool(test4_results),
            "duration": duration,
            "sql_results": test4_results if test4_results else []
        }


async def main():
    """Main test function"""
    # Use the existing database path
    db_path = "C:/Users/arjun/Desktop/PSPreport/power_data.db"
    
    if not os.path.exists(db_path):
        logger.error(f"❌ Database not found at {db_path}")
        return
        
    logger.info(f"🔍 Using database: {db_path}")
    
    # Create tester
    tester = FewShotExampleTester(db_path)
    
    # Run comprehensive test
    results = await tester.run_comprehensive_test()
    
    # Print detailed results
    if results.get('sql_results'):
        logger.info("\n📋 DETAILED SQL RESULTS:")
        for i, result in enumerate(results['sql_results'], 1):
            logger.info(f"   Query {i}: {result['query']}")
            logger.info(f"   SQL: {result['sql'][:100]}...")
            logger.info(f"   Confidence: {result['confidence']:.2f}")
            logger.info(f"   Success: {'✅' if result['success'] else '❌'}")
            logger.info("   " + "-" * 40)
            
    logger.info("🎯 Few-Shot Example Retrieval Test Complete!")


if __name__ == "__main__":
    asyncio.run(main())
