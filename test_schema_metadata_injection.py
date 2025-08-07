#!/usr/bin/env python3
"""
Test Script for Schema Metadata Injection
Validates the new schema metadata injection functionality and its impact on SQL generation accuracy
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.core.schema_metadata import SchemaMetadataExtractor
from backend.core.semantic_engine import SemanticEngine
from backend.core.llm_provider import create_llm_provider
from backend.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SchemaMetadataTester:
    """Test schema metadata injection functionality"""
    
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
        
        # Initialize schema extractor
        self.schema_extractor = SchemaMetadataExtractor(db_path)
        
        # Initialize semantic engine with schema metadata
        self.semantic_engine = SemanticEngine(self.llm_provider, db_path)
        
    async def test_schema_metadata_extraction(self):
        """Test schema metadata extraction"""
        logger.info("🔍 Testing Schema Metadata Extraction...")
        
        try:
            # Extract schema metadata
            metadata = self.schema_extractor.get_schema_metadata()
            
            logger.info(f"✅ Schema metadata extracted successfully")
            logger.info(f"   Tables: {len(metadata['tables'])}")
            logger.info(f"   Columns: {len(metadata['columns'])}")
            logger.info(f"   Relationships: {len(metadata['relationships'])}")
            logger.info(f"   Constraints: {len(metadata['constraints'])}")
            logger.info(f"   Indexes: {len(metadata['indexes'])}")
            logger.info(f"   Sample data tables: {len(metadata['sample_data'])}")
            
            # Test schema context building
            test_query = "What is the monthly growth of Energy Met of all regions in 2024?"
            schema_context = self.schema_extractor.build_schema_prompt_context(test_query)
            
            logger.info(f"✅ Schema context built successfully")
            logger.info(f"   Context length: {len(schema_context)} characters")
            
            # Test relevant schema context
            relevant_context = self.schema_extractor.get_relevant_schema_context(test_query)
            
            logger.info(f"✅ Relevant schema context extracted")
            logger.info(f"   Relevant tables: {relevant_context['relevant_tables']}")
            logger.info(f"   Relevant columns: {len(relevant_context['relevant_columns'])}")
            logger.info(f"   Keywords: {relevant_context['query_analysis']['keywords']}")
            logger.info(f"   Entities: {relevant_context['query_analysis']['entities']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Schema metadata extraction failed: {e}")
            return False
            
    async def test_semantic_engine_integration(self):
        """Test semantic engine integration with schema metadata"""
        logger.info("🔍 Testing Semantic Engine Integration...")
        
        try:
            # Initialize semantic engine
            await self.semantic_engine.initialize()
            
            # Test query
            test_query = "What is the monthly growth of Energy Met of all regions in 2024?"
            
            # Extract semantic context
            semantic_context = await self.semantic_engine.extract_semantic_context(test_query)
            
            logger.info(f"✅ Semantic context extracted successfully")
            logger.info(f"   Intent: {semantic_context.intent}")
            logger.info(f"   Confidence: {semantic_context.confidence:.2f}")
            logger.info(f"   Business entities: {len(semantic_context.business_entities)}")
            logger.info(f"   Domain concepts: {semantic_context.domain_concepts}")
            
            # Retrieve schema context
            schema_context = await self.semantic_engine.retrieve_schema_context(semantic_context)
            
            logger.info(f"✅ Schema context retrieved successfully")
            primary_table = schema_context.get('primary_table')
            if primary_table:
                logger.info(f"   Primary table: {primary_table.get('name', 'Unknown')}")
            else:
                logger.info(f"   Primary table: None (no relevant tables found)")
            logger.info(f"   Available tables: {len(schema_context.get('available_tables', []))}")
            
            # Generate SQL with schema metadata injection
            sql_result = await self.semantic_engine.generate_contextual_sql(
                test_query, semantic_context, schema_context
            )
            
            logger.info(f"✅ SQL generated with schema metadata injection")
            logger.info(f"   SQL: {sql_result.get('sql', 'None')[:200]}...")
            logger.info(f"   Confidence: {sql_result.get('confidence', 0.0):.2f}")
            logger.info(f"   Explanation: {sql_result.get('explanation', 'None')}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Semantic engine integration failed: {e}")
            return False
            
    async def test_sql_accuracy_improvement(self):
        """Test SQL accuracy improvement with schema metadata injection"""
        logger.info("🔍 Testing SQL Accuracy Improvement...")
        
        test_queries = [
            "What is the monthly growth of Energy Met of all regions in 2024?",
            "Show me the total energy consumption by state for 2024",
            "What is the average daily energy shortage across all regions?",
            "Compare energy generation between Eastern and Western regions"
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
        """Run comprehensive test of schema metadata injection"""
        logger.info("🚀 Starting Comprehensive Schema Metadata Injection Test")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Test 1: Schema metadata extraction
        test1_success = await self.test_schema_metadata_extraction()
        
        # Test 2: Semantic engine integration
        test2_success = await self.test_semantic_engine_integration()
        
        # Test 3: SQL accuracy improvement
        test3_results = await self.test_sql_accuracy_improvement()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("=" * 60)
        logger.info("📊 COMPREHENSIVE TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"✅ Schema Metadata Extraction: {'PASS' if test1_success else 'FAIL'}")
        logger.info(f"✅ Semantic Engine Integration: {'PASS' if test2_success else 'FAIL'}")
        logger.info(f"✅ SQL Accuracy Improvement: {'PASS' if test3_results else 'FAIL'}")
        logger.info(f"⏱️  Total Duration: {duration:.2f} seconds")
        
        if test3_results:
            success_rate = sum(1 for r in test3_results if r['success']) / len(test3_results) * 100
            avg_confidence = sum(r['confidence'] for r in test3_results) / len(test3_results)
            logger.info(f"📈 SQL Generation Success Rate: {success_rate:.1f}%")
            logger.info(f"📈 Average Confidence: {avg_confidence:.2f}")
            
        logger.info("=" * 60)
        
        return {
            "schema_extraction": test1_success,
            "semantic_integration": test2_success,
            "sql_accuracy": bool(test3_results),
            "duration": duration,
            "sql_results": test3_results if test3_results else []
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
    tester = SchemaMetadataTester(db_path)
    
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
            
    logger.info("🎯 Schema Metadata Injection Test Complete!")


if __name__ == "__main__":
    asyncio.run(main())
