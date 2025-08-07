#!/usr/bin/env python3
"""
Test script for Multi-Step Query Planning functionality.
Tests the adaptive query decomposition and planning capabilities.
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
from core.query_planner import QueryComplexity, QueryDecomposer, MultiStepQueryPlanner
from config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiStepPlanningTester:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.settings = get_settings()
        # Create LLM provider with correct parameters
        self.llm_provider = create_llm_provider(
            provider_type="ollama",
            model="llama3.2:3b",
            base_url="http://localhost:11434"
        )
        self.semantic_engine = SemanticEngine(self.llm_provider, db_path)
        self.query_planner = MultiStepQueryPlanner(self.llm_provider)
        
    async def initialize(self):
        """Initialize the semantic engine"""
        await self.semantic_engine.initialize()
        logger.info("Multi-step planning tester initialized successfully")
    
    async def test_query_complexity_analysis(self):
        """Test query complexity analysis"""
        logger.info("🔍 Testing Query Complexity Analysis...")
        
        test_queries = [
            ("What is the average energy shortage?", QueryComplexity.SIMPLE),
            ("Show me energy consumption by region", QueryComplexity.MODERATE),
            ("Compare average energy shortage vs consumption by region over time", QueryComplexity.COMPLEX),
            ("What is the trend of energy shortage growth compared to consumption patterns across different regions and time periods?", QueryComplexity.VERY_COMPLEX)
        ]
        
        results = []
        for query, expected_complexity in test_queries:
            complexity = self.semantic_engine._analyze_query_complexity(query)
            is_correct = complexity.value == expected_complexity.value
            results.append({
                "query": query,
                "expected": expected_complexity.value,
                "actual": complexity.value,
                "correct": is_correct
            })
            
            logger.info(f"   Query: {query[:50]}...")
            logger.info(f"   Expected: {expected_complexity.value}, Actual: {complexity.value}, ✅: {is_correct}")
        
        correct_count = sum(1 for r in results if r["correct"])
        success_rate = (correct_count / len(results)) * 100
        
        logger.info(f"📊 Complexity Analysis Results: {correct_count}/{len(results)} correct ({success_rate:.1f}%)")
        return results
    
    async def test_query_decomposition(self):
        """Test query decomposition functionality"""
        logger.info("🔍 Testing Query Decomposition...")
        
        complex_queries = [
            "Compare average energy shortage vs consumption by region over time",
            "What is the trend of energy shortage growth compared to consumption patterns across different regions?",
            "Show me the monthly growth rate of energy consumption by region and state"
        ]
        
        results = []
        for query in complex_queries:
            logger.info(f"   Testing decomposition for: {query[:60]}...")
            
            try:
                # Create a simple schema context for testing
                schema_context = {
                    "primary_table": {
                        "name": "FactAllIndiaDailySummary",
                        "info": {
                            "description": "Daily energy data",
                            "key_metrics": ["EnergyMet", "EnergyShortage", "EnergyRequirement"]
                        }
                    },
                    "related_tables": [
                        {
                            "name": "DimRegions",
                            "info": {"description": "Region information"}
                        },
                        {
                            "name": "DimDates", 
                            "info": {"description": "Date information"}
                        }
                    ]
                }
                
                # Test decomposition
                plan = await self.query_planner.decomposer.decompose_query(query, schema_context)
                
                results.append({
                    "query": query,
                    "plan_id": plan.query_id,
                    "complexity": plan.complexity.value,
                    "steps_count": len(plan.steps),
                    "confidence": plan.confidence,
                    "success": True
                })
                
                logger.info(f"   ✅ Generated plan with {len(plan.steps)} steps (confidence: {plan.confidence:.2f})")
                
                # Log step details
                for step in plan.steps:
                    logger.info(f"      Step {step.step_id}: {step.description} ({step.intent})")
                    
            except Exception as e:
                logger.error(f"   ❌ Failed to decompose query: {e}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        successful_decompositions = sum(1 for r in results if r.get("success", False))
        success_rate = (successful_decompositions / len(results)) * 100
        
        logger.info(f"📊 Decomposition Results: {successful_decompositions}/{len(results)} successful ({success_rate:.1f}%)")
        return results
    
    async def test_multi_step_execution(self):
        """Test multi-step query execution"""
        logger.info("🔍 Testing Multi-Step Query Execution...")
        
        test_query = "Compare average energy shortage vs consumption by region over time"
        
        try:
            # Create schema context
            schema_context = {
                "primary_table": {
                    "name": "FactAllIndiaDailySummary",
                    "info": {
                        "description": "Daily energy data",
                        "key_metrics": ["EnergyMet", "EnergyShortage", "EnergyRequirement"]
                    }
                },
                "related_tables": [
                    {
                        "name": "DimRegions",
                        "info": {"description": "Region information"}
                    },
                    {
                        "name": "DimDates", 
                        "info": {"description": "Date information"}
                    }
                ]
            }
            
            # Test multi-step execution
            execution_result = await self.query_planner.plan_and_execute(test_query, schema_context)
            
            if execution_result["success"]:
                logger.info(f"   ✅ Multi-step execution successful")
                logger.info(f"   Plan ID: {execution_result['plan'].query_id}")
                logger.info(f"   Complexity: {execution_result['plan'].complexity.value}")
                logger.info(f"   Steps: {len(execution_result['plan'].steps)}")
                logger.info(f"   Confidence: {execution_result['confidence']:.2f}")
                
                # Log execution results
                for result in execution_result["execution_results"]:
                    status = "✅" if result.get("success", False) else "❌"
                    logger.info(f"      {status} Step {result['step_id']}: {result['description']}")
                
                return {
                    "success": True,
                    "plan_id": execution_result['plan'].query_id,
                    "steps_count": len(execution_result['plan'].steps),
                    "confidence": execution_result['confidence'],
                    "execution_results": execution_result["execution_results"]
                }
            else:
                logger.error(f"   ❌ Multi-step execution failed: {execution_result.get('error', 'Unknown error')}")
                return {"success": False, "error": execution_result.get('error', 'Unknown error')}
                
        except Exception as e:
            logger.error(f"   ❌ Multi-step execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_semantic_engine_integration(self):
        """Test integration with semantic engine"""
        logger.info("🔍 Testing Semantic Engine Integration...")
        
        complex_query = "Compare average energy shortage vs consumption by region over time"
        
        try:
            # Extract semantic context
            semantic_context = await self.semantic_engine.extract_semantic_context(complex_query)
            logger.info(f"   ✅ Semantic context extracted (confidence: {semantic_context.confidence:.2f})")
            
            # Retrieve schema context
            schema_context = await self.semantic_engine.retrieve_schema_context(semantic_context)
            logger.info(f"   ✅ Schema context retrieved")
            
            # Generate SQL with multi-step planning
            sql_result = await self.semantic_engine.generate_contextual_sql(
                complex_query, semantic_context, schema_context
            )
            
            if sql_result.get("sql"):
                logger.info(f"   ✅ SQL generated successfully")
                logger.info(f"   SQL: {sql_result['sql'][:200]}...")
                logger.info(f"   Confidence: {sql_result['confidence']:.2f}")
                logger.info(f"   Validation passed: {sql_result.get('validation_passed', False)}")
                
                # Check if multi-step planning was used
                planning_details = sql_result.get("planning_details")
                if planning_details:
                    logger.info(f"   Multi-step planning used: {planning_details['steps_count']} steps")
                    logger.info(f"   Plan complexity: {planning_details['complexity']}")
                else:
                    logger.info(f"   Single-step approach used")
                
                return {
                    "success": True,
                    "sql_generated": True,
                    "confidence": sql_result['confidence'],
                    "validation_passed": sql_result.get('validation_passed', False),
                    "planning_used": planning_details is not None
                }
            else:
                logger.error(f"   ❌ Failed to generate SQL")
                return {"success": False, "error": "No SQL generated"}
                
        except Exception as e:
            logger.error(f"   ❌ Semantic engine integration failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        logger.info("🚀 Starting Comprehensive Multi-Step Planning Test")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Initialize
        await self.initialize()
        
        # Run tests
        complexity_results = await self.test_query_complexity_analysis()
        decomposition_results = await self.test_query_decomposition()
        execution_results = await self.test_multi_step_execution()
        integration_results = await self.test_semantic_engine_integration()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate overall success rate
        total_tests = len(complexity_results) + len(decomposition_results) + 1 + 1  # +1 for execution, +1 for integration
        successful_tests = (
            sum(1 for r in complexity_results if r["correct"]) +
            sum(1 for r in decomposition_results if r.get("success", False)) +
            (1 if execution_results.get("success", False) else 0) +
            (1 if integration_results.get("success", False) else 0)
        )
        
        overall_success_rate = (successful_tests / total_tests) * 100
        
        # Print results
        logger.info("=" * 60)
        logger.info("📊 COMPREHENSIVE TEST RESULTS")
        logger.info("=" * 60)
        
        logger.info(f"✅ Query Complexity Analysis: {sum(1 for r in complexity_results if r['correct'])}/{len(complexity_results)} correct")
        logger.info(f"✅ Query Decomposition: {sum(1 for r in decomposition_results if r.get('success', False))}/{len(decomposition_results)} successful")
        logger.info(f"✅ Multi-Step Execution: {'✅ PASS' if execution_results.get('success', False) else '❌ FAIL'}")
        logger.info(f"✅ Semantic Engine Integration: {'✅ PASS' if integration_results.get('success', False) else '❌ FAIL'}")
        
        logger.info(f"⏱️  Total Duration: {duration:.2f} seconds")
        logger.info(f"📈 Overall Success Rate: {overall_success_rate:.1f}%")
        
        logger.info("=" * 60)
        
        return {
            "complexity_results": complexity_results,
            "decomposition_results": decomposition_results,
            "execution_results": execution_results,
            "integration_results": integration_results,
            "overall_success_rate": overall_success_rate,
            "duration": duration
        }

async def main():
    """Main test function"""
    db_path = "C:/Users/arjun/Desktop/PSPreport/power_data.db"
    
    tester = MultiStepPlanningTester(db_path)
    results = await tester.run_comprehensive_test()
    
    print("\n" + "=" * 60)
    print("🎯 FINAL RESULTS")
    print("=" * 60)
    print(f"Query Complexity Analysis: {'✅ PASS' if results['complexity_results'][0]['correct'] else '❌ FAIL'}")
    print(f"Query Decomposition: {'✅ PASS' if results['decomposition_results'][0].get('success', False) else '❌ FAIL'}")
    print(f"Multi-Step Execution: {'✅ PASS' if results['execution_results'].get('success', False) else '❌ FAIL'}")
    print(f"Semantic Engine Integration: {'✅ PASS' if results['integration_results'].get('success', False) else '❌ FAIL'}")
    print(f"Overall Success Rate: {results['overall_success_rate']:.1f}%")
    print("=" * 60)
    
    if results['overall_success_rate'] >= 80:
        print("🎉 Multi-step query planning is working correctly!")
    else:
        print("⚠️  Some issues detected. Check logs for details.")

if __name__ == "__main__":
    asyncio.run(main())
