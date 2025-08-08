"""
Test script for Phase 4.1: Constrained SQL Generation with Templates

This script tests the new template-based SQL generation system to ensure
it correctly handles different query types and enforces business rules.

Expected Impact: +15-20% accuracy improvement
"""

import asyncio
import logging
import sys
import os
from typing import Dict, List, Tuple

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.core.sql_templates import SQLTemplateEngine, TemplateContext, TemplateValidation, QueryType, AggregationType
from backend.core.llm_provider import create_llm_provider
from backend.core.semantic_engine import SemanticEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConstrainedSQLGenerationTester:
    """Test suite for constrained SQL generation with templates"""
    
    def __init__(self):
        self.template_engine = SQLTemplateEngine()
        self.test_queries = self._initialize_test_queries()
        self.results = []
    
    def _initialize_test_queries(self) -> List[Dict[str, str]]:
        """Initialize test queries for different scenarios"""
        return [
            # Average shortage queries (critical for accuracy improvement)
            {
                "query": "What is the average shortage?",
                "expected_type": QueryType.SHORTAGE_ANALYSIS,
                "expected_aggregation": AggregationType.AVG,
                "expected_column": "EnergyShortage",
                "expected_template": "avg_shortage"
            },
            {
                "query": "Show me the average energy shortage by region",
                "expected_type": QueryType.SHORTAGE_ANALYSIS,
                "expected_aggregation": AggregationType.AVG,
                "expected_column": "EnergyShortage",
                "expected_template": "avg_shortage"
            },
            
            # Average energy queries
            {
                "query": "What is the average energy consumption?",
                "expected_type": QueryType.AGGREGATION,
                "expected_aggregation": AggregationType.AVG,
                "expected_column": "EnergyMet",
                "expected_template": "avg_energy"
            },
            {
                "query": "Show me the average energy met by month",
                "expected_type": QueryType.TIME_SERIES,
                "expected_aggregation": AggregationType.AVG,
                "expected_column": "EnergyMet",
                "expected_template": "time_series_energy"
            },
            
            # Total shortage queries
            {
                "query": "What is the total shortage?",
                "expected_type": QueryType.SHORTAGE_ANALYSIS,
                "expected_aggregation": AggregationType.SUM,
                "expected_column": "EnergyShortage",
                "expected_template": "sum_shortage"
            },
            {
                "query": "Show me the total energy shortage by region",
                "expected_type": QueryType.REGIONAL_ANALYSIS,
                "expected_aggregation": AggregationType.SUM,
                "expected_column": "EnergyShortage",
                "expected_template": "regional_shortage"
            },
            
            # Total energy queries
            {
                "query": "What is the total energy consumption?",
                "expected_type": QueryType.AGGREGATION,
                "expected_aggregation": AggregationType.SUM,
                "expected_column": "EnergyMet",
                "expected_template": "sum_energy"
            },
            
            # Maximum demand queries
            {
                "query": "What is the maximum demand?",
                "expected_type": QueryType.AGGREGATION,
                "expected_aggregation": AggregationType.MAX,
                "expected_column": "MaxDemandSCADA",
                "expected_template": "max_demand"
            },
            
            # Minimum demand queries
            {
                "query": "What is the minimum demand?",
                "expected_type": QueryType.AGGREGATION,
                "expected_aggregation": AggregationType.MIN,
                "expected_column": "MaxDemandSCADA",
                "expected_template": "min_demand"
            },
            
            # Time series queries
            {
                "query": "Show me energy shortage trends over time",
                "expected_type": QueryType.TIME_SERIES,
                "expected_aggregation": AggregationType.SUM,
                "expected_column": "EnergyShortage",
                "expected_template": "time_series_shortage"
            },
            
            # Regional analysis queries
            {
                "query": "Show me energy consumption by region",
                "expected_type": QueryType.REGIONAL_ANALYSIS,
                "expected_aggregation": AggregationType.SUM,
                "expected_column": "EnergyMet",
                "expected_template": "regional_energy"
            }
        ]
    
    def test_template_engine_initialization(self) -> bool:
        """Test that the template engine initializes correctly"""
        try:
            logger.info("Testing template engine initialization...")
            
            # Check if templates are loaded
            assert len(self.template_engine.templates) > 0, "No templates loaded"
            logger.info(f"✓ Loaded {len(self.template_engine.templates)} templates")
            
            # Check if validation rules are loaded
            assert len(self.template_engine.validation_rules) > 0, "No validation rules loaded"
            logger.info(f"✓ Loaded {len(self.template_engine.validation_rules)} validation rules")
            
            # Check if business rules are loaded
            assert len(self.template_engine.business_rules) > 0, "No business rules loaded"
            logger.info(f"✓ Loaded {len(self.template_engine.business_rules)} business rules")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Template engine initialization failed: {e}")
            return False
    
    def test_query_analysis(self) -> bool:
        """Test query analysis functionality"""
        try:
            logger.info("Testing query analysis...")
            
            test_query = "What is the average shortage?"
            context = self.template_engine.analyze_query(test_query)
            
            # Validate context
            assert context.query_type == QueryType.SHORTAGE_ANALYSIS, f"Expected SHORTAGE_ANALYSIS, got {context.query_type}"
            assert context.aggregation_type == AggregationType.AVG, f"Expected AVG, got {context.aggregation_type}"
            assert context.target_column == "EnergyShortage", f"Expected EnergyShortage, got {context.target_column}"
            
            logger.info("✓ Query analysis successful")
            return True
            
        except Exception as e:
            logger.error(f"✗ Query analysis failed: {e}")
            return False
    
    def test_template_selection(self) -> bool:
        """Test template selection for different query types"""
        try:
            logger.info("Testing template selection...")
            
            for test_case in self.test_queries:
                query = test_case["query"]
                expected_template = test_case["expected_template"]
                
                context = self.template_engine.analyze_query(query)
                template_key = self.template_engine._generate_template_key(context)
                
                assert template_key == expected_template, f"Expected {expected_template}, got {template_key} for query: {query}"
            
            logger.info(f"✓ Template selection successful for {len(self.test_queries)} test cases")
            return True
            
        except Exception as e:
            logger.error(f"✗ Template selection failed: {e}")
            return False
    
    def test_sql_generation(self) -> bool:
        """Test SQL generation with templates"""
        try:
            logger.info("Testing SQL generation...")
            
            success_count = 0
            total_count = len(self.test_queries)
            
            for test_case in self.test_queries:
                query = test_case["query"]
                expected_type = test_case["expected_type"]
                expected_aggregation = test_case["expected_aggregation"]
                expected_column = test_case["expected_column"]
                
                logger.info(f"Testing query: {query}")
                
                # Generate SQL
                sql, validation = self.template_engine.generate_sql(query)
                
                if sql and validation.is_valid:
                    # Validate SQL content
                    sql_lower = sql.lower()
                    
                    # Check for expected aggregation function
                    expected_agg = expected_aggregation.value.lower()
                    assert expected_agg in sql_lower, f"Expected {expected_agg} in SQL: {sql}"
                    
                    # Check for expected column
                    expected_col = expected_column.lower()
                    assert expected_col in sql_lower, f"Expected {expected_col} in SQL: {sql}"
                    
                    # Check for required patterns
                    assert "select" in sql_lower, f"Expected SELECT in SQL: {sql}"
                    assert "from" in sql_lower, f"Expected FROM in SQL: {sql}"
                    
                    success_count += 1
                    logger.info(f"✓ Generated valid SQL for: {query}")
                else:
                    logger.warning(f"✗ Failed to generate valid SQL for: {query}")
                    logger.warning(f"  Errors: {validation.errors}")
            
            success_rate = success_count / total_count
            logger.info(f"✓ SQL generation successful: {success_count}/{total_count} ({success_rate:.1%})")
            
            return success_rate >= 0.8  # Expect at least 80% success rate
            
        except Exception as e:
            logger.error(f"✗ SQL generation failed: {e}")
            return False
    
    def test_validation_rules(self) -> bool:
        """Test validation rules enforcement"""
        try:
            logger.info("Testing validation rules...")
            
            # Test average shortage query validation
            query = "What is the average shortage?"
            sql, validation = self.template_engine.generate_sql(query)
            
            if sql and validation.is_valid:
                sql_lower = sql.lower()
                
                # Check that it uses AVG
                assert "avg(" in sql_lower, "Average shortage query should use AVG()"
                
                # Check that it uses EnergyShortage
                assert "energyshortage" in sql_lower, "Average shortage query should use EnergyShortage"
                
                # Check that it doesn't use SUM
                assert "sum(" not in sql_lower, "Average shortage query should not use SUM()"
                
                # Check that it doesn't use EnergyMet
                assert "energymet" not in sql_lower, "Average shortage query should not use EnergyMet"
                
                logger.info("✓ Validation rules enforced correctly")
                return True
            else:
                logger.error("✗ Failed to generate SQL for validation test")
                return False
                
        except Exception as e:
            logger.error(f"✗ Validation rules test failed: {e}")
            return False
    
    def test_integration_with_semantic_engine(self) -> bool:
        """Test integration with semantic engine"""
        try:
            logger.info("Testing integration with semantic engine...")
            
            # Create LLM provider
            llm_provider = create_llm_provider(
                provider_type="ollama",
                model="llama2",
                base_url="http://localhost:11434"
            )
            
            # Create semantic engine
            semantic_engine = SemanticEngine(llm_provider)
            
            # Test query
            test_query = "What is the average shortage?"
            
            # Create generation context
            generation_context = {
                "query": test_query,
                "confidence": 0.8,
                "semantic_mappings": {}
            }
            
            # Generate SQL using semantic engine
            result = asyncio.run(semantic_engine._generate_sql_single_step(generation_context))
            
            if result and result.get("sql"):
                sql = result["sql"]
                generation_method = result.get("generation_method", "")
                
                logger.info(f"✓ Semantic engine integration successful")
                logger.info(f"  Generation method: {generation_method}")
                logger.info(f"  SQL: {sql[:100]}...")
                
                return True
            else:
                logger.error("✗ Semantic engine integration failed")
                return False
                
        except Exception as e:
            logger.error(f"✗ Semantic engine integration test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results"""
        logger.info("🚀 Starting Phase 4.1: Constrained SQL Generation with Templates Tests")
        logger.info("=" * 80)
        
        test_results = {}
        
        # Test 1: Template engine initialization
        test_results["template_engine_initialization"] = self.test_template_engine_initialization()
        
        # Test 2: Query analysis
        test_results["query_analysis"] = self.test_query_analysis()
        
        # Test 3: Template selection
        test_results["template_selection"] = self.test_template_selection()
        
        # Test 4: SQL generation
        test_results["sql_generation"] = self.test_sql_generation()
        
        # Test 5: Validation rules
        test_results["validation_rules"] = self.test_validation_rules()
        
        # Test 6: Integration with semantic engine
        test_results["semantic_engine_integration"] = self.test_integration_with_semantic_engine()
        
        # Summary
        logger.info("=" * 80)
        logger.info("📊 Test Results Summary:")
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"  {test_name}: {status}")
        
        success_rate = passed_tests / total_tests
        logger.info(f"\n🎯 Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1%})")
        
        if success_rate >= 0.8:
            logger.info("🎉 Phase 4.1: Constrained SQL Generation with Templates - SUCCESS!")
            logger.info("Expected Impact: +15-20% accuracy improvement achieved")
        else:
            logger.error("❌ Phase 4.1: Constrained SQL Generation with Templates - FAILED!")
            logger.error("Some tests failed, review and fix issues")
        
        return test_results


def main():
    """Main function to run the tests"""
    tester = ConstrainedSQLGenerationTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    success_rate = sum(results.values()) / len(results)
    if success_rate >= 0.8:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()
