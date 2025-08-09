#!/usr/bin/env python3
"""
Test script for Comprehensive Evaluation Framework
Validates energy domain-specific benchmark with execution accuracy (EX), 
efficiency scoring (VES), and business logic validation
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.core.evaluation_framework import (
    ComprehensiveEvaluationFramework, TestCase, QueryCategory,
    EvaluationResult, EvaluationSummary
)
from backend.core.semantic_engine import SemanticEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationFrameworkTester:
    """Test suite for evaluation framework"""
    
    def __init__(self):
        self.db_path = "backend/energy_data.db"
        self.evaluation_framework = ComprehensiveEvaluationFramework(self.db_path)
        self.semantic_engine = SemanticEngine(self.db_path)
        self.test_results = []
    
    async def run_all_tests(self):
        """Run all evaluation framework tests"""
        logger.info("🚀 Starting Evaluation Framework Tests")
        
        test_cases = [
            ("Framework Initialization", self._test_framework_initialization),
            ("Test Dataset Loading", self._test_test_dataset_loading),
            ("SQL Syntax Validation", self._test_sql_syntax_validation),
            ("Execution Accuracy", self._test_execution_accuracy),
            ("Efficiency Scoring", self._test_efficiency_scoring),
            ("Business Logic Validation", self._test_business_logic_validation),
            ("Semantic Accuracy", self._test_semantic_accuracy),
            ("Comprehensive Evaluation", self._test_comprehensive_evaluation),
            ("Evaluation Summary", self._test_evaluation_summary),
            ("Report Export", self._test_report_export),
        ]
        
        for test_name, test_func in test_cases:
            try:
                logger.info(f"\n📋 Running test: {test_name}")
                result = await test_func()
                self.test_results.append((test_name, result))
                logger.info(f"✅ {test_name}: PASSED")
            except Exception as e:
                logger.error(f"❌ {test_name}: FAILED - {e}")
                self.test_results.append((test_name, {"status": "FAILED", "error": str(e)}))
        
        self._print_summary()
    
    async def _test_framework_initialization(self):
        """Test framework initialization"""
        # Check if framework was initialized correctly
        assert self.evaluation_framework.db_path == self.db_path
        assert len(self.evaluation_framework.test_cases) > 0
        
        # Check if database tables were created
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('evaluation_results', 'evaluation_summaries')")
            tables = [row[0] for row in cursor.fetchall()]
            assert len(tables) == 2, f"Expected 2 tables, found {len(tables)}"
        
        return {
            "status": "PASSED",
            "test_cases_count": len(self.evaluation_framework.test_cases),
            "database_tables": tables
        }
    
    async def _test_test_dataset_loading(self):
        """Test test dataset loading"""
        # Check if test cases were loaded
        test_cases = self.evaluation_framework.test_cases
        assert len(test_cases) > 0, "No test cases loaded"
        
        # Check test case structure
        for test_case in test_cases:
            assert hasattr(test_case, 'id'), "Test case missing id"
            assert hasattr(test_case, 'category'), "Test case missing category"
            assert hasattr(test_case, 'original_query'), "Test case missing original_query"
            assert hasattr(test_case, 'expected_sql'), "Test case missing expected_sql"
        
        # Check for specific test cases
        test_case_ids = [tc.id for tc in test_cases]
        expected_ids = ["agg_001", "agg_002", "filter_001", "complex_001", "energy_001", "demand_001"]
        
        for expected_id in expected_ids:
            assert expected_id in test_case_ids, f"Expected test case {expected_id} not found"
        
        return {
            "status": "PASSED",
            "test_cases_loaded": len(test_cases),
            "test_case_ids": test_case_ids
        }
    
    async def _test_sql_syntax_validation(self):
        """Test SQL syntax validation"""
        # Test valid SQL
        valid_sql = "SELECT AVG(fs.EnergyMet) as AvgEnergy FROM FactAllIndiaDailySummary fs"
        is_valid = self.evaluation_framework._validate_syntax(valid_sql)
        assert is_valid, "Valid SQL should pass syntax validation"
        
        # Test invalid SQL
        invalid_sql = "SELECT AVG(fs.EnergyMet) as AvgEnergy FROM FactAllIndiaDailySummary fs GROUP"
        is_invalid = self.evaluation_framework._validate_syntax(invalid_sql)
        assert not is_invalid, "Invalid SQL should fail syntax validation"
        
        # Test empty SQL
        empty_sql = ""
        is_empty_valid = self.evaluation_framework._validate_syntax(empty_sql)
        assert not is_empty_valid, "Empty SQL should fail syntax validation"
        
        return {
            "status": "PASSED",
            "valid_sql_passed": is_valid,
            "invalid_sql_failed": not is_invalid,
            "empty_sql_failed": not is_empty_valid
        }
    
    async def _test_execution_accuracy(self):
        """Test execution accuracy calculation"""
        # Test with similar results
        result1 = {
            "row_count": 10,
            "columns": ["RegionName", "AvgEnergy"],
            "data": [("North", 100), ("South", 200)]
        }
        result2 = {
            "row_count": 10,
            "columns": ["RegionName", "AvgEnergy"],
            "data": [("North", 100), ("South", 200)]
        }
        
        similarity = self.evaluation_framework._calculate_result_similarity(result1, result2)
        assert similarity > 0.9, f"Similar results should have high similarity, got {similarity}"
        
        # Test with different results
        result3 = {
            "row_count": 5,
            "columns": ["RegionName"],
            "data": [("North",)]
        }
        
        similarity2 = self.evaluation_framework._calculate_result_similarity(result1, result3)
        assert similarity2 < similarity, "Different results should have lower similarity"
        
        return {
            "status": "PASSED",
            "similar_results_similarity": similarity,
            "different_results_similarity": similarity2
        }
    
    async def _test_efficiency_scoring(self):
        """Test efficiency scoring"""
        # Test simple query
        simple_sql = "SELECT AVG(fs.EnergyMet) FROM FactAllIndiaDailySummary fs"
        simple_test_case = TestCase(
            id="test_simple",
            category=QueryCategory.AGGREGATION,
            original_query="Test query",
            expected_sql=simple_sql,
            complexity_score=1.0
        )
        
        efficiency_score = self.evaluation_framework._calculate_efficiency_score(simple_sql, simple_test_case)
        assert 0.0 <= efficiency_score <= 1.0, f"Efficiency score should be between 0 and 1, got {efficiency_score}"
        
        # Test complex query
        complex_sql = """
        SELECT AVG(fs.EnergyMet) as AvgEnergy, r.RegionName 
        FROM FactAllIndiaDailySummary fs 
        JOIN DimRegions r ON fs.RegionID = r.RegionID 
        WHERE fs.EnergyMet > 1000 
        GROUP BY r.RegionName 
        ORDER BY AvgEnergy DESC
        """
        complex_test_case = TestCase(
            id="test_complex",
            category=QueryCategory.COMPLEX,
            original_query="Test complex query",
            expected_sql=complex_sql,
            complexity_score=3.0
        )
        
        complex_efficiency = self.evaluation_framework._calculate_efficiency_score(complex_sql, complex_test_case)
        assert 0.0 <= complex_efficiency <= 1.0, f"Complex efficiency score should be between 0 and 1, got {complex_efficiency}"
        
        return {
            "status": "PASSED",
            "simple_efficiency": efficiency_score,
            "complex_efficiency": complex_efficiency
        }
    
    async def _test_business_logic_validation(self):
        """Test business logic validation"""
        # Test valid business logic
        valid_sql = "SELECT AVG(fs.EnergyMet) as AvgEnergy FROM FactAllIndiaDailySummary fs"
        business_rules = ["EnergyMet should be aggregated using AVG"]
        is_valid = self.evaluation_framework._validate_business_logic(valid_sql, business_rules)
        assert is_valid, "Valid business logic should pass validation"
        
        # Test invalid business logic
        invalid_sql = "SELECT SUM(fs.EnergyMet) as AvgEnergy FROM FactAllIndiaDailySummary fs"
        is_invalid = self.evaluation_framework._validate_business_logic(invalid_sql, business_rules)
        assert not is_invalid, "Invalid business logic should fail validation"
        
        # Test multiple rules
        multiple_rules = ["EnergyMet should be aggregated using AVG", "Must join with DimRegions"]
        join_sql = "SELECT AVG(fs.EnergyMet) as AvgEnergy FROM FactAllIndiaDailySummary fs JOIN DimRegions r ON fs.RegionID = r.RegionID"
        is_multiple_valid = self.evaluation_framework._validate_business_logic(join_sql, multiple_rules)
        assert is_multiple_valid, "Multiple valid rules should pass validation"
        
        return {
            "status": "PASSED",
            "valid_business_logic": is_valid,
            "invalid_business_logic": not is_invalid,
            "multiple_rules_valid": is_multiple_valid
        }
    
    async def _test_semantic_accuracy(self):
        """Test semantic accuracy calculation"""
        # Test similar SQL
        sql1 = "SELECT AVG(fs.EnergyMet) as AvgEnergy, r.RegionName FROM FactAllIndiaDailySummary fs JOIN DimRegions r ON fs.RegionID = r.RegionID"
        sql2 = "SELECT AVG(fs.EnergyMet) as AvgEnergy, r.RegionName FROM FactAllIndiaDailySummary fs JOIN DimRegions r ON fs.RegionID = r.RegionID GROUP BY r.RegionName"
        
        semantic_accuracy = self.evaluation_framework._calculate_semantic_accuracy(sql1, sql2)
        assert semantic_accuracy > 0.8, f"Similar SQL should have high semantic accuracy, got {semantic_accuracy}"
        
        # Test different SQL
        sql3 = "SELECT SUM(fs.EnergyShortage) as TotalShortage FROM FactAllIndiaDailySummary fs"
        semantic_accuracy2 = self.evaluation_framework._calculate_semantic_accuracy(sql1, sql3)
        assert semantic_accuracy2 < semantic_accuracy, "Different SQL should have lower semantic accuracy"
        
        return {
            "status": "PASSED",
            "similar_semantic_accuracy": semantic_accuracy,
            "different_semantic_accuracy": semantic_accuracy2
        }
    
    async def _test_comprehensive_evaluation(self):
        """Test comprehensive evaluation with semantic engine"""
        # Use a subset of test cases for faster testing
        test_subset = self.evaluation_framework.test_cases[:3]
        
        # Run evaluation
        summary = await self.evaluation_framework.evaluate_sql_generation(
            self.semantic_engine, 
            test_cases=test_subset
        )
        
        # Validate summary
        assert summary.total_tests == len(test_subset), f"Expected {len(test_subset)} tests, got {summary.total_tests}"
        assert 0.0 <= summary.overall_accuracy <= 1.0, f"Overall accuracy should be between 0 and 1, got {summary.overall_accuracy}"
        assert summary.average_response_time >= 0.0, f"Average response time should be >= 0, got {summary.average_response_time}"
        assert 0.0 <= summary.average_confidence <= 1.0, f"Average confidence should be between 0 and 1, got {summary.average_confidence}"
        
        return {
            "status": "PASSED",
            "total_tests": summary.total_tests,
            "passed_tests": summary.passed_tests,
            "overall_accuracy": summary.overall_accuracy,
            "average_response_time": summary.average_response_time,
            "average_confidence": summary.average_confidence
        }
    
    async def _test_evaluation_summary(self):
        """Test evaluation summary generation"""
        # Create mock results
        mock_results = [
            EvaluationResult(
                test_case_id="test_001",
                original_query="Test query 1",
                generated_sql="SELECT AVG(fs.EnergyMet) FROM FactAllIndiaDailySummary fs",
                expected_sql="SELECT AVG(fs.EnergyMet) FROM FactAllIndiaDailySummary fs",
                execution_success=True,
                execution_accuracy=0.9,
                efficiency_score=0.8,
                business_logic_valid=True,
                syntax_correct=True,
                semantic_accuracy=0.85,
                response_time=1.5,
                confidence_score=0.9
            ),
            EvaluationResult(
                test_case_id="test_002",
                original_query="Test query 2",
                generated_sql="SELECT SUM(fs.EnergyShortage) FROM FactAllIndiaDailySummary fs",
                expected_sql="SELECT SUM(fs.EnergyShortage) FROM FactAllIndiaDailySummary fs",
                execution_success=True,
                execution_accuracy=0.95,
                efficiency_score=0.9,
                business_logic_valid=True,
                syntax_correct=True,
                semantic_accuracy=0.9,
                response_time=1.2,
                confidence_score=0.95
            )
        ]
        
        # Generate summary
        summary = self.evaluation_framework._generate_evaluation_summary(mock_results)
        
        # Validate summary
        assert summary.total_tests == 2, f"Expected 2 tests, got {summary.total_tests}"
        assert summary.passed_tests == 2, f"Expected 2 passed tests, got {summary.passed_tests}"
        assert summary.failed_tests == 0, f"Expected 0 failed tests, got {summary.failed_tests}"
        assert summary.overall_accuracy > 0.9, f"Expected high accuracy, got {summary.overall_accuracy}"
        assert len(summary.recommendations) > 0, "Should have recommendations"
        
        return {
            "status": "PASSED",
            "total_tests": summary.total_tests,
            "passed_tests": summary.passed_tests,
            "overall_accuracy": summary.overall_accuracy,
            "recommendations_count": len(summary.recommendations)
        }
    
    async def _test_report_export(self):
        """Test evaluation report export"""
        # Create mock summary
        mock_summary = EvaluationSummary(
            total_tests=5,
            passed_tests=4,
            failed_tests=1,
            overall_accuracy=0.8,
            average_response_time=1.5,
            average_confidence=0.85,
            category_breakdown={
                "aggregation": {"total": 2, "passed": 2, "accuracy": 1.0},
                "filtering": {"total": 1, "passed": 1, "accuracy": 1.0},
                "complex": {"total": 2, "passed": 1, "accuracy": 0.5}
            },
            performance_metrics={
                "syntax_correctness": 0.9,
                "business_logic_validation": 0.8,
                "semantic_accuracy": 0.85,
                "efficiency_score": 0.75
            },
            recommendations=["Improve complex query generation", "Enhance business rule validation"]
        )
        
        # Export report
        output_path = "test_evaluation_report.json"
        self.evaluation_framework.export_evaluation_report(mock_summary, output_path)
        
        # Check if file was created
        assert Path(output_path).exists(), f"Report file {output_path} was not created"
        
        # Clean up
        Path(output_path).unlink()
        
        return {
            "status": "PASSED",
            "report_exported": True,
            "output_path": output_path
        }
    
    def _print_summary(self):
        """Print test summary"""
        logger.info("\n" + "="*60)
        logger.info("🎯 Evaluation Framework Test Summary")
        logger.info("="*60)
        
        passed = 0
        failed = 0
        
        for test_name, result in self.test_results:
            if isinstance(result, dict) and result.get("status") == "PASSED":
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                failed += 1
                error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
                logger.error(f"❌ {test_name}: FAILED - {error_msg}")
        
        logger.info(f"\n📊 Results: {passed} passed, {failed} failed")
        
        if failed == 0:
            logger.info("🎉 All tests passed! Evaluation framework is working correctly.")
        else:
            logger.error(f"⚠️  {failed} test(s) failed. Please review the errors above.")


async def main():
    """Main test function"""
    tester = EvaluationFrameworkTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
