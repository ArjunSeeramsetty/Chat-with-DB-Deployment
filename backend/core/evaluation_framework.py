#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework for SQL Generation Accuracy
Implements energy domain-specific benchmark with execution accuracy (EX), 
efficiency scoring (VES), and business logic validation
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import pandas as pd
from sqlglot import parse, exp
from sqlglot.errors import ParseError

logger = logging.getLogger(__name__)


class EvaluationMetric(Enum):
    """Evaluation metrics for SQL generation"""
    EXECUTION_ACCURACY = "execution_accuracy"
    EFFICIENCY_SCORE = "efficiency_score"
    BUSINESS_LOGIC_VALIDATION = "business_logic_validation"
    SYNTAX_CORRECTNESS = "syntax_correctness"
    SEMANTIC_ACCURACY = "semantic_accuracy"
    RESPONSE_TIME = "response_time"
    CONFIDENCE_SCORE = "confidence_score"


class QueryCategory(Enum):
    """Categories of queries for evaluation"""
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    JOINING = "joining"
    GROUPING = "grouping"
    SORTING = "sorting"
    COMPLEX = "complex"
    COMPARISON = "comparison"
    TREND_ANALYSIS = "trend_analysis"
    ENERGY_CONSUMPTION = "energy_consumption"
    ENERGY_GENERATION = "energy_generation"
    ENERGY_SHORTAGE = "energy_shortage"
    DEMAND_ANALYSIS = "demand_analysis"


@dataclass
class TestCase:
    """Represents a test case for evaluation"""
    id: str
    category: QueryCategory
    original_query: str
    expected_sql: str
    expected_result: Optional[Dict[str, Any]] = None
    business_rules: List[str] = field(default_factory=list)
    complexity_score: float = 1.0
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Represents the result of an evaluation"""
    test_case_id: str
    original_query: str
    generated_sql: str
    expected_sql: str
    execution_success: bool
    execution_accuracy: float
    efficiency_score: float
    business_logic_valid: bool
    syntax_correct: bool
    semantic_accuracy: float
    response_time: float
    confidence_score: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_result: Optional[Dict[str, Any]] = None
    expected_result: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EvaluationSummary:
    """Summary of evaluation results"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_accuracy: float
    average_response_time: float
    average_confidence: float
    category_breakdown: Dict[str, Dict[str, float]]
    performance_metrics: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class ComprehensiveEvaluationFramework:
    """
    Comprehensive evaluation framework for SQL generation accuracy
    Implements energy domain-specific benchmark with execution accuracy (EX), 
    efficiency scoring (VES), and business logic validation
    """
    
    def __init__(self, db_path: str, test_dataset_path: Optional[str] = None):
        self.db_path = db_path
        self.test_dataset_path = test_dataset_path or "test_data/energy_domain_benchmark.json"
        self.test_cases: List[TestCase] = []
        self.evaluation_results: List[EvaluationResult] = []
        
        # Load test dataset
        self._load_test_dataset()
        
        # Initialize database connection
        self._init_database()
    
    def _load_test_dataset(self):
        """Load energy domain test dataset"""
        try:
            if Path(self.test_dataset_path).exists():
                with open(self.test_dataset_path, 'r') as f:
                    data = json.load(f)
                    self.test_cases = [TestCase(**case) for case in data.get("test_cases", [])]
                logger.info(f"Loaded {len(self.test_cases)} test cases from {self.test_dataset_path}")
            else:
                logger.warning(f"Test dataset not found at {self.test_dataset_path}, creating default dataset")
                self._create_default_test_dataset()
        except Exception as e:
            logger.error(f"Error loading test dataset: {e}")
            self._create_default_test_dataset()
    
    def _create_default_test_dataset(self):
        """Create default energy domain test dataset"""
        self.test_cases = [
            # Aggregation queries
            TestCase(
                id="agg_001",
                category=QueryCategory.AGGREGATION,
                original_query="What is the average energy consumption by region?",
                expected_sql="SELECT AVG(fs.EnergyMet) as AvgEnergy, r.RegionName FROM FactAllIndiaDailySummary fs JOIN DimRegions r ON fs.RegionID = r.RegionID GROUP BY r.RegionName",
                business_rules=["EnergyMet should be aggregated using AVG", "Must join with DimRegions for region names"],
                complexity_score=1.5,
                description="Average energy consumption aggregation by region",
                tags=["aggregation", "energy_consumption", "region"]
            ),
            TestCase(
                id="agg_002",
                category=QueryCategory.AGGREGATION,
                original_query="Show me the total energy shortage across all states",
                expected_sql="SELECT SUM(fs.EnergyShortage) as TotalShortage FROM FactAllIndiaDailySummary fs",
                business_rules=["EnergyShortage should be summed for total", "No grouping needed for overall total"],
                complexity_score=1.0,
                description="Total energy shortage calculation",
                tags=["aggregation", "energy_shortage", "total"]
            ),
            
            # Filtering queries
            TestCase(
                id="filter_001",
                category=QueryCategory.FILTERING,
                original_query="Show energy consumption for the last 30 days",
                expected_sql="SELECT fs.EnergyMet, d.Date FROM FactAllIndiaDailySummary fs JOIN DimDates d ON fs.DateID = d.DateID WHERE d.Date >= date('now', '-30 days') ORDER BY d.Date",
                business_rules=["Must filter by date range", "Should join with DimDates for date information"],
                complexity_score=2.0,
                description="Energy consumption for last 30 days",
                tags=["filtering", "date_range", "energy_consumption"]
            ),
            
            # Complex queries
            TestCase(
                id="complex_001",
                category=QueryCategory.COMPLEX,
                original_query="What is the trend of energy shortage by region over the past month, showing only regions with shortage > 1000?",
                expected_sql="SELECT AVG(fs.EnergyShortage) as AvgShortage, r.RegionName, d.Date FROM FactAllIndiaDailySummary fs JOIN DimRegions r ON fs.RegionID = r.RegionID JOIN DimDates d ON fs.DateID = d.DateID WHERE d.Date >= date('now', '-30 days') GROUP BY r.RegionName, d.Date HAVING AVG(fs.EnergyShortage) > 1000 ORDER BY r.RegionName, d.Date",
                business_rules=["Must filter by date range", "Must group by region and date", "Must filter by shortage threshold", "Must join multiple tables"],
                complexity_score=3.5,
                description="Complex trend analysis with filtering and grouping",
                tags=["complex", "trend_analysis", "filtering", "grouping"]
            ),
            
            # Energy-specific queries
            TestCase(
                id="energy_001",
                category=QueryCategory.ENERGY_CONSUMPTION,
                original_query="Which regions have the highest energy consumption?",
                expected_sql="SELECT r.RegionName, SUM(fs.EnergyMet) as TotalConsumption FROM FactAllIndiaDailySummary fs JOIN DimRegions r ON fs.RegionID = r.RegionID GROUP BY r.RegionName ORDER BY TotalConsumption DESC LIMIT 10",
                business_rules=["Must aggregate EnergyMet by region", "Must order by consumption descending", "Should limit results"],
                complexity_score=2.0,
                description="Top regions by energy consumption",
                tags=["energy_consumption", "ranking", "region"]
            ),
            
            # Demand analysis
            TestCase(
                id="demand_001",
                category=QueryCategory.DEMAND_ANALYSIS,
                original_query="What is the maximum demand by state?",
                expected_sql="SELECT s.StateName, MAX(fs.MaxDemandSCADA) as MaxDemand FROM FactAllIndiaDailySummary fs JOIN DimStates s ON fs.StateID = s.StateID GROUP BY s.StateName ORDER BY MaxDemand DESC",
                business_rules=["Must use MaxDemandSCADA column", "Must join with DimStates", "Must group by state"],
                complexity_score=2.0,
                description="Maximum demand analysis by state",
                tags=["demand_analysis", "maximum", "state"]
            )
        ]
        
        # Save default dataset
        self._save_test_dataset()
    
    def _save_test_dataset(self):
        """Save test dataset to file"""
        try:
            Path(self.test_dataset_path).parent.mkdir(parents=True, exist_ok=True)
            data = {
                "test_cases": [
                    {
                        "id": case.id,
                        "category": case.category.value,
                        "original_query": case.original_query,
                        "expected_sql": case.expected_sql,
                        "expected_result": case.expected_result,
                        "business_rules": case.business_rules,
                        "complexity_score": case.complexity_score,
                        "description": case.description,
                        "tags": case.tags
                    }
                    for case in self.test_cases
                ]
            }
            with open(self.test_dataset_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.test_cases)} test cases to {self.test_dataset_path}")
        except Exception as e:
            logger.error(f"Error saving test dataset: {e}")
    
    def _init_database(self):
        """Initialize database connection and create evaluation tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS evaluation_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        test_case_id TEXT NOT NULL,
                        original_query TEXT NOT NULL,
                        generated_sql TEXT NOT NULL,
                        expected_sql TEXT NOT NULL,
                        execution_success BOOLEAN NOT NULL,
                        execution_accuracy REAL NOT NULL,
                        efficiency_score REAL NOT NULL,
                        business_logic_valid BOOLEAN NOT NULL,
                        syntax_correct BOOLEAN NOT NULL,
                        semantic_accuracy REAL NOT NULL,
                        response_time REAL NOT NULL,
                        confidence_score REAL NOT NULL,
                        errors TEXT,
                        warnings TEXT,
                        execution_result TEXT,
                        expected_result TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS evaluation_summaries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        total_tests INTEGER NOT NULL,
                        passed_tests INTEGER NOT NULL,
                        failed_tests INTEGER NOT NULL,
                        overall_accuracy REAL NOT NULL,
                        average_response_time REAL NOT NULL,
                        average_confidence REAL NOT NULL,
                        category_breakdown TEXT NOT NULL,
                        performance_metrics TEXT NOT NULL,
                        recommendations TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("Evaluation database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing evaluation database: {e}")
    
    async def evaluate_sql_generation(
        self, 
        semantic_engine, 
        test_cases: Optional[List[TestCase]] = None
    ) -> EvaluationSummary:
        """
        Evaluate SQL generation accuracy and performance
        
        Args:
            semantic_engine: The semantic engine to evaluate
            test_cases: Optional list of test cases to evaluate (uses default if None)
        
        Returns:
            EvaluationSummary: Summary of evaluation results
        """
        if test_cases is None:
            test_cases = self.test_cases
        
        logger.info(f"🚀 Starting comprehensive evaluation with {len(test_cases)} test cases")
        
        results = []
        for test_case in test_cases:
            try:
                logger.info(f"📋 Evaluating test case: {test_case.id} - {test_case.description}")
                
                # Generate SQL using semantic engine
                start_time = time.time()
                
                # Create semantic context for the query
                semantic_context = {
                    "intent": "aggregation",
                    "confidence": 0.8,
                    "semantic_mappings": {},
                    "business_entities": [],
                    "temporal_context": {},
                    "domain_concepts": []
                }
                
                # Create schema context
                schema_context = {
                    "primary_table": "FactAllIndiaDailySummary",
                    "relationships": []
                }
                
                sql_result = await semantic_engine.generate_contextual_sql(
                    natural_language_query=test_case.original_query,
                    semantic_context=semantic_context,
                    schema_context=schema_context
                )
                response_time = time.time() - start_time
                
                generated_sql = sql_result.get("sql", "")
                confidence_score = sql_result.get("confidence", 0.0)
                
                # Evaluate the generated SQL
                evaluation_result = await self._evaluate_single_result(
                    test_case, generated_sql, response_time, confidence_score
                )
                
                results.append(evaluation_result)
                
                # Store result in database
                await self._store_evaluation_result(evaluation_result)
                
                logger.info(f"✅ Test case {test_case.id}: {'PASSED' if evaluation_result.execution_success else 'FAILED'}")
                
            except Exception as e:
                logger.error(f"❌ Error evaluating test case {test_case.id}: {e}")
                # Create failed result
                failed_result = EvaluationResult(
                    test_case_id=test_case.id,
                    original_query=test_case.original_query,
                    generated_sql="",
                    expected_sql=test_case.expected_sql,
                    execution_success=False,
                    execution_accuracy=0.0,
                    efficiency_score=0.0,
                    business_logic_valid=False,
                    syntax_correct=False,
                    semantic_accuracy=0.0,
                    response_time=0.0,
                    confidence_score=0.0,
                    errors=[str(e)]
                )
                results.append(failed_result)
        
        # Generate summary
        summary = self._generate_evaluation_summary(results)
        
        # Store summary in database
        await self._store_evaluation_summary(summary)
        
        logger.info(f"🎯 Evaluation completed: {summary.passed_tests}/{summary.total_tests} passed ({summary.overall_accuracy:.1%} accuracy)")
        
        return summary
    
    async def _evaluate_single_result(
        self, 
        test_case: TestCase, 
        generated_sql: str, 
        response_time: float, 
        confidence_score: float
    ) -> EvaluationResult:
        """Evaluate a single SQL generation result"""
        
        # Initialize result
        result = EvaluationResult(
            test_case_id=test_case.id,
            original_query=test_case.original_query,
            generated_sql=generated_sql,
            expected_sql=test_case.expected_sql,
            execution_success=False,
            execution_accuracy=0.0,
            efficiency_score=0.0,
            business_logic_valid=False,
            syntax_correct=False,
            semantic_accuracy=0.0,
            response_time=response_time,
            confidence_score=confidence_score
        )
        
        # 1. Syntax validation
        result.syntax_correct = self._validate_syntax(generated_sql)
        
        # 2. Execution accuracy
        if result.syntax_correct:
            result.execution_success, result.execution_accuracy = await self._validate_execution(
                generated_sql, test_case.expected_sql
            )
        
        # 3. Efficiency scoring
        result.efficiency_score = self._calculate_efficiency_score(generated_sql, test_case)
        
        # 4. Business logic validation
        result.business_logic_valid = self._validate_business_logic(generated_sql, test_case.business_rules)
        
        # 5. Semantic accuracy
        result.semantic_accuracy = self._calculate_semantic_accuracy(generated_sql, test_case.expected_sql)
        
        return result
    
    def _validate_syntax(self, sql: str) -> bool:
        """Validate SQL syntax using sqlglot"""
        try:
            if not sql.strip():
                return False
            parse(sql)
            return True
        except ParseError:
            return False
    
    async def _validate_execution(self, generated_sql: str, expected_sql: str) -> Tuple[bool, float]:
        """Validate SQL execution and compare with expected result"""
        try:
            # Execute generated SQL
            generated_result = await self._execute_sql(generated_sql)
            
            # Execute expected SQL
            expected_result = await self._execute_sql(expected_sql)
            
            # Compare results
            if generated_result is None or expected_result is None:
                return False, 0.0
            
            # Calculate accuracy based on result similarity
            accuracy = self._calculate_result_similarity(generated_result, expected_result)
            
            return True, accuracy
            
        except Exception as e:
            logger.error(f"Error validating execution: {e}")
            return False, 0.0
    
    async def _execute_sql(self, sql: str) -> Optional[Dict[str, Any]]:
        """Execute SQL and return result"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Use EXPLAIN to validate execution without actually running
                explain_sql = f"EXPLAIN {sql}"
                cursor = conn.execute(explain_sql)
                result = cursor.fetchall()
                
                # If EXPLAIN succeeds, try to get actual result (with LIMIT for safety)
                safe_sql = self._make_sql_safe(sql)
                cursor = conn.execute(safe_sql)
                result = cursor.fetchall()
                
                return {
                    "row_count": len(result),
                    "columns": [description[0] for description in cursor.description] if cursor.description else [],
                    "data": result[:10]  # Limit to first 10 rows for comparison
                }
                
        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            return None
    
    def _make_sql_safe(self, sql: str) -> str:
        """Make SQL safe for execution by adding LIMIT if not present"""
        try:
            parsed = parse(sql)
            if not any(isinstance(node, exp.Limit) for node in parsed.walk()):
                return f"{sql} LIMIT 100"
            return sql
        except:
            return f"{sql} LIMIT 100"
    
    def _calculate_result_similarity(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> float:
        """Calculate similarity between two query results"""
        if not result1 or not result2:
            return 0.0
        
        # Compare row counts
        row_count_similarity = min(result1["row_count"], result2["row_count"]) / max(result1["row_count"], result2["row_count"]) if max(result1["row_count"], result2["row_count"]) > 0 else 1.0
        
        # Compare column structures
        columns1 = set(result1.get("columns", []))
        columns2 = set(result2.get("columns", []))
        column_similarity = len(columns1.intersection(columns2)) / len(columns1.union(columns2)) if columns1.union(columns2) else 1.0
        
        # Overall similarity
        similarity = (row_count_similarity + column_similarity) / 2
        return similarity
    
    def _calculate_efficiency_score(self, sql: str, test_case: TestCase) -> float:
        """Calculate efficiency score based on SQL complexity and performance"""
        try:
            parsed = parse(sql)
            if not parsed:
                return 0.0
            
            # Get the first parsed statement
            stmt = parsed[0] if isinstance(parsed, list) else parsed
            
            # Count operations
            operations = {
                "joins": len(list(stmt.find_all(exp.Join))),
                "aggregations": len(list(stmt.find_all(exp.AggFunc))),
                "filters": len(list(stmt.find_all(exp.Where))),
                "groupings": len(list(stmt.find_all(exp.Group))),
                "orderings": len(list(stmt.find_all(exp.Order)))
            }
            
            # Calculate complexity score
            complexity = sum(operations.values()) * 0.2
            
            # Normalize by expected complexity
            normalized_complexity = min(complexity / test_case.complexity_score, 1.0)
            
            # Efficiency score (higher is better)
            efficiency = 1.0 - (normalized_complexity * 0.3)
            
            return max(efficiency, 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating efficiency score: {e}")
            return 0.0
    
    def _validate_business_logic(self, sql: str, business_rules: List[str]) -> bool:
        """Validate SQL against business rules"""
        try:
            parsed = parse(sql)
            sql_lower = sql.lower()
            
            # Check each business rule
            for rule in business_rules:
                rule_lower = rule.lower()
                
                # Check for required aggregation functions
                if "should be aggregated using avg" in rule_lower and "avg(" not in sql_lower:
                    return False
                if "should be aggregated using sum" in rule_lower and "sum(" not in sql_lower:
                    return False
                if "should be aggregated using count" in rule_lower and "count(" not in sql_lower:
                    return False
                if "should be aggregated using max" in rule_lower and "max(" not in sql_lower:
                    return False
                if "should be aggregated using min" in rule_lower and "min(" not in sql_lower:
                    return False
                
                # Check for required operations
                if "must join" in rule_lower and "join" not in sql_lower:
                    return False
                if "must group" in rule_lower and "group" not in sql_lower:
                    return False
                if "must order" in rule_lower and "order" not in sql_lower:
                    return False
                if "must filter" in rule_lower and "where" not in sql_lower:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating business logic: {e}")
            return False
    
    def _calculate_semantic_accuracy(self, generated_sql: str, expected_sql: str) -> float:
        """Calculate semantic accuracy by comparing SQL structures"""
        try:
            generated_parsed = parse(generated_sql)
            expected_parsed = parse(expected_sql)
            
            if not generated_parsed or not expected_parsed:
                return 0.0
            
            # Get the first parsed statement for each
            generated_stmt = generated_parsed[0] if isinstance(generated_parsed, list) else generated_parsed
            expected_stmt = expected_parsed[0] if isinstance(expected_parsed, list) else expected_parsed
            
            # Compare basic structure
            generated_tables = set()
            expected_tables = set()
            
            for table in generated_stmt.find_all(exp.Table):
                generated_tables.add(table.name.lower())
            
            for table in expected_stmt.find_all(exp.Table):
                expected_tables.add(table.name.lower())
            
            table_similarity = len(generated_tables.intersection(expected_tables)) / len(generated_tables.union(expected_tables)) if generated_tables.union(expected_tables) else 1.0
            
            # Compare columns
            generated_columns = set()
            expected_columns = set()
            
            for column in generated_stmt.find_all(exp.Column):
                generated_columns.add(column.name.lower())
            
            for column in expected_stmt.find_all(exp.Column):
                expected_columns.add(column.name.lower())
            
            column_similarity = len(generated_columns.intersection(expected_columns)) / len(generated_columns.union(expected_columns)) if generated_columns.union(expected_columns) else 1.0
            
            # Overall semantic accuracy
            semantic_accuracy = (table_similarity + column_similarity) / 2
            return semantic_accuracy
            
        except Exception as e:
            logger.error(f"Error calculating semantic accuracy: {e}")
            return 0.0
    
    def _generate_evaluation_summary(self, results: List[EvaluationResult]) -> EvaluationSummary:
        """Generate evaluation summary from results"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.execution_success)
        failed_tests = total_tests - passed_tests
        
        # Calculate overall metrics
        overall_accuracy = sum(r.execution_accuracy for r in results) / total_tests if total_tests > 0 else 0.0
        average_response_time = sum(r.response_time for r in results) / total_tests if total_tests > 0 else 0.0
        average_confidence = sum(r.confidence_score for r in results) / total_tests if total_tests > 0 else 0.0
        
        # Category breakdown
        category_breakdown = {}
        for result in results:
            # Find test case category
            test_case = next((tc for tc in self.test_cases if tc.id == result.test_case_id), None)
            if test_case:
                # Handle both enum and string cases
                if hasattr(test_case.category, 'value'):
                    category = test_case.category.value
                else:
                    category = str(test_case.category)
                
                if category not in category_breakdown:
                    category_breakdown[category] = {"total": 0, "passed": 0, "accuracy": 0.0}
                
                category_breakdown[category]["total"] += 1
                if result.execution_success:
                    category_breakdown[category]["passed"] += 1
        
        # Calculate category accuracies
        for category in category_breakdown:
            total = category_breakdown[category]["total"]
            passed = category_breakdown[category]["passed"]
            category_breakdown[category]["accuracy"] = passed / total if total > 0 else 0.0
        
        # Performance metrics
        performance_metrics = {
            "syntax_correctness": sum(1 for r in results if r.syntax_correct) / total_tests if total_tests > 0 else 0.0,
            "business_logic_validation": sum(1 for r in results if r.business_logic_valid) / total_tests if total_tests > 0 else 0.0,
            "semantic_accuracy": sum(r.semantic_accuracy for r in results) / total_tests if total_tests > 0 else 0.0,
            "efficiency_score": sum(r.efficiency_score for r in results) / total_tests if total_tests > 0 else 0.0
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, performance_metrics)
        
        return EvaluationSummary(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            overall_accuracy=overall_accuracy,
            average_response_time=average_response_time,
            average_confidence=average_confidence,
            category_breakdown=category_breakdown,
            performance_metrics=performance_metrics,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, results: List[EvaluationResult], performance_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Overall accuracy recommendations
        if performance_metrics["syntax_correctness"] < 0.9:
            recommendations.append("Improve SQL syntax validation and error handling")
        
        if performance_metrics["business_logic_validation"] < 0.8:
            recommendations.append("Enhance business rule validation and enforcement")
        
        if performance_metrics["semantic_accuracy"] < 0.85:
            recommendations.append("Improve semantic understanding and query intent recognition")
        
        if performance_metrics["efficiency_score"] < 0.7:
            recommendations.append("Optimize query generation for better efficiency")
        
        # Category-specific recommendations
        category_performance = {}
        for result in results:
            test_case = next((tc for tc in self.test_cases if tc.id == result.test_case_id), None)
            if test_case:
                # Handle both enum and string cases
                if hasattr(test_case.category, 'value'):
                    category = test_case.category.value
                else:
                    category = str(test_case.category)
                
                if category not in category_performance:
                    category_performance[category] = []
                category_performance[category].append(result.execution_accuracy)
        
        for category, accuracies in category_performance.items():
            avg_accuracy = sum(accuracies) / len(accuracies)
            if avg_accuracy < 0.8:
                recommendations.append(f"Focus on improving {category} query generation (current accuracy: {avg_accuracy:.1%})")
        
        if not recommendations:
            recommendations.append("System performing well across all metrics - maintain current performance")
        
        return recommendations
    
    async def _store_evaluation_result(self, result: EvaluationResult):
        """Store evaluation result in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO evaluation_results (
                        test_case_id, original_query, generated_sql, expected_sql,
                        execution_success, execution_accuracy, efficiency_score,
                        business_logic_valid, syntax_correct, semantic_accuracy,
                        response_time, confidence_score, errors, warnings,
                        execution_result, expected_result
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.test_case_id, result.original_query, result.generated_sql,
                    result.expected_sql, result.execution_success, result.execution_accuracy,
                    result.efficiency_score, result.business_logic_valid, result.syntax_correct,
                    result.semantic_accuracy, result.response_time, result.confidence_score,
                    json.dumps(result.errors), json.dumps(result.warnings),
                    json.dumps(result.execution_result), json.dumps(result.expected_result)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing evaluation result: {e}")
    
    async def _store_evaluation_summary(self, summary: EvaluationSummary):
        """Store evaluation summary in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO evaluation_summaries (
                        total_tests, passed_tests, failed_tests, overall_accuracy,
                        average_response_time, average_confidence, category_breakdown,
                        performance_metrics, recommendations
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    summary.total_tests, summary.passed_tests, summary.failed_tests,
                    summary.overall_accuracy, summary.average_response_time,
                    summary.average_confidence, json.dumps(summary.category_breakdown),
                    json.dumps(summary.performance_metrics), json.dumps(summary.recommendations)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing evaluation summary: {e}")
    
    def get_evaluation_history(self, limit: int = 10) -> List[EvaluationSummary]:
        """Get evaluation history from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT total_tests, passed_tests, failed_tests, overall_accuracy,
                           average_response_time, average_confidence, category_breakdown,
                           performance_metrics, recommendations, timestamp
                    FROM evaluation_summaries
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
                
                summaries = []
                for row in cursor.fetchall():
                    summary = EvaluationSummary(
                        total_tests=row[0],
                        passed_tests=row[1],
                        failed_tests=row[2],
                        overall_accuracy=row[3],
                        average_response_time=row[4],
                        average_confidence=row[5],
                        category_breakdown=json.loads(row[6]),
                        performance_metrics=json.loads(row[7]),
                        recommendations=json.loads(row[8]),
                        timestamp=datetime.fromisoformat(row[9])
                    )
                    summaries.append(summary)
                
                return summaries
        except Exception as e:
            logger.error(f"Error getting evaluation history: {e}")
            return []
    
    def export_evaluation_report(self, summary: EvaluationSummary, output_path: str):
        """Export evaluation report to file"""
        try:
            report = {
                "evaluation_summary": {
                    "total_tests": summary.total_tests,
                    "passed_tests": summary.passed_tests,
                    "failed_tests": summary.failed_tests,
                    "overall_accuracy": summary.overall_accuracy,
                    "average_response_time": summary.average_response_time,
                    "average_confidence": summary.average_confidence,
                    "timestamp": summary.timestamp.isoformat()
                },
                "category_breakdown": summary.category_breakdown,
                "performance_metrics": summary.performance_metrics,
                "recommendations": summary.recommendations
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Evaluation report exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting evaluation report: {e}")
