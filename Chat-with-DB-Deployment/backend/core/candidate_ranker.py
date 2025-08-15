"""
Candidate Ranking Module
Ranks multiple SQL candidates by syntactic and semantic plausibility
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp, parse

logger = logging.getLogger(__name__)


class RankingCriteria(Enum):
    """Criteria for ranking SQL candidates"""

    SYNTAX_CORRECTNESS = "syntax_correctness"
    SCHEMA_COMPLIANCE = "schema_compliance"
    QUERY_COMPLEXITY = "query_complexity"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    EXECUTION_SAFETY = "execution_safety"


@dataclass
class RankedCandidate:
    """A ranked SQL candidate"""

    sql: str
    rank: int
    score: float
    criteria_scores: Dict[RankingCriteria, float]
    confidence: float
    errors: List[str]
    warnings: List[str]


class CandidateRanker:
    """Ranks SQL candidates by multiple criteria"""

    def __init__(
        self,
        schema_info: Dict[str, List[str]],
        example_queries: Optional[List[Dict[str, Any]]] = None,
    ):
        self.schema_info = schema_info
        self.example_queries = example_queries or []
        self.criteria_weights = {
            RankingCriteria.SYNTAX_CORRECTNESS: 0.3,
            RankingCriteria.SCHEMA_COMPLIANCE: 0.25,
            RankingCriteria.QUERY_COMPLEXITY: 0.15,
            RankingCriteria.SEMANTIC_SIMILARITY: 0.2,
            RankingCriteria.EXECUTION_SAFETY: 0.1,
        }

    def rank_candidates(
        self, candidates: List[str], query: str, query_analysis: Any
    ) -> List[RankedCandidate]:
        """
        Rank SQL candidates by multiple criteria
        """
        ranked_candidates = []

        for i, sql in enumerate(candidates):
            try:
                # Calculate scores for each criterion
                criteria_scores = {}

                # Syntax correctness
                criteria_scores[RankingCriteria.SYNTAX_CORRECTNESS] = (
                    self._score_syntax_correctness(sql)
                )

                # Schema compliance
                criteria_scores[RankingCriteria.SCHEMA_COMPLIANCE] = (
                    self._score_schema_compliance(sql)
                )

                # Query complexity
                criteria_scores[RankingCriteria.QUERY_COMPLEXITY] = (
                    self._score_query_complexity(sql)
                )

                # Semantic similarity to examples
                criteria_scores[RankingCriteria.SEMANTIC_SIMILARITY] = (
                    self._score_semantic_similarity(sql, query)
                )

                # Execution safety
                criteria_scores[RankingCriteria.EXECUTION_SAFETY] = (
                    self._score_execution_safety(sql)
                )

                # Calculate overall score
                overall_score = sum(
                    criteria_scores[criteria] * self.criteria_weights[criteria]
                    for criteria in RankingCriteria
                )

                # Generate errors and warnings
                errors, warnings = self._extract_errors_and_warnings(
                    sql, criteria_scores
                )

                # Calculate confidence
                confidence = self._calculate_confidence(criteria_scores, errors)

                ranked_candidates.append(
                    RankedCandidate(
                        sql=sql,
                        rank=0,  # Will be set after sorting
                        score=overall_score,
                        criteria_scores=criteria_scores,
                        confidence=confidence,
                        errors=errors,
                        warnings=warnings,
                    )
                )

            except Exception as e:
                logger.error(f"Error ranking candidate {i}: {e}")
                # Add failed candidate with low score
                ranked_candidates.append(
                    RankedCandidate(
                        sql=sql,
                        rank=0,
                        score=0.0,
                        criteria_scores={},
                        confidence=0.0,
                        errors=[f"Ranking failed: {str(e)}"],
                        warnings=[],
                    )
                )

        # Sort by score and assign ranks
        ranked_candidates.sort(key=lambda x: x.score, reverse=True)
        for i, candidate in enumerate(ranked_candidates):
            candidate.rank = i + 1

        return ranked_candidates

    def _score_syntax_correctness(self, sql: str) -> float:
        """Score SQL syntax correctness"""
        try:
            parse_tree = parse(sql, dialect="sqlite")
            if parse_tree and len(parse_tree) > 0:
                return 1.0
            else:
                return 0.0
        except Exception:
            return 0.0

    def _score_schema_compliance(self, sql: str) -> float:
        """Score schema compliance"""
        try:
            parse_tree = parse(sql, dialect="sqlite")
            if not parse_tree or len(parse_tree) == 0:
                return 0.0

            # Use first statement from parse tree
            statement = parse_tree[0] if parse_tree else None
            if not statement:
                return 0.0

            # Extract table and column references
            tables = set()
            columns = set()

            for table in statement.find_all(exp.Table):
                tables.add(str(table.name).lower())

            for column in statement.find_all(exp.Column):
                if column.table:
                    columns.add(f"{str(column.table)}.{column.name}".lower())
                else:
                    columns.add(column.name.lower())

            # Check tables exist
            table_score = 0.0
            if tables:
                existing_tables = [t.lower() for t in self.schema_info.keys()]
                table_matches = sum(1 for table in tables if table in existing_tables)
                table_score = table_matches / len(tables)

            # Check columns exist (basic check)
            column_score = 0.0
            if columns:
                column_matches = 0
                for column_ref in columns:
                    if "." in column_ref:
                        table_name, col = column_ref.split(".", 1)
                        if table_name in self.schema_info:
                            table_columns = [
                                c.lower() for c in self.schema_info[table_name]
                            ]
                            if col in table_columns:
                                column_matches += 1
                column_score = column_matches / len(columns)

            return (table_score + column_score) / 2

        except Exception:
            return 0.0

    def _score_query_complexity(self, sql: str) -> float:
        """Score query complexity (prefer simpler queries)"""
        try:
            parse_tree = parse(sql, dialect="sqlite")
            if not parse_tree or len(parse_tree) == 0:
                return 0.0

            # Use first statement from parse tree
            statement = parse_tree[0] if parse_tree else None
            if not statement:
                return 0.0

            # Count complexity factors
            complexity_factors: float = 0.0

            # Subqueries
            subqueries = list(statement.find_all(exp.Subquery))
            complexity_factors += len(subqueries) * 2.0

            # CTEs
            ctes = list(statement.find_all(exp.CTE))
            complexity_factors += len(ctes) * 1.5

            # Multiple joins
            joins = list(statement.find_all(exp.Join))
            if len(joins) > 3:
                complexity_factors += (len(joins) - 3) * 0.5

            # Window functions
            window_functions = list(statement.find_all(exp.Window))
            complexity_factors += len(window_functions) * 1.5

            # Calculate score (lower complexity = higher score)
            max_complexity = 10.0
            complexity_score = max(0.0, 1.0 - (complexity_factors / max_complexity))

            return complexity_score

        except Exception:
            return 0.5  # Neutral score if parsing fails

    def _score_semantic_similarity(self, sql: str, query: str) -> float:
        """Score semantic similarity to example queries"""
        if not self.example_queries:
            return 0.5  # Neutral score if no examples

        try:
            # Extract key patterns from SQL
            sql_patterns = self._extract_sql_patterns(sql)

            best_similarity = 0.0
            for example in self.example_queries:
                if "sql" in example:
                    example_patterns = self._extract_sql_patterns(example["sql"])
                    similarity = self._calculate_pattern_similarity(
                        sql_patterns, example_patterns
                    )
                    best_similarity = max(best_similarity, similarity)

            return best_similarity

        except Exception:
            return 0.5

    def _extract_sql_patterns(self, sql: str) -> Dict[str, int]:
        """Extract patterns from SQL for similarity comparison"""
        patterns = {}

        # Count SQL keywords
        keywords = [
            "SELECT",
            "FROM",
            "WHERE",
            "GROUP BY",
            "ORDER BY",
            "JOIN",
            "LEFT JOIN",
            "INNER JOIN",
        ]
        for keyword in keywords:
            patterns[f"keyword_{keyword.lower()}"] = len(
                re.findall(rf"\b{keyword}\b", sql, re.IGNORECASE)
            )

        # Count functions
        functions = ["COUNT", "SUM", "AVG", "MAX", "MIN", "ROUND"]
        for func in functions:
            patterns[f"function_{func.lower()}"] = len(
                re.findall(rf"\b{func}\b", sql, re.IGNORECASE)
            )

        # Count operators
        operators = ["=", ">", "<", ">=", "<=", "!=", "LIKE", "IN"]
        for op in operators:
            patterns[f"operator_{op.lower()}"] = len(
                re.findall(rf"\b{re.escape(op)}\b", sql, re.IGNORECASE)
            )

        return patterns

    def _calculate_pattern_similarity(
        self, patterns1: Dict[str, int], patterns2: Dict[str, int]
    ) -> float:
        """Calculate similarity between two pattern dictionaries"""
        all_keys = set(patterns1.keys()) | set(patterns2.keys())
        if not all_keys:
            return 0.0

        total_diff = 0
        for key in all_keys:
            val1 = patterns1.get(key, 0)
            val2 = patterns2.get(key, 0)
            total_diff += abs(val1 - val2)

        max_possible_diff = sum(
            max(patterns1.get(k, 0), patterns2.get(k, 0)) for k in all_keys
        )
        if max_possible_diff == 0:
            return 1.0

        similarity = 1.0 - (total_diff / max_possible_diff)
        return max(0.0, similarity)

    def _score_execution_safety(self, sql: str) -> float:
        """Score execution safety"""
        # Check for dangerous operations
        dangerous_patterns = [
            (r"\bDROP\b", 0.0),
            (r"\bDELETE\b", 0.0),
            (r"\bUPDATE\b", 0.0),
            (r"\bINSERT\b", 0.0),
            (r"\bALTER\b", 0.0),
            (r"\bCREATE\b", 0.0),
        ]

        for pattern, penalty in dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return penalty

        # Check for potentially expensive operations
        expensive_patterns = [
            (r"\bCROSS JOIN\b", 0.7),
            (r"\bCARTESIAN JOIN\b", 0.7),
        ]

        for pattern, penalty in expensive_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return penalty

        return 1.0  # Safe query

    def _extract_errors_and_warnings(
        self, sql: str, criteria_scores: Dict[RankingCriteria, float]
    ) -> Tuple[List[str], List[str]]:
        """Extract errors and warnings based on criteria scores"""
        errors = []
        warnings = []

        # Syntax errors
        if criteria_scores.get(RankingCriteria.SYNTAX_CORRECTNESS, 0) < 0.5:
            errors.append("SQL syntax appears incorrect")

        # Schema violations
        if criteria_scores.get(RankingCriteria.SCHEMA_COMPLIANCE, 0) < 0.5:
            errors.append("SQL references non-existent tables or columns")
        elif criteria_scores.get(RankingCriteria.SCHEMA_COMPLIANCE, 0) < 0.8:
            warnings.append("Some schema references may be incorrect")

        # Complexity warnings
        if criteria_scores.get(RankingCriteria.QUERY_COMPLEXITY, 0) < 0.3:
            warnings.append("Query is very complex and may be difficult to understand")

        # Safety warnings
        if criteria_scores.get(RankingCriteria.EXECUTION_SAFETY, 0) < 0.5:
            errors.append("Query contains potentially dangerous operations")

        return errors, warnings

    def _calculate_confidence(
        self, criteria_scores: Dict[RankingCriteria, float], errors: List[str]
    ) -> float:
        """Calculate overall confidence based on criteria scores and errors"""
        if errors:
            return 0.0

        # Weighted average of criteria scores
        total_weight = sum(self.criteria_weights.values())
        weighted_sum = sum(
            criteria_scores.get(criteria, 0.5) * self.criteria_weights[criteria]
            for criteria in RankingCriteria
        )

        return weighted_sum / total_weight if total_weight > 0 else 0.0
