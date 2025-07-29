"""
SQL validation using sqlglot and sqlfluff for robust parser-based validation
Sprint 5: Integrate sqlfluff fix; enable auto-repair with guard-rails Syntax error rate ↓ 90%
"""

import asyncio
import logging
import os
import re
import shutil
import tempfile
from typing import List, Optional, Tuple

import sqlfluff
import sqlglot
from sqlfluff.api import fix, lint
from sqlfluff.core import FluffConfig, Linter

from backend.core.types import ValidationResult

logger = logging.getLogger(__name__)


class EnhancedSQLValidator:
    """
    Enhanced parser-based validation pipeline using sqlglot and sqlfluff for robust validation.
    Sprint 5: Integrated sqlfluff fix with auto-repair and guard-rails.
    """

    def __init__(self, schema_info: Optional[dict] = None, llm_provider=None):
        # Configure sqlfluff for SQLite dialect with basic settings
        self.linter = Linter(dialect="sqlite")
        self.schema_info = schema_info or {}
        self.llm_provider = llm_provider

        # Sprint 5: Guard-rails for auto-repair
        self.max_repair_attempts = 3
        self.repair_confidence_threshold = 0.7
        self.dangerous_patterns = [
            r"\bDROP\b",
            r"\bDELETE\b",
            r"\bUPDATE\b",
            r"\bINSERT\b",
            r"\bALTER\b",
            r"\bCREATE\b",
            r"\bTRUNCATE\b",
        ]

    def validate_sql(self, sql: str) -> ValidationResult:
        """
        Complete parser-based validation pipeline with Sprint 5 enhancements:
        1. Parse via sqlglot (dialect=sqlite)
        2. Lint with sqlfluff; auto-fix style errors
        3. SQLite-specific validation
        4. Auto-repair with guard-rails
        """
        errors = []
        warnings = []
        fixed_sql = None
        confidence = 1.0

        try:
            # Step 1: Parse via sqlglot (dialect=sqlite)
            parse_valid = self._validate_syntax(sql)
            if not parse_valid[0]:
                errors.extend(parse_valid[1])
                confidence -= 0.4

            # Step 2: Lint with sqlfluff; auto-fix style errors
            lint_result = self._lint_and_fix(sql)
            if lint_result[0]:  # If linting successful
                fixed_sql = lint_result[1]
                warnings.extend(lint_result[2])
                if fixed_sql and fixed_sql != sql:
                    confidence += 0.1  # Bonus for successful auto-fix
            else:
                warnings.extend(lint_result[2])
                confidence -= 0.2

            # Step 3: SQLite-specific validation
            sqlite_valid = self._validate_sqlite_specific(fixed_sql or sql)
            if not sqlite_valid[0]:
                errors.extend(sqlite_valid[1])
                confidence -= 0.3

            # Step 4: Schema validation
            if self.schema_info:
                schema_valid = self._validate_schema_compliance(fixed_sql or sql)
                if not schema_valid[0]:
                    errors.extend(schema_valid[1])
                    confidence -= 0.3

            # Step 5: Security validation with guard-rails
            security_valid = self._validate_security_with_guardrails(fixed_sql or sql)
            if not security_valid[0]:
                errors.extend(security_valid[1])
                confidence -= 0.5  # Security violations are critical

            # Step 6: Auto-repair with guard-rails (Sprint 5)
            if errors and confidence < self.repair_confidence_threshold:
                repair_result = self._attempt_auto_repair(sql, errors)
                if repair_result[0]:
                    fixed_sql = repair_result[1]
                    warnings.append("SQL was auto-repaired")
                    confidence += 0.2
                    # Re-validate the repaired SQL
                    repair_validation = self.validate_sql(fixed_sql)
                    if repair_validation.is_valid:
                        errors = repair_validation.errors
                        warnings.extend(repair_validation.warnings)
                        confidence = max(confidence, repair_validation.confidence)

            is_valid = len(errors) == 0
            confidence = max(0.0, min(1.0, confidence))

            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                fixed_sql=fixed_sql,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation failed: {str(e)}"],
                warnings=[],
                confidence=0.0,
            )

    def _validate_syntax(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Parse via sqlglot (dialect=sqlite); if parse fails, return errors.
        """
        try:
            parsed = sqlglot.parse_one(sql, dialect="sqlite")
            if parsed is None:
                return False, ["SQL could not be parsed"]
            return True, []
        except Exception as e:
            return False, [f"Syntax error: {str(e)}"]

    def _lint_and_fix(self, sql: str) -> Tuple[bool, Optional[str], List[str]]:
        """
        Lint with sqlfluff; auto-fix style errors using sqlfluff fix.
        Sprint 5: Enhanced with sqlfluff fix integration.
        """
        try:
            # Lint the SQL
            result = self.linter.lint_string(sql)

            warnings = []
            for violation in result.violations:
                warnings.append(f"Style: {violation.description}")

            # Sprint 5: Use sqlfluff fix for auto-repair
            fixed_sql = None
            if warnings:
                try:
                    # Use sqlfluff fix to automatically fix style issues
                    fix_result = fix.fix_string(sql, dialect="sqlite")
                    if fix_result and len(fix_result) > 0:
                        fixed_sql = fix_result[0]
                        logger.info("Applied sqlfluff auto-fixes")
                    else:
                        # Fallback to manual fixes if sqlfluff fix fails
                        fixed_sql = self._manual_style_fix(sql, warnings)
                except Exception as e:
                    logger.warning(
                        f"sqlfluff fix failed: {e}, falling back to manual fixes"
                    )
                    fixed_sql = self._manual_style_fix(sql, warnings)

            return True, fixed_sql, warnings

        except Exception as e:
            return False, None, [f"Linting error: {str(e)}"]

    def _manual_style_fix(self, sql: str, warnings: List[str]) -> str:
        """
        Manual style fixes as fallback when sqlfluff fix fails.
        """
        fixed_sql = sql

        # Fix common style issues
        for warning in warnings:
            if "L010" in warning:  # Keywords should be consistently upper case
                # Convert keywords to uppercase
                keywords = [
                    "SELECT",
                    "FROM",
                    "WHERE",
                    "JOIN",
                    "ON",
                    "GROUP BY",
                    "ORDER BY",
                    "HAVING",
                    "LIMIT",
                    "OFFSET",
                ]
                for keyword in keywords:
                    fixed_sql = re.sub(
                        rf"\b{keyword}\b", keyword, fixed_sql, flags=re.IGNORECASE
                    )

            elif "L030" in warning:  # Inconsistent indentation
                # Basic indentation fix
                lines = fixed_sql.split("\n")
                fixed_lines = []
                indent_level = 0
                for line in lines:
                    stripped = line.strip()
                    if stripped:
                        if stripped.upper().startswith(
                            ("SELECT", "FROM", "WHERE", "JOIN", "ON")
                        ):
                            indent_level = 0
                        elif stripped.upper().startswith(("AND", "OR")):
                            indent_level = 1
                        fixed_lines.append("  " * indent_level + stripped)
                    else:
                        fixed_lines.append("")
                fixed_sql = "\n".join(fixed_lines)

        return fixed_sql

    def _validate_sqlite_specific(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Validate SQLite-specific requirements.
        """
        errors = []
        sql_upper = sql.upper()

        # Check for non-SQLite functions
        non_sqlite_functions = ["YEAR(", "MONTH(", "DAY(", "DATE_FORMAT(", "IFNULL("]
        for func in non_sqlite_functions:
            if func in sql_upper:
                errors.append(f"Non-SQLite function: {func}")

        # Check for MySQL-specific syntax
        if "LIMIT" in sql_upper and "OFFSET" in sql_upper:
            # This is actually valid SQLite, but check for MySQL-style LIMIT
            if re.search(r"LIMIT\s+\d+\s*,\s*\d+", sql_upper):
                errors.append("MySQL-style LIMIT syntax detected")

        return len(errors) == 0, errors

    def _validate_schema_compliance(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Validate schema compliance if schema info is available.
        """
        if not self.schema_info:
            return True, []

        errors = []
        sql_upper = sql.upper()

        # Extract table names from SQL
        table_pattern = r"\bFROM\s+(\w+)"
        tables = re.findall(table_pattern, sql_upper)

        # Check if tables exist in schema
        for table in tables:
            if table.lower() not in [t.lower() for t in self.schema_info.keys()]:
                errors.append(f"Table '{table}' not found in schema")

        return len(errors) == 0, errors

    def _validate_security_with_guardrails(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Security validation with Sprint 5 guard-rails.
        """
        errors = []
        sql_upper = sql.upper()

        # Check for dangerous patterns with case-insensitive regex
        for pattern in self.dangerous_patterns:
            if re.search(pattern, sql_upper):
                errors.append(
                    f"Security violation: {pattern.strip()} operations are not allowed"
                )

        # Check for multiple statements
        if ";" in sql and sql.count(";") > 1:
            errors.append("Multiple SQL statements are not allowed")

        # Check for comments that might contain SQL injection
        if "/*" in sql or "--" in sql:
            errors.append("SQL comments are not allowed for security reasons")

        return len(errors) == 0, errors

    def _attempt_auto_repair(
        self, sql: str, errors: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Attempt auto-repair with Sprint 5 guard-rails.
        """
        try:
            # Only attempt repair for syntax errors, not security violations
            syntax_errors = [
                e for e in errors if "syntax" in e.lower() or "parse" in e.lower()
            ]
            if not syntax_errors:
                return False, None

            # Use sqlfluff fix for syntax repair
            fix_result = fix.fix_string(sql, dialect="sqlite")
            if fix_result and len(fix_result) > 0:
                repaired_sql = fix_result[0]

                # Validate the repaired SQL
                validation = self._validate_syntax(repaired_sql)
                if validation[0]:
                    return True, repaired_sql

            return False, None

        except Exception as e:
            logger.warning(f"Auto-repair failed: {e}")
            return False, None

    async def validate_and_fix_async(
        self, sql: str, max_attempts: int = 3
    ) -> Tuple[str, ValidationResult]:
        """
        Async version of validate and fix with Sprint 5 enhancements.
        """
        current_sql = sql
        attempt = 0

        while attempt < max_attempts:
            result = self.validate_sql(current_sql)

            if result.is_valid:
                return current_sql, result

            if result.fixed_sql and result.fixed_sql != current_sql:
                current_sql = result.fixed_sql
                attempt += 1
            else:
                break

        return current_sql, result


class SQLSandbox:
    """
    Provides a safe environment for testing SQL execution.
    """

    def __init__(self, original_db_path: str):
        self.original_db_path = original_db_path

    def test_execution(self, sql: str) -> Tuple[bool, str]:
        """
        Execute in sandbox; capture exceptions; feed back.
        """
        temp_db_path = None

        try:
            # Create temporary copy of database
            temp_db_path = self._create_temp_db()

            # Test execution
            import sqlite3

            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()

            # Set timeout and limits
            conn.execute("PRAGMA busy_timeout = 5000")

            # Execute query with limits
            cursor.execute(sql)

            # Fetch a few rows to test
            rows = cursor.fetchmany(10)

            conn.close()

            return True, f"Execution successful, returned {len(rows)} rows"

        except Exception as e:
            return False, f"Execution failed: {str(e)}"

        finally:
            # Clean up temporary database
            if temp_db_path and os.path.exists(temp_db_path):
                try:
                    os.remove(temp_db_path)
                except:
                    pass

    def _create_temp_db(self) -> str:
        """
        Create a temporary copy of the database.
        """
        temp_fd, temp_path = tempfile.mkstemp(suffix=".db")
        os.close(temp_fd)

        shutil.copy2(self.original_db_path, temp_path)
        return temp_path
