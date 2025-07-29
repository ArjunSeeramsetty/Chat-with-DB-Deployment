"""
SQL Validation and Parsing Module
Provides deterministic SQL validation using multiple parsers and schema checking
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import sqlglot
import sqlparse
from sqlglot import parse, exp
from sqlglot.errors import ParseError

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of SQL validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    fixed_sql: Optional[str] = None
    confidence: float = 0.0
    parse_tree: Optional[Any] = None
    schema_violations: List[str] = None

class SQLValidator:
    """Comprehensive SQL validator using multiple parsing approaches"""
    
    def __init__(self, schema_info: Optional[Dict[str, List[str]]] = None):
        self.schema_info = schema_info or {}
        self.sqlite_keywords = {
            'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT',
            'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'OUTER JOIN',
            'UNION', 'UNION ALL', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP',
            'ALTER', 'INDEX', 'TABLE', 'VIEW', 'TRIGGER', 'WITH', 'CTE'
        }
    
    def validate_sql(self, sql: str, dialect: str = "sqlite") -> ValidationResult:
        """
        Comprehensive SQL validation using multiple approaches
        """
        errors = []
        warnings = []
        fixed_sql = None
        parse_tree = None
        schema_violations = []
        
        # Step 1: Basic syntax validation with sqlglot
        try:
            parse_tree = parse(sql, dialect=dialect)
            if not parse_tree or len(parse_tree) == 0:
                errors.append("SQL parsing failed - no valid parse tree generated")
        except ParseError as e:
            errors.append(f"SQL parsing error: {str(e)}")
            # Try to fix common issues
            fixed_sql = self._attempt_sql_fix(sql)
            if fixed_sql:
                try:
                    parse_tree = parse(fixed_sql, dialect=dialect)
                    warnings.append("SQL was automatically fixed")
                except ParseError:
                    errors.append("SQL fix attempt failed")
        
        # Step 2: SQLite-specific validation
        sqlite_errors = self._validate_sqlite_compatibility(sql)
        errors.extend(sqlite_errors)
        
        # Step 3: Schema validation if schema info is available
        if self.schema_info and parse_tree and len(parse_tree) > 0:
            schema_violations = self._validate_schema_compliance(parse_tree[0])  # Use first statement
            if schema_violations:
                errors.extend(schema_violations)
        
        # Step 4: Security validation
        security_errors = self._validate_security(sql)
        errors.extend(security_errors)
        
        # Step 5: Calculate confidence
        confidence = self._calculate_confidence(sql, errors, warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            fixed_sql=fixed_sql,
            confidence=confidence,
            parse_tree=parse_tree,
            schema_violations=schema_violations
        )
    
    def _attempt_sql_fix(self, sql: str) -> Optional[str]:
        """Attempt to fix common SQL syntax issues"""
        fixed = sql
        
        # Fix common issues
        fixed = re.sub(r'\bSELECT\s+FROM\b', 'SELECT * FROM', fixed, flags=re.IGNORECASE)
        fixed = re.sub(r'\bGROUP BY\s+ORDER BY\b', 'GROUP BY 1 ORDER BY', fixed, flags=re.IGNORECASE)
        
        # Add missing semicolon
        if not fixed.strip().endswith(';'):
            fixed = fixed.strip() + ';'
        
        return fixed if fixed != sql else None
    
    def _validate_sqlite_compatibility(self, sql: str) -> List[str]:
        """Validate SQLite-specific compatibility"""
        errors = []
        
        # Check for unsupported SQLite features
        unsupported_patterns = [
            (r'\bFULL OUTER JOIN\b', 'FULL OUTER JOIN is not supported in SQLite'),
            (r'\bRIGHT JOIN\b', 'RIGHT JOIN is not supported in SQLite'),
            (r'\bCROSS APPLY\b', 'CROSS APPLY is not supported in SQLite'),
            (r'\bOUTER APPLY\b', 'OUTER APPLY is not supported in SQLite'),
        ]
        
        for pattern, message in unsupported_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                errors.append(message)
        
        return errors
    
    def _validate_schema_compliance(self, parse_tree) -> List[str]:
        """Validate that all tables and columns exist in schema"""
        violations = []
        
        try:
            # Extract table and column references from parse tree
            tables = set()
            columns = set()
            
            for table in parse_tree.find_all(exp.Table):
                tables.add(table.name.lower())
            
            for column in parse_tree.find_all(exp.Column):
                if column.table:
                    columns.add(f"{column.table}.{column.name}".lower())
                else:
                    columns.add(column.name.lower())
            
            # Check tables exist
            for table in tables:
                if table not in [t.lower() for t in self.schema_info.keys()]:
                    violations.append(f"Table '{table}' not found in schema")
            
            # Check columns exist (basic check)
            for column_ref in columns:
                if '.' in column_ref:
                    table, col = column_ref.split('.', 1)
                    if table in self.schema_info:
                        table_columns = [c.lower() for c in self.schema_info[table]]
                        if col not in table_columns:
                            violations.append(f"Column '{column_ref}' not found in schema")
        
        except Exception as e:
            violations.append(f"Schema validation error: {str(e)}")
        
        return violations
    
    def _validate_security(self, sql: str) -> List[str]:
        """Basic security validation"""
        errors = []
        
        # Check for potentially dangerous operations
        dangerous_patterns = [
            (r'\bDROP\b', 'DROP operations are not allowed'),
            (r'\bDELETE\b', 'DELETE operations are not allowed'),
            (r'\bUPDATE\b', 'UPDATE operations are not allowed'),
            (r'\bINSERT\b', 'INSERT operations are not allowed'),
            (r'\bALTER\b', 'ALTER operations are not allowed'),
            (r'\bCREATE\b', 'CREATE operations are not allowed'),
        ]
        
        for pattern, message in dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                errors.append(message)
        
        return errors
    
    def _calculate_confidence(self, sql: str, errors: List[str], warnings: List[str]) -> float:
        """Calculate confidence score based on validation results"""
        base_confidence = 1.0
        
        # Reduce confidence for each error
        for error in errors:
            if 'parsing error' in error.lower():
                base_confidence -= 0.4
            elif 'schema' in error.lower():
                base_confidence -= 0.3
            elif 'security' in error.lower():
                base_confidence -= 0.5
            else:
                base_confidence -= 0.2
        
        # Reduce confidence for warnings
        for warning in warnings:
            base_confidence -= 0.1
        
        return max(0.0, base_confidence)
    
    def format_sql(self, sql: str) -> str:
        """Format SQL for better readability"""
        try:
            return sqlparse.format(sql, reindent=True, keyword_case='upper')
        except Exception:
            return sql 