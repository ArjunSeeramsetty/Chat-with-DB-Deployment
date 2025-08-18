"""
Multi-Layer Validation and Self-Correction System
Implements 4-layer validation: syntax → business rules → dry run → result reasonableness
Phase 4.2: Multi-Layer Validation and Self-Correction
"""

import asyncio
import logging
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import sqlglot
import yaml
from sqlglot import parse
from sqlglot.errors import ParseError

from backend.core.types import ValidationResult
from backend.core.validator import EnhancedSQLValidator

logger = logging.getLogger(__name__)


@dataclass
class ValidationLayer:
    """Represents a validation layer result"""
    name: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    confidence: float
    execution_time: float
    metadata: Dict[str, Any] = None


@dataclass
class MultiLayerValidationResult:
    """Result of multi-layer validation"""
    overall_valid: bool
    layers: Dict[str, ValidationLayer]
    confidence: float
    fixed_sql: Optional[str] = None
    correction_attempts: int = 0
    total_execution_time: float = 0.0


class MultiLayerValidator:
    """
    Multi-layer validation system with self-correction capabilities
    Implements 4-layer validation: syntax → business rules → dry run → result reasonableness
    """
    
    def __init__(self, schema_info: Optional[dict] = None, db_path: Optional[str] = None):
        self.schema_info = schema_info or {}
        self.db_path = db_path
        self.enhanced_validator = EnhancedSQLValidator(schema_info)
        
        # Load business rules
        self.business_rules = self._load_business_rules()
        
        # Validation thresholds
        self.confidence_thresholds = {
            "syntax": 0.8,
            "business_rules": 0.7,
            "dry_run": 0.6,
            "reasonableness": 0.5
        }
        
        # Self-correction settings
        self.max_correction_attempts = 3
        self.correction_confidence_threshold = 0.6
        
    def _load_business_rules(self) -> Dict[str, Any]:
        """Load business rules from configuration"""
        try:
            config_path = Path("config/business_rules.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning("Business rules file not found, using defaults")
                return self._get_default_business_rules()
        except Exception as e:
            logger.error(f"Failed to load business rules: {e}")
            return self._get_default_business_rules()
    
    def _get_default_business_rules(self) -> Dict[str, Any]:
        """Get default business rules"""
        return {
            "validation": {
                "sqlite_compatibility": True,
                "security_checks": True,
                "schema_validation": True,
                "max_query_length": 1000
            },
            "schema_mapping": {
                "energy_columns": {
                    "FactAllIndiaDailySummary": {
                        "EnergyMet": "energy",
                        "EnergyShortage": "energy"
                    }
                },
                "power_columns": {
                    "FactAllIndiaDailySummary": {
                        "MaxDemandSCADA": "power"
                    }
                }
            }
        }
    
    async def validate_multi_layer(self, sql: str, context: Optional[Dict[str, Any]] = None) -> MultiLayerValidationResult:
        """
        Perform multi-layer validation with self-correction
        """
        start_time = asyncio.get_event_loop().time()
        layers = {}
        correction_attempts = 0
        current_sql = sql
        
        # Layer 1: Syntax Validation
        syntax_layer = await self._validate_syntax_layer(current_sql)
        layers["syntax"] = syntax_layer
        
        if not syntax_layer.is_valid:
            # Attempt syntax correction
            corrected_sql = await self._attempt_syntax_correction(current_sql, syntax_layer.errors)
            if corrected_sql:
                current_sql = corrected_sql
                correction_attempts += 1
                # Re-validate syntax
                syntax_layer = await self._validate_syntax_layer(current_sql)
                layers["syntax"] = syntax_layer
        
        # Layer 2: Business Rules Validation
        business_layer = await self._validate_business_rules_layer(current_sql, context)
        layers["business_rules"] = business_layer
        
        if not business_layer.is_valid:
            # Attempt business rules correction
            corrected_sql = await self._attempt_business_rules_correction(current_sql, business_layer.errors)
            if corrected_sql:
                current_sql = corrected_sql
                correction_attempts += 1
                # Re-validate business rules
                business_layer = await self._validate_business_rules_layer(current_sql, context)
                layers["business_rules"] = business_layer
        
        # Layer 3: Dry Run Validation
        dry_run_layer = await self._validate_dry_run_layer(current_sql)
        layers["dry_run"] = dry_run_layer
        
        if not dry_run_layer.is_valid:
            # Attempt dry run correction
            corrected_sql = await self._attempt_dry_run_correction(current_sql, dry_run_layer.errors)
            if corrected_sql:
                current_sql = corrected_sql
                correction_attempts += 1
                # Re-validate dry run
                dry_run_layer = await self._validate_dry_run_layer(current_sql)
                layers["dry_run"] = dry_run_layer
        
        # Layer 4: Result Reasonableness Validation
        reasonableness_layer = await self._validate_reasonableness_layer(current_sql, context)
        layers["reasonableness"] = reasonableness_layer
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(layers)
        
        # Determine overall validity
        overall_valid = all(layer.is_valid for layer in layers.values())
        
        total_execution_time = asyncio.get_event_loop().time() - start_time
        
        return MultiLayerValidationResult(
            overall_valid=overall_valid,
            layers=layers,
            confidence=overall_confidence,
            fixed_sql=current_sql if current_sql != sql else None,
            correction_attempts=correction_attempts,
            total_execution_time=total_execution_time
        )
    
    async def _validate_syntax_layer(self, sql: str) -> ValidationLayer:
        """Layer 1: Syntax Validation"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Use existing enhanced validator for syntax validation
            result = self.enhanced_validator.validate_sql(sql)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return ValidationLayer(
                name="syntax",
                is_valid=result.is_valid,
                errors=result.errors,
                warnings=result.warnings,
                confidence=result.confidence,
                execution_time=execution_time,
                metadata={"parse_tree": "available" if result.is_valid else None}
            )
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return ValidationLayer(
                name="syntax",
                is_valid=False,
                errors=[f"Syntax validation failed: {str(e)}"],
                warnings=[],
                confidence=0.0,
                execution_time=execution_time
            )
    
    async def _validate_business_rules_layer(self, sql: str, context: Optional[Dict[str, Any]] = None) -> ValidationLayer:
        """Layer 2: Business Rules Validation"""
        start_time = asyncio.get_event_loop().time()
        errors = []
        warnings = []
        confidence = 1.0
        
        try:
            # Parse SQL to extract components
            parsed = parse(sql, dialect="sqlite")
            if not parsed:
                errors.append("Failed to parse SQL for business rules validation")
                confidence = 0.0
            else:
                # Check column usage against business rules
                column_errors = self._validate_column_usage(parsed)
                errors.extend(column_errors)
                
                # Check aggregation functions
                aggregation_errors = self._validate_aggregation_functions(parsed)
                errors.extend(aggregation_errors)
                
                # Check table relationships
                relationship_errors = self._validate_table_relationships(parsed)
                errors.extend(relationship_errors)
                
                # Check query complexity
                complexity_warnings = self._validate_query_complexity(parsed)
                warnings.extend(complexity_warnings)
                
                # Adjust confidence based on errors
                if errors:
                    confidence -= 0.2 * len(errors)
                if warnings:
                    confidence -= 0.05 * len(warnings)
                
                confidence = max(0.0, min(1.0, confidence))
        
        except Exception as e:
            errors.append(f"Business rules validation failed: {str(e)}")
            confidence = 0.0
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return ValidationLayer(
            name="business_rules",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=confidence,
            execution_time=execution_time
        )
    
    async def _validate_dry_run_layer(self, sql: str) -> ValidationLayer:
        """Layer 3: Dry Run Validation"""
        start_time = asyncio.get_event_loop().time()
        errors = []
        warnings = []
        confidence = 1.0
        
        try:
            # Create temporary database for dry run
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
                temp_db_path = temp_db.name
            
            # Copy schema to temporary database
            if self.db_path and Path(self.db_path).exists():
                # Copy schema structure (not data)
                schema_sql = self._extract_schema_sql()
                if schema_sql:
                    conn = sqlite3.connect(temp_db_path)
                    try:
                        # Execute schema creation
                        for statement in schema_sql.split(';'):
                            if statement.strip():
                                conn.execute(statement)
                        conn.commit()
                        
                        # Try to execute the query (EXPLAIN to avoid actual execution)
                        explain_sql = f"EXPLAIN {sql}"
                        cursor = conn.execute(explain_sql)
                        cursor.fetchall()  # Execute but don't fetch results
                        
                        # If we get here, the query is valid for dry run
                        confidence = 1.0
                        
                    except sqlite3.Error as e:
                        errors.append(f"Dry run failed: {str(e)}")
                        confidence = 0.0
                    finally:
                        conn.close()
                else:
                    warnings.append("Could not extract schema for dry run validation")
                    confidence = 0.8
            else:
                warnings.append("No database path provided for dry run validation")
                confidence = 0.7
            
            # Clean up temporary database
            try:
                Path(temp_db_path).unlink()
            except:
                pass
                
        except Exception as e:
            errors.append(f"Dry run validation failed: {str(e)}")
            confidence = 0.0
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return ValidationLayer(
            name="dry_run",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=confidence,
            execution_time=execution_time
        )
    
    async def _validate_reasonableness_layer(self, sql: str, context: Optional[Dict[str, Any]] = None) -> ValidationLayer:
        """Layer 4: Result Reasonableness Validation"""
        start_time = asyncio.get_event_loop().time()
        errors = []
        warnings = []
        confidence = 1.0
        
        try:
            # Check for reasonable query patterns
            reasonableness_checks = self._perform_reasonableness_checks(sql, context)
            
            for check_name, (is_reasonable, message) in reasonableness_checks.items():
                if not is_reasonable:
                    if check_name in ["critical", "high"]:
                        errors.append(f"Reasonableness check failed: {message}")
                        confidence -= 0.3
                    else:
                        warnings.append(f"Reasonableness warning: {message}")
                        confidence -= 0.1
            
            confidence = max(0.0, min(1.0, confidence))
            
        except Exception as e:
            errors.append(f"Reasonableness validation failed: {str(e)}")
            confidence = 0.0
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return ValidationLayer(
            name="reasonableness",
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence=confidence,
            execution_time=execution_time
        )
    
    def _validate_column_usage(self, parsed) -> List[str]:
        """Validate column usage against business rules"""
        errors = []
        
        # Extract columns from parsed SQL
        columns = self._extract_columns(parsed)
        
        # Check against business rules
        schema_mapping = self.business_rules.get("schema_mapping", {})
        
        # Common foreign key and metadata columns that are allowed
        allowed_foreign_keys = {
            'RegionID', 'StateID', 'DateID', 'CountryID', 'SourceID', 'MechanismID',
            'RegionName', 'StateName', 'CountryName', 'SourceName', 'MechanismName'
        }
        
        for column in columns:
            # Check if column exists in schema mapping
            found = False
            for table_type, tables in schema_mapping.items():
                for table_name, table_columns in tables.items():
                    if column in table_columns:
                        found = True
                        break
                if found:
                    break
            
            # If not found in business rules, check if it's an allowed foreign key
            if not found and column in allowed_foreign_keys:
                found = True  # Allow foreign key columns
            
            if not found:
                errors.append(f"Column '{column}' not found in business rules schema")
        
        return errors
    
    def _validate_aggregation_functions(self, parsed) -> List[str]:
        """Validate aggregation functions against business rules"""
        errors = []
        
        # Extract aggregation functions
        aggregations = self._extract_aggregations(parsed)
        
        # Check aggregation function usage
        for agg_func, column in aggregations:
            # Check if aggregation function is appropriate for column type
            column_type = self._get_column_type(column)
            if column_type:
                if not self._is_valid_aggregation(agg_func, column_type):
                    errors.append(f"Invalid aggregation '{agg_func}' for column '{column}' of type '{column_type}'")
        
        return errors
    
    def _validate_table_relationships(self, parsed) -> List[str]:
        """Validate table relationships"""
        errors = []
        
        # Extract tables
        tables = self._extract_tables(parsed)
        
        # Check for required joins
        if len(tables) > 1:
            joins = self._extract_joins(parsed)
            if not joins:
                errors.append("Multiple tables referenced but no JOIN clauses found")
        
        return errors
    
    def _validate_query_complexity(self, parsed) -> List[str]:
        """Validate query complexity"""
        warnings = []
        
        # Check for complex patterns
        sql_str = str(parsed)
        
        # Check for subqueries
        if "SELECT" in sql_str.upper() and sql_str.upper().count("SELECT") > 1:
            warnings.append("Query contains subqueries - may impact performance")
        
        # Check for multiple aggregations
        agg_count = len(re.findall(r'\b(SUM|AVG|COUNT|MAX|MIN)\b', sql_str.upper()))
        if agg_count > 3:
            warnings.append("Query contains multiple aggregations - may impact performance")
        
        return warnings
    
    def _perform_reasonableness_checks(self, sql: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Tuple[bool, str]]:
        """Perform reasonableness checks on the query"""
        checks = {}
        
        # Check query length
        if len(sql) > 1000:
            checks["query_length"] = (False, "Query is too long (>1000 characters)")
        else:
            checks["query_length"] = (True, "Query length is reasonable")
        
        # Check for dangerous patterns
        dangerous_patterns = [
            (r"\bDROP\b", "DROP operations"),
            (r"\bDELETE\b", "DELETE operations"),
            (r"\bUPDATE\b", "UPDATE operations"),
            (r"\bINSERT\b", "INSERT operations"),
            (r"\bALTER\b", "ALTER operations"),
            (r"\bCREATE\b", "CREATE operations"),
        ]
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                pattern_name = pattern.lower().replace('\\b', '').replace('\\', '')
                checks[f"dangerous_{pattern_name}"] = (False, f"Query contains {description}")
            else:
                pattern_name = pattern.lower().replace('\\b', '').replace('\\', '')
                checks[f"dangerous_{pattern_name}"] = (True, f"No {description} found")
        
        # Check for required components
        if "SELECT" not in sql.upper():
            checks["select_clause"] = (False, "Query must contain SELECT clause")
        else:
            checks["select_clause"] = (True, "SELECT clause present")
        
        if "FROM" not in sql.upper():
            checks["from_clause"] = (False, "Query must contain FROM clause")
        else:
            checks["from_clause"] = (True, "FROM clause present")
        
        return checks
    
    async def _attempt_syntax_correction(self, sql: str, errors: List[str]) -> Optional[str]:
        """Attempt to correct syntax errors"""
        try:
            # Use existing enhanced validator's auto-repair
            result = self.enhanced_validator.validate_sql(sql)
            if result.fixed_sql:
                return result.fixed_sql
        except Exception as e:
            logger.warning(f"Syntax correction failed: {e}")
        
        return None
    
    async def _attempt_business_rules_correction(self, sql: str, errors: List[str]) -> Optional[str]:
        """Attempt to correct business rules violations"""
        try:
            # Simple corrections based on error patterns
            corrected_sql = sql
            
            for error in errors:
                if "Invalid aggregation" in error:
                    # Try to fix aggregation function
                    corrected_sql = self._fix_aggregation_function(corrected_sql, error)
                elif "Column not found" in error:
                    # Try to fix column name
                    corrected_sql = self._fix_column_name(corrected_sql, error)
            
            return corrected_sql if corrected_sql != sql else None
        except Exception as e:
            logger.warning(f"Business rules correction failed: {e}")
            return None
    
    async def _attempt_dry_run_correction(self, sql: str, errors: List[str]) -> Optional[str]:
        """Attempt to correct dry run errors"""
        try:
            # Simple corrections for common dry run issues
            corrected_sql = sql
            
            for error in errors:
                if "no such table" in error.lower():
                    # Try to fix table name
                    corrected_sql = self._fix_table_name(corrected_sql, error)
                elif "no such column" in error.lower():
                    # Try to fix column name
                    corrected_sql = self._fix_column_name(corrected_sql, error)
            
            return corrected_sql if corrected_sql != sql else None
        except Exception as e:
            logger.warning(f"Dry run correction failed: {e}")
            return None
    
    def _calculate_overall_confidence(self, layers: Dict[str, ValidationLayer]) -> float:
        """Calculate overall confidence from all layers"""
        if not layers:
            return 0.0
        
        # Weighted average based on layer importance
        weights = {
            "syntax": 0.3,
            "business_rules": 0.3,
            "dry_run": 0.2,
            "reasonableness": 0.2
        }
        
        total_confidence = 0.0
        total_weight = 0.0
        
        for layer_name, layer in layers.items():
            weight = weights.get(layer_name, 0.1)
            total_confidence += layer.confidence * weight
            total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.0
    
    def _extract_schema_sql(self) -> Optional[str]:
        """Extract schema SQL from the database"""
        if not self.db_path or not Path(self.db_path).exists():
            return None
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'")
            schema_statements = []
            
            for row in cursor.fetchall():
                if row[0]:
                    schema_statements.append(row[0])
            
            conn.close()
            return '; '.join(schema_statements)
        except Exception as e:
            logger.error(f"Failed to extract schema: {e}")
            return None
    
    def _extract_columns(self, parsed) -> List[str]:
        """Extract column names from parsed SQL"""
        columns = []
        sql_str = str(parsed)
        
        # Extract column names using more sophisticated regex patterns
        # Pattern 1: Column names in SELECT clause (with or without table alias)
        # This pattern looks for: table.column or just column, but excludes aliases
        select_pattern = r'\b(?:SELECT|,)\s+(?:DISTINCT\s+)?(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)(?:\s+AS\s+[A-Za-z_][A-Za-z0-9_]*)?\b'
        select_matches = re.findall(select_pattern, sql_str, re.IGNORECASE)
        columns.extend(select_matches)
        
        # Pattern 2: Column names in WHERE clause (only actual columns, not aliases)
        where_pattern = r'\bWHERE\s+.*?(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)\s*[=<>!]'
        where_matches = re.findall(where_pattern, sql_str, re.IGNORECASE)
        columns.extend(where_matches)
        
        # Pattern 3: Column names in GROUP BY clause
        group_pattern = r'\bGROUP\s+BY\s+.*?(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)\b'
        group_matches = re.findall(group_pattern, sql_str, re.IGNORECASE)
        columns.extend(group_matches)
        
        # Pattern 4: Column names in ORDER BY clause
        order_pattern = r'\bORDER\s+BY\s+.*?(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)\b'
        order_matches = re.findall(order_pattern, sql_str, re.IGNORECASE)
        columns.extend(order_matches)
        
        # Pattern 5: Column names in JOIN conditions
        join_pattern = r'\bJOIN\s+.*?\bON\s+.*?(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)\s*[=<>!]'
        join_matches = re.findall(join_pattern, sql_str, re.IGNORECASE)
        columns.extend(join_matches)
        
        # Filter out SQL keywords, aliases, and common non-column words
        sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'ORDER', 'HAVING', 'JOIN', 'ON', 'AND', 'OR', 'NOT',
            'IN', 'EXISTS', 'BETWEEN', 'LIKE', 'IS', 'NULL', 'DISTINCT', 'AS', 'ASC', 'DESC', 'LIMIT',
            'OFFSET', 'UNION', 'ALL', 'INTERSECT', 'EXCEPT', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
            'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'COALESCE', 'NULLIF', 'CAST', 'CONVERT'
        }
        
        # Common table aliases and non-column identifiers
        non_columns = {
            'fs', 'r', 'd', 's', 't', 'c', 'e', 'g', 'b', 'f',  # Common table aliases
            'quoted', 'Column', 'naked', 'identifier', 'function', 'name', 'bracketed', 'start', 'end',
            'expression', 'reference', 'dot', 'whitespace', 'comma', 'keyword', 'alias'
        }
        
        # Filter out keywords, aliases, and duplicates
        filtered_columns = []
        for col in columns:
            # Skip if it's a SQL keyword
            if col.upper() in sql_keywords:
                continue
            # Skip if it's a common non-column identifier
            if col.lower() in non_columns:
                continue
            # Skip if it's too short or numeric
            if len(col) <= 1 or col.isdigit():
                continue
            # Skip if it's already in the list
            if col in filtered_columns:
                continue
            # Skip if it looks like an alias (starts with common alias patterns)
            if col.lower().startswith(('avg', 'sum', 'count', 'max', 'min', 'total', 'average')):
                continue
            
            filtered_columns.append(col)
        
        return filtered_columns
    
    def _extract_aggregations(self, parsed) -> List[Tuple[str, str]]:
        """Extract aggregation functions from parsed SQL"""
        aggregations = []
        sql_str = str(parsed)
        # Extract aggregation functions using regex
        agg_matches = re.findall(r'\b(SUM|AVG|COUNT|MAX|MIN)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', sql_str, re.IGNORECASE)
        return [(agg.upper(), col) for agg, col in agg_matches]
    
    def _extract_tables(self, parsed) -> List[str]:
        """Extract table names from parsed SQL"""
        tables = []
        sql_str = str(parsed)
        # Extract table names using regex (simplified)
        table_matches = re.findall(r'\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)\b', sql_str, re.IGNORECASE)
        return list(set(table_matches))
    
    def _extract_joins(self, parsed) -> List[str]:
        """Extract JOIN clauses from parsed SQL"""
        joins = []
        sql_str = str(parsed)
        # Extract JOIN clauses using regex
        join_matches = re.findall(r'\bJOIN\s+[A-Za-z_][A-Za-z0-9_]*\b', sql_str, re.IGNORECASE)
        return join_matches
    
    def _get_column_type(self, column: str) -> Optional[str]:
        """Get column type from business rules"""
        schema_mapping = self.business_rules.get("schema_mapping", {})
        
        for table_type, tables in schema_mapping.items():
            for table_name, table_columns in tables.items():
                if column in table_columns:
                    return table_columns[column]
        
        return None
    
    def _is_valid_aggregation(self, agg_func: str, column_type: str) -> bool:
        """Check if aggregation function is valid for column type"""
        valid_aggregations = {
            "energy": ["SUM", "AVG", "MAX", "MIN"],
            "power": ["MAX", "MIN", "AVG"],
            "ratio": ["AVG", "MAX", "MIN"],
            "percentage": ["AVG", "MAX", "MIN"],
            "time": ["SUM", "MAX", "MIN", "AVG"],
            "outage": ["SUM", "MAX", "MIN", "AVG"]
        }
        
        return agg_func.upper() in valid_aggregations.get(column_type, [])
    
    def _fix_aggregation_function(self, sql: str, error: str) -> str:
        """Fix aggregation function based on error"""
        # Simple fix - replace with AVG if SUM is invalid
        if "SUM" in error and "energy" in error.lower():
            sql = re.sub(r'\bSUM\b', 'AVG', sql, flags=re.IGNORECASE)
        return sql
    
    def _fix_column_name(self, sql: str, error: str) -> str:
        """Fix column name based on error"""
        # Extract column name from error and try to find similar column
        column_match = re.search(r"Column '([^']+)'", error)
        if column_match:
            column_name = column_match.group(1)
            # Try to find similar column in schema
            similar_column = self._find_similar_column(column_name)
            if similar_column:
                sql = re.sub(rf'\b{column_name}\b', similar_column, sql, flags=re.IGNORECASE)
        return sql
    
    def _fix_table_name(self, sql: str, error: str) -> str:
        """Fix table name based on error"""
        # Extract table name from error and try to find similar table
        table_match = re.search(r"table '([^']+)'", error)
        if table_match:
            table_name = table_match.group(1)
            # Try to find similar table in schema
            similar_table = self._find_similar_table(table_name)
            if similar_table:
                sql = re.sub(rf'\b{table_name}\b', similar_table, sql, flags=re.IGNORECASE)
        return sql
    
    def _find_similar_column(self, column_name: str) -> Optional[str]:
        """Find similar column in schema"""
        schema_mapping = self.business_rules.get("schema_mapping", {})
        
        # Simple similarity check
        for table_type, tables in schema_mapping.items():
            for table_name, table_columns in tables.items():
                for col in table_columns.keys():
                    if column_name.lower() in col.lower() or col.lower() in column_name.lower():
                        return col
        
        return None
    
    def _find_similar_table(self, table_name: str) -> Optional[str]:
        """Find similar table in schema"""
        schema_mapping = self.business_rules.get("schema_mapping", {})
        
        # Simple similarity check
        for table_type, tables in schema_mapping.items():
            for table in tables.keys():
                if table_name.lower() in table.lower() or table.lower() in table_name.lower():
                    return table
        
        return None
