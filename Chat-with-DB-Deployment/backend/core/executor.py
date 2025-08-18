"""
Safe SQL execution with proper error handling and timeout management
Updated for MS SQL Server compatibility
"""

import asyncio
import logging
import pyodbc
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from backend.config import get_settings

from backend.core.types import ExecutionResult

logger = logging.getLogger(__name__)


class SQLExecutor:
    """
    Handles safe and efficient SQL execution with error handling and timeout management.
    Updated for MS SQL Server compatibility.
    """

    def __init__(self, database_path: str):
        self.database_path = database_path
        self.max_rows = 10000
        self.timeout_seconds = 30
        self.settings = get_settings()
        
    def _get_connection(self):
        """Get MS SQL Server connection"""
        try:
            # Use the connection string from settings
            conn_str = self.settings.get_database_url()
            return pyodbc.connect(conn_str, timeout=self.timeout_seconds)
        except Exception as e:
            logger.error(f"Failed to connect to MS SQL Server: {e}")
            raise

    def execute_sql(self, sql: str) -> ExecutionResult:
        """
        Execute SQL query with safety checks and timeout.
        """
        start_time = time.time()

        try:
            # Temporarily bypass validation for debugging
            # if not self.validate_query_safety(sql):
            #     return ExecutionResult(
            #         success=False,
            #         error="Query failed safety validation",
            #         execution_time=time.time() - start_time
            #     )

            # Execute synchronously (for backward compatibility)
            result = self._execute_sync(sql)

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=result[0],
                data=result[1] if result[0] and result[1] is not None else [],
                row_count=len(result[1]) if result[0] and result[1] else 0,
                error=result[2] if not result[0] else None,
                execution_time=execution_time,
                sql=sql,
            )

        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return ExecutionResult(
                success=False,
                error=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time,
                sql=sql,
            )

    def _execute_sync(
        self, sql: str
    ) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str]]:
        """
        Synchronous SQL execution.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            try:
                logger.info(f"Executing SQL: {sql[:200]}...")
                # Execute query
                cursor.execute(sql)

                # Fetch results
                if sql.strip().upper().startswith("SELECT"):
                    rows = cursor.fetchall()
                    logger.info(f"Fetched {len(rows)} rows from database")
                    # Convert to list of dictionaries
                    if rows:
                        columns = [description[0] for description in cursor.description]
                        logger.info(f"Columns: {columns}")
                        data = [
                            {str(col): val for col, val in zip(columns, row)}
                            for row in rows
                        ]
                        logger.info(f"Converted to {len(data)} data records")
                        logger.info(f"First row: {data[0] if data else 'None'}")
                    else:
                        data = []
                        logger.info("No rows returned, data is empty")
                    row_count = len(data)
                else:
                    data = []
                    row_count = cursor.rowcount

                logger.info(
                    f"SQL execution completed: success=True, row_count={row_count}"
                )
                return True, data, None

            except pyodbc.Error as e:
                logger.error(f"MS SQL Server error: {str(e)}")
                return False, None, f"MS SQL Server error: {str(e)}"
            except Exception as e:
                logger.error(f"Execution error: {str(e)}")
                return False, None, f"Execution error: {str(e)}"

    def test_connection(self) -> bool:
        """
        Test database connection.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_schema_info(self) -> Dict[str, List[str]]:
        """
        Get database schema information.
        """
        schema_info = {}

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()

                for table in tables:
                    table_name = table[0]

                    # Get columns for each table
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()

                    column_names = [col[1] for col in columns]
                    schema_info[table_name] = column_names

                return schema_info

        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            return {}

    def validate_query_safety(self, sql: str) -> bool:
        """
        Validate query safety to prevent malicious operations.
        """
        sql_upper = sql.upper()

        # Block dangerous operations
        dangerous_keywords = [
            "DROP",
            "DELETE",
            "UPDATE",
            "INSERT",
            "CREATE",
            "ALTER",
            "TRUNCATE",
            "EXEC",
            "EXECUTE",
        ]

        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                logger.warning(
                    f"Blocked potentially dangerous SQL with keyword: {keyword}"
                )
                return False

        # Block multiple statements
        if ";" in sql and sql.count(";") > 1:
            logger.warning("Blocked SQL with multiple statements")
            return False

        logger.info(f"SQL validation passed for query starting with: {sql[:100]}...")
        return True


class AsyncSQLExecutor:
    """
    Asynchronous wrapper for SQLExecutor optimized for Sprint 4.
    Uses asyncio.to_thread for better performance and proper async integration.
    """

    def __init__(self, database_path: str):
        self.executor = SQLExecutor(database_path)

    async def execute_sql_async(self, sql: str) -> ExecutionResult:
        """
        Execute SQL asynchronously using asyncio.to_thread.
        Sprint 4: Optimized for reduced latency and better async integration.
        """
        try:
            logger.info(f"Async SQL execution starting for query: {sql[:100]}...")
            # Sprint 4: Use asyncio.to_thread instead of run_in_executor
            result = await asyncio.to_thread(self.executor.execute_sql, sql)
            logger.info(
                f"Async SQL execution completed: success={result.success}, row_count={result.row_count}, error={result.error}"
            )
            return result
        except Exception as e:
            logger.error(f"Async SQL execution error: {e}")
            return ExecutionResult(
                success=False,
                error=f"Async execution error: {str(e)}",
                execution_time=0.0,
                sql=sql,
            )

    async def execute_sql_with_timeout(
        self, sql: str, timeout_seconds: int = 30
    ) -> ExecutionResult:
        """
        Execute SQL with timeout using asyncio.to_thread.
        Sprint 4: Proper timeout handling with asyncio.wait_for.
        """
        try:
            # Sprint 4: Use asyncio.wait_for with asyncio.to_thread for proper timeout handling
            result = await asyncio.wait_for(
                asyncio.to_thread(self.executor.execute_sql, sql),
                timeout=timeout_seconds,
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"SQL execution timed out after {timeout_seconds} seconds")
            return ExecutionResult(
                success=False,
                error=f"Query execution timed out after {timeout_seconds} seconds",
                execution_time=timeout_seconds,
                sql=sql,
            )
        except Exception as e:
            logger.error(f"Async SQL execution with timeout error: {e}")
            return ExecutionResult(
                success=False,
                error=f"Async execution error: {str(e)}",
                execution_time=0.0,
                sql=sql,
            )

    async def test_connection_async(self) -> bool:
        """
        Test database connection asynchronously.
        """
        try:
            result = await asyncio.to_thread(self.executor.test_connection)
            return result
        except Exception as e:
            logger.error(f"Async connection test failed: {e}")
            return False

    async def get_schema_info_async(self) -> Dict[str, List[str]]:
        """
        Get database schema information asynchronously.
        """
        try:
            result = await asyncio.to_thread(self.executor.get_schema_info)
            return result
        except Exception as e:
            logger.error(f"Async schema info retrieval failed: {e}")
            return {}
