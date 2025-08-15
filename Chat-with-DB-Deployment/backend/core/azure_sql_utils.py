"""
Azure SQL Server specific utilities and optimizations
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import text, create_engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.engine import Engine

from backend.config import get_settings
from backend.core.database import get_database_engine

logger = logging.getLogger(__name__)


class AzureSQLUtils:
    """Utility class for Azure SQL Server operations"""
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = None
    
    def get_engine(self) -> Engine:
        """Get the database engine"""
        if self.engine is None:
            self.engine = get_database_engine()
        return self.engine
    
    def test_azure_connection(self) -> Dict[str, Any]:
        """Test Azure SQL connection and return detailed status"""
        try:
            engine = self.get_engine()
            with engine.connect() as conn:
                # Test basic connectivity
                result = conn.execute(text("SELECT @@VERSION, @@SERVERNAME, DB_NAME(), @@SPID"))
                version, server, database, spid = result.fetchone()
                
                # Test Azure SQL specific features - use a more compatible approach
                try:
                    # Use CAST to ensure we get a compatible data type
                    result = conn.execute(text("SELECT CAST(SERVERPROPERTY('EngineEdition') AS INT)"))
                    engine_edition = result.fetchone()[0]
                except Exception as e:
                    logger.warning(f"Failed to get EngineEdition: {e}")
                    # Fallback: check if it's Azure SQL by looking for Azure-specific features
                    try:
                        result = conn.execute(text("SELECT COUNT(*) FROM sys.dm_db_resource_stats"))
                        engine_edition = 5  # Assume Azure SQL if this view exists
                    except:
                        engine_edition = 0  # Unknown
                
                # Check if it's Azure SQL Database
                is_azure = engine_edition == 5
                
                return {
                    "success": True,
                    "connected": True,
                    "server": server,
                    "database": database,
                    "session_id": spid,
                    "version": version[:100] + "..." if len(version) > 100 else version,
                    "engine_edition": engine_edition,
                    "is_azure_sql": is_azure,
                    "message": "Azure SQL connection successful"
                }
                
        except Exception as e:
            logger.error(f"Azure SQL connection test failed: {e}")
            return {
                "success": False,
                "connected": False,
                "error": str(e),
                "is_azure_sql": False
            }
    
    def get_azure_server_info(self) -> Dict[str, Any]:
        """Get detailed Azure SQL Server information"""
        try:
            engine = self.get_engine()
            with engine.connect() as conn:
                # Get server properties
                queries = {
                    "version": "SELECT @@VERSION",
                    "server_name": "SELECT @@SERVERNAME",
                    "database_name": "SELECT DB_NAME()",
                    "session_id": "SELECT @@SPID",
                    "connection_count": "SELECT COUNT(*) FROM sys.dm_exec_connections",
                    "database_size": "SELECT SUM(size * 8.0 / 1024) AS size_mb FROM sys.database_files",
                    "max_connections": "SELECT @@MAX_CONNECTIONS",
                    "tempdb_size": "SELECT SUM(size * 8.0 / 1024) AS size_mb FROM tempdb.sys.database_files"
                }
                
                info = {}
                for key, query in queries.items():
                    try:
                        result = conn.execute(text(query))
                        value = result.fetchone()[0]
                        info[key] = value
                    except Exception as e:
                        logger.warning(f"Failed to get {key}: {e}")
                        info[key] = None
                
                return {
                    "success": True,
                    "server_info": info
                }
                
        except Exception as e:
            logger.error(f"Failed to get Azure server info: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def optimize_azure_connection(self) -> Dict[str, Any]:
        """Set optimal Azure SQL connection parameters"""
        try:
            engine = self.get_engine()
            with engine.connect() as conn:
                # Set Azure SQL specific session options for better performance
                optimizations = [
                    ("SET ARITHABORT ON", "Arithmetic abort enabled"),
                    ("SET NUMERIC_ROUNDABORT OFF", "Numeric round abort disabled"),
                    ("SET ANSI_PADDING ON", "ANSI padding enabled"),
                    ("SET ANSI_WARNINGS ON", "ANSI warnings enabled"),
                    ("SET CONCAT_NULL_YIELDS_NULL ON", "Concatenation with NULL yields NULL"),
                    ("SET ANSI_NULLS ON", "ANSI NULLs enabled"),
                    ("SET QUOTED_IDENTIFIER ON", "Quoted identifier enabled"),
                    ("SET NOCOUNT ON", "Row count messages disabled"),
                    ("SET TRANSACTION ISOLATION LEVEL READ COMMITTED", "Transaction isolation level set"),
                ]
                
                applied = []
                for query, description in optimizations:
                    try:
                        conn.execute(text(query))
                        applied.append(description)
                    except Exception as e:
                        logger.warning(f"Failed to apply {description}: {e}")
                
                return {
                    "success": True,
                    "optimizations_applied": applied,
                    "message": f"Applied {len(applied)} Azure SQL optimizations"
                }
                
        except Exception as e:
            logger.error(f"Failed to optimize Azure connection: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def execute_azure_query(self, query: str, params: Optional[Dict] = None, 
                           timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute a query with Azure SQL optimizations"""
        try:
            engine = self.get_engine()
            
            # Set command timeout if specified
            if timeout:
                engine.execute(text(f"SET LOCK_TIMEOUT {timeout * 1000}"))
            
            with engine.connect() as conn:
                # Apply Azure SQL optimizations
                self.optimize_azure_connection()
                
                # Execute query
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                
                # Process results
                if result.returns_rows:
                    columns = result.keys()
                    rows = [dict(zip(columns, row)) for row in result.fetchall()]
                    return {
                        "success": True,
                        "rows": rows,
                        "row_count": len(rows),
                        "columns": list(columns),
                        "query_type": "SELECT"
                    }
                else:
                    return {
                        "success": True,
                        "rows_affected": result.rowcount,
                        "query_type": "DML",
                        "message": "Query executed successfully"
                    }
                    
        except OperationalError as e:
            logger.error(f"Azure SQL operational error: {e}")
            return {
                "success": False,
                "error": f"Database operation failed: {str(e)}",
                "error_type": "operational"
            }
        except Exception as e:
            logger.error(f"Azure SQL unexpected error: {e}")
            return {
                "success": False,
                "error": f"Unexpected database error: {str(e)}",
                "error_type": "unknown"
            }
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific table"""
        try:
            query = """
            SELECT 
                c.COLUMN_NAME,
                c.DATA_TYPE,
                c.IS_NULLABLE,
                c.COLUMN_DEFAULT,
                c.CHARACTER_MAXIMUM_LENGTH,
                c.NUMERIC_PRECISION,
                c.NUMERIC_SCALE,
                CASE WHEN pk.COLUMN_NAME IS NOT NULL THEN 'YES' ELSE 'NO' END AS IS_PRIMARY_KEY
            FROM INFORMATION_SCHEMA.COLUMNS c
            LEFT JOIN (
                SELECT ku.COLUMN_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS AS tc
                INNER JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS ku
                    ON tc.CONSTRAINT_TYPE = 'PRIMARY KEY' 
                    AND tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
                WHERE ku.TABLE_NAME = :table_name
            ) pk ON c.COLUMN_NAME = pk.COLUMN_NAME
            WHERE c.TABLE_NAME = :table_name
            ORDER BY c.ORDINAL_POSITION
            """
            
            return self.execute_azure_query(query, {"table_name": table_name})
            
        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_table_row_count(self, table_name: str) -> Dict[str, Any]:
        """Get row count for a specific table"""
        try:
            query = f"SELECT COUNT(*) AS row_count FROM {table_name}"
            return self.execute_azure_query(query)
            
        except Exception as e:
            logger.error(f"Failed to get row count for {table_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def check_azure_performance(self) -> Dict[str, Any]:
        """Check Azure SQL performance metrics"""
        try:
            engine = self.get_engine()
            with engine.connect() as conn:
                # Get performance metrics
                metrics = {}
                
                # Connection count
                try:
                    result = conn.execute(text("SELECT COUNT(*) FROM sys.dm_exec_connections"))
                    metrics["active_connections"] = result.fetchone()[0]
                except Exception as e:
                    logger.warning(f"Failed to get connection count: {e}")
                    metrics["active_connections"] = "Not available"
                
                # Memory usage
                try:
                    result = conn.execute(text("SELECT SUM(pages_kb) / 1024.0 AS memory_mb FROM sys.dm_os_memory_clerks"))
                    memory_result = result.fetchone()
                    if memory_result and memory_result[0] is not None:
                        metrics["memory_usage_mb"] = round(float(memory_result[0]), 2)
                    else:
                        metrics["memory_usage_mb"] = "Not available"
                except Exception as e:
                    logger.warning(f"Failed to get memory usage: {e}")
                    metrics["memory_usage_mb"] = "Not available"
                
                # CPU usage (using a more compatible query)
                try:
                    # Use a simpler query that avoids datetime2 columns
                    result = conn.execute(text("""
                        SELECT TOP 1 
                            CAST(cpu_percent AS FLOAT) as cpu_percent
                        FROM sys.dm_db_resource_stats 
                        WHERE cpu_percent IS NOT NULL
                        ORDER BY end_time DESC
                    """))
                    cpu_result = result.fetchone()
                    if cpu_result and cpu_result[0] is not None:
                        metrics["cpu_percent"] = round(float(cpu_result[0]), 2)
                    else:
                        metrics["cpu_percent"] = "Not available"
                except Exception as e:
                    logger.warning(f"Failed to get CPU usage: {e}")
                    metrics["cpu_percent"] = "Not available"
                
                # Database size
                try:
                    result = conn.execute(text("SELECT SUM(size * 8.0 / 1024) AS size_mb FROM sys.database_files"))
                    size_result = result.fetchone()
                    if size_result and size_result[0] is not None:
                        metrics["database_size_mb"] = round(float(size_result[0]), 2)
                    else:
                        metrics["database_size_mb"] = "Not available"
                except Exception as e:
                    logger.warning(f"Failed to get database size: {e}")
                    metrics["database_size_mb"] = "Not available"
                
                return {
                    "success": True,
                    "performance_metrics": metrics
                }
                
        except Exception as e:
            logger.error(f"Failed to check Azure performance: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Convenience functions
def get_azure_sql_utils() -> AzureSQLUtils:
    """Get Azure SQL utilities instance"""
    return AzureSQLUtils()


def test_azure_connection() -> Dict[str, Any]:
    """Test Azure SQL connection"""
    utils = get_azure_sql_utils()
    return utils.test_azure_connection()


def get_azure_server_info() -> Dict[str, Any]:
    """Get Azure SQL server information"""
    utils = get_azure_sql_utils()
    return utils.get_azure_server_info()


def execute_azure_query(query: str, params: Optional[Dict] = None, 
                       timeout: Optional[int] = None) -> Dict[str, Any]:
    """Execute a query with Azure SQL optimizations"""
    utils = get_azure_sql_utils()
    return utils.execute_azure_query(query, params, timeout)
