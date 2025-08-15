"""
Database connection and session management
Supports both SQLite and Azure SQL Server with SQLAlchemy
"""

import logging
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DisconnectionError
from sqlalchemy.engine import Engine

from backend.config import get_settings

logger = logging.getLogger(__name__)

# Global database engine and session factory
_engine = None
_SessionLocal = None


def get_database_engine():
    """Get or create the database engine"""
    global _engine
    
    if _engine is None:
        settings = get_settings()
        
        try:
            database_url = settings.get_database_url()
            logger.info(f"Connecting to database: {database_url.split('@')[0]}@***")
            
            # Engine configuration optimized for Azure SQL
            engine_kwargs = {
                'echo': settings.debug,  # SQL logging in debug mode
                'poolclass': QueuePool,
                'pool_pre_ping': True,
                'pool_recycle': 1800,  # 30 minutes for Azure SQL
                'pool_timeout': 30,
                'max_overflow': 20,
            }
            
            # Azure SQL Server specific configuration
            if settings.is_azure_sql():
                engine_kwargs.update({
                    'pool_size': 20,  # Larger pool for Azure SQL
                    'pool_recycle': 1800,  # 30 minutes for Azure SQL
                    'connect_args': {
                        'timeout': settings.mssql_connection_timeout,
                        'command_timeout': settings.mssql_command_timeout,
                        'autocommit': False,
                        'MultipleActiveResultSets': True,  # Enable MARS
                        'ApplicationIntent': 'ReadWrite',
                    }
                })
                
                # Add Azure SQL specific event listeners
                def _on_connect(dbapi_connection, connection_record):
                    """Set Azure SQL specific connection properties"""
                    try:
                        # Set session-level settings for Azure SQL
                        cursor = dbapi_connection.cursor()
                        cursor.execute("SET ARITHABORT ON")
                        cursor.execute("SET NUMERIC_ROUNDABORT OFF")
                        cursor.execute("SET ANSI_PADDING ON")
                        cursor.execute("SET ANSI_WARNINGS ON")
                        cursor.execute("SET CONCAT_NULL_YIELDS_NULL ON")
                        cursor.execute("SET ANSI_NULLS ON")
                        cursor.execute("SET QUOTED_IDENTIFIER ON")
                        cursor.close()
                        logger.debug("Azure SQL connection properties set")
                    except Exception as e:
                        logger.warning(f"Failed to set Azure SQL properties: {e}")
                
                # Register the connection event
                event.listen(Engine, "connect", _on_connect)
                
            else:
                # SQLite configuration
                engine_kwargs.update({
                    'pool_size': 10,
                    'pool_recycle': 3600,  # 1 hour for SQLite
                })
            
            _engine = create_engine(database_url, **engine_kwargs)
            
            # Test connection
            with _engine.connect() as conn:
                if settings.is_azure_sql():
                    result = conn.execute(text("SELECT @@VERSION, @@SERVERNAME, DB_NAME()"))
                    version, server, db = result.fetchone()
                    logger.info(f"Connected to Azure SQL Server: {server} - Database: {db}")
                    logger.info(f"Version: {version[:100]}...")
                else:
                    result = conn.execute(text("SELECT sqlite_version()"))
                    version = result.fetchone()[0]
                    logger.info(f"Connected to SQLite: {version}")
            
            logger.info("Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database engine: {str(e)}")
            raise
    
    return _engine


def get_database_session() -> Session:
    """Get a new database session"""
    global _SessionLocal
    
    if _SessionLocal is None:
        engine = get_database_engine()
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
    
    return _SessionLocal()


def test_database_connection() -> bool:
    """Test database connection and return success status"""
    try:
        engine = get_database_engine()
        with engine.connect() as conn:
            if get_settings().is_azure_sql():
                # Test Azure SQL specific functionality
                conn.execute(text("SELECT 1, @@VERSION"))
            else:
                conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return False


def close_database_connection():
    """Close database connections and cleanup"""
    global _engine, _SessionLocal
    
    if _engine:
        _engine.dispose()
        _engine = None
    
    if _SessionLocal:
        _SessionLocal = None
    
    logger.info("Database connections closed")


# Dependency for FastAPI
def get_db() -> Session:
    """Dependency to get database session for FastAPI routes"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()


# Context manager for database sessions
class DatabaseSession:
    """Context manager for database sessions"""
    
    def __init__(self):
        self.session: Optional[Session] = None
    
    def __enter__(self) -> Session:
        self.session = get_database_session()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type is not None:
                self.session.rollback()
            self.session.close()


# Health check function
def get_database_health() -> Dict[str, Any]:
    """Get database health status for monitoring"""
    try:
        is_connected = test_database_connection()
        settings = get_settings()
        
        health_info = {
            "status": "healthy" if is_connected else "unhealthy",
            "database_type": settings.database_type,
            "connected": is_connected,
            "pool_size": 20 if settings.is_azure_sql() else 10,
            "pool_overflow": 20,
        }
        
        # Add Azure SQL specific info
        if settings.is_azure_sql():
            try:
                engine = get_database_engine()
                with engine.connect() as conn:
                    # Get Azure SQL server info
                    result = conn.execute(text("SELECT @@SERVERNAME, @@VERSION, DB_NAME()"))
                    server, version, db = result.fetchone()
                    health_info.update({
                        "azure_server": server,
                        "azure_database": db,
                        "azure_version": version[:50] + "..." if len(version) > 50 else version,
                        "is_azure": True
                    })
            except Exception as e:
                health_info.update({
                    "azure_error": str(e),
                    "is_azure": True
                })
        else:
            health_info["is_azure"] = False
            
        return health_info
        
    except Exception as e:
        logger.error(f"Error checking database health: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "connected": False,
            "is_azure": False,
        }


def execute_azure_sql_query(query: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    """Execute a query specifically for Azure SQL with proper error handling"""
    try:
        engine = get_database_engine()
        with engine.connect() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            
            # Fetch results
            if result.returns_rows:
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
                return {
                    "success": True,
                    "rows": rows,
                    "row_count": len(rows),
                    "columns": list(columns)
                }
            else:
                return {
                    "success": True,
                    "rows_affected": result.rowcount,
                    "message": "Query executed successfully"
                }
                
    except OperationalError as e:
        logger.error(f"Azure SQL operational error: {e}")
        return {
            "success": False,
            "error": f"Database operation failed: {str(e)}",
            "error_type": "operational"
        }
    except DisconnectionError as e:
        logger.error(f"Azure SQL connection error: {e}")
        return {
            "success": False,
            "error": f"Database connection lost: {str(e)}",
            "error_type": "connection"
        }
    except Exception as e:
        logger.error(f"Azure SQL unexpected error: {e}")
        return {
            "success": False,
            "error": f"Unexpected database error: {str(e)}",
            "error_type": "unknown"
        }
