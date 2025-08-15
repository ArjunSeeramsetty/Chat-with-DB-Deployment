#!/usr/bin/env python3
"""
Data Migration Script: SQLite to SQL Server
Moves data from existing SQLite database to Microsoft SQL Server
"""

import os
import sys
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging
from typing import Dict, List, Any
import argparse

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SQLiteToMSSQLMigrator:
    """Migrate data from SQLite to SQL Server"""
    
    def __init__(self, sqlite_path: str, mssql_url: str):
        self.sqlite_path = sqlite_path
        self.mssql_url = mssql_url
        self.sqlite_conn = None
        self.mssql_engine = None
        
    def connect_databases(self):
        """Establish connections to both databases"""
        try:
            # Connect to SQLite
            self.sqlite_conn = sqlite3.connect(self.sqlite_path)
            logger.info(f"Connected to SQLite: {self.sqlite_path}")
            
            # Connect to SQL Server
            self.mssql_engine = create_engine(
                self.mssql_url,
                echo=False,
                pool_pre_ping=True
            )
            
            # Test connection
            with self.mssql_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Connected to SQL Server")
            
        except Exception as e:
            logger.error(f"Failed to connect to databases: {e}")
            raise
    
    def get_sqlite_tables(self) -> List[str]:
        """Get list of tables from SQLite database"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return tables
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get table schema information"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        cursor.close()
        
        schema = {
            'name': table_name,
            'columns': []
        }
        
        for col in columns:
            schema['columns'].append({
                'name': col[1],
                'type': col[2],
                'not_null': bool(col[3]),
                'default_value': col[4],
                'primary_key': bool(col[5])
            })
        
        return schema
    
    def map_sqlite_type_to_mssql(self, sqlite_type: str) -> str:
        """Map SQLite data types to SQL Server types"""
        type_mapping = {
            'INTEGER': 'INT',
            'REAL': 'FLOAT',
            'TEXT': 'NVARCHAR(MAX)',
            'BLOB': 'VARBINARY(MAX)',
            'NUMERIC': 'DECIMAL(18,2)',
            'BOOLEAN': 'BIT',
            'DATE': 'DATE',
            'DATETIME': 'DATETIME2',
            'TIMESTAMP': 'DATETIME2',
            'VARCHAR': 'NVARCHAR(255)',
            'CHAR': 'NCHAR(1)',
            'DOUBLE': 'FLOAT',
            'FLOAT': 'FLOAT',
            'DECIMAL': 'DECIMAL(18,2)',
            'MONEY': 'MONEY'
        }
        
        # Handle type with length (e.g., VARCHAR(255))
        base_type = sqlite_type.split('(')[0].upper()
        return type_mapping.get(base_type, 'NVARCHAR(MAX)')
    
    def create_mssql_table(self, schema: Dict[str, Any]):
        """Create table in SQL Server based on SQLite schema"""
        table_name = schema['name']
        columns = []
        
        for col in schema['columns']:
            col_name = col['name']
            col_type = self.map_sqlite_type_to_mssql(col['type'])
            col_def = f"[{col_name}] {col_type}"
            
            if col['not_null']:
                col_def += " NOT NULL"
            
            if col['primary_key']:
                col_def += " PRIMARY KEY"
            
            columns.append(col_def)
        
        create_sql = f"""
        IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[{table_name}]') AND type in (N'U'))
        BEGIN
            CREATE TABLE [dbo].[{table_name}] (
                {',\n                '.join(columns)}
            )
        END
        """
        
        try:
            with self.mssql_engine.connect() as conn:
                conn.execute(text(create_sql))
                conn.commit()
            logger.info(f"Created table: {table_name}")
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to create table {table_name}: {e}")
            raise
    
    def migrate_table_data(self, table_name: str, batch_size: int = 1000):
        """Migrate data from SQLite table to SQL Server"""
        try:
            # Read data from SQLite in batches
            offset = 0
            total_rows = 0
            
            while True:
                # Read batch from SQLite
                query = f"SELECT * FROM {table_name} LIMIT {batch_size} OFFSET {offset}"
                df = pd.read_sql_query(query, self.sqlite_conn)
                
                if df.empty:
                    break
                
                # Clean column names (remove brackets if present)
                df.columns = [col.replace('[', '').replace(']', '') for col in df.columns]
                
                # Write batch to SQL Server
                df.to_sql(
                    table_name,
                    self.mssql_engine,
                    if_exists='append',
                    index=False,
                    method='multi',
                    chunksize=batch_size
                )
                
                batch_rows = len(df)
                total_rows += batch_rows
                offset += batch_size
                
                logger.info(f"Migrated {batch_rows} rows from {table_name} (total: {total_rows})")
                
                # If we got fewer rows than batch_size, we're done
                if batch_rows < batch_size:
                    break
            
            logger.info(f"Completed migration of {total_rows} rows from {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to migrate data from {table_name}: {e}")
            raise
    
    def create_indexes(self, table_name: str):
        """Create common indexes for better performance"""
        try:
            # Common index patterns for energy data
            index_sqls = []
            
            if 'Date' in table_name or 'date' in table_name:
                index_sqls.append(f"CREATE INDEX IX_{table_name}_Date ON [dbo].[{table_name}] ([Date])")
            
            if 'Region' in table_name or 'region' in table_name:
                index_sqls.append(f"CREATE INDEX IX_{table_name}_Region ON [dbo].[{table_name}] ([RegionID])")
            
            if 'State' in table_name or 'state' in table_name:
                index_sqls.append(f"CREATE INDEX IX_{table_name}_State ON [dbo].[{table_name}] ([StateID])")
            
            # Create indexes
            with self.mssql_engine.connect() as conn:
                for index_sql in index_sqls:
                    try:
                        conn.execute(text(index_sql))
                        logger.info(f"Created index: {index_sql}")
                    except SQLAlchemyError as e:
                        logger.warning(f"Failed to create index: {e}")
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to create indexes for {table_name}: {e}")
    
    def migrate_all(self, batch_size: int = 1000):
        """Migrate all tables from SQLite to SQL Server"""
        try:
            tables = self.get_sqlite_tables()
            logger.info(f"Found {len(tables)} tables to migrate: {tables}")
            
            for table_name in tables:
                logger.info(f"Starting migration of table: {table_name}")
                
                # Get schema
                schema = self.get_table_schema(table_name)
                
                # Create table in SQL Server
                self.create_mssql_table(schema)
                
                # Migrate data
                self.migrate_table_data(table_name, batch_size)
                
                # Create indexes
                self.create_indexes(table_name)
                
                logger.info(f"Completed migration of table: {table_name}")
            
            logger.info("All tables migrated successfully!")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
    
    def close_connections(self):
        """Close database connections"""
        if self.sqlite_conn:
            self.sqlite_conn.close()
        if self.mssql_engine:
            self.mssql_engine.dispose()
        logger.info("Database connections closed")

def main():
    """Main migration function"""
    parser = argparse.ArgumentParser(description='Migrate SQLite to SQL Server')
    parser.add_argument('--sqlite-path', required=True, help='Path to SQLite database file')
    parser.add_argument('--mssql-url', required=True, help='SQL Server connection URL')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for data migration')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be migrated without executing')
    
    args = parser.parse_args()
    
    try:
        # Initialize migrator
        migrator = SQLiteToMSSQLMigrator(args.sqlite_path, args.mssql_url)
        
        if args.dry_run:
            # Show migration plan
            migrator.connect_databases()
            tables = migrator.get_sqlite_tables()
            logger.info("=== MIGRATION PLAN ===")
            for table_name in tables:
                schema = migrator.get_table_schema(table_name)
                logger.info(f"Table: {table_name}")
                logger.info(f"  Columns: {len(schema['columns'])}")
                for col in schema['columns']:
                    mssql_type = migrator.map_sqlite_type_to_mssql(col['type'])
                    logger.info(f"    {col['name']}: {col['type']} -> {mssql_type}")
            migrator.close_connections()
            
        else:
            # Execute migration
            migrator.connect_databases()
            migrator.migrate_all(args.batch_size)
            migrator.close_connections()
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
