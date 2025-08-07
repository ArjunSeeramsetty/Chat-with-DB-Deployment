"""
Schema Metadata Injection Module
Provides rich schema information to improve SQL generation accuracy
"""

import sqlite3
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class SchemaMetadataExtractor:
    """
    Extracts comprehensive schema metadata from SQLite database
    Provides rich context for SQL generation
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._schema_cache = None
        self._last_refresh = None
        self._cache_duration = 300  # 5 minutes cache
        
    def get_schema_metadata(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive schema metadata including tables, columns, types, relationships
        
        Args:
            force_refresh: Force refresh of cached schema data
            
        Returns:
            Dictionary containing complete schema metadata
        """
        # Check if cache is valid
        if not force_refresh and self._is_cache_valid():
            return self._schema_cache
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                metadata = {
                    "tables": self._extract_table_info(conn),
                    "columns": self._extract_column_info(conn),
                    "relationships": self._extract_relationship_info(conn),
                    "constraints": self._extract_constraint_info(conn),
                    "indexes": self._extract_index_info(conn),
                    "sample_data": self._extract_sample_data(conn),
                    "statistics": self._extract_statistics(conn),
                    "extracted_at": datetime.now().isoformat()
                }
                
                self._schema_cache = metadata
                self._last_refresh = datetime.now()
                
                logger.info(f"Schema metadata extracted successfully: {len(metadata['tables'])} tables, {len(metadata['columns'])} columns")
                return metadata
                
        except Exception as e:
            logger.error(f"Failed to extract schema metadata: {e}")
            return self._get_fallback_metadata()
            
    def _is_cache_valid(self) -> bool:
        """Check if cached schema data is still valid"""
        if not self._schema_cache or not self._last_refresh:
            return False
            
        age = (datetime.now() - self._last_refresh).total_seconds()
        return age < self._cache_duration
        
    def _extract_table_info(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        """Extract detailed table information"""
        tables = []
        
        # Get table list
        cursor = conn.execute("""
            SELECT name, sql FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        
        for row in cursor.fetchall():
            table_name, create_sql = row
            
            # Get row count
            try:
                count_cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = count_cursor.fetchone()[0]
            except:
                row_count = 0
                
            tables.append({
                "name": table_name,
                "create_sql": create_sql,
                "row_count": row_count,
                "description": self._get_table_description(table_name)
            })
            
        return tables
        
    def _extract_column_info(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        """Extract detailed column information"""
        columns = []
        
        for table_name in self._get_table_names(conn):
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            
            for row in cursor.fetchall():
                cid, name, data_type, not_null, default_val, pk = row
                
                # Get sample values for the column
                sample_values = self._get_sample_values(conn, table_name, name)
                
                columns.append({
                    "table_name": table_name,
                    "name": name,
                    "data_type": data_type,
                    "not_null": bool(not_null),
                    "default_value": default_val,
                    "primary_key": bool(pk),
                    "sample_values": sample_values,
                    "description": self._get_column_description(table_name, name)
                })
                
        return columns
        
    def _extract_relationship_info(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        """Extract foreign key relationships"""
        relationships = []
        
        for table_name in self._get_table_names(conn):
            try:
                cursor = conn.execute(f"PRAGMA foreign_key_list({table_name})")
                
                for row in cursor.fetchall():
                    # Handle variable column counts in PRAGMA foreign_key_list
                    if len(row) >= 8:
                        id, seq, ref_table, from_col, to_col, on_update, on_delete, match = row[:8]
                    elif len(row) >= 6:
                        id, seq, ref_table, from_col, to_col = row[:6]
                        on_update, on_delete, match = None, None, None
                    else:
                        # Skip rows with insufficient data
                        continue
                    
                    relationships.append({
                        "from_table": table_name,
                        "from_column": from_col,
                        "to_table": ref_table,
                        "to_column": to_col,
                        "on_update": on_update,
                        "on_delete": on_delete,
                        "match": match
                    })
            except Exception as e:
                logger.warning(f"Could not extract relationships for table {table_name}: {e}")
                
        return relationships
        
    def _extract_constraint_info(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        """Extract constraint information"""
        constraints = []
        
        for table_name in self._get_table_names(conn):
            try:
                # Get unique constraints
                cursor = conn.execute(f"PRAGMA index_list({table_name})")
                
                for row in cursor.fetchall():
                    # Handle variable column counts in PRAGMA index_list
                    if len(row) >= 3:
                        seq, name, unique = row[:3]
                    elif len(row) >= 2:
                        seq, name = row[:2]
                        unique = False
                    else:
                        continue
                        
                    if name.startswith("sqlite_autoindex_"):
                        continue
                        
                    # Get columns in this index
                    try:
                        index_cursor = conn.execute(f"PRAGMA index_info({name})")
                        index_columns = []
                        for index_row in index_cursor.fetchall():
                            if len(index_row) >= 3:
                                index_columns.append(index_row[2])
                            elif len(index_row) >= 1:
                                index_columns.append(index_row[0])
                    except Exception as e:
                        logger.warning(f"Could not get index info for {name}: {e}")
                        index_columns = []
                        
                    constraints.append({
                        "table_name": table_name,
                        "constraint_name": name,
                        "constraint_type": "UNIQUE" if unique else "INDEX",
                        "columns": index_columns
                    })
            except Exception as e:
                logger.warning(f"Could not extract constraints for table {table_name}: {e}")
                
        return constraints
        
    def _extract_index_info(self, conn: sqlite3.Connection) -> List[Dict[str, Any]]:
        """Extract index information"""
        indexes = []
        
        for table_name in self._get_table_names(conn):
            try:
                cursor = conn.execute(f"PRAGMA index_list({table_name})")
                
                for row in cursor.fetchall():
                    # Handle variable column counts in PRAGMA index_list
                    if len(row) >= 3:
                        seq, name, unique = row[:3]
                    elif len(row) >= 2:
                        seq, name = row[:2]
                        unique = False
                    else:
                        continue
                    
                    # Get index columns
                    try:
                        index_cursor = conn.execute(f"PRAGMA index_info({name})")
                        columns = []
                        for index_row in index_cursor.fetchall():
                            if len(index_row) >= 3:
                                columns.append(index_row[2])
                            elif len(index_row) >= 1:
                                columns.append(index_row[0])
                    except Exception as e:
                        logger.warning(f"Could not get index info for {name}: {e}")
                        columns = []
                        
                    indexes.append({
                        "table_name": table_name,
                        "index_name": name,
                        "unique": bool(unique),
                        "columns": columns
                    })
            except Exception as e:
                logger.warning(f"Could not extract indexes for table {table_name}: {e}")
                
        return indexes
        
    def _extract_sample_data(self, conn: sqlite3.Connection) -> Dict[str, List[Any]]:
        """Extract sample data from each table"""
        sample_data = {}
        
        for table_name in self._get_table_names(conn):
            try:
                cursor = conn.execute(f"SELECT * FROM {table_name} LIMIT 5")
                rows = cursor.fetchall()
                
                if rows:
                    # Get column names
                    column_names = [description[0] for description in cursor.description]
                    sample_data[table_name] = {
                        "columns": column_names,
                        "rows": [list(row) for row in rows]
                    }
            except Exception as e:
                logger.warning(f"Could not extract sample data from {table_name}: {e}")
                
        return sample_data
        
    def _extract_statistics(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Extract database statistics"""
        stats = {
            "total_tables": 0,
            "total_columns": 0,
            "total_rows": 0,
            "table_sizes": {},
            "column_types": {}
        }
        
        table_names = self._get_table_names(conn)
        stats["total_tables"] = len(table_names)
        
        for table_name in table_names:
            try:
                # Get row count
                count_cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = count_cursor.fetchone()[0]
                stats["total_rows"] += row_count
                stats["table_sizes"][table_name] = row_count
                
                # Get column info
                cursor = conn.execute(f"PRAGMA table_info({table_name})")
                for row in cursor.fetchall():
                    data_type = row[2]
                    stats["total_columns"] += 1
                    stats["column_types"][data_type] = stats["column_types"].get(data_type, 0) + 1
                    
            except Exception as e:
                logger.warning(f"Could not get statistics for {table_name}: {e}")
                
        return stats
        
    def _get_table_names(self, conn: sqlite3.Connection) -> List[str]:
        """Get list of table names"""
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        return [row[0] for row in cursor.fetchall()]
        
    def _get_sample_values(self, conn: sqlite3.Connection, table_name: str, column_name: str, limit: int = 5) -> List[Any]:
        """Get sample values for a column"""
        try:
            cursor = conn.execute(f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL LIMIT {limit}")
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.warning(f"Could not get sample values for {table_name}.{column_name}: {e}")
            return []
            
    def _get_table_description(self, table_name: str) -> str:
        """Get human-readable description for a table"""
        descriptions = {
            "FactStateDailyEnergy": "Daily energy consumption data by state",
            "FactAllIndiaDailySummary": "Daily summary of energy data across India",
            "FactGenerationData": "Power generation data and statistics",
            "FactTransmissionData": "Transmission network data and metrics",
            "DimStates": "State dimension table with state information",
            "DimRegions": "Region dimension table with regional classifications",
            "DimDates": "Date dimension table for temporal analysis"
        }
        return descriptions.get(table_name, f"Table containing {table_name} data")
        
    def _get_column_description(self, table_name: str, column_name: str) -> str:
        """Get human-readable description for a column"""
        descriptions = {
            "EnergyMet": "Actual energy consumption in MWh",
            "EnergyShortage": "Energy shortage/deficit in MWh",
            "MaximumDemand": "Peak power demand in MW",
            "GenerationAmount": "Power generation amount in MWh",
            "ActualDate": "The actual date for the data point",
            "StateName": "Name of the state",
            "RegionName": "Name of the region"
        }
        return descriptions.get(column_name, f"Column {column_name} in {table_name}")
        
    def _get_fallback_metadata(self) -> Dict[str, Any]:
        """Return fallback metadata when extraction fails"""
        return {
            "tables": [],
            "columns": [],
            "relationships": [],
            "constraints": [],
            "indexes": [],
            "sample_data": {},
            "statistics": {},
            "extracted_at": datetime.now().isoformat(),
            "error": "Schema extraction failed, using fallback"
        }
        
    def build_schema_prompt_context(self, query: str) -> str:
        """
        Build schema context for LLM prompts
        
        Args:
            query: User query to analyze
            
        Returns:
            Formatted schema context string
        """
        metadata = self.get_schema_metadata()
        
        context_parts = []
        
        # Add table information
        context_parts.append("## DATABASE SCHEMA")
        for table in metadata["tables"]:
            context_parts.append(f"Table: {table['name']}")
            context_parts.append(f"  Description: {table['description']}")
            context_parts.append(f"  Row count: {table['row_count']}")
            
        # Add column information
        context_parts.append("\n## COLUMNS")
        for column in metadata["columns"]:
            context_parts.append(f"{column['table_name']}.{column['name']} ({column['data_type']})")
            context_parts.append(f"  Description: {column['description']}")
            if column['primary_key']:
                context_parts.append("  Primary Key: Yes")
            if column['sample_values']:
                context_parts.append(f"  Sample values: {column['sample_values'][:3]}")
                
        # Add relationships
        if metadata["relationships"]:
            context_parts.append("\n## RELATIONSHIPS")
            for rel in metadata["relationships"]:
                context_parts.append(f"{rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']}")
                
        # Add constraints
        if metadata["constraints"]:
            context_parts.append("\n## CONSTRAINTS")
            for constraint in metadata["constraints"]:
                context_parts.append(f"{constraint['table_name']}: {constraint['constraint_type']} on {', '.join(constraint['columns'])}")
                
        return "\n".join(context_parts)
        
    def get_relevant_schema_context(self, query: str) -> Dict[str, Any]:
        """
        Get schema context most relevant to the query
        
        Args:
            query: User query to analyze
            
        Returns:
            Dictionary with relevant schema information
        """
        metadata = self.get_schema_metadata()
        
        # Simple keyword matching to find relevant tables/columns
        query_lower = query.lower()
        relevant_tables = []
        relevant_columns = []
        
        # Find relevant tables
        for table in metadata["tables"]:
            if any(keyword in query_lower for keyword in table["name"].lower().split("_")):
                relevant_tables.append(table["name"])
                
        # Find relevant columns
        for column in metadata["columns"]:
            if any(keyword in query_lower for keyword in column["name"].lower().split("_")):
                relevant_columns.append(column)
                
        return {
            "relevant_tables": relevant_tables,
            "relevant_columns": relevant_columns,
            "full_schema": metadata,
            "query_analysis": {
                "keywords": self._extract_keywords(query),
                "entities": self._extract_entities(query)
            }
        }
        
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        # Simple keyword extraction
        keywords = []
        query_lower = query.lower()
        
        # Energy-related keywords
        energy_keywords = ["energy", "power", "consumption", "generation", "demand", "shortage"]
        for keyword in energy_keywords:
            if keyword in query_lower:
                keywords.append(keyword)
                
        # Time-related keywords
        time_keywords = ["monthly", "daily", "yearly", "growth", "trend", "period"]
        for keyword in time_keywords:
            if keyword in query_lower:
                keywords.append(keyword)
                
        # Location-related keywords
        location_keywords = ["region", "state", "area", "location"]
        for keyword in location_keywords:
            if keyword in query_lower:
                keywords.append(keyword)
                
        return keywords
        
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query"""
        entities = []
        query_lower = query.lower()
        
        # Extract potential table names
        table_names = [table["name"] for table in self.get_schema_metadata()["tables"]]
        for table_name in table_names:
            if table_name.lower() in query_lower:
                entities.append(table_name)
                
        # Extract potential column names
        column_names = [col["name"] for col in self.get_schema_metadata()["columns"]]
        for column_name in column_names:
            if column_name.lower() in query_lower:
                entities.append(column_name)
                
        return entities
