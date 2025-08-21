"""
Cloud-Ready Semantic Engine for MS SQL Server
Implements advanced semantic understanding with vector search and context awareness
Updated for cloud deployment with MS SQL Server compatibility
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import pyodbc

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pandas as pd

from backend.core.types import QueryAnalysis, IntentType, QueryType
from backend.core.llm_provider import LLMProvider
from backend.config import get_settings

logger = logging.getLogger(__name__)


class SemanticConfidence(Enum):
    """Confidence levels for semantic analysis"""
    HIGH = "high"       # 0.8+
    MEDIUM = "medium"   # 0.6-0.8
    LOW = "low"         # 0.4-0.6
    VERY_LOW = "very_low"  # <0.4


@dataclass
class SemanticContext:
    """Enhanced semantic context with business understanding"""
    intent: IntentType
    confidence: float
    business_entities: List[Dict[str, Any]]
    domain_concepts: List[str]
    temporal_context: Optional[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    semantic_mappings: Dict[str, str]
    vector_similarity: float


@dataclass
class MDLRelationship:
    """Modeling Definition Language relationship"""
    source_table: str
    target_table: str
    join_type: str
    join_conditions: List[str]
    relationship_type: str  # one-to-one, one-to-many, many-to-many
    business_meaning: str


@dataclass
class DomainModel:
    """Energy domain model with business semantics"""
    tables: Dict[str, Dict[str, Any]]
    relationships: List[MDLRelationship]
    business_glossary: Dict[str, str]
    metrics: Dict[str, Dict[str, Any]]
    dimensions: Dict[str, Dict[str, Any]]


class CloudSemanticEngine:
    """
    Cloud-ready semantic engine with MS SQL Server compatibility
    Integrates vector search, domain modeling, and intelligent context retrieval
    """
    
    def __init__(self, llm_provider: LLMProvider, db_connection_string: str = None):
        self.llm_provider = llm_provider
        self.db_connection_string = db_connection_string or get_settings().get_database_url()
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_client = QdrantClient(":memory:")  # In-memory for development
        self.domain_model: Optional[DomainModel] = None
        
        # Initialize settings
        self.settings = get_settings()
        
        # Initialize vector store collections
        self._initialize_vector_store()
        
        # Load domain model
        self._load_domain_model()
        
        logger.info("Cloud Semantic Engine initialized successfully")
    
    def _initialize_vector_store(self):
        """Initialize vector database collections"""
        try:
            # Create collection for schema semantics
            self.vector_client.create_collection(
                collection_name="schema_semantics",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            
            # Create collection for query patterns
            self.vector_client.create_collection(
                collection_name="query_patterns",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            
            # Create collection for business context
            self.vector_client.create_collection(
                collection_name="business_context",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            
            logger.info("Vector store collections created successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def _load_domain_model(self):
        """Load energy domain model with business semantics"""
        # Energy domain model definition
        energy_tables = {
            "FactAllIndiaDailySummary": {
                "business_name": "National Energy Summary",
                "description": "Daily energy generation and consumption data at national level",
                "key_metrics": ["EnergyMet", "EnergyRequirement", "Surplus", "Deficit", "EnergyShortage"],
                "dimensions": ["RegionID", "DateID"],
                "grain": "daily_region"
            },
            "FactStateDailyEnergy": {
                "business_name": "State Energy Data",
                "description": "Daily energy consumption and availability by state",
                "key_metrics": ["EnergyAvailability", "EnergyConsumption", "PeakDemand"],
                "dimensions": ["StateID", "DateID"],
                "grain": "daily_state"
            },
            "FactDailyGenerationBreakdown": {
                "business_name": "Generation Source Breakdown",
                "description": "Daily energy generation by source type (coal, solar, wind, etc.)",
                "key_metrics": ["GenerationMW", "Capacity", "PLF"],
                "dimensions": ["GenerationSourceID", "StateID", "DateID"],
                "grain": "daily_source_state"
            }
        }
        
        # Create domain model
        self.domain_model = DomainModel(
            tables=energy_tables,
            relationships=[],
            business_glossary={},
            metrics={},
            dimensions={}
        )
        
        logger.info("Domain model loaded successfully")
    
    async def initialize(self):
        """Initialize the semantic engine with domain knowledge"""
        try:
            # Populate vector store with domain knowledge
            await self._populate_vector_store()
            logger.info("Cloud Semantic Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Cloud Semantic Engine: {e}")
            raise
    
    async def _populate_vector_store(self):
        """Populate vector store with domain knowledge"""
        try:
            # Add table information to vector store
            for table_name, table_info in self.domain_model.tables.items():
                # Create vector for table
                table_text = f"Table: {table_name} - {table_info['business_name']} - {table_info['description']}"
                table_embedding = self.sentence_transformer.encode(table_text)
                
                # Add to vector store
                self.vector_client.upsert(
                    collection_name="schema_semantics",
                    points=[
                        PointStruct(
                            id=hash(table_name),
                            vector=table_embedding.tolist(),
                            payload={"table": table_name, "info": table_info}
                        )
                    ]
                )
            
            logger.info("Vector store populated with domain knowledge")
            
        except Exception as e:
            logger.error(f"Failed to populate vector store: {e}")
            raise
    
    async def get_schema_metadata(self) -> Dict[str, Any]:
        """Get schema metadata from MS SQL Server"""
        try:
            if not self.db_connection_string:
                logger.warning("No database connection string available")
                return {}
            
            # Connect to MS SQL Server
            conn = pyodbc.connect(self.db_connection_string)
            cursor = conn.cursor()
            
            # Get table information
            cursor.execute("""
                SELECT 
                    t.TABLE_NAME,
                    c.COLUMN_NAME,
                    c.DATA_TYPE,
                    c.IS_NULLABLE,
                    c.COLUMN_DEFAULT
                FROM INFORMATION_SCHEMA.TABLES t
                JOIN INFORMATION_SCHEMA.COLUMNS c ON t.TABLE_NAME = c.TABLE_NAME
                WHERE t.TABLE_TYPE = 'BASE TABLE'
                ORDER BY t.TABLE_NAME, c.ORDINAL_POSITION
            """)
            
            rows = cursor.fetchall()
            
            # Organize schema information
            schema_info = {}
            for row in rows:
                table_name = row[0]
                column_name = row[1]
                data_type = row[2]
                is_nullable = row[3]
                column_default = row[4]
                
                if table_name not in schema_info:
                    schema_info[table_name] = {"columns": [], "metadata": {}}
                
                schema_info[table_name]["columns"].append({
                    "name": column_name,
                    "type": data_type,
                    "nullable": is_nullable == "YES",
                    "default": column_default
                })
            
            conn.close()
            
            logger.info(f"Retrieved schema metadata for {len(schema_info)} tables")
            return schema_info
            
        except Exception as e:
            logger.error(f"Failed to get schema metadata: {e}")
            return {}
    
    def get_semantic_context(self, query: str) -> SemanticContext:
        """Get semantic context for a query"""
        try:
            # Analyze query intent
            intent = self._analyze_intent(query)
            
            # Extract business entities
            business_entities = self._extract_business_entities(query)
            
            # Get domain concepts
            domain_concepts = self._get_domain_concepts(query)
            
            # Get temporal context
            temporal_context = self._extract_temporal_context(query)
            
            # Get relationships
            relationships = self._get_relationships(query)
            
            # Get semantic mappings
            semantic_mappings = self._get_semantic_mappings(query)
            
            # Calculate vector similarity
            vector_similarity = self._calculate_vector_similarity(query)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                intent, business_entities, domain_concepts, 
                temporal_context, relationships, vector_similarity
            )
            
            return SemanticContext(
                intent=intent,
                confidence=confidence,
                business_entities=business_entities,
                domain_concepts=domain_concepts,
                temporal_context=temporal_context,
                relationships=relationships,
                semantic_mappings=semantic_mappings,
                vector_similarity=vector_similarity
            )
            
        except Exception as e:
            logger.error(f"Failed to get semantic context: {e}")
            # Return default context
            return SemanticContext(
                intent=IntentType.DATA_RETRIEVAL,
                confidence=0.5,
                business_entities=[],
                domain_concepts=[],
                temporal_context=None,
                relationships=[],
                semantic_mappings={},
                vector_similarity=0.0
            )
    
    def _analyze_intent(self, query: str) -> IntentType:
        """Analyze query intent"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "show", "get", "find", "list"]):
            return IntentType.DATA_RETRIEVAL
        elif any(word in query_lower for word in ["compare", "vs", "versus"]):
            return IntentType.COMPARISON
        elif any(word in query_lower for word in ["trend", "over time", "growth"]):
            return IntentType.TREND_ANALYSIS
        elif any(word in query_lower for word in ["total", "sum", "average", "count"]):
            return IntentType.AGGREGATION
        else:
            return IntentType.DATA_RETRIEVAL
    
    def _extract_business_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract business entities from query"""
        entities = []
        query_lower = query.lower()
        
        # Extract energy-related entities
        energy_terms = ["energy", "power", "electricity", "generation", "consumption"]
        for term in energy_terms:
            if term in query_lower:
                entities.append({
                    "type": "energy_concept",
                    "value": term,
                    "confidence": 0.8
                })
        
        # Extract shortage-related entities
        if "shortage" in query_lower:
            entities.append({
                "type": "metric",
                "value": "EnergyShortage",
                "confidence": 0.9
            })
        
        # Extract region-related entities
        if "region" in query_lower:
            entities.append({
                "type": "dimension",
                "value": "RegionName",
                "confidence": 0.8
            })
        
        return entities
    
    def _get_domain_concepts(self, query: str) -> List[str]:
        """Get relevant domain concepts"""
        concepts = []
        query_lower = query.lower()
        
        # Add energy domain concepts
        if "energy" in query_lower:
            concepts.extend(["energy_management", "power_systems", "grid_operations"])
        
        if "shortage" in query_lower:
            concepts.extend(["demand_supply", "grid_reliability", "energy_security"])
        
        if "region" in query_lower:
            concepts.extend(["geographic_analysis", "regional_planning"])
        
        return concepts
    
    def _extract_temporal_context(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract temporal context from query"""
        query_lower = query.lower()
        
        # Extract year
        if "2025" in query:
            return {"year": 2025, "type": "specific_year"}
        
        # Extract month
        if "june" in query_lower:
            return {"month": 6, "type": "specific_month"}
        
        return None
    
    def _get_relationships(self, query: str) -> List[Dict[str, Any]]:
        """Get relevant table relationships"""
        relationships = []
        
        # Add common relationships
        relationships.append({
            "source": "FactAllIndiaDailySummary",
            "target": "DimRegions",
            "type": "many_to_one",
            "condition": "RegionID"
        })
        
        relationships.append({
            "source": "FactAllIndiaDailySummary",
            "target": "DimDates",
            "type": "many_to_one",
            "condition": "DateID"
        })
        
        # Add business logic for energy shortage queries
        if "shortage" in query.lower():
            relationships.append({
                "source": "FactAllIndiaDailySummary",
                "target": "EnergyShortage",
                "type": "metric",
                "condition": "Use EnergyShortage column for shortage analysis"
            })
        
        return relationships
    
    def _get_semantic_mappings(self, query: str) -> Dict[str, str]:
        """Get semantic mappings for query"""
        mappings = {}
        query_lower = query.lower()
        
        # Map business terms to technical terms
        if "energy shortage" in query_lower:
            mappings["energy_shortage"] = "EnergyShortage"
            # Don't restrict to specific region names for shortage queries
            if "all regions" in query_lower or "regions" in query_lower:
                mappings["all_regions"] = "Remove region filter"
        
        if "region" in query_lower and "all" in query_lower:
            mappings["all_regions"] = "Remove region filter"
        
        if "june" in query_lower:
            mappings["june"] = "Month = 6"
        
        if "2025" in query:
            mappings["2025"] = "Year = 2025"
        
        return mappings
    
    def _calculate_vector_similarity(self, query: str) -> float:
        """Calculate vector similarity for query"""
        try:
            # Encode query
            query_embedding = self.sentence_transformer.encode(query)
            
            # Search for similar vectors
            results = self.vector_client.search(
                collection_name="schema_semantics",
                query_vector=query_embedding.tolist(),
                limit=1
            )
            
            if results:
                return results[0].score
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Failed to calculate vector similarity: {e}")
            return 0.5
    
    def _calculate_confidence(self, intent: IntentType, business_entities: List, 
                            domain_concepts: List, temporal_context: Optional[Dict], 
                            relationships: List, vector_similarity: float) -> float:
        """Calculate overall confidence score"""
        confidence = 0.0
        
        # Base confidence from intent
        if intent == IntentType.DATA_RETRIEVAL:
            confidence += 0.3
        
        # Add confidence from business entities
        confidence += min(len(business_entities) * 0.2, 0.4)
        
        # Add confidence from domain concepts
        confidence += min(len(domain_concepts) * 0.1, 0.2)
        
        # Add confidence from temporal context
        if temporal_context:
            confidence += 0.2
        
        # Add confidence from relationships
        confidence += min(len(relationships) * 0.1, 0.2)
        
        # Add confidence from vector similarity
        confidence += vector_similarity * 0.3
        
        return min(confidence, 1.0)

    async def generate_contextual_sql(self, query: str, semantic_context: Dict[str, Any], options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate contextual SQL using semantic understanding and domain knowledge.
        This method is required by the enhanced RAG service.
        """
        try:
            logger.info(f"🔍 CloudSemanticEngine: Generating contextual SQL for query: {query[:100]}...")
            
            # Extract business entities and intent
            business_entities = self._extract_business_entities(query)
            intent = self._analyze_intent(query)
            domain_concepts = self._get_domain_concepts(query)
            temporal_context = self._extract_temporal_context(query)
            relationships = self._get_relationships(query)
            semantic_mappings = self._get_semantic_mappings(query)
            vector_similarity = self._calculate_vector_similarity(query)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                intent, business_entities, domain_concepts, 
                temporal_context, relationships, vector_similarity
            )
            
            # Generate SQL based on intent and context
            sql = self._generate_sql_from_context(
                query, business_entities, intent, domain_concepts,
                temporal_context, relationships, semantic_mappings
            )
            
            if sql:
                logger.info(f"✅ CloudSemanticEngine: SQL generated successfully with confidence {confidence:.2f}")
                return {
                    "sql": sql,
                    "confidence": confidence,
                    "business_entities": business_entities,
                    "intent": intent.value if hasattr(intent, 'value') else str(intent),
                    "semantic_mappings": semantic_mappings,
                    "relationships": relationships
                }
            else:
                logger.warning(f"⚠️ CloudSemanticEngine: No SQL generated")
                return {
                    "sql": None,
                    "confidence": confidence,
                    "error": "Unable to generate SQL from context"
                }
                
        except Exception as e:
            logger.error(f"❌ CloudSemanticEngine: Error generating contextual SQL: {e}")
            return {
                "sql": None,
                "confidence": 0.0,
                "error": f"Error: {str(e)}"
            }
    
    def _generate_sql_from_context(self, query: str, business_entities: List, intent: IntentType,
                                  domain_concepts: List, temporal_context: Optional[Dict],
                                  relationships: List, semantic_mappings: Dict) -> Optional[str]:
        """Generate SQL from semantic context"""
        try:
            # Simple SQL generation based on context
            if "energy shortage" in query.lower():
                # Generate energy shortage query
                sql = """
                SELECT r.RegionName, ROUND(SUM(fs.EnergyShortage), 2) AS TotalEnergyShortage
                FROM FactAllIndiaDailySummary fs
                JOIN DimRegions r ON fs.RegionID = r.RegionID
                JOIN DimDates d ON fs.DateID = d.DateID
                """
                
                # Add temporal constraints
                if temporal_context and temporal_context.get("year"):
                    sql += f" WHERE DATEPART(YEAR, d.ActualDate) = {temporal_context['year']}"
                
                sql += " GROUP BY r.RegionName ORDER BY TotalEnergyShortage DESC"
                return sql
            
            elif "generation" in query.lower() and "source" in query.lower():
                # Generate generation by source query
                sql = """
                SELECT dgs.SourceName, ROUND(SUM(fdgb.GenerationAmount), 2) AS TotalGeneration
                FROM FactDailyGenerationBreakdown fdgb
                JOIN DimGenerationSources dgs ON fdgb.GenerationSourceID = dgs.GenerationSourceID
                JOIN DimDates dt ON fdgb.DateID = dt.DateID
                """
                
                # Add temporal constraints
                if temporal_context and temporal_context.get("year"):
                    sql += f" WHERE DATEPART(YEAR, dt.ActualDate) = {temporal_context['year']}"
                
                sql += " GROUP BY dgs.SourceName ORDER BY TotalGeneration DESC"
                return sql
            
            elif "time block" in query.lower() or "hourly" in query.lower():
                # Generate time block query
                if "generation" in query.lower():
                    sql = """
                    SELECT dgs.SourceName, ftbg.BlockNumber, ROUND(SUM(ftbg.GenerationOutput), 2) AS TotalGeneration
                    FROM FactTimeBlockGeneration ftbg
                    JOIN DimGenerationSources dgs ON ftbg.GenerationSourceID = dgs.GenerationSourceID
                    JOIN DimDates dt ON ftbg.DateID = dt.DateID
                    """
                else:
                    sql = """
                    SELECT FORMAT(d.ActualDate, 'yyyy-MM-dd') AS Date, ftbpd.BlockTime, 
                           ROUND(SUM(ftbpd.TotalGeneration), 2) AS TotalGeneration
                    FROM FactTimeBlockPowerData ftbpd
                    JOIN DimDates d ON ftbpd.DateID = d.DateID
                    """
                
                # Add temporal constraints
                if temporal_context and temporal_context.get("year"):
                    sql += f" WHERE DATEPART(YEAR, dt.ActualDate) = {temporal_context['year']}"
                
                sql += " GROUP BY dgs.SourceName, ftbg.BlockNumber ORDER BY TotalGeneration DESC" if "generation" in query.lower() else " GROUP BY d.ActualDate, ftbpd.BlockTime ORDER BY d.ActualDate, ftbpd.BlockNumber"
                return sql
            
            # Default fallback
            return None
            
        except Exception as e:
            logger.error(f"Error generating SQL from context: {e}")
            return None
