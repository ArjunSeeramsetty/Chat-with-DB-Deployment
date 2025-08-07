"""
Semantic Engine for Enhanced SQL Generation
Implements advanced semantic understanding with vector search and context awareness
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pandas as pd

from backend.core.types import QueryAnalysis, IntentType, QueryType
from backend.core.llm_provider import LLMProvider

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


class SemanticEngine:
    """
    Advanced semantic engine with business context understanding
    Integrates vector search, domain modeling, and intelligent context retrieval
    """
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_client = QdrantClient(":memory:")  # In-memory for development
        self.domain_model: Optional[DomainModel] = None
        self._initialize_vector_store()
        
    async def initialize(self):
        """Initialize the semantic engine with domain knowledge"""
        await self._load_domain_model()
        await self._populate_vector_store()
        logger.info("Semantic engine initialized successfully")
        
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
            
    async def _load_domain_model(self):
        """Load energy domain model with business semantics"""
        # Energy domain model definition
        energy_tables = {
            "FactAllIndiaDailySummary": {
                "business_name": "National Energy Summary",
                "description": "Daily energy generation and consumption data at national level",
                "key_metrics": ["EnergyMet", "EnergyRequirement", "Surplus", "Deficit"],
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
        
        energy_relationships = [
            MDLRelationship(
                source_table="FactAllIndiaDailySummary",
                target_table="DimRegions",
                join_type="INNER",
                join_conditions=["f.RegionID = d.RegionID"],
                relationship_type="many-to-one",
                business_meaning="Each daily summary belongs to a specific region"
            ),
            MDLRelationship(
                source_table="FactStateDailyEnergy",
                target_table="DimStates",
                join_type="INNER",
                join_conditions=["f.StateID = d.StateID"],
                relationship_type="many-to-one",
                business_meaning="Each state energy record belongs to a specific state"
            )
        ]
        
        business_glossary = {
            "energy_met": "Total energy supplied to meet demand",
            "energy_requirement": "Total energy demand/requirement",
            "surplus": "Excess energy available beyond requirement",
            "deficit": "Shortfall in energy supply compared to requirement",
            "plf": "Plant Load Factor - capacity utilization percentage",
            "peak_demand": "Maximum energy demand in a day",
            "generation": "Energy production from various sources",
            "renewable": "Energy from solar, wind, hydro sources",
            "thermal": "Energy from coal, gas, nuclear sources"
        }
        
        energy_metrics = {
            "growth_rate": {
                "calculation": "((current - previous) / previous) * 100",
                "unit": "percentage",
                "business_meaning": "Period-over-period percentage change"
            },
            "capacity_utilization": {
                "calculation": "generation / installed_capacity",
                "unit": "percentage", 
                "business_meaning": "Efficiency of power plant utilization"
            }
        }
        
        self.domain_model = DomainModel(
            tables=energy_tables,
            relationships=energy_relationships,
            business_glossary=business_glossary,
            metrics=energy_metrics,
            dimensions={}
        )
        
        logger.info("Energy domain model loaded successfully")
        
    async def _populate_vector_store(self):
        """Populate vector store with domain knowledge"""
        if not self.domain_model:
            return
            
        # Index schema semantics
        schema_points = []
        for table_name, table_info in self.domain_model.tables.items():
            # Create embeddings for table and column semantics
            table_text = f"{table_info['business_name']}: {table_info['description']}"
            embedding = self.sentence_transformer.encode(table_text).tolist()
            
            schema_points.append(PointStruct(
                id=len(schema_points),
                vector=embedding,
                payload={
                    "type": "table",
                    "name": table_name,
                    "business_name": table_info['business_name'],
                    "description": table_info['description'],
                    "metrics": table_info['key_metrics'],
                    "dimensions": table_info['dimensions']
                }
            ))
            
        # Index business glossary
        for term, definition in self.domain_model.business_glossary.items():
            embedding = self.sentence_transformer.encode(f"{term}: {definition}").tolist()
            schema_points.append(PointStruct(
                id=len(schema_points),
                vector=embedding,
                payload={
                    "type": "glossary",
                    "term": term,
                    "definition": definition
                }
            ))
            
        # Upload to vector store
        self.vector_client.upsert(
            collection_name="schema_semantics",
            points=schema_points
        )
        
        # Index common query patterns
        query_patterns = [
            "monthly growth of energy generation by region",
            "state wise energy consumption analysis",
            "renewable energy capacity utilization trends",
            "peak demand forecasting by state",
            "thermal vs renewable generation comparison",
            "energy deficit analysis by region",
            "seasonal energy demand patterns"
        ]
        
        pattern_points = []
        for i, pattern in enumerate(query_patterns):
            embedding = self.sentence_transformer.encode(pattern).tolist()
            pattern_points.append(PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "pattern": pattern,
                    "category": self._categorize_pattern(pattern)
                }
            ))
            
        self.vector_client.upsert(
            collection_name="query_patterns",
            points=pattern_points
        )
        
        logger.info(f"Populated vector store with {len(schema_points)} schema points and {len(pattern_points)} pattern points")
        
    def _categorize_pattern(self, pattern: str) -> str:
        """Categorize query patterns for better retrieval"""
        pattern_lower = pattern.lower()
        if any(word in pattern_lower for word in ["growth", "trend", "change"]):
            return "trend_analysis"
        elif any(word in pattern_lower for word in ["comparison", "vs", "compare"]):
            return "comparison"
        elif any(word in pattern_lower for word in ["forecast", "predict"]):
            return "forecasting"
        else:
            return "descriptive"
            
    async def extract_semantic_context(self, natural_language_query: str) -> SemanticContext:
        """
        Extract rich semantic context from natural language query
        Uses vector search and LLM analysis for comprehensive understanding
        """
        try:
            # Step 1: Vector similarity search for relevant context
            query_embedding = self.sentence_transformer.encode(natural_language_query).tolist()
            
            # Search schema semantics
            schema_results = self.vector_client.search(
                collection_name="schema_semantics",
                query_vector=query_embedding,
                limit=5
            )
            
            # Search query patterns  
            pattern_results = self.vector_client.search(
                collection_name="query_patterns", 
                query_vector=query_embedding,
                limit=3
            )
            
            # Step 2: LLM-powered semantic analysis
            semantic_analysis = await self._llm_semantic_analysis(
                natural_language_query, schema_results, pattern_results
            )
            
            # Step 3: Extract business entities and relationships
            business_entities = self._extract_business_entities(
                natural_language_query, schema_results
            )
            
            # Step 4: Determine confidence level
            confidence = self._calculate_semantic_confidence(
                schema_results, pattern_results, semantic_analysis
            )
            
            # Step 5: Build semantic mappings
            semantic_mappings = self._build_semantic_mappings(
                natural_language_query, business_entities, schema_results
            )
            
            return SemanticContext(
                intent=self._determine_intent(semantic_analysis),
                confidence=confidence,
                business_entities=business_entities,
                domain_concepts=self._extract_domain_concepts(semantic_analysis),
                temporal_context=self._extract_temporal_context(natural_language_query),
                relationships=self._extract_relationships(business_entities),
                semantic_mappings=semantic_mappings,
                vector_similarity=max([r.score for r in schema_results]) if schema_results else 0.0
            )
            
        except Exception as e:
            logger.error(f"Failed to extract semantic context: {e}")
            # Return fallback context
            return SemanticContext(
                intent=IntentType.DATA_RETRIEVAL,
                confidence=0.3,
                business_entities=[],
                domain_concepts=[],
                temporal_context=None,
                relationships=[],
                semantic_mappings={},
                vector_similarity=0.0
            )
            
    async def _llm_semantic_analysis(self, query: str, schema_results: List, pattern_results: List) -> Dict[str, Any]:
        """Use LLM for semantic analysis"""
        try:
            prompt = f"""
            Analyze the following natural language query for semantic understanding:
            
            Query: "{query}"
            
            Available schema information:
            {json.dumps(schema_results, indent=2)}
            
            Query patterns found:
            {json.dumps(pattern_results, indent=2)}
            
            Provide analysis in JSON format with:
            - intent: (aggregation, filtering, time_series, comparison, growth)
            - confidence: (0.0-1.0)
            - business_entities: [list of business concepts]
            - domain_concepts: [list of domain-specific terms]
            - temporal_context: {{"time_period": "monthly", "year": 2024}}
            - relationships: [list of table relationships needed]
            
            Return only valid JSON:
            """
            
            response = await self.llm_provider.generate(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            try:
                return json.loads(response_text.strip())
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON, using fallback")
                return {
                    "intent": "aggregation",
                    "confidence": 0.5,
                    "business_entities": [],
                    "domain_concepts": [],
                    "temporal_context": None,
                    "relationships": []
                }
                
        except Exception as e:
            logger.error(f"LLM semantic analysis failed: {e}")
            return {
                "intent": "aggregation",
                "confidence": 0.3,
                "business_entities": [],
                "domain_concepts": [],
                "temporal_context": None,
                "relationships": []
            }
            
    def _extract_business_entities(self, query: str, schema_results: List) -> List[Dict[str, Any]]:
        """Extract business entities from query using schema context"""
        entities = []
        query_lower = query.lower()
        
        # Extract from schema results
        for result in schema_results:
            payload = result.payload
            if payload['type'] == 'table':
                # Check if table is relevant
                if any(metric.lower() in query_lower for metric in payload.get('metrics', [])):
                    entities.append({
                        "type": "table",
                        "name": payload['name'],
                        "business_name": payload['business_name'],
                        "relevance_score": result.score,
                        "matched_metrics": [
                            m for m in payload.get('metrics', []) 
                            if m.lower() in query_lower
                        ]
                    })
                    
        # Extract domain-specific entities
        domain_entities = {
            "regions": ["northern", "southern", "eastern", "western", "north eastern"],
            "sources": ["coal", "solar", "wind", "hydro", "nuclear", "thermal", "renewable"],
            "metrics": ["generation", "consumption", "demand", "capacity", "energy met"],
            "time_periods": ["monthly", "daily", "yearly", "quarterly", "annual"]
        }
        
        for entity_type, keywords in domain_entities.items():
            matched = [kw for kw in keywords if kw in query_lower]
            if matched:
                entities.append({
                    "type": entity_type,
                    "values": matched,
                    "confidence": len(matched) / len(keywords)
                })
                
        return entities
        
    def _calculate_semantic_confidence(self, schema_results: List, pattern_results: List, llm_analysis: Dict) -> float:
        """Calculate overall confidence in semantic understanding"""
        confidence_factors = []
        
        # Vector similarity confidence
        if schema_results:
            max_schema_score = max([r.score for r in schema_results])
            confidence_factors.append(max_schema_score * 0.4)
            
        if pattern_results:
            max_pattern_score = max([r.score for r in pattern_results])
            confidence_factors.append(max_pattern_score * 0.3)
            
        # LLM analysis confidence
        llm_confidence = len(llm_analysis.get('key_entities', [])) * 0.1
        confidence_factors.append(min(llm_confidence, 0.3))
        
        return min(sum(confidence_factors), 1.0)
        
    def _determine_intent(self, llm_analysis: Dict) -> IntentType:
        """Determine query intent from semantic analysis"""
        intent_mapping = {
            "data_retrieval": IntentType.DATA_RETRIEVAL,
            "trend_analysis": IntentType.TREND_ANALYSIS,
            "comparison": IntentType.COMPARISON,
            "aggregation": IntentType.DATA_RETRIEVAL
        }
        
        intent_str = llm_analysis.get('intent', 'data_retrieval')
        return intent_mapping.get(intent_str, IntentType.DATA_RETRIEVAL)
        
    def _extract_domain_concepts(self, llm_analysis: Dict) -> List[str]:
        """Extract domain-specific concepts"""
        concepts = []
        concepts.extend(llm_analysis.get('key_entities', []))
        concepts.extend(llm_analysis.get('metric_indicators', []))
        return list(set(concepts))
        
    def _extract_temporal_context(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract temporal context from query"""
        query_lower = query.lower()
        
        temporal_indicators = {
            "monthly": {"period": "month", "aggregation": "monthly"},
            "daily": {"period": "day", "aggregation": "daily"},
            "yearly": {"period": "year", "aggregation": "yearly"},
            "quarterly": {"period": "quarter", "aggregation": "quarterly"},
            "annual": {"period": "year", "aggregation": "yearly"}
        }
        
        for indicator, context in temporal_indicators.items():
            if indicator in query_lower:
                return context
                
        # Extract specific years/dates
        import re
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            return {
                "period": "year", 
                "specific_year": int(year_match.group(1)),
                "aggregation": "yearly"
            }
            
        return None
        
    def _extract_relationships(self, business_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between business entities"""
        relationships = []
        
        # Find table entities
        table_entities = [e for e in business_entities if e.get('type') == 'table']
        
        if self.domain_model:
            for relationship in self.domain_model.relationships:
                # Check if both tables are relevant
                source_relevant = any(
                    e['name'] == relationship.source_table for e in table_entities
                )
                target_relevant = any(
                    e['name'] == relationship.target_table for e in table_entities
                )
                
                if source_relevant or target_relevant:
                    relationships.append({
                        "source": relationship.source_table,
                        "target": relationship.target_table,
                        "type": relationship.relationship_type,
                        "join_conditions": relationship.join_conditions,
                        "business_meaning": relationship.business_meaning
                    })
                    
        return relationships
        
    def _build_semantic_mappings(self, query: str, business_entities: List, schema_results: List) -> Dict[str, str]:
        """Build semantic mappings from natural language to schema elements"""
        mappings = {}
        query_lower = query.lower()
        
        # Map common terms to schema elements
        term_mappings = {
            "energy met": "EnergyMet",
            "energy requirement": "EnergyRequirement", 
            "generation": "GenerationMW",
            "consumption": "EnergyConsumption",
            "capacity": "Capacity",
            "demand": "PeakDemand",
            "surplus": "Surplus",
            "deficit": "Deficit"
        }
        
        for term, column in term_mappings.items():
            if term in query_lower:
                mappings[term] = column
                
        # Map from schema results
        for result in schema_results:
            if result.payload['type'] == 'glossary':
                term = result.payload['term']
                if term in query_lower:
                    mappings[term] = result.payload['definition']
                    
        return mappings
        
    async def retrieve_schema_context(self, semantic_context: SemanticContext) -> Dict[str, Any]:
        """Retrieve relevant schema context based on semantic understanding"""
        
        # Use business entities to determine relevant tables
        relevant_tables = []
        for entity in semantic_context.business_entities:
            if entity.get('type') == 'table':
                table_name = entity['name']
                if table_name in self.domain_model.tables:
                    table_info = self.domain_model.tables[table_name]
                    relevant_tables.append({
                        "name": table_name,
                        "info": table_info,
                        "relevance_score": entity.get('relevance_score', 0.0)
                    })
                    
        # Sort by relevance
        relevant_tables.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Build comprehensive schema context
        schema_context = {
            "primary_table": relevant_tables[0] if relevant_tables else None,
            "related_tables": relevant_tables[1:3] if len(relevant_tables) > 1 else [],
            "relationships": semantic_context.relationships,
            "domain_model": self.domain_model,
            "confidence": semantic_context.confidence
        }
        
        return schema_context
        
    async def generate_contextual_sql(
        self, 
        natural_language_query: str, 
        semantic_context: Union[SemanticContext, Dict], 
        schema_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate SQL with full semantic context"""
        
        # Handle both SemanticContext object and dictionary
        if isinstance(semantic_context, SemanticContext):
            # Use SemanticContext object directly
            generation_context = {
                "query": natural_language_query,
                "intent": semantic_context.intent.value,
                "confidence": semantic_context.confidence,
                "semantic_mappings": semantic_context.semantic_mappings,
                "business_entities": semantic_context.business_entities,
                "temporal_context": semantic_context.temporal_context,
                "primary_table": schema_context.get("primary_table"),
                "relationships": schema_context.get("relationships", []),
                "domain_concepts": semantic_context.domain_concepts
            }
            confidence = semantic_context.confidence
        else:
            # Handle dictionary case
            generation_context = {
                "query": natural_language_query,
                "intent": semantic_context.get("intent", "aggregation"),
                "confidence": semantic_context.get("confidence", 0.5),
                "semantic_mappings": semantic_context.get("semantic_mappings", {}),
                "business_entities": semantic_context.get("business_entities", []),
                "temporal_context": semantic_context.get("temporal_context", {}),
                "primary_table": schema_context.get("primary_table"),
                "relationships": schema_context.get("relationships", []),
                "domain_concepts": semantic_context.get("domain_concepts", [])
            }
            confidence = semantic_context.get("confidence", 0.5)
        
        # Use LLM for contextual SQL generation
        sql_prompt = self._build_sql_generation_prompt(generation_context)
        
        try:
            sql_response = await self.llm_provider.generate(sql_prompt)
            response_text = sql_response.content if hasattr(sql_response, 'content') else str(sql_response)
            
            # Parse and validate SQL response
            sql_result = self._parse_sql_response(response_text)
            
            return {
                "sql": sql_result.get("sql", ""),
                "explanation": sql_result.get("explanation", ""),
                "confidence": confidence,
                "context_used": generation_context,
                "semantic_mappings": generation_context.get("semantic_mappings", {})
            }
            
        except Exception as e:
            logger.error(f"Contextual SQL generation failed: {e}")
            return {
                "sql": "",
                "explanation": "Failed to generate SQL",
                "confidence": 0.0,
                "error": str(e)
            }
            
    def _build_sql_generation_prompt(self, context: Dict[str, Any]) -> str:
        """Build comprehensive prompt for SQL generation"""
        
        primary_table = context.get("primary_table")
        table_info = primary_table["info"] if primary_table else {}
        
        prompt = f"""
        Generate a complete SQLite-compatible SQL query based on semantic analysis:
        
        Natural Language Query: "{context['query']}"
        
        Intent: {context['intent']}
        Confidence: {context['confidence']:.2f}
        
        Primary Table: {primary_table['name'] if primary_table else 'Unknown'}
        Table Description: {table_info.get('description', '')}
        Available Metrics: {', '.join(table_info.get('key_metrics', []))}
        Available Dimensions: {', '.join(table_info.get('dimensions', []))}
        
        Semantic Mappings:
        {json.dumps(context['semantic_mappings'], indent=2)}
        
        Temporal Context: {context.get('temporal_context', 'None')}
        
        Business Entities:
        {json.dumps(context['business_entities'], indent=2)}
        
        Domain Concepts: {', '.join(context['domain_concepts'])}
        
        IMPORTANT REQUIREMENTS:
        1. Generate a COMPLETE SQL query starting with SELECT
        2. Use SQLite-compatible syntax only
        3. Use strftime() for date functions (e.g., strftime('%Y-%m', d.ActualDate))
        4. Avoid window functions (LAG, LEAD, ROW_NUMBER) - use subqueries instead
        5. Use self-joins for growth calculations
        6. Use proper table aliases (f for fact tables, d for dimension tables)
        7. Use SUM(), AVG(), MAX(), MIN() for aggregations
        8. Use GROUP BY for grouping
        9. Use ORDER BY for sorting
        10. Include all necessary JOINs, WHERE clauses, and GROUP BY
        
        For growth calculations, use this exact pattern:
        - Use LEFT JOIN with a subquery to get previous month data
        - Calculate growth as: ((current - previous) / previous) * 100
        - Use CASE statements to handle division by zero
        
        Example structure for monthly growth:
        SELECT 
            r.RegionName,
            strftime('%Y-%m', d.ActualDate) as Month,
            SUM(fs.EnergyMet) as TotalEnergyMet,
            prev.PreviousMonthEnergy,
            CASE 
                WHEN prev.PreviousMonthEnergy > 0 
                THEN ((SUM(fs.EnergyMet) - prev.PreviousMonthEnergy) / prev.PreviousMonthEnergy) * 100 
                ELSE 0 
            END as GrowthRate
        FROM FactAllIndiaDailySummary fs
        JOIN DimRegions r ON fs.RegionID = r.RegionID
        JOIN DimDates d ON fs.DateID = d.DateID
        LEFT JOIN (
            SELECT 
                r2.RegionName,
                strftime('%Y-%m', d2.ActualDate) as Month,
                SUM(fs2.EnergyMet) as PreviousMonthEnergy
            FROM FactAllIndiaDailySummary fs2
            JOIN DimRegions r2 ON fs2.RegionID = r2.RegionID
            JOIN DimDates d2 ON fs2.DateID = d2.DateID
            WHERE strftime('%Y', d2.ActualDate) = '2024'
            GROUP BY r2.RegionName, strftime('%Y-%m', d2.ActualDate)
        ) prev ON r.RegionName = prev.RegionName 
            AND strftime('%Y-%m', d.ActualDate) = date(prev.Month || '-01', '+1 month')
        WHERE strftime('%Y', d.ActualDate) = '2024'
        GROUP BY r.RegionName, strftime('%Y-%m', d.ActualDate)
        ORDER BY r.RegionName, Month
        
        CRITICAL: Use the exact table names and column names from the schema:
        - FactAllIndiaDailySummary (alias: fs) - contains EnergyMet column
        - DimRegions (alias: r) - contains RegionName column  
        - DimDates (alias: d) - contains ActualDate column (NOT Date)
        - Use fs.EnergyMet for energy metrics
        - Use r.RegionName for region names
        - Use d.ActualDate for dates (NOT d.Date)
        - Use fs.RegionID for joining with DimRegions
        - Use fs.DateID for joining with DimDates
        - Use fs2.RegionID and fs2.DateID for subquery joins
        
        IMPORTANT: All JOINs must use the correct foreign key relationships:
        - JOIN DimRegions r ON fs.RegionID = r.RegionID
        - JOIN DimDates d ON fs.DateID = d.DateID
        - In subqueries: JOIN DimRegions r2 ON fs2.RegionID = r2.RegionID
        - In subqueries: JOIN DimDates d2 ON fs2.DateID = d2.DateID
        
        Generate a complete SQL query that:
        1. Starts with SELECT and includes all necessary clauses
        2. Uses the most relevant table and columns
        3. Applies appropriate joins based on relationships
        4. Includes proper temporal filtering if needed
        5. Uses semantic mappings for column selection
        6. Applies business logic understanding
        7. Uses SQLite-compatible syntax only
        8. Is complete and executable
        9. Uses the exact table and column names specified above
        
        Return only the SQL query without any explanation, formatting, or code blocks:
        """
        
        return prompt
        
    def _parse_sql_response(self, response: str) -> Dict[str, Any]:
        """Parse SQL generation response"""
        try:
            # Try to parse as JSON first
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Fallback: extract SQL from text response
            sql = self._extract_sql_from_text(response)
            
            return {
                "sql": sql,
                "explanation": "Generated from text response",
                "tables_used": [],
                "columns_used": [],
                "business_logic": "Basic extraction"
            }
    
    def _extract_sql_from_text(self, text: str) -> str:
        """Extract SQL from text response with multiple fallback strategies"""
        # Strategy 1: Look for SQL code blocks
        sql_blocks = []
        
        # Find ```sql blocks
        start_markers = ["```sql", "```SQL", "```"]
        end_markers = ["```"]
        
        for start_marker in start_markers:
            start_idx = text.find(start_marker)
            if start_idx != -1:
                start_pos = start_idx + len(start_marker)
                # Find the end marker
                for end_marker in end_markers:
                    end_idx = text.find(end_marker, start_pos)
                    if end_idx != -1:
                        sql_block = text[start_pos:end_idx].strip()
                        if sql_block and ('SELECT' in sql_block.upper() or 'FROM' in sql_block.upper()):
                            sql_blocks.append(sql_block)
                        break
        
        # Strategy 2: Look for SQL after "Return only the SQL query:"
        if "Return only the SQL query:" in text:
            parts = text.split("Return only the SQL query:")
            if len(parts) > 1:
                sql_part = parts[1].strip()
                # Extract until the next newline or end
                lines = sql_part.split('\n')
                sql_lines = []
                for line in lines:
                    if line.strip() and ('SELECT' in line.upper() or 'FROM' in line.upper() or line.strip().startswith('--')):
                        sql_lines.append(line)
                    elif sql_lines and line.strip():
                        sql_lines.append(line)
                    elif sql_lines and not line.strip():
                        break
                if sql_lines:
                    sql_blocks.append('\n'.join(sql_lines))
        
        # Strategy 3: Extract lines that look like SQL (improved)
        lines = text.split('\n')
        sql_lines = []
        in_sql = False
        sql_started = False
        
        for line in lines:
            line_upper = line.upper().strip()
            line_stripped = line.strip()
            
            # Check if this line starts SQL
            if 'SELECT' in line_upper and not sql_started:
                in_sql = True
                sql_started = True
            elif in_sql and line_upper.startswith(('SELECT', 'FROM', 'JOIN', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'UNION', 'WITH')):
                in_sql = True
            elif in_sql and line_stripped and not line_stripped.startswith('--'):
                # Continue if it looks like SQL (has keywords or is part of a statement)
                if any(keyword in line_upper for keyword in ['SELECT', 'FROM', 'JOIN', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'UNION', 'WITH', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AS', 'ON', 'AND', 'OR', 'IN', 'LIKE', 'BETWEEN', 'IS', 'NULL', 'NOT', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'STRFTIME', 'DATE', 'LEFT', 'RIGHT', 'INNER', 'OUTER']):
                    in_sql = True
                elif line_stripped.endswith(';') or line_stripped.endswith(','):
                    in_sql = True
                elif any(char in line_stripped for char in ['(', ')', '=', '>', '<', '+', '-', '*', '/', '||']):
                    in_sql = True
                elif line_stripped.startswith(('AND', 'OR', 'ON', 'IN', 'LIKE', 'BETWEEN')):
                    in_sql = True
                else:
                    # Check if this might be the end of SQL
                    if not any(char in line_stripped for char in ['(', ')', '=', '>', '<', '+', '-', '*', '/', '||', ',', ';']):
                        # If we have a substantial SQL block, continue; otherwise stop
                        if len(sql_lines) < 3:
                            in_sql = False
                        elif not any(keyword in line_upper for keyword in ['SELECT', 'FROM', 'JOIN', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'UNION', 'WITH', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AS', 'ON', 'AND', 'OR', 'IN', 'LIKE', 'BETWEEN', 'IS', 'NULL', 'NOT', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'STRFTIME', 'DATE', 'LEFT', 'RIGHT', 'INNER', 'OUTER']):
                            in_sql = False
            
            if in_sql:
                sql_lines.append(line)
        
        if sql_lines:
            sql_blocks.append('\n'.join(sql_lines))
        
        # Return the best SQL block found
        if sql_blocks:
            # Prefer the longest SQL block that starts with SELECT
            valid_blocks = [block for block in sql_blocks if block.strip().upper().startswith('SELECT')]
            if valid_blocks:
                return max(valid_blocks, key=len)
            else:
                # If no block starts with SELECT, return the longest one
                return max(sql_blocks, key=len)
        
        # Strategy 4: Last resort - extract anything that looks like SQL
        words = text.split()
        sql_words = []
        in_sql = False
        
        for i, word in enumerate(words):
            word_upper = word.upper()
            if 'SELECT' in word_upper:
                in_sql = True
            elif in_sql and word_upper in ['FROM', 'JOIN', 'WHERE', 'GROUP', 'ORDER', 'HAVING', 'UNION', 'WITH']:
                in_sql = True
            elif in_sql and word.strip().endswith(';'):
                sql_words.append(word)
                break
            
            if in_sql:
                sql_words.append(word)
        
        if sql_words:
            return ' '.join(sql_words)
        
        # If all else fails, return empty string
        return ""