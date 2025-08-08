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
from backend.core.schema_metadata import SchemaMetadataExtractor
from backend.core.few_shot_examples import FewShotExampleRepository, FewShotExampleRetriever
from backend.core.query_planner import MultiStepQueryPlanner, QueryComplexity
from backend.core.sql_templates import SQLTemplateEngine, TemplateContext, TemplateValidation

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
    
    def __init__(self, llm_provider: LLMProvider, db_path: str = None):
        self.llm_provider = llm_provider
        self.db_path = db_path
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_client = QdrantClient(":memory:")  # In-memory for development
        self.domain_model: Optional[DomainModel] = None
        
        # Initialize schema metadata extractor if db_path is provided
        self.schema_extractor = None
        if db_path:
            self.schema_extractor = SchemaMetadataExtractor(db_path)
            
        # Initialize few-shot example repository and retriever
        self.few_shot_repository = None
        self.few_shot_retriever = None
        if db_path:
            self.few_shot_repository = FewShotExampleRepository(db_path)
            self.few_shot_retriever = FewShotExampleRetriever(self.few_shot_repository)
            
        # Initialize multi-step query planner
        self.query_planner = MultiStepQueryPlanner(llm_provider)
        
        # Initialize SQL template engine for constrained SQL generation
        self.sql_template_engine = SQLTemplateEngine()
            
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
        """Use LLM for semantic analysis with robust error handling and fallbacks"""
        try:
            # Convert ScoredPoint objects to serializable dictionaries
            def convert_scored_points(results):
                converted = []
                for result in results:
                    if hasattr(result, 'payload') and hasattr(result, 'score'):
                        converted.append({
                            'payload': result.payload,
                            'score': float(result.score)
                        })
                    else:
                        converted.append(result)
                return converted
            
            schema_results_serializable = convert_scored_points(schema_results)
            pattern_results_serializable = convert_scored_points(pattern_results)
            
            prompt = f"""
            Analyze the following natural language query for semantic understanding.
            
            Query: "{query}"
            
            Available schema information:
            {json.dumps(schema_results_serializable, indent=2, default=str)}
            
            Query patterns found:
            {json.dumps(pattern_results_serializable, indent=2, default=str)}
            
            You must respond with ONLY valid JSON in this exact format:
            {{
                "intent": "aggregation",
                "confidence": 0.8,
                "business_entities": ["energy", "shortage", "region"],
                "domain_concepts": ["energy_met", "energy_shortage"],
                "temporal_context": {{"time_period": "daily", "year": 2024}},
                "relationships": []
            }}
            
            Rules:
            - intent must be one of: "aggregation", "filtering", "time_series", "comparison", "growth"
            - confidence must be a number between 0.0 and 1.0
            - business_entities must be a list of strings
            - domain_concepts must be a list of strings
            - temporal_context must be an object or null
            - relationships must be a list
            
            Return ONLY the JSON object, no other text:
            """
            
            # Wrap LLM call with robust error handling
            try:
                response = await self.llm_provider.generate(prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                if not response_text or response.error:
                    logger.warning(f"LLM returned empty response or error: {response.error}")
                    return self._fallback_entity_extraction(query)
                
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return self._fallback_entity_extraction(query)
            
            # Clean the response text
            response_text = response_text.strip()
            
            # Remove any markdown formatting
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON response
            try:
                parsed_response = json.loads(response_text)
                
                # Validate required fields
                required_fields = ['intent', 'confidence', 'business_entities', 'domain_concepts', 'relationships']
                for field in required_fields:
                    if field not in parsed_response:
                        raise ValueError(f"Missing required field: {field}")
                
                return parsed_response
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}. Response: {response_text[:200]}...")
                return self._fallback_entity_extraction(query)
                
        except Exception as e:
            logger.error(f"LLM semantic analysis failed: {e}")
            return self._fallback_entity_extraction(query)
    
    def _fallback_entity_extraction(self, query: str) -> Dict[str, Any]:
        """Fallback entity extraction when LLM fails"""
        query_lower = query.lower()
        
        # Extract entities based on keywords
        entities = []
        if any(word in query_lower for word in ['energy', 'power', 'electricity']):
            entities.append('energy')
        if any(word in query_lower for word in ['shortage', 'deficit', 'lack']):
            entities.append('shortage')
        if any(word in query_lower for word in ['consumption', 'usage', 'demand']):
            entities.append('consumption')
        if any(word in query_lower for word in ['region', 'state', 'area']):
            entities.append('region')
        if any(word in query_lower for word in ['date', 'time', 'period']):
            entities.append('date')
        
        # Determine intent
        intent = "aggregation"
        if any(word in query_lower for word in ['compare', 'vs', 'versus']):
            intent = "comparison"
        elif any(word in query_lower for word in ['trend', 'growth', 'change']):
            intent = "time_series"
        elif any(word in query_lower for word in ['filter', 'where', 'condition']):
            intent = "filtering"
        
        # Determine confidence based on entity count
        confidence = min(0.3 + (len(entities) * 0.1), 0.7)
        
        return {
            "intent": intent,
            "confidence": confidence,
            "business_entities": entities,
            "domain_concepts": entities,  # Use same entities as domain concepts
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
        
        # Handle case where domain_model might be None
        if self.domain_model and hasattr(self.domain_model, 'tables'):
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
        else:
            # Fallback: use schema metadata if available
            if self.schema_extractor:
                try:
                    schema_metadata = self.schema_extractor.get_schema_metadata()
                    for table in schema_metadata.get('tables', []):
                        relevant_tables.append({
                            "name": table['name'],
                            "info": {
                                "description": table['description'],
                                "row_count": table['row_count'],
                                "key_metrics": [],
                                "dimensions": []
                            },
                            "relevance_score": 0.5
                        })
                except Exception as e:
                    logger.warning(f"Could not retrieve schema metadata: {e}")
                    
        # Sort by relevance
        relevant_tables.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Build comprehensive schema context
        schema_context = {
            "primary_table": relevant_tables[0] if relevant_tables else None,
            "related_tables": relevant_tables[1:3] if len(relevant_tables) > 1 else [],
            "relationships": semantic_context.relationships,
            "domain_model": self.domain_model if self.domain_model else {},
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
        
        # Check if query is complex enough to warrant multi-step planning
        complexity = self._analyze_query_complexity(natural_language_query)
        
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            logger.info(f"Using multi-step query planning for {complexity.value} query")
            return await self._generate_sql_with_multi_step_planning(
                natural_language_query, generation_context, schema_context
            )
        else:
            # Use existing single-step approach for simple/moderate queries
            logger.info(f"Using single-step SQL generation for {complexity.value} query")
            return await self._generate_sql_single_step(generation_context)
    
    def _analyze_query_complexity(self, query: str) -> QueryComplexity:
        """Analyze query complexity to determine if multi-step planning is needed"""
        query_lower = query.lower()
        
        # Comprehensive aggregation keywords including synonyms
        aggregation_keywords = [
            'average', 'avg', 'sum', 'count', 'total', 'maximum', 'minimum', 
            'max', 'min', 'mean', 'median', 'mode', 'variance', 'std', 'standard deviation'
        ]
        
        # Comprehensive entity indicators including schema/column synonyms
        entity_indicators = [
            'region', 'state', 'date', 'generation', 'consumption', 'demand', 
            'shortage', 'deficit', 'supply', 'requirement', 'availability', 
            'capacity', 'utilization', 'efficiency', 'performance', 'trend',
            'growth', 'change', 'pattern', 'distribution', 'comparison'
        ]
        
        # Count complexity indicators
        aggregation_count = sum(kw in query_lower for kw in aggregation_keywords)
        entity_count = sum(ind in query_lower for ind in entity_indicators)
        
        complexity_score = 0
        
        # Aggregation scoring - more lenient
        if aggregation_count > 1:
            complexity_score += 2
        elif aggregation_count == 1:
            complexity_score += 0  # Single aggregation = SIMPLE
        
        # Entity scoring - more lenient
        if entity_count > 2:
            complexity_score += 2
        elif entity_count > 1:
            complexity_score += 1
        elif entity_count == 1:
            complexity_score += 0  # Single entity = SIMPLE
        
        # Time-based analysis
        if any(word in query_lower for word in ['trend', 'growth', 'change', 'over time', 'monthly', 'yearly', 'daily', 'weekly']):
            complexity_score += 1
        
        # Comparison logic - compositional queries (exclude 'by' as it's used for grouping)
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference', 'ratio', 'and']):
            complexity_score += 1
        
        # Additional rule: compositional queries with "vs", "and" get at least COMPLEX
        if any(word in query_lower for word in ("compare", "vs", "and")) and complexity_score < 2:
            complexity_score = 2  # at least COMPLEX
        
        # Conditional logic
        if any(word in query_lower for word in ['if', 'when', 'where', 'condition', 'filter']):
            complexity_score += 1
        
        # Grouping and aggregation combinations
        if 'group by' in query_lower or 'grouped' in query_lower:
            complexity_score += 1
        
        # Determine complexity level with adjusted thresholds
        if complexity_score >= 3:
            return QueryComplexity.VERY_COMPLEX
        elif complexity_score >= 2:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 1:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    async def _generate_sql_with_multi_step_planning(
        self, 
        query: str, 
        generation_context: Dict[str, Any], 
        schema_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate SQL using multi-step query planning for complex queries with robust error handling"""
        try:
            # Use the multi-step query planner with error handling
            try:
                planning_result = await self.query_planner.plan_and_execute(query, schema_context)
                
                if planning_result["success"]:
                    # Extract the final SQL from the planning result
                    # For now, we'll use the existing SQL generation as the final step
                    final_sql_result = await self._generate_sql_single_step(generation_context)
                    
                    return {
                        "sql": final_sql_result.get("sql", ""),
                        "explanation": f"Generated using multi-step planning with {len(planning_result['plan'].steps)} steps",
                        "confidence": planning_result["confidence"],
                        "context_used": generation_context,
                        "semantic_mappings": generation_context.get("semantic_mappings", {}),
                        "validation_passed": final_sql_result.get("validation_passed", False),
                        "planning_details": {
                            "plan_id": planning_result["plan"].query_id,
                            "complexity": planning_result["plan"].complexity.value,
                            "steps_count": len(planning_result["plan"].steps),
                            "execution_results": planning_result["execution_results"]
                        }
                    }
                else:
                    logger.warning("Multi-step planning failed, falling back to single-step approach")
                    return await self._generate_sql_single_step(generation_context)
                    
            except Exception as e:
                logger.error(f"Multi-step planning failed: {e}, falling back to single-step approach")
                return await self._generate_sql_single_step(generation_context)
                
        except Exception as e:
            logger.error(f"Multi-step planning failed: {e}, falling back to single-step approach")
            return await self._generate_sql_single_step(generation_context)
    
    async def _generate_sql_single_step(self, generation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL using constrained templates with fallback to LLM"""
        query = generation_context.get('query', '')
        
        # Phase 4.1: Try constrained SQL generation with templates first
        try:
            logger.info(f"Attempting constrained SQL generation with templates for query: {query}")
            template_sql, template_validation = self.sql_template_engine.generate_sql(query)
            
            if template_sql and template_validation.is_valid:
                logger.info(f"Template-based SQL generation successful with confidence: {template_validation.confidence}")
                return {
                    "sql": template_sql,
                    "confidence": template_validation.confidence,
                    "generation_method": "template",
                    "validation": {
                        "is_valid": template_validation.is_valid,
                        "errors": template_validation.errors,
                        "warnings": template_validation.warnings
                    },
                    "template_context": {
                        "query_type": "template_generated",
                        "validation_rules_applied": len(self.sql_template_engine.validation_rules)
                    }
                }
            else:
                logger.warning(f"Template generation failed: {template_validation.errors}")
                
        except Exception as e:
            logger.error(f"Template generation failed: {e}")
        
        # Fallback to LLM-based generation with enhanced validation
        logger.info("Falling back to LLM-based SQL generation")
        sql_prompt = self._build_sql_generation_prompt(generation_context)
        
        try:
            # Wrap LLM call with robust error handling
            try:
                sql_response = await self.llm_provider.generate(sql_prompt)
                response_text = sql_response.content if hasattr(sql_response, 'content') else str(sql_response)
                
                if not response_text or sql_response.error:
                    logger.warning(f"LLM returned empty response or error: {sql_response.error}")
                    return self._fallback_sql_generation(generation_context)
                
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return self._fallback_sql_generation(generation_context)
            
            # Parse and validate SQL response
            sql_result = self._parse_sql_response(response_text)
            generated_sql = sql_result.get("sql", "")
            
            # Enhanced validation with template-based rules
            query_lower = query.lower()
            
            # Determine expected aggregation function and column for validation
            aggregation_function = "SUM"  # Default
            if any(word in query_lower for word in ["average", "avg", "mean"]):
                aggregation_function = "AVG"
            elif any(word in query_lower for word in ["maximum", "max", "highest"]):
                aggregation_function = "MAX"
            elif any(word in query_lower for word in ["minimum", "min", "lowest"]):
                aggregation_function = "MIN"
            
            energy_column = "EnergyMet"  # Default
            if "shortage" in query_lower:
                energy_column = "EnergyShortage"
            elif "deficit" in query_lower:
                energy_column = "EnergyShortage"
            elif "demand" in query_lower:
                energy_column = "MaximumDemand"
            elif "generation" in query_lower:
                energy_column = "GenerationAmount"
            
            # Validate the generated SQL using template-based rules
            is_valid = self._validate_sql_against_instructions(generated_sql, aggregation_function, energy_column, query_lower)
            
            if not is_valid and generated_sql:
                logger.warning(f"Generated SQL failed validation, attempting retry with more explicit instructions")
                
                # Retry with more explicit instructions based on template rules
                retry_prompt = f"""
You are a SQL expert. The previous SQL generation was incorrect.

User request: "{query}"

CRITICAL REQUIREMENTS:
- Use {aggregation_function}() function
- Use {energy_column} column
- Do NOT use SUM() if average is requested
- Do NOT use EnergyMet if shortage is requested

EXAMPLE FOR THIS QUERY:
SELECT {aggregation_function}(fs.{energy_column}) AS Result
FROM FactAllIndiaDailySummary fs
JOIN DimRegions r ON fs.RegionID = r.RegionID
JOIN DimDates d ON fs.DateID = d.DateID

Generate ONLY the SQL query:
"""
                
                try:
                    retry_response = await self.llm_provider.generate(retry_prompt)
                    retry_text = retry_response.content if hasattr(retry_response, 'content') else str(retry_response)
                    
                    if not retry_text or retry_response.error:
                        logger.warning(f"Retry LLM call failed: {retry_response.error}")
                    else:
                        retry_sql_result = self._parse_sql_response(retry_text)
                        retry_sql = retry_sql_result.get("sql", "")
                        
                        # Validate retry SQL
                        retry_is_valid = self._validate_sql_against_instructions(retry_sql, aggregation_function, energy_column, query_lower)
                        
                        if retry_is_valid:
                            generated_sql = retry_sql
                            logger.info("Retry SQL generation successful")
                        else:
                            logger.error("Both initial and retry SQL generation failed validation")
                            
                except Exception as e:
                    logger.error(f"Retry LLM call failed: {e}")
            
            return {
                "sql": generated_sql,
                "confidence": 0.7 if is_valid else 0.3,
                "generation_method": "llm_with_validation",
                "validation": {
                    "is_valid": is_valid,
                    "errors": [] if is_valid else ["SQL validation failed"],
                    "warnings": []
                },
                "template_context": {
                    "query_type": "llm_generated",
                    "validation_rules_applied": 0
                }
            }
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return self._fallback_sql_generation(generation_context)
    
    def _fallback_sql_generation(self, generation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback SQL generation when LLM fails"""
        query_lower = generation_context['query'].lower()
        
        # Generate basic SQL based on query keywords
        sql = "SELECT * FROM FactAllIndiaDailySummary LIMIT 10"
        
        # Try to generate more specific SQL based on keywords
        if any(word in query_lower for word in ['average', 'avg', 'mean']):
            if 'shortage' in query_lower:
                sql = "SELECT AVG(EnergyShortage) as avg_shortage FROM FactAllIndiaDailySummary"
            elif 'consumption' in query_lower:
                sql = "SELECT AVG(EnergyMet) as avg_consumption FROM FactAllIndiaDailySummary"
            else:
                sql = "SELECT AVG(EnergyMet) as avg_energy FROM FactAllIndiaDailySummary"
        elif any(word in query_lower for word in ['total', 'sum']):
            if 'shortage' in query_lower:
                sql = "SELECT SUM(EnergyShortage) as total_shortage FROM FactAllIndiaDailySummary"
            elif 'consumption' in query_lower:
                sql = "SELECT SUM(EnergyMet) as total_consumption FROM FactAllIndiaDailySummary"
            else:
                sql = "SELECT SUM(EnergyMet) as total_energy FROM FactAllIndiaDailySummary"
        elif 'compare' in query_lower or 'vs' in query_lower:
            sql = "SELECT Region, AVG(EnergyShortage) as avg_shortage, AVG(EnergyMet) as avg_consumption FROM FactAllIndiaDailySummary GROUP BY Region"
        
        return {
            "sql": sql,
            "explanation": "Generated using fallback SQL generation",
            "confidence": 0.3,
            "context_used": generation_context,
            "semantic_mappings": generation_context.get("semantic_mappings", {}),
            "validation_passed": True,
            "error": "LLM failed, used fallback generation"
        }
        
    def _build_sql_generation_prompt(self, context: Dict[str, Any]) -> str:
        """Build comprehensive prompt for SQL generation with schema metadata injection and few-shot examples"""
        
        primary_table = context.get("primary_table")
        table_info = primary_table.get("info", {}) if primary_table else {}
        
        # Add schema metadata injection if available
        schema_context = ""
        if self.schema_extractor:
            try:
                schema_context = self.schema_extractor.build_schema_prompt_context(context['query'])
            except Exception as e:
                logger.warning(f"Failed to build schema context: {e}")
        
        # Add few-shot examples if available
        few_shot_examples = ""
        if self.few_shot_retriever:
            try:
                examples = self.few_shot_retriever.retrieve_examples_for_query(
                    context['query'], 
                    max_examples=3,
                    min_similarity=0.3
                )
                if examples:
                    few_shot_examples = self.few_shot_retriever.format_examples_for_prompt(examples)
                    logger.info(f"Retrieved {len(examples)} few-shot examples for query")
            except Exception as e:
                logger.warning(f"Failed to retrieve few-shot examples: {e}")
        
        # CRITICAL FIX: Detect aggregation function and column from query
        query_lower = context['query'].lower()
        
        # Determine aggregation function with more robust detection
        aggregation_function = "SUM"  # Default
        aggregation_keywords = []
        
        if any(word in query_lower for word in ["average", "avg", "mean"]):
            aggregation_function = "AVG"
            aggregation_keywords = [word for word in ["average", "avg", "mean"] if word in query_lower]
        elif any(word in query_lower for word in ["maximum", "max", "highest"]):
            aggregation_function = "MAX"
            aggregation_keywords = [word for word in ["maximum", "max", "highest"] if word in query_lower]
        elif any(word in query_lower for word in ["minimum", "min", "lowest"]):
            aggregation_function = "MIN"
            aggregation_keywords = [word for word in ["minimum", "min", "lowest"] if word in query_lower]
        elif any(word in query_lower for word in ["total", "sum"]):
            aggregation_function = "SUM"
            aggregation_keywords = [word for word in ["total", "sum"] if word in query_lower]
        
        # Determine energy column based on query keywords with more robust detection
        energy_column = "EnergyMet"  # Default
        column_keywords = []
        
        if "shortage" in query_lower:
            energy_column = "EnergyShortage"
            column_keywords.append("shortage")
        elif "deficit" in query_lower:
            energy_column = "EnergyShortage"
            column_keywords.append("deficit")
        elif "demand" in query_lower:
            energy_column = "MaximumDemand"
            column_keywords.append("demand")
        elif "generation" in query_lower:
            energy_column = "GenerationAmount"
            column_keywords.append("generation")
        elif "consumption" in query_lower:
            energy_column = "EnergyMet"
            column_keywords.append("consumption")
        elif "energy met" in query_lower or "energy_met" in query_lower:
            energy_column = "EnergyMet"
            column_keywords.append("energy_met")
        
        # FIXED: Simplified, focused prompts for specific query types
        if "average" in query_lower and "shortage" in query_lower:
            # Specialized prompt for average shortage queries
            prompt = f"""
You are a SQL expert for an analytics application.
Your ONLY task is to generate the SQL for the following user request.

User request: "{context['query']}"

MANDATORY SQL REQUIREMENTS:
- Use AVG(fs.EnergyShortage) as AverageShortage
- Use table: FactAllIndiaDailySummary (alias fs)
- Join with DimRegions (alias r) on fs.RegionID = r.RegionID
- Join with DimDates (alias d) on fs.DateID = d.DateID

FORBIDDEN:
- Do NOT use SUM() anywhere in the query.
- Do NOT use EnergyMet column.
- Do NOT include growth calculations or any logic except simple AVG on EnergyShortage.
- Do NOT include complex subqueries or window functions.

EXAMPLE OUTPUT:
SELECT AVG(fs.EnergyShortage) AS AverageShortage
FROM FactAllIndiaDailySummary fs
JOIN DimRegions r ON fs.RegionID = r.RegionID
JOIN DimDates d ON fs.DateID = d.DateID

Return ONLY the SQL code above. Nothing else.
"""
            return prompt
        
        elif "average" in query_lower and "energy" in query_lower:
            # Specialized prompt for average energy queries
            prompt = f"""
You are a SQL expert for an analytics application.
Your ONLY task is to generate the SQL for the following user request.

User request: "{context['query']}"

MANDATORY SQL REQUIREMENTS:
- Use AVG(fs.EnergyMet) as AverageEnergy
- Use table: FactAllIndiaDailySummary (alias fs)
- Join with DimRegions (alias r) on fs.RegionID = r.RegionID
- Join with DimDates (alias d) on fs.DateID = d.DateID

FORBIDDEN:
- Do NOT use SUM() anywhere in the query.
- Do NOT use EnergyShortage column unless specifically requested.
- Do NOT include growth calculations or any logic except simple AVG on EnergyMet.

EXAMPLE OUTPUT:
SELECT AVG(fs.EnergyMet) AS AverageEnergy
FROM FactAllIndiaDailySummary fs
JOIN DimRegions r ON fs.RegionID = r.RegionID
JOIN DimDates d ON fs.DateID = d.DateID

Return ONLY the SQL code above. Nothing else.
"""
            return prompt
        
        elif "total" in query_lower and "shortage" in query_lower:
            # Specialized prompt for total shortage queries
            prompt = f"""
You are a SQL expert for an analytics application.
Your ONLY task is to generate the SQL for the following user request.

User request: "{context['query']}"

MANDATORY SQL REQUIREMENTS:
- Use SUM(fs.EnergyShortage) as TotalShortage
- Use table: FactAllIndiaDailySummary (alias fs)
- Join with DimRegions (alias r) on fs.RegionID = r.RegionID
- Join with DimDates (alias d) on fs.DateID = d.DateID

FORBIDDEN:
- Do NOT use AVG() anywhere in the query.
- Do NOT use EnergyMet column.
- Do NOT include growth calculations or any logic except simple SUM on EnergyShortage.

EXAMPLE OUTPUT:
SELECT SUM(fs.EnergyShortage) AS TotalShortage
FROM FactAllIndiaDailySummary fs
JOIN DimRegions r ON fs.RegionID = r.RegionID
JOIN DimDates d ON fs.DateID = d.DateID

Return ONLY the SQL code above. Nothing else.
"""
            return prompt
        
        # Default prompt for other query types (simplified)
        prompt = f"""
You are a SQL expert for an analytics application.
Generate a complete SQLite-compatible SQL query based on the following request.

User request: "{context['query']}"

{schema_context}

{few_shot_examples}

IMPORTANT REQUIREMENTS:
1. Generate a COMPLETE SQL query starting with SELECT
2. Use SQLite-compatible syntax only
3. Use strftime() for date functions (e.g., strftime('%Y-%m', d.ActualDate))
4. Avoid window functions (LAG, LEAD, ROW_NUMBER) - use subqueries instead
5. Use proper table aliases (fs for fact tables, r for regions, d for dates)
6. Use {aggregation_function}() for aggregations
7. Use {energy_column} column for energy metrics
8. Use GROUP BY for grouping
9. Use ORDER BY for sorting
10. Include all necessary JOINs, WHERE clauses, and GROUP BY

CRITICAL: Use the exact table names and column names from the schema:
- FactAllIndiaDailySummary (alias: fs) - contains EnergyMet, EnergyShortage columns
- DimRegions (alias: r) - contains RegionName column  
- DimDates (alias: d) - contains ActualDate column (NOT Date)
- Use fs.{energy_column} for energy metrics
- Use r.RegionName for region names
- Use d.ActualDate for dates (NOT d.Date)
- Use fs.RegionID for joining with DimRegions
- Use fs.DateID for joining with DimDates

Generate the SQL query:
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

    def _validate_sql_against_instructions(self, sql: str, aggregation_function: str, energy_column: str, query_lower: str) -> bool:
        """Validate if the generated SQL follows the business rules"""
        sql_lower_check = sql.lower()
        
        # Check if correct aggregation function is used
        expected_agg = aggregation_function.lower()
        if expected_agg not in sql_lower_check:
            logger.warning(f"Expected {expected_agg}() but found different aggregation in SQL: {sql[:200]}...")
            return False
        
        # Check if correct column is used
        if energy_column.lower() not in sql_lower_check:
            logger.warning(f"Expected {energy_column} column but found different column in SQL: {sql[:200]}...")
            return False
        
        # Additional checks for specific query types
        if "average" in query_lower and "shortage" in query_lower:
            # For average shortage queries, check for forbidden patterns
            if "sum(" in sql_lower_check:
                logger.warning(f"Found forbidden SUM() in average shortage query: {sql[:200]}...")
                return False
            if "energymet" in sql_lower_check:
                logger.warning(f"Found forbidden EnergyMet column in average shortage query: {sql[:200]}...")
                return False
        
        return True