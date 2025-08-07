"""
Wren AI Integration for Advanced Semantic Layer
Implements MDL support, advanced vector search, and semantic layer integration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import yaml
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from backend.core.types import QueryAnalysis, IntentType, QueryType
from backend.core.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class MDLNodeType(Enum):
    """MDL node types"""
    MODEL = "model"
    SOURCE = "source"
    METRIC = "metric"
    DIMENSION = "dimension"
    RELATIONSHIP = "relationship"


@dataclass
class MDLNode:
    """MDL node representation"""
    node_type: MDLNodeType
    name: str
    definition: Dict[str, Any]
    metadata: Dict[str, Any]
    relationships: List[str] = None


@dataclass
class MDLRelationship:
    """Enhanced MDL relationship with business semantics"""
    source_model: str
    target_model: str
    join_type: str  # one_to_one, one_to_many, many_to_many
    join_conditions: List[str]
    business_meaning: str
    cardinality: str
    is_required: bool = True


@dataclass
class MDLSchema:
    """Complete MDL schema representation"""
    models: Dict[str, MDLNode]
    relationships: List[MDLRelationship]
    sources: Dict[str, MDLNode]
    metrics: Dict[str, MDLNode]
    dimensions: Dict[str, MDLNode]
    business_glossary: Dict[str, str]
    metadata: Dict[str, Any]


class WrenAIIntegration:
    """
    Wren AI Integration for advanced semantic layer
    Implements MDL support, advanced vector search, and semantic layer integration
    """
    
    def __init__(self, llm_provider: LLMProvider, mdl_path: Optional[str] = None):
        self.llm_provider = llm_provider
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_client = QdrantClient(":memory:")
        self.mdl_schema: Optional[MDLSchema] = None
        self.mdl_path = mdl_path or "mdl/"
        self._initialize_advanced_vector_store()
        
    async def initialize(self):
        """Initialize Wren AI integration with MDL support"""
        await self._load_mdl_schema()
        await self._populate_advanced_vector_store()
        await self._initialize_semantic_layer()
        logger.info("Wren AI integration initialized successfully")
        
    def _initialize_advanced_vector_store(self):
        """Initialize advanced vector database collections"""
        try:
            # Advanced collections for Wren AI integration
            collections = {
                "mdl_models": VectorParams(size=384, distance=Distance.COSINE),
                "mdl_relationships": VectorParams(size=384, distance=Distance.COSINE),
                "mdl_metrics": VectorParams(size=384, distance=Distance.COSINE),
                "mdl_dimensions": VectorParams(size=384, distance=Distance.COSINE),
                "semantic_context": VectorParams(size=384, distance=Distance.COSINE),
                "query_patterns": VectorParams(size=384, distance=Distance.COSINE),
                "business_entities": VectorParams(size=384, distance=Distance.COSINE)
            }
            
            for collection_name, vector_config in collections.items():
                try:
                    self.vector_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=vector_config
                    )
                except Exception as e:
                    logger.warning(f"Collection {collection_name} may already exist: {e}")
                    
            logger.info("Advanced vector store collections created successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced vector store: {e}")
            raise
            
    async def _load_mdl_schema(self):
        """Load MDL schema from files or create default energy domain schema"""
        try:
            # Try to load from MDL files
            mdl_files = list(Path(self.mdl_path).glob("*.yaml")) + list(Path(self.mdl_path).glob("*.yml"))
            
            if mdl_files:
                await self._load_mdl_from_files(mdl_files)
            else:
                await self._create_default_energy_mdl()
                
        except Exception as e:
            logger.error(f"Failed to load MDL schema: {e}")
            await self._create_default_energy_mdl()
            
    async def _load_mdl_from_files(self, mdl_files: List[Path]):
        """Load MDL schema from YAML files"""
        models = {}
        relationships = []
        sources = {}
        metrics = {}
        dimensions = {}
        business_glossary = {}
        
        for mdl_file in mdl_files:
            try:
                with open(mdl_file, 'r') as f:
                    mdl_content = yaml.safe_load(f)
                    
                # Parse MDL content based on Wren AI format
                if 'models' in mdl_content:
                    for model_name, model_def in mdl_content['models'].items():
                        models[model_name] = MDLNode(
                            node_type=MDLNodeType.MODEL,
                            name=model_name,
                            definition=model_def,
                            metadata=model_def.get('metadata', {})
                        )
                        
                if 'relationships' in mdl_content:
                    for rel_def in mdl_content['relationships']:
                        relationships.append(MDLRelationship(
                            source_model=rel_def['source'],
                            target_model=rel_def['target'],
                            join_type=rel_def.get('type', 'one_to_many'),
                            join_conditions=rel_def['conditions'],
                            business_meaning=rel_def.get('business_meaning', ''),
                            cardinality=rel_def.get('cardinality', 'many'),
                            is_required=rel_def.get('required', True)
                        ))
                        
            except Exception as e:
                logger.error(f"Failed to parse MDL file {mdl_file}: {e}")
                
        self.mdl_schema = MDLSchema(
            models=models,
            relationships=relationships,
            sources=sources,
            metrics=metrics,
            dimensions=dimensions,
            business_glossary=business_glossary,
            metadata={}
        )
        
        logger.info(f"Loaded MDL schema with {len(models)} models and {len(relationships)} relationships")
        
    async def _create_default_energy_mdl(self):
        """Create default energy domain MDL schema"""
        models = {
            "FactAllIndiaDailySummary": MDLNode(
                node_type=MDLNodeType.MODEL,
                name="FactAllIndiaDailySummary",
                definition={
                    "columns": {
                        "RegionID": {"type": "integer", "description": "Region identifier"},
                        "DateID": {"type": "integer", "description": "Date identifier"},
                        "EnergyMet": {"type": "decimal", "description": "Energy met in MWh"},
                        "EnergyRequirement": {"type": "decimal", "description": "Energy requirement in MWh"},
                        "Surplus": {"type": "decimal", "description": "Energy surplus in MWh"},
                        "Deficit": {"type": "decimal", "description": "Energy deficit in MWh"}
                    },
                    "description": "Daily energy summary at national level",
                    "grain": "daily_region"
                },
                metadata={"business_name": "National Energy Summary"}
            ),
            "FactStateDailyEnergy": MDLNode(
                node_type=MDLNodeType.MODEL,
                name="FactStateDailyEnergy",
                definition={
                    "columns": {
                        "StateID": {"type": "integer", "description": "State identifier"},
                        "DateID": {"type": "integer", "description": "Date identifier"},
                        "EnergyMet": {"type": "decimal", "description": "Energy met in MWh"},
                        "EnergyConsumption": {"type": "decimal", "description": "Energy consumption in MWh"},
                        "PeakDemand": {"type": "decimal", "description": "Peak demand in MW"}
                    },
                    "description": "Daily energy data by state",
                    "grain": "daily_state"
                },
                metadata={"business_name": "State Energy Data"}
            )
        }
        
        relationships = [
            MDLRelationship(
                source_model="FactAllIndiaDailySummary",
                target_model="DimRegions",
                join_type="many_to_one",
                join_conditions=["f.RegionID = d.RegionID"],
                business_meaning="Each daily summary belongs to a specific region",
                cardinality="many",
                is_required=True
            ),
            MDLRelationship(
                source_model="FactStateDailyEnergy",
                target_model="DimStates",
                join_type="many_to_one",
                join_conditions=["f.StateID = d.StateID"],
                business_meaning="Each state energy record belongs to a specific state",
                cardinality="many",
                is_required=True
            )
        ]
        
        business_glossary = {
            "energy_met": "Total energy supplied to meet demand",
            "energy_requirement": "Total energy demand/requirement",
            "surplus": "Excess energy available beyond requirement",
            "deficit": "Shortfall in energy supply compared to requirement",
            "peak_demand": "Maximum energy demand in a day",
            "generation": "Energy production from various sources"
        }
        
        self.mdl_schema = MDLSchema(
            models=models,
            relationships=relationships,
            sources={},
            metrics={},
            dimensions={},
            business_glossary=business_glossary,
            metadata={"domain": "energy", "version": "1.0"}
        )
        
        logger.info("Created default energy domain MDL schema")
        
    async def _populate_advanced_vector_store(self):
        """Populate advanced vector store with MDL knowledge"""
        if not self.mdl_schema:
            return
            
        # Index MDL models
        model_points = []
        for model_name, model_node in self.mdl_schema.models.items():
            model_text = f"{model_name}: {model_node.definition.get('description', '')}"
            embedding = self.sentence_transformer.encode(model_text).tolist()
            
            model_points.append(PointStruct(
                id=len(model_points),
                vector=embedding,
                payload={
                    "type": "mdl_model",
                    "name": model_name,
                    "description": model_node.definition.get('description', ''),
                    "grain": model_node.definition.get('grain', ''),
                    "columns": list(model_node.definition.get('columns', {}).keys()),
                    "business_name": model_node.metadata.get('business_name', model_name)
                }
            ))
            
        # Index MDL relationships
        relationship_points = []
        for rel in self.mdl_schema.relationships:
            rel_text = f"{rel.source_model} to {rel.target_model}: {rel.business_meaning}"
            embedding = self.sentence_transformer.encode(rel_text).tolist()
            
            relationship_points.append(PointStruct(
                id=len(relationship_points),
                vector=embedding,
                payload={
                    "type": "mdl_relationship",
                    "source_model": rel.source_model,
                    "target_model": rel.target_model,
                    "join_type": rel.join_type,
                    "business_meaning": rel.business_meaning,
                    "cardinality": rel.cardinality
                }
            ))
            
        # Index business glossary
        glossary_points = []
        for term, definition in self.mdl_schema.business_glossary.items():
            embedding = self.sentence_transformer.encode(f"{term}: {definition}").tolist()
            glossary_points.append(PointStruct(
                id=len(glossary_points),
                vector=embedding,
                payload={
                    "type": "business_glossary",
                    "term": term,
                    "definition": definition
                }
            ))
            
        # Insert into vector store
        if model_points:
            self.vector_client.upsert(collection_name="mdl_models", points=model_points)
        if relationship_points:
            self.vector_client.upsert(collection_name="mdl_relationships", points=relationship_points)
        if glossary_points:
            self.vector_client.upsert(collection_name="business_entities", points=glossary_points)
            
        logger.info(f"Populated vector store with {len(model_points)} models, {len(relationship_points)} relationships, {len(glossary_points)} glossary terms")
        
    async def _initialize_semantic_layer(self):
        """Initialize semantic layer with Wren AI capabilities"""
        # Initialize semantic processing capabilities
        logger.info("Semantic layer initialized with Wren AI capabilities")
        
    async def extract_semantic_context(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Extract rich semantic context using Wren AI integration
        Combines MDL knowledge, vector search, and business context
        """
        try:
            # Step 1: MDL-aware vector search
            query_embedding = self.sentence_transformer.encode(natural_language_query).tolist()
            
            # Search across all collections
            search_results = {}
            collections = ["mdl_models", "mdl_relationships", "business_entities", "query_patterns"]
            
            for collection in collections:
                try:
                    results = self.vector_client.search(
                        collection_name=collection,
                        query_vector=query_embedding,
                        limit=5
                    )
                    search_results[collection] = results
                except Exception as e:
                    logger.warning(f"Search failed for collection {collection}: {e}")
                    
            # Step 2: MDL relationship inference
            mdl_context = await self._infer_mdl_context(natural_language_query, search_results)
            
            # Step 3: Business entity extraction
            business_entities = self._extract_business_entities(natural_language_query, search_results)
            
            # Step 4: Semantic confidence calculation
            confidence = self._calculate_semantic_confidence(search_results, mdl_context)
            
            return {
                "mdl_context": mdl_context,
                "business_entities": business_entities,
                "search_results": search_results,
                "confidence": confidence,
                "semantic_mappings": self._build_semantic_mappings(natural_language_query, business_entities)
            }
            
        except Exception as e:
            logger.error(f"Semantic context extraction failed: {e}")
            return {
                "mdl_context": {},
                "business_entities": [],
                "search_results": {},
                "confidence": 0.0,
                "semantic_mappings": {}
            }
            
    async def _infer_mdl_context(self, query: str, search_results: Dict) -> Dict[str, Any]:
        """Infer MDL context from query and search results"""
        mdl_context = {
            "relevant_models": [],
            "relevant_relationships": [],
            "join_paths": [],
            "business_meaning": ""
        }
        
        # Extract relevant models
        if "mdl_models" in search_results:
            for result in search_results["mdl_models"]:
                mdl_context["relevant_models"].append({
                    "name": result.payload["name"],
                    "description": result.payload["description"],
                    "score": result.score
                })
                
        # Extract relevant relationships
        if "mdl_relationships" in search_results:
            for result in search_results["mdl_relationships"]:
                mdl_context["relevant_relationships"].append({
                    "source": result.payload["source_model"],
                    "target": result.payload["target_model"],
                    "business_meaning": result.payload["business_meaning"],
                    "score": result.score
                })
                
        # Infer join paths
        mdl_context["join_paths"] = self._infer_join_paths(mdl_context["relevant_models"], mdl_context["relevant_relationships"])
        
        return mdl_context
        
    def _infer_join_paths(self, models: List, relationships: List) -> List[List[str]]:
        """Infer possible join paths between models"""
        join_paths = []
        
        # Simple path inference - can be enhanced with graph algorithms
        for rel in relationships:
            join_paths.append([rel["source"], rel["target"]])
            
        return join_paths
        
    def _extract_business_entities(self, query: str, search_results: Dict) -> List[Dict[str, Any]]:
        """Extract business entities from query and search results"""
        entities = []
        
        # Extract from business glossary
        if "business_entities" in search_results:
            for result in search_results["business_entities"]:
                entities.append({
                    "type": "business_term",
                    "term": result.payload["term"],
                    "definition": result.payload["definition"],
                    "score": result.score
                })
                
        return entities
        
    def _calculate_semantic_confidence(self, search_results: Dict, mdl_context: Dict) -> float:
        """Calculate semantic confidence based on search results and MDL context"""
        confidence = 0.0
        
        # Base confidence from search results
        total_results = sum(len(results) for results in search_results.values())
        if total_results > 0:
            confidence += min(total_results / 10.0, 0.3)
            
        # MDL context confidence
        if mdl_context["relevant_models"]:
            confidence += 0.3
            
        if mdl_context["relevant_relationships"]:
            confidence += 0.2
            
        if mdl_context["join_paths"]:
            confidence += 0.2
            
        return min(confidence, 1.0)
        
    def _build_semantic_mappings(self, query: str, business_entities: List) -> Dict[str, str]:
        """Build semantic mappings from natural language to schema elements"""
        mappings = {}
        query_lower = query.lower()
        
        # Map business terms to schema elements
        for entity in business_entities:
            if entity["type"] == "business_term":
                term = entity["term"]
                if term in query_lower:
                    mappings[term] = entity["definition"]
                    
        return mappings
        
    async def generate_mdl_aware_sql(self, query: str, semantic_context: Dict) -> Dict[str, Any]:
        """Generate SQL using MDL-aware semantic context"""
        try:
            # Use MDL context to enhance SQL generation
            mdl_context = semantic_context.get("mdl_context", {})
            business_entities = semantic_context.get("business_entities", [])
            
            # Build enhanced prompt with MDL context
            prompt = self._build_mdl_aware_prompt(query, mdl_context, business_entities)
            
            # Generate SQL using LLM
            response = await self.llm_provider.generate(prompt)
            
            # Extract SQL from response
            response_text = response.content if hasattr(response, 'content') else str(response)
            extracted_sql = self._extract_sql_from_response(response_text)
            
            return {
                "sql": extracted_sql,
                "mdl_context": mdl_context,
                "confidence": semantic_context.get("confidence", 0.0),
                "business_entities": business_entities
            }
            
        except Exception as e:
            logger.error(f"MDL-aware SQL generation failed: {e}")
            return {
                "sql": "",
                "mdl_context": {},
                "confidence": 0.0,
                "business_entities": []
            }
    
    def _extract_sql_from_response(self, response_text: str) -> str:
        """Extract SQL from LLM response text"""
        # Strategy 1: Look for SQL code blocks
        sql_blocks = []
        
        # Find ```sql blocks
        start_markers = ["```sql", "```SQL", "```"]
        end_markers = ["```"]
        
        for start_marker in start_markers:
            start_idx = response_text.find(start_marker)
            if start_idx != -1:
                start_pos = start_idx + len(start_marker)
                # Find the end marker
                for end_marker in end_markers:
                    end_idx = response_text.find(end_marker, start_pos)
                    if end_idx != -1:
                        sql_block = response_text[start_pos:end_idx].strip()
                        if sql_block and ('SELECT' in sql_block.upper() or 'FROM' in sql_block.upper()):
                            sql_blocks.append(sql_block)
                        break
        
        # Strategy 2: Extract lines that look like SQL (improved)
        lines = response_text.split('\n')
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
        
        # Strategy 3: Last resort - extract anything that looks like SQL
        words = response_text.split()
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
        
    def _build_mdl_aware_prompt(self, query: str, mdl_context: Dict, business_entities: List) -> str:
        """Build MDL-aware prompt for SQL generation"""
        prompt = f"""
        Generate a complete SQLite-compatible SQL query for the following query using MDL context:
        
        Query: "{query}"
        
        MDL Context:
        - Relevant Models: {[m['name'] for m in mdl_context.get('relevant_models', [])]}
        - Relationships: {[f"{r['source']} -> {r['target']}" for r in mdl_context.get('relevant_relationships', [])]}
        - Join Paths: {mdl_context.get('join_paths', [])}
        
        Business Entities:
        {chr(10).join([f"- {e['term']}: {e['definition']}" for e in business_entities])}
        
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
        2. Uses the appropriate MDL models and relationships
        3. Follows the business semantics
        4. Includes proper joins based on the relationships
        5. Handles the business entities correctly
        6. Uses SQLite-compatible syntax only
        7. Is complete and executable
        8. Uses the exact table and column names specified above
        
        Return only the SQL query without any explanation, formatting, or code blocks:
        """
        
        return prompt 