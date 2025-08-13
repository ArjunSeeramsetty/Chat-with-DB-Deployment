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
    
    def __init__(self, llm_provider: LLMProvider, mdl_path: Optional[str] = None, db_path: Optional[str] = None):
        self.llm_provider = llm_provider
        # Try to initialize sentence transformer; fall back to a simple embedder if offline/rate-limited
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            self._embed = lambda text: self.sentence_transformer.encode(text).tolist()
        except Exception as e:
            logger.warning(f"Falling back to simple hashing embedder due to model load issue: {e}")
            import hashlib
            def _hash_embed(text: str) -> List[float]:
                # Deterministic 384-dim pseudo-embedding via hashing
                h = hashlib.sha256(text.encode('utf-8', errors='ignore')).digest()
                # Repeat bytes to reach 384 dims
                vec = list(h) * ((384 // len(h)) + 1)
                vec = vec[:384]
                # Normalize to 0..1
                return [v / 255.0 for v in vec]
            self._embed = _hash_embed
        self.vector_client = QdrantClient(":memory:")
        self.mdl_schema: Optional[MDLSchema] = None
        self.mdl_path = mdl_path or "mdl/"
        self.db_path = db_path
        self._initialize_advanced_vector_store()
        
    async def initialize(self):
        """Initialize Wren AI integration with MDL support"""
        await self._load_mdl_schema()
        # Augment MDL from SQLite database schema if available
        try:
            if self.db_path:
                await self._augment_mdl_from_sqlite()
                # Persist augmented MDL so it can be inspected and reused
                try:
                    await self._export_mdl_to_json("config/power-sector-mdl.json")
                except Exception as e:
                    logger.warning(f"Failed to export augmented MDL: {e}")
        except Exception as e:
            logger.warning(f"Failed to augment MDL from SQLite: {e}")
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
                "mdl_columns": VectorParams(size=384, distance=Distance.COSINE),
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
        """Load MDL schema from YAML/JSON files or create default energy domain schema"""
        try:
            mdl_dir = Path(self.mdl_path)
            config_dir = Path("config")
            # Gather YAML and JSON candidates from mdl/ and config/
            yaml_files = list(mdl_dir.glob("*.yaml")) + list(mdl_dir.glob("*.yml"))
            json_files = list(mdl_dir.glob("*.json")) + list(config_dir.glob("*.json"))

            loaded_any = False
            if yaml_files:
                await self._load_mdl_from_files(yaml_files)
                loaded_any = True
            if json_files:
                await self._load_mdl_from_json_files(json_files)
                loaded_any = True

            if not loaded_any:
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

    async def _load_mdl_from_json_files(self, json_files: List[Path]):
        """Load MDL schema from JSON files (Wren-style or simplified)."""
        # Merge semantics across multiple JSON files
        models = {} if not self.mdl_schema else dict(self.mdl_schema.models)
        relationships = [] if not self.mdl_schema else list(self.mdl_schema.relationships)
        sources = {} if not self.mdl_schema else dict(self.mdl_schema.sources)
        metrics = {} if not self.mdl_schema else dict(self.mdl_schema.metrics)
        dimensions = {} if not self.mdl_schema else dict(self.mdl_schema.dimensions)
        business_glossary = {} if not self.mdl_schema else dict(self.mdl_schema.business_glossary)

        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    j = json.load(f)
            except Exception as e:
                logger.warning(f"Could not parse JSON MDL file {jf}: {e}")
                continue

            # Models: accept either dict {models:{name:{...}}} or list {models:[{name:'', columns:[...] ...}]}
            if isinstance(j.get('models'), dict):
                for model_name, model_def in j['models'].items():
                    col_map: Dict[str, Any] = {}
                    # Handle columns as list or dict
                    cols = model_def.get('columns', {})
                    if isinstance(cols, list):
                        for c in cols:
                            if isinstance(c, dict) and 'name' in c:
                                col_map[c['name']] = {k: v for k, v in c.items() if k != 'name'}
                            elif isinstance(c, str):
                                col_map[c] = {"type": ""}
                    elif isinstance(cols, dict):
                        col_map = cols
                    models[model_name] = MDLNode(
                        node_type=MDLNodeType.MODEL,
                        name=model_name,
                        definition={
                            **{k: v for k, v in model_def.items() if k != 'columns'},
                            "columns": col_map
                        },
                        metadata=model_def.get('metadata', {})
                    )
            elif isinstance(j.get('models'), list):
                for m in j['models']:
                    name = m.get('name') or m.get('model')
                    if not name:
                        continue
                    col_map: Dict[str, Any] = {}
                    cols = m.get('columns', {})
                    if isinstance(cols, list):
                        for c in cols:
                            if isinstance(c, dict) and 'name' in c:
                                col_map[c['name']] = {k: v for k, v in c.items() if k != 'name'}
                            elif isinstance(c, str):
                                col_map[c] = {"type": ""}
                    elif isinstance(cols, dict):
                        col_map = cols
                    models[name] = MDLNode(
                        node_type=MDLNodeType.MODEL,
                        name=name,
                        definition={
                            **{k: v for k, v in m.items() if k not in ('columns','name','model')},
                            "columns": col_map
                        },
                        metadata=m.get('metadata', {})
                    )

            # Relationships: support top-level 'relationships' list
            rels = j.get('relationships') or []
            for rel in rels:
                try:
                    source = rel.get('source') or rel.get('source_model') or rel.get('from')
                    target = rel.get('target') or rel.get('target_model') or rel.get('to')
                    conds = rel.get('conditions') or rel.get('join_conditions') or []
                    join_type = rel.get('type') or rel.get('join_type') or 'one_to_many'
                    business_meaning = rel.get('business_meaning', '')
                    cardinality = rel.get('cardinality', 'many')
                    if source and target and conds:
                        relationships.append(MDLRelationship(
                            source_model=source,
                            target_model=target,
                            join_type=join_type,
                            join_conditions=conds,
                            business_meaning=business_meaning,
                            cardinality=cardinality,
                            is_required=rel.get('required', False)
                        ))
                except Exception:
                    continue

            # Metrics (list or dict)
            j_metrics = j.get('metrics') or []
            if isinstance(j_metrics, dict):
                for mname, mdef in j_metrics.items():
                    metrics[mname] = MDLNode(node_type=MDLNodeType.METRIC, name=mname, definition=mdef, metadata={})
            elif isinstance(j_metrics, list):
                for mobj in j_metrics:
                    mname = mobj.get('name')
                    if not mname:
                        continue
                    metrics[mname] = MDLNode(node_type=MDLNodeType.METRIC, name=mname, definition={k: v for k, v in mobj.items() if k != 'name'}, metadata={})

            # Dimensions (optional)
            j_dims = j.get('dimensions') or {}
            if isinstance(j_dims, dict):
                for dname, ddef in j_dims.items():
                    dimensions[dname] = MDLNode(node_type=MDLNodeType.DIMENSION, name=dname, definition=ddef, metadata={})

            # Business glossary (optional)
            j_gloss = j.get('business_glossary') or {}
            if isinstance(j_gloss, dict):
                business_glossary.update(j_gloss)

        self.mdl_schema = MDLSchema(
            models=models,
            relationships=relationships,
            sources=sources,
            metrics=metrics,
            dimensions=dimensions,
            business_glossary=business_glossary,
            metadata={}
        )
        logger.info(f"Loaded JSON MDL with {len(models)} models, {len(relationships)} relationships, {len(metrics)} metrics")
        
    async def _create_default_energy_mdl(self):
        """Create default energy domain MDL schema aligned to actual DB columns"""
        # Core models (facts)
        models = {
            "FactAllIndiaDailySummary": MDLNode(
                node_type=MDLNodeType.MODEL,
                name="FactAllIndiaDailySummary",
                definition={
                    "columns": {
                        "RegionID": {"type": "integer", "description": "Region identifier"},
                        "DateID": {"type": "integer", "description": "Date identifier"},
                        "EnergyMet": {"type": "decimal", "description": "Energy met (consumption) in MU"},
                        "EnergyShortage": {"type": "decimal", "description": "Energy shortage in MU"},
                        "MaxDemandSCADA": {"type": "decimal", "description": "Region-level peak demand (MW)"},
                        "CentralSectorOutage": {"type": "decimal", "description": "Central sector outage (MU)"},
                        "StateSectorOutage": {"type": "decimal", "description": "State sector outage (MU)"},
                        "PrivateSectorOutage": {"type": "decimal", "description": "Private sector outage (MU)"}
                    },
                    "description": "Daily energy summary by region (includes India aggregate)",
                    "grain": "daily_region"
                },
                metadata={"business_name": "National/Region Energy Summary"}
            ),
            "FactStateDailyEnergy": MDLNode(
                node_type=MDLNodeType.MODEL,
                name="FactStateDailyEnergy",
                definition={
                    "columns": {
                        "StateID": {"type": "integer", "description": "State identifier"},
                        "DateID": {"type": "integer", "description": "Date identifier"},
                        "EnergyMet": {"type": "decimal", "description": "Energy met (consumption) in MU"},
                        "EnergyShortage": {"type": "decimal", "description": "Energy shortage in MU"},
                        "MaximumDemand": {"type": "decimal", "description": "State-level peak demand (MW)"}
                    },
                    "description": "Daily energy metrics by state",
                    "grain": "daily_state"
                },
                metadata={"business_name": "State Energy Metrics"}
            ),
            # Dimensions
            "DimRegions": MDLNode(
                node_type=MDLNodeType.MODEL,
                name="DimRegions",
                definition={
                    "columns": {
                        "RegionID": {"type": "integer", "description": "Region identifier"},
                        "RegionName": {"type": "string", "description": "Region name (India, Northern Region, etc.)"}
                    },
                    "description": "Regions dimension"
                },
                metadata={"business_name": "Regions"}
            ),
            "DimStates": MDLNode(
                node_type=MDLNodeType.MODEL,
                name="DimStates",
                definition={
                    "columns": {
                        "StateID": {"type": "integer", "description": "State identifier"},
                        "StateName": {"type": "string", "description": "State name"}
                    },
                    "description": "States dimension"
                },
                metadata={"business_name": "States"}
            ),
            "DimDates": MDLNode(
                node_type=MDLNodeType.MODEL,
                name="DimDates",
                definition={
                    "columns": {
                        "DateID": {"type": "integer", "description": "Date identifier"},
                        "ActualDate": {"type": "date", "description": "Actual calendar date"}
                    },
                    "description": "Dates dimension"
                },
                metadata={"business_name": "Dates"}
            ),
        }

        # Relationships (business semantics + join keys)
        relationships = [
            MDLRelationship(
                source_model="FactAllIndiaDailySummary",
                target_model="DimRegions",
                join_type="many_to_one",
                join_conditions=["fs.RegionID = r.RegionID"],
                business_meaning="Each daily region record belongs to one region",
                cardinality="many",
                is_required=True,
            ),
            MDLRelationship(
                source_model="FactAllIndiaDailySummary",
                target_model="DimDates",
                join_type="many_to_one",
                join_conditions=["fs.DateID = d.DateID"],
                business_meaning="Each daily region record belongs to one date",
                cardinality="many",
                is_required=True,
            ),
            MDLRelationship(
                source_model="FactStateDailyEnergy",
                target_model="DimStates",
                join_type="many_to_one",
                join_conditions=["fs.StateID = s.StateID"],
                business_meaning="Each daily state record belongs to one state",
                cardinality="many",
                is_required=True,
            ),
            MDLRelationship(
                source_model="FactStateDailyEnergy",
                target_model="DimDates",
                join_type="many_to_one",
                join_conditions=["fs.DateID = d.DateID"],
                business_meaning="Each daily state record belongs to one date",
                cardinality="many",
                is_required=True,
            ),
        ]

        # Business glossary
        business_glossary = {
            "energy_met": "Total energy supplied to meet demand",
            "energy_shortage": "Unmet energy demand",
            "peak_demand_state": "Maximum demand measured for a state (MaximumDemand)",
            "peak_demand_region": "Maximum demand measured for a region (MaxDemandSCADA)",
            "consumption": "EnergyMet (MU)",
        }

        # Metrics and dimensions (semantic layer)
        metrics = {
            "energy_consumption_region": {
                "model": "FactAllIndiaDailySummary",
                "expression": "SUM(fs.EnergyMet)",
                "description": "Total energy consumption (MU) by region",
            },
            "energy_shortage_region": {
                "model": "FactAllIndiaDailySummary",
                "expression": "SUM(fs.EnergyShortage)",
                "description": "Total energy shortage (MU) by region",
            },
            "peak_demand_region": {
                "model": "FactAllIndiaDailySummary",
                "expression": "MAX(fs.MaxDemandSCADA)",
                "description": "Peak demand (MW) for a region",
            },
            "outage_total_region": {
                "model": "FactAllIndiaDailySummary",
                "expression": "SUM(COALESCE(fs.CentralSectorOutage,0)+COALESCE(fs.StateSectorOutage,0)+COALESCE(fs.PrivateSectorOutage,0))",
                "description": "Total outage (MU) for a region",
            },
            "outage_central_region": {
                "model": "FactAllIndiaDailySummary",
                "expression": "SUM(COALESCE(fs.CentralSectorOutage,0))",
                "description": "Central sector outage (MU) for a region",
            },
            "outage_state_region": {
                "model": "FactAllIndiaDailySummary",
                "expression": "SUM(COALESCE(fs.StateSectorOutage,0))",
                "description": "State sector outage (MU) for a region",
            },
            "outage_private_region": {
                "model": "FactAllIndiaDailySummary",
                "expression": "SUM(COALESCE(fs.PrivateSectorOutage,0))",
                "description": "Private sector outage (MU) for a region",
            },
            "energy_consumption_state": {
                "model": "FactStateDailyEnergy",
                "expression": "SUM(fs.EnergyMet)",
                "description": "Total energy consumption (MU) by state",
            },
            "energy_shortage_state": {
                "model": "FactStateDailyEnergy",
                "expression": "SUM(fs.EnergyShortage)",
                "description": "Total energy shortage (MU) by state",
            },
            "peak_demand_state": {
                "model": "FactStateDailyEnergy",
                "expression": "MAX(fs.MaximumDemand)",
                "description": "Peak demand (MW) for a state",
            },
            # Transmission / International link flow (example generic metrics)
            "transmission_flow_total": {
                "model": "FactTransmissionLinkFlow",
                "expression": "SUM(fs.Flow)",
                "description": "Total transmission flow",
                "granularity": ["year", "month", "day"],
            },
            "international_link_flow_total": {
                "model": "FactInternationalTransmissionLinkFlow",
                "expression": "SUM(fs.Flow)",
                "description": "Total international link flow",
                "granularity": ["year", "month", "day"],
            },
            # Time-block tables (sub-daily awareness)
            "timeblock_power_total": {
                "model": "FactTimeBlockPowerData",
                "expression": "SUM(fs.Power)",
                "description": "Total power by time-block",
                "granularity": ["time_block", "day", "month", "year"],
            },
            "timeblock_generation_total": {
                "model": "FactTimeBlockGeneration",
                "expression": "SUM(fs.Generation)",
                "description": "Total generation by time-block",
                "granularity": ["time_block", "day", "month", "year"],
            },
        }

        dimensions = {
            "region_name": {"model": "DimRegions", "column": "r.RegionName"},
            "state_name": {"model": "DimStates", "column": "s.StateName"},
            "month": {"model": "DimDates", "expression": "strftime('%Y-%m', d.ActualDate)"},
            "year": {"model": "DimDates", "expression": "strftime('%Y', d.ActualDate)"},
            "day": {"model": "DimDates", "expression": "strftime('%Y-%m-%d', d.ActualDate)"},
        }

        self.mdl_schema = MDLSchema(
            models=models,
            relationships=relationships,
            sources={},
            metrics={k: MDLNode(node_type=MDLNodeType.METRIC, name=k, definition=v, metadata={}) for k, v in metrics.items()},
            dimensions={k: MDLNode(node_type=MDLNodeType.DIMENSION, name=k, definition=v, metadata={}) for k, v in dimensions.items()},
            business_glossary=business_glossary,
            metadata={"domain": "energy", "version": "1.1"}
        )

        logger.info("Created default energy domain MDL schema (models, relationships, metrics, dimensions)")

    async def _augment_mdl_from_sqlite(self):
        """Read SQLite schema and add any missing tables/columns/relationships into MDL."""
        import sqlite3
        if not self.db_path:
            return
        if not self.mdl_schema:
            # Initialize empty schema if not present
            self.mdl_schema = MDLSchema(models={}, relationships=[], sources={}, metrics={}, dimensions={}, business_glossary={}, metadata={})

        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            # Get table names
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [r[0] for r in cur.fetchall()]

            # Load units mapping if available (DimUnits + MetaTableColumnUnits)
            unit_map: Dict[str, Dict[str, str]] = {}
            try:
                if "DimUnits" in tables and "MetaTableColumnUnits" in tables:
                    cur.execute(
                        """
                        SELECT mtc.SchemaName, mtc.TableName, mtc.ColumnName, du.UnitName, du.UnitSymbol, du.UnitCategory
                        FROM MetaTableColumnUnits mtc
                        JOIN DimUnits du ON mtc.UnitID = du.UnitID
                        """
                    )
                    for row in cur.fetchall():
                        schema_name = str(row[0])  # 'main' by default in SQLite
                        tname = str(row[1])
                        cname = str(row[2])
                        uname = str(row[3])
                        unit_map.setdefault(tname, {})[cname] = uname
            except Exception:
                unit_map = {}

            # Load columns per table
            table_to_cols: Dict[str, Dict[str, Any]] = {}
            for t in tables:
                cur.execute(f"PRAGMA table_info('{t}')")
                cols = [dict(name=row[1], type=row[2]) for row in cur.fetchall()]
                # Units from metadata tables when present (no guessing)
                # Prefer exact table/column case; fallback to case-insensitive match
                table_units = unit_map.get(t, {})
                if not table_units:
                    for ut, mapping in unit_map.items():
                        if ut.lower() == t.lower():
                            table_units = mapping
                            break
                table_to_cols[t] = {
                    c["name"]: {
                        "type": c["type"],
                        **({"unit": (table_units.get(c["name"]) or next((v for k,v in table_units.items() if k.lower()==c["name"].lower()), None))}
                           if (table_units.get(c["name"]) or any(k.lower()==c["name"].lower() for k in table_units.keys())) else {})
                    }
                    for c in cols
                }

                # Compute heuristic hints
                def _pick_preferred_time(col_names: List[str]) -> Optional[str]:
                    prefs = ["ActualDate", "Timestamp", "TimeStamp", "DateTime", "Time", "BlockTime", "TimeBlock"]
                    for p in prefs:
                        if p in col_names:
                            return p
                    # try lowercase contains
                    lowers = {c.lower(): c for c in col_names}
                    for p in ["actualdate", "timestamp", "datetime", "time", "blocktime", "timeblock"]:
                        if p in lowers:
                            return lowers[p]
                    return None

                def _pick_preferred_values(col_names: List[str]) -> List[str]:
                    result: List[str] = []
                    for c in col_names:
                        cl = c.lower()
                        if cl.endswith("id"):
                            continue
                        if any(k in cl for k in ["energymet", "energyshortage", "maxdemandscada", "maximumdemand"]):
                            result.append(c)
                        if any(k in cl for k in ["flow", "mw", "power"]):
                            result.append(c)
                        if any(k in cl for k in ["generationamount", "totalgeneration", "generation_mw", "generationmu", "generation"]):
                            result.append(c)
                    # de-dup preserving order
                    out = []
                    for x in result:
                        if x not in out:
                            out.append(x)
                    return out

                col_names_list = list(table_to_cols[t].keys())
                # basic semantic role and hints
                role = "dimension" if t.lower().startswith("dim") else "fact"
                # preferred categorical column guess
                cat_guess = None
                for candidate in ["RegionName", "StateName", "CountryName", "SourceName", "LineIdentifier", "MechanismName"]:
                    if candidate in col_names_list:
                        cat_guess = candidate
                        break
                # collect per-column unit hints (from metadata mapping only)
                unit_hints = {cn: cd.get("unit") for cn, cd in table_to_cols[t].items() if cd.get("unit")}
                hints = {
                    "preferred_time_column": _pick_preferred_time(col_names_list),
                    "preferred_value_columns": _pick_preferred_values(col_names_list),
                    "has_date_fk": ("DateID" in col_names_list),
                    "preferred_category": cat_guess,
                    "role": role,
                    "units": unit_hints,
                }

                if t not in self.mdl_schema.models:
                    self.mdl_schema.models[t] = MDLNode(
                        node_type=MDLNodeType.MODEL,
                        name=t,
                        definition={
                            "columns": table_to_cols[t],
                            "description": f"Auto-imported from SQLite: {t}",
                            "hints": hints,
                            "synonyms": [t, t.replace("_", " "), t.lower(), t.replace("_", "").lower()],
                        },
                        metadata={"business_name": t.replace("Fact", "").replace("Dim", "").strip()}
                    )
                else:
                    # Merge missing columns
                    existing_cols = self.mdl_schema.models[t].definition.setdefault("columns", {})
                    existing_cols.update({k: v for k, v in table_to_cols[t].items() if k not in existing_cols})
                    # Merge hints
                    existing_hints = self.mdl_schema.models[t].definition.setdefault("hints", {})
                    for hk, hv in hints.items():
                        if hk not in existing_hints or not existing_hints[hk]:
                            existing_hints[hk] = hv
                    # Ensure synonyms exists
                    syns = self.mdl_schema.models[t].definition.setdefault("synonyms", [])
                    for s in [t, t.replace("_", " "), t.lower(), t.replace("_", "").lower()]:
                        if s not in syns:
                            syns.append(s)

            # Load FK relationships
            for t in tables:
                cur.execute(f"PRAGMA foreign_key_list('{t}')")
                for row in cur.fetchall():
                    ref_table = row[2]
                    from_col = row[3]
                    to_col = row[4]
                    # Build relationship node
                    join = f"{t}.{from_col} = {ref_table}.{to_col}"
                    rel = MDLRelationship(
                        source_model=t,
                        target_model=ref_table,
                        join_type="many_to_one",
                        join_conditions=[join],
                        business_meaning=f"FK: {t}.{from_col} -> {ref_table}.{to_col}",
                        cardinality="many",
                        is_required=False,
                    )
                    # Avoid duplicates
                    existing = [r for r in self.mdl_schema.relationships if r.source_model == rel.source_model and r.target_model == rel.target_model and r.join_conditions == rel.join_conditions]
                    if not existing:
                        self.mdl_schema.relationships.append(rel)
        finally:
            conn.close()

    async def _export_mdl_to_json(self, out_path: str):
        """Export the in-memory MDL schema to a JSON file for offline inspection and reuse."""
        if not self.mdl_schema:
            return
        def _node_to_dict(n: MDLNode) -> Dict[str, Any]:
            return {
                "name": n.name,
                "definition": n.definition,
                "metadata": n.metadata,
            }
        data = {
            "models": {k: _node_to_dict(v) for k, v in (self.mdl_schema.models or {}).items()},
            "relationships": [
                {
                    "source": r.source_model,
                    "target": r.target_model,
                    "type": r.join_type,
                    "conditions": r.join_conditions,
                    "business_meaning": r.business_meaning,
                    "cardinality": r.cardinality,
                    "required": r.is_required,
                }
                for r in (self.mdl_schema.relationships or [])
            ],
            "metrics": {k: _node_to_dict(v) for k, v in (self.mdl_schema.metrics or {}).items()},
            "dimensions": {k: _node_to_dict(v) for k, v in (self.mdl_schema.dimensions or {}).items()},
            "business_glossary": self.mdl_schema.business_glossary or {},
            "metadata": self.mdl_schema.metadata or {},
        }
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported augmented MDL to {p}")
        
    async def _populate_advanced_vector_store(self):
        """Populate advanced vector store with MDL knowledge"""
        if not self.mdl_schema:
            return
            
        # Index MDL models
        model_points = []
        for model_name, model_node in self.mdl_schema.models.items():
            model_text = f"{model_name}: {model_node.definition.get('description', '')}"
            embedding = self._embed(model_text)
            
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
            embedding = self._embed(rel_text)
            
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
            
        # Index metrics
        metric_points = []
        for metric_name, metric_node in self.mdl_schema.metrics.items():
            metric_text = f"{metric_name}: {metric_node.definition.get('description', '')} in {metric_node.definition.get('model', '')}"
            embedding = self._embed(metric_text)

            metric_points.append(PointStruct(
                id=len(metric_points),
                vector=embedding,
                payload={
                    "type": "mdl_metric",
                    "name": metric_name,
                    "model": metric_node.definition.get('model', ''),
                    "expression": metric_node.definition.get('expression', ''),
                    "description": metric_node.definition.get('description', ''),
                }
            ))

        # Index dimensions
        dimension_points = []
        for dim_name, dim_node in self.mdl_schema.dimensions.items():
            dim_text = f"{dim_name}: {dim_node.definition}"
            embedding = self._embed(dim_text)
            dimension_points.append(PointStruct(
                id=len(dimension_points),
                vector=embedding,
                payload={
                    "type": "mdl_dimension",
                    "name": dim_name,
                    "definition": dim_node.definition,
                }
            ))

        # Index business glossary
        glossary_points = []
        for term, definition in self.mdl_schema.business_glossary.items():
            embedding = self._embed(f"{term}: {definition}")
            glossary_points.append(PointStruct(
                id=len(glossary_points),
                vector=embedding,
                payload={
                    "type": "business_glossary",
                    "term": term,
                    "definition": definition
                }
            ))
            
        # Index columns
        column_points = []
        for model_name, model_node in self.mdl_schema.models.items():
            cols = (model_node.definition or {}).get('columns', {})
            for col_name, col_def in cols.items():
                ctext = f"{model_name}.{col_name}: {col_def}"
                embedding = self._embed(ctext)
                column_points.append(PointStruct(
                    id=len(column_points),
                    vector=embedding,
                    payload={
                        "type": "mdl_column",
                        "table": model_name,
                        "column": col_name,
                        "definition": col_def,
                    }
                ))

        # Insert into vector store
        if model_points:
            self.vector_client.upsert(collection_name="mdl_models", points=model_points)
        if relationship_points:
            self.vector_client.upsert(collection_name="mdl_relationships", points=relationship_points)
        if metric_points:
            self.vector_client.upsert(collection_name="mdl_metrics", points=metric_points)
        if dimension_points:
            self.vector_client.upsert(collection_name="mdl_dimensions", points=dimension_points)
        if column_points:
            self.vector_client.upsert(collection_name="mdl_columns", points=column_points)
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
            query_embedding = self._embed(natural_language_query)
            
            # Search across all collections
            search_results = {}
            collections = ["mdl_models", "mdl_relationships", "mdl_columns", "mdl_metrics", "mdl_dimensions", "business_entities", "query_patterns"]
            
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
            # Prefer models whose synonyms appear explicitly in the query
            try:
                ql = (natural_language_query or "").lower()
                explicit = []
                for name, node in (self.mdl_schema.models or {}).items():
                    syns = (node.definition or {}).get("synonyms", [])
                    if any(s for s in syns if isinstance(s, str) and s and s.lower() in ql):
                        explicit.append({"name": name, "score": 1.0})
                if explicit:
                    # put explicit models to the front
                    existing = mdl_context.get("relevant_models", [])
                    mdl_context["relevant_models"] = explicit + existing
            except Exception:
                pass
            
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
        mappings: Dict[str, str] = {}
        query_lower = query.lower()

        # Map business terms to schema elements
        for entity in business_entities:
            if entity.get("type") == "business_term":
                term = entity.get("term", "")
                if term and term in query_lower:
                    mappings[term] = entity.get("definition", "")

        # Detect explicit table mentions to guide dynamic model selection
        try:
            import re
            tokens = set(re.split(r"[^A-Za-z0-9_.]+", query_lower))
            mentioned = []
            if self.mdl_schema and self.mdl_schema.models:
                for table, node in self.mdl_schema.models.items():
                    t_low = table.lower()
                    syns = (node.definition or {}).get('synonyms', [])
                    # token or substring match on name/synonyms/sanitized
                    candidates = set([t_low, t_low.replace('_','')]) | {str(s).lower() for s in syns}
                    if any((c in tokens) for c in candidates) or any(c in query_lower for c in candidates):
                        mentioned.append(table)
            if mentioned:
                mappings["explicit_tables"] = ",".join(sorted(set(mentioned)))
        except Exception:
            pass

        return mappings
        
    async def generate_mdl_aware_sql(self, query: str, semantic_context: Dict) -> Dict[str, Any]:
        """Generate SQL using MDL-aware semantic context"""
        try:
            # Use MDL context to enhance SQL generation
            mdl_context = semantic_context.get("mdl_context", {})
            business_entities = semantic_context.get("business_entities", [])
            semantic_mappings = semantic_context.get("semantic_mappings", {})
            
            # Build enhanced prompt with MDL context
            prompt = self._build_mdl_aware_prompt(query, mdl_context, business_entities)
            # Hint explicit tables (if any) for dynamic selection
            if semantic_mappings.get("explicit_tables"):
                prompt += f"\n\nPrefer using these explicitly mentioned tables if it makes sense: {semantic_mappings['explicit_tables']}\n"
            
            # Generate SQL using LLM
            response = await self.llm_provider.generate(prompt)
            
            # Extract SQL from response
            response_text = response.content if hasattr(response, 'content') else str(response)
            extracted_sql = self._extract_sql_from_response(response_text)

            # Post-process using MDL hints to correct common model/time issues, include original query for temporal refinement
            mdl_ctx_with_query = dict(mdl_context or {})
            mdl_ctx_with_query['original_query'] = query
            try:
                mdl_ctx_with_query['explicit_tables'] = semantic_mappings.get('explicit_tables')
            except Exception:
                pass
            extracted_sql = self._postprocess_sql_with_mdl_hints(extracted_sql, mdl_ctx_with_query)
            
            # If explicit tables are mentioned and SQL does not use them, synthesize a correct SQL using MDL
            try:
                explicit = semantic_mappings.get("explicit_tables")
                if explicit:
                    wanted = [t.strip() for t in explicit.split(',') if t.strip()]
                    lower_sql = (extracted_sql or '').lower()
                    chosen = next((t for t in wanted if t.lower() in lower_sql), None)
                    if not chosen and wanted:
                        synthesized = self._synthesize_sql_for_table(wanted[0], query)
                        if synthesized:
                            extracted_sql = synthesized
            except Exception:
                pass

            # Attach MDL-driven visualization defaults for UI (xAxis/yAxis/groupBy) when detectable
            plot_options = self._infer_plot_options_from_sql(extracted_sql)
            if not plot_options:
                # Try metric-driven defaults if SQL heuristics failed
                plot_options = self._infer_plot_options_from_metric(query, mdl_context)

            return {
                "sql": extracted_sql,
                "mdl_context": mdl_context,
                "confidence": semantic_context.get("confidence", 0.0),
                "business_entities": business_entities,
                "plot": {"chartType": "bar", "options": plot_options} if plot_options else None,
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

    def _postprocess_sql_with_mdl_hints(self, sql: str, mdl_context: Dict) -> str:
        """Use MDL model hints to fix common column/time issues in the generated SQL.
        Alias-aware and model-aware adjustments.
        """
        try:
            if not sql or not mdl_context:
                return sql
            import re as _re
            original_query = mdl_context.get('original_query') or mdl_context.get('query') or ''
            ql = original_query.lower()
            # If explicit tables requested and SQL doesn't use them, synthesize
            try:
                explicit = mdl_context.get('explicit_tables')
                if explicit:
                    wanted = [t.strip() for t in explicit.split(',') if t.strip()]
                    lw = [(t.lower(), t) for t in wanted]
                    lower_sql = (sql or '').lower()
                    if not any(tl in lower_sql for tl, _ in lw):
                        synth = self._synthesize_sql_for_table(wanted[0], mdl_context.get('original_query', ''))
                        if synth:
                            return synth
            except Exception:
                pass
            # Detect fact table and alias actually used
            m = _re.search(r"FROM\s+(\w+)\s+([a-zA-Z]\\w*)", sql, _re.IGNORECASE)
            fact_table = m.group(1) if m else None
            fact_alias = m.group(2) if m else "fs"

            # Pull hints for detected table; fallback to top relevant model
            hints = {}
            if fact_table and self.mdl_schema and self.mdl_schema.models:
                node = self.mdl_schema.models.get(fact_table)
                hints = (node.definition or {}).get('hints', {}) if node else {}
            if not hints:
                relevant_models = [m.get('name') for m in mdl_context.get('relevant_models', []) if isinstance(m, dict)]
                if relevant_models:
                    node = self.mdl_schema.models.get(relevant_models[0]) if self.mdl_schema and self.mdl_schema.models else None
                    hints = (node.definition or {}).get('hints', {}) if node else {}

            preferred_values = hints.get('preferred_value_columns') or []
            preferred_time = hints.get('preferred_time_column')
            has_date_fk = hints.get('has_date_fk', False)

            fixed = sql
            table_lower = (fact_table or '').lower()
            # 1) Replace EnergyMet in non-energy models with preferred value column
            is_energy_model = ('allindiadailysummary' in table_lower) or ('statedailyenergy' in table_lower)
            if ('energymet' in fixed.lower()) and not is_energy_model and preferred_values:
                pv = preferred_values[0]
                fixed = _re.sub(rf"\b{_re.escape(fact_alias)}\.EnergyMet\b", fact_alias + '.' + pv, fixed)
                fixed = _re.sub(r"\bEnergyMet\b", pv, fixed)

            # 2) If SQL uses d.ActualDate without DateID in model, switch to preferred time column and drop DimDates join
            if preferred_time and not has_date_fk and 'd.ActualDate' in fixed:
                fixed = _re.sub(r"JOIN\s+DimDates\s+d\s+ON\s+[^\s]+", "", fixed, flags=_re.IGNORECASE)
                fixed = fixed.replace('d.ActualDate', f'{fact_alias}.{preferred_time}')
                fixed = _re.sub(r"strftime\('%Y-%m',\s*d\.ActualDate\)", f"strftime('%Y-%m', {fact_alias}.{preferred_time})", fixed, flags=_re.IGNORECASE)
                fixed = _re.sub(r"strftime\('%Y',\s*d\.ActualDate\)", f"strftime('%Y', {fact_alias}.{preferred_time})", fixed, flags=_re.IGNORECASE)

            # 2b) Model-specific time column enforcement
            if ('factinternationaltransmissionlinkflow' in table_lower or 'facttransmissionlinkflow' in table_lower) and preferred_time:
                # ensure we use fs.<preferred_time> and not d.ActualDate
                fixed = _re.sub(r"JOIN\s+DimDates\s+d\s+ON\s+[^\s]+", "", fixed, flags=_re.IGNORECASE)
                fixed = _re.sub(r"strftime\('\%Y-\%m',\s*d\.ActualDate\)", f"strftime('%Y-%m', {fact_alias}.{preferred_time})", fixed, flags=_re.IGNORECASE)
                fixed = _re.sub(r"strftime\('\%Y',\s*d\.ActualDate\)", f"strftime('%Y', {fact_alias}.{preferred_time})", fixed, flags=_re.IGNORECASE)
                fixed = fixed.replace('d.ActualDate', f'{fact_alias}.{preferred_time}')
            if ('facttimeblockpowerdata' in table_lower or 'facttimeblockgeneration' in table_lower) and preferred_time:
                fixed = _re.sub(r"JOIN\s+DimDates\s+d\s+ON\s+[^\s]+", "", fixed, flags=_re.IGNORECASE)
                fixed = _re.sub(r"strftime\('\%Y-\%m',\s*d\.ActualDate\)", f"strftime('%Y-%m', {fact_alias}.{preferred_time})", fixed, flags=_re.IGNORECASE)
                fixed = _re.sub(r"strftime\('\%Y',\s*d\.ActualDate\)", f"strftime('%Y', {fact_alias}.{preferred_time})", fixed, flags=_re.IGNORECASE)
                fixed = fixed.replace('d.ActualDate', f'{fact_alias}.{preferred_time}')

            # 2c) Country exchange model: ensure joins and date usage (this model typically has DateID and CountryID)
            if 'factcountrydailyexchange' in table_lower:
                # Ensure DimCountries join when CountryName appears
                if 'CountryName' in fixed or 'countryname' in fixed.lower():
                    if not _re.search(r"JOIN\s+DimCountries\s+dc\s+ON\s+[^\s]+", fixed, _re.IGNORECASE):
                        fixed = fixed.replace(f"FROM {fact_table} {fact_alias}", f"FROM {fact_table} {fact_alias} JOIN DimCountries dc ON {fact_alias}.CountryID = dc.CountryID")
                # Ensure DimDates join when strftime on d.ActualDate used
                if "strftime('" in fixed and 'd.ActualDate' in fixed and not _re.search(r"JOIN\s+DimDates\s+d\s+ON\s+[^\s]+", fixed, _re.IGNORECASE):
                    fixed = fixed.replace(f"FROM {fact_table} {fact_alias}", f"FROM {fact_table} {fact_alias} JOIN DimDates d ON {fact_alias}.DateID = d.DateID")

            # 3) Clean f.Table.Column → alias.Column noise
            fixed = _re.sub(r"\bf\.[A-Za-z_]\w+\.", fact_alias + '.', fixed)

            # 3b) Normalize DimGenerationSources alias to dgs for consistency
            if _re.search(r"JOIN\s+DimGenerationSources\s+(\w+)\s+ON", fixed, _re.IGNORECASE):
                fixed = _re.sub(r"JOIN\s+DimGenerationSources\s+\w+\s+ON", "JOIN DimGenerationSources dgs ON", fixed, flags=_re.IGNORECASE)
                fixed = _re.sub(r"\bgs\.SourceName\b", "dgs.SourceName", fixed)
                # Ensure GROUP BY uses dgs.SourceName instead of generic 'Source'
                fixed = _re.sub(r"GROUP\s+BY\s+Source\s*,\s*Month", "GROUP BY dgs.SourceName, Month", fixed, flags=_re.IGNORECASE)

            # 4) Temporal refinement: inject month/day filters if mentioned in query text available via mdl_context
            try:
                if isinstance(original_query, str) and original_query:
                    # Map month names
                    month_map = {
                        'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05', 'june': '06',
                        'july': '07', 'august': '08', 'september': '09', 'october': '10', 'november': '11', 'december': '12'
                    }
                    month_num = None
                    for name, num in month_map.items():
                        if _re.search(rf"\b{name}\b", ql):
                            month_num = num
                            break
                    # Detect year
                    year_m = _re.search(r"\b(19|20)\d{2}\b", ql)
                    # Determine the time expression used in SQL
                    time_expr = None
                    if 'd.ActualDate' in fixed and has_date_fk:
                        time_expr = 'd.ActualDate'
                    elif preferred_time:
                        time_expr = f"{fact_alias}.{preferred_time}"
                    # Inject month filter if month specified and time_expr exists
                    if time_expr and month_num:
                        # Only inject if a month filter is not already present
                        if _re.search(r"strftime\('%m',\s*" + _re.escape(time_expr) + r"\)\s*=\s*'\d{2}'", fixed) is None:
                            # Add to WHERE clause; if WHERE exists, append AND, else create WHERE
                            if _re.search(r"\bWHERE\b", fixed, _re.IGNORECASE):
                                fixed = _re.sub(r"\bWHERE\b", lambda m: m.group(0) + f" strftime('%m', {time_expr}) = '{month_num}' AND", fixed, count=1, flags=_re.IGNORECASE)
                            else:
                                fixed += f" WHERE strftime('%m', {time_expr}) = '{month_num}'"
                    # Inject day filter for patterns like '15th June 2025'
                    day_m = _re.search(r"\b(\d{1,2})(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+((19|20)\d{2})\b", ql)
                    if time_expr and day_m:
                        day = day_m.group(1).zfill(2)
                        mon = month_map.get(day_m.group(3), None)
                        yr = day_m.group(4)
                        if mon and yr:
                            date_iso = f"{yr}-{mon}-{day}"
                            if _re.search(r"date\(\s*" + _re.escape(time_expr) + r"\s*\)\s*=\s*'\d{4}-\d{2}-\d{2}'", fixed) is None:
                                if _re.search(r"\bWHERE\b", fixed, _re.IGNORECASE):
                                    fixed = _re.sub(r"\bWHERE\b", lambda m: m.group(0) + f" date({time_expr}) = '{date_iso}' AND", fixed, count=1, flags=_re.IGNORECASE)
                                else:
                                    fixed += f" WHERE date({time_expr}) = '{date_iso}'"
            except Exception:
                pass

            # 5) Model-selection heuristics: override FAIDS/FSDE when domain keywords indicate specific models
            try:
                lower_sql_all = fixed.lower()
                # Generation by source → FactDailyGenerationBreakdown
                if (('generation' in ql and ('by source' in ql or 'source' in ql)) or ('renewable' in ql)) and ('factdailygenerationbreakdown' not in lower_sql_all):
                    synth = self._synthesize_sql_for_table('FactDailyGenerationBreakdown', original_query)
                    if synth:
                        return synth
                # Time-block queries → FactTimeBlockPowerData or FactTimeBlockGeneration
                if any(k in ql for k in ['time block', 'time-block', 'hourly', 'intraday', '15 min', '15-minute']):
                    target_table = 'FactTimeBlockGeneration' if ('generation' in ql or 'source' in ql) else 'FactTimeBlockPowerData'
                    if target_table.lower() not in lower_sql_all:
                        synth = self._synthesize_sql_for_table(target_table, original_query)
                        if not synth:
                            # Fallback minimal COUNT(*) to satisfy execution when hints unavailable
                            synth = f"SELECT COUNT(*) AS Value FROM {target_table}"
                        return synth
                # Transmission link flow / energy flow
                if any(k in ql for k in ['transmission', 'link flow', 'link-flow', 'interregional flow', 'inter-regional flow', 'flow', 'energy flow']):
                    # International if country present or 'international' keyword
                    intl = ('international' in ql) or any(c in ql for c in ['bangladesh', 'nepal', 'bhutan', 'sri lanka', 'china', 'myanmar', 'pakistan'])
                    target_table = 'FactInternationalTransmissionLinkFlow' if intl else 'FactTransmissionLinkFlow'
                    if target_table.lower() not in lower_sql_all:
                        synth = self._synthesize_sql_for_table(target_table, original_query)
                        if not synth:
                            synth = f"SELECT COUNT(*) AS Value FROM {target_table}"
                        return synth
                # Country exchange → FactCountryDailyExchange
                if any(k in ql for k in ['exchange', 'cross border', 'cross-border', 'import', 'export', 'international']) and ('factcountrydailyexchange' not in lower_sql_all) and any(c in ql for c in ['bangladesh','nepal','bhutan','sri','china','myanmar','pakistan']):
                    synth = self._synthesize_sql_for_table('FactCountryDailyExchange', original_query)
                    if synth:
                        return synth
            except Exception:
                pass

            return fixed
        except Exception:
            return sql
        
    def _build_mdl_aware_prompt(self, query: str, mdl_context: Dict, business_entities: List) -> str:
        """Build MDL-aware prompt for SQL generation"""
        # Build model-specific hints section
        model_hints_lines = []
        try:
            relevant_models = [m['name'] for m in mdl_context.get('relevant_models', [])]
        except Exception:
            relevant_models = []
        for mname in relevant_models:
            try:
                node = self.mdl_schema.models.get(mname)
                hints = (node.definition or {}).get('hints', {}) if node else {}
                pvals = hints.get('preferred_value_columns') or []
                ptime = hints.get('preferred_time_column')
                has_date_fk = hints.get('has_date_fk', False)
                if pvals or ptime or has_date_fk:
                    model_hints_lines.append(f"- {mname}: value={pvals[:3]} time={ptime or 'n/a'} date_fk={bool(has_date_fk)}")
            except Exception:
                continue
        model_hints_block = "\n".join(model_hints_lines)

        prompt = f"""
        Generate a complete SQLite-compatible SQL query for the following query using MDL context:
        
        Query: "{query}"
        
        MDL Context:
        - Relevant Models: {[m['name'] for m in mdl_context.get('relevant_models', [])]}
        - Relationships: {[f"{r['source']} -> {r['target']}" for r in mdl_context.get('relevant_relationships', [])]}
        - Join Paths: {mdl_context.get('join_paths', [])}

        MODEL HINTS:
        {model_hints_block}
        
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
        
        QUERY PATTERNS BY COMPLEXITY:
        
        FOR SIMPLE QUERIES (total, sum, single value):
        Example: "What is total energy consumption in 2025?"
        SELECT SUM(fs.EnergyMet) as TotalEnergyConsumption
        FROM FactAllIndiaDailySummary fs
        JOIN DimDates d ON fs.DateID = d.DateID
        WHERE strftime('%Y', d.ActualDate) = '2025'
        
        Example: "What is energy consumption of Northern Region in 2025?"
        SELECT SUM(fs.EnergyMet) as EnergyConsumption
        FROM FactAllIndiaDailySummary fs
        JOIN DimRegions r ON fs.RegionID = r.RegionID
        JOIN DimDates d ON fs.DateID = d.DateID
        WHERE r.RegionName = 'Northern Region' AND strftime('%Y', d.ActualDate) = '2025'

        Example: "What is energy shortage in 2025?" (no state/region mentioned → default to India total)
        SELECT SUM(fs.EnergyShortage) as TotalEnergyShortage
        FROM FactAllIndiaDailySummary fs
        JOIN DimRegions r ON fs.RegionID = r.RegionID
        JOIN DimDates d ON fs.DateID = d.DateID
        WHERE r.RegionName = 'India' AND strftime('%Y', d.ActualDate) = '2025'
        
        FOR BREAKDOWN QUERIES (by region/state):
        Example: "What is energy consumption by region in 2025?"
        SELECT r.RegionName, SUM(fs.EnergyMet) as TotalEnergyConsumption
        FROM FactAllIndiaDailySummary fs
        JOIN DimRegions r ON fs.RegionID = r.RegionID
        JOIN DimDates d ON fs.DateID = d.DateID
        WHERE strftime('%Y', d.ActualDate) = '2025'
        GROUP BY r.RegionName
        ORDER BY TotalEnergyConsumption DESC
        
        FOR TIME-SERIES QUERIES (monthly, yearly):
        Example: "What is monthly energy consumption in 2025?"
        SELECT strftime('%Y-%m', d.ActualDate) as Month, SUM(fs.EnergyMet) as MonthlyEnergyConsumption
        FROM FactAllIndiaDailySummary fs
        JOIN DimDates d ON fs.DateID = d.DateID
        WHERE strftime('%Y', d.ActualDate) = '2025'
        GROUP BY strftime('%Y-%m', d.ActualDate)
        ORDER BY Month
        
        ONLY use complex subqueries and growth calculations when explicitly asked for growth/change rates!
        
        CRITICAL: Use the exact table names and column names from the schema:
        
        FOR REGION-LEVEL QUERIES (multiple regions):
        - FactAllIndiaDailySummary (alias: fs) - contains EnergyMet, EnergyShortage, MaxDemandSCADA, EveningPeakDemandMet columns
        - DimRegions (alias: r) - contains RegionName column  
        - Use fs.EnergyMet for energy consumption
        - Use fs.MaxDemandSCADA for region-level peak demand (NOT MaximumDemand)
        - Use fs.EnergyShortage for energy shortage
        - For outage: use these columns as applicable and SUM them as needed:
          - fs.CentralSectorOutage, fs.StateSectorOutage, fs.PrivateSectorOutage
          - For "total outage" use SUM(COALESCE(fs.CentralSectorOutage,0)+COALESCE(fs.StateSectorOutage,0)+COALESCE(fs.PrivateSectorOutage,0))
        
        FOR STATE-LEVEL QUERIES (multiple states):
        - FactStateDailyEnergy (alias: fs) - contains EnergyMet, MaximumDemand, EnergyShortage columns
        - DimStates (alias: s) - contains StateName column
        - Use fs.EnergyMet for energy consumption  
        - Use fs.MaximumDemand for state-level peak demand
        - Use fs.EnergyShortage for energy shortage
        
        COMMON:
        - DimDates (alias: d) - contains ActualDate column (NOT Date, NOT Year)
        - If the chosen FACT table has a DateID column, JOIN DimDates and use d.ActualDate for dates.
        - If the chosen FACT table does NOT have DateID, use the model's preferred time column from MODEL HINTS (e.g., Timestamp, TimeBlock). Do NOT reference d.ActualDate without a join.
        - For year filtering: WHERE strftime('%Y', d.ActualDate) = '2025'
        - For month filtering: WHERE strftime('%Y-%m', d.ActualDate) = '2025-01'
            - For monthly average requests: include month in SELECT using strftime('%Y-%m', d.ActualDate) AS Month, then GROUP BY Month and use AVG(...) for metrics
        - Use fs.RegionID for joining with DimRegions
        - Use fs.StateID for joining with DimStates  
        - Use fs.DateID for joining with DimDates

        NEVER RULES:
        - NEVER use EnergyMet outside FactAllIndiaDailySummary or FactStateDailyEnergy
        - NEVER reference d.ActualDate unless JOIN DimDates is present and the FACT has DateID
        - NEVER invent columns not listed in MODEL HINTS or schema columns

        MODEL-SPECIFIC EXAMPLES:
        - FactInternationalTransmissionLinkFlow (alias: fs)
          If there is no DateID, use the model's intrinsic time column (e.g., fs.Timestamp) for strftime().
          SELECT ROUND(SUM(fs.Flow), 2) AS TotalFlow
          FROM FactInternationalTransmissionLinkFlow fs
          WHERE strftime('%Y', fs.Timestamp) = '2024'

        - FactTransmissionLinkFlow (alias: fs)
          SELECT ROUND(SUM(fs.Flow), 2) AS TotalFlow
          FROM FactTransmissionLinkFlow fs
          WHERE strftime('%Y', fs.Timestamp) = '2024'

        - FactTimeBlockPowerData (alias: fs)
          SELECT strftime('%Y-%m', fs.TimeBlock) AS Month, ROUND(SUM(fs.Power), 2) AS TotalPower
          FROM FactTimeBlockPowerData fs
          WHERE strftime('%Y', fs.TimeBlock) = '2025'
          GROUP BY Month ORDER BY Month

        - FactTimeBlockGeneration (alias: fs)
          SELECT strftime('%Y-%m', fs.TimeBlock) AS Month, ROUND(SUM(fs.Generation), 2) AS TotalGeneration
          FROM FactTimeBlockGeneration fs
          WHERE strftime('%Y', fs.TimeBlock) = '2025'
          GROUP BY Month ORDER BY Month
        
        IMPORTANT: Do NOT mix state and region tables! Choose ONE based on the query:
        - For "states in region" queries → Use FactStateDailyEnergy + DimStates + JOIN with DimRegions  
        - For "regions" queries → Use FactAllIndiaDailySummary + DimRegions
        
        NEVER use columns like f.Year, f.Month, d.Year, d.Month - these do not exist!
        NEVER use fs.MaximumDemand with FactAllIndiaDailySummary - use fs.MaxDemandSCADA!
        NEVER use fs.MaxDemandSCADA with FactStateDailyEnergy - use fs.MaximumDemand!
        ALWAYS use strftime() functions for date extraction.

        METRIC/TABLE SELECTION RULES:
        - If the query mentions a state or implies states-in-region: use FactStateDailyEnergy (fs) + DimStates (s). Metrics: fs.MaximumDemand (peak), fs.EnergyMet (consumption), fs.EnergyShortage (energy shortage)
        - If the query mentions a region, India, or no entity: use FactAllIndiaDailySummary (fs) + DimRegions (r). Metrics: fs.MaxDemandSCADA (peak), fs.EnergyMet (consumption), fs.EnergyShortage (energy shortage)
        - If the query mentions shortage with no entity: return All-India total → SUM(fs.EnergyShortage) with r.RegionName = 'India'
        
        IMPORTANT: All JOINs must use the correct foreign key relationships:
        - JOIN DimRegions r ON fs.RegionID = r.RegionID
        - JOIN DimDates d ON fs.DateID = d.DateID
        - In subqueries: JOIN DimRegions r2 ON fs2.RegionID = r2.RegionID
        - In subqueries: JOIN DimDates d2 ON fs2.DateID = d2.DateID
        
        Generate a complete SQL query that:
        1. Starts with SELECT and includes all necessary clauses
        2. Uses the SIMPLEST pattern that answers the question - avoid unnecessary complexity
        3. Uses the appropriate MDL models and relationships
        4. Follows the business semantics
        5. Includes proper joins based on the relationships
        6. Handles the business entities correctly
        7. Uses SQLite-compatible syntax only
        8. Is complete and executable
        9. Uses the exact table and column names specified above
        10. AVOID subqueries unless explicitly needed for growth/comparison calculations
        
        MANDATORY JOIN REQUIREMENTS:
        - ALWAYS JOIN DimDates when filtering by year, month, or any date
        - ALWAYS JOIN DimRegions when filtering by region name
        - ALWAYS JOIN DimStates when filtering by state name
        - Use consistent table aliases: fs (fact), r (regions), s (states), d (dates)
        
        DEFAULT ENTITY HANDLING:
        - If the query does NOT mention a specific state or region and asks for shortage/consumption, return the All-India total
        - Use FactAllIndiaDailySummary with JOIN DimRegions and filter r.RegionName = 'India'
        - For 'energy shortage' use fs.EnergyShortage; for 'peak shortage' use fs.PeakShortage
        
        IMPORTANT: Start with the simplest query pattern first. Only add complexity if required by the question.
        
        Return only the SQL query without any explanation, formatting, or code blocks:
        """
        
        return prompt 

    def _infer_plot_options_from_sql(self, sql: str) -> Optional[Dict[str, Any]]:
        """Lightweight heuristic to propose plot options (xAxis/yAxis/groupBy) from SQL.
        Helps UI render grouped series consistently.
        """
        try:
            import re as _re
            lower = (sql or "").lower()
            opts: Dict[str, Any] = {}
            # xAxis detection: Month/Year/Day
            if "strftime('%y-%m'".replace("%y", "%Y").lower() in lower or "strftime('%y-%m'" in lower:
                opts["xAxis"] = "Month"
            elif "strftime('%y'".replace("%y", "%Y").lower() in lower:
                opts["xAxis"] = "Year"
            elif "strftime('%y-%m-%d'".replace("%y", "%Y").lower() in lower:
                opts["xAxis"] = "Day"
            # groupBy detection for common dims
            if "group by r.regionname" in lower:
                opts["groupBy"] = "RegionName"
            elif "group by s.statename" in lower:
                opts["groupBy"] = "StateName"
            elif "group by dgs.sourcename" in lower or "group by gs.sourcename" in lower:
                opts["groupBy"] = "SourceName"
            elif "group by dc.countryname" in lower:
                opts["groupBy"] = "CountryName"
            # yAxis detection: pick first alias if present
            m = _re.search(r"\sAS\s+([A-Za-z_][A-Za-z0-9_]*)", sql, _re.IGNORECASE)
            if m:
                opts["yAxis"] = [m.group(1)]
                opts["valueLabel"] = m.group(1)
            else:
                # fallback common metric alias
                for alias in ["Value", "TotalEnergy", "TotalFlow", "TotalGeneration", "Central", "MonthlyEnergyConsumption"]:
                    if alias.lower() in lower:
                        opts["yAxis"] = [alias]
                        opts["valueLabel"] = alias
                        break
            return opts if (opts.get("xAxis") and opts.get("yAxis")) else None
        except Exception:
            return None

    def _infer_plot_options_from_metric(self, query: str, mdl_context: Dict) -> Optional[Dict[str, Any]]:
        """Fallback plot defaults based on metric/domain keywords and MDL hints."""
        try:
            ql = (query or "").lower()
            opts: Dict[str, Any] = {}
            # xAxis preference by temporal words
            if any(k in ql for k in ["monthly", "per month", "by month"]):
                opts["xAxis"] = "Month"
            elif any(k in ql for k in ["daily", "per day", "day wise", "day-wise"]):
                opts["xAxis"] = "Day"
            elif any(k in ql for k in ["yearly", "per year", "annual"]):
                opts["xAxis"] = "Year"
            else:
                opts["xAxis"] = "Month"

            # groupBy preference by entity
            if any(k in ql for k in ["all regions", "by region", "regionwise", "region wise"]):
                opts["groupBy"] = "RegionName"
            elif any(k in ql for k in ["all states", "by state", "statewise", "state wise"]):
                opts["groupBy"] = "StateName"
            elif any(k in ql for k in ["by source", "generation by source", "source wise"]):
                opts["groupBy"] = "SourceName"
            elif any(k in ql for k in ["country", "countries"]):
                opts["groupBy"] = "CountryName"

            # yAxis label from metric keywords
            if any(k in ql for k in ["shortage"]):
                opts["yAxis"] = ["TotalEnergyShortage"]
                opts["valueLabel"] = "TotalEnergyShortage"
            elif any(k in ql for k in ["maximum demand", "peak demand"]):
                opts["yAxis"] = ["PeakDemand"]
                opts["valueLabel"] = "PeakDemand"
            elif any(k in ql for k in ["outage"]):
                opts["yAxis"] = ["Outage"]
                opts["valueLabel"] = "Outage"
            elif any(k in ql for k in ["generation"]):
                opts["yAxis"] = ["TotalGeneration"]
                opts["valueLabel"] = "TotalGeneration"
            else:
                opts["yAxis"] = ["Value"]
                opts["valueLabel"] = "Value"

            return opts
        except Exception:
            return None

    def _synthesize_sql_for_table(self, table_name: str, query: str) -> Optional[str]:
        """Build MDL-aware SQL for a specific table when explicitly requested."""
        try:
            import re
            schema = getattr(self, 'mdl_schema', None)
            node = schema.models.get(table_name) if (schema and schema.models) else None
            hints = (node.definition or {}).get('hints', {}) if node else {}
            pvals = hints.get('preferred_value_columns') or []
            ptime = hints.get('preferred_time_column')
            has_date_fk = hints.get('has_date_fk', False)
            value_col = None
            tl = table_name.lower()
            # Choose value column by domain
            cols = []
            try:
                cols = list((node.definition or {}).get('columns', {}).keys()) if node and node.definition else []
            except Exception:
                cols = []
            lower_map = {c.lower(): c for c in cols}
            def first_existing(*names: str) -> Optional[str]:
                for n in names:
                    if n.lower() in lower_map:
                        return lower_map[n.lower()]
                return None
            def find_contains(*needles: str) -> Optional[str]:
                for lc, orig in lower_map.items():
                    if all(n in lc for n in needles) and not lc.endswith('id'):
                        return orig
                return None
            if 'factdailygenerationbreakdown' in tl:
                value_col = first_existing('GenerationAmount') or find_contains('generation', 'amount') or find_contains('generation') or (pvals[0] if pvals else None)
            elif 'facttimeblockpowerdata' in tl:
                value_col = first_existing('DemandMet', 'TotalGeneration') or find_contains('demand') or find_contains('generation') or (pvals[0] if pvals else None)
            elif 'facttimeblockgeneration' in tl:
                value_col = first_existing('GenerationOutput', 'Generation') or find_contains('generation') or (pvals[0] if pvals else None)
            elif 'factinternationaltransmissionlinkflow' in tl or 'facttransmissionlinkflow' in tl:
                value_col = first_existing('EnergyExchanged', 'NetImportEnergy', 'MaxLoading', 'MaxImport', 'MaxExport') or find_contains('energy') or find_contains('flow') or find_contains('loading') or (pvals[0] if pvals else None)
            elif pvals:
                value_col = pvals[0]
            if not value_col:
                return None
 
            ql = (query or '').lower()
            is_monthly = any(k in ql for k in ["monthly", "per month", "by month"])
            year_m = re.search(r"\b(19|20)\d{2}\b", ql)
            year = year_m.group(0) if year_m else None
            # Detect explicit full date
            day_full = None
            m_full = re.search(r"\b(20\d{2}|19\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b", ql)
            if m_full:
                day_full = m_full.group(0)
 
            # Table-specific joins
            joins = []
            if 'factdailygenerationbreakdown' in tl:
                joins.append("JOIN DimGenerationSources dgs ON fs.GenerationSourceID = dgs.GenerationSourceID")
            if has_date_fk and (is_monthly or year or day_full):
                joins.append("JOIN DimDates d ON fs.DateID = d.DateID")
 
            if is_monthly:
                if has_date_fk:
                    sql = (
                        f"SELECT "
                        + ("dgs.SourceName AS Source, " if 'factdailygenerationbreakdown' in tl else "")
                        + "strftime('%Y-%m', d.ActualDate) AS Month, "
                        + f"ROUND(SUM(fs.{value_col}), 2) AS Value FROM {table_name} fs "
                        + (" ".join(j for j in joins) + " " if joins else "")
                        + (f"WHERE strftime('%Y', d.ActualDate) = '{year}' " if year else "")
                        + ("GROUP BY " + ("dgs.SourceName, " if 'factdailygenerationbreakdown' in tl else "") + "Month ORDER BY Month")
                    )
                elif ptime:
                    sql = (
                        f"SELECT "
                        + ("dgs.SourceName AS Source, " if 'factdailygenerationbreakdown' in tl else "")
                        + f"strftime('%Y-%m', fs.{ptime}) AS Month, "
                        + f"ROUND(SUM(fs.{value_col}), 2) AS Value FROM {table_name} fs "
                        + (" ".join(j for j in joins if not j.startswith("JOIN DimDates")) + " " if joins else "")
                        + (f"WHERE strftime('%Y', fs.{ptime}) = '{year}' " if year else "")
                        + ("GROUP BY " + ("dgs.SourceName, " if 'factdailygenerationbreakdown' in tl else "") + "Month ORDER BY Month")
                    )
                else:
                    return None
            else:
                if has_date_fk:
                    filters = []
                    if year:
                        filters.append(f"strftime('%Y', d.ActualDate) = '{year}'")
                    if day_full:
                        filters.append(f"date(d.ActualDate) = '{day_full}'")
                    where_clause = (" WHERE " + " AND ".join(filters)) if filters else ""
                    sql = (
                        f"SELECT ROUND(SUM(fs.{value_col}), 2) AS Value FROM {table_name} fs "
                        + (" ".join(j for j in joins if not j.startswith("JOIN DimDates")) + " " if joins else "")
                        + "JOIN DimDates d ON fs.DateID = d.DateID"
                        + where_clause
                    )
                elif ptime:
                    filters = []
                    if year:
                        filters.append(f"strftime('%Y', fs.{ptime}) = '{year}'")
                    if day_full:
                        filters.append(f"date(fs.{ptime}) = '{day_full}'")
                    where_clause = (" WHERE " + " AND ".join(filters)) if filters else ""
                    sql = (
                        f"SELECT ROUND(SUM(fs.{value_col}), 2) AS Value FROM {table_name} fs "
                        + (" ".join(j for j in joins) + " " if joins else "")
                        + where_clause
                    )
                else:
                    return None
            return sql
        except Exception:
            return None