"""
Enhanced RAG Service with Wren AI Integration
Implements advanced semantic processing with MDL support and vector search
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import os

from backend.core.types import QueryAnalysis, IntentType, QueryType, ProcessingMode, ContextInfo, SchemaInfo
from backend.core.llm_provider import LLMProvider, create_llm_provider
from backend.core.cloud_semantic_engine import CloudSemanticEngine  # ADDED - cloud-ready version
from backend.core.wren_ai_integration import WrenAIIntegration
from backend.core.assembler import SQLAssembler
from backend.core.validator import EnhancedSQLValidator
from backend.core.executor import AsyncSQLExecutor, SQLExecutor  # RESTORED for MS SQL Server compatibility
from backend.core.advanced_retrieval import AdvancedRetrieval, ContextualRetrieval, RetrievalResult
from backend.core.intent import IntentAnalyzer
from backend.core.schema_linker import SchemaLinker
from backend.core.agentic_framework import VisualizationAgent
from backend.core.entity_loader import get_entity_loader
from backend.services.agentic_rag_service import AgenticRAGService
from backend.config import get_settings
from backend.core.temporal_processor import TemporalConstraintProcessor, TemporalConstraints

logger = logging.getLogger(__name__)


class EnhancedRAGService:
    """
    Enhanced RAG Service with Wren AI Integration
    Combines semantic processing, MDL support, and advanced vector search
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.settings = get_settings()
        
        # Initialize LLM provider
        if self.settings.llm_provider_type.lower() == "gemini":
            # Use Gemini-specific configuration
            self.llm_provider = create_llm_provider(
                provider_type=self.settings.llm_provider_type,
                api_key=self.settings.google_api_key,  # Use google_api_key instead of gemini_api_key
                model=self.settings.gemini_model,
                base_url=None,  # Gemini doesn't use base_url
                enable_gpu=self.settings.enable_gpu_acceleration
            )
        else:
            # Use generic LLM configuration
            self.llm_provider = create_llm_provider(
                provider_type=self.settings.llm_provider_type,
                api_key=self.settings.llm_api_key,
                model=self.settings.llm_model,
                base_url=self.settings.llm_base_url,
                enable_gpu=self.settings.enable_gpu_acceleration
            )
        
        # Initialize semantic components
        if os.path.exists(db_path):
            self.semantic_engine = CloudSemanticEngine(self.llm_provider, db_path)
            self.wren_ai_integration = WrenAIIntegration(self.llm_provider, mdl_path="config/", db_path=db_path)
        else:
            logger.warning(f"EnhancedRAGService: database path '{db_path}' not found – running in no-DB mode (testing)")
            # Use CloudSemanticEngine with MS SQL Server connection for testing
            try:
                self.semantic_engine = CloudSemanticEngine(self.llm_provider, self.settings.get_database_url())
                logger.info("Using CloudSemanticEngine with MS SQL Server connection")
            except Exception as e:
                logger.warning(f"Failed to initialize CloudSemanticEngine: {e}, using stub")
                # Fallback to stub if MS SQL Server connection fails
                async def stub_initialize():
                    logger.info("Stub semantic engine initialize called")
                    return
                async def stub_generate_contextual_sql(*args, **kwargs):
                    logger.info("Stub generate_contextual_sql called")
                    return {}
                self.semantic_engine = type("StubSE", (), {
                    "initialize": stub_initialize,
                    "generate_contextual_sql": stub_generate_contextual_sql
                })()
            
            # Initialize Wren AI integration with proper MDL path
            self.wren_ai_integration = WrenAIIntegration(self.llm_provider, mdl_path="config/", db_path=db_path)
        
        # Initialize core components
        self.sql_assembler = SQLAssembler(self.llm_provider)
        self.sql_validator = EnhancedSQLValidator()
        self.sql_executor = AsyncSQLExecutor(db_path)  # RESTORED for MS SQL Server compatibility
        
        # Initialize feedback storage
        # self.feedback_storage = FeedbackStorage(settings.feedback_db_path)  # Removed for cloud deployment
        
        # Initialize advanced retrieval system
        self.advanced_retrieval = AdvancedRetrieval()
        self.contextual_retrieval = ContextualRetrieval(self.advanced_retrieval)
        
        # Centralized temporal processor
        self.temporal_processor = TemporalConstraintProcessor({})
        
        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "semantic_enhancement_rate": 0.0,
            "average_response_time": 0.0,
            "mdl_usage_rate": 0.0,
            "vector_search_success_rate": 0.0,
            "hybrid_retrieval_rate": 0.0
        }
        
        self._initialized = False
        
    async def initialize(self):
        """Initialize all components"""
        if self._initialized:
            return
            
        try:
            # Initialize semantic engine (ensures vector collections are created & populated)
            await self.semantic_engine.initialize()
            
            # Initialize Wren AI integration
            await self.wren_ai_integration.initialize()
            
            # Initialize advanced retrieval with documents
            await self._initialize_advanced_retrieval()
            
            self._initialized = True
            logger.info("Enhanced RAG Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced RAG Service: {e}")
            raise
            
    async def _initialize_advanced_retrieval(self):
        """Initialize advanced retrieval system with documents"""
        try:
            # Get documents from various sources
            documents = await self._collect_documents()
            
            # Add documents to advanced retrieval system
            self.advanced_retrieval.add_documents(documents)
            
            logger.info(f"Initialized advanced retrieval with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced retrieval: {e}")
            
    async def _collect_documents(self) -> List[Dict[str, Any]]:
        """Collect documents from various sources for advanced retrieval"""
        documents = []
        
        try:
            # Add schema information
            schema_docs = await self._get_schema_documents()
            documents.extend(schema_docs)
            
            # Add business rules
            business_docs = await self._get_business_documents()
            documents.extend(business_docs)
            
            # Add historical queries
            historical_docs = await self._get_historical_documents()
            documents.extend(historical_docs)
            
            # Add feedback examples
            feedback_docs = await self._get_feedback_documents()
            documents.extend(feedback_docs)
            
            logger.info(f"Collected {len(documents)} documents for advanced retrieval")
            
        except Exception as e:
            logger.error(f"Failed to collect documents: {e}")
            
        return documents
        
    async def _get_schema_documents(self) -> List[Dict[str, Any]]:
        """Get schema-related documents"""
        documents = []
        
        try:
            # Get schema metadata
            schema_metadata = await self.semantic_engine.get_schema_metadata()

            # Support both list-based and dict-based schema formats
            if isinstance(schema_metadata, dict) and 'tables' in schema_metadata:
                # Newer extractor format: {'tables': [...], 'columns': [...], ...}
                for table in schema_metadata.get('tables', []):
                    table_name = table.get('name') or table.get('table') or ''
                    columns = table.get('columns', [])
                    # columns might be list of dicts or list of strings
                    if columns and isinstance(columns[0], dict):
                        col_names = [c.get('name') for c in columns if isinstance(c, dict)]
                    else:
                        col_names = [str(c) for c in columns]
                    description = table.get('description', '')

                    content = f"Table: {table_name}\nColumns: {', '.join(col_names)}\nDescription: {description}"

                    documents.append({
                        "content": content,
                        "source": "schema",
                        "metadata": {
                            "table_name": table_name,
                            "type": "schema",
                            "columns": col_names
                        }
                    })
            elif isinstance(schema_metadata, dict):
                # Legacy format: {table_name: {columns: [...], description: ''}}
                for table_name, table_info in schema_metadata.items():
                    columns = table_info.get('columns', [])
                    if columns and isinstance(columns[0], dict):
                        col_names = [c.get('name') for c in columns if isinstance(c, dict)]
                    else:
                        col_names = [str(c) for c in columns]
                    description = table_info.get('description', '')

                    content = f"Table: {table_name}\nColumns: {', '.join(col_names)}\nDescription: {description}"

                    documents.append({
                        "content": content,
                        "source": "schema",
                        "metadata": {
                            "table_name": table_name,
                            "type": "schema",
                            "columns": col_names
                        }
                    })
                
        except Exception as e:
            logger.error(f"Failed to get schema documents: {e}")
            
        return documents
        
    async def _get_business_documents(self) -> List[Dict[str, Any]]:
        """Get business rule documents"""
        documents = []
        
        try:
            # Add business rules and domain knowledge
            business_rules = [
                {
                    "content": "Energy Met represents actual energy consumption in megawatt hours",
                    "source": "business_rules",
                    "metadata": {"type": "business_rule", "domain": "energy"}
                },
                {
                    "content": "Energy Shortage represents unmet energy demand in megawatt hours",
                    "source": "business_rules", 
                    "metadata": {"type": "business_rule", "domain": "energy"}
                },
                {
                    "content": "States are grouped into regions: Northern, Southern, Eastern, Western, North Eastern, and Central",
                    "source": "business_rules",
                    "metadata": {"type": "business_rule", "domain": "geography"}
                }
            ]
            
            documents.extend(business_rules)
            
        except Exception as e:
            logger.error(f"Failed to get business documents: {e}")
            
        return documents
        
    async def _get_historical_documents(self) -> List[Dict[str, Any]]:
        """Get historical query documents"""
        documents = []
        
        try:
            # Get historical queries from feedback storage
            # This would be populated as the system is used
            pass
            
        except Exception as e:
            logger.error(f"Failed to get historical documents: {e}")
            
        return documents
        
    async def _get_feedback_documents(self) -> List[Dict[str, Any]]:
        """Get feedback-based documents"""
        documents = []
        
        try:
            # Get successful feedback examples
            # This would be populated as the system is used
            pass
            
        except Exception as e:
            logger.error(f"Failed to get feedback documents: {e}")
            
        return documents
        
    def _auto_repair_sql(self, sql: str) -> str:
        """Lightweight SQL auto-repair for common LLM mistakes.
        Schema-driven generalization:
        - Fix mixed aliases like f.Fs. -> fs.
        - Auto-inject JOINs for any referenced alias matching known dimension tables/keys from schema.
        - Remove stray GROUP BY/ORDER BY if only aggregates are selected.
        - If query implies monthly and filters a year, ensure monthly grouping present.
        """
        try:
            import re

            repaired = sql or ""
            repaired = re.sub(r"\s+", " ", repaired).strip()
            if not repaired:
                return repaired

            # 1) Fix alias typos
            for pattern, repl in [
                (r"\bf\.[Ff]s\.", "fs."),
                (r"\bfs\.[Ff]s\.", "fs."),
                (r"\bf\.[Rr]\.", "r."),
                (r"\bf\.[Dd]\.", "d."),
                (r"\bf\.[Ss]\.", "s."),
            ]:
                repaired = re.sub(pattern, repl, repaired)

            # Extract fact alias
            fact = re.search(r"FROM\s+(\w+)\s+([a-zA-Z]\w*)", repaired, re.IGNORECASE)
            fact_alias = fact.group(2) if fact else None

            def inject_after_from(sql_text: str, join_sql: str) -> str:
                m = re.search(r"FROM\s+\w+\s+[a-zA-Z]\\w*", sql_text, re.IGNORECASE)
                if not m:
                    return sql_text
                pos = m.end()
                return sql_text[:pos] + " " + join_sql + " " + sql_text[pos:]

            # 2) Inject JOINs (schema-driven)
            try:
                schema_exec = SQLExecutor(self.db_path)
                schema = schema_exec.get_schema_info()
            except Exception:
                schema = {}
            # Infer foreign keys by common name conventions
            fk_map = {
                'DimDates': ('d', 'DateID'),
                'DimRegions': ('r', 'RegionID'),
                'DimStates': ('s', 'StateID'),
            }
            for table, (alias, key) in fk_map.items():
                if f" {alias}." in repaired and f"JOIN {table}" not in repaired and fact_alias and key in repaired:
                    repaired = inject_after_from(repaired, f"JOIN {table} {alias} ON {fact_alias}.{key} = {alias}.{key}")

            # 3) Remove stray GROUP BY/ORDER BY if only aggregates
            if re.search(r"GROUP\s+BY", repaired, re.IGNORECASE):
                sel = re.search(r"SELECT\s+(.*?)\s+FROM", repaired, re.IGNORECASE)
                select_expr = sel.group(1) if sel else ""
                # Enhanced detection of non-aggregate columns that should be grouped
                has_group_cols = any(k in select_expr for k in [
                    " r.", " s.", " d.", " dc.", " dtl.", " dgs.", " ftb.", " ftbg.",  # Table aliases
                    "DATEPART(", "FORMAT(", "RegionName", "StateName", "CountryName", "LineIdentifier", 
                    "SourceName", "BlockTime", "BlockNumber", "Year", "Month", "DayOfMonth"  # Common dimension columns
                ])
                only_aggs = bool(re.search(r"\b(SUM|AVG|MAX|MIN|COUNT)\s*\(", select_expr, re.IGNORECASE)) and not has_group_cols
                if only_aggs:
                    repaired = re.sub(r"GROUP\s+BY\s+[^;]+", "", repaired, flags=re.IGNORECASE)
                    repaired = re.sub(r"ORDER\s+BY\s+[^;]+", "", repaired, flags=re.IGNORECASE)

            # 4) Enforce monthly grouping if hinted by query content captured earlier in pipeline
            # We infer from existing SQL: if it filters a full year and uses aggregate without any Month grouping, add Month
            if (
                ("strftime('%Y', d.ActualDate)" in repaired or "DATEPART(YEAR, d.ActualDate)" in repaired)
                and re.search(r"\b(AVG|SUM|MAX|MIN)\s*\(", repaired, re.IGNORECASE)
                and "dt.Month" not in repaired and ("strftime('%m'" not in repaired and "DATEPART(MONTH" not in repaired and "FORMAT(" not in repaired)
            ):
                # Inject dt.Month into SELECT, GROUP BY, ORDER BY
                # Ensure DimDates alias is dt or d; normalize to dt
                repaired = re.sub(r"\bJOIN\s+DimDates\s+([a-zA-Z]\w*)\b", "JOIN DimDates dt", repaired, flags=re.IGNORECASE)
                # Add Month to SELECT list after first SELECT clause (support multiline)
                repaired = re.sub(r"(SELECT\s+)(.+?)(\s+FROM)", r"\1\2, dt.Month\3", repaired, flags=re.IGNORECASE|re.DOTALL)
                # Add GROUP BY dt.Month if GROUP BY present, else create one with any name column if present
                if re.search(r"GROUP\s+BY", repaired, re.IGNORECASE):
                    repaired = re.sub(r"GROUP\s+BY\s+", "GROUP BY dt.Month, ", repaired, flags=re.IGNORECASE)
                else:
                    # Try to detect a name column alias d./r./s. name
                    name_group = "r.RegionName"
                    if "s.StateName" in repaired:
                        name_group = "s.StateName"
                    repaired = re.sub(r"(WHERE\s+[^;]+)", r"\1 GROUP BY " + name_group + ", dt.Month", repaired, flags=re.IGNORECASE)
                # Ensure ORDER BY dt.Month exists
                if not re.search(r"ORDER\s+BY", repaired, re.IGNORECASE):
                    repaired += " ORDER BY dt.Month"

            return repaired.strip()
        except Exception:
            return sql

    def _validate_and_correct_sql(self, sql: str, query: str) -> str:
        """Validate tables/columns against schema and correct common issues.
        Generalized to any schema while preserving energy-specific heuristics as a fallback.
        Rules:
        - Replace unknown columns with best-guess metric in the actual table; if none, drop token.
        - If explicit table names appear in the natural query, prefer generating a simple SELECT using discovered columns.
        - If shortage/consumption/peak words appear and energy tables exist, prefer correct metric mapping.
        - For no-entity shortages, default to All-India on region table if present.
        """
        try:
            import re
            repaired = sql or ""

            # Short-circuit: if SQL already targets specialized non-energy fact tables, preserve it.
            # Avoid rewriting to FactAllIndiaDailySummary/FactStateDailyEnergy.
            try:
                m_ft = re.search(r"FROM\s+(\w+)\s+[a-zA-Z]\\w*", repaired, re.IGNORECASE)
                fact_used = (m_ft.group(1) if m_ft else None) or ""
                keep_tables = {
                    "FactTransmissionLinkFlow",
                    "FactInternationalTransmissionLinkFlow",
                    "FactTimeBlockPowerData",
                    "FactTimeBlockGeneration",
                    "FactDailyGenerationBreakdown",
                }
                if fact_used in keep_tables:
                    return repaired.strip()
            except Exception:
                pass
            if not repaired.strip():
                # Build minimal query if user referenced explicit table name
                ql = (query or "").lower()
                explicit_table = None
                # Try to match any table name in schema
                schema_exec = SQLExecutor(self.db_path)
                schema = schema_exec.get_schema_info()
                for table in schema.keys():
                    if table.lower() in ql:
                        explicit_table = table
                        break
                if explicit_table:
                    cols = schema.get(explicit_table, [])
                    select_cols = ", ".join(cols[:6]) if cols else '*'
                    return f"SELECT {select_cols} FROM {explicit_table} LIMIT 100"
                return repaired

            # Build schema map using SQLExecutor
            schema_exec = SQLExecutor(self.db_path)
            schema = schema_exec.get_schema_info()  # {table: [columns]}

            # Capture table aliases
            alias_map = {}  # alias -> table
            for kind, table, alias in re.findall(r"\b(FROM|JOIN)\s+(\w+)\s+([a-zA-Z]\\w*)", repaired, flags=re.IGNORECASE):
                alias_map[alias] = table

            # Extract alias.column references
            col_refs = re.findall(r"\b([a-zA-Z]\\w*)\.([A-Za-z_]\\w*)", repaired)

            # Helper to pick metric by intent
            ql = (query or "").lower()
            def pick_metric(default: str = "EnergyMet") -> str:
                if "shortage" in ql:
                    return "EnergyShortage"
                if "maximum demand" in ql or "peak demand" in ql or "max demand" in ql:
                    return "MaximumDemand"
                return default

            # Remove or replace unknown columns
            for alias, col in set(col_refs):
                table = alias_map.get(alias)
                if not table:
                    continue
                valid_cols = set(schema.get(table, []))
                if col not in valid_cols:
                    # Replace with best metric guess
                    new_col = pick_metric("EnergyMet")
                    if new_col in valid_cols:
                        repaired = re.sub(fr"\b{re.escape(alias)}\.{re.escape(col)}\b", f"{alias}.{new_col}", repaired)
                    else:
                        # If still invalid, drop this select expression token (simple heuristic)
                        repaired = re.sub(fr"\b{re.escape(alias)}\.{re.escape(col)}\b\s*(AS\s+[A-Za-z_]\w*)?", "", repaired)

            # Detect entities (state/region) using entity loader
            try:
                loader = get_entity_loader()
            except Exception:
                loader = None

            # Enforce state vs region table selection
            # Collect candidate states using list + mappings
            detected_states = []
            if loader:
                states_list = set([s for s in (loader.get_indian_states() or [])])
                state_map = loader.get_state_name_mappings() or {}
                state_aliases = set(state_map.keys()) | set(state_map.values()) | states_list
                for name in state_aliases:
                    if name and name.lower() in ql:
                        # Normalize to proper case via mapping
                        proper = state_map.get(name.lower(), name)
                        detected_states.append(proper)

            # Detect region via mappings + list with word-boundary matching, prefer longest
            detected_region = None
            if loader:
                import re as _re
                region_map = loader.get_region_name_mappings() or {}
                regions = loader.get_indian_regions() or []
                # Build candidate (alias, canonical) list including direct names
                candidates = []
                for alias, canonical in (region_map or {}).items():
                    if alias:
                        candidates.append((alias.strip(), canonical))
                for rname in regions:
                    if rname:
                        candidates.append((rname.strip(), rname))
                # Sort by alias length desc to prefer longer matches
                candidates.sort(key=lambda x: len(x[0]), reverse=True)
                for alias, canonical in candidates:
                    a = alias.lower()
                    # Use word-boundary matching for safety
                    pattern = r"\b" + _re.escape(a) + r"\b"
                    if _re.search(pattern, ql):
                        detected_region = loader.get_proper_region_name(canonical) or canonical
                        break

            # Extract year from query if present
            year_match = re.search(r"\b(19|20)\d{2}\b", ql)
            year_val = year_match.group(0) if year_match else None
            logger.info(f"Temporal refinement: Query '{ql}', detected year: {year_val}")

            # Extract month from query if present (e.g., June 2025)
            month_val = None
            month_map = {
                'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05', 'june': '06',
                'july': '07', 'august': '08', 'september': '09', 'october': '10', 'november': '11', 'december': '12'
            }
            for name, num in month_map.items():
                if re.search(rf"\b{name}\b", ql):
                    month_val = num
                    logger.info(f"Temporal refinement: Detected month '{name}' -> '{month_val}' in query")
                    break

            # Extract day from query if present (e.g., 15th June 2025)
            day_val = None
            day_match = re.search(r"\b(\d{1,2})(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+((19|20)\d{2})\b", ql)
            if day_match:
                day_val = day_match.group(1).zfill(2)
                month_val = month_map.get(day_match.group(3), month_val)
                year_val = year_val or day_match.group(4)

            # Apply temporal refinement to existing SQL if needed
            if (month_val or day_val) and year_val:
                logger.info(f"Temporal refinement: Applying filters - month: {month_val}, day: {day_val}, year: {year_val}")
                
                # Check if temporal filters are already present
                has_month_filter = re.search(r"(strftime\('%m',\s*[^)]+\)|DATEPART\(MONTH,\s*[^)]+\)|FORMAT\(\s*[^,]+,\s*'MM'\))\s*=\s*'\d{1,2}'", repaired, re.IGNORECASE)
                has_day_filter = re.search(r"(date\(\s*[^)]+\)|CONVERT\(date,\s*[^)]+\))\s*=\s*'\d{4}-\d{2}-\d{2}'", repaired, re.IGNORECASE)
                
                logger.info(f"Temporal refinement: Existing filters - month: {has_month_filter}, day: {has_day_filter}")
                
                # Determine the time column to use
                time_col = None
                logger.info(f"Temporal refinement: Checking SQL for time columns: {repaired}")
                if 'd.ActualDate' in repaired:
                    time_col = 'd.ActualDate'
                    logger.info(f"Temporal refinement: Found d.ActualDate in SQL")
                elif 'dt.ActualDate' in repaired:
                    time_col = 'dt.ActualDate'
                    logger.info(f"Temporal refinement: Found dt.ActualDate in SQL")
                elif 'fs.Timestamp' in repaired:
                    time_col = 'fs.Timestamp'
                    logger.info(f"Temporal refinement: Found fs.Timestamp in SQL")
                elif 'fs.TimeBlock' in repaired:
                    time_col = 'fs.TimeBlock'
                    logger.info(f"Temporal refinement: Found fs.TimeBlock in SQL")
                elif 'd.DateID' in repaired or 'dt.DateID' in repaired:
                    # For cases where SQL uses DateID join, we need to use the actual date column
                    # Check if DimDates is joined and infer the correct time column
                    if 'JOIN DimDates d ' in repaired or 'JOIN DimDates dt ' in repaired:
                        if 'JOIN DimDates d ' in repaired:
                            time_col = 'd.ActualDate'
                        else:
                            time_col = 'dt.ActualDate'
                        logger.info(f"Temporal refinement: Inferred time column from DateID join: {time_col}")
                    else:
                        logger.warning(f"Temporal refinement: Found DateID but no DimDates join")
                
                logger.info(f"Temporal refinement: Using time column: {time_col}")
                
                if time_col:
                    # Add month filter if not present
                    if month_val and not has_month_filter:
                        if re.search(r"\bWHERE\b", repaired, re.IGNORECASE):
                            # Insert month filter after existing WHERE
                            month_filter = f"DATEPART(MONTH, {time_col}) = {month_val}"
                            repaired = re.sub(r"\bWHERE\b", f"WHERE {month_filter} AND", repaired, count=1, flags=re.IGNORECASE)
                            logger.info(f"Temporal refinement: Added month filter after WHERE: {month_filter}")
                        else:
                            # Add WHERE clause with month filter
                            month_filter = f"DATEPART(MONTH, {time_col}) = {month_val}"
                            repaired += f" WHERE {month_filter}"
                            logger.info(f"Temporal refinement: Added WHERE clause with month filter: {month_filter}")
                    
                    # Add day filter if not present
                    if day_val and not has_day_filter:
                        date_iso = f"{year_val}-{month_val}-{day_val}"
                        day_filter = f"CONVERT(date, {time_col}) = '{date_iso}'"
                        if re.search(r"\bWHERE\b", repaired, re.IGNORECASE):
                            # Insert day filter after existing WHERE
                            repaired = re.sub(r"\bWHERE\b", f"WHERE {day_filter} AND", repaired, count=1, flags=re.IGNORECASE)
                            logger.info(f"Temporal refinement: Added day filter after WHERE: {day_filter}")
                        else:
                            # Add WHERE clause with day filter
                            repaired += f" WHERE {day_filter}"
                            logger.info(f"Temporal refinement: Added WHERE clause with day filter: {day_filter}")
                    
                    logger.info(f"Temporal refinement: Final SQL after refinement: {repaired}")
                else:
                    logger.warning(f"Temporal refinement: No time column found in SQL: {repaired}")
            else:
                logger.info(f"Temporal refinement: Skipping - month: {month_val}, day: {day_val}, year: {year_val}")

            def _determine_agg_from_query(query_lower: str) -> str:
                if any(k in query_lower for k in ["average", "avg", "mean"]):
                    return "AVG"
                if any(k in query_lower for k in ["maximum", "max", "highest", "peak"]):
                    return "MAX"
                if any(k in query_lower for k in ["minimum", "min", "lowest"]):
                    return "MIN"
                if any(k in query_lower for k in ["count", "number of", "how many"]):
                    return "COUNT"
                return "SUM"

            def build_state_sql(state_name: str, metric: str) -> str:
                agg = _determine_agg_from_query(ql)
                alias = "Value"
                where_parts = [f"s.StateName = '{state_name}'"]
                if year_val:
                    where_parts.append(f"DATEPART(YEAR, d.ActualDate) = {year_val}")
                if month_val or wants_monthly:
                    # If user said 'monthly' but didn't specify month name, do not filter by month
                    # Grouping handled by caller; here we only filter when an explicit month is present
                    if month_val:
                        where_parts.append(f"DATEPART(MONTH, d.ActualDate) = {month_val}")
                if day_val:
                    where_parts.append(f"CONVERT(date, d.ActualDate) = '{year_val}-{month_val}-{day_val}'")
                where_clause = " AND ".join(where_parts)
                if wants_monthly:
                    return (
                        "SELECT FORMAT(d.ActualDate, 'yyyy-MM') AS Month, "
                        f"ROUND({agg}(fs.{metric}), 2) AS {alias} "
                        "FROM FactStateDailyEnergy fs "
                        "JOIN DimStates s ON fs.StateID = s.StateID "
                        "JOIN DimDates d ON fs.DateID = d.DateID "
                        f"WHERE {where_clause} "
                        "GROUP BY FORMAT(d.ActualDate, 'yyyy-MM') ORDER BY FORMAT(d.ActualDate, 'yyyy-MM')"
                    )
                else:
                    return (
                        f"SELECT {agg}(fs.{metric}) AS {alias} "
                        f"FROM FactStateDailyEnergy fs "
                        f"JOIN DimStates s ON fs.StateID = s.StateID "
                        f"JOIN DimDates d ON fs.DateID = d.DateID "
                        f"WHERE {where_clause}"
                    )

            def build_region_sql(region_name: str, metric: str, *, monthly: bool = False, use_avg: bool = False) -> str:
                agg = _determine_agg_from_query(ql)
                alias = "Value"

                where_parts = [f"r.RegionName = '{region_name}'"]
                if year_val:
                    where_parts.append(f"DATEPART(YEAR, d.ActualDate) = {year_val}")
                if month_val:
                    where_parts.append(f"DATEPART(MONTH, d.ActualDate) = {month_val}")
                if day_val:
                    where_parts.append(f"CONVERT(date, d.ActualDate) = '{year_val}-{month_val}-{day_val}'")
                where_clause = " AND ".join(where_parts)

                if monthly:
                    return (
                        "SELECT FORMAT(d.ActualDate, 'yyyy-MM') AS Month, "
                        f"ROUND({agg}(fs.{metric}), 2) AS {alias} "
                        "FROM FactAllIndiaDailySummary fs "
                        "JOIN DimRegions r ON fs.RegionID = r.RegionID "
                        "JOIN DimDates d ON fs.DateID = d.DateID "
                        f"WHERE {where_clause} "
                        "GROUP BY FORMAT(d.ActualDate, 'yyyy-MM') "
                        "ORDER BY FORMAT(d.ActualDate, 'yyyy-MM')"
                    )
                else:
                    return (
                        f"SELECT {agg}(fs.{metric}) AS {alias} "
                        "FROM FactAllIndiaDailySummary fs "
                        "JOIN DimRegions r ON fs.RegionID = r.RegionID "
                        "JOIN DimDates d ON fs.DateID = d.DateID "
                        f"WHERE {where_clause}"
                    )

            def _build_outage_expr_for_table(schema_cols: List[str], outage_kind: str) -> Tuple[str, str]:
                cols_lower_to_original = {c.lower(): c for c in schema_cols}
                has = set(c.lower() for c in schema_cols)

                def find_col(*needles: str) -> Optional[str]:
                    for c in has:
                        if all(n in c for n in needles):
                            return cols_lower_to_original[c]
                    return None

                if outage_kind == "central":
                    chosen = find_col("central", "outage") or find_col("centraloutage")
                    if chosen:
                        return f"COALESCE(fs.{chosen}, 0)", "Central"
                if outage_kind == "state":
                    chosen = find_col("state", "outage") or find_col("stateoutage")
                    if chosen:
                        return f"COALESCE(fs.{chosen}, 0)", "State"
                if outage_kind == "private":
                    chosen = find_col("private", "outage") or find_col("priv", "outage")
                    if chosen:
                        return f"COALESCE(fs.{chosen}, 0)", "Private"

                # total or fallback: sum all outage-like columns
                outage_like = [cols_lower_to_original[c] for c in has if "outage" in c]
                if outage_like:
                    parts = [f"COALESCE(fs.{c}, 0)" for c in outage_like]
                    return " + ".join(parts), ("Total" if outage_kind == "total" else "Outage")
                # ultimate fallback
                return "0", ("Total" if outage_kind == "total" else outage_kind.capitalize())

            def build_outage_sql(
                fact_table: str,
                outage_kind: str,
                *,
                monthly: bool = False,
                group_by: Optional[str] = None,
                region_name: Optional[str] = None,
            ) -> str:
                """Generic outage SQL builder. Detects outage columns at table level, supports monthly/group-by/filters."""
                agg = _determine_agg_from_query(ql)
                schema_cols = schema.get(fact_table, [])
                expr, alias = _build_outage_expr_for_table(schema_cols, outage_kind)

                join_parts: List[str] = []
                # Need dates for filtering/grouping
                join_parts.append("JOIN DimDates d ON fs.DateID = d.DateID")
                # Regions join if used
                if group_by == "r.RegionName" or region_name is not None:
                    join_parts.append("JOIN DimRegions r ON fs.RegionID = r.RegionID")

                where_parts: List[str] = []
                if region_name:
                    where_parts.append(f"r.RegionName = '{region_name}'")
                if year_val:
                    where_parts.append(f"DATEPART(YEAR, d.ActualDate) = {year_val}")
                where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

                if monthly:
                    select_month = "FORMAT(d.ActualDate, 'yyyy-MM') AS Month"
                    group_cols = []
                    if group_by:
                        group_cols.append(group_by)
                    group_cols.append("Month")
                    group_by_sql = ", ".join(group_cols)
                    order_by_sql = group_by_sql
                    select_prefix = (group_by + ", " if group_by else "") + select_month
                    return (
                        f"SELECT {select_prefix}, ROUND({agg}({expr}), 2) AS {alias} "
                        f"FROM {fact_table} fs " + " ".join(join_parts) + where_clause + " "
                        f"GROUP BY {group_by_sql} ORDER BY {order_by_sql}"
                    )
                else:
                    select_prefix = (group_by + ", " if group_by else "")
                    end_group = f" GROUP BY {group_by}" if group_by else ""
                    end_order = f" ORDER BY {group_by}" if group_by else ""
                    return (
                        f"SELECT {select_prefix}ROUND({agg}({expr}), 2) AS {alias} "
                        f"FROM {fact_table} fs " + " ".join(join_parts) + where_clause + end_group + end_order
                    )

            def build_region_outage_sql(region_name: str, outage_kind: str, *, monthly: bool = False) -> str:
                return build_outage_sql(
                    "FactAllIndiaDailySummary", outage_kind, monthly=monthly, group_by=None, region_name=region_name
                )

            # Interpret intent for grouping/aggregation
            wants_monthly = ("monthly" in ql) or ("per month" in ql) or ("by month" in ql)
            wants_average = ("average" in ql) or ("avg" in ql) or ("mean" in ql)
            wants_all_regions = any(p in ql for p in ["all regions", "across regions", "by region", "regionwise", "region wise"])
            wants_all_states = any(p in ql for p in ["all states", "across states", "by state", "statewise", "state wise"])

            # Helpers to build ALL-regions/states grouped SQL
            def build_all_regions_sql(metric: str, monthly: bool, use_avg: bool) -> str:
                agg = _determine_agg_from_query(ql)
                alias = "Value"
                where_year = f"WHERE DATEPART(YEAR, d.ActualDate) = {year_val}" if year_val else ""
                if monthly:
                    return (
                        "SELECT r.RegionName, FORMAT(d.ActualDate, 'yyyy-MM') AS Month, "
                        f"ROUND({agg}(fs.{metric}), 2) AS {alias} "
                        "FROM FactAllIndiaDailySummary fs "
                        "JOIN DimRegions r ON fs.RegionID = r.RegionID "
                        "JOIN DimDates d ON fs.DateID = d.DateID "
                        f"{where_year} "
                        "GROUP BY r.RegionName, FORMAT(d.ActualDate, 'yyyy-MM') "
                        "ORDER BY r.RegionName, FORMAT(d.ActualDate, 'yyyy-MM')"
                    )
                else:
                    return (
                        "SELECT r.RegionName, "
                        f"ROUND({agg}(fs.{metric}), 2) AS {alias} "
                        "FROM FactAllIndiaDailySummary fs "
                        "JOIN DimRegions r ON fs.RegionID = r.RegionID "
                        "JOIN DimDates d ON fs.DateID = d.DateID "
                        f"{where_year} "
                        "GROUP BY r.RegionName "
                        "ORDER BY r.RegionName"
                    )

            def build_all_regions_outage_sql(outage_kind: str, monthly: bool) -> str:
                return build_outage_sql(
                    "FactAllIndiaDailySummary", outage_kind, monthly=monthly, group_by="r.RegionName", region_name=None
                )

            def build_all_states_sql(metric: str, monthly: bool, use_avg: bool) -> str:
                agg = _determine_agg_from_query(ql)
                alias = "Value"
                where_year = f"WHERE DATEPART(YEAR, d.ActualDate) = {year_val}" if year_val else ""
                if monthly:
                    # If query mentions a specific region, restrict states to that region
                    region_filter_sql = ""
                    if detected_region:
                        region_filter_sql = f" AND r.RegionName = '{detected_region}' "
                    return (
                        "SELECT s.StateName, FORMAT(d.ActualDate, 'yyyy-MM') AS Month, "
                        f"ROUND({agg}(fs.{metric}), 2) AS {alias} "
                        "FROM FactStateDailyEnergy fs "
                        "JOIN DimStates s ON fs.StateID = s.StateID "
                        "JOIN DimDates d ON fs.DateID = d.DateID "
                        "JOIN DimRegions r ON s.RegionID = r.RegionID "
                        f"{where_year}{region_filter_sql} "
                        "GROUP BY s.StateName, FORMAT(d.ActualDate, 'yyyy-MM') "
                        "ORDER BY s.StateName, FORMAT(d.ActualDate, 'yyyy-MM')"
                    )
                else:
                    region_filter_sql = ""
                    if detected_region:
                        region_filter_sql = f" AND r.RegionName = '{detected_region}' "
                    return (
                        "SELECT s.StateName, "
                        f"ROUND({agg}(fs.{metric}), 2) AS {alias} "
                        "FROM FactStateDailyEnergy fs "
                        "JOIN DimStates s ON fs.StateID = s.StateID "
                        "JOIN DimDates d ON fs.DateID = d.DateID "
                        "JOIN DimRegions r ON s.RegionID = r.RegionID "
                        f"{where_year}{region_filter_sql} "
                        "GROUP BY s.StateName "
                        "ORDER BY s.StateName"
                    )

            # International exchange (country) detection
            is_exchange = any(k in ql for k in ["exchange", "import", "export", "international", "cross border", "cross-border"])
            mentioned_country = None
            if is_exchange:
                # naive extraction: pick any capitalized token present; prioritize known countries if schema has DimCountries
                import re as _re
                tokens = _re.findall(r"[A-Za-z][A-Za-z\-]+", query or "")
                # Prefer Bangladesh if present (common case reported)
                for cand in tokens:
                    if cand.lower() in {"bangladesh", "nepal", "bhutan", "sri", "sri-lanka", "china", "myanmar", "pakistan"}:
                        mentioned_country = "Bangladesh" if cand.lower()=="bangladesh" else cand.title()
                        break
                # Fallback: detect any token ending with 'stan' or typical country naming; leave None if unsure

            def _find_col_case_insensitive(cols: List[str], *needles: str) -> Optional[str]:
                lower_map = {c.lower(): c for c in cols}
                for lc, orig in lower_map.items():
                    if all(n in lc for n in needles):
                        return orig
                return None

            def build_country_exchange_sql(country_name: str, *, monthly: bool = False) -> Optional[str]:
                fact_name = None
                # Prefer FactCountryDailyExchange, else InternationalTransmission
                if "FactCountryDailyExchange" in schema:
                    fact_name = "FactCountryDailyExchange"
                elif "FactInternationalTransmissionLinkFlow" in schema:
                    fact_name = "FactInternationalTransmissionLinkFlow"
                if not fact_name:
                    return None

                fcols = schema.get(fact_name, [])
                dcols = schema.get("DimDates", [])
                ccols = schema.get("DimCountries", [])

                # Choose exchange expression
                expr = (
                    _find_col_case_insensitive(fcols, "net", "exchange")
                    or _find_col_case_insensitive(fcols, "total", "exchange")
                    or _find_col_case_insensitive(fcols, "exchange")
                    or (
                        # try import/export sum
                        (lambda imp, exp: (imp and exp and f"COALESCE(fx.{imp},0)+COALESCE(fx.{exp},0)") or None)(
                            _find_col_case_insensitive(fcols, "import"),
                            _find_col_case_insensitive(fcols, "export"),
                        )
                    )
                )
                if not expr:
                    return None

                # Determine date filtering
                has_date_join = "DateID" in fcols and "DateID" in dcols
                date_join = "JOIN DimDates d ON fx.DateID = d.DateID" if has_date_join else ""
                date_filter = f"DATEPART(YEAR, d.ActualDate) = {year_val}" if (year_val and has_date_join) else None

                # Determine country filter/join
                where_parts: List[str] = []
                has_country_join = "CountryID" in fcols and "CountryID" in ccols
                cn_name_col = _find_col_case_insensitive(ccols, "country", "name") if ccols else None
                fx_country_name_col = _find_col_case_insensitive(fcols, "country", "name") if fcols else None

                if has_country_join and cn_name_col:
                    country_join = f"JOIN DimCountries c ON fx.CountryID = c.CountryID"
                    where_parts.append(f"c.{cn_name_col} = '{country_name}'")
                elif fx_country_name_col:
                    country_join = ""
                    where_parts.append(f"fx.{fx_country_name_col} = '{country_name}'")
                else:
                    # Cannot filter by country safely
                    return None

                if date_filter:
                    where_parts.append(date_filter)
                where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

                if monthly and has_date_join:
                    return (
                        "SELECT FORMAT(d.ActualDate, 'yyyy-MM') AS Month, "
                        f"ROUND(SUM({expr}), 2) AS TotalExchange "
                        f"FROM {fact_name} fx "
                        f"{country_join} {date_join}"
                        f"{where_clause} "
                        "GROUP BY Month ORDER BY Month"
                    )
                else:
                    return (
                        f"SELECT ROUND(SUM({expr}), 2) AS TotalExchange "
                        f"FROM {fact_name} fx "
                        f"{country_join} {date_join}"
                        f"{where_clause}"
                    )

            # Generation by source detection and builder
            is_generation = ("generation" in ql)
            mentions_source = ("by source" in ql) or ("source" in ql)

            def _find_any(cols: List[str], *needles: str) -> Optional[str]:
                lower_map = {c.lower(): c for c in cols}
                for lc, orig in lower_map.items():
                    if all(n in lc for n in needles):
                        return orig
                return None

            def _pick_generation_value_column(fact_cols: List[str]) -> Optional[str]:
                # Prioritized picking of a numeric generation value column
                cols_lower = [c.lower() for c in fact_cols]
                name_map = {c.lower(): c for c in fact_cols}
                # 1) Exact popular names
                for exact in ("generationamount", "totalgeneration", "generation_mw", "generationmu", "generation"):
                    if exact in name_map:
                        return name_map[exact]
                # 2) Contains both 'generation' and 'amount'
                for lc in cols_lower:
                    if "generation" in lc and "amount" in lc and not lc.endswith("id"):
                        return name_map[lc]
                # 3) Contains 'generation' and unit hints
                for lc in cols_lower:
                    if "generation" in lc and any(u in lc for u in ["mw", "mu", "mwh", "kwh"]) and not lc.endswith("id"):
                        return name_map[lc]
                # 4) Generic 'amount' not an id
                for lc in cols_lower:
                    if "amount" in lc and not lc.endswith("id"):
                        return name_map[lc]
                # 5) Last resort: any non-ID column containing 'gen'
                for lc in cols_lower:
                    if "gen" in lc and not lc.endswith("id"):
                        return name_map[lc]
                return None

            def build_generation_by_source_sql(monthly: bool) -> Optional[str]:
                fact = None
                if "FactDailyGenerationBreakdown" in schema:
                    fact = "FactDailyGenerationBreakdown"
                elif "FactTimeBlockGeneration" in schema:
                    fact = "FactTimeBlockGeneration"
                if not fact:
                    return None

                fcols = schema.get(fact, [])
                dcols = schema.get("DimDates", [])
                gs_cols = schema.get("DimGenerationSources", [])

                gen_col = _pick_generation_value_column(fcols)
                if not gen_col:
                    return None

                # Join to dates if possible
                has_date_join = ("DateID" in fcols and "DateID" in dcols)
                date_join = "JOIN DimDates d ON fg.DateID = d.DateID" if has_date_join else ""
                date_filter = f"DATEPART(YEAR, d.ActualDate) = {year_val}" if (year_val and has_date_join) else None

                # Join to generation sources if possible
                has_gs_join = ("GenerationSourceID" in fcols and "GenerationSourceID" in gs_cols)
                # Force alias to dgs for consistency with tests
                dgs_join = "JOIN DimGenerationSources dgs ON fg.GenerationSourceID = dgs.GenerationSourceID" if has_gs_join else ""
                # Prefer SourceName column when available
                source_name_col = _find_any(gs_cols, "source", "name") if has_gs_join else None

                where_parts: List[str] = []
                if "renewable" in ql:
                    category_col = _find_any(gs_cols, "category") if has_gs_join else None
                    if category_col:
                        where_parts.append(f"dgs.{category_col} = 'Renewable'")
                if date_filter:
                    where_parts.append(date_filter)
                where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

                if monthly and has_date_join:
                    # Always expose dgs.SourceName for grouping when join exists; fallback to ID otherwise
                    group_key = (has_gs_join and (source_name_col and "dgs." + source_name_col or "dgs.GenerationSourceID")) or "fg.GenerationSourceID"
                    select_key = (has_gs_join and (source_name_col and "dgs." + source_name_col + " AS Source, " or "dgs.GenerationSourceID AS Source, ")) or ""
                    group_clause = ("dgs." + source_name_col) if (has_gs_join and source_name_col) else "Source"
                    return (
                        "SELECT "
                        + select_key
                        + "FORMAT(d.ActualDate, 'yyyy-MM') AS Month, "
                        + f"ROUND(SUM(fg.{gen_col}), 2) AS TotalGeneration "
                        + f"FROM {fact} fg "
                        + (dgs_join + " ")
                        + (date_join + " ")
                        + where_clause + " "
                        + ("GROUP BY " + (group_clause + ", " if select_key else "") + "Month ORDER BY " + (group_clause + ", " if select_key else "") + "Month")
                    )
                else:
                    select_key = (has_gs_join and (source_name_col and "dgs." + source_name_col + " AS Source, " or "dgs.GenerationSourceID AS Source, ")) or ""
                    group_key = ("Source" if select_key else "")
                    return (
                        "SELECT " + select_key + f"ROUND(SUM(fg.{gen_col}), 2) AS TotalGeneration "
                        + f"FROM {fact} fg "
                        + (dgs_join + " ")
                        + (date_join + " ")
                        + where_clause + " "
                        + ("GROUP BY " + group_key if select_key else "")
                    )

            # Outage detection and kind
            is_outage = ("outage" in ql)
            outage_kind = "total"
            if is_outage:
                if "central" in ql:
                    outage_kind = "central"
                elif "state" in ql:
                    outage_kind = "state"
                elif "private" in ql:
                    outage_kind = "private"

            if is_exchange and mentioned_country:
                sql_country = build_country_exchange_sql(mentioned_country, monthly=wants_monthly)
                if sql_country:
                    repaired = sql_country
                else:
                    # fall back to default behavior if unable to build
                    pass
            elif is_generation and mentions_source:
                sql_gen = build_generation_by_source_sql(wants_monthly)
                if sql_gen:
                    repaired = sql_gen
            elif wants_all_regions:
                if is_outage:
                    repaired = build_all_regions_outage_sql(outage_kind, wants_monthly)
                else:
                    region_metric = (
                        "EnergyShortage" if "shortage" in ql else (
                        "MaxDemandSCADA" if ("maximum demand" in ql or "max demand" in ql or "peak demand" in ql) else "EnergyMet")
                    )
                    repaired = build_all_regions_sql(region_metric, wants_monthly, wants_average)
            elif wants_all_states:
                state_metric = (
                    "EnergyShortage" if "shortage" in ql else (
                    "MaximumDemand" if ("maximum demand" in ql or "max demand" in ql or "peak demand" in ql) else "EnergyMet")
                )
                repaired = build_all_states_sql(state_metric, wants_monthly, wants_average)
            elif detected_states:
                state_name = detected_states[0]
                metric = pick_metric("EnergyMet")
                repaired = build_state_sql(state_name, metric)
            else:
                # Determine region-level metric from query
                if is_outage:
                    target_region = detected_region or "India"
                    repaired = build_region_outage_sql(target_region, outage_kind, monthly=wants_monthly)
                else:
                    region_metric = (
                        "EnergyShortage" if "shortage" in ql else (
                        "MaxDemandSCADA" if ("maximum demand" in ql or "max demand" in ql or "peak demand" in ql) else "EnergyMet")
                    )
                    if detected_region:
                        repaired = build_region_sql(detected_region, region_metric, monthly=wants_monthly, use_avg=wants_average)
                    else:
                        # No explicit region: default to All-India for any region-level metric
                        repaired = build_region_sql("India", region_metric, monthly=wants_monthly, use_avg=wants_average)

            # Remove unrelated metrics from SELECT (e.g., MaxDemandSCADA when asking consumption)
            if "consumption" in ql or ("energy" in ql and "shortage" not in ql and not detected_states):
                # Drop any MaxDemandSCADA aggregates from SELECT list
                repaired = re.sub(r",\s*SUM\([^)]*MaxDemandSCADA[^)]*\)\s+AS\s+[A-Za-z_]\w*", "", repaired, flags=re.IGNORECASE)

            return repaired
        except Exception:
            return sql

    async def advanced_retrieve(self, query: str, context: Dict[str, Any] = None, 
                              top_k: int = 5) -> List[RetrievalResult]:
        """
        Perform advanced retrieval using hybrid search
        """
        try:
            if context:
                # Use contextual retrieval
                results = await self.contextual_retrieval.retrieve_with_context(
                    query, context, top_k
                )
            else:
                # Use basic hybrid search
                results = self.advanced_retrieval.hybrid_search(query, top_k)
                
            # Update statistics
            if results:
                self.stats["hybrid_retrieval_rate"] = (
                    (self.stats["hybrid_retrieval_rate"] * (self.stats["total_requests"] - 1) + 1) 
                    / self.stats["total_requests"]
                ) if self.stats["total_requests"] > 0 else 1.0
                
            return results
            
        except Exception as e:
            logger.error(f"Advanced retrieval failed: {e}")
            return []
            
    async def process_query_enhanced(self, query: str, processing_mode: str = "adaptive", 
                                   session_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        Process query with enhanced semantic understanding and Wren AI integration.
        Implements a unified, step-based pipeline:
        1) Semantic analysis (MDL-aware)
        2) Multi-candidate SQL generation (Wren MDL, Semantic Engine, Traditional/Template, Heuristic)
        3) Auto-repair + MDL/business-rule validation
        4) Execution probing and scoring
        5) Best-candidate selection
        6) Visualization recommendation
        """
        start_time = datetime.now()
        
        try:
            # Ensure initialization
            if not self._initialized:
                await self.initialize()
                
            # Extract semantic context using Wren AI integration
            semantic_context = await self.wren_ai_integration.extract_semantic_context(query) if hasattr(self, 'wren_ai_integration') else {}
            
            # Extract temporal constraints once
            temporal_constraints = self.temporal_processor.extract_temporal_constraints(query)
            
            # Unified enhanced pipeline
            logger.info(f"🚀 Starting unified enhanced pipeline for query: {query}")
            try:
                unified_result = await self._run_unified_enhanced_pipeline(query, semantic_context)
                result = unified_result
                logger.info(f"🚀 Unified pipeline result: success={result.get('success')}, method={result.get('processing_method')}")
            except Exception as e:
                logger.error(f"🚀 Unified pipeline failed with exception: {e}")
                unified_result = {"success": False, "error": f"Unified pipeline exception: {str(e)}"}
                result = unified_result
            
            # Fallback strategy
            if not result.get("success", False):
                logger.info(f"🚀 Unified pipeline failed, falling back to adaptive strategy")
                confidence = semantic_context.get("confidence", 0.0) if isinstance(semantic_context, dict) else 0.0
                actual_mode = self._determine_processing_mode(confidence, processing_mode)
                if actual_mode == ProcessingMode.SEMANTIC_FIRST:
                    result = await self._process_semantic_first(query, semantic_context)
                elif actual_mode == ProcessingMode.HYBRID:
                    result = await self._process_hybrid(query, semantic_context)
                elif actual_mode == ProcessingMode.AGENTIC_WORKFLOW:
                    result = await self._process_agentic(query, semantic_context)
                else:
                    result = await self._process_traditional(query, semantic_context)
            
            # Ensure temporal constraints are applied regardless of path
            if result and result.get("sql"):
                sql_with_temporal = self.temporal_processor.apply_temporal_constraints(result["sql"], temporal_constraints)
                if sql_with_temporal != result["sql"]:
                    result["sql"] = sql_with_temporal
            
            # Step 4: Update statistics (feedback storage disabled for cloud deployment)
            self._update_statistics(start_time, semantic_context, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced query processing failed: {e}")
            # Error feedback storage disabled for cloud deployment
            return {
                "success": False,
                "error": str(e),
                "processing_method": "enhanced_rag"
            }

    async def _run_unified_enhanced_pipeline(self, query: str, semantic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Unified step-based pipeline with multi-candidate generation, validation, and selection."""
        # Step A: Generate candidates in parallel
        candidates = await self._generate_candidate_sqls(query, semantic_context)
        if not candidates:
            return {"success": False, "error": "No SQL candidates generated", "processing_method": "enhanced_unified"}
        
        logger.info(f"🎯 Generated {len(candidates)} candidates:")
        for i, cand in enumerate(candidates):
            logger.info(f"  {i+1}. {cand.get('source')}: {cand.get('sql', '')[:100]}...")

        # Step B: Validate, repair, and probe-execute to score candidates
        scored = await self._validate_and_score_candidates(query, candidates, semantic_context)
        if not scored:
            return {"success": False, "error": "No valid candidates after validation", "processing_method": "enhanced_unified"}

        # Pick best by score, but relax for flow/time-block: prefer specialized-table candidates
        ql = (query or '').lower()
        def is_special(sql: str) -> bool:
            ls = (sql or '').lower()
            return any(t in ls for t in [
                'factinternationaltransmissionlinkflow',
                'facttransmissionlinkflow',
                'facttimeblockpowerdata',
                'facttimeblockgeneration',
            ])
        if any(k in ql for k in ["time block", "hourly", "15 min", "15-minute", "intraday", "flow", "link flow", "linkflow", "energy exchanged"]):
            specials = [c for c in scored if is_special(c.get('sql',''))]
            best = max(specials or scored, key=lambda x: x.get("score", 0.0))
        else:
            best = max(scored, key=lambda x: x.get("score", 0.0))
        executed = best.get("execution")
        
        logger.info(f"🏆 Selected best candidate:")
        logger.info(f"  - Source: {best.get('source')}")
        logger.info(f"  - Score: {best.get('score', 0.0)}")
        logger.info(f"  - SQL: {best.get('sql', '')[:200]}...")
        logger.info(f"  - Success: {executed.success if executed else 'Unknown'}")

        # Step C: Visualization
        plot = None
        try:
            if executed and executed.success and executed.data:
                viz_agent = VisualizationAgent()
                viz_dict = await viz_agent._generate_visualization(executed.data, query)
                if viz_dict:
                    plot = {"chartType": viz_dict.get("chart_type", "bar"), "options": viz_dict.get("config", {})}
        except Exception:  # noqa: E722
            plot = None

        # Step D: Build keyword summary
        keyword_summary = self._build_keyword_summary(query, semantic_context)

        if executed and executed.success:
            return {
                "success": True,
                "sql": best.get("sql"),
                "data": executed.data,
                "plot": plot,
                "confidence": best.get("score", 0.7),
                "processing_method": "enhanced_unified",
                "candidate_count": len(candidates),
                "selected_candidate_source": best.get("source"),
                "keyword_summary": keyword_summary,
            }
        else:
            return {
                "success": False,
                "error": (executed.error if executed else "Execution failed"),
                "sql": best.get("sql"),
                "data": [],
                "processing_method": "enhanced_unified_failed",
                "candidate_count": len(candidates),
                "selected_candidate_source": best.get("source"),
                "keyword_summary": keyword_summary,
            }

    async def _generate_candidate_sqls(self, query: str, semantic_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multiple SQL candidates from different strategies."""
        candidates: List[Dict[str, Any]] = []

        # Candidate 1: Wren MDL-aware
        try:
            logger.info(f"🔍 Generating Wren MDL candidate for query: {query[:100]}...")
            wren_result = await self.wren_ai_integration.generate_mdl_aware_sql(query, semantic_context)
            logger.info(f"📋 Wren MDL result: {wren_result}")
            
            if wren_result and wren_result.get("sql"):
                candidates.append({"source": "wren_mdl", "sql": wren_result["sql"]})
                logger.info(f"✅ Wren MDL candidate added successfully")
            else:
                logger.warning(f"⚠️  Wren MDL returned no SQL: {wren_result}")
        except Exception as e:
            logger.error(f"❌ Wren MDL candidate failed with exception: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

        # Candidate 2: Semantic Engine
        try:
            logger.info(f"🔍 Generating Semantic Engine candidate for query: {query[:100]}...")
            sem = await self.semantic_engine.generate_contextual_sql(query, semantic_context, {})
            logger.info(f"📋 Semantic Engine result: {sem}")
            
            if sem and sem.get("sql"):
                candidates.append({"source": "semantic_engine", "sql": sem["sql"]})
                logger.info(f"✅ Semantic Engine candidate added successfully")
            else:
                logger.warning(f"⚠️  Semantic Engine returned no SQL: {sem}")
        except Exception as e:
            logger.error(f"❌ Semantic Engine candidate failed with exception: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

        # Candidate 3: Traditional/Assembler
        try:
            from backend.core.intent import IntentAnalyzer
            
            # Use proper intent analysis instead of hardcoded values
            intent_analyzer = IntentAnalyzer()
            analysis = await intent_analyzer.analyze_intent(query, self.llm_provider)
            
            # Create context with proper analysis and schema linker
            schema_map = {}
            try:
                executor = SQLExecutor(self.db_path)
                schema_map = executor.get_schema_info()  # {table: [columns]}
            except Exception as e:
                logger.warning(f"SQLExecutor unavailable, using lightweight schema introspection: {e}")
                # Lightweight schema introspection from SQLite - REMOVED for cloud deployment
                # try:
                #     import sqlite3
                #     conn = sqlite3.connect(self.db_path)
                #     cur = conn.cursor()
                #     cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                #     tables = [row[0] for row in cur.fetchall()]
                #     for tname in tables:
                #         try:
                #             cur.execute(f"PRAGMA table_info({tname})")
                #             cols = [r[1] for r in cur.fetchall()]
                #             schema_map[tname] = cols
                #         except Exception:
                #             schema_map[tname] = []
                #     conn.close()
                # except Exception as e2:
                #     logger.error(f"Lightweight schema introspection failed: {e2}")
                logger.info("Schema introspection will use main database connection for cloud deployment")
            context = ContextInfo(
                query_analysis=analysis,
                schema_info=SchemaInfo(tables=schema_map or {}),
                dimension_values={},
                user_mappings=[],
                memory_context=None,
                schema_linker=SchemaLinker(schema_map or {}, self.db_path, self.llm_provider),
                llm_provider=self.llm_provider,
            )
            sql_result = self.sql_assembler.generate_sql(query, analysis, context)
            if sql_result and hasattr(sql_result, 'sql') and sql_result.sql:
                candidates.append({"source": "assembler", "sql": sql_result.sql})
        except Exception as e:
            logger.warning(f"Assembler candidate failed: {e}")

        # Candidate 4: Heuristic builder via validation-correction (handles all regions/states, India default)
        try:
            heuristic_sql = self._validate_and_correct_sql("", query)
            if heuristic_sql and heuristic_sql.strip():
                candidates.append({"source": "heuristic", "sql": heuristic_sql})
        except Exception as e:
            logger.warning(f"Heuristic candidate failed: {e}")

        # Candidate 5: Explicit-table builder using MDL hints
        try:
            explicit_tables_str = (semantic_context or {}).get("semantic_mappings", {}).get("explicit_tables")
            detected_tables: set = set()
            if explicit_tables_str:
                detected_tables |= {t.strip() for t in explicit_tables_str.split(',') if t.strip()}
            # Regex fallback: scan query text for table-like tokens starting with 'Fact'
            import re as _re
            for t in _re.findall(r"fact[a-zA-Z0-9_]+", (query or ''), flags=_re.IGNORECASE):
                detected_tables.add(t)
            for tname in detected_tables:
                sql = self._build_explicit_table_sql(tname, query)
                if sql:
                    # Mark this candidate as explicitly matching the requested table
                    candidates.append({
                        "source": "explicit_builder", 
                        "sql": sql,
                        "explicit_match": True,  # CRITICAL: Mark for explicit table selection
                        "explicit_table": tname.lower()  # Track which table was matched
                    })
                    logger.info(f"✅ Explicit table builder candidate added for {tname} with explicit_match=True")
        except Exception as e:
            logger.warning(f"Explicit-table builder failed: {e}")

        # Candidate 6: Domain-enforced synthesis for known patterns (time-block and link flow)
        try:
            ql = (query or '').lower()
            # Sub-daily → time-block
            if any(k in ql for k in ["time block", "time-block", "hourly", "15 min", "15-minute", "intraday"]):
                target = "FactTimeBlockGeneration" if any(k in ql for k in ["by source", "source", "generation"]) else "FactTimeBlockPowerData"
                if hasattr(self.wren_ai_integration, '_synthesize_sql_for_table'):
                    synth = self.wren_ai_integration._synthesize_sql_for_table(target, query)
                    if synth:
                        candidates.append({"source": "domain_enforced", "sql": synth})
                        logger.info(f"✅ Domain-enforced candidate added for {target}")
            
            # International link flow
            if ("flow" in ql or "link flow" in ql or "linkflow" in ql) and (("international" in ql) or any(c in ql for c in ["bangladesh","nepal","bhutan","myanmar", "sri lanka", "sri-lanka", "pakistan", "china"])):
                if hasattr(self.wren_ai_integration, '_synthesize_sql_for_table'):
                    synth = self.wren_ai_integration._synthesize_sql_for_table("FactInternationalTransmissionLinkFlow", query)
                    if synth:
                        candidates.append({"source": "domain_enforced", "sql": synth})
                        logger.info(f"✅ Domain-enforced candidate added for FactInternationalTransmissionLinkFlow")
            
            # Domestic transmission link flow
            if ("flow" in ql or "link flow" in ql or "linkflow" in ql) and ("transmission" in ql or "link" in ql):
                if hasattr(self.wren_ai_integration, '_synthesize_sql_for_table'):
                    synth = self.wren_ai_integration._synthesize_sql_for_table("FactTransmissionLinkFlow", query)
                    if synth:
                        candidates.append({"source": "domain_enforced", "sql": synth})
                        logger.info(f"✅ Domain-enforced candidate added for FactTransmissionLinkFlow")
        except Exception as e:
            logger.warning(f"Domain-enforced synthesis failed: {e}")

        # Log final candidate summary
        logger.info(f"🎯 Generated {len(candidates)} SQL candidates:")
        for i, candidate in enumerate(candidates):
            logger.info(f"  {i+1}. {candidate['source']}: {candidate['sql'][:100]}...")

        # Deduplicate by normalized SQL
        seen = set()
        unique: List[Dict[str, Any]] = []
        for c in candidates:
            key = (c.get("sql") or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    async def _validate_and_score_candidates(self, query: str, candidates: List[Dict[str, Any]], semantic_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Auto-repair, validate against schema/business rules, and probe execute to score candidates.
        Prefers candidates that use explicitly mentioned table names in the question/semantic mappings.
        """
        scored: List[Dict[str, Any]] = []
        # Detect explicit tables from semantic context (Wren MDL mapping)
        explicit_tables: set = set()
        try:
            explicit_tables_str = (semantic_context or {}).get("semantic_mappings", {}).get("explicit_tables")
            if explicit_tables_str:
                explicit_tables = {t.strip().lower() for t in explicit_tables_str.split(",") if t.strip()}
        except Exception:
            explicit_tables = set()
        
        # Primary detection: scan query text for table-like tokens starting with 'Fact'
        if not explicit_tables:
            try:
                import re as _re
                detected = _re.findall(r"fact[a-zA-Z0-9_]+", (query or ''), flags=_re.IGNORECASE)
                explicit_tables = {t.lower() for t in detected}
                logger.info(f"🔍 Regex detected explicit tables: {explicit_tables}")
            except Exception as e:
                logger.warning(f"Regex detection failed: {e}")
        
        # Secondary detection: check against MDL model names
        if not explicit_tables:
            try:
                ql_detect = (query or "").lower()
                mdl = getattr(self, 'wren_ai_integration', None)
                schema = getattr(mdl, 'mdl_schema', None)
                model_names = list(schema.models.keys()) if (schema and getattr(schema, 'models', None)) else []
                for tname in model_names:
                    if tname and tname.lower() in ql_detect:
                        explicit_tables.add(tname.lower())
                logger.info(f"🔍 MDL model detection found: {explicit_tables}")
            except Exception as e:
                logger.warning(f"MDL model detection failed: {e}")
        
        logger.info(f"🎯 Final explicit tables detected: {explicit_tables}")

        # Detect sub-daily intent
        ql_global = (query or "").lower()
        is_sub_daily = any(k in ql_global for k in ["time block", "time-block", "hourly", "15 min", "15-minute", "intraday"])

        for cand in candidates:
            raw_sql = cand.get("sql") or ""
            
            # Skip auto-repair for Wren MDL and explicit_builder candidates - they are already correct
            if cand.get("source") in ["wren_mdl", "explicit_builder"]:
                repaired = raw_sql  # Use original SQL without modification
                logger.info(f"🔒 Preserving {cand.get('source')} SQL without auto-repair: {raw_sql[:100]}...")
            else:
                # Auto-repair and MDL/business-rule correction for other candidates
                repaired = self._auto_repair_sql(raw_sql)
            # If sub-daily, force-correct to time-block models when candidate uses daily models
            if is_sub_daily:
                lower_rs = repaired.lower()
                if ("factallindiadailysummary" in lower_rs) or ("factstatedailyenergy" in lower_rs) or ("factdailygenerationbreakdown" in lower_rs):
                    # Prefer power vs generation variant
                    target = "FactTimeBlockGeneration" if any(k in ql_global for k in ["by source", "source", "generation"]) else "FactTimeBlockPowerData"
                    synth = self.wren_ai_integration._synthesize_sql_for_table(target, query) if hasattr(self.wren_ai_integration, '_synthesize_sql_for_table') else None
                    if synth:
                        repaired = synth
            else:
                # Transmission / International link flow enforcement
                lrq = ql_global
                lower_rs = repaired.lower()
                wants_flow = ("flow" in lrq) or ("link flow" in lrq) or ("linkflow" in lrq)
                if wants_flow and ("international" in lrq or any(c in lrq for c in ["bangladesh","nepal","bhutan","myanmar"])):
                    if "factinternationaltransmissionlinkflow" not in lower_rs and hasattr(self.wren_ai_integration, '_synthesize_sql_for_table'):
                        synth = self.wren_ai_integration._synthesize_sql_for_table("FactInternationalTransmissionLinkFlow", query)
                        if synth:
                            repaired = synth
                elif wants_flow and ("transmission" in lrq or "link" in lrq):
                    if "facttransmissionlinkflow" not in lower_rs and hasattr(self.wren_ai_integration, '_synthesize_sql_for_table'):
                        synth = self.wren_ai_integration._synthesize_sql_for_table("FactTransmissionLinkFlow", query)
                        if synth:
                            repaired = synth
            # Always apply temporal refinement and validation, but preserve explicit table usage when appropriate
            # Skip validation/correction for Wren MDL and explicit_builder candidates - they are already correct
            if cand.get("source") in ["wren_mdl", "explicit_builder"]:
                corrected = repaired  # Use repaired SQL without further modification
                logger.info(f"🔒 Preserving {cand.get('source')} SQL without validation/correction")
            else:
                corrected = self._validate_and_correct_sql(repaired, query)
            # Enforce explicit table usage if requested in the query and not present in SQL
            try:
                if explicit_tables:
                    lower_corr = (corrected or "").lower()
                    if not any(t in lower_corr for t in explicit_tables):
                        first_explicit = next(iter(explicit_tables))
                        # Map to canonical case if present in MDL
                        mdl = getattr(self, 'wren_ai_integration', None)
                        schema = getattr(mdl, 'mdl_schema', None)
                        canonical = None
                        if schema and getattr(schema, 'models', None):
                            for k in schema.models.keys():
                                if k.lower() == first_explicit:
                                    canonical = k
                                    break
                        target_table = canonical or first_explicit
                        if hasattr(self.wren_ai_integration, '_synthesize_sql_for_table'):
                            synthesized = self.wren_ai_integration._synthesize_sql_for_table(target_table, query)
                            if synthesized:
                                corrected = synthesized
            except Exception:
                pass

            # Schema-driven join injection using FK patterns (lightweight)
            try:
                if "JOIN DimDates" not in corrected and ("strftime('%Y'" in corrected or "DATEPART(YEAR" in corrected):
                    # If any date function is present but no DimDates join, inject it for DateID models
                    if any(t in corrected for t in ["FactAllIndiaDailySummary", "FactStateDailyEnergy"]):
                        corrected = corrected.replace(" FROM ", " FROM ")
                        # naive injection after first FROM <fact> fs
                        corrected = corrected.replace(" FROM FactAllIndiaDailySummary fs ", " FROM FactAllIndiaDailySummary fs JOIN DimDates d ON fs.DateID = d.DateID ")
                        corrected = corrected.replace(" FROM FactStateDailyEnergy fs ", " FROM FactStateDailyEnergy fs JOIN DimDates d ON fs.DateID = d.DateID ")
                if "JOIN DimRegions" not in corrected and ".RegionName" in corrected:
                    corrected = corrected.replace(" FROM FactAllIndiaDailySummary fs ", " FROM FactAllIndiaDailySummary fs JOIN DimRegions r ON fs.RegionID = r.RegionID ")
                if "JOIN DimStates" not in corrected and ".StateName" in corrected:
                    corrected = corrected.replace(" FROM FactStateDailyEnergy fs ", " FROM FactStateDailyEnergy fs JOIN DimStates s ON fs.StateID = s.StateID ")
            except Exception:
                pass

            # Probe execution to get score
            try:
                # Attempt execution
                execution = await self.sql_executor.execute_sql_async(corrected)
            except Exception as e:
                execution = type("Exec", (), {"success": False, "error": str(e), "data": []})

            # Scoring heuristics
            row_count = len(execution.data) if getattr(execution, 'data', None) else 0
            success_score = 1.0 if execution.success else 0.0
            coverage_score = 0.0
            ql = (query or "").lower()
            if any(k in ql for k in ["monthly", "per month", "by month"]):
                coverage_score += 0.4 if ("strftime('%Y-%m'" in corrected or "FORMAT(" in corrected or "DATEPART(YEAR" in corrected or "dt.Month" in corrected) else 0.0
            if any(k in ql for k in ["all regions", "regionwise", "by region"]):
                coverage_score += 0.3 if "r.RegionName" in corrected and "GROUP BY" in corrected else 0.0
            if any(k in ql for k in ["all states", "statewise", "by state"]):
                coverage_score += 0.3 if "s.StateName" in corrected and "GROUP BY" in corrected else 0.0
            size_score = min(row_count / 50.0, 0.3)  # cap

            # Prefer explicit table usage when present
            explicit_bonus = 0.0
            lower_sql = corrected.lower()
            if explicit_tables:
                if any((t in lower_sql) for t in explicit_tables):
                    explicit_bonus += 3.0
                else:
                    explicit_bonus -= 1.5
            # Strongly penalize daily models for sub-daily queries
            if is_sub_daily:
                if any(t in lower_sql for t in ["factallindiadailysummary", "factstatedailyenergy", "factdailygenerationbreakdown"]):
                    explicit_bonus -= 3.0
                if any(t in lower_sql for t in ["facttimeblockpowerdata", "facttimeblockgeneration"]):
                    explicit_bonus += 2.0

            # Prefer fact tables matched by domain intent words
            table_hint_bonus = 0.0
            if "exchange" in ql or "international" in ql or "bangladesh" in ql:
                if "factcountrydailyexchange" in corrected.lower() or "factinternationaltransmissionlinkflow" in corrected.lower():
                    table_hint_bonus += 0.4
                else:
                    table_hint_bonus -= 0.2
            if "generation" in ql or "renewable" in ql or "thermal" in ql:
                if "factdailygenerationbreakdown" in corrected.lower():
                    table_hint_bonus += 0.4
                else:
                    table_hint_bonus -= 0.2
            if any(k in ql for k in ["time block", "hourly", "15 min", "15-minute", "intraday"]):
                if "facttimeblockpowerdata" in corrected.lower() or "facttimeblockgeneration" in corrected.lower():
                    table_hint_bonus += 0.4
                else:
                    table_hint_bonus -= 0.2

            # Bias: Prefer wren_mdl strongly; boost explicit_builder for explicit table requests; boost domain_enforced
            mdl_bias = 0.4 if (cand.get("source") == "wren_mdl" and execution.success) else 0.0
            # Increase explicit_builder_bonus to make it more competitive when explicit tables are requested
            explicit_builder_bonus = 0.3 if (cand.get("source") == "explicit_builder" and execution.success and explicit_tables) else 0.1
            # Only a small bias for domain_enforced; correctness should come from table hints below
            domain_enforced_bonus = 0.1 if (cand.get("source") == "domain_enforced" and execution.success) else 0.0
            # Penalize disallowed EnergyMet usage on non-energy models for wren_mdl
            disallowed_penalty = 0.0
            if cand.get("source") == "wren_mdl":
                import re as _re
                mm = _re.search(r"FROM\s+(\w+)\s+([a-zA-Z]\\w*)", corrected, _re.IGNORECASE)
                ftable = (mm.group(1).lower() if mm else "")
                if 'energymet' in corrected.lower() and not ('allindiadailysummary' in ftable or 'statedailyenergy' in ftable):
                    disallowed_penalty -= 0.5

            # Flow-specific table preference
            wants_flow = ("flow" in ql) or ("link flow" in ql) or ("linkflow" in ql)
            if wants_flow:
                if ("factinternationaltransmissionlinkflow" in lower_sql) or ("facttransmissionlinkflow" in lower_sql):
                    table_hint_bonus += 1.0
                if ("factallindiadailysummary" in lower_sql) or ("factstatedailyenergy" in lower_sql):
                    table_hint_bonus -= 3.0  # hard penalty: wrong table for flow queries

            # Hard guardrails: drop candidates using daily tables for sub-daily or flow intents
            hard_penalty = 0.0
            if is_sub_daily and ("factallindiadailysummary" in lower_sql or "factstatedailyenergy" in lower_sql or "factdailygenerationbreakdown" in lower_sql):
                hard_penalty -= 5.0
            if wants_flow and ("factallindiadailysummary" in lower_sql or "factstatedailyenergy" in lower_sql):
                hard_penalty -= 5.0

            score = success_score + coverage_score + size_score + explicit_bonus + table_hint_bonus + mdl_bias + explicit_builder_bonus + domain_enforced_bonus + disallowed_penalty + hard_penalty

            scored.append({
                "source": cand.get("source"),
                "sql": corrected,
                "execution": execution,
                "score": score,
                "explicit_match": bool(explicit_tables and any(t in (corrected or "").lower() for t in explicit_tables)),
            })

        # If explicit tables were requested and any candidate matched them successfully, prefer those
        if explicit_tables:
            explicit_success = [c for c in scored if c.get("explicit_match") and getattr(c.get("execution"), 'success', False)]
            if explicit_success:
                # STRONGLY prefer explicit_builder when explicit tables are requested
                explicit_success.sort(key=lambda c: (0 if c.get("source") == "explicit_builder" else (1 if c.get("source") == "wren_mdl" else (2 if c.get("source") == "assembler" else 3)), -c.get("score", 0)))
                logger.info(f"🎯 Explicit table requested - returning {len(explicit_success)} explicit matches, preferring explicit_builder")
                return explicit_success
        # If sub-daily, restrict to time-block tables when available
        if is_sub_daily:
            timeblock_only = [c for c in scored if any(t in (c.get("sql") or "").lower() for t in ["facttimeblockpowerdata", "facttimeblockgeneration"])]
            timeblock_success = [c for c in timeblock_only if getattr(c.get("execution"), 'success', False)]
            if timeblock_success:
                # Return sorted by score
                return sorted(timeblock_success, key=lambda c: -c.get("score", 0))
        # Sort; but for flow/time-block intents, disqualify daily-table candidates entirely
        def is_disqualified(c):
            sql_l = (c.get("sql") or "").lower()
            if any(k in ql for k in ["time block", "hourly", "15 min", "15-minute", "intraday", "flow", "link flow", "linkflow", "energy exchanged"]):
                if any(t in sql_l for t in ["factallindiadailysummary", "factstatedailyenergy", "factdailygenerationbreakdown"]):
                    return True
            return False
        filtered = [c for c in scored if not is_disqualified(c)]

        # If nothing remains and the intent is flow/time-block, attempt to synthesize a correct-table candidate
        if not filtered and any(k in ql for k in ["time block", "hourly", "15 min", "15-minute", "intraday", "flow", "link flow", "linkflow", "energy exchanged"]):
            try:
                target = None
                if any(k in ql for k in ["time block", "hourly", "15 min", "15-minute", "intraday"]):
                    target = "FactTimeBlockGeneration" if any(k in ql for k in ["generation", "by source", "source"]) else "FactTimeBlockPowerData"
                elif any(k in ql for k in ["flow", "link flow", "linkflow", "energy exchanged"]):
                    intl = ("international" in ql) or any(c in ql for c in ["bangladesh","nepal","bhutan","myanmar","sri lanka","sri-lanka","pakistan","china"]) or ("energy exchanged" in ql)
                    target = "FactInternationalTransmissionLinkFlow" if intl else "FactTransmissionLinkFlow"
                if target:
                    synth = None
                    if hasattr(self.wren_ai_integration, '_synthesize_sql_for_table'):
                        synth = self.wren_ai_integration._synthesize_sql_for_table(target, query)
                    # If synthesizer failed, fall back to minimal COUNT(*) to ensure executability
                    if not synth:
                        synth = f"SELECT COUNT(*) AS Value FROM {target}"
                    exec_res = await self.sql_executor.execute_sql_async(synth)
                    filtered = [{"source": "domain_enforced", "sql": synth, "execution": exec_res, "score": 1.0, "explicit_match": False}]
            except Exception:
                pass

        return sorted(filtered if filtered else scored, key=lambda c: -c.get("score", 0))

    def _build_explicit_table_sql(self, table_name: str, query: str) -> Optional[str]:
        """Build a safe SELECT for explicitly mentioned tables using MDL hints."""
        try:
            import re
            mdl = getattr(self, 'wren_ai_integration', None)
            schema = getattr(mdl, 'mdl_schema', None)
            node = schema.models.get(table_name) if (schema and schema.models) else None
            hints = (node.definition or {}).get('hints', {}) if node else {}
            pvals = hints.get('preferred_value_columns') or []
            ptime = hints.get('preferred_time_column')
            has_date_fk = hints.get('has_date_fk', False)
            value_col = pvals[0] if pvals else None
            # Fallback: ask Wren MDL synthesizer to build correct SQL when hints are incomplete
            if not value_col and mdl and hasattr(mdl, '_synthesize_sql_for_table'):
                try:
                    synth = mdl._synthesize_sql_for_table(table_name, query)
                    if synth:
                        return synth
                except Exception:
                    pass
            if not value_col:
                return None

            ql = (query or '').lower()
            is_monthly = any(k in ql for k in ["monthly", "per month", "by month"])
            year_m = re.search(r"\b(19|20)\d{2}\b", ql)
            year = year_m.group(0) if year_m else None

            if is_monthly:
                if has_date_fk:
                    return (
                        f"SELECT FORMAT(d.ActualDate, 'yyyy-MM') AS Month, ROUND(SUM(fs.{value_col}), 2) AS Value "
                        f"FROM {table_name} fs JOIN DimDates d ON fs.DateID = d.DateID "
                        + (f"WHERE DATEPART(YEAR, d.ActualDate) = {year} " if year else "")
                        + "GROUP BY FORMAT(d.ActualDate, 'yyyy-MM') ORDER BY FORMAT(d.ActualDate, 'yyyy-MM')"
                    )
                elif ptime:
                    return (
                        f"SELECT FORMAT(fs.{ptime}, 'yyyy-MM') AS Month, ROUND(SUM(fs.{value_col}), 2) AS Value "
                        f"FROM {table_name} fs "
                        + (f"WHERE DATEPART(YEAR, fs.{ptime}) = {year} " if year else "")
                        + "GROUP BY FORMAT(fs.{ptime}, 'yyyy-MM') ORDER BY FORMAT(fs.{ptime}, 'yyyy-MM')"
                    )
            else:
                if has_date_fk:
                    return (
                        f"SELECT ROUND(SUM(fs.{value_col}), 2) AS Value FROM {table_name} fs JOIN DimDates d ON fs.DateID = d.DateID "
                        + (f"WHERE DATEPART(YEAR, d.ActualDate) = {year}" if year else "")
                    )
                elif ptime:
                    return (
                        f"SELECT ROUND(SUM(fs.{value_col}), 2) AS Value FROM {table_name} fs "
                        + (f"WHERE DATEPART(YEAR, fs.{ptime}) = {year}" if year else "")
                    )
            return None
        except Exception:
            return None

    def _build_keyword_summary(self, query: str, semantic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize extracted/available/missing keywords for observability."""
        ql = (query or "").lower()
        available = []
        missing = []

        # Basic checks for time granularity and entities
        if any(k in ql for k in ["monthly", "per month", "by month", "month-wise"]):
            available.append("monthly")
        else:
            missing.append("monthly")
        if any(k in ql for k in ["2020", "2021", "2022", "2023", "2024", "2025"]):
            available.append("year")
        else:
            missing.append("year")
        if any(k in ql for k in ["region", "regions", "northern", "southern", "eastern", "western", "north eastern",
                                  "north", "south", "east", "west", "north east", "india", "the country"]):
            available.append("region_or_india")
        else:
            missing.append("region_or_india")
        if any(k in ql for k in ["state", "states"]):
            available.append("state")
        else:
            missing.append("state")

        # Include MDL context surface
        mdl_models = [m.get("name") for m in semantic_context.get("mdl_context", {}).get("relevant_models", [])]
        return {"available": sorted(set(available)), "missing": sorted(set(missing)), "mdl_models": mdl_models}
            
    async def process_feedback(self, session_id: str, feedback_data: Dict) -> Dict[str, Any]:
        """
        Process feedback for continuous learning and improvement
        DISABLED for cloud deployment - feedback storage not available
        """
        logger.warning("Feedback processing disabled for cloud deployment")
        return {
            "success": False,
            "error": "Feedback processing disabled for cloud deployment"
        }
            
    async def _store_query_feedback(self, query: str, result: Dict, semantic_context: Dict,
                                  session_id: str, user_id: str, start_time: datetime):
        """Store feedback for a processed query - DISABLED for cloud deployment"""
        logger.debug("Query feedback storage disabled for cloud deployment")
        pass
            
    async def _store_error_feedback(self, query: str, error_message: str,
                                  session_id: str, user_id: str, start_time: datetime):
        """Store feedback for a failed query - DISABLED for cloud deployment"""
        logger.debug("Error feedback storage disabled for cloud deployment")
        pass
            
    async def _analyze_feedback_for_learning(self, feedback_record: Any) -> Dict[str, Any]:
        """Analyze feedback for learning insights - DISABLED for cloud deployment"""
        logger.debug("Feedback analysis disabled for cloud deployment")
        return {}
            
    async def get_feedback_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive feedback analytics - DISABLED for cloud deployment"""
        logger.warning("Feedback analytics disabled for cloud deployment")
        return {
            "success": False,
            "error": "Feedback analytics disabled for cloud deployment"
        }
            
    async def get_similar_feedback(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Get similar feedback for a query - DISABLED for cloud deployment"""
        logger.warning("Similar feedback retrieval disabled for cloud deployment")
        return {
            "success": False,
            "error": "Similar feedback retrieval disabled for cloud deployment"
        }
            
    async def _process_semantic_first(self, query: str, semantic_context: Dict) -> Dict[str, Any]:
        """Process query using semantic-first approach with Wren AI integration"""
        try:
            # Use Wren AI integration for MDL-aware SQL generation
            wren_result = await self.wren_ai_integration.generate_mdl_aware_sql(query, semantic_context)
            
            if wren_result.get("sql") and wren_result.get("confidence", 0.0) > 0.7:
                # Execute the generated SQL
                repaired_sql = self._validate_and_correct_sql(self._auto_repair_sql(wren_result["sql"]), query)
                execution_result = await self.sql_executor.execute_sql_async(repaired_sql)

                # Attempt visualization recommendation via agentic visualization logic
                plot = None
                try:
                    if execution_result.success and execution_result.data:
                        viz_agent = VisualizationAgent()
                        viz_dict = await viz_agent._generate_visualization(execution_result.data, query)
                        if viz_dict:
                            plot = {
                                "chartType": viz_dict.get("chart_type", "bar"),
                                "options": viz_dict.get("config", {})
                            }
                except Exception:
                    plot = None

                # Check if execution was successful
                if execution_result.success:
                    return {
                        "success": True,
                        "sql": repaired_sql,
                        "data": execution_result.data,
                        "plot": plot,
                        "confidence": wren_result["confidence"],
                        "processing_method": "wren_ai_mdl",
                        "mdl_context": wren_result.get("mdl_context", {}),
                        "business_entities": wren_result.get("business_entities", [])
                    }
                else:
                    # SQL execution failed, return error
                    return {
                        "success": False,
                        "error": f"SQL execution failed: {execution_result.error}",
                        "sql": repaired_sql,
                        "data": [],
                        "confidence": wren_result["confidence"],
                        "processing_method": "wren_ai_mdl_failed",
                        "mdl_context": wren_result.get("mdl_context", {}),
                        "business_entities": wren_result.get("business_entities", [])
                    }
            else:
                # Fallback to semantic engine
                return await self._process_with_semantic_engine(query, semantic_context)
                
        except Exception as e:
            logger.error(f"Semantic-first processing failed: {e}")
            return await self._process_with_semantic_engine(query, semantic_context)
            
    async def _process_hybrid(self, query: str, semantic_context: Dict) -> Dict[str, Any]:
        """Process query using hybrid approach combining multiple methods"""
        try:
            # Try Wren AI first
            wren_result = await self.wren_ai_integration.generate_mdl_aware_sql(query, semantic_context)
            
            # Try semantic engine
            semantic_result = await self._process_with_semantic_engine(query, semantic_context)
            
            # Choose the best result based on confidence
            if wren_result.get("confidence", 0.0) > semantic_result.get("confidence", 0.0):
                return {
                    "success": True,
                    "sql": wren_result["sql"],
                    "data": semantic_result.get("data", []),
                    "confidence": wren_result["confidence"],
                    "processing_method": "hybrid_wren_ai",
                    "mdl_context": wren_result.get("mdl_context", {}),
                    "business_entities": wren_result.get("business_entities", [])
                }
            else:
                return semantic_result
                
        except Exception as e:
            logger.error(f"Hybrid processing failed: {e}")
            return await self._process_traditional(query, semantic_context)
            
    async def _process_with_semantic_engine(self, query: str, semantic_context: Dict) -> Dict[str, Any]:
        """Process query using semantic engine"""
        try:
            # Use semantic engine for context-aware SQL generation
            semantic_result = await self.semantic_engine.generate_contextual_sql(
                query, semantic_context, {}
            )
            
            if semantic_result.get("sql"):
                # Execute the generated SQL
                repaired_sql = self._validate_and_correct_sql(self._auto_repair_sql(semantic_result["sql"]), query)
                execution_result = await self.sql_executor.execute_sql_async(repaired_sql)

                # Attempt visualization recommendation via agentic visualization logic
                plot = None
                try:
                    if execution_result.success and execution_result.data:
                        viz_agent = VisualizationAgent()
                        viz_dict = await viz_agent._generate_visualization(execution_result.data, query)
                        if viz_dict:
                            plot = {
                                "chartType": viz_dict.get("chart_type", "bar"),
                                "options": viz_dict.get("config", {})
                            }
                except Exception:
                    plot = None

                # Check if execution was successful
                if execution_result.success:
                    return {
                        "success": True,
                        "sql": repaired_sql,
                        "data": execution_result.data,
                        "plot": plot,
                        "confidence": semantic_result.get("confidence", 0.0),
                        "processing_method": "semantic_engine",
                        "semantic_context": semantic_context
                    }
                else:
                    # SQL execution failed, return error
                    return {
                        "success": False,
                        "error": f"SQL execution failed: {execution_result.error}",
                        "sql": repaired_sql,
                        "data": [],
                        "confidence": semantic_result.get("confidence", 0.0),
                        "processing_method": "semantic_engine_failed",
                        "semantic_context": semantic_context
                    }
            else:
                return await self._process_traditional(query, semantic_context)
                
        except Exception as e:
            logger.error(f"Semantic engine processing failed: {e}")
            return await self._process_traditional(query, semantic_context)
            
    async def _process_traditional(self, query: str, semantic_context: Dict) -> Dict[str, Any]:
        """Process query using traditional approach"""
        try:
            # Use proper intent analysis instead of hardcoded values
            
            intent_analyzer = IntentAnalyzer()
            analysis = await intent_analyzer.analyze_intent(query, self.llm_provider)
            
            context = ContextInfo(
                query_analysis=analysis,
                schema_info=SchemaInfo(tables={}),
                dimension_values={},
                user_mappings=[],
                memory_context=None,
                schema_linker=None,
                llm_provider=self.llm_provider,
            )
            sql_result = self.sql_assembler.generate_sql(query, analysis, context)
            
            if sql_result and hasattr(sql_result, 'sql') and sql_result.sql:
                # Execute the generated SQL
                repaired_sql = self._validate_and_correct_sql(self._auto_repair_sql(sql_result.sql), query)
                execution_result = await self.sql_executor.execute_sql_async(repaired_sql)
                
                # Check if execution was successful
                if execution_result.success:
                    return {
                        "success": True,
                        "sql": repaired_sql,
                        "data": execution_result.data,
                        "confidence": 0.6,  # Lower confidence for traditional approach
                        "processing_method": "traditional",
                        "semantic_context": semantic_context
                    }
                else:
                    # SQL execution failed, return error
                    return {
                        "success": False,
                        "error": f"SQL execution failed: {execution_result.error}",
                        "sql": repaired_sql,
                        "data": [],
                        "confidence": 0.6,
                        "processing_method": "traditional_failed",
                        "semantic_context": semantic_context
                    }
            else:
                return {
                    "success": False,
                    "error": "SQL generation failed",
                    "processing_method": "traditional"
                }
                
        except Exception as e:
            logger.error(f"Traditional processing failed: {e}")
            return {
                "success": False,
                "error": str(e) or "traditional_processing_error",
                "processing_method": "traditional"
            }
            
    async def _process_agentic(self, query: str, semantic_context: Dict) -> Dict[str, Any]:
        """Process query using agentic workflow"""
        try:
            # Import agentic service
            from backend.services.agentic_rag_service import AgenticRAGService
            
            agentic_service = AgenticRAGService(self.db_path)
            result = await agentic_service.process_query_agentic(query)
            
            # Add semantic context to result
            result["semantic_context"] = semantic_context
            result["processing_method"] = "agentic_workflow"
            
            return result
            
        except Exception as e:
            logger.error(f"Agentic processing failed: {e}")
            return await self._process_hybrid(query, semantic_context)
            
    def _determine_processing_mode(self, confidence: float, requested_mode: str) -> ProcessingMode:
        """Determine the actual processing mode based on confidence and request"""
        if requested_mode == "adaptive":
            if confidence >= 0.8:
                return ProcessingMode.SEMANTIC_FIRST
            elif confidence >= 0.6:
                return ProcessingMode.HYBRID
            elif confidence >= 0.4:
                return ProcessingMode.AGENTIC_WORKFLOW
            else:
                return ProcessingMode.TRADITIONAL
        else:
            return ProcessingMode(requested_mode)
            
    def _update_statistics(self, start_time: datetime, semantic_context: Dict, result: Dict):
        """Update service statistics"""
        self.stats["total_requests"] += 1
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        if self.stats["total_requests"] > 0:
            self.stats["average_response_time"] = (
                (self.stats["average_response_time"] * (self.stats["total_requests"] - 1) + response_time) 
                / self.stats["total_requests"]
            )
        
        # Update semantic enhancement rate
        if result.get("processing_method") in ["wren_ai_mdl", "semantic_engine", "hybrid_wren_ai"]:
            if self.stats["total_requests"] > 0:
                self.stats["semantic_enhancement_rate"] = (
                    (self.stats["semantic_enhancement_rate"] * (self.stats["total_requests"] - 1) + 1) 
                    / self.stats["total_requests"]
                )
            
        # Update MDL usage rate
        if semantic_context.get("mdl_context", {}).get("relevant_models"):
            if self.stats["total_requests"] > 0:
                self.stats["mdl_usage_rate"] = (
                    (self.stats["mdl_usage_rate"] * (self.stats["total_requests"] - 1) + 1) 
                    / self.stats["total_requests"]
                )
            
        # Update vector search success rate
        if semantic_context.get("search_results"):
            if self.stats["total_requests"] > 0:
                self.stats["vector_search_success_rate"] = (
                    (self.stats["vector_search_success_rate"] * (self.stats["total_requests"] - 1) + 1) 
                    / self.stats["total_requests"]
                )
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "success": True,
            "statistics": self.stats,
            "system_status": {
                "semantic_engine": "operational" if self._initialized else "initializing",
                "wren_ai_integration": "operational" if self._initialized else "initializing",
                "vector_database": "operational" if self._initialized else "initializing",
                "mdl_support": "enabled" if self._initialized else "disabled",
                "accuracy_target": "85-90%"
            }
        }