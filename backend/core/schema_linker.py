"""
Schema Linking Module
Provides intelligent linking between natural language tokens and database schema elements
"""

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.core.entity_loader import get_entity_loader

logger = logging.getLogger(__name__)


@dataclass
class SchemaLink:
    """Represents a link between NL token and schema element"""

    nl_token: str
    schema_element: str
    element_type: str  # 'table', 'column', 'value'
    confidence: float
    context: str


@dataclass
class SchemaContext:
    """Schema context for query processing"""

    schema_info: Dict[str, List[str]]
    links: List[SchemaLink]
    relevant_tables: List[str]
    relevant_columns: List[str]
    confidence: float


class SchemaLinker:
    """
    Links natural language queries to database schema elements.
    """

    def __init__(
        self, schema_info: Dict[str, List[str]], db_path: str, llm_provider=None
    ):
        self.schema_info = schema_info
        self.db_path = db_path
        self.llm_provider = llm_provider
        self.entity_loader = get_entity_loader()

        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=1000
        )

        # Load business rules
        self.business_rules = self._load_business_rules()

        # Build schema embeddings
        self.schema_elements: List[str] = []
        self.element_types: List[str] = []
        self.schema_vectors = None
        self._build_schema_embeddings()

    def _load_business_rules(self) -> Dict[str, Any]:
        """Load business rules from configuration"""
        try:
            with open("config/business_rules.yaml", "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load business rules: {e}")
            return {}

    def _get_energy_column_for_table(
        self, table_name: str, query_type: str = "energy"
    ) -> str:
        """Get the correct column name based on table and query type"""
        if not self.business_rules:
            return ""  # No fallback - return empty string

        schema_mapping = self.business_rules.get("schema_mapping", {})

        # Try new categorized structure first
        if query_type == "power":
            power_columns = schema_mapping.get("power_columns", {})
            if table_name in power_columns:
                # Return first available column for the table
                return list(power_columns[table_name].keys())[0]
        elif query_type == "energy":
            energy_columns = schema_mapping.get("energy_columns", {})
            if table_name in energy_columns:
                # For energy columns, prefer more specific ones
                available_columns = list(energy_columns[table_name].keys())
                # Prefer EnergyShortage over EnergyMet if available
                if "EnergyShortage" in available_columns:
                    return "EnergyShortage"
                elif "EnergyMet" in available_columns:
                    return "EnergyMet"
                else:
                    # Return first available column
                    return available_columns[0] if available_columns else ""
        elif query_type == "ratio":
            ratio_columns = schema_mapping.get("ratio_columns", {})
            if table_name in ratio_columns:
                return list(ratio_columns[table_name].keys())[0]
        elif query_type == "percentage":
            percentage_columns = schema_mapping.get("percentage_columns", {})
            if table_name in percentage_columns:
                return list(percentage_columns[table_name].keys())[0]
        elif query_type == "time":
            time_columns = schema_mapping.get("time_columns", {})
            if table_name in time_columns:
                return list(time_columns[table_name].keys())[0]
        elif query_type == "outage":
            outage_columns = schema_mapping.get("outage_columns", {})
            if table_name in outage_columns:
                return list(outage_columns[table_name].keys())[0]
        elif query_type == "transmission":
            transmission_columns = schema_mapping.get("transmission_columns", {})
            if table_name in transmission_columns:
                return list(transmission_columns[table_name].keys())[0]
        elif query_type == "exchange":
            exchange_columns = schema_mapping.get("exchange_columns", {})
            if table_name in exchange_columns:
                return list(exchange_columns[table_name].keys())[0]

        # Fallback to legacy mappings
        if query_type == "evening_demand":
            evening_demand_columns = schema_mapping.get("evening_demand_columns", {})
            return evening_demand_columns.get(table_name, "")
        elif query_type == "demand":
            power_demand_columns = schema_mapping.get("power_demand_columns", {})
            return power_demand_columns.get(table_name, "")
        elif query_type == "drawal_schedule":
            drawal_schedule_columns = schema_mapping.get("drawal_schedule_columns", {})
            return drawal_schedule_columns.get(table_name, "")
        elif query_type == "actual_drawal":
            actual_drawal_columns = schema_mapping.get("actual_drawal_columns", {})
            return actual_drawal_columns.get(table_name, "")
        elif query_type == "shortage":
            shortage_columns = schema_mapping.get("shortage_columns", {})
            return shortage_columns.get(table_name, "")
        elif query_type == "transmission":
            transmission_flow_columns = schema_mapping.get(
                "transmission_flow_columns", {}
            )
            return transmission_flow_columns.get(table_name, "")
        elif query_type == "exchange":
            exchange_columns = schema_mapping.get("exchange_columns", {})
            return exchange_columns.get(table_name, "")

        # No fallback - return empty string to indicate no match found
        return ""

    def _get_column_data_type(self, table_name: str, column_name: str) -> str:
        """Get the data type of a specific column"""
        if not self.business_rules:
            return ""  # No fallback - return empty string

        schema_mapping = self.business_rules.get("schema_mapping", {})

        # Check each category
        for category, tables in schema_mapping.items():
            if category.endswith("_columns") and table_name in tables:
                if column_name in tables[table_name]:
                    return tables[table_name][column_name]

        return ""  # No fallback - return empty string

    def _get_appropriate_aggregation_functions(self, data_type: str) -> List[str]:
        """Get appropriate aggregation functions based on data type"""
        if not data_type:
            return []  # Return empty list if no data type found

        aggregation_rules = {
            "power": [
                "MAX",
                "MIN",
                "AVG",
            ],  # Power values - can't sum across time blocks
            "energy": ["SUM", "MAX", "MIN", "AVG"],  # Energy values - can sum
            "ratio": ["AVG", "MAX", "MIN"],  # Ratios - average makes sense
            "percentage": [
                "AVG",
                "MAX",
                "MIN",
            ],  # Percentages (like frequency durations) - average makes sense
            "time": ["SUM", "MAX", "MIN", "AVG"],  # Time durations - can sum
            "outage": ["SUM", "MAX", "MIN", "AVG"],  # Outage values - can sum
            "transmission": [
                "MAX",
                "MIN",
                "AVG",
            ],  # Transmission values (including loading percentages)
            "exchange": ["MAX", "MIN", "AVG"],  # Exchange values
        }

        return aggregation_rules.get(data_type, [])

    def _load_dimension_values(self) -> Dict[str, List[str]]:
        """Load dimension values from database tables"""
        dimension_values = {}

        if not self.db_path:
            logger.warning("No database path provided, using static dimension values")
            return self._get_static_dimension_values()

        logger.info(f"Loading dimension values from database: {self.db_path}")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Define dimension tables and their name columns
            dimension_tables = {
                "DimCountries": "CountryName",
                "DimRegions": "RegionName",
                "DimStates": "StateName",
                "DimTransmissionLines": "LineIdentifier",
                "DimExchangeMechanisms": "MechanismName",
                "DimGenerationSources": "SourceName",
                "DimUnits": "UnitName",
            }

            for table_name, name_column in dimension_tables.items():
                try:
                    # Check if table exists
                    cursor.execute(
                        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
                    )
                    if cursor.fetchone():
                        # Get distinct values from the name column
                        cursor.execute(
                            f"SELECT DISTINCT {name_column} FROM {table_name} WHERE {name_column} IS NOT NULL"
                        )
                        values = [row[0].lower() for row in cursor.fetchall() if row[0]]
                        dimension_values[table_name] = values
                        logger.info(
                            f"Loaded {len(values)} values from {table_name}: {values[:5]}..."
                        )  # Show first 5 values
                    else:
                        logger.warning(f"Table {table_name} not found in database")
                        # Use static values for missing table
                        dimension_values[table_name] = (
                            self._get_static_values_for_table(table_name)
                        )
                except Exception as e:
                    logger.error(f"Error loading values from {table_name}: {e}")
                    # Fallback to static values for this table
                    dimension_values[table_name] = self._get_static_values_for_table(
                        table_name
                    )

            conn.close()
            logger.info(
                f"Successfully loaded dimension values for {len(dimension_values)} tables"
            )
            return dimension_values

        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            logger.info("Falling back to static dimension values")
            return self._get_static_dimension_values()

    def _get_static_dimension_values(self) -> Dict[str, List[str]]:
        """Get static dimension values as fallback using centralized entity loader"""
        return {
            "DimCountries": ["india"],
            "DimRegions": self.entity_loader.get_indian_regions(),
            "DimStates": self.entity_loader.get_indian_states(),
            "DimTransmissionLines": ["transmission line 1", "transmission line 2"],
            "DimExchangeMechanisms": ["exchange mechanism 1", "exchange mechanism 2"],
            "DimGenerationSources": [
                "thermal",
                "hydro",
                "nuclear",
                "renewable",
                "solar",
                "wind",
                "biomass",
            ],
            "DimUnits": ["mw", "mwh", "gwh", "tw", "twh"],
        }

    def _get_static_values_for_table(self, table_name: str) -> List[str]:
        """Get static dimension values for a specific table"""
        dimension_mapping = {
            "DimStates": "StateName",
            "DimRegions": "RegionName",
            "DimGenerationSources": "SourceName",
            "DimTransmissionLines": "LineIdentifier",  # Fixed: actual column name
            "DimExchangeMechanisms": "MechanismName",  # Fixed: actual column name
            "DimCountries": "CountryName",
            "DimTimeBlocks": "TimeBlockName",
            "DimUnits": "UnitName",
        }

        column_name = dimension_mapping.get(table_name)
        if not column_name:
            return []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT DISTINCT {column_name} FROM {table_name} ORDER BY {column_name}"
            )
            values = [row[0] for row in cursor.fetchall() if row[0]]
            conn.close()
            return values
        except Exception as e:
            logger.error(f"Error loading values from {table_name}: {e}")
            return []

    def _build_schema_embeddings(self):
        """Build embeddings for schema elements"""
        self.schema_elements = []
        self.element_types = []

        # Add tables
        for table_name in self.schema_info.keys():
            self.schema_elements.append(table_name.lower())
            self.element_types.append("table")

        # Add columns
        for table_name, columns in self.schema_info.items():
            for column in columns:
                self.schema_elements.append(f"{table_name}.{column}".lower())
                self.element_types.append("column")

        # Load dimension values from database
        dimension_values = self._load_dimension_values()

        # Add dimension values
        for table_name, values in dimension_values.items():
            for value in values:
                # Add the raw value
                self.schema_elements.append(value.lower())
                self.element_types.append("value")

                # Add table.value format for better disambiguation
                self.schema_elements.append(f"{table_name}.{value}".lower())
                self.element_types.append("value")

        # Build TF-IDF vectors
        if self.schema_elements:
            self.schema_vectors = self.vectorizer.fit_transform(self.schema_elements)
            logger.info(
                f"Built embeddings for {len(self.schema_elements)} schema elements"
            )
        else:
            logger.warning("No schema elements found for embedding")

    def link_query_to_schema(self, query: str, query_analysis: Any) -> SchemaContext:
        """
        Link natural language query to schema elements
        """
        links = []
        relevant_tables = set()
        relevant_columns = set()

        # Extract potential entities from query
        nl_tokens = self._extract_nl_tokens(query)

        # Link each token to schema elements
        for token in nl_tokens:
            token_links = self._find_schema_links(token, query_analysis)
            links.extend(token_links)

            # Track relevant elements
            for link in token_links:
                if link.element_type == "table":
                    relevant_tables.add(link.schema_element)
                elif link.element_type == "column":
                    relevant_columns.add(link.schema_element)

        # Add direct entity links from query analysis
        if hasattr(query_analysis, "entities") and query_analysis.entities:
            for entity in query_analysis.entities:
                entity_links = self._find_schema_links(entity.lower(), query_analysis)
                links.extend(entity_links)

                # Track relevant elements
                for link in entity_links:
                    if link.element_type == "table":
                        relevant_tables.add(link.schema_element)
                    elif link.element_type == "column":
                        relevant_columns.add(link.schema_element)

        # Add context-based links
        context_links = self._add_context_links(query, query_analysis)
        links.extend(context_links)

        # Calculate overall confidence
        confidence = self._calculate_context_confidence(links)

        return SchemaContext(
            schema_info=self.schema_info,
            links=links,
            relevant_tables=list(relevant_tables),
            relevant_columns=list(relevant_columns),
            confidence=confidence,
        )

    def _extract_nl_tokens(self, query: str) -> List[str]:
        """Extract meaningful tokens from natural language query"""
        # Remove common words and extract meaningful tokens
        stop_words = {
            "what",
            "is",
            "the",
            "of",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "and",
            "or",
            "but",
            "from",
            "by",
            "as",
            "are",
            "were",
            "was",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
        }

        # Extract words and phrases
        words = re.findall(r"\b[a-zA-Z]+\b", query.lower())
        meaningful_tokens = [
            word for word in words if word not in stop_words and len(word) > 2
        ]

        # Add multi-word phrases
        phrases = re.findall(r"\b[a-zA-Z]+\s+[a-zA-Z]+\b", query.lower())
        meaningful_tokens.extend(phrases)

        return list(set(meaningful_tokens))

    def _find_schema_links(self, token: str, query_analysis: Any) -> List[SchemaLink]:
        """Find schema links for a given token"""
        links = []

        # Direct string matching
        for i, element in enumerate(self.schema_elements):
            if token in element or element in token:
                confidence = 0.8 if token == element else 0.6
                links.append(
                    SchemaLink(
                        nl_token=token,
                        schema_element=element,
                        element_type=self.element_types[i],
                        confidence=confidence,
                        context="direct_match",
                    )
                )

        # Semantic similarity using TF-IDF
        if hasattr(self, "schema_vectors") and self.schema_elements:
            try:
                token_vector = self.vectorizer.transform([token])
                similarities = cosine_similarity(
                    token_vector, self.schema_vectors
                ).flatten()

                # Find top matches
                top_indices = np.argsort(similarities)[-3:][::-1]
                for idx in top_indices:
                    if similarities[idx] > 0.3:  # Threshold for semantic similarity
                        links.append(
                            SchemaLink(
                                nl_token=token,
                                schema_element=self.schema_elements[idx],
                                element_type=self.element_types[idx],
                                confidence=similarities[idx],
                                context="semantic_similarity",
                            )
                        )
            except Exception as e:
                logger.warning(f"Semantic similarity failed: {e}")

        return links

    def _add_context_links(self, query: str, query_analysis: Any) -> List[SchemaLink]:
        """Add context-based links based on query analysis"""
        links = []

        # Link based on query type
        if hasattr(query_analysis, "query_type"):
            if query_analysis.query_type.value == "state":
                # Link to state-related tables and columns
                for table in ["DimStates", "FactStateDailyEnergy"]:
                    if table in self.schema_info:
                        links.append(
                            SchemaLink(
                                nl_token="state",
                                schema_element=table,
                                element_type="table",
                                confidence=0.9,
                                context="query_type",
                            )
                        )

            elif query_analysis.query_type.value == "region":
                # Link to region-related tables and columns
                for table in ["DimRegions", "FactAllIndiaDailySummary"]:
                    if table in self.schema_info:
                        links.append(
                            SchemaLink(
                                nl_token="region",
                                schema_element=table,
                                element_type="table",
                                confidence=0.9,
                                context="query_type",
                            )
                        )

        # Link based on intent
        if hasattr(query_analysis, "intent"):
            if query_analysis.intent.value == "trend_analysis":
                # Link to time-related columns
                for table, columns in self.schema_info.items():
                    for column in columns:
                        if any(
                            time_word in column.lower()
                            for time_word in [
                                "date",
                                "time",
                                "year",
                                "month",
                                "quarter",
                            ]
                        ):
                            links.append(
                                SchemaLink(
                                    nl_token="trend",
                                    schema_element=f"{table}.{column}",
                                    element_type="column",
                                    confidence=0.7,
                                    context="intent",
                                )
                            )

        return links

    def _calculate_context_confidence(self, links: List[SchemaLink]) -> float:
        """Calculate overall confidence for schema context"""
        if not links:
            return 0.0

        # Weight by confidence and context type
        total_weight: float = 0.0
        weighted_sum: float = 0.0

        for link in links:
            weight = 1.0
            if link.context == "direct_match":
                weight = 2.0
            elif link.context == "query_type":
                weight = 1.5
            elif link.context == "semantic_similarity":
                weight = 1.0

            weighted_sum += link.confidence * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_enhanced_prompt_context(self, schema_context: SchemaContext) -> str:
        """Generate enhanced prompt context with schema links"""
        context_parts = []

        # Add schema information
        context_parts.append("Available Tables and Columns:")
        for table, columns in self.schema_info.items():
            context_parts.append(f"  {table}: {', '.join(columns)}")

        # Add schema links
        if schema_context.links:
            context_parts.append("\nSchema Links (NL → Schema):")
            for link in sorted(
                schema_context.links, key=lambda x: x.confidence, reverse=True
            )[:10]:
                context_parts.append(
                    f"  '{link.nl_token}' → {link.schema_element} ({link.element_type}, confidence: {link.confidence:.2f})"
                )

        # Add relevant elements
        if schema_context.relevant_tables:
            context_parts.append(
                f"\nRelevant Tables: {', '.join(schema_context.relevant_tables)}"
            )

        if schema_context.relevant_columns:
            context_parts.append(
                f"Relevant Columns: {', '.join(schema_context.relevant_columns)}"
            )

        return "\n".join(context_parts)

    def find_similar_columns(
        self, user_query: str, table_name: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Find the most similar columns to user query words using TF-IDF similarity.

        Args:
            user_query: The user's natural language query
            table_name: The table to search columns in
            top_k: Number of top similar columns to return

        Returns:
            List of tuples (column_name, similarity_score) sorted by similarity
        """
        try:
            # Get all columns for the table
            all_columns = self._get_all_columns_for_table(table_name)
            if not all_columns:
                logger.warning(f"No columns found for table {table_name}")
                return []

            # Preprocess user query - extract meaningful words
            query_words = self._extract_query_words(user_query)
            if not query_words:
                logger.warning("No meaningful words extracted from query")
                return []

            # Create documents for TF-IDF: user query + all column names
            documents = [user_query.lower()] + [col.lower() for col in all_columns]

            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),  # Use unigrams and bigrams
                min_df=1,
                max_df=0.95,
            )

            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(documents)

            # Calculate similarity between query and each column
            query_vector = tfidf_matrix[0:1]  # First document is the query
            column_vectors = tfidf_matrix[1:]  # Rest are columns

            similarities = cosine_similarity(query_vector, column_vectors).flatten()

            # Create list of (column_name, similarity_score) tuples
            column_similarities = list(zip(all_columns, similarities))

            # Sort by similarity score (descending) and return top_k
            column_similarities.sort(key=lambda x: x[1], reverse=True)

            logger.info(
                f"Top {top_k} similar columns for query '{user_query}' in table {table_name}:"
            )
            for col, score in column_similarities[:top_k]:
                logger.info(f"  {col}: {score:.3f}")

            return column_similarities[:top_k]

        except Exception as e:
            logger.error(f"Error finding similar columns: {e}")
            return []

    def _get_all_columns_for_table(self, table_name: str) -> List[str]:
        """Get all column names for a given table from schema_info."""
        if not self.schema_info:
            return []

        return self.schema_info.get(table_name, [])

    def _extract_query_words(self, user_query: str) -> List[str]:
        """Extract meaningful words from user query for column matching."""
        # Convert to lowercase and split
        words = user_query.lower().split()

        # Remove common stop words and short words
        stop_words = {
            "what",
            "is",
            "the",
            "of",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "and",
            "or",
            "but",
            "a",
            "an",
            "as",
            "from",
            "into",
            "during",
            "including",
            "until",
            "against",
            "among",
            "throughout",
            "despite",
            "towards",
            "upon",
            "concerning",
            "to",
            "in",
            "for",
            "of",
            "with",
            "by",
            "total",
            "sum",
            "maximum",
            "minimum",
            "average",
            "max",
            "min",
            "avg",
        }

        meaningful_words = []
        for word in words:
            # Remove punctuation and clean
            clean_word = "".join(c for c in word if c.isalnum())
            if (
                len(clean_word) > 2
                and clean_word not in stop_words
                and not clean_word.isdigit()
            ):
                meaningful_words.append(clean_word)

        return meaningful_words

    def get_best_table_match(
        self, user_query: str, query_analysis: Any = None
    ) -> Tuple[str, float]:
        """
        Automatically identify the best table for a user query.

        Args:
            user_query: The user's natural language query
            query_analysis: Optional query analysis for context

        Returns:
            Tuple of (best_table_name, confidence_score)
        """
        query_lower = user_query.lower()

        # Define table patterns and their keywords
        table_patterns = {
            "FactAllIndiaDailySummary": {
                "keywords": [
                    "region",
                    "regional",
                    "all india",
                    "national",
                    "countrywide",
                ],
                "metrics": ["energy", "demand", "shortage", "met", "scada"],
                "priority": 1.0,
            },
            "FactStateDailyEnergy": {
                "keywords": [
                    "state",
                    "states",
                    "maharashtra",
                    "karnataka",
                    "tamil nadu",
                    "gujarat",
                ],
                "metrics": ["energy", "demand", "shortage", "met"],
                "priority": 0.9,
            },
            "FactDailyGenerationBreakdown": {
                "keywords": [
                    "generation",
                    "power plant",
                    "thermal",
                    "hydro",
                    "renewable",
                    "coal",
                    "gas",
                    "nuclear",
                    "solar",
                    "wind",
                    "biomass",
                ],
                "metrics": [
                    "generation",
                    "output",
                    "capacity",
                    "coal",
                    "gas",
                    "nuclear",
                    "solar",
                    "wind",
                    "biomass",
                ],
                "priority": 0.8,
            },
            "FactExchangeData": {
                "keywords": ["exchange", "import", "export", "trading", "bilateral"],
                "metrics": ["import", "export", "energy", "trading"],
                "priority": 0.7,
            },
            "FactTransmissionData": {
                "keywords": ["transmission", "loading", "corridor", "line"],
                "metrics": ["loading", "capacity", "transmission"],
                "priority": 0.6,
            },
        }

        best_table = None
        best_score = 0.0

        for table_name, pattern in table_patterns.items():
            score = 0.0

            # Check keyword matches
            keywords: List[str] = pattern["keywords"]
            keyword_matches = sum(1 for keyword in keywords if keyword in query_lower)
            if keyword_matches > 0:
                score += keyword_matches * 0.3

            # Check metric matches
            metrics: List[str] = pattern["metrics"]
            metric_matches = sum(1 for metric in metrics if metric in query_lower)
            if metric_matches > 0:
                score += metric_matches * 0.4

            # Apply priority multiplier
            priority: float = float(pattern["priority"])
            score *= priority

            # Special handling for generation queries
            if any(
                word in query_lower
                for word in [
                    "generation",
                    "coal",
                    "gas",
                    "nuclear",
                    "solar",
                    "wind",
                    "biomass",
                    "thermal",
                    "hydro",
                ]
            ):
                if table_name == "FactDailyGenerationBreakdown":
                    score += 0.8  # High priority for generation queries
                elif table_name == "FactAllIndiaDailySummary":
                    score -= 0.5  # Lower priority for non-generation queries

            # Special handling for state vs region queries
            if "state" in query_lower or any(
                state in query_lower
                for state in ["maharashtra", "karnataka", "tamil nadu"]
            ):
                if table_name == "FactStateDailyEnergy":
                    score += 0.5
                elif table_name == "FactAllIndiaDailySummary":
                    score -= 0.3

            # Special handling for region queries
            if "region" in query_lower or "all regions" in query_lower:
                if table_name == "FactAllIndiaDailySummary":
                    score += 0.5
                elif table_name == "FactStateDailyEnergy":
                    score -= 0.3

            if score > best_score:
                best_score = score
                best_table = table_name

        # Fallback to default table if no good match
        if not best_table or best_score < 0.1:
            best_table = "FactAllIndiaDailySummary"
            best_score = 0.1

        logger.info(
            f"Auto-selected table '{best_table}' with confidence {best_score:.2f} for query: {user_query[:50]}..."
        )
        return best_table, best_score

    def get_table_config(self, table_name: str, user_query: str) -> Dict[str, str]:
        """
        Get automatic table configuration including join clause and column names.

        Args:
            table_name: The selected table name
            user_query: The user query for context

        Returns:
            Dictionary with table configuration
        """
        query_lower = user_query.lower()

        # Define table configurations
        table_configs = {
            "FactAllIndiaDailySummary": {
                "join_clause": "JOIN DimRegions d ON f.RegionID = d.RegionID",
                "name_column": "RegionName",
                "dimension_table": "DimRegions",
                "dimension_alias": "d",
                "id_column": "RegionID",
            },
            "FactStateDailyEnergy": {
                "join_clause": "JOIN DimStates d ON f.StateID = d.StateID",
                "name_column": "StateName",
                "dimension_table": "DimStates",
                "dimension_alias": "d",
                "id_column": "StateID",
            },
            "FactDailyGenerationBreakdown": {
                "join_clause": "JOIN DimRegions d ON f.RegionID = d.RegionID JOIN DimGenerationSources gs ON f.GenerationSourceID = gs.GenerationSourceID",
                "name_column": "RegionName",
                "dimension_table": "DimRegions",
                "dimension_alias": "d",
                "id_column": "RegionID",
            },
            "FactExchangeData": {
                "join_clause": "JOIN DimCountries d ON f.CountryID = d.CountryID",
                "name_column": "CountryName",
                "dimension_table": "DimCountries",
                "dimension_alias": "d",
                "id_column": "CountryID",
            },
            "FactTransmissionData": {
                "join_clause": "JOIN DimTransmissionLines d ON f.LineID = d.LineID",
                "name_column": "LineName",
                "dimension_table": "DimTransmissionLines",
                "dimension_alias": "d",
                "id_column": "LineID",
            },
            "FactGenerationData": {
                "join_clause": "JOIN DimPowerPlants d ON f.PlantID = d.PlantID",
                "name_column": "PlantName",
                "dimension_table": "DimPowerPlants",
                "dimension_alias": "d",
                "id_column": "PlantID",
            },
        }

        config = table_configs.get(
            table_name,
            {
                "join_clause": "",
                "name_column": "Name",
                "dimension_table": "",
                "dimension_alias": "d",
                "id_column": "ID",
            },
        )

        # Auto-detect if query is asking for "all" entities
        if "all" in query_lower and any(
            word in query_lower for word in ["regions", "states", "countries"]
        ):
            config["where_clause"] = ""  # No filter - include all
        else:
            # Let the assembler handle specific entity filtering
            config["where_clause"] = ""

        logger.info(
            f"Auto-configured table '{table_name}': join='{config['join_clause']}', name_col='{config['name_column']}'"
        )
        return config

    def get_best_column_match(
        self,
        user_query: str,
        table_name: str,
        query_type: Optional[str] = None,
        llm_provider=None,
    ) -> str:
        """
        Get the best column match for a user query, with LLM fallback for edge cases.

        Args:
            user_query: The user's natural language query
            table_name: The table to search columns in
            query_type: Optional query type for business rules fallback
            llm_provider: Optional LLM provider for edge-case column linking (Hook #2)

        Returns:
            Best matching column name or empty string if no match found
        """
        # Get all columns for the specific table only
        all_columns = self._get_all_columns_for_table(table_name)
        if not all_columns:
            logger.warning(f"No columns found for table {table_name}")
            return ""

        query_lower = user_query.lower()

        # Check for exact matches or partial matches within the table's columns
        exact_matches = []
        partial_matches = []

        for column in all_columns:
            column_lower = column.lower()
            # Check if column name is contained in query or vice versa
            if column_lower in query_lower or any(
                word in column_lower for word in query_lower.split() if len(word) > 3
            ):

                # Check if this is an exact match (column name is a complete word in query)
                if column_lower in query_lower:
                    exact_matches.append(column)
                else:
                    partial_matches.append(column)

        # Special handling for "energy met" queries
        if "energy met" in query_lower:
            # Prioritize EnergyMet for "energy met" queries
            if "EnergyMet" in all_columns:
                logger.info(f"Found 'energy met' in query, prioritizing EnergyMet")
                return "EnergyMet"

        # Special handling for energy demand queries
        if "energy demand" in query_lower or (
            "energy" in query_lower and "demand" in query_lower
        ):
            # Prioritize EnergyMet for energy demand queries
            if "EnergyMet" in all_columns:
                logger.info(f"Found 'energy demand' in query, prioritizing EnergyMet")
                return "EnergyMet"

        # Special handling for shortage queries
        if "shortage" in query_lower:
            # Prioritize EnergyShortage over Shortage for shortage queries
            if "EnergyShortage" in all_columns:
                logger.info(f"Found 'shortage' in query, prioritizing EnergyShortage")
                return "EnergyShortage"
            elif "Shortage" in all_columns:
                logger.info(f"Found 'shortage' in query, using Shortage")
                return "Shortage"

        # Special handling for "Maximum Import" queries
        if "maximum" in query_lower and "import" in query_lower:
            # Prioritize MaxImport for maximum import queries
            if "MaxImport" in all_columns:
                logger.info(f"Found 'maximum import' in query, prioritizing MaxImport")
                return "MaxImport"

        # Special handling for "Maximum Export" queries
        if "maximum" in query_lower and "export" in query_lower:
            # Prioritize MaxExport for maximum export queries
            if "MaxExport" in all_columns:
                logger.info(f"Found 'maximum export' in query, prioritizing MaxExport")
                return "MaxExport"

        # Enhanced special handling for import/export queries
        if "import" in query_lower:
            if "maximum" in query_lower or "max" in query_lower:
                if "MaxImport" in all_columns:
                    logger.info(
                        f"Found 'maximum import' pattern in query, prioritizing MaxImport"
                    )
                    return "MaxImport"
            elif "energy" in query_lower or "daily" in query_lower:
                if "ImportEnergy" in all_columns:
                    logger.info(
                        f"Found 'import energy' pattern in query, prioritizing ImportEnergy"
                    )
                    return "ImportEnergy"
            elif "net" in query_lower:
                if "NetImportEnergy" in all_columns:
                    logger.info(
                        f"Found 'net import' pattern in query, prioritizing NetImportEnergy"
                    )
                    return "NetImportEnergy"

        if "export" in query_lower:
            if "maximum" in query_lower or "max" in query_lower:
                if "MaxExport" in all_columns:
                    logger.info(
                        f"Found 'maximum export' pattern in query, prioritizing MaxExport"
                    )
                    return "MaxExport"
            elif "energy" in query_lower or "daily" in query_lower:
                if "ExportEnergy" in all_columns:
                    logger.info(
                        f"Found 'export energy' pattern in query, prioritizing ExportEnergy"
                    )
                    return "ExportEnergy"

        # Special handling for transmission loading queries
        if "loading" in query_lower:
            if "maximum" in query_lower or "max" in query_lower:
                if "MaxLoading" in all_columns:
                    logger.info(
                        f"Found 'maximum loading' pattern in query, prioritizing MaxLoading"
                    )
                    return "MaxLoading"
            elif "minimum" in query_lower or "min" in query_lower:
                if "MinLoading" in all_columns:
                    logger.info(
                        f"Found 'minimum loading' pattern in query, prioritizing MinLoading"
                    )
                    return "MinLoading"
            elif "average" in query_lower or "avg" in query_lower:
                if "AvgLoading" in all_columns:
                    logger.info(
                        f"Found 'average loading' pattern in query, prioritizing AvgLoading"
                    )
                    return "AvgLoading"

        # Prioritize matches: prefer general columns over specific ones
        def prioritize_columns(columns):
            """Prioritize general columns over specific ones"""
            if not columns:
                return []

            # Define priority order for common patterns
            priority_patterns = [
                # Energy demand columns (highest priority for energy demand queries)
                lambda col: col.lower() in ["energymet", "demandmet"],
                # Maximum import/export columns (high priority for "maximum import/export" queries)
                lambda col: col.lower() in ["maximport", "maxexport"],
                # Maximum demand columns (high priority for "maximum demand" queries)
                lambda col: col.lower() in ["maxdemandscada", "maximumdemand"],
                # Energy shortage (high priority for shortage queries)
                lambda col: col.lower() == "energyshortage",
                # Then specific time-based columns (lower priority)
                lambda col: any(
                    word in col.lower()
                    for word in ["evening", "peak", "solar", "nonsolar"]
                ),
                # Then other specific columns (lowest priority)
                lambda col: any(
                    word in col.lower()
                    for word in ["schedule", "actual", "overunder", "net", "shortage"]
                ),
                # Default: no specific priority
                lambda col: True,
            ]

            prioritized = []
            remaining_columns = columns.copy()

            for pattern in priority_patterns:
                matching = [col for col in remaining_columns if pattern(col)]
                prioritized.extend(matching)
                # Remove matched columns from the remaining list
                remaining_columns = [
                    col for col in remaining_columns if col not in matching
                ]

            return prioritized

        # Apply prioritization to exact and partial matches
        if exact_matches:
            prioritized_exact = prioritize_columns(exact_matches)
            logger.info(f"Found exact matches (prioritized): {prioritized_exact}")
            return prioritized_exact[0]
        elif partial_matches:
            prioritized_partial = prioritize_columns(partial_matches)
            logger.info(f"Found partial matches (prioritized): {prioritized_partial}")
            return prioritized_partial[0]

        # Try similarity-based matching with lower threshold
        similar_columns = self.find_similar_columns(user_query, table_name, top_k=3)

        if (
            similar_columns and similar_columns[0][1] > 0.01
        ):  # Lowered threshold for testing
            best_column = similar_columns[0][0]
            logger.info(
                f"Using similarity-based column match: {best_column} (score: {similar_columns[0][1]:.3f})"
            )
            return best_column

        # Hook #2: LLM Edge-Case Column Linking
        if llm_provider and hasattr(llm_provider, "generate"):
            logger.info(f"No column match found, trying LLM edge-case column linking")
            try:
                # Create a simple prompt for column selection
                prompt = f"""Given the user query "{user_query}" and table "{table_name}" with columns {all_columns}, which column best matches the query? Return only the column name or "UNKNOWN"."""

                # For now, we'll use a simple approach without async to avoid complexity
                # In a production system, this would be properly async
                logger.info(
                    f"LLM Hook #2: Attempting column linking for query: {user_query}"
                )
                # Note: This is a simplified version - in production, you'd want proper async handling
                return ""  # Skip for now to avoid async issues
            except Exception as e:
                logger.warning(f"LLM Hook #2 (Column Linking) failed: {e}")
                return ""

        # No fallback - return empty string to indicate no match found
        logger.warning(
            f"No column match found for query '{user_query}' in table {table_name}"
        )
        return ""
