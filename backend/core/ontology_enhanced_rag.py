#!/usr/bin/env python3
"""
Ontology-Enhanced RAG System

This module implements an advanced RAG system that integrates the energy ontology
with the existing retrieval system to provide context-aware, business rule-enforced
retrieval for improved SQL generation accuracy.

Key Features:
1. Ontology-aware retrieval using energy domain concepts
2. Business rule enforcement during retrieval
3. Context-aware search using domain relationships
4. Domain-specific example curation
5. Rule validation for retrieved examples
6. Performance optimization for ontology queries
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer

from .energy_ontology import EnergyOntology, EnergyConcept, EnergyDomain, EnergyConceptType
from .few_shot_examples import QueryExample, FewShotExampleRepository, FewShotExampleRetriever

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Retrieval strategies for ontology-enhanced RAG"""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    ONTOLOGY_CONCEPT = "ontology_concept"
    BUSINESS_RULE = "business_rule"
    DOMAIN_RELATIONSHIP = "domain_relationship"
    HYBRID = "hybrid"


@dataclass
class OntologyRetrievalContext:
    """Context for ontology-enhanced retrieval"""
    query: str
    detected_concepts: List[EnergyConcept] = field(default_factory=list)
    detected_domains: List[EnergyDomain] = field(default_factory=list)
    business_rules: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    confidence: float = 0.0
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID


@dataclass
class OntologyEnhancedExample:
    """Enhanced example with ontology information"""
    example: QueryExample
    ontology_concepts: List[EnergyConcept] = field(default_factory=list)
    business_rules: List[str] = field(default_factory=list)
    domain_relevance: float = 0.0
    rule_compliance: float = 1.0
    context_similarity: float = 0.0


class OntologyEnhancedRAG:
    """
    Ontology-enhanced RAG system that integrates energy domain ontology
    with the existing retrieval system for improved accuracy and context awareness.
    """

    def __init__(self, db_path: str, ontology: EnergyOntology = None):
        self.db_path = db_path
        self.ontology = ontology or EnergyOntology(db_path)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize existing RAG components
        self.few_shot_repository = FewShotExampleRepository(db_path)
        self.few_shot_retriever = FewShotExampleRetriever(self.few_shot_repository)
        
        # Initialize ontology-enhanced components
        self._initialize_ontology_enhanced_database()
        
        logger.info("Ontology-enhanced RAG system initialized")

    def _initialize_ontology_enhanced_database(self):
        """Initialize database tables for ontology-enhanced features"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create ontology-enhanced examples table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ontology_enhanced_examples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        example_id INTEGER NOT NULL,
                        ontology_concepts TEXT,
                        business_rules TEXT,
                        domain_relevance REAL DEFAULT 0.0,
                        rule_compliance REAL DEFAULT 1.0,
                        context_similarity REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (example_id) REFERENCES query_examples(id)
                    )
                """)
                
                # Create concept mappings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS concept_mappings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        concept_id TEXT NOT NULL,
                        concept_name TEXT NOT NULL,
                        domain TEXT NOT NULL,
                        synonyms TEXT,
                        database_mapping TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create business rule validations table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS business_rule_validations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        example_id INTEGER NOT NULL,
                        rule_id TEXT NOT NULL,
                        rule_name TEXT NOT NULL,
                        validation_result BOOLEAN,
                        validation_score REAL DEFAULT 0.0,
                        validation_details TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (example_id) REFERENCES query_examples(id)
                    )
                """)
                
                # Create indexes for efficient retrieval
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ontology_examples_concepts 
                    ON ontology_enhanced_examples(ontology_concepts)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ontology_examples_domain_relevance 
                    ON ontology_enhanced_examples(domain_relevance)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_concept_mappings_domain 
                    ON concept_mappings(domain)
                """)
                
                conn.commit()
                logger.info("Ontology-enhanced database tables initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize ontology-enhanced database: {e}")

    def analyze_query_ontology(self, query: str) -> OntologyRetrievalContext:
        """
        Analyze query using energy ontology to extract concepts, domains, and business rules.
        
        Args:
            query: Natural language query
            
        Returns:
            OntologyRetrievalContext with detected ontology information
        """
        context = OntologyRetrievalContext(query=query)
        
        try:
            # Detect energy concepts in the query
            detected_concepts = self.ontology.suggest_concepts(query)
            context.detected_concepts = detected_concepts
            
            # Extract domains from detected concepts
            domains = set()
            for concept in detected_concepts:
                domains.add(concept.domain)
            context.detected_domains = list(domains)
            
            # Extract business rules relevant to the query
            business_rules = []
            for domain in context.detected_domains:
                domain_rules = self.ontology.get_business_rules_for_domain(domain)
                for rule in domain_rules:
                    if self._is_rule_relevant_to_query(rule, query):
                        business_rules.append(rule.id)
            context.business_rules = business_rules
            
            # Extract relationships from detected concepts
            relationships = []
            for concept in detected_concepts:
                related_concepts = self.ontology.get_related_concepts(concept.name)
                for related in related_concepts:
                    relationship_key = f"{concept.id}_to_{related.id}"
                    relationships.append(relationship_key)
            context.relationships = relationships
            
            # Calculate confidence based on concept detection
            context.confidence = min(1.0, len(detected_concepts) / 5.0)
            
            # Determine retrieval strategy
            context.strategy = self._determine_retrieval_strategy(context)
            
            logger.info(f"Ontology analysis completed: {len(detected_concepts)} concepts, "
                       f"{len(context.detected_domains)} domains, {len(business_rules)} rules")
            
        except Exception as e:
            logger.error(f"Failed to analyze query ontology: {e}")
            context.confidence = 0.0
            context.strategy = RetrievalStrategy.SEMANTIC_SIMILARITY
            
        return context

    def _is_rule_relevant_to_query(self, rule: Any, query: str) -> bool:
        """Check if a business rule is relevant to the query"""
        query_lower = query.lower()
        rule_name_lower = rule.name.lower()
        rule_desc_lower = rule.description.lower()
        
        # Check if rule name or description contains keywords from the query
        query_words = set(query_lower.split())
        rule_words = set(rule_name_lower.split() + rule_desc_lower.split())
        
        # Calculate relevance based on word overlap
        overlap = len(query_words.intersection(rule_words))
        return overlap > 0

    def _determine_retrieval_strategy(self, context: OntologyRetrievalContext) -> RetrievalStrategy:
        """Determine the best retrieval strategy based on context"""
        if len(context.detected_concepts) > 3 and len(context.business_rules) > 2:
            return RetrievalStrategy.HYBRID
        elif len(context.detected_concepts) > 2:
            return RetrievalStrategy.ONTOLOGY_CONCEPT
        elif len(context.business_rules) > 1:
            return RetrievalStrategy.BUSINESS_RULE
        elif len(context.relationships) > 1:
            return RetrievalStrategy.DOMAIN_RELATIONSHIP
        else:
            return RetrievalStrategy.SEMANTIC_SIMILARITY

    def retrieve_ontology_enhanced_examples(
        self, 
        query: str, 
        max_examples: int = 5,
        min_similarity: float = 0.3
    ) -> List[OntologyEnhancedExample]:
        """
        Retrieve examples enhanced with ontology information.
        
        Args:
            query: Natural language query
            max_examples: Maximum number of examples to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of ontology-enhanced examples
        """
        try:
            # Analyze query using ontology
            ontology_context = self.analyze_query_ontology(query)
            
            # Retrieve examples based on strategy
            if ontology_context.strategy == RetrievalStrategy.HYBRID:
                examples = self._retrieve_hybrid_examples(query, ontology_context, max_examples, min_similarity)
            elif ontology_context.strategy == RetrievalStrategy.ONTOLOGY_CONCEPT:
                examples = self._retrieve_concept_based_examples(query, ontology_context, max_examples, min_similarity)
            elif ontology_context.strategy == RetrievalStrategy.BUSINESS_RULE:
                examples = self._retrieve_rule_based_examples(query, ontology_context, max_examples, min_similarity)
            elif ontology_context.strategy == RetrievalStrategy.DOMAIN_RELATIONSHIP:
                examples = self._retrieve_relationship_based_examples(query, ontology_context, max_examples, min_similarity)
            else:
                examples = self._retrieve_semantic_examples(query, ontology_context, max_examples, min_similarity)
            
            # Enhance examples with ontology information
            enhanced_examples = []
            for example in examples:
                enhanced_example = self._enhance_example_with_ontology(example, ontology_context)
                if enhanced_example.domain_relevance >= min_similarity:
                    enhanced_examples.append(enhanced_example)
            
            # Sort by relevance and return top examples
            enhanced_examples.sort(key=lambda x: x.domain_relevance, reverse=True)
            return enhanced_examples[:max_examples]
            
        except Exception as e:
            logger.error(f"Failed to retrieve ontology-enhanced examples: {e}")
            return []

    def _retrieve_hybrid_examples(
        self, 
        query: str, 
        context: OntologyRetrievalContext, 
        max_examples: int, 
        min_similarity: float
    ) -> List[QueryExample]:
        """Retrieve examples using hybrid strategy combining multiple approaches"""
        examples = []
        
        # Get semantic examples
        semantic_examples = self._retrieve_semantic_examples(query, context, max_examples // 2, min_similarity)
        examples.extend(semantic_examples)
        
        # Get concept-based examples
        concept_examples = self._retrieve_concept_based_examples(query, context, max_examples // 2, min_similarity)
        examples.extend(concept_examples)
        
        # Remove duplicates
        seen_ids = set()
        unique_examples = []
        for example in examples:
            if example.id not in seen_ids:
                seen_ids.add(example.id)
                unique_examples.append(example)
        
        return unique_examples[:max_examples]

    def _retrieve_concept_based_examples(
        self, 
        query: str, 
        context: OntologyRetrievalContext, 
        max_examples: int, 
        min_similarity: float
    ) -> List[QueryExample]:
        """Retrieve examples based on ontology concepts"""
        examples = []
        
        try:
            # Get examples for each detected concept
            for concept in context.detected_concepts:
                concept_examples = self._get_examples_for_concept(concept, max_examples // len(context.detected_concepts))
                examples.extend(concept_examples)
            
            # Filter by similarity
            filtered_examples = []
            for example in examples:
                similarity = self._calculate_concept_similarity(example, context.detected_concepts)
                if similarity >= min_similarity:
                    filtered_examples.append(example)
            
            return filtered_examples[:max_examples]
            
        except Exception as e:
            logger.error(f"Failed to retrieve concept-based examples: {e}")
            return []

    def _retrieve_rule_based_examples(
        self, 
        query: str, 
        context: OntologyRetrievalContext, 
        max_examples: int, 
        min_similarity: float
    ) -> List[QueryExample]:
        """Retrieve examples based on business rules"""
        examples = []
        
        try:
            # Get examples for each business rule
            for rule_id in context.business_rules:
                rule_examples = self._get_examples_for_business_rule(rule_id, max_examples // len(context.business_rules))
                examples.extend(rule_examples)
            
            # Filter by rule compliance
            filtered_examples = []
            for example in examples:
                compliance = self._calculate_rule_compliance(example, context.business_rules)
                if compliance >= min_similarity:
                    filtered_examples.append(example)
            
            return filtered_examples[:max_examples]
            
        except Exception as e:
            logger.error(f"Failed to retrieve rule-based examples: {e}")
            return []

    def _retrieve_relationship_based_examples(
        self, 
        query: str, 
        context: OntologyRetrievalContext, 
        max_examples: int, 
        min_similarity: float
    ) -> List[QueryExample]:
        """Retrieve examples based on domain relationships"""
        examples = []
        
        try:
            # Get examples for each relationship
            for relationship in context.relationships:
                relationship_examples = self._get_examples_for_relationship(relationship, max_examples // len(context.relationships))
                examples.extend(relationship_examples)
            
            # Filter by relationship relevance
            filtered_examples = []
            for example in examples:
                relevance = self._calculate_relationship_relevance(example, context.relationships)
                if relevance >= min_similarity:
                    filtered_examples.append(example)
            
            return filtered_examples[:max_examples]
            
        except Exception as e:
            logger.error(f"Failed to retrieve relationship-based examples: {e}")
            return []

    def _retrieve_semantic_examples(
        self, 
        query: str, 
        context: OntologyRetrievalContext, 
        max_examples: int, 
        min_similarity: float
    ) -> List[QueryExample]:
        """Retrieve examples using semantic similarity (fallback)"""
        try:
            similar_examples = self.few_shot_repository.search_similar_examples(
                query, 
                limit=max_examples * 2,
                min_confidence=0.7,
                only_successful=True
            )
            
            # Filter by similarity threshold
            filtered_examples = [
                example for example, similarity in similar_examples 
                if similarity >= min_similarity
            ]
            
            return filtered_examples[:max_examples]
            
        except Exception as e:
            logger.error(f"Failed to retrieve semantic examples: {e}")
            return []

    def _enhance_example_with_ontology(
        self, 
        example: QueryExample, 
        context: OntologyRetrievalContext
    ) -> OntologyEnhancedExample:
        """Enhance an example with ontology information"""
        enhanced_example = OntologyEnhancedExample(example=example)
        
        try:
            # Extract ontology concepts from the example
            example_concepts = self.ontology.suggest_concepts(example.natural_query)
            enhanced_example.ontology_concepts = example_concepts
            
            # Calculate domain relevance
            domain_relevance = self._calculate_domain_relevance(example, context.detected_domains)
            enhanced_example.domain_relevance = domain_relevance
            
            # Calculate rule compliance
            rule_compliance = self._calculate_rule_compliance(example, context.business_rules)
            enhanced_example.rule_compliance = rule_compliance
            
            # Calculate context similarity
            context_similarity = self._calculate_context_similarity(example, context)
            enhanced_example.context_similarity = context_similarity
            
            # Extract business rules
            business_rules = []
            for concept in example_concepts:
                for rule_id in concept.validation_rules:
                    if rule_id in self.ontology.business_rules:
                        business_rules.append(rule_id)
            enhanced_example.business_rules = business_rules
            
        except Exception as e:
            logger.error(f"Failed to enhance example with ontology: {e}")
            enhanced_example.domain_relevance = 0.0
            enhanced_example.rule_compliance = 0.0
            enhanced_example.context_similarity = 0.0
        
        return enhanced_example

    def _calculate_domain_relevance(self, example: QueryExample, detected_domains: List[EnergyDomain]) -> float:
        """Calculate domain relevance score for an example"""
        if not detected_domains:
            return 0.0
        
        try:
            # Extract concepts from example
            example_concepts = self.ontology.suggest_concepts(example.natural_query)
            
            # Count concepts that match detected domains
            matching_concepts = 0
            for concept in example_concepts:
                if concept.domain in detected_domains:
                    matching_concepts += 1
            
            # Calculate relevance score
            if example_concepts:
                relevance = matching_concepts / len(example_concepts)
            else:
                relevance = 0.0
            
            return min(1.0, relevance)
            
        except Exception as e:
            logger.error(f"Failed to calculate domain relevance: {e}")
            return 0.0

    def _calculate_rule_compliance(self, example: QueryExample, business_rules: List[str]) -> float:
        """Calculate business rule compliance score for an example"""
        if not business_rules:
            return 1.0
        
        try:
            # Check compliance with each business rule
            compliant_rules = 0
            for rule_id in business_rules:
                if rule_id in self.ontology.business_rules:
                    rule = self.ontology.business_rules[rule_id]
                    if self._check_rule_compliance(example, rule):
                        compliant_rules += 1
            
            # Calculate compliance score
            compliance = compliant_rules / len(business_rules) if business_rules else 1.0
            return min(1.0, compliance)
            
        except Exception as e:
            logger.error(f"Failed to calculate rule compliance: {e}")
            return 0.0

    def _calculate_context_similarity(self, example: QueryExample, context: OntologyRetrievalContext) -> float:
        """Calculate context similarity between example and query context"""
        try:
            # Extract concepts from example
            example_concepts = self.ontology.suggest_concepts(example.natural_query)
            
            # Calculate concept overlap
            example_concept_ids = {concept.id for concept in example_concepts}
            context_concept_ids = {concept.id for concept in context.detected_concepts}
            
            if not context_concept_ids:
                return 0.0
            
            overlap = len(example_concept_ids.intersection(context_concept_ids))
            similarity = overlap / len(context_concept_ids)
            
            return min(1.0, similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate context similarity: {e}")
            return 0.0

    def _check_rule_compliance(self, example: QueryExample, rule: Any) -> bool:
        """Check if an example complies with a business rule"""
        try:
            # Simple rule compliance check based on rule type
            if rule.rule_type == "validation":
                # Check if the example SQL contains the required elements
                sql_lower = example.generated_sql.lower()
                rule_lower = rule.expression.lower()
                
                # Basic compliance checks
                if "positive" in rule_lower and "negative" in sql_lower:
                    return False
                if "required" in rule_lower and "null" in sql_lower:
                    return False
                
                return True
            else:
                return True
                
        except Exception as e:
            logger.error(f"Failed to check rule compliance: {e}")
            return True

    def _get_examples_for_concept(self, concept: EnergyConcept, limit: int) -> List[QueryExample]:
        """Get examples for a specific concept"""
        try:
            # Search for examples that contain the concept name or synonyms
            search_terms = [concept.name] + concept.synonyms
            
            examples = []
            for term in search_terms:
                term_examples = self.few_shot_repository.search_similar_examples(
                    term, 
                    limit=limit // len(search_terms),
                    min_confidence=0.6,
                    only_successful=True
                )
                examples.extend([example for example, _ in term_examples])
            
            return examples[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get examples for concept {concept.name}: {e}")
            return []

    def _get_examples_for_business_rule(self, rule_id: str, limit: int) -> List[QueryExample]:
        """Get examples for a specific business rule"""
        try:
            # This would require storing rule-example mappings
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Failed to get examples for business rule {rule_id}: {e}")
            return []

    def _get_examples_for_relationship(self, relationship: str, limit: int) -> List[QueryExample]:
        """Get examples for a specific relationship"""
        try:
            # This would require storing relationship-example mappings
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Failed to get examples for relationship {relationship}: {e}")
            return []

    def _calculate_concept_similarity(self, example: QueryExample, concepts: List[EnergyConcept]) -> float:
        """Calculate similarity between example and concepts"""
        try:
            example_concepts = self.ontology.suggest_concepts(example.natural_query)
            
            if not concepts or not example_concepts:
                return 0.0
            
            # Calculate concept overlap
            example_concept_ids = {concept.id for concept in example_concepts}
            query_concept_ids = {concept.id for concept in concepts}
            
            overlap = len(example_concept_ids.intersection(query_concept_ids))
            similarity = overlap / len(query_concept_ids)
            
            return min(1.0, similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate concept similarity: {e}")
            return 0.0

    def _calculate_relationship_relevance(self, example: QueryExample, relationships: List[str]) -> float:
        """Calculate relationship relevance score for an example"""
        if not relationships:
            return 0.0
        
        try:
            # Extract concepts from example
            example_concepts = self.ontology.suggest_concepts(example.natural_query)
            
            # Count relationships that are relevant to the example
            relevant_relationships = 0
            for relationship in relationships:
                # Check if the relationship involves concepts from the example
                for concept in example_concepts:
                    if concept.id in relationship:
                        relevant_relationships += 1
                        break
            
            # Calculate relevance score
            relevance = relevant_relationships / len(relationships) if relationships else 0.0
            return min(1.0, relevance)
            
        except Exception as e:
            logger.error(f"Failed to calculate relationship relevance: {e}")
            return 0.0

    def format_ontology_enhanced_examples(self, examples: List[OntologyEnhancedExample]) -> str:
        """Format ontology-enhanced examples for inclusion in prompts"""
        if not examples:
            return ""
        
        formatted_examples = []
        for i, enhanced_example in enumerate(examples, 1):
            example = enhanced_example.example
            
            # Format ontology information
            ontology_info = ""
            if enhanced_example.ontology_concepts:
                concept_names = [c.name for c in enhanced_example.ontology_concepts]
                ontology_info = f"Concepts: {', '.join(concept_names)}"
            
            if enhanced_example.business_rules:
                ontology_info += f" | Rules: {', '.join(enhanced_example.business_rules)}"
            
            formatted_example = f"""
Example {i} (Domain Relevance: {enhanced_example.domain_relevance:.2f}, Rule Compliance: {enhanced_example.rule_compliance:.2f}):
Query: {example.natural_query}
SQL: {example.generated_sql}
{ontology_info}
Confidence: {example.confidence:.2f}
Success: {'Yes' if example.success else 'No'}
"""
            formatted_examples.append(formatted_example)
        
        return "\n".join(formatted_examples)

    def validate_examples_against_business_rules(self, examples: List[OntologyEnhancedExample]) -> List[OntologyEnhancedExample]:
        """Validate examples against business rules and return compliant ones"""
        validated_examples = []
        
        for enhanced_example in examples:
            try:
                # Check rule compliance
                if enhanced_example.rule_compliance >= 0.8:  # High compliance threshold
                    validated_examples.append(enhanced_example)
                else:
                    logger.warning(f"Example {enhanced_example.example.id} failed rule compliance check: {enhanced_example.rule_compliance}")
                    
            except Exception as e:
                logger.error(f"Failed to validate example {enhanced_example.example.id}: {e}")
        
        return validated_examples

    def get_ontology_statistics(self) -> Dict[str, Any]:
        """Get statistics about the ontology-enhanced RAG system"""
        try:
            stats = {
                "total_concepts": len(self.ontology.concepts),
                "total_relationships": len(self.ontology.relationships),
                "total_business_rules": len(self.ontology.business_rules),
                "domains": [domain.value for domain in EnergyDomain],
                "retrieval_strategies": [strategy.value for strategy in RetrievalStrategy]
            }
            
            # Get database statistics
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count ontology-enhanced examples
                cursor.execute("SELECT COUNT(*) FROM ontology_enhanced_examples")
                stats["ontology_enhanced_examples"] = cursor.fetchone()[0]
                
                # Count concept mappings
                cursor.execute("SELECT COUNT(*) FROM concept_mappings")
                stats["concept_mappings"] = cursor.fetchone()[0]
                
                # Count business rule validations
                cursor.execute("SELECT COUNT(*) FROM business_rule_validations")
                stats["business_rule_validations"] = cursor.fetchone()[0]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get ontology statistics: {e}")
            return {}
