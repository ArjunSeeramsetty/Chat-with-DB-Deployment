#!/usr/bin/env python3
"""
Test script for Ontology-Enhanced RAG System
Validates the integration of energy ontology with the existing RAG system
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.core.ontology_enhanced_rag import (
    OntologyEnhancedRAG, OntologyRetrievalContext, OntologyEnhancedExample,
    RetrievalStrategy
)
from backend.core.energy_ontology import EnergyOntology, EnergyDomain
from backend.core.few_shot_examples import QueryExample

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OntologyEnhancedRAGTester:
    """Test suite for ontology-enhanced RAG system"""

    def __init__(self):
        self.db_path = "backend/energy_data.db"
        self.ontology = EnergyOntology(self.db_path)
        self.rag_system = OntologyEnhancedRAG(self.db_path, self.ontology)
        self.test_results = []

    async def run_all_tests(self):
        """Run all ontology-enhanced RAG tests"""
        logger.info("🚀 Starting Ontology-Enhanced RAG Tests")

        test_cases = [
            ("System Initialization", self._test_system_initialization),
            ("Query Ontology Analysis", self._test_query_ontology_analysis),
            ("Retrieval Strategy Determination", self._test_retrieval_strategy_determination),
            ("Concept-Based Retrieval", self._test_concept_based_retrieval),
            ("Rule-Based Retrieval", self._test_rule_based_retrieval),
            ("Relationship-Based Retrieval", self._test_relationship_based_retrieval),
            ("Hybrid Retrieval", self._test_hybrid_retrieval),
            ("Example Enhancement", self._test_example_enhancement),
            ("Business Rule Validation", self._test_business_rule_validation),
            ("Example Formatting", self._test_example_formatting),
            ("Statistics Generation", self._test_statistics_generation),
        ]

        for test_name, test_func in test_cases:
            try:
                logger.info(f"\n📋 Running test: {test_name}")
                result = await test_func()
                self.test_results.append((test_name, result))
                logger.info(f"✅ {test_name}: PASSED")
            except Exception as e:
                logger.error(f"❌ {test_name}: FAILED - {e}")
                self.test_results.append((test_name, {"status": "FAILED", "error": str(e)}))

        self._print_summary()

    async def _test_system_initialization(self):
        """Test system initialization"""
        # Check if RAG system was initialized correctly
        assert self.rag_system.db_path == self.db_path
        assert self.rag_system.ontology is not None
        assert self.rag_system.few_shot_repository is not None
        assert self.rag_system.few_shot_retriever is not None

        return {
            "status": "PASSED",
            "ontology_initialized": True,
            "few_shot_repository_initialized": True,
            "few_shot_retriever_initialized": True
        }

    async def _test_query_ontology_analysis(self):
        """Test query ontology analysis"""
        # Test with energy-related query
        query = "What is the average energy consumption by region in 2025?"
        context = self.rag_system.analyze_query_ontology(query)
        
        # Check if context was created
        assert context.query == query
        assert isinstance(context.detected_concepts, list)
        assert isinstance(context.detected_domains, list)
        assert isinstance(context.business_rules, list)
        assert isinstance(context.relationships, list)
        assert 0.0 <= context.confidence <= 1.0
        assert isinstance(context.strategy, RetrievalStrategy)
        
        # Check if concepts were detected
        assert len(context.detected_concepts) > 0, "No concepts detected for energy query"
        
        # Check if domains were detected
        assert len(context.detected_domains) > 0, "No domains detected for energy query"

        return {
            "status": "PASSED",
            "concepts_detected": len(context.detected_concepts),
            "domains_detected": len(context.detected_domains),
            "business_rules_detected": len(context.business_rules),
            "relationships_detected": len(context.relationships),
            "confidence": context.confidence,
            "strategy": context.strategy.value
        }

    async def _test_retrieval_strategy_determination(self):
        """Test retrieval strategy determination"""
        # Test different query types to see strategy selection
        
        # Test hybrid strategy (multiple concepts and rules)
        hybrid_query = "What is the maximum energy demand and shortage by state with generation output?"
        hybrid_context = self.rag_system.analyze_query_ontology(hybrid_query)
        
        # Test concept-based strategy
        concept_query = "Show me power generation from solar sources"
        concept_context = self.rag_system.analyze_query_ontology(concept_query)
        
        # Test semantic strategy (fallback)
        semantic_query = "Get some data"
        semantic_context = self.rag_system.analyze_query_ontology(semantic_query)
        
        strategies = [hybrid_context.strategy, concept_context.strategy, semantic_context.strategy]
        
        return {
            "status": "PASSED",
            "hybrid_strategy": hybrid_context.strategy.value,
            "concept_strategy": concept_context.strategy.value,
            "semantic_strategy": semantic_context.strategy.value,
            "strategies_determined": len(strategies)
        }

    async def _test_concept_based_retrieval(self):
        """Test concept-based retrieval"""
        query = "energy consumption by region"
        context = self.rag_system.analyze_query_ontology(query)
        
        # Test concept-based retrieval
        examples = self.rag_system._retrieve_concept_based_examples(
            query, context, max_examples=3, min_similarity=0.1
        )
        
        # Check if examples were retrieved
        assert isinstance(examples, list)
        
        return {
            "status": "PASSED",
            "examples_retrieved": len(examples),
            "concept_based_retrieval_working": True
        }

    async def _test_rule_based_retrieval(self):
        """Test rule-based retrieval"""
        query = "maximum demand greater than shortage"
        context = self.rag_system.analyze_query_ontology(query)
        
        # Test rule-based retrieval
        examples = self.rag_system._retrieve_rule_based_examples(
            query, context, max_examples=3, min_similarity=0.1
        )
        
        # Check if examples were retrieved
        assert isinstance(examples, list)
        
        return {
            "status": "PASSED",
            "examples_retrieved": len(examples),
            "rule_based_retrieval_working": True
        }

    async def _test_relationship_based_retrieval(self):
        """Test relationship-based retrieval"""
        query = "state belongs to region"
        context = self.rag_system.analyze_query_ontology(query)
        
        # Test relationship-based retrieval
        examples = self.rag_system._retrieve_relationship_based_examples(
            query, context, max_examples=3, min_similarity=0.1
        )
        
        # Check if examples were retrieved
        assert isinstance(examples, list)
        
        return {
            "status": "PASSED",
            "examples_retrieved": len(examples),
            "relationship_based_retrieval_working": True
        }

    async def _test_hybrid_retrieval(self):
        """Test hybrid retrieval"""
        query = "energy consumption and generation by region with demand analysis"
        context = self.rag_system.analyze_query_ontology(query)
        
        # Test hybrid retrieval
        examples = self.rag_system._retrieve_hybrid_examples(
            query, context, max_examples=5, min_similarity=0.1
        )
        
        # Check if examples were retrieved
        assert isinstance(examples, list)
        
        return {
            "status": "PASSED",
            "examples_retrieved": len(examples),
            "hybrid_retrieval_working": True
        }

    async def _test_example_enhancement(self):
        """Test example enhancement with ontology"""
        # Create a mock example
        mock_example = QueryExample(
            id=1,
            natural_query="energy consumption by region",
            generated_sql="SELECT AVG(EnergyMet) FROM FactAllIndiaDailySummary",
            confidence=0.8,
            success=True,
            execution_time=0.5
        )
        
        # Create a mock context
        context = OntologyRetrievalContext(
            query="energy consumption by region",
            detected_concepts=self.ontology.suggest_concepts("energy consumption"),
            detected_domains=[EnergyDomain.CONSUMPTION],
            business_rules=[],
            relationships=[],
            confidence=0.7,
            strategy=RetrievalStrategy.ONTOLOGY_CONCEPT
        )
        
        # Test example enhancement
        enhanced_example = self.rag_system._enhance_example_with_ontology(mock_example, context)
        
        # Check if example was enhanced
        assert enhanced_example.example == mock_example
        assert isinstance(enhanced_example.ontology_concepts, list)
        assert isinstance(enhanced_example.business_rules, list)
        assert 0.0 <= enhanced_example.domain_relevance <= 1.0
        assert 0.0 <= enhanced_example.rule_compliance <= 1.0
        assert 0.0 <= enhanced_example.context_similarity <= 1.0
        
        return {
            "status": "PASSED",
            "example_enhanced": True,
            "ontology_concepts": len(enhanced_example.ontology_concepts),
            "domain_relevance": enhanced_example.domain_relevance,
            "rule_compliance": enhanced_example.rule_compliance,
            "context_similarity": enhanced_example.context_similarity
        }

    async def _test_business_rule_validation(self):
        """Test business rule validation"""
        # Create mock enhanced examples
        mock_examples = []
        for i in range(3):
            mock_example = QueryExample(
                id=i+1,
                natural_query=f"energy consumption example {i+1}",
                generated_sql=f"SELECT AVG(EnergyMet) FROM FactAllIndiaDailySummary",
                confidence=0.8,
                success=True,
                execution_time=0.5
            )
            
            enhanced_example = OntologyEnhancedExample(
                example=mock_example,
                ontology_concepts=[],
                business_rules=[],
                domain_relevance=0.8,
                rule_compliance=0.9,
                context_similarity=0.7
            )
            mock_examples.append(enhanced_example)
        
        # Test business rule validation
        validated_examples = self.rag_system.validate_examples_against_business_rules(mock_examples)
        
        # Check if validation worked
        assert isinstance(validated_examples, list)
        assert len(validated_examples) <= len(mock_examples)
        
        return {
            "status": "PASSED",
            "examples_validated": len(validated_examples),
            "validation_working": True
        }

    async def _test_example_formatting(self):
        """Test example formatting"""
        # Create mock enhanced examples
        mock_examples = []
        for i in range(2):
            mock_example = QueryExample(
                id=i+1,
                natural_query=f"energy consumption example {i+1}",
                generated_sql=f"SELECT AVG(EnergyMet) FROM FactAllIndiaDailySummary",
                confidence=0.8,
                success=True,
                execution_time=0.5
            )
            
            enhanced_example = OntologyEnhancedExample(
                example=mock_example,
                ontology_concepts=self.ontology.suggest_concepts("energy consumption"),
                business_rules=["positive_energy_values"],
                domain_relevance=0.8,
                rule_compliance=0.9,
                context_similarity=0.7
            )
            mock_examples.append(enhanced_example)
        
        # Test example formatting
        formatted_examples = self.rag_system.format_ontology_enhanced_examples(mock_examples)
        
        # Check if formatting worked
        assert isinstance(formatted_examples, str)
        assert len(formatted_examples) > 0
        assert "Example 1" in formatted_examples
        assert "Example 2" in formatted_examples
        
        return {
            "status": "PASSED",
            "examples_formatted": len(mock_examples),
            "formatting_working": True,
            "formatted_length": len(formatted_examples)
        }

    async def _test_statistics_generation(self):
        """Test statistics generation"""
        # Test statistics generation
        stats = self.rag_system.get_ontology_statistics()
        
        # Check if statistics were generated
        assert isinstance(stats, dict)
        assert "total_concepts" in stats
        assert "total_relationships" in stats
        assert "total_business_rules" in stats
        assert "domains" in stats
        assert "retrieval_strategies" in stats
        
        return {
            "status": "PASSED",
            "statistics_generated": True,
            "total_concepts": stats.get("total_concepts", 0),
            "total_relationships": stats.get("total_relationships", 0),
            "total_business_rules": stats.get("total_business_rules", 0)
        }

    def _print_summary(self):
        """Print test summary"""
        logger.info("\n" + "="*60)
        logger.info("🎯 Ontology-Enhanced RAG Test Summary")
        logger.info("="*60)

        passed = 0
        failed = 0

        for test_name, result in self.test_results:
            if isinstance(result, dict) and result.get("status") == "PASSED":
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                failed += 1
                error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
                logger.error(f"❌ {test_name}: FAILED - {error_msg}")

        logger.info(f"\n📊 Results: {passed} passed, {failed} failed")

        if failed == 0:
            logger.info("🎉 All tests passed! Ontology-enhanced RAG system is working correctly.")
        else:
            logger.error(f"⚠️  {failed} test(s) failed. Please review the errors above.")


async def main():
    """Main test function"""
    tester = OntologyEnhancedRAGTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())






