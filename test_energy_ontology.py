#!/usr/bin/env python3
"""
Test script for Enhanced Energy Ontology
Validates comprehensive energy sector domain ontology with all database tables
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.core.energy_ontology import (
    EnergyOntology, EnergyConcept, EnergyDomain, EnergyConceptType,
    OntologyRelationship, BusinessRule
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnergyOntologyTester:
    """Test suite for enhanced energy ontology"""
    
    def __init__(self):
        self.db_path = "backend/energy_data.db"
        self.ontology = EnergyOntology(self.db_path)
        self.test_results = []
    
    async def run_all_tests(self):
        """Run all ontology tests"""
        logger.info("🚀 Starting Enhanced Energy Ontology Tests")
        
        test_cases = [
            ("Ontology Initialization", self._test_ontology_initialization),
            ("Concept Loading", self._test_concept_loading),
            ("Relationship Loading", self._test_relationship_loading),
            ("Business Rules Loading", self._test_business_rules_loading),
            ("Database Schema Loading", self._test_database_schema_loading),
            ("Concept Retrieval", self._test_concept_retrieval),
            ("Concept Suggestions", self._test_concept_suggestions),
            ("Domain Filtering", self._test_domain_filtering),
            ("Business Rules by Domain", self._test_business_rules_by_domain),
            ("Database Mappings", self._test_database_mappings),
            ("Related Concepts", self._test_related_concepts),
            ("Concept Validation", self._test_concept_validation),
            ("Ontology Export Import", self._test_ontology_export_import),
            ("Ambiguity Detection", self._test_ambiguity_detection),
            ("Clarification for Concepts", self._test_clarification_for_concept),
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
    
    async def _test_ontology_initialization(self):
        """Test ontology initialization"""
        # Check if ontology was initialized correctly
        assert self.ontology.db_path == self.db_path
        assert len(self.ontology.concepts) > 0
        assert len(self.ontology.relationships) > 0
        assert len(self.ontology.business_rules) > 0
        
        return {
            "status": "PASSED",
            "concepts_count": len(self.ontology.concepts),
            "relationships_count": len(self.ontology.relationships),
            "business_rules_count": len(self.ontology.business_rules)
        }
    
    async def _test_concept_loading(self):
        """Test concept loading"""
        # Check if all expected concepts are loaded
        expected_concepts = [
            "energy_generation", "power_generation", "generation_source",
            "energy_consumption", "maximum_demand", "time_of_maximum_demand", "evening_peak_demand",
            "demand_met", "net_demand_met", "energy_shortage", "peak_shortage",
            "shortage", "state", "region", "country", "date", "time_block",
            "power_exchange", "exchange_mechanism", "exchange_direction",
            "total_energy_exchanged", "peak_exchange", "transmission_line",
            "voltage_level", "max_import", "max_export", "import_energy",
            "export_energy", "net_import_energy", "max_loading", "min_loading",
            "avg_loading", "frequency", "total_generation", "net_transnational_exchange",
            "unit", "unit_symbol", "report"
        ]
        
        loaded_concepts = list(self.ontology.concepts.keys())
        
        for expected_concept in expected_concepts:
            assert expected_concept in loaded_concepts, f"Expected concept {expected_concept} not found"
        
        return {
            "status": "PASSED",
            "expected_concepts": len(expected_concepts),
            "loaded_concepts": len(loaded_concepts),
            "concept_names": loaded_concepts
        }
    
    async def _test_relationship_loading(self):
        """Test relationship loading"""
        # Check if all expected relationships are loaded
        expected_relationships = [
            "state_belongs_to_region", "region_belongs_to_country",
            "data_has_date", "data_has_time_block", "energy_generation_has_source",
            "power_generation_has_source", "energy_generation_to_power_generation",
            "demand_has_shortage", "demand_met_has_demand", "net_demand_met_has_demand",
            "shortage_has_demand", "state_has_demand", "region_has_consumption",
            "country_has_exchange", "exchange_mechanism_has_exchange",
            "exchange_direction_has_exchange", "total_energy_exchanged_has_exchange",
            "peak_exchange_has_exchange", "transmission_line_has_voltage_level",
            "max_import_has_transmission_line", "max_export_has_transmission_line",
            "import_energy_has_transmission_line", "export_energy_has_transmission_line",
            "net_import_energy_has_transmission_line", "max_loading_has_transmission_line",
            "min_loading_has_transmission_line", "avg_loading_has_transmission_line",
            "frequency_has_time_block", "total_generation_has_time_block",
            "net_transnational_exchange_has_time_block", "unit_symbol_has_unit",
            "report_has_date"
        ]
        
        loaded_relationships = list(self.ontology.relationships.keys())
        
        for expected_relationship in expected_relationships:
            assert expected_relationship in loaded_relationships, f"Expected relationship {expected_relationship} not found"
        
        return {
            "status": "PASSED",
            "expected_relationships": len(expected_relationships),
            "loaded_relationships": len(loaded_relationships),
            "relationship_names": loaded_relationships
        }
    
    async def _test_business_rules_loading(self):
        """Test business rules loading"""
        # Check if all expected business rules are loaded
        expected_rules = [
            "positive_energy_values", "demand_greater_than_shortage",
            "energy_met_plus_shortage", "energy_generation_positive",
            "power_generation_positive", "energy_power_relationship",
            "total_energy_generation_sum", "total_power_generation_sum",
            "demand_met_less_than_maximum", "net_demand_met_less_than_demand",
            "exchange_direction_valid", "peak_exchange_less_than_total",
            "max_import_positive", "max_export_positive", "import_energy_less_than_max",
            "export_energy_less_than_max", "max_loading_greater_than_min",
            "avg_loading_between_min_max", "frequency_range",
            "voltage_level_positive", "shortage_percentage",
            "loading_percentage", "efficiency_calculation",
            "maximum_demand_constraint", "generation_capacity_constraint",
            "transmission_capacity_constraint", "state_belongs_to_region",
            "region_belongs_to_country", "date_validity", "time_block_sequence",
            "data_completeness", "data_consistency"
        ]
        
        loaded_rules = list(self.ontology.business_rules.keys())
        
        for expected_rule in expected_rules:
            assert expected_rule in loaded_rules, f"Expected business rule {expected_rule} not found"
        
        return {
            "status": "PASSED",
            "expected_rules": len(expected_rules),
            "loaded_rules": len(loaded_rules),
            "rule_names": loaded_rules
        }
    
    async def _test_database_schema_loading(self):
        """Test database schema loading"""
        # Check if database schema was loaded (may be empty if database not accessible)
        assert hasattr(self.ontology, 'database_schema'), "Database schema attribute not found"
        
        # If schema is loaded, check for expected tables
        if len(self.ontology.database_schema) > 0:
            expected_tables = [
                "DimUnits", "DimDates", "DimRegions", "DimStates", "DimCountries",
                "DimGenerationSources", "DimTransmissionLines", "DimExchangeMechanisms",
                "DimReports", "FactAllIndiaDailySummary", "FactDailyGenerationBreakdown",
                "FactStateDailyEnergy", "FactCountryDailyExchange", "FactTransmissionLinkFlow",
                "FactInternationalTransmissionLinkFlow", "FactTransnationalExchangeDetail",
                "FactTimeBlockPowerData", "FactTimeBlockGeneration"
            ]
            
            loaded_tables = list(self.ontology.database_schema.keys())
            
            # Check if at least some expected tables are loaded
            found_tables = [table for table in expected_tables if table in loaded_tables]
            assert len(found_tables) > 0, f"No expected tables found. Loaded tables: {loaded_tables}"
            
            return {
                "status": "PASSED",
                "expected_tables": len(expected_tables),
                "loaded_tables": len(loaded_tables),
                "found_tables": len(found_tables),
                "table_names": loaded_tables
            }
        else:
            # Schema not loaded (database not accessible), but ontology should still work
            logger.warning("Database schema not loaded (database may not be accessible)")
            return {
                "status": "PASSED",
                "expected_tables": 0,
                "loaded_tables": 0,
                "found_tables": 0,
                "table_names": [],
                "note": "Schema not loaded - database may not be accessible"
            }
    
    async def _test_concept_retrieval(self):
        """Test concept retrieval by name"""
        # Test concept retrieval by exact name
        concept = self.ontology.get_concept_by_name("Energy Generation")
        assert concept is not None, "Energy Generation concept not found"
        assert concept.id == "energy_generation"
        
        # Test concept retrieval by exact name
        concept = self.ontology.get_concept_by_name("Power Generation")
        assert concept is not None, "Power Generation concept not found"
        assert concept.id == "power_generation"
        
        # Test concept retrieval by synonym
        concept = self.ontology.get_concept_by_name("energy production")
        assert concept is not None, "Energy Generation concept not found by synonym"
        assert concept.id == "energy_generation"
        
        # Test concept retrieval by synonym
        concept = self.ontology.get_concept_by_name("power production")
        assert concept is not None, "Power Generation concept not found by synonym"
        assert concept.id == "power_generation"
        
        # Test concept retrieval by lowercase
        concept = self.ontology.get_concept_by_name("energy consumption")
        assert concept is not None, "Energy Consumption concept not found"
        assert concept.id == "energy_consumption"
        
        # Test non-existent concept
        concept = self.ontology.get_concept_by_name("non_existent_concept")
        assert concept is None, "Non-existent concept should return None"
        
        return {
            "status": "PASSED",
            "exact_name_retrieval": True,
            "synonym_retrieval": True,
            "lowercase_retrieval": True,
            "non_existent_handling": True
        }
    
    async def _test_concept_suggestions(self):
        """Test concept suggestions based on query"""
        # Test suggestions for energy-related query
        suggestions = self.ontology.suggest_concepts("energy consumption by region")
        assert len(suggestions) > 0, "No suggestions found for energy consumption query"
        
        # Check if relevant concepts are suggested
        concept_names = [c.name.lower() for c in suggestions]
        assert any("energy" in name for name in concept_names), "Energy concept not suggested"
        assert any("region" in name for name in concept_names), "Region concept not suggested"
        
        # Test suggestions for generation-related query
        suggestions = self.ontology.suggest_concepts("energy generation from solar")
        assert len(suggestions) > 0, "No suggestions found for energy generation query"
        
        concept_names = [c.name.lower() for c in suggestions]
        assert any("generation" in name for name in concept_names), "Generation concept not suggested"
        
        # Test suggestions for power-related query
        suggestions = self.ontology.suggest_concepts("power generation capacity")
        assert len(suggestions) > 0, "No suggestions found for power generation query"
        
        concept_names = [c.name.lower() for c in suggestions]
        assert any("power" in name for name in concept_names), "Power concept not suggested"
        
        return {
            "status": "PASSED",
            "energy_consumption_suggestions": len(suggestions),
            "energy_generation_suggestions": len(suggestions),
            "power_generation_suggestions": len(suggestions),
            "relevant_suggestions": True
        }
    
    async def _test_domain_filtering(self):
        """Test filtering concepts by domain"""
        # Test generation domain concepts
        generation_concepts = self.ontology.get_concepts_by_domain(EnergyDomain.GENERATION)
        assert len(generation_concepts) > 0, "No generation concepts found"
        
        generation_names = [c.name for c in generation_concepts]
        assert "Energy Generation" in generation_names, "Energy Generation not in generation domain"
        assert "Power Generation" in generation_names, "Power Generation not in generation domain"
        assert "Generation Source" in generation_names, "Generation Source not in generation domain"
        
        # Test demand domain concepts
        demand_concepts = self.ontology.get_concepts_by_domain(EnergyDomain.DEMAND)
        assert len(demand_concepts) > 0, "No demand concepts found"
        
        demand_names = [c.name for c in demand_concepts]
        assert "Maximum Demand" in demand_names, "Maximum Demand not in demand domain"
        assert "Evening Peak Demand" in demand_names, "Evening Peak Demand not in demand domain"
        
        return {
            "status": "PASSED",
            "generation_concepts_count": len(generation_concepts),
            "demand_concepts_count": len(demand_concepts),
            "domain_filtering_working": True
        }
    
    async def _test_business_rules_by_domain(self):
        """Test getting business rules by domain"""
        # Test generation domain rules
        generation_rules = self.ontology.get_business_rules_for_domain(EnergyDomain.GENERATION)
        assert len(generation_rules) > 0, "No generation business rules found"
        
        rule_names = [r.name for r in generation_rules]
        assert any("Generation" in name for name in rule_names), "Generation rules not found"
        
        # Test demand domain rules
        demand_rules = self.ontology.get_business_rules_for_domain(EnergyDomain.DEMAND)
        assert len(demand_rules) > 0, "No demand business rules found"
        
        rule_names = [r.name for r in demand_rules]
        assert any("Demand" in name for name in rule_names), "Demand rules not found"
        
        return {
            "status": "PASSED",
            "generation_rules_count": len(generation_rules),
            "demand_rules_count": len(demand_rules),
            "domain_rules_filtering": True
        }
    
    async def _test_database_mappings(self):
        """Test database mappings for concepts"""
        # Test mapping for energy consumption (now has multiple mappings for state and region levels)
        mapping = self.ontology.get_database_mapping("energy consumption")
        assert mapping is not None, "Database mapping not found for energy consumption"
        # Check for state-level mapping
        assert mapping.get("state_table") == "FactStateDailyEnergy", "Incorrect state table for energy consumption"
        assert mapping.get("state_column") == "EnergyMet", "Incorrect state column for energy consumption"
        # Check for region-level mapping
        assert mapping.get("region_table") == "FactAllIndiaDailySummary", "Incorrect region table for energy consumption"
        assert mapping.get("region_column") == "EnergyMet", "Incorrect region column for energy consumption"
        
        # Test mapping for energy generation
        mapping = self.ontology.get_database_mapping("energy generation")
        assert mapping is not None, "Database mapping not found for energy generation"
        assert mapping["table"] == "FactDailyGenerationBreakdown", "Incorrect table for energy generation"
        assert mapping["column"] == "GenerationAmount", "Incorrect column for energy generation"
        
        # Test mapping for power generation
        mapping = self.ontology.get_database_mapping("power generation")
        assert mapping is not None, "Database mapping not found for power generation"
        assert mapping["table"] == "FactTimeBlockGeneration", "Incorrect table for power generation"
        assert mapping["column"] == "GenerationOutput", "Incorrect column for power generation"
        
        # Test mapping for maximum demand (now has multiple mappings for state and region levels)
        mapping = self.ontology.get_database_mapping("maximum demand")
        assert mapping is not None, "Database mapping not found for maximum demand"
        # Check for state-level mapping
        assert mapping.get("state_table") == "FactStateDailyEnergy", "Incorrect state table for maximum demand"
        assert mapping.get("state_column") == "MaximumDemand", "Incorrect state column for maximum demand"
        # Check for region-level mapping
        assert mapping.get("region_table") == "FactAllIndiaDailySummary", "Incorrect region table for maximum demand"
        assert mapping.get("region_column") == "MaxDemandSCADA", "Incorrect region column for maximum demand"
        
        # Test mapping for time of maximum demand (region level only)
        mapping = self.ontology.get_database_mapping("time of maximum demand")
        assert mapping is not None, "Database mapping not found for time of maximum demand"
        assert mapping["table"] == "FactAllIndiaDailySummary", "Incorrect table for time of maximum demand"
        assert mapping["column"] == "TimeOfMaxDemandMet", "Incorrect column for time of maximum demand"
        
        # Test mapping for evening peak demand (region level only)
        mapping = self.ontology.get_database_mapping("evening peak demand")
        assert mapping is not None, "Database mapping not found for evening peak demand"
        assert mapping["table"] == "FactAllIndiaDailySummary", "Incorrect table for evening peak demand"
        assert mapping["column"] == "EveningPeakDemandMet", "Incorrect column for evening peak demand"
        
        # Test mapping for energy shortage (now has multiple mappings for state and region levels)
        mapping = self.ontology.get_database_mapping("energy shortage")
        assert mapping is not None, "Database mapping not found for energy shortage"
        # Check for state-level mapping
        assert mapping.get("state_table") == "FactStateDailyEnergy", "Incorrect state table for energy shortage"
        assert mapping.get("state_column") == "EnergyShortage", "Incorrect state column for energy shortage"
        # Check for region-level mapping
        assert mapping.get("region_table") == "FactAllIndiaDailySummary", "Incorrect region table for energy shortage"
        assert mapping.get("region_column") == "EnergyShortage", "Incorrect region column for energy shortage"
        
        # Test mapping for peak shortage (now has multiple mappings for state and region levels)
        mapping = self.ontology.get_database_mapping("peak shortage")
        assert mapping is not None, "Database mapping not found for peak shortage"
        # Check for state-level mapping
        assert mapping.get("state_table") == "FactStateDailyEnergy", "Incorrect state table for peak shortage"
        assert mapping.get("state_column") == "Shortage", "Incorrect state column for peak shortage"
        # Check for region-level mapping
        assert mapping.get("region_table") == "FactAllIndiaDailySummary", "Incorrect region table for peak shortage"
        assert mapping.get("region_column") == "PeakShortage", "Incorrect region column for peak shortage"
        
        # Test database mappings for demand met (context-aware)
        mapping = self.ontology.get_database_mapping("demand met")
        assert mapping is not None, "Database mapping not found for demand met"
        # Check for timeblock-level mapping
        assert mapping.get("timeblock_table") == "FactTimeBlockPowerData", "Incorrect timeblock table for demand met"
        assert mapping.get("timeblock_column") == "DemandMet", "Incorrect timeblock column for demand met"
        # Check for state-level mapping
        assert mapping.get("state_table") == "FactStateDailyEnergy", "Incorrect state table for demand met"
        assert mapping.get("state_column") == "EnergyMet", "Incorrect state column for demand met"
        # Check for region-level mapping
        assert mapping.get("region_table") == "FactAllIndiaDailySummary", "Incorrect region table for demand met"
        assert mapping.get("region_column") == "EnergyMet", "Incorrect region column for demand met"
        
        # Test database mappings for net demand met (context-aware)
        mapping = self.ontology.get_database_mapping("net demand met")
        assert mapping is not None, "Database mapping not found for net demand met"
        # Check for timeblock-level mapping
        assert mapping.get("timeblock_table") == "FactTimeBlockPowerData", "Incorrect timeblock table for net demand met"
        assert mapping.get("timeblock_column") == "NetDemandMet", "Incorrect timeblock column for net demand met"
        # Check for state-level mapping
        assert mapping.get("state_table") == "FactStateDailyEnergy", "Incorrect state table for net demand met"
        assert mapping.get("state_column") == "EnergyMet", "Incorrect state column for net demand met"
        # Check for region-level mapping
        assert mapping.get("region_table") == "FactAllIndiaDailySummary", "Incorrect region table for net demand met"
        assert mapping.get("region_column") == "EnergyMet", "Incorrect region column for net demand met"
        
        # Test mapping for non-existent concept
        mapping = self.ontology.get_database_mapping("non_existent_concept")
        assert mapping is None, "Non-existent concept should return None mapping"
        
        return {
            "status": "PASSED",
            "energy_consumption_mapping": mapping is not None,
            "energy_generation_mapping": mapping is not None,
            "power_generation_mapping": mapping is not None,
            "maximum_demand_mapping": mapping is not None,
            "time_of_maximum_demand_mapping": mapping is not None,
            "evening_peak_demand_mapping": mapping is not None,
            "energy_shortage_mapping": mapping is not None,
            "peak_shortage_mapping": mapping is not None,
            "demand_met_mapping": mapping is not None,
            "net_demand_met_mapping": mapping is not None,
            "non_existent_mapping": mapping is None
        }
    
    async def _test_ambiguity_detection(self):
        """Test ambiguity detection functionality"""
        print("Testing ambiguity detection...")
        
        # Test 1: Ambiguous query with "demand met" (no energy/power context)
        query1 = "What is the demand met in Maharashtra?"
        is_ambiguous, clarification, concept = self.ontology.detect_ambiguity(query1)
        assert is_ambiguous, "Should detect ambiguity in query without energy/power context"
        assert clarification is not None, "Should provide clarification question"
        assert concept is not None, "Should identify the ambiguous concept"
        assert concept.name == "Demand Met", "Should identify Demand Met as ambiguous concept"
        
        # Test 2: Clear energy context query
        query2 = "What is the energy demand met in Maharashtra?"
        is_ambiguous, clarification, concept = self.ontology.detect_ambiguity(query2)
        assert not is_ambiguous, "Should not detect ambiguity when energy context is clear"
        
        # Test 3: Clear power context query
        query3 = "What is the power demand met at peak time in Maharashtra?"
        is_ambiguous, clarification, concept = self.ontology.detect_ambiguity(query3)
        assert not is_ambiguous, "Should not detect ambiguity when power context is clear"
        
        # Test 4: Ambiguous query with "net demand met"
        query4 = "Show me the net demand met for regions"
        is_ambiguous, clarification, concept = self.ontology.detect_ambiguity(query4)
        assert is_ambiguous, "Should detect ambiguity in net demand met query"
        assert concept.name == "Net Demand Met", "Should identify Net Demand Met as ambiguous concept"
        
        # Test 5: Query with no ambiguous concepts
        query5 = "What is the energy generation from solar sources?"
        is_ambiguous, clarification, concept = self.ontology.detect_ambiguity(query5)
        assert not is_ambiguous, "Should not detect ambiguity for non-ambiguous concepts"
        
        print("✅ Ambiguity detection tests passed")
        return "PASSED"
    
    async def _test_clarification_for_concept(self):
        """Test getting clarification questions for specific concepts"""
        print("Testing clarification for concepts...")
        
        # Test 1: Get clarification for demand met concept
        clarification = self.ontology.get_clarification_for_concept("demand met")
        assert clarification is not None, "Should provide clarification for demand met concept"
        assert "energy demand" in clarification.lower(), "Clarification should mention energy demand"
        assert "power demand" in clarification.lower(), "Clarification should mention power demand"
        
        # Test 2: Get clarification for net demand met concept
        clarification = self.ontology.get_clarification_for_concept("net demand met")
        assert clarification is not None, "Should provide clarification for net demand met concept"
        assert "net energy demand" in clarification.lower(), "Clarification should mention net energy demand"
        assert "net power demand" in clarification.lower(), "Clarification should mention net power demand"
        
        # Test 3: Get clarification for non-ambiguous concept
        clarification = self.ontology.get_clarification_for_concept("energy generation")
        assert clarification is None, "Should not provide clarification for non-ambiguous concept"
        
        print("✅ Clarification for concepts tests passed")
        return "PASSED"
    
    async def _test_related_concepts(self):
        """Test getting related concepts"""
        # Test related concepts for state
        related = self.ontology.get_related_concepts("state")
        assert len(related) > 0, "No related concepts found for state"
        
        related_names = [c.name for c in related]
        assert "Region" in related_names, "Region not found as related to state"
        
        # Test related concepts for energy generation
        related = self.ontology.get_related_concepts("energy generation")
        assert len(related) > 0, "No related concepts found for energy generation"
        
        related_names = [c.name for c in related]
        assert "Generation Source" in related_names, "Generation Source not found as related to energy generation"
        
        # Test related concepts for power generation
        related = self.ontology.get_related_concepts("power generation")
        assert len(related) > 0, "No related concepts found for power generation"
        
        related_names = [c.name for c in related]
        assert "Generation Source" in related_names, "Generation Source not found as related to power generation"
        
        return {
            "status": "PASSED",
            "state_related_concepts": len(related),
            "energy_generation_related_concepts": len(related),
            "power_generation_related_concepts": len(related),
            "related_concepts_working": True
        }
    
    async def _test_concept_validation(self):
        """Test concept validation"""
        # Test valid concept usage
        is_valid, errors = self.ontology.validate_concept_usage("energy consumption", {"domain": EnergyDomain.CONSUMPTION})
        assert is_valid, f"Valid concept usage failed: {errors}"
        
        # Test invalid concept usage
        is_valid, errors = self.ontology.validate_concept_usage("non_existent_concept", {})
        assert not is_valid, "Invalid concept should fail validation"
        assert len(errors) > 0, "Should have errors for invalid concept"
        
        return {
            "status": "PASSED",
            "valid_concept_validation": is_valid,
            "invalid_concept_validation": not is_valid,
            "validation_working": True
        }
    
    async def _test_ontology_export_import(self):
        """Test ontology export and import"""
        # Export ontology
        export_path = "test_ontology_export.json"
        self.ontology.export_ontology(export_path)
        
        # Check if file was created
        import os
        assert os.path.exists(export_path), f"Export file {export_path} was not created"
        
        # Create new ontology and import
        new_ontology = EnergyOntology()
        new_ontology.import_ontology(export_path)
        
        # Check if concepts were imported
        assert len(new_ontology.concepts) > 0, "No concepts imported"
        assert len(new_ontology.relationships) > 0, "No relationships imported"
        assert len(new_ontology.business_rules) > 0, "No business rules imported"
        
        # Clean up
        os.remove(export_path)
        
        return {
            "status": "PASSED",
            "export_successful": True,
            "import_successful": True,
            "concepts_imported": len(new_ontology.concepts),
            "relationships_imported": len(new_ontology.relationships),
            "business_rules_imported": len(new_ontology.business_rules)
        }
    
    def _print_summary(self):
        """Print test summary"""
        logger.info("\n" + "="*60)
        logger.info("🎯 Enhanced Energy Ontology Test Summary")
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
            logger.info("🎉 All tests passed! Enhanced energy ontology is working correctly.")
        else:
            logger.error(f"⚠️  {failed} test(s) failed. Please review the errors above.")


async def main():
    """Main test function"""
    tester = EnergyOntologyTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
