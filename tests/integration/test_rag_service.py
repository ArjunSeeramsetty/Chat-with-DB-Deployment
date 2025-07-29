"""
Integration tests for RAG service components
Sprint 6: PyTest matrix and GitHub Actions CI
"""
import pytest
import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.core.types import QueryRequest
from backend.services.rag_service import get_rag_service
from backend.config import get_settings

class TestRAGServiceIntegration:
    """Test RAG service integration with all components"""
    
    @pytest.fixture
    def settings(self):
        """Get application settings"""
        return get_settings()
    
    @pytest.fixture
    def rag_service(self, settings):
        """Create RAG service instance for testing"""
        return get_rag_service(settings.database_path)
    
    @pytest.mark.integration
    @pytest.mark.sprint1
    def test_rag_service_initialization(self, rag_service):
        """Test RAG service initialization with all components"""
        assert rag_service is not None
        assert hasattr(rag_service, 'intent_analyzer')
        assert hasattr(rag_service, 'sql_assembler')
        assert hasattr(rag_service, 'enhanced_validator')
        assert hasattr(rag_service, 'async_sql_executor')
    
    @pytest.mark.integration
    @pytest.mark.sprint2
    @pytest.mark.clarification
    def test_clarification_flow_integration(self, rag_service):
        """Test clarification flow integration"""
        # Test ambiguous query that should trigger clarification
        request = QueryRequest(
            question="What is the energy?",
            user_id="test_user",
            processing_mode="balanced",
            clarification_attempt_count=0
        )
        
        async def test_clarification():
            response = await rag_service.process_query(request)
            # Should either succeed or request clarification
            assert response is not None
            assert hasattr(response, 'success')
            assert hasattr(response, 'clarification_needed')
        
        asyncio.run(test_clarification())
    
    @pytest.mark.integration
    @pytest.mark.sprint3
    def test_entity_loader_integration(self, rag_service):
        """Test entity loader integration"""
        # Test that entity loader is working
        assert hasattr(rag_service.intent_analyzer, 'entity_loader')
        assert hasattr(rag_service.sql_assembler, 'entity_loader')
        assert hasattr(rag_service.schema_linker, 'entity_loader')
    
    @pytest.mark.integration
    @pytest.mark.sprint4
    @pytest.mark.performance
    def test_async_executor_integration(self, rag_service):
        """Test async executor integration"""
        # Test that async executor is working
        assert hasattr(rag_service, 'async_sql_executor')
        
        async def test_async_execution():
            result = await rag_service.async_sql_executor.execute_sql_async("SELECT 1 as test")
            assert result.success is True
            assert result.data is not None
        
        asyncio.run(test_async_execution())
    
    @pytest.mark.integration
    @pytest.mark.sprint5
    @pytest.mark.validation
    def test_enhanced_validator_integration(self, rag_service):
        """Test enhanced validator integration"""
        # Test that enhanced validator is working
        assert hasattr(rag_service, 'enhanced_validator')
        
        # Test validation of a simple query
        test_sql = "SELECT StateName FROM FactStateDailyEnergy"
        result = rag_service.enhanced_validator.validate_sql(test_sql)
        assert result.is_valid is True
        assert result.confidence > 0.8
    
    @pytest.mark.integration
    @pytest.mark.sprint5
    @pytest.mark.security
    def test_security_validation_integration(self, rag_service):
        """Test security validation integration"""
        # Test security validation
        dangerous_sql = "DROP TABLE FactStateDailyEnergy"
        result = rag_service.enhanced_validator.validate_sql(dangerous_sql)
        assert result.is_valid is False
        assert result.confidence < 0.6
        assert any("DROP" in error for error in result.errors)
    
    @pytest.mark.integration
    @pytest.mark.e2e
    def test_end_to_end_query_processing(self, rag_service):
        """Test end-to-end query processing"""
        request = QueryRequest(
            question="What is the energy consumption?",
            user_id="test_user",
            processing_mode="balanced",
            clarification_attempt_count=0
        )
        
        async def test_e2e():
            response = await rag_service.process_query(request)
            assert response is not None
            assert hasattr(response, 'success')
            assert hasattr(response, 'sql_query')
            assert hasattr(response, 'confidence')
        
        asyncio.run(test_e2e())

class TestComponentIntegration:
    """Test integration between different components"""
    
    @pytest.mark.integration
    @pytest.mark.sprint1
    def test_intent_analyzer_integration(self):
        """Test intent analyzer integration with other components"""
        from backend.core.intent import IntentAnalyzer
        
        analyzer = IntentAnalyzer()
        
        # Test query analysis
        query = "What is the energy consumption in Maharashtra?"
        analysis = analyzer.analyze_query(query)
        
        assert analysis is not None
        assert hasattr(analysis, 'query_type')
        assert hasattr(analysis, 'entities')
        assert hasattr(analysis, 'confidence')
    
    @pytest.mark.integration
    @pytest.mark.sprint2
    def test_sql_assembler_integration(self):
        """Test SQL assembler integration"""
        from backend.core.assembler import SQLAssembler
        
        assembler = SQLAssembler()
        
        # Test SQL generation
        from backend.core.types import QueryAnalysis, QueryType
        analysis = QueryAnalysis(
            query_type=QueryType.STATE,
            entities=['maharashtra'],
            confidence=0.9
        )
        
        sql_result = assembler.generate_sql("What is the energy consumption?", analysis)
        assert sql_result is not None
    
    @pytest.mark.integration
    @pytest.mark.sprint3
    def test_entity_loader_integration(self):
        """Test entity loader integration"""
        from backend.core.entity_loader import get_entity_loader
        
        loader = get_entity_loader()
        
        # Test entity retrieval
        states = loader.get_indian_states()
        regions = loader.get_indian_regions()
        
        assert len(states) > 0
        assert len(regions) > 0
        assert "maharashtra" in [s.lower() for s in states]
        assert "northern region" in [r.lower() for r in regions]
    
    @pytest.mark.integration
    @pytest.mark.sprint4
    @pytest.mark.performance
    def test_async_executor_integration(self):
        """Test async executor integration"""
        from backend.core.executor import AsyncSQLExecutor
        from backend.config import get_settings
        
        settings = get_settings()
        executor = AsyncSQLExecutor(settings.database_path)
        
        async def test_executor():
            result = await executor.execute_sql_async("SELECT 1 as test")
            assert result.success is True
            assert result.data is not None
        
        asyncio.run(test_executor())
    
    @pytest.mark.integration
    @pytest.mark.sprint5
    @pytest.mark.validation
    def test_validator_integration(self):
        """Test validator integration"""
        from backend.core.validator import EnhancedSQLValidator
        
        schema_info = {
            'FactStateDailyEnergy': ['StateID', 'EnergyMet', 'DemandMet', 'StateName'],
            'FactAllIndiaDailySummary': ['RegionID', 'MaxDemandSCADA', 'CentralSectorOutage', 'RegionName']
        }
        
        validator = EnhancedSQLValidator(schema_info)
        
        # Test validation
        result = validator.validate_sql("SELECT StateName FROM FactStateDailyEnergy")
        assert result.is_valid is True
        assert result.confidence > 0.8 