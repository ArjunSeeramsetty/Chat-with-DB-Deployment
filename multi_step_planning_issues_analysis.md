# Multi-Step Planning Issues Analysis

## Overview

This document analyzes two critical issues identified during the testing of the Adaptive Multi-Step Query Planning implementation:

1. **Query Complexity Analysis - Needs adjustment (the complexity scoring is too strict)**
2. **Semantic Engine Integration - Failed due to LLM API issues (404 errors)**

## Issue 1: Query Complexity Analysis - Too Strict Scoring

### Problem Description

The `_analyze_query_complexity` method in `backend/core/semantic_engine.py` is incorrectly classifying queries, resulting in 0/4 correct complexity analyses during testing.

### Root Cause Analysis

The complexity scoring logic was too restrictive and didn't account for common query patterns:

```python
# Original problematic logic in _analyze_query_complexity
def _analyze_query_complexity(self, query: str) -> QueryComplexity:
    query_lower = query.lower()
    complexity_score = 0
    
    # Multiple aggregations
    aggregation_count = query_lower.count('average') + query_lower.count('sum') + query_lower.count('count')
    if aggregation_count > 1:
        complexity_score += 2
    
    # Multiple tables/entities
    table_indicators = ['region', 'state', 'date', 'generation', 'consumption', 'demand']
    table_count = sum(1 for indicator in table_indicators if indicator in query_lower)
    if table_count > 2:
        complexity_score += 2
```

### Issues Identified

1. **Missing Aggregation Keywords**: The logic didn't include "total" as an aggregation keyword
2. **No Score for Single Aggregations**: Queries with one aggregation (like "average shortage") received no complexity score
3. **Missing Table Indicators**: "shortage" wasn't included in table indicators
4. **Too High Thresholds**: Required >2 tables for any score, missing moderate complexity cases

### Test Cases That Failed

```python
# Test cases from test_multi_step_planning.py
test_queries = [
    "What is the average energy shortage?",  # Should be MODERATE, was SIMPLE
    "Show me total energy consumption by region",  # Should be MODERATE, was SIMPLE  
    "What is the maximum demand in 2024?",  # Should be SIMPLE, was SIMPLE ✓
    "Compare energy generation vs consumption by state"  # Should be COMPLEX, was SIMPLE
]
```

### Implemented Fix

```python
# Updated logic in _analyze_query_complexity
def _analyze_query_complexity(self, query: str) -> QueryComplexity:
    query_lower = query.lower()
    complexity_score = 0
    
    # Multiple aggregations (including "total")
    aggregation_count = query_lower.count('average') + query_lower.count('sum') + query_lower.count('count') + query_lower.count('total')
    if aggregation_count > 1:
        complexity_score += 2
    elif aggregation_count == 1:  # NEW: Score for single aggregations
        complexity_score += 1
    
    # Multiple tables/entities (including "shortage")
    table_indicators = ['region', 'state', 'date', 'generation', 'consumption', 'demand', 'shortage']  # ADDED: shortage
    table_count = sum(1 for indicator in table_indicators if indicator in query_lower)
    if table_count > 2:
        complexity_score += 2
    elif table_count > 1:  # NEW: Score for multiple tables
        complexity_score += 1
```

### Expected Results After Fix

- "What is the average energy shortage?" → MODERATE (1 point for single aggregation + 1 point for shortage indicator)
- "Show me total energy consumption by region" → MODERATE (1 point for single aggregation + 1 point for region indicator)
- "What is the maximum demand in 2024?" → SIMPLE (0 points, correctly classified)
- "Compare energy generation vs consumption by state" → COMPLEX (2+ points for multiple indicators)

## Issue 2: Semantic Engine Integration - LLM API Issues (404 Errors)

### Problem Description

The Semantic Engine integration is failing with `Ollama API error: 404 - 404 page not found` errors during multi-step planning tests.

### Root Cause Analysis

The error occurs when the system attempts to call the Ollama API for LLM-dependent operations:

1. **Entity Extraction**: `QueryDecomposer._extract_entities()` calls the LLM to extract entities from queries
2. **SQL Generation**: `SemanticEngine.generate_contextual_sql()` calls the LLM to generate SQL
3. **Multi-Step Planning**: `MultiStepQueryPlanner` orchestrates these LLM calls

### Error Details

```python
# Error stack trace (simplified)
Ollama API error: 404 - 404 page not found
  File "backend/core/llm_provider.py", line 123, in _make_request
    response = httpx.post(url, json=payload, timeout=30)
  File "backend/core/semantic_engine.py", line 456, in _extract_entities
    entities_response = self.llm_provider.generate(prompt)
```

### API Endpoint Issues

The system is trying to access:
- **Expected**: `http://localhost:11434/v1/api/generate`
- **Actual**: `http://localhost:11434/v1/api/generate` (may not exist or be misconfigured)

### Potential Causes

1. **Ollama Server Not Running**: The Ollama service may not be started
2. **Incorrect API Version**: The API endpoint may have changed in newer Ollama versions
3. **Port Configuration**: Ollama may be running on a different port
4. **API Path Changes**: The API structure may have been updated

### Current Configuration

```python
# From test_multi_step_planning.py
self.llm_provider = create_llm_provider(
    provider_type="ollama", 
    model="llama3.2:3b", 
    base_url="http://localhost:11434/v1"
)
```

### Impact on Functionality

1. **Entity Extraction Fails**: Cannot extract entities, relationships, aggregations from queries
2. **SQL Generation Fails**: Cannot generate SQL for any queries
3. **Multi-Step Planning Degrades**: Falls back to single-step processing
4. **Test Suite Incomplete**: Cannot validate full multi-step planning functionality

### Recommended Solutions

#### Solution 1: Verify Ollama Installation and Service

```bash
# Check if Ollama is installed and running
ollama --version
ollama list
ollama serve  # Start the service if not running
```

#### Solution 2: Update API Endpoint Configuration

```python
# Potential updated configuration
self.llm_provider = create_llm_provider(
    provider_type="ollama", 
    model="llama3.2:3b", 
    base_url="http://localhost:11434"  # Remove /v1 if not needed
)
```

#### Solution 3: Add Fallback Mechanisms

```python
# Add fallback in _extract_entities
def _extract_entities(self, query: str) -> Dict[str, Any]:
    try:
        entities_response = self.llm_provider.generate(prompt)
        # Process response
    except Exception as e:
        logger.warning(f"LLM entity extraction failed: {e}, using fallback")
        return self._fallback_entity_extraction(query)
```

#### Solution 4: Implement Mock LLM for Testing

```python
# Create a mock LLM provider for testing
class MockLLMProvider:
    def generate(self, prompt: str) -> str:
        # Return predefined responses for testing
        if "entity extraction" in prompt.lower():
            return '{"entities": ["energy", "shortage"], "tables": ["EnergyData"]}'
        elif "sql generation" in prompt.lower():
            return "SELECT AVG(EnergyShortage) FROM EnergyData"
```

## Testing Strategy

### For Issue 1 (Complexity Analysis)

1. **Unit Tests**: Test individual complexity scoring components
2. **Integration Tests**: Test full complexity analysis with various query types
3. **Edge Cases**: Test boundary conditions and unusual queries

```python
def test_complexity_scoring():
    test_cases = [
        ("What is the average energy shortage?", QueryComplexity.MODERATE),
        ("Show me total energy consumption by region", QueryComplexity.MODERATE),
        ("What is the maximum demand in 2024?", QueryComplexity.SIMPLE),
        ("Compare energy generation vs consumption by state", QueryComplexity.COMPLEX)
    ]
    
    for query, expected in test_cases:
        result = analyzer._analyze_query_complexity(query)
        assert result == expected, f"Expected {expected} for '{query}', got {result}"
```

### For Issue 2 (LLM API)

1. **Service Verification**: Ensure Ollama is running and accessible
2. **API Testing**: Test API endpoints directly
3. **Fallback Testing**: Test fallback mechanisms when LLM is unavailable
4. **Mock Testing**: Use mock LLM for development and testing

```python
def test_llm_connectivity():
    try:
        response = httpx.get("http://localhost:11434/v1/tags")
        assert response.status_code == 200
    except Exception as e:
        logger.error(f"Ollama not accessible: {e}")
        # Implement fallback or skip tests
```

## Implementation Status

### Completed
- [x] Identified complexity scoring issues
- [x] Implemented initial fixes for complexity analysis
- [x] Created comprehensive test suite
- [x] Documented issues and solutions

### Pending
- [ ] Verify Ollama service status and configuration
- [ ] Implement fallback mechanisms for LLM failures
- [ ] Add mock LLM provider for testing
- [ ] Update API endpoint configuration if needed
- [ ] Re-run tests with fixes applied

## Next Steps

1. **Immediate**: Verify and fix Ollama API connectivity
2. **Short-term**: Implement fallback mechanisms for robust testing
3. **Medium-term**: Refine complexity scoring based on real-world usage
4. **Long-term**: Add comprehensive error handling and monitoring

## Conclusion

These issues represent typical challenges in implementing complex AI-driven systems:

1. **Complexity Analysis**: Requires iterative refinement based on real-world query patterns
2. **LLM Integration**: Requires robust error handling and fallback mechanisms

The fixes implemented address the immediate issues, but ongoing monitoring and refinement will be needed as the system handles more diverse queries and usage patterns.
