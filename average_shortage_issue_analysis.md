# Average Shortage SQL Generation Issue Analysis

## Overview

The current system has a critical issue where SQL generation for "average shortage" queries is not following explicit instructions, resulting in incorrect SQL being generated.

## Current Issue: SQL Generation Not Following Explicit Instructions

The problem is that despite having very explicit instructions in the prompt, the LLM is still generating incorrect SQL for "average shortage" queries.

### 1. The Problem

When a user asks: *"What is the average daily energy shortage across all regions?"*

**Expected SQL:**
```sql
SELECT AVG(fs.EnergyShortage) as AverageShortage 
FROM FactAllIndiaDailySummary fs 
JOIN DimRegions r ON fs.RegionID = r.RegionID 
JOIN DimDates d ON fs.DateID = d.DateID
```

**Actual SQL Generated:**
```sql
SELECT 
    r.RegionName,
    strftime('%Y-%m', d.ActualDate) as Month,
    SUM(fs.EnergyMet) as TotalEnergyMet,  -- ❌ WRONG: Should be AVG(fs.EnergyShortage)
    prev.PreviousMonthEnergy,
    CASE 
        WHEN prev.PreviousMonthEnergy > 0 
        THEN ((SUM(fs.EnergyMet) - prev.PreviousMonthEnergy) / prev.PreviousMonthEnergy) * 100 
        ELSE 0 
    END as GrowthRate
FROM FactAllIndiaDailySummary fs
-- ... rest of complex growth calculation query
```

### 2. Root Cause Analysis

The issue stems from several factors:

#### A. LLM Not Following Explicit Instructions

Despite having this explicit instruction in the prompt:
```python
# From the current code
aggregation_instructions = f"""
CRITICAL AGGREGATION AND COLUMN SELECTION RULES:

QUERY ANALYSIS:
- Original query: "{context['query']}"
- Query contains aggregation keywords: {aggregation_keywords}  # ['average']
- Query contains column keywords: {column_keywords}  # ['shortage']

MANDATORY RULES:
- Use aggregation function: {aggregation_function}() (NOT SUM() unless specifically requested)  # AVG()
- Use energy column: {energy_column} (NOT EnergyMet unless specifically requested)  # EnergyShortage

SPECIFIC MAPPINGS:
- For "average shortage" queries: Use AVG(fs.EnergyShortage)
- For "average energy" queries: Use AVG(fs.EnergyMet)
"""
```

The LLM is still generating `SUM(fs.EnergyMet)` instead of `AVG(fs.EnergyShortage)`.

#### B. Prompt Overwhelm

The prompt is extremely long and complex, which may be causing the LLM to:
1. **Lose focus** on the critical aggregation instructions
2. **Default to familiar patterns** (like growth calculations)
3. **Ignore specific instructions** in favor of the example provided

#### C. Conflicting Examples

The prompt includes this example:
```sql
Example structure for monthly growth:
SELECT 
    r.RegionName,
    strftime('%Y-%m', d.ActualDate) as Month,
    SUM(fs.EnergyMet) as TotalEnergyMet,  -- ❌ This example uses SUM(EnergyMet)
    -- ... complex growth calculation
```

This example is **contradicting** the explicit instructions and may be confusing the LLM.

### 3. Current Detection Logic (Working Correctly)

The detection logic is working perfectly:

```python
# This part is working correctly
query_lower = context['query'].lower()

# Aggregation detection
if any(word in query_lower for word in ["average", "avg", "mean"]):
    aggregation_function = "AVG"  # ✅ Correctly detected
    aggregation_keywords = [word for word in ["average", "avg", "mean"] if word in query_lower]

# Column detection  
if "shortage" in query_lower:
    energy_column = "EnergyShortage"  # ✅ Correctly detected
    column_keywords.append("shortage")
```

For the query *"What is the average daily energy shortage across all regions?"*:
- `aggregation_function = "AVG"` ✅
- `energy_column = "EnergyShortage"` ✅
- `aggregation_keywords = ["average"]` ✅
- `column_keywords = ["shortage"]` ✅

### 4. The Real Problem: LLM Response Generation

The issue is in the LLM's response generation, not the detection logic. The LLM is:

1. **Ignoring explicit instructions** about using `AVG()` and `EnergyShortage`
2. **Following the wrong example** (growth calculation with `SUM(EnergyMet)`)
3. **Generating complex queries** when simple aggregation is requested

### 5. Sample Code Showing the Issue

Here's what's happening in the code:

```python
# This detection works perfectly
def _build_sql_generation_prompt(self, context: Dict[str, Any]) -> str:
    query_lower = context['query'].lower()
    
    # ✅ Detection works correctly
    if any(word in query_lower for word in ["average", "avg", "mean"]):
        aggregation_function = "AVG"
        aggregation_keywords = [word for word in ["average", "avg", "mean"] if word in query_lower]
    
    if "shortage" in query_lower:
        energy_column = "EnergyShortage"
        column_keywords.append("shortage")
    
    # ✅ Instructions are explicit
    aggregation_instructions = f"""
    MANDATORY RULES:
    - Use aggregation function: {aggregation_function}()  # AVG()
    - Use energy column: {energy_column}  # EnergyShortage
    
    SPECIFIC MAPPINGS:
    - For "average shortage" queries: Use AVG(fs.EnergyShortage)
    """
    
    # ❌ But the LLM ignores these instructions
    prompt = f"""
    {aggregation_instructions}
    
    # ❌ This example contradicts the instructions
    Example structure for monthly growth:
    SELECT SUM(fs.EnergyMet) as TotalEnergyMet  # Wrong example!
    """
```

### 6. Solution Approaches

To fix this issue, we need to:

#### A. Simplify the Prompt

Remove conflicting examples and focus on the specific query type:

```python
def _build_sql_generation_prompt(self, context: Dict[str, Any]) -> str:
    # ... detection logic ...
    
    # Simplified, focused prompt
    if "average" in query_lower and "shortage" in query_lower:
        prompt = f"""
        Generate a simple SQL query for average shortage calculation.
        
        Query: "{context['query']}"
        
        REQUIRED SQL STRUCTURE:
        SELECT AVG(fs.EnergyShortage) as AverageShortage
        FROM FactAllIndiaDailySummary fs
        JOIN DimRegions r ON fs.RegionID = r.RegionID
        JOIN DimDates d ON fs.DateID = d.DateID
        
        DO NOT USE:
        - SUM() function
        - EnergyMet column
        - Complex growth calculations
        
        Generate ONLY the SQL query:
        """
```

#### B. Add Post-Processing Validation

Add validation to check if the generated SQL follows the rules:

```python
def _validate_sql_against_instructions(self, sql: str, aggregation_function: str, energy_column: str) -> bool:
    sql_lower = sql.lower()
    
    # Check if correct aggregation function is used
    expected_agg = aggregation_function.lower()
    if expected_agg not in sql_lower:
        logger.warning(f"Expected {expected_agg}() but found different aggregation")
        return False
    
    # Check if correct column is used
    if energy_column.lower() not in sql_lower:
        logger.warning(f"Expected {energy_column} column but found different column")
        return False
    
    return True
```

#### C. Use Few-Shot Examples

Add specific examples for average shortage queries:

```python
# Add to few-shot examples
average_shortage_example = QueryExample(
    natural_query="What is the average daily energy shortage?",
    generated_sql="SELECT AVG(fs.EnergyShortage) as AverageShortage FROM FactAllIndiaDailySummary fs",
    confidence=0.95,
    success=True,
    tags=["average", "shortage", "simple_aggregation"]
)
```

### 7. Immediate Fix

The quickest fix would be to modify the prompt to be more direct and remove conflicting examples:

```python
# Remove the complex growth example and focus on the specific query type
if "average" in query_lower and "shortage" in query_lower:
    prompt = f"""
    Generate a simple SQL query for average shortage calculation.
    
    Query: "{context['query']}"
    
    REQUIRED: Use AVG(fs.EnergyShortage)
    FORBIDDEN: Use SUM(fs.EnergyMet)
    
    Generate the SQL query:
    """
```

This would eliminate the confusion and force the LLM to generate the correct SQL for average shortage queries.

## Test Results Summary

The test results show:
- **Detection Logic**: ✅ Working correctly
- **Prompt Instructions**: ✅ Explicit and clear
- **LLM Response**: ❌ Not following instructions
- **Success Rate**: 0% (all tests failed)

## Next Steps

1. **Implement simplified prompts** for specific query types
2. **Add post-processing validation** to catch incorrect SQL
3. **Add more few-shot examples** for average shortage queries
4. **Test with different LLM models** to see if it's model-specific
5. **Consider prompt engineering techniques** like chain-of-thought or few-shot prompting

## Files Affected

- `backend/core/semantic_engine.py` - Main logic for SQL generation
- `test_average_shortage_fix.py` - Test script to reproduce the issue
- `TODO.md` - Update to mark this as a critical issue to fix

## Priority

This is a **critical issue** that affects the core functionality of the application. Users expecting average shortage calculations are getting incorrect results, which could lead to wrong business decisions.
