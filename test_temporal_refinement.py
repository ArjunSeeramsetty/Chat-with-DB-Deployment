#!/usr/bin/env python3
"""
Test script to verify temporal refinement functionality
"""
import sys
import asyncio
sys.path.append('.')

from backend.services.enhanced_rag_service import EnhancedRAGService

async def test_temporal_refinement():
    """Test the temporal refinement method directly"""
    
    # Initialize the service
    service = EnhancedRAGService("data/power_sector.db")
    await service.initialize()
    
    # Test SQL without month filtering
    test_sql = """SELECT s.StateName, ROUND(MAX(fs.EnergyMet), 2) AS Value 
    FROM FactStateDailyEnergy fs 
    JOIN DimStates s ON fs.StateID = s.StateID 
    JOIN DimDates d ON fs.DateID = d.DateID 
    WHERE strftime('%Y', d.ActualDate) = '2025' 
    GROUP BY s.StateName 
    ORDER BY s.StateName"""
    
    # Test query with month reference
    test_query = "What is the maximum Energy Met of all states in June 2025?"
    
    print("Original SQL:")
    print(test_sql)
    print("\nQuery:")
    print(test_query)
    
    # Apply temporal refinement
    refined_sql = service._validate_and_correct_sql(test_sql, test_query)
    
    print("\nRefined SQL:")
    print(refined_sql)
    
    # Check if month filtering was added
    if "strftime('%m'" in refined_sql:
        print("\n✅ SUCCESS: Month filtering was added")
    else:
        print("\n❌ FAILED: Month filtering was not added")

if __name__ == "__main__":
    asyncio.run(test_temporal_refinement())
