#!/usr/bin/env python3
"""
Simple test to verify the monthly aggregation fix.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.core.assembler import SQLAssembler

def test_time_period_detection():
    """Test time period detection in assembler"""
    print("=== Testing Time Period Detection ===")
    assembler = SQLAssembler()
    
    test_cases = [
        {
            "query": "What is the total energy consumption of all regions in 2024?",
            "expected": "none",
            "description": "Query without time period should return 'none'"
        },
        {
            "query": "What is the monthly energy consumption of all regions in 2024?", 
            "expected": "monthly",
            "description": "Query with 'monthly' should return 'monthly'"
        },
        {
            "query": "Show daily energy consumption for states",
            "expected": "daily", 
            "description": "Query with 'daily' should return 'daily'"
        },
        {
            "query": "What is energy consumption by month for regions?",
            "expected": "monthly",
            "description": "Query with 'by month' should return 'monthly'"
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        result = assembler._determine_time_period(test_case["query"])
        
        print(f"Test {i}: {test_case['description']}")
        print(f"Query: {test_case['query']}")
        print(f"Expected: {test_case['expected']}")
        print(f"Got: {result}")
        
        if result == test_case["expected"]:
            print("✅ PASSED")
        else:
            print("❌ FAILED")
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 All tests PASSED! The monthly aggregation fix is working correctly.")
    else:
        print("❌ Some tests failed.")

if __name__ == "__main__":
    test_time_period_detection()



