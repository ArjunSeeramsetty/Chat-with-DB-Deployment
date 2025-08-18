"""
Run One Query - Deployment Version
Simple script to test individual queries with the Cloud-Ready SemanticEngine
"""

import requests
import json
import sys


def test_query(query: str, user_id: str = "tester", session_id: str = "sess-single"):
    """Test a single query and display results"""
    
    print(f"🧪 Testing Query: {query}")
    print("=" * 80)
    
    # Use HTTP request to test the running application
    url = "http://localhost:8000/api/v1/ask-enhanced"
    payload = {
        "question": query,
        "user_id": user_id,
        "session_id": session_id
    }
    
    try:
        print("📡 Sending request to:", url)
        response = requests.post(url, json=payload, timeout=30)
        res = response.json()
        
        print(f"✅ Response received (Status: {response.status_code})")
        
        # Display results
        success = res.get('success', False)
        sql = res.get('sql_query') or res.get('sql', '')
        data = res.get('data', [])
        error = res.get('error')
        
        print(f"\n📊 Results:")
        print(f"   Success: {success}")
        print(f"   Rows returned: {len(data)}")
        print(f"   Processing mode: {res.get('processing_mode', 'unknown')}")
        print(f"   Candidate source: {res.get('selected_candidate_source', 'unknown')}")
        
        if sql:
            print(f"\n📝 Generated SQL:")
            print(f"   {sql}")
        
        if error:
            print(f"\n❌ Error: {error}")
        
        if data:
            print(f"\n📋 Data Preview (first 3 rows):")
            for i, row in enumerate(data[:3]):
                print(f"   Row {i+1}: {row}")
            if len(data) > 3:
                print(f"   ... and {len(data) - 3} more rows")
        
        # Check semantic insights
        semantic_insights = res.get('semantic_insights', {})
        if semantic_insights:
            print(f"\n🧠 Semantic Insights:")
            print(f"   Confidence: {semantic_insights.get('confidence', 'N/A')}")
            print(f"   MDL Models: {len(semantic_insights.get('mdl_context', {}).get('relevant_models', []))}")
            print(f"   Business Entities: {len(semantic_insights.get('business_entities', []))}")
        
        return res
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the backend is running on http://localhost:8000")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_one_query.py \"Your query here\"")
        print("Example: python run_one_query.py \"What is the energy shortage of all regions in June 2025?\"")
        sys.exit(1)
    
    query = sys.argv[1]
    test_query(query)


if __name__ == '__main__':
    main()
