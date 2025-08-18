# Test Scripts for Cloud-Ready SemanticEngine

This folder contains comprehensive test scripts for testing the Cloud-Ready SemanticEngine with MS SQL Server integration.

## 🚀 Available Test Scripts

### 1. `run_one_query.py` - Single Query Tester
**Purpose**: Test individual queries quickly and see detailed results
**Usage**: 
```bash
python run_one_query.py "Your query here"
```

**Example**:
```bash
python run_one_query.py "What is the energy shortage of all regions in June 2025?"
```

**Features**:
- Tests single queries
- Shows generated SQL
- Displays data preview
- Shows semantic insights
- Performance metrics

### 2. `test_all_fact_tables.py` - Comprehensive Fact Table Tests
**Purpose**: Test all major fact tables and their relationships
**Usage**:
```bash
python test_all_fact_tables.py
```

**Tests Covered**:
- ✅ FactAllIndiaDailySummary (Energy shortage, regions)
- ✅ FactStateDailyEnergy (State-level data)
- ✅ FactDailyGenerationBreakdown (Generation by source)
- ✅ FactCountryDailyExchange (International exchanges)
- ✅ FactInternationalTransmissionLinkFlow (Transmission data)
- ✅ FactTransmissionLinkFlow (Domestic transmission)
- ✅ FactTimeBlockPowerData (Time-block power data)
- ✅ FactTimeBlockGeneration (Time-block generation)

### 3. `test_exchange_and_timeblock.py` - Exchange & Timeblock Tests
**Purpose**: Focused testing of exchange and timeblock functionality
**Usage**:
```bash
python test_exchange_and_timeblock.py
```

**Tests Covered**:
- Country exchange monthly grouping
- International link flow
- Transmission link flow
- Time-block power hourly data
- Time-block generation hourly data

## 🧪 Running the Tests

### Prerequisites
1. **Backend Running**: Ensure the backend is running on `http://localhost:8000`
2. **MS SQL Server**: Azure SQL Database should be accessible
3. **Dependencies**: Install required packages

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run All Tests
```bash
# Test all fact tables
python test_all_fact_tables.py

# Test exchange and timeblock
python test_exchange_and_timeblock.py

# Test single query
python run_one_query.py "Your query here"
```

## 📊 Test Results Interpretation

### Success Indicators
- ✅ **PASS**: Query executed successfully with correct SQL
- ❌ **FAIL**: Query failed or generated incorrect SQL

### What Each Test Validates
1. **SQL Generation**: Correct table selection and joins
2. **Data Retrieval**: Actual data returned from MS SQL Server
3. **Semantic Understanding**: Business logic correctly interpreted
4. **Performance**: Response time within acceptable limits

### Expected Results
- **Success Rate**: 90%+ for production-ready deployment
- **Response Time**: < 5 seconds for complex queries
- **Data Accuracy**: Correct joins and business logic

## 🔧 Customizing Tests

### Adding New Test Cases
1. **Edit Test Files**: Add new `TestCase` objects
2. **Define Requirements**: Specify required SQL parts
3. **Set Expectations**: Define minimum rows and success criteria

### Example Test Case
```python
TestCase(
    case_id="custom_test",
    query="Your custom query here",
    required_sql_parts=[
        "from your_table",
        "join your_dimension",
        "your_required_column"
    ],
    min_rows=1,
    description="Description of what this test validates"
)
```

## 🚨 Troubleshooting

### Common Issues
1. **Connection Error**: Backend not running on port 8000
2. **Timeout**: MS SQL Server connection issues
3. **Empty Results**: Database tables not populated

### Debug Steps
1. Check backend status: `http://localhost:8000/health`
2. Verify MS SQL Server connection
3. Check firewall and network settings
4. Review backend logs for errors

## 📈 Performance Monitoring

### Metrics Tracked
- **Response Time**: Query execution duration
- **Success Rate**: Percentage of passed tests
- **Data Volume**: Number of rows returned
- **Processing Mode**: Semantic engine mode used

### Performance Targets
- **Simple Queries**: < 2 seconds
- **Complex Queries**: < 5 seconds
- **Overall Success Rate**: > 90%

## 🎯 Next Steps

After running tests:
1. **Review Results**: Identify any failing tests
2. **Analyze Failures**: Understand root causes
3. **Optimize**: Improve semantic engine performance
4. **Deploy**: Ready for cloud deployment

---

**Note**: These tests validate the Cloud-Ready SemanticEngine's ability to work with MS SQL Server and generate intelligent SQL from natural language queries.
