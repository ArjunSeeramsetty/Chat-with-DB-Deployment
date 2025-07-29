# Chat with DB - Energy Data Analysis System

A comprehensive natural language query system for Indian energy data analysis with modular architecture.

## 🏗️ **Project Structure**

```
Chat with DB new/
├── 📄 start_backend.py                    # Backend startup script
├── 📄 requirements.txt                     # Python dependencies
├── 📄 README.md                           # This file
├── 📁 config/                             # Configuration
│   └── 📄 business_rules.yaml             # Business rules
├── 📁 backend/                            # Backend application
│   ├── 📄 main.py                         # FastAPI application
│   ├── 📄 config.py                       # Application settings
│   ├── 📁 core/                           # Core business logic
│   │   ├── 📄 types.py                    # Data models
│   │   ├── 📄 intent.py                   # Query analysis
│   │   ├── 📄 assembler.py                # SQL assembly
│   │   ├── 📄 validator.py                # SQL validation
│   │   ├── 📄 executor.py                 # SQL execution
│   │   ├── 📄 llm_provider.py            # LLM integration
│   │   ├── 📄 schema_linker.py           # Schema linking
│   │   └── 📄 candidate_ranker.py        # SQL ranking
│   ├── 📁 api/                            # API layer
│   │   ├── 📄 routes.py                   # API endpoints
│   │   ├── 📄 deps.py                     # Dependencies
│   │   └── 📄 schemas.py                  # API schemas
│   └── 📁 services/                       # Business services
│       └── 📄 rag_service.py              # Main RAG service
└── 📁 agent-ui/                          # Frontend application
    ├── 📄 package.json                    # Node.js dependencies
    ├── 📁 public/                         # Static assets
    │   ├── 📄 index.html                  # Main HTML
    │   ├── 📄 manifest.json               # PWA manifest
    │   └── 📄 favicon.ico                 # Favicon
    └── 📁 src/                            # React source
        ├── 📄 App.modular.js              # Main app component
        ├── 📁 components/                  # React components
        │   ├── 📄 QueryInput.js           # Query input
        │   ├── 📄 ClarificationInterface.js # Clarification UI
        │   ├── 📄 DataVisualization.js    # Chart/table display
        │   └── 📄 ChartBuilder.js         # Chart configuration
        ├── 📁 hooks/                      # Custom hooks
        │   ├── 📄 useQueryService.js      # API service
        │   └── 📄 useClarification.js    # Clarification logic
        └── 📁 reducers/                   # State management
            └── 📄 appReducer.js           # Main reducer
```

## 🚀 **Quick Start**

### **Backend Setup**
```bash
cd "Chat with DB new"
pip install -r requirements.txt
python start_backend.py
```

### **Frontend Setup**
```bash
cd "Chat with DB new/agent-ui"
npm install
npm start
```

## 🔧 **Key Features**

### **Backend Features:**
- ✅ **Direct SQL Generation**: Bypasses complex templates for clear queries
- ✅ **Intelligent Query Analysis**: Extracts intent, entities, and metrics
- ✅ **LLM Integration**: Uses Ollama for clarification questions
- ✅ **SQL Validation**: Ensures generated SQL is safe and correct
- ✅ **Modular Architecture**: Clean separation of concerns

### **Frontend Features:**
- ✅ **Modular React Components**: Clean, maintainable code
- ✅ **Interactive Charts**: Bar, line, pie charts with ChartBuilder
- ✅ **Clarification Interface**: User-friendly question/answer flow
- ✅ **Real-time Data**: Live updates from backend
- ✅ **Responsive Design**: Works on all devices

## 📊 **Database Configuration**

The system uses the database from:
- **Database Path**: `C:/Users/arjun/Desktop/PSPreport/power_data.db`
- **Configuration**: Set in `backend/config.py`
- **Access**: Direct file path access (no local copy needed)

### **Database Schema:**
- **FactStateDailyEnergy**: State-level daily energy data
- **FactAllIndiaDailySummary**: Regional energy summaries
- **DimStates**: State dimension table
- **DimRegions**: Region dimension table
- **DimDates**: Date dimension table

## 🎯 **Query Examples**

### **Working Queries:**
- "What is the total energy consumption of all states in 2024?"
- "Show me the maximum energy demand in 2025"
- "What is the outage in 2024?"
- "Which state has the highest energy consumption?"

### **Features:**
- ✅ **Direct SQL Generation** for clear queries
- ✅ **Smart Clarification** for ambiguous queries
- ✅ **Chart Visualization** with multiple chart types
- ✅ **Table View** with sortable columns
- ✅ **Export Capabilities** for data analysis

## 🔍 **System Architecture**

### **Backend Flow:**
1. **Query Analysis** → Intent detection and entity extraction
2. **Confidence Calculation** → Determines if clarification needed
3. **Direct SQL Generation** → For clear, high-confidence queries
4. **LLM Clarification** → For ambiguous queries
5. **SQL Execution** → Safe database queries
6. **Data Processing** → Format for frontend consumption

### **Frontend Flow:**
1. **Query Input** → Natural language interface
2. **Clarification Interface** → Interactive Q&A if needed
3. **Data Visualization** → Charts and tables
4. **Chart Builder** → Custom chart configuration
5. **State Management** → Redux-like state management

## 🛠️ **Technologies Used**

### **Backend:**
- **FastAPI**: Modern Python web framework
- **SQLite**: Lightweight database (external path)
- **Ollama**: Local LLM for clarifications
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

### **Frontend:**
- **React**: Modern JavaScript framework
- **Recharts**: Chart library
- **Material-UI**: Component library
- **Axios**: HTTP client
- **React Hooks**: State management

## 📈 **Performance**

- **Query Response Time**: ~2-3 seconds for direct queries
- **Clarification Flow**: ~5-10 seconds with LLM
- **Chart Rendering**: Real-time with smooth animations
- **Database Access**: Direct file path access

## 🎉 **Success Metrics**

The system successfully:
- ✅ **Generates SQL directly** for clear queries
- ✅ **Handles clarifications** intelligently
- ✅ **Visualizes data** with multiple chart types
- ✅ **Provides real-time** query processing
- ✅ **Maintains modular** architecture
- ✅ **Uses external database** path efficiently

## 🔮 **Future Enhancements**

- **Advanced Analytics**: Trend analysis and forecasting
- **Multi-language Support**: Hindi and other Indian languages
- **Mobile App**: React Native version
- **Advanced Charts**: 3D visualizations and heatmaps
- **Export Features**: PDF reports and Excel exports

---

**Status**: ✅ **Production Ready** - System is fully functional with direct SQL generation and intelligent clarification handling.

**Database**: Uses external database from `C:/Users/arjun/Desktop/PSPreport/power_data.db` 