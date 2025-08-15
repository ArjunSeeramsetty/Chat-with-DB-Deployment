# Chat with DB - Deployment Repository

This is a **deployment-ready** version of the Chat with DB Energy Data Analysis System, containing only the essential files needed for production deployment.

## 🚀 **Quick Deployment Guide**

### **Prerequisites**
- Python 3.8+ 
- Node.js 16+ and npm
- Access to the energy database at `C:/Users/arjun/Desktop/PSPreport/power_data.db`

### **Backend Setup (FastAPI)**
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the backend server
python start_backend.py
```

The backend will start on `http://localhost:8000`

### **Frontend Setup (React)**
```bash
cd agent-ui

# Install Node.js dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

## 📁 **Repository Structure**

```
Chat-with-DB-Deployment/
├── 📄 requirements.txt              # Python dependencies
├── 📄 start_backend.py              # Backend startup script
├── 📄 pyproject.toml               # Python project configuration
├── 📄 README.md                    # Original project README
├── 📄 DEPLOYMENT_README.md         # This deployment guide
├── 📁 backend/                     # Backend application
│   ├── 📄 main.py                  # FastAPI application
│   ├── 📄 config.py                # Application settings
│   ├── 📄 __init__.py              # Python package init
│   ├── 📁 core/                    # Core business logic
│   │   ├── 📄 types.py             # Data models
│   │   ├── 📄 intent.py            # Query analysis
│   │   ├── 📄 assembler.py         # SQL assembly
│   │   ├── 📄 executor.py          # SQL execution
│   │   ├── 📄 llm_provider.py      # LLM integration
│   │   ├── 📄 schema_linker.py     # Schema linking
│   │   ├── 📄 validator.py         # SQL validation
│   │   └── 📄 candidate_ranker.py  # SQL ranking
│   ├── 📁 api/                     # API layer
│   │   ├── 📄 routes.py            # API endpoints
│   │   ├── 📄 schemas.py           # API schemas
│   │   └── 📄 __init__.py          # Python package init
│   └── 📁 services/                # Business services
│       ├── 📄 rag_service.py       # Main RAG service
│       └── 📄 __init__.py          # Python package init
└── 📁 agent-ui/                    # Frontend application
    ├── 📄 package.json             # Node.js dependencies
    ├── 📄 package-lock.json        # Locked dependency versions
    ├── 📄 README.md                # Frontend README
    ├── 📁 src/                     # React source code
    │   ├── 📄 App.js               # Main app component
    │   ├── 📄 index.js             # App entry point
    │   ├── 📁 components/          # React components
    │   ├── 📁 hooks/               # Custom React hooks
    │   └── 📁 reducers/            # State management
    └── 📁 public/                  # Static assets
```

## 🔧 **Configuration**

### **Database Configuration**
The system is configured to use the external database at:
```
C:/Users/arjun/Desktop/PSPreport/power_data.db
```

**Important**: Update the database path in `backend/config.py` if deploying to a different location.

### **Environment Variables**
Create a `.env` file in the root directory with:
```env
# Database
DATABASE_PATH=C:/Users/arjun/Desktop/PSPreport/power_data.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# LLM Configuration (if using external LLM)
LLM_PROVIDER_TYPE=ollama
LLM_MODEL=llama3.2:3b
LLM_BASE_URL=http://localhost:11434
```

## 🚀 **Production Deployment**

### **Backend Deployment**
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Configure database path** in `backend/config.py`
3. **Start with production server**:
   ```bash
   # Using gunicorn (recommended for production)
   pip install gunicorn
   gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   
   # Or using uvicorn directly
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

### **Frontend Deployment**
1. **Build production version**:
   ```bash
   cd agent-ui
   npm run build
   ```
2. **Serve static files** from the `build` directory using nginx, Apache, or any static file server

### **Docker Deployment** (Optional)
Create a `Dockerfile` in the root directory:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "start_backend.py"]
```

## 📊 **API Endpoints**

Once deployed, the following endpoints will be available:

- **Health Check**: `GET /api/v1/health`
- **Query Processing**: `POST /api/v1/ask`
- **API Documentation**: `GET /docs` (Swagger UI)
- **ReDoc Documentation**: `GET /redoc`

## 🔍 **Monitoring & Logs**

- **Application logs** are configured in `backend/main.py`
- **Health check endpoint** at `/api/v1/health`
- **Request correlation IDs** are automatically generated for tracking

## 🛠️ **Troubleshooting**

### **Common Issues**

1. **Database Connection Error**
   - Verify database path in `backend/config.py`
   - Ensure database file exists and is accessible

2. **Port Already in Use**
   - Change port in `backend/config.py` or use environment variable `API_PORT`

3. **Frontend Build Errors**
   - Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`

4. **LLM Connection Issues**
   - Check Ollama service is running if using local LLM
   - Verify API keys if using external LLM services

### **Logs**
Check the console output for detailed error messages and request logs.

## 📞 **Support**

For deployment issues, check:
1. **Backend logs** in the console
2. **Frontend console** in browser developer tools
3. **Network tab** for API request/response details

## 🎯 **Next Steps**

After successful deployment:
1. **Test all API endpoints** using the Swagger UI at `/docs`
2. **Verify frontend functionality** with sample queries
3. **Configure monitoring** and logging as needed
4. **Set up CI/CD pipeline** for automated deployments

---

**Status**: ✅ **Deployment Ready** - This repository contains all essential files for production deployment.

**Last Updated**: August 2025
