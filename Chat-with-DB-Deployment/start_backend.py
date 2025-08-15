#!/usr/bin/env python3
"""
Backend startup script for the Text-to-SQL RAG System
"""

import uvicorn
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Start the FastAPI backend server"""
    
    # Set default configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    workers = int(os.getenv("API_WORKERS", "1"))
    
    print(f"🚀 Starting Text-to-SQL RAG System Backend")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print(f"👥 Workers: {workers}")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/api/v1/health")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "backend.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 