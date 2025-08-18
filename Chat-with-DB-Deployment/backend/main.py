"""
Main FastAPI application with modular architecture
"""

import logging
import time
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import router
from backend.config import get_settings

# Configure simple logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create FastAPI application with all middleware and routes"""

    settings = get_settings()

    app = FastAPI(
        title="Text-to-SQL RAG System",
        description="Intelligent natural language to SQL conversion system",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()

        # Generate correlation ID
        correlation_id = request.headers.get(
            "X-Correlation-ID", f"req_{int(start_time * 1000)}"
        )

        # Log request
        logger.info(
            f"Request started - {request.method} {request.url} - Correlation ID: {correlation_id}"
        )

        # Process request
        try:
            response = await call_next(request)

            # Log response
            processing_time = time.time() - start_time
            logger.info(
                f"Request completed - Status: {response.status_code} - Time: {processing_time:.3f}s - Correlation ID: {correlation_id}"
            )

            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            return response

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"Request failed - Error: {str(e)} - Time: {processing_time:.3f}s - Correlation ID: {correlation_id}",
                exc_info=True,
            )
            raise

    # Add global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "correlation_id": request.headers.get("X-Correlation-ID", "unknown"),
                "timestamp": time.time(),
            },
        )

    # Include routes
    app.include_router(router)

    return app


# Create the application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
