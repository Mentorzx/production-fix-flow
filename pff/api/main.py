from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from pff.config import settings
from pff.utils import logger

from .routers import executions, health, sequences, websocket
from . import auth

"""
PFF API main application module.

Configures and initializes the FastAPI application with all
routers, middleware, and event handlers.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI app.
    Handles startup and shutdown logic.
    """
    # Startup logic
    logger.info("Iniciando PFF API v1.1.0")

    # Start WebSocket Redis listener
    from .routers.websocket import start_redis_listener

    await start_redis_listener()

    # Initialize cache warmup if configured
    if hasattr(settings, "CACHE_WARMUP") and settings.CACHE_WARMUP:
        logger.info("Aquecendo cache...")
        # Add cache warmup logic here

    logger.success("PFF API iniciada com sucesso")

    yield  # Application runs here

    # Shutdown logic
    logger.info("Encerrando PFF API...")

    try:
        from .routers.websocket import stop_redis_listener
        await stop_redis_listener()
    except Exception as e:
        logger.warning(f"Redis listener cleanup error (non-critical): {e}")

    try:
        from pff.config import rds
        if rds:
            rds.close()
            logger.debug("Redis connection closed successfully")
    except Exception as e:
        logger.warning(f"Redis cleanup error (non-critical): {e}")

    logger.success("PFF API encerrada")


# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="PFF API",
    version="1.1.0",
    description="Production Fix Flow API - Backend for triggering sequences and querying results with AI validation",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,  # ← Incluindo o lifespan
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(settings, "CORS_ORIGINS", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(health.router)
app.include_router(sequences.router, prefix="/sequences", tags=["sequences"])
app.include_router(executions.router, prefix="/executions", tags=["executions"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])


@app.get("/")
@limiter.limit("100/minute")
async def root(request: Request):
    """
    Root endpoint with API information.

    Returns basic API info and available endpoints.
    Rate limit: 100 requests per minute.
    """
    return {
        "message": "PFF API is running",
        "version": "1.1.0",
        "environment": (
            settings.ENVIRONMENT if hasattr(settings, "ENVIRONMENT") else "production"
        ),
        "endpoints": {
            "auth": "/api/v1/auth",
            "health": "/health",
            "sequences": "/sequences",
            "executions": "/executions",
            "websocket": "/ws/{client_id}",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


@app.get("/info")
@limiter.limit("100/minute")
async def info(request: Request):
    """
    Detailed API information endpoint.

    Returns configuration info and service status.
    Rate limit: 100 requests per minute.
    """
    return {
        "api": {
            "name": "PFF API",
            "version": "1.1.0",
            "description": "Production Fix Flow API",
        },
        "services": {
            "redis": {
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT,
                "databases": {
                    "executions": getattr(settings, "REDIS_DB_EXECUTIONS", 5),
                    "pubsub": getattr(settings, "REDIS_DB_PUBSUB", 2),
                    "cache": getattr(settings, "REDIS_DB_CACHE", 0),
                },
            },
            "celery": {
                "broker": settings.CELERY_BROKER_URL,
                "backend": settings.CELERY_RESULT_BACKEND,
            },
        },
        "paths": {
            "output": str(settings.OUTPUTS_DIR),
            "logs": str(settings.LOGS_DIR),
            "config": str(settings.CONFIG_DIR),
        },
    }
