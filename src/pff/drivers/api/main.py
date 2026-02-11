from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from pff.shared import logger
from pff.shared.core.config import settings

from . import auth
from .routers import executions, health, sequences, websocket

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

    logger.info("Iniciando PFF API v1.1.0")
    from pff.shared.core.cache import apply_cache_settings_from_config

    apply_cache_settings_from_config()

    from .routers.websocket import start_redis_listener

    await start_redis_listener()

    if hasattr(settings, "CACHE_WARMUP") and settings.CACHE_WARMUP:
        logger.info("Aquecendo cache...")

    logger.success("PFF API iniciada com sucesso")

    yield

    logger.info("Encerrando PFF API...")

    try:
        from .routers.websocket import stop_redis_listener

        await stop_redis_listener()
    except Exception as e:
        logger.warning(f"Redis listener cleanup error (non-critical): {e}")

    try:
        from pff.shared.core.config import get_redis_client

        try:
            client = get_redis_client(db=5, decode_responses=True)
            client.close()
        except Exception:
            pass
            logger.debug("Redis connection closed successfully")
    except Exception as e:
        logger.warning(f"Redis cleanup error (non-critical): {e}")

    logger.success("PFF API encerrada")


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="PFF API",
    version="1.1.0",
    description="Production Fix Flow API - Backend for triggering sequences and querying results with AI validation",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


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
