import time
from datetime import datetime, timezone

import asyncpg
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from pff.shared.core.config import settings

router = APIRouter()


@router.get("/health")
def healthcheck():
    """
    Basic health check for the API (fast, no dependencies).

    Returns:
        dict: A dictionary indicating the health status of the API.
              Example: {"status": "ok"}
    """
    return {"status": "ok"}


@router.get("/health/detailed")
async def healthcheck_detailed():
    """
    Detailed health check with database and redis connectivity.

    Checks:
    - API responsiveness
    - PostgreSQL connection
    - Redis connection (if USE_REDIS=true)
    - Timestamp and uptime

    Returns:
        dict: Detailed health status of all components
    """
    start_time = time.time()
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }

    try:
        conn = await asyncpg.connect(settings.DATABASE_URL_ASYNC, timeout=5)
        await conn.execute("SELECT 1")
        await conn.close()
        health_status["checks"]["postgres"] = {
            "status": "healthy",
            "message": "Connected",
        }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["postgres"] = {"status": "unhealthy", "error": str(e)}

    if settings.USE_REDIS:
        try:
            import redis

            r = redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=5)
            r.ping()
            health_status["checks"]["redis"] = {
                "status": "healthy",
                "message": "Connected",
            }
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["checks"]["redis"] = {"status": "unhealthy", "error": str(e)}
    else:
        health_status["checks"]["redis"] = {
            "status": "disabled",
            "message": "USE_REDIS=false",
        }

    elapsed_ms = (time.time() - start_time) * 1000
    health_status["response_time_ms"] = round(elapsed_ms, 2)

    if health_status["status"] == "healthy":
        return JSONResponse(content=health_status, status_code=status.HTTP_200_OK)
    elif health_status["status"] == "degraded":
        return JSONResponse(content=health_status, status_code=status.HTTP_200_OK)
    else:
        return JSONResponse(content=health_status, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
