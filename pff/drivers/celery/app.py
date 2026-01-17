from __future__ import annotations

from celery import Celery

from pff import settings

celery_config = {
    key.lower().replace("celery_", ""): value
    for key, value in settings.model_dump().items()
    if key.startswith("CELERY_")
}

celery_app = Celery("pff")
celery_app.conf.update(celery_config)
celery_app.autodiscover_tasks(settings.CELERY_TASK_AUTODISCOVER)


@celery_app.task(name="pff.quick_ping")
def quick_ping() -> str:
    """Responde rapidamente com 'pong'."""
    return "pong"
