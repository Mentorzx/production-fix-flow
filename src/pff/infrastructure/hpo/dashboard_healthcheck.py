"""Dashboard healthcheck utilities for HPO infrastructure."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from pff.shared.acceleration.asyncio_runner import run_coroutine_sync
from pff.shared.clients.http_client import check_http_status


def _status_url(bind: str, port: int) -> str:
    return f"http://{bind}:{port}/api/status"


def _check_once(url: str, *, request_timeout_s: float = 1.0) -> bool:
    """Execute check once.



    Args:

        url: Input value used by this callable.

        request_timeout_s: Optional input value.



    Returns:

        Return value produced by the callable.

    """

    try:
        return bool(
            run_coroutine_sync(
                check_http_status(url, timeout_s=request_timeout_s),
                timeout_s=request_timeout_s + 0.5,
            )
        )
    except Exception:
        return False


def is_dashboard_healthy(*, bind: str, port: int, timeout_s: float) -> bool:
    """Poll dashboard status endpoint until timeout."""
    url = _status_url(bind, port)
    deadline = time.time() + max(timeout_s, 0.1)
    while time.time() < deadline:
        if _check_once(url):
            return True
        time.sleep(0.25)
    return False


def start_dashboard_healthcheck_thread(
    *,
    bind: str,
    port: int,
    timeout_s: float,
    on_success: Callable[[], None] | None = None,
    on_timeout: Callable[[], None] | None = None,
) -> None:
    """Run dashboard healthcheck in a daemon thread."""

    def _worker() -> None:
        """Execute worker."""

        if is_dashboard_healthy(bind=bind, port=port, timeout_s=timeout_s):
            if on_success:
                on_success()
            return
        if on_timeout:
            on_timeout()

    thread = threading.Thread(target=_worker, daemon=True, name="hpo-healthcheck")
    thread.start()
