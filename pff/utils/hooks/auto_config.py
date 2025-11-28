import os
import sys
import warnings
from pathlib import Path

# Get project root - this module is in pff/utils/hooks/
_PROJECT_ROOT = Path(__file__).parents[3]


def apply_permanent_configurations():
    # Only filter sklearn Pipeline FutureWarning - this is noisy during normal operation
    warnings.filterwarnings(
        "ignore",
        message=".*Pipeline instance is not fitted yet.*",
        category=FutureWarning,
    )
    # Filter distributed/dask UserWarnings that are not actionable
    warnings.filterwarnings("ignore", category=UserWarning, module="distributed")

    if sys.platform == "win32":
        try:
            import asyncio

            if not isinstance(
                asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy
            ):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except (ImportError, RuntimeError):
            pass  # not critical

    env_path = _PROJECT_ROOT / ".env"

    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key not in os.environ:
                        os.environ[key] = value
        except Exception:
            pass


apply_permanent_configurations()
