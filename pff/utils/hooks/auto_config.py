import os
import sys
import warnings
from pathlib import Path


def apply_permanent_configurations():
    warnings.filterwarnings(
        "ignore",
        message=".*Pipeline instance is not fitted yet.*",
        category=FutureWarning,
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
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

    env_path = Path(".env")

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
