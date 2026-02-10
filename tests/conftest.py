"""
Pytest configuration and shared fixtures.

This conftest.py provides:
- Database fixtures (PostgreSQL with asyncpg)
- Environment configuration fixtures
- Common test utilities and mocks
"""

import asyncio
import logging
import os
import os as _os
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio

# Set Numba env vars BEFORE importing torch or anything else
# We allow CUDA visibility by default if present
if "CUDA_VISIBLE_DEVICES" not in _os.environ:
    # Do not force -1 if the user hasn't set it
    pass

# We increase Numba threads for tests, but keep it deterministic if needed
# By default, use all cores or a safe subset (e.g., 10)
# Force set to 10 to avoid "currently have X, trying to set Y" errors if env differs
recommended_threads = "10"
# Always overwrite to ensure consistency across all tests
_os.environ["NUMBA_NUM_THREADS"] = recommended_threads
_os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")  # Safe default for tests

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from loguru import logger  # noqa: E402

from pff.shared.system.resource_manager import configure_numba_threads  # noqa: E402

# Configure Numba once for the whole session
try:
    configure_numba_threads()
except Exception:
    pass

# Load test environment variables
TEST_ENV = Path(__file__).parent / ".env.test"
if TEST_ENV.exists():
    load_dotenv(TEST_ENV)
else:
    load_dotenv()  # Fallback to root .env


# ─── Pytest Configuration ────────────────────────────────────────────


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external dependencies)")
    config.addinivalue_line(
        "markers", "integration: Integration tests (database, external services)"
    )
    config.addinivalue_line("markers", "slow: Slow tests (>1s execution time)")
    config.addinivalue_line("markers", "asyncio: Async tests using asyncio")


# ─── Event Loop Fixture ──────────────────────────────────────────────

# pytest-asyncio >=0.23 uses 'loop_scope' instead of custom event_loop fixture.
# Session-scoped event loop is configured via pytest.ini with:
#   asyncio_default_fixture_loop_scope = "session"


@pytest.fixture(scope="session")
def event_loop_policy():
    """Return the default event loop policy for session-scoped async tests."""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="function")
def new_event_loop():
    """Provide a fresh event loop for tests that need isolation."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ─── Loguru Caplog Fixture ───────────────────────────────────────────


@pytest.fixture(autouse=True)
def caplog_for_loguru(caplog):
    """
    Make Loguru logs compatible with pytest's caplog fixture using a safe sink.
    """

    def safe_forward(message):
        # We use standard logging to emit the record so caplog can see it
        # We must avoid using the logger itself to avoid recursion if caplog is involved
        msg = message.record["message"]
        level = message.record["level"].no
        name = message.record["name"]

        # Manually create and handle the record to the root logger or specific logger
        logger_obj = logging.getLogger(name)
        record = logger_obj.makeRecord(name, level, "(unknown file)", 0, msg, None, None)

        # Pytest intercepts logs by adding a LogCaptureHandler to the loggers it tracks
        from _pytest.logging import LogCaptureHandler

        for handler in logging.root.handlers + logger_obj.handlers:
            if isinstance(handler, LogCaptureHandler):
                handler.emit(record)

    handler_id = logger.add(safe_forward, format="{message}", level=0)
    yield caplog
    try:
        logger.remove(handler_id)
    except ValueError:
        # Handler might have been removed already by a test that reloads/cleans the logger
        pass


# ─── Environment Fixtures ────────────────────────────────────────────


@pytest.fixture(scope="session")
def test_root_dir() -> Path:
    """Return the root directory of the test suite."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def project_root_dir() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).parents[1]


@pytest.fixture
def temp_env_vars(monkeypatch):
    """Provide temporary environment variables for testing.

    Usage:
        def test_something(temp_env_vars):
            temp_env_vars["SECRET_KEY"] = "test-secret"
            # SECRET_KEY is now set to "test-secret" for this test only
    """

    class TempEnv:
        def __init__(self):
            self._env_vars: dict[str, str] = {}

        def __setitem__(self, key: str, value: str):
            self._env_vars[key] = value
            monkeypatch.setenv(key, value)

        def __getitem__(self, key: str) -> str:
            return self._env_vars[key]

    return TempEnv()


# ─── Database Fixtures (PostgreSQL) ──────────────────────────────────


@pytest_asyncio.fixture(loop_scope="function")
async def db_connection() -> AsyncGenerator:
    """Provide async database connection for tests.

    NOTE: Requires asyncpg and PostgreSQL to be running.
    This fixture creates a connection to the test database and rolls back
    all changes after each test.
    """
    try:
        import asyncpg
    except ImportError:
        pytest.skip("asyncpg not installed")

    database_url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("No database URL configured")

    conn = await asyncpg.connect(database_url)

    # Start transaction for test isolation
    transaction = conn.transaction()
    await transaction.start()

    try:
        yield conn
    finally:
        # Rollback transaction (cleanup)
        await transaction.rollback()
        await conn.close()


# ─── Mock Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def mock_redis(monkeypatch):
    """Mock Redis client for testing without real Redis instance."""

    class MockRedis:
        def __init__(self, *args, **kwargs):
            self._data = {}

        def get(self, key):
            return self._data.get(key)

        def set(self, key, value, ex=None):
            self._data[key] = value
            return True

        def delete(self, *keys):
            for key in keys:
                self._data.pop(key, None)
            return len(keys)

        def exists(self, *keys):
            return sum(1 for k in keys if k in self._data)

        def flushdb(self):
            self._data.clear()
            return True

    import redis

    monkeypatch.setattr(redis, "Redis", MockRedis)
    return MockRedis()


# ─── DSLFM / PC Fixtures ─────────────────────────────────────────────


@pytest.fixture(scope="session")
def synthetic_rules_path(test_root_dir: Path) -> Path:
    """Return path to the synthetic rule set used in DSLFM/PC tests."""
    return test_root_dir / "fixtures" / "synthetic_rules.tsv"


@pytest.fixture
def synthetic_kg_triples() -> torch.Tensor:
    """Provide a small batch of synthetic triples for DSLFM tests."""
    return torch.tensor(
        [
            [0, 0, 1],
            [2, 1, 3],
            [4, 2, 5],
            [1, 3, 0],
        ],
        dtype=torch.long,
    )


# ─── Cache Cleanup Fixtures ──────────────────────────────────────────


@pytest.fixture(autouse=True)
def cleanup_disk_cache():
    """Clear disk cache before each test to prevent interference.

    Only removes test-sensitive caches (aggregated_rules, hpo_config).
    Preserves infrastructure caches (triton, ingest) that are expensive
    to rebuild and not test-sensitive.
    """
    import shutil

    try:
        from pff.shared.core.config import settings

        cache_base = settings.OUTPUTS_DIR / ".cache"
        cache_dirs = [
            cache_base / "aggregated_rules",
            cache_base / "hpo_config",
        ]
    except ImportError:
        cache_dirs = []

    for cache_dir in cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

    yield

    for cache_dir in cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def cleanup_aiofile_contexts():
    """Clear aiofile TLS contexts to prevent event loop issues between tests."""
    yield
    # Clean up aiofile contexts after each test
    try:
        from pff.shared.core.file_manager import _MSGSPEC_TLS

        if hasattr(_MSGSPEC_TLS, "aio_contexts"):
            _MSGSPEC_TLS.aio_contexts = {}
    except (ImportError, AttributeError):
        pass


# ─── Performance Fixtures ────────────────────────────────────────────


@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer for performance testing.

    Usage:
        def test_performance(benchmark_timer):
            with benchmark_timer("operation_name") as timer:
                # code to benchmark
                pass
            assert timer.elapsed < 1.0  # Should take less than 1 second
    """
    import time
    from contextlib import contextmanager

    class Timer:
        def __init__(self, name: str):
            self.name = name
            self.elapsed: float = 0.0

    @contextmanager
    def timer(name: str = "operation"):
        start = time.perf_counter()
        result = Timer(name)
        try:
            yield result
        finally:
            result.elapsed = time.perf_counter() - start
            print(f"\n  {name}: {result.elapsed:.4f}s")

    return timer
