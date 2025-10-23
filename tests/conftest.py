"""
Pytest configuration and shared fixtures.

This conftest.py provides:
- Database fixtures (PostgreSQL with asyncpg)
- Environment configuration fixtures
- Common test utilities and mocks
"""

import os
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from dotenv import load_dotenv

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
    config.addinivalue_line("markers", "integration: Integration tests (database, external services)")
    config.addinivalue_line("markers", "slow: Slow tests (>1s execution time)")
    config.addinivalue_line("markers", "asyncio: Async tests using asyncio")


# ─── Event Loop Fixture ──────────────────────────────────────────────


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests (session-scoped)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


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


@pytest_asyncio.fixture
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
            print(f"\n⏱️  {name}: {result.elapsed:.4f}s")

    return timer
