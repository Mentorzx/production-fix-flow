"""
Tests for Configuration Management (pff/config.py)

Tests cover:
- Environment variable loading
- PostgreSQL connection URL generation
- Settings validation
"""

import os
import pytest
from pff.config import Settings


@pytest.mark.unit
class TestSettings:
    """Test Settings configuration class."""

    def test_default_settings_load(self):
        """Test that Settings loads with default values."""
        settings = Settings()

        # Verify core paths exist
        assert settings.ROOT_DIR.exists()
        assert settings.DATA_DIR.name == "data"
        assert settings.LOGS_DIR.name == "logs"

        # Verify Redis defaults
        assert settings.REDIS_HOST == "localhost"
        assert settings.REDIS_PORT == 6379

    def test_postgres_url_generation(self):
        """Test PostgreSQL URL generation from settings."""
        settings = Settings(
            POSTGRES_HOST="testhost",
            POSTGRES_PORT=5433,
            POSTGRES_DB="testdb",
            POSTGRES_USER="testuser",
            POSTGRES_PASSWORD="testpass123"
        )

        expected_url = "postgresql://testuser:testpass123@testhost:5433/testdb"
        assert settings.DATABASE_URL == expected_url

        expected_async_url = "postgresql+asyncpg://testuser:testpass123@testhost:5433/testdb"
        assert settings.DATABASE_URL_ASYNC == expected_async_url

    def test_postgres_url_from_env(self, monkeypatch):
        """Test PostgreSQL URL generation from environment variables."""
        monkeypatch.setenv("POSTGRES_HOST", "envhost")
        monkeypatch.setenv("POSTGRES_PORT", "5555")
        monkeypatch.setenv("POSTGRES_DB", "envdb")
        monkeypatch.setenv("POSTGRES_USER", "envuser")
        monkeypatch.setenv("POSTGRES_PASSWORD", "envpass")

        # Create new Settings instance to pick up env vars
        settings = Settings()

        assert "envhost" in settings.DATABASE_URL
        assert "5555" in settings.DATABASE_URL
        assert "envdb" in settings.DATABASE_URL
        assert "envuser" in settings.DATABASE_URL
        assert "envpass" in settings.DATABASE_URL

    def test_celery_broker_url(self):
        """Test Celery broker URL generation."""
        settings = Settings(REDIS_HOST="celeryhost", REDIS_PORT=6380)

        assert settings.CELERY_BROKER_URL == "redis://celeryhost:6380/0"
        assert settings.CELERY_RESULT_BACKEND == "redis://celeryhost:6380/1"

    def test_security_defaults(self):
        """Test security-related settings have defaults."""
        settings = Settings()

        # These should have defaults (will be overridden by .env in production)
        assert settings.SECRET_KEY is not None
        assert settings.API_KEY is not None
        assert settings.ACCESS_TOKEN_EXPIRE_MINUTES > 0

    def test_api_version(self):
        """Test API version is set."""
        settings = Settings()
        assert settings.API_VERSION is not None
        assert isinstance(settings.API_VERSION, str)


@pytest.mark.unit
class TestCeleryConfig:
    """Test Celery configuration."""

    def test_celery_accept_content_coercion_from_list(self):
        """Test CELERY_ACCEPT_CONTENT handles list input."""
        settings = Settings(CELERY_ACCEPT_CONTENT=["json", "pickle"])
        assert settings.CELERY_ACCEPT_CONTENT == ["json", "pickle"]

    def test_celery_accept_content_coercion_from_json_string(self):
        """Test CELERY_ACCEPT_CONTENT handles JSON string input."""
        settings = Settings(CELERY_ACCEPT_CONTENT='["json", "msgpack"]')
        assert settings.CELERY_ACCEPT_CONTENT == ["json", "msgpack"]

    def test_celery_accept_content_coercion_from_csv_string(self):
        """Test CELERY_ACCEPT_CONTENT handles comma-separated string input."""
        settings = Settings(CELERY_ACCEPT_CONTENT="json, pickle, msgpack")
        assert "json" in settings.CELERY_ACCEPT_CONTENT
        assert "pickle" in settings.CELERY_ACCEPT_CONTENT
        assert "msgpack" in settings.CELERY_ACCEPT_CONTENT

    def test_celery_task_queues_defined(self):
        """Test Celery task queues are properly defined."""
        settings = Settings()

        assert len(settings.CELERY_TASK_QUEUES) >= 3
        queue_names = [q.name for q in settings.CELERY_TASK_QUEUES]
        assert "default" in queue_names
        assert "high" in queue_names
        assert "low" in queue_names
