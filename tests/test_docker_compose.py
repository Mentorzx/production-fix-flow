"""
Docker Compose Tests - Sprint 10

Tests docker-compose.yml configuration:
- Service definitions
- Dependencies
- Health checks
- Volumes and networks
- Environment variables
"""

import subprocess
from pathlib import Path

import pytest
import yaml


class TestDockerComposeFile:
    """Test docker-compose.yml file structure."""

    def test_docker_compose_exists(self):
        """Verify docker-compose.yml exists."""
        compose_file = Path("docker-compose.yml")
        assert compose_file.exists(), "docker-compose.yml not found"

    def test_docker_compose_valid_yaml(self):
        """Verify docker-compose.yml is valid YAML."""
        with open("docker-compose.yml") as f:
            config = yaml.safe_load(f)

        assert config is not None, "docker-compose.yml is empty"
        assert "services" in config, "Missing 'services' key"
        assert "version" in config, "Missing 'version' key"


class TestDockerComposeServices:
    """Test service definitions in docker-compose.yml."""

    @pytest.fixture
    def compose_config(self):
        """Load docker-compose.yml configuration."""
        with open("docker-compose.yml") as f:
            return yaml.safe_load(f)

    def test_has_postgres_service(self, compose_config):
        """Verify PostgreSQL service is defined."""
        services = compose_config["services"]
        assert "postgres" in services, "Missing postgres service"

        postgres = services["postgres"]
        assert "pgvector/pgvector:pg16" in postgres["image"], "Wrong PostgreSQL image"

    def test_has_redis_service(self, compose_config):
        """Verify Redis service is defined."""
        services = compose_config["services"]
        assert "redis" in services, "Missing redis service"

        redis = services["redis"]
        assert "redis:7" in redis["image"], "Wrong Redis image"

    def test_has_api_service(self, compose_config):
        """Verify API service is defined."""
        services = compose_config["services"]
        assert "api" in services, "Missing api service"

        api = services["api"]
        assert "build" in api, "API service missing build config"
        assert "context" in api["build"], "API build missing context"

    def test_has_celery_worker_service(self, compose_config):
        """Verify Celery worker service is defined."""
        services = compose_config["services"]
        assert "celery-worker" in services, "Missing celery-worker service"

        celery = services["celery-worker"]
        assert "celery" in celery["command"], "Celery command not found"

    def test_services_have_healthchecks(self, compose_config):
        """Verify critical services have health checks."""
        services = compose_config["services"]

        # PostgreSQL healthcheck
        assert "healthcheck" in services["postgres"], "Postgres missing healthcheck"
        assert "pg_isready" in services["postgres"]["healthcheck"]["test"][1], \
            "Postgres healthcheck not using pg_isready"

        # Redis healthcheck
        assert "healthcheck" in services["redis"], "Redis missing healthcheck"

        # API healthcheck
        assert "healthcheck" in services["api"], "API missing healthcheck"

    def test_api_depends_on_postgres_and_redis(self, compose_config):
        """Verify API service depends on postgres and redis."""
        services = compose_config["services"]
        api = services["api"]

        assert "depends_on" in api, "API missing depends_on"

        depends = api["depends_on"]
        assert "postgres" in depends, "API doesn't depend on postgres"
        assert "redis" in depends, "API doesn't depend on redis"

        # Check for service_healthy condition
        assert depends["postgres"]["condition"] == "service_healthy", \
            "Postgres dependency not using service_healthy"

    def test_services_expose_correct_ports(self, compose_config):
        """Verify services expose correct ports."""
        services = compose_config["services"]

        # PostgreSQL
        assert "5432:5432" in services["postgres"]["ports"], \
            "Postgres not exposing port 5432"

        # Redis
        assert "6379:6379" in services["redis"]["ports"], \
            "Redis not exposing port 6379"

        # API
        assert "8000:8000" in services["api"]["ports"], \
            "API not exposing port 8000"


class TestDockerComposeVolumes:
    """Test volume configurations."""

    @pytest.fixture
    def compose_config(self):
        """Load docker-compose.yml configuration."""
        with open("docker-compose.yml") as f:
            return yaml.safe_load(f)

    def test_has_named_volumes(self, compose_config):
        """Verify named volumes are defined."""
        assert "volumes" in compose_config, "Missing volumes section"

        volumes = compose_config["volumes"]
        assert "postgres_data" in volumes, "Missing postgres_data volume"
        assert "redis_data" in volumes, "Missing redis_data volume"

    def test_postgres_uses_volume(self, compose_config):
        """Verify PostgreSQL uses persistent volume."""
        postgres = compose_config["services"]["postgres"]

        assert "volumes" in postgres, "Postgres missing volumes"
        volumes = postgres["volumes"]

        # Check for postgres_data volume mount
        postgres_data_found = any("postgres_data:" in v for v in volumes)
        assert postgres_data_found, "Postgres not using postgres_data volume"

    def test_api_has_log_volume(self, compose_config):
        """Verify API mounts logs directory."""
        api = compose_config["services"]["api"]

        assert "volumes" in api, "API missing volumes"
        volumes = api["volumes"]

        # Check for logs mount
        logs_found = any("./logs:/app/logs" in v for v in volumes)
        assert logs_found, "API not mounting logs directory"


class TestDockerComposeEnvironment:
    """Test environment variable configurations."""

    @pytest.fixture
    def compose_config(self):
        """Load docker-compose.yml configuration."""
        with open("docker-compose.yml") as f:
            return yaml.safe_load(f)

    def test_postgres_environment_vars(self, compose_config):
        """Verify PostgreSQL has required environment variables."""
        postgres = compose_config["services"]["postgres"]

        assert "environment" in postgres, "Postgres missing environment"
        env = postgres["environment"]

        assert "POSTGRES_USER" in env, "Missing POSTGRES_USER"
        assert "POSTGRES_PASSWORD" in env, "Missing POSTGRES_PASSWORD"
        assert "POSTGRES_DB" in env, "Missing POSTGRES_DB"

    def test_api_environment_vars(self, compose_config):
        """Verify API has required environment variables."""
        api = compose_config["services"]["api"]

        assert "environment" in api, "API missing environment"
        env = api["environment"]

        required_vars = [
            "DATABASE_URL",
            "DATABASE_URL_ASYNC",
            "SECRET_KEY",
            "API_KEY",
            "PFF_ENV"
        ]

        for var in required_vars:
            assert var in env, f"API missing {var} environment variable"

    def test_api_production_environment(self, compose_config):
        """Verify API is configured for production."""
        api = compose_config["services"]["api"]
        env = api["environment"]

        assert env["PFF_ENV"] == "production", "API not set to production environment"


class TestDockerComposeNetworks:
    """Test network configurations."""

    @pytest.fixture
    def compose_config(self):
        """Load docker-compose.yml configuration."""
        with open("docker-compose.yml") as f:
            return yaml.safe_load(f)

    def test_has_custom_network(self, compose_config):
        """Verify custom network is defined."""
        assert "networks" in compose_config, "Missing networks section"

        networks = compose_config["networks"]
        assert "pff-network" in networks, "Missing pff-network"

    def test_services_use_custom_network(self, compose_config):
        """Verify services use custom network."""
        services = compose_config["services"]

        for service_name in ["postgres", "redis", "api", "celery-worker"]:
            service = services[service_name]
            assert "networks" in service, f"{service_name} missing networks"
            assert "pff-network" in service["networks"], \
                f"{service_name} not using pff-network"


class TestDockerComposeResourceLimits:
    """Test resource limit configurations."""

    @pytest.fixture
    def compose_config(self):
        """Load docker-compose.yml configuration."""
        with open("docker-compose.yml") as f:
            return yaml.safe_load(f)

    def test_api_has_resource_limits(self, compose_config):
        """Verify API service has resource limits."""
        api = compose_config["services"]["api"]

        assert "deploy" in api, "API missing deploy configuration"
        assert "resources" in api["deploy"], "API missing resources"
        assert "limits" in api["deploy"]["resources"], "API missing resource limits"

        limits = api["deploy"]["resources"]["limits"]
        assert "cpus" in limits, "API missing CPU limit"
        assert "memory" in limits, "API missing memory limit"


class TestDockerComposeValidation:
    """Test docker-compose validation (requires docker-compose installed)."""

    @pytest.mark.slow
    @pytest.mark.skipif(
        subprocess.run(["which", "docker"], capture_output=True).returncode != 0,
        reason="Docker not installed"
    )
    def test_docker_compose_config_valid(self):
        """Test docker-compose config is valid."""
        result = subprocess.run(
            ["docker", "compose", "config"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"docker-compose config invalid:\n{result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
