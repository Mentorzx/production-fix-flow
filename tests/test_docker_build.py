"""
Docker Build Tests - Sprint 10

Tests Dockerfile configuration and build process:
- Multi-stage build structure
- Security (non-root user)
- Image size optimization
- Required files present
"""

import subprocess
from pathlib import Path

import pytest


class TestDockerfile:
    """Test Dockerfile configuration and structure."""

    def test_dockerfile_exists(self):
        """Verify Dockerfile exists in project root."""
        dockerfile = Path("Dockerfile")
        assert dockerfile.exists(), "Dockerfile not found in project root"
        assert dockerfile.is_file(), "Dockerfile is not a file"

    def test_dockerfile_has_multi_stage(self):
        """Verify Dockerfile uses multi-stage build."""
        dockerfile = Path("Dockerfile").read_text()

        # Check for builder stage
        assert "FROM python:3.13-slim AS builder" in dockerfile, "Missing builder stage"

        # Check for runtime stage
        assert "FROM python:3.13-slim AS runtime" in dockerfile, "Missing runtime stage"

    def test_dockerfile_uses_poetry(self):
        """Verify Dockerfile uses Poetry for dependency management."""
        dockerfile = Path("Dockerfile").read_text()

        assert "POETRY_VERSION" in dockerfile, "Missing Poetry version"
        assert "poetry install" in dockerfile, "Missing poetry install command"

    def test_dockerfile_creates_nonroot_user(self):
        """Verify Dockerfile creates non-root user for security."""
        dockerfile = Path("Dockerfile").read_text()

        assert "groupadd -r pff" in dockerfile, "Missing group creation"
        assert "useradd -r -g pff pff" in dockerfile, "Missing user creation"
        assert "USER pff" in dockerfile, "Missing USER directive"

    def test_dockerfile_has_healthcheck(self):
        """Verify Dockerfile includes healthcheck."""
        dockerfile = Path("Dockerfile").read_text()

        assert "HEALTHCHECK" in dockerfile, "Missing HEALTHCHECK directive"
        assert "/health" in dockerfile, "Healthcheck not using /health endpoint"

    def test_dockerfile_exposes_port_8000(self):
        """Verify Dockerfile exposes correct port."""
        dockerfile = Path("Dockerfile").read_text()

        assert "EXPOSE 8000" in dockerfile, "Missing EXPOSE 8000 directive"

    def test_dockerfile_has_working_directory(self):
        """Verify Dockerfile sets working directory."""
        dockerfile = Path("Dockerfile").read_text()

        assert "WORKDIR /app" in dockerfile, "Missing WORKDIR /app directive"

    def test_dockerfile_copies_venv_from_builder(self):
        """Verify runtime stage copies .venv from builder."""
        dockerfile = Path("Dockerfile").read_text()

        assert "COPY --from=builder /app/.venv /app/.venv" in dockerfile, \
            "Missing .venv copy from builder"

    def test_dockerfile_sets_production_env(self):
        """Verify Dockerfile sets production environment variables."""
        dockerfile = Path("Dockerfile").read_text()

        assert "PFF_ENV=production" in dockerfile, "Missing PFF_ENV=production"


class TestDockerignore:
    """Test .dockerignore configuration."""

    def test_dockerignore_exists(self):
        """Verify .dockerignore exists."""
        dockerignore = Path(".dockerignore")
        assert dockerignore.exists(), ".dockerignore not found"

    def test_dockerignore_excludes_tests(self):
        """Verify .dockerignore excludes test directories."""
        dockerignore = Path(".dockerignore").read_text()

        assert "tests/" in dockerignore, "tests/ not excluded"
        assert ".pytest_cache/" in dockerignore, ".pytest_cache/ not excluded"

    def test_dockerignore_excludes_venv(self):
        """Verify .dockerignore excludes virtual environments."""
        dockerignore = Path(".dockerignore").read_text()

        assert ".venv/" in dockerignore, ".venv/ not excluded"
        assert "venv/" in dockerignore, "venv/ not excluded"

    def test_dockerignore_excludes_git(self):
        """Verify .dockerignore excludes .git directory."""
        dockerignore = Path(".dockerignore").read_text()

        assert ".git/" in dockerignore, ".git/ not excluded"

    def test_dockerignore_excludes_logs(self):
        """Verify .dockerignore excludes logs and outputs."""
        dockerignore = Path(".dockerignore").read_text()

        assert "logs/" in dockerignore, "logs/ not excluded"
        assert "outputs/" in dockerignore, "outputs/ not excluded"
        assert "*.log" in dockerignore, "*.log not excluded"

    def test_dockerignore_excludes_data(self):
        """Verify .dockerignore excludes large data files."""
        dockerignore = Path(".dockerignore").read_text()

        assert "data/" in dockerignore, "data/ not excluded"
        assert "*.zip" in dockerignore, "*.zip not excluded"
        assert "*.parquet" in dockerignore, "*.parquet not excluded"


class TestDockerBuild:
    """Test Docker build process (requires Docker installed)."""

    @pytest.mark.slow
    @pytest.mark.skipif(
        subprocess.run(["which", "docker"], capture_output=True).returncode != 0,
        reason="Docker not installed"
    )
    def test_docker_build_succeeds(self):
        """Test Docker build completes successfully."""
        result = subprocess.run(
            ["docker", "build", "-t", "pff:test", "--target", "runtime", "."],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )

        assert result.returncode == 0, f"Docker build failed:\n{result.stderr}"

    @pytest.mark.slow
    @pytest.mark.skipif(
        subprocess.run(["which", "docker"], capture_output=True).returncode != 0,
        reason="Docker not installed"
    )
    def test_docker_image_size_reasonable(self):
        """Test Docker image size is reasonable (<5GB with all ML dependencies)."""
        # Build image first
        subprocess.run(
            ["docker", "build", "-t", "pff:test", "--target", "runtime", "."],
            capture_output=True,
            timeout=600
        )

        # Get image size
        result = subprocess.run(
            ["docker", "images", "pff:test", "--format", "{{.Size}}"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            size_str = result.stdout.strip()
            # Parse size (e.g., "800MB" or "4.69GB")
            # Increased limit to 5GB due to ML dependencies (Ray, PyTorch, etc.)
            if "GB" in size_str:
                size_gb = float(size_str.replace("GB", ""))
                assert size_gb < 5.0, f"Image too large: {size_str} (target: <5GB)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
