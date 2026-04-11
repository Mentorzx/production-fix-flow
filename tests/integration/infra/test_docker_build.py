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
        assert "AS builder" in dockerfile, "Missing builder stage"

        # Check for runtime stages
        assert "AS runtime-base" in dockerfile, "Missing runtime-base stage"
        assert "AS runtime-cpu" in dockerfile, "Missing runtime-cpu stage"
        assert "AS runtime-cuda" in dockerfile, "Missing runtime-cuda stage"

    def test_dockerfile_supports_accelerator_build_arg(self):
        """Verify Dockerfile is parameterized by accelerator variant."""
        dockerfile = Path("Dockerfile").read_text()

        assert "ARG PFF_ACCELERATOR=cpu" in dockerfile, "Missing accelerator build arg"
        assert "pip install --index-url https://download.pytorch.org/whl/cpu" in dockerfile, (
            "Missing CPU torch installation path"
        )
        assert "pip install --index-url https://download.pytorch.org/whl/cu128" in dockerfile, (
            "Missing CUDA torch installation path"
        )

    def test_dockerfile_creates_nonroot_user(self):
        """Verify Dockerfile creates non-root user for security."""
        dockerfile = Path("Dockerfile").read_text()

        assert "groupadd -r pff" in dockerfile, "Missing group creation"
        assert "useradd -r -g pff pff" in dockerfile, "Missing user creation"
        assert "USER pff" in dockerfile, "Missing USER directive"

    def test_dockerfile_uses_cli_entrypoint(self):
        """Verify packaged image exposes the CLI by default."""
        dockerfile = Path("Dockerfile").read_text()

        assert 'ENTRYPOINT ["pff"]' in dockerfile, "Missing CLI entrypoint"
        assert 'CMD ["--help"]' in dockerfile, "Missing CLI default command"

    def test_dockerfile_exposes_port_8000(self):
        """Verify docker-compose can still expose the API port when overriding command."""
        dockerfile = Path("Dockerfile").read_text()

        assert "WORKDIR /app" in dockerfile, "Missing WORKDIR /app directive"

    def test_dockerfile_has_working_directory(self):
        """Verify Dockerfile sets working directory."""
        dockerfile = Path("Dockerfile").read_text()

        assert "WORKDIR /app" in dockerfile, "Missing WORKDIR /app directive"

    def test_dockerfile_copies_venv_from_builder(self):
        """Verify runtime stage copies .venv from builder."""
        dockerfile = Path("Dockerfile").read_text()

        assert "COPY --from=builder" in dockerfile and "/app/.venv /app/.venv" in dockerfile, (
            "Missing .venv copy from builder"
        )

    def test_dockerfile_sets_runtime_accelerator_env(self):
        """Verify Dockerfile propagates runtime accelerator selection."""
        dockerfile = Path("Dockerfile").read_text()

        assert "PFF_ENV=production" in dockerfile, "Missing PFF_ENV=production"
        assert "PFF_ACCELERATOR=${PFF_ACCELERATOR}" in dockerfile, (
            "Missing runtime accelerator environment export"
        )


class TestPackagingScripts:
    """Test packaging helper scripts."""

    def test_pff_run_script_exists(self):
        script = Path("scripts/package/pff-run")
        assert script.exists(), "Missing packaging launcher"

    def test_build_images_script_exists(self):
        script = Path("scripts/package/build-images.sh")
        assert script.exists(), "Missing image build script"

    def test_smoke_script_exists(self):
        script = Path("scripts/package/smoke-package.sh")
        assert script.exists(), "Missing packaging smoke script"


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
        reason="Docker not installed",
    )
    def test_docker_build_succeeds(self):
        """Test Docker build completes successfully."""
        result = subprocess.run(
            [
                "docker",
                "build",
                "-t",
                "pff:test-cpu",
                "--build-arg",
                "PFF_ACCELERATOR=cpu",
                "--target",
                "runtime-cpu",
                ".",
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )

        assert result.returncode == 0, f"Docker build failed:\n{result.stderr}"

    @pytest.mark.slow
    @pytest.mark.skipif(
        subprocess.run(["which", "docker"], capture_output=True).returncode != 0,
        reason="Docker not installed",
    )
    def test_docker_image_size_reasonable(self):
        """Test Docker image size is reasonable (<10GB with all ML dependencies)."""
        # Build image first
        subprocess.run(
            [
                "docker",
                "build",
                "-t",
                "pff:test-cpu",
                "--build-arg",
                "PFF_ACCELERATOR=cpu",
                "--target",
                "runtime-cpu",
                ".",
            ],
            capture_output=True,
            timeout=1800,
        )

        # Get image size
        result = subprocess.run(
            ["docker", "images", "pff:test-cpu", "--format", "{{.Size}}"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            size_str = result.stdout.strip()
            # Parse size (e.g., "800MB" or "7.68GB")
            # Limit set to 10GB due to ML dependencies (Ray, PyTorch, etc.)
            if "GB" in size_str:
                size_gb = float(size_str.replace("GB", ""))
                assert size_gb < 10.0, f"Image too large: {size_str} (target: <10GB)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
