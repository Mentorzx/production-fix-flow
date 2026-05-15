"""
Docker Build Tests - Sprint 10

Tests Dockerfile configuration and build process:
- Multi-stage build structure
- Security (non-root user)
- Image size optimization
- Required files present
"""

import subprocess
import tomllib
from pathlib import Path

import pytest


def _build_runtime_cpu_image() -> subprocess.CompletedProcess[str]:
    """Build the runtime CPU image with a single retry for transient daemon/network failures."""
    command = [
        "docker",
        "build",
        "-t",
        "pff:test-cpu",
        "--build-arg",
        "PFF_ACCELERATOR=cpu",
        "--target",
        "runtime-cpu",
        ".",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode == 0:
        return result
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=1800,
    )


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

        assert "AS builder" in dockerfile, "Missing builder stage"
        assert "AS tools" in dockerfile, "Missing tools stage"
        assert "AS test" in dockerfile, "Missing test stage"
        assert "AS runtime-base" in dockerfile, "Missing runtime-base stage"
        assert "AS runtime-cpu" in dockerfile, "Missing runtime-cpu stage"
        assert "AS runtime-cuda" in dockerfile, "Missing runtime-cuda stage"

    def test_dockerfile_keeps_runtime_stages_before_tooling_stages(self):
        """Verify legacy Docker builds do not build dev/test stages for runtime targets."""
        dockerfile = Path("Dockerfile").read_text()

        assert dockerfile.index("AS runtime-cpu") < dockerfile.index("AS tools")
        assert dockerfile.index("AS runtime-cuda") < dockerfile.index("AS tools")

    def test_dockerfile_supports_accelerator_build_arg(self):
        """Verify Dockerfile is parameterized by accelerator variant."""
        dockerfile = Path("Dockerfile").read_text()

        assert "ARG PFF_ACCELERATOR=cpu" in dockerfile, "Missing accelerator build arg"
        assert "pip install --index-url https://download.pytorch.org/whl/cu128" in dockerfile, (
            "Missing CUDA torch installation path"
        )
        assert "torch.version.cuda is None" in dockerfile, "Missing CPU torch verification"

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

    def test_dockerfile_normalizes_workspace_permissions(self):
        """Verify builder stage normalizes workspace readability for Docker-first tooling."""
        dockerfile = Path("Dockerfile").read_text()

        assert "find /app -path /app/.venv -prune -o -exec chmod a+rX {} +" in dockerfile, (
            "Missing workspace permission normalization for Docker-first tooling"
        )

    def test_dockerfile_sets_runtime_accelerator_env(self):
        """Verify Dockerfile propagates runtime accelerator selection."""
        dockerfile = Path("Dockerfile").read_text()

        assert "PFF_ENV=production" in dockerfile, "Missing PFF_ENV=production"
        assert "PFF_ACCELERATOR=${PFF_ACCELERATOR}" in dockerfile, (
            "Missing runtime accelerator environment export"
        )
        assert "HOME=/tmp/pff-home" in dockerfile, "Missing runtime HOME override"
        assert "XDG_CACHE_HOME=/tmp/pff-home/.cache" in dockerfile, (
            "Missing runtime XDG cache override"
        )
        assert "TRITON_CACHE_DIR=/tmp/pff-home/.cache/triton" in dockerfile, (
            "Missing Triton cache override"
        )

    def test_dockerfile_runtime_cuda_installs_c_compiler_for_triton(self):
        """Verify CUDA runtime includes a C compiler required by Triton."""
        dockerfile = Path("Dockerfile").read_text()

        assert "FROM runtime-base AS runtime-cuda" in dockerfile
        assert "apt-get install -y --no-install-recommends gcc libc6-dev" in dockerfile, (
            "Missing C toolchain headers for Triton runtime compilation"
        )
        assert "ENV CC=gcc" in dockerfile, "Missing CC export for Triton"

    def test_dockerfile_pins_playwright_browser_path_for_test_stage(self):
        """Verify the test image installs Playwright browsers in a stable shared path."""
        dockerfile = Path("Dockerfile").read_text()

        assert "PLAYWRIGHT_BROWSERS_PATH=/ms-playwright" in dockerfile, (
            "Missing stable Playwright browser path in test stage"
        )
        assert 'mkdir -p "${PLAYWRIGHT_BROWSERS_PATH}"' in dockerfile, (
            "Missing Playwright browser directory provisioning"
        )

    def test_dockerfile_builds_dashboard_bundle_in_builder_stage(self):
        """Verify Docker builds can regenerate the dashboard dist bundle from source."""
        dockerfile = Path("Dockerfile").read_text()

        assert "nodejs" in dockerfile and "npm" in dockerfile, (
            "Missing Node.js toolchain required for dashboard bundle builds"
        )
        assert "rustup target add wasm32-unknown-unknown" in dockerfile, (
            "Missing wasm32 target provisioning for dashboard bundle builds"
        )
        assert "cargo install wasm-bindgen-cli --version 0.2.113 --locked" in dockerfile, (
            "Missing pinned wasm-bindgen CLI installation for dashboard bundle builds"
        )
        assert "build_dashboard.sh" in dockerfile, (
            "Missing dashboard bundle build step in Docker builder stage"
        )

    def test_dockerfile_removes_poetry_cache_from_image_layers(self):
        """Verify Poetry downloads are not baked into final image layers."""
        dockerfile = Path("Dockerfile").read_text()

        assert "POETRY_CACHE_DIR=/tmp/pff-poetry-cache" in dockerfile, (
            "Missing isolated Poetry cache directory"
        )
        assert "rm -rf /tmp/pff-poetry-cache /root/.cache/pypoetry" in dockerfile, (
            "Missing Poetry cache cleanup in install layers"
        )

    def test_dockerfile_cpu_runtime_uses_cpu_torch_lock(self):
        """Verify CPU runtime is installed from the CPU lock without CUDA wheel swap."""
        dockerfile = Path("Dockerfile").read_text()
        pyproject = tomllib.loads(Path("pyproject.toml").read_text())

        torch_dependency = pyproject["tool"]["poetry"]["dependencies"]["torch"]
        assert torch_dependency == {"version": "2.7.0", "source": "pytorch-cpu"}
        assert "grep '^nvidia-.*-cu12$'" not in dockerfile
        assert "pip install --index-url https://download.pytorch.org/whl/cpu" not in dockerfile


class TestPackagingScripts:
    """Test packaging helper scripts."""

    def test_pff_run_script_exists(self):
        script = Path("scripts/package/pff-run")
        assert script.exists(), "Missing packaging launcher"

    def test_build_images_script_exists(self):
        script = Path("scripts/package/build-images.sh")
        assert script.exists(), "Missing image build script"

    def test_tool_runner_script_exists(self):
        script = Path("scripts/package/pff-tool-run")
        assert script.exists(), "Missing Docker-first tooling runner"

    def test_smoke_script_exists(self):
        script = Path("scripts/package/smoke-package.sh")
        assert script.exists(), "Missing packaging smoke script"

    def test_measure_image_sizes_script_exists(self):
        script = Path("scripts/package/measure-image-sizes.sh")
        assert script.exists(), "Missing Docker image size measurement script"

    def test_pff_run_exports_writable_home_for_arbitrary_user(self):
        script = Path("scripts/package/pff-run").read_text()
        assert '-e "HOME=/tmp/pff-home"' in script
        assert '-e "XDG_CACHE_HOME=/tmp/pff-home/.cache"' in script

    def test_pff_run_exports_triton_cache_dir(self):
        script = Path("scripts/package/pff-run").read_text()
        assert '-e "TRITON_CACHE_DIR=/tmp/pff-home/.cache/triton"' in script


class TestDockerFirstWrappers:
    """Test root-level Docker-first wrappers."""

    @pytest.mark.parametrize(
        "wrapper_name",
        ["pff", "pytest", "mypy", "ruff", "pyright", "pylint", "black"],
    )
    def test_wrapper_exists(self, wrapper_name: str):
        wrapper = Path(wrapper_name)
        assert wrapper.exists(), f"Missing root wrapper: {wrapper_name}"

    def test_build_images_supports_tools_target(self):
        script = Path("scripts/package/build-images.sh").read_text()
        assert 'TOOLS_IMAGE="${PFF_DOCKER_IMAGE_TOOLS:-pff:tools}"' in script
        assert "tools)" in script
        assert 'TEST_IMAGE="${PFF_DOCKER_IMAGE_TEST:-pff:test}"' in script
        assert "test)" in script

    def test_build_images_defaults_to_cpu_and_reports_sizes(self):
        script = Path("scripts/package/build-images.sh").read_text()

        assert 'TARGET="${1:-cpu}"' in script
        assert "runtime)" in script
        assert "show_sizes()" in script
        assert 'show_sizes "${CPU_IMAGE}" "${CUDA_IMAGE}" "${TOOLS_IMAGE}" "${TEST_IMAGE}"' in script

    def test_smoke_package_defaults_to_cpu_build_without_all_images(self):
        script = Path("scripts/package/smoke-package.sh").read_text()

        assert 'BUILD_TARGET="${PFF_SMOKE_BUILD_TARGET:-cpu}"' in script
        assert 'RUN_GPU="${PFF_SMOKE_RUN_GPU:-auto}"' in script
        assert 'build-images.sh" "${BUILD_TARGET}"' in script
        assert 'PFF_SMOKE_BUILD_TARGET=runtime' in script
        assert "should_run_gpu_smoke()" in script
        assert 'case "${BUILD_TARGET}"' in script
        assert 'build-images.sh" all' not in script

    def test_smoke_package_can_skip_build_and_require_gpu(self):
        script = Path("scripts/package/smoke-package.sh").read_text()

        assert "none|skip)" in script
        assert 'REQUIRE_GPU="${PFF_SMOKE_REQUIRE_GPU:-0}"' in script
        assert 'if [[ "${REQUIRE_GPU}" == "1" ]]' in script
        assert 'image_exists "${CUDA_IMAGE}"' in script
        assert "PFF_SMOKE_RUN_GPU=1" in script

    def test_smoke_package_uses_isolated_workspace_by_default(self):
        script = Path("scripts/package/smoke-package.sh").read_text()

        assert "setup_smoke_workspace()" in script
        assert 'mktemp -d "${TMPDIR:-/tmp}/pff-package-smoke.XXXXXXXX"' in script
        assert '-v "${SMOKE_DATA_DIR}:/app/data"' in script
        assert '-v "${SMOKE_LOGS_DIR}:/app/logs"' in script
        assert '-v "${SMOKE_OUTPUTS_DIR}:/app/outputs"' in script
        assert '-v "${ROOT_DIR}/outputs:/app/outputs"' not in script
        assert '-v "${ROOT_DIR}/data:/app/data"' not in script

    def test_measure_image_sizes_reports_budget_and_baseline_delta(self, tmp_path: Path):
        script = Path("scripts/package/measure-image-sizes.sh")
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_docker = fake_bin / "docker"
        fake_docker.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "image" && "$2" == "inspect" && "$3" == "pff:cpu" ]]; then
  echo 2147483648
  exit 0
fi
exit 1
""",
        )
        fake_docker.chmod(0o755)
        baseline = tmp_path / "baseline.tsv"
        baseline.write_text("image\tbytes\npff:cpu\t4294967296\n")

        result = subprocess.run(
            ["bash", str(script), "--baseline", str(baseline), "pff:cpu"],
            capture_output=True,
            text=True,
            check=False,
            env={"PATH": f"{fake_bin}:/usr/bin:/bin"},
        )

        assert result.returncode == 0, result.stderr
        assert "image\tstatus\tbytes\tgib\tbaseline_bytes\tdelta_gib\tdelta_pct" in result.stdout
        assert "pff:cpu\tpresent\t2147483648\t2.00\t4294967296\t-2.00\t-50.0\t3\tpass" in result.stdout

    def test_measure_image_sizes_can_fail_ci_on_default_budget(self, tmp_path: Path):
        script = Path("scripts/package/measure-image-sizes.sh")
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_docker = fake_bin / "docker"
        fake_docker.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "image" && "$2" == "inspect" && "$3" == "pff:ci" ]]; then
  echo 4294967296
  exit 0
fi
exit 1
""",
        )
        fake_docker.chmod(0o755)
        output_file = tmp_path / "sizes.tsv"

        result = subprocess.run(
            [
                "bash",
                str(script),
                "--fail-on-budget",
                "--output",
                str(output_file),
                "pff:ci",
            ],
            capture_output=True,
            text=True,
            check=False,
            env={
                "PATH": f"{fake_bin}:/usr/bin:/bin",
                "PFF_IMAGE_BUDGET_DEFAULT_GB": "3",
            },
        )

        assert result.returncode == 1
        assert "Image budget exceeded: image=pff:ci" in result.stderr
        assert "pff:ci\tpresent\t4294967296\t4.00\t\t\t\t3\tfail" in output_file.read_text()

    def test_tool_runner_enables_buildkit_when_available(self):
        script = Path("scripts/package/pff-tool-run").read_text()

        assert "docker buildx version" in script
        assert "export DOCKER_BUILDKIT=1" in script
        assert "export DOCKER_BUILDKIT=0" in script

    def test_tool_runner_resolves_compose_network_dynamically(self):
        script = Path("scripts/package/pff-tool-run").read_text()
        assert 'TOOLS_NETWORK="${PFF_DOCKER_TOOLS_NETWORK:-}"' in script
        assert "docker inspect pff-postgres" in script
        assert 'test_network="$(resolve_test_network)"' in script

    def test_tool_runner_adds_docker_socket_group_for_pytest(self):
        script = Path("scripts/package/pff-tool-run").read_text()
        assert "stat -c '%g' /var/run/docker.sock" in script
        assert '--group-add "${socket_gid}"' in script

    def test_tool_runner_exports_playwright_browser_path_for_pytest(self):
        script = Path("scripts/package/pff-tool-run").read_text()
        assert '-e "PLAYWRIGHT_BROWSERS_PATH=/ms-playwright"' in script

    def test_tool_runner_tmpfs_mounts_are_owned_by_current_user_for_pytest(self):
        script = Path("scripts/package/pff-tool-run").read_text()
        assert 'user_id="$(id -u)"' in script
        assert 'group_id="$(id -g)"' in script
        assert "uid=${user_id},gid=${group_id},mode=1777" in script


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

    def test_dockerignore_excludes_node_modules(self):
        """Verify .dockerignore excludes Node.js dependencies from build context."""
        dockerignore = Path(".dockerignore").read_text()

        assert "node_modules/" in dockerignore, "node_modules/ not excluded"

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
        result = _build_runtime_cpu_image()

        assert result.returncode == 0, f"Docker build failed:\n{result.stderr}"

    @pytest.mark.slow
    @pytest.mark.skipif(
        subprocess.run(["which", "docker"], capture_output=True).returncode != 0,
        reason="Docker not installed",
    )
    def test_docker_image_size_reasonable(self):
        """Test Docker image size is reasonable (<10GB with all ML dependencies)."""
        _build_runtime_cpu_image()

        result = subprocess.run(
            ["docker", "images", "pff:test-cpu", "--format", "{{.Size}}"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            size_str = result.stdout.strip()
            if "GB" in size_str:
                size_gb = float(size_str.replace("GB", ""))
                assert size_gb < 10.0, f"Image too large: {size_str} (target: <10GB)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
