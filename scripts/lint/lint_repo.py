#!/usr/bin/env python3
"""Unified repo-wide lint/guardrail pipeline for PFF.

Orchestrates Python, Dashboard (JS/React), Rust, Docs/Config linters
with scope detection, caching, and reporting.

Usage:
    poetry run python scripts/lint/lint_repo.py --fix          # autofix all
    poetry run python scripts/lint/lint_repo.py --check        # CI mode (no writes)
    poetry run python scripts/lint/lint_repo.py --changed-only # only git-changed files
    poetry run python scripts/lint/lint_repo.py --full         # everything
    poetry run python scripts/lint/lint_repo.py --fail-fast    # stop on first failure
    poetry run python scripts/lint/lint_repo.py --clean        # purge lint caches first
    poetry run python scripts/lint/lint_repo.py --no-clean     # skip cache purge
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DASHBOARD_DIR = REPO_ROOT / "src" / "pff" / "infrastructure" / "hpo" / "dashboard"
RUST_DIR = REPO_ROOT / "src" / "pff_rust"
CONFIG_DIR = REPO_ROOT / "config"
SCRIPTS_DIR = REPO_ROOT / "scripts" / "lint"

PYTHON_DIRS = ["src/", "tests/", "scripts/"]
DOC_GLOBS = ["*.md", "*.yaml", "*.yml", "*.sh"]

LINT_CACHE_DIRS = [
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".pyright",
    ".pylint.d",
    ".eslintcache",
]

PYCACHE_NAME = "__pycache__"


@dataclass
class LintResult:
    """Result of a single linter run."""

    tool: str
    scope: str
    returncode: int
    errors: int = 0
    warnings: int = 0
    duration: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class LintReport:
    """Aggregated lint report."""

    results: list[LintResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def has_errors(self) -> bool:
        """Execute has errors.



        Returns:

            Return value produced by the callable.

        """

        return any(r.returncode != 0 and not r.skipped for r in self.results)

    @property
    def total_duration(self) -> float:
        """Execute total duration.



        Returns:

            Return value produced by the callable.

        """

        return self.end_time - self.start_time

    def print_report(self) -> None:
        """Print formatted report table."""
        print("\n" + "=" * 78)
        print(f"{'LINT REPORT':^78}")
        print("=" * 78)
        print(f"{'Tool':<22} {'Scope':<12} {'Status':<10} {'Errors':<8} {'Warns':<8} {'Time':<8}")
        print("-" * 78)
        for r in self.results:
            if r.skipped:
                status = "SKIP"
            elif r.returncode == 0:
                status = "PASS"
            else:
                status = "FAIL"
            note = f" ({r.skip_reason})" if r.skipped and r.skip_reason else ""
            print(
                f"{r.tool:<22} {r.scope:<12} {status:<10} "
                f"{r.errors:<8} {r.warnings:<8} {r.duration:.1f}s{note}"
            )
        print("-" * 78)
        total_err = sum(r.errors for r in self.results if not r.skipped)
        total_warn = sum(r.warnings for r in self.results if not r.skipped)
        passed = sum(1 for r in self.results if r.returncode == 0 and not r.skipped)
        failed = sum(1 for r in self.results if r.returncode != 0 and not r.skipped)
        skipped = sum(1 for r in self.results if r.skipped)
        verdict = "PASS" if not self.has_errors else "FAIL"
        print(
            f"Total: {passed} passed, {failed} failed, {skipped} skipped | "
            f"{total_err} errors, {total_warn} warnings | "
            f"{self.total_duration:.1f}s | {verdict}"
        )
        print("=" * 78)


def _has_cmd(cmd: str) -> bool:
    """Check if a command is available on PATH."""
    return shutil.which(cmd) is not None


def _run(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int = 300,
) -> tuple[int, str, str]:
    """Run a subprocess, return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd or REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 2, "", f"TIMEOUT after {timeout}s"
    except FileNotFoundError:
        return 2, "", f"Command not found: {cmd[0]}"


def _count_issues(stdout: str, stderr: str) -> tuple[int, int]:
    """Heuristic error/warning counter from linter output."""
    combined = stdout + stderr
    errors = 0
    warnings = 0
    for line in combined.splitlines():
        ll = line.lower()
        if "error" in ll or ": e" in ll:
            errors += 1
        elif "warning" in ll or ": w" in ll:
            warnings += 1
    return errors, warnings


def _get_changed_files() -> list[str]:
    """Get list of files changed vs default branch or HEAD."""
    rc, out, _ = _run(["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD"])
    if rc != 0:
        rc, out, _ = _run(["git", "diff", "--name-only", "--diff-filter=ACMR"])
    return [f for f in out.strip().splitlines() if f]


def _has_scope(changed: list[str] | None, prefixes: list[str]) -> bool:
    """Check if any changed file matches the given prefixes."""
    if changed is None:
        return True
    return any(any(f.startswith(p) for p in prefixes) for f in changed)


def _has_dashboard_changes(changed: list[str] | None) -> bool:
    return _has_scope(changed, ["src/pff/infrastructure/hpo/dashboard/"])


def _has_rust_changes(changed: list[str] | None) -> bool:
    return _has_scope(changed, ["src/pff_rust/", "Cargo.toml"])


def _has_doc_changes(changed: list[str] | None) -> bool:
    """Execute has doc changes.



    Args:

        changed: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    if changed is None:
        return True
    return any(
        f.endswith((".md", ".yaml", ".yml", ".sh")) or f.startswith("config/") for f in changed
    )


# ---------------------------------------------------------------------------
# Lint runners
# ---------------------------------------------------------------------------


def run_ruff_check(fix: bool, changed: list[str] | None) -> LintResult:
    """Run ruff linter."""
    cmd = ["poetry", "run", "ruff", "check"]
    if fix:
        cmd.append("--fix")
    cmd.extend(PYTHON_DIRS)
    t0 = time.monotonic()
    rc, out, err = _run(cmd)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("ruff check", "python", rc, errors, warnings, dur)


def run_ruff_format(fix: bool, changed: list[str] | None) -> LintResult:
    """Run canonical Python formatter (Black) aligned with CI."""
    cmd = ["poetry", "run", "black", "src/pff/"]
    if not fix:
        cmd.append("--check")
    t0 = time.monotonic()
    rc, out, err = _run(cmd)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("black format", "python", rc, errors, warnings, dur)


def run_stdlib_json_guard() -> LintResult:
    """Block stdlib json imports; enforce orjson/msgspec usage."""
    cmd = [
        "poetry",
        "run",
        "pytest",
        "-q",
        "tests/architecture/test_no_stdlib_json.py",
    ]
    t0 = time.monotonic()
    rc, out, err = _run(cmd, timeout=120)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("stdlib-json guard", "python", rc, errors, warnings, dur)


def run_mypy() -> LintResult:
    """Run mypy type checker."""
    cmd = ["poetry", "run", "mypy", "src/"]
    t0 = time.monotonic()
    rc, out, err = _run(cmd, timeout=600)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("mypy", "python", rc, errors, warnings, dur)


def run_pyright() -> LintResult:
    """Run pyright type checker."""
    if not _has_cmd("pyright"):
        rc2, _, _ = _run(["poetry", "run", "pyright", "--version"])
        if rc2 != 0:
            return LintResult("pyright", "python", 0, skipped=True, skip_reason="not installed")
    cmd = ["poetry", "run", "pyright", "src/"]
    t0 = time.monotonic()
    rc, out, err = _run(cmd, timeout=600)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("pyright", "python", rc, errors, warnings, dur)


def run_pylint() -> LintResult:
    """Run pylint on src/pff/."""
    cmd = [
        "poetry",
        "run",
        "pylint",
        "src/pff/",
        "--rcfile=pyproject.toml",
        "--errors-only",
        "-j0",
    ]
    t0 = time.monotonic()
    rc, out, err = _run(cmd, timeout=600)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("pylint", "python", rc, errors, warnings, dur)


def run_bandit() -> LintResult:
    """Run bandit security linter."""
    cmd = ["poetry", "run", "bandit", "-r", "src/pff/", "-lll", "-q"]
    t0 = time.monotonic()
    rc, out, err = _run(cmd, timeout=300)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("bandit", "python", rc, errors, warnings, dur)


def run_pip_audit() -> LintResult:
    """Run pip-audit for supply-chain checks."""
    cmd = ["poetry", "run", "pip-audit"]
    t0 = time.monotonic()
    rc, out, err = _run(cmd, timeout=300)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("pip-audit", "python", rc, errors, warnings, dur)


def run_log_lint(fix: bool) -> LintResult:
    """Run log-lint compliance checker."""
    log_lint_script = SCRIPTS_DIR / "log_lint.py"
    if not log_lint_script.exists():
        return LintResult("log-lint", "python", 0, skipped=True, skip_reason="script missing")
    cmd = ["poetry", "run", "python", str(log_lint_script)]
    if fix:
        cmd.append("--fix")
    else:
        cmd.append("--check")
    cmd.append("src/pff/")
    t0 = time.monotonic()
    rc, out, err = _run(cmd, timeout=120)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("log-lint", "python", rc, errors, warnings, dur)


# --- Dashboard ---


def run_eslint(fix: bool) -> LintResult:
    """Run ESLint on dashboard."""
    if not (DASHBOARD_DIR / "node_modules").exists():
        return LintResult("eslint", "dashboard", 0, skipped=True, skip_reason="no node_modules")
    cmd = ["npm", "run", "lint"]
    t0 = time.monotonic()
    rc, out, err = _run(cmd, cwd=DASHBOARD_DIR)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("eslint", "dashboard", rc, errors, warnings, dur)


def run_prettier(fix: bool) -> LintResult:
    """Run Prettier on dashboard."""
    if not (DASHBOARD_DIR / "node_modules").exists():
        return LintResult("prettier", "dashboard", 0, skipped=True, skip_reason="no node_modules")
    cmd = ["npx", "prettier"]
    if fix:
        cmd.extend(["--write", "static/"])
    else:
        cmd.extend(["--check", "static/"])
    t0 = time.monotonic()
    rc, out, err = _run(cmd, cwd=DASHBOARD_DIR)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("prettier", "dashboard", rc, errors, warnings, dur)


def run_stylelint(fix: bool) -> LintResult:
    """Run Stylelint on dashboard CSS."""
    if not (DASHBOARD_DIR / "node_modules").exists():
        return LintResult("stylelint", "dashboard", 0, skipped=True, skip_reason="no node_modules")
    cmd = ["npx", "stylelint", "static/css/**/*.css"]
    if fix:
        cmd.append("--fix")
    t0 = time.monotonic()
    rc, out, err = _run(cmd, cwd=DASHBOARD_DIR)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("stylelint", "dashboard", rc, errors, warnings, dur)


def run_tsc() -> LintResult:
    """Run TypeScript type checking on dashboard."""
    if not (DASHBOARD_DIR / "node_modules").exists():
        return LintResult("tsc", "dashboard", 0, skipped=True, skip_reason="no node_modules")
    cmd = ["npm", "run", "typecheck"]
    t0 = time.monotonic()
    rc, out, err = _run(cmd, cwd=DASHBOARD_DIR)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("tsc", "dashboard", rc, errors, warnings, dur)


# --- Rust ---


def run_cargo_fmt(fix: bool) -> LintResult:
    """Run cargo fmt."""
    cmd = ["cargo", "fmt", "--manifest-path", str(RUST_DIR / "Cargo.toml")]
    if not fix:
        cmd.append("--check")
    t0 = time.monotonic()
    rc, out, err = _run(cmd)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("cargo fmt", "rust", rc, errors, warnings, dur)


def run_cargo_clippy() -> LintResult:
    """Run cargo clippy."""
    cmd = [
        "cargo",
        "clippy",
        "--manifest-path",
        str(RUST_DIR / "Cargo.toml"),
        "--",
        "-D",
        "warnings",
    ]
    t0 = time.monotonic()
    rc, out, err = _run(cmd, timeout=600)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("cargo clippy", "rust", rc, errors, warnings, dur)


def run_cargo_audit() -> LintResult:
    """Run cargo audit."""
    if not _has_cmd("cargo-audit"):
        return LintResult("cargo audit", "rust", 0, skipped=True, skip_reason="not installed")
    cmd = ["cargo", "audit", "--file", str(RUST_DIR / "Cargo.lock")]
    t0 = time.monotonic()
    rc, out, err = _run(cmd)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("cargo audit", "rust", rc, errors, warnings, dur)


def run_cargo_deny() -> LintResult:
    """Run cargo deny."""
    if not _has_cmd("cargo-deny"):
        return LintResult("cargo deny", "rust", 0, skipped=True, skip_reason="not installed")
    cmd = [
        "cargo",
        "deny",
        "--manifest-path",
        str(RUST_DIR / "Cargo.toml"),
        "-c",
        str(RUST_DIR / "deny.toml"),
        "check",
    ]
    t0 = time.monotonic()
    rc, out, err = _run(cmd)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("cargo deny", "rust", rc, errors, warnings, dur)


# --- Docs/Config ---


def run_yamllint() -> LintResult:
    """Run yamllint on config/."""
    if not _has_cmd("yamllint"):
        return LintResult("yamllint", "config", 0, skipped=True, skip_reason="not installed")
    cmd = ["yamllint", "-c", str(REPO_ROOT / ".yamllint.yml"), str(CONFIG_DIR)]
    t0 = time.monotonic()
    rc, out, err = _run(cmd)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("yamllint", "config", rc, errors, warnings, dur)


def run_shellcheck() -> LintResult:
    """Run shellcheck on shell scripts."""
    if not _has_cmd("shellcheck"):
        return LintResult("shellcheck", "scripts", 0, skipped=True, skip_reason="not installed")
    sh_files = list(REPO_ROOT.rglob("*.sh"))
    if not sh_files:
        return LintResult("shellcheck", "scripts", 0, skipped=True, skip_reason="no .sh files")
    cmd = ["shellcheck", "--severity=warning"] + [str(f) for f in sh_files[:50]]
    t0 = time.monotonic()
    rc, out, err = _run(cmd)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("shellcheck", "scripts", rc, errors, warnings, dur)


def run_markdownlint() -> LintResult:
    """Run markdownlint-cli2 on docs."""
    if not _has_cmd("markdownlint-cli2"):
        return LintResult("markdownlint", "docs", 0, skipped=True, skip_reason="not installed")
    cmd = [
        "markdownlint-cli2",
        "**/*.md",
        "--config",
        str(REPO_ROOT / ".markdownlint.jsonc"),
    ]
    t0 = time.monotonic()
    rc, out, err = _run(cmd)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("markdownlint", "docs", rc, errors, warnings, dur)


# --- Guardrail ---


def run_guardrail() -> LintResult:
    """Run unified guardrail checks."""
    guardrail_script = SCRIPTS_DIR / "guardrail.py"
    if not guardrail_script.exists():
        return LintResult("guardrail", "dashboard", 0, skipped=True, skip_reason="script missing")
    cmd = ["poetry", "run", "python", str(guardrail_script), "--check"]
    t0 = time.monotonic()
    rc, out, err = _run(cmd, timeout=120)
    dur = time.monotonic() - t0
    errors, warnings = _count_issues(out, err)
    return LintResult("guardrail", "dashboard", rc, errors, warnings, dur)


# ---------------------------------------------------------------------------
# Cache cleaning
# ---------------------------------------------------------------------------


def clean_lint_caches() -> list[str]:
    """Remove lint/tool caches. Returns list of removed paths."""
    removed = []

    for cache_name in LINT_CACHE_DIRS:
        for cache_path in REPO_ROOT.rglob(cache_name):
            if cache_path.is_dir() and "node_modules" not in str(cache_path):
                try:
                    sz = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
                    shutil.rmtree(cache_path, ignore_errors=True)
                    removed.append(f"{cache_path.relative_to(REPO_ROOT)} ({sz // 1024}KB)")
                except OSError:
                    pass

    for pycache in REPO_ROOT.rglob(PYCACHE_NAME):
        if pycache.is_dir() and ".venv" not in str(pycache) and "node_modules" not in str(pycache):
            try:
                shutil.rmtree(pycache, ignore_errors=True)
                removed.append(str(pycache.relative_to(REPO_ROOT)))
            except OSError:
                pass

    dashboard_eslint_cache = DASHBOARD_DIR / ".eslintcache"
    if dashboard_eslint_cache.exists():
        dashboard_eslint_cache.unlink(missing_ok=True)
        removed.append("dashboard/.eslintcache")

    guardrail_cache = REPO_ROOT / "outputs" / ".cache" / "guardrail"
    if guardrail_cache.is_dir():
        shutil.rmtree(guardrail_cache, ignore_errors=True)
        removed.append("outputs/.cache/guardrail")

    return removed


# ---------------------------------------------------------------------------
# Auto-fix pass (silent — runs all fixers before checking)
# ---------------------------------------------------------------------------


def _run_autofix_pass(changed: list[str] | None) -> None:
    """Run all auto-fixers silently. Called before the check pass so fixes
    from one tool benefit checks from another (e.g. ruff fix -> mypy happy)."""
    print("\n=== Auto-fix pass ===")
    fixed: list[str] = []

    if _has_scope(changed, PYTHON_DIRS):
        rc, _, _ = _run(["poetry", "run", "ruff", "check", "--fix", "--quiet"] + PYTHON_DIRS)
        if rc == 0:
            fixed.append("ruff check --fix")
        rc, _, _ = _run(["poetry", "run", "ruff", "format"] + PYTHON_DIRS)
        if rc == 0:
            fixed.append("ruff format")

    if _has_dashboard_changes(changed) and (DASHBOARD_DIR / "node_modules").exists():
        rc, _, _ = _run(
            ["npx", "prettier", "--write", "--log-level=warn", "static/"],
            cwd=DASHBOARD_DIR,
        )
        if rc == 0:
            fixed.append("prettier --write")
        rc, _, _ = _run(
            ["npx", "stylelint", "static/css/**/*.css", "--fix", "--quiet"],
            cwd=DASHBOARD_DIR,
        )
        if rc == 0:
            fixed.append("stylelint --fix")

    if _has_rust_changes(changed):
        rc, _, _ = _run(["cargo", "fmt", "--manifest-path", str(RUST_DIR / "Cargo.toml")])
        if rc == 0:
            fixed.append("cargo fmt")
        rc, _, _ = _run(
            [
                "cargo",
                "clippy",
                "--manifest-path",
                str(RUST_DIR / "Cargo.toml"),
                "--fix",
                "--allow-dirty",
                "--allow-staged",
            ],
            timeout=600,
        )
        if rc == 0:
            fixed.append("cargo clippy --fix")

    if _has_scope(changed, PYTHON_DIRS):
        log_lint_script = SCRIPTS_DIR / "log_lint.py"
        if log_lint_script.exists():
            rc, _, _ = _run(
                ["poetry", "run", "python", str(log_lint_script), "--fix", "src/pff/"],
                timeout=120,
            )
            if rc == 0:
                fixed.append("log-lint --fix")

    if fixed:
        print(f"  Applied: {', '.join(fixed)}")
    else:
        print("  No auto-fixers applied.")
    print()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main() -> int:
    """Execute main.



    Returns:

        Return value produced by the callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    parser = argparse.ArgumentParser(
        description="PFF unified lint pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fix", action="store_true", help="Autofix where possible")
    mode.add_argument("--check", action="store_true", help="Check-only (CI mode)")
    parser.add_argument("--changed-only", action="store_true", help="Lint only git-changed files")
    parser.add_argument("--full", action="store_true", help="Run all linters including slow ones")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="Purge lint caches first (default)",
    )
    parser.add_argument("--no-clean", action="store_true", help="Skip cache purge")
    args = parser.parse_args()

    fix = args.fix
    fail_fast = args.fail_fast
    do_clean = args.clean and not args.no_clean

    changed: list[str] | None = None
    if args.changed_only:
        changed = _get_changed_files()
        if not changed:
            print("No changed files detected. Nothing to lint.")
            return 0
        print(f"Linting {len(changed)} changed file(s)...")

    report = LintReport()
    report.start_time = time.monotonic()

    if do_clean:
        removed = clean_lint_caches()
        if removed:
            print(f"Cleaned {len(removed)} cache(s): {', '.join(removed[:5])}")
            if len(removed) > 5:
                print(f"  ... and {len(removed) - 5} more")

    if fix:
        _run_autofix_pass(changed)

    # After autofix pass, all checks run in check-only mode to report residuals.
    check_fix = False

    def _add(result: LintResult) -> bool:
        """Add result, return True if should stop."""
        report.results.append(result)
        if result.skipped:
            return False
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"  [{status}] {result.tool} ({result.scope}) - {result.duration:.1f}s")
        return fail_fast and result.returncode != 0

    # --- Python ---
    if _has_scope(changed, PYTHON_DIRS):
        print("\n--- Python (check) ---")
        if _add(run_ruff_check(check_fix, changed)):
            report.end_time = time.monotonic()
            report.print_report()
            return 1
        if _add(run_ruff_format(check_fix, changed)):
            report.end_time = time.monotonic()
            report.print_report()
            return 1
        if _add(run_stdlib_json_guard()):
            report.end_time = time.monotonic()
            report.print_report()
            return 1

        _add(run_mypy())
        if fail_fast and report.has_errors:
            report.end_time = time.monotonic()
            report.print_report()
            return 1

        _add(run_pyright())
        if fail_fast and report.has_errors:
            report.end_time = time.monotonic()
            report.print_report()
            return 1

        _add(run_pylint())
        _add(run_bandit())

        if args.full:
            _add(run_pip_audit())

        _add(run_log_lint(check_fix))

    # --- Dashboard ---
    if _has_dashboard_changes(changed):
        print("\n--- Dashboard (check) ---")
        _add(run_eslint(check_fix))
        if fail_fast and report.has_errors:
            report.end_time = time.monotonic()
            report.print_report()
            return 1
        _add(run_prettier(check_fix))
        _add(run_stylelint(check_fix))
        _add(run_tsc())
        _add(run_guardrail())

    # --- Rust ---
    if _has_rust_changes(changed):
        print("\n--- Rust (check) ---")
        _add(run_cargo_fmt(check_fix))
        if fail_fast and report.has_errors:
            report.end_time = time.monotonic()
            report.print_report()
            return 1
        _add(run_cargo_clippy())
        if args.full:
            _add(run_cargo_audit())
            _add(run_cargo_deny())

    # --- Docs/Config ---
    if _has_doc_changes(changed):
        print("\n--- Docs/Config ---")
        _add(run_yamllint())
        _add(run_shellcheck())
        if args.full:
            _add(run_markdownlint())

    report.end_time = time.monotonic()
    report.print_report()
    return 1 if report.has_errors else 0


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    sys.exit(main())
