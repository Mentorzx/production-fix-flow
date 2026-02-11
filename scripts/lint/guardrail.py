#!/usr/bin/env python3
"""Unified guardrail for PFF dashboard and lint pipeline.

Routes cache through CacheManager (pff.shared) instead of raw JSON files.
Used by: lint_repo.py, build_dashboard.sh

Version: 6.0.0
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DASHBOARD_DIR = REPO_ROOT / "src" / "pff" / "infrastructure" / "hpo" / "dashboard"

GUARD_VERSION = "6.0.0"

SOURCE_GLOBS = ["**/*.jsx", "**/*.js", "**/*.css", "**/*.html"]
SOURCE_EXCLUDES = {"node_modules", "dist", "build", ".cache"}
CONFIG_FILES = [
    "package.json",
    "eslint.config.cjs",
    "tsconfig.guard.json",
    ".prettierrc.json",
    ".stylelintrc.json",
]

MAX_BUNDLE_BYTES = 950_000
_CACHE_TTL = 30 * 24 * 3600  # 30 days


def _get_disk_cache():
    """Lazy-load DiskCache from pff.shared for persistent cross-process cache."""
    from pff.shared.core.cache import DiskCache

    cache_dir = REPO_ROOT / "outputs" / ".cache" / "guardrail"
    return DiskCache(root=cache_dir, purge_older_than=_CACHE_TTL)


def _compute_fingerprint() -> str:  # type: ignore[return]
    """Compute aggregate fingerprint of dashboard sources via BLAKE3."""
    from pff.shared.core.cache.utils import FunctionCallHasher

    hasher = FunctionCallHasher()

    source_files: list[Path] = []
    for glob_pat in SOURCE_GLOBS:
        for p in DASHBOARD_DIR.rglob(glob_pat.lstrip("**/")):
            if any(exc in p.parts for exc in SOURCE_EXCLUDES):
                continue
            if p.is_file():
                source_files.append(p)

    for cfg_name in CONFIG_FILES:
        cfg = DASHBOARD_DIR / cfg_name
        if cfg.exists():
            source_files.append(cfg)

    parts: list[str] = [GUARD_VERSION]
    for src in sorted(source_files):
        rel = src.relative_to(DASHBOARD_DIR)
        content = src.read_bytes()
        from hashlib import blake2b

        digest = blake2b(content, digest_size=16).hexdigest()
        parts.append(f"{rel}:{digest}")

    return hasher.hash_function_call(lambda: None, *parts)


def _cached_guardrail_check(
    fingerprint: str,
    verbose: bool = True,
) -> dict:  # type: ignore[type-arg]
    """Run guardrail checks with DiskCache persistence.

    The DiskCache decorator keys on (fingerprint,) — if the
    fingerprint matches a cached entry, checks are skipped entirely.
    """
    disk = _get_disk_cache()

    @disk(ttl=_CACHE_TTL)
    def _check(fp: str) -> dict:
        all_passed, results = check_guardrails(verbose=verbose)
        return {
            "version": GUARD_VERSION,
            "fingerprint": fp,
            "timestamp": time.time(),
            "passed": all_passed,
            "checks": results,
        }

    return _check(fingerprint)


def _run_check(cmd: list[str], label: str) -> tuple[bool, str]:
    """Run a check command, return (passed, output)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=DASHBOARD_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        passed = result.returncode == 0
        output = result.stdout + result.stderr
        return passed, output.strip()[:500]
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, str(e)


def check_guardrails(verbose: bool = True) -> tuple[bool, dict]:
    """Run all guardrail checks. Returns (all_passed, results_dict)."""
    results = {}

    if not (DASHBOARD_DIR / "node_modules").exists():
        if verbose:
            print("Guardrail: node_modules missing, skipping dashboard checks")
        return True, {"skipped": "no node_modules"}

    passed, out = _run_check(["npm", "run", "lint"], "eslint")
    results["eslint"] = {"passed": passed, "output": out}
    if verbose:
        print(f"  Guardrail ESLint: {'PASS' if passed else 'FAIL'}")

    passed, out = _run_check(["npm", "run", "typecheck"], "typecheck")
    results["typecheck"] = {"passed": passed, "output": out}
    if verbose:
        print(f"  Guardrail TypeCheck: {'PASS' if passed else 'FAIL'}")

    bundle = DASHBOARD_DIR / "dist" / "dashboard.js"
    if bundle.exists():
        size = bundle.stat().st_size
        passed = size <= MAX_BUNDLE_BYTES
        results["bundle_size"] = {
            "passed": passed,
            "size_bytes": size,
            "max_bytes": MAX_BUNDLE_BYTES,
        }
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(
                f"  Guardrail BundleSize: {status} ({size // 1024}KB / {MAX_BUNDLE_BYTES // 1024}KB)"
            )
    else:
        results["bundle_size"] = {"passed": True, "skipped": "no bundle"}

    all_passed = all(
        r.get("passed", True) for r in results.values() if isinstance(r, dict)
    )
    return all_passed, results


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="PFF unified guardrail (v6 — DiskCache)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run checks (use cache if valid)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    args = parser.parse_args()

    if not DASHBOARD_DIR.exists():
        print("Dashboard directory not found, nothing to guard.")
        return 0

    fingerprint = _compute_fingerprint()

    if args.force:
        # Purge cached entries to force re-run
        disk = _get_disk_cache()
        disk.purge()

    if not args.quiet:
        print(f"Guardrail v{GUARD_VERSION}: fingerprint={fingerprint[:12]}...")

    result = _cached_guardrail_check(
        fingerprint,
        verbose=not args.quiet,
    )
    all_passed = result.get("passed", False)

    if all_passed:
        if not args.quiet:
            print("Guardrail: all checks passed (cached)")
        return 0
    else:
        if not args.quiet:
            print("Guardrail: checks FAILED")
        return 1


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    sys.exit(main())
