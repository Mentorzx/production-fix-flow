#!/usr/bin/env python3
"""Deep cleanup: terminates PFF-owned processes and purges caches.

Extends the existing CleanupEngine with standalone process management.
Safe to call multiple times (idempotent).

Usage:
    poetry run python scripts/cleanup_deep.py           # interactive
    poetry run python scripts/cleanup_deep.py --dry-run  # preview only
    poetry run python scripts/cleanup_deep.py -y         # no confirmation
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "outputs" / ".cache"
HPO_PID_DIR = CACHE_DIR / "hpo"

PFF_PROCESS_PATTERNS = ["pff", "dashboard", "hpo"]


def _read_pid_file(pid_path: Path) -> int | None:
    """Read a PID from a pidfile. Returns None if invalid."""
    try:
        text = pid_path.read_text().strip()
        return int(text) if text.isdigit() else None
    except (OSError, ValueError):
        return None


def _is_pid_running(pid: int) -> bool:
    """Check if a PID is alive."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _kill_pid(pid: int, dry_run: bool = False) -> bool:
    """Send SIGTERM, wait, then SIGKILL if needed. Returns True if killed."""
    if not _is_pid_running(pid):
        return False
    if dry_run:
        print(f"  [DRY-RUN] Would kill PID {pid}")
        return True
    try:
        os.kill(pid, signal.SIGTERM)
        import time

        for _ in range(50):
            time.sleep(0.1)
            if not _is_pid_running(pid):
                print(f"  Terminated PID {pid} (SIGTERM)")
                return True
        os.kill(pid, signal.SIGKILL)
        print(f"  Killed PID {pid} (SIGKILL)")
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _cleanup_pid_files(dry_run: bool = False) -> list[str]:
    """Find and kill processes tracked by PFF pidfiles."""
    actions = []

    dashboard_pid_path = HPO_PID_DIR / "dashboard_server.pid"
    if dashboard_pid_path.exists():
        pid = _read_pid_file(dashboard_pid_path)
        if pid and _is_pid_running(pid):
            killed = _kill_pid(pid, dry_run)
            if killed:
                actions.append(f"dashboard (PID {pid})")
        if not dry_run:
            dashboard_pid_path.unlink(missing_ok=True)
            actions.append("removed dashboard_server.pid")

    for pid_file in HPO_PID_DIR.glob("*.pid") if HPO_PID_DIR.exists() else []:
        pid = _read_pid_file(pid_file)
        if pid and _is_pid_running(pid):
            killed = _kill_pid(pid, dry_run)
            if killed:
                actions.append(f"{pid_file.stem} (PID {pid})")
        if not dry_run:
            pid_file.unlink(missing_ok=True)

    return actions


def _cleanup_child_processes(dry_run: bool = False) -> list[str]:
    """Find and kill PFF child processes via psutil (if available)."""
    actions = []
    try:
        import psutil
    except ImportError:
        actions.append("psutil not available, skipping child process scan")
        return actions

    current_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = proc.info["pid"]
            if pid == current_pid:
                continue
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            name = (proc.info.get("name") or "").lower()

            is_pff = any(pat in cmdline or pat in name for pat in PFF_PROCESS_PATTERNS)
            if not is_pff:
                continue

            is_user_owned = proc.username() == os.getenv("USER", "")
            if not is_user_owned:
                continue

            if dry_run:
                actions.append(f"[DRY-RUN] Would kill: {name} (PID {pid})")
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                    actions.append(f"Terminated: {name} (PID {pid})")
                except psutil.TimeoutExpired:
                    proc.kill()
                    actions.append(f"Killed: {name} (PID {pid})")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return actions


def _cleanup_caches(dry_run: bool = False) -> list[str]:
    """Purge lint and guardrail caches — delegates to lint_repo.clean_lint_caches()."""
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "lint"))
    if dry_run:
        from lint_repo import LINT_CACHE_DIRS

        actions = [f"[DRY-RUN] Would remove lint caches: {', '.join(LINT_CACHE_DIRS)}"]
        guardrail_cache = CACHE_DIR / "guardrail"
        if guardrail_cache.exists():
            actions.append(f"[DRY-RUN] Would remove: {guardrail_cache.relative_to(REPO_ROOT)}")
        return actions

    from lint_repo import clean_lint_caches

    removed = clean_lint_caches()
    return [f"Removed: {r}" for r in removed] if removed else []


def main() -> int:
    parser = argparse.ArgumentParser(description="PFF deep cleanup + process termination")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without executing")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    print("PFF Deep Cleanup")
    print("=" * 50)

    if not args.yes and not args.dry_run:
        confirm = input("This will kill PFF processes and purge caches. Continue? [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            print("Aborted.")
            return 0

    all_actions: list[str] = []

    print("\n1. Checking PID files...")
    all_actions.extend(_cleanup_pid_files(args.dry_run))

    print("2. Scanning child processes...")
    all_actions.extend(_cleanup_child_processes(args.dry_run))

    print("3. Purging caches...")
    all_actions.extend(_cleanup_caches(args.dry_run))

    print(f"\n{'DRY-RUN ' if args.dry_run else ''}Summary:")
    if all_actions:
        for action in all_actions:
            print(f"  - {action}")
    else:
        print("  Nothing to clean.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
