"""
Robust Background Process Management (SOTA).

This module provides a context manager for running background sidecar processes
(like dashboards, proxy servers, or workers) with SOTA safety guarantees:

1.  **Orphan Prevention (Linux):** Uses `prctl(PR_SET_PDEATHSIG)` to ensure the
    child process terminates instantly if the parent process crashes hard
    (e.g., SIGKILL, OOM Killer).
2.  **Tree Killing:** Uses `psutil` to recursively kill the entire process tree
    (child + grandchildren) on exit, preventing orphaned worker threads.
3.  **Signal Safety:** Detaches the child from the terminal group (`start_new_session=True`)
    so Ctrl+C doesn't kill it before the parent can gracefully shut it down.
4.  **Automatic Cleanup:** Registers `atexit` handlers as a failsafe.

Usage:
    async with BackgroundProcess(
        ["python", "-m", "server"],
        name="dashboard"
    ) as proc:
        await do_work()
"""

import atexit
import platform
import subprocess
import threading
from typing import Any

from pff.shared.core.logging import logger

try:
    import psutil
except ImportError:
    psutil = None


class BackgroundProcess:
    """
    Context manager for a robust background process.
    """

    def __init__(
        self,
        command: list[str],
        name: str = "background_process",
        graceful_timeout: float = 5.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ):
        """
        Initialize the background process manager.

        Args:
            command: Command to execute (list of args)
            name: Human-readable name for logging
            graceful_timeout: Seconds to wait for SIGTERM before SIGKILL
            cwd: Working directory
            env: Environment variables override
        """
        self.command = command
        self.name = name
        self.graceful_timeout = graceful_timeout
        self.cwd = cwd
        self.env = env
        self.process: subprocess.Popen | None = None
        self._finalizer_registered = False
        self._lock = threading.Lock()

    def _preexec_fn(self) -> None:
        """
        Linux-only: Set PR_SET_PDEATHSIG to SIGTERM.
        This ensures the kernel sends SIGTERM to the child if the parent dies.
        """
        if platform.system() == "Linux":
            try:
                import ctypes

                libc = ctypes.CDLL("libc.so.6")
                PR_SET_PDEATHSIG = 1
                SIGTERM = 15
                libc.prctl(PR_SET_PDEATHSIG, SIGTERM)
            except Exception:
                pass

    def start(self) -> None:
        """Start the background process."""
        if self.process is not None:
            return

        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=self.cwd,
                env=self.env,
                preexec_fn=self._preexec_fn if platform.system() == "Linux" else None,
                start_new_session=True,
            )
            logger.debug(f"Background process started: name={self.name}, pid={self.process.pid}")

            if not self._finalizer_registered:
                atexit.register(self.stop)
                self._finalizer_registered = True

        except Exception as e:
            logger.error(f"Failed to start {self.name}: {e}")
            raise

    def stop(self) -> None:
        """Stop the process and its children recursively."""
        with self._lock:
            if self.process is None:
                return

            pid = self.process.pid
            logger.debug(f"Stopping background process: name={self.name}, pid={pid}")

            children = self._collect_children(pid)
            self._terminate_process()
            self._terminate_children(children)

            self.process = None
            logger.info(f"Processo {self.name} encerrado")

    def _collect_children(self, pid: int) -> list[Any]:
        """Execute collect children.



        Args:

            pid: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if not psutil:
            return []
        try:
            parent = psutil.Process(pid)
            return parent.children(recursive=True)
        except psutil.NoSuchProcess:
            return []

    def _terminate_process(self) -> None:
        """Execute terminate process."""

        if self.process is None:
            return
        try:
            self.process.terminate()
        except ProcessLookupError:
            return
        try:
            self.process.wait(timeout=self.graceful_timeout)
        except subprocess.TimeoutExpired:
            logger.warning(f"{self.name} did not respond to SIGTERM, forcing SIGKILL")
            try:
                self.process.kill()
            except ProcessLookupError:
                pass

    def _terminate_children(self, children: list[Any]) -> None:
        """Execute terminate children.



        Args:

            children: Input value used by this callable.

        """

        if not children or not psutil:
            return
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(children, timeout=1.0)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

    def __enter__(self) -> "BackgroundProcess":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    async def __aenter__(self) -> "BackgroundProcess":
        self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
