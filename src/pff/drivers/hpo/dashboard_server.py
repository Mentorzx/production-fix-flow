"""CLI entrypoint for HPO dashboard server.

Composition root for the dashboard inbound adapter.
"""

from __future__ import annotations

import argparse

from pff.infrastructure.hpo.dashboard.server import run_server


def main() -> None:
    """Parse CLI args and run dashboard server."""
    parser = argparse.ArgumentParser(description="Peak State HPO Dashboard Server")
    parser.add_argument("--port", type=int, default=8766, help="Server port")
    parser.add_argument("--bind", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument(
        "--parent-pid", type=int, default=None, help="Parent PID for watchdog"
    )
    args = parser.parse_args()
    run_server(port=args.port, parent_pid=args.parent_pid, bind=args.bind)


if __name__ == "__main__":
    main()
