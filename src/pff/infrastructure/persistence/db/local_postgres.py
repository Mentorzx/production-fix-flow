"""Local PostgreSQL bootstrap for development/runtime environments.

Ensures a local PostgreSQL server is available for HPO storage when the
configured storage URL points to localhost/127.0.0.1.
"""

from __future__ import annotations

import socket
import subprocess
import os
import shutil
from pathlib import Path
from urllib.parse import urlsplit

from pff.shared import logger

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _is_local_host(host: str) -> bool:
    """Return True when host points to local machine."""
    return host.strip().lower() in _LOCAL_HOSTS


def _quote_sql_literal(value: str) -> str:
    """Quote literal for SQL statements."""
    return value.replace("'", "''")


def _run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run command with stdout/stderr captured."""
    return subprocess.run(args, check=True, text=True, capture_output=True)


def _detect_pg_bin_dir() -> Path:
    """Resolve Postgres bin directory from pgserver package."""
    try:
        import pgserver.postgres_server as pgserver_runtime
    except ImportError as exc:  # pragma: no cover - covered by higher-level error path
        raise RuntimeError(
            "Embedded Postgres tooling is unavailable. Install dependency 'pgserver'."
        ) from exc

    pg_bin_path = getattr(pgserver_runtime, "POSTGRES_BIN_PATH", None)
    if pg_bin_path is None:
        raise RuntimeError(
            "Embedded Postgres tooling is unavailable. Missing POSTGRES_BIN_PATH in pgserver."
        )

    pg_bin_dir = Path(str(pg_bin_path))
    if not pg_bin_dir.exists():
        raise RuntimeError(f"Embedded Postgres bin directory not found: {pg_bin_dir}")
    return pg_bin_dir


def _is_tcp_open(host: str, port: int, timeout_s: float = 1.0) -> bool:
    """Check if TCP endpoint is reachable."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _parse_storage_url(storage_url: str) -> tuple[str, int, str, str, str]:
    """Parse PostgreSQL URL into host, port, user, password, dbname."""
    parsed = urlsplit(storage_url)
    host = parsed.hostname or "localhost"
    port = int(parsed.port or 5432)
    user = parsed.username or "postgres"
    password = parsed.password or ""
    dbname = parsed.path.lstrip("/") or "postgres"
    return host, port, user, password, dbname


def _ensure_initialized(pg_bin: Path, pgdata: Path, user: str) -> None:
    """Initialize cluster if PG_VERSION is missing."""
    pg_version = pgdata / "PG_VERSION"
    if pg_version.exists():
        return

    pgdata.parent.mkdir(parents=True, exist_ok=True)
    if pgdata.exists():
        shutil.rmtree(pgdata, ignore_errors=True)
    args = [
        str(pg_bin / "initdb"),
        "-D",
        str(pgdata),
        "--auth=trust",
        "--auth-local=trust",
        "--encoding=UTF8",
        "--username",
        user,
    ]
    _run_cmd(args)
    logger.info(f"Postgres local inicializado em {pgdata}")


def _ensure_started(
    pg_bin: Path, pgdata: Path, host: str, port: int, logfile: Path
) -> None:
    """Start PostgreSQL if not running."""
    status_args = [str(pg_bin / "pg_ctl"), "-D", str(pgdata), "status"]
    status = subprocess.run(status_args, text=True, capture_output=True)
    if status.returncode == 0:
        return

    logfile.parent.mkdir(parents=True, exist_ok=True)
    start_args = [
        str(pg_bin / "pg_ctl"),
        "-D",
        str(pgdata),
        "-l",
        str(logfile),
        "-o",
        f"-h {host} -p {port}",
        "start",
    ]
    _run_cmd(start_args)
    logger.info(f"Postgres local iniciado em {host}:{port}")


def _psql_exec(pg_bin: Path, conn_url: str, sql: str) -> str:
    """Run SQL with psql and return stdout."""
    args = [str(pg_bin / "psql"), conn_url, "-v", "ON_ERROR_STOP=1", "-tAc", sql]
    proc = _run_cmd(args)
    return proc.stdout.strip()


def _ensure_role_and_db(
    pg_bin: Path, *, host: str, port: int, user: str, password: str, dbname: str
) -> None:
    """Ensure role and database exist and are aligned with configured credentials."""
    admin_url = f"postgresql://{user}@{host}:{port}/postgres"
    user_esc = _quote_sql_literal(user)
    pass_esc = _quote_sql_literal(password)
    db_esc = _quote_sql_literal(dbname)

    role_exists = _psql_exec(
        pg_bin,
        admin_url,
        f"SELECT 1 FROM pg_roles WHERE rolname='{user_esc}';",
    )
    if role_exists != "1":
        _psql_exec(pg_bin, admin_url, f'CREATE ROLE "{user}" LOGIN SUPERUSER;')
    _psql_exec(
        pg_bin, admin_url, f"ALTER ROLE \"{user}\" WITH LOGIN PASSWORD '{pass_esc}';"
    )

    db_exists = _psql_exec(
        pg_bin,
        admin_url,
        f"SELECT 1 FROM pg_database WHERE datname='{db_esc}';",
    )
    if db_exists != "1":
        _psql_exec(pg_bin, admin_url, f'CREATE DATABASE "{dbname}" OWNER "{user}";')


def ensure_local_postgres_ready(storage_url: str) -> None:
    """Ensure local Postgres server is available when storage host is local."""
    host, port, user, password, dbname = _parse_storage_url(storage_url)
    if not _is_local_host(host):
        return
    if _is_tcp_open(host, port):
        return

    pg_bin = _detect_pg_bin_dir()
    base_dir = Path(
        os.environ.get(
            "PFF_LOCAL_POSTGRES_DATA_DIR",
            str(Path.home() / ".local" / "share" / "pff" / "postgres"),
        )
    )
    pgdata = base_dir / "pgdata"
    logfile = Path(
        os.environ.get(
            "PFF_LOCAL_POSTGRES_LOG_FILE",
            str(base_dir / "postgres.log"),
        )
    )

    _ensure_initialized(pg_bin, pgdata, user)
    _ensure_started(pg_bin, pgdata, host, port, logfile)
    _ensure_role_and_db(
        pg_bin, host=host, port=port, user=user, password=password, dbname=dbname
    )

    if not _is_tcp_open(host, port):
        raise RuntimeError(
            f"Embedded Postgres bootstrap failed. Endpoint still unavailable: {host}:{port}"
        )
