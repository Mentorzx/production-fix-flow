"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/shared/core/config.py

"""

import os
import sys
import warnings
from pathlib import Path

import orjson
import redis
from kombu import Exchange, Queue
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parents[2]


def _find_repo_root(start: Path) -> Path:
    """Best-effort repo root detection.

    In a source checkout, we use the directory containing pyproject.toml.
    In an installed package, this typically falls back to site-packages.
    """

    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start.parent


ROOT_DIR = _find_repo_root(PACKAGE_DIR)


def _detect_clean_mode(argv: list[str]) -> bool:
    """Execute detect clean mode.



    Args:

        argv: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    for arg in argv[1:]:
        if arg.startswith("-"):
            continue
        return arg == "clean"
    return False


if os.environ.get("PFF_CLEAN_MODE") is None and _detect_clean_mode(sys.argv):
    os.environ["PFF_CLEAN_MODE"] = "1"


def apply_permanent_configurations() -> None:
    """Apply runtime warning filters and platform-specific defaults."""
    if os.environ.get("PFF_CLEAN_MODE") != "1":
        pycache_prefix = ROOT_DIR / "outputs" / ".cache" / "pycache"
        try:
            pycache_prefix.mkdir(parents=True, exist_ok=True)
            os.environ["PYTHONPYCACHEPREFIX"] = str(pycache_prefix)
        except Exception:
            os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    warnings.filterwarnings(
        "ignore",
        message=".*Pipeline instance is not fitted yet.*",
        category=FutureWarning,
    )
    warnings.filterwarnings("ignore", category=UserWarning, module="distributed")

    if sys.platform == "win32":
        try:
            import asyncio

            if not isinstance(
                asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy
            ):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except (ImportError, RuntimeError):
            pass


class Settings(BaseSettings):
    """Manages application-wide configuration settings using Pydantic.
    This class centralizes all configuration parameters for the application. It
    inherits from `pydantic_settings.BaseSettings`, which allows it to automatically
    read settings from environment variables and a `.env` file located at the
    project root.
    The settings are structured into logical groups:
    - Directory Paths: Core application directories.
    - Redis Configuration: Connection details for the Redis server.
    - Celery Configuration: Settings for the Celery distributed task queue.
    Attributes:
        ROOT_DIR (Path): The absolute path to the project's root directory.
        DATA_DIR (Path): Path to the directory for storing data files.
        OUTPUTS_DIR (Path): Path to the directory for storing output files.
        LOGS_DIR (Path): Path to the directory for log files.
        MODELS_DIR (Path): Path to the directory for storing machine learning models.
        CONFIG_DIR (Path): Path to the configuration directory.
        CACHE_DIR (Path): Path to the application's cache directory.
        DEFAULT_MANIFEST_PATH (Path): Default path to the data manifest file.
        PIP_CACHE_DIR (Path): Path to the system's pip cache directory.
        REDIS_HOST (str): Hostname for the Redis server.
        REDIS_PORT (int): Port number for the Redis server.
        CELERY_BROKER_URL (str): Computed property for the Celery broker URL.
        CELERY_RESULT_BACKEND (str): Computed property for the Celery result backend URL.
        CELERY_ACCEPT_CONTENT (list[str]): A list of accepted content types for Celery.
        CELERY_TASK_SERIALIZER (str): The default serializer for Celery tasks.
        CELERY_RESULT_SERIALIZER (str): The default serializer for Celery task results.
         CELERY_TIMEZONE (str): The timezone used by Celery.
         CELERY_TASK_ACKS_LATE (bool): If true, tasks are acknowledged after execution.
         CELERY_TASK_REJECT_ON_WORKER_LOST (bool): If true, tasks are rejected if the
             worker process is lost.
         CELERY_TASK_DEFAULT_QUEUE (str): The default queue for Celery tasks.
         CELERY_TASK_QUEUES (list[Queue]): Configuration for Celery task queues.
         CELERY_TASK_AUTODISCOVER (list[str]): List of modules for Celery to
             auto-discover tasks from.
    Methods:
        coerce_accept_content(cls, v): A class method validator that coerces the
            `CELERY_ACCEPT_CONTENT` value from a string (JSON or comma-separated)
            into a list of strings.
    """

    ROOT_DIR: Path = ROOT_DIR
    PACKAGE_DIR: Path = PACKAGE_DIR
    DATA_DIR: Path = ROOT_DIR / "data"
    OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
    LOGS_DIR: Path = ROOT_DIR / "logs"
    MODELS_DIR: Path = DATA_DIR / "models"
    CONFIG_DIR: Path = ROOT_DIR / "config"
    CACHE_DIR: Path = OUTPUTS_DIR / ".cache"
    PATTERNS_DIR: Path = PACKAGE_DIR / "domain" / "kg" / "patterns"
    UTILS_DIR: Path = PACKAGE_DIR / "shared"

    DEFAULT_MANIFEST_PATH: Path = DATA_DIR / "manifest.yaml"
    PIP_CACHE_DIR: Path = Path.home() / ".cache" / "pip"

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    USE_REDIS: bool = True

    @property
    def REDIS_URL(self) -> str:
        """Redis connection URL for redis-py."""
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    API_VERSION: str = "1.1.0"

    REDIS_DB_EXECUTIONS: int = 5
    REDIS_DB_PUBSUB: int = 2
    REDIS_DB_CACHE: int = 0

    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])

    WS_HEARTBEAT_INTERVAL: int = 30
    WS_CONNECTION_TIMEOUT: int = 300

    CACHE_WARMUP: bool = False
    CACHE_TTL_DEFAULT: int = 3600
    PFF_ACCELERATOR: str = "auto"

    BATCH_SIZE_DEFAULT: int = 10
    BATCH_TIMEOUT: int = 300

    MAX_UPLOAD_SIZE: int = 104857600

    SECRET_KEY: str = "CHANGE_ME_SURELY_IN_PRODUCTION_32_CHAR_MIN"
    API_KEY: str = "CHANGE_ME_SURELY_16_CHAR_MIN"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "pff_db"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "CHANGE_ME_PASSWORD"

    celery_broker_url_override: str | None = Field(
        default=None, validation_alias="CELERY_BROKER_URL"
    )
    celery_result_backend_override: str | None = Field(
        default=None, validation_alias="CELERY_RESULT_BACKEND"
    )

    @property
    def DATABASE_URL(self) -> str:
        """Async PostgreSQL connection URL for asyncpg."""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def DATABASE_URL_ASYNC(self) -> str:
        """Async PostgreSQL connection URL for asyncpg (explicit asyncpg driver)."""
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def CELERY_BROKER_URL(self) -> str:
        """Execute CELERY BROKER URL.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if self.celery_broker_url_override:
            return self.celery_broker_url_override
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        """Execute CELERY RESULT BACKEND.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if self.celery_result_backend_override:
            return self.celery_result_backend_override
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/1"

    CELERY_ACCEPT_CONTENT: list[str] = Field(default_factory=lambda: ["json"])
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_TIMEZONE: str = "UTC"
    CELERY_TASK_ACKS_LATE: bool = True
    CELERY_TASK_REJECT_ON_WORKER_LOST: bool = True
    CELERY_TASK_DEFAULT_QUEUE: str = "default"
    CELERY_TASK_QUEUES: list[Queue] = [
        Queue("default", Exchange("default"), routing_key="default"),
        Queue("high", Exchange("high"), routing_key="high"),
        Queue("low", Exchange("low"), routing_key="low"),
    ]
    CELERY_TASK_AUTODISCOVER: list[str] = ["pff"]

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    @field_validator("CELERY_ACCEPT_CONTENT", mode="before")
    @classmethod
    def coerce_accept_content(cls, v):
        """Execute coerce accept content.



        Args:

            v: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if isinstance(v, (list, tuple)):
            return list(v)
        try:
            parsed = orjson.loads(v)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [s.strip() for s in str(v).split(",") if s.strip()]

    @field_validator("SECRET_KEY", "API_KEY", "POSTGRES_PASSWORD", mode="after")
    @classmethod
    def ensure_not_placeholder(cls, value: str) -> str:
        """Execute ensure not placeholder.



        Args:

            value: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if os.environ.get("PFF_CLEAN_MODE") == "1":
            return value
        if "CHANGE_ME" in value:
            raise ValueError(
                "Sensitive configuration values must be provided via "
                "environment variables or config files."
            )
        return value

    @field_validator("PFF_ACCELERATOR", mode="after")
    @classmethod
    def validate_accelerator(cls, value: str) -> str:
        """Normalize the runtime accelerator contract."""
        normalized = str(value).strip().lower()
        if normalized not in {"auto", "cpu", "cuda"}:
            raise ValueError("PFF_ACCELERATOR must be one of: auto, cpu, cuda.")
        return normalized

    @property
    def HPO_CONFIG(self) -> dict:
        """Load HPO config."""
        if OPTIMIZATION_CONFIG_PATH.exists():
            try:
                import polars as pl

                from pff.shared.core.file_manager import FileManager

                cfg = FileManager.read(OPTIMIZATION_CONFIG_PATH, return_native=True)
                if isinstance(cfg, pl.DataFrame):
                    if not cfg.is_empty():
                        return cfg.to_dicts()[0]
                    return {}
                return cfg or {}
            except Exception:
                return {}
        return {}

    @property
    def MODEL_CONFIG(self) -> dict:
        """Load DSLFM config."""
        if DSLFM_CONFIG_PATH.exists():
            try:
                import polars as pl

                from pff.shared.core.file_manager import FileManager

                cfg = FileManager.read(DSLFM_CONFIG_PATH, return_native=True)
                if isinstance(cfg, pl.DataFrame):
                    return cfg.to_dict(as_series=False)
                return cfg or {}
            except Exception:
                return {}
        return {}


settings = Settings()
if os.environ.get("PFF_CLEAN_MODE") == "1":
    settings.CACHE_DIR = settings.ROOT_DIR / "outputs" / ".cache"
    settings.LOGS_DIR = settings.ROOT_DIR / "logs"
_redis_clients: dict[tuple[int, bool], redis.Redis] = {}


def get_redis_client(db: int = 0, *, decode_responses: bool = True) -> redis.Redis:
    """Return a cached Redis client using Settings for host/port."""
    key = (db, decode_responses)
    client = _redis_clients.get(key)
    if client is not None:
        return client
    from redis.connection import ConnectionPool

    pool = ConnectionPool(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=db,
        max_connections=50,
        socket_timeout=5,
        socket_connect_timeout=2,
        decode_responses=decode_responses,
    )
    client = redis.Redis(connection_pool=pool)
    _redis_clients[key] = client
    return client


CONFIG_ROOT: Path = settings.CONFIG_DIR

MODELS_DIR: Path = CONFIG_ROOT / "models"
ENSEMBLE_CONFIG_PATH: Path = MODELS_DIR / "ensemble.yaml"
PC_CONFIG_PATH: Path = MODELS_DIR / "pc.yaml"
AUTOFEEDING_CONFIG_PATH: Path = MODELS_DIR / "autofeeding.yaml"
DSLFM_TRAINING_CONFIG_PATH: Path = MODELS_DIR / "dslfm.yaml"
DSLFM_CONFIG_PATH: Path = MODELS_DIR / "dslfm.yaml"
KG_PIPELINE_CONFIG_PATH: Path = MODELS_DIR / "kg.yaml"
RULE_FILTER_CONFIG_PATH: Path = KG_PIPELINE_CONFIG_PATH

HPO_DIR: Path = CONFIG_ROOT / "hpo"
ADAPTIVE_LEARNING_CONFIG_PATH: Path = HPO_DIR / "adaptive_learning.yaml"
OPTIMIZATION_CONFIG_PATH: Path = HPO_DIR / "optimization.yaml"
ENSEMBLE_HPO_CONFIG_PATH: Path = HPO_DIR / "optimization.yaml"
RULE_FILTER_HPO_CONFIG_PATH: Path = KG_PIPELINE_CONFIG_PATH

INFRA_DIR: Path = CONFIG_ROOT / "infra"
API_HOSTS_CONFIG_PATH: Path = INFRA_DIR / "api_hosts.yaml"
API_HOSTS_TEMPLATE_PATH: Path = INFRA_DIR / "api_hosts.yaml.example"
POSTGRES_CONFIG_PATH: Path = INFRA_DIR / "postgres.yaml"
SEQUENCES_CONFIG_PATH: Path = INFRA_DIR / "sequences.yaml"
VALIDATOR_CONFIG_PATH: Path = INFRA_DIR / "validator.yaml"
INGESTION_CONFIG_PATH: Path = INFRA_DIR / "ingestion.yaml"
LINE_SERVICE_CONFIG_PATH: Path = INFRA_DIR / "line_service.yaml"
PERFORMANCE_CONFIG_PATH: Path = INFRA_DIR / "performance.yaml"
CLEANUP_CONFIG_PATH: Path = INFRA_DIR / "cleanup.yaml"
CACHE_CONFIG_PATH: Path = INFRA_DIR / "cache.yaml"
ACCELERATION_CONFIG_PATH: Path = INFRA_DIR / "acceleration.yaml"

OBSERVABILITY_DIR: Path = CONFIG_ROOT / "observability"
EXPLAINABILITY_CONFIG_PATH: Path = OBSERVABILITY_DIR / "explainability.yaml"
TRAINING_METRICS_CONFIG_PATH: Path = OBSERVABILITY_DIR / "training_metrics.yaml"
METRICS_IMPROVEMENT_CONFIG_PATH: Path = OBSERVABILITY_DIR / "metrics_improvement.json"

AUDIT_DIR: Path = CONFIG_ROOT / "audit"
AUDIT_CONFIG_PATH: Path = AUDIT_DIR / "audit.yaml"
AUDIT_REPORT_SCHEMA_V1_PATH: Path = AUDIT_DIR / "audit_report.schema.v1.json"

__all__ = [
    "Settings",
    "settings",
    "get_redis_client",
    "CONFIG_ROOT",
    "CLEANUP_CONFIG_PATH",
    "MODELS_DIR",
    "ENSEMBLE_CONFIG_PATH",
    "AUTOFEEDING_CONFIG_PATH",
    "DSLFM_TRAINING_CONFIG_PATH",
    "DSLFM_CONFIG_PATH",
    "RULE_FILTER_CONFIG_PATH",
    "KG_PIPELINE_CONFIG_PATH",
    "HPO_DIR",
    "ADAPTIVE_LEARNING_CONFIG_PATH",
    "OPTIMIZATION_CONFIG_PATH",
    "ENSEMBLE_HPO_CONFIG_PATH",
    "RULE_FILTER_HPO_CONFIG_PATH",
    "INFRA_DIR",
    "API_HOSTS_CONFIG_PATH",
    "API_HOSTS_TEMPLATE_PATH",
    "LINE_SERVICE_CONFIG_PATH",
    "POSTGRES_CONFIG_PATH",
    "SEQUENCES_CONFIG_PATH",
    "VALIDATOR_CONFIG_PATH",
    "INGESTION_CONFIG_PATH",
    "PERFORMANCE_CONFIG_PATH",
    "CACHE_CONFIG_PATH",
    "ACCELERATION_CONFIG_PATH",
    "OBSERVABILITY_DIR",
    "EXPLAINABILITY_CONFIG_PATH",
    "TRAINING_METRICS_CONFIG_PATH",
    "METRICS_IMPROVEMENT_CONFIG_PATH",
    "AUDIT_DIR",
    "AUDIT_CONFIG_PATH",
    "AUDIT_REPORT_SCHEMA_V1_PATH",
]
