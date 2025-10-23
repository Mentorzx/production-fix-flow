from pathlib import Path

import orjson
import redis
from kombu import Exchange, Queue
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).parents[1]


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
        PYC_CLAUSE_DIR (Path): Path for PyClause-related outputs.
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
        CELERY_TASK_REJECT_ON_WORKER_LOST (bool): If true, tasks are rejected if the worker process is lost.
        CELERY_TASK_DEFAULT_QUEUE (str): The default queue for Celery tasks.
        CELERY_TASK_QUEUES (list[Queue]): Configuration for Celery task queues.
        CELERY_TASK_AUTODISCOVER (list[str]): List of modules for Celery to auto-discover tasks from.
    Methods:
        coerce_accept_content(cls, v): A class method validator that coerces the
            `CELERY_ACCEPT_CONTENT` value from a string (JSON or comma-separated)
            into a list of strings.
    """

    ROOT_DIR: Path = ROOT_DIR
    DATA_DIR: Path = ROOT_DIR / "data"
    OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
    LOGS_DIR: Path = ROOT_DIR / "logs"
    MODELS_DIR: Path = DATA_DIR / "models"
    PYCLAUSE_DIR: Path = OUTPUTS_DIR / "pyclause"
    CONFIG_DIR: Path = ROOT_DIR / "config"
    CACHE_DIR: Path = ROOT_DIR / ".cache"
    PATTERNS_DIR: Path = ROOT_DIR / "pff" / "validators" / "patterns"
    UTILS_DIR: Path = ROOT_DIR / "pff" / "utils"

    DEFAULT_MANIFEST_PATH: Path = DATA_DIR / "manifest.yaml"
    PIP_CACHE_DIR: Path = Path.home() / ".cache" / "pip"

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    USE_REDIS: bool = True  # Enable/disable Redis (set to False on Windows or if Redis unavailable)

    @property
    def REDIS_URL(self) -> str:
        """Redis connection URL for redis-py."""
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    # API Settings
    API_VERSION: str = "1.1.0"

    # Redis databases for different purposes
    REDIS_DB_EXECUTIONS: int = 5
    REDIS_DB_PUBSUB: int = 2
    REDIS_DB_CACHE: int = 0

    # CORS settings
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])

    # WebSocket settings
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_CONNECTION_TIMEOUT: int = 300

    # Cache settings
    CACHE_WARMUP: bool = False
    CACHE_TTL_DEFAULT: int = 3600

    # Batch processing
    BATCH_SIZE_DEFAULT: int = 10
    BATCH_TIMEOUT: int = 300

    # File size limits
    MAX_UPLOAD_SIZE: int = 104857600  # 100MB

    # API Security
    SECRET_KEY: str = "CHANGE_ME_32_BYTES_RANDOM"
    API_KEY: str = "CHANGE_ME_API_KEY"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # PostgreSQL Configuration
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "pff_production"
    POSTGRES_USER: str = "pff_user"
    POSTGRES_PASSWORD: str = "CHANGE_ME_POSTGRES_PASSWORD"

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
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    @property
    def CELERY_RESULT_BACKEND(self) -> str:
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

    # Pydantic -> .env
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    @field_validator("CELERY_ACCEPT_CONTENT", mode="before")
    @classmethod
    def coerce_accept_content(cls, v):
        if isinstance(v, (list, tuple)):
            return list(v)
        try:
            parsed = orjson.loads(v)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [s.strip() for s in str(v).split(",") if s.strip()]


settings = Settings()
rds = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=5)
