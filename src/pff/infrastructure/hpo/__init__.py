"""Infrastructure adapters for HPO."""

from __future__ import annotations


def configure_optuna_logging(level: int | None = None) -> None:
    """Silence Optuna's internal logger to avoid format leaks.

    Redirects Optuna messages through PFF's loguru pipeline so they
    appear with a consistent format instead of raw ``[I ...]`` lines.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    optuna_logger = optuna.logging.get_logger("optuna")
    optuna_logger.propagate = False
    for h in list(optuna_logger.handlers):
        optuna_logger.removeHandler(h)


configure_optuna_logging()

__all__ = ["configure_optuna_logging"]
