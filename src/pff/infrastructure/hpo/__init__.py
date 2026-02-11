"""Infrastructure adapters for HPO."""

from __future__ import annotations


def configure_optuna_logging(level: int | None = None) -> None:
    """Set Optuna verbosity to WARNING by default (configurable)."""
    import optuna

    target = optuna.logging.WARNING if level is None else int(level)
    optuna.logging.set_verbosity(target)


__all__ = ["configure_optuna_logging"]
