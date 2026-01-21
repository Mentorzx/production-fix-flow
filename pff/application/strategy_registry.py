"""Registry for KGC training strategies (plugin-friendly)."""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from typing import TYPE_CHECKING

from pff.shared import logger
from pff.shared.factory import GenericFactory

from .errors import StrategyResolutionError

if TYPE_CHECKING:
    from .learn_use_case import TrainingStrategy


class KGCStrategyRegistry(GenericFactory["TrainingStrategy"]):
    """Registry of KGC training strategies with entrypoint discovery."""

    def __init__(self) -> None:
        self._strategies: dict[str, type[TrainingStrategy]] = {}
        self._entrypoints_loaded = False

    def register(
        self,
        name: str,
        strategy_class: Any = None,
    ) -> Any:
        """Register a strategy class or act as a decorator."""
        if strategy_class is not None:
            self._strategies[name] = strategy_class
            return strategy_class

        def decorator(cls: Any) -> Any:
            self._strategies[name] = cls
            return cls

        return decorator

    def _load_entrypoints(self) -> None:
        if self._entrypoints_loaded:
            return
        try:
            entry_points = metadata.entry_points()
            group = (
                entry_points.select(group="pff.kgc_strategies")
                if hasattr(entry_points, "select")
                else entry_points.get("pff.kgc_strategies", [])
            )
            for ep in group:
                try:
                    cls = ep.load()
                except Exception as exc:
                    logger.warning(f"Failed to load KGC strategy entrypoint '{ep.name}': {exc}")
                    continue
                if ep.name not in self._strategies:
                    self._strategies[ep.name] = cls
        finally:
            self._entrypoints_loaded = True

    def get(self, name: str) -> type[TrainingStrategy]:
        """Return the strategy class for a given name."""
        self._load_entrypoints()
        strategy = self._strategies.get(name)
        if strategy is None:
            available = ", ".join(sorted(self._strategies)) or "<none>"
            raise StrategyResolutionError(
                f"Unknown training strategy '{name}'. Available: {available}"
            )
        return strategy

    def create(self, name: str, *args, **kwargs) -> TrainingStrategy:
        """Instantiate a strategy by name."""
        strategy_class = self.get(name)
        return strategy_class(*args, **kwargs)

    def available(self) -> list[str]:
        """List available strategy names."""
        self._load_entrypoints()
        return sorted(self._strategies)


_REGISTRY: KGCStrategyRegistry | None = None


def get_strategy_registry() -> KGCStrategyRegistry:
    """Return the global strategy registry singleton."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = KGCStrategyRegistry()
    return _REGISTRY
