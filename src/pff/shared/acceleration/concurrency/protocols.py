"""Base executor protocol and abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

Args = tuple[Any, ...]


class BaseExecutor(ABC):
    """Abstract base class for all executors."""

    @abstractmethod
    def map(
        self,
        fn: Callable[..., Any],
        args_list: Iterable[Args],
        *,
        desc: str | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Execute function across argument tuples."""
        ...

    @abstractmethod
    def submit(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Submit a single callable for asynchronous execution."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources held by the executor."""
        ...
