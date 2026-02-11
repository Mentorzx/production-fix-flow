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
    ) -> list[Any]: ...

    @abstractmethod
    def submit(self, fn: Callable[..., Any], *args: Any) -> Any: ...

    @abstractmethod
    def shutdown(self) -> None: ...
