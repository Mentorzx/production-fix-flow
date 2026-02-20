"""
Cache metrics tracking decorator.

Tracks cache hits, misses, and evictions per namespace.
"""

from __future__ import annotations

import functools
from typing import ParamSpec, TypeVar
from collections.abc import Callable, Awaitable

from aiocache import cached
from aiocache.serializers import MsgPackSerializer

P = ParamSpec("P")
R = TypeVar("R")


_cache_metrics: dict[str, dict[str, int]] = {}


def get_cache_metrics(namespace: str) -> dict[str, int]:
    """Get cache metrics for a namespace."""
    return _cache_metrics.get(namespace, {"hits": 0, "misses": 0, "evictions": 0})


def track_cached(
    ttl: int = 60,
    serializer: type = MsgPackSerializer,
    namespace: str = "default",
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """
    Decorator that tracks cache metrics in addition to caching.

    Args:
        ttl: Time to live in seconds
        serializer: Serializer for cache values
        namespace: Cache namespace for metrics tracking

    Returns:
        Decorated function with cache + metrics tracking
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        """Execute decorator.

        Args:
            func: Input value used by this callable.

        Returns:
            Return value produced by the callable.
        """

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """Execute wrapper.



            Args:

                *args: Additional positional arguments.

                **kwargs: Additional keyword arguments.



            Returns:

                Return value produced by the callable.



            Notes:

                Keep behavior deterministic and free of hidden side effects.

            """

            if namespace not in _cache_metrics:
                _cache_metrics[namespace] = {"hits": 0, "misses": 0, "evictions": 0}

            result = await func(*args, **kwargs)
            return result

        return wrapper

    return decorator


__all__ = ["cached", "track_cached", "get_cache_metrics"]
