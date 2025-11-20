import threading
from typing import Any

class ThreadContext:
    """
    A wrapper around threading.local() to manage thread-local state safely.
    """
    def __init__(self):
        self._local = threading.local()

    def set(self, key: str, value: Any) -> None:
        setattr(self._local, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self._local, key, default)

    def has(self, key: str) -> bool:
        return hasattr(self._local, key)

    def clear(self) -> None:
        self._local = threading.local()

# Global instance for general use
context = ThreadContext()
