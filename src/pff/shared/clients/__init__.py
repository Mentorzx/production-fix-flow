"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/shared/clients/__init__.py

"""

from typing import TYPE_CHECKING, Any

from .http_client import HttpClient

if TYPE_CHECKING:
    from .http_client import API

__all__ = ["HttpClient", "API"]


def __getattr__(name: str) -> Any:
    if name == "API":
        from .http_client import API

        return API
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
