"""FAISS import helper with warning suppression for third-party deprecations."""

from __future__ import annotations

import warnings
from typing import Any


def import_faiss() -> tuple[Any | None, bool]:
    """Import FAISS while suppressing known third-party deprecation warnings."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=r".*numpy\\.core\\._multiarray_umath.*",
            )
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=r".*SwigPyPacked.*__module__.*",
            )
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=r".*SwigPyObject.*__module__.*",
            )
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=r".*swigvarlink.*__module__.*",
            )
            import faiss

        return faiss, True
    except Exception as exc:
        raise RuntimeError(f"Failed to import FAISS: {exc} (Error Loud).") from exc
