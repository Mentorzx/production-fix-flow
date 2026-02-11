"""Materializer Strategy pattern for bundle-to-native conversion.

Materializers convert ParquetBundle objects back to their native Python
representations based on the bundle's parsed_kind.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..bundles import ParquetBundle


class Materializer(ABC):
    """Base class for materializers (Strategy pattern).

    Each materializer handles a specific parsed_kind and converts
    the bundle back to its native Python representation.
    """

    @property
    @abstractmethod
    def parsed_kind(self) -> str:
        """The parsed_kind this materializer handles."""
        ...

    @abstractmethod
    def materialize(self, bundle: ParquetBundle) -> Any:
        """Convert bundle to native Python object.

        Args:
            bundle: ParquetBundle to materialize.

        Returns:
            Native Python object (DataFrame, dict, str, bytes, etc.).
        """
        ...
