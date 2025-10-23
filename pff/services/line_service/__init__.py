"""
LineService - High-level façade for HTTP operations

Refactored in Sprint 4 into specialized modules for better maintainability.
"""

from .base import LineServiceBase, capture_collector
from .queries import LineServiceQueries
from .mutations import LineServiceMutations
from .cancellation import LineServiceCancellation


class LineService(
    LineServiceCancellation,
    LineServiceMutations,
    LineServiceQueries,
    LineServiceBase,
):
    """
    High-level façade around all async HTTP operations used by PFF projects.
    Enhanced with caching, circuit breaking, and request coalescing patterns.

    Combines functionality from:
    - LineServiceBase: Infrastructure (circuit breakers, coalescing, helpers)
    - LineServiceQueries: GET operations (enquiry, party, contract, product)
    - LineServiceMutations: POST/PATCH/DELETE operations (status updates, create, delete)
    - LineServiceCancellation: Cancellation operations (soft-cancel)

    This class maintains 100% backward compatibility with the original LineService.
    All public methods and attributes remain unchanged.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize LineService with all functionality.

        Args:
            **kwargs: Optional arguments passed to HttpClient initialization.
        """
        # Call base class init (only once via MRO)
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """String representation of LineService."""
        return f"<LineService(breakers={len([b for b in dir(self) if 'breaker' in b])})>"


__all__ = ["LineService", "capture_collector"]
