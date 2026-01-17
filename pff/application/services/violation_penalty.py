"""
Violation Penalty Calculator - DEPRECATED LOCATION.

This module re-exports from the new location for backward compatibility.
Please update imports to use:
    from pff.application.services.business_service.shared import (
        PenaltyConfig,
        ViolationPenaltyCalculator,
    )
"""

from pff.application.services.business_service.shared.violation_penalty import (
    PenaltyConfig,
    ViolationPenaltyCalculator,
)

__all__ = [
    "PenaltyConfig",
    "ViolationPenaltyCalculator",
]
