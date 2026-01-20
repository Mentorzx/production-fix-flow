"""
Validation Event Observer - DEPRECATED LOCATION.

This module re-exports from the new location for backward compatibility.
Please update imports to use:
    from pff.application.services.business_service.shared import (
        ValidationObserver,
        CompositeValidationObserver,
    )
"""

from pff.application.services.business_service.shared.validation_observer import (
    CompositeValidationObserver,
    LoggingValidationObserver,
    MetricsValidationObserver,
    ValidationEvent,
    ValidationEventType,
    ValidationObserver,
)

__all__ = [
    "ValidationEventType",
    "ValidationEvent",
    "ValidationObserver",
    "LoggingValidationObserver",
    "MetricsValidationObserver",
    "CompositeValidationObserver",
]
