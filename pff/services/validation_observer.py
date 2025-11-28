"""
Validation Event Observer - DEPRECATED LOCATION.

This module re-exports from the new location for backward compatibility.
Please update imports to use:
    from pff.services.business_service.shared import ValidationObserver, CompositeValidationObserver
"""

from pff.services.business_service.shared.validation_observer import (
    ValidationEventType,
    ValidationEvent,
    ValidationObserver,
    LoggingValidationObserver,
    MetricsValidationObserver,
    CompositeValidationObserver,
)

__all__ = [
    "ValidationEventType",
    "ValidationEvent",
    "ValidationObserver",
    "LoggingValidationObserver",
    "MetricsValidationObserver",
    "CompositeValidationObserver",
]
