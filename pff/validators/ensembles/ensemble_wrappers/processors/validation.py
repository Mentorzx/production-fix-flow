"""
Validation utilities for symbolic feature processors.

This module implements the Command pattern for validation operations
and provides comprehensive data validation capabilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import ValidationCommand


class ValidateViolationsListCommand(ValidationCommand):
    """Command to validate violations list format."""

    def validate(self, data: Any, context: dict[str, Any]) -> bool:
        """Validate violations list format."""
        if not isinstance(data, (list, tuple)):
            return False

        n_rules = context.get("n_rules", 0)
        if n_rules == 0:
            # Nothing to validate against
            return True

        try:
            for v in data:
                arr = np.asarray(v)
                if arr.ndim != 1:
                    return False
                if arr.shape[0] != n_rules:
                    return False
            return True
        except Exception:
            return False


class ValidateFeatureMatrixCommand(ValidationCommand):
    """Command to validate feature matrix format."""

    def validate(self, data: Any, context: dict[str, Any]) -> bool:
        """Validate feature matrix format."""
        try:
            if isinstance(data, list):
                # List of arrays - convert to single array
                if not data:
                    return False
                arr = np.vstack(data)
            else:
                arr = np.asarray(data)

            # Check dimensions
            if arr.ndim != 2:
                return False

            # Check data type
            if arr.dtype not in (np.int8, np.int16, np.int32, np.int64, np.float32, np.float64):
                # Try to convert to int8
                try:
                    arr = arr.astype(np.int8)
                except (ValueError, TypeError):
                    return False

            # Check for invalid values
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                return False

            # Store validated array back to context
            context["validated_data"] = arr
            return True

        except Exception:
            return False


class ValidateConfigurationCommand(ValidationCommand):
    """Command to validate processor configuration."""

    def validate(self, data: Any, context: dict[str, Any]) -> bool:
        """Validate processor configuration."""
        config = context.get("config", {})
        if not config:
            return False

        # Check required fields
        required_fields = ["rules_path"]
        for field in required_fields:
            if field not in config:
                return False

        # Check field types and ranges
        if "min_confidence_threshold" in config:
            threshold = config["min_confidence_threshold"]
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                return False

        if "n_groups" in config:
            n_groups = config["n_groups"]
            if not isinstance(n_groups, int) or n_groups <= 0:
                return False

        if "boost_factor" in config:
            boost_factor = config["boost_factor"]
            if not isinstance(boost_factor, (int, float)) or boost_factor <= 0:
                return False

        return True


class ValidateRulesCommand(ValidationCommand):
    """Command to validate rules format."""

    def validate(self, data: Any, context: dict[str, Any]) -> bool:
        """Validate rules format."""
        if not isinstance(data, list):
            return False

        if not data:
            # Empty rules list is valid
            return True

        try:
            for i, rule in enumerate(data):
                if not isinstance(rule, dict):
                    return False

                # Check required structure
                if "head" not in rule or "body" not in rule:
                    return False

                # Check confidence if present
                if "confidence" in rule:
                    confidence = rule["confidence"]
                    if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                        return False

                # Validate head structure
                head = rule["head"]
                if not isinstance(head, dict):
                    return False
                if not all(k in head for k in ["predicate", "subject", "object"]):
                    return False

                # Validate body structure
                body = rule["body"]
                if not isinstance(body, list):
                    return False
                for atom in body:
                    if not isinstance(atom, dict):
                        return False
                    if not all(k in atom for k in ["predicate", "subject", "object"]):
                        return False

            return True

        except Exception:
            return False


class ValidateSamplesCommand(ValidationCommand):
    """Command to validate input samples format."""

    def validate(self, data: Any, context: dict[str, Any]) -> bool:
        """Validate input samples format."""
        if not isinstance(data, list):
            return False

        if not data:
            # Empty list is valid
            return True

        try:
            for i, sample in enumerate(data):
                # Handle different sample formats
                if isinstance(sample, np.ndarray):
                    if sample.ndim != 2 or sample.shape[1] != 3:
                        return False
                elif isinstance(sample, list):
                    if not sample:
                        return False
                    for triple in sample:
                        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                            return False
                        # Check that all elements are stringifiable
                        if not all(str(elem) for elem in triple):
                            return False
                else:
                    return False

            return True

        except Exception:
            return False


class ValidationManager:
    """Manager for validation commands and operations."""

    def __init__(self):
        self.commands: dict[str, ValidationCommand] = {}
        self._register_default_commands()

    def register_command(self, name: str, command: ValidationCommand) -> None:
        """Register a validation command."""
        self.commands[name] = command

    def validate(self, name: str, data: Any, context: dict[str, Any]) -> bool:
        """Validate data using a specific command."""
        command = self.commands.get(name)
        if command:
            return command.validate(data, context)
        return False

    def validate_all(self, data: Any, context: dict[str, Any]) -> dict[str, bool]:
        """Validate data using all registered commands."""
        results = {}
        for name, command in self.commands.items():
            try:
                results[name] = command.validate(data, context)
            except Exception:
                results[name] = False
        return results

    def get_validation_summary(self, data: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Get a comprehensive validation summary."""
        results = self.validate_all(data, context)

        summary = {
            "overall_valid": all(results.values()),
            "individual_results": results,
            "failed_validations": [name for name, valid in results.items() if not valid],
            "passed_validations": [name for name, valid in results.items() if valid],
        }

        return summary

    def _register_default_commands(self) -> None:
        """Register default validation commands."""
        self.register_command("violations_list", ValidateViolationsListCommand())
        self.register_command("feature_matrix", ValidateFeatureMatrixCommand())
        self.register_command("configuration", ValidateConfigurationCommand())
        self.register_command("rules", ValidateRulesCommand())
        self.register_command("samples", ValidateSamplesCommand())


def create_validation_manager() -> ValidationManager:
    """Create a validation manager with default commands."""
    return ValidationManager()