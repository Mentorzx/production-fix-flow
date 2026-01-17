"""Tests for HPO domain models and KGE model resolution."""

import pytest
from pff.domain.hpo.models import (
    KGE_MODEL_DSLFM,
    KGE_MODEL_ALIASES,
    resolve_kge_model,
)


class TestKGEModelConstants:
    """Tests for KGE model constants."""

    def test_kge_model_dslfm_constant(self):
        """Verify DSLFM constant is defined."""
        assert KGE_MODEL_DSLFM is not None
        assert isinstance(KGE_MODEL_DSLFM, str)

    def test_kge_model_dslfm_value(self):
        """Verify DSLFM constant value."""
        assert KGE_MODEL_DSLFM.lower() == "dslfm"

    def test_kge_model_aliases_is_dict(self):
        """Verify aliases is a dictionary."""
        assert isinstance(KGE_MODEL_ALIASES, dict)

    def test_kge_model_aliases_contains_dslfm(self):
        """Verify aliases contains dslfm variants."""
        assert "dslfm" in KGE_MODEL_ALIASES
        assert "dslfm-kgc" in KGE_MODEL_ALIASES
        assert "dslfm_kgc" in KGE_MODEL_ALIASES


class TestResolveKGEModel:
    """Tests for resolve_kge_model function."""

    def test_resolve_kge_model_dslfm(self):
        """Verify DSLFM model resolution."""
        result = resolve_kge_model("dslfm")
        assert result == KGE_MODEL_DSLFM

    def test_resolve_kge_model_dslfm_kgc_hyphen(self):
        """Verify DSLFM-KGC with hyphen resolution."""
        result = resolve_kge_model("dslfm-kgc")
        assert result == KGE_MODEL_DSLFM

    def test_resolve_kge_model_dslfm_kgc_underscore(self):
        """Verify DSLFM_KGC with underscore resolution."""
        result = resolve_kge_model("dslfm_kgc")
        assert result == KGE_MODEL_DSLFM

    def test_resolve_kge_model_case_insensitive(self):
        """Verify model resolution is case insensitive."""
        result_lower = resolve_kge_model("dslfm")
        result_upper = resolve_kge_model("DSLFM")
        result_mixed = resolve_kge_model("DsLfM")
        assert result_lower == result_upper == result_mixed

    def test_resolve_kge_model_unknown_raises(self):
        """Verify unknown model raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            resolve_kge_model("unknown_model")
        assert "Unknown KGE model" in str(exc_info.value)
        assert "Valid options" in str(exc_info.value)

    def test_resolve_kge_model_none_raises(self):
        """Verify None raises AttributeError."""
        with pytest.raises(AttributeError):
            resolve_kge_model(None)

    def test_resolve_kge_model_empty_string_raises(self):
        """Verify empty string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            resolve_kge_model("")
        assert "Unknown KGE model" in str(exc_info.value)


class TestKGEModelAliasesConsistency:
    """Tests for KGE model aliases consistency."""

    def test_all_aliases_resolve_to_dslfm(self):
        """Verify all aliases resolve to DSLFM."""
        for alias, resolved in KGE_MODEL_ALIASES.items():
            assert resolved == KGE_MODEL_DSLFM

    def test_aliases_are_lowercase(self):
        """Verify all alias keys are lowercase."""
        for alias in KGE_MODEL_ALIASES.keys():
            assert alias == alias.lower()

    def test_underscore_hyphen_equivalence(self):
        """Verify underscore and hyphen variants are equivalent."""
        # Both dslfm_kgc and dslfm-kgc should resolve to same value
        assert KGE_MODEL_ALIASES.get("dslfm_kgc") == KGE_MODEL_ALIASES.get("dslfm-kgc")
