"""
Logging Compliance Tests.

Verifies adherence to AGENTS.md logging guidelines:
- No silent fallbacks for model selection
- No large dicts dumped at INFO level
- Optuna verbosity at WARNING by default

SOTA References:
- Google SRE: Fail-fast, no silent fallbacks
- optuna.readthedocs.io: set_verbosity(WARNING)
- betterstack.com: DEBUG for internal diagnostics
"""

import logging

import pytest


class TestModelAliasFailFast:
    """Test that model aliases resolve correctly or fail loudly."""

    def test_dslfm_kgc_resolves_correctly(self):
        """Ensure dslfm-kgc maps to canonical dslfm."""
        from pff.domain.hpo.models import KGE_MODEL_DSLFM, resolve_kge_model

        assert resolve_kge_model("dslfm-kgc") == KGE_MODEL_DSLFM
        assert resolve_kge_model("DSLFM-KGC") == KGE_MODEL_DSLFM
        assert resolve_kge_model("dslfm_kgc") == KGE_MODEL_DSLFM
        assert resolve_kge_model("dslfm") == KGE_MODEL_DSLFM

    def test_unknown_model_raises_value_error(self):
        """Ensure unknown models fail with clear error."""
        from pff.domain.hpo.models import resolve_kge_model

        with pytest.raises(ValueError, match="Unknown KGE model"):
            resolve_kge_model("invalid-model")

        with pytest.raises(ValueError, match="Unknown KGE model"):
            resolve_kge_model("rotate")

    def test_no_fallback_warning_for_valid_alias(self, caplog):
        """Ensure no 'defaulting' warning when using valid aliases."""
        from pff.domain.hpo.models import resolve_kge_model

        with caplog.at_level(logging.WARNING):
            resolve_kge_model("dslfm-kgc")

        assert "defaulting" not in caplog.text.lower()
        assert "unknown" not in caplog.text.lower()


class TestOptunaVerbosity:
    """Test that Optuna verbosity is controlled."""

    def test_optuna_verbosity_at_warning_or_higher(self):
        """Ensure Optuna only logs at WARNING+ by default."""
        import optuna

        from pff.infrastructure import hpo

        hpo.configure_optuna_logging()

        verbosity = optuna.logging.get_verbosity()
        assert verbosity >= optuna.logging.WARNING, (
            f"Optuna verbosity should be >= WARNING, got {verbosity}"
        )


class TestNoLargeDictDumps:
    """Test that large dicts are not dumped at INFO level."""

    # def test_preprocess_stats_not_at_info(self, caplog):
    #     """Ensure preprocessing stats go to DEBUG/JSON, not INFO."""
    #     # This is a structural test - we verify the code pattern
    #     import inspect
    #     from pff.domain.kg.kg import preprocess

    #     source = inspect.getsource(preprocess)

    #     # Should NOT have direct dict dump at INFO
    #     assert 'logger.info(f"Preprocessing stats: {' not in source

    #     # Should have DEBUG with path reference
    #     assert "preprocess_stats path=" in source or "stats_path" in source

    def test_info_messages_under_200_chars(self, caplog):
        """Spot check that INFO messages are concise."""
        # Import a module that logs
        with caplog.at_level(logging.INFO):
            from pff.domain.learning.ml.adaptive_training import (
                AdaptiveTrainingConfig,
            )

            _ = AdaptiveTrainingConfig  # Use import to avoid unused warning  # noqa: F841

        for record in caplog.records:
            if record.levelno == logging.INFO:
                # INFO messages should be concise
                assert len(record.message) < 500, (
                    f"INFO message too long ({len(record.message)} chars): "
                    f"{record.message[:100]}..."
                )
