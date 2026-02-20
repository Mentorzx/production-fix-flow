"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/ops/test_cleanup_strategies.py

"""

from pff.infrastructure.cleanup.strategies.builtin import (
    DeepCleanup,
    MLCleanup,
    ShutdownCleanup,
    StandardCleanup,
)


def test_standard_cleanup_builds_commands():
    """Execute test standard cleanup builds commands."""

    commands = StandardCleanup().build_commands()
    assert commands, "Standard cleanup should build commands"


def test_deep_cleanup_extends_standard():
    """Execute test deep cleanup extends standard."""

    std_len = len(StandardCleanup().build_commands())
    deep_len = len(DeepCleanup().build_commands())
    assert deep_len > std_len


def test_deep_cleanup_includes_kg_tables():
    """Execute test deep cleanup includes kg tables."""

    labels = [cmd.label for cmd in DeepCleanup().build_commands()]
    assert any("Knowledge Graph" in label for label in labels)
    assert any("mappings" in label for label in labels)
    assert any("embeddings" in label for label in labels)
    assert any("Optuna" in label for label in labels)


def test_ml_cleanup_focuses_on_ml():
    """Execute test ml cleanup focuses on ml."""

    labels = [cmd.label for cmd in MLCleanup().build_commands()]
    assert any("ML" in label or "DSLFM" in label for label in labels)


def test_shutdown_cleanup_minimal():
    """Execute test shutdown cleanup minimal."""

    labels = [cmd.label for cmd in ShutdownCleanup().build_commands()]
    assert len(labels) == 2
