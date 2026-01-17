"""Tests for backward compatibility cleanup shim."""


def test_shim_exports_build_engine():
    """Verify shim exports build_engine for CLI compatibility."""
    from pff.infrastructure.cleanup import build_engine

    assert callable(build_engine)


def test_shim_exports_main():
    """Verify shim exports main entry point."""
    from pff.infrastructure.cleanup import main

    assert callable(main)


def test_shim_exports_api_subset():
    """Verify shim exposes the primary cleanup API symbols."""
    from pff.infrastructure import cleanup

    expected = [
        "CleanupEngine",
        "CleanupCommand",
        "CleanupStrategy",
        "StandardCleanup",
        "DeepCleanup",
        "MLCleanup",
        "ShutdownCleanup",
        "build_engine",
        "load_cleanup_config",
        "main",
    ]
    for name in expected:
        assert hasattr(cleanup, name), f"Missing export: {name}"
