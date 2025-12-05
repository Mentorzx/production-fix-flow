"""Tests for backward compatibility cleanup shim."""


def test_shim_exports_build_engine():
    """Verify shim exports build_engine for CLI compatibility."""
    from pff.utils.cleanup import build_engine

    assert callable(build_engine)


def test_shim_exports_main():
    """Verify shim exports main entry point."""
    from pff.utils.cleanup import main

    assert callable(main)


def test_shim_exports_api_subset():
    """Verify shim exposes the primary cleanup API symbols."""
    import pff.utils.cleanup as cleanup

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
