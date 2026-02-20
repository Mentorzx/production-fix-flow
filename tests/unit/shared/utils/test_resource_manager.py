"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/test_resource_manager.py

"""

from __future__ import annotations

from pff.shared.system import resource_manager as rm


def test_get_memory_safe_workers_returns_zero_when_multiproc_unavailable(
    monkeypatch,
) -> None:
    """Execute test get memory safe workers returns zero when multiproc unavailable.



    Args:

        monkeypatch: Input value used by this callable.



    Raises:

        Exception: Propagates domain-specific failures with context.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    def _raise_permission_error():
        raise PermissionError("multiprocessing not allowed")

    monkeypatch.setattr(rm, "_MULTIPROC_AVAILABLE", None)
    monkeypatch.setattr(rm.mp, "get_context", _raise_permission_error)

    assert rm.get_memory_safe_workers(chunk_size=1000) == 0


def test_get_auto_dataloader_workers_returns_zero_for_small_dataset() -> None:
    """Execute test get auto dataloader workers returns zero for small dataset."""

    workers = rm.get_auto_dataloader_workers(
        dataset_size=10,
        batch_size=4,
        min_workers=1,
        max_workers=4,
    )
    assert workers == 0


def test_get_auto_dataloader_workers_respects_bounds(monkeypatch) -> None:
    """Execute test get auto dataloader workers respects bounds.



    Args:

        monkeypatch: Input value used by this callable.

    """

    monkeypatch.setattr(rm, "get_memory_safe_workers", lambda chunk_size: 16)
    workers = rm.get_auto_dataloader_workers(
        dataset_size=1000,
        batch_size=32,
        min_workers=2,
        max_workers=8,
    )
    assert workers == 8


def test_resource_manager_validates_input_ranges() -> None:
    """Execute test resource manager validates input ranges.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    rm.reset_resource_manager_for_tests()
    try:
        rm.ResourceManager(cpu_usage_percent=0.0, memory_usage_percent=90.0)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        rm.ResourceManager(cpu_usage_percent=90.0, memory_usage_percent=101.0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_hardware_detector_classification_is_stable(monkeypatch) -> None:
    """Execute test hardware detector classification is stable.



    Args:

        monkeypatch: Input value used by this callable.

    """

    monkeypatch.setattr(
        rm,
        "_load_classification_thresholds",
        lambda: rm.HardwareClassificationThresholds(
            mid_min_ram_gb=7.0, high_min_ram_gb=24.0, high_requires_gpu=True
        ),
    )
    assert rm.HardwareDetector._classify_machine(6.0, has_gpu=False) == "low_spec"
    assert rm.HardwareDetector._classify_machine(12.0, has_gpu=False) == "mid_spec"
    assert rm.HardwareDetector._classify_machine(32.0, has_gpu=True) == "high_spec"


def test_calculate_limits_validates_max_workers() -> None:
    """Execute test calculate limits validates max workers.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    rm.reset_resource_manager_for_tests()
    manager = rm.ResourceManager()
    try:
        manager.calculate_limits(
            task_count=1,
            estimated_task_size=1,
            min_workers=2,
            max_workers=1,
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_calculate_limits_keeps_expected_bounds(monkeypatch) -> None:
    """Execute test calculate limits keeps expected bounds.



    Args:

        monkeypatch: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    class _Memory:
        total = 16 * 1024**3
        available = 8 * 1024**3
        used = 8 * 1024**3
        percent = 50.0

    monkeypatch.setattr(
        rm.HardwareDetector,
        "detect",
        lambda: rm.HardwareProfile(
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            cpu_cores=4,
            cpu_threads=8,
            has_gpu=False,
            gpu_memory_gb=None,
            is_wsl=False,
            platform="Linux",
            profile_name="mid_spec",
        ),
    )
    monkeypatch.setattr(rm.psutil, "virtual_memory", lambda: _Memory())
    monkeypatch.setattr(
        rm,
        "_load_resource_manager_tuning",
        lambda: rm.ResourceManagerTuning(
            per_worker_overhead_mb=1,
            default_task_size_bytes=100,
            min_concurrent_tasks=1,
            max_concurrent_tasks=1000,
            ideal_batch_multiplier=2,
            max_batch_fraction_of_concurrency=0.5,
            pending_futures_multiplier=2,
            telemetry_ttl_ms=0,
        ),
    )

    manager = rm.ResourceManager(cpu_usage_percent=90.0, memory_usage_percent=90.0)
    limits = manager.calculate_limits(
        task_count=100,
        estimated_task_size=100,
        shared_data_size=0,
        min_workers=1,
        max_workers=4,
    )

    assert limits.total_cpus == 8
    assert limits.optimal_workers == 4
    assert limits.max_batch_size == 8
    assert limits.max_pending_futures == 8
