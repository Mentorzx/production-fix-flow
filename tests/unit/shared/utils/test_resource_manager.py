from __future__ import annotations

from pff.shared.system import resource_manager as rm


def test_get_memory_safe_workers_returns_zero_when_multiproc_unavailable(
    monkeypatch,
) -> None:
    def _raise_permission_error():
        raise PermissionError("multiprocessing not allowed")

    monkeypatch.setattr(rm, "_MULTIPROC_AVAILABLE", None)
    monkeypatch.setattr(rm.mp, "get_context", _raise_permission_error)

    assert rm.get_memory_safe_workers(chunk_size=1000) == 0


def test_get_auto_dataloader_workers_returns_zero_for_small_dataset() -> None:
    workers = rm.get_auto_dataloader_workers(
        dataset_size=10,
        batch_size=4,
        min_workers=1,
        max_workers=4,
    )
    assert workers == 0


def test_get_auto_dataloader_workers_respects_bounds(monkeypatch) -> None:
    monkeypatch.setattr(rm, "get_memory_safe_workers", lambda chunk_size: 16)
    workers = rm.get_auto_dataloader_workers(
        dataset_size=1000,
        batch_size=32,
        min_workers=2,
        max_workers=8,
    )
    assert workers == 8


def test_resource_manager_validates_input_ranges() -> None:
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
