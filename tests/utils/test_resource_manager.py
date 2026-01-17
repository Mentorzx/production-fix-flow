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
