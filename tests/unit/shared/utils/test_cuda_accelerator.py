"""Tests for runtime accelerator resolution."""

from __future__ import annotations

from pff.shared.system import cuda


def test_force_cpu_accelerator_hides_cuda(monkeypatch) -> None:
    """PFF_ACCELERATOR=cpu should always disable CUDA usage."""
    monkeypatch.setenv("PFF_ACCELERATOR", "cpu")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    accelerator, reason = cuda.configure_accelerator_environment()

    assert accelerator == "cpu"
    assert reason == "forced_cpu"
    assert cuda.os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert cuda.is_cuda_available() is False


def test_auto_accelerator_enables_allocator_defaults(monkeypatch) -> None:
    """Auto mode should install allocator defaults before CUDA init."""
    monkeypatch.setenv("PFF_ACCELERATOR", "auto")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(cuda, "_has_visible_cuda_devices", lambda: False)

    accelerator, reason = cuda.configure_accelerator_environment()

    assert accelerator == "cpu"
    assert reason == "no_visible_cuda_device"
    assert "expandable_segments:True" in cuda.os.environ["PYTORCH_CUDA_ALLOC_CONF"]


def test_cuda_request_falls_back_when_torch_cannot_use_cuda(monkeypatch) -> None:
    """Forced CUDA should degrade to CPU with a stable reason when unavailable."""
    monkeypatch.setenv("PFF_ACCELERATOR", "cuda")
    monkeypatch.setattr(cuda, "_has_visible_cuda_devices", lambda: True)
    monkeypatch.setattr(cuda, "_torch_reports_cuda", lambda: False)

    accelerator, reason = cuda.resolve_accelerator()

    assert accelerator == "cpu"
    assert reason == "cuda_requested_but_torch_unavailable"
