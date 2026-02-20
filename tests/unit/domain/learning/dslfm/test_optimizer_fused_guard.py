from __future__ import annotations

from pff.domain.learning.dslfm.kgc_manager import _should_enable_fused_adamw


def test_should_disable_fused_adamw_for_mixed_signatures() -> None:
    assert (
        _should_enable_fused_adamw(
            is_cuda=True,
            optimizer_fused=None,
            param_signatures={("cuda", "torch.float32"), ("cpu", "torch.float32")},
        )
        is False
    )


def test_should_enable_fused_adamw_for_single_cuda_signature() -> None:
    assert (
        _should_enable_fused_adamw(
            is_cuda=True,
            optimizer_fused=True,
            param_signatures={("cuda", "torch.float32")},
        )
        is True
    )
