from __future__ import annotations

import torch
import pytest
from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig
from pff.domain.learning.dslfm.kgc_manager import DSLFMKGCManager, KGCTrainingConfig


class MockPersistencePort:
    def save_checkpoint(self, data, filename):
        pass

    def load_checkpoint(self, filename, map_location=None):
        return None


def test_tf32_config_enables_torch_setting() -> None:
    """Regression test: Ensures manager properly sets torch matmul precision."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    m_cfg = DSLFMKGCConfig(num_entities=10, num_relations=2)

    # Test with TF32 enabled
    t_cfg_on = KGCTrainingConfig(tf32=True, matmul_precision="medium")
    _ = DSLFMKGCManager(m_cfg, t_cfg_on, persistence_port=MockPersistencePort(), device=device)
    assert torch.get_float32_matmul_precision() == "medium"

    # Test with TF32 disabled
    t_cfg_off = KGCTrainingConfig(tf32=False)
    # We need to manually reset first because torch settings are global
    torch.set_float32_matmul_precision("high")
    _ = DSLFMKGCManager(m_cfg, t_cfg_off, persistence_port=MockPersistencePort(), device=device)
    # If tf32 is False, it should NOT change the current setting (which we set to high)
    assert torch.get_float32_matmul_precision() == "high"
