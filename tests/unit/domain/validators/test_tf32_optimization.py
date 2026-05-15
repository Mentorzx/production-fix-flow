"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/validators/test_tf32_optimization.py

"""

from __future__ import annotations

import torch
import pytest
from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig
from pff.domain.learning.dslfm.kgc_manager import DSLFMKGCManager, KGCTrainingConfig


class MockPersistencePort:
    """Represent MockPersistencePort."""

    def save_checkpoint(self, data, filename):
        """Execute save checkpoint.



        Args:

            data: Input value used by this callable.

            filename: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        pass

    def load_checkpoint(self, filename, map_location=None):
        """Execute load checkpoint.



        Args:

            filename: Input value used by this callable.

            map_location: Optional input value.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        return None


def test_tf32_config_enables_torch_setting() -> None:
    """Regression test: Ensures manager properly sets torch matmul precision."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_cfg = DSLFMKGCConfig(num_entities=10, num_relations=2)

    # Test with TF32 enabled
    torch.set_float32_matmul_precision("high")
    t_cfg_on = KGCTrainingConfig(tf32=True, matmul_precision="medium")
    _ = DSLFMKGCManager(m_cfg, t_cfg_on, persistence_port=MockPersistencePort(), device=device)
    expected_precision = "medium" if device.type == "cuda" else "high"
    assert torch.get_float32_matmul_precision() == expected_precision

    # Test with TF32 disabled
    t_cfg_off = KGCTrainingConfig(tf32=False)
    # We need to manually reset first because torch settings are global
    torch.set_float32_matmul_precision("high")
    _ = DSLFMKGCManager(m_cfg, t_cfg_off, persistence_port=MockPersistencePort(), device=device)
    # If tf32 is False, it should NOT change the current setting (which we set to high)
    assert torch.get_float32_matmul_precision() == "high"
