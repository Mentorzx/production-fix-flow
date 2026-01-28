import torch
import numpy as np
from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig
from pff.domain.learning.dslfm.kgc_manager import DSLFMKGCManager, KGCTrainingConfig


class MockPersistencePort:
    def save_checkpoint(self, data, filename):
        pass

    def load_checkpoint(self, filename, map_location=None):
        return None


def test_triton_filtered_equivalence():
    """Verify that Triton Corrected Ranking matches Eager Filtered Ranking exactly."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return

    device = torch.device("cuda")
    m_cfg = DSLFMKGCConfig(num_entities=1000, num_relations=10)
    # Force Triton off for baseline
    t_cfg_eager = KGCTrainingConfig(epochs=0)

    manager = DSLFMKGCManager(
        m_cfg, t_cfg_eager, persistence_port=MockPersistencePort(), device=device
    )

    # Setup triples
    train_triples = np.array([[0, 0, 1], [0, 0, 2]], dtype=np.int64)
    valid_triples = np.array([[0, 0, 1]], dtype=np.int64)
    manager._build_filter_dict(train_triples, np.zeros((0, 3), dtype=np.int64))

    valid_tensor = torch.from_numpy(valid_triples).to(device)

    # 1. Get Eager Ranks
    # We manually call with Triton disabled
    with torch.no_grad():
        # Temporarily mock is_triton_available to return False
        from unittest.mock import patch

        with patch(
            "pff.shared.acceleration.triton_kernels.is_triton_available", return_value=False
        ):
            metrics_eager = manager.model.evaluate(
                valid_tensor, filter_fn=manager._mask_known_tails
            )

    # 2. Get Triton Ranks
    with torch.no_grad():
        metrics_triton = manager.model.evaluate(valid_tensor, filter_fn=manager._mask_known_tails)

    print(f"Eager MRR: {metrics_eager['mrr']:.4f}")
    print(f"Triton MRR: {metrics_triton['mrr']:.4f}")

    assert abs(metrics_eager["mrr"] - metrics_triton["mrr"]) < 1e-6
    assert abs(metrics_eager["hits@1"] - metrics_triton["hits@1"]) < 1e-6


if __name__ == "__main__":
    test_triton_filtered_equivalence()
