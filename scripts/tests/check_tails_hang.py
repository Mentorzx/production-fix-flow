import torch
from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig
from pff.domain.learning.dslfm.kgc_manager import DSLFMKGCManager, KGCTrainingConfig


class MockPersistencePort:
    def save_checkpoint(self, data, filename):
        pass

    def load_checkpoint(self, filename, map_location=None):
        return None


def check_simple():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    m_cfg = DSLFMKGCConfig(num_entities=100, num_relations=10, entity_dim=16, feature_dim=16)
    t_cfg = KGCTrainingConfig(epochs=0)

    manager = DSLFMKGCManager(m_cfg, t_cfg, persistence_port=MockPersistencePort(), device=device)

    h = torch.tensor([0], device=device)
    r = torch.tensor([0], device=device)
    t = torch.tensor([1], device=device)
    scores = torch.randn(1, 100, device=device)
    cands = torch.arange(100, device=device)

    # Prepopulate latents
    manager.model.precompute_entity_latents()

    print("Calling _mask_known_tails (masking mode)...")
    manager._mask_known_tails(scores.clone(), h, r, cands, t, False)
    print("Done masking mode.")

    true_scores = scores.gather(1, (t - 0).unsqueeze(1)).squeeze(1)
    print("Calling _mask_known_tails (correction mode)...")
    corr = manager._mask_known_tails(true_scores, h, r, torch.empty(0, device=device), t, True)
    print(f"Done correction mode. Correction: {corr}")


if __name__ == "__main__":
    check_simple()
