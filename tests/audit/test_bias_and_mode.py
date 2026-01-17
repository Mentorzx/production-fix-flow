import torch
from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCModel, DSLFMKGCConfig


def test_decoder_bias_and_mode_switching():
    """Verify Identity Bias magnitude and Mode Switching logic."""

    config_bi = DSLFMKGCConfig(num_entities=100, num_relations=5, feature_dim=64)
    model_bi = DSLFMKGCModel(config_bi)
    assert model_bi.decoder.use_bilinear, "Should be Bilinear for dim=64"
    assert hasattr(model_bi.decoder, "feature_bilinear"), "Missing bilinear layer"

    config_dot = DSLFMKGCConfig(num_entities=100, num_relations=5, feature_dim=256)
    model_dot = DSLFMKGCModel(config_dot)
    assert not model_dot.decoder.use_bilinear, "Should NOT be Bilinear for dim=256"

    print("\n[CHECK] Mode switching logic verified.")

    W = model_bi.decoder.W.detach()
    R, K, _ = W.shape

    diag_mask = torch.eye(K, dtype=torch.bool).unsqueeze(0).expand(R, K, K)
    diag_vals = W[diag_mask]
    off_vals = W[~diag_mask]

    diag_mean = diag_vals.mean().item()
    off_mean = off_vals.mean().item()

    print(f"\n[BIAS] Diagonal Mean: {diag_mean:.6f} (Target ~0.1)")
    print(f"[BIAS] Off-Diag Mean: {off_mean:.6f} (Target ~0.0)")

    bias_ratio = abs(diag_mean) / (abs(off_mean) + 1e-9)
    print(f"[BIAS] Ratio: {bias_ratio:.2f}")

    assert diag_mean > 0.08, "Identity bias seems missing or too weak"
    assert abs(off_mean) < 0.02, "Off-diagonal noise too high"


if __name__ == "__main__":
    test_decoder_bias_and_mode_switching()
