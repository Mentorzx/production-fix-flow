import torch

from pff.validators.transe.core import _apply_self_adversarial_weights


def test_self_adversarial_weights_matches_softmax_weighting():
    losses = torch.tensor([[1.0, 2.0], [0.5, 0.1]], dtype=torch.float32)
    neg_scores = torch.tensor([[0.2, 0.4], [0.0, 1.0]], dtype=torch.float32)
    temperature = 1.3

    weighted = _apply_self_adversarial_weights(losses, neg_scores, temperature)
    manual = torch.softmax(neg_scores * temperature, dim=1)
    manual = (losses * manual).sum(dim=1)

    assert torch.allclose(weighted, manual, atol=1e-6)
