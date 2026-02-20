"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/learning/dslfm/test_sbm_decoder_bias.py

"""

import torch

from pff.domain.learning.dslfm.sbm_decoder import StochasticBlockmodelDecoder


def test_score_all_tails_applies_relation_bias() -> None:
    """Ensure relation bias broadcasts correctly in score_all_tails."""
    decoder = StochasticBlockmodelDecoder(
        num_communities=2,
        feature_dim=2,
        num_relations=3,
    )
    decoder.community_weight.data.fill_(0.0)
    decoder.feature_weight.data.fill_(0.0)
    decoder.relation_bias.data = torch.tensor([1.0, 2.0, 3.0])

    z_head = torch.zeros(2, 2)
    f_head = torch.zeros(2, 2)
    relations = torch.tensor([0, 2])
    all_z = torch.zeros(5, 2)
    all_f = torch.zeros(5, 2)

    scores = decoder.score_all_tails(
        z_head=z_head,
        f_head=f_head,
        relations=relations,
        all_z=all_z,
        all_f=all_f,
    )

    assert scores.shape == (2, 5)
    assert torch.allclose(scores[0], torch.full((5,), 1.0))
    assert torch.allclose(scores[1], torch.full((5,), 3.0))
