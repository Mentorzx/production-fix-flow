from __future__ import annotations

import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel
from pff.domain.learning.dslfm.neg_sampling import NSCachingSampler, SamplerConfig, SamplerType


class _FakeCacheManager:
    def __init__(self, payload: torch.Tensor) -> None:
        self.payload = payload
        self.saved: dict[str, torch.Tensor] = {}

    def get(self, key: str):
        if key.endswith("_5"):
            return None
        if key == "nsc_tensor_2_3":
            return self.payload
        return None

    def set(self, key: str, value: torch.Tensor) -> None:
        self.saved[key] = value


def test_nscaching_sampler_sanitizes_legacy_cached_ids() -> None:
    config = SamplerConfig(
        sampler_type=SamplerType.NSCACHING,
        num_entities=5,
        num_triples=2,
        cache_size=3,
        sample_ratio=1.0,
    )
    sampler = NSCachingSampler(config)
    sampler.cache_manager = _FakeCacheManager(
        torch.tensor([[0, 5, 9], [10, 11, 12]], dtype=torch.long)
    )

    heads = torch.tensor([0, 1], dtype=torch.long)
    rels = torch.tensor([0, 0], dtype=torch.long)
    tails = torch.tensor([1, 2], dtype=torch.long)
    idx = torch.tensor([0, 1], dtype=torch.long)

    negatives = sampler.sample_negatives(heads, rels, tails, num_negatives=2, triple_indices=idx)
    assert negatives.shape == (2, 2)
    assert int(negatives.min().item()) >= 0
    assert int(negatives.max().item()) < 5


def test_encode_entities_modulo_guard_for_out_of_range_ids() -> None:
    model = DSLFMKGCModel(
        DSLFMKGCConfig(
            num_entities=4,
            num_relations=2,
            entity_dim=8,
            feature_dim=8,
            max_communities=4,
            hidden_dim=8,
            use_bert_relations=False,
            lambda_logic=0.0,
            lambda_pc=0.0,
        )
    )
    output = model.encode_entities(torch.tensor([0, 1, 7], dtype=torch.long))
    assert output["communities"].shape[0] == 3
