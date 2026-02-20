"""CUDA regression tests for DSLFM device alignment."""

from __future__ import annotations

import pytest
import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_encode_entities_aligns_entity_ids_to_model_device() -> None:
    """Entity IDs from CPU should be aligned to CUDA model device before embedding lookup."""
    config = DSLFMKGCConfig(
        num_entities=128,
        num_relations=8,
        num_triples=64,
        entity_dim=16,
        feature_dim=16,
        max_communities=8,
        hidden_dim=32,
        lambda_pc=0.0,
        lambda_logic=0.0,
        use_bert_relations=False,
    )
    model = DSLFMKGCModel(config).to("cuda")
    cpu_ids = torch.tensor([0, 3, 7, 9], dtype=torch.long, device="cpu")

    latents = model.encode_entities(cpu_ids)

    assert latents["communities"].device.type == "cuda"
    assert latents["features"].device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compute_loss_aligns_cpu_inputs_to_cuda_model() -> None:
    """compute_loss should align CPU triples to CUDA model device."""
    config = DSLFMKGCConfig(
        num_entities=128,
        num_relations=8,
        num_triples=64,
        entity_dim=16,
        feature_dim=16,
        max_communities=8,
        hidden_dim=32,
        lambda_pc=0.0,
        lambda_logic=0.0,
        use_bert_relations=False,
    )
    model = DSLFMKGCModel(config).to("cuda")
    heads = torch.tensor([1, 2, 3, 4], dtype=torch.long, device="cpu")
    relations = torch.tensor([0, 1, 2, 3], dtype=torch.long, device="cpu")
    tails = torch.tensor([2, 3, 4, 5], dtype=torch.long, device="cpu")

    loss_dict = model.compute_loss(heads, relations, tails)

    assert "loss" in loss_dict
    assert loss_dict["loss"].device.type == "cuda"
