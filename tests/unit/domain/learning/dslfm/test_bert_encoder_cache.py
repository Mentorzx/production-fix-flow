from __future__ import annotations

import torch

from pff.domain.learning.dslfm import bert_encoder as bert_module


def test_relation_text_encoder_reuses_cached_hf_artifacts(monkeypatch) -> None:
    """freeze_bert=True should reuse process-local HF model/tokenizer cache."""

    class DummyBert(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Cfg", (), {"hidden_size": 8})()
            self.weight = torch.nn.Parameter(torch.ones(1))

    class DummyAutoModel:
        calls = 0

        @classmethod
        def from_pretrained(cls, _name: str):
            cls.calls += 1
            return DummyBert()

    class DummyAutoTokenizer:
        calls = 0

        @classmethod
        def from_pretrained(cls, _name: str):
            cls.calls += 1
            return object()

    monkeypatch.setattr(bert_module, "TRANSFORMERS_AVAILABLE", True, raising=True)
    monkeypatch.setattr(bert_module, "AutoModel", DummyAutoModel, raising=True)
    monkeypatch.setattr(bert_module, "AutoTokenizer", DummyAutoTokenizer, raising=True)
    monkeypatch.setattr(bert_module, "_HF_MODEL_CACHE", {}, raising=True)
    monkeypatch.setattr(bert_module, "_HF_TOKENIZER_CACHE", {}, raising=True)

    enc_a = bert_module.RelationTextEncoder(
        model_name="bert-base-uncased",
        hidden_dim=8,
        freeze_bert=True,
    )
    enc_b = bert_module.RelationTextEncoder(
        model_name="bert-base-uncased",
        hidden_dim=8,
        freeze_bert=True,
    )

    assert DummyAutoModel.calls == 1
    assert DummyAutoTokenizer.calls == 1
    assert enc_a.bert is enc_b.bert
    assert enc_a.tokenizer is enc_b.tokenizer
