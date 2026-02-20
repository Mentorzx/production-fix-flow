from __future__ import annotations

import torch
from types import SimpleNamespace

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCModel


class _DummyANN:
    def __init__(self, scores: torch.Tensor, indices: torch.Tensor) -> None:
        self._scores = scores
        self._indices = indices

    def _score_faiss_candidates(self, _z_h, _f_h, _r, _k):
        return self._scores, self._indices


def test_ann_rerank_uses_candidates_and_handles_missing_true_tail() -> None:
    cand_scores = torch.tensor(
        [[0.9, 0.2, 0.1], [0.1, 0.2, 0.3]],
        dtype=torch.float32,
    )
    cand_idx = torch.tensor([[5, 2, 7], [1, 4, 9]], dtype=torch.int64)
    dummy = _DummyANN(cand_scores, cand_idx)

    base_ranks = torch.ones(2, dtype=torch.int32)
    tails = torch.tensor([2, 8], dtype=torch.int64)
    true_scores = torch.tensor([0.2, 0.15], dtype=torch.float32)

    ranks = DSLFMKGCModel._evaluate_batch_with_faiss(
        dummy,
        base_ranks=base_ranks,
        heads=torch.zeros(2, dtype=torch.int64),
        z_h=torch.zeros(2, 1),
        f_h=torch.zeros(2, 1),
        relations=torch.zeros(2, dtype=torch.int64),
        tails=tails,
        true_scores=true_scores,
        filter_fn=None,
        faiss_candidate_k=3,
        rerank_top_k=None,
    )

    assert ranks.tolist() == [2, 4]


def test_ann_rerank_respects_rerank_top_k() -> None:
    cand_scores = torch.tensor([[0.9, 0.2, 0.1]], dtype=torch.float32)
    cand_idx = torch.tensor([[5, 2, 7]], dtype=torch.int64)
    dummy = _DummyANN(cand_scores, cand_idx)

    ranks = DSLFMKGCModel._evaluate_batch_with_faiss(
        dummy,
        base_ranks=torch.ones(1, dtype=torch.int32),
        heads=torch.zeros(1, dtype=torch.int64),
        z_h=torch.zeros(1, 1),
        f_h=torch.zeros(1, 1),
        relations=torch.zeros(1, dtype=torch.int64),
        tails=torch.tensor([2], dtype=torch.int64),
        true_scores=torch.tensor([0.2], dtype=torch.float32),
        filter_fn=None,
        faiss_candidate_k=3,
        rerank_top_k=2,
    )

    assert ranks.item() == 2


def test_evaluate_disables_faiss_for_small_graphs() -> None:
    class _DummyModel:
        def __init__(self) -> None:
            self.entity_embedding = torch.nn.Embedding(10, 2)
            self.config = SimpleNamespace(num_entities=10)
            self.used_faiss = False

        def eval(self) -> None:
            return None

        def _resolve_eval_latents(self, batch_size: int, refresh_cache: bool):
            del batch_size, refresh_cache
            return torch.zeros(10, 1), torch.zeros(10, 1)

        def _ensure_faiss_index(self, _all_f: torch.Tensor) -> None:
            self.used_faiss = True

        def encode_entities(self, h: torch.Tensor):
            return {
                "communities": torch.zeros(h.shape[0], 1),
                "features": torch.zeros(h.shape[0], 1),
            }

        def _compute_true_scores(self, **kwargs):
            z_h = kwargs["z_h"]
            return torch.zeros(z_h.shape[0], dtype=torch.float32)

        def _can_use_triton_for_eval(self, _device) -> bool:
            return False

        def _evaluate_batch_full_scan(self, **kwargs):
            return kwargs["base_ranks"]

        def _compute_ranking_metrics(self, all_ranks: torch.Tensor):
            mrr = (1.0 / all_ranks).mean().item()
            return {
                "mrr": mrr,
                "hits@1": 1.0,
                "hits@3": 1.0,
                "hits@10": 1.0,
                "ap@10": 1.0,
            }

    model = _DummyModel()
    eval_triples = torch.tensor([[0, 0, 1], [1, 0, 2]], dtype=torch.long)
    metrics = DSLFMKGCModel.evaluate(
        model,
        eval_triples,
        batch_size=2,
        filter_fn=None,
        use_faiss_eval=True,
        faiss_candidate_k=4,
    )

    assert model.used_faiss is False
    assert metrics["mrr"] == 1.0
