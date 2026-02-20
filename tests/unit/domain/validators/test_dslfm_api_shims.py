"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/validators/test_dslfm_api_shims.py

"""

from __future__ import annotations

import numpy as np

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig
from pff.domain.learning.dslfm.kgc_manager import KGCTrainingConfig
from pff.domain.learning.dslfm.validator import DSLFMValidator


def test_dslfm_validator_facade_trains_on_tiny_batch() -> None:
    """Execute test dslfm validator facade trains on tiny batch.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    train_triples = np.array([[0, 0, 1], [1, 0, 2]], dtype=np.int64)
    valid_triples = np.array([[0, 0, 1]], dtype=np.int64)

    model_cfg = DSLFMKGCConfig(
        num_entities=3,
        num_relations=1,
        entity_dim=8,
        feature_dim=8,
        max_communities=4,
        hidden_dim=16,
        lambda_logic=0.0,
        lambda_pc=0.0,
    )
    train_cfg = KGCTrainingConfig(
        epochs=1,
        batch_size=2,
        effective_batch_size=2,
        learning_rate=1e-2,
        validate_every=1,
        early_stopping_patience=1,
        mixed_precision=False,
        num_workers=0,
        pin_memory=False,
        eval_batch_size=2,
    )

    validator = DSLFMValidator(model_cfg, train_cfg)
    stats = validator.train_and_validate(train_triples, valid_triples)

    assert stats["best_metrics"]["mrr"] >= 0.0
