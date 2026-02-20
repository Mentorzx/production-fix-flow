"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/utils/test_audit_pc2_graph_constraints.py

"""

from __future__ import annotations

import torch

from pff.domain.audit.graph_constraints import GraphConstraintsValidator
from pff.domain.audit.pc2_auditor import pc2_log_prob_pairwise
from pff.domain.learning.pc.npc import NeuralProbabilisticCircuit


def test_pc2_auditor_pairwise_log_prob_matches_direct_call() -> None:
    """Execute test pc2 auditor pairwise log prob matches direct call.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    torch.manual_seed(0)
    pc = NeuralProbabilisticCircuit(num_attrs=3, pruning_threshold=0.0)
    z_head = torch.rand(4, 3)
    z_tail = torch.rand(4, 3)

    audit = pc2_log_prob_pairwise(pc, z_head=z_head, z_tail=z_tail).log_prob

    combined = 0.5 * (z_head + z_tail).clamp(pc.smoothing_epsilon, 1.0 - pc.smoothing_epsilon)
    attr_probs = torch.stack([combined, 1.0 - combined], dim=-1)
    labels = torch.ones(4, dtype=torch.long)
    direct = pc.log_prob(attr_probs, labels)

    assert torch.allclose(audit, direct)


def test_graph_constraints_validator_emits_shacl_like_report() -> None:
    """Execute test graph constraints validator emits shacl like report.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    triples = [
        {"s": "doc", "p": "/a/y/*", "o": "1", "json_pointer": "/a/y/0"},
        {"s": "doc", "p": "/a/y/*", "o": "2", "json_pointer": "/a/y/1"},
    ]
    constraints = {"max_cardinality_by_predicate": {"/a/y/*": 1}}
    report = GraphConstraintsValidator(constraints=constraints).validate(triples)

    assert report
    assert any(
        item.get("constraint") == "max_cardinality" for item in report if isinstance(item, dict)
    )
