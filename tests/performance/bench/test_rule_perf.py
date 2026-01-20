import time

from pff.application.services.business_service.rule_validator import (
    Rule,
    find_rule_violations_indexed,
)
from pff.application.services.business_service.triple_index import TripleIndex


def bench_rule_validation():
    # 1. Create large synthetic dataset
    # Rule: Parent(X, Y) AND Parent(Y, Z) -> Grandparent(X, Z)
    triples = []
    # 1000 Parent(i, i+1) triples
    for i in range(1000):
        triples.append((f"Person_{i}", "Parent", f"Person_{i + 1}"))

    # Add some grandparents
    for i in range(0, 998, 2):
        triples.append((f"Person_{i}", "Grandparent", f"Person_{i + 2}"))

    index = TripleIndex(triples)
    rule = Rule(
        id="gp_rule",
        head={"predicate": "Grandparent", "args": ["X", "Z"]},
        body=[
            {"predicate": "Parent", "args": ["X", "Y"]},
            {"predicate": "Parent", "args": ["Y", "Z"]},
        ],
        confidence=1.0,
        source="manual",
    )

    violations = []
    start = time.perf_counter()
    find_rule_violations_indexed(rule.body, triples, index, 0, {}, violations, rule)
    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000
    print(f"Rule Validation Baseline (1000 triples, 2 body preds): {elapsed_ms:.2f}ms")
    print(f"Violations found: {len(violations)}")
    return elapsed_ms


if __name__ == "__main__":
    bench_rule_validation()
