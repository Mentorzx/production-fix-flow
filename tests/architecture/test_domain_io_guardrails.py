from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_ROOT = REPO_ROOT / "pff" / "domain"

FORBIDDEN_PATTERNS = [
    re.compile(r"\bFileManager\b"),
    re.compile(r"\basyncio\b"),
    re.compile(r"\bthreading\b"),
    re.compile(r"\bmultiprocessing\b"),
    re.compile(r"\brequests\b"),
]

ALLOWLIST = {
    "pff/domain/learning/dslfm/bert_encoder.py",
    "pff/domain/learning/dslfm/dslfm_kgc.py",
    "pff/domain/learning/dslfm/metrics_reporter.py",
    "pff/domain/learning/dslfm/checkpoint_manager.py",
    "pff/domain/learning/dslfm/kgc_manager.py",
    "pff/domain/learning/ml/adaptive_training.py",
    "pff/domain/learning/ml/ann_evaluator.py",
    "pff/domain/learning/ml/base_trainer.py",
    "pff/domain/learning/ml/model_factory.py",
    "pff/domain/learning/ml/training_observer.py",
    "pff/domain/kg/builder.py",
    "pff/domain/kg/config.py",
    "pff/domain/kg/data_loader.py",
    "pff/domain/kg/data_optimizer.py",
    "pff/domain/kg/pipeline.py",
    "pff/domain/kg/preprocess.py",
    "pff/domain/kg/preprocessing/config.py",
    "pff/domain/kg/preprocessing/pipeline.py",
    "pff/domain/audit/evt.py",
    "pff/domain/audit/bench.py",
    "pff/domain/audit/manifest.py",
    "pff/domain/audit/profile.py",
    "pff/domain/audit/report.py",
    "pff/domain/audit/schema.py",
}


def _iter_python_files() -> list[Path]:
    return [path for path in DOMAIN_ROOT.rglob("*.py") if path.is_file()]


def test_domain_io_concurrency_guardrails() -> None:
    """Prevent new I/O or concurrency imports in domain without explicit allowlist."""
    violations: list[str] = []
    for path in _iter_python_files():
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        if rel_path in ALLOWLIST:
            continue
        content = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(content):
                violations.append(f"{rel_path}: matched {pattern.pattern}")
                break

    assert (
        not violations
    ), "New domain imports violate I/O/concurrency guardrails:\n" + "\n".join(
        sorted(violations)
    )
