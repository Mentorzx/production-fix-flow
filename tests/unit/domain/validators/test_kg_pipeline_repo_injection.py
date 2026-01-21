from __future__ import annotations

from pff.domain.kg.config import KGConfig
from pff.domain.kg.pipeline import KGPipeline


class _DummySplitsRepo:
    pass


def test_pipeline_injects_splits_repo_into_preprocessor() -> None:
    config = KGConfig("config/models/kg.yaml")
    splits_repo = _DummySplitsRepo()

    pipeline = KGPipeline(config, splits_repo=splits_repo)

    assert pipeline.preprocessor.splits_repo is splits_repo
