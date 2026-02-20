"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/validators/test_kg_pipeline_repo_injection.py

"""

from __future__ import annotations

from typing import Any, cast

from pff.domain.kg.config import KGConfig
from pff.domain.kg.pipeline import KGPipeline
from pff.shared.core.file_manager import FileManager


class _DummySplitsRepo:
    pass


def test_pipeline_injects_splits_repo_into_preprocessor() -> None:
    """Execute test pipeline injects splits repo into preprocessor.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    config = KGConfig("config/models/kg.yaml")
    splits_repo = _DummySplitsRepo()

    pipeline = KGPipeline(config, splits_repo=cast(Any, splits_repo))

    assert pipeline.preprocessor.splits_repo is splits_repo


def test_pipeline_propagates_file_manager_injection() -> None:
    """Execute test pipeline propagates file manager injection.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    config = KGConfig("config/models/kg.yaml")
    file_manager = FileManager()

    pipeline = KGPipeline(config, file_manager=file_manager)

    assert pipeline.file_manager is file_manager
    assert pipeline.preprocessor.file_manager is file_manager
    assert pipeline.metrics_calculator.file_manager is file_manager
