"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/validators/test_kg_pipeline_checkpoint_fallback.py

"""

from __future__ import annotations

from pathlib import Path

import pytest

from pff.domain.kg.config import KGConfig
from pff.domain.kg.pipeline import KGPipeline


class _FailingCheckpointRepo:
    async def get_checkpoint(self, *args, **kwargs):
        """Execute get checkpoint.



        Args:

            *args: Additional positional arguments.

            **kwargs: Additional keyword arguments.

        """

        raise RuntimeError("db unavailable")

    async def save_checkpoint(self, *args, **kwargs):
        """Execute save checkpoint.



        Args:

            *args: Additional positional arguments.

            **kwargs: Additional keyword arguments.

        """

        raise RuntimeError("db unavailable")


@pytest.mark.asyncio
async def test_checkpoint_fallback_when_db_unavailable(tmp_path: Path) -> None:
    """Execute test checkpoint fallback when db unavailable.



    Args:

        tmp_path: Input value used by this callable.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    config = KGConfig("config/models/kg.yaml")
    pipeline = KGPipeline(config)
    pipeline.checkpoints_repo = _FailingCheckpointRepo()

    assert await pipeline._load_checkpoint("preprocess") is None
    await pipeline._save_checkpoint("preprocess", "running", progress=0.1)
