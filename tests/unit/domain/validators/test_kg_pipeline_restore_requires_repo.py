"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/domain/validators/test_kg_pipeline_restore_requires_repo.py

"""

from __future__ import annotations

import pytest

from pff.domain.kg.config import KGConfig
from pff.domain.kg.pipeline import KGPipeline


@pytest.mark.asyncio
async def test_restore_parquets_requires_repo() -> None:
    """Execute test restore parquets requires repo.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    config = KGConfig("config/models/kg.yaml")
    pipeline = KGPipeline(config)

    with pytest.raises(RuntimeError, match="splits_repo not available"):
        await pipeline._restore_parquets_from_postgres()
