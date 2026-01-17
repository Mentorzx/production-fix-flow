"""Generic HPO objective facade.

Provides a stable entrypoint to run DSLFM trials from the HPO strategies without
depending on fragmented trial modules. Delegates to the existing
`TrialEvaluationPipeline` to preserve behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from pff import settings
from pff.shared.core.logger import logger
from pff.infrastructure.hpo.trials.pipeline import TrialEvaluationPipeline
from pff.infrastructure.hpo.trials.artifacts import TrialArtifactManager
from pff.infrastructure.hpo.trials.postgres_store import HpoPostgresStore


def run_dslfm_objective(
    params: dict[str, Any],
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    *,
    target_entity_ratio: float = 1.0,
    trial_number: int = 0,
    output_root: str | Path | None = None,
    trial: Any | None = None,
    study_name: str | None = None,
    store: HpoPostgresStore | None = None,
) -> float:
    """Run a single DSLFM trial and return its composite score."""
    logger.info("Executando objetivo DSLFM (facade)")
    if store is None:
        raise ValueError("HPO trials require a Postgres store")
    if output_root is None:
        trial_output_root = settings.CACHE_DIR / "hpo" / "dslfm_trials"
    else:
        trial_output_root = Path(output_root)
        if not trial_output_root.is_absolute():
            trial_output_root = settings.CACHE_DIR / trial_output_root
    pipeline = TrialEvaluationPipeline(
        params=params,
        train_df=train_df,
        valid_df=valid_df,
        target_entity_ratio=target_entity_ratio,
        trial_number=trial_number,
        trial_output_root=trial_output_root,
        trial=trial,
        artifact_manager=TrialArtifactManager(
            base_dir=None,
            study_name=study_name,
            store=store,
        ),
    )
    return pipeline.run()
