"""Hyperparameter optimization use case."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pff.application.ports.hpo import HpoRunnerPort


class OptimizeUseCase:
    """Run the Optuna-based optimization pipeline."""

    def __init__(self, runner: HpoRunnerPort) -> None:
        """Execute init.



        Args:

            runner: Input value used by this callable.

        """

        self._runner = runner

    def execute(
        self,
        *,
        n_trials: int = 50,
        strategy: str = "optuna",
        enable_mlflow: bool = False,
        enable_visualization: bool = False,
        study_name: str | None = None,
        output_dir: Path | None = None,
        target_entity_ratio: float = 0.7,
        kge_model: str = "dslfm",
        use_synthetic_if_dslfm: bool = False,
        no_update_config: bool = False,
        no_bert: bool = False,
        resume_mode: bool | None = None,
        reset_state: bool = False,
    ) -> dict[str, Any]:
        """Execute the optimization workflow.

        Args:
            n_trials: Number of trials.
            strategy: Optimization backend strategy.
            enable_mlflow: Whether to enable MLflow tracking.
            enable_visualization: Whether to enable visualization callbacks.
            study_name: Optional Optuna study name.
            output_dir: Optional output directory.
            target_entity_ratio: Target entity ratio.
            kge_model: KGE model name.
            use_synthetic_if_dslfm: Use synthetic data for DSLFM.
            no_update_config: Skip updating dslfm.yaml with best params.
            no_bert: Disable BERT relation encoder for HPO.
            resume_mode: Resume mode override.
            reset_state: Reset Optuna state flag.

        Returns:
            Optimization result dictionary.
        """
        return self._runner.run(
            n_trials=n_trials,
            strategy=strategy,
            enable_mlflow=enable_mlflow,
            enable_visualization=enable_visualization,
            study_name=study_name,
            output_dir=output_dir,
            target_entity_ratio=target_entity_ratio,
            kge_model=kge_model,
            use_synthetic_if_dslfm=use_synthetic_if_dslfm,
            no_update_config=no_update_config,
            no_bert=no_bert,
            resume_mode=resume_mode,
            reset_state=reset_state,
        )
