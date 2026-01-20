#!/usr/bin/env python3
"""
Visualization Module - Automatic Plot Generation (Using Polars)

Automatically generates and saves visualization plots for optimization results:
- Optimization history (progress over time)
- Parameter importance (which params matter most)
- Contour plots (2D parameter relationships)
- Parallel coordinates (all parameters at once)
- Hyperparameter correlation heatmap
- Trial states (completed, pruned, failed)
- Best trial comparison

Uses Polars instead of Pandas for efficient data manipulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pff.shared import logger
from pff.shared.core.config import settings
from pff.shared.core.file_manager import FileManager

from .strategies.base import OptimizationResult


class OptimizationVisualizer:
    """
    Automatic visualization generation for optimization results.

    Creates comprehensive visualizations that are automatically saved
    and can be logged as MLflow artifacts.
    """

    def __init__(self, output_dir: Path | None = None):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots (default: ./outputs/optimization/plots)
        """
        self.output_dir = output_dir or (settings.OUTPUTS_DIR / "optimization" / "plots")
        self.file_manager = FileManager()
        self.file_manager.ensure_dir(self.output_dir)

        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if visualization libraries are available."""
        self.has_plotly = False
        self.has_matplotlib = False
        self.has_seaborn = False
        self.has_optuna_viz = False

        try:
            import plotly
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            self.has_plotly = True
            self.plotly = plotly
            self.px = px
            self.go = go
            self.make_subplots = make_subplots
            logger.debug("Plotly available for visualization")
        except ImportError:
            logger.warning("Plotly not installed. Install with: pip install plotly")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            self.has_matplotlib = True
            self.has_seaborn = True
            self.matplotlib = __import__("matplotlib.pyplot")
            self.plt = plt
            self.sns = sns
            logger.debug("Matplotlib/Seaborn available for visualization")
        except ImportError:
            logger.warning("Matplotlib not installed. Install with: pip install matplotlib seaborn")

        self.has_optuna_viz = self.has_plotly
        if self.has_optuna_viz:
            try:
                from optuna.visualization import (
                    plot_contour,
                    plot_edf,
                    plot_optimization_history,
                    plot_parallel_coordinate,
                    plot_param_importances,
                    plot_slice,
                )

                self.optuna_plot_funcs = {
                    "optimization_history": plot_optimization_history,
                    "param_importances": plot_param_importances,
                    "contour": plot_contour,
                    "parallel_coordinate": plot_parallel_coordinate,
                    "slice": plot_slice,
                    "edf": plot_edf,
                }
                logger.debug("Optuna visualization functions available")
            except ImportError:
                self.has_optuna_viz = False
                logger.debug("Optuna visualization not available")

    def _save_plotly_html(self, fig: Any, output_file: Path) -> None:
        """Persist a Plotly figure as HTML via the utils FileManager.

        Args:
            fig: Plotly figure instance.
            output_file: Destination HTML path.
        """
        html = fig.to_html(full_html=True, include_plotlyjs="cdn")
        self.file_manager.write_text(html, output_file)

    def _save_matplotlib_png(
        self,
        output_file: Path,
        *,
        dpi: int = 300,
        bbox_inches: str = "tight",
    ) -> None:
        """Persist the current Matplotlib figure as PNG via the utils FileManager.

        Args:
            output_file: Destination PNG path.
            dpi: Render resolution.
            bbox_inches: Matplotlib bbox mode.
        """
        import io

        buffer = io.BytesIO()
        self.plt.savefig(buffer, format="png", dpi=dpi, bbox_inches=bbox_inches)
        self.file_manager.write_bytes(buffer.getvalue(), output_file)

    def generate_all_plots(
        self,
        result: OptimizationResult,
        study: Any | None = None,
        top_n_trials: int = 20,
    ) -> dict[str, Path]:
        """
        Generate all visualization plots.

        Args:
            result: Optimization result
            study: Optuna study object (optional, for better plots)
            top_n_trials: Number of top trials to highlight

        Returns:
            Dictionary mapping plot names to file paths
        """
        logger.info("Gerando visualizacoes de otimizacao...")

        artifacts = {}

        artifacts.update(self.plot_optimization_history(result))

        artifacts.update(self.plot_parameter_importances(result, study))

        artifacts.update(self.plot_contour_plots(result, study))

        artifacts.update(self.plot_parallel_coordinates(result, study))

        artifacts.update(self.plot_trial_states(result))

        artifacts.update(self.plot_parameter_correlations(result))

        artifacts.update(self.plot_best_trials(result, top_n=top_n_trials))

        artifacts.update(self.plot_optimization_landscape_3d(result, study))

        artifacts.update(self.create_summary_dashboard(result))

        logger.success(f"Gerados {len(artifacts)} graficos de visualizacao")

        return artifacts

    def plot_optimization_history(
        self,
        result: OptimizationResult,
    ) -> dict[str, Path]:
        """
        Plot optimization history (score over trials).

        Args:
            result: Optimization result

        Returns:
            Dictionary with plot file paths
        """
        artifacts = {}

        if self.has_plotly:
            try:
                if self.has_optuna_viz:
                    logger.debug("Skipping Optuna optimization history (study not provided)")
                else:
                    fig = self.make_subplots(
                        rows=2,
                        cols=1,
                        subplot_titles=(
                            "Optimization Progress",
                            "Best Score Evolution",
                        ),
                        vertical_spacing=0.1,
                    )

                    history = result.study.optimization_history if hasattr(result, "study") else []
                    if not history:
                        history = result.get_optimization_history()

                    trial_numbers = [h[0] for h in history]
                    scores = [h[1] for h in history]

                    fig.add_trace(
                        self.go.Scatter(
                            x=trial_numbers,
                            y=scores,
                            mode="lines+markers",
                            name="Score",
                            line=dict(color="blue", width=2),
                            marker=dict(size=4),
                        ),
                        row=1,
                        col=1,
                    )

                    best_scores = []
                    current_best = float("-inf")
                    for score in scores:
                        current_best = max(current_best, score)
                        best_scores.append(current_best)

                    fig.add_trace(
                        self.go.Scatter(
                            x=trial_numbers,
                            y=best_scores,
                            mode="lines",
                            name="Best Score",
                            line=dict(color="red", width=3),
                        ),
                        row=2,
                        col=1,
                    )

                    fig.update_layout(
                        title="Optimization History",
                        showlegend=True,
                        height=800,
                    )

                    output_file = self.output_dir / "optimization_history.html"
                    self._save_plotly_html(fig, output_file)
                    artifacts["optimization_history"] = output_file

            except Exception as e:
                logger.warning(f"Failed to create optimization history plot: {e}")

        return artifacts

    def plot_parameter_importances(
        self,
        result: OptimizationResult,
        study: Any | None = None,
    ) -> dict[str, Path]:
        """
        Plot parameter importance scores.

        Args:
            result: Optimization result
            study: Optuna study object

        Returns:
            Dictionary with plot file paths
        """
        artifacts = {}

        if self.has_plotly and self.has_optuna_viz and study:
            try:
                fig = self.optuna_plot_funcs["param_importances"](study)

                output_file = self.output_dir / "parameter_importances.html"
                self._save_plotly_html(fig, output_file)
                artifacts["parameter_importances"] = output_file

            except Exception as e:
                logger.warning(f"Failed to create parameter importance plot: {e}")

        elif self.has_matplotlib:
            try:
                importances = result.get_param_importances()

                if importances:
                    params = list(importances.keys())
                    scores = list(importances.values())

                    sorted_indices = np.argsort(scores)[::-1]
                    params = [params[i] for i in sorted_indices]
                    scores = [scores[i] for i in sorted_indices]

                    self.plt.figure(figsize=(10, 6))
                    self.plt.barh(params, scores)
                    self.plt.xlabel("Importance")
                    self.plt.title("Parameter Importance")
                    self.plt.tight_layout()

                    output_file = self.output_dir / "parameter_importances.png"
                    self._save_matplotlib_png(output_file, dpi=300, bbox_inches="tight")
                    self.plt.close()

                    artifacts["parameter_importances"] = output_file

            except Exception as e:
                logger.warning(f"Failed to create parameter importance plot: {e}")

        return artifacts

    def plot_contour_plots(
        self,
        result: OptimizationResult,
        study: Any | None = None,
    ) -> dict[str, Path]:
        """
        Plot contour plots for parameter pairs.

        Args:
            result: Optimization result
            study: Optuna study object

        Returns:
            Dictionary with plot file paths
        """
        artifacts = {}

        if self.has_plotly and self.has_optuna_viz and study:
            try:
                fig = self.optuna_plot_funcs["contour"](study)

                output_file = self.output_dir / "contour_plots.html"
                self._save_plotly_html(fig, output_file)
                artifacts["contour_plots"] = output_file

            except Exception as e:
                logger.warning(f"Failed to create contour plots: {e}")

        return artifacts

    def plot_parallel_coordinates(
        self,
        result: OptimizationResult,
        study: Any | None = None,
    ) -> dict[str, Path]:
        """
        Plot parallel coordinates for all parameters.

        Args:
            result: Optimization result
            study: Optuna study object

        Returns:
            Dictionary with plot file paths
        """
        artifacts = {}

        if self.has_plotly and self.has_optuna_viz and study:
            try:
                fig = self.optuna_plot_funcs["parallel_coordinate"](study)

                output_file = self.output_dir / "parallel_coordinates.html"
                self._save_plotly_html(fig, output_file)
                artifacts["parallel_coordinates"] = output_file

            except Exception as e:
                logger.warning(f"Failed to create parallel coordinates plot: {e}")

        return artifacts

    def plot_trial_states(self, result: OptimizationResult) -> dict[str, Path]:
        """
        Plot distribution of trial states.

        Args:
            result: Optimization result

        Returns:
            Dictionary with plot file paths
        """
        artifacts = {}

        if self.has_matplotlib:
            try:
                trials = (
                    result.get("trials", [])
                    if isinstance(result, dict)
                    else getattr(result, "trials", [])
                )
                states = [
                    (
                        t.get("state", "COMPLETE")
                        if isinstance(t, dict)
                        else getattr(t, "state", "COMPLETE")
                    )
                    for t in trials
                ]
                state_counts = pl.Series(states).value_counts()

                self.plt.figure(figsize=(8, 8))
                colors = ["#2ecc71", "#e74c3c", "#f39c12", "#95a5a6"]
                try:
                    if "state" in state_counts.columns and "count" in state_counts.columns:
                        labels = (
                            state_counts["state"].to_list()
                            if "state" in state_counts.columns
                            else list(state_counts.select(pl.all().first()).to_numpy()[:, 0])
                        )
                        values = (
                            state_counts["count"].to_list()
                            if "count" in state_counts.columns
                            else list(state_counts.select(pl.all().first()).to_numpy()[:, 1])
                        )
                    elif len(state_counts.columns) >= 2:
                        labels = state_counts[:, 0].to_list()
                        values = state_counts[:, 1].to_list()
                    else:
                        col_names = state_counts.columns
                        labels = (
                            state_counts[col_names[0]].to_list()
                            if len(col_names) > 0
                            else list(state_counts.to_numpy()[:, 0])
                        )
                        values = (
                            state_counts[col_names[1]].to_list()
                            if len(col_names) > 1
                            else list(state_counts.to_numpy()[:, 1])
                        )
                except Exception as e:
                    logger.warning(f"Error extracting state counts: {e}")
                    labels = ["COMPLETE", "PRUNED", "FAILED"]
                    values = [len(states), 0, 0]

                wedges, texts, autotexts = self.plt.pie(
                    values,
                    labels=labels,
                    autopct="%1.1f%%",
                    colors=colors[: len(state_counts)],
                    startangle=90,
                )

                self.plt.title("Trial States Distribution")
                self.plt.axis("equal")

                output_file = self.output_dir / "trial_states.png"
                self._save_matplotlib_png(output_file, dpi=300, bbox_inches="tight")
                self.plt.close()

                artifacts["trial_states"] = output_file

            except Exception as e:
                logger.warning(f"Failed to create trial states plot: {e}")

        return artifacts

    def plot_parameter_correlations(
        self,
        result: OptimizationResult,
    ) -> dict[str, Path]:
        """
        Plot correlation heatmap between parameters and score.

        Args:
            result: Optimization result

        Returns:
            Dictionary with plot file paths
        """
        artifacts = {}

        if self.has_matplotlib and self.has_seaborn:
            try:
                trials = (
                    result.get("trials", [])
                    if isinstance(result, dict)
                    else getattr(result, "trials", [])
                )
                trials_dicts = [
                    (t.get("params", {}) if isinstance(t, dict) else getattr(t, "params", {}))
                    for t in trials
                ]
                trials_df = pl.DataFrame(trials_dicts)

                scores = pl.Series(
                    [
                        (t.get("value", 0) if isinstance(t, dict) else getattr(t, "value", 0))
                        for t in trials
                    ]
                )
                trials_df = trials_df.with_columns(score=scores)

                numeric_df = trials_df.select(pl.col(pl.Float64, pl.Int64))

                if len(numeric_df.columns) > 1:
                    import pyarrow.compute as pc

                    corr_data = pc.corr(numeric_df.to_arrow())

                    self.plt.figure(figsize=(12, 10))
                    self.sns.heatmap(
                        corr_data,
                        annot=True,
                        cmap="coolwarm",
                        center=0,
                        square=True,
                    )
                    self.plt.title("Parameter Correlations")
                    self.plt.tight_layout()

                    output_file = self.output_dir / "parameter_correlations.png"
                    self._save_matplotlib_png(output_file, dpi=300, bbox_inches="tight")
                    self.plt.close()

                    artifacts["parameter_correlations"] = output_file

            except Exception as e:
                logger.warning(f"Failed to create correlation plot: {e}")

        return artifacts

    def plot_optimization_landscape_3d(
        self,
        result: OptimizationResult,
        study: Any | None = None,
    ) -> dict[str, Path]:
        """
        Plot a 3D landscape of score vs three hyperparameters.

        The axes are selected using a variance-based heuristic on numeric parameters
        (`std / abs(mean)`). The surface is rendered with Plotly Mesh3d and the best
        trial is highlighted.

        Args:
            result: Optimization result
            study: Optuna study object

        Returns:
            Dictionary with plot file paths
        """
        artifacts = {}

        if self.has_plotly:
            try:
                trials = (
                    result.get("trials", [])
                    if isinstance(result, dict)
                    else getattr(result, "trials", [])
                )
                if not trials:
                    return artifacts

                data = []
                for t in trials:
                    params = (
                        t.get("params", {}) if isinstance(t, dict) else getattr(t, "params", {})
                    )
                    score = t.get("value", 0) if isinstance(t, dict) else getattr(t, "value", 0)
                    if score is None:
                        score = 0
                    row = params.copy()
                    row["score"] = score
                    row["trial_number"] = (
                        t.get("number", 0) if isinstance(t, dict) else getattr(t, "number", 0)
                    )
                    data.append(row)

                df = pl.DataFrame(data)

                param_cols = [c for c in df.columns if c not in ["score", "trial_number"]]
                if len(param_cols) < 3:
                    logger.warning("Need at least 3 parameters for 3D landscape plot")
                    return artifacts

                variances = {}
                for col in param_cols:
                    try:
                        if df[col].dtype in [pl.Float64, pl.Int64]:
                            std = df[col].std()
                            mean = df[col].mean()
                            if mean != 0:
                                variances[col] = std / abs(mean)
                            else:
                                variances[col] = 0
                    except Exception as exc:
                        logger.debug(f"3d_landscape_skip_param col={col} error={exc}")

                top_params = sorted(variances.items(), key=lambda x: x[1], reverse=True)[:3]
                x_col, y_col, z_col = [p[0] for p in top_params]

                pdf = df.to_pandas()

                mesh = self.go.Mesh3d(
                    x=pdf[x_col],
                    y=pdf[y_col],
                    z=pdf[z_col],
                    intensity=pdf["score"],
                    colorscale="Viridis",
                    colorbar_title="Score",
                    opacity=0.8,
                    alphahull=0,
                )

                best_idx = int(pdf["score"].idxmax())
                best_point = self.go.Scatter3d(
                    x=[pdf.loc[best_idx, x_col]],
                    y=[pdf.loc[best_idx, y_col]],
                    z=[pdf.loc[best_idx, z_col]],
                    mode="markers",
                    marker=dict(size=6, color="red", symbol="diamond"),
                    name="Best trial",
                )

                fig = self.go.Figure(data=[mesh, best_point])
                fig.update_layout(
                    title=f"Optimization Landscape (Top 3 Params: {x_col}, {y_col}, {z_col})",
                    scene=dict(
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        zaxis_title=z_col,
                    ),
                    height=900,
                )

                fig.update_layout(
                    scene=dict(
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        zaxis_title=z_col,
                    ),
                    height=900,
                )

                output_file = self.output_dir / "optimization_landscape_3d.html"
                self._save_plotly_html(fig, output_file)
                artifacts["optimization_landscape_3d"] = output_file
                logger.success(f"Grafico 3D gerado: {output_file}")

            except Exception as e:
                logger.warning(f"Failed to create 3D landscape plot: {e}")

        return artifacts

    def plot_best_trials(
        self,
        result: OptimizationResult,
        top_n: int = 20,
    ) -> dict[str, Path]:
        """
        Plot comparison of top N best trials.

        Args:
            result: Optimization result
            top_n: Number of top trials to show

        Returns:
            Dictionary with plot file paths
        """
        artifacts = {}

        if self.has_matplotlib:
            try:
                trials = (
                    result.get("trials", [])
                    if isinstance(result, dict)
                    else getattr(result, "trials", [])
                )

                sorted_trials = sorted(
                    trials,
                    key=lambda t: (
                        t.get("value", 0) if isinstance(t, dict) else getattr(t, "value", 0)
                    ),
                    reverse=True,
                )[:top_n]

                if len(sorted_trials) > 1:
                    all_params = set()
                    for trial in sorted_trials:
                        params = (
                            trial.get("params", {})
                            if isinstance(trial, dict)
                            else getattr(trial, "params", {})
                        )
                        all_params.update(params.keys())

                    n_params = len(all_params)
                    if n_params > 0:
                        fig, axes = self.plt.subplots(
                            min(n_params, 4),
                            1,
                            figsize=(12, 3 * min(n_params, 4)),
                            squeeze=False,
                        )

                        max_params = min(n_params, 4)
                        param_list = list(all_params)[:max_params]

                        for idx, param in enumerate(param_list):
                            row, col = idx // 1, idx % 1
                            ax = axes[row, col]

                            values = []
                            for t in sorted_trials:
                                if isinstance(t, dict):
                                    values.append(t.get("params", {}).get(param, 0))
                                else:
                                    values.append(getattr(t, "params", {}).get(param, 0))
                            trial_nums = list(range(len(sorted_trials)))

                            ax.bar(trial_nums, values)
                            ax.set_title(f"{param} (Top {len(sorted_trials)} Trials)")
                            ax.set_xlabel("Trial Rank")
                            ax.set_ylabel("Value")

                        self.plt.tight_layout()

                        output_file = self.output_dir / "best_trials_comparison.png"
                        self._save_matplotlib_png(output_file, dpi=300, bbox_inches="tight")
                        self.plt.close()

                        artifacts["best_trials_comparison"] = output_file

            except Exception as e:
                logger.warning(f"Failed to create best trials comparison plot: {e}")

        return artifacts

    def create_summary_dashboard(
        self,
        result: OptimizationResult,
    ) -> dict[str, Path]:
        """
        Create a comprehensive summary dashboard.

        Args:
            result: Optimization result

        Returns:
            Dictionary with dashboard file path
        """
        artifacts = {}

        if self.has_plotly:
            try:
                summary_stats = {
                    "Total Trials": (
                        result.get("n_trials", 0)
                        if isinstance(result, dict)
                        else getattr(result, "n_trials", 0)
                    ),
                    "Best Score": (
                        f"{result.get('best_value', 0):.4f}"
                        if isinstance(result, dict)
                        else f"{getattr(result, 'best_value', 0):.4f}"
                    ),
                    "Optimization Time": (
                        f"{result.get('optimization_time', 0):.2f}s"
                        if isinstance(result, dict)
                        else f"{getattr(result, 'optimization_time', 0):.2f}s"
                    ),
                    "Framework": (
                        result.get("framework", "unknown")
                        if isinstance(result, dict)
                        else getattr(result, "framework", "unknown")
                    ),
                }

                trials = (
                    result.get("trials", [])
                    if isinstance(result, dict)
                    else getattr(result, "trials", [])
                )
                states = [
                    (
                        t.get("state", "COMPLETE")
                        if isinstance(t, dict)
                        else getattr(t, "state", "COMPLETE")
                    )
                    for t in trials
                ]
                state_counts = pl.Series(states).value_counts()

                try:
                    if "state" in state_counts.columns and "count" in state_counts.columns:
                        completed_df = state_counts.filter(pl.col("state") == "COMPLETE")
                        n_completed = (
                            completed_df["count"].to_list()[0]
                            if "count" in completed_df.columns and len(completed_df) > 0
                            else (
                                completed_df.select(pl.all()).to_numpy()[0, 1]
                                if len(completed_df) > 0 and completed_df.shape[1] > 1
                                else 0
                            )
                        )

                        pruned_df = state_counts.filter(pl.col("state") == "PRUNED")
                        n_pruned = (
                            pruned_df["count"].to_list()[0]
                            if "count" in pruned_df.columns and len(pruned_df) > 0
                            else (
                                pruned_df.select(pl.all()).to_numpy()[0, 1]
                                if len(pruned_df) > 0 and pruned_df.shape[1] > 1
                                else 0
                            )
                        )
                    else:
                        completed_idx = None
                        pruned_idx = None
                        for i, col in enumerate(state_counts.columns):
                            col_data = state_counts[:, i].to_list()
                            if "COMPLETE" in col_data:
                                completed_idx = i
                            if "PRUNED" in col_data:
                                pruned_idx = i
                        n_completed = (
                            state_counts[:, completed_idx].to_list()[0]
                            if completed_idx is not None
                            else 0
                        )
                        n_pruned = (
                            state_counts[:, pruned_idx].to_list()[0]
                            if pruned_idx is not None
                            else 0
                        )
                except Exception as e:
                    logger.warning(f"Error extracting trial counts: {e}")
                    n_completed = states.count("COMPLETE")
                    n_pruned = states.count("PRUNED")

                n_trials = (
                    result.get("n_trials", len(trials))
                    if isinstance(result, dict)
                    else getattr(result, "n_trials", len(trials))
                )
                n_failed = n_trials - n_completed - n_pruned

                fig = self.make_subplots(
                    rows=3,
                    cols=2,
                    subplot_titles=(
                        "Optimization Summary",
                        "Trial States",
                        "Best Parameters",
                        "Performance Metrics",
                        "Notes",
                        "",
                    ),
                    specs=[
                        [{"type": "table"}, {"type": "pie"}],
                        [{"type": "table", "colspan": 2}, None],
                        [{"type": "table", "colspan": 2}, None],
                    ],
                    vertical_spacing=0.12,
                )

                fig.add_trace(
                    self.go.Table(
                        header=dict(
                            values=["Metric", "Value"],
                            fill_color="paleturquoise",
                            align="left",
                        ),
                        cells=dict(
                            values=[
                                list(summary_stats.keys()),
                                list(summary_stats.values()),
                            ],
                            fill_color="lavender",
                            align="left",
                        ),
                    ),
                    row=1,
                    col=2,
                )

                max_params = 10
                param_names = list(result.best_params.keys())[:max_params]
                param_values = [result.best_params[k] for k in param_names]

                fig.add_trace(
                    self.go.Pie(
                        labels=["Parameter", "Value"],
                        values=param_values,
                        marker=dict(colors=["#2ecc71", "#e74c3c", "#f39c12"]),
                    ),
                    row=1,
                    col=2,
                )

                fig.add_trace(
                    self.go.Table(
                        header=dict(
                            values=["Parameter", "Value"],
                            fill_color="lightblue",
                            align="left",
                        ),
                        cells=dict(
                            values=[param_names, param_values],
                            fill_color="lightcyan",
                            align="left",
                        ),
                    ),
                    row=2,
                    col=1,
                )

                metrics_data = {
                    "Metric": ["Completed", "Pruned", "Failed", "Success Rate"],
                    "Value": [
                        n_completed,
                        n_pruned,
                        n_failed,
                        f"{(n_completed / result.n_trials * 100):.1f}%",
                    ],
                }

                fig.add_trace(
                    self.go.Table(
                        header=dict(
                            values=["Metric", "Value"],
                            fill_color="lightgreen",
                            align="left",
                        ),
                        cells=dict(
                            values=[metrics_data["Metric"], metrics_data["Value"]],
                            fill_color="lightyellow",
                            align="left",
                        ),
                    ),
                    row=3,
                    col=1,
                )

                fig.update_layout(
                    title=f"Optimization Summary - {result.framework}",
                    height=1000,
                    showlegend=False,
                )

                output_file = self.output_dir / "summary_dashboard.html"
                self._save_plotly_html(fig, output_file)
                artifacts["summary_dashboard"] = output_file

            except Exception as e:
                logger.warning(f"Failed to create summary dashboard: {e}")
        return artifacts

    def generate_report(
        self,
        result: OptimizationResult,
        output_format: str = "html",
    ) -> Path:
        """
        Generate comprehensive optimization report.

        Args:
            result: Optimization result
            output_format: Output format ('html' or 'pdf')

        Returns:
            Path to report file
        """
        artifacts = self.generate_all_plots(result)

        if output_format == "html":
            report_html = self._create_html_report(result, artifacts)

            output_file = self.output_dir / "optimization_report.html"
            self.file_manager.write_text(report_html, output_file)

            logger.success(f"Relatório de otimização salvo: {output_file}")
            return output_file

        else:
            logger.warning(f"Output format '{output_format}' not yet implemented")
            return self.output_dir

    def _create_html_report(
        self,
        result: OptimizationResult,
        artifacts: dict[str, Path],
    ) -> str:
        """
        Create comprehensive HTML report.

        Args:
            result: Optimization result
            artifacts: Generated plot artifacts

        Returns:
            HTML string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Optimization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; }}
                .best-params {{ background: #e8f5e9; padding: 15px; border-left: 4px solid #4caf50; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #fff; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                .metric-label {{ font-size: 12px; color: #7f8c8d; }}
                .plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
                .plot-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                iframe {{ width: 100%; height: 500px; border: none; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1> Optimization Report</h1>

            <div class="summary">
                <h2>Summary</h2>
                <div class="metric">
                    <div class="metric-value">{result.best_value:.4f}</div>
                    <div class="metric-label">Best Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result.n_trials}</div>
                    <div class="metric-label">Total Trials</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result.optimization_time:.2f}s</div>
                    <div class="metric-label">Optimization Time</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result.framework}</div>
                    <div class="metric-label">Framework</div>
                </div>
            </div>

            <div class="best-params">
                <h2>Best Parameters</h2>
                <pre>{self.file_manager.json_dumps(result.best_params, sort_keys=True)}</pre>
            </div>

            <h2> Visualizations</h2>
            <div class="plots">
        """

        for plot_name, plot_path in artifacts.items():
            if self.file_manager.exists(plot_path):
                html += f"""
                <div class="plot-card">
                    <h3>{plot_name.replace("_", " ").title()}</h3>
                """

                if str(plot_path).endswith(".html"):
                    html += f'<iframe src="{plot_path}"></iframe>'
                else:
                    html += f'<img src="{plot_path}" alt="{plot_name}">'

                html += "</div>"

        html += """
            </div>
        </body>
        </html>
        """

        return html
