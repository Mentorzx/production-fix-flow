from __future__ import annotations

import numpy as np
import polars as pl


def default_anyburl_metrics(conf_threshold: float, support_threshold: float) -> dict[str, float]:
    """Default AnyBURL metrics payload."""
    return {
        "rule_count": 0.0,
        "avg_confidence": 0.0,
        "avg_support": 0.0,
        "high_confidence_ratio": 0.0,
        "applied_conf_threshold": float(conf_threshold),
        "applied_support_threshold": float(support_threshold),
    }


def compute_entity_quality_scores(train_df: pl.DataFrame, valid_df: pl.DataFrame) -> dict[str, float]:
    """Blend connectivity signals into a normalized entity quality score."""

    def _count(df: pl.DataFrame, column: str, alias: str) -> pl.DataFrame:
        if df.is_empty():
            return pl.DataFrame({"entity": [], alias: []})
        return df.group_by(column).agg(pl.len().alias(alias)).rename({column: "entity"})

    def _relation_span(df: pl.DataFrame, column: str, alias: str) -> pl.DataFrame:
        if df.is_empty():
            return pl.DataFrame({"entity": [], alias: []})
        return df.group_by(column).agg(pl.n_unique("p").alias(alias)).rename({column: "entity"})

    train_out = _count(train_df, "s", "train_out")
    train_in = _count(train_df, "o", "train_in")
    valid_out = _count(valid_df, "s", "valid_out")
    valid_in = _count(valid_df, "o", "valid_in")
    train_rel = _relation_span(train_df, "s", "train_rel")
    valid_rel = _relation_span(valid_df, "s", "valid_rel")

    # Use coalesce joins to avoid duplicate column issues
    stats = (
        train_out.lazy()
        .join(train_in.lazy(), on="entity", how="full", coalesce=True)
        .join(valid_out.lazy(), on="entity", how="full", coalesce=True)
        .join(valid_in.lazy(), on="entity", how="full", coalesce=True)
        .join(train_rel.lazy(), on="entity", how="full", coalesce=True)
        .join(valid_rel.lazy(), on="entity", how="full", coalesce=True)
        .with_columns([pl.all().exclude("entity").fill_null(0.0)])
        .with_columns(
            [
                (pl.col("train_out") + pl.col("train_in")).alias("train_total"),
                (pl.col("valid_out") + pl.col("valid_in")).alias("valid_total"),
            ]
        )
        .with_columns(
            [
                (pl.col("train_total") + 1.0).log1p().alias("train_signal"),
                (pl.col("valid_total") + 1.0).log1p().alias("valid_signal"),
                (pl.col("train_rel") + pl.col("valid_rel") + 1.0).log1p().alias("relation_signal"),
            ]
        )
        .with_columns(
            (
                0.55 * pl.col("train_signal")
                + 0.25 * pl.col("valid_signal")
                + 0.20 * pl.col("relation_signal")
            ).alias("quality_raw")
        )
        .select(["entity", "quality_raw"])
        .collect()
    )

    if stats.is_empty():
        return {}

    min_val = float(stats["quality_raw"].min())
    max_val = float(stats["quality_raw"].max())
    span = max(max_val - min_val, 1e-9)

    stats = stats.with_columns(((pl.col("quality_raw") - min_val) / span).alias("quality_score"))

    return {row["entity"]: float(row["quality_score"]) for row in stats.select(["entity", "quality_score"]).to_dicts()}
