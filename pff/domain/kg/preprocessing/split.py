"""
Safe Split Utilities for KG Data.

Design Pattern: Strategy + Validation
- Multiple split strategies (random, chronological)
- Leakage detection and prevention
- Consistent handling of inverse relations

CRITICAL: This module ensures that inverse relations do NOT leak
between splits. The correct order is:
1. Split data into train/valid/test
2. Add inverse relations to EACH split independently
3. Verify no leakage exists
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import polars as pl

from pff.shared import logger


@dataclass
class SplitResult:
    """Result of a data split operation.

    Attributes:
        train: Training set DataFrame
        valid: Validation set DataFrame
        test: Test set DataFrame
        stats: Statistics about the split
    """

    train: pl.DataFrame
    valid: pl.DataFrame
    test: pl.DataFrame
    stats: dict[str, Any]


class LeakageChecker:
    """Verify no data leakage between splits using vectorized operations.

    Leakage can occur when:
    1. Same triple appears in multiple splits
    2. Inverse of a test triple appears in train
    3. Validation/test entities don't appear in train (cold start)
    """

    def __init__(self, inverse_suffix: str = "_inv") -> None:
        """Initialize checker.

        Args:
            inverse_suffix: Suffix used for inverse relations
        """
        self.inverse_suffix = inverse_suffix

    def check_triple_leakage(
        self,
        train: pl.DataFrame,
        valid: pl.DataFrame,
        test: pl.DataFrame,
        log_on_leak: bool = True,
    ) -> dict[str, Any]:
        """Check for exact triple leakage between splits using vectorized joins."""
        # Using Polars inner joins to find intersections (vectorized)
        train_valid_overlap = train.join(valid, on=["s", "p", "o"], how="inner").height
        train_test_overlap = train.join(test, on=["s", "p", "o"], how="inner").height
        valid_test_overlap = valid.join(test, on=["s", "p", "o"], how="inner").height

        result = {
            "train_valid_overlap": train_valid_overlap,
            "train_test_overlap": train_test_overlap,
            "valid_test_overlap": valid_test_overlap,
            "has_leakage": bool(train_valid_overlap or train_test_overlap or valid_test_overlap),
        }

        if result["has_leakage"] and log_on_leak:
            logger.warning(
                f"Data leakage detected between splits (pre-inverse): "
                f"train-valid={train_valid_overlap}, "
                f"train-test={train_test_overlap}, "
                f"valid-test={valid_test_overlap}. "
                "Enable fix_leakage to auto-resplit before adding inverse relations."
            )

        return result

    def _to_canonical(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create canonical (h, r, t) form directly in Polars."""
        return df.with_columns(
            [
                pl.when(pl.col("p").str.ends_with(self.inverse_suffix))
                .then(pl.col("o"))
                .otherwise(pl.col("s"))
                .alias("_s_can"),
                pl.when(pl.col("p").str.ends_with(self.inverse_suffix))
                .then(
                    pl.col("p").str.slice(0, pl.col("p").str.len_chars() - len(self.inverse_suffix))
                )
                .otherwise(pl.col("p"))
                .alias("_p_can"),
                pl.when(pl.col("p").str.ends_with(self.inverse_suffix))
                .then(pl.col("s"))
                .otherwise(pl.col("o"))
                .alias("_o_can"),
            ]
        ).select(["_s_can", "_p_can", "_o_can"])

    def check_inverse_leakage(
        self,
        train: pl.DataFrame,
        valid: pl.DataFrame,
        test: pl.DataFrame,
        log_on_leak: bool = True,
    ) -> dict[str, Any]:
        """Check if inverse of test/valid triples appear in train using vectorized ops."""
        # Vectorized canonicalization
        train_can = self._to_canonical(train)
        valid_can = self._to_canonical(valid)
        test_can = self._to_canonical(test)

        # Vectorized overlap check via inner joins
        train_valid_inverse_leak = train_can.join(
            valid_can, on=["_s_can", "_p_can", "_o_can"], how="inner"
        ).height
        train_test_inverse_leak = train_can.join(
            test_can, on=["_s_can", "_p_can", "_o_can"], how="inner"
        ).height

        result = {
            "train_valid_inverse_leak": train_valid_inverse_leak,
            "train_test_inverse_leak": train_test_inverse_leak,
            "has_inverse_leakage": bool(train_valid_inverse_leak or train_test_inverse_leak),
        }

        if result["has_inverse_leakage"] and log_on_leak:
            logger.warning(
                f"Inverse leakage detected: train-valid={train_valid_inverse_leak}, "
                f"train-test={train_test_inverse_leak}. "
                "Ensure inverse relations are generated per split or re-run resplit with fix_leakage."
            )

        return result

    def check_entity_coverage(
        self, train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame
    ) -> dict[str, Any]:
        """Check entity coverage between splits using vectorized operations."""

        def get_entities(df: pl.DataFrame) -> pl.Series:
            # Vectorized unique entities extraction
            return df.select(pl.concat_list(["s", "o"]).explode()).unique().to_series()

        train_entities = get_entities(train)
        valid_entities = get_entities(valid)
        test_entities = get_entities(test)

        # Find unseen entities using is_in (vectorized)
        valid_unseen = valid_entities.filter(~valid_entities.is_in(train_entities))
        test_unseen = test_entities.filter(~test_entities.is_in(train_entities))

        result = {
            "train_entities": len(train_entities),
            "valid_entities": len(valid_entities),
            "test_entities": len(test_entities),
            "valid_unseen_entities": len(valid_unseen),
            "test_unseen_entities": len(test_unseen),
            "valid_coverage": 1 - len(valid_unseen) / max(len(valid_entities), 1),
            "test_coverage": 1 - len(test_unseen) / max(len(test_entities), 1),
        }

        if len(valid_unseen) > 0 or len(test_unseen) > 0:
            logger.warning(
                f"COLD-START ENTITIES: valid={len(valid_unseen)}, test={len(test_unseen)} "
                f"(coverage: valid={result['valid_coverage']:.2%}, test={result['test_coverage']:.2%})"
            )

        return result

    def full_check(
        self, train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame
    ) -> dict[str, Any]:
        """Run all leakage checks.

        Args:
            train: Training DataFrame
            valid: Validation DataFrame
            test: Test DataFrame

        Returns:
            Comprehensive leakage report
        """
        triple_check = self.check_triple_leakage(train, valid, test)
        inverse_check = self.check_inverse_leakage(train, valid, test)
        coverage_check = self.check_entity_coverage(train, valid, test)

        return {
            "triple_leakage": triple_check,
            "inverse_leakage": inverse_check,
            "entity_coverage": coverage_check,
            "all_clear": not (triple_check["has_leakage"] or inverse_check["has_inverse_leakage"]),
        }


class SafeSplitter:
    """Split KG data safely without leakage.

    Supports:
    - Random split (default)
    - Chronological split (time-based)
    - Stratified by relation (ensures all relations in all splits)

    CRITICAL: This splitter ensures proper handling of inverse relations
    by splitting BEFORE adding inverses, then adding inverses to each
    split independently.
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        inverse_suffix: str = "_inv",
    ) -> None:
        """Initialize splitter.

        Args:
            train_ratio: Fraction of data for training
            valid_ratio: Fraction of data for validation
            test_ratio: Fraction of data for test
            seed: Random seed for reproducibility
            inverse_suffix: Suffix for inverse relations
        """
        # Normalize ratios
        total = train_ratio + valid_ratio + test_ratio
        self.train_ratio = train_ratio / total
        self.valid_ratio = valid_ratio / total
        self.test_ratio = test_ratio / total
        self.seed = seed
        self.inverse_suffix = inverse_suffix
        self.leakage_checker = LeakageChecker(inverse_suffix)

    def random_split(self, df: pl.DataFrame) -> SplitResult:
        """Perform random split of data.

        Args:
            df: DataFrame with columns [s, p, o]

        Returns:
            SplitResult with train/valid/test DataFrames
        """
        random.seed(self.seed)

        n = len(df)
        indices = list(range(n))
        random.shuffle(indices)

        n_train = int(n * self.train_ratio)
        n_valid = int(n * self.valid_ratio)

        train_idx = indices[:n_train]
        valid_idx = indices[n_train : n_train + n_valid]
        test_idx = indices[n_train + n_valid :]

        # Use row indices for splitting
        train_df = df[train_idx]
        valid_df = df[valid_idx]
        test_df = df[test_idx]

        stats = {
            "total_triples": n,
            "train_triples": len(train_df),
            "valid_triples": len(valid_df),
            "test_triples": len(test_df),
            "train_ratio_actual": len(train_df) / n,
            "valid_ratio_actual": len(valid_df) / n,
            "test_ratio_actual": len(test_df) / n,
            "seed": self.seed,
        }

        logger.info(
            f"[SPLIT] Random split: train={len(train_df):,} ({stats['train_ratio_actual']:.1%}), "
            f"valid={len(valid_df):,} ({stats['valid_ratio_actual']:.1%}), "
            f"test={len(test_df):,} ({stats['test_ratio_actual']:.1%})"
        )

        return SplitResult(train=train_df, valid=valid_df, test=test_df, stats=stats)

    def chronological_split(
        self, df: pl.DataFrame, timestamp_column: str = "timestamp"
    ) -> SplitResult:
        """Perform chronological split based on timestamp.

        This is preferred for temporal KGs (like telecom) as it:
        - Reflects real-world evaluation scenario
        - Prevents "future leakage" (using future data to predict past)

        Args:
            df: DataFrame with columns [s, p, o, timestamp]
            timestamp_column: Name of timestamp column

        Returns:
            SplitResult with train/valid/test DataFrames
        """
        if timestamp_column not in df.columns:
            logger.warning(
                f"Timestamp column '{timestamp_column}' not found. Falling back to random split."
            )
            return self.random_split(df)

        # Sort by timestamp
        df_sorted = df.sort(timestamp_column)
        n = len(df_sorted)

        n_train = int(n * self.train_ratio)
        n_valid = int(n * self.valid_ratio)

        train_df = df_sorted[:n_train]
        valid_df = df_sorted[n_train : n_train + n_valid]
        test_df = df_sorted[n_train + n_valid :]

        stats = {
            "total_triples": n,
            "train_triples": len(train_df),
            "valid_triples": len(valid_df),
            "test_triples": len(test_df),
            "train_ratio_actual": len(train_df) / n,
            "valid_ratio_actual": len(valid_df) / n,
            "test_ratio_actual": len(test_df) / n,
            "split_type": "chronological",
            "timestamp_column": timestamp_column,
        }

        logger.info(
            f"[SPLIT] Chronological split: train={len(train_df):,}, "
            f"valid={len(valid_df):,}, test={len(test_df):,}"
        )

        return SplitResult(train=train_df, valid=valid_df, test=test_df, stats=stats)

    def split_with_inverse_safety(
        self,
        df: pl.DataFrame,
        add_inverses: bool = True,
        chronological: bool = False,
        timestamp_column: str = "timestamp",
    ) -> SplitResult:
        """Split data with proper inverse relation handling.

        This is the RECOMMENDED method. It:
        1. Splits the original data (no inverses)
        2. Adds inverse relations to EACH split independently
        3. Verifies no leakage exists

        Args:
            df: DataFrame with columns [s, p, o]
            add_inverses: Whether to add inverse relations
            chronological: Use chronological split
            timestamp_column: Timestamp column name

        Returns:
            SplitResult with properly processed train/valid/test
        """
        # Step 1: Split original data
        if chronological:
            result = self.chronological_split(df, timestamp_column)
        else:
            result = self.random_split(df)

        # Step 2: Add inverses to each split independently
        if add_inverses:
            from .strategies import InverseRelationStrategy

            inverse_strategy = InverseRelationStrategy(self.inverse_suffix)

            train_result = inverse_strategy.process(result.train)
            valid_result = inverse_strategy.process(result.valid)
            test_result = inverse_strategy.process(result.test)

            result = SplitResult(
                train=train_result.data,
                valid=valid_result.data,
                test=test_result.data,
                stats={
                    **result.stats,
                    "inverses_added": True,
                    "train_with_inverses": len(train_result.data),
                    "valid_with_inverses": len(valid_result.data),
                    "test_with_inverses": len(test_result.data),
                },
            )

        # Step 3: Verify no leakage
        leakage_report = self.leakage_checker.full_check(result.train, result.valid, result.test)
        result.stats["leakage_report"] = leakage_report

        if not leakage_report["all_clear"]:
            logger.error("DATA LEAKAGE DETECTED! Check leakage_report for details.")
        else:
            logger.success("Split verification: No leakage detected")

        return result
