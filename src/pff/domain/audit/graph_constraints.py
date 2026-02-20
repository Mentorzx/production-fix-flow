"""Deterministic graph constraints (SHACL-like) for audit workflows.

Design patterns:
    - Interpreter: evaluates a small, declarative constraint language over triples.
    - Adapter: emits a SHACL-like validation report structure consumable by the
      audit report Builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True)
class GraphConstraintViolation:
    """A single deterministic constraint violation over triples."""

    focus_node: str
    result_path: str
    value: str | None
    constraint: str
    message: str
    json_pointer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Execute to dict.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        payload = {
            "focus_node": self.focus_node,
            "result_path": self.result_path,
            "value": self.value,
            "constraint": self.constraint,
            "message": self.message,
        }
        if self.json_pointer is not None:
            payload["json_pointer"] = self.json_pointer
        return payload


class GraphConstraintsValidator:
    """Validate triples against a small set of deterministic constraints."""

    def __init__(self, *, constraints: dict[str, Any]) -> None:
        """Execute init.



        Args:

            constraints: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self._constraints = constraints

    @staticmethod
    def _build_triples_frame(triples: list[dict[str, Any]]) -> pl.DataFrame | None:
        """Execute build triples frame.



        Args:

            triples: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        try:
            frame = pl.DataFrame(triples)
            return frame.with_columns(
                [
                    pl.col("s").cast(pl.Utf8).fill_null(""),
                    pl.col("p").cast(pl.Utf8).fill_null(""),
                    pl.col("o").cast(pl.Utf8).fill_null(""),
                ]
            )
        except Exception:
            return None

    @staticmethod
    def _json_pointer_expr(df: pl.DataFrame) -> pl.Expr:
        """Execute json pointer expr.



        Args:

            df: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if "json_pointer" in df.columns:
            return pl.col("json_pointer").cast(pl.Utf8)
        return pl.lit(None).alias("json_pointer")

    def _collect_forbidden_violations(
        self,
        *,
        df: pl.DataFrame,
        forbidden_predicates: set[str],
    ) -> list[dict[str, Any]]:
        """Execute collect forbidden violations.



        Args:

            df: Input value used by this callable.

            forbidden_predicates: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if not forbidden_predicates:
            return []
        bad_preds_df = df.filter(pl.col("p").is_in(forbidden_predicates))
        if bad_preds_df.is_empty():
            return []
        v_df = bad_preds_df.select(
            pl.col("s").alias("focus_node"),
            pl.col("p").alias("result_path"),
            pl.col("o").alias("value"),
            pl.lit("forbidden_predicate").alias("constraint"),
            (pl.lit("Forbidden predicate used: predicate=") + pl.col("p")).alias("message"),
            self._json_pointer_expr(df),
        )
        return v_df.to_dicts()

    def _collect_allowed_value_violations(
        self,
        *,
        df: pl.DataFrame,
        allowed_by_predicate: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Execute collect allowed value violations.



        Args:

            df: Input value used by this callable.

            allowed_by_predicate: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        violations: list[dict[str, Any]] = []
        for predicate, allowed in allowed_by_predicate.items():
            if not allowed:
                continue
            allowed_list = list(map(str, allowed))
            bad_values_df = df.filter(
                (pl.col("p") == predicate) & (~pl.col("o").is_in(allowed_list))
            )
            if bad_values_df.is_empty():
                continue
            v_df = bad_values_df.select(
                pl.col("s").alias("focus_node"),
                pl.col("p").alias("result_path"),
                pl.col("o").alias("value"),
                pl.lit("allowed_values").alias("constraint"),
                (
                    pl.lit("Value not allowed for predicate: predicate=")
                    + pl.col("p")
                    + pl.lit(" value=")
                    + pl.col("o")
                ).alias("message"),
                self._json_pointer_expr(df),
            )
            violations.extend(v_df.to_dicts())
        return violations

    @staticmethod
    def _collect_max_cardinality_violations(
        *,
        df: pl.DataFrame,
        max_cardinality_by_predicate: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Execute collect max cardinality violations.



        Args:

            df: Input value used by this callable.

            max_cardinality_by_predicate: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if not max_cardinality_by_predicate:
            return []
        counts = df.group_by(["s", "p"]).len().rename({"len": "count"})
        violations: list[dict[str, Any]] = []
        for predicate, limit in max_cardinality_by_predicate.items():
            try:
                limit_int = int(limit)
            except (TypeError, ValueError):
                continue
            if limit_int < 0:
                continue
            over_limit = counts.filter((pl.col("p") == predicate) & (pl.col("count") > limit_int))
            if over_limit.is_empty():
                continue
            v_df = over_limit.select(
                pl.col("s").alias("focus_node"),
                pl.col("p").alias("result_path"),
                pl.lit(None).alias("value"),
                pl.lit("max_cardinality").alias("constraint"),
                (
                    pl.lit("Max cardinality exceeded: predicate=")
                    + pl.col("p")
                    + pl.lit(" count=")
                    + pl.col("count").cast(pl.Utf8)
                    + pl.lit(f" limit={limit_int}")
                ).alias("message"),
                pl.lit(None).alias("json_pointer"),
            )
            violations.extend(v_df.to_dicts())
        return violations

    def validate(
        self,
        triples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate triples and return a SHACL-like validation report.

        Uses Polars for vectorized validation of constraints.
        """
        if not triples:
            return []

        df = self._build_triples_frame(triples)
        if df is None:
            return []

        max_card_by_p = self._constraints.get("max_cardinality_by_predicate", {})
        allowed_by_p = self._constraints.get("allowed_values_by_predicate", {})
        forbidden_p = set(self._constraints.get("forbidden_predicates", []) or [])

        violations: list[dict[str, Any]] = []
        violations.extend(
            self._collect_forbidden_violations(df=df, forbidden_predicates=forbidden_p)
        )
        violations.extend(
            self._collect_allowed_value_violations(df=df, allowed_by_predicate=allowed_by_p)
        )
        violations.extend(
            self._collect_max_cardinality_violations(
                df=df,
                max_cardinality_by_predicate=max_card_by_p,
            )
        )

        return violations
