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
        self._constraints = constraints

    def validate(
        self,
        triples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate triples and return a SHACL-like validation report.

        Uses Polars for vectorized validation of constraints.
        """
        if not triples:
            return []

        try:
            df = pl.DataFrame(triples)

            df = df.with_columns(
                [
                    pl.col("s").cast(pl.Utf8).fill_null(""),
                    pl.col("p").cast(pl.Utf8).fill_null(""),
                    pl.col("o").cast(pl.Utf8).fill_null(""),
                ]
            )
        except Exception:
            return []

        max_card_by_p = self._constraints.get("max_cardinality_by_predicate", {})
        allowed_by_p = self._constraints.get("allowed_values_by_predicate", {})
        forbidden_p = set(self._constraints.get("forbidden_predicates", []) or [])

        violations: list[dict[str, Any]] = []

        if forbidden_p:
            bad_preds_df = df.filter(pl.col("p").is_in(forbidden_p))
            if not bad_preds_df.is_empty():
                v_df = bad_preds_df.select(
                    pl.col("s").alias("focus_node"),
                    pl.col("p").alias("result_path"),
                    pl.col("o").alias("value"),
                    pl.lit("forbidden_predicate").alias("constraint"),
                    (pl.lit("Forbidden predicate used: predicate=") + pl.col("p")).alias("message"),
                    pl.col("json_pointer").cast(pl.Utf8)
                    if "json_pointer" in df.columns
                    else pl.lit(None).alias("json_pointer"),
                )
                violations.extend(v_df.to_dicts())

        for pred, allowed in allowed_by_p.items():
            if not allowed:
                continue
            allowed_list = list(map(str, allowed))

            bad_values_df = df.filter((pl.col("p") == pred) & (~pl.col("o").is_in(allowed_list)))

            if not bad_values_df.is_empty():
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
                    pl.col("json_pointer").cast(pl.Utf8)
                    if "json_pointer" in df.columns
                    else pl.lit(None).alias("json_pointer"),
                )
                violations.extend(v_df.to_dicts())

        if max_card_by_p:
            counts = df.group_by(["s", "p"]).len().rename({"len": "count"})

            for pred, limit in max_card_by_p.items():
                try:
                    limit_int = int(limit)
                except (TypeError, ValueError):
                    continue

                if limit_int < 0:
                    continue

                over_limit = counts.filter((pl.col("p") == pred) & (pl.col("count") > limit_int))

                if not over_limit.is_empty():
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
