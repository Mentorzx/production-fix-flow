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

    def validate(  # noqa: PLR0912
        self,
        triples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate triples and return a SHACL-like validation report.

        Uses Polars for vectorized validation of constraints.
        """
        if not triples:
            return []

        # Convert to Polars DataFrame for vectorized operations
        # We enforce string types for s, p, o to match the domain model
        try:
            df = pl.DataFrame(triples)
            # Ensure required columns exist even if triples is partial (unlikely given type hint)
            # and cast to string.
            df = df.with_columns(
                [
                    pl.col("s").cast(pl.Utf8).fill_null(""),
                    pl.col("p").cast(pl.Utf8).fill_null(""),
                    pl.col("o").cast(pl.Utf8).fill_null(""),
                ]
            )
        except Exception:
            # Fallback for malformed inputs
            return []

        max_card_by_p = self._constraints.get("max_cardinality_by_predicate", {})
        allowed_by_p = self._constraints.get("allowed_values_by_predicate", {})
        forbidden_p = set(self._constraints.get("forbidden_predicates", []) or [])

        violations: list[GraphConstraintViolation] = []

        # 1. Forbidden Predicates
        if forbidden_p:
            bad_preds = df.filter(pl.col("p").is_in(forbidden_p))
            if not bad_preds.is_empty():
                for row in bad_preds.iter_rows(named=True):
                    violations.append(
                        GraphConstraintViolation(
                            focus_node=row["s"],
                            result_path=row["p"],
                            value=row["o"],
                            constraint="forbidden_predicate",
                            message=f"Forbidden predicate used: predicate={row['p']}",
                            json_pointer=(
                                str(row["json_pointer"])
                                if row.get("json_pointer")
                                else None
                            ),
                        )
                    )

        # 2. Allowed Values
        # Iterate over constraints (schema size) rather than data size
        for pred, allowed in allowed_by_p.items():
            if not allowed:
                continue
            allowed_set = {str(v) for v in allowed}
            # Find rows with this predicate BUT object NOT in allowed
            bad_values = df.filter(
                (pl.col("p") == pred) & (~pl.col("o").is_in(allowed_set))
            )
            if not bad_values.is_empty():
                for row in bad_values.iter_rows(named=True):
                    violations.append(
                        GraphConstraintViolation(
                            focus_node=row["s"],
                            result_path=row["p"],
                            value=row["o"],
                            constraint="allowed_values",
                            message=(
                                f"Value not allowed for predicate: "
                                f"predicate={row['p']} value={row['o']}"
                            ),
                            json_pointer=(
                                str(row["json_pointer"])
                                if row.get("json_pointer")
                                else None
                            ),
                        )
                    )

        # 3. Max Cardinality
        if max_card_by_p:
            # Count occurrences of (s, p)
            # group_by is efficient
            counts = df.group_by(["s", "p"]).count()

            for pred, limit in max_card_by_p.items():
                try:
                    limit_int = int(limit)
                except (TypeError, ValueError):
                    continue

                if limit_int < 0:
                    continue

                # Filter for this predicate where count > limit
                over_limit = counts.filter(
                    (pl.col("p") == pred) & (pl.col("count") > limit_int)
                )

                if not over_limit.is_empty():
                    for row in over_limit.iter_rows(named=True):
                        violations.append(
                            GraphConstraintViolation(
                                focus_node=row["s"],
                                result_path=row["p"],
                                value=None,
                                constraint="max_cardinality",
                                message=(
                                    f"Max cardinality exceeded: predicate={row['p']} "
                                    f"count={row['count']} limit={limit_int}"
                                ),
                            )
                        )

        return [v.to_dict() for v in violations]
