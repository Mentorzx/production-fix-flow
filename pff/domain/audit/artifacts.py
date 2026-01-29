"""Audit artifact directory layout.

The audit pipeline writes only under `outputs/` per the project contract.
This module defines a stable, versionable directory layout rooted at:

    outputs/audit/<run_id>/

Subdirectories are designed to mirror the roadmap layers:
    - canonical/: JSON canonicalization + provenance tables
    - schema/: JSON Schema validation outputs
    - profile/: statistical profiling + drift vs baseline
    - graph/: graph-level findings and repairs
    - report/: final audit_report.json and attachments
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from collections.abc import Callable


@dataclass(frozen=True)
class AuditArtifactPaths:
    """Resolved paths for a single audit run.

    Attributes:
        run_root: Root directory for the run: outputs/audit/<run_id>.
        canonical_dir: Canonicalization/provenance outputs.
        schema_dir: JSON Schema validation outputs.
        profile_dir: Statistical profile + drift outputs.
        graph_dir: Graph/neuro-symbolic outputs.
        report_dir: Final report outputs.
        report_path: Path for the final audit report JSON.
    """

    run_root: Path
    canonical_dir: Path
    schema_dir: Path
    profile_dir: Path
    graph_dir: Path
    report_dir: Path
    report_path: Path

    @classmethod
    def for_run(
        cls,
        *,
        outputs_dir: Path,
        run_id: str,
        report_filename: str = "audit_report.json",
    ) -> AuditArtifactPaths:
        """Construct the canonical artifact layout for a run.

        Args:
            outputs_dir: Root outputs directory.
            run_id: Stable run identifier.
            report_filename: Filename for the final report.

        Returns:
            AuditArtifactPaths instance.
        """
        run_root = outputs_dir / "audit" / run_id
        canonical_dir = run_root / "canonical"
        schema_dir = run_root / "schema"
        profile_dir = run_root / "profile"
        graph_dir = run_root / "graph"
        report_dir = run_root / "report"
        report_path = report_dir / report_filename
        return cls(
            run_root=run_root,
            canonical_dir=canonical_dir,
            schema_dir=schema_dir,
            profile_dir=profile_dir,
            graph_dir=graph_dir,
            report_dir=report_dir,
            report_path=report_path,
        )

    def ensure(self, ensure_dir: Callable[[Path], None]) -> None:
        """Ensure all directories in the layout exist.

        Args:
            ensure_dir: Callable that ensures a directory exists.
        """
        ensure_dir(self.run_root)
        ensure_dir(self.canonical_dir)
        ensure_dir(self.schema_dir)
        ensure_dir(self.profile_dir)
        ensure_dir(self.graph_dir)
        ensure_dir(self.report_dir)
