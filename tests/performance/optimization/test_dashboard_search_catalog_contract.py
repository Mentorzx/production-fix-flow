"""Static checks for dashboard search catalog richness contract."""

from __future__ import annotations

from pff.shared.core.config import settings

_CATALOG_PATH = (
    settings.PACKAGE_DIR
    / "infrastructure"
    / "hpo"
    / "dashboard"
    / "static"
    / "js"
    / "search"
    / "catalog.js"
)


def test_search_catalog_builds_detailed_descriptions() -> None:
    """Execute test search catalog builds detailed descriptions.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    assert _CATALOG_PATH.exists(), "catalog.js missing"
    content = _CATALOG_PATH.read_text(encoding="utf-8", errors="ignore")

    assert "buildDetailedDescription" in content, "Catalog must build rich descriptions"
    assert "registryMeta?.simple" in content, "Catalog must include simple hint text"
    assert "registryMeta?.tech" in content, "Catalog must include technical hint text"
    assert "describeExtra(registryMeta?.extra)" in content, "Catalog must index extra hint fields"
    assert "Seção:" in content, "Catalog descriptions must include section context"


def test_search_catalog_indexes_local_optima_diagnostics() -> None:
    """The local-optima diagnostics card must be reachable via search."""
    assert _CATALOG_PATH.exists(), "catalog.js missing"
    content = _CATALOG_PATH.read_text(encoding="utf-8", errors="ignore")

    assert "forecast-study-local-optima" in content, (
        "Catalog must include the local-optima diagnostics entry."
    )
    assert 'chartKey: "local_optima"' in content, (
        "Catalog must map the diagnostics entry to ChartRegistry local_optima metadata."
    )
