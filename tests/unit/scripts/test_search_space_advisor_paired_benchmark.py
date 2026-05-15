from __future__ import annotations

from scripts.benchmarks.search_space_advisor_paired_benchmark import run_benchmark


def test_paired_benchmark_compares_tpe_advisor_and_ablations() -> None:
    payload = run_benchmark(
        seeds=[3, 5],
        n_trials=12,
        advisor_period=6,
        min_advisor_trials=6,
    )

    policies = {row["policy"]: row for row in payload["policies"]}

    assert payload["n_trials"] == 12
    assert payload["seeds"] == [3, 5]
    assert "random" in policies
    assert "tpe_pure" in policies
    assert "gp_bo" in policies
    assert "advisor_full" in policies
    assert "advisor_gp_portfolio" in policies
    assert "advisor_static_gp" in policies
    assert "advisor_static_gp_guarded" in policies
    assert "advisor_no_bootstrap" in policies
    assert str(payload["claim_candidate_policy"]).startswith("advisor_")
    assert policies["tpe_pure"]["mean_delta_vs_tpe"] == 0.0
    assert "mean_delta_vs_gp_bo" in policies["advisor_full"]
    assert isinstance(payload["universal_superiority_claim_supported"], bool)
    assert all("best_value" in row for row in payload["runs"])
