from __future__ import annotations

from scripts.benchmarks.search_space_advisor_paired_benchmark import (
    _space_patch_changed,
    run_benchmark,
)


def test_paired_benchmark_compares_tpe_advisor_and_ablations() -> None:
    payload = run_benchmark(
        seeds=[3, 5],
        scenarios=["smooth_kgc", "narrow_ridge"],
        n_trials=12,
        advisor_period=6,
        min_advisor_trials=6,
        isolate_advisor=False,
    )

    policies = {row["policy"]: row for row in payload["policies"]}

    assert payload["n_trials"] == 12
    assert payload["seeds"] == [3, 5]
    assert payload["scenarios"] == ["smooth_kgc", "narrow_ridge"]
    assert "random" in policies
    assert "tpe_pure" in policies
    assert "gp_bo" in policies
    assert "tpe_hyperband" in policies
    assert "advisor_full" in policies
    assert "advisor_gp_portfolio" in policies
    assert "advisor_static_gp" in policies
    assert "advisor_static_gp_guarded" in policies
    assert "advisor_edge_gated_gp" in policies
    assert "advisor_embedding_upper_gp" in policies
    assert "advisor_domain_edge_gp" in policies
    assert "advisor_trust_region_gp" in policies
    assert "advisor_no_bootstrap" in policies
    assert str(payload["claim_candidate_policy"]).startswith("advisor_")
    assert policies["tpe_pure"]["mean_delta_vs_tpe"] == 0.0
    assert "mean_delta_vs_gp_bo" in policies["advisor_full"]
    assert "mean_delta_vs_gp_bo_ci95" in policies["advisor_full"]
    assert "wilcoxon_greater_vs_gp_bo_pvalue" in policies["advisor_full"]
    assert policies["tpe_hyperband"]["sampler"] == "TPESampler"
    assert "friedman_pvalue" in payload
    assert "holm_vs_tpe_pure" in payload
    assert "holm_vs_gp_bo" in payload
    assert payload["claim_decision"]["candidate_policy"] == payload["claim_candidate_policy"]
    assert payload["claim_decision"]["scope"] == "synthetic_paired_benchmark_only"
    assert isinstance(payload["claim_decision"]["sota_vs_gp_bo_supported"], bool)
    assert any(row["policy"] == "advisor_full" for row in payload["holm_vs_gp_bo"])
    assert len(payload["scenario_summaries"]) == 2
    assert isinstance(payload["universal_superiority_claim_supported"], bool)
    assert all("best_value" in row and "scenario" in row for row in payload["runs"])
    assert all(len(row["best_curve"]) == payload["n_trials"] for row in payload["runs"])


def test_paired_benchmark_can_run_focused_gp_advisor_subset() -> None:
    payload = run_benchmark(
        seeds=[3],
        scenarios=["edge_capacity"],
        n_trials=12,
        advisor_period=6,
        min_advisor_trials=6,
        isolate_advisor=False,
        policy_names=["gp_bo", "advisor_embedding_upper_gp"],
    )

    policies = {row["policy"]: row for row in payload["policies"]}

    assert payload["policy_names"] == ["gp_bo", "advisor_embedding_upper_gp"]
    assert set(policies) == {"gp_bo", "advisor_embedding_upper_gp"}
    assert policies["advisor_embedding_upper_gp"]["mean_delta_vs_tpe"] is None
    assert policies["advisor_embedding_upper_gp"]["mean_delta_vs_gp_bo"] is not None
    assert payload["friedman_pvalue"] is None
    assert payload["holm_vs_tpe_pure"] == []
    assert len(payload["holm_vs_gp_bo"]) == 1
    assert payload["holm_vs_gp_bo"][0]["policy"] == "advisor_embedding_upper_gp"
    assert "holm_adjusted_pvalue" in payload["holm_vs_gp_bo"][0]
    assert not payload["claim_decision"]["sota_vs_gp_bo_supported"]
    assert all(len(row["best_curve"]) == payload["n_trials"] for row in payload["runs"])


def test_space_patch_changed_ignores_noop_recommendations() -> None:
    previous = {"embedding_dim": {"type": "int", "low": 64.0, "high": 512.0}}
    same = {"embedding_dim": {"type": "int", "low": 64.0, "high": 512.0}}
    changed = {"embedding_dim": {"type": "int", "low": 64.0, "high": 768.0}}

    assert not _space_patch_changed(previous, same, ["embedding_dim"])
    assert _space_patch_changed(previous, changed, ["embedding_dim"])
