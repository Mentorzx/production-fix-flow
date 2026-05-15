from __future__ import annotations

from pathlib import Path

from scripts.research.deep_research_advisor_artifacts import (
    build_article,
    plot_advisor_evidence,
    plot_advisor_flow,
)


def _sample_metrics() -> dict[str, object]:
    return {
        "reliability_summary": {
            "validation_pass_rate": 1.0,
            "validation_pass_wilson_lb": 0.84,
            "mean_confidence_score": 0.38,
            "high_confidence_rate": 0.0,
        },
        "self_audit": {
            "prefixes_evaluated": 2,
            "directional_signals_total": 4,
            "directional_hit_rate": 0.5,
            "directional_hit_rate_wilson_lb": 0.15,
        },
        "objective": {
            "sampler": "tpe",
            "direction_input": "maximize",
            "direction_normalized": "maximize",
            "dashboard_best_value": 0.261075,
            "hpo_summary_best_value": 0.255363,
            "optimization_time_seconds": 41.86,
        },
        "trial_counts": {
            "hpo_summary_n_trials": 15,
            "completed_trials_in_dashboard_payload": 14,
            "advisor_completed_trials": 14,
        },
        "data_info": {
            "source": "real",
            "n_train": 10231,
            "n_valid": 1240,
            "n_entities": 5470,
            "n_predicates": 30,
        },
        "advisor": {
            "advisor_version": "2.3.0",
            "n_recommendations": 20,
            "search_space_coverage_ratio": 1.0,
            "action_counts": {"keep": 9, "fix": 6, "expand_lower": 3, "reduce_categories": 2},
        },
        "figures": {
            "flow": "figures/figura0_fluxo_advisor.png",
            "advisor_evidence": "figures/figura1_evidencias_advisor.png",
            "importances_actions": "figures/figura2_importancias_acoes.png",
            "topk_distributions": "figures/figura3_topk_distribuicoes.png",
            "reliability": "figures/figura4_confiabilidade.png",
            "pareto": "figures/figura5_pareto_qualidade_tempo.png",
            "survival": "figures/figura6_sobrevivencia_direcional.png",
        },
        "pareto_quality_time": {
            "pareto_front_size": 2,
            "pareto_trial_ids": [8, 11],
        },
        "ablations": [
            {
                "name": "full_no_adaptive",
                "metadata_compute_time_ms": 60.0,
                "n_recommendations": 20,
                "action_counts": {"keep": 9, "fix": 6},
                "mean_confidence_score": 0.58,
                "validation_pass_wilson_lb": 0.84,
                "directional_hit_rate_wilson_lb": 0.36,
            },
            {
                "name": "no_bootstrap",
                "metadata_compute_time_ms": 17.0,
                "n_recommendations": 20,
                "action_counts": {"keep": 9, "fix": 6},
                "mean_confidence_score": 0.43,
                "validation_pass_wilson_lb": 0.84,
                "directional_hit_rate_wilson_lb": 0.36,
            },
        ],
        "paired_benchmark": {
            "n_trials": 50,
            "seeds": [11, 17],
            "advisor_period": 10,
            "universal_superiority_claim_supported": False,
            "policies": [
                {
                    "policy": "tpe_pure",
                    "sampler": "TPESampler",
                    "mean_best_value": 0.8,
                    "mean_delta_vs_tpe": 0.0,
                    "mean_delta_vs_gp_bo": -0.1,
                    "wins_vs_tpe": 0,
                    "wilcoxon_greater_pvalue": None,
                },
                {
                    "policy": "advisor_full",
                    "sampler": "TPESampler",
                    "mean_best_value": 0.84,
                    "mean_delta_vs_tpe": 0.04,
                    "mean_delta_vs_gp_bo": -0.06,
                    "wins_vs_tpe": 1,
                    "wilcoxon_greater_pvalue": 0.25,
                },
            ],
        },
        "source_artifacts": {
            "dashboard": "outputs/.cache/hpo/dashboard_data.json",
            "hpo_summary": "outputs/optimization/kg_dslfm/hpo_summary.json",
            "advisor_audit": "outputs/benches/search_space_advisor/deep_research_audit_20260506.json",
            "technical_doc": "src/pff/infrastructure/hpo/SEARCH_SPACE_ADVISOR.md",
        },
        "docker_experiment_command": "./scripts/package/pff-run hpo --trials 50 --no-update-config --no-bert --no-dashboard --study-name deep_research_advisor_real_20260506",
        "audit_command": "python scripts/benchmarks/search_space_advisor_audit.py --input outputs/.cache/hpo/dashboard_data.json",
    }


def test_build_article_inlines_examples_and_cleans_figure_captions() -> None:
    article = build_article(_sample_metrics(), [])

    assert "Confiabilidade matemática de um Search Space Advisor" in article
    assert "## Exemplos operacionais das fórmulas" not in article
    assert article.count("Exemplo prático.") >= 5
    assert "## Prova basal e evidência empírica" in article
    assert "**Definição 1 — Estado do Advisor.**" in article
    assert "**Lema 1 — Top-k não vazio e limitado.**" in article
    assert "**Teorema 1 — Admissibilidade estrutural condicional.**" in article
    assert "**Potencial de proteção intelectual.**" in article
    assert "## Tabela 10 — ablations e sensibilidade do Advisor" in article
    assert "## Tabela 11 — benchmark pareado TPE, GP-BO e Advisor" in article
    assert "não foi satisfeita" in article
    assert "`no_bootstrap`" in article
    assert "![Fluxo lógico do Search Space Advisor.]" in article
    assert "## Figuras geradas a partir do payload do Advisor" in article
    assert "Evolução do objetivo e incumbente" not in article
    assert "![Figura 1 -" not in article
    assert "Fonte: elaborado pelo autor" not in article
    assert "*Fonte: elaboração própria" in article


def test_plot_advisor_flow_writes_image(tmp_path: Path) -> None:
    output = tmp_path / "fluxo.png"

    plot_advisor_flow(output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_advisor_evidence_writes_image(tmp_path: Path) -> None:
    output = tmp_path / "evidencias.png"
    recommendations = [
        {"action": "keep", "confidence": "low", "validation": {"passed": True}},
        {"action": "fix", "confidence": "medium", "validation": {"passed": True}},
        {"action": "expand_lower", "confidence": "medium", "validation": {"passed": False}},
    ]
    audit = {
        "metadata": {
            "search_space_coverage_ratio": 1.0,
            "reliability_summary": {
                "validation_pass_wilson_lb": 0.5,
                "mean_confidence_score": 0.42,
            },
            "self_audit": {"directional_hit_rate_wilson_lb": 0.25},
        }
    }

    plot_advisor_evidence(recommendations, audit, output)

    assert output.exists()
    assert output.stat().st_size > 0
