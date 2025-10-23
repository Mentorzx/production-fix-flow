from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from joblib import dump

from pff.config import settings
from pff.utils import logger
from pff.validators.ensembles.ensemble_wrappers import TransEWrapper
from pff.validators.ensembles.kgc_metrics import KGCEvaluator, KGCMetrics


@dataclass
class BaselineModel:
    name: str
    model: Any
    description: str
    is_fitted: bool = False


class BaselineComparator:
    def __init__(self, kg_config_path: str, transe_config_path: str, rules_path: str):
        self.kg_config_path = kg_config_path
        self.transe_config_path = transe_config_path
        self.rules_path = rules_path
        self.baselines: dict[str, BaselineModel] = {}
        self.results: dict[str, dict[str, float]] = {}
        self.evaluator = None

    def add_individual_models(self) -> None:
        logger.info("🧩 Adicionando modelos individuais como baselines...")
        transe_model = TransEWrapper(
            kg_config_path=self.kg_config_path,
            transe_config_path=self.transe_config_path,
        )
        self.baselines["TransE_Individual"] = BaselineModel(
            name="TransE_Individual",
            model=transe_model,
            description="Heterogeneous Graph Transformer isolado",
        )

    def create_all_baselines(self) -> None:
        logger.info("🔧 Criando todos os modelos baseline...")
        self.add_individual_models()
        logger.info(f"✅ {len(self.baselines)} modelos baseline criados:")
        for name, baseline in self.baselines.items():
            logger.info(f"   - {name}: {baseline.description}")

    def fit_all_baselines(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info("🏋️  Treinando todos os modelos baseline...")
        for name, baseline in self.baselines.items():
            logger.info(f"   Treinando {name}...")
            try:
                baseline.model.fit(X_train, y_train)
                baseline.is_fitted = True
                logger.info(f"   ✅ {name} treinado com sucesso")
            except Exception as e:
                logger.error(f"   ❌ Erro ao treinar {name}: {e}")
                baseline.is_fitted = False

    def evaluate_all_baselines(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, dict[str, float]]:
        logger.info("📊 Avaliando todos os modelos baseline...")
        results = {}
        for name, baseline in self.baselines.items():
            if not baseline.is_fitted:
                logger.warning(f"   ⚠️  Pulando {name} (não treinado)")
                continue
            logger.info(f"   Avaliando {name}...")
            try:
                y_pred = baseline.model.predict(X_test)
                y_proba = baseline.model.predict_proba(X_test)
                y_scores = y_proba[:, 1]
                metrics = KGCMetrics.calculate_all_metrics(y_test, y_scores, y_pred)
                results[name] = metrics
                logger.info(
                    f"   ✅ {name}: MRR={metrics['mrr']:.4f}, Hits@10={metrics['hits@10']:.4f}"
                )
            except Exception as e:
                logger.error(f"   ❌ Erro ao avaliar {name}: {e}")
                results[name] = {}

        self.results = results
        return results

    def compare_with_target_model(
        self,
        target_model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "TargetModel",
    ) -> dict[str, Any]:
        logger.info(f"🎯 Comparando baselines com {model_name}...")
        evaluator = KGCEvaluator(target_model)
        target_results = evaluator.evaluate_detailed(X_test, y_test)
        target_metrics = target_results["kgc_metrics"]
        comparison = {
            "target_model": {"name": model_name, "metrics": target_metrics},
            "baselines": self.results,
            "improvements": {},
            "rankings": {},
        }
        for metric in ["mrr", "hits@1", "hits@10", "accuracy", "auc_roc"]:
            if metric in target_metrics:
                target_value = target_metrics[metric]
                metric_improvements = {}
                for baseline_name, baseline_metrics in self.results.items():
                    if metric in baseline_metrics:
                        baseline_value = baseline_metrics[metric]
                        if baseline_value > 0:
                            improvement = (
                                (target_value - baseline_value) / baseline_value
                            ) * 100
                            metric_improvements[baseline_name] = improvement
                comparison["improvements"][metric] = metric_improvements
        for metric in ["mrr", "hits@1", "hits@10", "accuracy"]:
            if metric in target_metrics:
                all_scores = [(model_name, target_metrics[metric])]
                for baseline_name, baseline_metrics in self.results.items():
                    if metric in baseline_metrics:
                        all_scores.append((baseline_name, baseline_metrics[metric]))
                all_scores.sort(key=lambda x: x[1], reverse=True)
                comparison["rankings"][metric] = all_scores

        self._print_comparison_report(comparison)
        return comparison

    def _print_comparison_report(self, comparison: dict[str, Any]) -> None:
        target_name = comparison["target_model"]["name"]
        target_metrics = comparison["target_model"]["metrics"]
        print("\n" + "=" * 80)
        print(f"📊 RELATÓRIO DE COMPARAÇÃO - {target_name}")
        print("=" * 80)
        print(f"\n🎯 Performance do {target_name}:")
        for metric, value in target_metrics.items():
            print(f"   {metric:12}: {value:.4f}")
        print("\n🏆 Rankings por Métrica:")
        for metric, ranking in comparison["rankings"].items():
            print(f"\n   {metric.upper()}:")
            for i, (model, score) in enumerate(ranking, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                marker = " ← TARGET" if model == target_name else ""
                print(f"   {emoji} {i}. {model:20}: {score:.4f}{marker}")
        print(f"\n📈 Melhorias do {target_name} sobre baselines:")
        for metric, improvements in comparison["improvements"].items():
            print(f"\n   {metric.upper()}:")
            sorted_improvements = sorted(
                improvements.items(), key=lambda x: x[1], reverse=True
            )
            for baseline, improvement in sorted_improvements:
                sign = "+" if improvement >= 0 else ""
                print(f"      vs {baseline:20}: {sign}{improvement:6.1f}%")
        print("=" * 80 + "\n")

    def create_comparison_plots(
        self, comparison: dict[str, Any], save_path: Path | None = None
    ) -> None:
        logger.info("📈 Criando gráficos de comparação...")
        metrics = ["mrr", "hits@1", "hits@10", "accuracy", "auc_roc"]
        available_metrics = [
            m for m in metrics if m in comparison["target_model"]["metrics"]
        ]
        models = [comparison["target_model"]["name"]]
        models.extend(list(self.results.keys()))
        plot_data = []
        for model in models:
            if model == comparison["target_model"]["name"]:
                model_metrics = comparison["target_model"]["metrics"]
            else:
                model_metrics = self.results.get(model, {})
            for metric in available_metrics:
                if metric in model_metrics:
                    plot_data.append(
                        {
                            "Model": model,
                            "Metric": metric,
                            "Score": model_metrics[metric],
                            "IsTarget": model == comparison["target_model"]["name"],
                        }
                    )
        # Polars DataFrame (5x-54x mais rápido que pandas)
        df = pl.DataFrame(plot_data).to_pandas()  # Convert para pandas apenas para plotting (seaborn requer pandas)
        if df.empty:
            logger.warning("Nenhum dado disponível para plotting")
            return
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Comparação de Performance dos Modelos", fontsize=16, fontweight="bold"
        )
        ax1 = axes[0, 0]
        pivot_df = df.pivot(index="Model", columns="Metric", values="Score")
        pivot_df.plot(kind="bar", ax=ax1, rot=45)
        ax1.set_title("Performance por Métrica")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2 = axes[0, 1]
        self._create_radar_chart(ax2, comparison, available_metrics)
        ax3 = axes[1, 0]
        improvements_data = []
        for metric, improvements in comparison["improvements"].items():
            for baseline, improvement in improvements.items():
                improvements_data.append(
                    {
                        "Baseline": baseline,
                        "Metric": metric,
                        "Improvement_%": improvement,
                    }
                )
        if improvements_data:
            # Polars DataFrame (5x-54x mais rápido que pandas)
            imp_df = pl.DataFrame(improvements_data).to_pandas()  # Convert para pandas apenas para seaborn (requer pandas)
            imp_pivot = imp_df.pivot(
                index="Baseline", columns="Metric", values="Improvement_%"
            )
            sns.heatmap(
                imp_pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax3
            )
            ax3.set_title("Melhorias (%) sobre Baselines")
        ax4 = axes[1, 1]
        ranking_data = []
        for metric, ranking in comparison["rankings"].items():
            for rank, (model, score) in enumerate(ranking, 1):
                ranking_data.append(
                    {"Model": model, "Metric": metric, "Rank": rank, "Score": score}
                )
        if ranking_data:
            # Polars DataFrame (5x-54x mais rápido que pandas)
            rank_df = pl.DataFrame(ranking_data).to_pandas()  # Convert para pandas apenas para seaborn (requer pandas)
            rank_pivot = rank_df.pivot(index="Model", columns="Metric", values="Rank")
            sns.heatmap(rank_pivot, annot=True, fmt=".0f", cmap="RdYlGn_r", ax=ax4)
            ax4.set_title("Rankings (1=melhor)")
        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"📊 Gráficos salvos em: {save_path}")
        else:
            plt.show()

    def _create_radar_chart(
        self, ax, comparison: dict[str, Any], metrics: list[str]
    ) -> None:
        target_name = comparison["target_model"]["name"]
        target_metrics = comparison["target_model"]["metrics"]
        best_baseline_metrics = {}
        for metric in metrics:
            best_score = 0
            for baseline_name, baseline_metrics in self.results.items():
                if metric in baseline_metrics and baseline_metrics[metric] > best_score:
                    best_score = baseline_metrics[metric]
            best_baseline_metrics[metric] = best_score
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Fechar o círculo
        target_values = [target_metrics.get(m, 0) for m in metrics] + [
            target_metrics.get(metrics[0], 0)
        ]
        baseline_values = [best_baseline_metrics.get(m, 0) for m in metrics] + [
            best_baseline_metrics.get(metrics[0], 0)
        ]
        ax.plot(
            angles, target_values, "o-", linewidth=2, label=target_name, color="red"
        )
        ax.fill(angles, target_values, alpha=0.25, color="red")
        ax.plot(
            angles,
            baseline_values,
            "o-",
            linewidth=2,
            label="Melhor Baseline",
            color="blue",
        )
        ax.fill(angles, baseline_values, alpha=0.25, color="blue")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title("Comparação Radar: Target vs Melhor Baseline")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

    def save_comparison_results(
        self, comparison: dict[str, Any], save_path: Path | None = None
    ) -> Path:
        save_path = save_path or settings.OUTPUTS_DIR / "baseline_comparison.joblib"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dump(
            {
                "comparison": comparison,
                "baselines": self.baselines,
                "results": self.results,
            },
            save_path,
        )
        logger.success(f"💾 Comparação salva em: {save_path}")
        return save_path


def run_comprehensive_comparison(
    ensemble_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    kg_config_path: str,
    transe_config_path: str,
    rules_path: str,
) -> dict[str, Any]:
    logger.info("🚀 Iniciando comparação abrangente...")
    comparator = BaselineComparator(kg_config_path, transe_config_path, rules_path)
    comparator.create_all_baselines()
    comparator.fit_all_baselines(X_train, y_train)
    comparator.evaluate_all_baselines(X_test, y_test)
    comparison = comparator.compare_with_target_model(
        ensemble_model, X_test, y_test, "StackingEnsemble"
    )
    plot_path = settings.OUTPUTS_DIR / "comparison_plots.png"
    comparator.create_comparison_plots(comparison, plot_path)
    comparator.save_comparison_results(comparison)

    logger.success("✅ Comparação abrangente concluída!")
    return comparison
