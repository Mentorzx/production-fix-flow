/**
 * Global search catalog for dashboard command palette.
 */

import { ChartRegistry } from "../domain/metrics/ChartRegistry.js";
import { buildSearchBlob, normalizeText } from "./normalization.js";

const ITEM_DEFS = [
  {
    id: "tab-overview",
    domId: "panel-overview",
    chartKey: "convergence",
    title: "Aba Monitoramento",
    description: "Navega para a aba de monitoramento e visão geral do estudo/trial.",
    tabId: "overview",
    viewMode: "study",
    sectionPath: "Navegação > Aba",
    aliases: ["overview", "monitoramento", "visao geral"],
    tags: ["tab", "navigation"],
  },
  {
    id: "tab-analysis",
    domId: "panel-analysis",
    chartKey: "fanova",
    title: "Aba Análise",
    description: "Navega para a aba de análise com gráficos de sensibilidade e diagnóstico.",
    tabId: "analysis",
    viewMode: "study",
    sectionPath: "Navegação > Aba",
    aliases: ["analysis", "analise"],
    tags: ["tab", "navigation"],
  },
  {
    id: "tab-advanced",
    domId: "panel-advanced",
    chartKey: "scatter_plot",
    title: "Aba Avançado",
    description: "Navega para a aba avançada com slices, métricas e telemetria.",
    tabId: "advanced",
    viewMode: "study",
    sectionPath: "Navegação > Aba",
    aliases: ["advanced", "avancado"],
    tags: ["tab", "navigation"],
  },
  {
    id: "tab-forecast",
    domId: "panel-forecast",
    chartKey: "regression_chart",
    title: "Aba Previsão",
    description: "Navega para a aba de previsão, regressão e advisor de search space.",
    tabId: "forecast",
    viewMode: "study",
    sectionPath: "Navegação > Aba",
    aliases: ["forecast", "previsao"],
    tags: ["tab", "navigation"],
  },
  {
    id: "overview-study-incumbent-trajectory",
    chartKey: "hypervolume",
    tabId: "overview",
    viewMode: "study",
    sectionPath: "Monitoramento > Visão Geral",
    aliases: ["incumbent trajectory", "best so far", "trajetoria incumbent"],
    tags: ["overview", "kpi", "mrr"],
  },
  {
    id: "overview-study-best-trial",
    chartKey: "params",
    tabId: "overview",
    viewMode: "study",
    sectionPath: "Monitoramento > Visão Geral",
    aliases: ["melhor trial", "best trial", "incumbent"],
    tags: ["overview", "params"],
  },
  {
    id: "overview-study-detailed-history",
    chartKey: "detailed_history",
    tabId: "overview",
    viewMode: "study",
    sectionPath: "Monitoramento > Histórico",
    aliases: ["ranking trials", "historico detalhado", "tabela de trials"],
    tags: ["tabela", "trials", "historico"],
  },
  {
    id: "overview-trial-learning-metrics",
    chartKey: "trial_learning_metrics",
    tabId: "overview",
    viewMode: "trial",
    sectionPath: "Monitoramento > Trial Atual",
    aliases: ["loss mcc mrr", "aprendizado trial"],
    tags: ["trial", "learning", "mcc", "mrr"],
  },
  {
    id: "overview-trial-fold-confusions",
    chartKey: "fold_confusions",
    tabId: "overview",
    viewMode: "trial",
    sectionPath: "Monitoramento > Trial Atual",
    aliases: ["matriz folds", "fold matrix"],
    tags: ["trial", "confusion", "fold"],
  },
  {
    id: "overview-trial-full-metrics-log",
    chartKey: "full_metrics_log",
    tabId: "overview",
    viewMode: "trial",
    sectionPath: "Monitoramento > Trial Atual",
    aliases: ["log epocas", "metrics per epoch"],
    tags: ["trial", "epoch", "metrics", "log"],
  },
  {
    id: "analysis-study-param-importance",
    chartKey: "fanova",
    tabId: "analysis",
    viewMode: "study",
    sectionPath: "Análise > Sensibilidade & Impacto",
    sectionKey: "analysis-sensitivity",
    aliases: ["importancia parametros", "fanova", "feature importance"],
    tags: ["analysis", "importance"],
  },
  {
    id: "analysis-study-correlation",
    chartKey: "correlation",
    tabId: "analysis",
    viewMode: "study",
    sectionPath: "Análise > Sensibilidade & Impacto",
    sectionKey: "analysis-sensitivity",
    aliases: ["correlacao", "heatmap correlacao"],
    tags: ["analysis", "correlation"],
  },
  {
    id: "analysis-study-parallel",
    chartKey: "parallel",
    tabId: "analysis",
    viewMode: "study",
    sectionPath: "Análise > Espaço de Busca & Interações",
    sectionKey: "analysis-search",
    aliases: ["coordenadas paralelas", "parallel coordinates"],
    tags: ["analysis", "search-space"],
  },
  {
    id: "analysis-study-interaction",
    chartKey: "interaction",
    tabId: "analysis",
    viewMode: "study",
    sectionPath: "Análise > Espaço de Busca & Interações",
    sectionKey: "analysis-search",
    aliases: ["interacoes", "interaction plot"],
    tags: ["analysis", "interactions"],
  },
  {
    id: "analysis-study-search-space-table",
    chartKey: "search_space_table",
    tabId: "analysis",
    viewMode: "study",
    sectionPath: "Análise > Espaço de Busca & Interações",
    sectionKey: "analysis-search",
    aliases: ["espaco de busca", "search space"],
    tags: ["analysis", "search-space", "table"],
  },
  {
    id: "analysis-study-pareto",
    chartKey: "pareto_front",
    tabId: "analysis",
    viewMode: "study",
    sectionPath: "Análise > Performance & Trade-offs",
    sectionKey: "analysis-tradeoffs",
    aliases: ["pareto", "tradeoff"],
    tags: ["analysis", "pareto"],
  },
  {
    id: "analysis-study-confusion-matrix",
    chartKey: "confusion_matrix",
    tabId: "analysis",
    viewMode: "study",
    sectionPath: "Análise > Performance & Trade-offs",
    sectionKey: "analysis-tradeoffs",
    aliases: ["matriz confusao", "confusion matrix"],
    tags: ["analysis", "classification"],
  },
  {
    id: "analysis-study-edf",
    chartKey: "edf",
    tabId: "analysis",
    viewMode: "study",
    sectionPath: "Análise > Performance & Trade-offs",
    sectionKey: "analysis-tradeoffs",
    aliases: ["edf", "funcao distribuicao"],
    tags: ["analysis", "distribution"],
  },
  {
    id: "analysis-study-hypervolume",
    chartKey: "hypervolume",
    tabId: "analysis",
    viewMode: "study",
    sectionPath: "Análise > Performance & Trade-offs",
    sectionKey: "analysis-tradeoffs",
    aliases: ["best so far", "incumbent curve"],
    tags: ["analysis", "best-so-far"],
  },
  {
    id: "analysis-study-contour",
    chartKey: "contour",
    tabId: "analysis",
    viewMode: "study",
    sectionPath: "Análise > Performance & Trade-offs",
    sectionKey: "analysis-tradeoffs",
    aliases: ["contour", "superficie resposta"],
    tags: ["analysis", "surface"],
  },
  {
    id: "analysis-study-timeline",
    chartKey: "timeline",
    tabId: "analysis",
    viewMode: "study",
    sectionPath: "Análise > Diagnóstico de Execução",
    sectionKey: "analysis-diagnostics",
    aliases: ["timeline", "duracao trials"],
    tags: ["analysis", "diagnostics", "timeline"],
  },
  {
    id: "analysis-study-structural",
    chartKey: "structural_metrics",
    tabId: "analysis",
    viewMode: "study",
    sectionPath: "Análise > Diagnóstico de Execução",
    sectionKey: "analysis-diagnostics",
    aliases: ["metricas estruturais", "complexidade"],
    tags: ["analysis", "diagnostics", "structure"],
  },
  {
    id: "analysis-study-latency-pareto",
    chartKey: "latency_pareto",
    tabId: "analysis",
    viewMode: "study",
    sectionPath: "Análise > Diagnóstico de Execução",
    sectionKey: "analysis-diagnostics",
    aliases: ["latencia qualidade", "latency quality"],
    tags: ["analysis", "latency", "pareto"],
  },
  {
    id: "analysis-study-pc-comparison",
    chartKey: "pc_comparison",
    tabId: "analysis",
    viewMode: "study",
    sectionPath: "Análise > Diagnóstico de Execução",
    sectionKey: "analysis-diagnostics",
    aliases: ["comparacao pc", "pc table"],
    tags: ["analysis", "pc2"],
  },
  {
    id: "analysis-trial-learning-curve",
    chartKey: "learning_curve",
    tabId: "analysis",
    viewMode: "trial",
    sectionPath: "Análise > Aprendizado & Convergência",
    sectionKey: "analysis-trial-learning",
    aliases: ["learning curve", "curva aprendizado"],
    tags: ["analysis", "trial"],
  },
  {
    id: "analysis-trial-elbo",
    chartKey: "elbo_breakdown",
    tabId: "analysis",
    viewMode: "trial",
    sectionPath: "Análise > Decomposição de Perda",
    sectionKey: "analysis-trial-loss",
    aliases: ["elbo", "recon kl"],
    tags: ["analysis", "trial", "loss"],
  },
  {
    id: "analysis-trial-pc2-metrics",
    chartKey: "pc2_metrics",
    tabId: "analysis",
    viewMode: "trial",
    sectionPath: "Análise > Decomposição de Perda",
    sectionKey: "analysis-trial-loss",
    aliases: ["pc2", "metricas pc2"],
    tags: ["analysis", "trial", "pc2"],
  },
  {
    id: "analysis-trial-terminal-log",
    chartKey: "terminal_log",
    tabId: "analysis",
    viewMode: "trial",
    sectionPath: "Análise > Logs & Histórico",
    sectionKey: "analysis-trial-logs",
    aliases: ["terminal", "logs"],
    tags: ["analysis", "trial", "logs"],
  },
  {
    id: "advanced-study-slice-lr",
    chartKey: "scatter_plot",
    tabId: "advanced",
    viewMode: "study",
    sectionPath: "Avançado > Análise Marginal",
    sectionKey: "advanced-marginal",
    aliases: ["slice lr", "learning rate x objetivo"],
    tags: ["advanced", "slice", "scatter"],
  },
  {
    id: "advanced-study-slice-embed",
    chartKey: "scatter_plot",
    tabId: "advanced",
    viewMode: "study",
    sectionPath: "Avançado > Análise Marginal",
    sectionKey: "advanced-marginal",
    aliases: ["slice embedding", "embed x objetivo"],
    tags: ["advanced", "slice", "scatter"],
  },
  {
    id: "advanced-study-duration-score",
    chartKey: "scatter_plot",
    tabId: "advanced",
    viewMode: "study",
    sectionPath: "Avançado > Dinâmica de Performance",
    sectionKey: "advanced-dynamics",
    aliases: ["duracao score", "duration score"],
    tags: ["advanced", "scatter", "duration"],
  },
  {
    id: "advanced-study-metrics-evolution",
    chartKey: "metrics_evolution",
    tabId: "advanced",
    viewMode: "study",
    sectionPath: "Avançado > Dinâmica de Performance",
    sectionKey: "advanced-dynamics",
    aliases: ["evolucao metricas", "metrics evolution"],
    tags: ["advanced", "metrics"],
  },
  {
    id: "advanced-trial-hardware",
    chartKey: "hardware_monitor",
    tabId: "advanced",
    viewMode: "trial",
    sectionPath: "Avançado > Saúde do Sistema",
    sectionKey: "advanced-health",
    aliases: ["hardware", "cpu gpu ram"],
    tags: ["advanced", "trial", "hardware"],
  },
  {
    id: "advanced-trial-gradient",
    chartKey: "gradient_health",
    tabId: "advanced",
    viewMode: "trial",
    sectionPath: "Avançado > Saúde do Sistema",
    sectionKey: "advanced-health",
    aliases: ["gradiente", "gradient health", "grad norm"],
    tags: ["advanced", "trial", "gradient"],
  },
  {
    id: "advanced-trial-raw-config",
    chartKey: "raw_config",
    tabId: "advanced",
    viewMode: "trial",
    sectionPath: "Avançado > Configuração",
    sectionKey: "advanced-config",
    aliases: ["config bruta", "raw config"],
    tags: ["advanced", "trial", "config"],
  },
  {
    id: "forecast-study-estimated-score",
    chartKey: "estimated_score",
    tabId: "forecast",
    viewMode: "study",
    sectionPath: "Previsão > Estimativas Futuras",
    sectionKey: "forecast-future",
    aliases: ["score final", "estimativa score"],
    tags: ["forecast", "projection"],
  },
  {
    id: "forecast-study-optimization-velocity",
    chartKey: "optimization_velocity",
    tabId: "forecast",
    viewMode: "study",
    sectionPath: "Previsão > Estimativas Futuras",
    sectionKey: "forecast-future",
    aliases: ["velocidade otimizacao", "optimization velocity"],
    tags: ["forecast", "velocity"],
  },
  {
    id: "forecast-study-regression-chart",
    chartKey: "regression_chart",
    tabId: "forecast",
    viewMode: "study",
    sectionPath: "Previsão > Tendência e Regressão",
    sectionKey: "forecast-regression",
    aliases: ["regressao score", "trend chart"],
    tags: ["forecast", "regression"],
  },
  {
    id: "forecast-study-regression-insights",
    chartKey: "regression_insights",
    tabId: "forecast",
    viewMode: "study",
    sectionPath: "Previsão > Tendência e Regressão",
    sectionKey: "forecast-regression",
    aliases: ["insights regressao", "r2 slope"],
    tags: ["forecast", "regression"],
  },
  {
    id: "forecast-study-local-optima",
    chartKey: "local_optima",
    tabId: "forecast",
    viewMode: "study",
    sectionPath: "Previsão > Estagnação & Exploração",
    sectionKey: "forecast-local-optima",
    aliases: ["minimo local", "otimo local", "estagnacao", "multi regiao", "local optima"],
    tags: ["forecast", "diagnostics", "stagnation"],
  },
  {
    id: "forecast-study-trial-diff",
    chartKey: "trial_diff",
    tabId: "forecast",
    viewMode: "study",
    sectionPath: "Previsão > Comparativo de Trials",
    sectionKey: "forecast-comparison",
    aliases: ["comparativo trials", "trial diff"],
    tags: ["forecast", "table", "trials"],
  },
  {
    id: "forecast-study-search-space-advisor",
    chartKey: "search_space_table",
    title: "Search Space Advisor",
    description:
      "Tabela de recomendações para expansão/redução/fix do espaço de busca com confiança e importância.",
    tabId: "forecast",
    viewMode: "study",
    sectionPath: "Previsão > Search Space Advisor",
    sectionKey: "forecast-advisor",
    aliases: ["advisor", "recomendacoes espaco busca", "search advisor"],
    tags: ["forecast", "advisor", "search-space"],
  },
  {
    id: "forecast-trial-loss-projection",
    chartKey: "loss_projection",
    tabId: "forecast",
    viewMode: "trial",
    sectionPath: "Previsão > Previsão do Trial",
    sectionKey: "forecast-trial",
    aliases: ["projecao loss", "loss projection"],
    tags: ["forecast", "trial", "projection"],
  },
  {
    id: "forecast-trial-generalization-gap",
    chartKey: "generalization_gap",
    tabId: "forecast",
    viewMode: "trial",
    sectionPath: "Previsão > Previsão do Trial",
    sectionKey: "forecast-trial",
    aliases: ["gap generalizacao", "generalization gap"],
    tags: ["forecast", "trial", "gap"],
  },
];

const dedupe = (values) => {
  const unique = new Set();
  for (const value of values || []) {
    const text = String(value || "").trim();
    if (!text) continue;
    unique.add(text);
  }
  return [...unique];
};

const autoAliases = (title, chartKey) => {
  const normalizedTitle = normalizeText(title);
  const compactTitle = normalizedTitle.replace(/\s+/g, " ");
  const keyAlias = String(chartKey || "").replace(/_/g, " ");
  return dedupe([title, compactTitle, keyAlias]);
};

const describeExtra = (extra) => {
  if (!Array.isArray(extra) || extra.length === 0) return "";
  return extra
    .map((item) => {
      const label = String(item?.label || "").trim();
      const value = String(item?.value || "").trim();
      if (!label && !value) return "";
      if (!label) return value;
      if (!value) return label;
      return `${label}: ${value}`;
    })
    .filter(Boolean)
    .join(" ");
};

const buildDetailedDescription = (item, registryMeta) => {
  const parts = [
    item.description,
    registryMeta?.simple,
    registryMeta?.tech,
    describeExtra(registryMeta?.extra),
    item.sectionPath ? `Seção: ${item.sectionPath}.` : "",
  ].filter(Boolean);
  return dedupe(parts).join(" ");
};

/**
 * Build global search catalog from chart registry metadata + explicit placement map.
 */
export const buildSearchCatalog = () => {
  return ITEM_DEFS.map((item) => {
    const registryMeta = ChartRegistry.get(item.chartKey, null) || {};
    const title = item.title || registryMeta?.title || item.id;
    const description = buildDetailedDescription(item, registryMeta);
    const aliases = dedupe([...(autoAliases(title, item.chartKey) || []), ...(item.aliases || [])]);
    const tags = dedupe([item.tabId, item.viewMode, ...(item.tags || [])]);
    const searchBlob = buildSearchBlob({ title, aliases, tags, description });

    return {
      id: item.id,
      domId: item.domId || `search-${item.id}`,
      title,
      description,
      aliases,
      tags,
      sectionPath: item.sectionPath || "",
      sectionKey: item.sectionKey || "",
      tabId: item.tabId || "overview",
      viewMode: item.viewMode || "study",
      titleNorm: normalizeText(title),
      aliasesNorm: aliases.map(normalizeText),
      tagsNorm: tags.map(normalizeText),
      descriptionNorm: normalizeText(description),
      searchBlob,
      searchBlobNorm: normalizeText(searchBlob),
    };
  });
};
