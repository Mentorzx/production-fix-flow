/**
 * Provide LocalOptimaDiagnosticsCard module functionality for the HPO dashboard.
 */

import { useMemo } from "react";

import { Microscope } from "../../../ui/icons.jsx";
import { Card } from "../../../ui/Card.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { formatMetricValue } from "../../../domain/metrics/Formatters.js";
import { Theme } from "../../../ui/Theme.js";

const STATUS_META = {
  exploring: {
    label: "Explorando",
    tone: Theme.semantic.success,
    bg: Theme.palette.vividGreen + "18",
    border: Theme.palette.vividGreen + "40",
  },
  stagnant: {
    label: "Estagnado",
    tone: Theme.semantic.error,
    bg: Theme.palette.red + "18",
    border: Theme.palette.red + "40",
  },
  multiple_regions: {
    label: "Multiplas Regioes Boas",
    tone: Theme.palette.cyberYellow,
    bg: Theme.palette.cyberYellow + "18",
    border: Theme.palette.cyberYellow + "40",
  },
  insufficient_evidence: {
    label: "Dados insuficientes",
    tone: Theme.ui.text.muted,
    bg: Theme.ui.surfaceHighlight,
    border: Theme.ui.border,
  },
};

const formatPct = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  return `${(numeric * 100).toFixed(2)}%`;
};

const fallbackDiagnostics = {
  status: "insufficient_evidence",
  stagnant: false,
  trials_since_improvement: 0,
  recent_range: null,
  best_trial_id: null,
  best_score: null,
  current_sampler: "Unknown",
  recommended_action: "Aguarde mais trials completos antes de concluir sobre minimos locais.",
  multi_region_evidence: {
    detected: false,
    region_count: 0,
    summary_labels: [],
    eligible_trials: 0,
    elite_trials: 0,
  },
  completed_trials: 0,
};

/**
 * Expose local optima diagnostics card for dashboard usage.
 */
export const LocalOptimaDiagnosticsCard = ({ diagnostics }) => {
  const model = useMemo(() => {
    const source = diagnostics && typeof diagnostics === "object" ? diagnostics : {};
    const merged = {
      ...fallbackDiagnostics,
      ...source,
      multi_region_evidence: {
        ...fallbackDiagnostics.multi_region_evidence,
        ...(source?.multi_region_evidence || {}),
      },
    };
    return merged;
  }, [diagnostics]);
  const statusMeta = STATUS_META[model.status] || STATUS_META.insufficient_evidence;
  const helpText = ChartRegistry.get("local_optima");
  const summaryLabels = Array.isArray(model.multi_region_evidence?.summary_labels)
    ? model.multi_region_evidence.summary_labels.slice(0, 2)
    : [];
  const multiRegionText = model.multi_region_evidence?.detected
    ? `${model.multi_region_evidence.region_count} regioes competitivas`
    : model.multi_region_evidence?.status === "insufficient_evidence"
      ? "Aguardando pelo menos 12 trials completos"
      : "Sem evidencia conservadora de regioes extras";

  return (
    <Card
      title="Estagnacao & Otimos Locais"
      icon={Microscope}
      className="h-full"
      helpText={helpText}
    >
      <div className="flex flex-col gap-4 h-full min-h-0">
        <div
          className="rounded-xl border px-4 py-3 flex items-center justify-between gap-3"
          style={{ backgroundColor: statusMeta.bg, borderColor: statusMeta.border }}
        >
          <div>
            <div
              className="text-[10px] uppercase tracking-[0.2em] font-black"
              style={{ color: Theme.ui.text.muted }}
            >
              Estado
            </div>
            <div className="text-lg font-black" style={{ color: statusMeta.tone }}>
              {statusMeta.label}
            </div>
          </div>
          <div className="text-right">
            <div
              className="text-[10px] uppercase tracking-[0.2em] font-black"
              style={{ color: Theme.ui.text.muted }}
            >
              Trials completos
            </div>
            <div className="text-lg font-black font-mono" style={{ color: Theme.ui.text.primary }}>
              {Number(model.completed_trials || 0)}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div
            className="rounded-xl border p-3"
            style={{ backgroundColor: Theme.ui.surface, borderColor: Theme.ui.border }}
          >
            <div className="text-[10px] uppercase tracking-widest" style={{ color: Theme.ui.text.muted }}>
              Sem melhora
            </div>
            <div className="text-xl font-black font-mono" style={{ color: Theme.ui.text.primary }}>
              {Number(model.trials_since_improvement || 0)}
            </div>
          </div>
          <div
            className="rounded-xl border p-3"
            style={{ backgroundColor: Theme.ui.surface, borderColor: Theme.ui.border }}
          >
            <div className="text-[10px] uppercase tracking-widest" style={{ color: Theme.ui.text.muted }}>
              Faixa recente
            </div>
            <div className="text-xl font-black font-mono" style={{ color: Theme.ui.text.primary }}>
              {formatPct(model.recent_range)}
            </div>
          </div>
          <div
            className="rounded-xl border p-3"
            style={{ backgroundColor: Theme.ui.surface, borderColor: Theme.ui.border }}
          >
            <div className="text-[10px] uppercase tracking-widest" style={{ color: Theme.ui.text.muted }}>
              Melhor trial
            </div>
            <div className="text-xl font-black font-mono" style={{ color: Theme.ui.text.primary }}>
              {model.best_trial_id ? `#${model.best_trial_id}` : "—"}
            </div>
            <div className="text-[10px] mt-1" style={{ color: Theme.ui.text.secondary }}>
              Score {formatMetricValue(model.best_score)}
            </div>
          </div>
          <div
            className="rounded-xl border p-3"
            style={{ backgroundColor: Theme.ui.surface, borderColor: Theme.ui.border }}
          >
            <div className="text-[10px] uppercase tracking-widest" style={{ color: Theme.ui.text.muted }}>
              Sampler
            </div>
            <div
              className="text-sm font-black"
              style={{ color: Theme.ui.text.primary, wordBreak: "break-word" }}
            >
              {model.current_sampler || "Unknown"}
            </div>
          </div>
        </div>

        <div
          className="rounded-xl border p-4"
          style={{ backgroundColor: Theme.ui.surface, borderColor: Theme.ui.border }}
        >
          <div
            className="text-[10px] uppercase tracking-[0.2em] font-black mb-2"
            style={{ color: Theme.ui.text.muted }}
          >
            Evidencia multi-regiao
          </div>
          <div className="text-sm font-semibold" style={{ color: Theme.ui.text.primary }}>
            {multiRegionText}
          </div>
          {summaryLabels.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {summaryLabels.map((label) => (
                <span
                  key={label}
                  className="px-2 py-1 rounded-full text-[10px] font-mono"
                  style={{
                    backgroundColor: Theme.ui.surfaceHighlight,
                    color: Theme.ui.text.secondary,
                    border: `1px solid ${Theme.ui.border}`,
                  }}
                >
                  {label}
                </span>
              ))}
            </div>
          )}
        </div>

        <div
          className="rounded-xl border p-4"
          style={{ backgroundColor: Theme.ui.surface, borderColor: Theme.ui.border }}
        >
          <div
            className="text-[10px] uppercase tracking-[0.2em] font-black mb-2"
            style={{ color: Theme.ui.text.muted }}
          >
            Acao recomendada
          </div>
          <div className="text-sm leading-relaxed" style={{ color: Theme.ui.text.primary }}>
            {model.recommended_action}
          </div>
        </div>

        <div className="text-[10px] leading-relaxed mt-auto" style={{ color: Theme.ui.text.muted }}>
          Heuristica operacional de HPO: este card sugere padroes de exploracao; nao prova
          matematicamente minimos locais.
        </div>
      </div>
    </Card>
  );
};
