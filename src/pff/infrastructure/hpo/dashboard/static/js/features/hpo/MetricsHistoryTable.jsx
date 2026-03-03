/**
 * Provide MetricsHistoryTable module functionality for the HPO dashboard.
 */

import { useMemo } from "react";
import { SortableTable } from "../../ui/SortableTable.jsx";
import { formatDuration, resolveMetricValue } from "../../domain/metrics/Formatters.js";
import {
  scoreColumn,
  lossColumn,
  durationColumn,
  efficiencyColumn,
  clfColumns,
  rankingColumns,
  computeDurationStats,
} from "../../domain/metrics/ColumnFactory.js";

/**
 * Expose metrics history table for dashboard usage.
 */
export const MetricsHistoryTable = ({
  data = [],
  compact = false,
  framed = true,
  type = "trial",
}) => {
  const isEvalEpoch = (t) =>
    resolveMetricValue(t, "mcc") != null ||
    resolveMetricValue(t, "accuracy") != null ||
    resolveMetricValue(t, "mrr") != null;

  const stats = useMemo(() => {
    let bId = -1,
      wId = -1;
    let minL = Infinity,
      maxL = -Infinity;
    let minDE = Infinity,
      maxDE = -Infinity;
    let minDT = Infinity,
      maxDT = -Infinity;
    let minDA = Infinity,
      maxDA = -Infinity;

    for (const t of data) {
      const l = resolveMetricValue(t, "loss");
      const d = resolveMetricValue(t, "duration");
      if (typeof l === "number") {
        minL = Math.min(minL, l);
        maxL = Math.max(maxL, l);
      }
      if (typeof d === "number") {
        minDA = Math.min(minDA, d);
        maxDA = Math.max(maxDA, d);
        if (isEvalEpoch(t)) {
          minDE = Math.min(minDE, d);
          maxDE = Math.max(maxDE, d);
        } else {
          minDT = Math.min(minDT, d);
          maxDT = Math.max(maxDT, d);
        }
      }
    }

    const isEligible = (t) => {
      const state = String(t?.state || "").toUpperCase();
      if (state && state !== "COMPLETE") return false;
      const s = resolveMetricValue(t, "score");
      return typeof s === "number" && Number.isFinite(s);
    };
    const eligible = data.filter(isEligible);
    const noWarm = eligible.filter((t) => !t.warmstart);
    const candidates = noWarm.length > 0 ? noWarm : eligible;
    if (candidates.length > 0) {
      const sorted = [...candidates].sort(
        (a, b) => resolveMetricValue(b, "score") - resolveMetricValue(a, "score")
      );
      bId = sorted[0]?.id;
      wId = sorted[sorted.length - 1]?.id;
    }

    return {
      bestId: bId,
      worstId: wId,
      minLoss: minL === Infinity ? 0 : minL,
      maxLoss: maxL === -Infinity ? 1 : maxL,
      minDurEval: minDE === Infinity ? 0 : minDE,
      maxDurEval: maxDE === -Infinity ? 1 : maxDE,
      minDurTrain: minDT === Infinity ? 0 : minDT,
      maxDurTrain: maxDT === -Infinity ? 1 : maxDT,
      minDurAll: minDA === Infinity ? 0 : minDA,
      maxDurAll: maxDA === -Infinity ? 1 : maxDA,
    };
  }, [data]);

  const durStats = useMemo(() => computeDurationStats(data), [data]);

  const columns = useMemo(() => {
    const { bestId, worstId, minLoss, maxLoss } = stats;
    const durRanges = stats;
    const cols = [];

    const idKey = type === "epoch" ? "epoch" : "id";
    cols.push({
      key: idKey,
      label: type === "epoch" ? "Época" : "Trial",
      sortable: true,
      align: "left",
      width: "120px",
      group: "overview",
      helpText: {
        tech: "Identificador sequencial do registro para rastrear a ordem de execução.",
        simple: "O número de chamada para não se perder.",
        extra: [{ label: "Uso", value: "ordem cronológica" }],
      },
      sortValue: (row) => (type === "epoch" ? row.epoch : row.id),
      render: (id, row) => {
        const displayId = type === "epoch" ? (row.epoch ?? id) : id;
        const isWarm = !!row.warmstart;
        const isBest = displayId === bestId;
        const isWorst = displayId === worstId && data.length > 2;
        const isPruned = row.state === "PRUNED";
        return (
          <div
            className="flex items-center gap-2 pl-2 border-l-2"
            style={{
              borderColor: isBest
                ? "var(--viz-palette-4-yellow)"
                : isPruned
                  ? "var(--viz-palette-5-red)"
                  : "transparent",
            }}
          >
            <span className="font-mono font-bold" style={{ color: "var(--viz-text-primary)" }}>
              {displayId}
            </span>
            {isPruned && (
              <span className="px-1 py-0.5 rounded-sm bg-red-500/10 text-red-500 text-[8px] font-bold border border-red-500/20">
                PRUNED
              </span>
            )}
            {isWarm && (
              <span className="px-1 py-0.5 rounded-sm bg-amber-500/10 text-amber-500 text-[8px] font-bold border border-amber-500/20">
                WARM
              </span>
            )}
            {isBest && (
              <span className="px-1 py-0.5 rounded-sm bg-lime-500/10 text-lime-400 text-[8px] font-bold border border-lime-500/20">
                ★ MELHOR
              </span>
            )}
            {isWorst && (
              <span className="px-1 py-0.5 rounded-sm bg-rose-500/10 text-rose-500 text-[8px] font-bold border border-rose-500/20">
                PIOR
              </span>
            )}
          </div>
        );
      },
    });

    const durCol = durationColumn({ type, compact, isEvalEpoch, durRanges });

    if (type === "epoch") {
      cols.push(lossColumn({ minLoss, maxLoss }));
      cols.push({ ...durCol, group: "overview" });
    } else {
      cols.push(scoreColumn());
      cols.push(lossColumn({ minLoss, maxLoss }));
    }

    cols.push(...clfColumns());
    cols.push(...rankingColumns());

    if (type === "trial") cols.push(durCol);
    cols.push(efficiencyColumn());

    return cols;
  }, [type, compact, stats, data.length]);

  const footerStats = useMemo(() => {
    if (!durStats) return null;
    const fmt = (v) => formatDuration(v, compact);
    return `Duração — Média: ${fmt(durStats.mean)} · Mediana: ${fmt(durStats.median)} · Erro: ±${fmt(durStats.stderr)}`;
  }, [durStats, compact]);

  return (
    <div className="w-full flex flex-col min-h-0 h-full">
      <SortableTable
        data={data}
        columns={columns}
        defaultSort={{
          key: type === "epoch" ? "epoch" : "score",
          direction: "desc",
        }}
        framed={framed}
        footerStats={footerStats}
        className="text-[10px] h-full"
      />
    </div>
  );
};
