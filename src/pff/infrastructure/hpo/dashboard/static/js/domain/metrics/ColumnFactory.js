/**
 * Shared column definitions for SortableTable-based metric tables.
 *
 * Centralizes rendering logic (DataBar, HeatmapCell) so that any table
 * consuming metric data uses the exact same visual language.
 */
import { DataBar, HeatmapCell } from "../../ui/TableVisualization.jsx";
import { formatDuration, formatMetricValue, resolveMetricValue } from "./Formatters.js";
import { MetricRegistry } from "./MetricRegistry.js";

const hint = (key) => MetricRegistry.get(key);

// ---------------------------------------------------------------------------
// Heatmap pill column (0→1, red→yellow→green)
// ---------------------------------------------------------------------------
export const heatmapColumn = (key, { group = "clf", width = "90px" } = {}) => ({
  key,
  label: key.toUpperCase(),
  sortable: true,
  align: "right",
  direction: "up",
  width,
  group,
  helpText: hint(key),
  sortValue: (row) => resolveMetricValue(row, key),
  render: (_v, row) => <HeatmapCell value={resolveMetricValue(row, key)} min={0} max={1} />,
});

// ---------------------------------------------------------------------------
// Score column (DataBar 0→1, blue)
// ---------------------------------------------------------------------------
export const scoreColumn = () => ({
  key: "score",
  label: "SCORE",
  sortable: true,
  align: "right",
  direction: "up",
  width: "140px",
  group: "overview",
  helpText: hint("score"),
  sortValue: (row) => resolveMetricValue(row, "score"),
  render: (_v, row) => (
    <DataBar
      value={resolveMetricValue(row, "score")}
      min={0}
      max={1}
      color="var(--viz-palette-1-blue)"
      format={formatMetricValue}
    />
  ),
});

// ---------------------------------------------------------------------------
// Loss column (DataBar dynamic, red, inverted — lower = fuller)
// ---------------------------------------------------------------------------
export const lossColumn = ({ minLoss, maxLoss }) => ({
  key: "loss",
  label: "LOSS",
  sortable: true,
  align: "right",
  direction: "down",
  width: "140px",
  group: "overview",
  helpText: hint("loss"),
  sortValue: (row) => resolveMetricValue(row, "loss"),
  render: (_v, row) => (
    <DataBar
      value={resolveMetricValue(row, "loss")}
      min={maxLoss}
      max={minLoss}
      color="var(--viz-palette-5-red)"
      format={formatMetricValue}
    />
  ),
});

// ---------------------------------------------------------------------------
// Duration column (DataBar dynamic, color by epoch type or orange for trial)
// ---------------------------------------------------------------------------
export const durationColumn = ({ type, compact, isEvalEpoch, durRanges }) => ({
  key: "duration",
  label: "Duração",
  sortable: true,
  align: "right",
  direction: "down",
  group: "efficiency",
  helpText: hint("duration"),
  sortValue: (row) => resolveMetricValue(row, "duration"),
  render: (_v, row) => {
    const val = resolveMetricValue(row, "duration");
    if (type === "epoch") {
      const isEval = isEvalEpoch(row);
      return (
        <DataBar
          value={val}
          min={isEval ? durRanges.minDurEval : durRanges.minDurTrain}
          max={isEval ? durRanges.maxDurEval : durRanges.maxDurTrain}
          color={isEval ? "var(--viz-palette-4-yellow)" : "var(--viz-palette-3-orange)"}
          format={(v) => formatDuration(v, compact)}
        />
      );
    }
    return (
      <DataBar
        value={val}
        min={durRanges.minDurAll}
        max={durRanges.maxDurAll}
        color="var(--viz-palette-3-orange)"
        format={(v) => formatDuration(v, compact)}
      />
    );
  },
});

// ---------------------------------------------------------------------------
// Efficiency column (score / duration, yellow %)
// ---------------------------------------------------------------------------
export const efficiencyColumn = () => ({
  key: "efficiency",
  label: "Eficiência",
  sortable: true,
  align: "right",
  direction: "up",
  group: "efficiency",
  helpText: hint("efficiency"),
  sortValue: (row) => {
    const score = resolveMetricValue(row, "score");
    const dur = resolveMetricValue(row, "duration");
    return score != null && dur != null && dur > 0 ? score / dur : row.efficiency || 0;
  },
  render: (_v, row) => {
    const score = resolveMetricValue(row, "score");
    const dur = resolveMetricValue(row, "duration");
    const eff =
      score != null && dur != null && dur > 0
        ? score / dur
        : (resolveMetricValue(row, "efficiency") ?? _v);
    return eff ? (
      <span className="font-mono" style={{ color: "var(--viz-palette-4-yellow)" }}>
        {(eff * 100).toFixed(2)}%
      </span>
    ) : (
      "—"
    );
  },
});

// ---------------------------------------------------------------------------
// Classification metric columns (mcc, accuracy, precision, recall, f1, auc, pr_auc)
// ---------------------------------------------------------------------------
export const CLF_KEYS = ["mcc", "accuracy", "precision", "recall", "f1", "auc", "pr_auc"];
export const clfColumns = () => CLF_KEYS.map((key) => heatmapColumn(key, { group: "clf" }));

// ---------------------------------------------------------------------------
// Ranking metric columns (mrr, hits1, hits3, hits10)
// ---------------------------------------------------------------------------
export const RANKING_KEYS = ["mrr", "hits1", "hits3", "hits10"];
export const rankingColumns = () =>
  RANKING_KEYS.map((key) => heatmapColumn(key, { group: "ranking", width: "auto" }));

// ---------------------------------------------------------------------------
// Duration stats computation (median, mean, stderr)
// ---------------------------------------------------------------------------
export const computeDurationStats = (data) => {
  const durations = [];
  for (const row of data) {
    const d = resolveMetricValue(row, "duration");
    if (typeof d === "number" && Number.isFinite(d)) durations.push(d);
  }
  if (durations.length === 0) return null;

  durations.sort((a, b) => a - b);
  const n = durations.length;
  const sum = durations.reduce((a, b) => a + b, 0);
  const mean = sum / n;
  const mid = Math.floor(n / 2);
  const median = n % 2 === 0 ? (durations[mid - 1] + durations[mid]) / 2 : durations[mid];
  const variance = durations.reduce((acc, d) => acc + (d - mean) ** 2, 0) / n;
  const stderr = Math.sqrt(variance) / Math.sqrt(n);

  return { median, mean, stderr, count: n };
};
