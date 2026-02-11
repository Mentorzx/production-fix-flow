import { useMemo } from "react";

import { Card, TableIcon, EmptyState } from "../../../ui/BaseComponents.jsx";
import { renderWithHints, renderParamWithHints } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { MetricRegistry } from "../../../domain/metrics/MetricRegistry.js";
import { formatMetricValue, formatDuration, formatParamValue, resolveMetricValue } from "../../../domain/metrics/Formatters.js";

const getDirection = (key, defaultDirection) => {
    if (key === "score") return defaultDirection;
    return MetricRegistry.get(key)?.direction || "up";
};

const pickTrials = (trials, direction) => {
    const completed = trials.filter((t) => t?.state === "COMPLETE" && t?.value != null);
    if (completed.length === 0) return [];

    const sorted = [...completed].sort((a, b) => {
        const av = a.value ?? 0;
        const bv = b.value ?? 0;
        return direction === "minimize" ? av - bv : bv - av;
    });

    const best = sorted[0];
    const worst = sorted[sorted.length - 1];
    const recent = [...completed].sort((a, b) => a.id - b.id).slice(-1)[0];

    const unique = [];
    [best, recent, worst].forEach((t) => {
        if (t && !unique.find((u) => u.id === t.id)) unique.push(t);
    });

    return unique.slice(0, 3);
};

export const TrialDiffTableCard = ({ trials, direction = "maximize" }) => {
    const { selected, rows } = useMemo(() => {
        const pool = Array.isArray(trials) ? trials : [];
        const chosen = pickTrials(pool, direction);

        if (chosen.length === 0) {
            return { selected: [], rows: [] };
        }

        const metricRows = [
            { key: "score", label: "Score", type: "metric" },
            { key: "mcc", label: "MCC", type: "metric" },
            { key: "mrr", label: "MRR", type: "metric" },
            { key: "accuracy", label: "Accuracy", type: "metric" },
            { key: "f1", label: "F1", type: "metric" },
            { key: "auc", label: "AUC", type: "metric" },
            { key: "loss", label: "Loss", type: "metric" },
            { key: "duration", label: "Duração", type: "metric" }
        ];

        const paramCounts = {};
        chosen.forEach((trial) => {
            Object.keys(trial.params || {}).forEach((key) => {
                paramCounts[key] = (paramCounts[key] || 0) + 1;
            });
        });

        const paramKeys = Object.entries(paramCounts)
            .sort((a, b) => b[1] - a[1])
            .map(([key]) => key)
            .slice(0, 6);

        const paramRows = paramKeys.map((key) => ({ key, label: key, type: "param" }));

        return { selected: chosen, rows: [...metricRows, ...paramRows] };
    }, [trials, direction]);

    const helpText = ChartRegistry.get("trial_diff");

    if (selected.length === 0) {
        return (
            <Card title="Comparativo de Trials" icon={TableIcon} className="h-full" helpText={helpText}>
                <EmptyState className="text-zinc-500">Sem trials completos para comparar</EmptyState>
            </Card>
        );
    }

    return (
        <Card title="Comparativo de Trials" icon={TableIcon} className="h-full" helpText={helpText}>
            <div className="absolute inset-0 p-0 overflow-auto custom-scrollbar">
                <table className="w-full text-left text-[10px]">
                    <thead className="bg-zinc-900 sticky top-0">
                        <tr>
                            <th className="p-2 border-b border-zinc-800 text-zinc-500 uppercase tracking-widest">Métrica</th>
                            {selected.map((trial, idx) => (
                                <th key={trial.id} className="p-2 border-b border-zinc-800">
                                    <div className="flex flex-col">
                                        <span className="text-zinc-300 font-mono">Trial #{trial.id}</span>
                                        <span className={`text-[9px] uppercase tracking-widest ${idx === 0 ? "text-lime-400" : idx === 1 ? "text-amber-400" : "text-rose-400"}`}>
                                            {idx === 0 ? "melhor" : idx === 1 ? "recente" : "pior"}
                                        </span>
                                        <span className="text-[9px] text-zinc-500">Score: {formatMetricValue(resolveMetricValue(trial, "score"))}</span>
                                    </div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="font-mono">
                        {rows.map((row) => {
                            const values = selected.map((trial) => {
                                if (row.type === "metric") return resolveMetricValue(trial, row.key);
                                return trial.params?.[row.key];
                            });

                            const numeric = values.map((v) => (typeof v === "number" ? v : null));
                            const directionKey = row.type === "metric" ? getDirection(row.key, direction) : null;
                            const bestValue = directionKey
                                ? numeric.reduce((acc, v) => {
                                    if (v == null) return acc;
                                    if (acc == null) return v;
                                    return directionKey === "down" ? Math.min(acc, v) : Math.max(acc, v);
                                }, null)
                                : null;

                            return (
                                <tr key={row.key} className="border-b border-zinc-800/50 hover:bg-zinc-800/30">
                                    <td className="p-2 text-orange-400">
                                        {row.type === "metric" ? renderWithHints(row.label) : renderParamWithHints(row.label)}
                                    </td>
                                    {values.map((value, idx) => {
                                        const isBest = bestValue != null && value === bestValue;
                                        const formatted = row.type === "metric"
                                            ? (row.key === "duration" ? formatDuration(value) : formatMetricValue(value))
                                            : formatParamValue(value);
                                        return (
                                            <td key={`${row.key}-${idx}`} className={`p-2 ${isBest ? "text-lime-300" : "text-zinc-300"}`}>
                                                {formatted}
                                            </td>
                                        );
                                    })}
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </Card>
    );
};
