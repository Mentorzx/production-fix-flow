/**
 * Provide InteractionPlotCard module functionality for the HPO dashboard.
 */

import { useMemo } from "react";
import { GitMerge } from "../../../ui/icons.jsx";
import { Card } from "../../../ui/Card.jsx";
import { EmptyState } from "../../../ui/EmptyStates.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { buildWeightedInteractionMatrix } from "../../../utils/statistics.js";

/**
 * Expose interaction plot card for dashboard usage.
 */
export const InteractionPlotCard = ({ trials, importances }) => {
  const { params, interactionMatrix } = useMemo(() => {
    if (!trials || trials.length < 5 || !importances) {
      return { params: [], interactionMatrix: {} };
    }
    const completed = trials.filter((t) => t.state === "COMPLETE" && t.value != null);
    const topParams = Object.entries(importances)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map((e) => e[0]);
    return {
      params: topParams,
      interactionMatrix: buildWeightedInteractionMatrix(completed, topParams),
    };
  }, [trials, importances]);

  const fallbackContract = ChartRegistry.get("interaction") || {
    title: "Interação",
    tech: "Sinergia entre pares.",
  };

  if (params.length < 2) {
    return (
      <Card
        title={fallbackContract.title}
        icon={GitMerge}
        className="h-full"
        helpText={fallbackContract}
      >
        <EmptyState className="text-sm">Dados insuficientes</EmptyState>
      </Card>
    );
  }

  const chartContract = fallbackContract;

  return (
    <Card title={chartContract.title} helpText={chartContract} className="h-full" icon={GitMerge}>
      <div className="h-full p-4 flex flex-col items-center justify-center overflow-auto custom-scrollbar">
        <div className="w-full max-w-[400px]">
          <div
            className="grid gap-1 mb-2"
            style={{ gridTemplateColumns: `repeat(${params.length + 1}, minmax(0, 1fr))` }}
          >
            <div></div>
            {params.map((p) => (
              <div
                key={p}
                className="text-[8px] font-mono text-zinc-500 uppercase text-center truncate w-full max-w-[40px] mx-auto"
                title={p}
              >
                {p.slice(0, 6)}
              </div>
            ))}
          </div>
          {params.map((row) => (
            <div
              key={row}
              className="grid gap-1 mb-1"
              style={{ gridTemplateColumns: `repeat(${params.length + 1}, minmax(0, 1fr))` }}
            >
              <div
                className="text-[8px] font-mono text-zinc-500 uppercase text-right pr-2 self-center truncate"
                title={row}
              >
                {row.slice(0, 6)}
              </div>
              {params.map((col) => {
                const val = interactionMatrix[row]?.[col] ?? 0;
                const isDiag = row === col;
                return (
                  <div
                    key={`${row}-${col}`}
                    className={`rounded-sm ${isDiag ? "bg-zinc-800" : "bg-orange-500"} flex items-center justify-center aspect-square w-full max-w-[40px] mx-auto border border-white/5 transition-all hover:scale-110`}
                    style={{ opacity: isDiag ? 1 : 0.2 + Math.abs(val) * 0.8 }}
                  >
                    <span
                      className={`text-[9px] font-bold ${isDiag ? "text-zinc-500" : "text-white"}`}
                    >
                      {isDiag ? "-" : val.toFixed(1)}
                    </span>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
};
