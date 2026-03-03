/**
 * Provide PCComparisonTableCard module functionality for the HPO dashboard.
 */

import { useMemo } from "react";
import { TableIcon } from "../../../ui/icons.jsx";
import { Card } from "../../../ui/Card.jsx";
import { renderWithHints } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { pearsonCorrelation } from "../../../utils/statistics.js";

/**
 * Expose pccomparison table card for dashboard usage.
 */
export const PCComparisonTableCard = ({ trials }) => {
  const pcParams = useMemo(
    () => ["max_circuit_depth", "lambda_pc", "rebuild_every", "pruning_threshold", "t_norm"],
    []
  );

  const analysis = useMemo(() => {
    if (!trials || trials.length === 0) return [];

    const completed = trials.filter((t) => t.state === "COMPLETE" && t.value != null);
    if (completed.length === 0) return [];

    const bestTrial = completed.reduce(
      (prev, curr) => (curr.value > prev.value ? curr : prev),
      completed[0]
    );

    return pcParams
      .map((param) => {
        const values = [];
        const scores = [];
        let type = "number";

        completed.forEach((t) => {
          let val = t.params?.[param];
          if (val !== undefined && val !== null) {
            if (typeof val === "string") {
              // Handle categorical for correlation? Skip for now or map
              type = "string";
            } else {
              values.push(val);
              scores.push(t.value);
            }
          }
        });

        const correlation = type === "number" ? pearsonCorrelation(values, scores) : null;
        const bestValue = bestTrial.params?.[param] ?? "—";

        return {
          param,
          bestValue,
          correlation,
          type,
        };
      })
      .filter((r) => r.bestValue !== "—"); // Only show params that exist
  }, [trials, pcParams]);

  return (
    <Card
      title="Análise PC (Probabilistic Circuits)"
      className="h-full"
      icon={TableIcon}
      helpText={ChartRegistry.get("pc_comparison")}
    >
      <div className="absolute inset-0 overflow-auto custom-scrollbar p-0">
        <table className="w-full text-left text-[10px] border-collapse">
          <thead className="bg-zinc-900 sticky top-0">
            <tr>
              <th className="p-2 border-b border-zinc-800">Parâmetro</th>
              <th className="p-2 border-b border-zinc-800 text-right">Melhor Valor</th>
              <th className="p-2 border-b border-zinc-800 text-right">Correlação</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {analysis.length > 0 ? (
              analysis.map((row) => (
                <tr key={row.param} className="dashboard-table-row border-b border-zinc-800/50">
                  <td className="p-2 text-zinc-300 truncate font-semibold">
                    {renderWithHints(row.param.replace(/_/g, " "))}
                  </td>
                  <td className="p-2 text-right">
                    <span style={{ color: "var(--viz-palette-4-yellow)" }}>
                      {typeof row.bestValue === "number" ? row.bestValue.toFixed(4) : row.bestValue}
                    </span>
                  </td>
                  <td className="p-2 text-right">
                    {row.type === "number" ? (
                      <span
                        style={{
                          color:
                            row.correlation > 0.3
                              ? "var(--viz-palette-2-green)"
                              : row.correlation < -0.3
                                ? "var(--viz-palette-5-red)"
                                : "var(--viz-text-muted)",
                        }}
                      >
                        {row.correlation.toFixed(2)}
                      </span>
                    ) : (
                      <span className="text-zinc-600">—</span>
                    )}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={3} className="p-4 text-center italic text-zinc-500">
                  Nenhum parâmetro PC detectado.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
};
