import { useMemo } from "react";

import { Card, TrendingUp, EmptyState } from "../../../ui/BaseComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { formatMetricValue } from "../../../domain/metrics/Formatters.js";
import { linearRegression } from "../../../utils/statistics.js";

const computeSummary = (trials) => {
  if (!trials || trials.length < 3) return null;

  const completed = trials
    .filter((t) => t.state === "COMPLETE" && t.value != null)
    .map((t) => ({ x: t.id, y: t.value }));
  if (completed.length < 3) return null;

  const { slope, intercept, r2, rmse } = linearRegression(completed);
  const correlation = Math.sqrt(Math.abs(r2)) * (slope >= 0 ? 1 : -1);

  const maxId = Math.max(...completed.map((c) => c.x));

  const projections = [1, 10, 25].map((offset) => {
    const trialId = maxId + offset;
    return {
      label: `+${offset}`,
      trial: trialId,
      score: slope * trialId + intercept,
    };
  });

  return {
    r2,
    slope,
    rmse,
    correlation,
    projections,
    n: completed.length,
  };
};

export const RegressionInsightsCard = ({ trials }) => {
  const summary = useMemo(() => computeSummary(trials), [trials]);
  const helpText = ChartRegistry.get("regression_insights");

  if (!summary) {
    return (
      <Card title="Resumo da Regressão" icon={TrendingUp} className="h-full" helpText={helpText}>
        <EmptyState className="text-zinc-500">Dados insuficientes para projeção</EmptyState>
      </Card>
    );
  }

  return (
    <Card title="Resumo da Regressão" icon={TrendingUp} className="h-full" helpText={helpText}>
      <div className="flex flex-col h-full gap-2 min-h-0">
        {/* Metrics Grid */}
        <div className="grid grid-cols-2 gap-2 flex-shrink-0">
          <div className="p-2 rounded-lg border border-zinc-800 bg-zinc-950/40 flex flex-col justify-center">
            <div className="text-[9px] uppercase tracking-widest text-zinc-500 mb-0.5">
              R² Check
            </div>
            <div className="text-sm font-bold text-orange-400 font-mono tracking-tight">
              {summary.r2.toFixed(3)}
            </div>
          </div>
          <div className="p-2 rounded-lg border border-zinc-800 bg-zinc-950/40 flex flex-col justify-center">
            <div className="text-[9px] uppercase tracking-widest text-zinc-500 mb-0.5">Slope</div>
            <div
              className={`text-sm font-bold font-mono tracking-tight ${summary.slope >= 0 ? "text-lime-400" : "text-rose-400"}`}
            >
              {summary.slope >= 0 ? "+" : ""}
              {summary.slope.toFixed(4)}
            </div>
          </div>
          <div className="p-2 rounded-lg border border-zinc-800 bg-zinc-950/40 flex flex-col justify-center">
            <div className="text-[9px] uppercase tracking-widest text-zinc-500 mb-0.5">RMSE</div>
            <div className="text-sm font-bold text-zinc-300 font-mono tracking-tight">
              {summary.rmse.toFixed(4)}
            </div>
          </div>
          <div className="p-2 rounded-lg border border-zinc-800 bg-zinc-950/40 flex flex-col justify-center">
            <div className="text-[9px] uppercase tracking-widest text-zinc-500 mb-0.5">Corr</div>
            <div className="text-sm font-bold text-zinc-300 font-mono tracking-tight">
              {summary.correlation.toFixed(3)}
            </div>
          </div>
        </div>

        {/* Projections Table */}
        <div className="p-2 rounded-lg border border-zinc-800 bg-zinc-950/40 flex-1 flex flex-col min-h-0">
          <div className="text-[9px] uppercase tracking-widest text-zinc-500 mb-2">
            Projeções Futuras
          </div>
          <div className="flex-1 overflow-auto min-h-0">
            <table className="w-full text-[10px] text-zinc-300">
              <thead>
                <tr className="text-zinc-600 border-b border-zinc-800/50">
                  <th className="text-left font-normal pb-1">Milestone</th>
                  <th className="text-right font-normal pb-1">Trial</th>
                  <th className="text-right font-normal pb-1">Score Est.</th>
                </tr>
              </thead>
              <tbody>
                {summary.projections.map((p) => (
                  <tr
                    key={p.label}
                    className="border-b border-zinc-800/30 last:border-0 hover:bg-white/5"
                  >
                    <td className="py-1.5 font-mono text-zinc-500">{p.label}</td>
                    <td className="py-1.5 text-right font-mono">#{p.trial}</td>
                    <td className="py-1.5 text-right font-mono font-bold text-orange-300">
                      {formatMetricValue(p.score)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-2 text-[8px] text-zinc-600 text-right">
            Base: {summary.n} datapoints
          </div>
        </div>
      </div>
    </Card>
  );
};
