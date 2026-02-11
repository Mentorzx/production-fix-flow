import { useMemo } from "react";
import { Card, GitMerge, EmptyState } from "../../../ui/BaseComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const InteractionPlotCard = ({ trials, importances }) => {
  const { params, interactions } = useMemo(() => {
    if (!trials || trials.length < 5 || !importances) return { params: [], interactions: [] };
    const completed = trials.filter((t) => t.state === "COMPLETE" && t.value != null);
    const topParams = Object.entries(importances)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map((e) => e[0]);
    const matrix = [];
    for (let i = 0; i < topParams.length; i++) {
      for (let j = 0; j < topParams.length; j++) {
        if (i === j) {
          matrix.push({ row: topParams[i], col: topParams[j], value: 1.0 });
          continue;
        }
        const pairs = completed
          .map((t) => ({
            a: t.params?.[topParams[i]],
            b: t.params?.[topParams[j]],
            score: t.value,
          }))
          .filter((p) => p.a != null && p.b != null);
        if (pairs.length < 3) {
          matrix.push({ row: topParams[i], col: topParams[j], value: 0 });
          continue;
        }
        const meanA = pairs.reduce((s, p) => s + p.a, 0) / pairs.length;
        const meanB = pairs.reduce((s, p) => s + p.b, 0) / pairs.length;
        let num = 0,
          denA = 0,
          denB = 0;
        pairs.forEach((p) => {
          const dA = p.a - meanA;
          const dB = p.b - meanB;
          num += dA * dB * p.score;
          denA += dA * dA;
          denB += dB * dB;
        });
        matrix.push({
          row: topParams[i],
          col: topParams[j],
          value: Math.abs(denA * denB) > 0 ? num / (Math.sqrt(denA) * Math.sqrt(denB)) : 0,
        });
      }
    }
    return { params: topParams, interactions: matrix };
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
                const cell = interactions.find((x) => x.row === row && x.col === col);
                const val = cell ? cell.value : 0;
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
