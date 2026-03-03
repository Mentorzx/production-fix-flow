/**
 * Provide ELBOBreakdownCard module functionality for the HPO dashboard.
 */

import { useMemo } from "react";
import { AreaChart, Area, XAxis, YAxis } from "recharts";
import { BarChart2 } from "../../../ui/icons.jsx";
import { DefaultTooltip, ChartContainer } from "../../../ui/ChartPrimitives.jsx";
import { Card } from "../../../ui/Card.jsx";
import { EmptyState } from "../../../ui/EmptyStates.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { Theme } from "../../../ui/Theme.js";
import { useSmoothedDomain } from "../../../ui/useSmoothedDomain.js";

/**
 * Expose elbobreakdown card for dashboard usage.
 */
export const ELBOBreakdownCard = ({ liveStatus }) => {
  const data = useMemo(() => {
    if (!liveStatus?.epoch_history || liveStatus.epoch_history.length === 0) return [];
    return liveStatus.epoch_history.map((e, idx) => ({
      epoch: e.epoch ?? e.id ?? idx + 1,
      recon: e.elbo_recon || 0,
      kl: e.elbo_kl || 0,
      total: (e.elbo_recon || 0) + (e.elbo_kl || 0),
    }));
  }, [liveStatus?.epoch_history]);

  const latest = useMemo(() => {
    if (data.length === 0) return { recon: 0, kl: 0, total: 0 };
    return data[data.length - 1];
  }, [data]);

  const hasData = liveStatus?.elbo_recon != null;
  const yDomain = useSmoothedDomain(
    data.flatMap((row) => [row.recon, row.kl, row.total]),
    { clampMin: 0, minSpan: 0.05 }
  );

  return (
    <Card
      title="ELBO Breakdown"
      icon={BarChart2}
      className="h-full"
      helpText={ChartRegistry.get("elbo_breakdown")}
    >
      {hasData ? (
        <div className="flex flex-col h-full p-3 gap-2">
          {/* Metrics Row (Compact) */}
          <div className="flex items-center justify-between border-b border-zinc-800/50 pb-2 flex-none">
            {/* Recon */}
            <div className="flex flex-col">
              <div className="flex items-baseline gap-2">
                <span className="text-xl font-mono text-rose-400 leading-none">
                  {latest.recon.toFixed(4)}
                </span>
                <span className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">
                  Recon
                </span>
              </div>
              <span className="text-[10px] text-zinc-600">
                {((latest.recon / latest.total) * 100).toFixed(1)}% weight
              </span>
            </div>

            {/* Divider */}
            <div className="h-6 w-px bg-zinc-800/50 mx-4" />

            {/* KL */}
            <div className="flex flex-col items-end">
              <div className="flex items-baseline gap-2">
                <span className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">
                  KL Div
                </span>
                <span className="text-xl font-mono text-orange-400 leading-none">
                  {latest.kl.toFixed(4)}
                </span>
              </div>
              <span className="text-[10px] text-zinc-600">
                {((latest.kl / latest.total) * 100).toFixed(1)}% weight
              </span>
            </div>
          </div>

          {/* Chart Area (Expanded) */}
          <div
            className="flex-1 min-h-[150px] w-full rounded border overflow-hidden"
            style={{
              backgroundColor: "var(--viz-bg-elevated)",
              borderColor: "var(--viz-border)",
            }}
          >
            <ChartContainer minHeight={150} className="h-full">
              <AreaChart data={data} margin={{ top: 12, right: 8, left: 8, bottom: 6 }}>
                <defs>
                  <linearGradient id="gradRecon" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={Theme.semantic.chart.recon} stopOpacity={0.24} />
                    <stop offset="100%" stopColor={Theme.semantic.chart.recon} stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="gradKL" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={Theme.semantic.chart.klDiv} stopOpacity={0.24} />
                    <stop offset="100%" stopColor={Theme.semantic.chart.klDiv} stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="epoch" hide />
                <YAxis hide domain={yDomain} />
                <DefaultTooltip />
                <Area
                  isAnimationActive={false}
                  type="monotone"
                  dataKey="recon"
                  stackId="1"
                  stroke={Theme.semantic.chart.recon}
                  fill="url(#gradRecon)"
                  strokeWidth={2}
                  name="Recon"
                />
                <Area
                  isAnimationActive={false}
                  type="monotone"
                  dataKey="kl"
                  stackId="1"
                  stroke={Theme.semantic.chart.klDiv}
                  fill="url(#gradKL)"
                  strokeWidth={2}
                  name="KL Div"
                />
              </AreaChart>
            </ChartContainer>
          </div>
        </div>
      ) : (
        <EmptyState className="text-sm">Aguardando dados...</EmptyState>
      )}
    </Card>
  );
};
