import { useMemo } from "react";
import { AreaChart, Area, XAxis, YAxis, Legend } from "recharts";
import {
  Card,
  BarChart2,
  EmptyState,
  DefaultTooltip,
  ChartContainer,
} from "../../../ui/BaseComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { renderWithHints } from "../../../ui/UIComponents.jsx";
import { Theme } from "../../../ui/Theme.js";

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
            className="flex-1 min-h-[120px] w-full rounded border overflow-hidden relative"
            style={{
              backgroundColor: "var(--viz-bg-elevated)",
              borderColor: "var(--viz-border)",
            }}
          >
            <div className="absolute top-1 right-2 text-[9px] text-zinc-600 font-mono z-10">
              HISTORY
            </div>
            <ChartContainer minHeight={120} className="h-full">
              <AreaChart data={data} margin={{ top: 20, right: 0, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="gradRecon" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={Theme.semantic.chart.recon} stopOpacity={0.2} />
                    <stop offset="95%" stopColor={Theme.semantic.chart.recon} stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gradKL" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={Theme.semantic.chart.klDiv} stopOpacity={0.2} />
                    <stop offset="95%" stopColor={Theme.semantic.chart.klDiv} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="epoch" hide />
                <YAxis hide domain={["auto", "auto"]} />
                <DefaultTooltip />
                <Legend
                  formatter={renderWithHints}
                  verticalAlign="top"
                  align="right"
                  height={18}
                  wrapperStyle={{ top: -8, fontSize: "10px" }}
                />
                <Area
                  type="monotone"
                  dataKey="recon"
                  stackId="1"
                  stroke={Theme.semantic.chart.recon}
                  fill="url(#gradRecon)"
                  strokeWidth={2}
                  name="Recon"
                />
                <Area
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
