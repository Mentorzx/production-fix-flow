/**
 * Provide PC2MetricsCard module functionality for the HPO dashboard.
 */

import { useId, useMemo } from "react";
import { XAxis, YAxis, LineChart, Line, Area } from "recharts";
import { GitMerge } from "../../../ui/icons.jsx";
import { DefaultTooltip, ChartContainer } from "../../../ui/ChartPrimitives.jsx";
import { Card } from "../../../ui/Card.jsx";
import { EmptyState } from "../../../ui/EmptyStates.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { useSmoothedDomain } from "../../../ui/useSmoothedDomain.js";

/**
 * Expose pc2 metrics card for dashboard usage.
 */
export const PC2MetricsCard = ({ liveStatus }) => {
  const gradientSuffix = useId().replace(/:/g, "");
  const data = useMemo(() => {
    if (!liveStatus?.epoch_history || liveStatus.epoch_history.length === 0) return [];
    return liveStatus.epoch_history.map((e, idx) => ({
      epoch: e.epoch ?? e.id ?? idx + 1,
      latency: e.pc2_latency || 0,
      active_rules: e.pc2_rules || 0,
    }));
  }, [liveStatus?.epoch_history]);

  const latest = useMemo(() => {
    if (data.length === 0) return { latency: 0, active_rules: 0 };
    return data[data.length - 1];
  }, [data]);

  const hasData = liveStatus?.pc2_rules != null;
  const yDomain = useSmoothedDomain(
    data.map((row) => row.latency),
    { clampMin: 0, minSpan: 0.05 }
  );

  return (
    <Card
      title="PC2 Performance"
      icon={GitMerge}
      className="h-full"
      helpText={ChartRegistry.get("pc2_metrics")}
    >
      {hasData ? (
        <div className="flex flex-col h-full p-3 gap-2">
          {/* Metrics Row (Compact) */}
          <div className="flex items-center justify-between border-b border-zinc-800/50 pb-2 flex-none">
            {/* Rules Badge aligned with label */}
            <div className="flex items-center gap-2">
              <span className="px-1.5 py-0.5 rounded bg-lime-500/10 border border-lime-500/20 text-lime-400 font-bold font-mono text-sm leading-none">
                {latest.active_rules}
              </span>
              <span className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">
                Rules
              </span>
            </div>

            {/* Value & Label inline */}
            <div className="flex items-baseline gap-2">
              <span className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">
                Latency
              </span>
              <div className="flex items-baseline">
                <span className="text-xl font-mono text-amber-400 leading-none">
                  {latest.latency.toFixed(2)}
                </span>
                <span className="text-[10px] text-zinc-500 ml-1">ms</span>
              </div>
            </div>
          </div>

          {/* Chart Column (Sparkline Expanded) */}
          <div
            className="flex-1 min-h-[150px] w-full rounded border overflow-hidden"
            style={{
              backgroundColor: "var(--viz-bg-elevated)",
              borderColor: "var(--viz-border)",
            }}
          >
            <ChartContainer minHeight={150} className="h-full">
              <LineChart data={data} margin={{ top: 12, right: 8, left: 8, bottom: 6 }}>
                <defs>
                  <linearGradient
                    id={`grad-pc2-latency-${gradientSuffix}`}
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop offset="0%" stopColor="#fbbf24" stopOpacity={0.24} />
                    <stop offset="100%" stopColor="#fbbf24" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="epoch" hide />
                <YAxis hide domain={yDomain} />
                <DefaultTooltip />
                <Area
                  isAnimationActive={false}
                  type="monotone"
                  dataKey="latency"
                  stroke="none"
                  fill={`url(#grad-pc2-latency-${gradientSuffix})`}
                  fillOpacity={1}
                  baseValue="dataMin"
                  legendType="none"
                />
                <Line
                  isAnimationActive={false}
                  type="monotone"
                  dataKey="latency"
                  stroke="#fbbf24"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                  name="Latency (ms)"
                />
              </LineChart>
            </ChartContainer>
          </div>
        </div>
      ) : (
        <EmptyState className="text-sm">Aguardando dados...</EmptyState>
      )}
    </Card>
  );
};
