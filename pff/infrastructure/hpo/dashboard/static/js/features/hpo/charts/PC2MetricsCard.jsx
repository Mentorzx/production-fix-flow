import { useMemo } from 'react';
import { XAxis, YAxis, Tooltip, LineChart, Line, Legend } from 'recharts';
import { Card, GitMerge, EmptyState, BaseTooltip, ChartContainer } from "../../../ui/BaseComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { renderLegendWithHints } from "../../../ui/UIComponents.jsx";

export const PC2MetricsCard = ({ liveStatus }) => {
    const data = useMemo(() => {
        if (!liveStatus?.epoch_history || liveStatus.epoch_history.length === 0) return [];
        return liveStatus.epoch_history.map((e, idx) => ({
            epoch: e.epoch ?? e.id ?? idx + 1,
            latency: e.pc2_latency || 0,
            active_rules: e.pc2_rules || 0
        }));
    }, [liveStatus?.epoch_history]);

    const latest = useMemo(() => {
        if (data.length === 0) return { latency: 0, active_rules: 0 };
        return data[data.length - 1];
    }, [data]);

    const hasData = liveStatus?.pc2_rules != null;

    return (
        <Card title="PC2 Performance" icon={GitMerge} className="h-full" helpText={ChartRegistry.get('pc2_metrics')}>
            {hasData ? (
                <div className="flex flex-col h-full p-3 gap-2">
                    {/* Metrics Row (Compact) */}
                    <div className="flex items-center justify-between border-b border-zinc-800/50 pb-2 flex-none">
                        {/* Rules Badge aligned with label */}
                        <div className="flex items-center gap-2">
                            <span className="px-1.5 py-0.5 rounded bg-lime-500/10 border border-lime-500/20 text-lime-400 font-bold font-mono text-sm leading-none">
                                {latest.active_rules}
                            </span>
                            <span className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Rules</span>
                        </div>

                        {/* Value & Label inline */}
                        <div className="flex items-baseline gap-2">
                            <span className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Latency</span>
                            <div className="flex items-baseline">
                                <span className="text-xl font-mono text-amber-400 leading-none">{latest.latency.toFixed(2)}</span>
                                <span className="text-[10px] text-zinc-500 ml-1">ms</span>
                            </div>
                        </div>
                    </div>

                    {/* Chart Column (Sparkline Expanded) */}
                    <div
                        className="flex-1 min-h-[120px] w-full rounded border overflow-hidden relative"
                        style={{
                            backgroundColor: 'var(--viz-bg-elevated)',
                            borderColor: 'var(--viz-border)',
                        }}
                    >
                        <div className="absolute top-1 right-2 text-[9px] text-zinc-600 font-mono z-10">HISTORY (AVG EPOCH)</div>
                        <ChartContainer minHeight={120} className="h-full">
                            <LineChart data={data} margin={{ top: 20, right: 5, left: 5, bottom: 5 }}>
                                <XAxis dataKey="epoch" hide />
                                <YAxis hide domain={['auto', 'auto']} />
                                <Tooltip content={<BaseTooltip />} />
                                <Legend formatter={renderLegendWithHints} verticalAlign="top" align="left" height={18} wrapperStyle={{ top: -8, fontSize: '10px' }} />
                                <Line type="monotone" dataKey="latency" stroke="#fbbf24" strokeWidth={2} dot={false} activeDot={{ r: 4 }} name="Latency (ms)" />
                            </LineChart>
                        </ChartContainer>
                    </div>
                </div>
            ) : <EmptyState className="text-sm">Aguardando dados...</EmptyState>}
        </Card>
    );
};
