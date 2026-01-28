import { useMemo } from 'react';
import { ComposedChart, Line, Area, XAxis, YAxis, Tooltip, Legend, Label } from 'recharts';
import { Theme } from "../../../ui/Theme.js";

import { Card, TrendingUp, DefaultCartesianGrid, ChartFrame, ChartContainer } from "../../../ui/BaseComponents.jsx";
import { renderLegendWithHints, ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { useStoreState } from "../../../store/store.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const IncumbentTrajectoryCard = ({ trials }) => {
    const { data } = useStoreState();
    const direction = data?.direction || 'maximize';

    const chartData = useMemo(() => {
        if (!trials || trials.length === 0) return [];
        let currentIncumbent = direction === 'minimize' ? Infinity : -Infinity;
        const eligible = [...trials]
            .filter(t => (t.state === 'COMPLETE' || t.state === 'RUNNING'))
            .filter(t => t.value != null)
            .sort((a, b) => a.id - b.id);

        return eligible.map((t, i, arr) => {
            const val = t.value || 0;
            if (direction === 'minimize') { if (val < currentIncumbent) currentIncumbent = val; }
            else { if (val > currentIncumbent) currentIncumbent = val; }
            return {
                id: t.id,
                index: i,
                value: val,
                movingAverage: arr.slice(Math.max(0, i - 4), i + 1).reduce((s, c) => s + (c.value || 0), 0) / Math.min(i + 1, 5),
                incumbent: currentIncumbent
            };
        });
    }, [trials, direction]);

    const chartContract = ChartRegistry.get('convergence') || { title: "Evolução" };

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div className="border p-3 rounded-xl shadow-2xl z-50 text-left font-mono" style={{ backgroundColor: Theme.ui.background, borderColor: Theme.ui.border, color: Theme.ui.text.secondary }}>
                    <div className="text-xs font-black mb-2 pb-1 border-b" style={{ borderColor: Theme.ui.border, color: Theme.ui.text.muted }}>TRIAL #{label}</div>
                    {payload.map((p, i) => (
                        <div key={i} className="flex items-center justify-between gap-4 text-xs mb-1">
                            <span style={{ color: p.color }}>{p.name}:</span>
                            <span style={{ color: Theme.ui.text.primary }}>{p.value?.toFixed(6)}</span>
                        </div>
                    ))}
                </div>
            );
        }
        return null;
    };

    return (
        <Card title={chartContract.title} icon={TrendingUp} className="h-full" helpText={chartContract}>
            <ChartFrame>
                <ChartContainer>
                    <ComposedChart data={chartData} margin={{ top: 20, right: 20, bottom: 50, left: 60 }}>
                        <DefaultCartesianGrid vertical={false} />
                        <XAxis
                            type="number"
                            dataKey="id"
                            domain={['dataMin', 'dataMax']}
                            tickCount={Math.min(chartData.length, 10)}
                            stroke={Theme.ui.text.secondary}
                            tick={{ fontSize: 10, fill: Theme.ui.text.secondary }}
                            tickFormatter={(v) => `#${v}`}
                            height={50}
                        >
                            <Label content={<ChartAxisLabel value="Trial" axis="x" />} />
                        </XAxis>
                        <YAxis stroke={Theme.ui.text.secondary} tick={{ fontSize: 10, fill: Theme.ui.text.secondary }} domain={['auto', 'auto']} width={60}>
                            <Label content={<ChartAxisLabel value="Objective" axis="y" />} position="insideLeft" />
                        </YAxis>
                        <Tooltip content={<CustomTooltip />} cursor={{ stroke: Theme.ui.grid }} />
                        <Legend verticalAlign="top" align="right" height={40} wrapperStyle={{ fontSize: '10px', top: -10 }} formatter={renderLegendWithHints} />
                        <Area
                            type="monotone"
                            dataKey="value"
                            name="Objetivo"
                            stroke={Theme.semantic.primary}
                            strokeWidth={2}
                            fill={Theme.semantic.primary}
                            fillOpacity={0.15}
                            connectNulls={false}
                            activeDot={{ r: 4, strokeWidth: 0 }}
                        />
                        <Line name="Média Móvel" type="monotone" dataKey="movingAverage" stroke={Theme.semantic.chart.movingAverage} strokeWidth={2} dot={false} strokeDasharray="4 4" />
                        <Line name="Melhor (Incumbent)" type="stepAfter" dataKey="incumbent" stroke={Theme.semantic.chart.incumbent} strokeWidth={3} dot={false} />
                    </ComposedChart>
                </ChartContainer>
            </ChartFrame>
        </Card>
    );
};
