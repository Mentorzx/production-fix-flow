import { useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, Cell, Label } from 'recharts';

import { Card, TargetIcon, colors, defaultChartMargins, DefaultCartesianGrid, DefaultTooltip, DefaultTooltipCursor, ChartFrame, ChartContainer, WithData } from "../../../ui/BaseComponents.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const ParetoFrontCard = ({ trials }) => {
    const data = useMemo(() => {
        if (!trials) return []; const completed = trials.filter(t => t.state === 'COMPLETE' && t.value > 0); const points = completed.map(t => ({ id: t.id, x: t.duration || 0, y: t.value || 0, isPareto: false })); points.forEach(p => { if (!points.some(other => (other.x <= p.x && other.y > p.y) || (other.x < p.x && other.y >= p.y))) p.isPareto = true; }); return points;
    }, [trials]);
    return (
        <Card title="Fronteira de Pareto" className="h-full" icon={TargetIcon} helpText={ChartRegistry.get('pareto_front')}>
            <ChartFrame>
                <WithData when={data.length > 0} empty="Sem dados">
                    <ChartContainer>
                        <ScatterChart margin={defaultChartMargins}>
                            <DefaultCartesianGrid />
                            <XAxis type="number" dataKey="x" stroke={colors.text} height={50}>
                                <Label content={<ChartAxisLabel value="Duração (s)" axis="x" />} />
                            </XAxis>
                            <YAxis type="number" dataKey="y" stroke={colors.text}>
                                <Label content={<ChartAxisLabel value="Score" axis="y" />} position="insideLeft" />
                            </YAxis>
                            <DefaultTooltip cursor={DefaultTooltipCursor} />
                            <Scatter name="Fronteira de Pareto" data={data}>
                                {data.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.isPareto ? colors.success : colors.primary} fillOpacity={entry.isPareto ? 1 : 0.4} />
                                ))}
                            </Scatter>
                        </ScatterChart>
                    </ChartContainer>
                </WithData>
            </ChartFrame>
        </Card>
    );
};
