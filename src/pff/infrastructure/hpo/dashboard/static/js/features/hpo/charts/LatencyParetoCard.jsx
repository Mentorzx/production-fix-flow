import { useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, Label } from 'recharts';

import { Card, TargetIcon, colors, defaultChartMargins, DefaultCartesianGrid, DefaultTooltip, DefaultTooltipCursor, ChartFrame, ChartContainer, WithData } from "../../../ui/BaseComponents.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const LatencyParetoCard = ({ trials }) => {
    const data = useMemo(() => { if (!trials) return []; return trials.filter(t => t.state === 'COMPLETE').map(t => ({ id: t.id, x: (t.metrics?.inference_latency || t.duration || 0), y: t.value || 0 })); }, [trials]);
    return (
        <Card title="Latência x Qualidade" className="h-full" icon={TargetIcon} helpText={ChartRegistry.get('latency_pareto')}>
            <ChartFrame>
                <WithData when={data.length > 0} empty="Sem dados">
                    <ChartContainer>
                        <ScatterChart margin={defaultChartMargins}>
                            <DefaultCartesianGrid />
                            <XAxis type="number" dataKey="x" stroke={colors.text} height={50}>
                                <Label content={<ChartAxisLabel value="Latência (ms)" axis="x" />} />
                            </XAxis>
                            <YAxis type="number" dataKey="y" stroke={colors.text}>
                                <Label content={<ChartAxisLabel value="Score" axis="y" />} position="insideLeft" />
                            </YAxis>
                            <DefaultTooltip cursor={DefaultTooltipCursor} />
                            <Scatter data={data} fill={colors.orange} />
                        </ScatterChart>
                    </ChartContainer>
                </WithData>
            </ChartFrame>
        </Card>
    );
};
