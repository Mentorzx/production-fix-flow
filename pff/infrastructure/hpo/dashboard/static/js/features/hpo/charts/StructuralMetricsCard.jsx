import { useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, Label } from 'recharts';

import { Card, Layers, colors, defaultChartMargins, DefaultCartesianGrid, DefaultTooltip, DefaultTooltipCursor, ChartFrame, ChartContainer, WithData } from "../../../ui/BaseComponents.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const StructuralMetricsCard = ({ trials }) => {
    const data = useMemo(() => { if (!trials || trials.length === 0) return []; return trials.filter(t => t.state === 'COMPLETE' && t.value > 0).map(t => ({ id: t.id, x: t.params?.embedding_dim || 0, y: t.value || 0 })); }, [trials]);
    return (
        <Card title="Métricas Estruturais" className="h-full" icon={Layers} helpText={ChartRegistry.get('structural_metrics')}>
            <ChartFrame>
                <WithData when={data.length > 0} empty="Sem dados">
                    <ChartContainer>
                        <ScatterChart margin={defaultChartMargins}>
                            <DefaultCartesianGrid />
                            <XAxis type="number" dataKey="x" stroke={colors.text} height={50}>
                                <Label content={<ChartAxisLabel value="Dimensão" axis="x" />} />
                            </XAxis>
                            <YAxis type="number" dataKey="y" stroke={colors.text}>
                                <Label content={<ChartAxisLabel value="Score" axis="y" />} position="insideLeft" />
                            </YAxis>
                            <DefaultTooltip cursor={DefaultTooltipCursor} />
                            <Scatter data={data} fill={colors.amber} />
                        </ScatterChart>
                    </ChartContainer>
                </WithData>
            </ChartFrame>
        </Card>
    );
};
