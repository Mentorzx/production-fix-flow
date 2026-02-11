import { useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, Label } from 'recharts';

import { Layers, colors, defaultChartMargins, DefaultCartesianGrid, DefaultTooltip, DefaultTooltipCursor } from "../../../ui/BaseComponents.jsx";
import { ChartCard } from "../../../ui/withChartCard.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";

export const StructuralMetricsCard = ({ trials }) => {
    const data = useMemo(() => { if (!trials || trials.length === 0) return []; return trials.filter(t => t.state === 'COMPLETE' && t.value > 0).map(t => ({ id: t.id, x: t.params?.embedding_dim || 0, y: t.value || 0 })); }, [trials]);
    return (
        <ChartCard title="Métricas Estruturais" icon={Layers} registryKey="structural_metrics" hasData={data.length > 0}>
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
        </ChartCard>
    );
};
