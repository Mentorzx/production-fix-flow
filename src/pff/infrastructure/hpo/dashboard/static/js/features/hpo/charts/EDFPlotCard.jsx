import { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, Label } from 'recharts';

import { TrendingUp, colors, DefaultCartesianGrid, DefaultTooltip } from "../../../ui/BaseComponents.jsx";
import { ChartCard } from "../../../ui/withChartCard.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";

export const EDFPlotCard = ({ filteredTrials }) => {
    const data = useMemo(() => {
        if (!filteredTrials) return [];
        const values = filteredTrials.filter(t => t.state === 'COMPLETE' && t.value != null).map(t => t.value).sort((a, b) => a - b);
        return values.map((val, i) => ({ x: val, y: (i + 1) / values.length }));
    }, [filteredTrials]);
    return (
        <ChartCard title="EDF Plot" icon={TrendingUp} registryKey="edf" hasData={data.length > 1} emptyText="Dados insuficientes" emptyClassName="text-sm" chartFrameClassName="pt-8">
            <LineChart data={data}>
                <DefaultCartesianGrid vertical={false} />
                <XAxis type="number" dataKey="x" stroke={colors.text} height={50}>
                    <Label content={<ChartAxisLabel value="Objetivo" axis="x" />} />
                </XAxis>
                <YAxis type="number" dataKey="y" stroke={colors.text}>
                    <Label content={<ChartAxisLabel value="EDF" axis="y" />} position="insideLeft" />
                </YAxis>
                <DefaultTooltip />
                <Line type="stepAfter" dataKey="y" stroke={colors.success} strokeWidth={2} dot={false} />
            </LineChart>
        </ChartCard>
    );
};
