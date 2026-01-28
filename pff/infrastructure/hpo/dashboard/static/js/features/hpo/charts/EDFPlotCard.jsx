import { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, Label } from 'recharts';

import { Card, TrendingUp, colors, DefaultCartesianGrid, DefaultTooltip, ChartFrame, ChartContainer, WithData } from "../../../ui/BaseComponents.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const EDFPlotCard = ({ filteredTrials }) => {
    const data = useMemo(() => {
        if (!filteredTrials) return [];
        const values = filteredTrials.filter(t => t.state === 'COMPLETE' && t.value != null).map(t => t.value).sort((a, b) => a - b);
        return values.map((val, i) => ({ x: val, y: (i + 1) / values.length }));
    }, [filteredTrials]);
    return (
        <Card title="EDF Plot" icon={TrendingUp} className="h-full" helpText={ChartRegistry.get('edf')}>
            <ChartFrame className="pt-8">
                <WithData when={data.length > 1} empty="Dados insuficientes" emptyClassName="text-sm">
                    <ChartContainer>
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
                    </ChartContainer>
                </WithData>
            </ChartFrame>
        </Card>
    );
};
