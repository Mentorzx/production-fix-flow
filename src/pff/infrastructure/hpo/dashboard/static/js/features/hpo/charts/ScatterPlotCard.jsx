import { ScatterChart, Scatter, XAxis, YAxis, Label } from 'recharts';

import { Card, Search, colors, defaultChartMargins, DefaultCartesianGrid, DefaultTooltip, DefaultTooltipCursor, ChartFrame, ChartContainer, WithData } from "../../../ui/BaseComponents.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const ScatterPlotCard = ({ title, data, xLabel, yLabel, action }) => {
    return (
        <Card title={title} icon={Search} className="h-full" action={action} helpText={ChartRegistry.get('scatter_plot')}>
            <ChartFrame>
                <WithData when={data && data.length > 0} empty="Aguardando dados...">
                    <ChartContainer>
                        <ScatterChart margin={defaultChartMargins}>
                            <DefaultCartesianGrid />
                            <XAxis type="number" dataKey="x" stroke={colors.text} height={50}>
                                <Label content={<ChartAxisLabel value={xLabel} axis="x" />} />
                            </XAxis>
                            <YAxis type="number" dataKey="y" stroke={colors.text}>
                                <Label content={<ChartAxisLabel value={yLabel} axis="y" />} position="insideLeft" />
                            </YAxis>
                            <DefaultTooltip cursor={DefaultTooltipCursor} />
                            <Scatter data={data} fill={colors.primary} />
                        </ScatterChart>
                    </ChartContainer>
                </WithData>
            </ChartFrame>
        </Card>
    );
};
