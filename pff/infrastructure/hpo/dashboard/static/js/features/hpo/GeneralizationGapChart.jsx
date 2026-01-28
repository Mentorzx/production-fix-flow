import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Label } from 'recharts';

import { colors, ChartContainer } from "../../ui/BaseComponents.jsx";
import { ChartAxisLabel } from "../../ui/UIComponents.jsx";

export const GeneralizationGapChart = ({ data = [] }) => {
    return (
        <div className="h-full w-full min-h-[300px]" style={{ minHeight: 300 }}>
            <ChartContainer className="min-h-[300px]">
                <AreaChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
                    <XAxis dataKey="epoch" stroke={colors.text} height={50}>
                        <Label content={<ChartAxisLabel value="Epoch" axis="x" />} />
                    </XAxis>
                    <YAxis stroke={colors.text}>
                        <Label content={<ChartAxisLabel value="Loss" axis="y" />} position="insideLeft" />
                    </YAxis>
                    <Tooltip contentStyle={{ backgroundColor: colors.tooltip }} />
                    <Area type="monotone" dataKey="loss" stroke={colors.primary} fill={colors.primary} fillOpacity={0.1} />
                </AreaChart>
            </ChartContainer>
        </div>
    );
};
