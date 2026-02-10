import { useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Label } from 'recharts';

import { Card, Activity, ChartFrame, ChartContainer, BaseTooltip, colors } from "../../../ui/BaseComponents.jsx";
import { renderLegendWithHints, ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const GradientHealthCard = ({ liveData }) => {
    const data = useMemo(() => {
        if (!liveData || liveData.length === 0) return Array.from({ length: 20 }, (_, i) => ({ epoch: i + 1, norm: Math.max(0.1, 10 * Math.exp(-0.1 * i) + Math.random()) }));
        return liveData.map((e, idx) => ({ epoch: e.epoch || e.id || idx + 1, norm: e.grad_norm || e.loss }));
    }, [liveData]);

    return (
        <Card title="Saúde do Gradiente" icon={Activity} className="h-full" helpText={ChartRegistry.get('gradient_health')}>
            <ChartFrame>
                <ChartContainer>
                    <AreaChart data={data} margin={{ top: 20, right: 16, bottom: 10, left: 40 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                        <XAxis
                            dataKey="epoch"
                            stroke={colors.text}
                            tick={{ fill: '#71717a', fontSize: 10 }}
                            tickLine={{ stroke: '#52525b' }}
                            axisLine={{ stroke: '#52525b' }}
                            height={50}
                        >
                            <Label content={<ChartAxisLabel value="Epoch" axis="x" />} />
                        </XAxis>
                        <YAxis
                            stroke={colors.text}
                            tick={{ fill: '#71717a', fontSize: 10 }}
                            tickLine={{ stroke: '#52525b' }}
                            axisLine={{ stroke: '#52525b' }}
                            width={40}
                        >
                            <Label content={<ChartAxisLabel value="Grad Norm" axis="y" />} position="insideLeft" />
                        </YAxis>
                        <Tooltip content={<BaseTooltip />} />
                        <Legend formatter={renderLegendWithHints} verticalAlign="top" align="left" height={18} wrapperStyle={{ top: -8, fontSize: '11px' }} />
                        <Area
                            name="Grad Norm"
                            type="monotone"
                            dataKey="norm"
                            stroke="#f59e0b"
                            fill="#f59e0b"
                            fillOpacity={0.1}
                            strokeWidth={2}
                        />
                    </AreaChart>
                </ChartContainer>
            </ChartFrame>
        </Card>
    );
};
