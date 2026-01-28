import { useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

import { Card, Activity, ChartFrame, ChartContainer, BaseTooltip } from "../../../ui/BaseComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const GradientHealthCard = ({ liveData }) => {
    const data = useMemo(() => {
        if (!liveData || liveData.length === 0) return Array.from({ length: 20 }, (_, i) => ({ id: i, norm: Math.max(0.1, 10 * Math.exp(-0.1 * i) + Math.random()) }));
        return liveData.map(e => ({ id: e.id || e.epoch, norm: e.grad_norm || e.loss }));
    }, [liveData]);

    return (
        <Card title="Saúde do Gradiente" icon={Activity} className="h-full" helpText={ChartRegistry.get('gradient_health')}>
            <ChartFrame>
                <ChartContainer>
                    <AreaChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                        <XAxis
                            dataKey="id"
                            stroke="#52525b"
                            tick={{ fill: '#71717a', fontSize: 10 }}
                            tickLine={{ stroke: '#52525b' }}
                            axisLine={{ stroke: '#52525b' }}
                        />
                        <YAxis
                            stroke="#52525b"
                            tick={{ fill: '#71717a', fontSize: 10 }}
                            tickLine={{ stroke: '#52525b' }}
                            axisLine={{ stroke: '#52525b' }}
                            width={40}
                        />
                        <Tooltip content={<BaseTooltip />} />
                        <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
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
