import { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, Legend, Label } from 'recharts';

import {
    Card,
    TrendingUp,
    ChartFrame,
    ChartContainer,
    WithData,
    DefaultCartesianGrid,
    DefaultTooltip,
    colors,
} from "../../../ui/BaseComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { renderLegendWithHints, ChartAxisLabel } from "../../../ui/UIComponents.jsx";

export const LossProjectionCard = ({ liveData }) => {
    const data = useMemo(() => {
        const rows = Array.isArray(liveData) ? liveData : [];
        if (rows.length === 0) return [];

        const base = rows
            .map((e, idx) => {
                if (!e || typeof e !== "object") return null;
                const epoch = typeof e.epoch === "number" ? e.epoch : idx + 1;
                const rawLoss = e.loss ?? e.train_loss ?? e.val_loss ?? e.binary_loss;
                const loss = Number.parseFloat(rawLoss);
                return Number.isFinite(loss) ? { epoch, loss } : null;
            })
            .filter(Boolean);

        if (base.length === 0) return [];
        const last = base[base.length - 1];
        const projection = Array.from({ length: 10 }, (_, i) => ({
            epoch: last.epoch + i + 1,
            loss: last.loss * Math.pow(0.95, i + 1),
        }));
        return [...base, ...projection];
    }, [liveData]);
    return (
        <Card title="Extrapolação de Perda" icon={TrendingUp} className="h-full" helpText={ChartRegistry.get('loss_projection')}>
            <ChartFrame className="p-2 h-full min-h-[120px]">
                <WithData when={data.length > 0} empty="Aguardando...">
                    <ChartContainer minHeight={120} className="h-full min-h-[120px]">
                        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 20, left: 40 }}>
                            <DefaultCartesianGrid />
                            <XAxis dataKey="epoch" height={24} tick={{ fill: colors.text }}>
                                <Label content={<ChartAxisLabel value="Epoch" axis="x" />} />
                            </XAxis>
                            <YAxis tick={{ fill: colors.text }} width={48}>
                                <Label content={<ChartAxisLabel value="Loss" axis="y" />} position="insideLeft" />
                            </YAxis>
                            <DefaultTooltip />
                            <Legend formatter={renderLegendWithHints} verticalAlign="top" align="left" height={18} wrapperStyle={{ top: -8 }} />
                            <Line
                                type="monotone"
                                dataKey="loss"
                                name="Loss projetada"
                                stroke={colors.primary}
                                strokeWidth={2}
                                dot={false}
                                strokeDasharray="5 5"
                            />
                        </LineChart>
                    </ChartContainer>
                </WithData>
            </ChartFrame>
        </Card>
    );
};
