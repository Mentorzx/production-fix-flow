import { useMemo } from 'react';
import { ComposedChart, Area, Line, XAxis, YAxis, Legend, Label } from 'recharts';

import { Card, TrendingUp, colors, DefaultCartesianGrid, DefaultTooltip, ChartContainer, WithData, ChartFrame } from "../../../ui/BaseComponents.jsx";
import { ChartAxisLabel, renderWithHints } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const LearningCurveChart = ({ data }) => {
    const rows = useMemo(() => {
        const items = Array.isArray(data) ? data : [];
        return items
            .map((row, idx) => {
                if (!row || typeof row !== "object") return null;

                const epoch = typeof row.epoch === "number"
                    ? row.epoch
                    : (typeof row.id === "number" ? row.id : idx + 1);

                const metrics = row.metrics && typeof row.metrics === "object" ? row.metrics : null;

                const parseValue = (v) => {
                    if (v === null || v === undefined) return null;
                    const n = parseFloat(v);
                    return Number.isFinite(n) ? n : null;
                };

                return {
                    epoch,
                    train_loss: parseValue(row.train_loss ?? row.loss ?? metrics?.train_loss ?? metrics?.loss),
                    val_loss: parseValue(row.val_loss ?? row.validation_loss ?? row.binary_loss ?? metrics?.val_loss ?? metrics?.validation_loss ?? metrics?.binary_loss),
                };
            })
            .filter(Boolean);
    }, [data]);

    const hasData = rows.length > 0 && rows.some((row) => row.train_loss != null || row.val_loss != null);

    return (
        <Card title="Curvas de Aprendizado" icon={TrendingUp} className="h-full" helpText={ChartRegistry.get('learning_curve')}>
            <ChartFrame>
                <WithData when={hasData} empty="Aguardando curvas de treino..." emptyClassName="text-zinc-500">
                    <ChartContainer>
                        <ComposedChart data={rows} margin={{ top: 20, right: 16, bottom: 10, left: 40 }}>
                            <defs>
                                <linearGradient id="gradTrainLoss" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor={colors.primary} stopOpacity={0.25} />
                                    <stop offset="95%" stopColor={colors.primary} stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <DefaultCartesianGrid />
                            <XAxis dataKey="epoch" stroke={colors.text} height={50}>
                                <Label content={<ChartAxisLabel value="Epoch" axis="x" />} />
                            </XAxis>
                            <YAxis stroke={colors.text}>
                                <Label content={<ChartAxisLabel value="Loss" axis="y" />} position="insideLeft" />
                            </YAxis>
                            <DefaultTooltip />
                            <Legend formatter={renderWithHints} verticalAlign="top" align="left" height={18} wrapperStyle={{ top: -8 }} />
                            <Area type="monotone" dataKey="train_loss" name="Train Loss" stroke={colors.primary} fill="url(#gradTrainLoss)" strokeWidth={2} dot={false} connectNulls />
                            <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke={colors.error} dot={false} connectNulls strokeDasharray="4 4" />
                        </ComposedChart>
                    </ChartContainer>
                </WithData>
            </ChartFrame>
        </Card>
    );
};
