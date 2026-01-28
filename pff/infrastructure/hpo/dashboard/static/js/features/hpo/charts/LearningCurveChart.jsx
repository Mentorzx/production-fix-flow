import { LineChart, Line, XAxis, YAxis } from 'recharts';
import { Label } from 'recharts';

import { Card, TrendingUp, colors, DefaultCartesianGrid, DefaultTooltip, ChartContainer, WithData, ChartFrame } from "../../../ui/BaseComponents.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const LearningCurveChart = ({ data }) => {
    const rows = Array.isArray(data) ? data : [];
    const hasData = rows.length > 0 && rows.some((row) => row && (row.train_loss != null || row.val_loss != null));

    return (
        <Card title="Curvas de Aprendizado" icon={TrendingUp} className="h-full" helpText={ChartRegistry.get('learning_curve')}>
            <ChartFrame>
                <WithData when={hasData} empty="Aguardando curvas de treino..." emptyClassName="text-zinc-500">
                    <ChartContainer>
                        <LineChart data={rows}>
                            <DefaultCartesianGrid />
                            <XAxis dataKey="epoch" stroke={colors.text} height={50}>
                                <Label content={<ChartAxisLabel value="Epoch" axis="x" />} />
                            </XAxis>
                            <YAxis stroke={colors.text}>
                                <Label content={<ChartAxisLabel value="Loss" axis="y" />} position="insideLeft" />
                            </YAxis>
                            <DefaultTooltip />
                            <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke={colors.primary} dot={false} />
                            <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke={colors.error} dot={false} />
                        </LineChart>
                    </ChartContainer>
                </WithData>
            </ChartFrame>
        </Card>
    );
};
