import { useMemo } from "react";
import { ComposedChart, Area, XAxis, YAxis, Legend, Label } from "recharts";
import { Theme } from "../../../ui/Theme.js";

import {
    Card,
    TrendingUp,
    DefaultCartesianGrid,
    DefaultTooltip,
    ChartFrame,
    ChartContainer,
    WithData,
} from "../../../ui/BaseComponents.jsx";
import { renderLegendWithHints, ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

const parseValue = (v) => {
    if (v === null || v === undefined) return null;
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : null;
};

export const TrialLearningMetricsCard = ({ liveData }) => {
    const data = useMemo(() => {
        const rows = Array.isArray(liveData) ? liveData : [];
        return rows
            .map((e, idx) => {
                if (!e || typeof e !== "object") return null;
                const epoch = typeof e.epoch === "number" ? e.epoch : idx + 1;
                return {
                    epoch,
                    loss: parseValue(e.loss),
                    mrr: parseValue(e.mrr),
                    mcc: parseValue(e.mcc),
                };
            })
            .filter(Boolean);
    }, [liveData]);

    const hasData = data.length > 1 && data.some((d) => d.loss != null || d.mrr != null || d.mcc != null);

    const helpText = ChartRegistry.get("trial_learning_metrics");

    return (
        <Card title="Loss + MCC/MRR" icon={TrendingUp} className="h-full" helpText={helpText}>
            <ChartFrame>
                <WithData when={hasData} empty="Aguardando histórico do trial..." emptyClassName="text-zinc-500">
                    <ChartContainer>
                        <ComposedChart data={data} margin={{ top: 20, right: 60, bottom: 50, left: 60 }}>
                            <defs>
                                <linearGradient id="gradLoss" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor={Theme.semantic.chart.loss} stopOpacity={0.4} />
                                    <stop offset="95%" stopColor={Theme.semantic.chart.loss} stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="gradMrr" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor={Theme.palette.neonBlue} stopOpacity={0.4} />
                                    <stop offset="95%" stopColor={Theme.palette.neonBlue} stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="gradMcc" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor={Theme.palette.vividGreen} stopOpacity={0.4} />
                                    <stop offset="95%" stopColor={Theme.palette.vividGreen} stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <DefaultCartesianGrid />
                            <XAxis dataKey="epoch" stroke={Theme.ui.text.secondary} height={50}>
                                <Label content={<ChartAxisLabel value="Epoch" axis="x" />} />
                            </XAxis>
                            <YAxis
                                yAxisId="loss"
                                stroke={Theme.semantic.chart.loss}
                                tick={{ fill: Theme.ui.text.secondary }}
                                domain={[0, "auto"]}
                                width={60}
                            >
                                <Label content={<ChartAxisLabel value="Loss" axis="y" />} position="insideLeft" />
                            </YAxis>
                            <YAxis
                                yAxisId="metric"
                                orientation="right"
                                stroke={Theme.palette.neonBlue}
                                tick={{ fill: Theme.ui.text.secondary }}
                                domain={[0, 1]}
                                width={60}
                            >
                                <Label content={<ChartAxisLabel value="Metrics" axis="y-right" />} position="insideRight" />
                            </YAxis>
                            <DefaultTooltip />
                            <Legend formatter={renderLegendWithHints} />

                            <Area
                                type="monotone"
                                yAxisId="loss"
                                dataKey="loss"
                                name="LOSS"
                                stroke={Theme.semantic.chart.loss}
                                fill="url(#gradLoss)"
                                strokeWidth={2}
                                connectNulls
                            />
                            <Area
                                type="monotone"
                                yAxisId="metric"
                                dataKey="mrr"
                                name="MRR"
                                stroke={Theme.palette.neonBlue}
                                fill="url(#gradMrr)"
                                strokeWidth={2}
                                connectNulls
                            />
                            <Area
                                type="monotone"
                                yAxisId="metric"
                                dataKey="mcc"
                                name="MCC"
                                stroke={Theme.palette.vividGreen}
                                fill="url(#gradMcc)"
                                strokeWidth={2}
                                connectNulls
                            />
                        </ComposedChart>
                    </ChartContainer>
                </WithData>
            </ChartFrame>
        </Card>
    );
};
