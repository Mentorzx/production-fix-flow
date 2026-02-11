import { useMemo } from "react";
import { ComposedChart, Line, Area, Bar, XAxis, YAxis, Legend, ReferenceLine, Label } from "recharts";
import { Theme } from "../../../ui/Theme.js";

import {
    Card,
    Activity,
    DefaultCartesianGrid,
    DefaultTooltip,
    ChartFrame,
    ChartContainer,
    WithData,
} from "../../../ui/BaseComponents.jsx";
import { ChartAxisLabel, renderWithHints } from "../../../ui/UIComponents.jsx";

const parseValue = (v) => {
    if (v === null || v === undefined) return null;
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : null;
};

export const GeneralizationGapCard = ({ liveData }) => {
    const data = useMemo(() => {
        const rows = Array.isArray(liveData) ? liveData : [];
        if (rows.length === 0) return [];

        let prevLoss = null;

        return rows
            .map((e, idx) => {
                if (!e || typeof e !== "object") return null;
                const epoch = typeof e.epoch === "number" ? e.epoch : idx + 1;

                // Allow 'loss' or 'train_loss' as the primary loss metric
                const rawLoss = e.loss ?? e.train_loss;
                const loss = parseValue(rawLoss);

                const metric = parseValue(e.mcc ?? e.mrr); // Prefer MCC, fallback to MRR

                let delta = null;
                if (loss !== null && prevLoss !== null) {
                    delta = prevLoss - loss; // Positive means improvement (loss went down)
                }
                if (loss !== null) prevLoss = loss;

                return {
                    epoch,
                    loss,
                    metric,
                    delta, // "Stability / Improvement Rate"
                };
            })
            .filter(Boolean);
    }, [liveData]);

    const hasData = data.length > 1;

    // Custom Tooltip for Dynamics
    const DynamicsTooltip = ({ active, payload, label }) => {
        if (!active || !payload || !payload.length) return null;

        const lossPayload = payload.find(p => p.dataKey === 'loss');
        const metricPayload = payload.find(p => p.dataKey === 'metric');
        const deltaPayload = payload.find(p => p.dataKey === 'delta');

        return (
            <div className="border p-3 rounded-xl shadow-2xl text-[10px]" style={{ backgroundColor: Theme.ui.background, borderColor: Theme.ui.border, color: Theme.ui.text.secondary }}>
                <div className="font-bold border-b pb-1 mb-1" style={{ borderColor: Theme.ui.border, color: Theme.ui.text.muted }}>
                    Epoch {label}
                </div>
                {lossPayload && (
                    <div className="flex items-center gap-2 mb-1">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: Theme.semantic.chart.loss }}></span>
                        <span style={{ color: Theme.ui.text.secondary }}>Loss:</span>
                        <span className="font-mono" style={{ color: Theme.ui.text.primary }}>{lossPayload.value?.toFixed(4)}</span>
                    </div>
                )}
                {deltaPayload && (
                    <div className="flex items-center gap-2 mb-1">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: Theme.palette.lime }}></span>
                        <span style={{ color: Theme.ui.text.secondary }}>Delta:</span>
                        <span className="font-mono" style={{ color: deltaPayload.value > 0 ? Theme.palette.vividGreen : Theme.palette.red }}>
                            {deltaPayload.value > 0 ? '▼' : '▲'} {Math.abs(deltaPayload.value)?.toFixed(4)}
                        </span>
                    </div>
                )}
                {metricPayload && (
                    <div className="flex items-center gap-2 border-t pt-1 mt-1" style={{ borderColor: Theme.ui.border }}>
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: Theme.semantic.chart.metric }}></span>
                        <span style={{ color: Theme.ui.text.secondary }}>MCC/MRR:</span>
                        <span className="font-mono" style={{ color: Theme.semantic.chart.metric }}>{metricPayload.value?.toFixed(4)}</span>
                    </div>
                )}
            </div>
        );
    };

    return (
        <Card title="Dinâmica de Otimização (Loss & Stability)" icon={Activity} className="h-full" helpText={{ tech: "Visualiza a taxa de convergência (Delta) e a correlação entre queda de Loss e ganho de Métrica (MCC/MRR).", simple: "Barras Teal = Melhora na Loss. Linha Laranja = Loss Absoluta. Linha Indigo = Métrica de Performance." }}>
            <ChartFrame className="p-3">
                <WithData when={hasData} empty="Aguardando dados de convergência..." emptyClassName="text-zinc-500">
                    <ChartContainer minHeight={0} className="min-h-0">
                        <ComposedChart data={data} margin={{ top: 22, right: 60, bottom: 18, left: 50 }}>
                            <DefaultCartesianGrid />
                            <XAxis dataKey="epoch" stroke={Theme.ui.text.secondary} height={32}>
                                <Label content={<ChartAxisLabel value="Epoch" axis="x" />} />
                            </XAxis>

                            {/* Left Axis: Loss */}
                            <YAxis
                                yAxisId="loss"
                                stroke={Theme.semantic.chart.loss}
                                tick={{ fill: Theme.ui.text.secondary }}
                                domain={['auto', 'auto']}
                                width={60}
                            >
                                <Label content={<ChartAxisLabel value="Loss" axis="y" />} position="insideLeft" />
                            </YAxis>

                            {/* Right Axis: Metric & Delta (Shared scale centered around 0 for small values) */}
                            <YAxis
                                yAxisId="metric"
                                orientation="right"
                                stroke={Theme.semantic.chart.metric}
                                tick={{ fill: Theme.ui.text.secondary }}
                                domain={['auto', 'auto']}
                                width={60}
                            >
                                <Label content={<ChartAxisLabel value="Stability" axis="y-right" offset={14} />} position="insideRight" />
                            </YAxis>

                            <DefaultTooltip content={<DynamicsTooltip />} />
                            <Legend
                                formatter={renderWithHints}
                                verticalAlign="top"
                                align="left"
                                height={18}
                                wrapperStyle={{ top: -8 }}
                            />
                            <ReferenceLine y={0} yAxisId="metric" stroke={Theme.ui.border} strokeDasharray="3 3" />

                            {/* Stability / Delta Bars (Teal=Good/Improvement, Rose=Bad/Regression) */}
                            <Bar
                                yAxisId="metric"
                                dataKey="delta"
                                name="Stability"
                                fill={Theme.palette.vividGreen}
                                barSize={4}
                                fillOpacity={0.6}
                            />

                            {/* Loss Area */}
                            <Area
                                type="monotone"
                                yAxisId="loss"
                                dataKey="loss"
                                name="Loss"
                                stroke={Theme.semantic.chart.loss}
                                fill={Theme.semantic.chart.loss}
                                fillOpacity={0.05}
                                strokeWidth={2}
                            />

                            {/* Metric Line */}
                            <Line
                                type="monotone"
                                yAxisId="metric"
                                dataKey="metric"
                                name="MCC/MRR"
                                stroke={Theme.semantic.chart.metric}
                                dot={false}
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
