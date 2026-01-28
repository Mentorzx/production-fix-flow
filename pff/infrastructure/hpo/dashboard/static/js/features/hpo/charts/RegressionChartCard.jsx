import { useMemo } from 'react';
import { ComposedChart, Scatter, Line, Area, XAxis, YAxis, Legend, Label, ReferenceLine } from 'recharts';
import { Card, TrendingUp, DefaultCartesianGrid, DefaultTooltip, ChartFrame, ChartContainer, WithData } from "../../../ui/BaseComponents.jsx";
import { renderLegendWithHints, ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { Theme } from "../../../ui/Theme.js";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const RegressionChartCard = ({ trials }) => {
    const { data, r2, slope } = useMemo(() => {
        if (!trials || trials.length < 3) return { data: [], r2: 0, slope: 0, projectedEnd: 0 };

        const completed = trials.filter(t => t.state === 'COMPLETE' && t.value != null).map(t => ({ x: t.id, y: t.value }));
        if (completed.length < 3) return { data: completed, r2: 0, slope: 0, projectedEnd: 0 };

        // 1. Linear Regression (Least Squares)
        const n = completed.length;
        const sumX = completed.reduce((a, b) => a + b.x, 0);
        const sumY = completed.reduce((a, b) => a + b.y, 0);
        const sumXY = completed.reduce((a, b) => a + b.x * b.y, 0);
        const sumXX = completed.reduce((a, b) => a + b.x * b.x, 0);
        const sumYY = completed.reduce((a, b) => a + b.y * b.y, 0);

        const slopeVal = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slopeVal * sumX) / n;

        // 2. R² Calculation
        const ssTot = sumYY - (sumY * sumY) / n;
        const ssRes = completed.reduce((a, b) => {
            const pred = slopeVal * b.x + intercept;
            return a + Math.pow(b.y - pred, 2);
        }, 0);
        const r2Val = 1 - (ssRes / ssTot);

        // 3. Prediction Bands (Standard Error)
        // Approx: pred +/- 1.96 * StdDev of Residuals
        const sigma = Math.sqrt(ssRes / (n - 2));
        const margin = 1.96 * sigma;

        // 4. Projection (Extend 20% or 10 trials)
        const maxId = Math.max(...completed.map(c => c.x));
        const extension = Math.max(10, Math.ceil(n * 0.2));
        const points = [];

        // Existing Points
        completed.forEach(p => {
            const trend = slopeVal * p.x + intercept;
            points.push({
                x: p.x,
                y: p.y,
                trend: trend,
                ci_low: trend - margin,
                ci_high: trend + margin,
                isProjection: false
            });
        });

        // Future Points (Projected)
        for (let i = 1; i <= extension; i++) {
            const nextX = maxId + i;
            const trend = slopeVal * nextX + intercept;
            // Expand uncertainty slightly for projection (simple heuristic: linear growth of margin)
            // const projectionMargin = margin * (1 + (i * 0.05)); 
            const projectionMargin = margin;

            points.push({
                x: nextX,
                trend: trend,
                ci_low: trend - projectionMargin,
                ci_high: trend + projectionMargin,
                isProjection: true
            });
        }

        return { data: points, r2: r2Val, slope: slopeVal };
    }, [trials]);

    const title = `Projeção de Tendência (R² = ${r2.toFixed(3)})`;
    const helpChart = { ...ChartRegistry.get('regression_chart'), simple: ChartRegistry.get('regression_chart')?.simple + ` O valor R² de ${r2.toFixed(3)} indica a força da tendência.` };

    return (
        <Card title={title} icon={TrendingUp} className="h-full" helpText={helpChart}>
            <ChartFrame>
                <WithData when={data.length > 0} empty="Dados insuficientes para regressão">
                    <ChartContainer>
                        <ComposedChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                            <DefaultCartesianGrid />
                            <XAxis
                                dataKey="x"
                                type="number"
                                stroke={Theme.ui.text.secondary}
                                domain={['dataMin', 'dataMax']}
                                tickCount={8}
                                height={50}
                            >
                                <Label content={<ChartAxisLabel value="Trial" axis="x" />} />
                            </XAxis>
                            <YAxis
                                stroke={Theme.ui.text.secondary}
                                domain={['auto', 'auto']}
                                width={60}
                            >
                                <Label content={<ChartAxisLabel value="Score" axis="y" />} position="insideLeft" />
                            </YAxis>

                            <DefaultTooltip
                                payloadUniqBy={(item) => item.dataKey}
                                labelFormatter={(v) => `Trial #${v}`}
                                filterNull={false}
                            />

                            <Legend formatter={renderLegendWithHints} verticalAlign="top" height={36} wrapperStyle={{ top: -10 }} />

                            {/* Prediction Band (Confidence) */}
                            <Area
                                name="Intervalo de Confiança (95%)"
                                dataKey={(d) => [d.ci_low, d.ci_high]}
                                stroke="none"
                                fill={Theme.semantic.warning}
                                fillOpacity={0.1}
                            />

                            {/* Scatter Points (Real Data) */}
                            <Scatter name="Trials (Reais)" dataKey="y" fill={Theme.semantic.primary} shape="circle" />

                            {/* Linear Regression Line */}
                            <Line
                                name={`Tendência (${slope > 0 ? '+' : ''}${slope.toFixed(5)}/trial)`}
                                type="monotone"
                                dataKey="trend"
                                stroke={Theme.semantic.success}
                                dot={(d) => d.isProjection ? { r: 2, fill: Theme.semantic.success, strokeWidth: 0 } : false}
                                strokeWidth={2}
                                strokeDasharray="5 5"
                            />

                            {/* Differentiate Projection Zone ?? Maybe a ReferenceLine at maxId */}
                            <ReferenceLine x={data.filter(d => !d.isProjection).pop()?.x} stroke={Theme.ui.border} strokeDasharray="3 3" label={{ value: "Hoje", position: 'insideTopRight', fill: Theme.ui.text.muted, fontSize: 10 }} />

                        </ComposedChart>
                    </ChartContainer>
                </WithData>
            </ChartFrame>
        </Card>
    );
};
