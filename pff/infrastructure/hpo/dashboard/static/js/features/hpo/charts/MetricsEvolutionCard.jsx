import { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, Legend, Label } from 'recharts';
import { useStoreState } from "../../../store/store.jsx";
import { resolveMetricValue } from "../../../domain/metrics/Formatters.js";

import { Card, TrendingUp, colors, DefaultCartesianGrid, DefaultTooltip, ChartFrame, ChartContainer } from "../../../ui/BaseComponents.jsx";
import { Theme } from "../../../ui/Theme.js";
import { renderLegendWithHints, ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const MetricsEvolutionCard = ({ trials }) => {
    const { filters } = useStoreState();

    const metricPalette = useMemo(() => [
        Theme.semantic.primary,
        Theme.semantic.success,
        Theme.semantic.warning,
        Theme.semantic.highlight,
        Theme.semantic.error,
        Theme.semantic.info,
        Theme.palette.lime,
        Theme.palette.teal,
        Theme.palette.hotOrange,
        Theme.palette.red
    ], []);

    const { data, keys, colorByKey } = useMemo(() => {
        if (!trials || trials.length === 0) return { data: [], keys: [], colorByKey: {} };

        const allowed = ['score', 'mrr', 'mcc', 'accuracy', 'f1', 'auc', 'precision', 'recall', 'hits1', 'hits3', 'hits10'];
        const colorByKey = Object.fromEntries(allowed.map((key, index) => [key, metricPalette[index % metricPalette.length]]));

        // Filter logic: Must be COMPLETE. Also skip Warmstart if filter says so.
        // Note: 'trials' prop might be raw. If we want strict adherence to global filters, we should check them.
        const filtered = trials.filter(t => {
            if (t.state !== 'COMPLETE') return false;
            // User requested: "pulando warmstart quando o filtro tiver ativo"
            // filters.includeWarmup defaults to false. If false, we skip warmstarts.
            if (!filters.includeWarmup && t.warmstart) return false;
            return true;
        });

        const proc = filtered.map(t => {
            const row = { id: t.id, score: t.value || 0 };
            allowed.forEach(k => {
                const val = resolveMetricValue(t, k);
                if (val !== null && typeof val === 'number') {
                    row[k] = val;
                }
            });
            return row;
        }).sort((a, b) => a.id - b.id);

        // Determine which keys actually have data to avoid empty lines
        const presentKeys = allowed.filter(k => k === 'score' || proc.some(r => r[k] !== undefined));

        return { data: proc, keys: presentKeys, colorByKey };
    }, [trials, filters.includeWarmup, metricPalette]);
    return (
        <Card title="Evolução de Métricas" icon={TrendingUp} className="h-full" helpText={ChartRegistry.get('metrics_evolution')}>
            <ChartFrame>
                <ChartContainer>
                    <LineChart data={data}>
                        <DefaultCartesianGrid />
                        <XAxis dataKey="id" stroke={colors.text} height={50}>
                            <Label content={<ChartAxisLabel value="Trial" axis="x" />} />
                        </XAxis>
                        <YAxis stroke={colors.text} domain={[0, 1]}>
                            <Label content={<ChartAxisLabel value="Score" axis="y" />} position="insideLeft" />
                        </YAxis>
                        <DefaultTooltip />
                        <Legend formatter={renderLegendWithHints} verticalAlign="bottom" align="center" wrapperStyle={{ paddingTop: 6 }} />
                        {keys.map(k => (
                            <Line
                                key={k}
                                type="monotone"
                                dataKey={k}
                                stroke={colorByKey?.[k] || colors.success}
                                dot={false}
                                name={k.toUpperCase()}
                            />
                        ))}
                    </LineChart>
                </ChartContainer>
            </ChartFrame>
        </Card>
    );
};
