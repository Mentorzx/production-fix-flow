/**
 * Provide MetricsEvolutionCard module functionality for the HPO dashboard.
 */

import { useId, useMemo } from "react";
import { LineChart, Line, Area, XAxis, YAxis, Legend, Label } from "recharts";
import { useStoreState } from "../../../store/store.jsx";
import { resolveMetricValue } from "../../../domain/metrics/Formatters.js";

import {
  Card,
  TrendingUp,
  colors,
  DefaultCartesianGrid,
  DefaultTooltip,
  ChartFrame,
  ChartContainer,
  getChartAreaGradientStops,
} from "../../../ui/BaseComponents.jsx";
import { Theme } from "../../../ui/Theme.js";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { InteractiveLegend, useLegendVisibility } from "../../../ui/ChartPrimitives.jsx";

/**
 * Expose metrics evolution card for dashboard usage.
 */
export const MetricsEvolutionCard = ({ trials }) => {
  const gradientSuffix = useId().replace(/:/g, "");
  const { filters } = useStoreState();
  const scoreStops = useMemo(
    () => getChartAreaGradientStops("primaryReadable", Theme.semantic.primary),
    []
  );

  const metricPalette = useMemo(
    () => [
      Theme.semantic.primary,
      Theme.semantic.success,
      Theme.semantic.warning,
      Theme.semantic.highlight,
      Theme.semantic.error,
      Theme.semantic.info,
      Theme.palette.lime,
      Theme.palette.teal,
      Theme.palette.hotOrange,
      Theme.palette.red,
    ],
    []
  );

  const { data, keys, colorByKey } = useMemo(() => {
    if (!trials || trials.length === 0) return { data: [], keys: [], colorByKey: {} };

    const allowed = [
      "score",
      "mrr",
      "mcc",
      "accuracy",
      "f1",
      "auc",
      "precision",
      "recall",
      "hits1",
      "hits3",
      "hits10",
    ];
    const colorByKey = Object.fromEntries(
      allowed.map((key, index) => [key, metricPalette[index % metricPalette.length]])
    );

    // Filter logic: Must be COMPLETE. Also skip Warmstart if filter says so.
    // Note: 'trials' prop might be raw. If we want strict adherence to global filters, we should check them.
    const filtered = trials.filter((t) => {
      if (t.state !== "COMPLETE") return false;
      // User requested: "pulando warmstart quando o filtro tiver ativo"
      // filters.includeWarmup defaults to false. If false, we skip warmstarts.
      if (!filters.includeWarmup && t.warmstart) return false;
      return true;
    });

    const proc = filtered
      .map((t) => {
        const row = { id: t.id, score: t.value || 0 };
        allowed.forEach((k) => {
          const val = resolveMetricValue(t, k);
          if (val !== null && typeof val === "number") {
            row[k] = val;
          }
        });
        return row;
      })
      .sort((a, b) => a.id - b.id);

    // Determine which keys actually have data to avoid empty lines
    const presentKeys = allowed.filter(
      (k) => k === "score" || proc.some((r) => r[k] !== undefined)
    );

    return { data: proc, keys: presentKeys, colorByKey };
  }, [trials, filters.includeWarmup, metricPalette]);
  const { hiddenKeys, toggleSeriesVisibility, isSeriesVisible } = useLegendVisibility(keys);
  return (
    <Card
      title="Evolução de Métricas"
      icon={TrendingUp}
      className="h-full"
      helpText={ChartRegistry.get("metrics_evolution")}
    >
      <ChartFrame>
        <ChartContainer>
          <LineChart data={data} margin={{ top: 20, right: 10, bottom: 10, left: 10 }}>
            <defs>
              <linearGradient id={`grad-score-${gradientSuffix}`} x1="0" y1="0" x2="0" y2="1">
                {scoreStops.map((stop, index) => (
                  <stop
                    key={`${gradientSuffix}-score-${index}`}
                    offset={stop.offset}
                    stopColor={stop.color}
                    stopOpacity={stop.opacity}
                  />
                ))}
              </linearGradient>
            </defs>
            <DefaultCartesianGrid />
            <XAxis dataKey="id" stroke={colors.text} height={50}>
              <Label content={<ChartAxisLabel value="Trial" axis="x" />} />
            </XAxis>
            <YAxis stroke={colors.text} domain={[0, 1]}>
              <Label content={<ChartAxisLabel value="Score" axis="y" />} position="insideLeft" />
            </YAxis>
            <DefaultTooltip />
            <Legend
              layout="horizontal"
              verticalAlign="top"
              align="right"
              height={28}
              iconSize={8}
              wrapperStyle={{ top: -10, whiteSpace: "nowrap", overflow: "hidden" }}
              content={(props) => (
                <InteractiveLegend
                  {...props}
                  hiddenKeys={hiddenKeys}
                  onToggleSeries={toggleSeriesVisibility}
                  seriesKeys={keys}
                  align="right"
                />
              )}
            />
            <Area
              isAnimationActive={false}
              type="monotone"
              dataKey="score"
              stroke="none"
              fill={`url(#grad-score-${gradientSuffix})`}
              fillOpacity={1}
              baseValue="dataMin"
              legendType="none"
              hide={!isSeriesVisible("score")}
            />
            {keys.map((k) => (
              <Line
                isAnimationActive={false}
                key={k}
                type="monotone"
                dataKey={k}
                stroke={colorByKey?.[k] || colors.success}
                dot={false}
                name={k.toUpperCase()}
                hide={!isSeriesVisible(k)}
              />
            ))}
          </LineChart>
        </ChartContainer>
      </ChartFrame>
    </Card>
  );
};
