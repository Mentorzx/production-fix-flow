/**
 * Provide RegressionChartCard module functionality for the HPO dashboard.
 */

import { useMemo } from "react";
import {
  ComposedChart,
  Scatter,
  Line,
  Area,
  XAxis,
  YAxis,
  Legend,
  Label,
  ReferenceLine,
} from "recharts";
import { TrendingUp } from "../../../ui/icons.jsx";
import { DefaultCartesianGrid, DefaultTooltip, ChartFrame, ChartContainer } from "../../../ui/ChartPrimitives.jsx";
import { Card } from "../../../ui/Card.jsx";
import { WithData } from "../../../ui/EmptyStates.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { Theme } from "../../../ui/Theme.js";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { linearRegression } from "../../../utils/statistics.js";
import { InteractiveLegend, useLegendVisibility } from "../../../ui/ChartPrimitives.jsx";

/**
 * Expose regression chart card for dashboard usage.
 */
export const RegressionChartCard = ({ trials, totalTrials = 50 }) => {
  const safeTotalTrials = useMemo(() => {
    const parsedTotalTrials = Number(totalTrials);
    return Number.isFinite(parsedTotalTrials) && parsedTotalTrials > 0
      ? Math.floor(parsedTotalTrials)
      : 50;
  }, [totalTrials]);

  const { data, r2, slope } = useMemo(() => {
    if (!trials || trials.length < 3) return { data: [], r2: 0, slope: 0, projectedEnd: 0 };

    const completed = trials
      .filter(
        (t) => t.state === "COMPLETE" && t.value != null && Number.isFinite(t.id) && t.id >= 1
      )
      .map((t) => ({ x: t.id, y: t.value }));
    if (completed.length < 3) return { data: completed, r2: 0, slope: 0, projectedEnd: 0 };

    const { slope: slopeVal, intercept, r2: r2Val, ssRes } = linearRegression(completed);
    const n = completed.length;

    const sigma = Math.sqrt(ssRes / (n - 2));
    const margin = 1.96 * sigma;

    const maxId = Math.max(...completed.map((c) => c.x));
    const points = [];

    completed.forEach((p) => {
      const trend = slopeVal * p.x + intercept;
      points.push({
        x: p.x,
        y: p.y,
        trend: trend,
        ci_low: trend - margin,
        ci_high: trend + margin,
        ci_band: [trend - margin, trend + margin],
        isProjection: false,
      });
    });

    const maxProjectionX = Math.max(maxId, safeTotalTrials);
    for (let nextX = maxId + 1; nextX <= maxProjectionX; nextX++) {
      const trend = slopeVal * nextX + intercept;
      const projectionMargin = margin;

      points.push({
        x: nextX,
        trend: trend,
        ci_low: trend - projectionMargin,
        ci_high: trend + projectionMargin,
        ci_band: [trend - projectionMargin, trend + projectionMargin],
        isProjection: true,
      });
    }

    return { data: points, r2: r2Val, slope: slopeVal };
  }, [trials, safeTotalTrials]);
  const { hiddenKeys, toggleSeriesVisibility, isSeriesVisible } = useLegendVisibility([
    "ci_band",
    "y",
    "trend",
  ]);

  const title = `Projeção de Tendência (R² = ${r2.toFixed(3)})`;
  const helpChart = {
    ...ChartRegistry.get("regression_chart"),
    simple:
      ChartRegistry.get("regression_chart")?.simple +
      ` O valor R² de ${r2.toFixed(3)} indica a força da tendência.`,
  };

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
                domain={[1, safeTotalTrials]}
                allowDataOverflow={true}
                tickCount={8}
                height={50}
              >
                <Label content={<ChartAxisLabel value="Trial" axis="x" />} />
              </XAxis>
              <YAxis
                stroke={Theme.ui.text.secondary}
                domain={[0, 1]}
                allowDataOverflow={true}
                width={60}
              >
                <Label content={<ChartAxisLabel value="Score" axis="y" />} position="insideLeft" />
              </YAxis>

              <DefaultTooltip
                payloadUniqBy={(item) => item.dataKey}
                labelFormatter={(v) => `Trial #${v}`}
                filterNull={false}
              />

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
                    seriesKeys={["ci_band", "y", "trend"]}
                    align="right"
                  />
                )}
              />

              {/* Prediction Band (Confidence) */}
              <Area
                isAnimationActive={false}
                name="Intervalo de Confiança (95%)"
                dataKey="ci_band"
                stroke="none"
                fill={Theme.semantic.warning}
                fillOpacity={0.1}
                hide={!isSeriesVisible("ci_band")}
              />

              {/* Scatter Points (Real Data) */}
              <Scatter
                isAnimationActive={false}
                name="Trials (Reais)"
                dataKey="y"
                fill={Theme.semantic.primary}
                shape="circle"
                hide={!isSeriesVisible("y")}
              />

              {/* Linear Regression Line */}
              <Line
                isAnimationActive={false}
                name={`Tendência (${slope > 0 ? "+" : ""}${slope.toFixed(5)}/trial)`}
                type="monotone"
                dataKey="trend"
                stroke={Theme.semantic.success}
                dot={(d) =>
                  d.isProjection ? { r: 2, fill: Theme.semantic.success, strokeWidth: 0 } : false
                }
                strokeWidth={2}
                strokeDasharray="5 5"
                hide={!isSeriesVisible("trend")}
              />

              {/* Differentiate Projection Zone ?? Maybe a ReferenceLine at maxId */}
              <ReferenceLine
                x={data.filter((d) => !d.isProjection).pop()?.x}
                stroke={Theme.ui.border}
                strokeDasharray="3 3"
                label={{
                  value: "Hoje",
                  position: "insideTopRight",
                  fill: Theme.ui.text.muted,
                  fontSize: 10,
                }}
              />
            </ComposedChart>
          </ChartContainer>
        </WithData>
      </ChartFrame>
    </Card>
  );
};
