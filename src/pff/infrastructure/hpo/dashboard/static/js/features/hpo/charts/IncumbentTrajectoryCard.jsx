/**
 * Provide IncumbentTrajectoryCard module functionality for the HPO dashboard.
 */

import { useId, useMemo } from "react";
import { LineChart, Line, Area, XAxis, YAxis, Tooltip, Label, Legend } from "recharts";
import { Theme } from "../../../ui/Theme.js";

import { Card, TrendingUp, DefaultCartesianGrid, ChartFrame, ChartContainer } from "../../../ui/BaseComponents.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { useStoreState } from "../../../store/store.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { InteractiveLegend, useLegendVisibility } from "../../../ui/ChartPrimitives.jsx";

/**
 * Expose incumbent trajectory card for dashboard usage.
 */
export const IncumbentTrajectoryCard = ({ trials }) => {
  const gradientSuffix = useId().replace(/:/g, "");
  const { data } = useStoreState();
  const direction = data?.direction || "maximize";
  const objectiveStops = useMemo(
    () => [
      { offset: "0%", color: Theme.semantic.primary, opacity: 0.46 },
      { offset: "55%", color: Theme.semantic.primary, opacity: 0.22 },
      { offset: "100%", color: Theme.semantic.primary, opacity: 0.04 },
    ],
    []
  );
  const movingAverageStops = useMemo(
    () => [
      { offset: "0%", color: Theme.semantic.chart.movingAverage, opacity: 0.3 },
      { offset: "100%", color: Theme.semantic.chart.movingAverage, opacity: 0.03 },
    ],
    []
  );
  const incumbentStops = useMemo(
    () => [
      { offset: "0%", color: Theme.semantic.chart.incumbent, opacity: 0.34 },
      { offset: "100%", color: Theme.semantic.chart.incumbent, opacity: 0.03 },
    ],
    []
  );
  const { hiddenKeys, toggleSeriesVisibility, isSeriesVisible } = useLegendVisibility([
    "value",
    "movingAverage",
    "incumbent",
  ]);

  const chartData = useMemo(() => {
    if (!trials || trials.length === 0) return [];
    let currentIncumbent = direction === "minimize" ? Infinity : -Infinity;
    const eligible = [...trials]
      .filter((t) => t.state === "COMPLETE" || t.state === "RUNNING")
      .filter((t) => t.value != null)
      .sort((a, b) => a.id - b.id);

    return eligible.map((t, i, arr) => {
      const val = t.value || 0;
      if (direction === "minimize") {
        if (val < currentIncumbent) currentIncumbent = val;
      } else {
        if (val > currentIncumbent) currentIncumbent = val;
      }
      return {
        id: t.id,
        index: i,
        value: val,
        movingAverage:
          arr.slice(Math.max(0, i - 4), i + 1).reduce((s, c) => s + (c.value || 0), 0) /
          Math.min(i + 1, 5),
        incumbent: currentIncumbent,
      };
    });
  }, [trials, direction]);

  const chartContract = ChartRegistry.get("convergence") || { title: "Evolução" };
  const yDomain = useMemo(() => {
    const values = chartData
      .flatMap((row) => [row.value, row.movingAverage, row.incumbent])
      .filter((v) => Number.isFinite(v));
    if (values.length === 0) return [0, 1];
    let min = Math.min(...values);
    let max = Math.max(...values);
    const span = Math.max(max - min, 0.01);
    const pad = span * 0.08;
    min -= pad;
    max += pad;
    if (max - min < 0.01) max = min + 0.01;
    return [min, max];
  }, [chartData]);

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div
          className="border p-3 rounded-xl shadow-2xl z-50 text-left font-mono"
          style={{
            backgroundColor: Theme.ui.background,
            borderColor: Theme.ui.border,
            color: Theme.ui.text.secondary,
          }}
        >
          <div
            className="text-xs font-black mb-2 pb-1 border-b"
            style={{ borderColor: Theme.ui.border, color: Theme.ui.text.muted }}
          >
            TRIAL #{label}
          </div>
          {payload.map((p, i) => (
            <div key={i} className="flex items-center justify-between gap-4 text-xs mb-1">
              <span style={{ color: p.color }}>{p.name}:</span>
              <span style={{ color: Theme.ui.text.primary }}>{p.value?.toFixed(6)}</span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <Card title={chartContract.title} icon={TrendingUp} className="h-full" helpText={chartContract}>
      <ChartFrame>
        <ChartContainer>
          <LineChart data={chartData} margin={{ top: 20, right: 20, bottom: 50, left: 60 }}>
            <defs>
              <linearGradient id={`grad-objective-${gradientSuffix}`} x1="0" y1="0" x2="0" y2="1">
                {objectiveStops.map((stop, index) => (
                  <stop
                    key={`${gradientSuffix}-objective-${index}`}
                    offset={stop.offset}
                    stopColor={stop.color}
                    stopOpacity={stop.opacity}
                  />
                ))}
              </linearGradient>
              <linearGradient
                id={`grad-moving-average-${gradientSuffix}`}
                x1="0"
                y1="0"
                x2="0"
                y2="1"
              >
                {movingAverageStops.map((stop, index) => (
                  <stop
                    key={`${gradientSuffix}-moving-average-${index}`}
                    offset={stop.offset}
                    stopColor={stop.color}
                    stopOpacity={stop.opacity}
                  />
                ))}
              </linearGradient>
              <linearGradient id={`grad-incumbent-${gradientSuffix}`} x1="0" y1="0" x2="0" y2="1">
                {incumbentStops.map((stop, index) => (
                  <stop
                    key={`${gradientSuffix}-incumbent-${index}`}
                    offset={stop.offset}
                    stopColor={stop.color}
                    stopOpacity={stop.opacity}
                  />
                ))}
              </linearGradient>
            </defs>
            <DefaultCartesianGrid vertical={false} />
            <XAxis
              type="number"
              dataKey="id"
              domain={["dataMin", "dataMax"]}
              tickCount={Math.min(chartData.length, 10)}
              stroke={Theme.ui.text.secondary}
              tick={{ fontSize: 10, fill: Theme.ui.text.secondary }}
              tickFormatter={(v) => `#${v}`}
              height={50}
            >
              <Label content={<ChartAxisLabel value="Trial" axis="x" />} />
            </XAxis>
            <YAxis
              stroke={Theme.ui.text.secondary}
              tick={{ fontSize: 10, fill: Theme.ui.text.secondary }}
              domain={yDomain}
              tickFormatter={(v) => (Number.isFinite(v) ? Number(v).toFixed(3) : v)}
              width={60}
            >
              <Label
                content={<ChartAxisLabel value="Objective" axis="y" />}
                position="insideLeft"
              />
            </YAxis>
            <Tooltip content={<CustomTooltip />} cursor={{ stroke: Theme.ui.grid }} />
            <Legend
              layout="horizontal"
              verticalAlign="top"
              align="right"
              height={28}
              iconSize={8}
              wrapperStyle={{ top: -8, whiteSpace: "nowrap", overflow: "hidden" }}
              content={(props) => (
                <InteractiveLegend
                  {...props}
                  hiddenKeys={hiddenKeys}
                  onToggleSeries={toggleSeriesVisibility}
                  seriesKeys={["value", "movingAverage", "incumbent"]}
                  align="right"
                />
              )}
            />
            <Area
              isAnimationActive={false}
              type="stepAfter"
              dataKey="incumbent"
              stroke="none"
              fill={`url(#grad-incumbent-${gradientSuffix})`}
              fillOpacity={0.52}
              connectNulls={false}
              baseValue={yDomain[0]}
              legendType="none"
              hide={!isSeriesVisible("incumbent")}
            />
            <Area
              isAnimationActive={false}
              type="monotone"
              dataKey="movingAverage"
              stroke="none"
              fill={`url(#grad-moving-average-${gradientSuffix})`}
              fillOpacity={0.48}
              connectNulls={false}
              baseValue={yDomain[0]}
              legendType="none"
              hide={!isSeriesVisible("movingAverage")}
            />
            <Area
              isAnimationActive={false}
              type="monotone"
              dataKey="value"
              stroke="none"
              fill={`url(#grad-objective-${gradientSuffix})`}
              fillOpacity={0.95}
              connectNulls={false}
              baseValue={yDomain[0]}
              legendType="none"
              hide={!isSeriesVisible("value")}
            />
            <Line
              isAnimationActive={false}
              name="Objetivo"
              type="monotone"
              dataKey="value"
              stroke={Theme.semantic.primary}
              strokeWidth={2}
              dot={false}
              connectNulls={false}
              hide={!isSeriesVisible("value")}
            />
            <Line
              isAnimationActive={false}
              name="Média Móvel"
              type="monotone"
              dataKey="movingAverage"
              stroke={Theme.semantic.chart.movingAverage}
              strokeWidth={2}
              dot={false}
              strokeDasharray="4 4"
              hide={!isSeriesVisible("movingAverage")}
            />
            <Line
              isAnimationActive={false}
              name="Melhor (Incumbent)"
              type="stepAfter"
              dataKey="incumbent"
              stroke={Theme.semantic.chart.incumbent}
              strokeWidth={3}
              dot={false}
              hide={!isSeriesVisible("incumbent")}
            />
          </LineChart>
        </ChartContainer>
      </ChartFrame>
    </Card>
  );
};
