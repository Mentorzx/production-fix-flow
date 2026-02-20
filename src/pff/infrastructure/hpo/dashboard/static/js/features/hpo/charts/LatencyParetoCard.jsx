/**
 * Provide LatencyParetoCard module functionality for the HPO dashboard.
 */

import { useMemo } from "react";
import { ScatterChart, Scatter, XAxis, YAxis, Label, Legend } from "recharts";

import {
  Card,
  TargetIcon,
  colors,
  defaultChartMargins,
  DefaultCartesianGrid,
  DefaultTooltip,
  DefaultTooltipCursor,
  ChartFrame,
  ChartContainer,
  WithData,
} from "../../../ui/BaseComponents.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { formatCompactTick } from "../../../ui/tickFormatters.js";
import { InteractiveLegend, useLegendVisibility } from "../../../ui/ChartPrimitives.jsx";

/**
 * Expose latency pareto card for dashboard usage.
 */
export const LatencyParetoCard = ({ trials }) => {
  const data = useMemo(() => {
    if (!trials) return [];
    return trials
      .filter((t) => t.state === "COMPLETE")
      .map((t) => ({
        id: t.id,
        x: t.metrics?.inference_latency || t.duration || 0,
        y: t.value || 0,
      }));
  }, [trials]);
  const { hiddenKeys, toggleSeriesVisibility, isSeriesVisible } = useLegendVisibility(["trials"]);
  return (
    <Card
      title="Latência x Qualidade"
      className="h-full"
      icon={TargetIcon}
      helpText={ChartRegistry.get("latency_pareto")}
    >
      <ChartFrame>
        <WithData when={data.length > 0} empty="Sem dados">
          <ChartContainer>
            <ScatterChart margin={defaultChartMargins}>
              <DefaultCartesianGrid />
              <XAxis
                type="number"
                dataKey="x"
                stroke={colors.text}
                height={50}
                tickFormatter={formatCompactTick}
              >
                <Label content={<ChartAxisLabel value="Latência (ms)" axis="x" />} />
              </XAxis>
              <YAxis
                type="number"
                dataKey="y"
                stroke={colors.text}
                tickFormatter={formatCompactTick}
              >
                <Label content={<ChartAxisLabel value="Score" axis="y" />} position="insideLeft" />
              </YAxis>
              <DefaultTooltip cursor={DefaultTooltipCursor} />
              <Legend
                layout="horizontal"
                verticalAlign="top"
                align="right"
                height={28}
                iconSize={8}
                wrapperStyle={{ top: -10, whiteSpace: "nowrap", overflow: "hidden" }}
                payload={[
                  {
                    value: "Trials",
                    type: "circle",
                    id: "trials",
                    color: colors.orange,
                    dataKey: "trials",
                  },
                ]}
                content={(props) => (
                  <InteractiveLegend
                    {...props}
                    hiddenKeys={hiddenKeys}
                    onToggleSeries={toggleSeriesVisibility}
                    seriesKeys={["trials"]}
                    align="right"
                  />
                )}
              />
              <Scatter
                isAnimationActive={false}
                data={data}
                fill={colors.orange}
                name="Trials"
                hide={!isSeriesVisible("trials")}
              />
            </ScatterChart>
          </ChartContainer>
        </WithData>
      </ChartFrame>
    </Card>
  );
};
