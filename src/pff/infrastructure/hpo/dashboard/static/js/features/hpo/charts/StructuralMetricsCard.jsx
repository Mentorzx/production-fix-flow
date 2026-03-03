/**
 * Provide StructuralMetricsCard module functionality for the HPO dashboard.
 */

import { useMemo } from "react";
import { ScatterChart, Scatter, XAxis, YAxis, Label, Legend } from "recharts";

import { Layers } from "../../../ui/icons.jsx";
import { colors, defaultChartMargins, DefaultCartesianGrid, DefaultTooltip, DefaultTooltipCursor } from "../../../ui/ChartPrimitives.jsx";
import { ChartCard } from "../../../ui/withChartCard.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { formatCompactTick } from "../../../ui/tickFormatters.js";
import { InteractiveLegend, useLegendVisibility } from "../../../ui/ChartPrimitives.jsx";

/**
 * Expose structural metrics card for dashboard usage.
 */
export const StructuralMetricsCard = ({ trials }) => {
  const data = useMemo(() => {
    if (!trials || trials.length === 0) return [];
    return trials
      .filter((t) => t.state === "COMPLETE" && t.value > 0)
      .map((t) => ({ id: t.id, x: t.params?.embedding_dim || 0, y: t.value || 0 }));
  }, [trials]);
  const { hiddenKeys, toggleSeriesVisibility, isSeriesVisible } = useLegendVisibility(["trials"]);
  return (
    <ChartCard
      title="Métricas Estruturais"
      icon={Layers}
      registryKey="structural_metrics"
      hasData={data.length > 0}
    >
      <ScatterChart margin={defaultChartMargins}>
        <DefaultCartesianGrid />
        <XAxis
          type="number"
          dataKey="x"
          stroke={colors.text}
          height={50}
          tickFormatter={formatCompactTick}
        >
          <Label content={<ChartAxisLabel value="Dimensão" axis="x" />} />
        </XAxis>
        <YAxis type="number" dataKey="y" stroke={colors.text} tickFormatter={formatCompactTick}>
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
              color: colors.amber,
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
          fill={colors.amber}
          name="Trials"
          hide={!isSeriesVisible("trials")}
        />
      </ScatterChart>
    </ChartCard>
  );
};
