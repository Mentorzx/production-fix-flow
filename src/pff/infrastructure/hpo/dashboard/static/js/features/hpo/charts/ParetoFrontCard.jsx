/**
 * Provide ParetoFrontCard module functionality for the HPO dashboard.
 */

import { useMemo } from "react";
import { ScatterChart, Scatter, XAxis, YAxis, Cell, Label } from "recharts";

import { TargetIcon } from "../../../ui/icons.jsx";
import { colors, defaultChartMargins, DefaultCartesianGrid, DefaultTooltip, DefaultTooltipCursor } from "../../../ui/ChartPrimitives.jsx";
import { ChartCard } from "../../../ui/withChartCard.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { formatCompactTick } from "../../../ui/tickFormatters.js";

/**
 * Expose pareto front card for dashboard usage.
 */
export const ParetoFrontCard = ({ trials }) => {
  const data = useMemo(() => {
    if (!trials) return [];
    const completed = trials.filter((t) => t.state === "COMPLETE" && t.value > 0);
    const points = completed.map((t) => ({
      id: t.id,
      x: t.duration || 0,
      y: t.value || 0,
      isPareto: false,
    }));
    points.forEach((p) => {
      if (
        !points.some(
          (other) => (other.x <= p.x && other.y > p.y) || (other.x < p.x && other.y >= p.y)
        )
      )
        p.isPareto = true;
    });
    return points;
  }, [trials]);
  return (
    <ChartCard
      title="Fronteira de Pareto"
      icon={TargetIcon}
      registryKey="pareto_front"
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
          <Label content={<ChartAxisLabel value="Duração (s)" axis="x" />} />
        </XAxis>
        <YAxis type="number" dataKey="y" stroke={colors.text} tickFormatter={formatCompactTick}>
          <Label content={<ChartAxisLabel value="Score" axis="y" />} position="insideLeft" />
        </YAxis>
        <DefaultTooltip cursor={DefaultTooltipCursor} />
        <Scatter isAnimationActive={false} name="Fronteira de Pareto" data={data}>
          {data.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={entry.isPareto ? colors.success : colors.primary}
              fillOpacity={entry.isPareto ? 1 : 0.4}
            />
          ))}
        </Scatter>
      </ScatterChart>
    </ChartCard>
  );
};
