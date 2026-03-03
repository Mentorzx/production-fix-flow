/**
 * Provide TimelinePlotCard module functionality for the HPO dashboard.
 */

import { useMemo } from "react";
import { BarChart, Bar, XAxis, YAxis, Cell, Label } from "recharts";

import { Clock } from "../../../ui/icons.jsx";
import { colors, DefaultCartesianGrid, ChartFrame, ChartContainer } from "../../../ui/ChartPrimitives.jsx";
import { Card } from "../../../ui/Card.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { formatCompactTick } from "../../../ui/tickFormatters.js";

/**
 * Expose timeline plot card for dashboard usage.
 */
export const TimelinePlotCard = ({ trials }) => {
  const data = useMemo(() => {
    if (!trials || trials.length === 0) return [];
    return trials
      .filter((t) => t.duration > 0)
      .map((t) => ({
        id: t.id,
        trialLabel: `#${t.id}`,
        duration: t.duration,
        state: t.state,
      }))
      .slice(-30);
  }, [trials]);
  return (
    <Card
      title="Timeline de Execução"
      icon={Clock}
      className="h-full"
      helpText={ChartRegistry.get("timeline")}
    >
      <ChartFrame>
        <ChartContainer minHeight={0} className="min-h-0">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 8, right: 16, bottom: 22, left: 8 }}
          >
            <DefaultCartesianGrid vertical={false} />
            <XAxis
              type="number"
              stroke={colors.text}
              tick={{ fontSize: 9 }}
              tickMargin={8}
              height={34}
              tickFormatter={formatCompactTick}
            >
              <Label content={<ChartAxisLabel value="Duração (s)" axis="x" />} />
            </XAxis>
            <YAxis
              type="category"
              dataKey="trialLabel"
              stroke={colors.text}
              tick={{ fontSize: 10 }}
              tickMargin={6}
              width={46}
            />
            <Bar
              isAnimationActive={false}
              dataKey="duration"
              fill={colors.primary}
              radius={[0, 4, 4, 0]}
              barSize={16}
            >
              {data.map((e) => (
                <Cell key={e.id} fill={e.state === "COMPLETE" ? colors.success : colors.error} />
              ))}
            </Bar>
          </BarChart>
        </ChartContainer>
      </ChartFrame>
    </Card>
  );
};
