import { useMemo } from "react";
import { BarChart, Bar, XAxis, YAxis, Cell, Label } from "recharts";

import {
  Card,
  Clock,
  colors,
  DefaultCartesianGrid,
  ChartFrame,
  ChartContainer,
} from "../../../ui/BaseComponents.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const TimelinePlotCard = ({ trials }) => {
  const data = useMemo(() => {
    if (!trials || trials.length === 0) return [];
    return trials
      .filter((t) => t.duration > 0)
      .map((t) => ({ id: t.id, duration: t.duration, state: t.state }))
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
        <ChartContainer>
          <BarChart data={data} layout="vertical">
            <DefaultCartesianGrid vertical={false} />
            <XAxis type="number" stroke={colors.text} tick={{ fontSize: 9 }} height={50}>
              <Label content={<ChartAxisLabel value="Duração (s)" axis="x" />} />
            </XAxis>
            <YAxis
              type="category"
              dataKey="id"
              stroke={colors.text}
              tick={{ fontSize: 9 }}
              width={30}
            >
              <Label content={<ChartAxisLabel value="Trial" axis="y" />} position="insideLeft" />
            </YAxis>
            <Bar dataKey="duration" fill={colors.primary} radius={[0, 4, 4, 0]}>
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
