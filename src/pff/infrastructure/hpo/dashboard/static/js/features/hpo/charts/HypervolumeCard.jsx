import { useMemo } from "react";
import { AreaChart, Area, XAxis, YAxis, Label } from "recharts";

import {
  Card,
  TargetIcon,
  colors,
  DefaultCartesianGrid,
  DefaultTooltip,
  ChartFrame,
  ChartContainer,
} from "../../../ui/BaseComponents.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const HypervolumeCard = ({ trials }) => {
  const data = useMemo(() => {
    if (!trials || trials.length === 0) return [];
    let best = -Infinity;
    return [...trials]
      .sort((a, b) => a.id - b.id)
      .map((t) => {
        best = Math.max(best, t.value || 0);
        return { id: t.id, hv: best };
      });
  }, [trials]);

  return (
    <Card
      title="Hypervolume"
      icon={TargetIcon}
      className="h-full"
      helpText={ChartRegistry.get("hypervolume")}
    >
      <ChartFrame>
        <ChartContainer>
          <AreaChart data={data}>
            <DefaultCartesianGrid />
            <XAxis dataKey="id" stroke={colors.text} height={50}>
              <Label content={<ChartAxisLabel value="Trial" axis="x" />} />
            </XAxis>
            <YAxis stroke={colors.text} domain={["auto", "auto"]}>
              <Label
                content={<ChartAxisLabel value="Hypervolume" axis="y" />}
                position="insideLeft"
              />
            </YAxis>
            <DefaultTooltip />
            <Area
              type="stepAfter"
              dataKey="hv"
              stroke={colors.success}
              fill={colors.success}
              fillOpacity={0.2}
              strokeWidth={2}
            />
          </AreaChart>
        </ChartContainer>
      </ChartFrame>
    </Card>
  );
};
