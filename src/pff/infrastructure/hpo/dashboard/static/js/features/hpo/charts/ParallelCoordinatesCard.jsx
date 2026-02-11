import { useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, Label } from "recharts";

import {
  Card,
  colors,
  DefaultCartesianGrid,
  DefaultTooltip,
  GitMerge,
  ChartFrame,
  ChartContainer,
  WithData,
} from "../../../ui/BaseComponents.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const ParallelCoordinatesCard = ({ trials }) => {
  const normalizedData = useMemo(() => {
    if (!trials || trials.length === 0) return [];
    const numericKeys = new Set();
    trials.forEach((t) => {
      Object.entries(t.params || {}).forEach(([k, v]) => {
        if (typeof v === "number") numericKeys.add(k);
      });
    });
    const keys = Array.from(numericKeys).slice(0, 6);
    const ranges = {};
    keys.forEach((k) => {
      ranges[k] = { min: Infinity, max: -Infinity };
    });
    ranges["value"] = { min: Infinity, max: -Infinity };
    trials.forEach((t) => {
      keys.forEach((k) => {
        const val = t.params[k];
        if (val !== undefined) {
          ranges[k].min = Math.min(ranges[k].min, val);
          ranges[k].max = Math.max(ranges[k].max, val);
        }
      });
      if (t.value !== undefined) {
        ranges["value"].min = Math.min(ranges["value"].min, t.value);
        ranges["value"].max = Math.max(ranges["value"].max, t.value);
      }
    });
    return [...keys, "value"].map((axis) => {
      const row = { name: axis.slice(0, 4).toUpperCase() };
      trials.forEach((t) => {
        let val = axis === "value" ? t.value : t.params[axis];
        if (val === undefined) val = 0;
        const range = ranges[axis];
        let norm = 0.5;
        if (range && range.max > range.min) {
          norm = (val - range.min) / (range.max - range.min);
        }
        row[t.id] = norm;
      });
      return row;
    });
  }, [trials]);

  return (
    <Card
      title="Coordenadas Paralelas"
      icon={GitMerge}
      className="h-full"
      helpText={ChartRegistry.get("parallel")}
    >
      <ChartFrame>
        <WithData when={normalizedData.length > 0} empty="Sem dados" emptyClassName="text-sm">
          <ChartContainer>
            <LineChart data={normalizedData}>
              <DefaultCartesianGrid vertical={true} horizontal={false} />
              <XAxis dataKey="name" stroke={colors.text} height={50}>
                <Label content={<ChartAxisLabel value="Parâmetros" axis="x" />} />
              </XAxis>
              <YAxis domain={[0, 1]} tick={false} tickLine={false} axisLine={false} width={40}>
                <Label content={<ChartAxisLabel value="Valor" axis="y" />} position="insideLeft" />
              </YAxis>
              <DefaultTooltip />
              {trials.slice(-20).map((t) => (
                <Line
                  key={t.id}
                  type="monotone"
                  dataKey={t.id}
                  stroke={t.value > 0.5 ? colors.success : colors.primary}
                  strokeWidth={1}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ChartContainer>
        </WithData>
      </ChartFrame>
    </Card>
  );
};
