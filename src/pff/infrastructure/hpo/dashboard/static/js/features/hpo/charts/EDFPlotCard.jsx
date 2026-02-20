/**
 * Provide EDFPlotCard module functionality for the HPO dashboard.
 */

import { useId, useMemo } from "react";
import { LineChart, Line, Area, XAxis, YAxis, Label } from "recharts";

import {
  TrendingUp,
  colors,
  DefaultCartesianGrid,
  DefaultTooltip,
} from "../../../ui/BaseComponents.jsx";
import { ChartCard } from "../../../ui/withChartCard.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";

/**
 * Expose edfplot card for dashboard usage.
 */
export const EDFPlotCard = ({ filteredTrials }) => {
  const gradientSuffix = useId().replace(/:/g, "");
  const data = useMemo(() => {
    if (!filteredTrials) return [];
    const values = filteredTrials
      .filter((t) => t.state === "COMPLETE" && t.value != null)
      .map((t) => t.value)
      .sort((a, b) => a - b);
    return values.map((val, i) => ({ x: val, y: (i + 1) / values.length }));
  }, [filteredTrials]);
  return (
    <ChartCard
      title="EDF Plot"
      icon={TrendingUp}
      registryKey="edf"
      hasData={data.length > 1}
      emptyText="Dados insuficientes"
      emptyClassName="text-sm"
      chartFrameClassName="pt-8"
    >
      <LineChart data={data}>
        <defs>
          <linearGradient id={`grad-edf-${gradientSuffix}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={colors.success} stopOpacity={0.24} />
            <stop offset="100%" stopColor={colors.success} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <DefaultCartesianGrid vertical={false} />
        <XAxis type="number" dataKey="x" stroke={colors.text} height={50}>
          <Label content={<ChartAxisLabel value="Objetivo" axis="x" />} />
        </XAxis>
        <YAxis type="number" dataKey="y" stroke={colors.text}>
          <Label content={<ChartAxisLabel value="EDF" axis="y" />} position="insideLeft" />
        </YAxis>
        <DefaultTooltip />
        <Area
          isAnimationActive={false}
          type="stepAfter"
          dataKey="y"
          stroke="none"
          fill={`url(#grad-edf-${gradientSuffix})`}
          fillOpacity={1}
          baseValue="dataMin"
          legendType="none"
        />
        <Line
          isAnimationActive={false}
          type="stepAfter"
          dataKey="y"
          stroke={colors.success}
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ChartCard>
  );
};
