/**
 * Provide ScatterPlotCard module functionality for the HPO dashboard.
 */

import { ScatterChart, Scatter, XAxis, YAxis, Label } from "recharts";

import { Search } from "../../../ui/icons.jsx";
import { colors, defaultChartMargins, DefaultCartesianGrid, DefaultTooltip, DefaultTooltipCursor, ChartFrame, ChartContainer } from "../../../ui/ChartPrimitives.jsx";
import { Card } from "../../../ui/Card.jsx";
import { WithData } from "../../../ui/EmptyStates.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

/**
 * Expose scatter plot card for dashboard usage.
 */
export const ScatterPlotCard = ({ title, data, xLabel, yLabel, action }) => {
  return (
    <Card
      title={title}
      icon={Search}
      className="h-full"
      action={action}
      helpText={ChartRegistry.get("scatter_plot")}
    >
      <ChartFrame>
        <WithData when={data && data.length > 0} empty="Aguardando dados...">
          <ChartContainer>
            <ScatterChart margin={defaultChartMargins}>
              <DefaultCartesianGrid />
              <XAxis type="number" dataKey="x" stroke={colors.text} height={50}>
                <Label content={<ChartAxisLabel value={xLabel} axis="x" />} />
              </XAxis>
              <YAxis type="number" dataKey="y" stroke={colors.text}>
                <Label content={<ChartAxisLabel value={yLabel} axis="y" />} position="insideLeft" />
              </YAxis>
              <DefaultTooltip cursor={DefaultTooltipCursor} />
              <Scatter isAnimationActive={false} data={data} fill={colors.primary} />
            </ScatterChart>
          </ChartContainer>
        </WithData>
      </ChartFrame>
    </Card>
  );
};
