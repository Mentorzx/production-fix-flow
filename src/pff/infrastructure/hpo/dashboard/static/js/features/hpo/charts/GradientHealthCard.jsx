/**
 * Provide GradientHealthCard module functionality for the HPO dashboard.
 */

import { useMemo } from "react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Legend, Label } from "recharts";

import { Activity } from "../../../ui/icons.jsx";
import { ChartFrame, ChartContainer, DefaultTooltip, colors } from "../../../ui/ChartPrimitives.jsx";
import { Card } from "../../../ui/Card.jsx";
import { WithData } from "../../../ui/EmptyStates.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { Theme } from "../../../ui/Theme.js";
import { formatCompactTick } from "../../../ui/tickFormatters.js";
import { useSmoothedDomain } from "../../../ui/useSmoothedDomain.js";
import { InteractiveLegend, useLegendVisibility } from "../../../ui/ChartPrimitives.jsx";

/**
 * Expose gradient health card for dashboard usage.
 */
export const GradientHealthCard = ({ liveData }) => {
  const data = useMemo(() => {
    if (!liveData || liveData.length === 0) return [];
    return liveData.map((e, idx) => ({
      epoch: e.epoch || e.id || idx + 1,
      norm: e.grad_norm || e.loss,
    }));
  }, [liveData]);
  const hasData = data.length > 0 && data.some((row) => row.norm != null);
  const yDomain = useSmoothedDomain(
    data.map((row) => row.norm),
    { clampMin: 0, minSpan: 0.05 }
  );
  const { hiddenKeys, toggleSeriesVisibility, isSeriesVisible } = useLegendVisibility(["norm"]);

  return (
    <Card
      title="Saúde do Gradiente"
      icon={Activity}
      className="h-full"
      helpText={ChartRegistry.get("gradient_health")}
    >
      <ChartFrame>
        <WithData
          when={hasData}
          empty="Aguardando grad_norm real..."
          emptyClassName="text-zinc-500"
        >
          <ChartContainer>
            <AreaChart data={data} margin={{ top: 20, right: 16, bottom: 10, left: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={Theme.ui.grid} vertical={false} />
              <XAxis
                dataKey="epoch"
                stroke={colors.text}
                tick={{ fill: Theme.ui.text.muted, fontSize: 10 }}
                tickLine={{ stroke: Theme.ui.border }}
                axisLine={{ stroke: Theme.ui.border }}
                height={50}
              >
                <Label content={<ChartAxisLabel value="Epoch" axis="x" />} />
              </XAxis>
              <YAxis
                stroke={colors.text}
                tick={{ fill: Theme.ui.text.muted, fontSize: 10 }}
                tickFormatter={formatCompactTick}
                tickLine={{ stroke: Theme.ui.border }}
                axisLine={{ stroke: Theme.ui.border }}
                width={72}
                domain={yDomain}
              >
                <Label
                  content={<ChartAxisLabel value="Grad Norm" axis="y" />}
                  position="insideLeft"
                />
              </YAxis>
              <DefaultTooltip />
              <Legend
                layout="horizontal"
                verticalAlign="top"
                align="right"
                height={28}
                iconSize={8}
                wrapperStyle={{
                  top: -8,
                  fontSize: "11px",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                }}
                content={(props) => (
                  <InteractiveLegend
                    {...props}
                    hiddenKeys={hiddenKeys}
                    onToggleSeries={toggleSeriesVisibility}
                    seriesKeys={["norm"]}
                    align="right"
                  />
                )}
              />
              <Area
                isAnimationActive={false}
                name="Grad Norm"
                type="monotone"
                dataKey="norm"
                stroke={Theme.semantic.chart.gradNorm}
                fill={Theme.semantic.chart.gradNorm}
                fillOpacity={0.1}
                strokeWidth={2}
                hide={!isSeriesVisible("norm")}
              />
            </AreaChart>
          </ChartContainer>
        </WithData>
      </ChartFrame>
    </Card>
  );
};
