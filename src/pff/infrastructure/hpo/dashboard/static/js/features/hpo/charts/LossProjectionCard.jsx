/**
 * Provide LossProjectionCard module functionality for the HPO dashboard.
 */

import { useId, useMemo } from "react";
import { LineChart, Line, Area, XAxis, YAxis, Legend, Label } from "recharts";

import {
  Card,
  TrendingUp,
  ChartFrame,
  ChartContainer,
  WithData,
  DefaultCartesianGrid,
  DefaultTooltip,
  colors,
  getChartAreaGradientStops,
} from "../../../ui/BaseComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { useSmoothedDomain } from "../../../ui/useSmoothedDomain.js";
import { InteractiveLegend, useLegendVisibility } from "../../../ui/ChartPrimitives.jsx";

const MAX_ABS_LOSS = 1e4;
const parseLoss = (v) => {
  const n = Number.parseFloat(v);
  if (!Number.isFinite(n)) return null;
  if (Math.abs(n) > MAX_ABS_LOSS) return null;
  return n;
};
const formatLossTick = (v) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 1000) return n.toExponential(1);
  const fixed = n.toFixed(3);
  return fixed.replace(/\.?0+$/, "");
};

/**
 * Expose loss projection card for dashboard usage.
 */
export const LossProjectionCard = ({ liveData, targetEpoch = null }) => {
  const gradientSuffix = useId().replace(/:/g, "");
  const observedStops = useMemo(
    () => getChartAreaGradientStops("primaryReadable", colors.primary),
    []
  );
  const projectedStops = useMemo(
    () => getChartAreaGradientStops("primarySubtle", colors.primary),
    []
  );
  const data = useMemo(() => {
    const rows = Array.isArray(liveData) ? liveData : [];
    if (rows.length === 0) return [];

    const base = rows
      .map((e, idx) => {
        if (!e || typeof e !== "object") return null;
        const epoch = typeof e.epoch === "number" ? e.epoch : idx + 1;
        const rawLoss = e.loss ?? e.train_loss ?? e.val_loss ?? e.binary_loss;
        const loss = parseLoss(rawLoss);
        return loss != null ? { epoch, loss } : null;
      })
      .filter(Boolean);

    if (base.length < 2) return [];
    const last = base[base.length - 1];
    const backendEpoch = Number(targetEpoch);
    const target =
      Number.isFinite(backendEpoch) && backendEpoch > 0
        ? Math.max(Math.floor(backendEpoch), last.epoch)
        : last.epoch + 10;
    const projectionSteps = Math.max(0, target - last.epoch);
    const projection = Array.from({ length: projectionSteps }, (_, i) => ({
      epoch: last.epoch + i + 1,
      projected_loss: last.loss * Math.pow(0.95, i + 1),
    }));

    const observed = base.map((row, idx) => ({
      epoch: row.epoch,
      observed_loss: row.loss,
      projected_loss: idx === base.length - 1 ? row.loss : null,
    }));

    return [...observed, ...projection];
  }, [liveData, targetEpoch]);
  const yDomain = useSmoothedDomain(
    data.flatMap((d) => [d.observed_loss, d.projected_loss]),
    { clampMin: 0, minSpan: 0.05 }
  );
  const { hiddenKeys, toggleSeriesVisibility, isSeriesVisible } = useLegendVisibility([
    "observed_loss",
    "projected_loss",
  ]);

  return (
    <Card
      title="Extrapolação de Perda"
      icon={TrendingUp}
      className="h-full"
      helpText={ChartRegistry.get("loss_projection")}
    >
      <ChartFrame className="p-2 h-full min-h-[200px]">
        <WithData when={data.length > 0} empty="Aguardando...">
          <ChartContainer minHeight={120} className="h-full min-h-[200px]">
            <LineChart data={data} margin={{ top: 28, right: 22, bottom: 24, left: 44 }}>
              <defs>
                <linearGradient
                  id={`grad-loss-observed-${gradientSuffix}`}
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="1"
                >
                  {observedStops.map((stop, index) => (
                    <stop
                      key={`${gradientSuffix}-observed-${index}`}
                      offset={stop.offset}
                      stopColor={stop.color}
                      stopOpacity={stop.opacity}
                    />
                  ))}
                </linearGradient>
                <linearGradient
                  id={`grad-loss-projected-${gradientSuffix}`}
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="1"
                >
                  {projectedStops.map((stop, index) => (
                    <stop
                      key={`${gradientSuffix}-projected-${index}`}
                      offset={stop.offset}
                      stopColor={stop.color}
                      stopOpacity={stop.opacity}
                    />
                  ))}
                </linearGradient>
              </defs>
              <DefaultCartesianGrid />
              <XAxis dataKey="epoch" height={36} tick={{ fill: colors.text }} tickMargin={8}>
                <Label
                  value="Epoch"
                  position="insideBottom"
                  offset={-8}
                  fill={colors.text}
                  fontSize={12}
                />
              </XAxis>
              <YAxis
                tick={{ fill: colors.text }}
                width={70}
                tickMargin={8}
                domain={yDomain}
                tickFormatter={formatLossTick}
              >
                <Label content={<ChartAxisLabel value="Loss" axis="y" offset={12} />} />
              </YAxis>
              <DefaultTooltip
                formatter={(value) => (Number.isFinite(value) ? Number(value).toFixed(4) : value)}
                labelFormatter={(label) => `Epoch ${label}`}
              />
              <Legend
                layout="horizontal"
                verticalAlign="top"
                align="right"
                height={28}
                iconSize={8}
                wrapperStyle={{ top: -4, whiteSpace: "nowrap", overflow: "hidden" }}
                payload={[
                  {
                    value: "Loss observada",
                    type: "line",
                    id: "observed_loss",
                    color: colors.primary,
                    dataKey: "observed_loss",
                  },
                  {
                    value: "Loss projetada",
                    type: "line",
                    id: "projected_loss",
                    color: colors.primary,
                    dataKey: "projected_loss",
                  },
                ]}
                content={(props) => (
                  <InteractiveLegend
                    {...props}
                    hiddenKeys={hiddenKeys}
                    onToggleSeries={toggleSeriesVisibility}
                    seriesKeys={["observed_loss", "projected_loss"]}
                    align="right"
                  />
                )}
              />
              <Area
                isAnimationActive={false}
                type="monotone"
                dataKey="observed_loss"
                stroke="none"
                fill={`url(#grad-loss-observed-${gradientSuffix})`}
                fillOpacity={1}
                connectNulls
                baseValue="dataMin"
                legendType="none"
                hide={!isSeriesVisible("observed_loss")}
              />
              <Area
                isAnimationActive={false}
                type="monotone"
                dataKey="projected_loss"
                stroke="none"
                fill={`url(#grad-loss-projected-${gradientSuffix})`}
                fillOpacity={1}
                connectNulls
                baseValue="dataMin"
                legendType="none"
                hide={!isSeriesVisible("projected_loss")}
              />
              <Line
                isAnimationActive={false}
                type="monotone"
                dataKey="observed_loss"
                name="Loss observada"
                stroke={colors.primary}
                strokeWidth={2}
                dot={false}
                connectNulls
                hide={!isSeriesVisible("observed_loss")}
              />
              <Line
                isAnimationActive={false}
                type="monotone"
                dataKey="projected_loss"
                name="Loss projetada"
                stroke={colors.primary}
                strokeWidth={2}
                dot={false}
                strokeDasharray="5 5"
                connectNulls
                hide={!isSeriesVisible("projected_loss")}
              />
            </LineChart>
          </ChartContainer>
        </WithData>
      </ChartFrame>
    </Card>
  );
};
