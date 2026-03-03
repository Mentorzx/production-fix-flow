/**
 * Provide LearningCurveChart module functionality for the HPO dashboard.
 */

import { useMemo } from "react";
import { ComposedChart, Area, Line, XAxis, YAxis, Legend, Label } from "recharts";

import { TrendingUp } from "../../../ui/icons.jsx";
import { colors, DefaultCartesianGrid, DefaultTooltip, ChartContainer, ChartFrame } from "../../../ui/ChartPrimitives.jsx";
import { Card } from "../../../ui/Card.jsx";
import { WithData } from "../../../ui/EmptyStates.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { useSmoothedDomain } from "../../../ui/useSmoothedDomain.js";
import { InteractiveLegend, useLegendVisibility } from "../../../ui/ChartPrimitives.jsx";

const parseValue = (v) => {
  if (v === null || v === undefined) return null;
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : null;
};
const MAX_ABS_LOSS = 1e4;
const parseLoss = (v) => {
  const n = parseValue(v);
  if (n == null) return null;
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
 * Expose learning curve chart for dashboard usage.
 */
export const LearningCurveChart = ({ data }) => {
  const rows = useMemo(() => {
    const items = Array.isArray(data) ? data : [];
    return items
      .map((row, idx) => {
        if (!row || typeof row !== "object") return null;

        const epoch =
          typeof row.epoch === "number" ? row.epoch : typeof row.id === "number" ? row.id : idx + 1;

        const metrics = row.metrics && typeof row.metrics === "object" ? row.metrics : null;
        return {
          epoch,
          train_loss: parseLoss(row.train_loss ?? row.loss ?? metrics?.train_loss ?? metrics?.loss),
          val_loss: parseLoss(
            row.val_loss ??
              row.validation_loss ??
              metrics?.val_loss ??
              metrics?.validation_loss ??
              row.eval_loss ??
              metrics?.eval_loss
          ),
        };
      })
      .filter(Boolean);
  }, [data]);

  const hasData =
    rows.length > 0 && rows.some((row) => row.train_loss != null || row.val_loss != null);
  const yDomain = useSmoothedDomain(
    rows.flatMap((row) => [row.train_loss, row.val_loss]),
    { clampMin: 0, minSpan: 0.05 }
  );
  const { hiddenKeys, toggleSeriesVisibility, isSeriesVisible } = useLegendVisibility([
    "train_loss",
    "val_loss",
  ]);

  return (
    <Card
      title="Curvas de Aprendizado"
      icon={TrendingUp}
      className="h-full"
      helpText={ChartRegistry.get("learning_curve")}
    >
      <ChartFrame>
        <WithData
          when={hasData}
          empty="Aguardando curvas de treino..."
          emptyClassName="text-zinc-500"
        >
          <ChartContainer>
            <ComposedChart data={rows} margin={{ top: 20, right: 16, bottom: 10, left: 40 }}>
              <defs>
                <linearGradient id="gradTrainLoss" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={colors.primary} stopOpacity={0.24} />
                  <stop offset="100%" stopColor={colors.primary} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <DefaultCartesianGrid />
              <XAxis dataKey="epoch" stroke={colors.text} height={50}>
                <Label content={<ChartAxisLabel value="Epoch" axis="x" />} />
              </XAxis>
              <YAxis stroke={colors.text} domain={yDomain} tickFormatter={formatLossTick}>
                <Label content={<ChartAxisLabel value="Loss" axis="y" />} position="insideLeft" />
              </YAxis>
              <DefaultTooltip />
              <Legend
                layout="horizontal"
                verticalAlign="top"
                align="right"
                height={28}
                iconSize={8}
                wrapperStyle={{ top: -8, whiteSpace: "nowrap", overflow: "hidden" }}
                content={(props) => (
                  <InteractiveLegend
                    {...props}
                    hiddenKeys={hiddenKeys}
                    onToggleSeries={toggleSeriesVisibility}
                    seriesKeys={["train_loss", "val_loss"]}
                    align="right"
                  />
                )}
              />
              <Area
                isAnimationActive={false}
                type="monotone"
                dataKey="train_loss"
                name="Train Loss"
                stroke={colors.primary}
                fill="url(#gradTrainLoss)"
                strokeWidth={2}
                dot={false}
                connectNulls
                hide={!isSeriesVisible("train_loss")}
              />
              <Line
                isAnimationActive={false}
                type="monotone"
                dataKey="val_loss"
                name="Val Loss"
                stroke={colors.error}
                dot={false}
                connectNulls
                strokeDasharray="4 4"
                hide={!isSeriesVisible("val_loss")}
              />
            </ComposedChart>
          </ChartContainer>
        </WithData>
      </ChartFrame>
    </Card>
  );
};
