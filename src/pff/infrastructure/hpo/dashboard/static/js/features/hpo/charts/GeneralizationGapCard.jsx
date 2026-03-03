/**
 * Provide GeneralizationGapCard module functionality for the HPO dashboard.
 */

import { useMemo } from "react";
import { ComposedChart, Line, Bar, XAxis, YAxis, Legend, ReferenceLine, Label } from "recharts";
import { Theme } from "../../../ui/Theme.js";

import { Activity } from "../../../ui/icons.jsx";
import { DefaultCartesianGrid, DefaultTooltip, ChartFrame, ChartContainer } from "../../../ui/ChartPrimitives.jsx";
import { Card } from "../../../ui/Card.jsx";
import { WithData } from "../../../ui/EmptyStates.jsx";
import { ChartAxisLabel } from "../../../ui/UIComponents.jsx";
import { useSmoothedDomain } from "../../../ui/useSmoothedDomain.js";
import { InteractiveLegend, useLegendVisibility } from "../../../ui/ChartPrimitives.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

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
const formatMetricTick = (v) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  const fixed = n.toFixed(3);
  return fixed.replace(/\.?0+$/, "");
};

/**
 * Expose generalization gap card for dashboard usage.
 */
export const GeneralizationGapCard = ({ liveData }) => {
  const data = useMemo(() => {
    const rows = Array.isArray(liveData) ? liveData : [];
    if (rows.length === 0) return [];

    return rows
      .map((e, idx) => {
        if (!e || typeof e !== "object") return null;
        const epoch = typeof e.epoch === "number" ? e.epoch : idx + 1;

        const payload = e.metrics && typeof e.metrics === "object" ? e.metrics : e;
        const trainLoss = parseLoss(payload.train_loss ?? payload.loss ?? payload.binary_loss);
        const valLoss = parseLoss(payload.val_loss ?? payload.validation_loss ?? payload.eval_loss);
        const metric = parseValue(payload.mcc ?? payload.mrr);
        const gap = trainLoss !== null && valLoss !== null ? valLoss - trainLoss : null;

        return {
          epoch,
          train_loss: trainLoss,
          val_loss: valLoss,
          metric,
          gap,
        };
      })
      .filter(Boolean);
  }, [liveData]);

  const hasData =
    data.length > 0 &&
    data.some((row) => row.train_loss != null || row.val_loss != null || row.gap != null);
  const lossDomain = useSmoothedDomain(
    data.flatMap((row) => [row.train_loss, row.val_loss, row.gap]),
    { minSpan: 0.05 }
  );
  const metricDomain = useSmoothedDomain(
    data.map((row) => row.metric),
    { minSpan: 0.01 }
  );
  const { hiddenKeys, toggleSeriesVisibility, isSeriesVisible } = useLegendVisibility([
    "gap",
    "train_loss",
    "val_loss",
    "metric",
  ]);

  return (
    <Card
      title="Gap de Generalização"
      icon={Activity}
      className="h-full"
      helpText={ChartRegistry.get("generalization_gap")}
    >
      <ChartFrame className="p-3">
        <WithData
          when={hasData}
          empty="Aguardando dados de convergência..."
          emptyClassName="text-zinc-500"
        >
          <ChartContainer minHeight={0} className="min-h-0">
            <ComposedChart data={data} margin={{ top: 28, right: 76, bottom: 24, left: 46 }}>
              <DefaultCartesianGrid />
              <XAxis dataKey="epoch" stroke={Theme.ui.text.secondary} height={38} tickMargin={8}>
                <Label
                  value="Epoch"
                  position="insideBottom"
                  offset={-8}
                  fill={Theme.ui.text.secondary}
                  fontSize={12}
                />
              </XAxis>

              <YAxis
                yAxisId="loss"
                stroke={Theme.semantic.chart.loss}
                tick={{ fill: Theme.ui.text.secondary }}
                domain={lossDomain}
                width={72}
                tickMargin={8}
                tickFormatter={formatLossTick}
              >
                <Label content={<ChartAxisLabel value="Loss / Gap" axis="y" offset={14} />} />
              </YAxis>

              <YAxis
                yAxisId="metric"
                orientation="right"
                stroke={Theme.semantic.chart.metric}
                tick={{ fill: Theme.ui.text.secondary }}
                domain={metricDomain}
                width={72}
                tickMargin={8}
                tickFormatter={formatMetricTick}
              >
                <Label content={<ChartAxisLabel value="MCC/MRR" axis="y-right" offset={14} />} />
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
                content={(props) => (
                  <InteractiveLegend
                    {...props}
                    hiddenKeys={hiddenKeys}
                    onToggleSeries={toggleSeriesVisibility}
                    seriesKeys={["gap", "train_loss", "val_loss", "metric"]}
                    align="right"
                  />
                )}
              />
              <ReferenceLine y={0} yAxisId="loss" stroke={Theme.ui.border} strokeDasharray="3 3" />

              <Bar
                isAnimationActive={false}
                yAxisId="loss"
                dataKey="gap"
                name="Gap de Generalização"
                fill={Theme.palette.vividGreen}
                barSize={6}
                fillOpacity={0.45}
                hide={!isSeriesVisible("gap")}
              />

              <Line
                isAnimationActive={false}
                type="monotone"
                yAxisId="loss"
                dataKey="train_loss"
                name="Loss de Treino"
                stroke={Theme.semantic.chart.loss}
                strokeWidth={2}
                dot={false}
                connectNulls
                hide={!isSeriesVisible("train_loss")}
              />
              <Line
                isAnimationActive={false}
                type="monotone"
                yAxisId="loss"
                dataKey="val_loss"
                name="Loss de Validação"
                stroke={Theme.semantic.error}
                strokeWidth={2}
                dot={false}
                strokeDasharray="4 4"
                connectNulls
                hide={!isSeriesVisible("val_loss")}
              />

              <Line
                isAnimationActive={false}
                type="monotone"
                yAxisId="metric"
                dataKey="metric"
                name="MCC/MRR"
                stroke={Theme.semantic.chart.metric}
                dot={false}
                strokeWidth={2}
                connectNulls
                hide={!isSeriesVisible("metric")}
              />
            </ComposedChart>
          </ChartContainer>
        </WithData>
      </ChartFrame>
    </Card>
  );
};
