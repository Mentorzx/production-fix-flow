/**
 * Provide TrialLearningMetricsCard module functionality for the HPO dashboard.
 */

import { useCallback, useMemo } from "react";
import { ComposedChart, Area, Line, XAxis, YAxis, Legend, Label } from "recharts";
import { Theme } from "../../../ui/Theme.js";

import { TrendingUp } from "../../../ui/icons.jsx";
import { DefaultCartesianGrid, DefaultTooltip, ChartFrame, ChartContainer } from "../../../ui/ChartPrimitives.jsx";
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

const hasValidationSignals = (payload) =>
  parseValue(payload?.mrr) != null ||
  parseValue(payload?.mcc) != null ||
  parseValue(payload?.auc) != null ||
  parseValue(payload?.pr_auc) != null ||
  parseValue(payload?.accuracy) != null;

const isEvaluationEpoch = (payload) =>
  hasValidationSignals(payload) ||
  parseValue(payload?.val_loss) != null ||
  parseValue(payload?.validation_loss) != null ||
  parseValue(payload?.eval_loss) != null ||
  parseValue(payload?.test_loss) != null;

/**
 * Expose trial learning metrics card for dashboard usage.
 */
export const TrialLearningMetricsCard = ({ liveData }) => {
  const data = useMemo(() => {
    const rows = Array.isArray(liveData) ? liveData : [];
    return rows
      .map((e, idx) => {
        if (!e || typeof e !== "object") return null;
        const payload = e.metrics && typeof e.metrics === "object" ? e.metrics : e;
        if (!isEvaluationEpoch(payload)) return null;
        const epoch = typeof e.epoch === "number" ? e.epoch : idx + 1;
        return {
          epoch,
          loss: parseLoss(payload.loss ?? payload.train_loss ?? payload.binary_loss),
          val_loss: parseLoss(payload.val_loss ?? payload.validation_loss ?? payload.eval_loss),
          mrr: parseValue(payload.mrr),
          mcc: parseValue(payload.mcc),
        };
      })
      .filter(Boolean);
  }, [liveData]);

  const hasData =
    data.length > 0 &&
    data.some((d) => d.loss != null || d.val_loss != null || d.mrr != null || d.mcc != null);
  const lossDomain = useSmoothedDomain(
    data.flatMap((d) => [d.loss, d.val_loss]),
    { clampMin: 0, minSpan: 0.05 }
  );
  const { hiddenKeys, toggleSeriesVisibility, isSeriesVisible } = useLegendVisibility([
    "loss",
    "val_loss",
    "mrr",
    "mcc",
  ]);
  const legendSeriesKeys = useMemo(() => ["loss", "val_loss", "mrr", "mcc"], []);
  const legendSeriesAliases = useMemo(
    () => ({
      loss: ["loss", "train loss", "perda treino", "perda de treino"],
      val_loss: ["val loss", "validation loss", "loss validação", "perda validação"],
      mrr: ["mrr", "mean reciprocal rank"],
      mcc: ["mcc", "matthews", "coeficiente de matthews"],
    }),
    []
  );
  const legendSeriesLabels = useMemo(
    () => ({
      loss: "LOSS",
      val_loss: "VAL LOSS",
      mrr: "MRR",
      mcc: "MCC",
    }),
    []
  );
  const renderLegend = useCallback(
    (props) => (
      <InteractiveLegend
        {...props}
        hiddenKeys={hiddenKeys}
        onToggleSeries={toggleSeriesVisibility}
        seriesKeys={legendSeriesKeys}
        seriesAliases={legendSeriesAliases}
        seriesLabels={legendSeriesLabels}
        align="right"
      />
    ),
    [hiddenKeys, toggleSeriesVisibility, legendSeriesAliases, legendSeriesKeys, legendSeriesLabels]
  );

  const helpText = ChartRegistry.get("trial_learning_metrics");

  return (
    <Card title="Loss + MCC/MRR" icon={TrendingUp} className="h-full" helpText={helpText}>
      <ChartFrame>
        <WithData
          when={hasData}
          empty="Aguardando histórico do trial..."
          emptyClassName="text-zinc-500"
        >
          <ChartContainer>
            <ComposedChart data={data} margin={{ top: 20, right: 60, bottom: 50, left: 60 }}>
              <defs>
                <linearGradient id="gradLoss" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={Theme.semantic.chart.loss} stopOpacity={0.24} />
                  <stop offset="100%" stopColor={Theme.semantic.chart.loss} stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="gradMrr" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={Theme.palette.neonBlue} stopOpacity={0.24} />
                  <stop offset="100%" stopColor={Theme.palette.neonBlue} stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="gradMcc" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={Theme.palette.vividGreen} stopOpacity={0.24} />
                  <stop offset="100%" stopColor={Theme.palette.vividGreen} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <DefaultCartesianGrid />
              <XAxis dataKey="epoch" stroke={Theme.ui.text.secondary} height={50}>
                <Label content={<ChartAxisLabel value="Epoch" axis="x" />} />
              </XAxis>
              <YAxis
                yAxisId="loss"
                stroke={Theme.semantic.chart.loss}
                tick={{ fill: Theme.ui.text.secondary }}
                domain={lossDomain}
                width={60}
                tickFormatter={formatLossTick}
              >
                <Label content={<ChartAxisLabel value="Loss" axis="y" />} position="insideLeft" />
              </YAxis>
              <YAxis
                yAxisId="metric"
                orientation="right"
                stroke={Theme.palette.neonBlue}
                tick={{ fill: Theme.ui.text.secondary }}
                domain={[0, 1]}
                width={60}
              >
                <Label
                  content={<ChartAxisLabel value="Metrics" axis="y-right" />}
                  position="insideRight"
                />
              </YAxis>
              <DefaultTooltip />
              <Legend
                layout="horizontal"
                verticalAlign="top"
                align="right"
                height={28}
                iconSize={8}
                wrapperStyle={{ top: -8, whiteSpace: "nowrap", overflow: "hidden" }}
                content={renderLegend}
              />

              <Area
                isAnimationActive={false}
                type="monotone"
                yAxisId="loss"
                dataKey="loss"
                name="LOSS"
                stroke={Theme.semantic.chart.loss}
                fill="url(#gradLoss)"
                strokeWidth={2}
                connectNulls
                hide={!isSeriesVisible("loss")}
              />
              <Line
                isAnimationActive={false}
                type="monotone"
                yAxisId="loss"
                dataKey="val_loss"
                name="VAL LOSS"
                stroke={Theme.semantic.error}
                strokeWidth={2}
                dot={false}
                connectNulls
                strokeDasharray="4 4"
                hide={!isSeriesVisible("val_loss")}
              />
              <Area
                isAnimationActive={false}
                type="monotone"
                yAxisId="metric"
                dataKey="mrr"
                name="MRR"
                stroke={Theme.palette.neonBlue}
                fill="url(#gradMrr)"
                strokeWidth={2}
                connectNulls
                hide={!isSeriesVisible("mrr")}
              />
              <Area
                isAnimationActive={false}
                type="monotone"
                yAxisId="metric"
                dataKey="mcc"
                name="MCC"
                stroke={Theme.palette.vividGreen}
                fill="url(#gradMcc)"
                strokeWidth={2}
                connectNulls
                hide={!isSeriesVisible("mcc")}
              />
            </ComposedChart>
          </ChartContainer>
        </WithData>
      </ChartFrame>
    </Card>
  );
};
