/**
 * Provide KpiRow module functionality for the HPO dashboard.
 */

import { TrialStatusCard } from "../features/hpo/charts/TrialStatusCard.jsx";
import { StatBadge } from "../ui/BaseComponents.jsx";
import { MetricRegistry } from "../domain/metrics/MetricRegistry.js";
import { useKpiMetrics, formatCompactDuration } from "../hooks/useKpiMetrics.js";
import { AnimatedNumberText } from "../ui/AnimatedNumberText.jsx";

const DurationValueNode = ({ parts, animationSeed = "" }) => {
  if (!parts) return "—";
  return (
    <div className="flex flex-col leading-none">
      <AnimatedNumberText
        value={parts.main}
        seed={animationSeed}
        className="text-4xl font-black font-mono tracking-tighter tabular-nums"
      />
      <div className="text-[10px] font-black font-mono uppercase tracking-[0.3em] opacity-40">
        {parts.unit}
      </div>
    </div>
  );
};

/**
 * Expose kpi row for dashboard usage.
 */
export const KpiRow = () => {
  const {
    viewMode,
    activeTab,
    data,
    trials,
    bestTrialNoWarmstart,
    objectiveDirection,
    bestSeries,
    bestDeltaPct,
    lastDurations,
    avgDuration,
    avgDurationDeltaPct,
    avgDurationParts,
    estimatedCompletion,
    etaDeltaPct,
    etaParts,
    latestTrialMetrics,
    liveMetricSeries,
    deltaFromSeries,
  } = useKpiMetrics();
  const animationSeed = `scope:${viewMode}|tab:${activeTab}`;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6" data-jackpot-force="true">
      <div className="h-44">
        <TrialStatusCard data={data} trials={trials} animationSeed={animationSeed} />
      </div>

      {viewMode === "study" ? (
        <>
          <div className="h-44">
            <StatBadge
              label="Melhor Global"
              value={
                bestTrialNoWarmstart?.value != null ? bestTrialNoWarmstart.value.toFixed(4) : "—"
              }
              subtext={
                bestTrialNoWarmstart?.id != null ? `TRIAL #${bestTrialNoWarmstart.id}` : undefined
              }
              color="lime"
              sparklineValues={bestSeries}
              deltaPct={bestDeltaPct}
              direction={objectiveDirection}
              helpText={MetricRegistry.get("incumbent")}
              animationSeed={animationSeed}
            />
          </div>
          <div className="h-44">
            <StatBadge
              label="Ritmo Médio"
              value={formatCompactDuration(avgDuration)}
              valueNode={
                avgDurationParts ? (
                  <DurationValueNode parts={avgDurationParts} animationSeed={animationSeed} />
                ) : null
              }
              subtext="ÚLTIMOS 5 TRIALS"
              color="orange"
              sparklineValues={lastDurations}
              deltaPct={avgDurationDeltaPct}
              direction="down"
              helpText={MetricRegistry.get("duration")}
              animationSeed={animationSeed}
            />
          </div>
          <div className="h-44">
            <StatBadge
              label="CONCLUSÃO ESTIMADA"
              value={formatCompactDuration(estimatedCompletion.eta)}
              valueNode={
                etaParts ? (
                  <DurationValueNode parts={etaParts} animationSeed={animationSeed} />
                ) : null
              }
              subtext="BASEADA NO RITMO ATUAL"
              color="orange"
              progress={estimatedCompletion.pct}
              deltaPct={etaDeltaPct}
              direction="down"
              helpText={MetricRegistry.get("eta")}
              animationSeed={animationSeed}
            />
          </div>
        </>
      ) : (
        <>
          <div className="h-44">
            <StatBadge
              label="Perda Atual"
              value={latestTrialMetrics.loss != null ? latestTrialMetrics.loss.toFixed(4) : "—"}
              color="rose"
              sparklineValues={liveMetricSeries.loss.slice(-20)}
              deltaPct={deltaFromSeries.loss}
              direction="down"
              helpText={MetricRegistry.get("loss")}
              animationSeed={animationSeed}
            />
          </div>
          <div className="h-44">
            <StatBadge
              label="MCC Atual"
              value={latestTrialMetrics.mcc != null ? latestTrialMetrics.mcc.toFixed(4) : "—"}
              color="orange"
              sparklineValues={liveMetricSeries.mcc.slice(-20)}
              deltaPct={deltaFromSeries.mcc}
              direction="up"
              helpText={MetricRegistry.get("mcc")}
              animationSeed={animationSeed}
            />
          </div>
          <div className="h-44">
            <StatBadge
              label="MRR Atual"
              value={latestTrialMetrics.mrr != null ? latestTrialMetrics.mrr.toFixed(4) : "—"}
              color="lime"
              sparklineValues={liveMetricSeries.mrr.slice(-20)}
              deltaPct={deltaFromSeries.mrr}
              direction="up"
              helpText={MetricRegistry.get("mrr")}
              animationSeed={animationSeed}
            />
          </div>
        </>
      )}
    </div>
  );
};
