/**
 * Provide OverviewTab module functionality for the HPO dashboard.
 */

import { useMemo } from "react";
import { useStore } from "../store/store.jsx";
import { resolveMetricValue } from "../domain/metrics/Formatters.js";
import { BestTrialCard } from "../features/hpo/charts/BestTrialCard.jsx";
import { IncumbentTrajectoryCard } from "../features/hpo/charts/IncumbentTrajectoryCard.jsx";
import { TrialLearningMetricsCard } from "../features/hpo/charts/TrialLearningMetricsCard.jsx";
import { FoldConfusionsCard } from "../features/hpo/charts/FoldConfusionsCard.jsx";
import { FullMetricsLogCard } from "../features/hpo/charts/FullMetricsLogCard.jsx";
import { DetailedHistoryCard } from "../features/hpo/charts/DetailedHistoryCard.jsx";

/**
 * Expose overview tab for dashboard usage.
 */
export const OverviewTab = () => {
  const { trials, filteredTrials, data, viewMode } = useStore();
  const rankingTrials = useMemo(
    () =>
      filteredTrials.filter((trial) => {
        if (!trial) return false;
        const state = String(trial.state || "").toUpperCase();
        return state === "COMPLETE" && Number.isFinite(Number(trial.value));
      }),
    [filteredTrials]
  );
  const bestRankingTrial = useMemo(() => {
    const eligible = rankingTrials.filter(
      (trial) => typeof resolveMetricValue(trial, "score") === "number"
    );
    if (eligible.length === 0) return { id: 0, value: 0, params: {} };
    const noWarm = eligible.filter((trial) => !trial?.warmstart);
    const pool = noWarm.length > 0 ? noWarm : eligible;
    const sorted = [...pool].sort(
      (a, b) => Number(resolveMetricValue(b, "score")) - Number(resolveMetricValue(a, "score"))
    );
    return sorted[0] || { id: 0, value: 0, params: {} };
  }, [rankingTrials]);

  const liveTrialData = useMemo(
    () => data.liveStatus?.epoch_history || [],
    [data.liveStatus?.epoch_history]
  );

  if (viewMode === "study") {
    return (
      <div className="grid grid-cols-12 gap-6 p-2">
        {/* Filters moved to GlobalFilterBar */}

        {/* Main Charts */}
        <div className="col-span-12 grid grid-cols-12 gap-6 min-h-0 lg:h-[480px] lg:grid-rows-1">
          <div
            className="col-span-12 lg:col-span-8 h-full min-h-0"
            id="search-overview-study-incumbent-trajectory"
            data-search-id="search-overview-study-incumbent-trajectory"
          >
            <IncumbentTrajectoryCard trials={filteredTrials} />
          </div>
          <div
            className="col-span-12 lg:col-span-4 h-full min-h-0"
            id="search-overview-study-best-trial"
            data-search-id="search-overview-study-best-trial"
          >
            <BestTrialCard trial={bestRankingTrial} delay={600} />
          </div>
        </div>

        {/* Trial Table */}
        <div
          className="col-span-12 min-h-0 lg:min-h-[320px]"
          id="search-overview-study-detailed-history"
          data-search-id="search-overview-study-detailed-history"
        >
          <DetailedHistoryCard trials={rankingTrials} />
        </div>
      </div>
    );
  }

  // View Mode: Trial
  return (
    <div className="grid grid-cols-12 gap-6 p-2">
      <div className="col-span-12 grid grid-cols-12 gap-6 min-h-0 lg:h-[480px] lg:grid-rows-1">
        <div
          className="col-span-12 lg:col-span-8 h-full min-h-0"
          id="search-overview-trial-learning-metrics"
          data-search-id="search-overview-trial-learning-metrics"
        >
          <TrialLearningMetricsCard liveData={liveTrialData} />
        </div>
        <div
          className="col-span-12 lg:col-span-4 h-full min-h-0"
          id="search-overview-trial-fold-confusions"
          data-search-id="search-overview-trial-fold-confusions"
        >
          <FoldConfusionsCard trials={trials} liveStatus={data.liveStatus} charts={data.charts} />
        </div>
      </div>

      <div
        className="col-span-12 min-h-0 lg:min-h-[320px]"
        id="search-overview-trial-full-metrics-log"
        data-search-id="search-overview-trial-full-metrics-log"
      >
        <FullMetricsLogCard liveStatus={data.liveStatus} />
      </div>
    </div>
  );
};
