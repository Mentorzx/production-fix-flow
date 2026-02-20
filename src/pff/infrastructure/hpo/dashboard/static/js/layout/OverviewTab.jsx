/**
 * Provide OverviewTab module functionality for the HPO dashboard.
 */

import { useMemo } from "react";
import { useStore } from "../store/store.jsx";
import { BestTrialCard } from "../features/hpo/charts/BestTrialCard.jsx";
import { IncumbentTrajectoryCard } from "../features/hpo/charts/IncumbentTrajectoryCard.jsx";
import {
  TrialLearningMetricsCard,
  FoldConfusionsCard,
  FullMetricsLogCard,
  DetailedHistoryCard,
} from "../features/hpo/charts/AllCharts.js";

/**
 * Expose overview tab for dashboard usage.
 */
export const OverviewTab = () => {
  const { trials, bestTrialNoWarmstart, filteredTrials, data, viewMode } = useStore();

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
            <BestTrialCard trial={bestTrialNoWarmstart} delay={600} />
          </div>
        </div>

        {/* Trial Table */}
        <div
          className="col-span-12 min-h-0 lg:min-h-[320px]"
          id="search-overview-study-detailed-history"
          data-search-id="search-overview-study-detailed-history"
        >
          <DetailedHistoryCard trials={filteredTrials} />
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
