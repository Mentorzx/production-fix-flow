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
        <div className="col-span-12 grid grid-cols-12 gap-6 min-h-0 lg:h-[480px] lg:grid-rows-1 animate-slide-up delay-100">
          <div className="col-span-12 lg:col-span-8 h-full min-h-0">
            <IncumbentTrajectoryCard trials={filteredTrials} />
          </div>
          <div className="col-span-12 lg:col-span-4 h-full min-h-0">
            <BestTrialCard trial={bestTrialNoWarmstart} delay={600} />
          </div>
        </div>

        {/* Trial Table */}
        <div className="col-span-12 animate-slide-up delay-200">
          <DetailedHistoryCard trials={filteredTrials} />
        </div>
      </div>
    );
  }

  // View Mode: Trial
  return (
    <div className="grid grid-cols-12 gap-6 p-2 animate-fade-in">
      <div className="col-span-12 grid grid-cols-12 gap-6 min-h-0 lg:h-[480px] lg:grid-rows-1 animate-slide-up delay-100">
        <div className="col-span-12 lg:col-span-8 h-full min-h-0">
          <TrialLearningMetricsCard liveData={liveTrialData} />
        </div>
        <div className="col-span-12 lg:col-span-4 h-full min-h-0">
          <FoldConfusionsCard trials={trials} liveStatus={data.liveStatus} charts={data.charts} />
        </div>
      </div>

      <div className="col-span-12 animate-slide-up delay-200">
        <FullMetricsLogCard liveStatus={data.liveStatus} />
      </div>
    </div>
  );
};
