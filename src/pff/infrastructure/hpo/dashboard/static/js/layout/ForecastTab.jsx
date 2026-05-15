/**
 * Provide ForecastTab module functionality for the HPO dashboard.
 */

import { useMemo } from "react";
import { useStore } from "../store/store.jsx";
import {
  EstimatedScoreCard,
  OptimizationVelocityCard,
} from "../features/hpo/charts/ForecastCards.jsx";
import { LossProjectionCard } from "../features/hpo/charts/LossProjectionCard.jsx";
import { RegressionChartCard } from "../features/hpo/charts/RegressionChartCard.jsx";
import { RegressionInsightsCard } from "../features/hpo/charts/RegressionInsightsCard.jsx";
import { TrialDiffTableCard } from "../features/hpo/charts/TrialDiffTableCard.jsx";
import { GeneralizationGapCard } from "../features/hpo/charts/GeneralizationGapCard.jsx";
import { LocalOptimaDiagnosticsCard } from "../features/hpo/charts/LocalOptimaDiagnosticsCard.jsx";
import { SearchSpaceAdvisorCard } from "../features/hpo/charts/SearchSpaceAdvisorCard.jsx";
import { CollapsibleSection } from "../ui/UIComponents.jsx";
import { Share2, TrendingUp, Sliders } from "../ui/icons.jsx";
import { DEFAULT_TOTAL_TRIALS } from "../ui/constants.js";
import { linearRegression } from "../utils/statistics.js";

/**
 * Expose forecast tab for dashboard usage.
 */
export const ForecastTab = () => {
  const { viewMode, filteredTrials, data } = useStore();

  const projections = useMemo(() => {
    if (viewMode !== "study") return { predictedValue: 0, slope: 0 };
    const history = filteredTrials
      .filter((t) => t.value != null)
      .sort((a, b) => a.id - b.id)
      .map((t) => ({ x: t.id, y: t.value }));
    if (history.length < 2) return { predictedValue: 0, slope: 0 };
    const { slope, intercept } = linearRegression(history);
    const total = data.totalTrials || DEFAULT_TOTAL_TRIALS;
    return { slope, predictedValue: slope * total + intercept };
  }, [filteredTrials, data.totalTrials, viewMode]);

  const liveTrialData = useMemo(
    () => data.liveStatus?.epoch_history || [],
    [data.liveStatus?.epoch_history]
  );
  const targetEpoch = useMemo(() => {
    const fromStatus = Number(data?.liveStatus?.total_epochs);
    if (Number.isFinite(fromStatus) && fromStatus > 0) return Math.floor(fromStatus);

    const fromParams = Number(data?.liveStatus?.params?.dslfm_epochs);
    if (Number.isFinite(fromParams) && fromParams > 0) return Math.floor(fromParams);

    return null;
  }, [data?.liveStatus?.total_epochs, data?.liveStatus?.params?.dslfm_epochs]);

  if (viewMode === "trial") {
    return (
      <div className="grid grid-cols-12 gap-6 animate-slide-right pb-10">
        <CollapsibleSection
          label="Previsão do Trial"
          icon={Share2}
          sectionKey="forecast-trial"
          contentClassName="grid grid-cols-12 gap-6"
        >
          <div
            className="col-span-12 h-[320px]"
            id="search-forecast-trial-loss-projection"
            data-search-id="search-forecast-trial-loss-projection"
          >
            <LossProjectionCard liveData={liveTrialData} targetEpoch={targetEpoch} />
          </div>
          <div
            className="col-span-12 h-[360px]"
            id="search-forecast-trial-generalization-gap"
            data-search-id="search-forecast-trial-generalization-gap"
          >
            <GeneralizationGapCard liveData={liveTrialData} />
          </div>
        </CollapsibleSection>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-12 gap-6 animate-slide-right pb-10">
      <CollapsibleSection
        label="Estimativas Futuras"
        icon={Share2}
        sectionKey="forecast-future"
        contentClassName="grid grid-cols-12 gap-6"
      >
        <div
          className="col-span-12 lg:col-span-4 h-[160px]"
          id="search-forecast-study-estimated-score"
          data-search-id="search-forecast-study-estimated-score"
        >
          <EstimatedScoreCard projection={projections} totalTrials={data.totalTrials} />
        </div>
        <div
          className="col-span-12 lg:col-span-8 h-[160px]"
          id="search-forecast-study-optimization-velocity"
          data-search-id="search-forecast-study-optimization-velocity"
        >
          <OptimizationVelocityCard projection={projections} />
        </div>
      </CollapsibleSection>

      <CollapsibleSection
        label="Tendência e Regressão"
        icon={TrendingUp}
        sectionKey="forecast-regression"
        contentClassName="grid grid-cols-12 gap-6"
      >
        <div
          className="col-span-12 lg:col-span-8 h-[450px]"
          id="search-forecast-study-regression-chart"
          data-search-id="search-forecast-study-regression-chart"
        >
          <RegressionChartCard
            trials={filteredTrials}
            totalTrials={data.totalTrials || DEFAULT_TOTAL_TRIALS}
          />
        </div>
        <div
          className="col-span-12 lg:col-span-4 h-[450px]"
          id="search-forecast-study-regression-insights"
          data-search-id="search-forecast-study-regression-insights"
        >
          <RegressionInsightsCard trials={filteredTrials} />
        </div>
      </CollapsibleSection>

      <CollapsibleSection
        label="Estagnacao & Exploracao"
        icon={Sliders}
        sectionKey="forecast-local-optima"
        contentClassName="grid grid-cols-12 gap-6"
      >
        <div
          className="col-span-12 h-[420px]"
          id="search-forecast-study-local-optima"
          data-search-id="search-forecast-study-local-optima"
        >
          <LocalOptimaDiagnosticsCard diagnostics={data.optimizationDiagnostics?.localOptima} />
        </div>
      </CollapsibleSection>

      <CollapsibleSection
        label="Comparativo de Trials"
        icon={Share2}
        sectionKey="forecast-comparison"
        contentClassName="grid grid-cols-12 gap-6"
      >
        <div
          className="col-span-12 h-[360px]"
          id="search-forecast-study-trial-diff"
          data-search-id="search-forecast-study-trial-diff"
        >
          <TrialDiffTableCard trials={filteredTrials} direction={data.direction} />
        </div>
      </CollapsibleSection>

      <CollapsibleSection
        label="Search Space Advisor"
        icon={Sliders}
        sectionKey="forecast-advisor"
        contentClassName="grid grid-cols-12 gap-6"
      >
        <div
          className="col-span-12 h-[520px]"
          id="search-forecast-study-search-space-advisor"
          data-search-id="search-forecast-study-search-space-advisor"
        >
          <SearchSpaceAdvisorCard
            advice={data.searchSpaceAdvice}
            searchSpace={data.searchSpace}
            trials={filteredTrials}
          />
        </div>
      </CollapsibleSection>
    </div>
  );
};
