/**
 * Provide AnalysisTab module functionality for the HPO dashboard.
 */

import { useMemo } from "react";

import { useStore } from "../store/store.jsx";
import { CollapsibleSection } from "../ui/UIComponents.jsx";
import {
  ParamImportanceCard,
  CorrelationMatrixCard,
  ParallelCoordinatesCard,
  InteractionPlotCard,
  SearchSpaceTableCard,
  ParetoFrontCard,
  ConfusionMatrixCard,
  EDFPlotCard,
  HypervolumeCard,
  ContourPlotCard,
  TimelinePlotCard,
  StructuralMetricsCard,
  LatencyParetoCard,
  PCComparisonTableCard,
  LearningCurveChart,
  ELBOBreakdownCard,
  PC2MetricsCard,
  TerminalLogCard,
} from "../features/hpo/charts/AllCharts.js";
import {
  Sliders,
  Activity,
  TargetIcon,
  GitMerge,
  Layers,
  TrendingUp,
  Terminal,
} from "../ui/BaseComponents.jsx";

/**
 * Expose analysis tab for dashboard usage.
 */
export const AnalysisTab = () => {
  const { viewMode, filteredTrials, data, bestTrialNoWarmstart } = useStore();
  const learningCurveData = useMemo(
    () => data.liveStatus?.epoch_history || [],
    [data.liveStatus?.epoch_history]
  );

  if (viewMode === "study") {
    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-slide-right pb-10">
        <CollapsibleSection
          label="Sensibilidade & Impacto"
          icon={Sliders}
          sectionKey="analysis-sensitivity"
          contentClassName="grid grid-cols-1 lg:grid-cols-2 gap-6"
        >
          <div
            className="lg:col-span-1 h-[400px]"
            id="search-analysis-study-param-importance"
            data-search-id="search-analysis-study-param-importance"
          >
            <ParamImportanceCard importances={data.importances} />
          </div>
          <div
            className="lg:col-span-1 h-[400px]"
            id="search-analysis-study-correlation"
            data-search-id="search-analysis-study-correlation"
          >
            <CorrelationMatrixCard trials={filteredTrials} />
          </div>
        </CollapsibleSection>

        <CollapsibleSection
          label="Espaço de Busca & Interações"
          icon={GitMerge}
          sectionKey="analysis-search"
          contentClassName="grid grid-cols-1 lg:grid-cols-2 gap-6"
        >
          <div
            className="lg:col-span-2 h-[450px]"
            id="search-analysis-study-parallel"
            data-search-id="search-analysis-study-parallel"
          >
            <ParallelCoordinatesCard trials={filteredTrials} />
          </div>
          <div
            className="lg:col-span-1 h-[350px]"
            id="search-analysis-study-interaction"
            data-search-id="search-analysis-study-interaction"
          >
            <InteractionPlotCard trials={filteredTrials} importances={data.importances} />
          </div>
          <div
            className="lg:col-span-1 h-[350px]"
            id="search-analysis-study-search-space-table"
            data-search-id="search-analysis-study-search-space-table"
          >
            <SearchSpaceTableCard searchSpace={data.searchSpace} />
          </div>
        </CollapsibleSection>

        <CollapsibleSection
          label="Performance & Trade-offs"
          icon={TargetIcon}
          sectionKey="analysis-tradeoffs"
          contentClassName="grid grid-cols-1 lg:grid-cols-2 gap-6"
        >
          <div
            className="lg:col-span-1 h-[350px]"
            id="search-analysis-study-pareto"
            data-search-id="search-analysis-study-pareto"
          >
            <ParetoFrontCard trials={filteredTrials} />
          </div>
          <div
            className="lg:col-span-1 h-[350px]"
            id="search-analysis-study-confusion-matrix"
            data-search-id="search-analysis-study-confusion-matrix"
          >
            <ConfusionMatrixCard liveStatus={data.liveStatus} bestTrial={bestTrialNoWarmstart} />
          </div>
          <div
            className="lg:col-span-1 h-[350px]"
            id="search-analysis-study-edf"
            data-search-id="search-analysis-study-edf"
          >
            <EDFPlotCard filteredTrials={filteredTrials} />
          </div>
          <div
            className="lg:col-span-1 h-[350px]"
            id="search-analysis-study-hypervolume"
            data-search-id="search-analysis-study-hypervolume"
          >
            <HypervolumeCard trials={filteredTrials} />
          </div>
          <div
            className="lg:col-span-2 h-[420px]"
            id="search-analysis-study-contour"
            data-search-id="search-analysis-study-contour"
          >
            <ContourPlotCard trials={filteredTrials} />
          </div>
        </CollapsibleSection>

        <CollapsibleSection
          label="Diagnóstico de Execução"
          icon={Activity}
          sectionKey="analysis-diagnostics"
          contentClassName="grid grid-cols-1 lg:grid-cols-2 gap-6"
        >
          <div
            className="lg:col-span-2 h-[320px]"
            id="search-analysis-study-timeline"
            data-search-id="search-analysis-study-timeline"
          >
            <TimelinePlotCard trials={filteredTrials} />
          </div>
          <div
            className="lg:col-span-1 h-[300px]"
            id="search-analysis-study-structural"
            data-search-id="search-analysis-study-structural"
          >
            <StructuralMetricsCard trials={filteredTrials} />
          </div>
          <div
            className="lg:col-span-1 h-[300px]"
            id="search-analysis-study-latency-pareto"
            data-search-id="search-analysis-study-latency-pareto"
          >
            <LatencyParetoCard trials={filteredTrials} />
          </div>
          <div
            className="lg:col-span-2 h-[300px]"
            id="search-analysis-study-pc-comparison"
            data-search-id="search-analysis-study-pc-comparison"
          >
            <PCComparisonTableCard trials={filteredTrials} />
          </div>
        </CollapsibleSection>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-slide-right pb-10">
      <CollapsibleSection
        label="Aprendizado & Convergência"
        icon={TrendingUp}
        sectionKey="analysis-trial-learning"
        contentClassName="grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        <div
          className="lg:col-span-2 h-[400px]"
          id="search-analysis-trial-learning-curve"
          data-search-id="search-analysis-trial-learning-curve"
        >
          <LearningCurveChart data={learningCurveData} />
        </div>
      </CollapsibleSection>

      <CollapsibleSection
        label="Decomposição de Perda"
        icon={Layers}
        sectionKey="analysis-trial-loss"
        contentClassName="grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        <div
          className="h-[340px]"
          id="search-analysis-trial-elbo"
          data-search-id="search-analysis-trial-elbo"
        >
          <ELBOBreakdownCard liveStatus={data.liveStatus} />
        </div>
        <div
          className="h-[340px]"
          id="search-analysis-trial-pc2-metrics"
          data-search-id="search-analysis-trial-pc2-metrics"
        >
          <PC2MetricsCard liveStatus={data.liveStatus} />
        </div>
      </CollapsibleSection>

      <CollapsibleSection
        label="Logs & Histórico"
        icon={Terminal}
        sectionKey="analysis-trial-logs"
        contentClassName="grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        <div
          className="lg:col-span-2"
          id="search-analysis-trial-terminal-log"
          data-search-id="search-analysis-trial-terminal-log"
        >
          <TerminalLogCard logs={data.liveStatus?.logs} />
        </div>
      </CollapsibleSection>
    </div>
  );
};
