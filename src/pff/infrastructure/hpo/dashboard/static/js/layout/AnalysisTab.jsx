import { useMemo } from "react";

import { useStore } from "../store/store.jsx";
import { SectionDivider } from "../ui/UIComponents.jsx";
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

export const AnalysisTab = () => {
  const { viewMode, filteredTrials, data, bestTrialNoWarmstart } = useStore();
  const learningCurveData = useMemo(
    () => data.liveStatus?.epoch_history || [],
    [data.liveStatus?.epoch_history]
  );

  if (viewMode === "study") {
    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-slide-right pb-10">
        <SectionDivider label="Sensibilidade & Impacto" icon={Sliders} />
        <div className="lg:col-span-1 h-[400px]">
          <ParamImportanceCard importances={data.importances} />
        </div>
        <div className="lg:col-span-1 h-[400px]">
          <CorrelationMatrixCard trials={filteredTrials} />
        </div>
        <SectionDivider label="Espaço de Busca & Interações" icon={GitMerge} />
        <div className="lg:col-span-2 h-[450px]">
          <ParallelCoordinatesCard trials={filteredTrials} />
        </div>
        <div className="lg:col-span-1 h-[350px]">
          <InteractionPlotCard trials={filteredTrials} importances={data.importances} />
        </div>
        <div className="lg:col-span-1 h-[350px]">
          <SearchSpaceTableCard searchSpace={data.searchSpace} />
        </div>
        <SectionDivider label="Performance & Trade-offs" icon={TargetIcon} />
        <div className="lg:col-span-1 h-[350px]">
          <ParetoFrontCard trials={filteredTrials} />
        </div>
        <div className="lg:col-span-1 h-[350px]">
          <ConfusionMatrixCard liveStatus={data.liveStatus} bestTrial={bestTrialNoWarmstart} />
        </div>
        <div className="lg:col-span-1 h-[350px]">
          <EDFPlotCard filteredTrials={filteredTrials} />
        </div>
        <div className="lg:col-span-1 h-[350px]">
          <HypervolumeCard trials={filteredTrials} />
        </div>
        <div className="lg:col-span-1 h-[350px]">
          <ContourPlotCard trials={filteredTrials} />
        </div>
        <SectionDivider label="Diagnóstico de Execução" icon={Activity} />
        <div className="lg:col-span-2 h-[250px]">
          <TimelinePlotCard trials={filteredTrials} />
        </div>
        <div className="lg:col-span-1 h-[300px]">
          <StructuralMetricsCard trials={filteredTrials} />
        </div>
        <div className="lg:col-span-1 h-[300px]">
          <LatencyParetoCard trials={filteredTrials} />
        </div>
        <div className="lg:col-span-1 h-[300px]">
          <PCComparisonTableCard trials={filteredTrials} />
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-slide-right pb-10">
      <SectionDivider label="Aprendizado & Convergência" icon={TrendingUp} />
      <div className="lg:col-span-2 h-[400px]">
        <LearningCurveChart data={learningCurveData} />
      </div>
      <SectionDivider label="Decomposição de Perda" icon={Layers} />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:col-span-2">
        <div className="h-[280px]">
          <ELBOBreakdownCard liveStatus={data.liveStatus} />
        </div>
        <div className="h-[280px]">
          <PC2MetricsCard liveStatus={data.liveStatus} />
        </div>
      </div>
      <SectionDivider label="Logs & Histórico" icon={Terminal} />
      <div className="lg:col-span-2">
        <TerminalLogCard logs={data.liveStatus?.logs} />
      </div>
    </div>
  );
};
