/**
 * Provide AdvancedTab module functionality for the HPO dashboard.
 */

import { useMemo } from "react";

import { useStore } from "../store/store.jsx";
import { CollapsibleSection } from "../ui/UIComponents.jsx";
import { ScatterPlotCard } from "../features/hpo/charts/ScatterPlotCard.jsx";
import { MetricsEvolutionCard } from "../features/hpo/charts/MetricsEvolutionCard.jsx";
import { HardwareMonitorCard } from "../features/hpo/charts/HardwareMonitorCard.jsx";
import { GradientHealthCard } from "../features/hpo/charts/GradientHealthCard.jsx";
import { RawConfigCard } from "../features/hpo/charts/RawConfigCard.jsx";
import { Microscope, TrendingUp, Cpu, Sliders } from "../ui/icons.jsx";

/**
 * Expose advanced tab for dashboard usage.
 */
export const AdvancedTab = () => {
  const { viewMode, filteredTrials, data, currentParams } = useStore();

  const completedTrials = useMemo(
    () => filteredTrials.filter((t) => t.state === "COMPLETE"),
    [filteredTrials]
  );
  const liveTrialData = useMemo(
    () => data.liveStatus?.epoch_history || [],
    [data.liveStatus?.epoch_history]
  );

  const { detectedParams, sliceDataLr, sliceDataEmbed } = useMemo(() => {
    const keys = new Set();
    for (const t of filteredTrials) {
      const p = t?.params;
      if (!p || typeof p !== "object") continue;
      for (const k of Object.keys(p)) keys.add(k);
    }

    const params = Array.from(keys);
    const lower = params.map((p) => ({ raw: p, lower: String(p).toLowerCase() }));

    const firstParam = params[0] ?? null;
    const lr =
      lower.find((p) => p.lower === "learning_rate")?.raw ||
      lower.find(
        (p) => p.lower.includes("learning_rate") || p.lower === "lr" || p.lower.endsWith("_lr")
      )?.raw ||
      lower.find((p) => p.lower.includes("lr"))?.raw ||
      firstParam;

    const embed =
      lower.find((p) => p.lower.includes("embed") || p.lower.includes("embedding"))?.raw ||
      (params.find((p) => p !== lr) ?? firstParam);

    const sliceDataLr = lr
      ? completedTrials
        .map((t) => ({ x: t?.params?.[lr], y: t.value ?? 0 }))
        .filter((p) => p.x != null)
      : [];

    const sliceDataEmbed = embed
      ? completedTrials
        .map((t) => ({ x: t?.params?.[embed], y: t.value ?? 0 }))
        .filter((p) => p.x != null)
      : [];

    const liveParams = data.liveStatus?.params || {};
    const lastEpoch = liveTrialData.length > 0 ? liveTrialData[liveTrialData.length - 1] : null;
    const liveScore = lastEpoch?.mrr ?? lastEpoch?.score ?? null;
    const livePoint = (key) => {
      const val = liveParams?.[key];
      if (val == null || liveScore == null) return null;
      const xVal = typeof val === "boolean" ? (val ? 1 : 0) : val;
      return { x: xVal, y: liveScore };
    };

    const liveLr = lr ? livePoint(lr) : null;
    const liveEmbed = embed ? livePoint(embed) : null;

    return {
      detectedParams: { lr, embed },
      sliceDataLr: liveLr ? [...sliceDataLr, liveLr] : sliceDataLr,
      sliceDataEmbed: liveEmbed ? [...sliceDataEmbed, liveEmbed] : sliceDataEmbed,
    };
  }, [completedTrials, filteredTrials, data.liveStatus?.params, liveTrialData]);

  if (viewMode === "study") {
    return (
      <div className="grid grid-cols-12 gap-6 animate-slide-right pb-10">
        <CollapsibleSection
          label="Análise Marginal"
          icon={Microscope}
          sectionKey="advanced-marginal"
          contentClassName="grid grid-cols-12 gap-6"
        >
          <div
            className="col-span-12 lg:col-span-6 min-h-[350px]"
            id="search-advanced-study-slice-lr"
            data-search-id="search-advanced-study-slice-lr"
          >
            <ScatterPlotCard
              title={`Slice Plot: ${detectedParams.lr || "Parâmetro"} × Objetivo`}
              data={sliceDataLr}
              xLabel={detectedParams.lr || "Parâmetro"}
              yLabel="Objetivo"
            />
          </div>
          <div
            className="col-span-12 lg:col-span-6 min-h-[350px]"
            id="search-advanced-study-slice-embed"
            data-search-id="search-advanced-study-slice-embed"
          >
            <ScatterPlotCard
              title={`Slice Plot: ${detectedParams.embed || "Parâmetro"} × Objetivo`}
              data={sliceDataEmbed}
              xLabel={detectedParams.embed || "Parâmetro"}
              yLabel="Objetivo"
            />
          </div>
        </CollapsibleSection>

        <CollapsibleSection
          label="Dinâmica de Performance"
          icon={TrendingUp}
          sectionKey="advanced-dynamics"
          contentClassName="grid grid-cols-12 gap-6"
        >
          <div
            className="col-span-12 lg:col-span-6 min-h-[350px]"
            id="search-advanced-study-duration-score"
            data-search-id="search-advanced-study-duration-score"
          >
            <ScatterPlotCard
              title="Duração × Score"
              data={completedTrials.map((t) => ({ x: t.duration, y: t.value }))}
              xLabel="Duração (s)"
              yLabel="Score"
            />
          </div>
          <div
            className="col-span-12 lg:col-span-6 min-h-[350px]"
            id="search-advanced-study-metrics-evolution"
            data-search-id="search-advanced-study-metrics-evolution"
          >
            <MetricsEvolutionCard trials={filteredTrials} />
          </div>
        </CollapsibleSection>
      </div>
    );
  }
  return (
    <div className="grid grid-cols-12 gap-6 animate-slide-right pb-10">
      <CollapsibleSection
        label="Saúde do Sistema"
        icon={Cpu}
        sectionKey="advanced-health"
        contentClassName="grid grid-cols-12 gap-6"
      >
        <div
          className="col-span-12 lg:col-span-6 min-h-[300px]"
          id="search-advanced-trial-hardware"
          data-search-id="search-advanced-trial-hardware"
        >
          <HardwareMonitorCard
            hardware={data.liveStatus?.hardware}
            history={data.liveStatus?.hardware_history}
          />
        </div>
        <div
          className="col-span-12 lg:col-span-6 min-h-[300px]"
          id="search-advanced-trial-gradient"
          data-search-id="search-advanced-trial-gradient"
        >
          <GradientHealthCard liveData={liveTrialData} />
        </div>
      </CollapsibleSection>

      <CollapsibleSection
        label="Configuração"
        icon={Sliders}
        sectionKey="advanced-config"
        contentClassName="grid grid-cols-12 gap-6"
      >
        <div
          className="col-span-12 min-h-[200px]"
          id="search-advanced-trial-raw-config"
          data-search-id="search-advanced-trial-raw-config"
        >
          <RawConfigCard config={currentParams} />
        </div>
      </CollapsibleSection>
    </div>
  );
};
