import { useMemo } from 'react';


import { useStore } from "../store/store.jsx";
import { SectionDivider } from "../ui/UIComponents.jsx";
import {
    ScatterPlotCard, MetricsEvolutionCard, HardwareMonitorCard,
    GradientHealthCard, RawConfigCard
} from "../features/hpo/charts/AllCharts.js";
import { Microscope, TrendingUp, Cpu, Sliders } from "../ui/BaseComponents.jsx";

export const AdvancedTab = () => {
    const { viewMode, filteredTrials, data, currentParams } = useStore();

    const completedTrials = useMemo(() => filteredTrials.filter(t => t.state === 'COMPLETE'), [filteredTrials]);
    const liveTrialData = useMemo(() => data.liveStatus?.epoch_history || [], [data.liveStatus?.epoch_history]);

    const { detectedParams, sliceDataLr, sliceDataEmbed } = useMemo(() => {
        const keys = new Set();
        for (const t of filteredTrials) {
            const p = t?.params;
            if (!p || typeof p !== 'object') continue;
            for (const k of Object.keys(p)) keys.add(k);
        }

        const params = Array.from(keys);
        const lower = params.map((p) => ({ raw: p, lower: String(p).toLowerCase() }));

        const firstParam = params[0] ?? null;
        const lr =
            lower.find((p) => p.lower === 'learning_rate')?.raw ||
            lower.find((p) => p.lower.includes('learning_rate') || p.lower === 'lr' || p.lower.endsWith('_lr'))?.raw ||
            lower.find((p) => p.lower.includes('lr'))?.raw ||
            firstParam;

        const embed =
            lower.find((p) => p.lower.includes('embed') || p.lower.includes('embedding'))?.raw ||
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

    if (viewMode === 'study') {
        return (
            <div className="grid grid-cols-12 gap-6 animate-slide-right pb-10">
                <SectionDivider label="Análise Marginal" icon={Microscope} />
                <div className="col-span-12 lg:col-span-6 min-h-[350px]">
                    <ScatterPlotCard title={`Slice Plot: ${detectedParams.lr || 'Parâmetro'} × Objetivo`} data={sliceDataLr} xLabel={detectedParams.lr || 'Parâmetro'} yLabel="Objetivo" />
                </div>
                <div className="col-span-12 lg:col-span-6 min-h-[350px]">
                    <ScatterPlotCard title={`Slice Plot: ${detectedParams.embed || 'Parâmetro'} × Objetivo`} data={sliceDataEmbed} xLabel={detectedParams.embed || 'Parâmetro'} yLabel="Objetivo" />
                </div>
                <SectionDivider label="Dinâmica de Performance" icon={TrendingUp} />
                <div className="col-span-12 lg:col-span-6 min-h-[350px]">
                    <ScatterPlotCard title="Duração × Score" data={completedTrials.map(t => ({ x: t.duration, y: t.value }))} xLabel="Duração (s)" yLabel="Score" />
                </div>
                <div className="col-span-12 lg:col-span-6 min-h-[350px]"><MetricsEvolutionCard trials={filteredTrials} /></div>
            </div>
        );
    }
    return (
        <div className="grid grid-cols-12 gap-6 animate-slide-right pb-10">
            <SectionDivider label="Saúde do Sistema" icon={Cpu} />
            <div className="col-span-12 lg:col-span-6 min-h-[300px]"><HardwareMonitorCard hardware={data.liveStatus?.hardware} history={data.liveStatus?.hardware_history} /></div>
            <div className="col-span-12 lg:col-span-6 min-h-[300px]"><GradientHealthCard liveData={liveTrialData} /></div>
            <SectionDivider label="Configuração" icon={Sliders} />
            <div className="col-span-12 min-h-[200px]"><RawConfigCard config={currentParams} /></div>
        </div>
    );
};
