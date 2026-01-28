import { useMemo } from 'react';


import { useStore } from "../store/store.jsx";
import { EstimatedScoreCard, OptimizationVelocityCard, LossProjectionCard, RegressionChartCard, RegressionInsightsCard, TrialDiffTableCard } from "../features/hpo/charts/AllCharts.js";
import { SectionDivider } from "../ui/UIComponents.jsx";
import { Share2, TrendingUp } from "../ui/BaseComponents.jsx";

export const ForecastTab = () => {
    const { filteredTrials, data } = useStore();

    const projections = useMemo(() => {
        // Strict filtering: Use filteredTrials only.
        const history = filteredTrials.filter(t => t.value != null).sort((a, b) => a.id - b.id).map(t => ({ x: t.id, y: t.value }));
        const n = history.length;
        if (n < 2) return { predictedValue: 0, slope: 0 };
        let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
        history.forEach(p => { sumX += p.x; sumY += p.y; sumXY += p.x * p.y; sumXX += p.x * p.x; });
        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        const total = data.totalTrials || 50;
        return { slope, predictedValue: slope * total + intercept };
    }, [filteredTrials, data.totalTrials]);

    return (
        <div className="grid grid-cols-12 gap-6 animate-slide-right pb-10">
            <SectionDivider label="Estimativas Futuras" icon={Share2} />

            {/* KPI Row - Top */}
            <div className="col-span-12 lg:col-span-4 h-[160px]"><EstimatedScoreCard projection={projections} totalTrials={data.totalTrials} /></div>
            <div className="col-span-12 lg:col-span-4 h-[160px]"><OptimizationVelocityCard projection={projections} /></div>
            <div className="col-span-12 lg:col-span-4 h-[160px]"><LossProjectionCard liveData={data.liveStatus?.epoch_history} /></div>

            <SectionDivider label="Tendência e Regressão" icon={TrendingUp} />

            <div className="col-span-12 lg:col-span-8 h-[450px]">
                <RegressionChartCard trials={filteredTrials} />
            </div>
            <div className="col-span-12 lg:col-span-4 h-[450px]">
                <RegressionInsightsCard trials={filteredTrials} />
            </div>

            <SectionDivider label="Comparativo de Trials" icon={Share2} />

            <div className="col-span-12 h-[360px]">
                <TrialDiffTableCard trials={filteredTrials} direction={data.direction} />
            </div>
        </div>
    );
};
