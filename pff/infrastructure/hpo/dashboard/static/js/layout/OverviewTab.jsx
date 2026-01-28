import { useMemo, useRef, useEffect, useState } from "react";
import { useStore } from "../store/store.jsx";
import { BestTrialCard } from "../features/hpo/charts/BestTrialCard.jsx";
import { IncumbentTrajectoryCard } from "../features/hpo/charts/IncumbentTrajectoryCard.jsx";
import { MetricsHistoryTable } from "../features/hpo/MetricsHistoryTable.jsx";
import { StatBadge, TableIcon } from "../ui/BaseComponents.jsx";
import { SectionDivider } from "../ui/UIComponents.jsx";
import { TrialLearningMetricsCard, GeneralizationGapCard, FoldConfusionsCard } from "../features/hpo/charts/AllCharts.js";

const TrialStatusCard = ({ data, trials }) => {
    const status = data.liveStatus || {};
    const progress = status.progress || 0;
    const lastTrialId = trials.length > 0 ? trials[trials.length - 1].id : 0;
    const currentTrial = status.trial_number != null ? (status.trial_number) : lastTrialId;
    const [pulseKey, setPulseKey] = useState(0);
    const prevProgress = useRef(progress);

    const formatTime = (s) => {
        if (!s || s <= 0) return '--:--';
        const m = Math.floor(s / 60); const sec = Math.floor(s % 60);
        return m > 0 ? `${m}m ${sec.toString().padStart(2, '0')}s` : `${sec}s`;
    };

    const eta = status.elapsed_seconds && progress > 0 ? (status.elapsed_seconds / (progress / 100)) - status.elapsed_seconds : null;

    useEffect(() => {
        if (Math.abs(progress - prevProgress.current) >= 0.5) {
            setPulseKey((k) => k + 1);
            prevProgress.current = progress;
        }
    }, [progress]);

    return (
        <div className="bg-zinc-900 border border-zinc-800 p-4 rounded-xl flex flex-col justify-between shadow-lg h-full min-h-[120px] transition-transform hover:scale-[1.01] duration-300">
            <div className="flex justify-between items-start opacity-70"><span className="text-[10px] font-bold uppercase tracking-widest text-orange-400">Progresso HPO</span></div>
            <div className="text-3xl font-bold font-mono my-2 text-white">Trial #{currentTrial} <span className="text-zinc-600 text-lg font-normal">/ {data.totalTrials || 50}</span></div>
            <div className="space-y-3 mt-1">
                <div className="h-2 w-full bg-zinc-800 rounded-full overflow-hidden">
                    <div key={pulseKey} className="h-full bg-linear-to-r from-orange-600 to-orange-400 transition-all duration-1000 shadow-[0_0_15px_rgba(99,102,241,0.4)]" style={{ width: `${progress}%` }}></div>
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[10px] font-mono tracking-tight">
                    <div className="flex justify-between border-r border-zinc-800 pr-2"><span className="text-zinc-500">Fold</span><span className="text-lime-400 font-bold">{status.cv_fold_id != null ? status.cv_fold_id : '—'}</span></div>
                    <div className="flex justify-between pl-2"><span className="text-zinc-500">ETA</span><span className="text-amber-400 font-bold">{formatTime(eta)}</span></div>
                </div>
            </div>
        </div>
    );
};

export const OverviewTab = () => {
    const { trials, bestTrialNoWarmstart, filteredTrials, data, viewMode } = useStore();

    const getLastAvailableMetric = (series, key) => {
        if (!Array.isArray(series) || series.length === 0) return null;
        for (let i = series.length - 1; i >= 0; i -= 1) {
            const value = series[i]?.[key];
            if (Number.isFinite(value)) return value;
        }
        return null;
    };

    const estimatedCompletion = useMemo(() => {
        const totalTrials = typeof data?.totalTrials === 'number' && data.totalTrials > 0 ? data.totalTrials : 50;
        const completed = Array.isArray(trials) ? trials.filter(t => t?.state === 'COMPLETE' && typeof t?.duration === 'number' && t.duration > 0.1) : [];
        if (completed.length < 2) return '—';

        const recent = completed.slice(-5);
        const avgDuration = recent.reduce((acc, t) => acc + t.duration, 0) / recent.length;
        if (!Number.isFinite(avgDuration) || avgDuration <= 0) return '—';

        const remaining = Math.max(0, totalTrials - completed.length);
        const etaSeconds = remaining * avgDuration;

        if (etaSeconds < 60) return `${Math.round(etaSeconds)}s`;
        if (etaSeconds < 3600) return `${Math.round(etaSeconds / 60)}m`;
        return `${(etaSeconds / 3600).toFixed(1)}h`;
    }, [data?.totalTrials, trials]);

    const liveTrialData = useMemo(() => data.liveStatus?.epoch_history || [], [data.liveStatus?.epoch_history]);
    const latestTrialMetrics = useMemo(() => ({
        loss: getLastAvailableMetric(liveTrialData, 'loss'),
        mcc: getLastAvailableMetric(liveTrialData, 'mcc'),
        mrr: getLastAvailableMetric(liveTrialData, 'mrr')
    }), [liveTrialData]);

    if (viewMode === 'study') {
        return (
            <div className="grid grid-cols-12 gap-6 p-2">
                {/* Filters moved to GlobalFilterBar */}


                {/* Top Stats */}
                <div className="col-span-12 grid grid-cols-2 lg:grid-cols-4 gap-4 animate-slide-up">
                    <div className="h-full"><StatBadge label="Melhor Global" value={bestTrialNoWarmstart.value?.toFixed(4) || 0} subtext={`Trial #${bestTrialNoWarmstart.id}`} color="lime" /></div>
                    <div className="h-full"><TrialStatusCard data={data} trials={trials} /></div>
                    <div className="h-full"><StatBadge label="Ritmo Médio" value={(() => {
                        const completed = trials.filter(t => t.state === 'COMPLETE' && typeof t.duration === 'number' && t.duration > 0.1);
                        if (completed.length < 2) return "—";
                        const recent = completed.slice(-5); const avgDuration = recent.reduce((a, t) => a + t.duration, 0) / recent.length;
                        if (avgDuration <= 0) return "—";
                        return avgDuration > 60 ? (avgDuration / 60).toFixed(1) + " min/trial" : (60 / avgDuration).toFixed(1) + " t/m";
                    })()} subtext="Últimos 5 trials" color="orange" /></div>
                    <div className="h-full"><StatBadge label="CONCLUSÃO ESTIMADA" value={estimatedCompletion} subtext="Baseada no ritmo atual" color="orange" /></div>
                </div>

                {/* Main Charts */}
                <div className="col-span-12 grid grid-cols-12 gap-6 h-[480px] animate-slide-up delay-100">
                    <div className="col-span-12 lg:col-span-8 h-full"><IncumbentTrajectoryCard trials={filteredTrials} /></div>
                    <div className="col-span-12 lg:col-span-4 h-full"><BestTrialCard trial={bestTrialNoWarmstart} delay={600} /></div>
                </div>

                {/* Trial Table */}
                <SectionDivider label="Ranking de Trials" icon={TableIcon} />
                <div className="col-span-12 animate-slide-up delay-200 min-h-[160px]">
                    <MetricsHistoryTable data={filteredTrials} type="trial" />
                </div>
            </div>
        );
    }

    // View Mode: Trial
    return (
        <div className="grid grid-cols-12 gap-6 p-2 animate-fade-in">
            <div className="col-span-12 grid grid-cols-2 lg:grid-cols-4 gap-4 animate-slide-up">
                <TrialStatusCard data={data} trials={trials} />
                <StatBadge label="Perda Atual" value={(latestTrialMetrics.loss != null ? latestTrialMetrics.loss.toFixed(4) : null) || '—'} color="rose" />
                <StatBadge label="MCC Atual" value={(latestTrialMetrics.mcc != null ? latestTrialMetrics.mcc.toFixed(4) : null) || '—'} color="orange" />
                <StatBadge label="MRR Atual" value={(latestTrialMetrics.mrr != null ? latestTrialMetrics.mrr.toFixed(4) : null) || '—'} color="lime" />
            </div>

            <div className="col-span-12 lg:col-span-8 h-[450px] animate-slide-up delay-100">
                <TrialLearningMetricsCard liveData={liveTrialData} />
            </div>
            <div className="col-span-12 lg:col-span-4 h-[450px] animate-slide-up delay-100">
                <FoldConfusionsCard trials={trials} liveStatus={data.liveStatus} charts={data.charts} />
            </div>

            <div className="col-span-12 h-[350px] animate-slide-up delay-200">
                <GeneralizationGapCard liveData={liveTrialData} />
            </div>
        </div>
    );
};
