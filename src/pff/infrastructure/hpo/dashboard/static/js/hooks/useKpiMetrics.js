/**
 * Computes all KPI metrics for the dashboard header row.
 *
 * Extracts heavy useMemo chains out of KpiRow so the component
 * focuses on presentation only.
 */
import { useMemo } from "react";
import { useStore } from "../store/store.jsx";
import { DEFAULT_TOTAL_TRIALS } from "../ui/constants.js";

const ROLLING_WINDOW_SIZE = 5;
const KPI_SPARKLINE_MAX_POINTS = 40;

const parseNumber = (v) => {
    if (v === null || v === undefined) return null;
    const n = Number.parseFloat(v);
    return Number.isFinite(n) ? n : null;
};

const calcDeltaPct = (current, prev) => {
    if (typeof current !== "number" || !Number.isFinite(current)) return null;
    if (typeof prev !== "number" || !Number.isFinite(prev)) return null;
    const denom = Math.abs(prev);
    if (denom < 1e-12) return null;
    return ((current - prev) / denom) * 100;
};

export const formatCompactDuration = (seconds) => {
    const s = typeof seconds === "number" && Number.isFinite(seconds) ? seconds : null;
    if (!s || s <= 0) return "—";
    if (s < 60) return `${Math.round(s)}s`;
    if (s < 3600) return `${(s / 60).toFixed(1)} min`;
    return `${(s / 3600).toFixed(1)}h`;
};

export const formatCompactDurationParts = (seconds) => {
    const s = typeof seconds === "number" && Number.isFinite(seconds) ? seconds : null;
    if (!s || s <= 0) return null;
    if (s < 60) return { main: `${Math.round(s)}`, unit: "s" };
    if (s < 3600) return { main: (s / 60).toFixed(1), unit: "min" };
    return { main: (s / 3600).toFixed(1), unit: "h" };
};

/**
 * @returns {Object} All computed KPI values ready for rendering.
 */
export const useKpiMetrics = () => {
    const { viewMode, trials, filteredTrials, data, bestTrialNoWarmstart } = useStore();
    const objectiveDirection = data?.direction === "minimize" ? "down" : "up";

    const completed = useMemo(() => {
        const items = Array.isArray(trials) ? trials : [];
        return items
            .filter(
                (t) =>
                    t &&
                    t.state === "COMPLETE" &&
                    typeof t.duration === "number" &&
                    Number.isFinite(t.duration) &&
                    t.duration > 0.1,
            )
            .sort((a, b) => a.id - b.id);
    }, [trials]);

    const lastDurations = useMemo(() => completed.slice(-10).map((t) => t.duration), [completed]);

    const avgDuration = useMemo(() => {
        const recent = completed.slice(-ROLLING_WINDOW_SIZE);
        if (recent.length < 2) return null;
        const sum = recent.reduce((acc, t) => acc + t.duration, 0);
        const avg = sum / recent.length;
        return Number.isFinite(avg) && avg > 0 ? avg : null;
    }, [completed]);

    const avgDurationPrev = useMemo(() => {
        const prevWindow = completed.slice(-(ROLLING_WINDOW_SIZE + 1), -1);
        if (prevWindow.length < 2) return null;
        const sum = prevWindow.reduce((acc, t) => acc + t.duration, 0);
        const avg = sum / prevWindow.length;
        return Number.isFinite(avg) && avg > 0 ? avg : null;
    }, [completed]);

    const estimatedCompletion = useMemo(() => {
        const totalTrials =
            typeof data?.totalTrials === "number" && data.totalTrials > 0
                ? data.totalTrials
                : DEFAULT_TOTAL_TRIALS;
        if (!avgDuration) return { eta: null, pct: 0, total: totalTrials };
        const remaining = Math.max(0, totalTrials - completed.length);
        const eta = remaining * avgDuration;
        const pct = totalTrials > 0 ? (completed.length / totalTrials) * 100 : 0;
        return { eta, pct, total: totalTrials };
    }, [avgDuration, completed.length, data?.totalTrials]);

    const estimatedCompletionPrev = useMemo(() => {
        const totalTrials =
            typeof data?.totalTrials === "number" && data.totalTrials > 0
                ? data.totalTrials
                : DEFAULT_TOTAL_TRIALS;
        if (!avgDurationPrev) return { eta: null, pct: 0, total: totalTrials };
        const completedPrevCount = Math.max(0, completed.length - 1);
        const remainingPrev = Math.max(0, totalTrials - completedPrevCount);
        const eta = remainingPrev * avgDurationPrev;
        const pct = totalTrials > 0 ? (completedPrevCount / totalTrials) * 100 : 0;
        return { eta, pct, total: totalTrials };
    }, [avgDurationPrev, completed.length, data?.totalTrials]);

    const bestSeries = useMemo(() => {
        const items = Array.isArray(filteredTrials) ? filteredTrials : [];
        const direction = data?.direction || "maximize";
        let incumbent = direction === "minimize" ? Infinity : -Infinity;
        return items
            .filter((t) => t && t.state === "COMPLETE" && typeof t.value === "number" && Number.isFinite(t.value))
            .sort((a, b) => a.id - b.id)
            .map((t) => {
                const val = t.value;
                if (direction === "minimize") incumbent = Math.min(incumbent, val);
                else incumbent = Math.max(incumbent, val);
                return incumbent;
            })
            .slice(-20);
    }, [filteredTrials, data?.direction]);

    const bestDeltaPct = useMemo(() => {
        if (!Array.isArray(bestSeries) || bestSeries.length < 2) return null;
        const prev = bestSeries[bestSeries.length - 2];
        const curr = bestSeries[bestSeries.length - 1];
        return calcDeltaPct(curr, prev);
    }, [bestSeries]);

    const avgDurationDeltaPct = useMemo(() => calcDeltaPct(avgDuration, avgDurationPrev), [avgDuration, avgDurationPrev]);

    const etaDeltaPct = useMemo(
        () => calcDeltaPct(estimatedCompletion.eta, estimatedCompletionPrev.eta),
        [estimatedCompletion.eta, estimatedCompletionPrev.eta],
    );

    const avgDurationParts = useMemo(() => formatCompactDurationParts(avgDuration), [avgDuration]);
    const etaParts = useMemo(() => formatCompactDurationParts(estimatedCompletion.eta), [estimatedCompletion.eta]);

    const liveTrialHistory = useMemo(() => {
        const rows = Array.isArray(data?.liveStatus?.epoch_history) ? data.liveStatus.epoch_history : [];
        return rows.slice(-KPI_SPARKLINE_MAX_POINTS);
    }, [data?.liveStatus?.epoch_history]);

    const liveMetricSeries = useMemo(() => {
        const loss = [];
        const mcc = [];
        const mrr = [];
        for (const e of liveTrialHistory) {
            if (!e || typeof e !== "object") continue;
            const payload = e.metrics && typeof e.metrics === "object" ? e.metrics : e;

            const lossVal = parseNumber(payload.loss ?? payload.train_loss ?? payload.val_loss ?? payload.binary_loss);
            const mccVal = parseNumber(payload.mcc);
            const mrrVal = parseNumber(payload.mrr);

            if (lossVal !== null) loss.push(lossVal);
            if (mccVal !== null) mcc.push(mccVal);
            if (mrrVal !== null) mrr.push(mrrVal);
        }
        return { loss, mcc, mrr };
    }, [liveTrialHistory]);

    const deltaFromSeries = useMemo(() => {
        const delta = (series) => {
            if (!Array.isArray(series) || series.length < 2) return null;
            const prev = series[series.length - 2];
            const curr = series[series.length - 1];
            return calcDeltaPct(curr, prev);
        };
        return {
            loss: delta(liveMetricSeries.loss),
            mcc: delta(liveMetricSeries.mcc),
            mrr: delta(liveMetricSeries.mrr),
        };
    }, [liveMetricSeries]);

    const latestTrialMetrics = useMemo(() => {
        const pickLast = (series) => {
            for (let i = series.length - 1; i >= 0; i -= 1) {
                const v = series[i];
                if (typeof v === "number" && Number.isFinite(v)) return v;
            }
            return null;
        };
        return {
            loss: pickLast(liveMetricSeries.loss),
            mcc: pickLast(liveMetricSeries.mcc),
            mrr: pickLast(liveMetricSeries.mrr),
        };
    }, [liveMetricSeries]);

    return {
        viewMode,
        data,
        trials,
        bestTrialNoWarmstart,
        objectiveDirection,
        bestSeries,
        bestDeltaPct,
        lastDurations,
        avgDuration,
        avgDurationDeltaPct,
        avgDurationParts,
        estimatedCompletion,
        etaDeltaPct,
        etaParts,
        latestTrialMetrics,
        liveMetricSeries,
        deltaFromSeries,
    };
};
