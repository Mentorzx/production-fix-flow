// @ts-check
import { useMemo, useRef, useEffect, useState } from "react";
import { Theme } from "../../../ui/Theme.js";
import { DEFAULT_TOTAL_TRIALS } from "../../../ui/constants.js";
import {
    extractSparklineData,
    buildSparklinePath,
    calculateTrend
} from "../../../ui/StyleUtils.js";

// MetricCard base com design PULSO
export const MetricCard = ({
    label,
    value,
    subtext = "",
    color = "orange",
    history = [],
    metricKey = "",
}) => {
    const themeColors = {
        rose: { main: Theme.palette.red, bg: "rgba(239,68,68,0.15)", border: "rgba(239,68,68,0.3)" },
        orange: { main: Theme.palette.hotOrange, bg: "rgba(249,115,22,0.15)", border: "rgba(249,115,22,0.3)" },
        lime: { main: Theme.palette.vividGreen, bg: "rgba(132,204,22,0.15)", border: "rgba(132,204,22,0.3)" },
        amber: { main: Theme.palette.cyberYellow, bg: "rgba(245,158,11,0.15)", border: "rgba(245,158,11,0.3)" },
    };

    const colors = themeColors[color] || themeColors.orange;
    const [flipKey, setFlipKey] = useState(0);
    const prevValue = useRef(value);

    useEffect(() => {
        if (prevValue.current !== value) {
            setFlipKey((k) => k + 1);
            prevValue.current = value;
        }
    }, [value]);

    const sparklineData = useMemo(() => {
        if (!metricKey || !history) return [];
        return extractSparklineData(history, metricKey);
    }, [history, metricKey]);

    const trend = useMemo(() => {
        if (sparklineData.length < 2) return { trend: "neutral", diff: "—" };
        const current = sparklineData[sparklineData.length - 1]?.y;
        const previous = sparklineData[sparklineData.length - 2]?.y;
        return calculateTrend(current, previous);
    }, [sparklineData]);

    const trendColors = {
        up: color === "rose" || color === "orange" ? "text-rose-400" : "text-lime-400",
        down: color === "rose" || color === "orange" ? "text-lime-400" : "text-rose-400",
        neutral: "text-zinc-500"
    };

    return (
        <div
            className="pff-card p-6 rounded-xl flex flex-col justify-between h-full min-h-[140px] transition-all duration-300 hover:scale-[1.02]"
            style={{
                backgroundColor: 'var(--bg-secondary)',
                border: `1px solid ${colors.border}`,
                boxShadow: `0 0 20px ${colors.border}`
            }}
        >
            <div className="flex justify-between items-start mb-2">
                <span
                    className="text-[10px] font-black uppercase tracking-[0.2em]"
                    style={{ color: colors.main }}
                >
                    {label}
                </span>
                {trend.trend !== "neutral" && (
                    <span className={`text-xs font-bold ${trendColors[trend.trend]}`}>
                        {trend.trend === "up" ? "↑" : "↓"} {trend.diff}
                    </span>
                )}
            </div>

            <div className="flex items-baseline gap-2">
                <div
                    key={flipKey}
                    className="text-4xl font-black font-mono tracking-tighter"
                    style={{ color: Theme.ui.text.primary }}
                >
                    {value}
                </div>
            </div>

            {subtext && (
                <div
                    className="text-[10px] font-bold uppercase mt-2"
                    style={{ color: colors.main, opacity: 0.7 }}
                >
                    {subtext}
                </div>
            )}

            {sparklineData.length > 1 && (
                <div className="mt-3 h-8">
                    <svg width="100%" height="100%" viewBox="0 0 100 20" preserveAspectRatio="none">
                        <path
                            d={buildSparklinePath(sparklineData, 100, 20)}
                            fill="none"
                            stroke={colors.main}
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                        />
                    </svg>
                </div>
            )}
        </div>
    );
};

// Cards específicos para modo Study
export const BestScoreCard = ({ bestTrial, trials }) => {
    const history = useMemo(() => {
        return trials
            .filter(t => t.state === 'COMPLETE' && t.value != null)
            .map(t => ({ x: t.id, y: t.value }))
            .sort((a, b) => a.x - b.x);
    }, [trials]);

    return (
        <MetricCard
            label="Melhor Global"
            value={bestTrial?.value?.toFixed(4) || '0.0000'}
            subtext={`Trial #${bestTrial?.id || 0}`}
            color="lime"
            history={history}
            metricKey="y"
        />
    );
};

export const PaceCard = ({ trials }) => {
    const { value, subtext } = useMemo(() => {
        const completed = trials.filter(t =>
            t.state === 'COMPLETE' &&
            typeof t.duration === 'number' &&
            t.duration > 0.1
        );

        if (completed.length < 2) {
            return { value: "—", subtext: "Aguardando dados" };
        }

        const recent = completed.slice(-5);
        const avgDuration = recent.reduce((a, t) => a + t.duration, 0) / recent.length;

        if (avgDuration <= 0) {
            return { value: "—", subtext: "Calculando..." };
        }

        const value = avgDuration > 60
            ? `${(avgDuration / 60).toFixed(1)} min/trial`
            : `${(60 / avgDuration).toFixed(1)} t/m`;

        return { value, subtext: "Últimos 5 trials" };
    }, [trials]);

    return (
        <MetricCard
            label="Ritmo Médio"
            value={value}
            subtext={subtext}
            color="orange"
        />
    );
};

export const CompletionCard = ({ trials, totalTrials = DEFAULT_TOTAL_TRIALS }) => {
    const { value, subtext } = useMemo(() => {
        const completed = trials.filter(t =>
            t.state === 'COMPLETE' &&
            typeof t.duration === 'number' &&
            t.duration > 0.1
        );

        if (completed.length < 2) {
            return { value: "—", subtext: "Estimando..." };
        }

        const recent = completed.slice(-5);
        const avgDuration = recent.reduce((a, t) => a + t.duration, 0) / recent.length;

        if (!avgDuration || avgDuration <= 0) {
            return { value: "—", subtext: "Calculando..." };
        }

        const remaining = Math.max(0, totalTrials - completed.length);
        const etaSeconds = remaining * avgDuration;

        let value;
        if (etaSeconds < 60) {
            value = `${Math.round(etaSeconds)}s`;
        } else if (etaSeconds < 3600) {
            value = `${Math.round(etaSeconds / 60)}m`;
        } else {
            value = `${(etaSeconds / 3600).toFixed(1)}h`;
        }

        return { value, subtext: "Baseada no ritmo atual" };
    }, [trials, totalTrials]);

    return (
        <MetricCard
            label="Conclusão Estimada"
            value={value}
            subtext={subtext}
            color="amber"
        />
    );
};

// Cards específicos para modo Trial
export const LossMetricCard = ({ value, history }) => (
    <MetricCard
        label="Perda Atual"
        value={value != null ? value.toFixed(4) : '—'}
        color="rose"
        history={history}
        metricKey="loss"
    />
);

export const MccMetricCard = ({ value, history }) => (
    <MetricCard
        label="MCC Atual"
        value={value != null ? value.toFixed(4) : '—'}
        color="orange"
        history={history}
        metricKey="mcc"
    />
);

export const MrrMetricCard = ({ value, history }) => (
    <MetricCard
        label="MRR Atual"
        value={value != null ? value.toFixed(4) : '—'}
        color="lime"
        history={history}
        metricKey="mrr"
    />
);
