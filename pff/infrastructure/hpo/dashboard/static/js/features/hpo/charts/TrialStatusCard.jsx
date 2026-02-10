// @ts-check
import { useState, useRef, useEffect } from "react";
import { Theme } from "../../../ui/Theme.js";
import { PortalTooltip } from "../../../ui/BaseComponents.jsx";

export const TrialStatusCard = ({ data, trials }) => {
    const status = data.liveStatus || {};
    const progress = status.progress || 0;
    const lastTrialId = trials.length > 0 ? trials[trials.length - 1].id : 1;
    const currentTrial = status.trial_number != null ? Number(status.trial_number) + 1 : lastTrialId;
    const currentEpoch = typeof status.current_epoch === "number" ? status.current_epoch : null;
    const totalEpochs = typeof status.total_epochs === "number" ? status.total_epochs : null;
    const [pulseKey, setPulseKey] = useState(0);
    const prevProgress = useRef(progress);

    const formatTime = (s) => {
        if (!s || s <= 0) return '--:--';
        const m = Math.floor(s / 60);
        const sec = Math.floor(s % 60);
        return m > 0 ? `${m}m ${sec.toString().padStart(2, '0')}s` : `${sec}s`;
    };

    const eta = status.elapsed_seconds && progress > 0 
        ? (status.elapsed_seconds / (progress / 100)) - status.elapsed_seconds 
        : null;

    useEffect(() => {
        if (Math.abs(progress - prevProgress.current) >= 0.5) {
            setPulseKey((k) => k + 1);
            prevProgress.current = progress;
        }
    }, [progress]);

    const tooltipContent = (
        <div className="w-72 border p-3 rounded-xl shadow-2xl text-[10px]" style={{ backgroundColor: Theme.ui.background, borderColor: Theme.ui.border, color: Theme.ui.text.secondary }}>
            <div className="space-y-2">
                <div>
                    <span className="text-[8px] font-black uppercase block mb-1" style={{ color: Theme.semantic.warning }}>Explicação Técnica</span>
                    <div className="leading-tight" style={{ color: Theme.ui.text.primary }}>
                        Progresso do HPO no trial atual, incluindo fold em execução, época atual (e total), e ETA estimado.
                    </div>
                </div>
                <div className="pt-2 border-t" style={{ borderColor: Theme.ui.border }}>
                    <span className="text-[8px] font-black uppercase block mb-1" style={{ color: Theme.palette.cyberYellow }}>Valores</span>
                    <div className="space-y-1">
                        <div className="text-[10px] leading-tight flex gap-2" style={{ color: Theme.ui.text.secondary }}>
                            <span className="font-semibold min-w-[72px]" style={{ color: Theme.palette.apricot }}>Trial:</span>
                            <span>{currentTrial}</span>
                        </div>
                        <div className="text-[10px] leading-tight flex gap-2" style={{ color: Theme.ui.text.secondary }}>
                            <span className="font-semibold min-w-[72px]" style={{ color: Theme.palette.apricot }}>Época:</span>
                            <span>{currentEpoch != null ? `${currentEpoch}${totalEpochs ? `/${totalEpochs}` : ""}` : "—"}</span>
                        </div>
                        <div className="text-[10px] leading-tight flex gap-2" style={{ color: Theme.ui.text.secondary }}>
                            <span className="font-semibold min-w-[72px]" style={{ color: Theme.palette.apricot }}>Fold:</span>
                            <span>{status.cv_fold_id != null ? status.cv_fold_id : "—"}</span>
                        </div>
                        <div className="text-[10px] leading-tight flex gap-2" style={{ color: Theme.ui.text.secondary }}>
                            <span className="font-semibold min-w-[72px]" style={{ color: Theme.palette.apricot }}>Progresso:</span>
                            <span>{typeof progress === "number" && Number.isFinite(progress) ? `${progress.toFixed(1)}%` : "—"}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );

    return (
        <PortalTooltip content={tooltipContent} className="block w-full h-full">
            <div
                className="p-5 rounded-2xl flex flex-col justify-between h-full min-h-[140px] cursor-help transition-transform duration-200 ease-out hover:scale-[1.015] hover:brightness-110"
                style={{
                    backgroundColor: Theme.ui.surface,
                    border: `1px solid ${Theme.palette.cyberYellow}33`,
                    boxShadow: `0 0 28px ${Theme.palette.cyberYellow}14`
                }}
            >
                <div className="flex justify-between items-start mb-2">
                    <span
                        className="text-[10px] font-black uppercase tracking-[0.2em]"
                        style={{ color: Theme.palette.hotOrange }}
                    >
                        Progresso HPO
                    </span>
                </div>

                <div className="flex items-baseline gap-2 mb-4">
                    <div
                        className="text-4xl font-black font-mono tracking-tighter"
                        style={{ color: Theme.ui.text.primary }}
                    >
                        Trial #{currentTrial}
                    </div>
                    <span
                        className="text-lg font-normal"
                        style={{ color: Theme.ui.text.secondary, opacity: 0.6 }}
                    >
                        / {data.totalTrials || 50}
                    </span>
                </div>

                <div className="space-y-3">
                    <div className="h-2 w-full rounded-full overflow-hidden" style={{ backgroundColor: Theme.ui.border }}>
                        <div
                            key={pulseKey}
                            className="h-full transition-all duration-1000"
                            style={{
                                width: `${progress}%`,
                                background: `linear-gradient(90deg, ${Theme.palette.hotOrange}, ${Theme.palette.cyberYellow})`,
                                boxShadow: `0 0 10px ${Theme.palette.hotOrange}`
                            }}
                        />
                    </div>

                    <div className="grid grid-cols-3 gap-x-4 gap-y-1 text-[10px] font-mono tracking-tight">
                        <div
                            className="flex justify-between border-r pr-2"
                            style={{ borderColor: Theme.ui.border }}
                        >
                            <span style={{ color: Theme.ui.text.secondary }}>Fold</span>
                            <span style={{ color: Theme.palette.vividGreen, fontWeight: "bold" }}>
                                {status.cv_fold_id != null ? status.cv_fold_id : "—"}
                            </span>
                        </div>
                        <div
                            className="flex justify-between border-r px-2"
                            style={{ borderColor: Theme.ui.border }}
                        >
                            <span style={{ color: Theme.ui.text.secondary }}>Época</span>
                            <span style={{ color: Theme.palette.neonBlue, fontWeight: "bold" }}>
                                {currentEpoch != null ? `${currentEpoch}${totalEpochs ? `/${totalEpochs}` : ""}` : "—"}
                            </span>
                        </div>
                        <div className="flex justify-between pl-2">
                            <span style={{ color: Theme.ui.text.secondary }}>ETA</span>
                            <span style={{ color: Theme.palette.cyberYellow, fontWeight: "bold" }}>
                                {formatTime(eta)}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </PortalTooltip>
    );
};
