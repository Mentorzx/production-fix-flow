// @ts-check
import { useState, useRef, useEffect } from "react";
import { Theme } from "../../../ui/Theme.js";

export const TrialStatusCard = ({ data, trials }) => {
    const status = data.liveStatus || {};
    const progress = status.progress || 0;
    const lastTrialId = trials.length > 0 ? trials[trials.length - 1].id : 0;
    const currentTrial = status.trial_number != null ? status.trial_number : lastTrialId;
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

    return (
        <div 
            className="p-6 rounded-xl flex flex-col justify-between h-full min-h-[140px] transition-all duration-300 hover:scale-[1.02]"
            style={{ 
                backgroundColor: 'var(--bg-secondary)', 
                border: '1px solid rgba(229, 197, 88, 0.3)',
                boxShadow: '0 0 20px rgba(229, 197, 88, 0.15)'
            }}
        >
            <div className="flex justify-between items-start mb-2">
                <span 
                    className="text-[10px] font-black uppercase tracking-[0.2em]"
                    style={{ color: Theme.palette.hotOrange }}
                >
                    Progresso HPO
                </span>
                <span 
                    className="text-xs font-mono"
                    style={{ color: Theme.ui.text.secondary }}
                >
                    {progress.toFixed(0)}%
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
                <div className="h-2 w-full rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bg-tertiary)' }}>
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
                
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[10px] font-mono tracking-tight">
                    <div 
                        className="flex justify-between border-r pr-2"
                        style={{ borderColor: 'var(--border-default)' }}
                    >
                        <span style={{ color: Theme.ui.text.secondary }}>Fold</span>
                        <span style={{ color: Theme.palette.vividGreen, fontWeight: 'bold' }}>
                            {status.cv_fold_id != null ? status.cv_fold_id : '—'}
                        </span>
                    </div>
                    <div className="flex justify-between pl-2">
                        <span style={{ color: Theme.ui.text.secondary }}>ETA</span>
                        <span style={{ color: Theme.palette.cyberYellow, fontWeight: 'bold' }}>
                            {formatTime(eta)}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
};
