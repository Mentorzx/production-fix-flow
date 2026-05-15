/**
 * Provide TrialStatusCard module functionality for the HPO dashboard.
 */

// @ts-check
import { useState, useRef, useEffect } from "react";
import { Theme } from "../../../ui/Theme.js";
import { PortalTooltip } from "../../../ui/PortalTooltip.jsx";
import { DEFAULT_TOTAL_TRIALS } from "../../../ui/constants.js";
import { AnimatedNumberText } from "../../../ui/AnimatedNumberText.jsx";

/**
 * Expose trial status card for dashboard usage.
 */
export const TrialStatusCard = ({ data, trials, animationSeed = "" }) => {
  const status = data.liveStatus || {};
  const progress = status.progress || 0;
  const sortedTrialIds = [...(Array.isArray(trials) ? trials : [])]
    .filter((trial) => trial && Number.isFinite(Number(trial.id)))
    .map((trial) => ({
      id: Math.trunc(Number(trial.id)),
      state: String(trial.state || "").toUpperCase(),
      warmstart: Boolean(trial.warmstart),
    }))
    .sort((a, b) => a.id - b.id);
  const activeTrialIds = sortedTrialIds
    .filter((trial) => trial.state === "RUNNING" || trial.state === "WAITING")
    .map((trial) => trial.id)
    .filter((trialId) => trialId > 0);
  const completedTrialsAll = sortedTrialIds.filter((trial) => {
    if (!trial || trial.id <= 0) return false;
    return trial.state === "COMPLETE";
  }).length;
  const lastTrialId = sortedTrialIds.length > 0 ? sortedTrialIds[sortedTrialIds.length - 1].id : 1;
  const totalTrials =
    Number.isFinite(Number(data.totalTrials)) && Number(data.totalTrials) > 0
      ? Math.trunc(Number(data.totalTrials))
      : DEFAULT_TOTAL_TRIALS;
  const liveTrialId =
    status.trial_number != null && Number.isFinite(Number(status.trial_number))
      ? Math.min(totalTrials, Math.max(1, Math.trunc(Number(status.trial_number)) + 1))
      : null;
  const nextTrialByCompletion = Math.min(totalTrials, Math.max(1, completedTrialsAll + 1));
  const currentTrial =
    activeTrialIds.length > 0
      ? Math.min(activeTrialIds[0], nextTrialByCompletion)
      : completedTrialsAll > 0
      ? nextTrialByCompletion
      : liveTrialId != null
      ? liveTrialId
      : nextTrialByCompletion > 0
      ? nextTrialByCompletion
      : lastTrialId;
  const currentEpoch = typeof status.current_epoch === "number" ? status.current_epoch : null;
  const totalEpochs = typeof status.total_epochs === "number" ? status.total_epochs : null;
  const totalFolds = Number.isFinite(Number(data.totalFolds))
    ? Math.max(1, Math.trunc(Number(data.totalFolds)))
    : null;
  const parsedFold =
    status.cv_fold_id != null && Number.isFinite(Number(status.cv_fold_id))
      ? Math.trunc(Number(status.cv_fold_id))
      : null;
  const displayFold =
    parsedFold == null
      ? null
      : totalFolds != null && (parsedFold < 0 || parsedFold >= totalFolds)
        ? 0
        : parsedFold;
  const [pulseKey, setPulseKey] = useState(0);
  const prevProgress = useRef(progress);
  const [clockTick, setClockTick] = useState(() => Date.now());

  const formatTime = (s) => {
    if (!s || s <= 0) return "--:--";
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return m > 0 ? `${m}m ${sec.toString().padStart(2, "0")}s` : `${sec}s`;
  };
  const formatElapsed = (s) => {
    if (!Number.isFinite(s) || s < 0) return "—";
    const totalMs = Math.floor(s * 1000);
    const hours = Math.floor(totalMs / 3_600_000);
    const minutes = Math.floor((totalMs % 3_600_000) / 60_000);
    const seconds = Math.floor((totalMs % 60_000) / 1000);
    const ms = totalMs % 1000;
    return `${String(hours).padStart(2, "0")}h ${String(minutes).padStart(2, "0")}m ${String(seconds).padStart(2, "0")}s ${String(ms).padStart(3, "0")}ms`;
  };

  const eta =
    status.elapsed_seconds && progress > 0
      ? status.elapsed_seconds / (progress / 100) - status.elapsed_seconds
      : null;
  const trialElapsed = Number(status.elapsed_seconds);
  const statusUpdatedAtMs = Date.parse(String(status.updated_at || ""));
  const driftSeconds = Number.isFinite(statusUpdatedAtMs)
    ? Math.max(0, (clockTick - statusUpdatedAtMs) / 1000)
    : 0;
  const trialElapsedLabel = formatElapsed(
    (Number.isFinite(trialElapsed) && trialElapsed > 0 ? trialElapsed : 0) + driftSeconds
  );

  useEffect(() => {
    if (Math.abs(progress - prevProgress.current) >= 0.5) {
      setPulseKey((k) => k + 1);
      prevProgress.current = progress;
    }
  }, [progress]);

  useEffect(() => {
    if (animationSeed == null || animationSeed === "") return;
    setPulseKey((k) => k + 1);
  }, [animationSeed]);

  useEffect(() => {
    const timerId = setInterval(() => setClockTick(Date.now()), 100);
    return () => clearInterval(timerId);
  }, []);

  const tooltipContent = (
    <div
      className="w-72 border p-3 rounded-xl shadow-2xl text-[10px]"
      style={{
        backgroundColor: Theme.ui.background,
        borderColor: Theme.ui.border,
        color: Theme.ui.text.secondary,
      }}
    >
      <div className="space-y-2">
        <div>
          <span
            className="text-[8px] font-black uppercase block mb-1"
            style={{ color: Theme.semantic.warning }}
          >
            Explicação Técnica
          </span>
          <div className="leading-tight" style={{ color: Theme.ui.text.primary }}>
            Progresso do HPO no trial atual, incluindo fold em execução, época atual (e total), e
            ETA estimado.
          </div>
        </div>
        <div className="pt-2 border-t" style={{ borderColor: Theme.ui.border }}>
          <span
            className="text-[8px] font-black uppercase block mb-1"
            style={{ color: Theme.palette.cyberYellow }}
          >
            Valores
          </span>
          <div className="space-y-1">
            <div
              className="text-[10px] leading-tight flex gap-2"
              style={{ color: Theme.ui.text.secondary }}
            >
              <span className="font-semibold min-w-[72px]" style={{ color: Theme.palette.apricot }}>
                Trial:
              </span>
              <span>{currentTrial}</span>
            </div>
            <div
              className="text-[10px] leading-tight flex gap-2"
              style={{ color: Theme.ui.text.secondary }}
            >
              <span className="font-semibold min-w-[72px]" style={{ color: Theme.palette.apricot }}>
                Época:
              </span>
              <span>
                {currentEpoch != null
                  ? `${currentEpoch}${totalEpochs ? `/${totalEpochs}` : ""}`
                  : "—"}
              </span>
            </div>
            <div
              className="text-[10px] leading-tight flex gap-2"
              style={{ color: Theme.ui.text.secondary }}
            >
              <span className="font-semibold min-w-[72px]" style={{ color: Theme.palette.apricot }}>
                Fold:
              </span>
              <span>{displayFold != null ? displayFold : "—"}</span>
            </div>
            <div
              className="text-[10px] leading-tight flex gap-2"
              style={{ color: Theme.ui.text.secondary }}
            >
              <span className="font-semibold min-w-[72px]" style={{ color: Theme.palette.apricot }}>
                Progresso:
              </span>
              <span>
                {typeof progress === "number" && Number.isFinite(progress)
                  ? `${progress.toFixed(1)}%`
                  : "—"}
              </span>
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
          boxShadow: `0 0 28px ${Theme.palette.cyberYellow}14`,
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
            className="text-[10px] font-mono tabular-nums"
            style={{ color: Theme.ui.text.secondary, opacity: 0.75 }}
          >
            {trialElapsedLabel}
          </span>
        </div>

        <div className="flex items-baseline gap-2 mb-4">
          <div
            key={`${pulseKey}-${currentTrial}`}
            className="text-4xl font-black font-mono tracking-tighter"
          >
            <span style={{ color: Theme.ui.text.primary }}>Trial #</span>
            <AnimatedNumberText
              value={currentTrial}
              seed={animationSeed}
              className="tabular-nums"
              style={{ color: Theme.ui.text.primary }}
            />
          </div>
          <span
            className="text-lg font-normal"
            style={{ color: Theme.ui.text.secondary, opacity: 0.6 }}
          >
            / {totalTrials}
          </span>
        </div>

        <div className="space-y-3">
          <div
            className="h-2 w-full rounded-full overflow-hidden"
            style={{ backgroundColor: Theme.ui.border }}
          >
            <div
              key={pulseKey}
              className="h-full transition-all duration-1000"
              style={{
                width: `${progress}%`,
                background: `linear-gradient(90deg, ${Theme.palette.hotOrange}, ${Theme.palette.cyberYellow})`,
                boxShadow: `0 0 10px ${Theme.palette.hotOrange}`,
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
                {displayFold != null ? displayFold : "—"}
              </span>
            </div>
            <div
              className="flex justify-between border-r px-2"
              style={{ borderColor: Theme.ui.border }}
            >
              <span style={{ color: Theme.ui.text.secondary }}>Época</span>
              <span style={{ color: Theme.palette.neonBlue, fontWeight: "bold" }}>
                {currentEpoch != null
                  ? `${currentEpoch}${totalEpochs ? `/${totalEpochs}` : ""}`
                  : "—"}
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
