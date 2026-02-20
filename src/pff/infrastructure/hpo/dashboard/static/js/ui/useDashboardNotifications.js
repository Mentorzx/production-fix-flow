import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { resolveMetricValue } from "../domain/metrics/Formatters.js";
import { CLF_KEYS, RANKING_KEYS } from "../domain/metrics/ColumnFactory.js";
import { MetricRegistry } from "../domain/metrics/MetricRegistry.js";

const MAX_TOASTS = 5;
const MAX_HISTORY = 120;
const TOAST_DURATION_MS = 7000;
const EPS = 1e-9;
const HISTORY_STORAGE_KEY = "pff-dashboard-notifications-history-v1";
const SEEN_STORAGE_KEY = "pff-dashboard-notifications-seen-v1";
const DISMISSED_STORAGE_KEY = "pff-dashboard-notifications-dismissed-v1";

const asFinite = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};

const metricValueFromTrial = (trial, key) => asFinite(resolveMetricValue(trial, key));

const RECORD_METRIC_KEYS = [...new Set([...CLF_KEYS, ...RANKING_KEYS])];
const METRIC_LABELS_PT = {
  accuracy: "Acurácia",
  precision: "Precisão",
  recall: "Recall",
  mcc: "MCC",
  f1: "F1",
  auc: "AUC",
  pr_auc: "PR-AUC",
  mrr: "MRR",
  hits1: "Hits@1",
  hits3: "Hits@3",
  hits10: "Hits@10",
};

const metricLabel = (key) => METRIC_LABELS_PT[key] || String(key).toUpperCase();
const metricDirection = (key) => MetricRegistry.get(key)?.direction || "up";
const formatPct = (value) => `${(Number(value) * 100).toFixed(1)}%`;

const normalizeLogEntry = (entry) => {
  if (entry == null) return null;
  if (typeof entry === "string") {
    return {
      timestamp: "",
      level: "WARNING",
      message: entry,
    };
  }
  if (typeof entry !== "object") return null;
  return {
    timestamp: String(entry.timestamp || entry.time || ""),
    level: String(entry.level || entry.severity || "WARNING").toUpperCase(),
    message: String(entry.message || entry.msg || ""),
  };
};

const levelToType = (level) => {
  const upper = String(level || "").toUpperCase();
  if (upper.includes("ERROR") || upper.includes("CRIT")) return "danger";
  if (upper.includes("WARN")) return "warning";
  if (upper.includes("SUCCESS")) return "success";
  return "warning";
};
const isExecutionAlertLevel = (level) => {
  const upper = String(level || "").toUpperCase();
  return upper.includes("WARN") || upper.includes("ERROR") || upper.includes("CRIT");
};

const getBestWorst = (completed, direction) => {
  if (completed.length === 0) return { best: null, worst: null };
  const maximize = direction !== "minimize";
  const best = completed.reduce((acc, cur) =>
    maximize ? (cur.value > acc.value ? cur : acc) : cur.value < acc.value ? cur : acc
  );
  const worst = completed.reduce((acc, cur) =>
    maximize ? (cur.value < acc.value ? cur : acc) : cur.value > acc.value ? cur : acc
  );
  return { best, worst };
};

const buildOverfitSignal = (liveStatus) => {
  const trial = asFinite(liveStatus?.trial_number);
  const rows = Array.isArray(liveStatus?.epoch_history) ? liveStatus.epoch_history : [];
  const evalPoints = rows
    .map((row) => {
      const train = asFinite(row?.train_loss ?? row?.loss);
      const val = asFinite(row?.val_loss ?? row?.validation_loss);
      const epoch = asFinite(row?.epoch);
      if (train == null || val == null || epoch == null) return null;
      return { train, val, epoch };
    })
    .filter(Boolean);

  if (evalPoints.length < 3) return null;
  const last3 = evalPoints.slice(-3);
  const [p0, p1, p2] = last3;
  const trainDown = p2.train <= p1.train + EPS && p1.train <= p0.train + EPS;
  const valUp = p2.val >= p1.val - EPS && p1.val >= p0.val - EPS;
  const gap = p2.val - p2.train;
  if (!trainDown || !valUp || gap < 0.15) return null;
  return {
    key: `overfit:${trial ?? "x"}:${p2.epoch}`,
    gap,
    epoch: p2.epoch,
    trial,
  };
};

const readStoredJson = (key, fallback) => {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw);
    return parsed ?? fallback;
  } catch {
    return fallback;
  }
};

/**
 * Centralized notification engine for dashboard events.
 */
export const useDashboardNotifications = (data) => {
  const [toasts, setToasts] = useState([]);
  const [dismissedMap, setDismissedMap] = useState(() => {
    const stored = readStoredJson(DISMISSED_STORAGE_KEY, {});
    return stored && typeof stored === "object" ? stored : {};
  });
  const [history, setHistory] = useState(() => {
    const storedDismissed = readStoredJson(DISMISSED_STORAGE_KEY, {});
    const stored = readStoredJson(HISTORY_STORAGE_KEY, []);
    if (!Array.isArray(stored)) return [];
    return stored
      .filter(
        (item) =>
          item && typeof item === "object" && item.id && item.key && !storedDismissed?.[item.key]
      )
      .slice(0, MAX_HISTORY);
  });
  const [seenMap, setSeenMap] = useState(() => {
    const stored = readStoredJson(SEEN_STORAGE_KEY, {});
    return stored && typeof stored === "object" ? stored : {};
  });
  const [nowMs, setNowMs] = useState(() => Date.now());
  const initializedRef = useRef(false);
  const seenRef = useRef(new Set());
  const prevRef = useRef({
    lastLogSig: null,
    bestId: null,
    worstId: null,
    metricRecords: {},
  });

  useEffect(() => {
    const timer = setInterval(() => setNowMs(Date.now()), 100);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    setToasts((items) => items.filter((item) => item.expiresAt > nowMs));
  }, [nowMs]);

  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history.slice(0, MAX_HISTORY)));
    } catch {
      // ignore storage errors
    }
  }, [history]);

  useEffect(() => {
    try {
      localStorage.setItem(SEEN_STORAGE_KEY, JSON.stringify(seenMap));
    } catch {
      // ignore storage errors
    }
  }, [seenMap]);

  useEffect(() => {
    try {
      localStorage.setItem(DISMISSED_STORAGE_KEY, JSON.stringify(dismissedMap));
    } catch {
      // ignore storage errors
    }
  }, [dismissedMap]);

  const emit = useCallback(
    (payload) => {
      const key = String(payload.key || `${payload.type}:${payload.title}:${payload.message}`);
      if (dismissedMap[key]) return;
      if (seenRef.current.has(key)) return;
      seenRef.current.add(key);
      if (seenRef.current.size > 1000) {
        const next = new Set(Array.from(seenRef.current).slice(-500));
        seenRef.current = next;
      }

      const createdAt = Date.now();
      const durationMs = payload.durationMs ?? TOAST_DURATION_MS;
      const note = {
        id: `${key}:${createdAt}`,
        key,
        title: payload.title,
        message: payload.message,
        type: payload.type || "warning",
        createdAt,
        expiresAt: createdAt + durationMs,
        durationMs,
      };

      setToasts((items) =>
        [note, ...items.filter((item) => item.key !== key)].slice(0, MAX_TOASTS)
      );
      setHistory((items) =>
        [note, ...items.filter((item) => item.key !== key)].slice(0, MAX_HISTORY)
      );
    },
    [dismissedMap]
  );

  useEffect(() => {
    const liveStatus = data?.liveStatus || {};
    const trialRows = Array.isArray(data?.trials) ? data.trials : [];
    const completed = trialRows.filter(
      (trial) => trial?.state === "COMPLETE" && asFinite(trial?.value) != null
    );
    const direction = data?.direction === "minimize" ? "minimize" : "maximize";

    const logs = Array.isArray(liveStatus?.logs)
      ? liveStatus.logs
      : Array.isArray(liveStatus?.recent_logs)
        ? liveStatus.recent_logs
        : [];
    const alertLogs = logs
      .map(normalizeLogEntry)
      .filter((log) => log && isExecutionAlertLevel(log.level));
    const lastLog = alertLogs.length > 0 ? alertLogs[alertLogs.length - 1] : null;
    const lastLogSig = lastLog
      ? `${lastLog.timestamp || ""}|${lastLog.level || ""}|${lastLog.message || ""}`
      : null;

    const { best, worst } = getBestWorst(completed, direction);

    const metricRecords = {};
    for (const key of RECORD_METRIC_KEYS) {
      const values = completed
        .map((trial) => metricValueFromTrial(trial, key))
        .filter((value) => value != null);
      if (values.length > 0) {
        const dir = metricDirection(key);
        metricRecords[key] = {
          key,
          label: metricLabel(key),
          direction: dir,
          value: dir === "down" ? Math.min(...values) : Math.max(...values),
        };
      }
    }

    const overfitSignal = buildOverfitSignal(liveStatus);
    const adviceMeta = data?.searchSpaceAdvice?.metadata || {};
    const reliability = adviceMeta?.reliability_summary || {};
    const selfAudit = adviceMeta?.self_audit || {};
    const adviceTrialCount =
      asFinite(adviceMeta?.n_completed_trials) ?? asFinite(completed.length) ?? 0;
    const sourceLastTrial =
      asFinite(selfAudit?.source_last_trial) ?? asFinite(adviceMeta?.last_trial) ?? -1;

    if (!initializedRef.current) {
      prevRef.current = {
        lastLogSig,
        bestId: best?.id ?? null,
        worstId: worst?.id ?? null,
        metricRecords,
      };
      initializedRef.current = true;
      return;
    }

    if (lastLogSig && lastLogSig !== prevRef.current.lastLogSig && lastLog?.message) {
      emit({
        key: `log:${lastLogSig}`,
        title: "Novo log de execução",
        message: String(lastLog.message),
        type: levelToType(lastLog.level),
      });
      prevRef.current.lastLogSig = lastLogSig;
    }

    if (best?.id != null && best.id !== prevRef.current.bestId) {
      emit({
        key: `best:${best.id}`,
        title: "Novo melhor trial",
        message: `Trial #${best.id} assumiu a liderança (${Number(best.value).toFixed(4)}).`,
        type: "success",
      });
      prevRef.current.bestId = best.id;
    }

    if (worst?.id != null && worst.id !== prevRef.current.worstId) {
      emit({
        key: `worst:${worst.id}`,
        title: "Novo pior trial",
        message: `Trial #${worst.id} registrou o pior score atual (${Number(worst.value).toFixed(4)}).`,
        type: "warning",
      });
      prevRef.current.worstId = worst.id;
    }

    for (const [key, info] of Object.entries(metricRecords)) {
      const prev = prevRef.current.metricRecords[key]?.value;
      const improved =
        prev == null
          ? true
          : info.direction === "down"
            ? info.value < prev - 1e-6
            : info.value > prev + 1e-6;
      if (improved) {
        if (key !== "score") {
          emit({
            key: `record:${key}:${info.value.toFixed(6)}`,
            title: `Recorde de ${info.label}`,
            message: `${info.label} atingiu novo pico: ${info.value.toFixed(4)}.`,
            type: "success",
          });
        }
      }
    }
    prevRef.current.metricRecords = metricRecords;

    if (overfitSignal) {
      emit({
        key: overfitSignal.key,
        title: "Sinal de overfitting",
        message: `Gap train/val elevado na época ${overfitSignal.epoch} (Δ=${overfitSignal.gap.toFixed(4)}).`,
        type: "danger",
      });
    }

    if (adviceTrialCount >= 10) {
      const validationLb = asFinite(reliability?.validation_pass_wilson_lb);
      if (validationLb != null && validationLb < 0.65) {
        emit({
          key: `advisor:validation_lb:${sourceLastTrial}:${validationLb.toFixed(3)}`,
          title: "Confiabilidade do advisor em queda",
          message: `Limite inferior de validação em ${formatPct(validationLb)}. Revise recomendações antes de aplicar.`,
          type: validationLb < 0.5 ? "danger" : "warning",
        });
      }

      const meanConfidence = asFinite(reliability?.mean_confidence_score);
      if (meanConfidence != null && meanConfidence < 0.45) {
        emit({
          key: `advisor:confidence:${sourceLastTrial}:${meanConfidence.toFixed(3)}`,
          title: "Confiança média baixa",
          message: `Confiança média das recomendações em ${formatPct(meanConfidence)}.`,
          type: "warning",
        });
      }
    }

    const villainsCount = asFinite(selfAudit?.villains_count);
    const villains = Array.isArray(selfAudit?.villains) ? selfAudit.villains : [];
    if (villainsCount != null && villainsCount > 0 && villains.length > 0) {
      const topVillain = villains[0] || {};
      const param = String(topVillain?.param_name || "?");
      const action = String(topVillain?.action || "?");
      const lb = asFinite(topVillain?.hit_rate_wilson_lb);
      emit({
        key: `advisor:self_audit_villain:${sourceLastTrial}:${param}:${action}:${lb ?? "x"}`,
        title: "Ação degradante detectada",
        message:
          lb != null
            ? `${param} (${action}) com Wilson LB=${formatPct(lb)}. Recomendações desse padrão estão sendo bloqueadas.`
            : `${param} (${action}) detectado como degradante no auto-audit.`,
        type: "danger",
      });
    }

    const blockedActions = asFinite(selfAudit?.blocked_actions_current);
    if (blockedActions != null && blockedActions > 0) {
      emit({
        key: `advisor:self_audit_blocked:${sourceLastTrial}:${blockedActions}`,
        title: "Bloqueio automático ativo",
        message: `${blockedActions} recomendação(ões) bloqueadas pelo auto-audit por baixa confiabilidade.`,
        type: "warning",
      });
    }
  }, [data, emit]);

  const dismissToast = useCallback(
    (id) => {
      setToasts((items) => items.filter((item) => item.id !== id));
      const note = history.find((item) => item.id === id);
      if (!note?.key) return;
      setSeenMap((prev) => (prev[note.key] ? prev : { ...prev, [note.key]: true }));
    },
    [history]
  );

  const markAllSeen = useCallback(() => {
    if (history.length === 0) return;
    setSeenMap((prev) => {
      const next = { ...prev };
      for (const item of history) {
        if (item?.key) next[item.key] = true;
      }
      return next;
    });
  }, [history]);

  const markSeen = useCallback(
    (id) => {
      if (!id) return;
      const note = history.find((item) => item.id === id);
      if (!note?.key) return;
      setSeenMap((prev) => (prev[note.key] ? prev : { ...prev, [note.key]: true }));
    },
    [history]
  );

  const clearHistory = useCallback(() => {
    setDismissedMap((prev) => {
      const next = { ...prev };
      for (const item of history) {
        if (item?.key) next[item.key] = true;
      }
      return next;
    });
    setHistory([]);
    setToasts([]);
  }, [history]);

  const removeHistoryItem = useCallback(
    (id) => {
      if (!id) return;
      const note = history.find((item) => item.id === id);
      if (note?.key) {
        setDismissedMap((prev) => (prev[note.key] ? prev : { ...prev, [note.key]: true }));
        setSeenMap((prev) => (prev[note.key] ? prev : { ...prev, [note.key]: true }));
      }
      setHistory((items) => items.filter((item) => item.id !== id));
      setToasts((items) => items.filter((item) => item.id !== id));
    },
    [history]
  );

  const unseenCount = useMemo(
    () => history.reduce((acc, item) => (seenMap[item.key] ? acc : acc + 1), 0),
    [history, seenMap]
  );

  return useMemo(
    () => ({
      toasts,
      history,
      unseenCount,
      nowMs,
      dismissToast,
      markAllSeen,
      markSeen,
      clearHistory,
      removeHistoryItem,
    }),
    [
      toasts,
      history,
      unseenCount,
      nowMs,
      dismissToast,
      markAllSeen,
      markSeen,
      clearHistory,
      removeHistoryItem,
    ]
  );
};
