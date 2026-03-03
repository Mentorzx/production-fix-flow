/**
 * Provide store module functionality for the HPO dashboard.
 */

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useMemo,
  useTransition,
  useRef,
} from "react";
import { RUNNING_STALENESS_MS } from "../ui/constants.js";

const StoreStateContext = createContext();
const StoreDispatchContext = createContext();

const INITIAL_DATA = {
  studyName: "Initializing...",
  trials: [],
  totalTrials: 0,
  liveStatus: null,
  updatedAt: new Date().toISOString(),
};

/**
 * Expose store provider for dashboard usage.
 */
export const StoreProvider = ({ children }) => {
  const [data, setData] = useState(INITIAL_DATA);
  const [activeTab, setActiveTab] = useState("overview");
  const [viewMode, setViewMode] = useState("study");
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());
  const [isRunning, setIsRunning] = useState(false);
  const [filters, setFilters] = useState({
    includeWarmup: false,
    includePruned: true,
    onlyComplete: false,
    minScore: null,
    maxScore: null,
  });

  const [isPending, startTransition] = useTransition();
  const pendingDataRef = useRef(null);
  const frameRef = useRef(0);

  // Stream de dados via SSE
  useEffect(() => {
    const eventSource = new EventSource("/api/events");

    const flushPendingData = () => {
      frameRef.current = 0;
      const pending = pendingDataRef.current;
      if (!pending) return;
      pendingDataRef.current = null;
      setData((prev) => {
        if (
          prev?.updatedAt === pending?.updatedAt &&
          prev?.liveStatus?.updated_at === pending?.liveStatus?.updated_at &&
          (prev?.trials?.length || 0) === (pending?.trials?.length || 0)
        ) {
          return prev;
        }
        return pending;
      });
    };

    eventSource.onmessage = (event) => {
      try {
        const jsonData = JSON.parse(event.data);
        pendingDataRef.current = jsonData;
        if (!frameRef.current) {
          frameRef.current = requestAnimationFrame(flushPendingData);
        }
      } catch (error) {
        console.debug("SSE parse error:", error);
      }
    };

    eventSource.onerror = (error) => {
      // EventSource reconnects automatically; avoid warning noise for transient disconnects.
      console.info("SSE connection interrupted; browser auto-reconnect is active.", error);
    };

    return () => {
      eventSource.close();
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = 0;
      }
      pendingDataRef.current = null;
    };
  }, []);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const candidate = data.liveStatus?.updated_at || data.updatedAt;
    if (!candidate) {
      setIsRunning(false);
      return;
    }
    const lastUpdate = new Date(candidate).getTime();
    const now = Date.now();
    // Consider running if update was less than 30s ago
    setIsRunning(now - lastUpdate < RUNNING_STALENESS_MS);
  }, [data.liveStatus?.updated_at, data.updatedAt]);

  // Selectors / Derived State
  const trials = useMemo(() => data.trials || [], [data.trials]);
  const sortedTrials = useMemo(() => [...trials].sort((a, b) => a.id - b.id), [trials]);

  const filteredTrials = useMemo(() => {
    return sortedTrials.filter((t) => {
      if (!filters.includeWarmup && t.warmstart) return false;
      if (!filters.includePruned && t.state === "PRUNED") return false;
      if (filters.onlyComplete && t.state !== "COMPLETE") return false;
      if (filters.minScore !== null && t.value < filters.minScore) return false;
      if (filters.maxScore !== null && t.value > filters.maxScore) return false;
      return true;
    });
  }, [sortedTrials, filters]);

  const bestTrial = useMemo(() => {
    const candidates = filteredTrials.filter(
      (t) =>
        String(t?.state || "").toUpperCase() === "COMPLETE" &&
        typeof t?.value === "number" &&
        Number.isFinite(t.value)
    );
    if (candidates.length === 0) return { id: 0, value: 0, params: {} };

    const direction = data.direction || "maximize";
    return candidates.reduce((prev, current) => {
      const prevVal = prev.value;
      const currVal = current.value;
      if (direction === "minimize") {
        return currVal < prevVal ? current : prev;
      }
      return currVal > prevVal ? current : prev;
    }, candidates[0]);
  }, [filteredTrials, data.direction]);

  const bestTrialNoWarmstart = useMemo(() => {
    const candidates = filteredTrials.filter(
      (t) =>
        String(t?.state || "").toUpperCase() === "COMPLETE" &&
        typeof t?.value === "number" &&
        Number.isFinite(t.value)
    );
    if (candidates.length === 0) return { id: 0, value: 0, params: {} };

    const noWarm = candidates.filter((t) => !t.warmstart);
    const pool = noWarm.length > 0 ? noWarm : candidates;

    const direction = data.direction || "maximize";
    return pool.reduce((prev, current) => {
      const prevVal = prev.value;
      const currVal = current.value;
      if (direction === "minimize") {
        return currVal < prevVal ? current : prev;
      }
      return currVal > prevVal ? current : prev;
    }, pool[0]);
  }, [filteredTrials, data.direction]);

  const currentTrialId = useMemo(() => {
    const totalTrials =
      Number.isFinite(Number(data.totalTrials)) && Number(data.totalTrials) > 0
        ? Math.trunc(Number(data.totalTrials))
        : 50;
    const completedTrialsAll = sortedTrials.filter((trial) => {
      if (!trial) return false;
      const state = String(trial.state || "").toUpperCase();
      return state === "COMPLETE";
    }).length;
    const nextTrialByCompletion = Math.min(
      totalTrials,
      Math.max(1, Math.trunc(completedTrialsAll) + 1)
    );

    const activeTrialIds = sortedTrials
      .filter((trial) => {
        if (!trial || !Number.isFinite(Number(trial.id))) return false;
        const state = String(trial.state || "").toUpperCase();
        return state === "RUNNING" || state === "WAITING";
      })
      .map((trial) => Math.trunc(Number(trial.id)))
      .filter((trialId) => Number.isFinite(trialId) && trialId > 0)
      .sort((a, b) => a - b);
    if (activeTrialIds.length > 0) return Math.min(activeTrialIds[0], nextTrialByCompletion);

    if (Number.isFinite(nextTrialByCompletion) && nextTrialByCompletion > 0) {
      return nextTrialByCompletion;
    }

    const liveId = data.liveStatus?.trial_number;
    if (liveId !== undefined && liveId !== null && Number.isFinite(Number(liveId))) {
      return Math.max(1, Math.trunc(Number(liveId)) + 1);
    }
    if (sortedTrials.length > 0) {
      const lastTrial = sortedTrials[sortedTrials.length - 1];
      if (lastTrial.state === "RUNNING" || lastTrial.state === "WAITING") return lastTrial.id;
      return lastTrial.id + 1;
    }
    return 1;
  }, [data.liveStatus, data.totalTrials, sortedTrials]);

  const currentParams = useMemo(() => {
    const liveId = data.liveStatus?.trial_number;
    if (liveId !== undefined && liveId + 1 === currentTrialId) {
      const liveParams = data.liveStatus?.params;
      if (liveParams && Object.keys(liveParams).length > 0) return liveParams;
    }
    const historyTrial = trials.find((t) => t.id === currentTrialId);
    if (historyTrial && historyTrial.params && Object.keys(historyTrial.params).length > 0)
      return historyTrial.params;
    if (trials.length > 0) {
      const lastTrial = trials[trials.length - 1];
      if (lastTrial.params && Object.keys(lastTrial.params).length > 0) return lastTrial.params;
    }
    return {};
  }, [data.liveStatus, trials, currentTrialId]);

  const stateValue = useMemo(
    () => ({
      data,
      activeTab,
      viewMode,
      currentTime,
      isRunning,
      filters,
      isPending,
      trials,
      sortedTrials,
      bestTrial,
      bestTrialNoWarmstart,
      filteredTrials,
      currentTrialId,
      currentParams,
    }),
    [
      data,
      activeTab,
      viewMode,
      currentTime,
      isRunning,
      filters,
      isPending,
      trials,
      sortedTrials,
      bestTrial,
      bestTrialNoWarmstart,
      filteredTrials,
      currentTrialId,
      currentParams,
    ]
  );

  const dispatchValue = useMemo(
    () => ({
      setData,
      // Tab switching must be immediate; transitions can starve under frequent SSE updates.
      setActiveTab: (tab) => setActiveTab(tab),
      // View switching must be immediate; wrapping in a transition can starve under frequent SSE updates.
      setViewMode: (mode) => setViewMode(mode),
      setFilters: (f) => startTransition(() => setFilters(f)),
    }),
    []
  );

  return (
    <StoreStateContext.Provider value={stateValue}>
      <StoreDispatchContext.Provider value={dispatchValue}>
        {children}
      </StoreDispatchContext.Provider>
    </StoreStateContext.Provider>
  );
};

/**
 * Expose use store state for dashboard usage.
 */
export const useStoreState = () => {
  const context = useContext(StoreStateContext);
  if (context === undefined) throw new Error("useStoreState must be used within a StoreProvider");
  return context;
};

/**
 * Expose use store dispatch for dashboard usage.
 */
export const useStoreDispatch = () => {
  const context = useContext(StoreDispatchContext);
  if (context === undefined)
    throw new Error("useStoreDispatch must be used within a StoreProvider");
  return context;
};

/**
 * Expose use store for dashboard usage.
 */
export const useStore = () => {
  const state = useStoreState();
  const dispatch = useStoreDispatch();
  return useMemo(() => ({ ...state, ...dispatch }), [state, dispatch]);
};
