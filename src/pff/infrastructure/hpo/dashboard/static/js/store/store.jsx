import { createContext, useContext, useState, useEffect, useMemo, useTransition } from 'react';
import { RUNNING_STALENESS_MS } from '../ui/constants.js';

const StoreStateContext = createContext();
const StoreDispatchContext = createContext();

const INITIAL_DATA = {
    studyName: "Initializing...",
    trials: [],
    totalTrials: 0,
    liveStatus: null,
    updatedAt: new Date().toISOString()
};

export const StoreProvider = ({ children }) => {
    const [data, setData] = useState(INITIAL_DATA);
    const [activeTab, setActiveTab] = useState('overview');
    const [viewMode, setViewMode] = useState('study');
    const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());
    const [isRunning, setIsRunning] = useState(false);
    const [filters, setFilters] = useState({
        includeWarmup: false,
        includePruned: true,
        onlyComplete: false,
        minScore: null,
        maxScore: null
    });

    const [isPending, startTransition] = useTransition();

    // Stream de dados via SSE
    useEffect(() => {
        const eventSource = new EventSource('/api/events');

        eventSource.onmessage = (event) => {
            try {
                const jsonData = JSON.parse(event.data);
                setData(jsonData);
            } catch (error) {
                console.warn("SSE parse error:", error);
            }
        };

        eventSource.onerror = (error) => {
            console.warn("SSE connection error, attempting reconnect in 3s...", error);
            eventSource.close();
            setTimeout(() => {
                // Reconnect logic implies remount or custom retry, 
                // but for now let's just log. The browser usually retries SSE automatically.
            }, 3000);
        };

        return () => eventSource.close();
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
        setIsRunning((now - lastUpdate) < RUNNING_STALENESS_MS);
    }, [data.liveStatus?.updated_at, data.updatedAt]);

    // Selectors / Derived State
    const trials = useMemo(() => data.trials || [], [data.trials]);
    const sortedTrials = useMemo(() => [...trials].sort((a, b) => a.id - b.id), [trials]);

    const filteredTrials = useMemo(() => {
        return sortedTrials.filter((t) => {
            if (!filters.includeWarmup && t.warmstart) return false;
            if (!filters.includePruned && t.state === 'PRUNED') return false;
            if (filters.onlyComplete && t.state !== 'COMPLETE') return false;
            if (filters.minScore !== null && t.value < filters.minScore) return false;
            if (filters.maxScore !== null && t.value > filters.maxScore) return false;
            return true;
        });
    }, [sortedTrials, filters]);

    const bestTrial = useMemo(() => {
        const candidates = filteredTrials.filter(
            (t) => typeof t?.value === 'number' && Number.isFinite(t.value)
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
            (t) => typeof t?.value === 'number' && Number.isFinite(t.value)
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
        const liveId = data.liveStatus?.trial_number;
        if (liveId !== undefined && liveId !== null) return liveId + 1;
        if (sortedTrials.length > 0) {
            const lastTrial = sortedTrials[sortedTrials.length - 1];
            if (lastTrial.state === 'RUNNING' || lastTrial.state === 'WAITING') return lastTrial.id;
            return lastTrial.id + 1;
        }
        return 1;
    }, [data.liveStatus, sortedTrials]);

    const currentParams = useMemo(() => {
        const liveId = data.liveStatus?.trial_number;
        if (liveId !== undefined && (liveId + 1) === currentTrialId) {
            const liveParams = data.liveStatus?.params;
            if (liveParams && Object.keys(liveParams).length > 0) return liveParams;
        }
        const historyTrial = trials.find(t => t.id === currentTrialId);
        if (historyTrial && historyTrial.params && Object.keys(historyTrial.params).length > 0) return historyTrial.params;
        if (trials.length > 0) {
            const lastTrial = trials[trials.length - 1];
            if (lastTrial.params && Object.keys(lastTrial.params).length > 0) return lastTrial.params;
        }
        return {};
    }, [data.liveStatus, trials, currentTrialId]);

    const stateValue = useMemo(() => ({
        data, activeTab, viewMode, currentTime, isRunning, filters, isPending,
        trials, sortedTrials, bestTrial, bestTrialNoWarmstart, filteredTrials, currentTrialId, currentParams
    }), [data, activeTab, viewMode, currentTime, isRunning, filters, isPending, trials, sortedTrials, bestTrial, bestTrialNoWarmstart, filteredTrials, currentTrialId, currentParams]);

    const dispatchValue = useMemo(() => ({
        setData,
        setActiveTab: (tab) => startTransition(() => setActiveTab(tab)),
        // View switching must be immediate; wrapping in a transition can starve under frequent SSE updates.
        setViewMode: (mode) => setViewMode(mode),
        setFilters: (f) => startTransition(() => setFilters(f))
    }), []);

    return (
        <StoreStateContext.Provider value={stateValue}>
            <StoreDispatchContext.Provider value={dispatchValue}>
                {children}
            </StoreDispatchContext.Provider>
        </StoreStateContext.Provider>
    );
};

export const useStoreState = () => {
    const context = useContext(StoreStateContext);
    if (context === undefined) throw new Error("useStoreState must be used within a StoreProvider");
    return context;
};

export const useStoreDispatch = () => {
    const context = useContext(StoreDispatchContext);
    if (context === undefined) throw new Error("useStoreDispatch must be used within a StoreProvider");
    return context;
};

export const useStore = () => {
    const state = useStoreState();
    const dispatch = useStoreDispatch();
    return useMemo(() => ({ ...state, ...dispatch }), [state, dispatch]);
};
