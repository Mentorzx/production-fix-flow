import { useTransition } from 'react';
import { useStore } from "../store/store.jsx";
import { DashboardHeader } from "./DashboardHeader.jsx";
import { OverviewTab } from "./OverviewTab.jsx";
import { AnalysisTab } from "./AnalysisTab.jsx";
import { AdvancedTab } from "./AdvancedTab.jsx";
import { ForecastTab } from "./ForecastTab.jsx";

import BackgroundGraph from "../ui/BackgroundGraph.jsx";
import { GlobalFilterBar } from "../ui/GlobalFilterBar.jsx";
import { useTheme } from "../ui/ThemeContext.jsx";

export const Dashboard = () => {
    const {
        data, activeTab, setActiveTab, viewMode, setViewMode, isRunning, currentTime
    } = useStore();
    const { theme } = useTheme();

    const [isPending, startTransition] = useTransition();

    const handleTabChange = (tabId) => {
        startTransition(() => {
            setActiveTab(tabId);
        });
    };

    return (
        <div className={`flex flex-col h-screen font-sans overflow-hidden relative ${theme === 'dark' ? 'pff-ambient' : ''}`} style={{ backgroundColor: 'var(--viz-bg-canvas)', color: 'var(--viz-text-secondary)' }}>
            <BackgroundGraph />
            <div className="absolute inset-0 flex flex-col z-10 pointer-events-none">
                <div className="pointer-events-auto w-full">
                    <DashboardHeader
                        activeTab={activeTab}
                        setActiveTab={handleTabChange}
                        viewMode={viewMode}
                        setViewMode={setViewMode}
                        isRunning={isRunning}
                        currentTime={currentTime}
                        data={data}
                    />
                    <GlobalFilterBar />
                </div>

                <main className={`flex-1 overflow-auto custom-scrollbar p-6 transition-opacity duration-300 pointer-events-auto ${isPending ? 'opacity-50' : 'opacity-100'}`}>
                    <div className="max-w-[1600px] mx-auto space-y-6">
                        {activeTab === 'overview' && (
                            <section role="tabpanel" id="panel-overview" aria-labelledby="tab-overview" tabIndex={0}>
                                <OverviewTab />
                            </section>
                        )}
                        {activeTab === 'analysis' && (
                            <section role="tabpanel" id="panel-analysis" aria-labelledby="tab-analysis" tabIndex={0}>
                                <AnalysisTab />
                            </section>
                        )}
                        {activeTab === 'advanced' && (
                            <section role="tabpanel" id="panel-advanced" aria-labelledby="tab-advanced" tabIndex={0}>
                                <AdvancedTab />
                            </section>
                        )}
                        {activeTab === 'forecast' && (
                            <section role="tabpanel" id="panel-forecast" aria-labelledby="tab-forecast" tabIndex={0}>
                                <ForecastTab />
                            </section>
                        )}
                    </div>
                </main>

                <footer className="flex-none h-8 px-4 flex items-center justify-between text-[10px] font-mono pointer-events-auto" style={{ backgroundColor: 'var(--viz-bg-surface)', borderTop: '1px solid var(--viz-border)', color: 'var(--viz-text-muted)' }}>
                    <div className="flex items-center gap-4">
                        <span className="flex items-center gap-1.5">
                            <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-lime-500 animate-pulse' : 'bg-zinc-700'}`}></span>
                            System: {isRunning ? 'ACTIVE' : 'IDLE'}
                        </span>
                        <span className="border-l border-zinc-800 pl-4">Architecture: SOTA ESM + Transitions</span>
                    </div>
                    <div>Last Update: {data.updatedAt || 'N/A'}</div>
                </footer>
            </div>
        </div >
    );
};
