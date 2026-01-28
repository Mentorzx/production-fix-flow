import { useCallback, useRef } from "react";
import { Activity, TrendingUp, Microscope, Sliders, Share2, Layers } from "../ui/BaseComponents.jsx";
import { ExportMenu } from "../ui/TableComponents.jsx";
import { ThemeToggle } from "../ui/ThemeToggle.jsx";

export const DashboardHeader = ({ activeTab, setActiveTab, viewMode, setViewMode, isRunning, currentTime, data }) => {
    const tabs = [
        { id: 'overview', icon: TrendingUp, label: 'Visão Geral' },
        { id: 'analysis', icon: Microscope, label: 'Análise' },
        { id: 'advanced', icon: Sliders, label: 'Avançado' },
        { id: 'forecast', icon: Share2, label: 'Previsão' }
    ];

    const tabRefs = useRef([]);

    const handleTabKeyDown = useCallback(
        (e, index) => {
            const key = e.key;
            if (!['ArrowLeft', 'ArrowRight', 'Home', 'End', 'Enter', ' '].includes(key)) return;
            e.preventDefault();

            let nextIndex = index;
            if (key === 'ArrowLeft') nextIndex = (index - 1 + tabs.length) % tabs.length;
            if (key === 'ArrowRight') nextIndex = (index + 1) % tabs.length;
            if (key === 'Home') nextIndex = 0;
            if (key === 'End') nextIndex = tabs.length - 1;

            const nextTab = tabs[nextIndex];
            if (nextTab) {
                setActiveTab(nextTab.id);
                tabRefs.current[nextIndex]?.focus?.();
            }
        },
        [setActiveTab, tabs]
    );

    return (
        <header className="flex-none h-16 backdrop-blur-md flex items-center justify-between px-6 z-100 relative" style={{ backgroundColor: 'color-mix(in srgb, var(--viz-bg-canvas), transparent 20%)', borderBottom: '1px solid var(--viz-border)' }}>
            <div className="flex items-center gap-4">
                <div className="relative w-9 h-9 rounded-lg flex items-center justify-center border shadow-inner" style={{ backgroundColor: 'var(--viz-bg-surface)', borderColor: 'var(--viz-border)' }}>
                    <Activity className={isRunning ? "text-orange-400" : ""} style={{ color: isRunning ? 'var(--viz-palette-3-orange)' : 'var(--viz-text-muted)' }} size={20} />
                </div>
                <div>
                    <div className="flex items-center gap-2">
                        <h1 className="text-lg font-bold tracking-tight" style={{ color: 'var(--viz-text-primary)' }}>Peak State</h1>
                        {data?.dashboardDebugMode && (
                            <span className="px-2 py-0.5 rounded-full border bg-amber-500/10 text-[9px] font-black uppercase tracking-widest text-amber-400" style={{ borderColor: 'var(--viz-palette-4-yellow)', color: 'var(--viz-palette-4-yellow)' }}>
                                Debug Mode
                            </span>
                        )}
                    </div>
                    <div className="text-[10px] uppercase tracking-widest flex items-center gap-2 font-mono" style={{ color: 'var(--viz-text-muted)' }}>
                        <span>Study: {data.studyName}</span>
                    </div>
                </div>
            </div>

            <div className="flex items-center gap-6">
                <nav
                    className="hidden md:flex p-1 rounded-lg border"
                    style={{ backgroundColor: 'var(--viz-bg-surface)', borderColor: 'var(--viz-border)' }}
                    role="tablist"
                    aria-label="Seções do dashboard"
                >
                    {tabs.map((tab, i) => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            onKeyDown={(e) => handleTabKeyDown(e, i)}
                            ref={(el) => {
                                tabRefs.current[i] = el;
                            }}
                            role="tab"
                            id={`tab-${tab.id}`}
                            aria-selected={activeTab === tab.id}
                            aria-controls={`panel-${tab.id}`}
                            tabIndex={activeTab === tab.id ? 0 : -1}
                            className={`flex items-center gap-2 px-4 py-1.5 text-[10px] font-bold rounded-md transition-all uppercase tracking-wide`}
                            style={{
                                backgroundColor: activeTab === tab.id ? 'var(--viz-bg-canvas)' : 'transparent',
                                color: activeTab === tab.id ? 'var(--viz-text-primary)' : 'var(--viz-text-muted)',
                                boxShadow: activeTab === tab.id ? '0 1px 2px rgba(0,0,0,0.1)' : 'none'
                            }}
                        >
                            <tab.icon size={14} />
                            <span>{tab.label}</span>
                        </button>
                    ))}
                </nav>

                <div className="flex items-center gap-4 border-l pl-6" style={{ borderColor: 'var(--viz-border)' }}>
                    <ThemeToggle />
                    {/* View Mode Toggle */}
                    <div className="flex p-1 rounded-lg border" style={{ backgroundColor: 'var(--viz-bg-surface)', borderColor: 'var(--viz-border)' }}>
                        <button
                            onClick={() => setViewMode('study')}
                            className={`flex items-center gap-2 px-3 py-1 rounded-sm text-[10px] font-bold uppercase transition-all`}
                            style={{
                                backgroundColor: viewMode === 'study' ? 'var(--viz-bg-canvas)' : 'transparent',
                                color: viewMode === 'study' ? 'var(--viz-text-primary)' : 'var(--viz-text-muted)',
                                boxShadow: viewMode === 'study' ? '0 1px 2px rgba(0,0,0,0.1)' : 'none'
                            }}
                        >
                            <Layers size={12} /> Estudo
                        </button>
                        <button
                            onClick={() => setViewMode('trial')}
                            className={`flex items-center gap-2 px-3 py-1 rounded-sm text-[10px] font-bold uppercase transition-all`}
                            style={{
                                color: viewMode === 'trial' ? 'var(--viz-palette-3-orange)' : 'var(--viz-text-muted)',
                                backgroundColor: viewMode === 'trial' ? 'rgba(213, 94, 0, 0.1)' : 'transparent',
                                border: viewMode === 'trial' ? '1px solid var(--viz-palette-3-orange)' : '1px solid transparent'
                            }}
                        >
                            <Activity size={12} /> Trial Atual
                        </button>
                    </div>

                    <div className="text-xs font-mono px-2 py-1 rounded-sm border" style={{ backgroundColor: 'var(--viz-bg-surface)', borderColor: 'var(--viz-border)', color: 'var(--viz-text-secondary)' }}>
                        {currentTime}
                    </div>
                    <ExportMenu data={data} />
                </div>
            </div>
        </header>
    );
};
