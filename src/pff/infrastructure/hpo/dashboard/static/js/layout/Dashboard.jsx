/**
 * Provide Dashboard module functionality for the HPO dashboard.
 */

// @ts-check
import { useState, useEffect, useMemo, useRef } from "react";
import { useStore } from "../store/store.jsx";
import { OverviewTab } from "./OverviewTab.jsx";
import { AnalysisTab } from "./AnalysisTab.jsx";
import { AdvancedTab } from "./AdvancedTab.jsx";
import { ForecastTab } from "./ForecastTab.jsx";
import { KpiRow } from "./KpiRow.jsx";
import { LeftSidebar } from "./LeftSidebar.jsx";
import { FloatingScopeSwitch } from "./FloatingScopeSwitch.jsx";

import BackgroundGraph from "../ui/BackgroundGraph.jsx";
import { useTheme } from "../ui/ThemeContext.jsx";
import { useDashboardNotifications } from "../ui/useDashboardNotifications.js";
import { NotificationToasts } from "../ui/NotificationToasts.jsx";
import { useJackpotAnimation } from "../ui/useJackpotAnimation.js";
import { SearchLauncherButton } from "../ui/SearchLauncherButton.jsx";
import { CommandPalette } from "../ui/CommandPalette.jsx";
import { buildSearchCatalog } from "../search/catalog.js";

/**
 * Expose dashboard for dashboard usage.
 */
export const Dashboard = () => {
  const { data, activeTab, setActiveTab, viewMode, setViewMode, isRunning } = useStore();
  const { theme } = useTheme();
  const {
    toasts,
    history,
    unseenCount,
    nowMs,
    dismissToast,
    markAllSeen,
    markSeen,
    clearHistory,
    removeHistoryItem,
  } = useDashboardNotifications(data);
  const [versionInfo, setVersionInfo] = useState({ version: "-", buildId: "-" });
  const [paletteOpen, setPaletteOpen] = useState(false);
  const dashboardRootRef = useRef(null);
  const launcherRef = useRef(null);
  const searchCatalog = useMemo(() => buildSearchCatalog(), []);
  const jackpotTriggerKey = [
    activeTab,
    viewMode,
    data?.updatedAt || "",
    data?.liveStatus?.trial_number ?? "",
    data?.liveStatus?.epoch ?? "",
    Array.isArray(data?.trials) ? data.trials.length : 0,
  ].join("|");

  useJackpotAnimation(dashboardRootRef, jackpotTriggerKey);

  useEffect(() => {
    // Load version info from the generated version.json
    fetch("/dist/version.json")
      .then((r) => r.json())
      .then((v) => setVersionInfo(v))
      .catch(() => setVersionInfo({ version: "dev", buildId: "local" }));
  }, []);

  const handleTabChange = (tabId) => {
    setActiveTab(tabId);
  };

  return (
    <div
      ref={dashboardRootRef}
      className={`flex flex-col h-screen font-sans overflow-hidden relative ${theme === "dark" ? "pff-ambient" : ""}`}
      style={{ backgroundColor: "var(--viz-bg-canvas)", color: "var(--viz-text-secondary)" }}
    >
      <BackgroundGraph />
      <NotificationToasts toasts={toasts} nowMs={nowMs} dismissToast={dismissToast} />
      <div className="absolute inset-0 z-10 flex">
        <LeftSidebar
          activeTab={activeTab}
          setActiveTab={handleTabChange}
          isRunning={isRunning}
          data={data}
          notifications={history}
          unseenNotifications={unseenCount}
          onMarkNotificationsSeen={markAllSeen}
          onMarkNotificationSeen={markSeen}
          onClearNotificationHistory={clearHistory}
          onRemoveNotificationItem={removeHistoryItem}
        />

        <div className="flex-1 min-w-0 flex flex-col relative">
          <div
            className="pointer-events-none absolute inset-0 z-0"
            style={{
              background:
                "radial-gradient(ellipse at 50% 28%, color-mix(in srgb, var(--viz-bg-canvas), transparent 88%) 0%, color-mix(in srgb, var(--viz-bg-canvas), transparent 72%) 68%, color-mix(in srgb, var(--viz-bg-canvas), transparent 56%) 100%)",
            }}
          />
          <FloatingScopeSwitch viewMode={viewMode} setViewMode={setViewMode} />
          <div className="absolute top-4 right-8 z-30">
            <SearchLauncherButton buttonRef={launcherRef} onClick={() => setPaletteOpen(true)} />
          </div>
          <CommandPalette
            open={paletteOpen}
            onOpenChange={setPaletteOpen}
            catalog={searchCatalog}
            setActiveTab={setActiveTab}
            setViewMode={setViewMode}
            launcherRef={launcherRef}
          />

          <main className="relative z-10 flex-1 overflow-y-scroll overflow-x-hidden custom-scrollbar px-6 pb-6 pt-24 transition-opacity duration-300 [scrollbar-gutter:stable] opacity-100">
            <div className="max-w-[1600px] mx-auto space-y-6">
              <KpiRow />
              {activeTab === "overview" && (
                <section
                  role="tabpanel"
                  id="panel-overview"
                  aria-labelledby="tab-overview"
                  tabIndex={0}
                >
                  <OverviewTab />
                </section>
              )}
              {activeTab === "analysis" && (
                <section
                  role="tabpanel"
                  id="panel-analysis"
                  aria-labelledby="tab-analysis"
                  tabIndex={0}
                >
                  <AnalysisTab />
                </section>
              )}
              {activeTab === "advanced" && (
                <section
                  role="tabpanel"
                  id="panel-advanced"
                  aria-labelledby="tab-advanced"
                  tabIndex={0}
                >
                  <AdvancedTab />
                </section>
              )}
              {activeTab === "forecast" && (
                <section
                  role="tabpanel"
                  id="panel-forecast"
                  aria-labelledby="tab-forecast"
                  tabIndex={0}
                >
                  <ForecastTab />
                </section>
              )}
            </div>
          </main>

          <footer
            className="flex-none h-8 px-4 flex items-center justify-between text-[10px] font-mono"
            style={{
              backgroundColor: "var(--viz-bg-surface)",
              borderTop: "1px solid var(--viz-border)",
              color: "var(--viz-text-muted)",
            }}
          >
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1.5">
                <span
                  className={`w-2 h-2 rounded-full ${isRunning ? "bg-lime-500 animate-pulse" : "bg-zinc-700"}`}
                ></span>
                System: {isRunning ? "ACTIVE" : "IDLE"}
              </span>
              <span className="border-l border-zinc-800 pl-4">
                Architecture: SOTA ESM + Transitions
              </span>
              <span className="border-l border-zinc-800 pl-4">
                Last Update:{" "}
                {data.updatedAt
                  ? new Date(data.updatedAt).toLocaleString("pt-BR", {
                      timeZone: "America/Sao_Paulo",
                    })
                  : "N/A"}
              </span>
            </div>
            <div style={{ color: "var(--viz-text-primary)" }}>v{versionInfo.version}</div>
          </footer>
        </div>
      </div>
    </div>
  );
};
