import { useCallback, useMemo, useRef } from "react";
import { useStore } from "../store/store.jsx";
import {
  Activity,
  TrendingUp,
  Microscope,
  Sliders,
  Share2,
  Layers,
} from "../ui/BaseComponents.jsx";
import { ExportMenu } from "../ui/TableComponents.jsx";
import { ThemeToggle } from "../ui/ThemeToggle.jsx";
import { Theme } from "../ui/Theme.js";
const hexToRgba = (hex, alpha = 1) => {
  let c;
  if (/^#([A-Fa-f0-9]{3}){1,2}$/.test(hex)) {
    c = hex.substring(1).split("");
    if (c.length === 3) {
      c = [c[0], c[0], c[1], c[1], c[2], c[2]];
    }
    c = "0x" + c.join("");
    return "rgba(" + [(c >> 16) & 255, (c >> 8) & 255, c & 255].join(",") + "," + alpha + ")";
  }
  return hex; // Fallback
};

// PULSO Logo Component with refined heartbeat animation
const PeakStateLogo = ({ isRunning }) => {
  // Use PULSO brand colors (desaturated gold/cyan)
  const baseColor = isRunning ? Theme.palette.cyberYellow : Theme.palette.cyan;
  const glowColor = isRunning
    ? hexToRgba(Theme.palette.cyberYellow, 0.6)
    : hexToRgba(Theme.palette.cyan, 0.6);

  return (
    <div
      className={`relative w-10 h-10 flex items-center justify-center pulso-logo ${isRunning ? "training" : ""}`}
    >
      {/* Outer glow ring */}
      <div
        className="pulso-logo-glow absolute inset-0 rounded-xl"
        style={{
          background: `radial-gradient(circle, ${glowColor} 0%, transparent 70%)`,
          filter: "blur(8px)",
        }}
      />

      {/* Inner container with gradient */}
      <div
        className="relative w-9 h-9 rounded-xl flex items-center justify-center border-2 overflow-hidden"
        style={{
          background: `linear-gradient(135deg, ${baseColor}15 0%, ${baseColor}30 50%, ${baseColor}50 100%)`,
          borderColor: baseColor,
          boxShadow: `0 0 20px ${glowColor}, inset 0 0 10px ${glowColor}`,
        }}
      >
        {/* Animated shimmer sweep */}
        <div
          className="pulso-logo-shimmer absolute inset-0 opacity-50"
          style={{
            background: `linear-gradient(45deg, transparent 30%, ${baseColor}50 50%, transparent 70%)`,
          }}
        />

        {/* Activity icon with heartbeat */}
        <Activity
          className="pulso-logo-icon"
          size={20}
          style={{
            color: baseColor,
          }}
        />
      </div>

      {/* Corner accents */}
      <div
        className="pulso-corner-dot absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full"
        style={{ backgroundColor: baseColor, boxShadow: `0 0 6px ${baseColor}` }}
      />
      <div
        className="pulso-corner-dot absolute -bottom-0.5 -left-0.5 w-1.5 h-1.5 rounded-full"
        style={{ backgroundColor: baseColor, boxShadow: `0 0 4px ${baseColor}` }}
      />
    </div>
  );
};

/** @typedef {{
 *  activeTab: string,
 *  setActiveTab: (id: string) => void,
 *  viewMode: string,
 *  setViewMode: (mode: string) => void,
 *  isRunning: boolean,
 *  currentTime: string,
 *  data: any
 * }} DashboardHeaderProps */

/** @param {DashboardHeaderProps} props */
export const DashboardHeader = ({
  activeTab,
  setActiveTab,
  viewMode,
  setViewMode,
  isRunning,
  currentTime,
  data,
}) => {
  const { filters, setFilters } = useStore();
  const tabs = useMemo(
    () => [
      { id: "overview", icon: TrendingUp, label: "Monitoramento" },
      { id: "analysis", icon: Microscope, label: "Análise" },
      { id: "advanced", icon: Sliders, label: "Avançado" },
      { id: "forecast", icon: Share2, label: "Previsão" },
    ],
    []
  );

  const tabRefs = useRef([]);

  const handleTabKeyDown = useCallback(
    (e, index) => {
      const key = e.key;
      if (!["ArrowLeft", "ArrowRight", "Home", "End", "Enter", " "].includes(key)) return;
      e.preventDefault();

      let nextIndex = index;
      if (key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
      if (key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
      if (key === "Home") nextIndex = 0;
      if (key === "End") nextIndex = tabs.length - 1;

      const nextTab = tabs[nextIndex];
      if (nextTab) {
        setActiveTab(nextTab.id);
        tabRefs.current[nextIndex]?.focus?.();
      }
    },
    [setActiveTab, tabs]
  );

  return (
    <header
      className="flex-none h-16 backdrop-blur-md flex items-center justify-between px-6 z-100 relative"
      style={{
        backgroundColor: "var(--viz-bg-surface)",
        borderBottom: "1px solid var(--viz-border)",
      }}
    >
      <div className="flex items-center gap-4">
        <PeakStateLogo isRunning={isRunning} />
        <div>
          <div className="flex items-center gap-2">
            <h1
              className="text-lg font-black tracking-tighter uppercase"
              style={{
                color: "var(--viz-text-primary)",
                fontFamily: "Space Grotesk, system-ui, sans-serif",
                letterSpacing: "0.15em",
                textShadow: `0 0 20px ${hexToRgba(Theme.palette.cyberYellow, 0.3)}`,
                background:
                  "linear-gradient(90deg, var(--viz-text-primary) 0%, var(--viz-palette-4-yellow) 50%, var(--viz-text-primary) 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}
            >
              PULSO
            </h1>
            {data?.dashboardDebugMode && (
              <span
                className="px-2 py-0.5 rounded-full border bg-amber-500/10 text-[9px] font-black uppercase tracking-widest text-amber-400"
                style={{
                  borderColor: "var(--viz-palette-4-yellow)",
                  color: "var(--viz-palette-4-yellow)",
                }}
              >
                Debug Mode
              </span>
            )}
          </div>
          <div
            className="text-[10px] uppercase tracking-widest flex items-center gap-2 font-mono"
            style={{ color: "var(--viz-text-muted)" }}
          >
            <span>Study: {data.studyName}</span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <nav
          className="hidden md:flex p-1 rounded-lg border"
          style={{ backgroundColor: "var(--viz-bg-surface)", borderColor: "var(--viz-border)" }}
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
              className={`btn-tab flex items-center gap-2 px-4 py-1.5 text-[10px] font-bold rounded-md uppercase tracking-wide`}
              style={{
                backgroundColor: activeTab === tab.id ? "var(--viz-bg-canvas)" : "transparent",
                color: activeTab === tab.id ? "var(--viz-text-primary)" : "var(--viz-text-muted)",
                boxShadow: activeTab === tab.id ? "0 1px 2px rgba(0,0,0,0.1)" : "none",
              }}
            >
              <tab.icon size={14} />
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>

        <div
          className="flex items-center gap-4 border-l pl-6"
          style={{ borderColor: "var(--viz-border)" }}
        >
          {/* Inline Filters (Hidden on small screens, or we can adapt) */}
          <div className="hidden lg:flex items-center gap-3 mr-4">
            <label
              className="flex items-center gap-2 cursor-pointer group select-none"
              title="Incluir Warmup"
            >
              <input
                type="checkbox"
                checked={filters.includeWarmup}
                onChange={(e) => setFilters({ ...filters, includeWarmup: e.target.checked })}
                className="w-3.5 h-3.5 rounded-sm border-zinc-600 bg-zinc-800 text-orange-500 focus:ring-0 focus:ring-offset-0"
              />
              <span className="text-[10px] uppercase font-bold text-zinc-500 group-hover:text-zinc-300 transition-colors">
                Warmup
              </span>
            </label>
            <label
              className="flex items-center gap-2 cursor-pointer group select-none"
              title="Incluir Pruned"
            >
              <input
                type="checkbox"
                checked={filters.includePruned}
                onChange={(e) => setFilters({ ...filters, includePruned: e.target.checked })}
                className="w-3.5 h-3.5 rounded-sm border-zinc-600 bg-zinc-800 text-orange-500 focus:ring-0 focus:ring-offset-0"
              />
              <span className="text-[10px] uppercase font-bold text-zinc-500 group-hover:text-zinc-300 transition-colors">
                Pruned
              </span>
            </label>
          </div>

          <ThemeToggle />
          {/* View Mode Toggle */}
          <div
            className="flex p-1 rounded-lg border"
            style={{ backgroundColor: "var(--viz-bg-surface)", borderColor: "var(--viz-border)" }}
          >
            <button
              onClick={() => setViewMode("study")}
              className={`btn-toggle flex items-center gap-2 px-3 py-1 rounded-sm text-[10px] font-bold uppercase transition-all`}
              style={{
                backgroundColor: viewMode === "study" ? "var(--viz-bg-canvas)" : "transparent",
                color: viewMode === "study" ? "var(--viz-text-primary)" : "var(--viz-text-muted)",
                boxShadow: viewMode === "study" ? "0 1px 2px rgba(0,0,0,0.1)" : "none",
              }}
            >
              <Layers size={12} /> Estudo
            </button>
            <button
              onClick={() => setViewMode("trial")}
              className={`btn-toggle flex items-center gap-2 px-3 py-1 rounded-sm text-[10px] font-bold uppercase transition-all`}
              style={{
                color:
                  viewMode === "trial" ? "var(--viz-palette-3-orange)" : "var(--viz-text-muted)",
                backgroundColor: viewMode === "trial" ? "rgba(213, 94, 0, 0.1)" : "transparent",
                border:
                  viewMode === "trial"
                    ? "1px solid var(--viz-palette-3-orange)"
                    : "1px solid transparent",
              }}
            >
              <Activity size={12} /> Trial Atual
            </button>
          </div>

          <div
            className="text-xs font-mono px-2 py-1 rounded-sm border"
            style={{
              backgroundColor: "var(--viz-bg-surface)",
              borderColor: "var(--viz-border)",
              color: "var(--viz-text-secondary)",
            }}
          >
            {currentTime}
          </div>
          <ExportMenu data={data} />
        </div>
      </div>
    </header>
  );
};
