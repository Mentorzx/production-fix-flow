/**
 * Provide LeftSidebar module functionality for the HPO dashboard.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "../store/store.jsx";
import {
  Activity,
  TrendingUp,
  Microscope,
  Sliders,
  Share2,
  AlertTriangle,
  CheckCircle,
  Bell,
  ChevronRight,
  Settings,
  X,
} from "../ui/BaseComponents.jsx";
import { ThemeToggle } from "../ui/ThemeToggle.jsx";
import { ExportMenu } from "../ui/TableComponents.jsx";
import { Theme } from "../ui/Theme.js";

const SIDEBAR_COLLAPSED_STORAGE_KEY = "pff-sidebar-collapsed";
const NOTIFICATION_FILTER_STORAGE_KEY = "pff-notification-filter";
const formatElapsedRuntime = (secondsTotal) => {
  const safeSeconds = Number.isFinite(secondsTotal) && secondsTotal > 0 ? secondsTotal : 0;
  const totalMs = Math.floor(safeSeconds * 1000);
  const days = Math.floor(totalMs / 86_400_000);
  const hours = Math.floor((totalMs % 86_400_000) / 3_600_000);
  const minutes = Math.floor((totalMs % 3_600_000) / 60_000);
  const seconds = Math.floor((totalMs % 60_000) / 1000);
  const ms = totalMs % 1000;
  return `${days}d ${String(hours).padStart(2, "0")}h ${String(minutes).padStart(2, "0")}m ${String(seconds).padStart(2, "0")}s ${String(ms).padStart(3, "0")}ms`;
};

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
  return hex;
};

const normalizeNotificationType = (type) => {
  const normalized = String(type || "warning").toLowerCase();
  if (normalized === "success" || normalized === "warning" || normalized === "danger") {
    return normalized;
  }
  return "warning";
};

const getNotificationMeta = (type) => {
  const normalized = String(type || "warning").toLowerCase();
  if (normalized === "success") {
    return {
      color: "var(--viz-palette-2-green)",
      Icon: CheckCircle,
      label: "SUCESSO",
    };
  }
  if (normalized === "warning") {
    return {
      color: "var(--viz-palette-4-yellow)",
      Icon: AlertTriangle,
      label: "ALERTA",
    };
  }
  if (normalized === "danger") {
    return {
      color: "var(--viz-palette-6-red)",
      Icon: AlertTriangle,
      label: "CRÍTICO",
    };
  }
  return {
    color: "var(--viz-palette-4-yellow)",
    Icon: AlertTriangle,
    label: "ALERTA",
  };
};

const PeakStateLogo = ({ isRunning, compact = false }) => {
  const baseColor = isRunning ? Theme.palette.cyberYellow : Theme.palette.cyan;
  const glowColor = isRunning
    ? hexToRgba(Theme.palette.cyberYellow, 0.6)
    : hexToRgba(Theme.palette.cyan, 0.6);
  const outerSize = compact ? 40 : 60;
  const innerSize = compact ? 30 : 36;
  const iconSize = compact ? 16 : 18;

  return (
    <div
      className={`relative flex items-center justify-center pulso-logo ${isRunning ? "training" : ""}`}
      style={{ width: `${outerSize}px`, height: `${outerSize}px` }}
    >
      <div
        className="pulso-logo-glow absolute inset-0 rounded-xl"
        style={{
          background: `radial-gradient(circle, ${glowColor} 0%, transparent 70%)`,
          filter: "blur(8px)",
        }}
      />
      <div
        className="relative rounded-xl flex items-center justify-center border-2 overflow-hidden"
        style={{
          width: `${innerSize}px`,
          height: `${innerSize}px`,
          background: `linear-gradient(135deg, ${baseColor}15 0%, ${baseColor}30 50%, ${baseColor}50 100%)`,
          borderColor: baseColor,
          boxShadow: `0 0 20px ${glowColor}, inset 0 0 10px ${glowColor}`,
        }}
      >
        <div
          className="pulso-logo-shimmer absolute inset-0 opacity-50"
          style={{
            background: `linear-gradient(45deg, transparent 30%, ${baseColor}50 50%, transparent 70%)`,
          }}
        />
        <Activity size={iconSize} style={{ color: baseColor }} />
      </div>
      <div
        className="pulso-corner-dot absolute top-0.5 right-0.5 w-2 h-2 rounded-full"
        style={{ backgroundColor: baseColor, boxShadow: `0 0 6px ${baseColor}` }}
      />
      <div
        className="pulso-corner-dot absolute bottom-0.5 left-0.5 w-1.5 h-1.5 rounded-full"
        style={{ backgroundColor: baseColor, boxShadow: `0 0 4px ${baseColor}` }}
      />
    </div>
  );
};

const SidebarFilterToggle = ({ label, checked, onToggle, accentColor }) => (
  <button
    type="button"
    onClick={onToggle}
    aria-pressed={checked}
    className="btn-toggle w-full rounded-xl border px-3 py-2.5 text-left"
    style={{
      "--viz-icon-active": accentColor,
      borderColor: checked ? accentColor : "var(--viz-border)",
      backgroundColor: checked
        ? `color-mix(in srgb, ${accentColor}, transparent 86%)`
        : "color-mix(in srgb, var(--viz-bg-surface), var(--viz-bg-canvas) 10%)",
      color: checked ? "var(--viz-text-primary)" : "var(--viz-text-muted)",
    }}
  >
    <span className="flex items-center gap-2.5">
      <span
        className="h-2.5 w-2.5 rounded-full transition-all duration-200"
        style={{
          backgroundColor: checked ? accentColor : "var(--viz-text-muted)",
          boxShadow: checked ? `0 0 10px ${accentColor}` : "none",
          opacity: checked ? 1 : 0.6,
        }}
      />
      <span className="text-[11px] font-bold uppercase tracking-wide">{label}</span>
      <span className="ml-auto text-[10px] font-mono">{checked ? "ON" : "OFF"}</span>
    </span>
  </button>
);

/** @typedef {{
 *  activeTab: string,
 *  setActiveTab: (id: string) => void,
 *  isRunning: boolean,
 *  data: any,
 *  notifications: any[],
 *  unseenNotifications: number,
 *  onMarkNotificationsSeen: () => void,
 *  onMarkNotificationSeen: (id: string) => void,
 *  onClearNotificationHistory: () => void,
 *  onRemoveNotificationItem: (id: string) => void
 * }} LeftSidebarProps */

/** @param {LeftSidebarProps} props */
export const LeftSidebar = ({
  activeTab,
  setActiveTab,
  isRunning,
  data,
  notifications = [],
  unseenNotifications = 0,
  onMarkNotificationsSeen = () => {},
  onMarkNotificationSeen = () => {},
  onClearNotificationHistory = () => {},
  onRemoveNotificationItem = () => {},
}) => {
  const { filters, setFilters } = useStore();
  const [isCollapsed, setIsCollapsed] = useState(() => {
    try {
      return localStorage.getItem(SIDEBAR_COLLAPSED_STORAGE_KEY) === "1";
    } catch {
      return false;
    }
  });
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [notificationFilter, setNotificationFilter] = useState(() => {
    try {
      const stored = localStorage.getItem(NOTIFICATION_FILTER_STORAGE_KEY);
      if (stored === "success" || stored === "warning" || stored === "danger") {
        return stored;
      }
      return "all";
    } catch {
      return "all";
    }
  });
  const [runtimeTick, setRuntimeTick] = useState(() => Date.now());
  const tabRefs = useRef([]);

  useEffect(() => {
    const timerId = setInterval(() => setRuntimeTick(Date.now()), 100);
    return () => clearInterval(timerId);
  }, []);

  const hpoElapsedSeconds = useMemo(() => {
    const trialRows = Array.isArray(data?.trials) ? data.trials : [];
    const completedDurations = trialRows
      .filter((trial) => trial?.state !== "RUNNING" && trial?.state !== "WAITING")
      .reduce((acc, trial) => {
        const duration = Number(trial?.duration);
        return Number.isFinite(duration) && duration > 0 ? acc + duration : acc;
      }, 0);

    const runningElapsed = Number(data?.liveStatus?.elapsed_seconds);
    const runningSeconds =
      Number.isFinite(runningElapsed) && runningElapsed > 0 ? runningElapsed : 0;

    const updatedAtMs = Date.parse(String(data?.liveStatus?.updated_at || ""));
    const driftSeconds =
      Number.isFinite(updatedAtMs) && isRunning
        ? Math.max(0, (runtimeTick - updatedAtMs) / 1000)
        : 0;
    return completedDurations + runningSeconds + driftSeconds;
  }, [
    data?.trials,
    data?.liveStatus?.elapsed_seconds,
    data?.liveStatus?.updated_at,
    isRunning,
    runtimeTick,
  ]);

  useEffect(() => {
    try {
      localStorage.setItem(SIDEBAR_COLLAPSED_STORAGE_KEY, isCollapsed ? "1" : "0");
    } catch {
      // Ignore storage errors in restricted browsers.
    }
  }, [isCollapsed]);

  useEffect(() => {
    try {
      localStorage.setItem(NOTIFICATION_FILTER_STORAGE_KEY, notificationFilter);
    } catch {
      // Ignore storage errors in restricted browsers.
    }
  }, [notificationFilter]);

  const tabs = useMemo(
    () => [
      {
        id: "overview",
        icon: TrendingUp,
        label: "Monitoramento",
        color: "var(--viz-palette-1-blue)",
      },
      { id: "analysis", icon: Microscope, label: "Análise", color: "var(--viz-palette-7-cyan)" },
      { id: "advanced", icon: Sliders, label: "Avançado", color: "var(--viz-palette-3-orange)" },
      { id: "forecast", icon: Share2, label: "Previsão", color: "var(--viz-palette-4-yellow)" },
    ],
    []
  );

  const notificationItems = useMemo(() => {
    const all = (Array.isArray(notifications) ? notifications : [])
      .slice(0, 120)
      .map((item) => ({ ...item, type: normalizeNotificationType(item?.type) }));
    if (notificationFilter === "all") return all;
    return all.filter((item) => item.type === notificationFilter);
  }, [notifications, notificationFilter]);

  const notificationCounts = useMemo(() => {
    const all = (Array.isArray(notifications) ? notifications : [])
      .slice(0, 120)
      .map((item) => normalizeNotificationType(item?.type));
    return {
      all: all.length,
      success: all.filter((type) => type === "success").length,
      warning: all.filter((type) => type === "warning").length,
      danger: all.filter((type) => type === "danger").length,
    };
  }, [notifications]);

  const handleTabKeyDown = useCallback(
    (e, index) => {
      const key = e.key;
      if (!["ArrowUp", "ArrowDown", "Home", "End", "Enter", " "].includes(key)) return;
      e.preventDefault();
      let nextIndex = index;
      if (key === "ArrowUp") nextIndex = (index - 1 + tabs.length) % tabs.length;
      if (key === "ArrowDown") nextIndex = (index + 1) % tabs.length;
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

  const handleToggleSettings = useCallback(() => {
    if (isCollapsed) {
      setIsCollapsed(false);
      requestAnimationFrame(() => setIsSettingsOpen(true));
      return;
    }
    setIsSettingsOpen((prev) => !prev);
  }, [isCollapsed]);

  return (
    <aside
      className={`h-full flex-none border-r transition-[width] duration-300 ease-out ${
        isCollapsed ? "w-[86px]" : "w-[320px]"
      } relative overflow-visible`}
      style={{
        background:
          "linear-gradient(180deg, color-mix(in srgb, var(--viz-bg-surface), var(--viz-bg-canvas) 12%) 0%, color-mix(in srgb, var(--viz-bg-surface), var(--viz-bg-canvas) 28%) 100%)",
        borderColor: "var(--viz-border)",
        boxShadow:
          "inset -1px 0 0 color-mix(in srgb, var(--viz-border), transparent 30%), 16px 0 44px rgba(0,0,0,0.24)",
      }}
    >
      <div className="h-full p-3 flex flex-col">
        <div
          className={`${
            isCollapsed ? "grid place-items-center" : "flex items-center justify-between"
          } gap-2 pb-3 min-h-[88px]`}
        >
          <div
            className={`min-w-0 overflow-hidden transition-[max-width,max-height,opacity,transform,filter] duration-350 ease-out origin-left ${
              isCollapsed
                ? "max-w-0 max-h-0 opacity-0 -translate-x-4 scale-95 blur-[1px] pointer-events-none"
                : "max-w-[310px] max-h-[92px] opacity-100 translate-x-0 scale-100 blur-0"
            }`}
          >
            <div className="flex items-center gap-3 min-w-0 pr-2 pl-2 py-2">
              <PeakStateLogo isRunning={isRunning} />
              <div className="min-w-0">
                <h1
                  className="text-[28px] leading-none font-black tracking-[0.22em] uppercase"
                  style={{
                    color: "var(--viz-text-primary)",
                    fontFamily: "Space Grotesk, system-ui, sans-serif",
                  }}
                >
                  Pulso
                </h1>
                <p
                  className="text-[10px] uppercase tracking-[0.18em] truncate"
                  style={{ color: "var(--viz-text-muted)" }}
                  title={data.studyName}
                >
                  Study: {data.studyName || "-"}
                </p>
              </div>
            </div>
          </div>
          <button
            type="button"
            onClick={() => {
              setIsCollapsed((prev) => !prev);
              setIsSettingsOpen(false);
            }}
            aria-pressed={isCollapsed}
            aria-label={isCollapsed ? "Expandir barra lateral" : "Recolher barra lateral"}
            className={`btn-toggle rounded-xl border p-2 inline-flex items-center justify-center ${
              isCollapsed ? "mx-auto" : ""
            }`}
            style={{
              "--viz-icon-active": "var(--viz-palette-4-yellow)",
              borderColor: "var(--viz-border)",
              backgroundColor: "var(--viz-bg-canvas)",
              color: "var(--viz-text-primary)",
            }}
          >
            <ChevronRight
              size={16}
              className={`transition-transform duration-300 ${isCollapsed ? "rotate-0" : "rotate-180"}`}
            />
          </button>
        </div>

        <div
          className={`h-px ${isCollapsed ? "mx-0 mb-3" : "mx-1 mb-4"}`}
          style={{
            background:
              "linear-gradient(90deg, transparent 0%, color-mix(in srgb, var(--viz-border), transparent 5%) 50%, transparent 100%)",
          }}
        />

        <div className="flex-1 min-h-0 flex flex-col">
          <div className={isCollapsed ? "pt-1" : "px-1 pt-1"}>
            {!isCollapsed && (
              <p
                className="mb-2 px-1 text-[10px] font-bold uppercase tracking-[0.16em]"
                style={{ color: "var(--viz-text-muted)" }}
              >
                VISÕES
              </p>
            )}
            <nav role="tablist" aria-orientation="vertical" aria-label="Visões do dashboard">
              <div className={`grid grid-cols-1 ${isCollapsed ? "gap-2" : "gap-2.5"}`}>
                {tabs.map((tab, index) => {
                  const isActive = activeTab === tab.id;
                  return (
                    <button
                      key={tab.id}
                      ref={(el) => {
                        tabRefs.current[index] = el;
                      }}
                      type="button"
                      id={`tab-${tab.id}`}
                      role="tab"
                      aria-selected={isActive}
                      aria-controls={`panel-${tab.id}`}
                      tabIndex={isActive ? 0 : -1}
                      onClick={() => setActiveTab(tab.id)}
                      onKeyDown={(e) => handleTabKeyDown(e, index)}
                      className={`btn-tab w-full rounded-xl border transition-all duration-200 ${
                        isCollapsed
                          ? "h-10 w-10 mx-auto px-0 flex items-center justify-center"
                          : "h-11 px-3.5 flex items-center gap-3"
                      }`}
                      title={isCollapsed ? tab.label : undefined}
                      style={{
                        "--viz-icon-active": tab.color,
                        borderColor: isActive ? tab.color : "var(--viz-border)",
                        backgroundColor: isActive
                          ? `color-mix(in srgb, ${tab.color}, transparent 86%)`
                          : "color-mix(in srgb, var(--viz-bg-surface), var(--viz-bg-canvas) 14%)",
                        color: isActive ? "var(--viz-text-primary)" : "var(--viz-text-muted)",
                        boxShadow: isActive ? `0 0 18px ${hexToRgba("#e5c558", 0.12)}` : "none",
                      }}
                    >
                      <tab.icon size={16} />
                      {!isCollapsed && (
                        <span className="text-[11px] font-bold uppercase tracking-[0.12em] leading-none">
                          {tab.label}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </nav>
          </div>

          <div
            className={`h-px ${isCollapsed ? "mx-0 my-3" : "mx-1 my-4"}`}
            style={{
              background:
                "linear-gradient(90deg, transparent 0%, color-mix(in srgb, var(--viz-border), transparent 5%) 50%, transparent 100%)",
            }}
          />

          <section
            className={`min-h-0 flex-1 ${isCollapsed ? "px-0 mb-2" : "px-1 mb-3"} flex flex-col`}
          >
            {isCollapsed ? (
              <div
                className="relative mx-auto h-10 w-10 rounded-xl border grid place-items-center"
                title={`Notificações não vistas (${unseenNotifications})`}
                style={{
                  borderColor: "var(--viz-border)",
                  color: "var(--viz-text-muted)",
                  background: "color-mix(in srgb, var(--viz-bg-surface), var(--viz-bg-canvas) 18%)",
                }}
              >
                <Bell size={15} style={{ color: "var(--viz-palette-4-yellow)" }} />
                {unseenNotifications > 0 && (
                  <span
                    className="absolute -right-1 -top-1 min-w-[16px] rounded-full px-1 text-center text-[9px] font-bold"
                    style={{
                      color: "#fff",
                      backgroundColor: "var(--viz-palette-6-red)",
                      boxShadow:
                        "0 0 12px color-mix(in srgb, var(--viz-palette-6-red), transparent 40%)",
                    }}
                  >
                    {Math.min(unseenNotifications, 99)}
                  </span>
                )}
              </div>
            ) : (
              <>
                <div className="mb-2 flex items-center gap-2 px-1">
                  <Bell size={14} style={{ color: "var(--viz-palette-4-yellow)" }} />
                  <p
                    className="text-[10px] font-bold uppercase tracking-[0.16em]"
                    style={{ color: "var(--viz-text-muted)" }}
                  >
                    Notificações
                  </p>
                  {unseenNotifications > 0 && (
                    <span
                      className="rounded-full px-1.5 py-0.5 text-[9px] font-bold tabular-nums"
                      style={{
                        color: "#fff",
                        backgroundColor: "var(--viz-palette-6-red)",
                        boxShadow:
                          "0 0 12px color-mix(in srgb, var(--viz-palette-6-red), transparent 35%)",
                      }}
                    >
                      {Math.min(unseenNotifications, 99)}
                    </span>
                  )}
                  <span
                    className="ml-auto rounded-lg border px-1.5 py-0.5 text-[9px] font-bold tabular-nums"
                    style={{
                      borderColor: "var(--viz-border)",
                      color: "var(--viz-text-primary)",
                      background:
                        "color-mix(in srgb, var(--viz-bg-surface), var(--viz-bg-canvas) 20%)",
                    }}
                  >
                    {notificationItems.length}
                  </span>
                </div>
                <div className="mb-2 px-1 flex flex-wrap items-center gap-1.5">
                  <div className="mr-auto flex flex-wrap items-center gap-1.5">
                    {[
                      { id: "all", label: "Tudo" },
                      { id: "success", label: "OK" },
                      { id: "warning", label: "Alert" },
                      { id: "danger", label: "Crit" },
                    ].map((filterType) => {
                      const isActive = notificationFilter === filterType.id;
                      return (
                        <button
                          key={filterType.id}
                          type="button"
                          onClick={() =>
                            setNotificationFilter((prev) =>
                              prev === filterType.id ? "all" : filterType.id
                            )
                          }
                          className="btn-toggle min-w-[58px] h-8 rounded-md border px-2 text-[9px] font-bold uppercase tracking-[0.08em] inline-flex items-center justify-center"
                          style={{
                            borderColor: isActive
                              ? "var(--viz-palette-1-blue)"
                              : "var(--viz-border)",
                            color: isActive ? "var(--viz-text-primary)" : "var(--viz-text-muted)",
                            backgroundColor: isActive
                              ? "color-mix(in srgb, var(--viz-palette-1-blue), transparent 86%)"
                              : "color-mix(in srgb, var(--viz-bg-surface), var(--viz-bg-canvas) 14%)",
                          }}
                        >
                          {filterType.label} ({notificationCounts[filterType.id] || 0})
                        </button>
                      );
                    })}
                  </div>
                  <button
                    type="button"
                    onClick={onMarkNotificationsSeen}
                    className="btn-toggle h-8 rounded-md border px-2.5 text-[9px] font-bold uppercase tracking-[0.1em] inline-flex items-center justify-center"
                    style={{
                      borderColor: "var(--viz-border)",
                      color: "var(--viz-text-primary)",
                      background:
                        "color-mix(in srgb, var(--viz-bg-surface), var(--viz-bg-canvas) 14%)",
                    }}
                  >
                    Visualizar tudo
                  </button>
                  <button
                    type="button"
                    onClick={onClearNotificationHistory}
                    className="btn-toggle h-8 rounded-md border px-2.5 text-[9px] font-bold uppercase tracking-[0.1em] inline-flex items-center justify-center"
                    style={{
                      borderColor: "color-mix(in srgb, var(--viz-palette-6-red), transparent 45%)",
                      color: "var(--viz-palette-6-red)",
                      background: "color-mix(in srgb, var(--viz-palette-6-red), transparent 92%)",
                    }}
                  >
                    Limpar
                  </button>
                </div>

                <div
                  className="min-h-[160px] flex-1 overflow-hidden rounded-xl border"
                  style={{
                    borderColor: "var(--viz-border)",
                    background:
                      "linear-gradient(180deg, color-mix(in srgb, var(--viz-bg-surface), transparent 10%) 0%, color-mix(in srgb, var(--viz-bg-canvas), transparent 20%) 100%)",
                  }}
                >
                  {notificationItems.length === 0 ? (
                    <div
                      className="flex h-full items-center justify-center px-4 text-center text-[10px] uppercase tracking-[0.14em]"
                      style={{ color: "var(--viz-text-muted)" }}
                    >
                      Nenhuma notificação relevante por enquanto.
                    </div>
                  ) : (
                    <div
                      className="h-full overflow-y-auto custom-scrollbar p-2 space-y-1.5"
                      onMouseEnter={onMarkNotificationsSeen}
                    >
                      {notificationItems.map((notification) => {
                        const meta = getNotificationMeta(notification.type);
                        const Icon = meta.Icon;
                        return (
                          <article
                            key={notification.id}
                            className="relative rounded-lg border px-2 py-1.5"
                            onMouseEnter={() => onMarkNotificationSeen(notification.id)}
                            style={{
                              borderColor: `color-mix(in srgb, ${meta.color}, transparent 70%)`,
                              backgroundColor: `color-mix(in srgb, ${meta.color}, transparent 93%)`,
                            }}
                          >
                            <span
                              className="absolute left-0.5 top-1.5 bottom-1.5 w-[2px] rounded-full"
                              style={{
                                backgroundColor: meta.color,
                                boxShadow: `0 0 8px ${meta.color}`,
                              }}
                            />
                            <div className="mb-0.5 flex items-center gap-1.5">
                              <Icon size={12} style={{ color: meta.color }} />
                              <p
                                className="truncate text-[10px] font-bold uppercase tracking-[0.12em]"
                                style={{ color: "var(--viz-text-primary)" }}
                              >
                                {notification.title}
                              </p>
                              <button
                                type="button"
                                className="btn-toggle inline-flex h-5 w-5 items-center justify-center rounded border"
                                onClick={() => onRemoveNotificationItem(notification.id)}
                                aria-label="Remover notificação"
                                style={{
                                  borderColor: "var(--viz-border)",
                                  background:
                                    "color-mix(in srgb, var(--viz-bg-surface), var(--viz-bg-canvas) 16%)",
                                  color: "var(--viz-text-muted)",
                                }}
                              >
                                <X size={10} />
                              </button>
                              <span
                                className="ml-auto text-[9px] font-mono"
                                style={{ color: "var(--viz-text-muted)" }}
                              >
                                {new Date(notification.createdAt).toLocaleTimeString("pt-BR", {
                                  hour12: false,
                                })}
                              </span>
                            </div>
                            <p
                              className="line-clamp-1 text-[10px] leading-relaxed"
                              style={{ color: "var(--viz-text-secondary)" }}
                            >
                              {notification.message}
                            </p>
                            <span
                              className="mt-0.5 inline-block text-[9px] font-bold uppercase tracking-[0.1em]"
                              style={{ color: meta.color }}
                            >
                              {meta.label}
                            </span>
                          </article>
                        );
                      })}
                    </div>
                  )}
                </div>
              </>
            )}
          </section>
        </div>

        <div
          className={`mt-auto ${isCollapsed ? "pt-3" : "pt-4"} space-y-2`}
          style={{
            borderTop: "1px solid color-mix(in srgb, var(--viz-border), transparent 24%)",
          }}
        >
          {!isCollapsed && (
            <div
              className="rounded-xl border px-3 py-2 flex items-center gap-3"
              style={{
                borderColor: "var(--viz-border)",
                backgroundColor: "color-mix(in srgb, var(--viz-bg-canvas), transparent 20%)",
                boxShadow:
                  "inset 0 1px 0 color-mix(in srgb, var(--viz-text-primary), transparent 96%)",
              }}
            >
              <span
                className={`inline-block w-2 h-2 rounded-full ${isRunning ? "animate-pulse" : ""}`}
                style={{
                  backgroundColor: isRunning
                    ? "var(--viz-palette-2-green)"
                    : "var(--viz-text-muted)",
                  boxShadow: isRunning
                    ? "0 0 10px color-mix(in srgb, var(--viz-palette-2-green), transparent 40%)"
                    : "none",
                }}
              />
              <span
                className="text-[10px] uppercase tracking-[0.14em]"
                style={{ color: "var(--viz-text-muted)" }}
              >
                Runtime
              </span>
              <span
                className="ml-auto text-[11px] font-mono tabular-nums"
                style={{ color: "var(--viz-text-primary)" }}
              >
                {formatElapsedRuntime(hpoElapsedSeconds)}
              </span>
            </div>
          )}

          <div className={`flex gap-2 ${isCollapsed ? "flex-col items-center" : "items-center"}`}>
            <ThemeToggle className="btn-toggle border rounded-xl p-2.5" />

            <button
              type="button"
              onClick={handleToggleSettings}
              aria-pressed={isSettingsOpen}
              aria-expanded={isSettingsOpen}
              aria-label="Abrir configuração e filtros"
              className={`btn-toggle border rounded-xl ${isCollapsed ? "p-2.5" : "px-3 py-2.5 flex-1"} ${
                isCollapsed ? "flex items-center justify-center" : "flex items-center gap-2"
              }`}
              style={{
                "--viz-icon-active": "var(--viz-palette-3-orange)",
                borderColor: isSettingsOpen ? "var(--viz-palette-3-orange)" : "var(--viz-border)",
                backgroundColor: isSettingsOpen
                  ? "color-mix(in srgb, var(--viz-palette-3-orange), transparent 84%)"
                  : "color-mix(in srgb, var(--viz-bg-surface), var(--viz-bg-canvas) 14%)",
                color: isSettingsOpen ? "var(--viz-text-primary)" : "var(--viz-text-muted)",
              }}
              title={isCollapsed ? "Configurações" : undefined}
            >
              <Settings size={16} />
              {!isCollapsed && (
                <>
                  <span className="text-[11px] font-bold uppercase tracking-[0.14em]">
                    Configuração
                  </span>
                  <ChevronRight
                    size={14}
                    className={`ml-auto transition-transform duration-300 ${isSettingsOpen ? "rotate-90" : "rotate-0"}`}
                  />
                </>
              )}
            </button>
          </div>

          <div
            className={`overflow-hidden transition-all duration-300 ease-out ${
              !isCollapsed && isSettingsOpen
                ? "max-h-[360px] opacity-100 translate-y-0"
                : "max-h-0 opacity-0 -translate-y-2 pointer-events-none"
            }`}
          >
            <div
              className="rounded-2xl border p-3 space-y-2.5"
              style={{
                borderColor: "var(--viz-border)",
                backgroundColor:
                  "color-mix(in srgb, var(--viz-bg-surface), var(--viz-bg-canvas) 12%)",
              }}
            >
              <p
                className="text-[10px] font-bold uppercase tracking-[0.16em]"
                style={{ color: "var(--viz-text-muted)" }}
              >
                Filtros
              </p>

              <SidebarFilterToggle
                label="Warmup"
                checked={filters.includeWarmup}
                onToggle={() => setFilters({ ...filters, includeWarmup: !filters.includeWarmup })}
                accentColor="var(--viz-palette-3-orange)"
              />

              <SidebarFilterToggle
                label="Pruned"
                checked={filters.includePruned}
                onToggle={() => setFilters({ ...filters, includePruned: !filters.includePruned })}
                accentColor="var(--viz-palette-4-yellow)"
              />

              <div className="h-px my-1" style={{ backgroundColor: "var(--viz-border)" }} />

              <div className="flex items-center justify-between gap-3">
                <div>
                  <p
                    className="text-[10px] font-bold uppercase tracking-[0.14em]"
                    style={{ color: "var(--viz-text-primary)" }}
                  >
                    Exportar
                  </p>
                  <p className="text-[10px]" style={{ color: "var(--viz-text-muted)" }}>
                    JSON, CSV, Parquet e Toon
                  </p>
                </div>
                <ExportMenu data={data} />
              </div>
            </div>
          </div>
        </div>
      </div>

      <div
        className={`absolute left-full top-3 ml-3 z-30 transition-[opacity,transform,filter] duration-350 ease-out ${
          isCollapsed
            ? "opacity-100 translate-x-0 scale-100 blur-0 pointer-events-none"
            : "opacity-0 -translate-x-5 scale-95 blur-[2px] pointer-events-none"
        }`}
      >
        <div className={`sidebar-brand-float ${isCollapsed ? "sidebar-brand-float-active" : ""}`}>
          <div
            className="flex items-center gap-3 px-4 py-3 rounded-2xl border min-h-[88px]"
            style={{
              borderColor: "var(--viz-border)",
              background:
                "linear-gradient(180deg, color-mix(in srgb, var(--viz-bg-surface), transparent 8%) 0%, color-mix(in srgb, var(--viz-bg-canvas), transparent 18%) 100%)",
              boxShadow:
                "0 12px 28px rgba(0,0,0,0.28), inset 0 1px 0 color-mix(in srgb, var(--viz-text-primary), transparent 94%)",
              backdropFilter: "blur(5px)",
            }}
          >
            <PeakStateLogo isRunning={isRunning} />
            <div className="min-w-0 max-w-[280px]">
              <h1
                className="text-[28px] leading-none font-black tracking-[0.2em] uppercase"
                style={{
                  color: "var(--viz-text-primary)",
                  fontFamily: "Space Grotesk, system-ui, sans-serif",
                }}
              >
                Pulso
              </h1>
              <p
                className="text-[10px] uppercase tracking-[0.16em] truncate"
                style={{ color: "var(--viz-text-muted)" }}
                title={data.studyName}
              >
                Study: {data.studyName || "-"}
              </p>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
};
