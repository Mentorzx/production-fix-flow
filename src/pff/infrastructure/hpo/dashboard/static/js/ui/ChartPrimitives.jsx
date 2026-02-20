/**
 * Provide ChartPrimitives module functionality for the HPO dashboard.
 */

import React from "react";
import { Tooltip, CartesianGrid } from "recharts";
import { Theme } from "./Theme.js";
import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { PortalTooltip } from "./PortalTooltip.jsx";
import { HintTooltipContent } from "./HintTooltipContent.jsx";

/**
 * Chart-level primitives: colors adapter, default grid/tooltip, responsive container.
 */

export const colors = {
  bg: Theme.ui.background,
  card: Theme.ui.surface,
  border: Theme.ui.border,
  text: Theme.ui.text.secondary,
  textHigh: Theme.ui.text.primary,
  primary: Theme.semantic.primary,
  orange: Theme.palette.hotOrange,
  success: Theme.semantic.success,
  lime: Theme.palette.lime,
  amber: Theme.palette.cyberYellow,
  error: Theme.semantic.error,
  warning: Theme.semantic.warning,
  grid: Theme.ui.grid,
  tooltip: Theme.ui.background,
};

/**
 * Expose default chart margins for dashboard usage.
 */
export const defaultChartMargins = { top: 25, right: 15, bottom: 35, left: 50 };

/**
 * Expose default tooltip style for dashboard usage.
 */
export const defaultTooltipStyle = {
  backgroundColor: Theme.ui.background,
  borderColor: Theme.ui.border,
  color: Theme.ui.text.primary,
  fontSize: "11px",
  borderRadius: "8px",
  boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.5)",
};

/**
 * Expose default cartesian grid for dashboard usage.
 */
export const DefaultCartesianGrid = React.memo((props) => (
  <CartesianGrid strokeDasharray="3 3" stroke={Theme.ui.grid} strokeOpacity={0.5} {...props} />
));

/**
 * Expose default tooltip for dashboard usage.
 */
export const DefaultTooltip = React.memo((props) => {
  const { wrapperStyle, ...rest } = props;
  return (
    <Tooltip
      contentStyle={defaultTooltipStyle}
      itemStyle={{ color: Theme.ui.text.secondary }}
      cursor={{ stroke: Theme.ui.grid, strokeDasharray: "3 3" }}
      wrapperStyle={{ zIndex: 60, ...wrapperStyle }}
      {...rest}
    />
  );
});

/**
 * Expose default tooltip cursor for dashboard usage.
 */
export const DefaultTooltipCursor = { strokeDasharray: "3 3", stroke: Theme.ui.grid };

/**
 * Expose standardized area-gradient presets for chart fills.
 */
export const getChartAreaGradientStops = (
  token = "primarySubtle",
  baseColor = Theme.semantic.primary
) => {
  const presets = Theme.gradients?.chartArea || {};
  const selected = presets[token] ||
    presets.primarySubtle || [
      { offset: "0%", color: "currentColor", opacity: 0.24 },
      { offset: "100%", color: "currentColor", opacity: 0.02 },
    ];
  return selected.map((stop) => ({
    ...stop,
    color: stop.color === "currentColor" ? baseColor : stop.color,
  }));
};

/**
 * Expose chart frame for dashboard usage.
 */
export const ChartFrame = React.memo(({ children, className = "" }) => (
  <div className={`relative w-full h-full min-h-0 min-w-0 ${className}`}>{children}</div>
));

/**
 * Expose chart container for dashboard usage.
 */
export const ChartContainer = React.memo(({ children, className = "", minHeight = 0 }) => {
  const containerRef = useRef(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const lastSizeRef = useRef({ width: 0, height: 0 });

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;

    const applyNextSize = (nextWidth, nextHeight) => {
      const roundedWidth = Math.max(0, Math.round(nextWidth));
      const roundedHeight = Math.max(0, Math.round(nextHeight));
      const last = lastSizeRef.current;
      if (Math.abs(last.width - roundedWidth) <= 1 && Math.abs(last.height - roundedHeight) <= 1) {
        return;
      }
      lastSizeRef.current = { width: roundedWidth, height: roundedHeight };
      setSize((prev) => {
        if (prev.width === roundedWidth && prev.height === roundedHeight) return prev;
        return { width: roundedWidth, height: roundedHeight };
      });
    };

    let frameId = 0;
    const measure = () => {
      const { width, height } = container.getBoundingClientRect();
      if (width > 0 && height > 0) {
        applyNextSize(width, height);
      } else {
        applyNextSize(0, 0);
      }
    };

    const scheduleMeasure = () => {
      if (frameId) cancelAnimationFrame(frameId);
      frameId = requestAnimationFrame(measure);
    };

    scheduleMeasure();
    window.addEventListener("resize", scheduleMeasure, { passive: true });
    return () => {
      if (frameId) cancelAnimationFrame(frameId);
      window.removeEventListener("resize", scheduleMeasure);
    };
  }, []);

  const ready = size.width > 0 && size.height > 0;
  const child = React.Children.only(children);

  return (
    <div
      ref={containerRef}
      className={`w-full h-full min-w-0 ${className}`}
      style={{ minWidth: 0, minHeight }}
    >
      {ready &&
        React.isValidElement(child) &&
        React.cloneElement(child, { width: size.width, height: size.height })}
    </div>
  );
});

const resolveLegendEntryKey = (entry) => {
  if (!entry || typeof entry !== "object") return "";
  const payloadDataKey = entry?.payload?.dataKey;
  if (typeof payloadDataKey === "string" || typeof payloadDataKey === "number") {
    return String(payloadDataKey);
  }
  const dataKey = entry.dataKey;
  if (typeof dataKey === "string" || typeof dataKey === "number") {
    return String(dataKey);
  }
  if (typeof entry.id === "string" || typeof entry.id === "number") {
    return String(entry.id);
  }
  if (typeof entry.value === "string" || typeof entry.value === "number") {
    return String(entry.value);
  }
  return "";
};

const isHiddenLegendEntry = (entry) => {
  if (!entry || typeof entry !== "object") return true;
  if (entry.type === "none" || entry.legendType === "none") return true;
  const rawValue = String(entry.value ?? "").trim();
  if (!rawValue) return true;
  return false;
};

const choosePreferredLegendEntry = (current, candidate) => {
  const currentValue = String(current?.value ?? "").trim();
  const candidateValue = String(candidate?.value ?? "").trim();
  const currentKey = resolveLegendEntryKey(current);
  const candidateKey = resolveLegendEntryKey(candidate);
  const currentLooksHuman = currentValue && currentValue !== currentKey;
  const candidateLooksHuman = candidateValue && candidateValue !== candidateKey;
  if (candidateLooksHuman && !currentLooksHuman) return candidate;
  if (candidateLooksHuman === currentLooksHuman && candidateValue.length > currentValue.length) {
    return candidate;
  }
  return current;
};

const normalizeLegendEntries = (payload) => {
  if (!Array.isArray(payload) || payload.length === 0) return [];
  const byToggleKey = new Map();
  for (const entry of payload) {
    if (isHiddenLegendEntry(entry)) continue;
    const toggleKey = resolveLegendEntryKey(entry);
    if (!toggleKey) continue;
    const previous = byToggleKey.get(toggleKey);
    byToggleKey.set(toggleKey, previous ? choosePreferredLegendEntry(previous, entry) : entry);
  }
  return [...byToggleKey.entries()].map(([toggleKey, entry]) => ({ toggleKey, entry }));
};

/**
 * Expose legend visibility hook for dashboard charts.
 */
export const useLegendVisibility = (seriesKeys = []) => {
  const normalizedSeriesKeys = useMemo(
    () =>
      Array.from(
        new Set(
          (Array.isArray(seriesKeys) ? seriesKeys : [])
            .map((value) => String(value || "").trim())
            .filter(Boolean)
        )
      ),
    [seriesKeys]
  );

  const [hiddenKeys, setHiddenKeys] = useState(() => new Set());

  useEffect(() => {
    setHiddenKeys((prev) => {
      if (normalizedSeriesKeys.length === 0) return new Set();
      const filtered = [...prev].filter((key) => normalizedSeriesKeys.includes(key));
      return filtered.length === prev.size ? prev : new Set(filtered);
    });
  }, [normalizedSeriesKeys]);

  const toggleSeriesVisibility = useCallback((seriesKey) => {
    const key = String(seriesKey || "").trim();
    if (!key) return;
    setHiddenKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const isSeriesVisible = useCallback(
    (seriesKey) => !hiddenKeys.has(String(seriesKey || "").trim()),
    [hiddenKeys]
  );

  return { hiddenKeys, toggleSeriesVisibility, isSeriesVisible };
};

/**
 * Expose interactive legend content for dashboard charts.
 */
export const InteractiveLegend = React.memo(
  ({
    payload,
    hiddenKeys,
    onToggleSeries,
    getHint,
    align = "right",
    seriesKeys = [],
  }) => {
    const hidden = hiddenKeys instanceof Set ? hiddenKeys : new Set();
    const persistedEntriesRef = useRef(new Map());
    const allowedKeys = useMemo(
      () =>
        new Set(
          (Array.isArray(seriesKeys) ? seriesKeys : [])
            .map((value) => String(value || "").trim())
            .filter(Boolean)
        ),
      [seriesKeys]
    );
    const safePayload = useMemo(() => {
      const normalized = normalizeLegendEntries(payload);
      for (const item of normalized) {
        persistedEntriesRef.current.set(item.toggleKey, item.entry);
      }
      let items = [...persistedEntriesRef.current.entries()].map(([toggleKey, entry]) => ({
        toggleKey,
        entry,
      }));
      if (allowedKeys.size > 0) {
        items = items.filter((item) => allowedKeys.has(item.toggleKey));
      }
      return items;
    }, [payload, allowedKeys]);
    const justifyClass =
      align === "left" ? "justify-start" : align === "center" ? "justify-center" : "justify-end";

    return (
      <ul className={`m-0 flex w-full flex-wrap items-center gap-2 ${justifyClass} p-0`}>
        {safePayload.map(({ toggleKey, entry }, index) => {
          const key = toggleKey || `legend-item-${index}`;
          const visible = !hidden.has(toggleKey);
          const dotColor = visible ? entry?.color || Theme.ui.text.secondary : Theme.ui.text.muted;
          const label = entry?.value ?? key;
          const legendHint =
            (typeof getHint === "function" ? getHint(toggleKey, entry) : null) || {
              tech: `Série "${String(label)}" do gráfico atual. Clique para ocultar/mostrar e validar o efeito na leitura.`,
              simple:
                "Passe o mouse para ver o contexto da série. Clique para comparar visualmente com as demais.",
              extra: [
                { label: "Atalho", value: "Clique: alternar visibilidade da série" },
                { label: "Estado", value: visible ? "Visível" : "Oculta" },
              ],
            };
          const renderedLabel = String(label);

          return (
            <li key={key} className="list-none">
              <PortalTooltip
                className="inline-flex"
                interactive={true}
                content={<HintTooltipContent hint={legendHint} value={label} extraValue={toggleKey} />}
              >
                <button
                  type="button"
                  onClick={() => onToggleSeries?.(toggleKey)}
                  className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] transition-colors duration-150"
                  style={{
                    borderColor: visible ? Theme.ui.border : Theme.ui.grid,
                    backgroundColor: visible
                      ? "rgba(10, 13, 34, 0.25)"
                      : "rgba(9, 12, 28, 0.45)",
                    color: visible ? Theme.ui.text.secondary : Theme.ui.text.muted,
                    opacity: visible ? 1 : 0.78,
                  }}
                  aria-pressed={!visible}
                  aria-label={`${visible ? "Ocultar" : "Mostrar"} série ${String(label)}`}
                >
                  <span
                    className="inline-block h-2 w-2 rounded-full"
                    style={{ backgroundColor: dotColor }}
                  />
                  <span className="inline-flex items-center gap-1">{renderedLabel}</span>
                </button>
              </PortalTooltip>
            </li>
          );
        })}
      </ul>
    );
  }
);
