import React, { useState, useEffect, useRef, useMemo } from "react";
import { Theme } from "./Theme.js";
import { PortalTooltip } from "./PortalTooltip.jsx";

/**
 * KPI stat badge with sparkline, delta tracking, and optional progress bar.
 */
export const StatBadge = React.memo(
  ({
    label,
    value,
    valueNode = null,
    subtext,
    color = "orange",
    sparklineValues = null,
    progress = null,
    deltaPct = null,
    direction = null,
    helpText = null,
  }) => {
    const themeColors = {
      orange: Theme.palette.hotOrange,
      lime: Theme.palette.lime,
      amber: Theme.palette.cyberYellow,
      rose: Theme.palette.red,
      zinc: Theme.palette.grey,
    };

    const activeColor = themeColors[color] || Theme.semantic.primary;
    const [flipKey, setFlipKey] = useState(0);
    const prevValue = useRef(value);

    useEffect(() => {
      if (prevValue.current !== value) {
        setFlipKey((k) => k + 1);
        prevValue.current = value;
      }
    }, [value]);

    const spark = useMemo(() => {
      if (!Array.isArray(sparklineValues) || sparklineValues.length < 2) return null;
      const nums = sparklineValues
        .map((v) => {
          const n = Number.parseFloat(v);
          return Number.isFinite(n) ? n : null;
        })
        .filter((v) => v !== null);
      if (nums.length < 2) return null;

      const w = 96;
      const h = 28;
      const min = Math.min(...nums);
      const max = Math.max(...nums);
      const range = Math.max(1e-9, max - min);
      const dx = w / (nums.length - 1);

      const points = nums
        .map((v, i) => {
          const x = i * dx;
          const y = h - ((v - min) / range) * h;
          return `${x.toFixed(2)},${y.toFixed(2)}`;
        })
        .join(" ");

      return { w, h, points };
    }, [sparklineValues]);

    const pct = useMemo(() => {
      const p = Number.parseFloat(progress);
      if (!Number.isFinite(p)) return null;
      return Math.max(0, Math.min(100, p));
    }, [progress]);

    const formattedDelta = useMemo(() => {
      const d = Number.parseFloat(deltaPct);
      if (!Number.isFinite(d)) return null;
      const rounded = Math.round(d * 10) / 10;
      const isFlat = Math.abs(rounded) < 0.05;
      const dir = direction === "down" ? "down" : "up";
      const improved = dir === "down" ? rounded < 0 : rounded > 0;
      const tone = isFlat ? "flat" : improved ? "good" : "bad";
      const text = isFlat ? "0.0%" : `${rounded > 0 ? "+" : ""}${rounded.toFixed(1)}%`;
      return { text, tone };
    }, [deltaPct, direction]);

    const deltaColor = useMemo(() => {
      if (!formattedDelta) return Theme.ui.text.muted;
      if (formattedDelta.tone === "flat") return Theme.ui.text.muted;
      if (formattedDelta.tone === "good") return Theme.palette.vividGreen;
      return Theme.palette.red;
    }, [formattedDelta]);

    const directionText = useMemo(() => {
      if (direction === "down") return { arrow: "↓", text: "Melhor se menor" };
      if (direction === "up") return { arrow: "↑", text: "Melhor se maior" };
      return null;
    }, [direction]);

    const tooltipContent = useMemo(() => {
      if (!helpText) return null;
      const isHelpTextObject = (value) =>
        typeof value === "object" && value !== null && "tech" in value;
      if (!isHelpTextObject(helpText)) return null;

      return (
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
                {helpText.tech}
              </div>
            </div>
            {helpText.simple && (
              <div className="pt-2 border-t" style={{ borderColor: Theme.ui.border }}>
                <span
                  className="text-[8px] font-black uppercase block mb-1"
                  style={{ color: Theme.semantic.success }}
                >
                  Para Leigos
                </span>
                <div
                  className="italic leading-tight border-l-2 pl-2"
                  style={{
                    color: Theme.palette.mint,
                    borderColor: Theme.palette.vividGreen + "33",
                  }}
                >
                  {helpText.simple}
                </div>
              </div>
            )}
            {Array.isArray(helpText.extra) && helpText.extra.length > 0 && (
              <div className="pt-2 border-t" style={{ borderColor: Theme.ui.border }}>
                <span
                  className="text-[8px] font-black uppercase block mb-1"
                  style={{ color: Theme.palette.cyberYellow }}
                >
                  Valores
                </span>
                <div className="space-y-1">
                  {helpText.extra.map((item, index) => (
                    <div
                      key={`${item.label}-${index}`}
                      className="text-[10px] leading-tight flex gap-2"
                      style={{ color: Theme.ui.text.secondary }}
                    >
                      <span
                        className="font-semibold min-w-[72px]"
                        style={{ color: Theme.palette.apricot }}
                      >
                        {item.label}:
                      </span>
                      <span>{item.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      );
    }, [helpText]);

    const card = (
      <div
        className="p-5 rounded-2xl border shadow-xl flex flex-col h-full min-h-[140px] transition-transform duration-200 ease-out hover:scale-[1.015] hover:brightness-110"
        style={{
          backgroundColor: Theme.ui.surface,
          borderColor: activeColor + "33",
          boxShadow: `0 0 28px ${activeColor}14`,
        }}
      >
        <div className="flex items-start justify-between gap-4">
          <span
            className="text-[10px] font-black uppercase tracking-[0.2em] opacity-40"
            style={{ color: activeColor }}
          >
            {label}
          </span>
          <span
            className="text-[10px] font-mono font-bold tracking-tight tabular-nums"
            style={{ color: deltaColor }}
          >
            {formattedDelta ? formattedDelta.text : "—"}
          </span>
        </div>

        <div className="mt-2 flex flex-col flex-1 min-h-0">
          <div className="flex items-center justify-between gap-4 min-w-0">
            <div
              key={flipKey}
              className={`${valueNode ? "pff-flip min-w-0" : "text-4xl font-black font-mono tracking-tighter pff-flip min-w-0 whitespace-nowrap"}`}
              style={{ color: Theme.ui.text.primary }}
            >
              {valueNode ?? value}
            </div>

            {spark && (
              <div className="shrink-0 opacity-80">
                <svg viewBox={`0 0 ${spark.w} ${spark.h}`} className="w-28 h-8">
                  <polyline
                    points={spark.points}
                    fill="none"
                    stroke={activeColor}
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    opacity="0.9"
                  />
                  <polygon
                    points={`${spark.points} ${spark.w},${spark.h} 0,${spark.h}`}
                    fill={activeColor}
                    fillOpacity="0.08"
                    stroke="none"
                  />
                </svg>
              </div>
            )}
          </div>

          {subtext && (
            <div
              className="mt-1 text-[10px] opacity-40 font-bold uppercase"
              style={{ color: activeColor }}
            >
              {subtext}
            </div>
          )}
        </div>

        <div className="mt-1 flex items-center justify-end min-h-[14px]">
          {directionText && (
            <span
              className="text-[9px] font-black uppercase tracking-widest opacity-30"
              style={{ color: Theme.ui.text.secondary }}
            >
              <span style={{ color: activeColor }}>{directionText.arrow}</span> {directionText.text}
            </span>
          )}
        </div>

        {pct !== null && (
          <div
            className="mt-3 h-1.5 w-full rounded-full overflow-hidden"
            style={{ backgroundColor: Theme.ui.border }}
          >
            <div
              className="h-full"
              style={{
                width: `${pct}%`,
                background: `linear-gradient(90deg, ${activeColor}, ${Theme.palette.cyberYellow})`,
              }}
            />
          </div>
        )}
      </div>
    );

    if (tooltipContent) {
      return (
        <PortalTooltip content={tooltipContent} className="block w-full h-full">
          {card}
        </PortalTooltip>
      );
    }

    return card;
  }
);
