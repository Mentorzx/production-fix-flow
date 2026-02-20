/**
 * Provide TableVisualization module functionality for the HPO dashboard.
 */

import React from "react";
import { Theme } from "./Theme";

/**
 * Tier 2: Micro-Visualizations for High Density Tables
 * Zero-dependency SVG implementations for max performance.
 */

// --- Sparkline (Mini Line Chart) ---
/**
 * Expose sparkline for dashboard usage.
 */
export const Sparkline = React.memo(
  ({
    data = [],
    width = 60,
    height = 20,
    color = Theme.semantic.primary,
    min = null,
    max = null,
  }) => {
    if (!data || data.length < 2) return <span className="text-zinc-600">-</span>;

    const values = data.filter((v) => typeof v === "number");
    if (values.length < 2) return <span className="text-zinc-600">-</span>;

    const minVal = min ?? Math.min(...values);
    const maxVal = max ?? Math.max(...values);
    const range = maxVal - minVal || 1;

    // SVG Polyline points
    const points = values
      .map((val, i) => {
        const x = (i / (values.length - 1)) * width;
        const normalizedVal = (val - minVal) / range;
        const y = height - normalizedVal * height; // Invert Y (SVG coords)
        return `${x},${y}`;
      })
      .join(" ");

    return (
      <svg width={width} height={height} className="overflow-visible">
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="opacity-80"
        />
        {/* End dot */}
        <circle
          cx={width}
          cy={height - ((values[values.length - 1] - minVal) / range) * height}
          r="2"
          fill={color}
        />
      </svg>
    );
  }
);

// --- DataBar (Horizontal Bar Background) ---
/**
 * Expose data bar for dashboard usage.
 */
export const DataBar = React.memo(
  ({
    value,
    min = 0,
    max = 1,
    color = null, // Auto-select based on value if null
    showValue = true,
    format = (v) => v?.toFixed?.(4) ?? v,
    invert = false, // Lower is better (e.g., loss)
  }) => {
    if (typeof value !== "number" || isNaN(value)) return <span className="text-zinc-500">—</span>;

    // Normalize 0-100% (if min===max, show full bar)
    const range = max - min;
    let ratio = range === 0 ? 1 : Math.max(0, Math.min(1, (value - min) / range));

    // Invert for "lower is better" metrics
    if (invert) ratio = 1 - ratio;

    const percentage = ratio * 100;

    // Auto-select color based on performance tier using project palette
    const barColor =
      color ||
      (() => {
        if (ratio > 0.8) return Theme.palette.mint; // Excellent
        if (ratio > 0.5) return Theme.palette.cyan; // Good
        if (ratio > 0.2) return Theme.palette.cyberYellow; // Fair
        return Theme.semantic.error; // Poor
      })();

    return (
      <div className="relative w-full h-full flex items-center justify-end px-2 min-w-[80px]">
        {/* Background Bar - tier based */}
        <div
          className="absolute left-0 top-1 bottom-1 rounded-r transition-all duration-300"
          style={{
            width: `${percentage}%`,
            backgroundColor: barColor,
            opacity: 0.2,
          }}
        />
        {/* Tier indicator dot */}
        <div
          className="absolute left-1 top-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full"
          style={{ backgroundColor: barColor }}
        />
        {/* Value Text */}
        {showValue && (
          <span
            className="relative z-10 font-mono text-xs font-medium"
            style={{
              color: barColor,
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {format(value)}
          </span>
        )}
      </div>
    );
  }
);

// --- HeatmapCell (SOTA Design: Progress Bar Background + Tier Colors) ---
/**
 * HeatmapCell with project palette - Tier system with background bar
 * @param {number} value - The metric value
 * @param {number} min - Minimum for normalization
 * @param {number} max - Maximum for normalization
 * @param {boolean} invert - If true, lower values are better (e.g., loss)
 * @param {string} tier - "high", "mid", "low" for direct color assignment
 */
export const HeatmapCell = React.memo(
  ({
    value,
    min = 0,
    max = 1,
    invert = false,
    tier = null,
    children,
    format = (v) => v?.toFixed?.(4) ?? v,
  }) => {
    if (typeof value !== "number" || isNaN(value)) {
      return children || <span className="text-zinc-500">—</span>;
    }

    // Project palette colors
    const PALETTE = {
      // High tier (>80%): Emerald/Mint - excellent performance
      high: { bg: Theme.palette.mint, text: Theme.palette.mint },
      // Mid tier (50-80%): Cyan/Blue - moderate performance
      mid: { bg: Theme.palette.cyan, text: Theme.palette.cyan },
      // Low tier (<50%): Grey - low performance
      low: { bg: Theme.palette.grey, text: Theme.palette.grey },
      // Alert tier (bad for inverted metrics): Rose/Red - warning
      alert: { bg: Theme.semantic.error, text: Theme.semantic.error },
    };

    // Calculate normalized ratio
    const range = max - min || 1;
    let ratio = Math.max(0, Math.min(1, (value - min) / range));

    // Invert for "lower is better" metrics (e.g., loss, duration)
    if (invert) {
      ratio = 1 - ratio;
    }

    // Determine tier
    let tierColors;
    if (tier && PALETTE[tier]) {
      tierColors = PALETTE[tier];
    } else if (ratio > 0.8) {
      tierColors = PALETTE.high;
    } else if (ratio > 0.5) {
      tierColors = PALETTE.mid;
    } else if (invert && value > max * 0.8) {
      // For inverted metrics: high values = alert
      tierColors = PALETTE.alert;
    } else {
      tierColors = PALETTE.low;
    }

    // Bar width based on ratio (always at least 5% for visibility)
    const barWidth = Math.max(5, ratio * 100);

    return (
      <div className="relative w-full h-full flex items-center justify-end px-2 min-w-[70px]">
        {/* Background progress bar */}
        <div
          className="absolute left-0 top-1 bottom-1 rounded-r transition-all duration-300"
          style={{
            width: `${barWidth}%`,
            backgroundColor: tierColors.bg,
            opacity: 0.15,
          }}
        />

        {/* Value text with tier color */}
        <span
          className="relative z-10 font-mono text-xs font-semibold"
          style={{
            color: tierColors.text,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {children || format(value)}
        </span>
      </div>
    );
  }
);
