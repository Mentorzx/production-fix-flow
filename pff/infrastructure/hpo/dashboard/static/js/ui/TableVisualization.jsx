import React from 'react';
import { Theme } from './Theme';

/**
 * Tier 2: Micro-Visualizations for High Density Tables
 * Zero-dependency SVG implementations for max performance.
 */

// --- Sparkline (Mini Line Chart) ---
export const Sparkline = React.memo(({ data = [], width = 60, height = 20, color = Theme.semantic.primary, min = null, max = null }) => {
    if (!data || data.length < 2) return <span className="text-zinc-600">-</span>;

    const values = data.filter(v => typeof v === 'number');
    if (values.length < 2) return <span className="text-zinc-600">-</span>;

    const minVal = min ?? Math.min(...values);
    const maxVal = max ?? Math.max(...values);
    const range = maxVal - minVal || 1;

    // SVG Polyline points
    const points = values.map((val, i) => {
        const x = (i / (values.length - 1)) * width;
        const normalizedVal = (val - minVal) / range;
        const y = height - (normalizedVal * height); // Invert Y (SVG coords)
        return `${x},${y}`;
    }).join(' ');

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
});

// --- DataBar (Horizontal Bar Background) ---
export const DataBar = React.memo(({ value, min = 0, max = 1, color = Theme.semantic.chart.metric, showValue = true, format = (v) => v?.toFixed(4) }) => {
    if (typeof value !== 'number') return <span className="text-zinc-500">—</span>;

    // Normalize 0-100% (if min===max, show full bar)
    const range = max - min;
    const percentage = range === 0 ? 100 : Math.max(0, Math.min(100, ((value - min) / range) * 100));

    return (
        <div className="relative w-full h-full flex items-center justify-end px-2">
            {/* Background Bar */}
            <div
                className="absolute left-0 top-1 bottom-1 rounded-r-sm opacity-30 transition-all duration-300"
                style={{
                    width: `${percentage}%`,
                    backgroundColor: color
                }}
            />
            {/* Value Text */}
            {showValue && (
                <span className="relative z-10 font-mono tracking-tighter" style={{ fontVariantNumeric: 'tabular-nums' }}>
                    {format(value)}
                </span>
            )}
        </div>
    );
});

// --- HeatmapCell (Rounded Pill with Red→Yellow→Green Gradient) ---
export const HeatmapCell = React.memo(({ value, min = 0, max = 1, children }) => {
    if (typeof value !== 'number') return children || <span className="text-zinc-500">—</span>;

    const range = max - min || 1;
    const normalized = Math.max(0, Math.min(1, (value - min) / range));

    // HSL hue interpolation: red(0°) → yellow(60°) → green(125°)
    const hue = normalized * 125;
    const bgAlpha = 0.14 + normalized * 0.16;
    const borderAlpha = 0.20 + normalized * 0.18;

    return (
        <div className="w-full h-full flex items-center justify-end">
            <span
                className="inline-flex items-center justify-center font-mono rounded-full px-2.5 py-0.5 text-[10px]"
                style={{
                    backgroundColor: `hsla(${hue}, 72%, 50%, ${bgAlpha})`,
                    border: `1px solid hsla(${hue}, 72%, 50%, ${borderAlpha})`,
                    color: `hsl(${hue}, 68%, 72%)`,
                    fontVariantNumeric: 'tabular-nums'
                }}
            >
                {children || value.toFixed(4)}
            </span>
        </div>
    );
});
