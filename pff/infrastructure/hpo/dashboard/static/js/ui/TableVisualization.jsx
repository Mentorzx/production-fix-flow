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

    // Normalize 0-100%
    const range = max - min || 1;
    const percentage = Math.max(0, Math.min(100, ((value - min) / range) * 100));

    return (
        <div className="relative w-full h-full flex items-center justify-end px-2">
            {/* Background Bar */}
            <div
                className="absolute left-0 top-1 bottom-1 rounded-r-sm opacity-20 transition-all duration-300"
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

// --- HeatmapCell (Background Color) ---
export const HeatmapCell = React.memo(({ value, min = 0, max = 1, colorScale = 'green', children }) => {
    if (typeof value !== 'number') return children || <span className="text-zinc-500">—</span>;

    // Normalize 0-1
    const range = max - min || 1;
    const normalized = Math.max(0, Math.min(1, (value - min) / range));

    // Simple Opacity-based Scale (Works well in Dark/Light mode)
    // Base colors from Theme (using semantic tokens)
    let baseColor = 'var(--viz-palette-2-green)'; // Default Success
    if (colorScale === 'red') baseColor = 'var(--viz-palette-5-red)';
    if (colorScale === 'blue') baseColor = 'var(--viz-palette-1-blue)';
    if (colorScale === 'orange') baseColor = 'var(--viz-palette-3-orange)';

    // Use opacity to create intensity
    // Min opacity 0.05 (visible) to 0.4 (readable text overlay)
    const opacity = 0.05 + (normalized * 0.35);

    return (
        <div
            className="w-full h-full flex items-center justify-end px-2"
            style={{ backgroundColor: `color-mix(in srgb, ${baseColor}, transparent ${100 - (opacity * 100)}%)` }}
        >
            <span className="font-mono" style={{ fontVariantNumeric: 'tabular-nums' }}>
                {children || value.toFixed(4)}
            </span>
        </div>
    );
});
