import React, { useState, useRef } from 'react';
import { createPortal } from 'react-dom';
import { ArrowUp, ArrowDown } from "./BaseComponents.jsx";
import { Theme } from "./Theme.js";
import { MetricRegistry } from "../domain/metrics/MetricRegistry.js";
import { ParamRegistry } from "../domain/metrics/ParamRegistry.js";

export const PortalTooltip = ({ content, children }) => {
    const [visible, setVisible] = useState(false);
    const [coords, setCoords] = useState({ top: 0, left: 0 });
    const triggerRef = useRef(null);

    const handleMouseEnter = () => {
        if (triggerRef.current) {
            const rect = triggerRef.current.getBoundingClientRect();
            setCoords({ top: rect.top - 8, left: rect.left + (rect.width / 2) });
            setVisible(true);
        }
    };

    return (
        <React.Fragment>
            <div ref={triggerRef} onMouseEnter={handleMouseEnter} onMouseLeave={() => setVisible(false)} className="inline-block">{children}</div>
            {visible && createPortal(
                <div className="fixed z-9999 pointer-events-none animate-in fade-in zoom-in-95 duration-100" style={{ top: coords.top, left: coords.left, transform: 'translate(-50%, -100%)' }}>{content}</div>,
                document.body
            )}
        </React.Fragment>
    );
};

export const SectionDivider = ({ label, icon: Icon }) => (
    <div className="col-span-full flex items-center gap-3 pb-2 border-b border-zinc-800/50 mt-8 mb-4 first:mt-0">
        {Icon && <div className="p-1.5 rounded-md bg-zinc-900 text-orange-400"><Icon size={14} /></div>}
        <h3 className="text-xs font-black uppercase tracking-[0.2em] text-zinc-500">{label}</h3>
    </div>
);

export const renderWithHints = (value, extraValue = null) => {
    const lower = String(value).toLowerCase();
    let key = lower;

    // Mapping normalization
    if (lower.includes('média móvel') || lower.includes('moving average')) { key = 'moving_average'; }
    else if (lower.includes('intervalo de confiança') || lower.includes('confidence interval')) { key = 'confidence_interval'; }
    else if (lower.includes('ci_high') || lower.includes('ci low') || lower.includes('ci_low') || lower.includes('ci high')) { key = 'confidence_interval'; }
    else if ((lower.includes('trials') || lower.includes('trial')) && (lower.includes('reais') || lower.includes('real'))) { key = 'real_trials'; }
    else if (lower.includes('trials') || lower.includes('trial')) { key = 'trial'; }
    else if (lower.includes('tendência') || lower.includes('trend')) { key = 'trend'; }
    else if (lower.includes('melhor trial') || lower.includes('melhor (incumbent)') || lower.includes('incumbent')) { key = 'incumbent'; }
    else if (lower.includes('objetivo') || lower.includes('objective')) { key = 'objective'; }
    else if (lower.includes('performance') || lower.includes('dimensão')) { key = 'performance_dim'; }
    else if (lower.includes('trade-off')) { key = 'latency_tradeoff'; }
    else if (lower.includes('latência') || lower.includes('latency')) { key = 'inference_time'; }
    else if (lower.includes('duração')) { key = 'duration'; }
    else if (lower.includes('época') || lower.includes('epoca')) { key = 'epoch'; }
    else if (
        lower.includes('val_loss')
        || lower.includes('val loss')
        || lower.includes('validation loss')
        || ((lower.includes('validação') || lower.includes('validacao')) && lower.includes('loss'))
    ) { key = 'val_loss'; }
    else if (
        lower.includes('train_loss')
        || lower.includes('train loss')
        || lower.includes('training loss')
    ) { key = 'train_loss'; }
    else if (lower.includes('loss')) { key = 'loss'; }
    else if (lower.includes('stability') || lower.includes('delta') || lower.includes('improvement')) { key = 'stability'; }
    else if (lower.includes('mcc/mrr') || (lower.includes('mcc') && lower.includes('mrr'))) { key = 'mrr'; }
    else if (lower.includes('mcc')) { key = 'mcc'; }
    else if (lower.includes('mrr')) { key = 'mrr'; }
    else if (lower.includes('métricas') || lower.includes('metricas') || lower.includes('metrics')) { key = 'metrics'; }
    else if (lower.includes('recon')) { key = 'recon'; }
    else if (lower.includes('kl')) { key = 'kl_div'; }
    else if (lower.includes('rules') || lower.includes('regras')) { key = 'rules'; }
    else if ((lower.includes('grad') || lower.includes('gradient')) && lower.includes('norm')) { key = 'grad_norm'; }
    else if (/\bcpu\b/.test(lower)) { key = 'cpu'; }
    else if (/\bgpu\b/.test(lower)) { key = 'gpu'; }
    else if (/\bvram\b/.test(lower)) { key = 'vram'; }
    else if (/\bram\b/.test(lower)) { key = 'ram'; }
    else if (lower.includes('valor')) { key = 'value'; }
    else if (lower.includes('score')) { key = 'score'; }
    else if (lower.includes('id') || lower.includes('trial #') || lower.includes('trial')) { key = 'id'; }
    else if (lower.includes('importância')) { key = 'importance'; }
    else if (lower.includes('params') || lower.includes('parâmetros') || lower.includes('parâmetro')) { key = 'params'; }

    const metricHints = MetricRegistry.getAll();
    const hint = metricHints[key] ?? metricHints[lower.replace(/ /g, '_')] ?? null;

    if (!hint || hint.tech === key) return <span className="ml-2 text-zinc-400">{value}</span>;

    const TooltipContent = (
        <div className="w-72 p-4 bg-zinc-950 border border-zinc-800 rounded-xl shadow-2xl text-left font-sans z-50">
            <div className="space-y-3">
                <div className="pb-2 border-b border-zinc-900 flex justify-between items-center">
                    <span className="text-[10px] font-black text-white uppercase tracking-wider">{value}</span>
                    {extraValue && <span className="text-[10px] font-mono font-bold text-amber-400">{extraValue}</span>}
                </div>
                <div>
                    <span className="text-[8px] font-black text-orange-500 uppercase block mb-1">Explicação Técnica</span>
                    <p className="text-[10px] text-zinc-300 leading-tight normal-case">{hint.tech}</p>
                </div>
                {hint.simple && (
                    <div className="pt-2 border-t border-zinc-900">
                        <span className="text-[8px] font-black text-lime-500 uppercase block mb-1">Para Leigos</span>
                        <p className="text-[10px] text-lime-400/80 italic leading-tight normal-case border-l-2 border-lime-500/20 pl-2">{hint.simple}</p>
                    </div>
                )}
                {Array.isArray(hint.extra) && hint.extra.length > 0 && (
                    <div className="pt-2 border-t border-zinc-900">
                        <span className="text-[8px] font-black text-amber-400 uppercase block mb-1">Valores</span>
                        <div className="space-y-1">
                            {hint.extra.map((item, index) => (
                                <div key={`${item.label}-${index}`} className="text-[10px] text-zinc-300 leading-tight flex gap-2">
                                    <span className="text-amber-300/90 font-semibold min-w-[72px]">{item.label}:</span>
                                    <span className="text-zinc-300">{item.value}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
                {extraValue && (
                    <div className="pt-2 border-t border-zinc-900">
                        <span className="text-[8px] font-black text-amber-500 uppercase block mb-1">Valor Atual</span>
                        <p className="text-[11px] text-zinc-100 font-mono bg-zinc-900/50 p-1 rounded-sm border border-zinc-800">{extraValue}</p>
                    </div>
                )}
            </div>
            <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1.5 w-3 h-3 bg-zinc-950 border-r border-b border-zinc-800 rotate-45 shadow-lg"></div>
        </div>
    );

    return (
        <span className="ml-2 group relative cursor-help inline-flex items-center gap-1">
            <PortalTooltip content={TooltipContent}>
                <span className="text-zinc-300 border-b border-dotted border-zinc-600 hover:text-orange-400 transition-colors whitespace-nowrap">
                    {value}
                </span>
            </PortalTooltip>
            {hint.direction === 'up' && <ArrowUp size={10} />}
            {hint.direction === 'down' && <ArrowDown size={10} />}
        </span>
    );
};

// Aliases for compatibility
export const renderLegendWithHints = (val) => renderWithHints(val);

export const renderParamWithHints = (value, extraValue = null) => {
    const lower = String(value).toLowerCase();
    let key = lower.replace(/ /g, '_');

    if (lower.includes('rerank')) { key = 'rerank_top_k'; }
    else if (lower.includes('contrastive') && lower.includes('temperature')) { key = 'contrastive_temperature'; }
    else if (lower.includes('num') && lower.includes('global') && lower.includes('negative')) { key = 'num_global_negatives'; }
    else if (lower.includes('kl') && lower.includes('weight')) { key = 'kl_weight'; }

    const paramHints = ParamRegistry.getAll();
    const hint = paramHints[key] ?? paramHints[lower] ?? null;

    if (!hint || hint.tech === key) return <span className="ml-2" style={{ color: Theme.ui.text.secondary }}>{value}</span>;

    const TooltipContent = (
        <div className="w-72 p-4 bg-zinc-950 border border-zinc-800 rounded-xl shadow-2xl text-left font-sans z-50">
            <div className="space-y-3">
                <div className="pb-2 border-b border-zinc-900 flex justify-between items-center">
                    <span className="text-[10px] font-black text-white uppercase tracking-wider">{value}</span>
                    {extraValue && <span className="text-[10px] font-mono font-bold text-amber-400">{extraValue}</span>}
                </div>
                <div>
                    <span className="text-[8px] font-black text-orange-500 uppercase block mb-1">Explicação Técnica</span>
                    <p className="text-[10px] text-zinc-300 leading-tight normal-case">{hint.tech}</p>
                </div>
                {hint.simple && (
                    <div className="pt-2 border-t border-zinc-900">
                        <span className="text-[8px] font-black text-lime-500 uppercase block mb-1">Para Leigos</span>
                        <p className="text-[10px] text-lime-400/80 italic leading-tight normal-case border-l-2 border-lime-500/20 pl-2">{hint.simple}</p>
                    </div>
                )}
                {Array.isArray(hint.extra) && hint.extra.length > 0 && (
                    <div className="pt-2 border-t border-zinc-900">
                        <span className="text-[8px] font-black text-amber-400 uppercase block mb-1">Valores</span>
                        <div className="space-y-1">
                            {hint.extra.map((item, index) => (
                                <div key={`${item.label}-${index}`} className="text-[10px] text-zinc-300 leading-tight flex gap-2">
                                    <span className="text-amber-300/90 font-semibold min-w-[72px]">{item.label}:</span>
                                    <span className="text-zinc-300">{item.value}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
                {extraValue && (
                    <div className="pt-2 border-t border-zinc-900">
                        <span className="text-[8px] font-black text-amber-500 uppercase block mb-1">Valor Atual</span>
                        <p className="text-[11px] text-zinc-100 font-mono bg-zinc-900/50 p-1 rounded-sm border border-zinc-800">{extraValue}</p>
                    </div>
                )}
            </div>
            <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1.5 w-3 h-3 bg-zinc-950 border-r border-b border-zinc-800 rotate-45 shadow-lg"></div>
        </div>
    );

    return (
        <span className="ml-2 group relative cursor-help inline-flex items-center gap-1">
            <PortalTooltip content={TooltipContent}>
                <span className="text-zinc-300 border-b border-dotted border-zinc-600 hover:text-orange-400 transition-colors whitespace-nowrap">
                    {value}
                </span>
            </PortalTooltip>
            {hint.direction === 'up' && <ArrowUp size={10} />}
            {hint.direction === 'down' && <ArrowDown size={10} />}
        </span>
    );
};

export const ChartAxisLabel = ({ viewBox, value, axis = 'x', offset = 0 }) => {
    const { x, y, width, height } = viewBox;

    const isYRight = axis === 'y-right';
    const isX = axis === 'x';

    // Y-Axis (Left): Position 0 to Width (e.g., 60px). aligned to the left of the axis line (x)
    // Y-Axis (Right): Position X to X+Width.
    // X-Axis: Position X to X+Width, below the chart.

    let foreignX, foreignY, foreignW, foreignH;
    let containerClass = "flex items-center justify-center w-full h-full";

    if (isX) {
        foreignX = x;
        foreignY = y + 20; // Push down below ticks
        foreignW = width;
        foreignH = 30;
    } else {
        // Y Axis (Left or Right) - We create a box for the label
        // Rotation center is the middle of this box
        const axisWidth = width || 60; // Fallback

        foreignW = height; // Swapped for rotation logic if we were using pure SVG transforms, but for foreignObject we keep it simple
        foreignH = axisWidth;

        // Actually, easiest way for HTML in SVG rotation is to place a box and rotate the content
        foreignW = axisWidth;
        foreignH = height;

        if (isYRight) {
            foreignX = x + 10 + offset;
        } else {
            foreignX = x - axisWidth - offset; // Place in the margin area left of axis line
        }
        foreignY = y;

        // Rotate text -90deg
        containerClass = "flex items-center justify-center w-full h-full -rotate-90 origin-center whitespace-nowrap";
    }

    return (
        <foreignObject x={foreignX} y={foreignY} width={foreignW} height={foreignH} style={{ overflow: 'visible' }}>
            <div className={containerClass}>
                {renderWithHints(value)}
            </div>
        </foreignObject>
    );
};
