import React, { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Tooltip, CartesianGrid } from 'recharts';
import { Theme } from './Theme.js';

// Adapter for legacy chart components to use SOTA Theme
export const colors = {
    bg: Theme.ui.background,
    card: Theme.ui.surface,
    border: Theme.ui.border,
    text: Theme.ui.text.secondary,
    textHigh: Theme.ui.text.primary,
    primary: Theme.semantic.primary, // Maps to Neon Blue
    orange: Theme.palette.hotOrange,
    success: Theme.semantic.success,
    lime: Theme.palette.lime,
    amber: Theme.palette.cyberYellow,
    error: Theme.semantic.error,
    warning: Theme.semantic.warning,
    grid: Theme.ui.grid,
    tooltip: Theme.ui.background,
};

export const defaultChartMargins = { top: 25, right: 15, bottom: 35, left: 50 };

export const defaultTooltipStyle = {
    backgroundColor: Theme.ui.background,
    borderColor: Theme.ui.border,
    color: Theme.ui.text.primary,
    fontSize: '11px',
    borderRadius: '8px',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.5)'
};

export const DefaultCartesianGrid = React.memo((props) => (
    <CartesianGrid strokeDasharray="3 3" stroke={Theme.ui.grid} strokeOpacity={0.5} {...props} />
));

export const DefaultTooltip = React.memo((props) => {
    const { wrapperStyle, ...rest } = props;
    return (
        <Tooltip
            contentStyle={defaultTooltipStyle}
            itemStyle={{ color: Theme.ui.text.secondary }}
            cursor={{ stroke: Theme.ui.grid, strokeDasharray: '3 3' }}
            wrapperStyle={{ zIndex: 60, ...wrapperStyle }}
            {...rest}
        />
    );
});

export const DefaultTooltipCursor = { strokeDasharray: '3 3', stroke: Theme.ui.grid };

export const IconBase = ({ d, className = "", size = 16, style = {} }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className} style={style}>{d}</svg>
);

export const Activity = (p) => <IconBase {...p} d={<path d="M22 12h-4l-3 9L9 3l-3 9H2" />} />;
export const RefreshCw = (p) => <IconBase {...p} d={<><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" /><path d="M21 3v5h-5" /><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" /><path d="M8 16H3v5" /></>} />;
export const Zap = (p) => <IconBase {...p} d={<path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />} />;
export const Clock = (p) => <IconBase {...p} d={<><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></>} />;
export const TrendingUp = (p) => <IconBase {...p} d={<polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />} />;
export const Layers = (p) => <IconBase {...p} d={<><polygon points="12 2 2 7 12 12 22 7 12 2" /><polyline points="2 17 12 22 22 17" /><polyline points="2 12 12 17 22 12" /></>} />;
export const TableIcon = (p) => <IconBase {...p} d={<path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18" />} />;
export const Sliders = (p) => <IconBase {...p} d={<><line x1="4" y1="21" x2="4" y2="14" /><line x1="4" y1="10" x2="4" y2="3" /><line x1="12" y1="21" x2="12" y2="12" /><line x1="12" y1="8" x2="12" y2="3" /><line x1="20" y1="21" x2="20" y2="16" /><line x1="20" y1="12" x2="20" y2="3" /><line x1="1" y1="14" x2="7" y2="14" /><line x1="9" y1="8" x2="15" y2="8" /><line x1="17" y1="16" x2="23" y2="16" /></>} />;
export const GitMerge = (p) => <IconBase {...p} d={<><circle cx="18" cy="18" r="3" /><circle cx="6" cy="6" r="3" /><path d="M6 21V9a9 9 0 0 0 9 9" /></>} />;
export const Download = (p) => <IconBase {...p} d={<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" />} />;
export const TargetIcon = (p) => <IconBase {...p} d={<><circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="6" /><circle cx="12" cy="12" r="2" /></>} />;
export const Cpu = (p) => <IconBase {...p} d={<><rect x="4" y="4" width="16" height="16" rx="2" ry="2" /><rect x="9" y="9" width="6" height="6" /><line x1="9" y1="1" x2="9" y2="4" /><line x1="15" y1="1" x2="15" y2="4" /><line x1="9" y1="20" x2="9" y2="23" /><line x1="15" y1="20" x2="15" y2="23" /><line x1="20" y1="9" x2="23" y2="9" /><line x1="20" y1="14" x2="23" y2="14" /><line x1="1" y1="9" x2="4" y2="9" /><line x1="1" y1="14" x2="4" y2="14" /></>} />;
export const Microscope = (p) => <IconBase {...p} d={<><path d="M6 18h8" /><path d="M3 22h18" /><path d="M14 22a7 7 0 1 0 0-14h-1" /><path d="M9 14h2" /><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z" /><path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3" /></>} />;
export const Share2 = (p) => <IconBase {...p} d={<><circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" /><line x1="8.59" y1="13.51" x2="15.42" y2="17.49" /><line x1="15.41" y1="6.51" x2="8.59" y2="10.49" /></>} />;
export const AlertTriangle = (p) => <IconBase {...p} d={<><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></>} />;
export const ChevronRight = (p) => <IconBase {...p} d={<polyline points="9 18 15 12 9 6" />} />;
export const Search = (p) => <IconBase {...p} d={<><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></>} />;
export const BarChart2 = (p) => <IconBase {...p} d={<><line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" /><line x1="6" y1="20" x2="6" y2="14" /></>} />;
export const Terminal = (p) => <IconBase {...p} d={<><polyline points="4 17 10 11 4 5" /><line x1="12" y1="19" x2="20" y2="19" /></>} />;
export const Info = (p) => <IconBase {...p} d={<><circle cx="12" cy="12" r="10" /><line x1="12" y1="16" x2="12" y2="12" /><line x1="12" y1="8" x2="12.01" y2="8" /></>} />;
export const HelpCircle = (p) => <IconBase {...p} d={<><circle cx="12" cy="12" r="10" /><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" /><line x1="12" y1="17" x2="12.01" y2="17" /></>} />;

export const ArrowUp = ({ className = "", size = 10 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={Theme.semantic.success} strokeWidth="3" className={className}><polyline points="18 15 12 9 6 15" /></svg>
);
export const ArrowDown = ({ className = "", size = 10 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={Theme.semantic.chart.loss} strokeWidth="3" className={className}><polyline points="6 9 12 15 18 9" /></svg>
);

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
                <div className="fixed z-9999 pointer-events-none animate-in fade-in zoom-in-95 duration-100" style={{ top: coords.top, left: coords.left, transform: 'translate(-50%, -100%)' }}>
                    {content}
                </div>,
                document.body
            )}
        </React.Fragment>
    );
};

export const HelpButton = React.memo(({ text }) => {
    const tooltipContent = (
        <div className="w-64 border p-3 rounded-xl shadow-2xl text-[10px]" style={{ backgroundColor: Theme.ui.background, borderColor: Theme.ui.border, color: Theme.ui.text.secondary }}>
            {typeof text === 'object' ? (
                <div className="space-y-2">
                    <div>
                        <span className="text-[8px] font-black uppercase block mb-1" style={{ color: Theme.semantic.warning }}>Explicação Técnica</span>
                        <div className="leading-tight" style={{ color: Theme.ui.text.primary }}>{text.tech}</div>
                    </div>
                    {text.simple && (
                        <div className="pt-2 border-t" style={{ borderColor: Theme.ui.border }}>
                            <span className="text-[8px] font-black uppercase block mb-1" style={{ color: Theme.semantic.success }}>Para Leigos</span>
                            <div className="italic leading-tight border-l-2 pl-2" style={{ color: Theme.palette.mint, borderColor: Theme.palette.vividGreen + '33' }}>{text.simple}</div>
                        </div>
                    )}
                    {Array.isArray(text.extra) && text.extra.length > 0 && (
                        <div className="pt-2 border-t" style={{ borderColor: Theme.ui.border }}>
                            <span className="text-[8px] font-black uppercase block mb-1" style={{ color: Theme.palette.cyberYellow }}>Valores</span>
                            <div className="space-y-1">
                                {text.extra.map((item, index) => (
                                    <div key={`${item.label}-${index}`} className="text-[10px] leading-tight flex gap-2" style={{ color: Theme.ui.text.secondary }}>
                                        <span className="font-semibold min-w-[72px]" style={{ color: Theme.palette.apricot }}>{item.label}:</span>
                                        <span>{item.value}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            ) : (
                text
            )}
        </div>
    );
    return (
        <div className="inline-block ml-1">
            <PortalTooltip content={tooltipContent}>
                <div className="cursor-help p-0.5 border rounded-sm text-[8px]" style={{ borderColor: Theme.ui.border, backgroundColor: Theme.ui.surfaceHighlight, color: Theme.ui.text.muted }}>?</div>
            </PortalTooltip>
        </div>
    );
});

export const Card = React.memo(({ children, className = "", title, icon: Icon, action, helpText }) => (
    <article className={`rounded-xl shadow-lg flex flex-col transition-all duration-300 relative overflow-hidden ${className}`} style={{ backgroundColor: Theme.ui.surface, border: `1px solid ${Theme.ui.border}` }}>
        <div className="pff-micro-orbit" aria-hidden="true" style={{ opacity: 0.05 }}></div>
        {(title || Icon) && (
            <header className="flex items-center justify-between px-5 py-3 border-b" style={{ borderColor: Theme.ui.border, backgroundColor: Theme.ui.surfaceHighlight + '40' }}>
                <div className="flex items-center gap-2.5">
                    {Icon && <div className="p-1 rounded-sm shadow-inner" style={{ backgroundColor: Theme.ui.background, color: Theme.semantic.warning }}><Icon size={14} /></div>}
                    <h3 className="font-black text-[10px] uppercase tracking-widest" style={{ color: Theme.ui.text.primary }}>{title}</h3>
                </div>
                <div className="flex items-center gap-2">
                    {helpText && <HelpButton text={helpText} />}
                    {action}
                </div>
            </header>
        )}
        <div className="p-5 flex-1 relative flex flex-col min-h-0">{children}</div>
    </article>
));

export const ChartFrame = React.memo(({ children, className = "" }) => (
    <div className={`relative p-5 w-full h-full min-h-[220px] min-w-0 ${className}`}>{children}</div>
));

export const ChartContainer = React.memo(({ children, className = "", minHeight = 200 }) => {
    const containerRef = useRef(null);
    const [size, setSize] = useState({ width: 0, height: 0 });

    useEffect(() => {
        const container = containerRef.current;
        if (!container) return undefined;
        const observer = new ResizeObserver((entries) => {
            const entry = entries[0];
            if (!entry) return;
            const { width, height } = entry.contentRect;
            if (width > 0 && height > 0) {
                setSize({ width, height });
            } else {
                setSize({ width: 0, height: 0 });
            }
        });
        observer.observe(container);
        return () => observer.disconnect();
    }, []);

    const ready = size.width > 0 && size.height > 0;
    const child = React.Children.only(children);

    return (
        <div
            ref={containerRef}
            className={`w-full h-full min-w-0 ${className}`}
            style={{ minWidth: 0, minHeight }}
        >
            {ready && React.isValidElement(child) && React.cloneElement(child, { width: size.width, height: size.height })}
        </div>
    );
});

export const StatBadge = React.memo(({ label, value, subtext, color = "orange" }) => {
    const themeColors = {
        orange: Theme.palette.hotOrange,
        lime: Theme.palette.vividGreen,
        amber: Theme.palette.cyberYellow,
        rose: Theme.palette.red,
        zinc: Theme.palette.grey
    };

    const activeColor = themeColors[color] || Theme.palette.neonBlue;
    const [flipKey, setFlipKey] = useState(0);
    const prevValue = useRef(value);

    useEffect(() => {
        if (prevValue.current !== value) {
            setFlipKey((k) => k + 1);
            prevValue.current = value;
        }
    }, [value]);

    return (
        <div className="p-6 rounded-2xl border shadow-xl flex flex-col justify-center min-h-[140px]" style={{ backgroundColor: Theme.ui.surface, borderColor: activeColor + '33' }}>
            <span className="text-[10px] font-black uppercase tracking-[0.2em] opacity-40 mb-2" style={{ color: activeColor }}>{label}</span>
            <div key={flipKey} className="text-4xl font-black font-mono tracking-tighter pff-flip" style={{ color: Theme.ui.text.primary }}>{value}</div>
            {subtext && <div className="text-[10px] opacity-40 font-bold uppercase mt-1" style={{ color: activeColor }}>{subtext}</div>}
        </div>
    );
});

export const EmptyState = React.memo(({ children, className = "" }) => (
    <div className={`h-full flex items-center justify-center italic text-xs ${className}`} style={{ color: Theme.ui.text.muted }}>{children}</div>
));

export const WithData = React.memo(({ when, empty, emptyClassName = "", children }) => (
    when ? children : <EmptyState className={emptyClassName}>{empty}</EmptyState>
));

export const BaseTooltip = React.memo(({ trigger, title, tech, className = "inline-block" }) => {
    const tooltipContent = (
        <div className="w-64 border p-3 rounded-xl shadow-2xl text-[10px]" style={{ backgroundColor: Theme.ui.background, borderColor: Theme.ui.border, color: Theme.ui.text.secondary }}>
            {title && <div className="font-bold border-b pb-1 mb-1" style={{ borderColor: Theme.ui.border, color: Theme.ui.text.primary }}>{title}</div>}
            {tech}
        </div>
    );

    return (
        <div className={className}>
            <PortalTooltip content={tooltipContent}>{trigger}</PortalTooltip>
        </div>
    );
});
