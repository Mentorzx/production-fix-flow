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

/** @type {React.FC<import('recharts').CartesianGridProps>} */
export const DefaultCartesianGrid = React.memo((props) => (
    <CartesianGrid strokeDasharray="3 3" stroke={Theme.ui.grid} strokeOpacity={0.5} {...props} />
));

/**
 * @typedef {Object} DefaultTooltipProps
 * @property {Object} [wrapperStyle]
 * @property {Object} [contentStyle]
 * @property {Object} [itemStyle]
 * @property {Object} [cursor]
 */

/** @type {React.FC<DefaultTooltipProps & import('recharts').TooltipProps<any, any>>} */
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

/**
 * @typedef {Object} IconBaseProps
 * @property {React.ReactNode} [d] - The SVG path content (provided by icon components)
 * @property {string} [className]
 * @property {number} [size]
 * @property {React.CSSProperties} [style]
 */

/** @type {React.FC<IconBaseProps>} */
export const IconBase = ({ d, className = "", size = 16, style = {} }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className} style={style}>{d}</svg>
);

/** @type {React.FC<IconBaseProps>} */
export const Activity = (p) => <IconBase {...p} d={<path d="M22 12h-4l-3 9L9 3l-3 9H2" />} />;
/** @type {React.FC<IconBaseProps>} */
export const RefreshCw = (p) => <IconBase {...p} d={<><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" /><path d="M21 3v5h-5" /><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" /><path d="M8 16H3v5" /></>} />;
/** @type {React.FC<IconBaseProps>} */
export const Zap = (p) => <IconBase {...p} d={<path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />} />;
/** @type {React.FC<IconBaseProps>} */
export const Clock = (p) => <IconBase {...p} d={<><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></>} />;
/** @type {React.FC<IconBaseProps>} */
export const TrendingUp = (p) => <IconBase {...p} d={<polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />} />;
/** @type {React.FC<IconBaseProps>} */
export const Layers = (p) => <IconBase {...p} d={<><polygon points="12 2 2 7 12 12 22 7 12 2" /><polyline points="2 17 12 22 22 17" /><polyline points="2 12 12 17 22 12" /></>} />;
/** @type {React.FC<IconBaseProps>} */
export const TableIcon = (p) => <IconBase {...p} d={<path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18" />} />;
/** @type {React.FC<IconBaseProps>} */
export const Sliders = (p) => <IconBase {...p} d={<><line x1="4" y1="21" x2="4" y2="14" /><line x1="4" y1="10" x2="4" y2="3" /><line x1="12" y1="21" x2="12" y2="12" /><line x1="12" y1="8" x2="12" y2="3" /><line x1="20" y1="21" x2="20" y2="16" /><line x1="20" y1="12" x2="20" y2="3" /><line x1="1" y1="14" x2="7" y2="14" /><line x1="9" y1="8" x2="15" y2="8" /><line x1="17" y1="16" x2="23" y2="16" /></>} />;
/** @type {React.FC<IconBaseProps>} */
export const GitMerge = (p) => <IconBase {...p} d={<><circle cx="18" cy="18" r="3" /><circle cx="6" cy="6" r="3" /><path d="M6 21V9a9 9 0 0 0 9 9" /></>} />;
/** @type {React.FC<IconBaseProps>} */
export const Download = (p) => <IconBase {...p} d={<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" />} />;
/** @type {React.FC<IconBaseProps>} */
export const TargetIcon = (p) => <IconBase {...p} d={<><circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="6" /><circle cx="12" cy="12" r="2" /></>} />;
/** @type {React.FC<IconBaseProps>} */
export const Cpu = (p) => <IconBase {...p} d={<><rect x="4" y="4" width="16" height="16" rx="2" ry="2" /><rect x="9" y="9" width="6" height="6" /><line x1="9" y1="1" x2="9" y2="4" /><line x1="15" y1="1" x2="15" y2="4" /><line x1="9" y1="20" x2="9" y2="23" /><line x1="15" y1="20" x2="15" y2="23" /><line x1="20" y1="9" x2="23" y2="9" /><line x1="20" y1="12" x2="23" y2="14" /><line x1="1" y1="9" x2="4" y2="9" /><line x1="1" y1="14" x2="4" y2="14" /></>} />;
/** @type {React.FC<IconBaseProps>} */
export const Microscope = (p) => <IconBase {...p} d={<><path d="M6 18h8" /><path d="M3 22h18" /><path d="M14 22a7 7 0 1 0 0-14h-1" /><path d="M9 14h2" /><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z" /><path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3" /></>} />;
/** @type {React.FC<IconBaseProps>} */
export const Share2 = (p) => <IconBase {...p} d={<><circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" /><line x1="8.59" y1="13.51" x2="15.42" y2="17.49" /><line x1="15.41" y1="6.51" x2="8.59" y2="10.49" /></>} />;
/** @type {React.FC<IconBaseProps>} */
export const AlertTriangle = (p) => <IconBase {...p} d={<><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></>} />;
/** @type {React.FC<IconBaseProps>} */
export const AlertOctagon = (p) => <IconBase {...p} d={<><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" /></>} />;
/** @type {React.FC<IconBaseProps>} */
export const CheckCircle = (p) => <IconBase {...p} d={<><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" /></>} />;
/** @type {React.FC<IconBaseProps>} */
export const X = (p) => <IconBase {...p} d={<><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></>} />;
/** @type {React.FC<IconBaseProps>} */
export const ChevronRight = (p) => <IconBase {...p} d={<polyline points="9 18 15 12 9 6" />} />;
/** @type {React.FC<IconBaseProps>} */
export const Search = (p) => <IconBase {...p} d={<><circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /></>} />;
/** @type {React.FC<IconBaseProps>} */
export const BarChart2 = (p) => <IconBase {...p} d={<><line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" /><line x1="6" y1="20" x2="6" y2="14" /></>} />;
/** @type {React.FC<IconBaseProps>} */
export const Terminal = (p) => <IconBase {...p} d={<><polyline points="4 17 10 11 4 5" /><line x1="12" y1="19" x2="20" y2="19" /></>} />;
/** @type {React.FC<IconBaseProps>} */
export const Info = (p) => <IconBase {...p} d={<><circle cx="12" cy="12" r="10" /><line x1="12" y1="16" x2="12" y2="12" /><line x1="12" y1="8" x2="12.01" y2="8" /></>} />;
/** @type {React.FC<IconBaseProps>} */
export const HelpCircle = (p) => <IconBase {...p} d={<><circle cx="12" cy="12" r="10" /><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" /><line x1="12" y1="17" x2="12.01" y2="17" /></>} />;

/**
 * @typedef {Object} ArrowProps
 * @property {string} [className]
 * @property {number} [size]
 */

/** @type {React.FC<ArrowProps>} */
export const ArrowUp = ({ className = "", size = 10 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={Theme.semantic.success} strokeWidth="3" className={className}><polyline points="18 15 12 9 6 15" /></svg>
);
/** @type {React.FC<ArrowProps>} */
export const ArrowDown = ({ className = "", size = 10 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={Theme.semantic.chart.loss} strokeWidth="3" className={className}><polyline points="6 9 12 15 18 9" /></svg>
);

/**
 * @typedef {Object} PortalTooltipProps
 * @property {React.ReactNode} content - The tooltip content
 * @property {React.ReactNode} children - The trigger element
 * @property {string} [className] - CSS class for the trigger wrapper
 */

/** @type {React.FC<PortalTooltipProps>} */
export const PortalTooltip = ({ content, children, className = "inline-block" }) => {
    const [visible, setVisible] = useState(false);
    const [coords, setCoords] = useState({ top: 0, left: 0 });
    const triggerRef = useRef(null);
    const tooltipRef = useRef(null);
    const isOverTrigger = useRef(false);
    const isOverTooltip = useRef(false);

    const checkHide = () => {
        if (!isOverTrigger.current && !isOverTooltip.current) {
            setVisible(false);
        }
    };

    const handleTriggerEnter = () => {
        isOverTrigger.current = true;
        if (triggerRef.current) {
            const rect = triggerRef.current.getBoundingClientRect();
            setCoords({ top: rect.top - 8, left: rect.left + (rect.width / 2) });
            setVisible(true);
        }
    };

    const handleTriggerLeave = () => {
        isOverTrigger.current = false;
        setTimeout(checkHide, 100);
    };

    const handleTooltipEnter = () => {
        isOverTooltip.current = true;
    };

    const handleTooltipLeave = () => {
        isOverTooltip.current = false;
        setTimeout(checkHide, 100);
    };

    return (
        <React.Fragment>
            <div
                ref={triggerRef}
                onMouseEnter={handleTriggerEnter}
                onMouseLeave={handleTriggerLeave}
                className={className}
                style={{ display: className?.includes('contents') ? 'contents' : undefined }}
            >
                {children}
            </div>
            {visible && createPortal(
                <div
                    ref={tooltipRef}
                    onMouseEnter={handleTooltipEnter}
                    onMouseLeave={handleTooltipLeave}
                    className="fixed z-[99999] animate-in fade-in zoom-in-95 duration-100 pointer-events-auto"
                    style={{ top: coords.top, left: coords.left, transform: 'translate(-50%, -100%)', filter: 'drop-shadow(0 4px 20px rgba(0,0,0,0.5))' }}
                >
                    {content}
                </div>,
                document.body
            )}
        </React.Fragment>
    );
};

/**
 * @typedef {Object} HelpTextItem
 * @property {string} label
 * @property {string} value
 *
 * @typedef {Object} HelpTextObject
 * @property {React.ReactNode} tech
 * @property {React.ReactNode} [simple]
 * @property {HelpTextItem[]} [extra]
 *
 * @typedef {Object} HelpButtonProps
 * @property {React.ReactNode | HelpTextObject} text
 */

/** @type {React.FC<HelpButtonProps>} */
export const HelpButton = React.memo(({ text }) => {
    /**
     * Type guard to check if text is a HelpTextObject
     * @param {React.ReactNode | HelpTextObject} value
     * @returns {value is HelpTextObject}
     */
    const isHelpTextObject = (value) => {
        return typeof value === 'object' && value !== null && 'tech' in value;
    };

    const tooltipContent = (
        <div className="w-64 border p-3 rounded-xl shadow-2xl text-[10px]" style={{ backgroundColor: Theme.ui.background, borderColor: Theme.ui.border, color: Theme.ui.text.secondary }}>
            {isHelpTextObject(text) ? (
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
        <div className="inline-block ml-2">
            <PortalTooltip content={tooltipContent}>
                <div
                    className="cursor-help w-6 h-6 flex items-center justify-center border-2 hover:scale-110 hover:brightness-125 transition-all duration-200"
                    style={{
                        borderColor: Theme.palette.cyberYellow,
                        backgroundColor: 'rgba(229, 197, 88, 0.1)',
                        color: Theme.palette.cyberYellow,
                        borderRadius: '6px',
                        boxShadow: '0 0 8px rgba(229, 197, 88, 0.3)'
                    }}
                >
                    <Info size={14} />
                </div>
            </PortalTooltip>
        </div>
    );
});

/**
 * @typedef {Object} CardProps
 * @property {React.ReactNode} children
 * @property {string} [className]
 * @property {string} [title]
 * @property {React.FC<IconBaseProps>} [icon]
 * @property {React.ReactNode} [action]
 * @property {React.ReactNode | Object} [helpText]
 * @property {boolean} [glow] - Whether to show a glow effect on the card
 */

export const StatBadge = React.memo(({ label, value, subtext, color = "orange" }) => {
    const themeColors = {
        orange: Theme.palette.hotOrange,
        lime: Theme.palette.lime,
        amber: Theme.palette.cyberYellow,
        rose: Theme.palette.red,
        zinc: Theme.palette.grey
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

    return (
        <div className="p-6 rounded-2xl border shadow-xl flex flex-col justify-center min-h-[140px]" style={{ backgroundColor: Theme.ui.surface, borderColor: activeColor + '33' }}>
            <span className="text-[10px] font-black uppercase tracking-[0.2em] opacity-40 mb-2" style={{ color: activeColor }}>{label}</span>
            <div key={flipKey} className="text-4xl font-black font-mono tracking-tighter pff-flip" style={{ color: Theme.ui.text.primary }}>{value}</div>
            {subtext && <div className="text-[10px] opacity-40 font-bold uppercase mt-1" style={{ color: activeColor }}>{subtext}</div>}
        </div>
    );
});

/** @type {React.FC<CardProps>} */
export const Card = React.memo(({ children, className = "", title, icon: Icon, action, helpText, glow }) => (
    <article
        className={`rounded-xl flex flex-col relative overflow-hidden card-edge ${className} ${glow ? 'card-edge-active' : ''}`}
        style={{ backgroundColor: Theme.ui.surface }}
    >
        <div className="pff-micro-orbit" aria-hidden="true" style={{ opacity: 0.05 }}></div>
        {(title || Icon) && (
            <header className="flex items-center justify-between px-5 py-5 border-b" style={{ borderColor: Theme.ui.border, backgroundColor: Theme.ui.surfaceHighlight + '40' }}>
                <div className="flex items-center gap-2.5">
                    {Icon && <div className="p-1 rounded-sm" style={{ backgroundColor: Theme.ui.background, color: Theme.semantic.warning }}><Icon size={14} /></div>}
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

/**
 * @typedef {Object} ChartFrameProps
 * @property {React.ReactNode} children
 * @property {string} [className]
 */

/** @type {React.FC<ChartFrameProps>} */
export const ChartFrame = React.memo(({ children, className = "" }) => (
    <div className={`relative p-5 w-full h-full min-h-[220px] min-w-0 ${className}`}>{children}</div>
));

/**
 * @typedef {Object} ChartContainerProps
 * @property {React.ReactElement} children
 * @property {string} [className]
 * @property {number} [minHeight]
 */

/** @type {React.FC<ChartContainerProps>} */
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


/**
 * @typedef {Object} EmptyStateProps
 * @property {React.ReactNode} children
 * @property {string} [className]
 */

/** @type {React.FC<EmptyStateProps>} */
export const EmptyState = React.memo(({ children, className = "" }) => (
    <div className={`h-full flex items-center justify-center italic text-xs ${className}`} style={{ color: Theme.ui.text.muted }}>{children}</div>
));

/**
 * @typedef {Object} WithDataProps
 * @property {boolean} when
 * @property {React.ReactNode} empty
 * @property {string} [emptyClassName]
 * @property {React.ReactNode} children
 */

/** @type {React.FC<WithDataProps>} */
export const WithData = React.memo(({ when, empty, emptyClassName = "", children }) => (
    when ? children : <EmptyState className={emptyClassName}>{empty}</EmptyState>
));

/**
 * @typedef {Object} BaseTooltipProps
 * @property {React.ReactNode} trigger
 * @property {string} [title]
 * @property {React.ReactNode} tech
 * @property {string} [className]
 */

/** @type {React.FC<BaseTooltipProps>} */
export const BaseTooltip = React.memo(({ trigger, title, tech, className = "inline-block" }) => {
    const tooltipContent = (
        <div className="w-64 border p-3 rounded-xl shadow-2xl text-[10px]" style={{ backgroundColor: Theme.ui.background, borderColor: Theme.ui.border, color: Theme.ui.text.secondary }}>
            {title && <div className="font-bold border-b pb-1 mb-1" style={{ borderColor: Theme.ui.border, color: Theme.ui.text.primary }}>{title}</div>}
            {tech}
        </div>
    );

    return (
        <div className={`${className} h-full w-full`}>
            <PortalTooltip content={tooltipContent} className="h-full w-full block">{trigger}</PortalTooltip>
        </div>
    );
});

/**
 * @typedef {Object} SkeletonProps
 * @property {string} [className]
 * @property {'text' | 'header' | 'metric' | 'chart' | 'matrix' | 'custom'} [variant]
 * @property {number} [delay]
 * @property {string} [width]
 * @property {string} [height]
 */

/** @type {React.FC<SkeletonProps>} */
export const Skeleton = React.memo(({
    className = "",
    variant = "text",
    delay = 0,
    width,
    height
}) => {
    const delayClass = delay > 0 ? `skeleton-delay-${delay}` : '';

    const variantClasses = {
        text: 'skeleton skeleton-text',
        header: 'skeleton skeleton-header',
        metric: 'skeleton skeleton-metric',
        chart: 'skeleton skeleton-chart',
        matrix: 'skeleton skeleton-matrix',
        custom: 'skeleton'
    };

    const style = {};
    if (width) style.width = width;
    if (height) style.height = height;

    return (
        <div
            className={`${variantClasses[variant]} ${delayClass} ${className}`}
            style={style}
            aria-hidden="true"
        />
    );
});

/**
 * @typedef {Object} EmptyStatePulsoProps
 * @property {string} [title] - Main message
 * @property {string} [subtitle] - Secondary message with guidance
 * @property {React.ComponentType<{size?: number, style?: React.CSSProperties}>} [icon] - Custom icon component
 * @property {'waiting' | 'idle' | 'success' | 'error'} [mood] - Visual mood
 * @property {string} [className]
 * @property {React.ReactNode} [children] - Additional content (CTA, etc)
 */

/** @type {React.FC<EmptyStatePulsoProps>} */
export const EmptyStatePulso = React.memo(({
    title = "Aguardando dados...",
    subtitle,
    icon: Icon,
    mood = "waiting",
    className = "",
    children
}) => {
    const moodConfig = {
        waiting: {
            iconColor: 'var(--viz-palette-4-yellow)',
            pulseClass: 'pulso-logo',
            textColor: Theme.ui.text.muted
        },
        idle: {
            iconColor: 'var(--viz-palette-7-cyan)',
            pulseClass: '',
            textColor: Theme.ui.text.muted
        },
        success: {
            iconColor: 'var(--viz-palette-2-green)',
            pulseClass: '',
            textColor: Theme.semantic.success
        },
        error: {
            iconColor: 'var(--viz-palette-5-red)',
            pulseClass: '',
            textColor: Theme.semantic.error
        }
    };

    const config = moodConfig[mood];

    return (
        <div className={`h-full flex flex-col items-center justify-center p-8 text-center ${className}`}>
            {/* PULSO Heartbeat Icon */}
            <div className={`relative w-16 h-16 mb-4 ${config.pulseClass}`}>
                {Icon ? (
                    <Icon size={32} style={{ color: config.iconColor }} />
                ) : (
                    <div
                        className="w-full h-full flex items-center justify-center"
                        style={{ color: config.iconColor }}
                    >
                        <svg
                            width="32"
                            height="32"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            className="pulso-logo-icon"
                        >
                            <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                        </svg>
                    </div>
                )}
            </div>

            {/* Title - Humanized */}
            <h3
                className="text-sm font-semibold mb-2"
                style={{ color: config.textColor }}
            >
                {title}
            </h3>

            {/* Subtitle - Helpful guidance */}
            {subtitle && (
                <p
                    className="text-xs max-w-xs"
                    style={{ color: Theme.ui.text.secondary, lineHeight: '1.5' }}
                >
                    {subtitle}
                </p>
            )}

            {/* Additional content (CTA, etc) */}
            {children && (
                <div className="mt-4">
                    {children}
                </div>
            )}
        </div>
    );
});

/**
 * Enhanced WithData that uses EmptyStatePulso
 * @typedef {Object} WithDataPulsoProps
 * @property {boolean} when
 * @property {string} [title]
 * @property {string} [subtitle]
 * @property {'waiting' | 'idle' | 'success' | 'error'} [mood]
 * @property {string} [className]
 * @property {React.ReactNode} children
 */

/** @type {React.FC<WithDataPulsoProps>} */
export const WithDataPulso = React.memo(({
    when,
    title = "Aguardando dados...",
    subtitle,
    mood = "waiting",
    className = "",
    children
}) => (
    when ? children : <EmptyStatePulso title={title} subtitle={subtitle} mood={mood} className={className} />
));
