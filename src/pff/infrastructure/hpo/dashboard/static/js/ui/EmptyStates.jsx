import React from 'react';
import { Theme } from './Theme.js';
import { PortalTooltip } from './PortalTooltip.jsx';

/**
 * Empty state components, data guards, skeleton loaders, and base tooltip.
 */

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
        <div className={`${className} h-full w-full`}>
            <PortalTooltip content={tooltipContent} className="h-full w-full block">{trigger}</PortalTooltip>
        </div>
    );
});

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

export const EmptyStatePulso = React.memo(({
    title = "Aguardando dados...",
    subtitle,
    icon: Icon,
    mood = "waiting",
    className = "",
    children
}) => {
    const moodConfig = {
        waiting: { iconColor: 'var(--viz-palette-4-yellow)', pulseClass: 'pulso-logo', textColor: Theme.ui.text.muted },
        idle: { iconColor: 'var(--viz-palette-7-cyan)', pulseClass: '', textColor: Theme.ui.text.muted },
        success: { iconColor: 'var(--viz-palette-2-green)', pulseClass: '', textColor: Theme.semantic.success },
        error: { iconColor: 'var(--viz-palette-5-red)', pulseClass: '', textColor: Theme.semantic.error }
    };
    const config = moodConfig[mood];

    return (
        <div className={`h-full flex flex-col items-center justify-center p-8 text-center ${className}`}>
            <div className={`relative w-16 h-16 mb-4 ${config.pulseClass}`}>
                {Icon ? (
                    <Icon size={32} style={{ color: config.iconColor }} />
                ) : (
                    <div className="w-full h-full flex items-center justify-center" style={{ color: config.iconColor }}>
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="pulso-logo-icon">
                            <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                        </svg>
                    </div>
                )}
            </div>
            <h3 className="text-sm font-semibold mb-2" style={{ color: config.textColor }}>{title}</h3>
            {subtitle && (
                <p className="text-xs max-w-xs" style={{ color: Theme.ui.text.secondary, lineHeight: '1.5' }}>{subtitle}</p>
            )}
            {children && <div className="mt-4">{children}</div>}
        </div>
    );
});

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
