import React from 'react';
import { ArrowUp, ArrowDown, Download, ChevronRight } from "./BaseComponents.jsx";
import { PortalTooltip } from "./UIComponents.jsx";
import { ExportService } from "../utils/ExportService.js";

export const SortedTableHeader = React.memo(({ label, sortKey, currentSort, onSort, helpText, directionHint, align = 'left' }) => {
    const active = currentSort?.key === sortKey;
    const direction = active ? currentSort.direction : 'none';

    const hintUp = directionHint === 'up';
    const hintDown = directionHint === 'down';

    // SOTA Theme Colors for Sort Indicators
    const upClass = active && direction === 'asc'
        ? "text-orange-400"
        : hintUp
            ? "text-emerald-400/90 group-hover:text-emerald-300"
            : "text-zinc-800 opacity-20";

    const downClass = active && direction === 'desc'
        ? "text-orange-400"
        : hintDown
            ? "text-rose-400/90 group-hover:text-rose-300"
            : "text-zinc-800 opacity-20";

    // Tooltip Content Component
    const TooltipContent = (
        <div className="w-64 border p-3 rounded-xl shadow-2xl text-left normal-case font-sans z-50" style={{ backgroundColor: 'var(--viz-bg-surface)', borderColor: 'var(--viz-border)' }}>
            <div className="space-y-2">
                <div>
                    <span className="text-[8px] font-black uppercase block mb-0.5" style={{ color: 'var(--viz-palette-3-orange)' }}>Explicação Técnica</span>
                    <p className="text-[10px] leading-tight" style={{ color: 'var(--viz-text-primary)' }}>{helpText?.tech || helpText}</p>
                </div>
                {helpText?.simple && (
                    <div className="pt-2 border-t" style={{ borderColor: 'var(--viz-border)' }}>
                        <span className="text-[8px] font-black uppercase block mb-0.5" style={{ color: 'var(--viz-palette-2-green)' }}>Para Leigos</span>
                        <p className="text-[10px] italic leading-tight border-l-2 pl-2" style={{ color: 'var(--viz-text-secondary)', borderColor: 'rgba(60, 180, 75, 0.2)' }}>{helpText.simple}</p>
                    </div>
                )}
                {Array.isArray(helpText?.extra) && helpText.extra.length > 0 && (
                    <div className="pt-2 border-t" style={{ borderColor: 'var(--viz-border)' }}>
                        <span className="text-[8px] font-black uppercase block mb-0.5" style={{ color: 'var(--viz-palette-4-yellow)' }}>Valores</span>
                        <div className="space-y-1">
                            {helpText.extra.map((item, index) => (
                                <div key={`${item.label}-${index}`} className="text-[10px] leading-tight flex gap-2">
                                    <span className="font-semibold min-w-[72px]" style={{ color: 'var(--viz-palette-4-yellow)' }}>{item.label}:</span>
                                    <span style={{ color: 'var(--viz-text-secondary)' }}>{item.value}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
                <div className="pt-2 border-t flex items-center justify-between" style={{ borderColor: 'var(--viz-border)' }}>
                    <span className="text-[8px] font-black uppercase" style={{ color: 'var(--viz-text-muted)' }}>Melhor direção</span>
                    <span className="text-[8px] font-mono" style={{ color: 'var(--viz-text-secondary)' }}>{directionHint === 'up' ? 'MAIOR (UP)' : directionHint === 'down' ? 'MENOR (DOWN)' : 'N/A'}</span>
                </div>
            </div>
            <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1 w-2 h-2 border-r border-b rotate-45" style={{ backgroundColor: 'var(--viz-bg-surface)', borderColor: 'var(--viz-border)' }}></div>
        </div>
    );

    // Flex layout determination based on alignment
    // Left Align: Icons on Right
    // Right Align: Icons on Left
    // Center Align: Icons on Left (default)
    const justifyClass = align === 'right' ? 'justify-end' : (align === 'center' ? 'justify-center' : 'justify-start');
    const flexDirection = align === 'right' ? 'flex-row' : 'flex-row-reverse';

    return (
        <div
            className={`flex items-center gap-2 select-none relative group h-full ${justifyClass} ${flexDirection} ${onSort ? 'cursor-pointer' : 'cursor-default'}`}
            onClick={() => onSort?.(sortKey)}
        >
            {/* Sort Indicators */}
            <div className="flex flex-col -space-y-1 flex-none">
                <ArrowUp size={14} className={upClass} />
                <ArrowDown size={14} className={downClass} />
            </div>

            <div className="flex items-center gap-1.5 truncate">
                {helpText ? (
                    <PortalTooltip content={TooltipContent}>
                        <span className={`text-[10px] font-black uppercase tracking-widest transition-colors border-b border-dotted hover:border-zinc-500 ${active ? 'text-orange-400' : 'text-zinc-500 group-hover:text-zinc-300'}`} style={{ borderColor: 'var(--viz-border)' }}>
                            {label}
                        </span>
                    </PortalTooltip>
                ) : (
                    <span className={`text-[10px] font-black uppercase tracking-widest transition-colors ${active ? 'text-orange-400' : 'text-zinc-500 group-hover:text-zinc-300'}`}>
                        {label}
                    </span>
                )}
            </div>
        </div>
    );
});

export const ExportMenu = React.memo(({ data, filename = "hpo_export" }) => {
    const [isOpen, setIsOpen] = React.useState(false);
    const handleExport = async (format) => {
        setIsOpen(false);
        await ExportService.export(format, data, filename);
    };
    return (
        <div className="relative">
            <button onClick={() => setIsOpen(!isOpen)} className="bg-zinc-900 border border-zinc-700 p-1.5 rounded-lg text-zinc-300 hover:bg-zinc-800"><Download size={14} /></button>
            {isOpen && (
                <div className="absolute right-0 top-full mt-2 w-32 bg-zinc-900 border border-zinc-800 rounded-xl shadow-2xl z-50 overflow-hidden">
                    {['json', 'csv', 'parquet', 'toon'].map(fmt => (
                        <button key={fmt} onClick={() => handleExport(fmt)} className="w-full text-left px-4 py-2 text-xs text-zinc-300 hover:bg-orange-500/10 hover:text-orange-300 uppercase font-bold">{fmt}</button>
                    ))}
                </div>
            )}
        </div>
    );
});

export const PaginationControls = React.memo(({ totalItems, currentPage, rowsPerPage, onPageChange, onRowsPerPageChange }) => {
    const totalPages = rowsPerPage === 'All' ? 1 : Math.ceil(totalItems / rowsPerPage);
    const options = [10, 20, 50, 100, 'All'];
    return (
        <div className="flex items-center justify-between px-4 py-3 border-t border-zinc-800 bg-zinc-900/30">
            <div className="text-[10px] text-zinc-500 font-mono">Total: {totalItems}</div>
            <div className="flex items-center gap-4">
                <div className="flex items-center gap-2 text-[10px] text-zinc-500 font-mono">
                    <span>Exibir:</span>
                    <select
                        value={rowsPerPage}
                        onChange={(e) => onRowsPerPageChange?.(e.target.value === 'All' ? 'All' : Number(e.target.value))}
                        className="bg-zinc-950 border border-zinc-800 rounded-md px-2 py-1 text-zinc-300"
                    >
                        {options.map((opt) => (
                            <option key={opt} value={opt}>{opt}</option>
                        ))}
                    </select>
                </div>
                <button onClick={() => onPageChange(currentPage - 1)} disabled={currentPage === 1} className="text-zinc-400 disabled:opacity-30"><ChevronRight className="rotate-180" size={14} /></button>
                <span className="text-[10px] text-zinc-400 font-mono">Página {currentPage} de {totalPages}</span>
                <button onClick={() => onPageChange(currentPage + 1)} disabled={currentPage === totalPages} className="text-zinc-400 disabled:opacity-30"><ChevronRight size={14} /></button>
            </div>
        </div>
    );
});
