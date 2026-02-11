import { useEffect, useMemo, useState } from "react";
import { Card } from "../../../ui/BaseComponents.jsx";
import { PaginationControls } from "../../../ui/TableComponents.jsx";
import { PortalTooltip } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

/* ── Level → visual config ────────────────────────────────────── */
const LEVEL_STYLE = {
    ERROR: { bg: "rgba(220, 38, 38, 0.12)", bgHover: "rgba(220, 38, 38, 0.22)", border: "rgba(220, 38, 38, 0.30)", text: "#fca5a5", badge: "#ef4444", badgeBg: "rgba(220, 38, 38, 0.25)" },
    WARNING: { bg: "rgba(202, 138, 4, 0.10)", bgHover: "rgba(202, 138, 4, 0.20)", border: "rgba(202, 138, 4, 0.25)", text: "#fde68a", badge: "#eab308", badgeBg: "rgba(202, 138, 4, 0.20)" },
    CRITICAL: { bg: "rgba(220, 38, 38, 0.18)", bgHover: "rgba(220, 38, 38, 0.30)", border: "rgba(220, 38, 38, 0.40)", text: "#fecaca", badge: "#dc2626", badgeBg: "rgba(220, 38, 38, 0.35)" },
};
const DEFAULT_STYLE = LEVEL_STYLE.WARNING;

const GRID_COLS = "7.5rem 3.2rem 7rem 1fr";

/* ── Badge component ──────────────────────────────────────────── */
const LevelBadge = ({ level, style }) => (
    <span
        className="shrink-0 font-semibold text-[9px] tracking-wider rounded px-1.5 py-[1px] uppercase select-none"
        style={{ color: style.badge, backgroundColor: style.badgeBg }}
    >
        {level === "WARNING" ? "WARN" : level}
    </span>
);

/* ── Message with syntax highlighting ─────────────────────────── */
const HighlightedMessage = ({ text, color }) => {
    const parts = (text || "").split(/(\s+)/);
    return (
        <span className="break-all" style={{ color }}>
            {parts.map((part, i) => {
                const kvMatch = part.match(/^([a-zA-Z_]+)=(.+)$/);
                if (kvMatch) {
                    return (
                        <span key={i}>
                            <span className="text-zinc-500">{kvMatch[1]}=</span>
                            <span className="text-cyan-400 font-semibold">{kvMatch[2]}</span>
                        </span>
                    );
                }
                if (/^[\d.]+$/.test(part.trim()) && part.trim() !== "") {
                    return <span key={i} className="text-amber-400">{part}</span>;
                }
                return <span key={i}>{part}</span>;
            })}
        </span>
    );
};

/* ── Log header tooltips ──────────────────────────────────────── */
const HEADER_HINTS = {
    timestamp: {
        tech: "Timestamp UTC do evento registrado pelo pipeline de treinamento.",
        simple: "Quando o evento aconteceu.",
        extra: [{ label: "Formato", value: "HH:mm:ss.SSS" }]
    },
    level: {
        tech: "Severidade do log: INFO (operacional), WARN (degradação), ERROR (falha), CRITICAL (sistema comprometido).",
        simple: "Quão grave é o aviso.",
        extra: [{ label: "WARN", value: "Atenção necessária" }, { label: "ERROR", value: "Falha no fluxo" }, { label: "CRITICAL", value: "Sistema em risco" }]
    },
    module: {
        tech: "Componente do pipeline que emitiu o log (manager, evaluator, objective, etc.).",
        simple: "Qual parte do sistema gerou o aviso."
    },
    message: {
        tech: "Conteúdo textual do evento com parâmetros destacados automaticamente (key=value em ciano, números em âmbar).",
        simple: "O que aconteceu, em detalhes."
    }
};

const LogHeaderCell = ({ label, hint }) => {
    const TooltipContent = (
        <div className="w-56 border p-2.5 rounded-xl shadow-2xl text-left normal-case font-sans z-50" style={{ backgroundColor: 'var(--viz-bg-surface)', borderColor: 'var(--viz-border)' }}>
            <div className="space-y-1.5">
                <div>
                    <span className="text-[8px] font-black uppercase block mb-0.5" style={{ color: 'var(--viz-palette-3-orange)' }}>Explicação</span>
                    <p className="text-[10px] leading-tight" style={{ color: 'var(--viz-text-primary)' }}>{hint.tech}</p>
                </div>
                {hint.simple && (
                    <div className="pt-1.5 border-t" style={{ borderColor: 'var(--viz-border)' }}>
                        <span className="text-[8px] font-black uppercase block mb-0.5" style={{ color: 'var(--viz-palette-2-green)' }}>Para Leigos</span>
                        <p className="text-[10px] italic leading-tight" style={{ color: 'var(--viz-text-secondary)' }}>{hint.simple}</p>
                    </div>
                )}
                {Array.isArray(hint.extra) && hint.extra.length > 0 && (
                    <div className="pt-1.5 border-t" style={{ borderColor: 'var(--viz-border)' }}>
                        <div className="space-y-0.5">
                            {hint.extra.map((item, i) => (
                                <div key={i} className="text-[10px] leading-tight flex gap-2">
                                    <span className="font-semibold min-w-[48px]" style={{ color: 'var(--viz-palette-4-yellow)' }}>{item.label}:</span>
                                    <span style={{ color: 'var(--viz-text-secondary)' }}>{item.value}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
    return (
        <PortalTooltip content={TooltipContent}>
            <span className="cursor-help border-b border-dotted border-zinc-700 hover:border-zinc-500 transition-colors">{label}</span>
        </PortalTooltip>
    );
};

/* ── Single log row (with hover highlight) ────────────────────── */
const LogRow = ({ entry }) => {
    const lvl = (entry.level || "WARNING").toUpperCase();
    const st = LEVEL_STYLE[lvl] || DEFAULT_STYLE;

    return (
        <div
            className="grid items-start gap-x-3 py-[3px] px-3 cursor-default relative"
            style={{
                gridTemplateColumns: GRID_COLS,
                backgroundColor: st.bg,
                borderLeft: `2px solid ${st.border}`,
                transition: "background-color 120ms ease, box-shadow 120ms ease",
            }}
            onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = st.bgHover;
                e.currentTarget.style.boxShadow = `inset 0 0 12px ${st.border}, 0 0 6px ${st.border}`;
                e.currentTarget.style.zIndex = "5";
            }}
            onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = st.bg;
                e.currentTarget.style.boxShadow = "";
                e.currentTarget.style.zIndex = "";
            }}
        >
            <span className="text-zinc-500 shrink-0 select-none tabular-nums tracking-tight">
                {entry.timestamp ? entry.timestamp.split(" ").pop()?.slice(0, 12) : "—"}
            </span>
            <LevelBadge level={lvl} style={st} />
            <span className="text-zinc-500 truncate select-none" title={entry.module || ""}>
                {entry.module || "—"}
            </span>
            <HighlightedMessage text={entry.message} color={st.text} />
        </div>
    );
};

/* ── Main component ───────────────────────────────────────────── */
export const TerminalLogCard = ({ logs }) => {
    const [currentPage, setCurrentPage] = useState(1);
    const [rowsPerPage, setRowsPerPage] = useState(20);

    const entries = useMemo(() => {
        const items = Array.isArray(logs) ? logs : [];
        return items
            .map((entry, i) => {
                if (entry && typeof entry === "object" && entry.message) {
                    return { ...entry, key: `o:${i}:${entry.timestamp || ""}` };
                }
                if (typeof entry === "string") {
                    return { key: `s:${i}`, timestamp: "", level: "WARNING", module: "", message: entry };
                }
                return null;
            })
            .filter(Boolean);
    }, [logs]);

    const errorCount = entries.filter(e => e.level === "ERROR" || e.level === "CRITICAL").length;
    const warnCount = entries.filter(e => e.level === "WARNING").length;

    const paginatedEntries = useMemo(() => {
        if (rowsPerPage === "All") return entries;
        const start = (currentPage - 1) * rowsPerPage;
        return entries.slice(start, start + rowsPerPage);
    }, [entries, currentPage, rowsPerPage]);

    const totalPages = rowsPerPage === "All" ? 1 : Math.max(1, Math.ceil(entries.length / rowsPerPage));
    useEffect(() => {
        if (currentPage > totalPages && totalPages > 0) {
            setCurrentPage(totalPages);
        }
    }, [currentPage, totalPages]);

    return (
        <Card
            title="Logs de Execução"
            icon={() => <span className="text-lime-400 font-mono text-[10px]">$</span>}
            helpText={ChartRegistry.get("terminal_log")}
            headerRight={
                entries.length > 0 && (
                    <div className="flex gap-3 text-[10px] font-mono">
                        {errorCount > 0 && (
                            <span className="text-rose-400">{errorCount} erro{errorCount > 1 ? "s" : ""}</span>
                        )}
                        {warnCount > 0 && (
                            <span className="text-yellow-400">{warnCount} warn{warnCount > 1 ? "s" : ""}</span>
                        )}
                    </div>
                )
            }
        >
            <div className="w-full flex flex-col font-mono text-[10px] rounded-lg border shadow-inner overflow-hidden"
                style={{ backgroundColor: "var(--viz-bg-elevated)", borderColor: "var(--viz-border)" }}
            >
                {entries.length > 0 ? (
                    <>
                        {/* Column header */}
                        <div
                            className="grid items-center gap-x-3 py-1.5 px-3 text-[9px] font-semibold uppercase tracking-widest text-zinc-600 select-none border-b"
                            style={{
                                gridTemplateColumns: GRID_COLS,
                                backgroundColor: "var(--viz-bg-elevated)",
                                borderColor: "var(--viz-border)",
                            }}
                        >
                            <LogHeaderCell label="Horário" hint={HEADER_HINTS.timestamp} />
                            <LogHeaderCell label="Nível" hint={HEADER_HINTS.level} />
                            <LogHeaderCell label="Módulo" hint={HEADER_HINTS.module} />
                            <LogHeaderCell label="Mensagem" hint={HEADER_HINTS.message} />
                        </div>

                        {/* Log rows */}
                        <div className="flex flex-col py-1">
                            {paginatedEntries.map((entry) => (
                                <LogRow key={entry.key} entry={entry} />
                            ))}
                        </div>

                        {/* Pagination */}
                        <PaginationControls
                            totalItems={entries.length}
                            currentPage={currentPage}
                            rowsPerPage={rowsPerPage}
                            onPageChange={setCurrentPage}
                            onRowsPerPageChange={(value) => {
                                setRowsPerPage(value);
                                setCurrentPage(1);
                            }}
                        />
                    </>
                ) : (
                    <div className="h-full flex flex-col items-center justify-center text-zinc-700 italic gap-2 py-8">
                        <div className="text-2xl">✓</div>
                        <div>Nenhum warning ou erro registrado</div>
                    </div>
                )}
            </div>
        </Card>
    );
};
