import { useMemo } from "react";
import { Card } from "../../../ui/BaseComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

const LogLine = ({ message, level }) => {
    // Basic detection
    const msgUpper = message.toUpperCase();
    const isErr = level === "ERROR" || msgUpper.includes("ERROR") || msgUpper.includes("FAILED");
    const isSuccess = level === "SUCCESS" || msgUpper.includes("SAVED") || msgUpper.includes("COMPLETED");
    const isWarn = level === "WARNING" || msgUpper.includes("WARNING");

    // Fallback color
    let baseClass = "text-zinc-300";
    if (isErr) baseClass = "text-rose-400";
    if (isSuccess) baseClass = "text-lime-400";
    if (isWarn) baseClass = "text-yellow-400";

    // Split by common separators to find key-value pairs
    // Regex matches: word=number, or common phrases
    const parts = message.split(/(\s+)/);

    return (
        <span className={`${baseClass} break-all`}>
            {parts.map((part, i) => {
                // Key=Value Pattern (e.g., loss=0.234, mrr=0.5)
                const kvMatch = part.match(/^([a-zA-Z0-9_@]+)=([0-9.]+)(%?)$/);
                if (kvMatch) {
                    return (
                        <span key={i}>
                            <span className="text-zinc-500">{kvMatch[1]}=</span>
                            <span className="text-cyan-400 font-bold">{kvMatch[2]}{kvMatch[3]}</span>
                        </span>
                    );
                }

                // Numbers alone (mostly IDs or metrics)
                if (!isNaN(parseFloat(part)) && isFinite(part) && part.trim() !== "") {
                    return <span key={i} className="text-amber-400">{part}</span>;
                }

                // Keywords Highlighting
                const lower = part.toLowerCase();
                if (lower.includes('checkpoint')) return <span key={i} className="text-purple-400 font-bold">{part}</span>;
                if (lower.includes('metricas')) return <span key={i} className="text-blue-400 font-bold">{part}</span>;
                if (lower.includes('epoch')) return <span key={i} className="text-yellow-500 font-bold">{part}</span>;
                if (lower.includes('ms/triple') || lower.includes('t/m')) return <span key={i} className="text-emerald-400">{part}</span>;

                return <span key={i}>{part}</span>;
            })}
        </span>
    );
};

export const TerminalLogCard = ({ logs }) => {
    const normalized = useMemo(() => {
        const items = Array.isArray(logs) ? logs : [];
        return items
            .map((entry, i) => {
                if (typeof entry === "string") {
                    return { key: `s:${i}:${entry.slice(0, 80)}`, ts: null, level: null, message: entry };
                }
                if (entry && typeof entry === "object") {
                    const ts = entry.timestamp ? new Date(entry.timestamp * 1000).toLocaleTimeString([], { hour12: false }) : null;
                    const level = entry.level ?? null;
                    const message = entry.message ?? String(entry);
                    return {
                        key: `o:${i}:${String(message).slice(0, 40)}`,
                        ts,
                        level,
                        message: String(message),
                    };
                }
                return { key: `u:${i}`, ts: null, level: null, message: String(entry) };
            })
            // Keep strictly the last N for performance
            .slice(-100);
    }, [logs]);

    return (
        <Card
            title="Logs de Execução"
            className="h-full"
            icon={() => <span className="text-lime-400 font-mono text-[10px]">$</span>}
            helpText={ChartRegistry.get("terminal_log")}
        >
            <div
                className="h-full w-full p-3 font-mono text-[10px] overflow-auto custom-scrollbar rounded-lg border shadow-inner"
                style={{ backgroundColor: 'var(--viz-bg-elevated)', borderColor: 'var(--viz-border)' }}
            >
                {normalized.length > 0 ? (
                    <div className="flex flex-col">
                        {normalized.map((log) => (
                            <div key={log.key} className="mb-0.5 leading-snug flex gap-2 hover:bg-zinc-900/50 -mx-2 px-2 py-0.5 rounded transition-colors">
                                <span className="text-zinc-600 shrink-0 select-none">
                                    {log.ts || new Date().toLocaleTimeString([], { hour12: false })}
                                </span>
                                <LogLine message={log.message} level={log.level} />
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="h-full flex flex-col items-center justify-center text-zinc-700 italic gap-2">
                        <div className="animate-pulse">Aguardando output...</div>
                    </div>
                )}
            </div>
        </Card>
    );
};
