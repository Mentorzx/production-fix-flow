import { BaseTooltip } from "../../ui/BaseComponents.jsx";

const formatPercent = (value, total) => {
    if (!Number.isFinite(total) || total <= 0) return "0.0%";
    return `${((value / total) * 100).toFixed(1)}%`;
};

const buildTooltip = (technical, lay, extra = []) => (
    <div className="space-y-2 text-[10px]">
        <div>
            <span className="font-semibold text-orange-400 uppercase text-[8px]">Explicação Técnica</span>
            <div className="text-zinc-300 leading-tight">{technical}</div>
        </div>
        <div>
            <span className="font-semibold text-lime-300 uppercase text-[8px]">Para Leigos</span>
            <div className="text-lime-400/80 italic leading-tight border-l-2 border-lime-500/20 pl-2">{lay}</div>
        </div>
        {Array.isArray(extra) && extra.length > 0 && (
            <div className="pt-2 border-t border-zinc-800">
                <span className="font-semibold text-amber-300 uppercase text-[8px]">Valores</span>
                <div className="space-y-1">
                    {extra.map((item, index) => (
                        <div key={`${item.label}-${index}`} className="flex gap-2 text-zinc-300">
                            <span className="text-amber-300/90 font-semibold min-w-[72px]">{item.label}:</span>
                            <span>{item.value}</span>
                        </div>
                    ))}
                </div>
            </div>
        )}
    </div>
);

export const ConfusionMatrix = ({ cm, compact = false }) => {
    if (!cm) return <div className="h-full flex items-center justify-center text-zinc-600 italic">Aguardando fold...</div>;
    const vp = Number(cm.vp ?? 0);
    const vn = Number(cm.vn ?? 0);
    const fp = Number(cm.fp ?? 0);
    const fn = Number(cm.fn ?? 0);
    const total = vp + vn + fp + fn;

    const cells = [
        {
            key: "vp",
            label: "VP",
            title: "Verdadeiro Positivo",
            value: vp,
            className: "bg-lime-500/10 border-lime-500/20 text-lime-400",
            technical: `Verdadeiro Positivo (VP) = ${vp}`,
            lay: "Acertou quando disse SIM.",
        },
        {
            key: "fp",
            label: "FP",
            title: "Falso Positivo",
            value: fp,
            className: "bg-rose-500/10 border-rose-500/20 text-rose-400",
            technical: `Falso Positivo (FP) = ${fp}`,
            lay: "Alarme falso: disse SIM sem precisar.",
        },
        {
            key: "fn",
            label: "FN",
            title: "Falso Negativo",
            value: fn,
            className: "bg-amber-500/10 border-amber-500/20 text-amber-400",
            technical: `Falso Negativo (FN) = ${fn}`,
            lay: "Passou batido: disse NÃO quando era SIM.",
        },
        {
            key: "vn",
            label: "VN",
            title: "Verdadeiro Negativo",
            value: vn,
            className: "bg-orange-500/10 border-orange-500/20 text-orange-400",
            technical: `Verdadeiro Negativo (VN) = ${vn}`,
            lay: "Acertou ao dizer NÃO.",
        },
    ];

    const cellPadding = compact ? "p-2" : "p-3";
    const labelSize = compact ? "text-[9px]" : "text-[11px]";
    const valueSize = compact ? "text-[clamp(18px,2.2vw,30px)]" : "text-[clamp(20px,5.2vw,42px)]";
    const gridGap = compact ? "gap-2" : "gap-3";

    return (
        <div className="h-full w-full flex flex-col min-h-0">
            <div className={`grid grid-cols-2 grid-rows-2 ${gridGap} h-full w-full flex-1 min-h-0`}>
                {cells.map((cell) => (
                    <BaseTooltip
                        key={cell.key}
                        title={`${cell.title} (${cell.label})`}
                        tech={buildTooltip(
                            cell.technical,
                            cell.lay,
                            total > 0
                                ? [
                                    { label: "Contagem", value: cell.value },
                                    { label: "Percentual", value: formatPercent(cell.value, total) },
                                ]
                                : []
                        )}
                        className="h-full w-full min-h-0 min-w-0 flex"
                        trigger={(
                            <div className={`border ${cellPadding} rounded-xl flex flex-col items-center justify-center h-full w-full min-h-0 min-w-0 overflow-hidden ${cell.className}`}>
                                <span className={`${labelSize} font-bold uppercase tracking-widest opacity-60 mb-1`}>{cell.label}</span>
                                <span className={`font-black ${valueSize} leading-none tabular-nums tracking-tight`}>
                                    {formatPercent(cell.value, total)}
                                </span>
                            </div>
                        )}
                    />
                ))}
            </div>
        </div>
    );
};
