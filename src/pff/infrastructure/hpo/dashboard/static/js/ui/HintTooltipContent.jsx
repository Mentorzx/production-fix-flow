/**
 * Reusable tooltip body for metric/param hint overlays.
 *
 * Renders: header (name + value), tech explanation, simple explanation,
 * extra key-value pairs, current value, and a decorative arrow.
 *
 * @param {Object} props
 * @param {Object} props.hint - Registry hint object { tech, simple?, extra?, direction? }
 * @param {*} props.value - Display label
 * @param {*} [props.extraValue] - Current numeric/formatted value
 */
export const HintTooltipContent = ({ hint, value, extraValue }) => (
    <div className="w-72 p-4 bg-zinc-950 border border-zinc-800 rounded-xl shadow-2xl text-left font-sans z-50">
        <div className="space-y-3">
            <div className="pb-2 border-b border-zinc-900 flex justify-between items-center">
                <span className="text-[10px] font-black text-white uppercase tracking-wider">{value}</span>
                {extraValue != null && <span className="text-[10px] font-mono font-bold text-amber-400">{extraValue}</span>}
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
            {extraValue != null && (
                <div className="pt-2 border-t border-zinc-900">
                    <span className="text-[8px] font-black text-amber-500 uppercase block mb-1">Valor Atual</span>
                    <p className="text-[11px] text-zinc-100 font-mono bg-zinc-900/50 p-1 rounded-sm border border-zinc-800">{extraValue}</p>
                </div>
            )}
        </div>
        <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1.5 w-3 h-3 bg-zinc-950 border-r border-b border-zinc-800 rotate-45 shadow-lg"></div>
    </div>
);
