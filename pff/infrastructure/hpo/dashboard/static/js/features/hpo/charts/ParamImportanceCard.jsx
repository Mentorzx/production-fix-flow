import { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, Cell, Label } from 'recharts';

import { Card, Sliders, colors, DefaultCartesianGrid, DefaultTooltip, ChartContainer, WithData } from "../../../ui/BaseComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { ParamRegistry } from "../../../domain/metrics/ParamRegistry.js";
import { PortalTooltip, ChartAxisLabel } from "../../../ui/UIComponents.jsx";

export const ParamImportanceCard = ({ importances }) => {
    const data = useMemo(() => { if (!importances) return []; return Object.entries(importances).map(([name, value]) => ({ name, value })).sort((a, b) => b.value - a.value); }, [importances]);

    const helpText = ChartRegistry.get('fanova', {
        title: "Importância de Parâmetros",
        tech: "Importância estimada (fANOVA/heurística) dos hiperparâmetros no valor objetivo.",
        simple: "Quais botões mais mexem na nota."
    });

    return (
        <Card title="Importância de Parâmetros" icon={Sliders} className="h-full" helpText={helpText}>
            <div className="w-full h-full min-h-[300px]">
                <WithData when={data.length > 0} empty="Sem dados fANOVA" emptyClassName="text-zinc-500">
                    <ChartContainer>
                        <BarChart layout="vertical" data={data} margin={{ left: 5, right: 20, top: 10, bottom: 5 }}>
                            <DefaultCartesianGrid horizontal={false} />
                            <XAxis type="number" stroke={colors.text} tick={{ fontSize: 9 }} height={50}>
                                <Label content={<ChartAxisLabel value="Importância" axis="x" />} />
                            </XAxis>
                            <YAxis
                                dataKey="name"
                                type="category"
                                stroke={colors.text}
                                width={100}
                                tick={{ fontSize: 9 }}
                                interval={0}
                                tickFormatter={(v) => String(v).length > 14 ? String(v).slice(0, 14) + '…' : v}
                            >
                                <Label content={<ChartAxisLabel value="Parâmetro" axis="y" />} position="insideLeft" />
                            </YAxis>
                            <DefaultTooltip />
                            <Bar dataKey="value" fill={colors.primary} radius={[0, 4, 4, 0]} barSize={12}>
                                {data.map((e, i) => (
                                    <Cell key={e.name} fill={i === 0 ? colors.success : colors.primary} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ChartContainer>
                </WithData>

                {data.length > 0 && (
                    <div className="mt-2 text-[9px] text-zinc-600 font-mono">
                        {data.slice(0, 5).map((p) => {
                            const hints = ParamRegistry.getAll();
                            const normKey = String(p.name).toLowerCase().replace(/ /g, '_');
                            const hint = hints[normKey] ?? hints[p.name] ?? null;
                            if (!hint) return null;
                            const TooltipContent = (
                                <div className="w-72 bg-zinc-950 border border-zinc-700 p-3 rounded-xl shadow-2xl text-left normal-case font-sans">
                                    <div className="space-y-2">
                                        <div>
                                            <span className="text-[8px] font-black text-orange-500 uppercase block mb-0.5">Explicação Técnica</span>
                                            <p className="text-[10px] text-zinc-200 leading-tight">{hint.tech}</p>
                                        </div>
                                        {hint.simple && (
                                            <div className="pt-2 border-t border-zinc-800">
                                                <span className="text-[8px] font-black text-lime-500 uppercase block mb-0.5">Para Leigos</span>
                                                <p className="text-[10px] text-lime-400/90 italic leading-tight border-l-2 border-lime-500/20 pl-2">{hint.simple}</p>
                                            </div>
                                        )}
                                        {Array.isArray(hint.extra) && hint.extra.length > 0 && (
                                            <div className="pt-2 border-t border-zinc-800">
                                                <span className="text-[8px] font-black text-amber-400 uppercase block mb-0.5">Valores</span>
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
                                    </div>
                                    <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1 w-2 h-2 bg-zinc-950 border-r border-b border-zinc-700 rotate-45"></div>
                                </div>
                            );
                            return (
                                <PortalTooltip key={p.name} content={TooltipContent}>
                                    <span className="mr-3 cursor-help border-b border-dotted border-zinc-700/50 hover:border-zinc-500">{p.name}</span>
                                </PortalTooltip>
                            );
                        })}
                    </div>
                )}
            </div>
        </Card>
    );
};
