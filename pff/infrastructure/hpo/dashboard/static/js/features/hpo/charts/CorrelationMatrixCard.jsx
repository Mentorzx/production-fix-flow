import { useMemo } from 'react';

import { Card, GitMerge, EmptyState } from "../../../ui/BaseComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const CorrelationMatrixCard = ({ trials }) => {
    const params = useMemo(() => {
        if (!trials || trials.length === 0) return [];
        const allKeys = new Set(); trials.forEach(t => Object.keys(t.params || {}).forEach(k => allKeys.add(k)));
        const numericalKeys = Array.from(allKeys).filter(k => { const vals = trials.map(t => t.params[k]).filter(v => typeof v === 'number'); return vals.length > trials.length * 0.8; });
        return [...numericalKeys.slice(0, 5), 'value'];
    }, [trials]);

    const correlations = useMemo(() => {
        if (!trials || trials.length < 2 || params.length < 2) return null;
        const matrix = {}; params.forEach(p1 => { matrix[p1] = {}; params.forEach(p2 => { if (p1 === p2) { matrix[p1][p2] = 1; } else { const v1 = trials.map(t => p1 === 'value' ? t.value : t.params[p1]); const v2 = trials.map(t => p2 === 'value' ? t.value : t.params[p2]); let sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0, pSum = 0, count = 0; for (let i = 0; i < v1.length; i++) { if (typeof v1[i] === 'number' && typeof v2[i] === 'number') { sum1 += v1[i]; sum2 += v2[i]; sum1Sq += v1[i] * v1[i]; sum2Sq += v2[i] * v2[i]; pSum += v1[i] * v2[i]; count++; } } if (count < 2) matrix[p1][p2] = 0; else { const num = pSum - (sum1 * sum2 / count); const den = Math.sqrt((sum1Sq - sum1 * sum1 / count) * (sum2Sq - sum2 * sum2 / count)); matrix[p1][p2] = den === 0 ? 0 : num / den; } } }); }); return matrix;
    }, [trials, params]);

    const chartContract = ChartRegistry.get('correlation') || { title: "Correlação", tech: "Pearson." };

    if (!correlations) {
        return (
            <Card title={chartContract.title} icon={GitMerge} className="h-full" helpText={chartContract}>
                <EmptyState className="text-sm">Sem dados suficientes</EmptyState>
            </Card>
        );
    }

    return (
        <Card title={chartContract.title} helpText={chartContract} className="h-full" icon={GitMerge}>
            <div className="h-full p-4 flex flex-col items-center justify-center overflow-auto custom-scrollbar">
                <div className="w-full min-w-[300px]">
                    <div className="grid gap-1 mb-2" style={{ gridTemplateColumns: `repeat(${params.length + 1}, minmax(0, 1fr))` }}>
                        <div></div>
                        {params.map(p => <div key={p} className="text-[8px] font-mono text-zinc-500 uppercase text-center truncate w-full max-w-[40px] mx-auto" title={p}>{p.slice(0, 4)}</div>)}
                    </div>
                    {params.map((row) => (
                        <div key={row} className="grid gap-1 mb-1" style={{ gridTemplateColumns: `repeat(${params.length + 1}, minmax(0, 1fr))` }}>
                            <div className="text-[8px] font-mono text-zinc-500 uppercase text-right pr-2 self-center truncate" title={row}>{row.slice(0, 4)}</div>
                            {params.map((col) => {
                                const val = correlations[row][col];
                                const color = val > 0 ? 'bg-lime-500' : 'bg-rose-500';
                                return <div key={`${row}-${col}`} className={`rounded-sm ${row === col ? 'bg-zinc-800' : color} flex items-center justify-center aspect-square w-full max-w-[40px] mx-auto`} style={{ opacity: row === col ? 1 : Math.abs(val) }}><span className="text-[9px] font-bold text-white">{row === col ? '-' : val.toFixed(1)}</span></div>;
                            })}
                        </div>
                    ))}

                </div>
            </div>
        </Card>
    );
};
