import { useMemo } from 'react';
import { SortableTable } from "../../ui/SortableTable.jsx";
import { formatDuration, formatMetricValue, resolveMetricValue } from "../../domain/metrics/Formatters.js";
import { MetricRegistry } from "../../domain/metrics/MetricRegistry.js";
import { Sparkline, DataBar, HeatmapCell } from "../../ui/TableVisualization.jsx";

export const MetricsHistoryTable = ({
    data = [],
    compact = false,
    type = 'trial' // 'trial' or 'epoch'
}) => {

    // 1. Encontrar recordes de forma isolada (Global stats for DataBar/Heatmap normalization)
    const { bestId, worstId, minScore, maxScore, minLoss, maxLoss } = useMemo(() => {
        let bId = -1, wId = -1;
        let minS = Infinity, maxS = -Infinity;
        let minL = Infinity, maxL = -Infinity;

        if (data.length > 0) {
            data.forEach(t => {
                const s = resolveMetricValue(t, 'score');
                const l = resolveMetricValue(t, 'loss'); // Assuming 'loss' key exists or resolved
                if (typeof s === 'number') {
                    if (s < minS) minS = s;
                    if (s > maxS) maxS = s;
                }
                if (typeof l === 'number') {
                    if (l < minL) minL = l;
                    if (l > maxL) maxL = l;
                }
            });

            const isEligible = (t) => {
                if (!t) return false;
                const state = String(t.state || '').toUpperCase();
                if (state && state !== 'COMPLETE') return false;
                const s = resolveMetricValue(t, 'score');
                return typeof s === 'number' && Number.isFinite(s);
            };

            const eligible = data.filter(isEligible);
            const eligibleNoWarm = eligible.filter(t => !t.warmstart);
            const candidates = eligibleNoWarm.length > 0 ? eligibleNoWarm : eligible;
            if (candidates.length > 0) {
                const sorted = [...candidates].sort((a, b) => (resolveMetricValue(b, 'score') - resolveMetricValue(a, 'score')));
                bId = sorted[0]?.id;
                wId = sorted[sorted.length - 1]?.id;
            }
        }
        return {
            bestId: bId, worstId: wId,
            minScore: minS === Infinity ? 0 : minS, maxScore: maxS === -Infinity ? 1 : maxS,
            minLoss: minL === Infinity ? 0 : minL, maxLoss: maxL === -Infinity ? 1 : maxL
        };
    }, [data]);

    // 2. Memoizar Definição de Colunas SOTA
    const columns = useMemo(() => {
        const cols = [];
        const getHint = (key) => MetricRegistry.get(key);

        // STICKY ID COLUMN
        const idKey = type === 'epoch' ? 'epoch' : 'id';
        cols.push({
            key: idKey, label: type === 'epoch' ? 'Época' : 'Trial', sortable: true, align: 'left', width: '120px', group: 'overview',
            helpText: {
                tech: "Identificador sequencial do registro para rastrear a ordem de execução.",
                simple: "O número de chamada para não se perder.",
                extra: [{ label: "Uso", value: "ordem cronológica" }]
            },
            sortValue: (row) => (type === 'epoch' ? row.epoch : row.id),
            render: (id, row) => {
                const displayId = type === 'epoch' ? (row.epoch ?? id) : id;
                const isWarm = !!(row.warmstart);
                const isBest = displayId === bestId;
                const isWorst = displayId === worstId && data.length > 2;
                const isPruned = row.state === 'PRUNED';
                return (
                    <div className="flex items-center gap-2 pl-2 border-l-2" style={{ borderColor: isBest ? 'var(--viz-palette-4-yellow)' : (isPruned ? 'var(--viz-palette-5-red)' : 'transparent') }}>
                        <span className="font-mono font-bold" style={{ color: 'var(--viz-text-primary)' }}>{displayId}</span>
                        {isPruned && <span className="px-1 py-0.5 rounded-sm bg-red-500/10 text-red-500 text-[8px] font-bold border border-red-500/20">PRUNED</span>}
                        {isWarm && <span className="px-1 py-0.5 rounded-sm bg-amber-500/10 text-amber-500 text-[8px] font-bold border border-amber-500/20">WARM</span>}
                        {isBest && <span className="px-1 py-0.5 rounded-sm bg-lime-500/10 text-lime-400 text-[8px] font-bold border border-lime-500/20">★ MELHOR</span>}
                        {isWorst && <span className="px-1 py-0.5 rounded-sm bg-rose-500/10 text-rose-500 text-[8px] font-bold border border-rose-500/20">PIOR</span>}
                    </div>
                );
            }
        });

        const durationColumn = {
            key: 'duration', label: 'Duração', sortable: true, align: 'right', direction: 'down', group: 'efficiency',
            helpText: getHint('duration'),
            sortValue: (row) => resolveMetricValue(row, 'duration'),
            render: (v, row) => <span className="font-mono text-zinc-400">{formatDuration(resolveMetricValue(row, 'duration'), compact)}</span>
        };

        const scoreColumn = {
            key: 'score', label: 'SCORE', sortable: true, align: 'right', direction: 'up', width: '140px', group: 'overview',
            helpText: getHint('score'),
            sortValue: (row) => resolveMetricValue(row, 'score'),
            render: (v, row) => {
                const val = resolveMetricValue(row, 'score');
                return (
                    <DataBar
                        value={val}
                        min={minScore}
                        max={maxScore}
                        color="var(--viz-palette-1-blue)"
                        format={formatMetricValue}
                    />
                );
            }
        };

        const lossColumn = {
            key: 'loss', label: 'LOSS', sortable: true, align: 'right', direction: 'down', width: '140px', group: 'overview',
            helpText: getHint('loss'),
            sortValue: (row) => resolveMetricValue(row, 'loss'),
            render: (v, row) => {
                const val = resolveMetricValue(row, 'loss');
                const history = row.history?.loss || [];
                return (
                    <div className="flex items-center justify-end gap-2">
                        {history.length > 2 && (
                            <Sparkline
                                data={history.slice(-20)}
                                width={40}
                                height={16}
                                color="var(--viz-palette-3-orange)"
                                min={minLoss}
                                max={maxLoss}
                            />
                        )}
                        <span className="font-mono" style={{ color: 'var(--viz-text-secondary)' }}>{formatMetricValue(val)}</span>
                    </div>
                );
            }
        };

        if (type === 'epoch') {
            cols.push({ ...durationColumn, group: 'overview' });
            cols.push(lossColumn);
        } else {
            cols.push(scoreColumn);
            cols.push(lossColumn);
        }

        ['mcc', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'pr_auc'].forEach(key => {
            cols.push({
                key, label: key.toUpperCase(), sortable: true, align: 'right', direction: 'up', width: '90px', group: 'clf',
                helpText: getHint(key),
                sortValue: (row) => resolveMetricValue(row, key),
                render: (v, row) => (
                    <HeatmapCell
                        value={resolveMetricValue(row, key)}
                        min={0} max={1}
                        colorScale="green"
                    />
                )
            });
        });

        ['mrr', 'hits1', 'hits3', 'hits10'].forEach(key => {
            cols.push({
                key, label: key.toUpperCase(), sortable: true, align: 'right', direction: 'up', group: 'ranking',
                helpText: getHint(key),
                sortValue: (row) => resolveMetricValue(row, key),
                render: (v, row) => (
                    key === 'mrr'
                        ? <HeatmapCell value={resolveMetricValue(row, key)} min={0} max={1} colorScale="blue" />
                        : <span className="font-mono text-zinc-400">{formatMetricValue(resolveMetricValue(row, key))}</span>
                )
            });
        });

        if (type === 'trial') {
            cols.push(durationColumn);
        }

        cols.push({
            key: 'efficiency', label: 'Eficiência', sortable: true, align: 'right', direction: 'up', group: 'efficiency',
            helpText: getHint('efficiency'),
            sortValue: (row) => {
                const score = resolveMetricValue(row, 'score');
                const dur = resolveMetricValue(row, 'duration');
                return (score != null && dur != null && dur > 0) ? (score / dur) : (row.efficiency || 0);
            },
            render: (v, row) => {
                const score = resolveMetricValue(row, 'score');
                const dur = resolveMetricValue(row, 'duration');
                const eff = (score != null && dur != null && dur > 0)
                    ? (score / dur)
                    : (resolveMetricValue(row, 'efficiency') ?? v);
                return eff ? <span className="font-mono" style={{ color: 'var(--viz-palette-4-yellow)' }}>{(eff * 100).toFixed(2)}%</span> : '—';
            }
        });

        return cols;
    }, [type, compact, bestId, worstId, data.length, minScore, maxScore, minLoss, maxLoss]);

    return (
        <div className="w-full h-full flex flex-col min-h-0">
            <SortableTable
                data={data}
                columns={columns}
                defaultSort={{ key: type === 'epoch' ? 'epoch' : 'id', direction: 'desc' }}
                className="text-[10px] bg-transparent! border-0! shadow-none!"
            />
        </div>
    );
};
