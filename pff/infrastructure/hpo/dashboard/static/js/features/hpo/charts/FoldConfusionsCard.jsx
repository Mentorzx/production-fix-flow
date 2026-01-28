import { useMemo } from "react";

import { Card, TableIcon, WithData } from "../../../ui/BaseComponents.jsx";
import { ConfusionMatrix } from "../ConfusionMatrix.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

const normalizeCm = (m) => {
    if (!m || typeof m !== "object") return null;
    if (m.vp != null || m.vn != null || m.fp != null || m.fn != null) {
        return {
            vp: Number(m.vp ?? 0),
            vn: Number(m.vn ?? 0),
            fp: Number(m.fp ?? 0),
            fn: Number(m.fn ?? 0),
        };
    }
    if (m.tp != null || m.tn != null || m.fp != null || m.fn != null) {
        return {
            vp: Number(m.tp ?? 0),
            vn: Number(m.tn ?? 0),
            fp: Number(m.fp ?? 0),
            fn: Number(m.fn ?? 0),
        };
    }
    return null;
};

export const FoldConfusionsCard = ({ trials, liveStatus, charts }) => {
    const items = useMemo(() => {
        const fromCharts = Array.isArray(charts?.confusion_matrices) ? charts.confusion_matrices : null;
        if (fromCharts && fromCharts.length > 0) {
            return fromCharts
                .map((row, idx) => {
                    const cm = normalizeCm(row?.confusion_matrix || row);
                    if (!cm) return null;
                    const fold = row?.cv_fold_id;
                    const epoch = row?.epoch;
                    const foldLabel = fold != null ? `FOLD ${Number(fold) + 1}` : `FOLD ${idx + 1}`;
                    const suffix = epoch != null ? ` (epoca ${epoch})` : "";
                    return {
                        key: `c:${idx}`,
                        title: `${foldLabel}${suffix}`,
                        cm,
                    };
                })
                .filter(Boolean);
        }

        const t = Array.isArray(trials) ? trials : [];

        const fromTrials = t
            .map((trial) => {
                const cm = normalizeCm(trial?.metrics?.confusion_matrix || trial?.metrics);
                if (!cm) return null;
                return { key: `t:${trial.id}`, title: `TRIAL #${trial.id}`, cm };
            })
            .filter(Boolean);

        const last3 = fromTrials.slice(-3);
        if (last3.length > 0) return last3;

        const live = normalizeCm(liveStatus?.confusion_matrix);
        if (live) return [{ key: "live", title: "TRIAL ATUAL", cm: live }];
        return [];
    }, [charts?.confusion_matrices, trials, liveStatus?.confusion_matrix]);

    const helpText = ChartRegistry.get("fold_confusions");

    return (
        <Card title="Matriz de Confusão (3 folds)" className="h-full" icon={TableIcon} helpText={helpText}>
            <WithData when={items.length > 0} empty="Aguardando validação..." emptyClassName="text-zinc-500">
                <div className={`grid gap-4 auto-rows-fr ${items.length === 1 ? "grid-cols-1" : "grid-cols-1 md:grid-cols-2 2xl:grid-cols-3"} h-full min-h-0`}>
                    {items.map((it) => (
                        <div key={it.key} className="flex flex-col gap-2 min-h-0 min-w-0">
                            <div className="text-[9px] font-black uppercase tracking-widest text-zinc-500 leading-tight">
                                {it.title}
                            </div>
                            <div className="flex-1 min-h-[220px] min-w-0">
                                <ConfusionMatrix cm={it.cm} compact />
                            </div>
                        </div>
                    ))}
                </div>
            </WithData>
        </Card>
    );
};
