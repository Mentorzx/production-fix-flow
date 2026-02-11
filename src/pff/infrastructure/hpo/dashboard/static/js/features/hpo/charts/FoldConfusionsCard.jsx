import { useEffect, useMemo, useRef, useState } from "react";

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
      const normalized = fromCharts
        .map((row, idx) => {
          const cm = normalizeCm(row?.confusion_matrix || row);
          if (!cm) return null;
          const fold = row?.cv_fold_id;
          const trialNumber = row?.trial_number;
          const epoch = row?.epoch;

          const trialLabel = trialNumber != null ? `TRIAL ${Number(trialNumber) + 1}` : "TRIAL ?";
          const foldLabel = fold != null ? `FOLD ${Number(fold)}` : `FOLD ${idx}`;
          const suffix = epoch != null ? ` (epoca ${epoch})` : "";

          const comboKey = `${trialNumber ?? "x"}:${fold ?? "x"}`;
          return {
            comboKey,
            key: `c:${comboKey}`,
            pillLabel: `${trialLabel} · ${foldLabel}`,
            title: `${trialLabel} · ${foldLabel}${suffix}`,
            cm,
          };
        })
        .filter(Boolean);

      const unique = [];
      const seen = new Set();
      for (let i = normalized.length - 1; i >= 0; i -= 1) {
        const it = normalized[i];
        if (!it) continue;
        if (seen.has(it.comboKey)) continue;
        seen.add(it.comboKey);
        unique.push(it);
        if (unique.length >= 3) break;
      }
      return unique.reverse();
    }

    const t = Array.isArray(trials) ? trials : [];

    const fromTrials = t
      .map((trial) => {
        const cm = normalizeCm(trial?.metrics?.confusion_matrix || trial?.metrics);
        if (!cm) return null;
        return {
          comboKey: `t:${trial.id}`,
          key: `t:${trial.id}`,
          pillLabel: `TRIAL ${trial.id}`,
          title: `TRIAL ${trial.id}`,
          cm,
        };
      })
      .filter(Boolean);

    const last3 = fromTrials.slice(-3);
    if (last3.length > 0) return last3;

    const live = normalizeCm(liveStatus?.confusion_matrix);
    if (live) return [{ key: "live", title: "TRIAL ATUAL", cm: live }];
    return [];
  }, [charts?.confusion_matrices, trials, liveStatus?.confusion_matrix]);

  const helpText = ChartRegistry.get("fold_confusions");

  const [activeKey, setActiveKey] = useState(null);
  const lastKeyRef = useRef(null);

  useEffect(() => {
    if (items.length === 0) {
      setActiveKey(null);
      lastKeyRef.current = null;
      return;
    }

    const lastKey = items[items.length - 1]?.key ?? null;
    if (!lastKey) return;

    const hasActive = activeKey && items.some((it) => it.key === activeKey);
    if (!hasActive) {
      setActiveKey(lastKey);
    } else if (activeKey === lastKeyRef.current) {
      // Follow the newest fold as it updates, unless the user selected an older one.
      setActiveKey(lastKey);
    }
    lastKeyRef.current = lastKey;
  }, [items, activeKey]);

  const active = useMemo(() => {
    if (!activeKey) return items.length > 0 ? items[items.length - 1] : null;
    return (
      items.find((it) => it.key === activeKey) ||
      (items.length > 0 ? items[items.length - 1] : null)
    );
  }, [items, activeKey]);

  return (
    <Card
      title="Matriz de Confusão (3 folds)"
      className="h-full"
      icon={TableIcon}
      helpText={helpText}
    >
      <WithData
        when={items.length > 0}
        empty="Aguardando validação..."
        emptyClassName="text-zinc-500"
      >
        <div className="flex flex-col gap-3 h-full min-h-0 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            {items.map((it) => {
              const selected = it.key === active?.key;
              return (
                <button
                  key={it.key}
                  type="button"
                  onClick={() => setActiveKey(it.key)}
                  className="px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest transition-colors border"
                  style={{
                    backgroundColor: selected ? "rgba(229, 197, 88, 0.12)" : "transparent",
                    borderColor: selected ? "var(--viz-palette-4-yellow)" : "var(--viz-border)",
                    color: selected ? "var(--viz-palette-4-yellow)" : "var(--viz-text-muted)",
                  }}
                  aria-pressed={selected}
                >
                  {it.pillLabel || it.title}
                </button>
              );
            })}
          </div>

          {active && (
            <div className="flex-1 min-h-0 min-w-0 flex flex-col gap-2">
              <div className="text-[9px] font-black uppercase tracking-widest text-zinc-500 leading-tight">
                {active.title}
              </div>
              <div className="flex-1 min-h-[220px] min-w-0">
                <ConfusionMatrix cm={active.cm} />
              </div>
            </div>
          )}
        </div>
      </WithData>
    </Card>
  );
};
