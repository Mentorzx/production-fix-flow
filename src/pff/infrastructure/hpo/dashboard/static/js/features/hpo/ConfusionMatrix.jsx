/**
 * Provide ConfusionMatrix module functionality for the HPO dashboard.
 */

import { BaseTooltip } from "../../ui/BaseComponents.jsx";
import { Theme } from "../../ui/Theme.js";

const formatPercent = (value, total) => {
  if (!Number.isFinite(total) || total <= 0) return "0.0%";
  return `${((value / total) * 100).toFixed(1)}%`;
};

const formatCount = (value) => Number(value || 0).toLocaleString("pt-BR");
const clamp01 = (value) => Math.max(0, Math.min(1, Number(value) || 0));

const ratio = (numerator, denominator) => {
  if (!Number.isFinite(denominator) || denominator <= 0) return 0;
  return clamp01((Number(numerator) || 0) / denominator);
};

const buildTooltip = (technical, lay, extra = []) => (
  <div className="space-y-2 text-[10px]">
    <div>
      <span className="font-semibold text-orange-400 uppercase text-[8px]">Explicação Técnica</span>
      <div className="text-zinc-300 leading-tight">{technical}</div>
    </div>
    <div>
      <span className="font-semibold text-lime-300 uppercase text-[8px]">Para Leigos</span>
      <div className="text-lime-400/80 italic leading-tight border-l-2 border-lime-500/20 pl-2">
        {lay}
      </div>
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

/**
 * Expose confusion matrix for dashboard usage.
 */
export const ConfusionMatrix = ({ cm, compact = false }) => {
  if (!cm)
    return (
      <div
        className="h-full flex items-center justify-center italic border rounded-xl"
        style={{
          color: Theme.ui.text.muted,
          borderColor: Theme.ui.border,
          background:
            "linear-gradient(160deg, color-mix(in srgb, var(--viz-bg-surface), transparent 5%) 0%, color-mix(in srgb, var(--viz-bg-canvas), transparent 20%) 100%)",
        }}
      >
        Aguardando fold...
      </div>
    );
  const vp = Number(cm.vp ?? 0);
  const vn = Number(cm.vn ?? 0);
  const fp = Number(cm.fp ?? 0);
  const fn = Number(cm.fn ?? 0);
  const total = vp + vn + fp + fn;
  const precision = ratio(vp, vp + fp);
  const recall = ratio(vp, vp + fn);
  const summary = [
    { label: "Total", value: formatCount(total) },
    { label: "Acurácia", value: formatPercent(vp + vn, total) },
    { label: "Precisão", value: `${(precision * 100).toFixed(1)}%` },
    { label: "Recall", value: `${(recall * 100).toFixed(1)}%` },
  ];

  const cells = [
    {
      key: "vp",
      label: "VP",
      title: "Verdadeiro Positivo",
      value: vp,
      toneVar: Theme.semantic.success,
      technical: `Verdadeiro Positivo (VP) = ${vp}`,
      lay: "Acertou quando disse SIM.",
    },
    {
      key: "fp",
      label: "FP",
      title: "Falso Positivo",
      value: fp,
      toneVar: Theme.semantic.error,
      technical: `Falso Positivo (FP) = ${fp}`,
      lay: "Alarme falso: disse SIM sem precisar.",
    },
    {
      key: "fn",
      label: "FN",
      title: "Falso Negativo",
      value: fn,
      toneVar: Theme.semantic.warning,
      technical: `Falso Negativo (FN) = ${fn}`,
      lay: "Passou batido: disse NÃO quando era SIM.",
    },
    {
      key: "vn",
      label: "VN",
      title: "Verdadeiro Negativo",
      value: vn,
      toneVar: Theme.semantic.info,
      technical: `Verdadeiro Negativo (VN) = ${vn}`,
      lay: "Acertou ao dizer NÃO.",
    },
  ];

  const cellPadding = compact ? "p-2.5" : "p-3.5";
  const labelSize = compact ? "text-[9px]" : "text-[11px]";
  const valueSize = compact ? "text-[clamp(18px,2.2vw,28px)]" : "text-[clamp(22px,5.2vw,52px)]";
  const countSize = compact ? "text-[11px]" : "text-[12px]";
  const gridGap = compact ? "gap-2" : "gap-3";

  return (
    <div className="h-full w-full flex flex-col min-h-0 items-center">
      {!compact && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3 w-full">
          {summary.map((item) => (
            <div
              key={item.label}
              className="rounded-lg px-2.5 py-2 border text-center"
              style={{
                borderColor: Theme.ui.border,
                background:
                  "linear-gradient(180deg, color-mix(in srgb, var(--viz-bg-surface), transparent 2%) 0%, color-mix(in srgb, var(--viz-bg-canvas), transparent 18%) 100%)",
              }}
            >
              <div className="text-[9px] font-black uppercase tracking-widest text-zinc-500">
                {item.label}
              </div>
              <div className="text-[12px] font-bold tabular-nums" style={{ color: Theme.ui.text.primary }}>
                {item.value}
              </div>
            </div>
          ))}
        </div>
      )}
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
            trigger={
              <div
                className={`relative border ${cellPadding} rounded-xl flex flex-col items-center justify-center h-full w-full min-h-0 min-w-0 overflow-hidden transition-all duration-300`}
                style={{
                  color: cell.toneVar,
                  borderColor: `color-mix(in srgb, ${cell.toneVar}, transparent 68%)`,
                  background: `linear-gradient(
                    155deg,
                    color-mix(in srgb, ${cell.toneVar}, transparent ${Math.round(
                      94 - clamp01(ratio(cell.value, total)) * 12
                    )}% ) 0%,
                    color-mix(in srgb, var(--viz-bg-surface), var(--viz-bg-canvas) 18%) 85%
                  )`,
                  boxShadow: `0 8px 16px color-mix(in srgb, ${cell.toneVar}, transparent 95%)`,
                }}
              >
                <div
                  className="absolute -top-10 -right-10 w-20 h-20 rounded-full pointer-events-none"
                  style={{
                    background: `radial-gradient(circle, color-mix(in srgb, ${cell.toneVar}, transparent 85%) 0%, transparent 72%)`,
                  }}
                />
                <span
                  className={`${labelSize} font-black uppercase tracking-[0.18em] opacity-80 mb-1`}
                >
                  {cell.label}
                </span>
                <span className={`${countSize} font-mono tabular-nums opacity-75 mb-1`}>
                  n={formatCount(cell.value)}
                </span>
                <span
                  className={`font-black ${valueSize} leading-none tabular-nums tracking-tight`}
                >
                  {formatPercent(cell.value, total)}
                </span>
              </div>
            }
          />
        ))}
      </div>
    </div>
  );
};
