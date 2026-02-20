/**
 * Provide UIComponents module functionality for the HPO dashboard.
 */

import { useState } from "react";
import { ArrowUp, ArrowDown } from "./BaseComponents.jsx";
import { PortalTooltip } from "./PortalTooltip.jsx";
export { PortalTooltip };
import { MetricRegistry } from "../domain/metrics/MetricRegistry.js";
import { ParamRegistry } from "../domain/metrics/ParamRegistry.js";
import { HintTooltipContent } from "./HintTooltipContent.jsx";
import { ChevronRight } from "./icons.jsx";

/**
 * Expose section divider for dashboard usage.
 */
export const SectionDivider = ({ label, icon: Icon }) => (
  <div className="col-span-full flex items-center gap-3 pb-2 border-b border-zinc-800/50 mt-8 mb-4 first:mt-0">
    {Icon && (
      <div className="p-1.5 rounded-md bg-zinc-900 text-orange-400">
        <Icon size={14} />
      </div>
    )}
    <h3 className="text-xs font-black uppercase tracking-[0.2em] text-zinc-500">{label}</h3>
  </div>
);

/**
 * Collapsible section wrapper with animated expand/collapse.
 */
export const CollapsibleSection = ({
  label,
  icon: Icon,
  children,
  defaultExpanded = true,
  sectionKey = "",
  className = "",
  contentClassName = "",
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <section
      className={`col-span-full first:mt-0 ${className}`}
      data-search-section={sectionKey || undefined}
    >
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="w-full relative flex items-center justify-between gap-3 px-2 py-2.5 group transition-colors duration-250"
        aria-expanded={expanded}
        aria-controls={sectionKey ? `${sectionKey}-content` : undefined}
        data-section-toggle={sectionKey || undefined}
        style={{
          border: "none",
          background: "transparent",
          backdropFilter: "none",
          boxShadow: "none",
        }}
      >
        <div
          className="pointer-events-none absolute inset-x-0 bottom-0 h-px"
          style={{
            background:
              "linear-gradient(90deg, transparent 0%, color-mix(in srgb, var(--viz-border), transparent 4%) 50%, transparent 100%)",
          }}
        />
        <div className="flex items-center gap-3">
          {Icon && (
            <div
              className="p-1.5 rounded-md transition-colors group-hover:text-yellow-300"
              style={{
                backgroundColor: "color-mix(in srgb, var(--viz-bg-canvas), transparent 6%)",
                color: "var(--viz-palette-3-orange)",
                border: "1px solid color-mix(in srgb, var(--viz-border), transparent 30%)",
              }}
            >
              <Icon size={14} />
            </div>
          )}
          <h3 className="text-xs font-black uppercase tracking-[0.2em] text-zinc-400 group-hover:text-zinc-100 transition-colors">
            {label}
          </h3>
        </div>
        <div
          className="flex items-center justify-center w-7 h-7 rounded-md transition-colors"
          style={{
            border: "1px solid color-mix(in srgb, var(--viz-border), transparent 28%)",
            backgroundColor: "color-mix(in srgb, var(--viz-bg-canvas), transparent 18%)",
            color: "var(--viz-text-secondary)",
            boxShadow: "inset 0 1px 0 color-mix(in srgb, var(--viz-text-primary), transparent 94%)",
          }}
        >
          <ChevronRight
            size={14}
            className={`transition-transform duration-300 ease-out ${expanded ? "rotate-90" : "rotate-0"}`}
          />
        </div>
      </button>

      <div
        id={sectionKey ? `${sectionKey}-content` : undefined}
        data-section-content={sectionKey || undefined}
        className="grid overflow-hidden transition-[grid-template-rows,opacity,margin-top] duration-300 ease-out"
        style={{
          gridTemplateRows: expanded ? "1fr" : "0fr",
          opacity: expanded ? 1 : 0,
          marginTop: expanded ? "14px" : "0px",
        }}
      >
        <div
          className={`min-h-0 overflow-hidden transition-transform duration-300 ease-out ${expanded ? "translate-y-0" : "-translate-y-1"} ${contentClassName}`}
        >
          {children}
        </div>
      </div>
    </section>
  );
};

/** Declarative lookup table for metric key normalization. Order matters — first match wins. */
const METRIC_KEY_MAP = [
  [/média móvel|moving average/i, "moving_average"],
  [
    /intervalo de confiança|confidence interval|ci_high|ci low|ci_low|ci high/i,
    "confidence_interval",
  ],
  [/(trials?|trial).*(reais|real)/i, "real_trials"],
  [/trials?/i, "trial"],
  [/tendência|trend/i, "trend"],
  [/melhor trial|melhor \(incumbent\)|incumbent/i, "incumbent"],
  [/objetivo|objective/i, "objective"],
  [/performance|dimensão/i, "performance_dim"],
  [/trade-off/i, "latency_tradeoff"],
  [/latência|latency/i, "inference_time"],
  [/duração/i, "duration"],
  [/época|epoca/i, "epoch"],
  [
    /loss.*(validação|validacao)|val_loss|val loss|validation loss|(validação|validacao).*loss/i,
    "val_loss",
  ],
  [/loss.*treino|train_loss|train loss|training loss/i, "train_loss"],
  [/gap.*generaliza(ç|c)[aã]o|generalization gap/i, "gap"],
  [/loss/i, "loss"],
  [/stability|delta|improvement/i, "stability"],
  [/mcc\/mrr|mcc.*mrr/i, "mrr"],
  [/mcc/i, "mcc"],
  [/mrr/i, "mrr"],
  [/métricas|metricas|metrics/i, "metrics"],
  [/recon/i, "recon"],
  [/kl/i, "kl_div"],
  [/rules|regras/i, "rules"],
  [/(grad|gradient).*norm/i, "grad_norm"],
  [/\bcpu\b/i, "cpu"],
  [/\bgpu\b/i, "gpu"],
  [/\bvram\b/i, "vram"],
  [/\bram\b/i, "ram"],
  [/valor/i, "value"],
  [/score/i, "score"],
  [/id|trial #|trial/i, "id"],
  [/importância/i, "importance"],
  [/params|parâmetros|parâmetro/i, "params"],
];

const PARAM_KEY_MAP = [
  [/rerank/i, "rerank_top_k"],
  [/contrastive.*temperature/i, "contrastive_temperature"],
  [/num.*global.*negative/i, "num_global_negatives"],
  [/kl.*weight/i, "kl_weight"],
];

/**
 * Normalize a label string to a registry key using a declarative lookup table.
 * Falls back to lower_snake_case of the input.
 */
const normalizeKey = (lower, keyMap) => {
  for (const [pattern, key] of keyMap) {
    if (pattern.test(lower)) return key;
  }
  return lower.replace(/ /g, "_");
};

/**
 * Generic hint renderer — single source of truth for metric/param hint tooltips.
 * Looks up the registry, renders HintTooltipContent inside PortalTooltip with directional arrows.
 */
const renderHints = (value, extraValue, registry, keyMap) => {
  const lower = String(value).toLowerCase();
  const key = normalizeKey(lower, keyMap);
  const hints = registry.getAll();
  const hint =
    hints[key] ??
    hints[lower.replace(/ /g, "_")] ?? {
      tech: `Indicador "${String(value)}" exibido na visualização atual. Use em conjunto com as demais séries para validar tendência, estabilidade e risco de regressão.`,
      simple:
        "Este rótulo identifica uma série do gráfico. Clique na legenda para ocultar/mostrar e compare o impacto visual com as outras linhas.",
      extra: [
        { label: "Interação", value: "Clique para ocultar/mostrar a série" },
        { label: "Leitura", value: "Compare tendência, inclinação e variação relativa" },
      ],
    };

  return (
    <span className="ml-2 group relative cursor-help inline-flex items-center gap-1">
      <PortalTooltip
        content={<HintTooltipContent hint={hint} value={value} extraValue={extraValue} />}
      >
        <span className="text-zinc-300 border-b border-dotted border-zinc-600 hover:text-orange-400 transition-colors whitespace-nowrap">
          {value}
        </span>
      </PortalTooltip>
      {hint.direction === "up" && <ArrowUp size={10} />}
      {hint.direction === "down" && <ArrowDown size={10} />}
    </span>
  );
};

/**
 * Expose render with hints for dashboard usage.
 */
export const renderWithHints = (value, extraValue = null) => {
  const maybeDataValue =
    extraValue && typeof extraValue === "object" && "value" in extraValue ? extraValue.value : null;
  return renderHints(value, maybeDataValue, MetricRegistry, METRIC_KEY_MAP);
};
/**
 * Expose render param with hints for dashboard usage.
 */
export const renderParamWithHints = (value, extraValue = null) =>
  renderHints(value, extraValue, ParamRegistry, PARAM_KEY_MAP);

/**
 * Expose chart axis label for dashboard usage.
 */
export const ChartAxisLabel = ({ viewBox, value, axis = "x", offset = 0 }) => {
  const { x, y, width, height } = viewBox;

  const isYRight = axis === "y-right";
  const isX = axis === "x";

  // Y-Axis (Left): Position 0 to Width (e.g., 60px). aligned to the left of the axis line (x)
  // Y-Axis (Right): Position X to X+Width.
  // X-Axis: Position X to X+Width, below the chart.

  let foreignX, foreignY, foreignW, foreignH;
  let containerClass = "flex items-center justify-center w-full h-full";

  if (isX) {
    foreignX = x;
    foreignY = y + 20; // Push down below ticks
    foreignW = width;
    foreignH = 30;
  } else {
    // Y Axis (Left or Right) - We create a box for the label
    // Rotation center is the middle of this box
    const axisWidth = width || 60; // Fallback

    foreignW = height; // Swapped for rotation logic if we were using pure SVG transforms, but for foreignObject we keep it simple
    foreignH = axisWidth;

    // Actually, easiest way for HTML in SVG rotation is to place a box and rotate the content
    foreignW = axisWidth;
    foreignH = height;

    if (isYRight) {
      foreignX = x + 10 + offset;
    } else {
      foreignX = x - axisWidth - offset; // Place in the margin area left of axis line
    }
    foreignY = y;

    // Rotate text -90deg
    containerClass =
      "flex items-center justify-center w-full h-full -rotate-90 origin-center whitespace-nowrap";
  }

  return (
    <foreignObject
      x={foreignX}
      y={foreignY}
      width={foreignW}
      height={foreignH}
      style={{ overflow: "visible" }}
    >
      <div className={containerClass}>{renderWithHints(value)}</div>
    </foreignObject>
  );
};
