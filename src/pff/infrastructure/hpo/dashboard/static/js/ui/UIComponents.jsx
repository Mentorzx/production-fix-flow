import { ArrowUp, ArrowDown } from "./BaseComponents.jsx";
import { PortalTooltip } from "./PortalTooltip.jsx";
export { PortalTooltip };
import { Theme } from "./Theme.js";
import { MetricRegistry } from "../domain/metrics/MetricRegistry.js";
import { ParamRegistry } from "../domain/metrics/ParamRegistry.js";
import { HintTooltipContent } from "./HintTooltipContent.jsx";

export const SectionDivider = ({ label, icon: Icon }) => (
    <div className="col-span-full flex items-center gap-3 pb-2 border-b border-zinc-800/50 mt-8 mb-4 first:mt-0">
        {Icon && <div className="p-1.5 rounded-md bg-zinc-900 text-orange-400"><Icon size={14} /></div>}
        <h3 className="text-xs font-black uppercase tracking-[0.2em] text-zinc-500">{label}</h3>
    </div>
);

/** Declarative lookup table for metric key normalization. Order matters — first match wins. */
const METRIC_KEY_MAP = [
    [/média móvel|moving average/i, 'moving_average'],
    [/intervalo de confiança|confidence interval|ci_high|ci low|ci_low|ci high/i, 'confidence_interval'],
    [/(trials?|trial).*(reais|real)/i, 'real_trials'],
    [/trials?/i, 'trial'],
    [/tendência|trend/i, 'trend'],
    [/melhor trial|melhor \(incumbent\)|incumbent/i, 'incumbent'],
    [/objetivo|objective/i, 'objective'],
    [/performance|dimensão/i, 'performance_dim'],
    [/trade-off/i, 'latency_tradeoff'],
    [/latência|latency/i, 'inference_time'],
    [/duração/i, 'duration'],
    [/época|epoca/i, 'epoch'],
    [/val_loss|val loss|validation loss|(validação|validacao).*loss/i, 'val_loss'],
    [/train_loss|train loss|training loss/i, 'train_loss'],
    [/loss/i, 'loss'],
    [/stability|delta|improvement/i, 'stability'],
    [/mcc\/mrr|mcc.*mrr/i, 'mrr'],
    [/mcc/i, 'mcc'],
    [/mrr/i, 'mrr'],
    [/métricas|metricas|metrics/i, 'metrics'],
    [/recon/i, 'recon'],
    [/kl/i, 'kl_div'],
    [/rules|regras/i, 'rules'],
    [/(grad|gradient).*norm/i, 'grad_norm'],
    [/\bcpu\b/i, 'cpu'],
    [/\bgpu\b/i, 'gpu'],
    [/\bvram\b/i, 'vram'],
    [/\bram\b/i, 'ram'],
    [/valor/i, 'value'],
    [/score/i, 'score'],
    [/id|trial #|trial/i, 'id'],
    [/importância/i, 'importance'],
    [/params|parâmetros|parâmetro/i, 'params'],
];

const PARAM_KEY_MAP = [
    [/rerank/i, 'rerank_top_k'],
    [/contrastive.*temperature/i, 'contrastive_temperature'],
    [/num.*global.*negative/i, 'num_global_negatives'],
    [/kl.*weight/i, 'kl_weight'],
];

/**
 * Normalize a label string to a registry key using a declarative lookup table.
 * Falls back to lower_snake_case of the input.
 */
const normalizeKey = (lower, keyMap) => {
    for (const [pattern, key] of keyMap) {
        if (pattern.test(lower)) return key;
    }
    return lower.replace(/ /g, '_');
};

/**
 * Generic hint renderer — single source of truth for metric/param hint tooltips.
 * Looks up the registry, renders HintTooltipContent inside PortalTooltip with directional arrows.
 */
const renderHints = (value, extraValue, registry, keyMap) => {
    const lower = String(value).toLowerCase();
    const key = normalizeKey(lower, keyMap);
    const hints = registry.getAll();
    const hint = hints[key] ?? hints[lower.replace(/ /g, '_')] ?? null;

    if (!hint || hint.tech === key) {
        return <span className="ml-2" style={{ color: Theme.ui.text.secondary }}>{value}</span>;
    }

    return (
        <span className="ml-2 group relative cursor-help inline-flex items-center gap-1">
            <PortalTooltip content={<HintTooltipContent hint={hint} value={value} extraValue={extraValue} />}>
                <span className="text-zinc-300 border-b border-dotted border-zinc-600 hover:text-orange-400 transition-colors whitespace-nowrap">
                    {value}
                </span>
            </PortalTooltip>
            {hint.direction === 'up' && <ArrowUp size={10} />}
            {hint.direction === 'down' && <ArrowDown size={10} />}
        </span>
    );
};

export const renderWithHints = (value, extraValue = null) => renderHints(value, extraValue, MetricRegistry, METRIC_KEY_MAP);
export const renderParamWithHints = (value, extraValue = null) => renderHints(value, extraValue, ParamRegistry, PARAM_KEY_MAP);

export const ChartAxisLabel = ({ viewBox, value, axis = 'x', offset = 0 }) => {
    const { x, y, width, height } = viewBox;

    const isYRight = axis === 'y-right';
    const isX = axis === 'x';

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
        containerClass = "flex items-center justify-center w-full h-full -rotate-90 origin-center whitespace-nowrap";
    }

    return (
        <foreignObject x={foreignX} y={foreignY} width={foreignW} height={foreignH} style={{ overflow: 'visible' }}>
            <div className={containerClass}>
                {renderWithHints(value)}
            </div>
        </foreignObject>
    );
};
