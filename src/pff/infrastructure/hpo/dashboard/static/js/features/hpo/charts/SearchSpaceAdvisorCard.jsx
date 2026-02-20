/**
 * SearchSpaceAdvisorCard - SOTA Hyperparameter Optimization Decision Panel
 *
 * Design Principles (SOTA):
 * 1. Visual Density: Strip plots + topology rulers for at-a-glance pattern recognition
 * 2. Decision-Driven: Action-oriented UI with clear affordances
 * 3. Progressive Disclosure: Drill-down panels for detailed technical analysis
 * 4. Semantic Color: Project palette for consistent visual language
 *
 * @frontend-specialist: This component implements "Chromatic Engineering" patterns
 * using the project's NEON_DARK_PALETTE and semantic tokens.
 */

import { useMemo, useState, useCallback, useEffect } from "react";
import {
  Card,
  Sliders,
  AlertTriangle,
  ArrowDown,
  Zap,
  Info,
  CheckCircle,
  X,
  TrendingUp,
  TargetIcon,
  WithData,
} from "../../../ui/BaseComponents.jsx";
import { renderParamWithHints } from "../../../ui/UIComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";
import { Theme } from "../../../ui/Theme.js";

// ============================================================================
// LOCALIZATION (PT-BR) - Glossário Técnico Localizado
// ============================================================================

const LABELS = {
  title: "Consultor de Espaço de Busca",
  subtitle: "Análise Inteligente de Hiperparâmetros",
  columns: {
    parameter: "Parâmetro",
    stats: "Estatísticas",
    density: "Densidade",
    topology: "Topologia",
    importance: "Importância",
    confidence: "Confiança",
    action: "Ação",
  },
  badges: {
    int: "INT",
    float: "FLOAT",
    log: "LOG",
    categorical: "CAT",
  },
  stats: {
    n: "n",
    mean: "μ",
    std: "σ",
    median: "med",
    skewness: "assim",
    kurtosis: "curt",
  },
  confidence: {
    high: "ALTA",
    medium: "MÉDIA",
    low: "BAIXA",
  },
  topology: {
    current: "Atual",
    proposal: "Proposta",
    expand: "Expandir",
    shrink: "Reduzir",
    shift: "Deslocar",
  },
  drillDown: {
    title: "Análise Detalhada",
    technicalAnalysis: "Análise Técnica",
    advancedStats: "Estatísticas Avançadas",
    estimatedImpact: "Impacto Estimado",
    bootstrap: "Bootstrap",
    interactions: "Interações",
    surrogate: "Surrogate",
    apply: "Aplicar Ajuste",
    dismiss: "Ignorar",
    bestRegion: "Melhor Região",
    trialDistribution: "Distribuição dos Trials",
    defaultRationale: "Recomendação automática baseada no desempenho observado.",
  },
  hints: {
    stats:
      "n = número de trials\nμ (mu) = média aritmética\nσ (sigma) = desvio padrão\n\nQuanto maior σ, mais dispersos estão os valores testados.",
    density:
      "Cada ponto azul = 1 trial.\nÁrea amarela = região dos top performers (q10-q90).\n\nPadrões:\n• Pontos agrupados à direita → expandir limite superior\n• Poucos pontos na área amarela → precisa mais exploração",
    topology:
      "Barra cinza = espaço de busca atual\nContorno colorido = proposta da IA\n\nVerde = expansão (mais opções)\nLaranja = redução (focar busca)\nSetas indicam direção da mudança",
    importance:
      "Quanto este parâmetro influencia o resultado final.\n\n>40%: Crítico - pequenas mudanças têm grande impacto\n10-40%: Moderado - vale a pena ajustar\n<10%: Baixo - pode fixar em valor constante",
    confidence:
      "Quanto a IA confia nesta recomendação:\n\n⚡ ALTA: Dados claros, evidência forte\n🎯 MÉDIA: Tendência visível, mas verificar\nℹ️ BAIXA: Poucos dados, recomendação incerta",
    actions: {
      expand_upper: "Aumentar o limite máximo permitido\nEx: learning_rate de 0.01 → 0.05",
      expand_lower: "Aumentar o limite mínimo permitido\nEx: warmup_steps de 100 → 50",
      narrow: "Reduzir o intervalo para focar a busca\nEx: batch_size de [16,512] → [64,256]",
      fix: "Fixar em valor constante (baixa importância)\nEx: dropout = 0.3 sempre",
      keep: "Manter configuração atual",
      reduce_categories: "Remover categorias com baixo desempenho para acelerar a convergência",
      change_distribution:
        "Trocar a distribuição (ex.: uniforme → log) para explorar regiões esquecidas",
    },
  },
};

const ACTION_LABELS = {
  expand_upper: "EXPANDIR \u2191",
  expand_lower: "EXPANDIR \u2193",
  narrow: "REDUZIR",
  fix: "FIXAR",
  reduce_categories: "REDUZIR CAT.",
  change_distribution: "ALTERAR DIST.",
  keep: "MANTER",
};

// ============================================================================
// COLOR SYSTEM - Semantic mappings from Theme
// ============================================================================

const CONFIDENCE_COLORS = {
  high: Theme.semantic.success,
  medium: Theme.semantic.warning,
  low: Theme.ui.text.muted,
};

const CONFIDENCE_ICONS = {
  high: Zap,
  medium: TargetIcon,
  low: Info,
};

const ACTION_COLORS = {
  expand_upper: Theme.palette.neonBlue,
  expand_lower: Theme.palette.neonBlue,
  narrow: Theme.palette.hotOrange,
  fix: Theme.palette.purple,
  reduce_categories: Theme.palette.hotOrange,
  change_distribution: Theme.palette.mint,
  keep: Theme.ui.text.muted,
};

const TOPOLOGY_COLORS = {
  current: Theme.palette.grey,
  expand: Theme.palette.vividGreen,
  shrink: Theme.palette.hotOrange,
  shift: Theme.palette.cyberYellow,
  bestRegion: Theme.palette.cyberYellow + "80",
  trialPoint: Theme.palette.neonBlue,
};

// ============================================================================
// FORMATTING HELPERS - Compact, Visual-First Formatting
// ============================================================================

const getParamType = (space) => {
  if (!space) return null;
  if (space.choices) return "categorical";
  if (space.log) return "log";
  if (space.low != null && space.high != null) {
    return Number.isInteger(space.low) && Number.isInteger(space.high) ? "int" : "float";
  }
  return null;
};

// ============================================================================
// SMART NUMBER FORMATTING - Prevents ugly scientific notation
// ============================================================================

/**
 * Scientific notation formatter: "3,4 × 10⁻³"
 */
const fmtScientific = (v, digits = 1) => {
  if (v === 0) return "0";
  const exp = Math.floor(Math.log10(Math.abs(v)));
  const mantissa = v / Math.pow(10, exp);
  const mantissaStr = mantissa.toFixed(digits).replace(".", ",");

  // Unicode superscript digits
  const superscripts = {
    0: "⁰",
    1: "¹",
    2: "²",
    3: "³",
    4: "⁴",
    5: "⁵",
    6: "⁶",
    7: "⁷",
    8: "⁸",
    9: "⁹",
    "-": "⁻",
  };

  const expStr = String(exp)
    .split("")
    .map((c) => superscripts[c] || c)
    .join("");
  return `${mantissaStr}×10${expStr}`;
};

/**
 * Smart number formatter:
 * - 0.01 to 999: decimal normal
 * - < 0.01 or > 999: notação científica "3,4 × 10⁻³"
 */
const fmtSmart = (v) => {
  if (v == null) return "—";
  if (typeof v !== "number") return String(v);
  if (isNaN(v)) return "—";
  if (v === 0) return "0";

  const absV = Math.abs(v);

  // Scientific notation for very small (< 0.01) or very large (> 999)
  if (absV < 0.01 || absV > 999) {
    return fmtScientific(v, 1);
  }

  // Normal decimal for medium values
  if (absV < 0.1) return v.toFixed(3);
  if (absV < 1) return v.toFixed(2);
  if (absV < 10) return v.toFixed(1);
  return v.toFixed(0);
};

/**
 * Format for delta/change display (compact)
 * Uses scientific notation for extremes: "3,4 × 10⁻³"
 */
const fmtDelta = (v) => {
  if (v == null) return "—";
  if (typeof v !== "number") return String(v);
  if (isNaN(v)) return "—";
  if (v === 0) return "0";

  const absV = Math.abs(v);

  // Scientific notation for very small (< 0.01) or very large (> 999)
  if (absV < 0.01 || absV > 999) {
    return fmtScientific(v, 1);
  }

  // Normal decimal for medium values
  if (absV < 0.1) return v.toFixed(3);
  if (absV < 1) return v.toFixed(2);
  if (absV < 10) return v.toFixed(1);
  return v.toFixed(0);
};

// ============================================================================
// SOTA VISUAL COMPONENTS - State-of-the-Art Data Visualization
// ============================================================================

/**
 * TrialDensityStrip - Strip plot with fixed generous width
 */
const TrialDensityStrip = ({ attempts, bestRegion, space }) => {
  const width = 140;
  const height = 36;
  const padding = 10;

  // Check for invalid/empty data
  const hasValidData =
    space && (space.low != null || space.choices) && attempts && attempts.count > 0;

  if (!hasValidData) {
    return (
      <div
        style={{ minHeight: height }}
        className="flex flex-col items-center justify-center opacity-40 w-full"
      >
        <span className="text-[10px] text-zinc-500">—</span>
        <span className="text-[7px] text-zinc-600 mt-0.5">sem dados</span>
      </div>
    );
  }

  const isCategorical = Array.isArray(space?.choices);
  const numericLow = typeof space?.low === "number" ? space.low : 0;
  const numericHigh = typeof space?.high === "number" ? space.high : 1;
  const min = isCategorical ? 0 : numericLow;
  const max = isCategorical ? (space?.choices?.length || 1) - 1 : numericHigh;
  const range = max - min || 1;

  const normalize = (val) => {
    if (isCategorical) {
      const idx = space.choices?.indexOf(val) ?? 0;
      return padding + (idx / Math.max(1, space.choices.length - 1)) * (width - 2 * padding);
    }
    const clamped = Math.max(min, Math.min(max, val));
    return padding + ((clamped - min) / range) * (width - 2 * padding);
  };

  const trialValues = (() => {
    if (!attempts) return [];

    if (isCategorical && attempts.distribution) {
      return Object.entries(attempts.distribution).flatMap(([val, count]) =>
        Array(count).fill(val)
      );
    }

    if (!isCategorical && Array.isArray(attempts.samples) && attempts.samples.length > 0) {
      return attempts.samples.map((value) => Number(value)).filter((value) => !Number.isNaN(value));
    }

    if (!isCategorical && attempts.stats) {
      const stats = attempts.stats;
      const anchors = [
        stats.min,
        stats.q10,
        stats.q25,
        stats.q50,
        stats.q75,
        stats.q90,
        stats.max,
      ].filter((value) => typeof value === "number" && !Number.isNaN(value));

      if (anchors.length >= 2) {
        const synthCount = Math.min(
          60,
          Math.max(
            anchors.length * 3,
            Math.min(attempts.count || stats.count || 0, 60) || anchors.length * 3
          )
        );
        const segments = anchors.length - 1;
        const perSegment = Math.max(2, Math.floor(synthCount / segments));
        const synthetic = [];

        for (let segment = 0; segment < segments; segment += 1) {
          const start = anchors[segment];
          const end = anchors[segment + 1];
          const segmentSpan = Math.abs(end - start) || 1;
          for (let step = 0; step < perSegment; step += 1) {
            const t = step / perSegment;
            const base = start + (end - start) * t;
            const jitterSeed = segment * perSegment + step;
            const jitter = ((jitterSeed % 5) - 2) * 0.01 * segmentSpan;
            const candidate = Math.max(min, Math.min(max, base + jitter));
            synthetic.push(candidate);
          }
        }

        synthetic.push(anchors[anchors.length - 1]);
        return synthetic;
      }

      if (typeof stats.mean === "number") {
        return [stats.mean];
      }
    }

    return [];
  })();

  const trials = trialValues
    .slice(0, 60)
    .map((value) => (isCategorical ? value : Number(value)))
    .filter((value) => (isCategorical ? true : !Number.isNaN(value)));

  const bestMin = bestRegion?.stats?.q10 ?? bestRegion?.stats?.min;
  const bestMax = bestRegion?.stats?.q90 ?? bestRegion?.stats?.max;

  // Count trials in best region
  const trialsInBest =
    bestMin != null && bestMax != null
      ? trials.filter((v) => v >= bestMin && v <= bestMax).length
      : 0;

  return (
    <div className="flex flex-col items-center w-full">
      <svg
        width="100%"
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="overflow-visible"
      >
        {/* Background track - subtle */}
        <line
          x1={padding}
          y1={height / 2}
          x2={width - padding}
          y2={height / 2}
          stroke={Theme.ui.border}
          strokeWidth={1}
          opacity={0.5}
        />

        {/* Best region highlight - semi-transparent band */}
        {bestMin != null && bestMax != null && (
          <rect
            x={normalize(bestMin)}
            y={4}
            width={Math.max(6, normalize(bestMax) - normalize(bestMin))}
            height={height - 8}
            fill={Theme.palette.cyberYellow}
            opacity={0.15}
            rx={3}
          />
        )}

        {/* Trial points - CLEARLY VISIBLE dots with jitter */}
        {trials.slice(0, 50).map((val, i) => {
          // Limit to 50 for performance
          const x = normalize(val);
          // Multi-level jitter to prevent overlap
          const jitterY = ((i * 7) % 11) - 5; // -5 to +5
          const opacity = Math.max(0.4, 0.9 - i / 100); // Fade older trials slightly

          // Highlight if in best region
          const isInBest = bestMin != null && bestMax != null && val >= bestMin && val <= bestMax;

          return (
            <circle
              key={i}
              cx={x}
              cy={height / 2 + jitterY}
              r={isInBest ? 2.5 : 2}
              fill={isInBest ? Theme.palette.cyberYellow : Theme.palette.neonBlue}
              opacity={isInBest ? 1 : opacity}
              stroke={isInBest ? Theme.palette.cyberYellow : "none"}
              strokeWidth={0.5}
            />
          );
        })}

        {/* Best region indicator line */}
        {bestMin != null && bestMax != null && (
          <>
            <line
              x1={normalize(bestMin)}
              y1={height / 2 - 6}
              x2={normalize(bestMin)}
              y2={height / 2 + 6}
              stroke={Theme.palette.cyberYellow}
              strokeWidth={1}
              opacity={0.8}
            />
            <line
              x1={normalize(bestMax)}
              y1={height / 2 - 6}
              x2={normalize(bestMax)}
              y2={height / 2 + 6}
              stroke={Theme.palette.cyberYellow}
              strokeWidth={1}
              opacity={0.8}
            />
          </>
        )}
      </svg>

      {/* Micro-label: count in best region */}
      {trialsInBest > 0 && (
        <span className="text-[9px] font-mono mt-1" style={{ color: Theme.palette.cyberYellow }}>
          {trialsInBest}/{trials.length} top
        </span>
      )}
    </div>
  );
};

/**
 * SpaceTopologyRuler - Fixed-width ruler with proper proportions
 */
const SpaceTopologyRuler = ({ current, proposal, action }) => {
  const width = 160; // Base width for the ruler viewBox
  const barHeight = 32;
  const axisY = 40;

  if (!current || (!current.low && !current.choices)) {
    return (
      <div
        className="w-full flex items-center justify-center text-[8px] text-zinc-500"
        style={{ minHeight: 30 }}
      >
        —
      </div>
    );
  }

  const isCategorical = !!current.choices;
  const isLogScale = current.log === true;

  // CRITICAL FIX: For log scale, ensure values are never 0 or negative
  const sanitizeForLog = (val) => {
    if (!isLogScale) return val;
    if (val == null || val <= 0 || isNaN(val)) return 1e-6; // Minimum positive value for log
    return val;
  };

  const currentLow = isCategorical ? 0 : sanitizeForLog(current.low ?? 0);
  const currentHigh = isCategorical
    ? (current.choices?.length || 1) - 1
    : sanitizeForLog(current.high ?? 1);

  // Calculate global bounds including proposal for consistent scaling
  let propLow = proposal?.new_low ?? proposal?.fix_value;
  let propHigh = proposal?.new_high ?? proposal?.fix_value;

  // Sanitize proposal values for log scale
  if (isLogScale) {
    propLow = sanitizeForLog(propLow);
    propHigh = sanitizeForLog(propHigh);
  }

  // Handle log scale normalization
  let globalMin, globalMax, normalize;

  if (isLogScale && !isCategorical) {
    // Log scale: use log values for proportional spacing
    const logMin = Math.log10(Math.min(currentLow, propLow ?? currentLow));
    const logMax = Math.log10(Math.max(currentHigh, propHigh ?? currentHigh));
    const logRange = logMax - logMin || 1;

    normalize = (val) => {
      const safeVal = sanitizeForLog(val);
      const logVal = Math.log10(safeVal);
      return ((logVal - logMin) / logRange) * (width - 20) + 10;
    };
    globalMin = Math.pow(10, logMin);
    globalMax = Math.pow(10, logMax);
  } else {
    // Linear scale (original behavior)
    globalMin = Math.min(currentLow, propLow ?? currentLow);
    globalMax = Math.max(currentHigh, propHigh ?? currentHigh);
    const range = globalMax - globalMin || 1;
    normalize = (val) => ((val - globalMin) / range) * (width - 20) + 10;
  }

  // Determine action color
  let actionColor = TOPOLOGY_COLORS.current;
  if (action?.includes("expand")) actionColor = TOPOLOGY_COLORS.expand;
  else if (action === "narrow") actionColor = TOPOLOGY_COLORS.shrink;
  else if (action === "fix") actionColor = TOPOLOGY_COLORS.shift;

  const curLeft = normalize(currentLow);
  const curRight = normalize(currentHigh);
  const curWidth = Math.max(4, curRight - curLeft);

  const hasProposal = action !== "keep" && (propLow != null || propHigh != null);
  const showMaxColumn = !isCategorical;
  const labelGridTemplate = (() => {
    if (showMaxColumn && hasProposal) {
      return "minmax(0,120px) minmax(0,1fr) minmax(0,120px)";
    }
    if (showMaxColumn || hasProposal) {
      return "repeat(2, minmax(0,120px))";
    }
    return "1fr";
  })();
  const labelRowStyle = {
    gridTemplateColumns: labelGridTemplate,
    paddingTop: "14px",
    marginTop: "10px",
    rowGap: "6px",
    columnGap: "12px",
  };
  const propLeft = hasProposal ? normalize(propLow ?? propHigh ?? currentLow) : curLeft;
  const propRight = hasProposal ? normalize(propHigh ?? propLow ?? currentHigh) : curRight;
  const propWidth = Math.max(4, propRight - propLeft);

  // Check for invalid calculations
  const isValid = !isNaN(curLeft) && !isNaN(curRight) && curWidth > 0;

  if (!isValid) {
    return (
      <div className="flex flex-col items-center justify-center w-full" style={{ minHeight: 30 }}>
        <span className="text-[9px] text-zinc-500">Dados inválidos</span>
      </div>
    );
  }

  // Contextual formatter for log scale values
  const fmtTopology = (v, isLog) => {
    if (v == null) return "—";
    if (isLog && (v === 0 || v <= 1e-10)) return "< 1×10⁻⁶"; // Show minimum safe value
    return fmtSmart(v);
  };

  return (
    <div className="flex flex-col w-full">
      {/* Bars area - INCREASED HEIGHT */}
      <svg
        width="100%"
        height={40}
        viewBox={`0 0 ${width} 40`}
        preserveAspectRatio="none"
        className="overflow-visible"
      >
        {/* Background track */}
        <rect
          x={5}
          y={axisY - barHeight / 2}
          width={width - 10}
          height={barHeight}
          fill={Theme.ui.background}
          rx={4}
          opacity={0.3}
        />

        {/* Current range - solid bar (thicker) */}
        <rect
          x={curLeft}
          y={axisY - barHeight / 2 + 2}
          width={curWidth}
          height={barHeight - 4}
          fill={TOPOLOGY_COLORS.current}
          rx={3}
          opacity={0.7}
        />

        {/* Proposal range - outline/ghost bar */}
        {hasProposal && (
          <>
            <rect
              x={propLeft}
              y={axisY - barHeight / 2}
              width={propWidth}
              height={barHeight}
              fill="none"
              stroke={actionColor}
              strokeWidth={2}
              rx={4}
              strokeDasharray={action === "fix" ? "4,3" : undefined}
            />
            {/* Direction arrows for expansion */}
            {action?.includes("expand") && propLeft < curLeft && (
              <polygon
                points={`${propLeft},${axisY} ${propLeft + 6},${axisY - 4} ${propLeft + 6},${axisY + 4}`}
                fill={actionColor}
              />
            )}
            {action?.includes("expand") && propRight > curRight && (
              <polygon
                points={`${propRight},${axisY} ${propRight - 6},${axisY - 4} ${propRight - 6},${axisY + 4}`}
                fill={actionColor}
              />
            )}
          </>
        )}
      </svg>

      {/* Labels row - GRID layout to prevent overlap */}
      <div className="grid items-start px-1 w-full" style={labelRowStyle}>
        {/* Left value (min) */}
        <div className="flex flex-col">
          <span
            className="text-[8px] uppercase tracking-wider"
            style={{ color: Theme.ui.text.muted }}
          >
            {isCategorical ? "vals" : "min"}
          </span>
          <span
            className="text-xs font-mono font-semibold whitespace-nowrap"
            style={{ color: Theme.ui.text.secondary }}
          >
            {isCategorical ? current.choices?.length : fmtTopology(globalMin, isLogScale)}
          </span>
        </div>

        {/* Center - proposal indicator */}
        {hasProposal && (
          <div className="flex flex-col items-center justify-center text-center whitespace-nowrap w-full">
            <span className="text-[8px] uppercase tracking-wider" style={{ color: actionColor }}>
              proposta
            </span>
          </div>
        )}

        {/* Right value (max) */}
        {showMaxColumn && (
          <div
            className="flex flex-col items-end text-right whitespace-nowrap max-w-full"
            style={{ textAlign: "right" }}
          >
            <span
              className="text-[8px] uppercase tracking-wider"
              style={{ color: Theme.ui.text.muted }}
            >
              max
            </span>
            <span
              className="text-xs font-mono font-semibold whitespace-nowrap"
              style={{ color: Theme.ui.text.secondary }}
            >
              {fmtTopology(globalMax, isLogScale)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * ImportanceBar - High-contrast gradient bar
 * UX Fix: Brighter colors on dark background for visibility
 */
const ImportanceBar = ({ value }) => {
  const pct = Math.min(100, Math.max(0, (value || 0) * 100));

  // High-contrast colors for dark background
  let gradientId = "importance-low";
  if (pct > 40) gradientId = "importance-high";
  else if (pct > 10) gradientId = "importance-med";

  return (
    <div className="flex items-center gap-3 w-full">
      <svg
        width="100%"
        height={14}
        viewBox="0 0 100 14"
        preserveAspectRatio="none"
        className="flex-1"
      >
        <defs>
          {/* High importance: Bright mint/cyan - VISIBLE on dark */}
          <linearGradient id="importance-high" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={Theme.palette.mint} />
            <stop offset="100%" stopColor={Theme.palette.cyan} />
          </linearGradient>
          {/* Medium importance: Cyber yellow to orange - VISIBLE */}
          <linearGradient id="importance-med" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={Theme.palette.cyberYellow} />
            <stop offset="100%" stopColor={Theme.palette.apricot} />
          </linearGradient>
          {/* Low importance: Subtle grey */}
          <linearGradient id="importance-low" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={Theme.palette.grey} />
            <stop offset="100%" stopColor={Theme.ui.border} />
          </linearGradient>
        </defs>
        {/* Background track */}
        <rect
          x={0}
          y={1}
          width={100}
          height={10}
          rx={5}
          fill={Theme.ui.background}
          stroke={Theme.ui.border}
          strokeWidth={0.5}
        />
        {/* Fill bar */}
        <rect
          x={0}
          y={1}
          width={Math.max(3, (pct / 100) * 100)}
          height={10}
          rx={5}
          fill={`url(#${gradientId})`}
        />
      </svg>
      <span
        className="text-xs font-mono font-bold tabular-nums"
        style={{
          color: pct > 10 ? Theme.palette.cyberYellow : Theme.ui.text.muted,
          minWidth: "40px",
        }}
      >
        {pct.toFixed(1)}%
      </span>
    </div>
  );
};

/**
 * ConfidenceBadge - Always shows descriptive text + icon
 * UX Fix: No ambiguity - full word always visible
 */
const ConfidenceBadge = ({ level }) => {
  const color = CONFIDENCE_COLORS[level] || CONFIDENCE_COLORS.low;
  const Icon = CONFIDENCE_ICONS[level] || Info;

  // Full descriptive labels
  const labels = {
    high: "ALTA",
    medium: "MÉDIA",
    low: "BAIXA",
  };

  const descriptions = {
    high: "Dados sólidos",
    medium: "Tendência visível",
    low: "Evidência fraca",
  };

  return (
    <div className="flex flex-col gap-0.5">
      <span
        className="inline-flex items-center gap-1.5 px-2 py-1 rounded"
        style={{
          backgroundColor: color + "15",
          border: `1px solid ${color}40`,
        }}
      >
        <Icon size={14} style={{ color }} />
        <span className="text-xs font-bold uppercase" style={{ color }}>
          {labels[level] || "BAIXA"}
        </span>
      </span>
      <span className="text-[9px] text-zinc-500 leading-tight">
        {descriptions[level] || "Dados insuficientes"}
      </span>
    </div>
  );
};

/**
 * SmartActionButton - Shows delta value directly (e.g., "0.01 → 0.02")
 * UX Fix: Eliminates cognitive load - user sees result before clicking
 * CRITICAL FIX: Uses smart formatting to avoid 0e+0 and ugly notation
 */
const SmartActionButton = ({ action, recommendation, onClick }) => {
  const color = ACTION_COLORS[action] || Theme.ui.text.muted;
  const isInteractive = onClick && action !== "keep";

  // Build smart label showing the actual change
  let smartLabel = ACTION_LABELS[action];
  let deltaText = null;
  let hasError = false;

  if (recommendation && action !== "keep") {
    const r = recommendation;

    // Check for invalid proposals (0 on log scale)
    const isInvalid = (v) => v === 0 && r.log_scale;

    if (action === "expand_upper" && r.new_high != null && r.old_high != null) {
      if (isInvalid(r.new_high)) hasError = true;
      deltaText = `${fmtDelta(r.old_high)} \u2192 ${fmtDelta(r.new_high)}`;
    } else if (action === "expand_lower" && r.new_low != null && r.old_low != null) {
      if (isInvalid(r.new_low)) hasError = true;
      deltaText = `${fmtDelta(r.old_low)} \u2192 ${fmtDelta(r.new_low)}`;
    } else if (action === "narrow" && r.new_low != null && r.new_high != null) {
      deltaText = `[${fmtDelta(r.new_low)}, ${fmtDelta(r.new_high)}]`;
    } else if (action === "fix" && r.fix_value != null) {
      if (isInvalid(r.fix_value)) hasError = true;
      deltaText = `= ${fmtSmart(r.fix_value)}`;
    } else if (action === "reduce_categories" && r.keep) {
      deltaText = `${r.keep.length} categorias`;
    } else if (action === "change_distribution") {
      deltaText = r.distribution || "log";
    }

    // Detect invalid expansion (same values)
    if (
      (action === "expand_upper" && r.old_high === r.new_high) ||
      (action === "expand_lower" && r.old_low === r.new_low)
    ) {
      hasError = true;
      deltaText = "valores iguais";
    }
  }

  return (
    <button
      onClick={onClick}
      disabled={!isInteractive || hasError}
      className={`
        flex flex-col items-center justify-center px-3 py-2 rounded
        transition-all duration-150 min-w-[90px] w-full
        ${isInteractive && !hasError ? "hover:brightness-125 hover:scale-105 active:scale-95 cursor-pointer" : "cursor-not-allowed opacity-50"}
      `}
      style={{
        backgroundColor: hasError
          ? Theme.semantic.error + "20"
          : color + (isInteractive ? "25" : "10"),
        border: `1px solid ${hasError ? Theme.semantic.error : color}${isInteractive ? "50" : "25"}`,
      }}
      title={hasError ? "Proposta inválida: verifique escala logarítmica" : ""}
    >
      {/* Action verb */}
      <span
        className="text-[10px] font-black uppercase tracking-wider leading-none"
        style={{ color: hasError ? Theme.semantic.error : color }}
      >
        {smartLabel}
      </span>
      {/* Delta preview */}
      {deltaText && (
        <span
          className="text-sm font-mono font-bold mt-1.5 leading-none"
          style={{
            color: hasError ? Theme.semantic.error : isInteractive ? "#fff" : Theme.ui.text.muted,
          }}
        >
          {deltaText}
        </span>
      )}
    </button>
  );
};

/**
 * ParameterBadge - Type indicator for parameter (INT/LOG/FLOAT/CAT)
 */
const ParameterBadge = ({ type }) => {
  if (!type) return null;

  const badgeColors = {
    int: Theme.palette.neonBlue,
    float: Theme.palette.cyan,
    log: Theme.palette.cyberYellow,
    categorical: Theme.palette.mint,
  };

  const label = LABELS.badges[type] || type.toUpperCase();
  const color = badgeColors[type] || Theme.palette.grey;

  return (
    <span
      className="inline-flex items-center px-2 py-0.5 rounded text-xs font-bold"
      style={{
        backgroundColor: color + "20",
        color,
        border: `1px solid ${color}40`,
      }}
    >
      {label}
    </span>
  );
};

// ============================================================================
// MODAL COMPONENTS - Drill-down and Preview
// ============================================================================

/**
 * ColumnHint - Detailed tooltip for column headers
 * Explains Greek letters, visualizations, and technical concepts
 */
const ColumnHint = ({ title, content, children }) => {
  const [showHint, setShowHint] = useState(false);

  return (
    <div
      className="relative inline-flex items-center gap-1 cursor-help"
      onMouseEnter={() => setShowHint(true)}
      onMouseLeave={() => setShowHint(false)}
    >
      {children}
      <Info size={12} style={{ color: Theme.ui.text.muted, opacity: 0.6 }} />

      {showHint && (
        <div
          className="absolute top-full left-0 mt-2 p-3 rounded-lg border z-50 w-72 shadow-xl"
          style={{
            backgroundColor: Theme.ui.surface,
            borderColor: Theme.ui.border,
            borderLeft: `3px solid ${Theme.palette.cyberYellow}`,
          }}
        >
          <h5 className="text-[11px] font-bold mb-2" style={{ color: Theme.palette.cyberYellow }}>
            {title}
          </h5>
          <div
            className="text-[11px] leading-relaxed whitespace-pre-line"
            style={{ color: Theme.ui.text.secondary }}
          >
            {content}
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * DrillDownPanel - Expanded technical analysis panel
 * Provides: Technical explanation, advanced stats, ROI estimation, action buttons
 */
const DrillDownPanel = ({ rec, onClose, onApply, onIgnore }) => {
  if (!rec) return null;

  const estimatedImpact = rec.estimated_improvement
    ? `+${(rec.estimated_improvement * 100).toFixed(1)}% no Score`
    : "Impacto indeterminado";

  const hasAdvancedStats = rec.attempts_summary?.stats?.skewness != null;
  const actionDescription = LABELS.hints.actions?.[rec.action] ?? null;
  const rationaleText = actionDescription || LABELS.drillDown.defaultRationale;
  const bootstrapSupport =
    rec.bootstrap_support != null ? `${Math.round(rec.bootstrap_support * 100)}%` : "—";
  const interactionStrength =
    rec.interaction_strength != null ? rec.interaction_strength.toFixed(3) : "—";
  const surrogateBounds = rec.surrogate_bounds
    ? `LCB_in=${fmtSmart(rec.surrogate_bounds.lcb_inside)}, UCB_out=${fmtSmart(rec.surrogate_bounds.ucb_outside)}`
    : "—";

  return (
    <div
      className="col-span-full p-4 rounded-lg border mt-2 mb-2"
      style={{
        backgroundColor: Theme.ui.background + "80",
        borderColor: Theme.ui.border,
        borderLeft: `3px solid ${ACTION_COLORS[rec.action] || Theme.palette.grey}`,
      }}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h4
            className="text-sm font-black uppercase tracking-wider mb-2"
            style={{ color: Theme.ui.text.primary }}
          >
            {LABELS.drillDown.title}: {rec.param_name}
          </h4>
          <p className="text-xs leading-relaxed" style={{ color: Theme.ui.text.secondary }}>
            {rationaleText}
          </p>
        </div>
        <button onClick={onClose} className="p-2 rounded hover:bg-white/10 transition-colors ml-4">
          <X size={18} style={{ color: Theme.ui.text.muted }} />
        </button>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        {/* ROI Card */}
        <div
          className="p-3 rounded border"
          style={{ backgroundColor: Theme.ui.surface, borderColor: Theme.ui.border }}
        >
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp size={14} style={{ color: Theme.palette.mint }} />
            <span
              className="text-[10px] uppercase tracking-wider"
              style={{ color: Theme.ui.text.muted }}
            >
              {LABELS.drillDown.estimatedImpact}
            </span>
          </div>
          <span className="text-base font-bold font-mono" style={{ color: Theme.palette.mint }}>
            {estimatedImpact}
          </span>
        </div>

        {/* Best Region Card */}
        <div
          className="p-3 rounded border"
          style={{ backgroundColor: Theme.ui.surface, borderColor: Theme.ui.border }}
        >
          <div className="flex items-center gap-2 mb-2">
            <TargetIcon size={14} style={{ color: Theme.palette.cyberYellow }} />
            <span
              className="text-[10px] uppercase tracking-wider"
              style={{ color: Theme.ui.text.muted }}
            >
              {LABELS.drillDown.bestRegion}
            </span>
          </div>
          <span className="text-sm font-mono" style={{ color: Theme.ui.text.secondary }}>
            {rec.best_region?.stats
              ? `[${Number(rec.best_region.stats.q10).toPrecision(3)}, ${Number(rec.best_region.stats.q90).toPrecision(3)}]`
              : "N/A"}
          </span>
        </div>

        {/* Advanced Stats */}
        {hasAdvancedStats && (
          <div
            className="p-3 rounded border"
            style={{ backgroundColor: Theme.ui.surface, borderColor: Theme.ui.border }}
          >
            <span
              className="text-[10px] uppercase tracking-wider block mb-2"
              style={{ color: Theme.ui.text.muted }}
            >
              {LABELS.drillDown.advancedStats}
            </span>
            <div className="text-xs font-mono space-y-1" style={{ color: Theme.ui.text.secondary }}>
              <div>
                {LABELS.stats.skewness}: {Number(rec.attempts_summary.stats.skewness).toFixed(2)}
              </div>
              <div>
                {LABELS.stats.kurtosis}: {Number(rec.attempts_summary.stats.kurtosis).toFixed(2)}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <div
          className="p-3 rounded border"
          style={{ backgroundColor: Theme.ui.surface, borderColor: Theme.ui.border }}
        >
          <span
            className="text-[10px] uppercase tracking-wider block mb-2"
            style={{ color: Theme.ui.text.muted }}
          >
            {LABELS.drillDown.bootstrap}
          </span>
          <div className="text-sm font-mono" style={{ color: Theme.ui.text.secondary }}>
            {bootstrapSupport}
          </div>
        </div>
        <div
          className="p-3 rounded border"
          style={{ backgroundColor: Theme.ui.surface, borderColor: Theme.ui.border }}
        >
          <span
            className="text-[10px] uppercase tracking-wider block mb-2"
            style={{ color: Theme.ui.text.muted }}
          >
            {LABELS.drillDown.interactions}
          </span>
          <div className="text-sm font-mono" style={{ color: Theme.ui.text.secondary }}>
            {interactionStrength}
          </div>
        </div>
        <div
          className="p-3 rounded border"
          style={{ backgroundColor: Theme.ui.surface, borderColor: Theme.ui.border }}
        >
          <span
            className="text-[10px] uppercase tracking-wider block mb-2"
            style={{ color: Theme.ui.text.muted }}
          >
            {LABELS.drillDown.surrogate}
          </span>
          <div className="text-[11px] font-mono" style={{ color: Theme.ui.text.secondary }}>
            {surrogateBounds}
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      {rec.action !== "keep" && (
        <div className="flex items-center gap-3">
          <button
            onClick={() => onApply?.(rec)}
            className="flex items-center gap-2 px-4 py-2 rounded text-xs font-bold uppercase tracking-wider transition-all hover:brightness-125"
            style={{
              backgroundColor: Theme.palette.vividGreen + "30",
              color: Theme.palette.vividGreen,
              border: `1px solid ${Theme.palette.vividGreen}60`,
            }}
          >
            <CheckCircle size={16} />
            {LABELS.drillDown.apply}
          </button>
          <button
            onClick={() => onIgnore?.(rec)}
            className="px-4 py-2 rounded text-xs font-bold uppercase tracking-wider transition-all hover:brightness-125"
            style={{
              backgroundColor: Theme.ui.background,
              color: Theme.ui.text.muted,
              border: `1px solid ${Theme.ui.border}`,
            }}
          >
            {LABELS.drillDown.dismiss}
          </button>
        </div>
      )}
    </div>
  );
};

/**
 * PatchPreviewModal - Full patch preview (existing, styled)
 */
const PatchPreviewModal = ({ patch, onApplyAll, onClose }) => {
  if (!patch || Object.keys(patch).length === 0) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ backgroundColor: "rgba(0,0,0,0.85)" }}
      onClick={onClose}
    >
      <div
        className="max-w-3xl w-full mx-4 rounded-xl p-6 border shadow-2xl"
        style={{ backgroundColor: Theme.ui.surface, borderColor: Theme.ui.border }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h3
            className="text-sm font-black uppercase tracking-widest"
            style={{ color: Theme.ui.text.primary }}
          >
            Preview do Patch Completo
          </h3>
          <button onClick={onClose} className="p-2 rounded hover:bg-white/10">
            <X size={20} style={{ color: Theme.ui.text.muted }} />
          </button>
        </div>

        <pre
          className="text-xs font-mono p-4 rounded-lg overflow-auto max-h-[400px] custom-scrollbar leading-relaxed"
          style={{ backgroundColor: Theme.ui.background, color: Theme.ui.text.secondary }}
        >
          {JSON.stringify(patch, null, 2)}
        </pre>

        <div className="mt-6 flex justify-end gap-3">
          <button
            onClick={onApplyAll}
            className="flex items-center gap-2 px-4 py-2 rounded text-xs font-bold uppercase tracking-wider transition-colors hover:brightness-125"
            style={{
              backgroundColor: Theme.semantic.success + "30",
              color: Theme.semantic.success,
              border: `1px solid ${Theme.semantic.success}60`,
            }}
          >
            <CheckCircle size={16} />
            Aplicar Tudo
          </button>
          <button
            className="px-4 py-2 rounded text-xs font-bold uppercase tracking-wider transition-colors hover:brightness-125"
            style={{
              backgroundColor: Theme.ui.background,
              color: Theme.ui.text.muted,
              border: `1px solid ${Theme.ui.border}`,
            }}
            onClick={onClose}
          >
            Fechar
          </button>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// MAIN COMPONENT - SearchSpaceAdvisorCard (SOTA Implementation)
// ============================================================================

/**
 * SearchSpaceAdvisorCard - SOTA Hyperparameter Decision Dashboard
 *
 * Architecture:
 * - Grid-based responsive layout (not table-based) for visual flexibility
 * - Progressive disclosure: summary row → drill-down panel
 * - Visual-first: strip plots + topology rulers replace text-heavy cells
 * - Color-coded actions with verb-first UX
 */
export const SearchSpaceAdvisorCard = ({ advice, _searchSpace, _trials }) => {
  const [patchData, setPatchData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [refreshingAdvice, setRefreshingAdvice] = useState(false);
  const [adviceOverride, setAdviceOverride] = useState(null);
  const [expandedRow, setExpandedRow] = useState(null);
  const [applyStatus, setApplyStatus] = useState(null);
  const [ignoredParams, setIgnoredParams] = useState(() => new Set());
  const [appliedParams, setAppliedParams] = useState(() => new Set());

  const effectiveAdvice = useMemo(() => {
    if (!adviceOverride) return advice || {};
    if (!advice) return adviceOverride;

    const overrideMeta = adviceOverride?.metadata || {};
    const incomingMeta = advice?.metadata || {};
    const overrideTrials = Number(overrideMeta.n_completed_trials ?? -1);
    const incomingTrials = Number(incomingMeta.n_completed_trials ?? -1);
    const overrideVersion = String(overrideMeta.advisor_version || "");
    const incomingVersion = String(incomingMeta.advisor_version || "");
    const overrideCompute = Number(overrideMeta.compute_time_ms ?? 0);
    const incomingCompute = Number(incomingMeta.compute_time_ms ?? 0);

    if (incomingTrials > overrideTrials) return advice;
    if (incomingMeta.forced_recompute === true && incomingTrials >= overrideTrials) return advice;
    if (
      incomingVersion === overrideVersion &&
      incomingTrials === overrideTrials &&
      incomingMeta.cache_hit === false &&
      incomingCompute >= overrideCompute
    ) {
      return advice;
    }
    return adviceOverride;
  }, [adviceOverride, advice]);
  const rawRecommendations = useMemo(
    () => effectiveAdvice?.recommendations || [],
    [effectiveAdvice]
  );
  const metadata = useMemo(() => effectiveAdvice?.metadata || {}, [effectiveAdvice]);
  const reliability = useMemo(() => metadata?.reliability_summary || {}, [metadata]);
  const selfAudit = useMemo(() => metadata?.self_audit || {}, [metadata]);
  const insufficient = metadata.insufficient_evidence === true;

  useEffect(() => {
    const ignored = Array.isArray(metadata.ignored_params) ? metadata.ignored_params : [];
    const applied = Array.isArray(metadata.applied_params) ? metadata.applied_params : [];
    setIgnoredParams(new Set(ignored));
    setAppliedParams(new Set(applied));
  }, [metadata]);

  const hiddenParams = useMemo(() => {
    const merged = new Set();
    ignoredParams.forEach((name) => merged.add(name));
    appliedParams.forEach((name) => merged.add(name));
    return merged;
  }, [ignoredParams, appliedParams]);

  const recommendations = useMemo(
    () => rawRecommendations.filter((rec) => !hiddenParams.has(rec.param_name)),
    [rawRecommendations, hiddenParams]
  );

  const handlePreviewPatch = useCallback(async () => {
    if (loading) return;
    setLoading(true);
    try {
      const resp = await fetch("/api/hpo/search-space-advice/patch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ recommendations }),
      });
      const data = await resp.json();
      setPatchData(data.patch || {});
    } catch {
      setPatchData({ error: "Failed to generate patch" });
    } finally {
      setLoading(false);
    }
  }, [recommendations, loading]);

  const handleRefreshAdvice = useCallback(async () => {
    if (loading || refreshingAdvice) return;
    setRefreshingAdvice(true);
    setApplyStatus(null);
    try {
      const resp = await fetch("/api/hpo/search-space-advice?refresh=1", {
        method: "GET",
        cache: "no-store",
      });
      const data = await resp.json();
      if (!resp.ok || !data || typeof data !== "object") {
        throw new Error(data?.detail || data?.error || "Falha ao recalcular recomendações.");
      }
      setAdviceOverride(data);
      setExpandedRow(null);
      setApplyStatus({
        type: "success",
        message: "Recomendacoes recalculadas com refresh forcado.",
      });
    } catch (error) {
      setApplyStatus({
        type: "error",
        message: error?.message || "Falha ao recalcular recomendações.",
      });
    } finally {
      setRefreshingAdvice(false);
    }
  }, [loading, refreshingAdvice]);

  const toggleRow = useCallback((paramName) => {
    setExpandedRow((prev) => (prev === paramName ? null : paramName));
  }, []);

  const applyRecommendations = useCallback(
    async (items) => {
      if (loading) return;
      setLoading(true);
      setApplyStatus(null);
      try {
        const resp = await fetch("/api/hpo/search-space-advice/apply", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ recommendations: items }),
        });
        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data?.detail || data?.error || "Falha ao aplicar ajustes");
        }
        const applied = Array.isArray(data.applied_params) ? data.applied_params : [];
        setAppliedParams((prev) => {
          const next = new Set(prev);
          applied.forEach((name) => next.add(name));
          return next;
        });
        setApplyStatus({
          type: "success",
          message: "Ajustes aplicados no YAML e no espaco de busca.",
        });
      } catch (error) {
        setApplyStatus({
          type: "error",
          message: error?.message || "Falha ao aplicar ajustes.",
        });
      } finally {
        setLoading(false);
      }
    },
    [loading]
  );

  const handleApply = useCallback(
    (rec) => {
      if (!rec) return;
      applyRecommendations([rec]);
    },
    [applyRecommendations]
  );

  const handleApplyAll = useCallback(() => {
    if (!recommendations.length) return;
    setPatchData(null);
    applyRecommendations(recommendations);
  }, [applyRecommendations, recommendations]);

  const handleIgnore = useCallback(
    async (rec) => {
      if (!rec || loading) return;
      setLoading(true);
      setApplyStatus(null);
      try {
        const resp = await fetch("/api/hpo/search-space-advice/ignore", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ param_names: [rec.param_name] }),
        });
        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data?.detail || data?.error || "Falha ao ignorar ajuste");
        }
        setIgnoredParams((prev) => {
          const next = new Set(prev);
          next.add(rec.param_name);
          return next;
        });
        setExpandedRow(null);
        setApplyStatus({ type: "success", message: "Ajuste ignorado no dashboard." });
      } catch (error) {
        setApplyStatus({
          type: "error",
          message: error?.message || "Falha ao ignorar ajuste.",
        });
      } finally {
        setLoading(false);
      }
    },
    [loading]
  );

  const helpText = ChartRegistry.get("search_space_advisor", {
    title: LABELS.title,
    tech: "Análise estatística de trials completos, importância de parâmetros (SHAP-like) e densidade top-k para recomendações inteligentes de espaço de busca. SOTA v2.4 com visualizações de topologia e drill-down.",
    simple: "Sugestões visuais e acionáveis para otimizar os hiperparâmetros.",
  });

  const actionSummary = useMemo(() => {
    const counts = {};
    for (const r of recommendations) {
      const a = r.action || "keep";
      counts[a] = (counts[a] || 0) + 1;
    }
    return counts;
  }, [recommendations]);

  const hasChanges = recommendations.some((r) => r.action !== "keep");

  return (
    <Card
      title={LABELS.title}
      icon={Sliders}
      className="h-full"
      helpText={helpText}
      headerRight={
        <div className="flex items-center gap-2">
          <button
            className="flex items-center gap-2 px-3 py-1.5 rounded text-xs font-bold uppercase tracking-wider transition-all hover:brightness-125 disabled:opacity-50"
            style={{
              backgroundColor: Theme.palette.neonBlue + "20",
              color: Theme.palette.neonBlue,
              border: `1px solid ${Theme.palette.neonBlue}50`,
            }}
            onClick={handleRefreshAdvice}
            disabled={loading || refreshingAdvice}
            title="Forca o backend a recalcular as recomendacoes ignorando cache."
          >
            {refreshingAdvice ? "..." : "Recalcular Agora"}
          </button>
          {hasChanges && (
            <button
              className="flex items-center gap-2 px-3 py-1.5 rounded text-xs font-bold uppercase tracking-wider transition-all hover:brightness-125 disabled:opacity-50"
              style={{
                backgroundColor: Theme.palette.hotOrange + "25",
                color: Theme.palette.hotOrange,
                border: `1px solid ${Theme.palette.hotOrange}50`,
              }}
              onClick={handlePreviewPatch}
              disabled={loading || refreshingAdvice}
            >
              {loading ? "..." : "Preview Patch"}
            </button>
          )}
        </div>
      }
    >
      {patchData && (
        <PatchPreviewModal
          patch={patchData}
          onApplyAll={handleApplyAll}
          onClose={() => setPatchData(null)}
        />
      )}

      {insufficient ? (
        <div className="flex flex-col items-center justify-center h-full gap-4 py-12">
          <AlertTriangle size={36} style={{ color: Theme.semantic.warning }} />
          <p className="text-sm text-center" style={{ color: Theme.ui.text.muted }}>
            Evidência insuficiente ({metadata.n_completed_trials || 0} trials completos).
            <br />
            Mínimo necessário: {metadata.min_trials_required || 3} trials.
          </p>
        </div>
      ) : (
        <div className="flex flex-col h-full min-h-0">
          {/* Metadata Header - LARGER fonts */}
          <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
            <div className="flex items-center gap-3">
              <span className="text-xs font-mono" style={{ color: Theme.ui.text.muted }}>
                {metadata.n_completed_trials || 0} trials
                <span className="mx-2">|</span>
                top-{metadata.n_top_k || 0}
                <span className="mx-2">|</span>
                {metadata.compute_time_ms || 0}ms
                {typeof reliability.mean_confidence_score === "number" && (
                  <>
                    <span className="mx-2">|</span>
                    conf {Math.round(reliability.mean_confidence_score * 100)}%
                  </>
                )}
                {typeof reliability.validation_pass_wilson_lb === "number" && (
                  <>
                    <span className="mx-2">|</span>
                    val-LB {Math.round(reliability.validation_pass_wilson_lb * 100)}%
                  </>
                )}
                {typeof reliability.actionable === "number" && (
                  <>
                    <span className="mx-2">|</span>
                    acoes {reliability.actionable}/{reliability.total || 0}
                  </>
                )}
                {typeof selfAudit.villains_count === "number" && (
                  <>
                    <span className="mx-2">|</span>
                    viloes {selfAudit.villains_count}
                  </>
                )}
                {metadata.cache_hit && <span style={{ color: Theme.palette.mint }}> (cache)</span>}
                {metadata.forced_recompute && (
                  <span style={{ color: Theme.palette.neonBlue }}> (refresh)</span>
                )}
              </span>
            </div>
            <div className="flex items-center gap-2">
              {Object.entries(actionSummary).map(([action, count]) => (
                <span
                  key={action}
                  className="text-[10px] font-mono px-2 py-1 rounded"
                  style={{
                    color: ACTION_COLORS[action] || Theme.ui.text.muted,
                    backgroundColor: (ACTION_COLORS[action] || Theme.ui.text.muted) + "15",
                  }}
                >
                  {ACTION_LABELS[action]?.split(" ")[0] || action}: {count}
                </span>
              ))}
            </div>
          </div>

          <div className="mb-3 grid gap-2 md:grid-cols-4">
            <div
              className="px-3 py-2 rounded text-xs font-mono"
              style={{
                backgroundColor: Theme.ui.surface,
                border: `1px solid ${Theme.ui.border}`,
                color: Theme.ui.text.secondary,
              }}
            >
              valid-LB{" "}
              {typeof reliability.validation_pass_wilson_lb === "number"
                ? `${Math.round(reliability.validation_pass_wilson_lb * 100)}%`
                : "-"}
            </div>
            <div
              className="px-3 py-2 rounded text-xs font-mono"
              style={{
                backgroundColor: Theme.ui.surface,
                border: `1px solid ${Theme.ui.border}`,
                color: Theme.ui.text.secondary,
              }}
            >
              high-conf-LB{" "}
              {typeof reliability.high_confidence_wilson_lb === "number"
                ? `${Math.round(reliability.high_confidence_wilson_lb * 100)}%`
                : "-"}
            </div>
            <div
              className="px-3 py-2 rounded text-xs font-mono"
              style={{
                backgroundColor: Theme.ui.surface,
                border: `1px solid ${Theme.ui.border}`,
                color: Theme.ui.text.secondary,
              }}
            >
              self-audit-LB{" "}
              {typeof selfAudit.directional_hit_rate_wilson_lb === "number"
                ? `${Math.round(selfAudit.directional_hit_rate_wilson_lb * 100)}%`
                : "-"}
            </div>
            <div
              className="px-3 py-2 rounded text-xs font-mono"
              style={{
                backgroundColor: Theme.ui.surface,
                border: `1px solid ${Theme.ui.border}`,
                color: Theme.ui.text.secondary,
              }}
            >
              bloqueios{" "}
              {typeof selfAudit.blocked_actions_current === "number"
                ? selfAudit.blocked_actions_current
                : 0}
            </div>
          </div>

          {applyStatus && (
            <div
              className="mb-3 text-xs font-mono px-3 py-2 rounded"
              style={{
                color:
                  applyStatus.type === "success" ? Theme.semantic.success : Theme.semantic.danger,
                backgroundColor:
                  applyStatus.type === "success"
                    ? Theme.semantic.success + "15"
                    : Theme.semantic.danger + "15",
                border: `1px solid ${Theme.ui.border}`,
              }}
            >
              {applyStatus.message}
            </div>
          )}

          <WithData
            when={recommendations.length > 0}
            empty="Nenhuma recomendação disponível"
            emptyClassName="text-zinc-500"
          >
            <div className="overflow-auto custom-scrollbar flex-1 min-h-0">
              {/* Grid Header - INCREASED font and spacing */}
              <div
                className="grid gap-4 px-4 py-4 text-sm font-bold uppercase tracking-wider sticky top-0 z-10"
                style={{
                  backgroundColor: Theme.ui.surface,
                  borderBottom: `2px solid ${Theme.ui.border}`,
                  color: Theme.ui.text.primary,
                  gridTemplateColumns:
                    "minmax(160px, 1fr) minmax(80px, 0.6fr) minmax(110px, 0.9fr) minmax(140px, 1fr) minmax(100px, 0.8fr) minmax(80px, 0.6fr) minmax(110px, 0.9fr)",
                }}
              >
                <div>{LABELS.columns.parameter}</div>
                <ColumnHint title={LABELS.columns.stats} content={LABELS.hints.stats}>
                  {LABELS.columns.stats}
                </ColumnHint>
                <ColumnHint title="Densidade de Tentativas" content={LABELS.hints.density}>
                  {LABELS.columns.density}
                </ColumnHint>
                <ColumnHint title="Topologia do Espaço" content={LABELS.hints.topology}>
                  {LABELS.columns.topology}
                </ColumnHint>
                <ColumnHint title={LABELS.columns.importance} content={LABELS.hints.importance}>
                  {LABELS.columns.importance}
                </ColumnHint>
                <div>{LABELS.columns.confidence}</div>
                <div>{LABELS.columns.action}</div>
              </div>

              {/* Grid Rows - EXPANDED with larger fonts */}
              <div className="flex flex-col">
                {recommendations.map((rec) => {
                  const isExpanded = expandedRow === rec.param_name;
                  const paramType = getParamType(rec.current_space);
                  const stats = rec.attempts_summary?.stats;

                  return (
                    <div key={rec.param_name} className="flex flex-col">
                      {/* Main Row - INCREASED font and spacing */}
                      <div
                        onClick={() => toggleRow(rec.param_name)}
                        className={`
                          grid gap-4 px-4 py-4 text-base cursor-pointer transition-all
                          hover:bg-white/5
                          ${isExpanded ? "bg-white/5" : ""}
                        `}
                        style={{
                          borderBottom: `1px solid ${Theme.ui.border}50`,
                          gridTemplateColumns:
                            "minmax(160px, 1fr) minmax(80px, 0.6fr) minmax(110px, 0.9fr) minmax(140px, 1fr) minmax(100px, 0.8fr) minmax(80px, 0.6fr) minmax(110px, 0.9fr)",
                        }}
                      >
                        {/* Parameter Column */}
                        <div className="flex flex-col gap-2">
                          <div
                            className="text-base font-semibold"
                            style={{ color: Theme.palette.cyberYellow }}
                          >
                            {renderParamWithHints(rec.param_name)}
                          </div>
                          <div className="flex items-center gap-2">
                            <ParameterBadge type={paramType} />
                            {rec.current_space?.log && <ParameterBadge type="log" />}
                          </div>
                          {rec.current_space?.low != null && rec.current_space?.high != null && (
                            <div
                              className="text-[11px] text-zinc-400 flex flex-wrap gap-3"
                              style={{ fontWeight: 400 }}
                            >
                              <span className="font-mono">
                                min {fmtSmart(rec.current_space.low)}
                              </span>
                              <span className="font-mono">
                                max {fmtSmart(rec.current_space.high)}
                              </span>
                            </div>
                          )}
                        </div>

                        {/* Stats Column */}
                        <div className="flex flex-col gap-1.5 text-sm font-mono">
                          <div style={{ color: Theme.ui.text.muted }}>
                            <span style={{ opacity: 0.6 }}>{LABELS.stats.n}=</span>
                            <span style={{ color: Theme.ui.text.secondary }}>
                              {rec.attempts_summary?.count || 0}
                            </span>
                          </div>
                          {stats?.mean != null && (
                            <div style={{ color: Theme.ui.text.muted }}>
                              <span style={{ opacity: 0.6 }}>{LABELS.stats.mean}=</span>
                              <span style={{ color: Theme.ui.text.secondary }}>
                                {Number(stats.mean).toPrecision(3)}
                              </span>
                            </div>
                          )}
                          {stats?.std != null && (
                            <div style={{ color: Theme.ui.text.muted }}>
                              <span style={{ opacity: 0.6 }}>{LABELS.stats.std}=</span>
                              <span style={{ color: Theme.ui.text.secondary }}>
                                {Number(stats.std).toPrecision(2)}
                              </span>
                            </div>
                          )}
                        </div>

                        {/* Trial Density Strip Plot */}
                        <div className="flex items-center">
                          <TrialDensityStrip
                            attempts={rec.attempts_summary}
                            bestRegion={rec.best_region}
                            space={rec.current_space}
                          />
                        </div>

                        {/* Space Topology Ruler */}
                        <div className="flex items-center">
                          <SpaceTopologyRuler
                            current={rec.current_space}
                            proposal={rec.recommendation}
                            action={rec.action}
                          />
                        </div>

                        {/* Importance Bar */}
                        <div className="flex items-center">
                          <ImportanceBar value={rec.importance} />
                        </div>

                        {/* Confidence Badge */}
                        <div className="flex items-center">
                          <ConfidenceBadge level={rec.confidence} />
                        </div>

                        {/* Smart Action Button - Shows delta directly */}
                        <div className="flex items-center justify-between gap-3">
                          <SmartActionButton
                            action={rec.action}
                            recommendation={rec.recommendation}
                            onClick={rec.action !== "keep" ? () => handleApply(rec) : undefined}
                          />
                          <ArrowDown
                            size={18}
                            className={`transition-transform ${isExpanded ? "rotate-180" : ""}`}
                            style={{ color: Theme.ui.text.muted }}
                          />
                        </div>
                      </div>

                      {/* Drill-down Panel */}
                      {isExpanded && (
                        <DrillDownPanel
                          rec={rec}
                          onClose={() => setExpandedRow(null)}
                          onApply={handleApply}
                          onIgnore={handleIgnore}
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </WithData>
        </div>
      )}
    </Card>
  );
};
