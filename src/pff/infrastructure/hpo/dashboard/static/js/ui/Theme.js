/**
 * SOTA Visual Architecture Theme Registry
 * Implements "Chromatic Engineering" principles for High-Density HPO Dashboards.
 *
 * Design Principles:
 * 1. Dark Mode SOTA: #121212 Background (OLED Smearing mitigation)
 * 2. Neon Palette: high-distinction categorical colors for N > 5 datasets
 * 3. Semantic Mappings: Standardized Success/Warning/Error states
 */

const OKABE_ITO_LIGHT_PALETTE = {
  blue: "#0072B2", // Strong Blue
  green: "#009E73", // Gluish Green (Teal-like)
  orange: "#D55E00", // Vermillion
  yellow: "#E69F00", // Orange-Yellow (Ocre)
  red: "#D92121", // Vivid Red
  purple: "#882255", // Reddish Purple
  cyan: "#007788", // Dark Cyan (Teal)
  grey: "#666666", // Neutral
  black: "#000000",
  white: "#FFFFFF",
};

const NEON_DARK_PALETTE = {
  neonBlue: "#4363d8",
  vividGreen: "#3cb44b",
  hotOrange: "#D4692A", // Desaturated 15%
  cyberYellow: "#E5C558", // Warm amber/gold, less saturated (was #ffe119)
  apricot: "#F6B26B", // Warm accent for labels/tooltips
  magenta: "#f032e6",
  cyan: "#5BA8B0", // Teal-cyan, less neon (was #42d4f4)
  mint: "#7EE7C2", // Soft mint for "Para Leigos" emphasis
  purple: "#911eb4",
  lime: "#bfef45",
  teal: "#469990",
  red: "#e6194B",
  grey: "#9A9A9A", // Warmer gray
  white: "#E8E8E8", // Off-white, less harsh (was #ffffff)
};

const SEMANTIC_TOKENS = {
  // Dynamic CSS Variable references for runtime switching
  primary: "var(--viz-palette-1-blue)",
  success: "var(--viz-palette-2-green)",
  warning: "var(--viz-palette-3-orange)",
  highlight: "var(--viz-palette-4-yellow)",
  error: "var(--viz-palette-5-red)",
  purple: "var(--viz-palette-6-purple)",
  info: "var(--viz-palette-7-cyan)",

  // UI Structural Tokens
  background: "var(--viz-bg-canvas)",
  surface: "var(--viz-bg-surface)",
  surfaceHighlight: "var(--viz-bg-elevated)",
  text: {
    primary: "var(--viz-text-primary)",
    secondary: "var(--viz-text-secondary)",
    muted: "var(--viz-text-muted)",
  },
  border: "var(--viz-border)",
  grid: "var(--viz-axis-line)",

  // Chart Specific Semantic Tokens
  chart: {
    loss: "var(--viz-palette-3-orange)", // Orange (Dark) / Vermillion (Light)
    metric: "var(--viz-palette-1-blue)", // Blue (Dark) / Strong Blue (Light)
    stability: "var(--viz-palette-7-cyan)", // Cyan (Dark) / Teal (Light)
    incumbent: "var(--viz-palette-4-yellow)", // Yellow (Dark) / Ocre (Light)
    movingAverage: "var(--viz-palette-2-green)", // Green (Dark) / Teal-Green (Light)
    gradNorm: "var(--viz-palette-4-yellow)", // Gradient norm / amber highlight
    recon: "var(--viz-palette-5-red)", // ELBO reconstruction loss
    klDiv: "var(--viz-palette-3-orange)", // ELBO KL divergence
  },

  // Hardware monitoring chart colors
  hardware: {
    cpu: "var(--viz-palette-3-orange)",
    gpu: "var(--viz-palette-5-red)",
    vram: "var(--viz-palette-6-purple)",
    ram: "var(--viz-palette-7-cyan)",
  },
};

/**
 * Expose theme for dashboard usage.
 */
export const Theme = Object.freeze({
  // Keep raw palettes for reference or static usage if needed
  palettes: {
    dark: NEON_DARK_PALETTE,
    light: OKABE_ITO_LIGHT_PALETTE,
  },

  // Components should use these semantic tokens
  semantic: SEMANTIC_TOKENS,
  ui: SEMANTIC_TOKENS, // Alias for backward compatibility with my previous refactor
  palette: NEON_DARK_PALETTE, // Fallback/Legacy direct access (try to avoid using this directly in new components)

  gradients: {
    // Updated to use CSS variables where possible or fallbacks
    magma: [
      { offset: "0%", color: "var(--viz-palette-3-orange)", opacity: 0.8 },
      { offset: "100%", color: "var(--viz-palette-4-yellow)", opacity: 0.1 },
    ],
    ocean: [
      { offset: "0%", color: "var(--viz-palette-1-blue)", opacity: 0.6 },
      { offset: "100%", color: "var(--viz-palette-7-cyan)", opacity: 0.0 },
    ],
    chartArea: {
      primarySubtle: [
        { offset: "0%", color: "currentColor", opacity: 0.24 },
        { offset: "100%", color: "currentColor", opacity: 0.02 },
      ],
      primaryReadable: [
        { offset: "0%", color: "currentColor", opacity: 0.36 },
        { offset: "100%", color: "currentColor", opacity: 0.09 },
      ],
      objectiveStrong: [
        { offset: "0%", color: "currentColor", opacity: 0.58 },
        { offset: "100%", color: "currentColor", opacity: 0.16 },
      ],
    },
  },
});
