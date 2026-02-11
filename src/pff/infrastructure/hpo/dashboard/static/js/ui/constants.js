/**
 * Dashboard-wide named constants extracted from magic numbers.
 * Import from here instead of hardcoding values in components.
 */

/** Fallback total trials when backend has not yet pushed config. Should match config/hpo/optimization.yaml defaults.n_trials. */
export const DEFAULT_TOTAL_TRIALS = 50;

/** Milliseconds before a RUNNING study is considered stale (no updates). */
export const RUNNING_STALENESS_MS = 30_000;
