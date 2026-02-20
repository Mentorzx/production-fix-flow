# Search Space Advisor

## How it works

The `SearchSpaceAdvisor` analyzes completed HPO trials and produces per-parameter
recommendations for adjusting the search space. It combines three signals:

1. **Empirical evidence (trials)** — selects the top-k trials (top 25% by default),
   computes statistics per parameter, and detects edge concentration (values clustering
   near bounds), tight concentration (low CV), or categorical dominance.

2. **Parameter importance (Optuna fANOVA)** — weights recommendations by importance.
   High-importance params near a bound trigger aggressive expansion; low-importance
   params may be fixed to save budget.

3. **Heuristics** —
   - Detects log-scale candidates by name (`lr`, `weight_decay`, etc.) or range ratio (>100x),
     and suggests `log_uniform` distribution when appropriate.
   - With low trial counts, uses dataset profile (`n_entities`, `n_relations`, `n_triples`, `density`)
     to bootstrap initial recommendations for `embedding_dim`, negative sampling, and regularization.

4. **Reliability guardrails (v2.3+)** —
   - Normalizes direction labels (`MAXIMIZE`, `StudyDirection.MAXIMIZE`, etc.) to avoid top-k inversion.
   - Respects explicit `log: false` and avoids log-space transforms when `low <= 0`.
   - Validates each recommendation before exposing it (`validation.passed`, `blocked_reason`).
   - Unsafe actions are downgraded to `keep` with `blocked_action` metadata.
   - Categorical recommendations are canonicalized to prevent `keep/remove` overlap.
   - Directional expansion (`expand_upper/lower`) is gated by monotonic trend support
     (Spearman sign agreement) to avoid edge-only false positives.
   - Periodic self-audit (every 10 completed trials) backtests directional actions on
     prefix/suffix slices and blocks low-reliability patterns only when both
     hit-rate `< 50%` and Wilson lower bound is below threshold.
   - Self-audit runs in lightweight mode (`enable_surrogate=false`, `enable_interactions=false`,
     `disable_internal_importances=true`) to cut overhead while preserving directional checks.
   - Adaptive performance mode can auto-disable expensive components (`interactions`, then
     `surrogate`, then internal importances) only when latency is high **and**
     validation reliability remains above a configurable Wilson-LB threshold.
   - Adaptive mode uses a short cooldown window to avoid oscillation (fast on/off flapping)
     between successive advice calls.

### Actions produced

| Action | Trigger | Effect |
|---|---|---|
| `expand_upper` | Top-k q90 near upper bound | Increase `high` by 50% of range |
| `expand_lower` | Top-k q10 near lower bound | Decrease `low` |
| `narrow` | Top-k tightly concentrated (CV < 0.15) | Shrink to [q10, q90] of top-k |
| `fix` | Low importance (< 5%) | Fix at top-k median |
| `reduce_categories` | One category dominates top-k (>60%) | Keep top 2+ categories |
| `change_distribution` | Name/range suggests log-scale | Switch to log-uniform |
| `keep` | No signal strong enough | No change |

### Cache

Results now use two layers:

- **L1 (memory)** via `CacheManager` for low-latency reads.
- **L2 (PostgreSQL)** via `HpoPostgresStore` to persist advice across restarts.

Cache identity uses:
`study_name + dataset_fingerprint + direction + advisor_version + last_trial + search_space_hash + objective_schema_hash`.

Metadata includes `cache_layer_hit: l1|l2|none`.

### Rust acceleration

- The advisor uses optional Rust acceleration for large Spearman computations (`fast_spearman_corr`)
  when available in `pff_rust`; fallback is deterministic pure-Python implementation.
- The Rust path is gated by vector size (`rust_spearman_min_len`, default: 512) to avoid
  Python↔Rust conversion overhead on small/medium samples.

### Multi-objective

For multi-objective studies, ranking is no longer based only on `values[0]`.
The advisor uses a hybrid projection:

- Pareto-aware ranking (non-dominated sorting),
- Hypervolume contribution (when objective count <= 3 and front size is tractable),
- Scalarized fallback when full Pareto/HV is not applicable.

Metadata includes `multiobjective_mode`, `objective_directions`, `objective_count`, and `hypervolume`.

## API

### GET `/api/hpo/search-space-advice?study_id=...`

Returns JSON with:
- `recommendations[]` — per-parameter analysis and suggestion
- `metadata` — study info, cache status, compute time
- `recommendations[].validation` — validation checks for each action
- `recommendations[].confidence_score` — numeric confidence score in `[0, 1]`
- `metadata.validation_flags` — global validation summary
- `metadata.reliability_summary` — aggregate quality counters for current advice
  (`validation_pass_wilson_lb` and `high_confidence_wilson_lb` use conservative Wilson lower bounds)
- `metadata.self_audit` — periodic directional backtest summary (`villains`, hit-rate LBs, and blocked actions)
- `metadata.importance_source` / `metadata.importance_quality` — external/internal/blended importance diagnostics
- `metadata.search_space_coverage_ratio` / `metadata.missing_params` / `metadata.distribution_conflicts`
- `metadata.acceleration` — runtime toggles (`surrogate_enabled`, `interactions_enabled`,
  `internal_importances_disabled`, `rust_spearman_available`, `rust_spearman_min_len`)
- `metadata.adaptive_performance` — adaptive controller diagnostics
  (`decision`, `degraded_count_before/after`, thresholds, previous latency/reliability)
- `recommendations[].scope` — `global` or `conditional` (support-aware recommendation scope)

Tip: add `?refresh=1` to force recomputation and ignore cached advice payload.
The dashboard card now includes a **Recalcular Agora** button that calls this refresh endpoint.

### POST `/api/hpo/search-space-advice/patch`

Body: `{"recommendations": [...]}` (from GET response).
Returns: `{"patch": {...}, "n_changes": N}` — a preview of config changes. Does NOT apply automatically.

Invalid recommendations (`validation.passed=false`) are excluded from patch generation.

## Reliability audit (offline)

Run:

```bash
poetry run python scripts/benchmarks/search_space_advisor_audit.py --min-prefix 8
```

The report includes:
- `backtest.directional_signals_hit_rate`
- `backtest.confidence_success_rate`
- `backtest.directional_breakdown` (per `param+action`)
- `backtest.villains` (groups with low hit-rate and enough evidence)

## How to apply a patch

1. Open the "Previsao de Estudo" tab in the dashboard.
2. Click "Preview Patch" to see the suggested config changes.
3. Copy the patch JSON and manually update your `config/hpo/optimization.yaml`
   search space section accordingly.
4. Re-run the study with the updated config.

The advisor never modifies config files automatically.

## Limitations

- Recommendations require a minimum number of completed trials (default: 5, from `config/hpo/optimization.yaml`) before
  producing empirical suggestions. With fewer than 10 trials, confidence is always "low" or "medium".
- Conditional dependencies are modeled by support-based scope (`global` vs `conditional`), but not yet by a full hierarchical conditional graph.
- Hypervolume is computed for up to 3 objectives in the advisor path; higher-dimensional fronts use scalarized fallback.
