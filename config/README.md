# Config Layout

Canonical configuration files now live in scoped subdirectories. Prefer these
paths (or the constants in `pff.shared.core.config`) instead of root-level files.

- `models/`
  - `ensemble.yaml` – legacy ensemble/meta-learner (compat only; DSLFM+PC2 is primary)
  - `autofeeding.yaml` – rule extraction/autofeeding pipeline
  - `kg.yaml` – KG pipeline + PC2 settings
  - `dslfm.yaml` – DSLFM-KGC configuration (adaptive batch size + PC2 fusion; training includes `num_workers`, `pin_memory`, `dataloader_prefetch_factor`, `dataloader_persistent_workers`, `cuda_cache_flush_steps`, `use_faiss_eval`, `train_heartbeat_interval_s`, `score_all_tails_chunk_size`, `use_compile`, `optimizer_fused`, `optimizer_foreach`; compile settings live under `compile`)
- `rule_filter` section inside `kg.yaml` – legacy rule filter defaults (keep only for backward compatibility)
- `hpo/`
  - `adaptive_learning.yaml` – threshold tuning and adaptive rules
  - `optimization.yaml` – HPO memory, storage (Optuna), sampler/pruner, bounds, and live dashboard toggles
- `infra/`
  - `api_hosts.yaml` (+ `.example`) – service endpoints
  - `postgres.yaml` – DB pool/retry/SSL settings
  - `sequences.yaml` – declarative workflow sequences (dispatcher)
  - `validator.yaml` – business-service thresholds/messages
  - `performance.yaml` – performance backends + I/O streaming thresholds + parquet-first ingest cache
- `observability/`
  - `explainability.yaml` – SHAP/interpretability settings
  - `training_metrics.yaml` – metrics logging toggles
  - `metrics_improvement.json` – target metrics/monitoring hints

Usage: import paths from `pff.shared.core.config` instead of hardcoding strings to
avoid breakage when reorganizing configs.

## HPO optimization.yaml highlights

- `storage`: Optuna backend + pooling knobs (RDBStorage) + `grpc_proxy` host/port.
- `sampler`/`pruner`: TPE multivariado + Hyperband/ASHA + PatientPruner (opcional `sampler.type: auto` com optunahub).
- `multi_objective`: ativa MOTPE/NSGA2 com `directions` e `secondary_metric`.
- `live_plots`: controla o dashboard HTML (Plotly) e seus intervalos; quando

  `enable_optuna_dashboard=true`, os PNGs ao vivo deixam de ser gerados. Use
  `dashboard_debug_mode=true` para manter o dashboard ativo sem HPO rodando. Use
  `dashboard_data_path` para forcar o local persistente do dashboard JSON.
