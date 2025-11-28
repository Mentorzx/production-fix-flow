# Config Layout

Canonical configuration files now live in scoped subdirectories. Prefer these
paths (or the constants in `pff.config`) instead of root-level files.

- `models/`
  - `ensemble.yaml` – ensemble/meta-learner + balancing settings
  - `autofeeding.yaml` – rule extraction/autofeeding pipeline
  - `oov.yaml` – OOV weighting and thresholds
  - `strategies/balanced_training_strategy.json`
  - `kg.yaml` – KG pipeline + AnyBURL/PyClause configs
  - `rotate.yaml` – RotatE + LightGBM hybrid config
  - `rule_filter` section inside `kg.yaml` – AnyBURL rule filter defaults + HPO ranges
- `hpo/`
  - `ensemble_hpo.yaml` – ensemble HPO/normalization bounds
  - `adaptive_learning.yaml` – threshold tuning and adaptive rules
  - `optimization.yaml` – HPO memory / warmstart controls
- `infra/`
  - `api_hosts.yaml` (+ `.example`) – service endpoints
  - `postgres.yaml` – DB pool/retry/SSL settings
  - `sequences.yaml` – declarative workflow sequences (dispatcher)
  - `validator.yaml` – business-service thresholds/messages
- `observability/`
  - `explainability.yaml` – SHAP/interpretability settings
  - `training_metrics.yaml` – metrics logging toggles
  - `metrics_improvement.json` – target metrics/monitoring hints

Usage: import paths from `pff.config` instead of hardcoding strings to
avoid breakage when reorganizing configs.
