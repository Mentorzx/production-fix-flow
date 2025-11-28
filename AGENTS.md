# PFF Codex Playbook

Version 13.0.0 • Updated 2025-11-26

> Scope: this file is `AGENTS.md` at the repository root and applies to the entire PFF project.

## 0. TL;DR for coding agents

- Always read `CONTEXT.md` before doing anything and update it after every change (problem, log, fix, outcome, timestamp).
- Always route filesystem, cache, HTTP, concurrency, serialization, hashing, and logging operations through `pff/utils/**`. Extend utils first, then consume from services/validators.
- Never hardcode thresholds, paths, tokens, or endpoints. Load them from `config/*.yaml`, `.env`, or `settings.py` using the FileManager.
- **Always run commands inside the project's virtual environment:**
  - Assume the working directory is the repository root (`/home/Alex/Development/PFF` or equivalent).
  - First, ensure the `.venv` is active (`source .venv/bin/activate`) or use `poetry run …` consistently.
  - Avoid repeating `cd /home/Alex/Development/PFF` or `source .venv/bin/activate` if the shell is already in the repo root and the virtualenv is active.
- Run fast, targeted tests after every edit. Default command: `poetry run pytest -m "not slow" -q`. Use targeted suites when touching ML/ensemble code.
- **Avoid running full production pipelines (`pff learn`, HPO scripts under `scripts/optimization/**`) as a default test step.** Only execute them in exceptional cases, when explicitly required, and after faster mock/smoke tests are green.
- Emit all artifacts under `outputs/`, mirror runtime logs under `logs/`, and clean up temporary directories (`outputs/temp`, caches, test DBs) when done.
- Respect the logging contract: `logger.info`/`logger.success` MUST be in Portuguese (Brazilian); `logger.warning`/`logger.error`/exceptions MUST be in English. No emojis.
- Think in English internally, produce outward responses/logs with the required language split.
- Prefer SOTA implementations (Numba/Ray/SIMD) and document the applied design patterns.
- **Use small synthetic test fixtures** (tens/hundreds of triples) under `tests/fixtures/` instead of loading real KG assets from `data/models/`.

## 1. Project overview & architecture

PFF is a production-grade neuro-symbolic platform that blends knowledge-graph embeddings (RotatE - SOTA), gradient-boosted models (LightGBM, XGBoost), and symbolic rule systems (AnyBURL, PyClause). It targets telecom-scale KG workloads with strict performance, reproducibility, and observability demands.

Repository map:

- `config/` – YAML specs for validators, ensembles, optimizers (always edit here before touching code).
- `data/models/` – Real KG parquet assets (read-only for code; never modify). **Tests MUST NOT depend on these.**
- `outputs/` – Canonical home for generated artifacts (models, metrics, plots). Tests must create temporary folders inside this tree.
- `pff/services/` – Business services that orchestrate validators and pipelines (MUST call utils).
- `pff/utils/` – Infrastructure layer (I/O, caching, concurrency, accelerators, logging). All lower-level operations flow through here.
- `pff/validators/` – ML and symbolic validators (RotatE, AnyBURL, ensembles) that depend on `pff/utils/**` for shared capabilities.
- `scripts/optimization/` – Optuna/HPO pipelines using real KG data; follow the utils layer and logging constraints.
- `tests/` – Fast unit/integration suites. Mark slow suites with `@pytest.mark.slow`; default automation excludes them.
- `tests/fixtures/` – Small synthetic datasets for fast, deterministic tests.

Services and validators MUST NOT directly touch the filesystem, network, or concurrency primitives; they rely on the utils layer for those concerns.

## 2. Setup & canonical commands

1. Install dependencies with Poetry: `poetry install --sync`.
2. **Always execute commands inside the project's virtual environment:**
   - Prefer `poetry run …` for scripts and tests, OR activate the local `.venv` once per shell session:
     - `source .venv/bin/activate`
   - Do NOT repeatedly call `cd /home/Alex/Development/PFF` or `source .venv/bin/activate` if the shell is already in the repo root and the virtualenv is active.
3. Fast default tests after any change: `poetry run pytest -m "not slow" -q`.
4. Smoke tests for ML subsystems:
   - `poetry run pytest tests/test_ensemble_wrappers.py -q`
   - `poetry run pytest tests/test_symbolic_features_fix.py -q`
   - `poetry run pytest tests/test_utils_hash.py -q` (fastest sanity check)
5. **Do NOT use full pipelines as routine tests:**
   - Avoid running `pff learn` or HPO/optimization scripts (`scripts/optimization/**`) as a "test after every change".
   - Only run them in exceptional cases (e.g., release validation, major refactors), and document the rationale in `CONTEXT.md` and the PR/commit description.
6. Optimizer dry-run (only when absolutely required and time permits): `poetry run python optimize_kg_real.py --dry-run` (respecting real-data constraints).

**✅ Prefer:**

- `poetry run pytest tests/test_utils_hash.py -q`
- `poetry run pytest tests/test_rotate_manager.py -q`
- `poetry run pytest tests/test_ensemble_wrappers.py -q`

**❌ Avoid:**

- `pytest` without `poetry run`
- `python optimize_kg_real.py` without `--dry-run`
- `pff learn` as a generic "test that things work"

Record the command you ran in commit/PR descriptions per the testing policy.

## 3. Mission

Deliver a production-grade neuro-symbolic platform with uncompromised reproducibility, observability, and speed. Every change MUST preserve or improve performance targets and keep optimizers/validators operational on real KG data. Never accept regressions in coverage, sparsity, latency, or determinism.

## 4. Core principles (non-negotiable)

1. **Utils-first architecture:** You MUST route all filesystem, cache, HTTP, concurrency, serialization, hashing, and logging concerns through `pff/utils/**`. Extending new behavior happens there first, with matching tests under `tests/utils/`.
2. **Configuration over hardcoding:** You MUST source every tunable (thresholds, limits, credentials, endpoints) from `config/*.yaml`, `.env`, or `settings.py`. Hardcoded literals are rejected.
3. **Outputs-only artifacts:** All generated files must live under `outputs/`. Delete or archive temporary outputs once they are no longer needed.
4. **Type safety + readability:** Use built-in annotations (`list[str]`, `dict[str, Any]`, etc.), only import `typing` items when a built-in does not exist, interpolate with f-strings only, and provide English Google-style docstrings (Args/Returns/Raises) for every function/class.
5. **Logging contract:** Success/info logs MUST be PT-BR; warnings/errors/exceptions MUST be EN. Emojis are forbidden. Show literal examples only when demonstrating formatting: `logger.info("Processo concluído com sucesso")`, `logger.warning("Symbolic contribution exceeds limit")`.
6. **Design-pattern-first:** Identify and apply Strategy, Factory, Builder, Command, Template Method, Adapter, Observer, Decorator, and DI wherever they clarify responsibilities. Name the patterns explicitly in module docstrings.
7. **SOTA bias & research workflow:** Prefer vectorized, Numba-accelerated, SIMD, or Ray-based approaches for hot loops. Before adopting third-party APIs, resolve documentation through MCP Context7 and justify the choice.
8. **Validation-first engineering:** Add guards to fail fast when coverage, weight floors, or rule dominance violate targets. Never persist "best" artifacts if diagnostics fail.
9. **Testing discipline:** Run fast, relevant tests after every change, document the command, and maintain coverage (>80% utils, >90% business services). Even documentation or config-only edits within scope require a programmatic check (lint or targeted tests).
10. **Language routing:** Interpret user inputs in English internally. Produce outward answers/logs following the PT-BR (info/success) and EN (warn/error) split.
11. **Pipeline ownership:** All adjustments, filters, or fixes to model behavior MUST live in the main pipeline (`pff/validators/**`). Hyperparameter optimization scripts (`scripts/optimization/**`) may only orchestrate trials/runs, persist artifacts, and render visualizations.
12. **Safe rollout:** Any change to scoring, thresholds, coverage, or ensemble weighting MUST be controlled via config-level flags or modes (e.g., `mode: "conservative" | "experimental"`) and backward-compatible by default.

## 5. Utils layer (mandatory usage)

| Module | Purpose | Typical usage |
|--------|---------|---------------|
| `pff/utils/file_manager.py` | Unified async/sync I/O for 13+ formats (msgspec JSON, Parquet, CSV, ZIP) with streaming safeguards. | Always load configs, datasets, and artifacts through this class. Extend it when you need new formats, then call it from higher layers. |
| `pff/utils/cache.py` | Multi-layer cache (memory/disk/HTTP) with eviction, compression, and rate-aware invalidation. | Wrap expensive API calls or derived datasets. Never roll your own caching logic in services. |
| `pff/utils/concurrency.py` | Adaptive concurrency manager (Ray vs process pools) with lazy submission to prevent OOM. | All parallel workloads (rule validation, feature extraction) must go through this abstraction. No raw `threading`/`multiprocessing` outside utils. |
| `pff/utils/http_client.py` | Resilient HTTP/2 client with retries, circuit breakers, failover. | Every outbound HTTP call (telemetry, ingestion). Configure auth via `.env` + config. |
| `pff/utils/loop_accelerator.py` | Strategy/Factory for vectorized, parallel, or Python fallback loops. | Speed up scalar loops in validators/services without duplicating logic. |
| `pff/utils/numba_kernels.py` | Pre-built Numba kernels for KG math and feature encoding. | Import these kernels instead of writing inline `@njit`. Add new kernels here with tests. |
| `pff/utils/symbolic_rule_accelerator.py` | Deterministic symbolics + Numba adapter for rule validation. | Use it to ensure symbolic features align with business-service results. |
| `pff/utils/hash.py` | Stable hashing utilities (`stable_hash`, `hash_bytes`). | Feature encoding, cache keys, reproducible seeds. Never use Python's built-in `hash()`. |
| `pff/utils/logger.py` | Structured logging helpers that enforce colorless PT-BR/EN output. | All logging must go through this module. |
| `pff/utils/hooks/auto_config.py` | Environment/bootstrap detection and auto-configuration. | CLI and orchestrators use it to load overrides safely. |
| `pff/utils/core/*` | Hardware detection, ML training profiles, metrics, cleanup hooks. | Access only via the published interfaces—no direct reimplementation. |
| `pff/utils/performance/training_observer.py` | Observer pattern for training events (epochs, metrics, checkpoints). | Use `CompositeObserver` to decouple trainers from logging backends (MLflow, console, DB). |
| `pff/utils/ml/kge_strategy.py` | Strategy Pattern for KGE models (RotatE). | Use `KGEModelStrategy` ABC for model abstraction. `RotatEStrategy` is the primary implementation. |
| `pff/utils/ml/model_factory.py` | Factory Pattern for model creation (KGE, LightGBM, XGBoost, CatBoost). SOTA: GPU auto-detect + gradient quantization. | Use `ModelFactory.create(ModelType.ROTATE, ...)` for centralized instantiation. LightGBM/XGBoost auto-detect GPU. |
| `pff/utils/ml/base_trainer.py` | Template Method Pattern for training loops. | Extend `BaseTrainer` and implement `_setup_model`, `_train_epoch`, `_validate`. Integrates with `TrainingObserver`. |
| `pff/validators/rotate/checkpoint_manager.py` | Checkpoint management for RotatE (SRP). | Use `RotatECheckpointManager` for save/load/cleanup of model checkpoints. |
| `pff/validators/rotate/contrastive.py` | Contrastive learning losses for KG embeddings. | Use `ContrastiveLearner` with InfoNCE/Triplet/NTXent/KG losses. Factory via `ContrastiveLossFactory`. |
| `pff/validators/rotate/core.py` | RotatE model with complex embeddings (SOTA). | Use `RotatEModel` for rotation-based KG embeddings. Better for sparse graphs and anti-symmetric relations. |
| `pff/validators/rotate/config.py` | RotatE configuration with Builder pattern. | Use `RotatEConfig` or `RotatEConfigBuilder` for fluent configuration. Load from `config/rotate.yaml`. |
| `pff/validators/rotate/manager.py` | RotatE training orchestrator. | Use `RotatEManager` for training, evaluation, and checkpointing. Primary KGE model for the pipeline. |
| `pff/validators/rotate/lightgbm_trainer.py` | RotatE+LightGBM hybrid trainer. | Use `RotatELightGBMTrainer` for hybrid model training combining RotatE embeddings with LightGBM. |
| `pff/validators/ensembles/metrics_reporter.py` | Metrics reporting extracted from AdvancedEnsembleTrainer (SRP). | Use `EnsembleMetricsReporter` for classification metrics, feature balance, JSON reports. |
| `pff/validators/ensembles/feature_balancer.py` | Feature balance validation extracted (SRP). | Use `EnsembleFeatureBalancer` with `BalanceConfig` to validate neural/symbolic ratio. |
| `pff/validators/ensembles/attention.py` | Attention mechanism for neural-symbolic synergy. | Use `AttentionEnsembleTransformer` (sklearn-compatible) or `AttentionFactory` for custom attention. |
| `pff/services/violation_penalty.py` | Violation penalty calculator extracted from ModelIntegration (SRP). | Use `ViolationPenaltyCalculator` with `PenaltyConfig` for score adjustments based on violations. Config from `validator.yaml`. |
| `pff/services/rule_builder.py` | Builder and Factory patterns for Rule construction. | Use `RuleBuilder` for fluent Rule construction, `RuleSourceFactory` for loading rules from files (JSON, TSV). |
| `pff/services/validation_observer.py` | Observer pattern for validation events. | Use `CompositeValidationObserver` to dispatch events to `LoggingValidationObserver`, `MetricsValidationObserver`. |
| `pff/services/business_service/` | Refactored Business Service package (modular structure). | Contains `core.py` (BusinessService), `models.py` (Rule, RuleViolation), `rule_engine.py` (RuleEngine), `rule_validator.py` (RuleValidator), `model_integration.py` (ModelIntegration), `triple_index.py` (TripleIndex). |

When adding new infrastructure, extend this table, describe the new module's purpose, and add regression tests to `tests/utils/`.

## 6. Code style requirements

- Provide English Google-style docstrings for every public function/class (`Args`, `Returns`, `Raises` complete).
- Use f-strings for all interpolation; `%` or `.format()` is forbidden.
- Avoid inline comments. Place concise block comments above complex logic if needed.
- Favor built-in types; import from `typing` only when mandatory (`Any`, `Protocol`, etc.).
- Order imports: stdlib, third-party, internal. Remove unused imports instantly.
- Read constants via the FileManager from `config/*.yaml`/`.env`/`settings.py`. Never read configs with raw `open()` or third-party loaders outside utils.
- After executing tests or scripts, clean temporary artifacts (logs, caches, database fixtures) to keep the repo deterministic.

## 7. Logging & monitoring

### 7.1. Log level purpose and language

| Level | Language | Purpose | Examples |
|-------|----------|---------|----------|
| `logger.info` | PT-BR | High-level process steps, user-facing summaries, key metrics at checkpoints, **epoch progress**, **training progress** | `logger.info("Iniciando treinamento RotatE: epocas=50, entidades=10000")`, `logger.info("Epoca 10/50: loss=0.234, MRR=0.42")` |
| `logger.success` | PT-BR | Major step completions (use sparingly) | `logger.success("Treinamento concluido: MRR=0.45, Hits@10=0.82")` |
| `logger.warning` | EN | Degraded states, fallbacks, missing optional data | `logger.warning("CUDA unavailable, falling back to CPU")` |
| `logger.error` | EN | Failures that stop or invalidate the current flow | `logger.error("Training failed: checkpoint corrupted at path=%s", path)` |
| `logger.debug` | EN | Detailed diagnostics, hardware info, shapes, timings, **resource limits**, **internal thresholds**, **adaptive parameters** (debug mode only) | `logger.debug("Batch shape: %s, device: %s", batch.shape, device)`, `logger.debug("Adaptive resource limits: %s", limits)` |

### 7.2. Structured logging rules

- **Info/success logs:** Focus on model metrics, core pipeline steps, and user-relevant progress (epochs, training steps, validation metrics). Include stable keys for comparison across runs.
- **Debug logs:** Use for hardware details, fine-grained timings, internal thresholds, raw tensor shapes, **adaptive resource calculations**, **internal optimization decisions**. These MUST NOT be emitted at info level in production.
- **Warnings:** Must contain remediation hints when possible.
- **Errors:** Must include enough context (paths, IDs, key parameters) to debug the issue.

### 7.3. Noise control (forbidden patterns)

Agents MUST NOT add logs that:

- Claim generic improvements without specific metrics: ❌ `logger.info("Melhoria de 40% na velocidade")`
- Restate obvious information without context
- Spam info-level with low-level debug details

**Replace with structured alternatives:**

```python
# ❌ BAD: Vague improvement claim
logger.info("Speed improved by 40%")

# ✅ GOOD: Structured with before/after metrics
logger.info("Tempo medio por epoca: %.3fs -> %.3fs (modelo=RotatE)", old_time, new_time)
```

### 7.4. Observability minimums

Any change affecting training loops, optimizers, ensemble logic, or symbolic coverage MUST expose:

- At least one before/after metric (e.g., epoch duration, rules filtered, coverage)
- Stable keys in logs for run comparison

### 7.5. General rules

- Persist runtime logs under `logs/` and mirror them into `outputs/logs/` for ephemeral runs.
- Metrics reports go to JSON files under component-specific folders (e.g., `outputs/ensemble/metrics_all.json`). No ad-hoc paths.
- Fallbacks and degraded modes must warn in EN with remediation hints.

## 8. Performance & design patterns

- Default to vectorized/Numba/SIMD/Ray implementations for hot paths. Reference the specific acceleration strategy in docstrings.
- Services/validators MUST describe the applied design patterns (Strategy, Factory, Command, Builder, Template Method, Adapter, Observer, Decorator, DI, etc.) in their module/class docstrings.
- Direct `threading`/`multiprocessing` usage is permitted only in the utils layer. Higher layers should request concurrency via `pff/utils/concurrency.py`.

## 9. Dependency guidance

- Before using/upgrading third-party libraries, resolve documentation via MCP Context7 (or internal references) and summarize key behaviors in the PR description or module docstring.
- Keep `poetry.lock` synchronized with `pyproject.toml`. When bumping versions, describe compatibility considerations and run targeted tests relevant to that dependency.
- Record the exact command used to verify dependency changes (e.g., `poetry run pytest tests/test_http_client.py -q`).

## 10. Testing policy

### 10.1. Test hierarchy

| Level | Type | Purpose | When to run |
|-------|------|---------|-------------|
| 0 | Static checks | Lint/type (if available) | Before commit |
| 1 | Unit tests | Pure utils, functions without external deps | After every change |
| 2 | Integration tests | Validators/services with mocked fixtures | After changes to business logic |
| 3 | End-to-end / real-data | `pff learn`, HPO scripts | Only for release validation, major refactors |

### 10.2. General rules

- Run fast, relevant tests after every change. Prefer focused unit/integration suites over broad-but-slow ones.
- Required coverage thresholds: >80% for utils, >90% for business-service paths. Never merge if coverage regresses.
- Even "simple" changes (config tweaks, doc adjustments that touch code paths) require a programmatic check (at minimum, the fastest applicable pytest command).
- Record the executed command in your commit/PR summary.

### 10.3. Mocking & fixtures policy

- Test datasets MUST be small (tens/hundreds of triples, not millions).
- Prefer synthetic or anonymized KGs under `tests/fixtures/**` or generated on-the-fly.
- Tests MUST NOT depend on large production assets under `data/models/**`.

## 11. Workflow checklist for agents

1. **Open a shell in the repo root and ensure the virtualenv is active:**
   - Working directory MUST be the repository root.
   - Activate the `.venv` once if needed: `source .venv/bin/activate` (or rely on `poetry run …`).
   - Do NOT keep repeating `cd` or `source` if they are already in effect.
2. **Read `CONTEXT.md`:** Summarize current investigations, logs, and past fixes. Assume the user already executed the latest run; verify whether prior fixes worked before proposing new changes.
3. **Update configuration first:** Add tunables to `config/*.yaml`, `.env`, or `settings.py` (via FileManager) before modifying code.
4. **Implement via utils + patterns:** Extend the utils layer when introducing new infrastructure, then integrate via Strategy/Factory/etc. Follows SOTA best practices.
5. **Design fast, mock-friendly tests instead of heavy pipelines:**
   - Prefer small, deterministic unit and integration tests that mock I/O, models, and external services.
   - Avoid using `pff learn` or full HPO pipelines as a testing shortcut. Only run them in rare, explicitly justified scenarios.
6. **Write/adjust tests and run them:** Use the canonical commands in §2 (fast pytest targets). Capture the exact command in your notes and commit/PR summary.
7. **Validate artifacts:** Ensure all outputs stay under `outputs/`, logs under `logs/`, and clean temporary files/caches/DB fixtures.
8. **Update `CONTEXT.md`:** Document the problem, log snippet, fix, observed outcome, and timestamp. This file is the canonical session memory.
9. **Update `AGENTS.md` when policies change:** If you alter global engineering rules or add new utils modules, reflect those changes here.

## 12. Known failure modes & how agents should react

| Failure | Cause | Action |
|---------|-------|--------|
| "Entity/relation mappings not found in any of: [...]" | HPO trial directory missing pyclause mappings | Check runtime_outputs paths; verify mappings are written via FileManager; confirm embedding generation succeeded |
| "Segmentation fault during RotatE training" | Invalid data (NaNs, shape mismatches) or CUDA config | Check for NaNs in input; verify torch.compile config; prefer Python exceptions over segfaults |
| "CUDA initialization failed: config[i] == get()->name()" | CUDA allocator reconfigured after initialization | Use `_CUDA_ALLOCATOR_CONFIGURED` flag; configure allocator ONCE before CUDA init |
| "LightGBM GPU error" | OpenCL unavailable | Fallback to `device="cpu"` with appropriate num_threads |
| "FileNotFoundError: training data not found" | Path mismatch or missing preprocessing | Check fallback paths; verify data pipeline completed |

## 13. Protected areas

The following areas require extra caution and explicit justification for changes:

| Area | Policy |
|------|--------|
| `data/models/**` | **Read-only.** Never modify production KG assets. |
| Global logging language contract | **Immutable.** PT-BR for info/success, EN for warn/error. |
| Top-level folder structure | **Do not rename/move** without coordinated migration plan. |
| `scripts/optimization/**` HPO scripts | Behavior changes require: updated configs, updated tests, documentation in CONTEXT.md and PR notes. |
| `pff/utils/**` core modules | Changes must include regression tests in `tests/utils/`. |

Follow this playbook every time you interact with the PFF repository. It is the authoritative contract for Codex agents working on this codebase.
