# PFF Codex Playbook

Version 12.0.0 • Updated 2025-11-19

> Scope: this file is `AGENTS.md` at the repository root and applies to the entire PFF project.

## 0. TL;DR for coding agents
- Always read `CONTEXT.md` before doing anything and update it after every change (problem, log, fix, outcome, timestamp).
- Always route filesystem, cache, HTTP, concurrency, serialization, hashing, and logging operations through `pff/utils/**`. Extend utils first, then consume from services/validators.
- Never hardcode thresholds, paths, tokens, or endpoints. Load them from `config/*.yaml`, `.env`, or `settings.py` using the FileManager.
- Run fast, targeted tests after every edit. Default command: `poetry run pytest -m "not slow" -q`. Use targeted suites such as `poetry run pytest tests/test_ensemble_wrappers.py -q` or `poetry run pytest tests/test_symbolic_features_fix.py -q` when touching ML/ensemble code.
- Emit all artifacts under `outputs/`, mirror runtime logs under `logs/`, and clean up temporary directories (`outputs/temp`, caches, test DBs) when done.
- Respect the logging contract: `logger.info`/`logger.success` MUST be in Portuguese (Brazilian); `logger.warning`/`logger.error`/exceptions MUST be in English. No emojis.
- Think in English internally, produce outward responses/logs with the required language split.
- Prefer SOTA implementations (Numba/Ray/SIMD) and document the applied design patterns.

## 1. Project overview & architecture
PFF is a production-grade neuro-symbolic platform that blends knowledge-graph embeddings (TransE), gradient-boosted models (LightGBM, XGBoost), and symbolic rule systems (AnyBURL, PyClause). It targets telecom-scale KG workloads with strict performance, reproducibility, and observability demands.

Repository map:
- `config/` – YAML specs for validators, ensembles, optimizers (always edit here before touching code).
- `data/models/` – Real KG parquet assets (read-only for code; never modify).
- `outputs/` – Canonical home for generated artifacts (models, metrics, plots). Tests must create temporary folders inside this tree.
- `pff/services/` – Business services that orchestrate validators and pipelines (MUST call utils).
- `pff/utils/` – Infrastructure layer (I/O, caching, concurrency, accelerators, logging). All lower-level operations flow through here.
- `pff/validators/` – ML and symbolic validators (TransE, AnyBURL, ensembles) that depend on `pff/utils/**` for shared capabilities.
- `scripts/optimization/` – Optuna/HPO pipelines using real KG data; follow the utils layer and logging constraints.
- `tests/` – Fast unit/integration suites. Mark slow suites with `@pytest.mark.slow`; default automation excludes them.

Services and validators MUST NOT directly touch the filesystem, network, or concurrency primitives; they rely on the utils layer for those concerns.

## 2. Setup & canonical commands
For Codex agents:
1. Install dependencies with Poetry: `poetry install --sync`.
2. Activate the virtualenv automatically via `poetry run …` (no direct `pip`).
3. Fast default tests after any change: `poetry run pytest -m "not slow" -q`.
4. Smoke tests for ML subsystems:
   - `poetry run pytest tests/test_ensemble_wrappers.py -q`
   - `poetry run pytest tests/test_symbolic_features_fix.py -q`
   - `poetry run pytest tests/test_utils_hash.py -q` (fastest sanity check)
5. Optimizer dry-run (only when absolutely required and time permits): `poetry run python optimize_kg_real.py --dry-run` (respecting real-data constraints).
Record the command you ran in commit/PR descriptions per the testing policy.

## 3. Mission
Deliver a production-grade neuro-symbolic platform with uncompromised reproducibility, observability, and speed. Every change MUST preserve or improve performance targets and keep optimizers/validators operational on real KG data. Never accept regressions in coverage, sparsity, latency, or determinism.

## 4. Core principles (non-negotiable)
## 4. Core principles (non-negotiable)
1. **Utils-first architecture:** You MUST route all filesystem, cache, HTTP, concurrency, serialization, hashing, and logging concerns through `pff/utils/**`. Extending new behavior happens there first, with matching tests under `tests/utils/`.
2. **Configuration over hardcoding:** You MUST source every tunable (thresholds, limits, credentials, endpoints) from `config/*.yaml`, `.env`, or `settings.py`. Hardcoded literals are rejected.
3. **Outputs-only artifacts:** All generated files must live under `outputs/`. Delete or archive temporary outputs once they are no longer needed.
4. **Type safety + readability:** Use built-in annotations (`list[str]`, `dict[str, Any]`, etc.), only import `typing` items when a built-in does not exist, interpolate with f-strings only, and provide English Google-style docstrings (Args/Returns/Raises) for every function/class.
5. **Logging contract:** Success/info logs MUST be PT-BR; warnings/errors/exceptions MUST be EN. Emojis are forbidden. Show literal examples only when demonstrating formatting: `logger.info("Processo concluído com sucesso")`, `logger.warning("Symbolic contribution exceeds limit")`.
6. **Design-pattern-first:** Identify and apply Strategy, Factory, Builder, Command, Template Method, Adapter, Observer, Decorator, and DI wherever they clarify responsibilities. Name the patterns explicitly in module docstrings.
7. **SOTA bias & research workflow:** Prefer vectorized, Numba-accelerated, SIMD, or Ray-based approaches for hot loops. Before adopting third-party APIs, resolve documentation through MCP Context7 and justify the choice.
8. **Validation-first engineering:** Add guards to fail fast when coverage, weight floors, or rule dominance violate targets. Never persist “best” artifacts if diagnostics fail.
9. **Testing discipline:** Run fast, relevant tests after every change, document the command, and maintain coverage (>80% utils, >90% business services). Even documentation or config-only edits within scope require a programmatic check (lint or targeted tests).
10. **Language routing:** Interpret user inputs in English internally. Produce outward answers/logs following the PT-BR (info/success) and EN (warn/error) split.
11. **Pipeline ownership:** All adjustments, filters, or fixes to model behavior MUST live in the main pipeline (`pff/validators/**`). Hyperparameter optimization scripts (`scripts/optimization/**`) may only orchestrate trials/runs, persist artifacts, and render visualizations.


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
| `pff/utils/hash.py` | Stable hashing utilities (`stable_hash`, `hash_bytes`). | Feature encoding, cache keys, reproducible seeds. Never use Python’s built-in `hash()`. |
| `pff/utils/logger.py` | Structured logging helpers that enforce colorless PT-BR/EN output. | All logging must go through this module. |
| `pff/utils/hooks/auto_config.py` | Environment/bootstrap detection and auto-configuration. | CLI and orchestrators use it to load overrides safely. |
| `pff/utils/core/*` | Hardware detection, ML training profiles, metrics, cleanup hooks. | Access only via the published interfaces—no direct reimplementation. |

When adding new infrastructure, extend this table, describe the new module’s purpose, and add regression tests to `tests/utils/`.

## 6. Code style requirements
- Provide English Google-style docstrings for every public function/class (`Args`, `Returns`, `Raises` complete).
- Use f-strings for all interpolation; `%` or `.format()` is forbidden.
- Avoid inline comments. Place concise block comments above complex logic if needed.
- Favor built-in types; import from `typing` only when mandatory (`Any`, `Protocol`, etc.).
- Order imports: stdlib, third-party, internal. Remove unused imports instantly.
- Read constants via the FileManager from `config/*.yaml`/`.env`/`settings.py`. Never read configs with raw `open()` or third-party loaders outside utils.
- After executing tests or scripts, clean temporary artifacts (logs, caches, database fixtures) to keep the repo deterministic.

## 7. Logging & monitoring
- Info/success PT-BR; warnings/errors/exceptions EN; no emojis. Examples: `logger.info("Processo concluído com sucesso")`, `logger.warning("Symbolic contribution exceeds limit")`.
- Long-running flows (optimizers, trainers) must emit periodic heartbeat logs (progress, ETA, trial number) and close with coverage/sparsity/memory summaries.
- Persist runtime logs under `logs/` and mirror them into `outputs/logs/` for ephemeral runs. Configure rotation + compression and remove job-specific files when done.
- Metrics reports go to JSON files under component-specific folders (e.g., `outputs/ensemble/metrics_all.json`). No ad-hoc paths.
- Fallbacks and degraded modes must warn in EN with remediation hints (“Check AnyBURL rules file”).

## 8. Performance & design patterns
- Default to vectorized/Numba/SIMD/Ray implementations for hot paths. Reference the specific acceleration strategy in docstrings.
- Services/validators MUST describe the applied design patterns (Strategy, Factory, Command, Builder, Template Method, Adapter, Observer, Decorator, DI, etc.) in their module/class docstrings.
- Direct `threading`/`multiprocessing` usage is permitted only in the utils layer. Higher layers should request concurrency via `pff/utils/concurrency.py`.

## 9. Dependency guidance
- Before using/upgrading third-party libraries, resolve documentation via MCP Context7 (or internal references) and summarize key behaviors in the PR description or module docstring.
- Keep `poetry.lock` synchronized with `pyproject.toml`. When bumping versions, describe compatibility considerations and run targeted tests relevant to that dependency.
- Record the exact command used to verify dependency changes (e.g., `poetry run pytest tests/test_http_client.py -q`).

## 10. Testing policy
- Run fast, relevant tests after every change. Prefer focused unit/integration suites over broad-but-slow ones.
- Required coverage thresholds: >80% for utils, >90% for business-service paths. Never merge if coverage regresses.
- Even “simple” changes (config tweaks, doc adjustments that touch code paths) require a programmatic check (at minimum, the fastest applicable pytest command).
- Record the executed command in your commit/PR summary.

## 11. Workflow checklist for agents
1. **Read `CONTEXT.md`:** Summarize current investigations, logs, and past fixes. Assume the user already executed the latest run; verify whether prior fixes worked before proposing new changes.
2. **Update configuration first:** Add tunables to `config/*.yaml`, `.env`, or `settings.py` (via FileManager) before modifying code.
3. **Implement via utils + patterns:** Extend the utils layer when introducing new infrastructure, then integrate via Strategy/Factory/etc. Follows SOTA best practices.
4. **Write/adjust tests and run them:** Use the canonical commands in §2. Capture the exact command in your notes.
5. **Validate artifacts:** Ensure all outputs stay under `outputs/`, logs under `logs/`, and clean temporary files/caches/DB fixtures.
6. **Update `CONTEXT.md`:** Document the problem, log snippet, fix, observed outcome, and timestamp. This file is the canonical session memory.
7. **Update `AGENTS.md` when policies change:** If you alter global engineering rules or add new utils modules, reflect those changes here.

Follow this playbook every time you interact with the PFF repository. It is the authoritative contract for Codex agents working on this codebase.
