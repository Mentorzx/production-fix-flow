# PFF Codex Playbook

Version 18.0.0 • Updated 2026-01-14

> Scope: this file is `.github/copilot-instructions.md` at the repository root and applies to the entire PFF project.

## 0. TL;DR for coding agents

- Always read `AGENTS.md` before doing anything and update it if policies change.
- Always route filesystem, cache, HTTP, concurrency, serialization, hashing, and logging operations through ports under `pff/application/ports/` and implementations under `pff/infrastructure/`.
- Never hardcode thresholds, paths, tokens, or endpoints. Load them from `config/*.yaml` using the FileManager.
- **Always run commands inside the project's virtual environment:**
  - Assume the working directory is the repository root (`/home/Alex/Development/PFF`).
  - Use `poetry run …` consistently.
- Run fast, targeted tests after every edit. Default command: `poetry run pytest -m "not slow" -q`.
- Emit all artifacts under `outputs/`, mirror runtime logs under `logs/`, and clean up temporary directories when done.
- Respect the logging contract: `logger.info`/`logger.success` MUST be in Portuguese (Brazilian); `logger.warning`/`logger.error`/exceptions MUST be in English.
- Prefer SOTA implementations (Numba/Triton/Ray/SIMD) and document the applied design patterns.

## 1. Project overview & architecture

PFF is a production-grade neuro-symbolic platform focused on Knowledge Graph Completion (KGC) using:
- **DSLFM-KGC:** Deep Sparse Latent Feature Model for representation learning.
- **Probabilistic Circuits (PC2):** Neural PC integration for uncertainty-aware aggregation.
- **Rules:** Symbolic rule integration for domain constraints.

Repository map:
- `config/` – YAML specs (always edit here before touching code).
- `data/models/` – Real KG assets (**read-only**; tests must not depend on these).
- `outputs/` – Canonical home for generated artifacts (models, metrics, plots).
- `pff/drivers/` – Composition roots / entrypoints (CLI, API, HPO).
- `pff/application/` – Use cases + ports (defines interfaces).
- `pff/domain/` – Core business/ML logic (DSLFM-KGC, PC2, rules).
- `pff/infrastructure/` – Adapters (DB, filesystem, external services).
- `pff/shared/` – Cross-cutting code used by 2+ production consumers.
- `tests/` – Unit, integration, and golden master tests.

**Rule:** `pff/domain/**` and `pff/application/**` MUST NOT touch filesystem/network/DB. Side effects live in `pff/infrastructure/**` and are reached through ports.

## 2. Coding conventions

- Provide English Google-style docstrings for every public function/class.
- Use f-strings for all interpolation.
- Avoid inline comments; use block comments for complex logic.
- Favor built-in types; use `typing` only when mandatory.
- Read constants via the FileManager from `config/*.yaml`.
- Clean temporary artifacts after tests or scripts.

## 3. Performance & design patterns

- Default to vectorized/Numba/Triton implementations for hot paths.
- Apply and name design patterns explicitly (Strategy, Factory, Observer, etc.).
- Direct `threading`/`multiprocessing` is permitted **only** in `pff/infrastructure/` or `pff/shared/acceleration/`.
- Use the `pff/infrastructure/profiling.py` for hotspot analysis.

## 4. Testing policy

- **Level 1 (Unit):** Pure domain/shared logic; no external services.
- **Level 2 (Integration):** Application + infrastructure with fixtures.
- **Level 3 (Golden master):** Characterize CLI/HPO behavior with normalized outputs.
- **Level 4 (End-to-end):** Full pipeline validation.

## 5. Protected areas

| Area | Rule |
|------|------|
| `config/**` | Any key change requires docs + config parsing test. |
| `data/models/**` | Read-only. No tests. No writes. |
| `outputs/**` | Only generated content. Never import from here. |
| `pff/drivers/**` | Composition root only; keep thin. |
| `pff/domain/**` | No side effects. No infra imports. |
| `pff/shared/**` + `pff/infrastructure/**` core | Must include regression tests. |

Follow this playbook every time you interact with the PFF repository. It is the authoritative contract for agents working on this codebase.
