# Production Fix Flow

[![CI/CD](https://github.com/Mentorzx/production-fix-flow/actions/workflows/ci.yml/badge.svg)](https://github.com/Mentorzx/production-fix-flow/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Type checking: mypy](https://img.shields.io/badge/type%20checking-mypy-2a6dbb.svg)](https://mypy-lang.org/)

Production Fix Flow is a Python backend platform for orchestrating operational API workflows with validation, storage and observability layers.

This repository is strongest as a systems-and-architecture project. It combines CLI and API entrypoints, YAML-driven orchestration, async services, storage backends, background execution and a large codebase organized around domain, application, infrastructure and shared layers.

## Why it matters

- It shows how I structure a large Python codebase around execution flows instead of one-off scripts.
- It includes operational concerns that matter in real environments: retries, logging, typed configuration, storage, background jobs and service boundaries.
- It also contains experimental components for ranking, learning and optimization, but this README positions the repository around what is clearly supported by the public codebase.

## What the repository includes

- YAML-driven workflow execution for multi-step operational sequences
- FastAPI endpoints for execution and health checks
- background processing with Celery and Redis
- PostgreSQL-oriented persistence and execution storage
- shared infrastructure for file handling, logging, caching and runtime management
- a sizeable test suite plus linting and type-checking configuration
- optional research and ML-related modules kept inside the same codebase

## Stack

- Python 3.12
- FastAPI, Uvicorn, HTTPX
- PostgreSQL, asyncpg, Redis, Celery
- Polars, DuckDB, PyArrow
- Pydantic, Poetry, pytest, mypy, Ruff
- optional acceleration and experimentation dependencies such as Ray, Triton, FAISS and Rust bindings

## Repository shape

```text
src/pff/
  application/      # orchestration use cases and service-level logic
  domain/           # domain logic, rules, validation, learning-related modules
  drivers/          # CLI, API and worker entrypoints
  infrastructure/   # persistence, observability, performance and system integrations
  shared/           # cross-cutting utilities for files, logging, runtime and cache
tests/
config/
scripts/
```

## Quick start

### 1. Install dependencies

```bash
poetry install
```

### 2. Prepare local configuration

```bash
cp .env.example .env
cp config/infra/api_hosts.yaml.example config/infra/api_hosts.yaml
```

### 3. Run from the CLI

```bash
poetry run python -m pff run data/manifest.yaml
```

### 4. Run the API

```bash
poetry run python -m pff api --host 0.0.0.0 --port 8000 --reload
```

## Useful commands

```bash
poetry run pytest -m "not slow" -q
poetry run ruff check .
poetry run mypy src
docker-compose up -d
```

## Notes on scope

- Some modules in this repository explore ranking, knowledge-graph and optimization ideas.
- Those components are part of the codebase, but this public README avoids claims that depend on unpublished benchmarks, internal datasets or paper-style comparisons.
- If you are evaluating the repository for backend work, the strongest signal is the architecture and operational structure, not any research-style score.

## Contact

Public profile: [github.com/Mentorzx](https://github.com/Mentorzx)
