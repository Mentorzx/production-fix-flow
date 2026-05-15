# PFF Knowledge Base (v18.0.0)

Este é o repositório central de conhecimento para o projeto **PFF (Production Fix Flow)**. Ele consolida a identidade do sistema, regras de engenharia, arquitetura e stack técnico.

---

## 🏗️ Identidade & Propósito

O PFF é um sistema inteligente de orquestração para automação de sequências complexas de chamadas API em produção, focado no domínio de **Telecom**.

Utiliza IA **Neuro-Simbólica** combinando:

- **DSLFM-KGC**: Deep Sparse Latent Feature Models para Knowledge Graph Completion.
- **PC2**: Probabilistic Circuits (Variant 2) para integração de lógica simbólica e incerteza.

---

## 🛠️ Engineering Guidelines (AGENTS.md)

As diretrizes abaixo são mandatórias para todos os agentes e desenvolvedores.

### 🐍 Coding Standards

- **Python 3.12+**: Utilize as funcionalidades mais recentes da linguagem.
- **Docstrings**: Obrigatórias para funções/classes públicas. Devem estar em **Inglês** (Google-style).
- **Tipagem**: Evite `typing` a menos que necessário; `Any` é o último recurso.
- **Configuração**: Nunca use hardcoding. Todas as variáveis ajustáveis devem vir de arquivos YAML em `config/**` e ser lidas via `FileManager`.
- **I/O Discipline**: Todas as operações de leitura/escrita de arquivos **DEVEM** passar pelo `FileManager` em `pff/shared/core/file_manager`.

### 📝 Logging & Monitoramento

- **Idiomas**:
    - **Info/Success**: Português (Brasil). Focado no progresso do usuário (ex: "Época 10/50 concluída").
    - **Warnings/Errors/Debug**: Inglês. Focado em depuração técnica e logs internos.
- **Estrutura**: Todo log deve conter `timestamp`, `component_name`, e `key_parameters`.
- **LocalStorage**: Logs são rotacionados em `logs/` e artefatos de saída ficam em `outputs/`.

### 🧪 Testing Policy

- **Frequência**: Rode testes após cada alteração.
- **Hierarquia**:
    - `0`: Static checks (lint/ruff).
    - `1`: Unit tests (shared/domain).
    - `2`: Integration (infrastructure/ports).
    - `3`: Golden Masters (comportamento CLI/HPO).
- **Comandos**:
    - `poetry run ruff check .`
    - `poetry run pytest -q`

---

## 📐 Arquitetura: Clean Layers

Seguimos uma estrutura de **Clean Architecture** com a regra de dependência apontando para dentro:

`pff/drivers` (Root) → `pff/application` (Use Cases/Ports) → `pff/domain` (ML Logic/Entities)

- **Infrastructure**: Implementa os ports definidos na camada application.
- **Shared**: Utilidades usadas por 2 ou mais consumidores de produção.
- **No-Shims Policy**: Não criamos camadas de compatibilidade. Se um path muda, a mudança é atômica (`git mv` + codemod).

---

## 📀 Estratégia de Armazenamento Tiered

O sistema segue a política **Parquet-Arrow-Postgres-First**:

| Tier            | Tecnologia        | Caso de Uso                                                    |
| --------------- | ----------------- | -------------------------------------------------------------- |
| **Archival**    | **Parquet**       | Dados tabulares em repouso (histórico, datasets, logs brutos). |
| **Hot Cache**   | **Arrow IPC**     | Cache local efêmero, zero-copy reads via `mmap`.               |
| **Operational** | **PostgreSQL**    | Estado relacional, task queues, triagens HPO (via `asyncpg`).  |
| **Out-of-Core** | **Lance/LanceDB** | Negative sampling em larga escala (>100M triples) via disco.   |

---

## 🚀 Comandos Canônicos

```bash

# Instalação
poetry install

# Qualidade de Código
poetry run ruff check .
poetry run ruff format .

# Testes Rápidos
poetry run pytest tests/audit/test_eval_protocol.py -q
```

---

> [!IMPORTANT]
> **Data models em `data/models/**` são estritamente READ-ONLY.** Todos os outputs devem ser escritos em `outputs/**`.
