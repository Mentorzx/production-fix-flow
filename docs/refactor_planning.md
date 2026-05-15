A tua **espinha dorsal tá certíssima** (Strangler Fig nas fases 0–2 + **Flag Day Cutover** na fase 3, **sem shims**). O que eu vou fazer abaixo é **tirar as contradições internas**, fechar os buracos (principalmente **“Fase 1 in-place” que estava movendo pasta**, e o **db/connection.py** que ficou órfão), e deixar o plano **executável como checklist de PR** sem depender de “fé”.

Segue o **planning perfeito completo (v2.1)**, já “PR-ready”.

---

# Plano de Refatoração — PFF (Production Fix Flow)

**Data:** 2025-12-23
**Status:** HARD STRICT MODE — PLANNING v2.1 (EXECUTÁVEL)
**Estratégia:** Strangler Fig (0–2) + **Flag Day Cutover** (3) — **NO SHIMS POST-CUTOVER**
**Meta:** reduzir complexidade e acoplamento **sem mudar comportamento observável** (CLI/API/artefatos/resultados), permitindo breaking **apenas** em import paths **internos** (formalizado em ADR).

---

## 0. Invariantes (Contrato que NÃO pode quebrar)

**Contrato Externo (não quebra):**

* CLI: nomes de comandos, flags, help text “essencial”, exit codes, estrutura de logs/outputs.
* API (FastAPI/WS): rotas, payloads, códigos, eventos WS.
* Artefatos: layout de `outputs/`, `logs/`, formatos (JSON/CSV/parquet), schemas e chaves.
* Determinismo “prático”: métricas do HPO/treinos dentro de tolerância, seed, dataset fixo de teste.

**Contrato Interno (pode quebrar na Fase 3):**

* Import paths Python (ex.: `pff.validators.*`, `pff.shared.*`, `pff.db.*`) **NÃO** são API pública.

> Isso vira lei com **ADR-007** (Public API) + **ADR-008** (No-Shims Policy).

---

## 1. Objetivos e Não-Objetivos

### Objetivos

1. Reduzir hotspots (CLI, callbacks, scripts) **sem alterar comportamento observável**.
2. Tornar dependências explícitas e testáveis (arquitetura como teste).
3. Preparar migração física (Fase 3) com baixo risco: **Golden Master + Freeze Import Rules + Codemod**.
4. Eliminar “lixão” de `utils/` (shared só quando faz sentido).
5. Melhorar governança (ADRs + regras de dependência).

### Não-Objetivos

* Trocar argparse→typer/click **sem ADR**.
* Reescrever tudo (“big bang”) ou microservices.
* Mexer no layout de `outputs/`/`logs/`.
* Introduzir GraphDB/RDF/SHACL sem prova.
* Criar compat layer permanente (shims).

---

## 2. Mudança Estratégica (v1.4 → v2.1)

**Antes (ruim):** shims eternos = “dor crônica”.
**Agora (bom):** preparar terreno e fazer **um corte limpo** = “dia de dor controlada”.

**Regra:**

* Fases 0–2 podem criar **novos módulos** e extrair código, **mas sem mudar paths antigos**.
* Fase 3 muda paths **de uma vez** e **deleta** os antigos.

> Tradução: nada de “pff/drivers/…” sendo usado antes do Flag Day. Pode existir, mas não vira dependência “ativa” até o cutover.

---

## 3. Arquitetura Alvo (Modular Monolith + Clean/Hexagonal pragmático)

```markdown
pff/
├── drivers/                 # CLI, API, Celery, HPO entrypoints (composition roots)
├── application/             # use cases + ports (interfaces)
├── domain/                  # lógica de negócio/ML/auditoria (pode depender de libs científicas)
├── infrastructure/          # adapters concretos: DB, FS, HTTP clients, reporting, etc.
└── shared/                  # cross-cutting real (>=2 consumidores prod)
```

### Regras de dependência (testáveis)

* drivers → application ✅
* application → domain ✅
* domain → shared ✅
* infrastructure → application ✅ (implementa ports)
* domain → infrastructure ❌
* application → infrastructure ❌ (depende de ports, não de implementação)

---

## 4. Estratégia de Prova (os “airbags”)

### 4.1 Golden Master (com anti-flake kit)

* CPU-only no CI
* Seed fixo (`PYTHONHASHSEED=0`, `--seed 42`)
* Dataset mínimo determinístico (fixture)
* Normalização de timestamps/paths/hostnames/durações
* Tolerâncias (HPO ~1e-3, CLI ~1e-5)

### 4.2 “Freeze” de arquitetura (Import Linter em modo baseline)

* **Passa no baseline atual**.
* Falha só se surgir **nova violação**.
* Depois da Fase 3: regras ficam **estritas**.

### 4.3 “No old namespaces” pós-cutover

* `pff.validators`, `pff.shared`, `pff.db` **não existem**
* nenhuma referência/import para esses namespaces

> **Importante:** evitar `grep` no teste (Windows/CI chora). Use varredura AST em Python.

---

## 5. Plano por Fases (executável)

## Fase 0 — Guardrails (sem refactor ainda)

**Objetivo:** criar o “cinto de segurança” antes de dirigir bêbado.

**Deliverables**

1. `tests/golden_master/` (CLI + HPO)
2. `tests/architecture/`:

   * freeze baseline (permitir o que já existe)
   * bloquear novas violações

**Checklist**

* [ ] fixtures mínimas (config + dataset pequeno)
* [ ] normalizadores (remove campos não determinísticos)
* [ ] CI CPU-only para golden master
* [ ] baseline de violações documentado

**Rollback:** deletar `tests/golden_master` e `tests/architecture`.

---

## Fase 1 — Refatorar hotspots **in-place** (sem mover pastas)

**Objetivo:** reduzir complexidade **sem** mexer nos paths “velhos”.

> Aqui é onde o teu plano original tinha a única treta séria: você dizia “sem mover pastas”, mas já movia pra `pff/drivers/…`. Na v2.1, Fase 1 extrai **módulos auxiliares**, mas mantém o entrypoint e o path principal.

### 1.1 CLI (`pff/cli.py`)

**Ação:** quebrar o arquivo em submódulos mantendo `pff/cli.py` como orquestrador (temporário).

Exemplo de destino (sem mudar o import externo ainda):

```markdown
pff/
├── cli.py                   # permanece, mas fica fino
└── cli_internal/
    ├── commands/
    ├── parsing.py
    ├── printers.py
    └── wiring.py
```

**DoD**

* [ ] `pff --help`, `pff learn --help`, `pff logs --help` iguais (golden master)
* [ ] `pff/cli.py` cai drasticamente de tamanho
* [ ] zero mudança em outputs/logs

### 1.2 Callbacks HPO (`scripts/optimization/callbacks.py`)

**Ação:** extrair para package interno, mantendo o arquivo original como facade temporária.

```markdown
scripts/optimization/
├── callbacks.py                 # fino (temporário)
└── callbacks_internal/
    ├── collectors.py
    ├── visualizers.py
    ├── observers.py
    └── configs.py
```

**DoD**

* [ ] `hpo.py --n-trials 2 ...` passa golden master (ou pelo menos invariantes)
* [ ] callback principal mais simples, menos args

### 1.3 `scripts/sync.py` e `scripts/test.py`

Mesma ideia: extrair para módulos internos, mantendo entrypoints.

---

## Fase 2 — Application layer (sem cutover físico ainda)

**Objetivo:** criar **use cases** e **ports** para “desacoplar por cima”, sem mover pastas.

**Deliverables**

```markdown
pff/application/
  ├── ports/
  ├── audit_use_case.py
  ├── learn_use_case.py
  ├── optimize_use_case.py
  └── sync_use_case.py
```

**Regra prática:** ports só aparecem quando existir pelo menos 1 implementação real + 1 consumidor. Senão vira abstração decorativa.

**Prioridade:** `audit_use_case.py` (cara do produto).

**DoD**

* [ ] CLI/HPO chamam use cases (não chamam “validators/services” diretamente)
* [ ] golden master continua passando
* [ ] freeze de arquitetura continua travando novas violações

---

## Fase 3 — Flag Day Cutover (o dia do “agora vai”)

**Objetivo:** mover tudo pra estrutura final **em commits atômicos**, reescrever imports **automaticamente**, deletar legado **no mesmo PR**, sem shims.

### Passo 1 — `git mv` (somente mover)

**Movimentos (exemplo)**

* `pff/validators/*` → `pff/domain/learning/*` e `pff/domain/kg/*`
* `pff/utils/audit/*` → `pff/domain/audit/*`
 * `pff/utils/core | acceleration | performance | system` → `pff/shared/*`
* **DB:** `pff/db/repositories` **e** `pff/db/connection.py` → `pff/infrastructure/persistence/db/*`
* `pff/services/*` → decidir:

  * se é orquestração: `pff/application/*` (provável)
  * se é regra de domínio: `pff/domain/*`

> **Correção importante:** teu texto movia `repositories` mas esquecia `connection.py`. Isso vira bug instantâneo no cutover.

### Passo 2 — Codemod de imports (sem regex frágil)

**Regra:** não usar regex cega pra trocar import em produção se der pra evitar.

**Recomendado:** codemod com **LibCST** (AST) para reescrever imports com segurança.

* Troca `from pff.validators.dslfm ...` → `from pff.domain.learning.dslfm ...`
* Troca `import pff.shared.core ...` → `import pff.shared.core ...`

**DoD**

* [ ] 0 imports antigos detectados por varredura AST
* [ ] projeto importa (compile/import test)

### Passo 3 — Entrypoints (pyproject)

* `pff = "pff.drivers.cli.main:main"`
* hpo entrypoints idem (se existirem)

### Passo 4 — Delete legado

* `git rm -r pff/validators pff/utils pff/db`
* remover também facades temporárias (`cli_internal` etc.) se tiverem sido realocadas

### Passo 5 — Validação “sem dó”

* Import tests
* Golden master CLI/HPO
* Import Linter agora **estrito**
* Smoke API

**Rollback:** `git revert` do(s) commits do cutover (por isso eles são atômicos).

---

## 6. Testes que substituem “torcida”

### 6.1 `tests/architecture/`

* `freeze_existing_imports_test.py` (Fase 0–2)
* `strict_clean_arch_test.py` (pós-Fase 3)

### 6.2 `tests/golden_master/`

* `test_cli.py`
* `test_hpo.py`
* fixtures mínimas
* normalização

### 6.3 “Old namespaces forbidden” via AST (cross-platform)

* varrer `ast.Import` e `ast.ImportFrom`
* falhar se module começa com `pff.validators`/`pff.shared`/`pff.db`

---

## 7. ADRs (governança mínima pra não virar guerra civil)

* **ADR-007:** Public API do PFF (CLI/API/artefatos vs imports)
* **ADR-008:** No-Shims Policy (proibido compat layer pós-cutover)
* ADR-005: regras de dependência
* ADR-003: outputs/logs imutáveis
* ADR-006: Clean pragmático (domain pode depender de PyTorch/NumPy/Optuna)

---

## 8. Backlog executável (DoD claro)

### P0

* P0-001 Guardrails (Import Linter freeze + baseline)
* P0-002 Golden Master (CLI + HPO)
* P0-003 Refactor CLI in-place (extrair módulos internos)
* P0-004 Refactor callbacks in-place

### P1

* P1-001 Application layer (use cases + ports)
* P1-002 mover dependências: CLI/HPO → use cases
* P1-003 Auditoria: confirmar se `utils/audit/pc2_auditor.py` é produto ou morto

### P2 (Cutover)

* P2-001 Flag Day (git mv)
* P2-002 Codemod AST (LibCST)
* P2-003 Entrypoints
* P2-004 Delete legado
* P2-005 Arquitetura estrita + “old namespaces forbidden”

---

## 9. Critérios de sucesso finais (o placar)

* [ ] CLI/API/artefatos idênticos (ou compatíveis dentro do contrato)
* [ ] Golden master passa (com tolerâncias)
* [ ] Sem `pff.validators`, `pff.shared`, `pff.db` no repo
* [ ] Sem imports antigos (AST scan)
* [ ] Import Linter estrito passa
* [ ] Cutover revertível via git

---

## 10. Nota honesta (a parte adulta do rolê)

Esse plano é “cirurgia com anestesia”: **Golden Master + Freeze** são o anestésico; o **Flag Day** é o corte; e **Sem Shims** é fechar sem deixar gaze dentro do paciente.

---
