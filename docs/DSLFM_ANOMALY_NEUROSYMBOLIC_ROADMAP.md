# Roadmap: DSLFM‑KGC + PC2 para Detecção de Anomalias e Validação Neuro‑Simbólica (PFF)

**Escopo:** este documento descreve, de forma prática e minuciosa, como adaptar a stack atual **DSLFM‑KGC + PC Variant 2 (PC2/NPC)** para operar bem em **classificação (SIM/NÃO)** e **detecção de anomalias**, preservando o que já funciona em **ranking/link prediction** (MRR/Hits) e seguindo estritamente o `AGENTS.md`.

**Repositório:** PFF (`/home/Alex/Development/PFF`)
**Pilha atual (confirmada no código):** DSLFM‑KGC (`pff/validators/dslfm`) + PC2/NPC (`pff/validators/pc/npc.py`) + HPO Optuna (`scripts/optimization/**`)
**Arquivos‑chave que serão impactados:** `pff learn` (`pff/cli.py`), HPO (`scripts/optimization/trials/*`), live metrics (`scripts/optimization/callbacks.py`), score composto (`scripts/optimization/trials/scoring.py`).

---

## 0) Objetivo e não‑objetivos

### 0.1 Objetivo (produto)

Transformar o modelo DSLFM‑KGC (bom em ordenação/ranking) em um componente que também:

1. **Classifica triplas** com métricas fortes para decisão binária (AUC/PR‑AUC/F1/precision/recall) de forma **global e comparável entre relações**.
2. **Detecta anomalias** (fatos implausíveis/ruidosos dentro do KG) com limiares robustos (sem heurística frágil), e com **validação neuro‑simbólica auditável**.
3. Mantém **determinismo por default** (mesma entrada ⇒ mesma saída) e melhora performance sem comprometer MRR/Hits.
4. Entrega um **laudo acionável em JSON** (“JSON→grafo→JSON”), apontando **onde** no JSON está o problema (`json_pointer`), **qual invariante** foi quebrado (schema/perfil/SHACL/PC2/EVT), **evidência histórica** (baseline/drift) e **reparo sugerido** (JSON Patch).

### 0.2 Não‑objetivos (por enquanto)

- Não substituir a modelagem central do DSLFM‑KGC por outro KGE “do zero”.
- Não depender de assets reais em `data/models/**` para testes.
- Não “inventar” I/O/locks/cache/threads fora de `pff/utils/**`.

---

## 1) Estado atual no PFF (o que já existe e onde)

### 1.1 DSLFM‑KGC (core neural‑gerativo)

**Local:** `pff/validators/dslfm/dslfm_kgc.py`
O modelo já contém as peças essenciais para a adaptação:

- **VAE + IBP**: latentes com distribuição (não apenas ponto) ⇒ habilita sinais gerativos (reconstrução, KL, incerteza).
- **SBM decoder** (Stochastic Blockmodel) para scoring de triplas.
- **PC2 integrado** quando `lambda_pc > 0` (ver seção 1.2).
- **Camada de lógica diferenciável** (opcional) em `pff/validators/dslfm/logic_layer.py` controlada por `lambda_logic`.

**Config principal:** `config/models/dslfm.yaml` (carregado via `FileManager` em `pff/validators/dslfm/dslfm_kgc.py:load_dslfm_kgc_settings`).

### 1.2 PC Variant 2 (PC2/NPC) como auditor probabilístico exato

**Local:** `pff/validators/pc/npc.py`
Implementação de **Neural Probabilistic Circuit (NPC)** com backbone HCLT e inferência exata.

**Integração no DSLFM:** `pff/validators/dslfm/dslfm_kgc.py`

- O DSLFM instancia `NeuralProbabilisticCircuit` quando `config.lambda_pc > 0`.
- O PC contribui via `pc_log`/`pc_penalty` em scoring/loss.

**Importante para o objetivo:** o PC2 pode ser usado como **auditor neuro‑simbólico** (probabilidade exata) para validar/contradizer sinais do DSLFM.

### 1.3 `pff learn` (pipeline principal)

**Local:** `pff/cli.py`
O comando `learn` já usa Strategy/Factory para selecionar a execução:

- `KGCTrainingStrategy`: treina DSLFM‑KGC com dados preprocessados de KG.
- `FullPipelineStrategy`: preprocess KG + DSLFM‑KGC.

**Hoje** o `learn` treina o modelo, salva checkpoints e reporta métricas de treino/validação, mas não fecha o ciclo completo de:

- calibrar scores para probabilidade global,
- estimar limiar estatístico robusto (EVT),
- produzir relatórios de anomalias,
- persistir artefatos de calibração/EVT com contratos estáveis para inferência.

### 1.4 HPO (Optuna) + score composto + live metrics

**Locais principais:**

- Trial pipeline: `scripts/optimization/trials/pipeline.py`
- Treino/eval: `scripts/optimization/trials/evaluator.py`
- Score composto: `scripts/optimization/trials/scoring.py`
- Seleção multi‑objetivo: `scripts/optimization/trials/selection.py`
- Live plots: `scripts/optimization/callbacks.py` (`LivePlotCallback`)

**O que já existe (relevante):**

- `scoring.py` já separa **rank vs classification vs efficiency** e suporta pesos via config.
- `evaluator.py` já computa métricas binárias (AUC/PR‑AUC/F1 etc) via corruptions.
- `LivePlotCallback` já plota MRR/Hits + AUC/PR‑AUC/F1 etc (métricas absolutas).

---

## 2) Diagnóstico: por que ranking forte e classificação fraca coexistem

Você descreveu um caso típico: **MRR/Hits altos**, mas **AUC/PR‑AUC ~ aleatório** e threshold muito baixo.

Isso acontece quando:

1. **Ranking é “por consulta”** (ex.: dado `(h, r, ?)` ordenar todos os `t`).

   Métricas como MRR/Hits não exigem que o score seja comparável entre relações; basta ordenar corretamente *dentro* de cada consulta.

2. **Classificação é “global”** (um threshold único ou poucos thresholds).

   AUC/PR‑AUC/F1 exigem que scores de triplas de **relações diferentes** sejam comparáveis.

3. **Distribuições de score variam por relação/direção**:

   Relações com base rate diferente, dificuldade diferente e semânticas diferentes geram logits com offsets/escala diferentes.

4. **OWA vs CWA (mundo aberto)**: negativos gerados por corrupção não são “falsos garantidos” em KGs reais; isso gera ruído nos rótulos binários e piora PR‑AUC.

**Consequência prática:** calibração global simples (ex.: sigmoide única) pode melhorar “probabilidade” e thresholds, mas **não necessariamente melhora AUC**, porque AUC é rank‑based. Para melhorar AUC/PR‑AUC em cenário multi‑relação, você precisa:

- **normalizar/calibrar por grupo** (por relação/direção), e/ou
- usar um **meta‑classificador** que incorpore features adicionais (embeddings, incerteza, PC2, etc) para reordenar exemplos entre relações.

---

## 3) Princípios de engenharia (AGENTS.md aplicado ao plano)

### 3.1 Config‑first

Todo knob novo (calibração, EVT, LR, thresholds, amostragem de anomalias, etc) vai para `config/*.yaml` (ex.: `config/models/dslfm.yaml` e `config/hpo/optimization.yaml`) e é carregado via `FileManager`.

### 3.2 Utils‑first (infra)

Tudo que for:

- persistência (artefatos de calibrador/EVT),
- cache (ex.: estatísticas por relação, global negatives se aplicável),
- hashing (chaves estáveis),
- paralelismo (se for necessário),

deve estar em `pff/utils/**` (ou usar o que já existe lá).

### 3.3 Determinismo por padrão

HPO já usa `stable_hash` + `set_global_seed` em `scripts/optimization/trials/pipeline.py`.
Para este roadmap, a regra é:

- determinismo ativado por default no pipeline;
- qualquer fonte de não‑determinismo (CUDA kernels, sampling, LR steps) precisa de guardrails e evidência em benchmark/teste.

### 3.4 Outputs‑only

Artefatos previstos:

- modelos/checkpoints: `outputs/dslfm_kgc/**` e `outputs/optimization/**`
- calibradores/EVT/anomalias: `outputs/dslfm_kgc/anomaly/**` (proposto)
- logs: `logs/**` e (quando aplicável) espelhado em `outputs/logs/**`.

### 3.5 Logging contract

Este documento não é log, mas toda implementação deve:

- `info/success`: PT‑BR (ex.: `logger.info("calibracao_concluida n=...")`)
- `warning/error/exception`: EN (ex.: `logger.error("EVT fit failed: ...")`)

---

## 4) Arquitetura alvo (camadas, padrões e contratos)

### 4.0 A peça que faltava: contratos explícitos para “laudo JSON” (camadas 0–2)

O roadmap original estava forte na **camada 3 (semântica em grafo)**, mas o produto final (“apontar trechos corrompidos no JSON”) exige fechar o loop:

- **Camada 0: Canonicalização + proveniência** (cada fato tem origem rastreável via JSON Pointer).
- **Camada 1: Validação mecânica** (JSON Schema, erros com paths).
- **Camada 2: Perfil estatístico + histórico** (baseline “museu dos saudáveis”, drift e anomalias por campo).
- **Camada 3: Semântica em grafo** (DSLFM‑KGC + calibração + EVT/POT + auditor PC2 + política).

Sem as camadas 0–2, você encontra uma anomalia no grafo… mas não consegue responder “onde no JSON eu mexo?” nem justificar com evidência comparável em produção.

```mermaid
flowchart TD
  J[JSON bruto] --> C0[Camada 0: canonicalizar + provenance (JSON Pointer)]
  C0 --> C1[Camada 1: validar JSON Schema]
  C0 --> C2[Camada 2: perfil estatístico + drift vs baseline]
  C0 --> G[Camada 3: extrair triplas (com provenance)]
  G --> N[DSLFM-KGC score + sinais VAE]
  N --> Cal[calibração (global/per-relation)]
  Cal --> EVT[EVT/POT (threshold + p-value)]
  N --> PC[PC2 auditor (exact prob)]
  C1 --> D[Política/decisão]
  C2 --> D
  EVT --> D
  PC --> D
  D --> R[Laudo final JSON + reparos sugeridos]
```

### 4.1 Contratos mínimos (input → artefatos → laudo)

#### 4.1.1 Input do auditor (documento)

O pipeline precisa tratar cada input como um **artefato versionado**:

- `document_id` (hash estável via `pff/utils/hash.py:stable_hash`)
- `schema_id` / `schema_version`
- `source_system` / `ingest_timestamp`

Isso garante reprodutibilidade e permite histórico comparável (drift, regressões).

#### 4.1.2 Canonicalização + Proveniência (Camada 0)

Objetivo: gerar um “formato canônico” que preserve 100% rastreabilidade.

**Fonte de verdade (PostgreSQL, recomendado para evitar arquivos intermediários):**

- `audit_runs`: `{run_id, document_id, baseline_id, meta}`
- `audit_canonical_records`: `{run_id, record_hash, json_pointer, field_path, key, value_type, normalized_value, raw_value}`
- `audit_triples`: `{run_id, triple_hash, s, p, o, json_pointer, record_hash}`

**Exports opcionais (somente quando necessário para debug/inspeção):**

- `outputs/audit/<run_id>/canonical/` pode conter dumps derivados (ex.: amostras JSONL/Parquet), mas o pipeline deve operar por default via PostgreSQL.

**Decisão importante:** toda tripla/fato precisa carregar `json_pointer` para fechar o ciclo “onde mexer”.

**Padrões recomendados:**

- JSON Pointer (RFC 6901) como formato de path
- JSON Patch (RFC 6902) para reparos sugeridos

#### 4.1.3 Validação mecânica (Camada 1)

Aplicar JSON Schema como “primeiro juiz”: rápido, determinístico e com paths rastreáveis.

**Fonte de verdade (PostgreSQL):**

- `audit_schema_reports`: `run_id`, `schema_id`, `schema_version`, `report` (JSONB)

**Exports opcionais (somente quando necessário):**

- `outputs/audit/<run_id>/schema/schema_report.json` com erros no formato:
  - `error_code`, `message`, `json_pointer`, `validator`, `validator_value`, `instance_snippet`

O laudo final deve sempre incluir os erros de schema (se existirem), porque isso costuma ser o “root cause” mais barato de corrigir.

#### 4.1.4 Perfil estatístico + baseline histórico (Camada 2)

Construir um “museu dos saudáveis”: estatísticas do passado/treino para comparar novos documentos.

**Fonte de verdade (PostgreSQL):**

- `audit_profile_baselines`: `baseline_id`, `profile` (JSONB), `digest` (JSONB)
- `audit_run_profiles`: `run_id`, `profile_current` (JSONB), `drift` (JSONB)

**Exports opcionais (somente quando necessário):**

- `outputs/audit/baselines/<baseline_id>/profile_baseline.json` + `profile_digest.json`
- `outputs/audit/<run_id>/profile/profile_current.json` + `profile_drift.json`

#### 4.1.5 Semântica em grafo (Camada 3)

Mantém o núcleo do roadmap:

- DSLFM‑KGC raw scores + sinais VAE
- calibração por relação + métricas (ECE/Brier/NLL)
- EVT/POT para thresholds robustos
- PC2 como auditor probabilístico exato
- política de decisão configurável

**Saídas em `outputs/audit/<run_id>/graph/`:**

- `graph_findings.json` (anomalias por tripla com evidências)
- `graph_repairs.json` (reparos sugeridos no espaço do grafo)

#### 4.1.6 Laudo final (contrato “produto”)

O laudo final é um JSON único para consumo por sistemas e por humanos.

**Saída em `outputs/audit/<run_id>/report/`:**

- `audit_report.json`

**Schema lógico sugerido (alto nível):**

- `meta`: ids, versões, paths, seeds, determinismo, duração
- `findings[]`: cada achado com:
 - `severity` (`info | warning | error`)
- `layer` (`schema | profile | graph | neuro_symbolic`)

  - `json_pointer` (obrigatório quando aplicável)
  - `entity_refs` (ids/labels do grafo quando aplicável)
  - `evidence` (scores calibrados, p‑value EVT, PC2 NLL, drift metrics, etc.)
  - `broken_invariants[]` (nomes + evidência)
  - `suggested_repairs[]` (JSON Patch ops + justificativa + impacto esperado)
- `summary`: contagens, top causas, top pointers afetados

O ponto central: **toda decisão do modelo precisa virar um achado rastreável com pointer e proposta de ação**.

#### 4.1.7 Schema formal do laudo (versão e retrocompatibilidade)

Para evitar drift entre produtores/consumidores, o contrato do laudo deve ser **formal e versionado**.

- Schema oficial (v1): `config/audit/audit_report.schema.v1.json`
- Campo obrigatório no laudo: `schema_version: 1`

Regras de compatibilidade:

- Adição de novos campos: sempre opcional e com defaults bem definidos.
- Remoção/renomeação: somente com bump de versão (`schema_version: 2`) e migração explícita.
- Consumidores devem validar `audit_report.json` contra o schema antes de ingerir.

#### 4.1.8 “Culpados mínimos”: algoritmo prescritivo para causa‑raiz (2–3 pointers)

O laudo precisa diferenciar:

- **sintomas** (muitos achados correlacionados) vs
- **causas acionáveis** (poucos `json_pointer` que explicam a maioria dos impactos).

Proposta de algoritmo (determinístico, rápido, auditável):

1. **Coletar candidatos** (poucos, por camada):
   - Schema: cada erro já traz `json_pointer` (candidato direto).
   - Perfil: selecionar top‑N campos por `drift_score` (ex.: PSI/JS/KS) com `json_pointer` do campo.
   - Grafo: agrupar achados por `json_pointer` (via proveniência das triplas) e calcular:
     - `impact = sum(severity_weight * anomaly_score)` por pointer
     - selecionar top‑N pointers por impacto.

2. **Reduzir para conjunto mínimo** via ablation gulosa (2–3 passos):
   - Defina uma função de risco agregada `R(report)` (ex.: soma ponderada de severidades + EVT tail p‑values + penalidade PC2).
   - Para cada candidato `p` (pointer), simule “remover/neutralizar” os fatos associados:
     - no espaço do grafo: mascarar/remover triplas cujo `json_pointer==p` e recomputar apenas os sinais necessários (cacheando embeddings/scores quando possível);
     - no espaço do schema/perfil: marcar como “corrigido” e recomputar métricas derivadas.
   - Escolha o pointer que mais reduz `R` (`ΔR` máximo), fixe no conjunto, e repita até:
     - `R` cair abaixo de um limiar configurado, ou
     - atingir `k_max=3`.

3. **Emitir no laudo**:
   - `summary.root_causes[]` com `{json_pointer, delta_risk, layers_impacted, evidence}`.

Observações:

- Este método não depende de inferências difíceis (ex.: marginal MAP) para funcionar.
- Quando PC2 estiver disponível, `R` pode incluir termos do PC2 (ex.: NLL) para priorizar violações “semânticas”.

### 4.2 Visão de alto nível

O objetivo é manter o **núcleo DSLFM‑KGC** e adicionar um **wrapper pós‑treino** que constrói um “modelo de decisão” para classificação/anomalia, sem degradar ranking.

```mermaid
flowchart TD
  A[Triples (train/valid/test)] --> B[DSLFM-KGC Training]
  B --> C[Raw Scoring + Latent Signals]
  C --> D[Calibration (global/per-relation)]
  D --> E[Anomaly Score (neg_log_prob / recon / KL / LR)]
  E --> F[EVT POT (threshold + tail p-value)]
  C --> G[PC2/NPC Auditor (exact prob)]
  F --> H[Decision Policy (matrix + costs)]
  G --> H
  H --> I[Artifacts + Reports (outputs/)]
  H --> J[Metrics (live + JSON)]
```

### 4.3 Componentes (reuso + novos)

#### (A) Extrator de sinais do DSLFM (novo, DSLFM‑specific)

**Responsabilidade:** dado um lote de triplas, extrair:

- score bruto do KGC (para ranking),
- sinais gerativos: reconstrução, KL (gaussian/IBP), incerteza,
- sinais PC2 (log‑prob / NLL / “consistência”),
- (opcional) likelihood regret.

**Sugestão de módulo (novo):**

- `pff/validators/dslfm/anomaly_signals.py` (Facade/Adapter)

#### (B) Calibração (reuso; mover para utils na implementação)

**Reuso existente:** `pff/validators/kg/calibration.py` (`ScoreCalibrator`, `find_optimal_threshold`).

**Objetivo da calibração aqui:** produzir probabilidades comparáveis globalmente e por grupo, suportando:

- `global` (único calibrador),
- `per_relation` (um calibrador por relação),
- `per_relation_direction` (se existir direção head/tail no protocolo).

**Sugestão de refactor (na implementação):**

- criar `pff/utils/ml/calibration.py` e mover/consolidar o calibrador para evitar duplicação e permitir uso por `pff learn` e HPO.

#### (C) EVT (novo, utils/ml)

**Responsabilidade:** limiar robusto baseado em cauda (POT/GPD) para:

- retornar threshold por grupo,
- retornar p‑value/risco de anomalia,

com determinismo e persistência via `FileManager`.

**Sugestão de módulo (novo):**

- `pff/utils/ml/evt.py` com Strategy/Builder para POT.

#### (D) Auditor neuro‑simbólico (reuso)

**Reuso existente:**

- PC2: `pff/validators/pc/npc.py`
- lógica diferenciável: `pff/validators/dslfm/logic_layer.py` (quando fizer sentido)
- agregação PC fallback: `pff/validators/pc/strategy.py` (para cenários de agregação)

**Objetivo aqui:** produzir um sinal de consistência lógica (probabilidade exata do PC2, violações/discordâncias).

#### (E) Política de decisão (novo, config‑driven)

**Responsabilidade:** combinar sinais em uma decisão final (ex.: NORMAL / ANOMALIA_ESTATISTICA / ANOMALIA_LOGICA / FALSO_POSITIVO_NEURAL).

**Padrão recomendado:** Strategy (policy pluggable) + Command (emissão de ações).

#### (F) Reporting (reuso)

**Reuso existente:**

- Observers: `pff/utils/performance/training_observer.py`
- DSLFM metrics reporter: `pff/validators/dslfm/metrics_reporter.py`
- HPO live metrics: `scripts/optimization/callbacks.py`

Objetivo: persistir artefatos e métricas com schema estável.

---

## 5) Calibração e classificação (do score para probabilidade confiável)

### 5.1 Por que “calibrar” ajuda e quando não ajuda

- Calibração global monotônica (Platt/isotônica) melhora **interpretação probabilística** e **threshold**, mas pode não melhorar AUC (rank‑based).
- Para melhorar AUC/PR‑AUC global em multi‑relação, o caminho SOTA prático é:
  - **calibração por grupo** (relação/direção) e/ou
  - **meta‑classificador** usando features extras além do score bruto.

### 5.2 Estratégia recomendada (em ordem de ROI)

1. **Per‑relation Platt/Isotonic** com fallback global (mínimo de amostras por relação).
2. **Normalização por relação** (ex.: z‑score/quantis por relação) antes da calibração.
3. **Meta‑classificador leve** (LogReg/LGBM) com features:
   - score bruto DSLFM,
   - PC2 log‑prob/NLL,
   - KL/recon (se disponível),
   - embeddings (opcional, com cuidado de custo).

### 5.3 Métricas adicionais recomendadas (além de AUC/PR‑AUC/F1)

Para saber se a “probabilidade” é utilizável de verdade:

- **Brier score**
- **Log loss (NLL)**
- **ECE/MCE** (Expected/Maximum Calibration Error)
- **PR‑AUC baseline** (prevalência) logada junto para interpretação.

---

## 6) EVT (Peaks‑Over‑Threshold) para limiar robusto de anomalia

### 6.1 Ideia

Escolha um score de anomalia (ex.: `anomaly_score = -log(p_calibrada)`), e ajuste GPD na cauda alta (POT).

**Resultado do EVT:**

- threshold estatístico para um FPR alvo (ex.: 0.1%),
- p‑value (quão extremo é o caso),

sem heurística manual frágil.

### 6.2 Recomendação prática no KG

EVT funciona melhor quando:

- o conjunto usado para ajustar POT contém majoritariamente “normal”,
- você faz isso **por relação** (distribuições diferem).

### 6.3 Artefatos a persistir

**Fonte de verdade (PostgreSQL):**

- `audit_calibration_models`: calibradores `__global__` + por relação (JSONB + métricas)
- `audit_evt_params`: params EVT `__global__` + por relação (JSONB)

**Exports opcionais (somente quando necessário):**

- `outputs/dslfm_kgc/anomaly/evt/evt_params_global.json`
- `outputs/dslfm_kgc/anomaly/evt/evt_params_by_relation.json`
- `outputs/dslfm_kgc/anomaly/evt/evt_fit_report.json` (diagnóstico: quantil u, N_u, xi, beta, warnings)

---

## 7) Likelihood Regret (LR) e sinais gerativos do VAE

### 7.1 Sinais já alinhados com a arquitetura do DSLFM

Como o DSLFM já produz latentes (mu/logvar) e KLs, os seguintes sinais são naturais:

- erro de reconstrução (ou proxy),
- KL gaussiano (posterior vs prior),
- KL IBP / sparsity indicators,
- incerteza (variância latente).

### 7.2 Likelihood Regret (LR) como “SOTA add‑on”

LR é útil quando o decoder é expressivo e reconstrução simples falha.

**Trade‑off:** LR exige otimizar um latente por exemplo ⇒ caro.
Para manter performance:

- computar LR apenas em amostra (fixa e determinística) na avaliação/HPO,
- e/ou em modo “forense” quando `anomaly_score` acima do EVT threshold.

Artefatos: `outputs/dslfm_kgc/anomaly/lr/` com params e relatórios.

---

## 8) Validação neuro‑simbólica (PC2 + regras) como matriz de decisão

PC2 fornece um segundo “modelo de verdade” probabilístico e exato.

### 8.0 Invariantes declarativas no grafo (SHACL/ShEx) como “camada simbólica determinística”

Além das regras mineradas e do PC2, para auditoria declarativa “de engenharia” vale adotar constraints explícitas:

- **SHACL** (W3C) para shapes e validação de constraints (cardinalidade, domínios, ranges, caminhos).
- **ShEx** para validação de expressões de shape (útil quando você quer contratos compactos).

No laudo, isso entra como um tipo de achado `layer=graph` (determinístico) que:

- identifica a violação
- aponta o(s) `json_pointer` associado(s) às arestas envolvidas
- sugere reparos (remover aresta, corrigir tipo, completar campo obrigatório)

#### 8.0.1 Mapeamento formal SHACL report → `json_pointer`

O SHACL normalmente produz um validation report com campos como:

- `sh:focusNode` (nó com problema)
- `sh:resultPath` (propriedade/caminho violado)
- `sh:value` / `sh:sourceConstraintComponent` / `sh:message`

Para fechar JSON→grafo→JSON, a camada 0 precisa persistir (por execução) tabelas de mapeamento:

- (Recomendado) PostgreSQL:
  - `audit_node_provenance` (a implementar no Sprint 4): `run_id`, `node_id`, `node_label`, `document_id`, `json_pointers` (JSONB)
  - `audit_triples` (já contém proveniência de aresta): `run_id`, `s`, `p`, `o`, `record_hash`, `json_pointer`

Regra de mapeamento:

- se o `sh:focusNode` corresponder a um `node_id`, use `node_provenance` para obter pointers candidatos;
- se o report apontar um caminho/propriedade equivalente à aresta (`resultPath`), refinar para pointers em `edge_provenance` compatíveis com `(focusNode, resultPath, value)`;
- todo achado SHACL no laudo deve incluir:
  - `json_pointer` (quando possível)
  - `entity_refs` (para auditoria do grafo)
  - `broken_invariants[]` com o shape/constraint componente.

### 8.1 Matriz de decisão recomendada (auditável)

| Sinal neural (DSLFM/EVT) | Auditor (PC2)       | Decisão               | Uso                 |
| ------------------------ | ------------------- | --------------------- | ------------------- |
| Normal                   | Suporta             | NORMAL                | seguir              |
| Anômalo (EVT)            | Violação/baixa prob | ANOMALIA_CRITICA      | alta prioridade     |
| Anômalo (EVT)            | Suporta alto        | FALSO_POSITIVO_NEURAL | revisão humana      |
| Normal                   | Violação forte      | ANOMALIA_LOGICA       | corrigir regra/dado |

### 8.2 Métricas neuro‑simbólicas úteis

- taxa de discordância DSLFM vs PC2
- top‑K anomalias com maior conflito
- correlação entre LR alto e PC2 baixo (sanidade)

---

## 9) Mudanças necessárias (por subsistema)

### 9.1 `pff learn` (pipeline principal)

**Onde:** `pff/cli.py` (Strategy) + `pff/validators/dslfm/kgc_manager.py` (train)
**Mudança proposta:** após `manager.train(...)`, executar estágio pós‑treino:

1. gerar dataset de calibração (positivos vs corruptions) com determinismo,
2. ajustar calibrador (global + por relação),
3. construir EVT (global + por relação),
4. produzir relatório de anomalias (top‑K) para `valid/test`,
5. persistir artefatos e métricas em `outputs/dslfm_kgc/anomaly/**`.

**Importante:** ranking (MRR/Hits) deve continuar calculado no protocolo atual, sem depender do calibrador.

### 9.2 HPO (Optuna) – objetivo e métricas

**Onde:** `scripts/optimization/trials/evaluator.py` + `scripts/optimization/trials/scoring.py`
**Mudanças propostas:**

- Computar classificação “global” e “por relação” (quando viável) no evaluator.
- Adicionar métricas de calibração (Brier/ECE) e EVT (p‑value stats) ao `user_attrs`.
- Atualizar `scoring.py` para incorporar um **anomaly/calibration block** (ou enriquecer o bloco classification), mantendo pesos via config.

### 9.3 Live metrics (plots em tempo real)

**Onde:** `scripts/optimization/callbacks.py` (`LivePlotCallback`)
**Mudanças propostas:**

- Expandir schema de métricas para incluir:
  - `ece`, `brier`, `logloss`
  - `anomaly_pr_auc`, `anomaly_precision_at_k`, `evt_threshold_q`, `evt_tail_mean`
  - métricas por relação agregadas (ex.: média ponderada por suporte)
- Garantir que o callback continue plotando métricas absolutas (0..1) sem normalização relativa.

### 9.4 Score composto (seleção de trials)

**Onde:** `scripts/optimization/trials/scoring.py` + `scripts/optimization/trials/selection.py`
**Mudanças propostas:**

- Adicionar pesos explícitos para:
  - ranking (MRR/Hits),
  - classificação (AUC/PR‑AUC/F1),
  - calibração/anomalia (ECE/Brier/EVT‑precision@K),
  - eficiência (tempo).
- Garantir compatibilidade: se métrica nova não existir, score degrade de forma determinística (via defaults em config), sem “optional metrics” escondidas.

---

## 10) Configuração proposta (YAML) – schema sugerido

Adicionar em `config/models/dslfm.yaml` (exemplo de schema, valores ilustrativos):

```yaml
anomaly:
  enabled: true
  score:
    strategy: neg_log_prob   # neg_log_prob | recon_kl | likelihood_regret | hybrid
    hybrid_weights:
      neg_log_prob: 1.0
      recon_error: 0.3
      kl_gaussian: 0.1
  calibration:
    scope: per_relation       # global | per_relation | per_relation_direction
    method: isotonic          # platt | isotonic | both
    min_samples_per_group: 500
    cv_folds: 5
  evt:
    enabled: true
    scope: per_relation
    pot_quantile_u: 0.98
    target_fpr: 0.001
    min_tail_samples: 200
  decision:
    top_k_report: 200
    cost_matrix:
      fp: 1.0
      fn: 5.0
      anomaly_fp: 2.0
      anomaly_fn: 10.0
```

**Regra:** nenhum valor hardcoded no código final; tudo lido via `FileManager`.

---

## 11) Protocolo de dados (classificação + anomalias) sem depender de dados reais

### 11.1 Geração de negativos (CWA local, determinístico)

Para validação binária:

- positivos: triplas reais do split `valid/test`
- negativos: corrupções controladas:
  - aleatória (baseline),
  - type‑aware (mesma “família”/comunidade),
  - hard negatives (próximos no embedding/ANN).

#### 11.1.1 Protocolo prescritivo de calibração (negativos/corrupções)

Para calibrar probabilidade e thresholds (sem leakage), o protocolo precisa ser fixo e reproducível.

Conjuntos (propostos):

- `calibration_fit`: derivado de `train` (ou `train+valid` quando permitido) e **nunca** do `test`.
- `calibration_eval`: subconjunto holdout (ex.: 10%) para medir Brier/ECE/NLL sem reusar exemplos.
- `anomaly_eval`: `valid/test` (somente avaliação e laudo).

Geração de negativos por positivo (por relação):

- Para cada tripla positiva `(h, r, t)`:
  - gere `k_tail` corrupções em cauda e `k_head` corrupções em cabeça, com proporção fixa:
    - `p_tail = 0.5`, `p_head = 0.5` (configurável)
  - amostre entidades de `entity_pool`:
    - por default: entidades vistas em `train` (evita vazamento por entidade “nova”)
    - opcional: entidades vistas em `train+valid` (modo experimental/config)
  - rejeite corrupções que colidam com um conjunto de “triplas conhecidas” (filtered setting), quando disponível.

Hard negatives (opcional e determinístico):

- selecionar `k_hard` entidades candidatas por proximidade (ANN) e amostrar deterministamente via seed
- usar hard negatives apenas na calibração (não no treino), para melhorar separação global

Seeds e determinismo:

 - `seed_calibration = stable_hash(document_id | baseline_id | schema_version | split_id)` (truncate 32 bits)
- RNG único por etapa (fit/eval) e por relação para evitar variação “intermitente”.

#### 11.1.2 Protocolo prescritivo de EVT/POT

EVT deve ser ajustado sobre exemplos majoritariamente “normais” e sem leakage:

- **Fit** do EVT: use apenas `calibration_fit` (ou um “normal pool” derivado do histórico).
- Score de anomalia recomendado: `anomaly_score = -log(p_calibrada)` (ou híbrido config‑driven).
- POT threshold `u`: quantil alto fixo (ex.: `0.98`) por relação, com `min_tail_samples` mínimo.
- Reportar: `u`, `N_u`, `xi`, `beta`, e diagnósticos de estabilidade (warnings) por relação.

O EVT não deve ser “auto‑tuned” no `test`: o `test` é somente para avaliação/laudo.

### 11.2 Injeção de anomalias sintéticas (para medir de verdade)

Criar um fixture pequeno em `tests/fixtures/**` e gerar:

- anomalias estruturais (ligações cross‑community),
- anomalias semânticas (atributos conflitantes, quando aplicável),
- anomalias “difíceis” (mesmo tipo, relação inválida).

Métricas de sucesso: PR‑AUC e Precision@K de anomalia (o analista só olha top‑K).

---

## 12) Plano de testes e benchmarks (gates)

### 12.1 Testes (unit/integration)

Adicionar/ajustar testes para cobrir:

- determinismo do pipeline (mesma seed ⇒ mesmos artefatos principais),
- calibrador por relação + fallback global,
- EVT fit determinístico e thresholds estáveis,
- schema de métricas no HPO (user_attrs com chaves estáveis),
- live metrics não quebra com métricas novas.

### 12.2 Benchmarks (antes/depois)

Benchmarks devem ir para `outputs/benchmarks/**` e medir:

- tempo de época (train),
- tempo de avaliação (ranking + classificação),
- custo extra do pós‑treino (calibração/EVT/LR),

com logs estruturados.

---

## 13) Plano de execução (milestones sugeridos)

1. **Baseline:** medir e registrar (tempo/época + métricas atuais) com um dataset sintético (fixtures).
2. **Per‑relation calibration:** implementar + métricas (ECE/Brier/threshold) + persistência.
3. **EVT POT:** implementar + relatório top‑K + determinismo.
4. **PC2 auditor:** integrar no relatório/decisão (sem mexer no ranking).
5. **HPO + live metrics + score composto:** ampliar métricas e score; validar com testes.
6. **Hardening/perf:** otimizar hot paths, revalidar com benches e evitar regressões de MRR.

### Sprints (checklists)

#### Sprint 0 — Contratos + baseline de execução (fundação)

- [x] Formalizar o contrato do laudo: validar `audit_report.json` contra `config/audit/audit_report.schema.v1.json` em runtime (fail‑fast, EN error).
- [x] Padronizar `run_id`, `document_id`, `baseline_id` via `pff/utils/hash.py` (stable keys) e persistir em `meta`.
- [x] Fixar layout de artefatos sob `outputs/audit/<run_id>/**` (sem paths ad‑hoc) e documentar no código via docstrings.
- [x] Adicionar um “smoke” determinístico de geração de laudo (fixture pequeno) e checar invariantes de schema_version/paths.
- [x] Registrar baseline “antes” (tempo/época + métricas atuais) em `outputs/benchmarks/**` com logs estruturados.

#### Sprint 1 — Camada 0 (canonicalização + proveniência)

- [x] Implementar canonicalização JSON→records com `json_pointer` (RFC 6901) e `record_hash` estável.
- [x] Implementar geração de triplas canônicas (`s=document_id`, `p=field_path`, `o=normalized_value`) carregando `json_pointer` em cada fato/aresta.
- [x] Persistir proveniência em PostgreSQL (`audit_canonical_records`, `audit_triples`) para fechar JSON→grafo→JSON (sem Parquet por default).
- [x] Garantir determinismo (mesmo JSON ⇒ mesmas triplas/hashes) e adicionar teste de propriedade.

#### Sprint 2 — Camadas 1–2 (schema mecânico + perfil estatístico + drift)

- [x] Validar JSON Schema (camada 1) e produzir `schema_report` com `json_pointer` e campos de depuração (persistência em PostgreSQL).
- [x] Construir baseline estatístico (“museu dos saudáveis”) por campo/pointer e persistir `profile` + `digest` em PostgreSQL.
- [x] Calcular drift vs baseline com métricas por tipo (numérico/categórico/missingness) e persistir em PostgreSQL.
- [x] Mapear achados de schema/perfil para `findings[]` no laudo e consolidar `summary.counts` via Builder.

#### Sprint 3 — Camada 3 (calibração + classificação + EVT/POT)

- [x] Implementar protocolo prescritivo de negativos/corrupções com seeds estáveis por split e por relação (determinístico).
- [x] Implementar calibração `global` + `per_relation` e calcular Brier/ECE/NLL (com config em `config/audit/audit.yaml`).
- [x] Implementar EVT/POT por relação: ajustar GPD na cauda e persistir params em PostgreSQL (`audit_evt_params`) com fallback `__global__`.
- [x] Emitir achados neuro‑simbólicos por tripla: `p_calibrada`, `anomaly_score`, `evt_p_value` e builders de findings.

#### Sprint 4 — Auditoria determinística (PC2 + SHACL) + explicação mínima

- [x] Integrar PC2 como auditor: expor helper para `log_prob`/NLL por tripla (evidência determinística) e permitir sinalizar discordâncias.
- [x] Adicionar constraints determinísticas no grafo (SHACL-like) e consumir validation report em formato estável.
- [x] Implementar mapeamento SHACL→`json_pointer` usando proveniência em `audit_triples` (json_pointer por aresta).
- [x] Implementar “culpados mínimos”: Builder determinístico para `summary.root_causes[]` (greedy coverage com risco).

#### Sprint 5 — Reparos sugeridos + integração (`pff learn` + HPO + live metrics)

- [x] Gerar `suggested_repairs[]` em JSON Patch (RFC 6902) e validar patch contra JSON Schema antes de recomendar (fail‑closed).
- [x] Persistir o laudo em PostgreSQL (`audit_reports`) e manter export opcional em `outputs/audit/**` quando necessário.
- [x] BusinessService/API: adicionar entrypoint `BusinessService.audit_document` (Postgres-first) e corrigir contract do FastAPI deps (`pff/api/deps.py`).
- [x] HPO: propagar métricas adicionais de calibração (`brier`, `nll`, `ece`, `f1`, `decision_threshold`) no `user_attrs` e payload de live metrics.
- [x] Bench/determinismo: cobrir determinismo/contratos via testes rápidos (utils) e manter gates de regressão por suite alvo.

---

## 14) Riscos e mitigação

- **Cache de negativos globais:** pode reduzir diversidade ⇒ risco de MRR. Mitigar com A/B e cache “por janela” (TTL) e/ou cache apenas de candidatos, não de labels.
- **EVT mal ajustado:** se “normal” contém muitas anomalias, cauda engana. Mitigar com robustez por relação e relatórios de ajuste.
- **Triton pode piorar:** manter gating por benchmark e fallback Torch sempre disponível.
- **Leak de calibração:** calibrar em cima do conjunto de teste invalida métricas. Mitigar com split claro e persistência de artefatos por split.
- **Reparo sugerido pode quebrar schema:** sempre validar JSON Patch contra JSON Schema antes de recomendar como “seguro”.
- **Conformal/garantias mal calibradas:** qualquer “garantia” deve ser calculada em conjunto de calibração separado e persistida como artefato com versão/seed.

---

## 15) Definition of Done (DoD)

- Dado um JSON novo, o pipeline gera um **laudo acionável**:
  1. aponta `json_pointer`(s) suspeitos,
  2. lista invariantes quebrados (schema/perfil/SHACL/PC2/EVT),
  3. inclui evidência histórica (baseline + drift),
  4. sugere reparos (JSON Patch) com validação mecânica,
  5. reporta score calibrado + risco EVT + auditoria PC2.
- BusinessService (API) e HPO persistem artefatos com schema estável em PostgreSQL, com export opcional em `outputs/audit/**` (sem caminhos ad‑hoc).
- HPO otimiza com score composto que inclui classificação/anomalia sem matar MRR/Hits.
- Live metrics exibe métricas novas sem normalização errada e sem spam.
- Testes rápidos relevantes passam (comandos registrados em `CONTEXT.md`).

---

## 16) Referências (reais, reproduzíveis)

- RFC 6901 (JSON Pointer): https://www.rfc-editor.org/rfc/rfc6901
- RFC 6902 (JSON Patch): https://www.rfc-editor.org/rfc/rfc6902
- JSON Schema (especificação): https://json-schema.org/
- W3C SHACL validation report (modelo de resultado): https://www.w3.org/TR/shacl/#validation-report
- ShEx (Shape Expressions): http://shex.io/
- scikit‑learn `IsotonicRegression` (calibração): https://scikit-learn.org/stable/modules/isotonic.html
- SciPy `scipy.stats.genpareto` (POT/GPD): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html
