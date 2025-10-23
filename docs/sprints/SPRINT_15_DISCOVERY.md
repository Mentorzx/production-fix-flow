# Sprint 15: Descoberta Crítica - Ensemble Base Score Constante

**Data:** 2025-10-23 00:30 BRT
**Status:** 🔴 CRITICAL FINDING
**Bug:** #4 (Scores Constantes) - STATUS REDUZIDO DE "CORRIGIDO" PARA "PARCIALMENTE CORRIGIDO"

---

## 🚨 Descoberta Crítica

Durante validação final do Sprint 15, descobrimos que o **Ensemble base score continua constante (0.3906)** independentemente do input, mesmo após todas as correções implementadas.

### Evidência

**Teste Realizado:**
```bash
$ pff run data/manifest.yaml
# Manifest contém 2 tasks: test.json e test1.json
```

**Resultados:**

| Input | Triplas | Violations | Base Score | Penalty | Final Score |
|-------|---------|------------|------------|---------|-------------|
| test.json | 1,125 | 156 | **0.3906** | -0.1982 | 0.1924 |
| test1.json | 294 | 158 | **0.3906** | -0.2007 | 0.1898 |

**Diferença de input:** 382% mais triplas (1125 vs 294)
**Diferença no base score:** 0.0000 (0.0%) ❌

---

## 🔍 Análise da Causa Raiz

### O Que Funciona (✅)

1. **Violations são detectadas corretamente:**
   - test.json: 156 violations
   - test1.json: 158 violations

2. **Penalty varia conforme violations:**
   - 156 violations (1.98%) → penalty -0.1982
   - 158 violations (2.01%) → penalty -0.2007

3. **Score FINAL varia:**
   - test.json: 0.1924
   - test1.json: 0.1898
   - **Diferença:** 0.0026 (1.4%) ✅

### O Que NÃO Funciona (❌)

1. **Ensemble base score é CONSTANTE:**
   - Sempre retorna 0.3906
   - Ignora diferença de 382% no número de triplas
   - Ignora conteúdo completamente diferente dos JSONs

2. **Symbolic features sempre 0:**
   - Log: "🔍 Symbolic Analysis: 0 regras ativas"
   - Impossível com 156 violations detectadas!
   - SymbolicFeatureExtractor não tem acesso às regras

3. **Features extraídas são constantes:**
   - "✅ Features: 24305 → 155 agrupadas"
   - Mesmo número de features para ambos os inputs
   - Provavelmente valores default/zeros

---

## 🧩 Por Que o Ensemble Ignora os Inputs?

### Arquitetura Atual (QUEBRADA)

```
Business Service
├─> Valida regras ✅ (156 violations detectadas)
├─> Extrai triples ✅ (1125 ou 294 triplas)
└─> Chama Ensemble.predict_proba([triples]) ❌

Ensemble sklearn Pipeline
├─> SymbolicFeatureExtractor.transform([triples])
│   └─> self.rules_ está VAZIO ❌
│   └─> Retorna features ZEROS (24305 → 155)
│
├─> TransEFeatureExtractor.transform([triples])
│   └─> Provavelmente também retorna valores constantes
│
├─> LightGBMFeatureExtractor.transform([triples])
│   └─> Recebe features ZEROS do Symbolic
│   └─> Também retorna valores constantes
│
└─> Meta-learner.predict(features)
    └─> Features SEMPRE iguais = Score SEMPRE 0.3906 ❌
```

### Diagrama do Problema

```
Input 1: 1125 triplas → [zeros] → Ensemble → 0.3906
Input 2: 294 triplas  → [zeros] → Ensemble → 0.3906
                         ^^^^^
                      MESMO ARRAY DE FEATURES!
```

---

## 📊 Re-classificação dos Bugs

### Bug #4: Scores Constantes

**Status Anterior:** ✅ CORRIGIDO
**Status Atual:** ⚠️ PARCIALMENTE CORRIGIDO

**O que foi corrigido:**
- ✅ Score FINAL varia via penalty (0.1924 vs 0.1898)
- ✅ Penalty é calculada corretamente (rate-based)
- ✅ Violations são passadas para predict_hybrid_score()

**O que NÃO foi corrigido:**
- ❌ Ensemble base score CONTINUA constante (0.3906)
- ❌ Features extraídas são SEMPRE iguais
- ❌ Symbolic features SEMPRE zeros

**Causa raiz:** Bugs #2 e #3 ainda não corrigidos

---

## 🎯 Impacto da Descoberta

### No Sprint 15

**Antes da descoberta:**
- Pensamos que Bug #4 estava corrigido
- Score variava de 0.391 → 0.1924
- Atribuímos variação ao penalty

**Depois da descoberta:**
- Bug #4 apenas PARCIALMENTE corrigido
- Variação vem APENAS do penalty, não do Ensemble
- Ensemble continua broken (retorna sempre 0.3906)

### Na Classificação do Sistema

**O que pensávamos:**
- XAI completo ✅
- Violations usadas corretamente ✅
- Scores variáveis ✅

**Realidade:**
- XAI completo ✅ (verdadeiro)
- Violations usadas via penalty ✅ (verdadeiro)
- Scores finais variáveis ✅ (verdadeiro via penalty)
- **Ensemble base BROKEN** ❌ (nova descoberta)

---

## 📝 Implicações

### 1. Sprint 15 Status

**Antes:** ✅ 3/5 bugs corrigidos (60%)
**Agora:** ⚠️ 2.5/5 bugs corrigidos (50%)

- ✅ Bug #1: Violations passadas (completo)
- ⚠️ Bug #4: Scores variam via penalty (parcial)
- ❌ Bug #2: Features dimensions (não corrigido)
- ❌ Bug #3: Symbolic features zeros (não corrigido)
- ❌ Bug #5: Arquitetura invertida (não corrigido)

### 2. Workaround Atual

**Enquanto Bugs #2 e #3 não forem corrigidos:**
- Ensemble base score = 0.3906 (constante, ignorado)
- Penalty varia conforme violations (funcional)
- Score final = 0.3906 - penalty (funcional mas subótimo)

**Problema:**
- Sistema depende 100% do penalty
- Ensemble ML não está contribuindo (apenas retorna constante)
- TransE, LightGBM, XGBoost não estão sendo usados efetivamente

### 3. Próximos Passos Críticos

**Sprint 16 - PRIORIDADE MÁXIMA:**

1. **Corrigir Bug #2 (Feature Dimensions)** - 6h estimadas
   - Remover SymbolicFeatureExtractor do Ensemble
   - Business Service extrai features unificadas
   - Ensemble recebe ndarray em vez de [triples]

2. **Corrigir Bug #3 (Symbolic Features)** - 2h estimadas
   - Implementar `_violations_to_binary_vector()`
   - Passar violation vector como features
   - Verificar que features variam entre inputs

3. **Validar Ensemble Base Varia** - 1h estimadas
   - Testar com test.json vs test1.json
   - Verificar que base score varia (não só final score)
   - Diferença esperada: >0.1 entre inputs válido/inválido

---

## 🔬 Testes de Validação

### Teste que EXPÕE o problema

```bash
# Execute com 2 inputs completamente diferentes
$ pff run data/manifest.yaml

# Grep apenas base scores
$ grep "Base Ensemble Score" logs/*.log

# RESULTADO ATUAL (ERRADO):
# Base Ensemble Score: 0.3906  (test.json - 1125 triplas)
# Base Ensemble Score: 0.3906  (test1.json - 294 triplas)

# RESULTADO ESPERADO (após Bug #2/#3 fix):
# Base Ensemble Score: 0.6500  (test.json - válido)
# Base Ensemble Score: 0.2800  (test1.json - inválido)
```

### Critério de Sucesso

**Sprint 16 será considerado sucesso quando:**
1. Base ensemble score varia entre inputs (diferença >0.1)
2. Symbolic features > 0 (não mais "0 regras ativas")
3. Features extraídas variam entre inputs
4. LightGBM, TransE contribuem com scores diferentes

---

## 📚 Referências

### Logs de Evidência

- `/tmp/pff_ensemble_comparison.log` - Teste com test.json e test1.json
- `/tmp/pff_penalty_test_v3.log` - Teste com penalty corrigida

### Documentos Relacionados

- `SPRINT_15_BUGS.md` - Bug #2 documenta feature dimensions
- `SPRINT_15_BUGS.md` - Bug #3 documenta symbolic features zeros
- `SPRINT_15_COMPLETE.md` - Status anterior (antes da descoberta)

### Código Afetado

- `pff/validators/ensembles/ensemble_wrappers/transformers.py:156-375` - SymbolicFeatureExtractor (REMOVER)
- `pff/validators/ensembles/ensemble_wrappers/model_wrappers.py` - predict_proba() (REFATORAR)
- `pff/services/business_service.py:715-890` - predict_hybrid_score() (ADICIONAR feature extraction)

---

## 🎓 Lições Aprendidas

### 1. Validação Inadequada

**Erro:** Testamos apenas com um input e declaramos bug corrigido
**Lição:** SEMPRE testar com múltiplos inputs variados

### 2. Penalty Mascarou o Problema

**Erro:** Score final variou (via penalty) e assumimos Ensemble funcionava
**Lição:** Validar CADA COMPONENTE separadamente, não apenas resultado final

### 3. Importância de Logs Detalhados

**Acerto:** Logs XAI permitiram identificar "Base Ensemble Score" constante
**Lição:** Logging estruturado é CRÍTICO para debugging

### 4. Workarounds Podem Esconder Bugs

**Observação:** Penalty funcional criou ilusão de sistema funcionando
**Lição:** Workarounds são temporários, não substituem fix real

---

## ✅ Ações Tomadas

1. **Documentação atualizada:**
   - ✅ SPRINT_15_DISCOVERY.md criado (este documento)
   - ⏳ SPRINT_15_COMPLETE.md será atualizado
   - ⏳ SPRINT_15_FIXES_IMPLEMENTED.md será atualizado

2. **Status dos bugs re-classificado:**
   - Bug #4: ✅ CORRIGIDO → ⚠️ PARCIALMENTE CORRIGIDO

3. **Sprint 16 planejado:**
   - Prioridade: Corrigir Bugs #2 e #3
   - Estimativa: 9h (6h + 2h + 1h validação)
   - Critério de sucesso: Base ensemble score variável

---

**Last Update:** 2025-10-23 00:45 BRT
**Discovered By:** User observation (test.json vs test1.json comparison)
**Impact:** HIGH - Ensemble ML não está funcionando efetivamente
**Priority:** Sprint 16 - MÁXIMA
**Status:** 🔴 CRITICAL - Requires immediate attention in next sprint
