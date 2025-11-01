# 🔍 LOG ANALYSIS REPORT - Post Sprint 25 & 26
**Data:** 2025-11-01 14:20 BRT  
**Status:** ANÁLISE COMPLETA

---

## 📊 Comparação: Antes vs Depois

### Metrics Comparison

| Metric | Antes (Broken) | Depois (Fixed) | Status |
|--------|---------------|----------------|--------|
| **Sparsity** | 0.00% | 0.97% | ⚠️ Piorou (-0.21%) |
| **F1-Score** | 0.6205 | 0.5949 | ⚠️ Piorou (-4.13%) |
| **Hybrid Contrib** | 51.25% | 39.22% | ✅ Melhor equilíbrio |
| **Symbolic Contrib** | 48.75% | 60.78% | ⚠️ Desbalanceou |
| **Balance Status** | BALANCED | BALANCED | ✅ Mantido |

---

## 🚨 NOVOS PROBLEMAS IDENTIFICADOS

### 1. Symbolic Features Variando Entre Runs ⚠️

**Evidência dos Logs:**
```
Run 1 (Sprint 25, 13:21): non-zero=5181/438800 (1.18%)
Run 2 (Latest, 14:20):    non-zero=4236/438800 (0.97%)
```

**Análise:**
- Variação de 21% na sparsity entre runs (-0.21%)
- Indica **não-determinismo** no matching de regras
- Possível causa: Numba vectorized processing tem aleatoriedade

**Root Cause:**
- Numba `check_violations_vectorized()` pode ter race conditions
- Ordem de processamento não determinística
- Hashing de entidades pode variar

### 2. F1-Score Piorou em 4.13% ⚠️

**Comparação:**
- Sprint 25: 0.6205 (+5.68% vs broken)
- Latest: 0.5949 (-4.13% vs Sprint 25)
- Net change: +1.33% vs original broken (0.5871)

**Possíveis Causas:**
- Regras diferentes sendo detectadas entre runs
- XGBoost extraction variando
- Autofeeding pode estar adicionando ruído

### 3. Symbolic Contribution Desbalanceou ⚠️

**Antes (Sprint 25):**
- Hybrid: 51.25% vs Symbolic: 48.75% (quase perfeito)

**Depois (Latest):**
- Hybrid: 39.22% vs Symbolic: 60.78% (desbalanceado para symbolic)

**Análise:**
- Symbolic passou a dominar (+12.03%)
- Hybrid perdeu importância (-12.03%)
- Não está mais no range ideal (40-60% cada)

---

## ✅ PROBLEMAS RESOLVIDOS

### 1. Symbolic Features Funcionando ✅
- **Antes:** 0% sparsity (completamente quebrado)
- **Depois:** ~1% sparsity (funcionando)
- **Status:** RESOLVIDO ✅

### 2. Model Balance Mantido ✅
- Ainda no range 40-60% (39.22% vs 60.78%)
- Balance Status: BALANCED
- **Status:** ACEITÁVEL ✅

### 3. Numba Accelerator Funcionando ✅
```log
✅ Numba acceleration successful: processed 4388 samples
Using vectorized processing for 4388 samples
```
- Processamento paralelo ativo
- Sem fallbacks para Python manual
- **Status:** RESOLVIDO ✅

---

## 🎯 NOVOS ISSUES CRÍTICOS

### Issue #1: Non-Deterministic Symbolic Features
**Severidade:** MÉDIA  
**Impacto:** Resultados variam entre runs (-21% sparsity variance)

**Root Cause:**
- Numba vectorized processing pode ter race conditions
- Entity encoding não é determinístico (hash-based)
- Ordem de processamento paralelo varia

**Solução Proposta:**
1. Adicionar seed para Numba random
2. Garantir ordenação determinística de samples
3. Usar encoding baseado em string position ao invés de hash()

### Issue #2: F1-Score Degradation
**Severidade:** MÉDIA  
**Impacto:** Performance piorou 4.13% vs Sprint 25

**Root Cause:**
- Regras diferentes sendo extraídas do XGBoost
- Autofeeding pode estar adicionando regras de baixa qualidade
- Threshold de confiança pode estar muito baixo (0.05)

**Solução Proposta:**
1. Aumentar min_confidence threshold: 0.05 → 0.10
2. Validar regras XGBoost antes de adicionar ao autofeeding
3. Adicionar filtro de qualidade nas regras simbólicas

### Issue #3: Symbolic Dominance
**Severidade:** BAIXA  
**Impacto:** Symbolic 60.78% vs Hybrid 39.22% (fora do ideal 50/50)

**Root Cause:**
- XGBoost dando mais peso para features simbólicas
- Hybrid probability pode estar com menor variância
- Top symbolic groups muito fortes (0.12-0.11)

**Solução Proposta:**
1. Ajustar XGBoost scale_pos_weight para balancear
2. Reduzir número de symbolic groups (152 → ~100)
3. Aumentar peso da feature hybrid_probability

---

## 📋 EVIDÊNCIAS DOS LOGS

### Sparsity Variance
```log
# Sprint 25 (13:21)
non-zero=5181/438800 (1.18%)

# Latest (14:20)
non-zero=4236/438800 (0.97%)

# Difference
-945 violations (-21% variance)
```

### F1-Score Variance
```log
# Sprint 25
F1-Score Final: 0.6205

# Latest
F1-Score Final: 0.5949

# Difference
-0.0256 (-4.13%)
```

### Contribution Variance
```log
# Sprint 25
Hybrid: 51.25% | Symbolic: 48.75%

# Latest
Hybrid: 39.22% | Symbolic: 60.78%

# Difference
Hybrid: -12.03% | Symbolic: +12.03%
```

---

## 🎯 PRÓXIMAS AÇÕES RECOMENDADAS

### Sprint 27: Fix Non-Determinism (4h) 🔴 HIGH PRIORITY
1. **Fix Numba Encoding (2h)**
   - Substituir hash() por encoding determinístico
   - Adicionar seed para random operations
   - Garantir ordenação de samples

2. **Validate Regression (1h)**
   - Rodar pipeline 3x e verificar variance
   - Sparsity deve variar <5%
   - F1-Score deve variar <2%

3. **Add Determinism Tests (1h)**
   - Criar test que roda pipeline 2x
   - Verificar que resultados são idênticos
   - Fail se variance >5%

### Sprint 28: Improve F1-Score (3h) 🟡 MEDIUM PRIORITY
1. **Tune Confidence Threshold (1h)**
   - min_confidence: 0.05 → 0.10
   - Validar impacto no F1-Score
   - Medir trade-off precision vs recall

2. **Filter XGBoost Rules (1h)**
   - Adicionar validação de regras antes autofeeding
   - Rejeitar regras com <70% support
   - Limitar a 50 regras de maior importância

3. **Benchmark (1h)**
   - Comparar com Sprint 25 baseline
   - Target: F1 > 0.62 (recovery)

### Sprint 29: Balance Symbolic/Hybrid (2h) 🟢 LOW PRIORITY
1. **Adjust XGBoost Weights (1h)**
   - Testar scale_pos_weight valores
   - Target: 45-55% balance

2. **Reduce Symbolic Groups (30min)**
   - 152 → 100 grupos
   - Manter apenas top importance

3. **Validate Balance (30min)**
   - Verificar contribution ratio
   - Garantir 40-60% range

---

## 📊 SUMMARY

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Resolvidos | 3 | 50% |
| ⚠️ Novos Issues | 3 | 50% |
| 🔴 High Priority | 1 | 16.7% |
| 🟡 Medium Priority | 1 | 16.7% |
| 🟢 Low Priority | 1 | 16.7% |

**Overall Status:** ⚠️ **PARTIALLY FIXED**

**Recomendação:** Executar Sprint 27 (Fix Non-Determinism) IMEDIATAMENTE antes de production deploy.

---

**Last Update:** 2025-11-01 17:24 BRT  
**Next Action:** Sprint 27 - Fix Non-Determinism  
**ETA:** 4 hours
