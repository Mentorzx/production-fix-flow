# Análise de Resultados - Pipeline Completo (2025-11-01 10:08)

## 🎯 Executive Summary

**Status**: ✅ **XGBoost Extraction FUNCIONANDO** | ⚠️ **Ensemble ainda INFERIOR ao TransE**

---

## 📊 Métricas Principais

### TransE (Baseline)
- **MRR**: 0.7898 ✅ **EXCELENTE** (+10% vs execução anterior 0.7161)
- **Status**: Modelo de referência, performance superior

### Ensemble Advanced
- **F1-Score**: 0.5871 ❌ **INFERIOR** (vs TransE MRR 0.7898 = **26% pior**)
- **Contribuição Híbrida**: 61.91% (TransE + LightGBM)
- **Contribuição Simbólica**: 38.09% (regras)
- **Status**: ❌ Ainda pior que modelo base (deveria ser >= TransE)

### XGBoost Rule Extraction
- **Árvores analisadas**: 35 (was: 100 nas execuções anteriores)
- **Regras extraídas**: 8 ✅ **FUNCIONANDO** (was: 0)
- **Status**: ✅ **BUG CORRIGIDO COM SUCESSO**

### Symbolic Features
- **Sparsidade**: 0.00% ❌ **CRÍTICO** (0/1,520 features)
- **Regras ativas (treino)**: 0 regras
- **Regras ativas (validação)**: 5-8 regras ⚠️ **MUITO BAIXO**
- **Status**: ❌ Features simbólicas NÃO estão funcionando no treino

---

## 🔍 Análise Detalhada

### 1. XGBoost Extraction - ✅ SUCESSO TÉCNICO

**Evidências:**
```
🌳 Analisando 35 árvores do XGBoost
✅ 8 regras extraídas do XGBoost
🎉 Total de regras extraídas: 8
```

**Impacto:**
- ✅ Correção técnica funcionou perfeitamente
- ✅ Node navigation via node_map operacional
- ✅ Feature count correto (n_features_in_=153)
- ⚠️ Apenas 35 árvores geradas (vs 100 esperado) - indica early stopping

**Por que apenas 35 árvores?**
- Configuração: `n_estimators=100`
- Early stopping: 15 rounds sem melhoria
- Conclusão: Modelo convergiu em 35 iterações

### 2. Symbolic Features - ❌ PROBLEMA CRÍTICO PERSISTENTE

**Sintomas:**
```
📊 Sparsidade: 0/1,520 (0.00%) não-zero  [TREINO]
🔍 Symbolic Analysis: 0 regras ativas      [TREINO]
🔍 Symbolic Analysis: 5-8 regras ativas    [VALIDAÇÃO]
```

**Problema IDENTIFICADO:**
- Regras NÃO estão matcheando durante o treino
- Durante validação, algumas regras matcheiam (5-8)
- Isso indica que o problema NÃO é no código de matching
- Problema REAL: **Dados de treino incompatíveis com regras aprendidas**

**Por que isso acontece?**
1. Regras aprendidas do AnyBURL são muito específicas
2. Dados de treino já foram vistos (overfitting das regras)
3. Business validator pode estar retornando sempre False

### 3. Ensemble Performance - ❌ AINDA INFERIOR

**Comparação:**
- TransE MRR: 0.7898
- Ensemble F1: 0.5871
- **Diferença**: -26% (Ensemble é 26% PIOR)

**Evolução histórica:**
| Execução | TransE | Ensemble F1 | Diferença | Status |
|----------|--------|-------------|-----------|--------|
| 02:33 (original) | 0.7161 | 0.6441 | -10% | Overfit |
| 03:24 (v1 conserv.) | 0.7167 | 0.5980 | -17% | Underfit |
| 03:40 (v2 balanced) | 0.7083 | 0.6284 | -11% | Melhor |
| **10:08 (XGBoost fix)** | **0.7898** | **0.5871** | **-26%** | **PIOR** |

**Possível causa da piora:**
- TransE melhorou significativamente (+10%)
- Ensemble piorou (-6%)
- Early stopping muito agressivo (35 árvores)
- Falta de regularização adequada

### 4. Balanceamento Híbrido/Simbólico - ⚠️ ACEITÁVEL

**Atual:**
- Híbrido: 61.91% (TransE + LightGBM)
- Simbólico: 38.09% (regras)

**Esperado:**
- 70/30 (híbrido/simbólico)

**Status:** ⚠️ Próximo do ideal mas invertido (deveria ser 70% híbrido)

---

## 🐛 Problemas Identificados

### P0 - CRÍTICO

1. **Symbolic Features com 0% sparsity no treino**
   - Regras não matcheiam dados de treino
   - Causa: Possível bug no business_service.validate()
   - Evidência: Funciona em validação (5-8 regras) mas não no treino (0)

2. **Ensemble 26% inferior ao TransE**
   - Ensemble deveria SEMPRE ser >= melhor modelo base
   - Stacking está piorando ao invés de melhorar
   - Early stopping muito agressivo (35 árvores)

### P1 - ALTO

3. **Early stopping muito agressivo**
   - 35 árvores geradas de 100 possíveis
   - Pode estar sub-fitando
   - Considerar aumentar early_stopping_rounds de 15 para 30

4. **TransE variabilidade alta**
   - MRR pulou de 0.7083 → 0.7898 (+11%)
   - Pode indicar instabilidade no treino
   - Verificar seed fixo

---

## 🎯 Ações Recomendadas

### Imediatas (P0)

1. **Investigar business_service.validate() no treino**
   ```python
   # Adicionar debug em SymbolicFeatureExtractor.transform()
   # Verificar se business_service está sendo chamado
   # Logar resultados de validate() para primeiras 10 amostras
   ```

2. **Testar ensemble SEM features simbólicas**
   ```bash
   # Desabilitar symbolic features temporariamente
   # Verificar se F1 melhora sem elas
   ```

3. **Aumentar early_stopping_rounds**
   ```yaml
   # config/ensemble.yaml
   early_stopping_rounds: 30  # was: 15
   ```

### Curto Prazo (P1)

4. **Fixar seed para reprodutibilidade**
   ```python
   # Adicionar random_state=42 em todos os modelos
   ```

5. **Reduzir regularização**
   ```yaml
   # Ensemble está underfitting
   reg_alpha: 0.05  # was: 0.1
   reg_lambda: 0.5  # was: 1.0
   ```

---

## 📈 Próximos Passos

### Sprint 22: Debug Symbolic Features (4h) - URGENTE

1. Adicionar logs detalhados em business_service.validate()
2. Verificar por que regras não matcheiam no treino
3. Testar matching manual com 10 amostras
4. Corrigir bug se encontrado

### Sprint 23: Otimizar Ensemble (4h)

1. Desabilitar symbolic features temporariamente
2. Ajustar early_stopping_rounds para 30
3. Reduzir regularização (reg_alpha=0.05, reg_lambda=0.5)
4. Validar que ensemble >= TransE

---

## 📝 Conclusão

### ✅ Sucessos

1. **XGBoost extraction funcionando**: 8 regras extraídas com sucesso
2. **Código robusto**: Node navigation e feature detection corretos
3. **TransE excelente**: MRR=0.7898 (+10% vs baseline)

### ❌ Falhas

1. **Symbolic features quebradas**: 0% sparsity no treino
2. **Ensemble inferior**: 26% pior que TransE (deveria ser melhor)
3. **Early stopping agressivo**: Apenas 35/100 árvores

### 🎯 Prioridade Máxima

**Corrigir symbolic features é CRÍTICO** - sem elas, o ensemble não tem vantagem sobre TransE puro.

---

**Arquivo gerado em**: 2025-11-01 10:15 BRT  
**Responsável**: Claude Code  
**Status**: ⚠️ XGBoost fix OK, mas symbolic features ainda quebradas
