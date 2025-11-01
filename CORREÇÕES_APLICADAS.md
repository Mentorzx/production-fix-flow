# Correções Aplicadas - 2025-11-01 06:30

## ⚠️ ATUALIZAÇÃO: Primeira Correção Causou Underfitting

Após análise dos logs de 03:24, identificamos que as configurações conservadoras causaram **underfitting**:
- Ensemble F1 caiu de 0.6333 → 0.5980 (piora de 6%)
- Sparsity caiu de 1.91% → 0.04% (48× pior!)
- XGBoost ainda retorna 0 regras (min_confidence=0.1 muito alto)

---

## ✅ Bugs Críticos Corrigidos

### 1. XGBoost Rule Extraction - TypeError
**Bug**: `argument of type 'int' is not iterable` em `_normalize_tree_node()`

**Causa Raiz**: Código tentava usar operador `in` sem verificar se `node` era dict

**Solução**:
```python
# ANTES (linha 59):
if "leaf" in node:  # ❌ Falha se node é int!
    
# DEPOIS:
if not isinstance(node, dict):
    return {}
if "leaf" in node:  # ✅ Seguro
```

**Debug Adicionado (v2)**:
- Log quantas regras extraídas na primeira árvore
- Log leaf nodes e porque são filtrados  
- Warning se 0 regras extraídas

**Teste**: `tests/test_xgboost_extraction_fix.py` (2/2 passing ✅)

---

## 🔧 Configurações Rebalanceadas (v2 - Correção de Underfitting)

### Estratégia: Sweet Spot entre Overfitting e Underfitting

| Parâmetro | Original (Overfit) | v1 Conserv. (Underfit) | **v2 Balanced** |
|-----------|-------------------|------------------------|-----------------|
| n_estimators | 400 | 50 | **100** ⬅️ |
| max_depth | 4 | 2 | **3** ⬅️ |
| reg_alpha (L1) | 0.005 | 1.0 | **0.1** ⬅️ |
| reg_lambda (L2) | 0.05 | 10.0 | **1.0** ⬅️ |
| min_child_weight | 5 | 10 | **7** ⬅️ |
| gamma | 0.005 | 0.1 | **0.05** ⬅️ |
| subsample | 0.9 | 0.7 | **0.8** ⬅️ |
| early_stopping | 30 | 10 | **15** ⬅️ |
| min_confidence | 0.01 | 0.1 | **0.05** ⬅️ |
| AnyBURL threshold | 0.01 | 0.05 | **0.02** ⬅️ |

### Arquivos Modificados
- `config/ensemble.yaml`
- `config/kg.yaml`

---

## 📊 Histórico de Performance

| Versão | Config | Ensemble F1 | vs TransE | Sparsity | XGB Rules | Status |
|--------|--------|-------------|-----------|----------|-----------|--------|
| Baseline | Original | 0.6333 | -14% | 1.91% | 0 | ❌ Overfitting |
| **v1** | **Conserv.** | **0.5980** | **-17%** | **0.04%** | **0** | ❌ **Underfitting** |
| **v2** | **Balanced** | **?** | **?** | **?** | **?** | ⏳ **A Testar** |

**TransE Baseline**: MRR=0.7167 (F1 estimado ~0.72)

**Objetivo v2**: Ensemble F1 >= 0.72 (igual ou melhor que TransE)

---

## 🔍 Debug de Rule Matching Adicionado

**Problema**: 0 regras ativas apesar de sparsity

**Logs adicionados**:
- Formato das chaves (head/body)
- Padrão da regra
- Sample de triplas disponíveis
- Diagnóstico de regras puladas

**Arquivo**: `pff/validators/ensembles/ensemble_wrappers/transformers.py`

---

## 🚀 Próximos Passos

### 1. Rodar Pipeline Novamente
```bash
pff run --manifest data/manifest.yaml
```

### 2. Validar Métricas Esperadas
- ✅ XGBoost: 20-100 regras extraídas (min_confidence=0.05)
- ✅ Ensemble F1: 0.65-0.73 (próximo de TransE)
- ✅ Sparsity: >1% (melhor que 0.04%)
- ✅ Symbolic activation: >0 regras ativas
- ✅ Balanceamento: ~70/30 híbrido/simbólico (mantido)

### 3. Se Ainda Não Funcionar
- Reduzir min_confidence para 0.02
- Aumentar n_estimators para 150
- Reduzir reg_alpha/reg_lambda em 50%

---

## 📁 Arquivos Modificados

### Código
- `pff/validators/ensembles/ensemble_rules_extractor.py` (TypeError fix + debug)
- `pff/validators/ensembles/ensemble_wrappers/transformers.py` (debug logs)

### Configuração
- `config/ensemble.yaml` (balanced params v2)
- `config/kg.yaml` (balanced AnyBURL threshold)

### Documentação
- `ANALYSIS_BUGS_OVERFITTING.md` (atualizado com nova execução)
- `CORREÇÕES_APLICADAS.md` (este arquivo)

### Testes
- `tests/test_xgboost_extraction_fix.py` (novo)

---

## 📝 Commits

1. **f2fca48** - Fix critical bugs: XGBoost extraction + ensemble overfitting (v1 - conservador demais)
2. **9c58a82** - Fix underfitting: rebalance ensemble configs + debug XGBoost extraction (v2 - balanceado)

**Total**: 108 arquivos modificados, 15,221 inserções, 1,560 deleções
