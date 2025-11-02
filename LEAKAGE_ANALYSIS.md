# 🚨 ANÁLISE CRÍTICA: Data Leakage + Feature Scaling Issues

**Data:** 2025-11-01  
**Status:** ⚠️ **3 BUGS CRÍTICOS IDENTIFICADOS**

---

## 📊 ANOMALIAS DETECTADAS

| Métrica | LightGBM | Ensemble | Delta | Status |
|---------|----------|----------|-------|--------|
| **AUC** | 0.9967 | 0.7677 | **-23%** | 🚨 CRÍTICO |
| **F1** | 0.9765 | 0.6823 | **-30%** | 🚨 CRÍTICO |
| **Accuracy** | 0.9766 | 0.6839 | **-30%** | 🚨 CRÍTICO |

**Interpretação:** LightGBM parece "perfeito" mas ensemble é "mediano" → **DATA LEAKAGE**

---

## 🔥 BUG #1: DATA LEAKAGE (CRÍTICO)

### Localização:
`pff/validators/transe/lightgbm_trainer.py:196-200`

### Código Problemático:
```python
for _ in range(num_negatives):
    # Random entities and relation
    head_idx = rng.randint(0, num_entities)      # ❌ TOTALMENTE ALEATÓRIO
    tail_idx = rng.randint(0, num_entities)      # ❌ TOTALMENTE ALEATÓRIO  
    rel_idx = rng.randint(0, num_relations)      # ❌ TOTALMENTE ALEATÓRIO
```

### Problema:
1. **Gera negativos que JÁ EXISTEM como positivos** (leak direto!)
2. **Negativos muito fáceis** de classificar (entidades aleatórias)
3. **AUC artificial de 0.996** porque modelo aprende padrão trivial

### Evidência:
- Train AUC ≈ Val AUC (0.996) = sem generalização
- Ensemble com negativos mais realistas: AUC cai para 0.76
- Gap de **30%** entre LightGBM standalone vs ensemble

### Correção Necessária:
```python
def generate_negative_samples_CORRECT(self, X_pos, y_pos, ...):
    """Generate hard negatives by corrupting existing triples."""
    
    # 1. Build set of known positive triples
    known_positives = set()
    for triple in self.train_triples:
        known_positives.add((triple[0], triple[1], triple[2]))
    
    X_neg = []
    max_attempts = 100
    
    for pos_idx in range(num_negatives):
        # Corrupt a random positive triple
        orig_triple = self.train_triples[pos_idx % len(self.train_triples)]
        h, r, t = orig_triple
        
        for attempt in range(max_attempts):
            # Randomly corrupt head OR tail (not both)
            if rng.random() < 0.5:
                # Corrupt head
                h_neg = rng.randint(0, num_entities)
                neg_triple = (h_neg, r, t)
            else:
                # Corrupt tail
                t_neg = rng.randint(0, num_entities)
                neg_triple = (h, r, t_neg)
            
            # ✅ CHECK: Negative is NOT in training set
            if neg_triple not in known_positives:
                # Extract features for this negative triple
                X_neg.append(self._extract_features(neg_triple))
                break
```

**Impact:** Reduzirá AUC de 0.996 para ~0.75-0.85 (realista)

---

## 🔥 BUG #2: FEATURE SCALING INCORRETA (ALTA)

### Evidência:
```
Top-10 features:
  Feature 288: 36253.04  ← 17x MAIOR que outras
  Feature 41:  2082.14
  Feature 171: 2013.20
```

### Problema:
1. **Feature 288 com magnitude absurda** (36k vs 2k)
2. **Symbolic dominance: 86.39%** (esperado: ~50-70%)
3. **Sem normalização consistente** entre híbridas e simbólicas

### Localização:
- Híbridas: `ProbaTransformer` retorna probabilidades [0,1] → OK ✅
- Simbólicas: `log1p(count_normalized)` → OK mas pode ter outliers ⚠️

### Hipótese:
- Feature 288 pode ser um **índice incorreto** (mapeamento bug)
- Ou uma feature simbólica com **muitas violações** concentradas

### Correção Necessária:
```python
# Em advanced_trainer.py, ANTES de fit:
from sklearn.preprocessing import StandardScaler

# Create pipeline with scaler
meta_learner = Pipeline([
    ('scaler', StandardScaler()),  # ✅ Normaliza TODAS as features
    ('xgboost', XGBClassifier(...))
])
```

**Impact:** Balanceará contribuição híbrida/simbólica para ~50/50

---

## ⚠️ BUG #3: FEATURE MAPPING INCORRETO (MÉDIA)

### Evidência:
```
Features totais: 152 agrupadas + 1 híbrida = 153
Feature reportada: 288  ← ÍNDICE IMPOSSÍVEL!
```

### Problema:
`get_feature_names_out()` retorna apenas 153 features, mas XGBoost reporta feature 288

### Hipóteses:
1. **Offset bug**: Híbrida tem index 0, simbólicas começam em 1 → confusão
2. **Cache antigo**: Feature importance de modelo antigo (antes refactor)
3. **LightGBM standalone**: Usa 484 features, ensemble usa 153 (mismatch)

### Teste:
```python
# Verificar get_feature_names_out() vs feature_importances_
feature_names = ensemble_model.named_steps['meta_learner'].get_feature_names_out()
print(f"Features declaradas: {len(feature_names)}")

importances = ensemble_model.named_steps['meta_learner'].feature_importances_
print(f"Importances shape: {importances.shape}")

# Deve ser igual!
assert len(feature_names) == len(importances)
```

---

## 📋 PLANO DE CORREÇÃO (PRIORIDADE)

### Sprint 28: Fix Data Leakage (URGENTE - 2h)

1. **Implementar negative sampling correto**
   - Arquivo: `lightgbm_trainer.py:151-230`
   - Lógica: Corrupção de triplas + verificação de existência
   - Teste: AUC deve cair para ~0.75-0.85

2. **Adicionar validação**
   - Verificar que negativos NÃO existem em train/val/test
   - Log: % de negativos únicos

3. **Re-treinar LightGBM**
   - Com negativos corretos
   - Esperado: AUC ~0.80, F1 ~0.75

### Sprint 29: Fix Feature Scaling (MÉDIA - 1h)

1. **Adicionar StandardScaler no ensemble**
   - Arquivo: `advanced_trainer.py:240-250`
   - Pipeline: scaler → XGBoost
   
2. **Investigar Feature 288**
   - Mapear índice → nome
   - Verificar se é bug ou outlier real

3. **Validar contribuições**
   - Híbrida: 40-60%
   - Simbólica: 40-60%

### Sprint 30: Feature Mapping Validation (BAIXA - 30min)

1. **Assert feature count**
   - len(feature_names) == len(importances)
   
2. **Log detalhado**
   - Top-10 features com NOMES completos
   - Não apenas índices

---

## 🎯 RESULTADOS ESPERADOS

### Antes (atual):
| Componente | AUC | F1 | Status |
|------------|-----|----|----|
| LightGBM | 0.9967 | 0.9765 | ❌ Leakage |
| Ensemble | 0.7677 | 0.6823 | ⚠️ OK |
| **Gap** | **-23%** | **-30%** | 🚨 CRÍTICO |

### Depois (esperado):
| Componente | AUC | F1 | Status |
|------------|-----|----|----|
| LightGBM | 0.80-0.85 | 0.75-0.80 | ✅ Realista |
| Ensemble | 0.78-0.82 | 0.72-0.76 | ✅ Melhor |
| **Gap** | **< 5%** | **< 5%** | ✅ NORMAL |

**Objetivo:** LightGBM e Ensemble devem ter performance **similar** (delta < 5%)

---

## 🔍 VALIDAÇÃO

### Checklist Pré-Correção:
- [ ] AUC LightGBM: 0.9967 ← ABSURDAMENTE ALTO
- [ ] AUC Ensemble: 0.7677 ← Gap de 23%
- [ ] Negative sampling: aleatório ← LEAK
- [ ] Feature scaling: inconsistente
- [ ] Feature mapping: index 288/153 ← BUG

### Checklist Pós-Correção:
- [ ] Negative sampling: corrupção + verificação ✅
- [ ] AUC LightGBM: 0.75-0.85 ✅ Realista
- [ ] AUC Ensemble: 0.78-0.82 ✅ Melhor
- [ ] Gap: < 5% ✅ Normal
- [ ] Contribuição balanceada: 40-60% híbrida ✅
- [ ] Feature mapping: correto ✅

---

**Próximo passo:** Implementar correção do negative sampling (Sprint 28)

**ETA:** 2 horas  
**Prioridade:** 🔥 CRÍTICA (bug de leakage invalida todas as métricas)

