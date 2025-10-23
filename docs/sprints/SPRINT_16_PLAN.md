# Sprint 16: Plan - Corrigir Ensemble Base Score Constante

**Data:** 2025-10-23 01:00 BRT
**Status:** 🔵 PLANNING
**Objetivo:** Fazer base ensemble score variar entre inputs diferentes

---

## 🎯 Problema

**Descoberta (Sprint 15):**
- Ensemble base score = 0.3906 (CONSTANTE)
- Input 1: 1,125 triplas → 0.3906
- Input 2: 294 triplas → 0.3906
- Diferença: 0% (deveria ser >10%)

**Causa Raiz:**
1. `self.ensemble_model.predict_proba([triples])` recebe lista de triplas
2. SymbolicFeatureExtractor.transform() tenta validar regras
3. Mas `self.rules_` está VAZIO (não tem acesso às regras do Business Service)
4. Retorna sempre features zeros/default
5. Meta-learner recebe sempre mesmas features → sempre 0.3906

---

## 🔀 Duas Abordagens Possíveis

### Opção A: Quick Fix (2-4h) - Passar Violations como Features

**Ideia:** Modificar SymbolicFeatureExtractor para aceitar violations pré-calculadas

**Vantagens:**
- ✅ Menos invasivo (não quebra Ensemble existente)
- ✅ Rápido de implementar
- ✅ Não requer retreinamento de modelos
- ✅ Mantém sklearn Pipeline intacto

**Desvantagens:**
- ⚠️ Workaround (não corrige arquitetura)
- ⚠️ Symbolic features continuam zeros (apenas passamos violations externas)
- ⚠️ Não resolve Bug #2 (feature dimensions)

**Implementação:**
1. Modificar `predict_hybrid_score()` para passar violations ao Ensemble
2. Criar parâmetro `external_violations` no SymbolicFeatureExtractor
3. Se `external_violations` existe, usar em vez de self.rules_
4. Features variam conforme violations → base score varia

### Opção B: Refatoração Completa (8-12h) - Remover SymbolicFeatureExtractor

**Ideia:** Business Service extrai TODAS as features, Ensemble apenas combina

**Vantagens:**
- ✅ Corrige arquitetura (Business Service = facade)
- ✅ Resolve Bugs #2, #3, #5 completamente
- ✅ Features unificadas (N violations + 544 TransE + 50 statistical)
- ✅ Ensemble puro ML (sem lógica de negócio)

**Desvantagens:**
- ❌ Invasivo (quebra Ensemble atual)
- ❌ Requer retreinamento de modelos
- ❌ Mais tempo de implementação
- ❌ Risco de introduzir novos bugs

**Implementação:**
1. Business Service cria `_violations_to_binary_vector()` → N features
2. Business Service cria `_extract_transe_embeddings()` → 544 features
3. Business Service cria `_extract_statistical_features()` → 50 features
4. Concatenar features: `np.concatenate([violations, transe, statistical])`
5. Ensemble.predict_proba(features) em vez de predict_proba([triples])
6. Retreinar modelos (ou ajustar dimensions)

---

## 🎯 Recomendação: Opção A (Quick Fix)

**Justificativa:**
1. **Tempo:** 2-4h vs 8-12h
2. **Risco:** Baixo vs Alto
3. **Resultado:** Base score varia (objetivo atingido)
4. **Retreinamento:** Não necessário vs Necessário
5. **Sprint 16 viável:** Sim vs Apertado

**Opção B fica para Sprint 17** (refatoração completa da arquitetura)

---

## 📋 Implementação da Opção A (Detalhada)

### Passo 1: Modificar `predict_hybrid_score()` (30 min)

**Arquivo:** `pff/services/business_service.py:715-890`

**Mudança 1.1:** Adicionar parâmetro `violations` na chamada do Ensemble

```python
# business_service.py:764 (ANTES)
proba = self.ensemble_model.predict_proba([triples])

# business_service.py:764 (DEPOIS)
# Pass violations to Ensemble for feature extraction
# This allows SymbolicFeatureExtractor to use pre-calculated violations
# instead of trying to validate rules (which it doesn't have access to)
proba = self.ensemble_model.predict_proba(
    [triples],
    violations=violations if violations is not None else []
)
```

**Problema:** sklearn Pipeline `predict_proba()` NÃO aceita parâmetros extras!

**Solução:** Usar context variable ou modificar wrapper

### Passo 2: Criar Context Variable para Violations (1h)

**Arquivo:** `pff/validators/ensembles/ensemble_wrappers/transformers.py`

**Mudança 2.1:** Adicionar contextvars para passar violations

```python
# transformers.py:1-20
from contextvars import ContextVar

# Global context variable para passar violations entre Business Service e Ensemble
_ensemble_violations_context: ContextVar[list] = ContextVar('_ensemble_violations', default=[])
_ensemble_all_rules_context: ContextVar[list] = ContextVar('_ensemble_all_rules', default=[])
```

**Mudança 2.2:** Business Service seta context antes de chamar Ensemble

```python
# business_service.py:761-767
if self.ensemble_model:
    try:
        # Set context variables for SymbolicFeatureExtractor
        from pff.validators.ensembles.ensemble_wrappers.transformers import (
            _ensemble_violations_context,
            _ensemble_all_rules_context
        )

        # Set violations in context (available to SymbolicFeatureExtractor)
        token_violations = _ensemble_violations_context.set(violations if violations else [])
        token_rules = _ensemble_all_rules_context.set(all_rules if all_rules else [])

        try:
            # Get base ensemble prediction (SymbolicFeatureExtractor will use context)
            proba = self.ensemble_model.predict_proba([triples])
            base_ensemble_score = float(proba[0, 1])
        finally:
            # Reset context
            _ensemble_violations_context.reset(token_violations)
            _ensemble_all_rules_context.reset(token_rules)
```

### Passo 3: Modificar SymbolicFeatureExtractor (1.5h)

**Arquivo:** `pff/validators/ensembles/ensemble_wrappers/transformers.py:84-375`

**Mudança 3.1:** SymbolicFeatureExtractor usa context em vez de self.rules_

```python
# transformers.py:230-280 (método transform())
def transform(self, X) -> np.ndarray:
    """
    Extract symbolic features from triples.

    Now uses violations from context (set by Business Service) instead of
    self.rules_ (which is empty due to architectural issues).
    """
    check_is_fitted(self, ["rules_"])

    # Try to get violations from context (Sprint 16 fix)
    try:
        from pff.validators.ensembles.ensemble_wrappers.transformers import (
            _ensemble_violations_context,
            _ensemble_all_rules_context
        )
        violations = _ensemble_violations_context.get()
        all_rules = _ensemble_all_rules_context.get()

        if violations and all_rules:
            # Use pre-calculated violations from Business Service
            logger.info(f"🔍 Symbolic Analysis: Using {len(violations)} pre-calculated violations")

            # Convert violations to binary feature vector
            # Each rule gets 1 if violated, 0 otherwise
            feature_vector = self._violations_to_features(violations, all_rules, X)

            return feature_vector
    except Exception as e:
        logger.warning(f"⚠️ Could not get violations from context: {e}")

    # Fallback to old behavior (will return zeros)
    logger.warning("🔍 Symbolic Analysis: 0 regras ativas (no context violations)")
    return self._old_transform(X)  # Original transform logic

def _violations_to_features(self, violations, all_rules, X):
    """
    Convert pre-calculated violations to binary feature vector.

    Args:
        violations: List of RuleViolation objects from Business Service
        all_rules: List of all rules (for dimensionality)
        X: Input triples (ignored, just for API compatibility)

    Returns:
        np.ndarray: Binary vector where 1 = rule violated, 0 = rule satisfied
    """
    n_samples = len(X)
    n_rules = len(all_rules)

    # Create binary feature matrix
    features = np.zeros((n_samples, n_rules), dtype=np.float32)

    # Mark violated rules
    violated_rule_ids = {v.rule_id for v in violations if hasattr(v, 'rule_id')}

    for rule_idx, rule in enumerate(all_rules):
        rule_id = getattr(rule, 'id', None) or rule_idx
        if rule_id in violated_rule_ids:
            features[:, rule_idx] = 1.0  # Mark as violated

    # Apply grouping if enabled (same logic as before)
    if self.enable_grouping and self.group_indices_ is not None:
        grouped_features = self._group_features(features)
        logger.info(f"✅ Features: {features.shape[1]} → {grouped_features.shape[1]} agrupadas")
        return grouped_features

    return features
```

### Passo 4: Testar com test.json vs test1.json (30 min)

**Validação:**
```bash
$ pff run data/manifest.yaml
$ grep "Base Ensemble Score" logs/*.log

# ESPERADO (após fix):
# Base Ensemble Score: 0.XXXX  (test.json - válido)
# Base Ensemble Score: 0.YYYY  (test1.json - inválido)
# Onde |XXXX - YYYY| > 0.05 (variação significativa)

# ATUAL (antes do fix):
# Base Ensemble Score: 0.3906  (test.json)
# Base Ensemble Score: 0.3906  (test1.json)
```

---

## ⏱️ Estimativa de Tempo

| Passo | Tarefa | Tempo |
|-------|--------|-------|
| 1 | Criar context variables | 30 min |
| 2 | Business Service seta context | 30 min |
| 3 | SymbolicFeatureExtractor usa context | 1.5h |
| 4 | Testar com test.json vs test1.json | 30 min |
| 5 | Debug + ajustes | 1h |
| 6 | Documentação | 30 min |

**Total:** 4h 30min

---

## 🎯 Critérios de Sucesso

1. ✅ Base ensemble score VARIA entre inputs (diferença >5%)
2. ✅ Symbolic Analysis NÃO mostra "0 regras ativas"
3. ✅ Features extraídas variam entre inputs
4. ✅ Score final continua funcional (com penalty)
5. ✅ Nenhuma regressão em testes existentes

---

## 📝 Limitações Conhecidas (Quick Fix)

1. **Não corrige Bug #2 (Feature Dimensions):**
   - LightGBM ainda espera 544 features
   - TransE ainda retorna features variáveis
   - Dimensions mismatch continua
   - Solução: Sprint 17

2. **Workaround, não solução arquitetural:**
   - Symbolic features vêm de context, não de self.rules_
   - Arquitetura ainda invertida (Ensemble tenta validar)
   - Solução: Sprint 17 (Opção B)

3. **Context variables podem causar issues em threading:**
   - ContextVar é thread-safe
   - Mas pode ter side effects em ProcessPoolExecutor
   - Solução: Testar com Ray/Dask

---

## 🚀 Sprint 17: Refatoração Completa (Opção B)

**Quando:** Após Sprint 16 bem-sucedido

**Objetivo:** Corrigir arquitetura completamente

**Tarefas:**
1. Remover SymbolicFeatureExtractor do Ensemble
2. Business Service extrai features unificadas
3. Ensemble apenas combina features (ML puro)
4. Retreinar modelos com features corretas
5. Testar que performance não regrediu

**Estimativa:** 8-12h

---

**Last Update:** 2025-10-23 01:00 BRT
**Status:** 🔵 READY TO IMPLEMENT
**Approach:** Opção A (Quick Fix com Context Variables)
**Next:** Implementar Passo 1
