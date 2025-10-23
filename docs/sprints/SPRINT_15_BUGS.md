# Sprint 15: Análise Completa de Bugs do Ensemble ML

**Data:** 2025-10-22
**Investigador:** Claude Code
**Contexto:** User reportou que "apesar de funcional, ele está com erros estratégicos e de implementação, que estão dando resultados errados"

---

## 🔴 RESUMO EXECUTIVO

**Sintomas observados:**
1. Ensemble sempre retorna score ~0.39 independentemente do input
2. Symbolic Analysis reporta "0 regras ativas" quando há 156 violações reais
3. Modelos individuais dão "mesma nota para tudo" (report do usuário)
4. Features: 24305 → 155 agrupadas, mas LightGBM recebe 544 features

**Causa raiz:**
- **DUPLICAÇÃO DE LÓGICA**: Business Service valida regras ✅ mas Ensemble tenta validar de novo ❌
- **DESCONEXÃO DE DADOS**: Violations detectadas não são passadas para o Ensemble
- **MISMATCH DE FEATURES**: Dimensões incompatíveis entre componentes

---

## 🐛 BUG #1: Duplicação de Lógica de Validação (CRÍTICO)

**Severidade:** 🔴 CRÍTICA
**Impacto:** Ensemble ignora validações reais, sempre retorna scores similares

### Descrição
Business Service valida regras corretamente (156 violações), mas passa apenas `triples` para o Ensemble. O Ensemble então **reimplementa a validação** no `SymbolicFeatureExtractor`, que:
1. Não tem acesso às regras carregadas no Business Service
2. Retorna 0 regras ativas
3. Contribui com zeros para o score final

### Locais do código

**Business Service (`business_service.py:852-902`):**
```python
def validate(self, input_data: dict | str) -> dict[str, Any]:
    # ...
    violations, satisfied_rules = self.rule_validator.validate_rules(
        all_rules, triples  # ✅ Valida corretamente - 156 violations
    )
    # ...
    hybrid_score = self.model_integration.predict_hybrid_score(triples)  # ❌ Passa só triples!
```

**Model Integration (`business_service.py:715-781`):**
```python
def predict_hybrid_score(self, triples: list[tuple[Any, str, Any]]) -> float:
    # ...
    proba = self.ensemble_model.predict_proba([triples])  # ❌ Passa só triples!
    ensemble_score = float(proba[0, 1])
```

**Symbolic Feature Extractor (`transformers.py:327-335`):**
```python
for i, rule in enumerate(rules):
    if SymbolicFeatureExtractor._rule_is_violated(rule, available_triples_set):
        sample_feature_vector[i] = 1  # ❌ Tenta validar de novo, falha
        violations += 1
# Log real: "🔍 Symbolic Analysis: 0 regras ativas"  ❌ SEMPRE 0!
```

### Fluxo Atual (ERRADO)
```
Business Service
├─> Valida regras: 156 violations ✅
├─> Chama ensemble.predict_proba([triples])
    └─> Ensemble
        ├─> SymbolicFeatureExtractor tenta validar regras DE NOVO
        ├─> Não tem regras do Business Service
        ├─> Retorna 0 regras ativas ❌
        ├─> LightGBM e TransE processam normalmente
        └─> Score final ~0.39 (sem component Symbolic)
└─> Retorna score 0.39 mas IGNORA as 156 violations reais
```

### Solução proposta
Business Service deve extrair features E PASSAR para o Ensemble:

```python
# Business Service extrair features
violations_vector = self._violations_to_feature_vector(violations)  # NEW
transe_features = self._extract_transe_features(triples)           # MOVE from Ensemble
statistical_features = self._extract_statistical_features(triples) # MOVE from Ensemble

# Combinar features
features = np.concatenate([violations_vector, transe_features, statistical_features])

# Ensemble recebe features prontas, NÃO triples
hybrid_score = self.model_integration.predict_hybrid_score(features)  # FIXED
```

---

## 🐛 BUG #2: Mismatch de Dimensões de Features (CRÍTICO)

**Severidade:** 🔴 CRÍTICA
**Impacto:** LightGBM recebe features de tamanho errado, resultando em predições constantes

### Descrição
- LightGBM foi treinado com **544 features** (embeddings TransE de dimensão 544)
- Transformer agrupa para **155 features** (24305 → 155)
- Features simbólicas teoricamente têm **24305 dimensões** (1 por regra AnyBURL)

**Log real:**
```
2025-10-22 19:10:10.152 | DEBUG | Predição LightGBM OK com 544 features
2025-10-22 19:10:17.575 | INFO  | ✅ Features: 24305 → 155 agrupadas
2025-10-22 19:10:17.577 | INFO  | 🔍 Symbolic Analysis: 0 regras ativas
```

### Locais do código

**LightGBM Wrapper (`model_wrappers.py:284-309`):**
```python
def _extract_features_from_triples(self, triples_list: list) -> np.ndarray:
    expected = getattr(self, "_expected_features", self.n_features_in_ or 544)  # 544
    # ...
    aggregated_features = np.mean(sample_features, axis=0)  # Média de embeddings
    return idx, aggregated_features.astype(np.float32)  # Shape: (n_samples, 544)
```

**Transformer (`transformers.py:230-237`):**
```python
features = self._apply_feature_grouping(binary_features)  # 24305 → 155
# ...
active_rules = np.sum(features > 0, axis=1)  # Conta regras ativas
logger.info(f"🔍 Symbolic Analysis: {active_rules[0]} regras ativas")  # SEMPRE 0!
```

### Problema
1. **LightGBM** espera 544 features (embeddings TransE)
2. **Symbolic** gera 24305 features (1 por regra), agrupa para 155
3. **Ensemble Pipeline** precisa concatenar ambos, mas dimensões não batem
4. **Resultado:** Features são zeradas ou truncadas, modelo sempre retorna ~0.39

### Solução proposta
Definir dimensões consistentes:

```python
# Arquitetura correta:
# - Symbolic features: N_violations (variável por sample)
# - TransE features: 544 (embedding dimension)
# - Statistical features: N_predicates (ex: 50)
# Total: N_violations + 544 + 50 = VARIÁVEL por amostra

# Pipeline sklearn deve lidar com dimensões variáveis ou:
# - Usar DictVectorizer para features esparsas
# - Ou fixar dimensões com padding/truncation
```

---

## 🐛 BUG #3: Symbolic Features Sempre Zeros (CRÍTICO)

**Severidade:** 🔴 CRÍTICA
**Impacto:** Componente Symbolic do Ensemble completamente desligado

### Descrição
`SymbolicFeatureExtractor._rule_is_violated()` sempre retorna `False` porque:
1. Não tem acesso às regras carregadas pelo Business Service
2. Tenta criar features baseadas em regras que não existem no seu escopo
3. Retorna vetor de zeros

### Evidências
**Log real:**
```
2025-10-22 19:10:17.577 | INFO | 🔍 Symbolic Analysis: 0 regras ativas
```

Com 128,319 regras AnyBURL carregadas e 156 violations detectadas, **0 regras ativas é IMPOSSÍVEL**.

### Locais do código

**Transformer (`transformers.py:213-237`):**
```python
def transform(self, X: list[list[tuple]], y=None) -> np.ndarray:
    if not self.rules_:
        logger.warning(
            "Nenhuma regra carregada no SymbolicFeatureExtractor. Retornando features vazias."
        )  # ❌ self.rules_ está vazio!
        return np.zeros((len(X), len(self.rules_)), dtype=np.float32)
```

**Rule Violation Check (`transformers.py:338-375`):**
```python
@staticmethod
def _rule_is_violated(rule: dict, available_triples: set) -> bool:
    # Lógica está correta, MAS:
    # - rules passadas estão no formato errado
    # - ou available_triples não contém as triplas certas
    # - ou regras não foram carregadas no fit()
```

### Solução proposta
**REMOVER** `SymbolicFeatureExtractor` do Ensemble. Business Service já valida regras:

```python
# NO Business Service:
violations_vector = self._violations_to_binary_vector(violations, all_rules)
# violations_vector[i] = 1 se regra i foi violada, 0 caso contrário

# Passar para Ensemble como feature pronta
features = {"violations": violations_vector, "triples": triples}
hybrid_score = self.ensemble.predict(features)
```

---

## 🐛 BUG #4: Scores Constantes (~0.39) (ALTO)

**Severidade:** 🟠 ALTA
**Impacto:** Ensemble não discrimina entre JSONs válidos e inválidos

### Descrição
Due aos bugs #1, #2, #3, o Ensemble sempre retorna scores similares:
- **Bug #1:** Violations não chegam ao Ensemble
- **Bug #2:** Features têm dimensões erradas
- **Bug #3:** Symbolic component sempre zero
- **Resultado:** Apenas TransE e LightGBM contribuem, com features sub-ótimas

### Evidências
**Log real:**
```
2025-10-22 19:10:17.611 | DEBUG | ✅ Ensemble score: 0.391
2025-10-22 19:10:47.826 | DEBUG | ✅ Ensemble score: 0.391  # EXATO mesmo score!
```

Dois JSONs diferentes (1125 triplas vs 294 triplas) → **MESMO SCORE 0.391**

### Explicação
Sem Symbolic features (Bug #3), o Ensemble depende apenas de:
1. **TransE:** Calcula scores baseados em embeddings
   - Problema: Embeddings são médias, sempre similares
2. **LightGBM:** Usa features de dimensão errada (Bug #2)
   - Problema: Features truncadas/zeradas → predição constante

**Pesos do Ensemble:**
- TransE: 30%
- LightGBM: 30%
- Symbolic: 20%  ❌ SEMPRE ZERO
- XGBoost: 20%

Com Symbolic = 0, apenas 60% do ensemble contribui, e mal.

### Solução proposta
Corrigir Bugs #1, #2, #3 para restaurar variabilidade nos scores.

---

## 🐛 BUG #5: Arquitetura Invertida (MÉDIO)

**Severidade:** 🟡 MÉDIA
**Impacto:** Dificulta manutenção e adiciona complexidade desnecessária

### Descrição
**User feedback:** "o business service deve ser o facade, a central, e não o ensemble. o ensemble que tem que mandar as coisas para o business, e não o contrário."

**Arquitetura atual (ERRADA):**
```
Business Service (facade) → Ensemble → SymbolicFeatureExtractor → tenta validar regras
```

**Arquitetura correta:**
```
Business Service (facade)
├─> Valida regras
├─> Extrai features
└─> Chama Ensemble apenas para combinar features (sem lógica de negócio)
```

### Solução proposta
- Business Service mantém TODA lógica de validação
- Ensemble é apenas um combinador ML de features
- Remover duplicação de lógica no Ensemble

---

## 📋 PLANO DE CORREÇÃO

### Fase 1: Testes que Expõem Bugs (4h)
- [ ] test_business_service_violations.py
  - Verificar que violations são detectadas
  - Verificar formato das violations
- [ ] test_ensemble_features_dimensions.py
  - Verificar dimensões de cada componente
  - Expor mismatch de features
- [ ] test_ensemble_score_variability.py
  - JSONs válidos devem ter scores >0.6
  - JSONs inválidos devem ter scores <0.4
  - Expor scores constantes

### Fase 2: Correção de Arquitetura (8h)
- [ ] Remover SymbolicFeatureExtractor do Ensemble
- [ ] Business Service extrai features completas:
  - violations_vector (from validate_rules)
  - transe_features (from TransE embeddings)
  - statistical_features (predicate counts)
- [ ] Ensemble recebe features prontas, não triples
- [ ] Ajustar dimensões para compatibilidade

### Fase 3: Validação (2h)
- [ ] Todos os testes passam
- [ ] `pff run` mostra scores variáveis
- [ ] Symbolic Analysis mostra regras ativas
- [ ] Ensemble discrimina válido vs inválido

---

## 🎯 MÉTRICAS DE SUCESSO

**Antes (atual):**
- Ensemble score: ~0.39 (constante)
- Symbolic Analysis: 0 regras ativas
- Violations detectadas: 156 (mas ignoradas)

**Depois (esperado):**
- Ensemble score: variável (0.2-0.8 dependendo do JSON)
- Symbolic Analysis: N regras ativas (onde N = violations)
- Violations detectadas: 156 (e usadas no score)

**Teste específico:**
```python
# JSON válido
score_valid = business_service.validate(valid_json)["hybrid_score"]
assert score_valid > 0.6

# JSON inválido
score_invalid = business_service.validate(invalid_json)["hybrid_score"]
assert score_invalid < 0.4

# Diferença significativa
assert abs(score_valid - score_invalid) > 0.3
```

---

**Próximos passos:** Criar testes, implementar correções, validar.
