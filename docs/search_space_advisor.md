# Painel Search Space Advisor (Optuna)

Este documento descreve, em detalhes, (i) os hiperparâmetros que o Optuna explora atualmente no PFF, (ii) a finalidade da tabela “Consultor de Espaço de Busca” exibida no dashboard de HPO, e (iii) as heurísticas empregadas para recomendar expansão, redução ou manutenção do espaço de busca.

## 1. Hiperparâmetros hoje expandidos pelo Optuna

A matriz `config/hpo/optimization.yaml` define explicitamente cada espaço de busca. Abaixo estão os blocos principais com os valores atuais.

### 1.1 Treinamento (`dslfm_kgc.training`)

| Hiperparâmetro          | Espaço                       | Observação                     |
| ------------------------ | ----------------------------- | -------------------------------- |
| `learning_rate`        | Contínuo ∈\[4.0e-4, 6.0e-4] | FloatDistribution log-friendly.  |
| `weight_decay`         | Contínuo ∈\[1.0e-6, 1.0e-4] | Regularização L2.              |
| `batch_size`           | Categórico {256, 512, 1024}  | Ajusta throughput/VRAM.          |
| `negative_sample_size` | Contínuo ∈\[384, 512]       | Número de negativos por update. |
| `grad_clip`            | Contínuo ∈\[1.0, 10.0]      | Limita norm de gradiente.        |
| `warmup_ratio`         | Contínuo ∈\[0.10, 0.20]     | Proporção de warmup linear.    |
| `epochs`               | Contínuo ∈\[120, 160]       | Máx. de épocas por trial.      |
| `use_compile`          | Booleano {false}              | Fixado (sem busca).              |

### 1.2 Arquitetura (`dslfm_kgc.architecture`)

| Hiperparâmetro     | Espaço                       | Observação                 |
| ------------------- | ----------------------------- | ---------------------------- |
| `feature_dim`     | Categórico {256, 512}        | Dimensão de features DSLFM. |
| `max_communities` | Categórico {128}             | Fixado.                      |
| `hidden_dim`      | Categórico {256, 512}        | Hidden size de MLPs.         |
| `kl_weight`       | Contínuo ∈\[1.0e-4, 1.0e-2] | Peso KL (ibp/variational).   |
| `sparsity_weight` | Contínuo ∈\[1.0e-6, 1.0e-2] | Penaliza ativações densas. |
| `ibp_alpha`       | Contínuo ∈\[1.0, 10.0]      | Tempera limites IBP.         |

### 1.3 Contraste (`dslfm_kgc.contrastive`)

| Hiperparâmetro          | Espaço                      | Observação        |
| ------------------------ | ---------------------------- | ------------------- |
| `temperature`          | Contínuo ∈\[0.025, 0.04]   | Escala InfoNCE.     |
| `margin`               | Contínuo ∈\[0.0, 0.05]     | Margin hinge.       |
| `num_negatives`        | Categórico {384}            | Fixado.             |
| `num_global_negatives` | Contínuo ∈\[96, 96]        | Valor fixo.         |
| `neg_sampler`          | Categórico {"degree_based"} | Única estratégia. |
| `self_adversarial`     | Booleano {true}              | Sempre ativo.       |

### 1.4 Amostragem (`dslfm_kgc.sampling`)

| Hiperparâmetro     | Espaço                 | Observação                        |
| ------------------- | ----------------------- | ----------------------------------- |
| `adv_temperature` | Contínuo ∈\[0.9, 1.8] | Temperatura adversarial.            |
| `hard_neg_ratio`  | Contínuo ∈\[0.0, 0.7] | Proporção de negativos difíceis. |

### 1.5 PC / Lógica / Low-Rank / FAISS

- **Probabilistic Circuits (`dslfm_kgc.pc`)**: `enabled` fixo em `false`, `num_latents` ∈ {64, 128}, `lambda_pc` = 0, `depth` = 4.
- **Lógica (`dslfm_kgc.logic`)**: `lambda_logic` ∈ \[0.03, 0.05], `t_norm` {"product"}.
- **Low Rank (`dslfm_kgc.low_rank`)**: `num_basis` ∈ {2, 4, 8, 16} com `enabled=true`.
- **FAISS (`dslfm_kgc.faiss`)**: `nlist` ∈ {256, 512, 1024, 2048}; `nprobe` ∈ {4, 8, 16, 32}; `eval_topk` ∈ {512, 1024, 2048}.

### 1.6 Faixas auxiliares

- **`adaptive_range_factors`**: impõe mínimo relativo (divisores) para `batch_size` e `num_negatives` após recomendações.
- **`hpo_bounds` e `metrics_bounds`**: definem envelopes adicionais para pesos simbólicos/neuronais, taxas de decisão e métricas alvo. Esses limites são usados pelo Advisor e pelos testes arquiteturais para validar as sugestões.

## 2. Objetivo e features da tabela “Consultor de Espaço de Busca”

O card `SearchSpaceAdvisorCard` (React) foi criado para transformar as recomendações do módulo `SearchSpaceAdvisor` em uma tabela acionável. Cada linha representa um hiperparâmetro com os seguintes elementos:

1. **Parâmetro** – Nome com tooltip contextual e badges automáticos (INT/FLOAT/LOG/CAT). Exibe também o par `min/max` atual extraído do espaço Optuna.
2. **Estatísticas** – `n`, média e desvio dos trials concluídos, alimentados por `attempts_summary`.
3. **Densidade** – Strip plot responsivo que desenha pontos (até 60) com jitter e realce da região top-k (q10–q90). A banda amarela sinaliza os limites da região “best performers”.
4. **Topologia** – Régua que sobrepõe o intervalo atual (barra cheia) e a proposta (contorno). Suporta escalas logarítmicas, setas de expansão e textos “min / proposta / max”.
5. **Importância** – Barra gradiente (0–100%) derivada de importâncias externas, internas ou misturadas. Quando habilitado, o Advisor combina importâncias externas do Optuna com importâncias internas calculadas por correlação/variância.
6. **Confiança** – Badge textual (ALTA/MÉDIA/BAIXA) colorida por nível.
7. **Ação** – Botão inteligente que mostra o delta (“0.001 → 0.005”, “= 0.3”, etc.). Há também um ícone de dropdown que abre o painel detalhado.

Features adicionais:

- **Resumo superior**: chips com contagem de cada ação (`EXPANDIR`, `REDUZIR`, etc.), total de trials/top-k e tempo de computação.
- **Drill-down**: painel expandido com impacto estimado, região ótima, estatísticas avançadas (assimetria, curtose) e botões “Aplicar Ajuste / Ignorar”. Agora todo o texto está em PT-BR.
- **Preview de patch**: botão “Preview Patch” chama `generate_search_space_patch` e exibe o diff JSON sugerido.

## 3. Fluxo de dados e análise

1. **Ingestão dos trials** – `SearchSpaceAdvisor.advise()` projeta o estudo para um score unificado, transforma os trials em `TrialSummary` e usa trials `COMPLETE` para recomendações empíricas. Trials `PRUNED` ainda podem alimentar surrogate e sinais auxiliares via `intermediate_values`.
2. **Fingerprint de dataset** – Usa os Parquets em `outputs/kg/**` para calcular número de entidades, relações, densidade etc. (`compute_dataset_profile_fingerprint`). Esse perfil é anexado às recomendações e reutilizado em heurísticas de “cold start”.
3. **Seleção top-k** – `_select_top_k` escolhe até 25% dos melhores trials (mínimo adaptativo `max(top_k_min, 5% de n_trials)`), respeitando a direção (`maximize`/`minimize`).
4. **Importâncias** – O driver pode enviar importâncias externas, mas o Advisor também consegue calcular importâncias internas. O resultado final pode ser `external`, `internal` ou `blended`, e modula confiança, prioridade de ordenação e algumas heurísticas (por exemplo, ações agressivas só sobem para confiança alta quando `importance > 0.1`).
5. **Análise numérica/categórica** – `_analyze_numeric_param` e `_analyze_categorical_param` combinam os valores brutos (`all_values`) e top-k (`top_k_values`) para gerar `attempts_summary`, regiões ideais, CV e proximidades em relação às bordas.
6. **Cache em memória e persistente** – Recomendações são memoizadas em duas camadas: L1 em memória e L2 em PostgreSQL. A identidade inclui `study_name + dataset_fingerprint + direction + advisor_version + last_trial + search_space_hash + objective_schema_hash`. Se nenhum trial novo entrou e o contexto continua equivalente, o Advisor responde com `cache_hit=true` sem recalcular.
7. **Dataset heuristics** – Quando `n_trials < min_trials_any` (default 5), o Advisor não tem evidências suficientes e deriva recomendações com base em `dataset_profile` (ajusta `embedding_dim`, `negative_sampling`, pesos de regularização etc.).

## 4. Heurísticas de recomendação

### 4.1 Parâmetros numéricos

- **Expansão superior** (`expand_upper`): dispara quando os top performers se acumulam perto do limite superior, mas a ação só passa se houver suporte monotônico suficiente. Para parâmetros sensíveis a custo, o critério é mais rígido.
- **Expansão inferior** (`expand_lower`): análogo, agora condicionado por suporte de tendência monotônica na direção oposta.
- **Apertar intervalo** (`narrow`): se o coeficiente de variação dos top-k (`std / |mean|`) ficar < 0.15 e houver evidência suficiente, o Advisor restringe o espaço para `q10..q90`. Se houver surrogate, ainda aplica um teste de segurança estilo BALLET antes de aceitar o estreitamento.
- **Fixar valor** (`fix`): quando `importance < 0.05` e nenhuma outra ação foi tomada, o Advisor sugere fixar no `q50` dos top-k.
- **Trocar distribuição** (`change_distribution`): se o nome/range sugerir escala logarítmica (p.ex. `learning_rate`, `weight_decay` ou `high / low > 100`) e o espaço não estiver marcado como `log`, a recomendação é converter para distribuição log-uniforme, desde que os limites sejam positivos.

### 4.2 Parâmetros categóricos

- **Reduzir categorias** (`reduce_categories`): identifica classes que concentram ≥60% dos top-k. Quando há surrogate, a redução pode ser guiada por UCB/LCB por categoria; sem surrogate, mantém as dominantes + runner-up. A ação ainda depende de evidência mínima no top-k.
- **Fixar categoria**: se a importância estiver <0.05, fixa na categoria mais frequente entre os top performers.

### 4.3 Recomendações guiadas por dataset

Em cenários de baixa evidência o Advisor aplica heurísticas determinísticas:

- **`embedding_dim`**: converge para o valor mais próximo de `2 * sqrt(n_entidades)` (limitado entre 64 e 1024) e sugere estreitamento ou redução de choices.
- **`negative_sample_size`**: cruza `n_triples` para escolher faixas-alvo (64/128/256) e estreita o intervalo.
- **Regularização (`lambda`, `weight_decay`, `dropout`)**: usa a densidade média do grafo para aplicar um intervalo seguro.

### 4.4 Geração e aplicação de patches

As sugestões são convertidas por `generate_search_space_patch`, que devolve um dicionário pronto para mergear no YAML do Optuna. O dashboard expõe o JSON para revisão antes de aplicar.

## 5. Como interpretar “Expandir” x “Manter”

- **Expandir (↑/↓)**: indica saturação nas bordas do espaço. Aceitar implica permitir valores fora do intervalo original, importante quando todas as tentativas “boas” estão agrupadas numa extremidade.
- **Reduzir / Fixar**: ocorre quando há concentração estatística. Ajuda a estabilizar o treino e liberar orçamento de HPO para outros parâmetros.
- **Manter**: significa que o Advisor não encontrou evidência clara (ou já ajustou via distribuição log). Mesmo assim, o painel mostra importâncias e confiança para que o operador saiba priorizar manualmente.

## 6. Interface com os dados em tempo real

- Cada execução do comando `pff hpo optimize ...` injeta novos trials e reavalia o painel. Caso apenas novas métricas (mas não novos trials) sejam registrados, o cache evita recomputações.
- O Advisor também inspeciona os dados de entrada (Parquet + métricas Optuna) para atualizar `dataset_profile`. Assim, o painel sempre reflete o estado atual do conjunto de dados, mesmo antes de haver evidência suficiente para recomendações baseadas em trials.

## 7. Funcionamento interno detalhado

Esta seção descreve a implementação atual do Advisor no código, com foco em fluxo de execução, cálculos e mecanismos de confiabilidade.

### 7.1 Pipeline completo

1. **Resolve configuração de runtime** – junta defaults do YAML `config/hpo/optimization.yaml` com overrides passados na chamada. Nessa etapa também marca se `enable_surrogate`, `enable_interactions` e `disable_internal_importances` foram explicitamente definidos.
2. **Normaliza a direção do objetivo** – rótulos como `MAXIMIZE`, `StudyDirection.MAXIMIZE`, `max` e `maximize` convergem para `maximize`; equivalentes de minimização convergem para `minimize`.
3. **Projeta estudos multiobjetivo** – antes de escolher top-k, o Advisor converte vetores de objetivos em um score escalar híbrido com ordenação Pareto, contribuição de hypervolume e fallback escalar quando necessário.
4. **Constrói `TrialSummary`** – trials com `params` válidos são convertidos para uma estrutura leve. Trials `COMPLETE` alimentam o corpo principal da análise; trials `PRUNED` podem entrar em caminhos auxiliares quando há `intermediate_values`.
5. **Consulta cache L1/L2** – a resposta é reutilizada quando `study_name`, `dataset_fingerprint`, `direction`, `advisor_version`, `last_trial`, `search_space_hash` e `objective_schema_hash` não mudaram.
6. **Decide entre heurística de baixa evidência e análise empírica** – com menos de `min_trials_any`, o Advisor gera apenas recomendações orientadas por `dataset_profile`; acima disso ele entra no caminho completo.
7. **Seleciona top-k** – usa apenas trials `COMPLETE`, respeitando a direção do objetivo e um `k` adaptativo.
8. **Constrói metadados por parâmetro** – tipa cada parâmetro como `float`, `int`, `categorical` ou `fixed`, resolve se o parâmetro deve operar em domínio log e extrai âncoras locais para o surrogate.
9. **Ajusta surrogate opcional** – quando há dados suficientes, treina uma `RandomForestRegressor` para sondagens locais e cálculo de incerteza aproximada.
10. **Calcula interações opcionais** – se o surrogate existe e `interactions` está habilitado, estima interações por SHAP e usa isso para bloquear fixações/estreitamentos frágeis.
11. **Resolve importâncias** – mistura importâncias externas do Optuna com importâncias internas calculadas pelo próprio Advisor.
12. **Atualiza estado de confiança temporal** – o `trust_bucket` registra se melhores trials recentes vêm batendo repetidamente nas bordas do espaço.
13. **Analisa parâmetro a parâmetro** – decide `expand_upper`, `expand_lower`, `narrow`, `fix`, `reduce_categories`, `change_distribution` ou `keep`.
14. **Faz bootstrap da ação** – reamostra os trials e mede quantas vezes a mesma ação reaparece.
15. **Valida a recomendação** – recomendações inseguras são automaticamente rebaixadas para `keep` e ganham metadados de bloqueio.
16. **Roda self-audit periódico** – backtesta ações direcionais em prefixes históricos contra suffixes futuros e bloqueia padrões ruins.
17. **Gera resumo agregado de confiabilidade** – calcula pass-rate, Wilson lower bound e estatísticas globais do payload.
18. **Atualiza o controlador adaptativo de performance** – se o Advisor estiver lento e confiável, ele degrada módulos caros de forma controlada.

### 7.2 Seleção top-k

O top-k é calculado por:

```text
k = max(min_k, floor(n_trials * top_k_fraction))
```

Na implementação atual, além do `top_k_min` configurado, existe um reforço adaptativo:

```text
adaptive_min_k = min(20, max(3, floor(0.05 * n_trials)))
effective_min_k = max(top_k_min, adaptive_min_k)
```

Com isso, `k` nunca fica pequeno demais quando o estudo já acumulou muitos trials.

### 7.3 Projeção multiobjetivo

Quando o estudo é multiobjetivo, o Advisor não usa apenas `values[0]`. O fluxo é:

1. Extrai o vetor de objetivos por trial.
2. Aplica o sinal da direção: objetivos de minimização são invertidos.
3. Normaliza cada dimensão para o intervalo `[0, 1]` por min-max no conjunto observado.
4. Calcula um score base como média dos objetivos normalizados.
5. Calcula ranks de Pareto por non-dominated sorting.
6. Para até 3 objetivos e frente de Pareto pequena o bastante, calcula contribuição de hypervolume.
7. Ajusta o score final do trial por:

```text
score_final = score_base + 0.15 * hv_bonus - 0.05 * rank_penalty
```

onde `hv_bonus` é a contribuição normalizada de hypervolume do ponto e `rank_penalty = max(0, rank - 1)`.

### 7.4 Importâncias externas, internas e blended

O Advisor aceita importâncias externas do Optuna, mas também calcula importâncias internas.

- **Numéricos**: usa `|rho_spearman|` entre valor do parâmetro e score ajustado.
- **Categóricos**: usa uma medida análoga a ANOVA por variância explicada:

```text
eta2 = between_var / total_var
```

Se houver importâncias externas e internas, o Advisor mistura ambas:

```text
score = alpha * external + (1 - alpha) * internal
```

onde `alpha` depende da cobertura da importância externa e é truncado para o intervalo `[0.2, 0.8]`. Depois a mistura é renormalizada sobre os parâmetros presentes no search space.

### 7.5 Trust bucket

O `trust_bucket` é um estado por parâmetro que acompanha se os melhores trials recentes melhoraram batendo perto das bordas inferior ou superior.

- Se um novo melhor trial aparece perto da borda superior, incrementa `upper_success`.
- Se aparece perto da borda inferior, incrementa `lower_success`.
- Se várias melhorias deixam de acontecer, o bucket acumula falhas e eventualmente reseta os contadores.

Esse estado não substitui os cálculos estatísticos, mas funciona como reforço de memória para expansões direcionais.

## 8. Cálculos por tipo de recomendação

### 8.1 Estatística numérica básica

Para cada parâmetro numérico, o Advisor calcula estatísticas sobre todos os valores observados e também sobre os valores do top-k:

- média
- desvio-padrão amostral
- mínimo e máximo
- quantis `q10`, `q25`, `q50`, `q75`, `q90`

Os quantis são obtidos por interpolação linear no índice `q * (n - 1)`.

### 8.2 Expansão superior e inferior

No domínio numérico efetivo, o Advisor calcula:

```text
upper_proximity = (q90_top - low) / span
lower_proximity = (q10_top - low) / span
span = high - low
```

Para parâmetros logarítmicos, `low`, `high`, `q10_top` e `q90_top` são avaliados em `log10` antes desses cálculos.

- **`expand_upper`** exige sinal de borda superior e suporte monotônico compatível.
- **`expand_lower`** exige sinal de borda inferior e suporte monotônico compatível.

O aumento de faixa é de meia faixa adicional:

```text
new_high = high + 0.5 * span
new_low = low - 0.5 * span
```

No domínio log, isso equivale a uma expansão multiplicativa quando os valores são desnormalizados de volta.

### 8.3 Gating monotônico com Spearman

As expansões direcionais não dependem só de edge concentration. O Advisor calcula a correlação de Spearman entre o valor do parâmetro e o score ajustado:

```text
rho = cov(rank(x), rank(y)) / sqrt(var(rank(x)) * var(rank(y)))
```

Regras atuais:

- `expand_upper` só passa se `rho >= 0.15`.
- `expand_lower` só passa se `rho <= -0.15`.
- Para parâmetros sensíveis a custo, `expand_upper` exige evidência mais forte: `rho >= 0.25` e `importance >= 0.1`.

Se a cardinalidade for baixa e não houver evidência monotônica suficiente, a expansão é bloqueada mesmo que os top performers estejam perto da borda.

### 8.4 Narrow por concentração

O Advisor mede a concentração dos top performers pelo coeficiente de variação:

```text
CV = std_top / max(abs(mean_top), 1e-12)
```

Se `CV < 0.15` e houver trials suficientes, ele propõe estreitar o espaço para `[q10_top, q90_top]`.

Se esse intervalo ficar estreito demais, menor que 10% da faixa total, o Advisor recentra uma janela mínima de 10% da faixa em torno do `q50_top`.

### 8.5 Segurança BALLET com surrogate

Quando existe surrogate, o `narrow` passa por um teste de segurança adicional:

1. O surrogate avalia 25 pontos uniformemente distribuídos na faixa atual.
2. Para cada ponto, estima média e desvio-padrão pela variância entre árvores da floresta.
3. Constrói:

```text
UCB = mean + 1.96 * std
LCB = mean - 1.96 * std
```

4. Aceita o estreitamento apenas se:

```text
max(LCB dentro da nova faixa) > max(UCB fora da nova faixa)
```

Se isso falhar, o `narrow` é trocado por `keep` com racional explicando que o surrogate ainda enxerga região externa potencialmente competitiva.

### 8.6 Fix por baixa importância

Se nenhuma ação mais forte foi disparada e a importância final do parâmetro for menor que `0.05`, o Advisor propõe `fix` no `q50` do top-k para parâmetros numéricos ou na categoria mais frequente do top-k para categóricos.

### 8.7 Change distribution

Quando o spec não força `log`, o Advisor pode sugerir `change_distribution` para `log_uniform` se:

- o nome do parâmetro sugerir escala logarítmica (`lr`, `learning_rate`, `weight_decay`, `lambda`, `kl_weight`, `min_delta`), ou
- a razão `high / low` for maior que `100`

Essa recomendação só é válida se `low > 0` e `high > 0`.

### 8.8 Cálculo categórico

Para parâmetros categóricos, o Advisor conta ocorrências no top-k e mede diversidade efetiva.

Entropia:

```text
H = -sum(p_i * log(p_i))
```

Número efetivo de categorias:

```text
N_eff = exp(H)
```

Com surrogate, ele avalia cada categoria em contexto âncora e mantém:

- a melhor categoria por `LCB`
- qualquer categoria cujo `UCB` ainda seja competitivo com esse melhor `LCB`

Sem surrogate, a regra empírica usa dominância no top-k e preserva pelo menos uma runner-up para evitar colapso prematuro do espaço.

## 9. Como a confiabilidade é calculada

O Advisor não usa uma única métrica de confiabilidade. Há várias camadas.

### 9.1 Incerteza heurística

A incerteza local de cada recomendação é:

```text
trial_factor = min(1.0, n_trials / 30)
top_k_factor = min(1.0, top_k_count / 20)
uncertainty = 1.0 - (trial_factor * top_k_factor)
```

Quanto mais trials e quanto maior o top-k observado, menor a incerteza.

### 9.2 Bootstrap support

Depois de decidir a ação final, o Advisor reamostra os trials com reposição por `bootstrap_samples = 50` e repete a análise.

O suporte da ação é:

```text
bootstrap_support = acertos / amostras_validas
```

onde “acerto” significa que a análise da amostra bootstrap reproduziu exatamente a mesma ação final do parâmetro.

### 9.3 Rótulo de confiança

O rótulo textual de confiança é reclassificado a partir do suporte bootstrap:

- `high` se `bootstrap_support >= 0.75`
- `medium` se `bootstrap_support >= 0.50`
- `low` caso contrário

Se não houver bootstrap, o Advisor mantém a confiança base inferida da evidência observada.

### 9.4 Confidence score numérico

Além do rótulo, cada recomendação recebe `confidence_score` em `[0, 1]`.

O score base do rótulo é:

- `high = 0.85`
- `medium = 0.60`
- `low = 0.35`

O suporte bootstrap é calibrado com prior Beta simplificado, usando prior neutro de peso `20` e média `0.5`:

```text
calibrated_support = (support * evidence_count + 0.5 * 20) / (evidence_count + 20)
```

Com suporte calibrado, o score final é:

```text
confidence_score = clamp(0, 1,
	0.45 * base_score + 0.55 * calibrated_support - 0.15 * uncertainty
)
```

Sem suporte bootstrap, mas com incerteza disponível, o Advisor usa:

```text
confidence_score = clamp(0, 1,
	base_score * (1 - 0.4 * uncertainty)
)
```

### 9.5 Validação dura do payload

Cada recomendação passa por validação estrutural e semântica:

- `expand_upper` falha se `new_high <= old_high`, se `new_high` não for finito ou se crescer acima do fator máximo configurado.
- `expand_lower` falha se `new_low >= old_low`, se cruzar para negativo quando o espaço original era não negativo ou se a expansão exceder o fator máximo.
- `narrow` falha se `new_low >= new_high`.
- `reduce_categories` falha se `keep` ficar vazio ou se houver sobreposição entre `keep` e `remove`.
- `fix` falha se o valor não existir.
- `change_distribution` para `log_uniform` falha se os limites não forem positivos ou se o intervalo for inválido.

Quando a validação falha, a ação é rebaixada para `keep`, mas a recomendação continua aparecendo com `blocked_action` e `blocked_reason` para auditoria.

### 9.6 Wilson lower bound

Para medir robustez agregada, o Advisor usa o limite inferior de Wilson com `z = 1.96`:

```text
LB = (p + z^2 / (2n) - z * sqrt((p * (1 - p) + z^2 / (4n)) / n)) / (1 + z^2 / n)
```

onde `p = successes / total`.

Esse cálculo aparece em dois lugares principais:

- `validation_pass_wilson_lb`
- `high_confidence_wilson_lb`

O objetivo é não confiar apenas na taxa observada bruta quando a amostra ainda é pequena.

### 9.7 Self-audit periódico

O self-audit reexecuta o Advisor em prefixes históricos e valida ações direcionais contra suffixes futuros.

Fluxo:

1. Seleciona prefixes periódicos no histórico.
2. Em cada prefix, roda um Advisor leve, com surrogate e interações desligados.
3. Pega apenas ações direcionais: `expand_upper` e `expand_lower`.
4. Mede o Spearman no suffix futuro para o mesmo parâmetro.
5. Marca acerto se:
   - `expand_upper` encontrar `rho > 0`
   - `expand_lower` encontrar `rho < 0`
6. Agrega acertos e totais por `param_name + action`.
7. Calcula `hit_rate` e `hit_rate_wilson_lb`.
8. Se um grupo tiver amostra suficiente, `hit_rate < 0.5` e `Wilson LB < self_audit_wilson_block`, ele vira um `villain`.
9. Recomendações atuais com o mesmo `param_name + action` são bloqueadas e rebaixadas para `keep`.

No projeto atual, `self_audit_wilson_block = 0.35`.

### 9.8 Resumo agregado de confiabilidade

O campo `metadata.reliability_summary` resume a qualidade do payload atual:

- `total`
- `actionable`
- `blocked`
- `validation_pass_rate`
- `validation_pass_wilson_lb`
- `mean_confidence_score`
- `high_confidence_rate`
- `high_confidence_wilson_lb`

Esse resumo é usado tanto para diagnóstico no dashboard quanto para permitir ou não a degradação adaptativa de custo.

## 10. Kernels, surrogate e aceleração

### 10.1 O Advisor usa kernels estatísticos?

Não. A implementação atual do pacote `search_space_advisor` não usa KDE, Gaussian kernel, RBF, Epanechnikov, tricube, triweight nem bandwidth selection.

O modelo auxiliar do Advisor é uma floresta aleatória, não um método kernelizado.

### 10.2 Surrogate usado pelo Advisor

O surrogate atual é:

- `RandomForestRegressor`
- `n_estimators = 64`
- `max_depth = 8`
- `random_state = 42`

Pré-processamento:

- categóricos com `OneHotEncoder(handle_unknown="ignore")`
- numéricos em passthrough
- parâmetros logarítmicos convertidos para domínio `log10`

O desvio-padrão preditivo usado pelo Advisor vem da dispersão das predições entre árvores individuais da floresta.

### 10.3 Interações

Se `enable_interactions = true` e o surrogate foi ajustado, o Advisor tenta calcular `shap_interaction_values` e reduz isso a uma força média de interação por par de parâmetros.

O limiar adaptativo é:

```text
interaction_threshold = max(0.05, 1.5 * mean(interaction_strengths))
```

Se um parâmetro estiver acima desse limiar, o Advisor evita `fix`, `narrow` e algumas reduções categóricas frágeis.

### 10.4 Aceleração Rust

O Advisor usa uma aceleração opcional em Rust para Spearman: `fast_spearman_corr`, exportada pelo pacote `pff_rust`.

Ela entra apenas quando:

- a extensão Rust está disponível
- `numpy` está disponível
- o vetor tem tamanho maior ou igual a `rust_spearman_min_len`

No projeto atual, `rust_spearman_min_len = 512`.

### 10.5 O que exatamente o “kernel” Rust faz

Embora a função viva no módulo Rust `shared::kernels`, ela não é um kernel de ML. Ela só acelera o cálculo da correlação de Spearman com tratamento explícito de empates:

1. Converte os vetores `x` e `y` em ranks médios (`rankdata` com tie handling).
2. Calcula média dos ranks.
3. Calcula covariância e variâncias dos ranks.
4. Retorna:

```text
rho = cov / sqrt(var_x * var_y)
```

Se houver `NaN`, `Inf`, cardinalidade insuficiente, tamanhos incompatíveis ou uma das séries for constante, a função retorna `None` e o Advisor cai no caminho Python puro, determinístico e equivalente.

## 11. Mapa rápido do código

Os arquivos-chave da implementação atual são:

- `src/pff/infrastructure/hpo/search_space_advisor/service.py` – orquestração principal.
- `src/pff/infrastructure/hpo/search_space_advisor/statistics.py` – quantis, Spearman, top-k, incerteza.
- `src/pff/infrastructure/hpo/search_space_advisor/analysis_numeric.py` – ações numéricas e teste BALLET.
- `src/pff/infrastructure/hpo/search_space_advisor/analysis_categorical.py` – ações categóricas.
- `src/pff/infrastructure/hpo/search_space_advisor/surrogate.py` – surrogate, UCB/LCB e interações.
- `src/pff/infrastructure/hpo/search_space_advisor/confidence.py` – confidence score.
- `src/pff/infrastructure/hpo/search_space_advisor/reliability.py` – Wilson LB e resumo agregado.
- `src/pff/infrastructure/hpo/search_space_advisor/self_audit.py` e `self_audit_runner.py` – auditoria temporal e bloqueios.
- `src/pff/infrastructure/hpo/search_space_advisor/cache.py` – cache L1/L2.
- `src/pff/infrastructure/hpo/search_space_advisor/multiobjective.py` – projeção multiobjetivo.
- `src/pff_rust/src/shared/kernels.rs` – implementação Rust de `fast_spearman_corr`.

---

**Resumo**: O Search Space Advisor cruza a definição de espaço do Optuna com os resultados dos trials e estatísticas do dataset para ranquear ações (Expandir/Reduzir/Fixar/Manter). O dashboard expõe esses cálculos com visualizações de densidade, régua topológica, importâncias e um painel detalhado em português, permitindo que operadores entendam o porquê de cada sugestão antes de aplicar patches no YAML de HPO.
