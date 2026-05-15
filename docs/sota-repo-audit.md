# Auditoria SOTA do repositório PFF

Data: 2026-05-15

## Escopo

Esta auditoria cobre arquitetura, organização de pastas, tecnologias, bibliografia operacional,
scripts, complexidade/guardrails, containers e dashboard desacoplado. Ela registra evidências
locais e fontes externas consultadas para orientar os ajustes priorizados.

## Evidência local

| Área                | Evidência                                                                                                                                                                                                                           | Decisão                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| Arquitetura          | `src/pff/{drivers,application,domain,infrastructure,shared}` e 18 testes em `tests/architecture/`                                                                                                                                | Estrutura compatível com Clean/Hexagonal; manter guardrails.                |
| Search Space Advisor | `src/pff/infrastructure/hpo/search_space_advisor/**`, versão 2.3.0, relatório em `docs/deep-research-report-abnt.pdf`, auditoria real 50-complete em `outputs/benches/search_space_advisor/deep_research_audit_20260506.json`        | Advisor alinhado ao anexo e com evidências empíricas no PDF; sem reivindicação universal contra GP-BO/BALLET. |
| Dashboard            | `npm run verify` em `src/pff/infrastructure/hpo/dashboard`                                                                                                                                                                       | Dashboard permanece desacoplado e sem barrel wrapper obsoleto.               |
| Docker               | `pff:cpu-lock-check` 2.84 GB / 2.64 GiB; `pff:cuda-lock-check` 8.45 GiB; `pff:tools-slim-check` 14.8 GB; `pff:test-slim-check` 15.9 GB; lock principal CPU e requisito público `torch==2.7.0`; `scripts/package/measure-image-sizes.sh`; gate CI com `load: true`, cache `type=gha` e TSV de orçamento; `docs/docker-runtime-matrix.md` | Runtime `./scripts/package/pff-run` permanece GPU-first; build padrão é CPU para economia; orçamento agora falha CI. |
| Static tooling       | `pyproject.toml` com mypy, pylint e pyright em Python 3.12                                                                                                                                                                         | Corrigido desalinhamento de Pyright 3.14 para 3.12.                          |
| Testes novos         | `tests/unit/test_pyproject_tooling_contract.py`                                                                                                                                                                                    | Garante que os analisadores estáticos sigam o runtime declarado.            |

## Fontes externas consultadas

| Tema         | Fonte                                                                                                                       | Uso na auditoria                                                                                                        |
| ------------ | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Optuna       | Documentação estável via Context7:`optuna.study.create_study`, `TPESampler`, `NSGAIISampler`, `constraints_func` | Confirmar defaults atuais: TPE para single-objective e NSGA-II para multi-objective quando sampler é `None`.         |
| scikit-learn | Documentação estável via Context7:`RandomForestRegressor`                                                              | Confirmar parâmetros e papel de `random_state`, `n_estimators`, `max_depth` e bootstrap no surrogate do Advisor. |
| Poetry       | Documentação oficial de dependency groups                                                                                 | Evitar grupos CPU/CUDA conflitantes: o Poetry resolve grupos juntos mesmo quando opcionais.                             |
| PyTorch      | Documentação oficial de instalação local                                                                                | Manter seleção explícita de wheel CPU no lock principal e wheel CUDA no target `cuda`.                             |
| Docker/GHA   | Documentação oficial do backend de cache GitHub Actions para Buildx e `docker/build-push-action`                         | Acoplar orçamento de imagem a build carregado no daemon com cache `type=gha`.                                           |
| ABNT         | Consulta web sobre NBR 14724:2024 e atualizações institucionais                                                           | Confirmar que o relatório deve seguir a edição 2024 para apresentação acadêmica.                                  |
| Evidência estatística | Demsar (JMLR 2006), Wilcoxon/Friedman/CD diagrams; Wolpert & Macready (IEEE TEC 1997), No Free Lunch | Definir por que uma vitória local do Advisor não sustenta reivindicação universal e qual protocolo mínimo sustenta uma reivindicação forte. |
| HPO moderno  | Bergstra et al. (NeurIPS 2011), Optuna `GPSampler`, Optuna `HyperbandPruner`, BOHB/Falkner et al. (ICML 2018)              | Comparar TPE, GP-BO e bandit/pruning com orçamento pareado e evitar conclusões com 5 sementes em um único cenário. |
| KGC          | RotatE/Sun et al. (ICLR 2019)                                                                                              | Priorizar ganhos orgânicos em ranking: capacidade, negativos auto-adversariais, padrões relacionais e avaliação filtrada. |

## Achados

1. **Arquitetura:** estado forte. Os guardrails cobrem I/O no domínio/aplicação, shared-first, parquet-first, logging e entrypoints.
2. **Advisor:** estado forte para a versão atual. O relatório ABNT agora usa figuras do Advisor em si: ações, confiança, validação, top-k, confiabilidade, Pareto e self-audit. A camada de diagnóstico de parâmetros fixos foi adicionada para alertar quando um valor congelado merece exploração ou quando há evidência suficiente para mantê-lo.
3. **Docker:** melhora material. O cache do Poetry não fica nas imagens finais, o build padrão evita criar `cuda`, `tools` e `test` sem necessidade, o runtime CPU usa a wheel `torch==2.7.0+cpu` sem pacotes `nvidia-*-cu12` ou `triton`, o requisito público permanece `torch==2.7.0` para aceitar CPU/CUDA, e há script TSV para comparar tamanhos contra baseline.
4. **Dashboard:** estado saudável. O verificador do bundle foi atualizado para o contrato sem barrel wrappers.
5. **Tooling:** corrigido. Pyright usava Python 3.14 enquanto o projeto declara Python 3.12.
6. **Risco remanescente:** o target CUDA foi buildado sem GPU física e valida import/package; ainda falta smoke real em host NVIDIA para confirmar `torch.cuda.is_available()`.
7. **Reivindicação SOTA do Advisor:** ainda não é defensável como universal nem como SOTA local robusto. O benchmark pareado foi ampliado para múltiplos cenários sintéticos PFF-KGC, com IC bootstrap, Wilcoxon pareado, Friedman e Holm. A melhor triagem curta com 3 seeds (`advisor_edge_gated_gp`) superou levemente GP-BO, mas esse sinal caiu em 20 seeds. A investigação mostrou que parte dos ganhos/perdas de `advisor_embedding_upper_gp` vinha de reinícios sem mudança material do espaço; o benchmark agora bloqueia esse caso como `no_material_change`. O No Free Lunch impede transformar sinais locais frágeis em superioridade geral.

## Por que a reivindicação universal ainda não sustenta

O resultado atual sustenta uma afirmação metodológica, não uma vitória SOTA: neste benchmark sintético, políticas Advisor conservadoras podem empatar GP-BO quando evitam reinícios sem mudança real, mas ainda não demonstram ganho real após a correção `no_material_change`. Ele não sustenta uma afirmação universal por quatro razões:

1. **Sem ganho limpo contra GP-BO:** no protocolo multi-cenário limpo com 20 seeds, 30 trials, Advisor isolado e políticas `gp_bo`, `advisor_edge_gated_gp`, `advisor_embedding_upper_gp`, todas as políticas Advisor empataram exatamente com GP-BO em 80/80 pares: média 0.750145, delta 0.000000, IC95 [0.000000, 0.000000], Wilcoxon/Holm nulos por deltas todos zero. Os skips foram `edge_gate`, `no_expand_upper`, `upper_edge_not_allowed`, `patch_not_allowed` e `no_material_change`; nenhum patch material gerou ganho verificável.
2. **Amostra estatística curta:** 3 seeds por cenário não bastam para reivindicação robusta contra múltiplas políticas; no mínimo, usar 20-30 sementes por cenário ou mais quando o efeito observado for pequeno.
3. **Cobertura ainda sintética:** os quatro cenários ajudam a testar formas de paisagem, mas a evidência precisa cobrir datasets reais, tamanhos de KG, ruído, budgets e famílias de sampler/pruner.
4. **Protocolo multi-algoritmo ainda incompleto:** Random, TPE, TPE+Hyperband, GP-BO, Advisor e ablations já estão no script; BOHB pleno e pós-testes Nemenyi/Holm ainda faltam para uma reivindicação forte completa.

Recorte positivo defensável: a política `advisor_embedding_upper_gp` só aplica a recomendação quando a borda superior influente é exclusivamente compatível com `embedding_dim` e o patch também é restrito a `embedding_dim`; além disso, agora só reinicia se o espaço realmente mudou. A hipótese científica remanescente é que o Advisor pode ser útil quando detectar capacidade subdimensionada fora do teto configurado, mas o protocolo atual ainda não demonstrou ganho real sobre GP-BO.

Correção metodológica: o protocolo não deve consumir orçamento quando o Advisor retorna `empty_patch` ou quando o gate bloqueia a intervenção. O benchmark foi ajustado para registrar o skip e ainda avaliar o trial corrente. Essa correção remove uma penalização artificial contra políticas conservadoras do Advisor.

Camada metodológica nova: parâmetros fixos agora recebem recomendação informativa com `action=keep` e `recommendation.diagnostic`. O Advisor diferencia `needs_exploration` (fixo importante, sensibilidade não estimável), `stable_fixed_value` (fixo pouco importante com evidência suficiente) e `watch_fixed_value` (evidência fraca). Isso cobre casos como `embedding_dim=512`: o valor não é promovido a ótimo sem varredura local, mas também não é explorado desnecessariamente quando a evidência o classifica como estável.

Hipótese testada e reprovada: uma variante `advisor_trust_region_gp`, inspirada por trust-region BO, foi adicionada ao benchmark como ablação. Na triagem curta multi-cenário, ela piorou o agregado contra GP-BO (média 0.732598; delta -0.051305; IC95 [-0.100455, -0.004219]; 3 vitórias e 9 derrotas), ficando abaixo de `advisor_static_gp`. Portanto, a próxima tentativa deve focar orçamento multi-fidelidade/BOHB ou integração mais fiel com o sampler vencedor, não estreitamento local ingênuo.

Baseline multi-fidelidade: `tpe_hyperband` foi adicionado com intermediários simulados e `HyperbandPruner` real do Optuna. Na triagem curta isolada mais recente, perdeu para GP-BO em todos os pares (média 0.665865; delta -0.118037; IC95 [-0.145598, -0.086562]). Isso confirma que o pruner é útil como baseline obrigatório, mas não resolveu o gargalo central deste protocolo.

Ferramenta metodológica nova: o benchmark agora aceita `--policies`, permitindo rodadas focadas, reproduzíveis e baratas contra baselines específicos sem remover o protocolo completo padrão. O artefato JSON também inclui correção Holm pareada (`holm_vs_tpe_pure` e `holm_vs_gp_bo`) além de Friedman/Wilcoxon, para controlar múltiplas comparações. O campo `claim_decision` aplica a regra mínima `positive_mean_delta_and_holm_adjusted_pvalue_below_0.05` e retorna `stop_reason=not_supported_by_paired_holm_test_vs_gp_bo` quando a reivindicação contra GP-BO não sustenta. Isso foi usado para confirmar que o sinal de `advisor_edge_gated_gp` com 3 seeds não generalizava, testar `advisor_embedding_upper_gp` e depois eliminar reinícios sem mudança material do espaço.

Critério para sustentar uma reivindicação forte, sem exagero:

- matriz pareada com pelo menos 6-10 cenários reais/sintéticos relevantes e 20-30 sementes por cenário;
- mesmo orçamento por política, mesmas seeds, mesmo timeout, mesmo número de workers e logs completos;
- baselines: Random, TPE puro, TPE+Hyperband, GP-BO, BOHB quando aplicável, Advisor completo e ablations;
- efeito reportado como média, mediana, intervalo de confiança/bootstrap, taxa de vitórias, Wilcoxon pareado e Friedman com pós-testes;
- reivindicação textual limitada ao domínio observado, por exemplo: "SOTA no benchmark PFF-KGC sob este orçamento e protocolo", não "universalmente superior".

## Plano orgânico para 0.7

Estado factual do estudo `deep_research_advisor_real50_gpu_20260506`:

- 50 trials completos em 60 entradas totais; 8 `RUNNING` antigos e 2 `PRUNED` não entram na evidência de qualidade.
- Melhor score observado: 0.469644; mediana dos completos: 0.452441; média: 0.449473.
- Melhor trial: `embedding_dim=512`, `t_norm=godel`, `batch_size=455`, `negative_sample_size=512`, `num_global_negatives=96`, `validate_every=3`, `learning_rate=0.001704`, `lambda_pc=0.030451`, `lambda_logic=0.023992`, `contrastive_temperature=0.056256`.
- Métricas do melhor trial: MRR 0.346742, Hits@10 0.493207, MCC 0.325509, AUC 0.710417, duração 728.5 s.
- Correlações com o objetivo nos 50 completos: `duration` negativa forte (Pearson -0.623), MRR/MCC positivas (~0.45), `learning_rate` Spearman +0.357, `num_global_negatives` +0.237, `batch_size` +0.232, `validate_every` -0.290, `rebuild_every` -0.293.

Interpretação: para chegar a 0.7 sem mexer artificialmente na fórmula, o ganho principal precisa vir de qualidade de ranking e classificação, não só de tempo. A função atual pesa fortemente o bloco de ranking; portanto, MRR por volta de 0.35 e Hits@10 por volta de 0.50 ainda deixam o score longe de 0.7.

| Fase | Objetivo | Mudança orgânica guiada pelo Advisor | Gate de aceitação |
| ---- | -------- | ------------------------------------- | ----------------- |
| 1 | Reduzir ruído e confirmar o vale bom | Aplicar `narrow/fix/reduce`: `embedding_dim=512`, `max_communities=64`, `t_norm in {godel, product}`, `batch_size=409..478`, `negative_sample_size=448..640`, `num_global_negatives=96`, `validate_every=2..3`, `early_stopping_patience=6..7`, `pruning_threshold=0.0108..0.0294`, `contrastive_temperature=0.056..0.075`. Não expandir `contrastive_temperature` acima de 0.08 porque o self-audit marcou `expand_upper` como vilão. | 30 trials novos; melhor score >=0.50 ou mediana top-10 >=0.47; sem NaN/Inf; comparação pareada contra o espaço anterior. |
| 2 | Explorar o único sinal de expansão forte | Expandir `learning_rate` de `0.002` para teto controlado `0.006..0.009` em escala log, mantendo clip/guard de overflow e rollback se `binary_loss`/NaN piorar. | 40 trials; Wilson/self-audit não pode bloquear `expand_upper`; melhor score >=0.55. |
| 3 | Aumentar capacidade com custo controlado | Testar `embedding_dim in {512, 768, 1024}` e `attr_hidden_dim in {256, 512}` só depois da fase 1 fixar o restante; manter GPU-first e medir duração/memória. | 30-50 trials; MRR >=0.42 ou Hits@10 >=0.58 sem duração mediana >1200 s. |
| 4 | Melhorar ranking por negativos | Trazer o padrão RotatE/self-adversarial para o espaço real: destravar `adversarial_temperature` e `hard_neg_ratio`, testar negativos relacionais/degree-based mais agressivos e `negative_sample_size` perto de 512/576 antes de 640. | MRR >=0.48, Hits@10 >=0.63 e MCC >=0.40 em validação filtrada. |
| 5 | Melhorar dados e avaliação | Rodar diagnóstico por relação: cobertura, relações raras, vazamento inverso, split temporal/estratificado, triples duplicados e qualidade de negativos. Corrigir dados antes de ampliar HPO se houver relação dominante ou label noise. | Relatório por relação; nenhum split leakage; score >=0.60 antes de ampliar arquitetura. |
| 6 | Comparação SOTA defensável | Rodar campanha pareada real com TPE, GP-BO, Hyperband/BOHB quando aplicável, Advisor e ablations no mesmo orçamento. | Score alvo >=0.70 em pelo menos 2 rodadas independentes ou melhoria estatisticamente significativa sobre baselines; relatório com Wilcoxon/Friedman e IC bootstrap. |

Próximo experimento recomendado: iniciar uma campanha curta de 30 trials com o espaço estreitado da Fase 1 e `learning_rate` expandido moderadamente (`1e-4..6e-3`) como único eixo agressivo. Se ela não passar de 0.50, não vale gastar uma madrugada em 200 trials; primeiro investigar dados/relações e gargalo do ranking.

## Priorização

| Prioridade | Achado                                                           | Impacto | Esforço | Risco  | Urgência | Decisão                                                                                                                                                                                                                      |
| ---------: | ---------------------------------------------------------------- | ------- | -------- | ------ | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|          1 | Imagem CPU carregava payload CUDA do lock                        | Alto    | Médio   | Baixo  | Alta      | Corrigido no Dockerfile e no lock; runtime CPU medido com `torch==2.7.0+cpu`, 2.84 GB no Docker CLI e 2.64 GiB por bytes.                                                                                                   |
|          2 | Build padrão criava imagens além do necessário                | Alto    | Baixo    | Baixo  | Alta      | Corrigido: target padrão `cpu`; `runtime`, `tools`, `test` e `all` são explícitos.                                                                                                                               |
|          3 | Ausência de medição reproduzível de tamanho                  | Médio  | Baixo    | Baixo  | Alta      | Corrigido com `scripts/package/measure-image-sizes.sh` e teste de baseline/delta.                                                                                                                                           |
|          4 | Relatório do Advisor não tinha prova formal explícita         | Alto    | Médio   | Baixo  | Alta      | Corrigido com definições, lemas, teorema, demonstrações, limites e PI sem afirmar patenteabilidade.                                                                                                                       |
|          5 | Pyright declarava Python 3.14 enquanto o projeto usa 3.12        | Médio  | Baixo    | Baixo  | Média    | Corrigido em `pyproject.toml` e coberto por teste de contrato.                                                                                                                                                              |
|          6 | Dashboard tinha verificador de bundle preso a arquivo legado     | Médio  | Baixo    | Baixo  | Média    | Corrigido em `verify_bundle.js`; `npm run verify` passou.                                                                                                                                                                 |
|          7 | Lock principal instalava CUDA antes da troca para CPU no builder | Médio  | Médio   | Baixo  | Média    | Corrigido: `poetry.lock` trava a wheel CPU, `pyproject.toml` exige `torch==2.7.0`, e o lock não trava `triton`/`nvidia-*-cu12`.                                                                                                          |
|          8 | Backtest do Advisor ainda usa recorte curto de 25 trials         | Alto    | Alto     | Médio | Média    | Corrigido como evidência 50-complete: estudo `deep_research_advisor_real50_gpu_20260506` auditado com 50 trials completos no dashboard, melhor objetivo 0.469644, 21 recomendações, 37 prefixos avaliados, hit-rate direcional 0.7903 e validação Wilson-LB 0.7733. O benchmark pareado sintético agora cobre múltiplos cenários, Random, TPE, TPE+Hyperband, GP-BO, Advisor e ablations, com IC bootstrap, vitórias pareadas, Wilcoxon, Friedman e Holm. Resultado honesto: as aparentes melhorias curtas contra GP-BO não sobreviveram a mais seeds e a auditoria de mudança material do espaço; portanto ainda não há reivindicação SOTA. PDF ABNT não deve reivindicar superioridade SOTA até haver evidência real mais forte. |
|          9 | Orçamento Docker ainda não roda como gate de CI real           | Médio  | Médio   | Médio | Baixa     | Corrigido; `.github/workflows/ci.yml` carrega `pff:ci` no daemon via `docker/build-push-action` com `load: true`, usa cache `type=gha,scope=pff-runtime-cpu`, roda `measure-image-sizes.sh --fail-on-budget` e publica `outputs/benches/docker/image-sizes-ci.tsv`. |

## Próximas melhorias priorizadas

1. Rodar smoke CUDA real em host NVIDIA para validar `torch.cuda.is_available()` no `pff:cuda`.
2. Rodar a suíte completa de arquitetura sempre que houver corte transversal em `src/pff/shared/**` ou Dockerfile.
3. Rodar o gate Docker no GitHub Actions para observar cache `type=gha` e orçamento em runner real.
