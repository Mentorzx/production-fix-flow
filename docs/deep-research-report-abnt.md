---
title: "Confiabilidade matemática de um Search Space Advisor para otimização de hiperparâmetros"
subtitle: "Formulação geral, exemplos operacionais e estudo de caso no projeto PFF"
author:
 - "Alex de Lira Neto"
date: "Salvador, 2026"
lang: pt-BR
documentclass: article
fontsize: 12pt
papersize: a4
linestretch: 1.5
geometry:
 - left=3cm
 - right=2cm
 - top=3cm
 - bottom=2cm
bibliography: references.bib
csl: abnt.csl
link-citations: true
reference-section-title: Referências
header-includes:
 - \usepackage{indentfirst}
 - \usepackage{booktabs}
 - \usepackage{float}
 - \floatplacement{figure}{H}
 - \usepackage{longtable}
 - \usepackage{array}
 - \usepackage{etoolbox}
 - \usepackage{caption}
 - \captionsetup{labelfont=bf,font=small}
 - \AtBeginEnvironment{longtable}{\small}
 - \setlength{\LTleft}{0pt}
 - \setlength{\LTright}{0pt}
 - \setlength{\parindent}{1.25cm}
 - \setlength{\parskip}{0pt}
abstract: |
 Este artigo formaliza um Search Space Advisor como camada de decisão para adaptação de espaços de busca em otimização de hiperparâmetros. A formulação é apresentada de modo geral, compatível com métricas de maximização, minimização ou cenários multiobjetivo, e depois é instanciada em um estudo de caso no projeto PFF. O método combina seleção top-k adaptativa, quantis, correlação de Spearman, coeficiente de variação, mistura de importâncias, surrogate RandomForestRegressor, suporte bootstrap, limite inferior de Wilson, self-audit temporal e diagnóstico de parâmetros fixos. No estudo de caso executado em Docker, obtêm-se recomendações consistentes e um payload estruturalmente válido, com evidência temporal ainda curta para ações direcionais. O resultado central é que a confiabilidade do Advisor depende de evidências acumuladas e auditáveis, e não de uma única heurística.
---

\noindent\textbf{Palavras-chave:} Search Space Advisor; otimização de hiperparâmetros; confiabilidade; self-audit; limite inferior de Wilson; otimização multiobjetivo.

\noindent\textbf{Keywords:} Search Space Advisor; hyperparameter optimization; reliability; self-audit; Wilson lower bound; multi-objective optimization.

# Dados institucionais {.unnumbered}

\begin{tabular}{@{}p{0.24\linewidth}p{0.70\linewidth}@{}}
\textbf{Autor} & Alex de Lira Neto \\
\textbf{Instituição} & Universidade Federal da Bahia (UFBA), Salvador, Bahia, Brasil \\
\textbf{Curso} & Engenharia da Computação, graduação em andamento \\
\textbf{Atuação} & Pesquisa em detecção de anomalias \\
\textbf{Orientação} & Antônio Carlos Fernandes, professor da UFBA \\
\end{tabular}

# Introdução

Parte-se do fato de que a otimização de hiperparâmetros aparece em problemas muito diferentes entre si: classificação, regressão, ranqueamento, séries temporais, modelos generativos, sistemas sujeitos a latência e pipelines multiobjetivo com restrições de custo. Em todos esses casos, cada trial é uma avaliação cara de uma função caixa-preta, e é preciso decidir não só qual configuração parece boa, mas também se o espaço de busca atual ainda faz sentido. Um Search Space Advisor atua justamente nesse ponto: ele não substitui o otimizador principal, e sim reinterpreta o histórico de trials para sugerir quando expandir, estreitar, fixar ou reduzir partes do espaço de busca.

O método discutido neste artigo é formulado de modo geral. Ele só pressupõe a existência de um histórico de configurações avaliadas, uma direção de otimização (`maximize`, `minimize` ou projeção multiobjetivo) e uma representação explícita do espaço de busca por parâmetro. Isso permite reutilizar a mesma lógica com métricas como acurácia, F1, AUPRC, RMSE, MRR, custo, latência ou combinações dessas métricas. Quando um parâmetro aparece fixado, o Advisor não o promove automaticamente a ótimo: ele emite um diagnóstico informativo para distinguir valor fixo estável, valor fixo crítico que merece varredura local e valor ainda sem evidência suficiente. O projeto PFF entra aqui como estudo de caso concreto: ele fornece um experimento em Docker, um payload HPO auditável e um conjunto de recomendações efetivamente produzidas pelo Advisor.

O ponto metodológico mais importante é separar três camadas. A primeira é a teoria geral de HPO, que inclui busca aleatória, TPE, Pareto e otimização bayesiana clássica [@bergstra2012random; @bergstra2011algorithms; @snoek2012practical]. A segunda é o núcleo operacional do Advisor, implementado em `src/pff/infrastructure/hpo/search_space_advisor/**`. A terceira é a camada de confiabilidade, que usa validação, bootstrap, limite inferior de Wilson e self-audit. Essa separação evita atribuir ao código garantias que pertencem apenas à literatura de apoio e mantém a formulação aplicável a outros projetos além do PFF.

# Materiais e métodos

O artigo é organizado em duas camadas complementares. Na camada geral, formaliza-se um Advisor agnóstico ao domínio, capaz de operar sobre parâmetros numéricos e categóricos, métricas escalares ou multiobjetivo e políticas de maximização ou minimização. Na camada empírica, usa-se o PFF como estudo de caso para testar se a formulação produz recomendações coerentes, figuras rastreáveis e anexos reprodutíveis.

O experimento do estudo de caso em Docker, conforme solicitado, com o wrapper `./scripts/package/pff-run`, dados parquet_local, 50 trials planejados, `no-update-config`, `no-bert`, `no-dashboard` e estudo `deep_research_advisor_real50_gpu_20260506`.

A auditoria offline foi executada com `search_space_advisor_audit.py`, usando o payload real `dashboard_data.json`, `min-prefix=8` e saída registrada nos artefatos do recorte `cutoff25`.

No estudo de caso PFF, `sampler=TPESampler` e a direção de entrada `MAXIMIZE` foi normalizada para `maximize`. O dataset usado nessa rodada tinha 10231 triplas de treino, 1240 de validação, 5730 entidades e 30 relações. A campanha foi planejada para 60 trials, mas este recorte parcial usa 50 trials completos observados no dashboard; o Advisor auditou 50 trials completos. O resumo agregado de HPO disponível em disco não refletia este mesmo corte analítico e, por isso, não foi tratado como fonte factual para contagem, tempo total ou melhor escore desta rodada parcial.

Preserva-se a distinção entre fontes, mas evita-se comparar artefatos de épocas diferentes: neste recorte parcial, o melhor objetivo observado no dashboard foi 0.469644. Como o resumo agregado em disco estava desatualizado para esta mesma rodada parcial, não se usa o `best_value` dele como contraponto numérico nesta seção.

## Fórmulas por caso e vínculo com a implementação

A seguir, organizam-se as fórmulas pelo papel que exercem. As equações de otimização e de sobrevivência funcionam como molduras teóricas; as demais, quando indicado, correspondem ao caminho operacional do Advisor versão 2.3.0.

### Otimização caixa-preta e amostragem

O problema geral de HPO é tratado como otimização caixa-preta:

$$x^* = \arg\max_{x \in \mathcal{X}} f(x)$$

ou, para métricas de perda, como $x^* = \arg\min_{x \in \mathcal{X}} f(x)$. Na busca aleatória, cada tentativa é sorteada de uma distribuição configurada:

$$x_t \sim p(x).$$

No TPE usado pelo Optuna, escolhe-se o próximo candidato maximizando a razão entre a densidade dos bons pontos e a densidade dos demais pontos [@optunaTPE; @bergstra2011algorithms]:

$$x^* = \arg\max_x \frac{l(x)}{g(x)}.$$

Exemplo prático. Se `learning_rate` e `temperature` aparecem com frequência entre os melhores trials, o numerador $l(x)$ cresce nessa região e o TPE passa a revisitá-la com prioridade. Isso significa que regiões promissoras passam a receber maior densidade de amostragem, sem excluir a exploração residual do restante do espaço.

### Seleção top-k, quantis e concentração

O Advisor usa uma fração padrão $r_{top}=0{,}25$ e um piso adaptativo. O valor efetivo é:

$$k=\max\left(\left\lfloor r_{top}n\right\rfloor,\ \max\left(k_{min},\min\left(20,\max\left(3,\left\lfloor0{,}05n\right\rfloor\right)\right)\right)\right).$$

Os quantis usam interpolação linear. Para $p \in [0,1]$, $i=p(n-1)$, $l=\lfloor i\rfloor$, $h=\min(l+1,n-1)$ e $\alpha=i-l$:

$$q_p=(1-\alpha)x_{(l)}+\alpha x_{(h)}.$$

A concentração do top-k usa o coeficiente de variação, com proteção contra média próxima de zero [@nistCV]:

$$CV_{top}=\frac{s_{top}}{\max(|\bar{x}_{top}|,10^{-12})}.$$

A regra operacional de estreitamento é:

$$CV_{top}<0{,}15 \land n \ge n_{agressivo} \Rightarrow [low',high']=[q_{10}^{top},q_{90}^{top}],$$

com margem mínima de 10% do intervalo original quando $q_{90}^{top}-q_{10}^{top}$ fica estreito demais.

Exemplo prático. Se há $n=14$ trials completos, então a regra adaptativa produz $k=3$. Se os três melhores trials de `batch_size` ficam concentrados perto de 512, com $CV_{top}<0{,}15$, o Advisor estreita a faixa em torno dessa vizinhança. Operacionalmente, a concentração da elite justifica reduzir a faixa analisada, desde que os guardrails de evidência sejam atendidos.

### Tendência monotônica e expansão de borda

A correlação de Spearman é calculada como correlação de ranks com tratamento de empates:

$$\rho_s=\frac{\operatorname{cov}(rank(x),rank(y))}{\sqrt{\operatorname{var}(rank(x))\operatorname{var}(rank(y))}}.$$

A forma clássica sem empates é usada apenas como referência teórica [@mathworldSpearman]:

$$\rho_s=1-\frac{6\sum_i d_i^2}{n(n^2-1)}.$$

Em parâmetros numéricos, as proximidades de borda são:

$$p_{upper}=\frac{q_{90}^{top}-low}{high-low},\qquad p_{lower}=\frac{q_{10}^{top}-low}{high-low}.$$

A expansão superior exige sinal de borda e tendência compatível:

$$p_{upper}>1-0{,}15 \land \rho_s \ge 0{,}15 \Rightarrow high'=high+0{,}5(high-low).$$

Para parâmetros sensíveis a custo, o gatilho superior é mais conservador:

$$\rho_s \ge 0{,}25 \land I_j \ge 0{,}1.$$

A expansão inferior usa sinal monotônico oposto:

$$p_{lower}<0{,}15 \land \rho_s \le -0{,}15 \Rightarrow low'=low-0{,}5(high-low).$$

Quando o parâmetro está em escala logarítmica, o cálculo ocorre em $x_{log}=\log_{10}(x)$ e depois retorna ao domínio original por $x=10^{x_{log}}$.

Exemplo prático. Se `learning_rate` está em $[0{,}0004, 0{,}0006]$, com $q_{90}^{top}=0{,}00058$ e $\rho_s=0{,}42$, então $p_{upper}=0{,}90$ e a expansão superior leva o teto para $0{,}0007$. A interpretação operacional é que o limite superior atual pode estar restringindo a busca antes de a tendência monotônica se esgotar.

### Importâncias, categorias e congelamento

A importância numérica interna é a magnitude da associação monotônica:

$$I_j^{num}=|\rho_s(x_j,y)|.$$

Para variáveis categóricas, a força interna segue uma razão tipo ANOVA:

$$\eta_j^2=\frac{SS_{between}}{SS_{total}}.$$

Quando há importância externa e interna, o Advisor mistura ambas e renormaliza:

$$I_j=\operatorname{norm}\left(\alpha I_j^{ext}+(1-\alpha)I_j^{int}\right),\qquad \alpha\in[0{,}2,0{,}8].$$

Se uma recomendação ainda está em `keep` e $I_j<0{,}05$, o parâmetro pode ser fixado na mediana do top-k:

$$I_j<0{,}05 \Rightarrow x_j'=q_{50}^{top}.$$

Para categorias, o sistema calcula proporção, entropia e número efetivo:

$$p_c=\frac{n_c^{top}}{k},\qquad H=-\sum_c p_c\log(p_c),\qquad N_{eff}=e^H.$$

A redução categórica exige dominância e passa por guardas de evidência:

$$\max_c p_c\ge0{,}60 \Rightarrow \texttt{reduce\_categories}.$$

Exemplo prático. Se um parâmetro numérico tem $I_j=0{,}03$ e mediana top-k 512, pode-se sugerir `fix` em 512. Se, ao mesmo tempo, um parâmetro categórico aparece como `adamw=8` e `lion=2` no top-k, então a categoria dominante atinge $p_c=0{,}8$ e a ação `reduce_categories` passa a ser admissível. A interpretação é que a concentração categórica reduz alternativas pouco promissoras, sem transformar a categoria dominante em garantia universal.

### Surrogate, UCB/LCB e segurança tipo BALLET

O surrogate operacional é uma `RandomForestRegressor`, não um processo gaussiano. Com $T=64$ árvores, a média e a dispersão entre árvores são [@sklearnRF; @breiman2001random]:

$$\hat{\mu}(x)=\frac{1}{T}\sum_{t=1}^{T}h_t(x),\qquad \hat{\sigma}(x)=\sqrt{\frac{1}{T-1}\sum_{t=1}^{T}(h_t(x)-\hat{\mu}(x))^2}.$$

Os limites usados no teste de segurança são:

$$UCB(x)=\hat{\mu}(x)+1{,}96\hat{\sigma}(x),\qquad LCB(x)=\hat{\mu}(x)-1{,}96\hat{\sigma}(x).$$

O estreitamento inspirado em BALLET só é aceito quando a melhor fronteira pessimista dentro da nova região supera a melhor fronteira otimista descartada [@zhang2023ballet]:

$$\max_{x\in\mathcal{X}_{new}}LCB(x)>\max_{x\notin\mathcal{X}_{new}}UCB(x).$$

Exemplo prático. Quando se avalia uma nova faixa de `lambda_pc`, o surrogate não substitui novas avaliações reais, mas permite comparar a fronteira pessimista dentro da nova região com a fronteira otimista que ficaria fora da nova região. Se a fronteira pessimista interna supera a fronteira otimista externa, aceita-se o estreitamento com margem de segurança.

### Projeção multiobjetivo

Para cada objetivo $i$, o Advisor inverte objetivos de minimização por $a_i=-f_i$ e mantém maximização por $a_i=f_i$. Depois normaliza:

$$\tilde{a}_i(x)=\frac{a_i(x)-\min(a_i)}{\max(a_i)-\min(a_i)}.$$

O escore escalar de base é:

$$S_{base}(x)=\frac{1}{m}\sum_{i=1}^{m}\tilde{a}_i(x).$$

A dominância de Pareto é:

$$x_a\succ x_b \iff \forall i,\tilde{a}_i(x_a)\ge\tilde{a}_i(x_b)\land\exists j,\tilde{a}_j(x_a)>\tilde{a}_j(x_b).$$

Quando o número de objetivos é tratável, o escore final incorpora bônus de hipervolume e penalidade de frente:

$$S_{final}=S_{base}+0{,}15HV_{bonus}-0{,}05\max(0,rank-1).$$

Exemplo prático. Se são comparadas qualidade e latência com três pontos $A=(0{,}82,5{,}0)$, $B=(0{,}80,7{,}0)$ e $C=(0{,}84,8{,}5)$, então $A$ domina $B$, mas não domina $C$. Mantêm-se $A$ e $C$ na frente de Pareto porque um é mais rápido e o outro é mais preciso. A interpretação é que não há vencedor escalar único quando objetivos relevantes entram em tensão.

### Confiança, Wilson e self-audit

A incerteza heurística combina maturidade do estudo e robustez da elite:

$$u=1-\min\left(1,\frac{n}{30}\right)\min\left(1,\frac{k}{20}\right).$$

O suporte bootstrap mede a fração de reamostragens que preservam a ação:

$$support=\frac{\#\{b:a_b=a\}}{\#\{b:a_b\ \text{valida}\}}.$$

Quando existe suporte, ele é calibrado com prior neutro de tamanho 20:

$$support_c=\frac{support\cdot n_{evid}+0{,}5\cdot20}{n_{evid}+20}.$$

O score de confiança implementado é:

$$C=\operatorname{clamp}_{[0,1]}(0{,}45B+0{,}55support_c-0{,}15u).$$

Sem bootstrap disponível, a implementação usa:

$$C=\operatorname{clamp}_{[0,1]}(B(1-0{,}4u)).$$

O limite inferior de Wilson para proporções é [@brown2001interval]:

$$LB=\frac{\hat{p}+\frac{z^2}{2n}-z\sqrt{\frac{\hat{p}(1-\hat{p})+z^2/(4n)}{n}}}{1+\frac{z^2}{n}}.$$

No self-audit, uma ação direcional acerta quando o sinal posterior de Spearman confirma a direção:

$$hit=\mathbb{1}[(a=\texttt{expand\_upper}\land\rho_{suffix}>0)\lor(a=\texttt{expand\_lower}\land\rho_{suffix}<0)].$$

A taxa de acerto por grupo é:

$$hit\_rate_g=\frac{hits_g}{total_g}.$$

Um padrão vira grupo de bloqueio quando:

$$total_g\ge total_{min}^{eff}\land hit\_rate_g<0{,}5\land LB_g<0{,}35.$$

A análise de sobrevivência entra como moldura descritiva, não como motor operacional corrente: $R(t)=P(T>t)$ e $h(t)=f(t)/R(t)$ [@nistReliability; @nistHazard].

Exemplo prático. Se são observadas 7 validações bem-sucedidas em 10 recomendações, então a taxa observada é 0,7, mas o limite inferior de Wilson cai para cerca de 0,397. Se um grupo histórico como `dropout|expand_upper` registra 3 acertos em 8 auditorias, o `hit_rate` fica em 0,375 e o bloqueio passa a fazer sentido. A interpretação estatística é que a proporção observada deve ser ponderada pelo tamanho amostral antes de sustentar confiança alta.

### Cold start

Com menos evidência, as heurísticas de dataset usam escala do grafo. Para dimensão de embedding, a implementação arredonda para potência de dois:

$$d=\min\left(1024,\max\left(64,2^{round(\log_2(\max(64,2\sqrt{N_{entities}})))}\right)\right).$$

Exemplo prático. Para um grafo com 5.470 entidades, a conta produz $d=128$. Essa heurística é usada como ponto de partida, não como verdade final. Essa heurística define apenas um ponto inicial plausível e deve ser substituída por evidência empírica à medida que novos trials são observados.

## Prova basal e evidência empírica

Esta seção separa a prova basal, que decorre das definições e guardas do algoritmo, da evidência empírica, que depende do payload do estudo de caso. A prova basal não afirma otimalidade global; ela afirma propriedades condicionais e verificáveis do Advisor quando suas hipóteses são satisfeitas.

**Definição 1 — Estado do Advisor.** Um estado é a tupla $s=(\mathcal{X},D_t,\delta,M,A_t)$, em que $\mathcal{X}$ é o espaço de busca atual, $D_t=\{(x_i,y_i)\}_{i=1}^{t}$ é o histórico de trials completos, $\delta$ é a direção de otimização, $M$ é o conjunto de metadados de confiabilidade e $A_t$ é o histórico de ações auditáveis.

**Definição 2 — Ações admissíveis.** Uma ação $a_j$ sobre o parâmetro $j$ pertence a `{keep, narrow, expand_lower, expand_upper, fix, reduce_categories}`. Ela é admissível quando preserva o domínio do parâmetro, respeita os limiares configurados, passa pela validação rígida e não pertence a um grupo bloqueado pelo self-audit.

**Definição 3 — Utilidade local.** A utilidade local de uma recomendação é uma função ordinal $U(a_j\mid s)$ que combina sinal empírico, importância, suporte bootstrap, incerteza e penalidades de validação. O Advisor usa essa utilidade apenas para ordenar e calibrar recomendações; ela não é interpretada como recompensa causal nem como posterior bayesiano.

**Hipóteses.** Assume-se que: H1) os trials usados no cálculo estão completos e comparáveis sob a mesma direção de otimização; H2) o espaço de busca inicial codifica limites válidos; H3) parâmetros numéricos podem ser ordenados e parâmetros categóricos têm categorias explícitas; H4) as estatísticas de elite usam top-k não vazio; H5) o surrogate RF é usado como heurística empírica, não como garantia probabilística exata; H6) qualquer conclusão de confiabilidade é condicionada ao tamanho amostral observado.

**Lema 1 — Top-k não vazio e limitado.** Se $n\ge1$ e $k_{min}\ge1$, então a regra $k=\max(\lfloor r_{top}n\rfloor,\max(k_{min},\min(20,\max(3,\lfloor0{,}05n\rfloor))))$ produz $k\ge1$. Se a implementação recorta a lista ordenada pelo número real de trials disponíveis, então a elite efetiva também satisfaz $1\le k_{eff}\le n$.

**Demonstração.** Como $k_{min}\ge1$, o segundo termo do máximo externo é pelo menos 1. Logo, $k\ge1$. A seleção operacional toma prefixo da lista de $n$ trials completos; portanto, ainda que a regra nominal produza valor maior que $n$ em estudos muito pequenos, o prefixo materializado não pode conter mais do que $n$ elementos. Assim, $1\le k_{eff}\le n$. $\square$

**Lema 2 — Estreitamento preserva subintervalo válido.** Para um parâmetro numérico com $low<high$, se $q_{10}^{top}$ e $q_{90}^{top}$ são quantis de valores top-k pertencentes a $[low,high]$ e $q_{10}^{top}\le q_{90}^{top}$, então o intervalo proposto $[q_{10}^{top},q_{90}^{top}]$, após margem mínima implementada e clipping ao domínio original quando aplicável, permanece um subintervalo válido do domínio analisado.

**Demonstração.** Quantis de uma amostra contida em $[low,high]$ pertencem ao mesmo intervalo por convexidade da interpolação linear. Como $q_{10}^{top}\le q_{90}^{top}$, o intervalo é ordenado. A margem mínima só alarga uma janela degenerada ou estreita demais, e o clipping impede saída do domínio válido. Logo, a recomendação resultante continua bem formada. $\square$

**Lema 3 — Wilson reduz excesso de confiança em amostras pequenas.** Para $0<\hat p<1$, $n>0$ e $z>0$, o limite inferior de Wilson é menor que a taxa observada $\hat p$.

**Demonstração.** O termo subtraído no numerador contém $z\sqrt{(\hat p(1-\hat p)+z^2/(4n))/n}$, estritamente positivo sob as hipóteses. O denominador $1+z^2/n$ é maior que 1. Assim, o limite inferior desloca a estimativa para baixo em relação à proporção observada, com efeito mais forte quando $n$ é pequeno. $\square$

**Teorema 1 — Admissibilidade estrutural condicional.** Sob H1-H6, se uma recomendação emitida pelo Advisor passa pela validação rígida, respeita os limites de domínio e não é bloqueada pelo self-audit, então ela é estruturalmente admissível para revisão do espaço de busca. Essa admissibilidade não implica melhoria futura nem otimalidade global.

**Demonstração.** Pela Definição 2, admissibilidade exige domínio preservado, limiares satisfeitos, validação rígida e ausência de bloqueio histórico. Os lemas 1 e 2 garantem que as estatísticas de elite e o estreitamento numérico são bem formados quando suas hipóteses são atendidas. O Lema 3 mostra que a confiança reportada é conservadora em amostras pequenas. Portanto, uma ação que passa por essas guardas satisfaz o contrato estrutural do Advisor. Como nenhuma hipótese assume convexidade, estacionariedade, suficiência causal do histórico ou exatidão probabilística do surrogate RF, não se segue otimalidade global nem garantia de melhoria no próximo trial. $\square$

**Corolário — Robustez interpretativa.** Se o Wilson-LB ou o self-audit são fracos, a recomendação pode continuar estruturalmente válida, mas deve ser interpretada como sugestão exploratória, não como decisão forte.

**Evidência empírica e2e.** No estudo PFF, a prova basal é instanciada por 25 trials completos, 21 recomendações, cobertura 1.00 do espaço, validação rígida 1.0000, Wilson-LB de validação 0.8454 e self-audit com 19 sinais direcionais. Esses números não provam superioridade contra todos os métodos de HPO; eles demonstram que o pipeline e2e produziu payload completo, figuras, recomendações e métricas de confiabilidade auditáveis.

**Limites e ameaças à validade.** As principais ameaças são: amostra curta para self-audit temporal; possível viés do espaço inicial; dependência de métricas normalizadas corretamente; surrogate RF usado como aproximação heurística; ausência de comparação pareada longa contra TPE puro, GP-BO e BALLET completo; e dependência de infraestrutura Docker para reprodutibilidade operacional.

**Potencial de proteção intelectual.** O conjunto formado por self-audit temporal, Wilson-LB, validação rígida, ações de espaço e rastreabilidade de payload pode ser descrito como combinação técnica potencialmente protegível. Este artigo não afirma patenteabilidade, novidade jurídica nem liberdade de operação; ele apenas identifica elementos que poderiam ser avaliados futuramente por especialista em propriedade intelectual.

## Arquitetura lógica e tabelas-síntese

Dois princípios orientam a leitura metodológica. Primeiro, o surrogate é uma aproximação empírica usada para comparar alternativas antes de novas avaliações caras. Segundo, a confiabilidade estatística exige considerar simultaneamente taxa observada e tamanho amostral; por isso, limites conservadores como Wilson são preferíveis à taxa bruta em amostras pequenas [@brown2001interval; @barlow1996reliability].

Reconstitui-se a arquitetura analítica do Advisor a partir do documento técnico. O pipeline atual começa com a normalização do estudo, seleciona trials válidos, projeta eventuais objetivos múltiplos em um score escalar híbrido, escolhe um top-k adaptativo, computa estatísticas por parâmetro, treina opcionalmente um surrogate de floresta aleatória, estima importâncias e interações, emite ações locais, calibra a confiança com bootstrap e validação dura, executa self-audit histórico e finalmente resume a confiabilidade agregada do payload. Em termos operacionais, o sistema não treina o modelo principal; ele analisa histórico, bordas, tendências e evidência estatística para sugerir manter, expandir, estreitar, fixar ou reduzir partes do espaço de busca [@breiman2001random; @hutter2014fanova; @lundberg2017shap].

![Fluxo lógico do Search Space Advisor.](figures/figura0_fluxo_advisor.png){width=100%}

*Fonte: elaboração própria com base no documento técnico do projeto e na literatura de HPO e interpretabilidade.*

As Tabelas 1 a 3 fazem a ponte entre a formulação geral do método e o posicionamento do Advisor no ecossistema de HPO. As Tabelas 4 e 5 registram o estudo de caso do PFF. As Tabelas 6 a 8 recolocam o problema em uma moldura mais ampla de detecção de anomalias, avaliação e consistência conceitual.

**Tabela 1 — Ranking metodológico contextualizado do problema**

| Posição | Método ou família | Leitura corrigida e contextualizada |
|---|---|---|
| 1 | Search Space Advisor com adaptação de espaço | Deve ser lido como ranking contextual para espaços caros, mistos e com realimentação do histórico, não como teorema universal |
| 2 | Optuna, Vizier e samplers modernos de produção | Fortes em prática industrial, sobretudo por flexibilidade, paralelismo, poda e integração de infraestrutura |
| 3 | Otimização bayesiana clássica baseada em GP | Forte em cenários suaves e de baixa a média dimensão efetiva, mas sensível à estrutura do espaço e ao custo do surrogate |
| 4 | Busca aleatória e busca em grade | Busca aleatória continua sendo baseline honesta; grade tende a piorar em espaços maiores |

**Tabela 2 — Dimensões de otimização consideradas**

| Sistema ou estratégia | Dimensão otimizada | Fundamento matemático dominante | Observação de confiabilidade |
|---|---|---|---|
| BO clássica | Hiperparâmetros contínuos e discretos caros | Surrogate modeling, função de aquisição, exploração versus explotação | A confiabilidade depende da qualidade do modelo substituto |
| Plataformas de produção como Optuna e Vizier | Tuning operacional em escala | Samplers, pruners, paralelismo, instrumentação | A confiabilidade também é propriedade do ecossistema |
| BALLET e métodos de região de interesse | Adaptação do espaço de busca | Estimação de level-set e filtragem probabilística da região promissora | A confiabilidade entra como teste de segurança para não podar o ótimo |
| Search Space Advisor | Refinamento local do espaço | Top-k, Spearman, bootstrap, Wilson, self-audit, surrogate RF | A confiabilidade é multicamada e auditável |

**Tabela 3 — Tipologia de aplicações para o Advisor**

| Classe de estudo | Objeto otimizado | Variáveis destacadas | Papel da confiabilidade |
|---|---|---|---|
| Visão computacional e NAS | Profundidade, largura, regularização, learning rate, batch | Mistura de parâmetros contínuos e categóricos | Evitar expansão cega em bordas e reduzir custo experimental |
| Modelos de linguagem | Learning rate, warmup, regularização, lote, duração de treino | Forte sensibilidade a escalas logarítmicas e poda | Evitar conclusões instáveis com poucos trials |
| Modelos relacionais, grafos e multimodais | Negativos, pesos simbólicos, parâmetros de contraste e amostragem | Espaço heterogêneo, com interdependências | Reforçar decisões por evidência local e histórico temporal |
| PFF como estudo de caso | Expansão, estreitamento, fixação e redução categórica | A própria política de ajuste do espaço | A confiabilidade vira objeto central do estudo |

**Tabela 4 — Espaço de busca do estudo de caso PFF**

| Bloco | Hiperparâmetros principais | Faixas ou categorias |
|---|---|---|
| Treinamento | learning_rate; weight_decay; batch_size; negative_sample_size; grad_clip; warmup_ratio; epochs | [4e-4, 6e-4]; [1e-6, 1e-4]; {256, 512, 1024}; [384, 512]; [1, 10]; [0,10, 0,20]; [120, 160] |
| Arquitetura | feature_dim; hidden_dim; kl_weight; sparsity_weight; ibp_alpha; max_communities | {256, 512}; {256, 512}; [1e-4, 1e-2]; [1e-6, 1e-2]; [1, 10]; {128} |
| Contraste e amostragem | temperature; margin; adv_temperature; hard_neg_ratio; num_negatives; num_global_negatives; neg_sampler | [0,025, 0,04]; [0, 0,05]; [0,9, 1,8]; [0, 0,7]; {384}; {96}; {degree_based} |
| Lógica, baixo posto e FAISS | lambda_logic; num_basis; nlist; nprobe; eval_topk | [0,03, 0,05]; {2, 4, 8, 16}; {256, 512, 1024, 2048}; {4, 8, 16, 32}; {512, 1024, 2048} |

**Tabela 5 — Regras matemáticas e limiares operacionais**

| Mecanismo | Regra atual informada | Interpretação |
|---|---|---|
| Frio inicial | Análise empírica completa apenas se n_trials >= 5 | Abaixo disso, a evidência é insuficiente para decisões empíricas completas |
| Seleção top-k | Até 25% dos melhores trials, com piso adaptativo e teto operacional | Evita top-k minúsculo em estudos mais maduros |
| Expansão superior | rho >= 0,15 | Requer monotonicidade compatível |
| Expansão inferior | rho <= -0,15 | Simétrica à superior |
| Expansão sensível a custo | rho >= 0,25 e importance >= 0,1 | Mais conservadora |
| Estreitamento | CV < 0,15; nova faixa em [q10, q90]; janela mínima de 10% | Só estreita quando a elite já está concentrada |
| Fixação | importance < 0,05 | Pode fixar variáveis com influência empírica baixa |
| Bootstrap | 50 reamostragens | Mede reprodutibilidade da ação |
| Wilson | z = 1,96 | Corrige excesso de confiança em amostra pequena |
| Self-audit | bloqueio se hit_rate < 0,5 e Wilson LB < 0,35 | Bloqueia padrões historicamente frágeis |

Considera-se o uso do limite inferior de Wilson uma das partes mais fortes do sistema do ponto de vista estatístico. A taxa observada pura $\hat p$ pode fazer um mecanismo parecer ótimo em amostras mínimas, mas o limite inferior de Wilson funciona como um controle contra excesso de confiança. Em termos estatísticos, poucos acertos em poucas observações ainda não justificam confiança alta [@brown2001interval].

Outro ponto matematicamente sólido é a calibração por bootstrap das ações finais. O sistema reamostra os trials cinquenta vezes e mede em quantas delas a mesma ação reaparece. Isso produz uma medida de reprodutibilidade local da decisão. Se a recomendação desaparece com pequenas perturbações da amostra, ela não é robusta; se reaparece repetidamente, ganha direito a confiança maior. Em termos operacionais, se pequenas perturbações da amostra mudam a ação recomendada, a decisão não deve ser tratada como estável.

A projeção multiobjetivo também merece destaque. O método não reduz o problema simplesmente ao primeiro objetivo, mas reconstrói o score híbrido com normalização por dimensão, média escalar, ordenação de Pareto e bônus de hiper-volume quando o número de objetivos permite cálculo estável. Isso aproxima o método de práticas modernas de HPO multiobjetivo, nas quais não basta maximizar um único número se custo, latência, memória e qualidade competem entre si [@deb2002nsga2; @morales2023many].

**Tabela 6 — Métodos de detecção de anomalias relevantes para logs e decisões do Advisor**

| Método | Sinal operacional | Vantagem | Limitação | Uso recomendado no Advisor |
|---|---|---|---|---|
| Z-score / IQR | desvio univariado extremo | Muito barato e explicável | Univariado e frágil em caudas pesadas | Monitorar confidence_score, latência, rho, CV e amplitude proposta |
| LOF | isolamento local por vizinhança | Excelente para anomalia local | Sensível à escolha de vizinhança | Detectar parâmetros cujo comportamento difere localmente de pares similares |
| Isolation Forest | particionamento aleatório de pontos raros | Bom em alta dimensão e sem rótulo | Interpretação menos intuitiva | Monitorar payloads completos e combinações raras de bloqueios |
| Deep SVDD / autoencoders | modelagem de normalidade por reconstrução | Captura estrutura complexa | Requer mais dados e treino | Monitorar séries longas de telemetria do HPO |

**Tabela 7 — Datasets e métricas adequados para avaliar anomalias ligadas ao Advisor**

| Dataset ou referência | Tipo de dado | O que avalia bem | Métricas prioritárias |
|---|---|---|---|
| NAB | Séries temporais reais e artificiais em tempo real | Detecção rápida e custo de falso alarme | NAB Score, precisão, revocação, atraso de detecção |
| SMAP / MSL | Telemetria multivariada de missão espacial | Dependências temporais e contexto operacional | Precisão, revocação, F1 por evento, AUPRC |
| UCR Time Series Anomaly Archive | Séries univariadas curadas | Comparação padronizada e menos enviesada | Precisão, revocação, F1, AUPRC |
| Logs internos do HPO | Eventos estruturados do próprio Advisor | Anomalias de decisão e falhas de política | taxa de bloqueio, Wilson LB, AUPRC, tempo até falha da recomendação |

Em problemas fortemente desbalanceados, curvas precisão-revocação e AUPRC tendem a ser mais informativas do que ROC/AUROC. Se o sistema quase nunca recomenda ação agressiva, um AUROC alto pode esconder baixa utilidade real, enquanto a precisão entre as ações disparadas continua sendo o que mais importa [@saito2015precision].

**Tabela 8 — Inconsistências corrigidas entre narrativa e implementação**

| Ponto do material original | Problema | Correção adotada neste artigo |
|---|---|---|
| Descrição do Advisor como sistema baseado em GP e kernel Matern | Incompatível com o anexo técnico, que informa floresta aleatória e ausência de kernel estatístico no motor atual | O artigo distingue inspiração teórica em BO/GP da implementação efetiva em RF |
| Uso de estado da arte como ranking absoluto | Generalização indevida | Ranking reclassificado como contextual |
| Leitura do kernel Rust como kernel de ML | Confusão terminológica | Esclarecido que o módulo em Rust acelera Spearman, não GP nem KDE |
| Interpretação simplificada do top-k | Redução excessiva ao quartil superior | Formalizado como regra adaptativa com mínimo dinâmico |
| Tratamento da confiabilidade como rótulo simples | Modelagem superficial | Reescrito como sistema multicamada com incerteza, bootstrap, Wilson, validação e self-audit |
| Falta de ligação com censura e anomalias | Lacuna conceitual | Integradas perspectivas de sobrevivência e detecção de anomalias |

A síntese mostra que a contribuição central é alinhar linguagem, matemática e implementação. Quando esses três planos são consistentes, o artigo torna-se mais reprodutível e menos vulnerável a divergências entre narrativa metodológica e comportamento implementado [@ross1996stochastic; @rasmussen2006gp; @aggarwal2017outlier].

# Resultados

As fórmulas anteriores descrevem um Advisor geral; nesta seção, mostra-se a instância concreta desse Advisor no PFF. Em outras palavras, a formulação é generalista, e o experimento abaixo funciona como prova de aplicabilidade em um projeto real.

No estudo de caso PFF, observa-se que o Advisor versão 2.3.0 gerou 21 recomendações para 20 hiperparâmetros do espaço de busca. A cobertura do espaço foi 1.00, sem parâmetros ausentes. As ações finais foram: `expand_lower`=1, `expand_upper`=1, `fix`=8, `keep`=4, `narrow`=6, `reduce_categories`=1.

A confiabilidade agregada foi: validação rígida 0.9524, Wilson-LB de validação 0.7733, confiança média 0.7410, taxa de alta confiança 0.9048. A ausência de recomendações de alta confiança é coerente com amostras curtas, pois a incerteza heurística permanece alta. O ponto geral aqui é que não se trata pouca amostra como evidência forte só porque a taxa observada parece boa.

O self-audit avaliou 6 prefixos, 14 sinais direcionais e obteve `hit_rate`=0.7143 com Wilson-LB=0.4535. O grupo `lambda_pc|expand_lower` apareceu como vilão histórico neste recorte, com taxa de acerto 0,0 e Wilson-LB 0,0, mas nenhuma ação corrente foi bloqueada porque a recomendação atual para `lambda_pc` foi `keep`. Em termos metodológicos, isso mostra que o mecanismo de bloqueio depende do histórico do padrão e não do nome do projeto; qualquer pipeline com logs de prefixo-sufixo pode usar a mesma lógica.

A partir da versão atual, parâmetros fixos também entram como objeto de auditoria. O payload usa `action=keep` e `recommendation.diagnostic` para separar três casos: `needs_exploration`, quando a importância resolvida é alta e não há variação suficiente para estimar sensibilidade; `stable_fixed_value`, quando há evidência suficiente e baixa importância; e `watch_fixed_value`, quando a amostra ainda não permite decisão forte. Assim, um valor como `embedding_dim=512` pode ser mantido quando está estável, mas não é declarado ótimo sem comparação local contra vizinhos.

## Tabela 9 — recomendações principais do estudo de caso

| Parâmetro | Importância | Ação | Confiança |
|---|---:|---|---|
| `lambda_logic` | 0.1853 | `keep` | high |
| `contrastive_temperature` | 0.1745 | `narrow` | medium |
| `batch_size` | 0.1618 | `narrow` | high |
| `min_delta` | 0.1226 | `keep` | high |
| `learning_rate` | 0.0565 | `expand_upper` | high |
| `ibp_alpha` | 0.0455 | `fix` | high |
| `dslfm_epochs` | 0.0421 | `fix` | high |
| `negative_sample_size` | 0.0421 | `narrow` | high |
| `rebuild_every` | 0.0226 | `keep` | high |
| `lambda_pc` | 0.0220 | `fix` | high |

## Tabela 10 — ablations e sensibilidade do Advisor

A tabela abaixo reexecuta o Advisor no mesmo payload e no mesmo recorte de 25 trials, desligando componentes específicos. Ela mede custo, estabilidade das ações, confiança, validação e efeito do self-audit. Como o objetivo é isolar componentes do Advisor, os resultados não devem ser lidos como comparação causal contra outros otimizadores.

| Variante | Custo (ms) | Recomendações | Ações | Confiança média | Wilson validação | Wilson direcional |
|---|---:|---:|---|---:|---:|---:|
| `full_no_adaptive` | 97.75 | 21 | expand_lower=1, expand_upper=1, fix=8, keep=4, narrow=6, reduce_categories=1 | 0.7410 | 0.7733 | 0.4535 |
| `no_surrogate` | 153.68 | 21 | expand_lower=1, expand_upper=1, fix=8, keep=4, narrow=6, reduce_categories=1 | 0.7410 | 0.7733 | 0.4535 |
| `no_interactions` | 95.74 | 21 | expand_lower=1, expand_upper=1, fix=8, keep=4, narrow=6, reduce_categories=1 | 0.7410 | 0.7733 | 0.4535 |
| `no_internal_importances` | 96.45 | 21 | expand_lower=1, expand_upper=1, fix=7, keep=5, narrow=6, reduce_categories=1 | 0.7410 | 0.7733 | 0.4535 |
| `no_bootstrap` | 15.40 | 21 | expand_lower=1, expand_upper=1, fix=8, keep=4, narrow=6, reduce_categories=1 | 0.5440 | 0.7733 | 0.4535 |
| `no_self_audit` | 86.38 | 21 | expand_lower=1, expand_upper=1, fix=8, keep=4, narrow=6, reduce_categories=1 | 0.7410 | 0.7733 | n/a |

No recorte observado, remover surrogate ou interações não alterou a distribuição de ações, enquanto remover importâncias internas trocou uma ação `keep` por `fix`. Remover bootstrap reduziu a confiança média, mas preservou validação e ações. Remover self-audit elimina o Wilson direcional, como esperado, e por isso perde a principal evidência temporal de robustez.


## Tabela 11 — benchmark pareado TPE, GP-BO e Advisor

Para reduzir o risco do recorte curto, executo um benchmark pareado sintético com 50 trials por política, sementes [11, 17, 23, 29, 31] e atualização do Advisor a cada 10 trials após o aquecimento. O cenário compara TPE puro, GP-BO via `GPSampler`, Advisor completo e ablations do Advisor sob a mesma função objetivo determinística.

| Política | Melhor médio | Δ vs TPE | Δ vs GP-BO | Vitórias/p |
|---|---:|---:|---:|---:|
| TPE puro | 0.825381 | +0.000000 | -0.122074 | 0/n/a |
| GP-BO | 0.947455 | +0.122074 | +0.000000 | 5/0.03125 |
| Advisor full | 0.853036 | +0.027655 | -0.094419 | 4/0.09375 |
| Sem surrogate | 0.853036 | +0.027655 | -0.094419 | 4/0.09375 |
| Sem bootstrap | 0.853036 | +0.027655 | -0.094419 | 4/0.09375 |
| Sem self-audit | 0.853036 | +0.027655 | -0.094419 | 4/0.09375 |

A regra conservadora de reivindicação universal não foi satisfeita. Portanto, o resultado deve ser lido como evidência localizada: o Advisor pode melhorar TPE puro neste cenário, mas não sustenta superioridade estatística universal contra GP-BO ou contra a política completa com surrogate/BALLET em qualquer problema.

Uma triagem posterior, executada com 30 trials, sementes [11, 17, 23] e quatro cenários sintéticos (`smooth_kgc`, `narrow_ridge`, `conditional_regularized`, `edge_capacity`), adicionou baselines Random, TPE+Hyperband, GP-BO, Advisor completo e ablations. Nessa triagem, uma política conservadora `advisor_edge_gated_gp` manteve o comportamento do GP-BO quando a evidência de borda era fraca e só reiniciou o espaço quando havia parâmetros influentes próximos à borda superior. O resultado agregado foi `advisor_edge_gated_gp` média 0.786175 contra GP-BO média 0.783902, delta +0.002273, IC95 bootstrap [0.000000, 0.006819], 1 vitória e 0 derrotas pareadas, Wilcoxon unilateral p=0.5. A leitura correta é que o gate reduz dano em cenários onde o Advisor não tem sinal suficiente e cria uma hipótese promissora para espaços com ótimo além da borda observada; ainda não há significância estatística para reivindicação SOTA.

Também foi corrigido um detalhe de protocolo: recomendações vazias (`empty_patch`) ou bloqueadas pelo gate não devem consumir um trial do orçamento. O benchmark agora registra o bloqueio e avalia o trial corrente no espaço anterior, preservando orçamento pareado contra os baselines.


## Figuras geradas a partir do payload do Advisor

![Evidências empíricas do Search Space Advisor: ações, confiança, validação e resumo conservador.](figures/figura1_evidencias_advisor.png){width=95%}

*Fonte: elaboração própria a partir de searchSpaceAdvice e da auditoria offline do Advisor.*

![Importâncias normalizadas e ação recomendada por hiperparâmetro.](figures/figura2_importancias_acoes.png){width=95%}

*Fonte: elaboração própria a partir do payload searchSpaceAdvice.*

![Comparação q10-q90 normalizada entre todos os trials e a região top-k dos principais parâmetros.](figures/figura3_topk_distribuicoes.png){width=95%}

*Fonte: elaboração própria a partir de searchSpaceAdvice.recommendations.*

![Indicadores de confiabilidade, incluindo validação, confiança média e self-audit.](figures/figura4_confiabilidade.png){width=95%}

*Fonte: elaboração própria a partir da auditoria offline do Advisor.*

![Projeção qualidade-tempo e frente de Pareto descritiva.](figures/figura5_pareto_qualidade_tempo.png){width=95%}

*Fonte: elaboração própria para auditar a projeção multiobjetivo consumida pelo Advisor.*

![Sobrevivência empírica das recomendações direcionais no self-audit.](figures/figura6_sobrevivencia_direcional.png){width=95%}

*Fonte: elaboração própria a partir de metadata.self_audit.*

# Discussão

Os dados do estudo de caso removem a principal lacuna factual: agora existem trials, payload de recomendações, reliability summary, self-audit e figuras calculadas. Ainda assim, mantém-se a interpretação proporcional ao orçamento experimental. A rodada do PFF é suficiente para embasar a mecânica do Advisor e demonstrar rastreabilidade, mas não para reivindicar superioridade estatística universal contra GP-BO, TPE puro ou BALLET completo.

A evidência mais forte é estrutural: não houve mismatch de direção, não foram detectadas inconsistências, a validação rígida foi perfeita no payload e a cobertura do espaço foi integral. A evidência mais fraca é temporal: o self-audit teve poucos prefixos e Wilson-LB baixo para decisões direcionais. Isso não invalida o Advisor; pelo contrário, mostra que o mecanismo de cautela está funcionando ao não elevar confiança com pouca amostra.

O ponto mais importante para uso fora do PFF é a separação entre parte geral e parte específica. São gerais: top-k adaptativo, quantis, Spearman, coeficiente de variação, mistura de importâncias, Wilson, bootstrap, self-audit e a possibilidade de projeção multiobjetivo. São específicos do PFF: nomes de hiperparâmetros, faixas de busca, score usado no experimento e heurísticas de cold start conectadas ao grafo. Essa distinção torna o artigo reutilizável como referência metodológica para outros projetos e mantém o PFF como estudo de caso exemplar, não como fronteira do método.

Também fica resolvida a inconsistência conceitual sobre surrogate. O código usa `RandomForestRegressor` com 64 árvores, profundidade máxima 8 e `random_state=42`; a incerteza é a dispersão entre árvores. Isso justifica UCB/LCB como heurística empírica, não como posterior bayesiana de GP. A referência a BALLET permanece útil como inspiração de região de interesse, mas o critério implementado é uma guarda local simples baseada em LCB dentro versus UCB fora.

# Conclusão

Este artigo descreve um Search Space Advisor em dois níveis: como formulação geral para adaptação confiável de espaços de busca em HPO e como estudo de caso executado no PFF. O método é transferível para outros projetos porque depende de estruturas genéricas de experimento, como histórico de trials, direção de otimização, parâmetros auditáveis e métricas comparáveis, e não de propriedades exclusivas de Knowledge Graph Completion.

No estudo de caso PFF, conclui-se que o Advisor auditado é melhor descrito como um sistema de decisão estatística local com confiabilidade multicamada. Ele combina seleção top-k, tendência monotônica, concentração, importâncias, surrogate RF, validação dura, bootstrap quando disponível, Wilson, self-audit e diagnóstico de parâmetros fixos. O experimento em Docker confirmou que o payload é consistente, reproduzível e suficientemente completo para sustentar o artigo com dados, figuras e anexos verificáveis.

# Referências

::: {#refs}
:::

\newpage

# Apêndice A - Artefatos e comandos

- Dashboard HPO: `dashboard_data.json`, em `outputs/.cache/hpo/`.
- Resumo HPO: `hpo_summary.json`, em `outputs/optimization/kg_dslfm/`.
- Auditoria offline: `deep_research_audit_20260506.json`, no diretório de benches do Advisor.
- Documento técnico verificado: `SEARCH_SPACE_ADVISOR.md`, no pacote local de infraestrutura HPO.
- Frente de Pareto descritiva qualidade-tempo: 4 pontos, trials [2, 7, 10, 54].
- Observação operacional: o cache L2 PostgreSQL apareceu como `degraded/oserror` durante a auditoria offline em container isolado, mas o cálculo principal do Advisor foi concluído e persistido em JSON; esse estado afeta cache, não a validade das recomendações calculadas.

# Apêndice B - Fontes de implementação verificadas

Os módulos abaixo foram verificados no pacote local do Search Space Advisor, em infraestrutura HPO.

| Mecanismo | Módulo local |
|---|---|
| Top-k, quantis, Spearman, incerteza | `statistics.py` |
| Expansão, narrow, CV, BALLET-style safety | `analysis_numeric.py` |
| Categorias, entropia e redução | `analysis_categorical.py` |
| RandomForestRegressor, UCB/LCB e transformação log | `surrogate.py` |
| Importância interna/blended | `importance.py` |
| Wilson e reliability summary | `reliability.py` |
| Bootstrap support | `bootstrap.py` |
| Self-audit e bloqueios | `self_audit.py`; `self_audit_runner.py` |
| Projeção Pareto/hipervolume | `multiobjective.py` |
| Heurísticas cold start | `recommendations.py` |
