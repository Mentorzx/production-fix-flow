Táticas de Hyperparameter Optimization para DSLFM-KGC em Grafos Esparsos sob a Ótica de Circuitos Probabilísticos Variante 2 (PC-V2): Uma Análise Rigorosa da Convergência e Estratégia

Nota de alinhamento: referências a modelos legados (ex.: TransE, RotatE, AnyBURL, LightGBM, XGBoost, CatBoost) são históricas e não refletem o stack atual, que é DSLFM-KGC + PC2.

1. Introdução e Contextualização do Problema

A fronteira da inteligência artificial simbólica e estatística encontra-se, atualmente, no desenvolvimento de modelos capazes de raciocínio complexo sobre estruturas de dados incompletas. A completação de grafos de conhecimento (KGC - Knowledge Graph Completion) representa o arquétipo deste desafio, exigindo que algoritmos predigam relações ausentes $(h, r,?)$ ou $(?, r, t)$ em grafos onde a esparsidade não é uma exceção, mas a norma dominante. Enquanto abordagens baseadas em embeddings vetoriais densos — como TransE, RotatE ou ComplEx — dominaram a última década de pesquisa, elas sofrem de uma opacidade inerente e de uma incapacidade crônica de modelar a incerteza estrutural de forma interpretável.

Neste cenário, o surgimento dos Deep Sparse Latent Feature Models (DSLFM-KGC), conforme delineado nos trabalhos recentes de Li et al. 1, marca uma mudança de paradigma. Ao integrar Variational Autoencoders (VAEs) com processos estocásticos não-paramétricos, especificamente o Indian Buffet Process (IBP), o DSLFM-KGC promete unir a capacidade de representação profunda das redes neurais com a interpretabilidade e a flexibilidade de adaptação de complexidade dos modelos Bayesianos não-paramétricos. No entanto, a introdução de tais componentes estocásticos, acoplada à necessidade de regularização lógica estrita para garantir a consistência das predições, introduz uma superfície de otimização de hiperparâmetros (HPO) de complexidade proibitiva.

Este relatório foca especificamente na integração de uma variante avançada de regularização estrutural: os Probabilistic Circuits Variant 2 (PC-V2). Definimos PC-V2 no contexto deste trabalho, baseando-nos na literatura emergente sobre circuitos tratáveis para contextos de regras 3, como uma arquitetura que modela explicitamente a distribuição de probabilidade sobre "contextos de regras" lógicas mineradas, permitindo inferência exata e eficiente sobre a validade semântica das triplas preditas. A "Variante 2" distingue-se pela sua capacidade de lidar com conjuntos de regras reduzidos e contextos dinâmicos, diferindo de abordagens anteriores que tentavam compilar todas as regras possíveis em um circuito estático massivo.

O problema central que este documento se propõe a resolver é a instabilidade do treinamento do DSLFM-KGC quando regularizado por PC-V2 em regimes de dados esparsos. Observa-se que as técnicas padrão de HPO falham frequentemente em convergir para soluções ótimas devido a conflitos dinâmicos entre a expansão de features do IBP (que exige um período de "aquecimento" ou burn-in) e as restrições rígidas impostas pelo PC-V2. Através de uma análise exaustiva e comparativa de métodos de otimização — Tree-structured Parzen Estimator (TPE), Non-dominated Sorting Genetic Algorithm II (NSGA-II), Population Based Training (PBT) e Hyperband — estabeleceremos um novo protocolo de otimização adaptativa. Argumentamos que a eficácia do HPO neste domínio depende não apenas da busca eficiente, mas da sincronização temporal entre o algoritmo de busca e as fases de transição de fase do modelo generativo subjacente.

2. Fundamentação Teórica e Arquitetural

Para dissecar as estratégias de otimização, é imperativo primeiro desconstruir a arquitetura do DSLFM-KGC e o papel regulatório do PC-V2, pois é a interação mecânica entre estes componentes que define a topologia da superfície de perda que o HPO deve navegar.

2.1 Deep Sparse Latent Feature Models (DSLFM)

O DSLFM opera sob a hipótese fundamental de que as triplas em um grafo de conhecimento são condicionalmente independentes dadas as estruturas de comunidade latentes das entidades envolvidas.5 Diferente de modelos de fatoração de tensores que assumem uma dimensão latente fixa $k$, o DSLFM emprega o Indian Buffet Process (IBP) como um prior sobre a matriz de alocação de features latentes $\mathbf{Z}$.

2.1.1 O Prior Indian Buffet Process (IBP) e a Dinâmica de Esparsidade

O IBP é um processo estocástico que define uma distribuição de probabilidade sobre matrizes binárias esparsas com um número infinito de colunas (features). A metáfora clássica envolve clientes (entidades) entrando em um restaurante indiano (espaço latente) e escolhendo pratos (features).2

A dinâmica matemática é regida pelo parâmetro de concentração $\alpha$. O número de novas features $K_{new}$ que o $i$-ésimo cliente experimenta segue uma distribuição Poisson:

$$K_{new} \sim \text{Poisson}\left(\frac{\alpha}{i}\right)$$

Para o HPO, isso é crucial: o parâmetro $\alpha$ não controla apenas a esparsidade final, mas a taxa de descoberta de complexidade do modelo. Um $\alpha$ mal ajustado pode levar a duas falhas catastróficas:

Colapso de Representação ($\alpha \to 0$): O modelo não gera features suficientes para capturar a semântica das relações, resultando em underfitting.

Explosão Combinatória ($\alpha \to \infty$): O modelo gera features excessivas, resultando em uma matriz $\mathbf{Z}$ densa, perda de interpretabilidade e overfitting massivo, além de estourar a memória da GPU.7

No DSLFM, a inferência não é feita via amostragem de Gibbs (que seria lenta), mas sim através de Inferência Variacional (VI) amortizada usando um VAE. O codificador neural projeta as entidades para os parâmetros variacionais da distribuição Beta-Bernoulli que aproxima o IBP.1 Aqui reside a primeira tensão de otimização: os gradientes do ELBO (Evidence Lower Bound) devem propagar através da amostragem discreta de $\mathbf{Z}$, exigindo relaxamentos contínuos (como Gumbel-Softmax ou aproximações de stick-breaking) cujos hiperparâmetros (temperatura $\tau$) são altamente sensíveis.1

2.2 Probabilistic Circuits Variant 2 (PC-V2): Regularização Estrutural

Enquanto o DSLFM foca na geração de features latentes, os Probabilistic Circuits (PCs) entram na arquitetura como garantidores de consistência e tratabilidade inferencial. A literatura recente de 2024 e 2025 3 destaca o uso de PCs para modelar distribuições complexas permitindo inferência exata (cálculo de verossimilhança e marginais) em tempo linear em relação ao tamanho do grafo computacional.

2.2.1 Definição de PC-V2: Contextos de Regras

Distinguimos a Variante 2 (PC-V2) baseada nos avanços descritos por Patil et al..3 Diferente de PCs genéricos usados apenas para estimativa de densidade de dados tabulares, o PC-V2 em KGC é projetado especificamente para:

Ingerir Regras Lógicas Mineradas: O sistema utiliza algoritmos como AnyBURL para extrair regras (e.g., $\forall x, y, z: Parent(x, y) \wedge Parent(y, z) \to Grandparent(x, z)$).

Modelar Contextos ($C$): Em vez de tratar todas as regras como universalmente válidas ou inválidas, o PC-V2 aprende uma distribuição $P_\theta(C)$ sobre "contextos" — subconjuntos de regras que são coerentes entre si.

 Inferência Tratável: O circuito representa a distribuição conjunta sobre triplas e contextos, permitindo calcular $P(Query | Context)$ exatamente.

A integração no DSLFM ocorre adicionando o negativo da log-verossimilhança do PC-V2 como um termo de regularização na função de perda global:

$$\mathcal{L}_{total} = \mathcal{L}_{ELBO}(DSLFM) + \lambda_{PC} \cdot \mathcal{L}_{PC-V2}(\mathbf{Z}, \text{Regras})$$

Onde $\lambda_{PC}$ é um hiperparâmetro que controla a força da consistência lógica. O desafio de otimização aqui é duplo: o PC-V2 deve ser aprendido simultaneamente (ou alternadamente) com o VAE, e seus parâmetros estruturais (profundidade, número de somas) interagem com a esparsidade de $\mathbf{Z}$. Se $\mathbf{Z}$ for muito ruidoso no início (fase de burn-in do IBP), o PC-V2 pode aprender correlações espúrias ou, inversamente, suprimir a exploração necessária do VAE ao penalizar configurações válidas mas ainda não consolidadas.

3. O Espaço de Hiperparâmetros e Dinâmicas de Treinamento

A complexidade do DSLFM-KGC com PC-V2 não reside apenas no número de hiperparâmetros, mas na natureza condicional e temporal de suas influências. A análise dos dados de benchmark 8 e das propriedades teóricas dos modelos componentes nos permite mapear o espaço de busca crítico.

3.1 Taxonomia dos Hiperparâmetros Críticos

A Tabela 1 apresenta os hiperparâmetros identificados como determinantes para a convergência e desempenho, baseados nas implementações de referência e nos snippets analisados.1

Categoria

Hiperparâmetro

Símbolo

Intervalo Típico

Comportamento e Impacto de Segunda Ordem

Otimização

Taxa de Aprendizado (Encoder)

$\eta_{enc}$

$10^{-5}$ a $10^{-3}$

Afeta a velocidade de convergência do VAE. Valores altos desestabilizam a divergência KL do IBP.

Otimização

Taxa de Aprendizado (PC)

$\eta_{pc}$

$10^{-4}$ a $10^{-2}$

Hipótese: Deve ser desacoplada de $\eta_{enc}$. O PC exige atualizações mais suaves para manter a estrutura probabilística válida.

Estrutura Latente

Prior Stick-Breaking

$\alpha_{qry}$

$10$ a $100$

Controla o número esperado de comunidades. Em WN18RR (hierárquico), $\alpha$ alto é necessário; em FB15k-237, menor.

Relaxamento

Temperatura Contrastiva

$\tau$

$0.02$ a $0.1$

Crítico para a função de perda contrastiva. $\tau$ baixo aguca a distribuição, mas pode causar vanishing gradients em grafos esparsos.

Regularização

Peso do PC-V2

$\lambda_{PC}$

$10^{-3}$ a $1.0$

Equilibra a fidelidade aos dados (reconstrução) vs. consistência lógica. $\lambda$ excessivo causa "rigidez" no modelo.

Estrutura PC

Profundidade do Circuito

$D_{pc}$

$2$ a $10$

Controla a complexidade das dependências lógicas capturadas. Profundidade excessiva retarda a inferência (tratabilidade).

Dinâmica

Épocas de Burn-in

$E_{burn}$

$50$ a $200$

Período onde o pruning de features do IBP é desativado para permitir exploração.

3.2 O Fenômeno de "Burn-in" e a Armadilha da Otimização Precoce

Um insight crucial derivado da literatura sobre Processos de Buffet Indiano é a necessidade de uma fase de burn-in.11 Durante as épocas iniciais, o modelo IBP tende a superestimar o número de features latentes ($K_+$) enquanto explora o espaço de configurações. Apenas após essa fase expansiva é que o termo de penalidade de esparsidade (KL Divergence) começa a podar efetivamente as features redundantes.

Isso cria uma curva de aprendizado não monotônica para a métrica de validação (MRR). Frequentemente, o MRR pode estagnar ou até cair ligeiramente durante a transição da fase "densa e ruidosa" para a fase "esparsa e estruturada".

Implicação para HPO: Algoritmos que dependem de early stopping agressivo baseados em curvas de aprendizado iniciais (como as primeiras 20 épocas) correm um risco severo de descartar configurações que possuem alto potencial de longo prazo, mas que sofrem de um início lento devido à complexidade da reorganização latente. Isso contradiz a premissa básica de eficiência do Hyperband em sua forma padrão.

4. Análise Comparativa de Estratégias de HPO

Nesta seção, avaliamos quatro paradigmas de HPO — TPE, NSGA-II, PBT e Hyperband — contra as necessidades específicas do DSLFM-KGC + PC-V2.

4.1 Tree-structured Parzen Estimator (TPE)

 O TPE é um método sequencial de Otimização Bayesiana que modela a densidade de probabilidade dos hiperparâmetros $p(\lambda | y)$ dados os resultados observados, dividindo as observações em grupos de "bom" e "mau" desempenho ($l(\lambda)$ e $g(\lambda)$).14

Mecanismo: Ao invés de modelar a superfície de resposta $y = f(\lambda)$ (como em Processos Gaussianos), o TPE otimiza a Expected Improvement (EI) através da razão $l(\lambda)/g(\lambda)$.

Adequação ao PC-V2: O TPE brilha em lidar com espaços de busca condicionais e heterogêneos (mistura de categóricos e contínuos). Por exemplo, a escolha da topologia do PC-V2 (e.g., "profundo" vs "largo") pode condicionar quais intervalos de $\lambda_{PC}$ são válidos. O TPE captura essas dependências naturalmente.

Limitação: Sendo sequencial, ele pode ser lento para explorar o espaço inicial e tende a ficar preso em mínimos locais se a exploração inicial for insuficiente. Além disso, em sua forma canônica, é mono-objetivo, o que força o pesquisador a agregar métricas (MRR e Esparsidade) em uma única escalar ponderada, muitas vezes de forma arbitrária.

4.2 Hyperband e a Controvérsia do Early Stopping

O Hyperband formula o HPO como um problema de alocação de recursos (bandit problem), descartando agressivamente configurações de baixo desempenho em estágios iniciais de treinamento (Successive Halving).16

O Conflito com IBP: Conforme identificado na análise de "burn-in", o Hyperband é perigoso para DSLFM. Snippets 18 sugerem que, embora eficiente para redes neurais convolucionais padrão, o Hyperband assume que a performance relativa das configurações é estável ao longo do tempo (i.e., se a config A é pior que B na época 10, ela provavelmente será pior na época 100).

Hipótese de Falha: Para o DSLFM, uma configuração com $\alpha$ alto pode ter desempenho pobre na época 10 (devido ao ruído de muitas features ativas) mas excelente na época 200 (após o PC-V2 organizar essas features em contextos válidos). O Hyperband eliminaria essa configuração prematuramente. Portanto, o uso de Hyperband exige um parâmetro de min_resources artificialmente alto, o que anula seus ganhos de eficiência.

4.3 Population Based Training (PBT)

O PBT evolui uma população de modelos, onde membros com baixo desempenho copiam os pesos e hiperparâmetros (com mutação) dos membros de alto desempenho.20

Incompatibilidade Estrutural: A cópia de pesos ("exploit") é problemática para modelos esparsos. Se um modelo $M_1$ (que aprendeu um conjunto de features latentes $Z_1$) copiar os pesos de $M_2$ (que opera sobre um conjunto $Z_2$ disjunto), a estrutura latente é quebrada catastróficamente.

Análise: O snippet 20 corrobora essa visão, afirmando que "dinâmicas de treinamento prejudicadas impedem modelos esparsos de compartilhar os mesmos hiperparâmetros ótimos". O PBT assume que os pesos são transferíveis suavemente, o que não é verdade quando a topologia do espaço latente (determinada pelo IBP) varia drasticamente entre indivíduos da população.

4.4 NSGA-II (Non-dominated Sorting Genetic Algorithm II)

O NSGA-II é um algoritmo evolutivo focado em otimização multi-objetivo, mantendo uma população diversa na fronteira de Pareto.21

Relevância Multi-Objetivo: Em KGC com PC-V2, temos objetivos conflitantes: (1) Maximizar MRR (precisão), (2) Minimizar o número de features ativas (esparsidade/interpretabilidade), e (3) Minimizar a violação de regras lógicas (consistência).

Vantagem: O NSGA-II não exige a agregação arbitrária desses objetivos. Ele permite descobrir configurações que oferecem compromissos distintos (e.g., um modelo levemente menos preciso, mas 90% mais esparso e logicamente consistente).

Sinergia com PC-V2: A natureza discreta/combinatória da estrutura do PC-V2 (quais regras incluir no contexto) mapeia-se bem para os operadores de mutação e crossover genéticos do NSGA-II.

5. Proposta de Estratégia Híbrida e Hipóteses Definidas

Com base na análise das falhas do Hyperband e PBT, e nas forças complementares do TPE e NSGA-II, propomos uma estratégia de otimização em dois estágios, denominada Híbrido NSGA-TPE com Consciência de Burn-in.

5.1 Hipóteses Formais para PC-V2

Antes de detalhar a estratégia, definimos as hipóteses que guiam a configuração do PC-V2:

Hipótese 1 (H1 - A Necessidade de Desacoplamento Temporal): A regularização do PC-V2 deve ser introduzida gradualmente (annealing). Aplicar restrições lógicas fortes $\lambda_{PC}$ no início do treinamento, quando o IBP ainda não estabeleceu features semânticas, leva a mínimos locais sub-ótimos onde o modelo "apaga" o conhecimento para satisfazer a lógica trivialmente.

Hipótese 2 (H2 - Estrutura Condicionada à Esparsidade): A profundidade ótima do circuito PC-V2 é inversamente proporcional à esparsidade alvo. Grafos latentes muito esparsos requerem circuitos mais profundos para capturar dependências de longo alcance que não são representadas explicitamente por arestas diretas.

Hipótese 3 (H3 - Sensibilidade da Temperatura Contrastiva): A temperatura $\tau$ no DSLFM controla a granularidade dos clusters latentes. Existe uma relação funcional ótima $\tau = f(\text{densidade do grafo})$ onde grafos mais esparsos exigem $\tau$ menor para forçar a discriminação.

5.2 O Protocolo de Otimização Proposto

A estratégia recomendada consiste em duas fases distintas para navegar o trade-off exploração/explotação sem cair nas armadilhas do IBP.

Fase 1: Exploração Topológica com NSGA-II (Coarse-Grained)

Nesta fase, o objetivo é mapear a Fronteira de Pareto e fixar os hiperparâmetros estruturais (discretos).

Algoritmo: NSGA-II.

Objetivos:.

Hiperparâmetros Alvo: Topologia do PC-V2, $\alpha$ do IBP (discretizado), Tamanho do Batch.

Configuração de Avaliação: Treinamento parcial, mas com uma salvaguarda crítica: Burn-in Protection. Cada indivíduo deve treinar por no mínimo $E_{burn}$ épocas (estimado em 50-80 baseada na estabilização da KL-divergence) antes de qualquer avaliação de fitness. Isso evita a eliminação de modelos IBP de convergência lenta.

Fase 2: Refinamento Paramétrico com TPE (Fine-Grained)

Uma vez selecionada uma região da Fronteira de Pareto (e.g., "modelos com esparsidade > 80%"), fixamos a estrutura e refinamos os parâmetros contínuos.

Algoritmo: TPE (com suporte a prioris).

Hiperparâmetros Alvo: Taxas de aprendizado ($\eta_{enc}, \eta_{pc}$), Temperatura $\tau$, Decaimento de peso, $\lambda_{PC}$.

Métrica: Uma função escalar derivada da preferência do usuário na fronteira de Pareto (e.g., $MRR - \beta \cdot (1-\text{Sparsity})$).

Justificativa: O TPE é mais eficiente em amostrar regiões contínuas densas perto do ótimo do que algoritmos genéticos.14

6. Análise Experimental e Discussão de Cenários

A aplicação desta estratégia requer adaptações sensíveis ao dataset em questão. A literatura e os snippets fornecem dados vitais sobre WN18RR e FB15k-237 que informam essas adaptações.

6.1 WN18RR: O Desafio Hierárquico e Esparso

O dataset WN18RR (WordNet) é caracterizado por uma estrutura hierárquica estrita e alta esparsidade.

Configuração Ótima Observada: Snippet 1 indica uma taxa de aprendizado inicial mais alta ($8 \times 10^{-5}$) e um número elevado de épocas (65) para convergência, comparado a outros datasets.

Papel do PC-V2: Em WN18RR, as regras lógicas (como assimetria e transitividade de hiperônimos) são fortes. O HPO deve permitir um $\lambda_{PC}$ alto. A estratégia NSGA-II tenderá a favorecer circuitos mais profundos que consigam codificar cadeias transitivas longas sem perder a tratabilidade.

Risco de TPE/Hyperband: O TPE padrão pode demorar a encontrar a região de $\tau$ baixo ($0.02$ 8) necessária para separar conceitos hierárquicos sutis. O NSGA-II, ao manter diversidade, preserva indivíduos com $\tau$ baixo mesmo que sua convergência inicial seja lenta.

6.2 FB15k-237: Densidade e Relações Complexas

O FB15k-237 (Freebase) possui muito mais tipos de relações e maior densidade média, mas carece de regras lógicas "limpas" (muitas exceções).

Configuração Ótima Observada: Taxa de aprendizado menor ($2 \times 10^{-5}$) e convergência rápida (15 épocas).1

Papel do PC-V2: Aqui, o PC-V2 atua mais como um regularizador suave ("soft constraint") para evitar overfitting. A estratégia de HPO deve favorecer $\lambda_{PC}$ menor.

Dinâmica IBP: Com maior densidade de dados, o IBP tende a gerar muitas features. O HPO deve focar em ajustar o $\alpha$ para evitar a explosão de memória. O TPE é particularmente eficaz aqui, pois a superfície de resposta é mais suave e menos propensa a armadilhas de burn-in longas do que no WN18RR.

6.3 Comparativo Sintético de Desempenho

Baseando-nos nas características dos algoritmos e nos requisitos do DSLFM, projetamos o seguinte perfil de desempenho (Tabela 2):

Método HPO

Custo Computacional

Capacidade de Lidar com Burn-in

Qualidade Final (Esparsidade + MRR)

Recomendação

Grid Search

Exorbitante

Alta (se executado totalmente)

Média

Não Recomendado

Hyperband

Baixo

Muito Baixa (risco de descarte prematuro)

Baixa (tende a modelos densos)

Não Recomendado para IBP

PBT

Médio

Média

Baixa (quebra de estrutura latente)

Não Recomendado

NSGA-II

Alto

Alta

Alta (excelente trade-off)

Recomendado para Fase 1

TPE

Médio

Média/Alta

Alta (excelente em ajuste fino)

Recomendado para Fase 2

Híbrido (Proposto)

Médio-Alto

Máxima

Estado-da-Arte

Estratégia Ouro

7. Conclusão e Diretrizes Futuras

A otimização de hiperparâmetros para arquiteturas neuro-simbólicas complexas como o DSLFM-KGC equipado com Probabilistic Circuits Variant 2 não é meramente uma tarefa de ajuste de curvas, mas um problema de orquestração de dinâmicas de aprendizado. A análise rigorosa demonstra que as premissas de estabilidade temporal e transferência de pesos, fundamentais para algoritmos populares como Hyperband e PBT, são violadas pelas características intrínsecas do Indian Buffet Process e pela rigidez estrutural dos Circuitos Probabilísticos.

A "melhor estratégia", portanto, não é um único algoritmo, mas o protocolo híbrido NSGA-II $\to$ TPE. Esta abordagem respeita a necessidade de exploração estrutural multi-objetivo nas fases iniciais (onde a esparsidade e a lógica competem com a precisão) e aproveita a eficiência Bayesiana para o refinamento final. Crucialmente, a introdução do conceito de "Burn-in Protection" no ciclo de avaliação de HPO é a chave para desbloquear o verdadeiro potencial destes modelos em grafos esparsos, garantindo que as estruturas latentes interpretáveis tenham tempo hábil para emergir do ruído estocástico inicial.

Para pesquisas futuras, sugere-se a investigação de meta-modelos que possam prever dinamicamente o ponto de transição de fase do IBP ($E_{burn}$) baseando-se em estatísticas do grafo de entrada, permitindo uma adaptação ainda mais fina dos recursos computacionais durante a otimização.

Referências Integradas

O presente relatório sintetiza informações e dados técnicos dos seguintes artefatos de pesquisa:

.1

Referências citadas

Deep Sparse Latent Feature Models for Knowledge Graph Completion - arXiv, acessado em dezembro 16, 2025, https://arxiv.org/pdf/2411.15694

Deep Sparse Latent Feature Models for Knowledge Graph Completion - arXiv, acessado em dezembro 16, 2025, https://arxiv.org/html/2411.15694v1

Probabilistic Circuits for Knowledge Graph Completion with Reduced Rule Sets - arXiv, acessado em dezembro 16, 2025, https://arxiv.org/pdf/2508.06706

Probabilistic Circuits for Knowledge Graph Completion with Reduced Rule Sets - arXiv, acessado em dezembro 16, 2025, https://arxiv.org/abs/2508.06706

[Literature Review] Deep Sparse Latent Feature Models for Knowledge Graph Completion, acessado em dezembro 16, 2025, https://www.themoonlight.io/en/review/deep-sparse-latent-feature-models-for-knowledge-graph-completion

The Convergent Indian Buffet Process - MDPI, acessado em dezembro 16, 2025, https://www.mdpi.com/2227-7390/13/23/3881

Assessing the effects of hyperparameters on knowledge graph embedding quality - PMC, acessado em dezembro 16, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC10164002/

DEEP SPARSE LATENT FEATURE MODELS FOR KNOWLEDGE GRAPH COMPLETION - OpenReview, acessado em dezembro 16, 2025, https://openreview.net/notes/edits/attachment?id=IJ2ChMvj29&name=pdf

1 INTRODUCTION - arXiv, acessado em dezembro 16, 2025, https://arxiv.org/html/2411.12256v1

Hyperparameters of our DSLFM-KGC for each dataset during training. - ResearchGate, acessado em dezembro 16, 2025, https://www.researchgate.net/figure/Hyperparameters-of-our-DSLFM-KGC-for-each-dataset-during-training_tbl4_386111526

Learning invariant features using the Transformed Indian Buffet Process - Austerweil Lab, acessado em dezembro 16, 2025, https://alab.psych.wisc.edu/papers/files/AusterweilGriffithsTIBPNIPS2010.pdf

Large Scale Nonparametric Bayesian Inference: Data Parallelisation in the Indian Buffet Process - MLG Cambridge, acessado em dezembro 16, 2025, https://mlg.eng.cam.ac.uk/pub/pdf/DosKnoMohGha09.pdf

The Indian Buffet Process: Scalable Inference and Extensions - MLG Cambridge, acessado em dezembro 16, 2025, https://mlg.eng.cam.ac.uk/pub/pdf/Dos09.pdf

DeepHyper: A Python Package for Massively Parallel Hyperparameter Optimization in Machine Learning - Journal of Open Source Software, acessado em dezembro 16, 2025, https://joss.theoj.org/papers/10.21105/joss.07975.pdf

FedHPO-Bench: A Benchmark Suite for Federated Hyperparameter Optimization - Proceedings of Machine Learning Research, acessado em dezembro 16, 2025, https://proceedings.mlr.press/v202/wang23n/wang23n.pdf

uai2025 - Tutorials, acessado em dezembro 16, 2025, https://www.auai.org/uai2025/tutorials

A Unified Framework for Gradient-based Hyperparameter Optimization and Meta-learning - UCL Discovery - University College London, acessado em dezembro 16, 2025, https://discovery.ucl.ac.uk/id/eprint/10129910/1/Thesis_Luca_Franceschi_UCL.pdf

Hyperband Tuned Deep Neural Network With Well Posed Stacked Sparse AutoEncoder for Detection of DDoS Attacks in Cloud - IEEE Xplore, acessado em dezembro 16, 2025, https://ieeexplore.ieee.org/iel7/6287639/8948470/09212425.pdf

A simple transfer-learning extension of Hyperband - Amazon Science, acessado em dezembro 16, 2025, https://assets.amazon.science/32/28/c0db3ecc45c199af2c37da3a2ec6/a-simple-transfer-learning-extension-of-hyperband.pdf

SINDy-RL for interpretable and efficient model-based reinforcement learning - PMC - NIH, acessado em dezembro 16, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC12663201/

Multi-Objective Hyperparameter Optimization in Machine Learning – An Overview - arXiv, acessado em dezembro 16, 2025, https://arxiv.org/html/2206.07438v3

Hyperparameter optimization: Classics, acceleration, online, multi-objective, and tools - AIMS Press, acessado em dezembro 16, 2025, https://www.aimspress.com/aimspress-data/mbe/2024/6/PDF/mbe-21-06-275.pdf

Neurosymbolic AI for Reasoning over Knowledge Graphs: A Survey - arXiv, acessado em dezembro 16, 2025, https://arxiv.org/html/2302.07200v3

Speeding Up Multi-Objective Hyperparameter Optimization by Task Similarity-Based Meta-Learning for the Tree-Structured Parzen Estimator - IJCAI, acessado em dezembro 16, 2025, https://www.ijcai.org/proceedings/2023/0487.pdf
