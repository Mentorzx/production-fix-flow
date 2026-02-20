/**
 * Provide ParamRegistry module functionality for the HPO dashboard.
 */

import { BaseRegistry } from "./BaseRegistry.js";

const PARAMS = {
  embedding_dim: {
    tech: "Tamanho do vetor latente; define a capacidade do espaço de representação.",
    simple: "A resolução do mapa onde o modelo desenha as relações.",
    extra: [{ label: "Impacto", value: "mais dimensão = mais memória e custo" }],
  },
  dslfm_epochs: {
    tech: "Número máximo de passagens completas pelo dataset durante o treino.",
    simple: "Quantas vezes o modelo relê o livro inteiro.",
    extra: [{ label: "Unidade", value: "épocas" }],
  },
  learning_rate: {
    tech: "Magnitude do passo na descida do gradiente; controla velocidade e estabilidade.",
    simple: "O acelerador do aprendizado: rápido demais derrapa, lento demais não chega.",
    extra: [{ label: "Escala", value: "valores pequenos são comuns" }],
  },
  negative_sample_size: {
    tech: "Quantidade de triplas negativas geradas por batch para contraste.",
    simple: "Quantos exemplos errados o modelo vê para aprender a dizer NÃO.",
    extra: [{ label: "Unidade", value: "triplas por batch" }],
  },
  regularization_weight: {
    tech: "Peso da penalidade L2; controla o tamanho dos parâmetros.",
    simple: "O freio que impede o modelo de exagerar nos números.",
    extra: [{ label: "Escala", value: "peso adimensional" }],
  },
  early_stopping_patience: {
    tech: "Número de épocas sem melhora antes de interromper o treino.",
    simple: "A paciência do treinador antes de dizer chega.",
    extra: [{ label: "Unidade", value: "épocas" }],
  },
  // --- New HPO Parameters ---
  adversarial_temperature: {
    tech: "Temperatura alpha usada na amostragem negativa adversarial (RotatE).",
    simple: "O quão difícil tornamos a distinção entre certo e errado.",
    extra: [{ label: "Faixa", value: "0.5 a 1.0+" }],
  },
  lambda_logic: {
    tech: "Peso (λ) do componente lógico/semântico na perda híbrida.",
    simple: "Quanto o modelo obedece às regras vs dados.",
    extra: [{ label: "Foco", value: "Neuro-Simbólico" }],
  },
  lambda_pc: {
    tech: "Peso (λ) do componente de Circuitos Probabilísticos (PC).",
    simple: "Importância dadas às restrições probabilísticas.",
    extra: [{ label: "Tipo", value: "Estrutural" }],
  },
  t_norm: {
    tech: "Norma triangular usada para operações lógicas difusas (fuzzification).",
    simple: "A matemática para combinar verdades parciais.",
    extra: [{ label: "Exemplos", value: "Lukasiewicz, Product" }],
  },
  attr_hidden_dim: {
    tech: "Dimensão da camada oculta para atributos de entidades.",
    simple: "Capacidade de processamento de características.",
    extra: [{ label: "Unidade", value: "neurônios" }],
  },
  pruning_threshold: {
    tech: "Limiar mínimo para manter uma conexão ou regra ativa.",
    simple: "A régua de corte para descartar o irrelevante.",
    extra: [{ label: "Efeito", value: "Sparsity" }],
  },
  rebuild_every: {
    tech: "Frequência (em épocas) de reconstrução da estrutura do circuito/grafo.",
    simple: "De quanto em quanto tempo o modelo faz uma reforma na casa.",
    extra: [{ label: "Unidade", value: "épocas" }],
  },
  max_circuit_depth: {
    tech: "Profundidade máxima permitida para o circuito probabilístico.",
    simple: "O limite de altura do prédio que o modelo pode construir.",
    extra: [{ label: "Impacto", value: "Complexidade" }],
  },
  min_delta: {
    tech: "Variação mínima necessária na métrica monitorada para resetar o early stopping.",
    simple: "O progresso mínimo que conta como melhoria.",
    extra: [{ label: "Unidade", value: "absoluta" }],
  },
  validate_every: {
    tech: "Intervalo de épocas entre rodadas de validação.",
    simple: "Com que frequência fazemos uma prova surpresa.",
    extra: [{ label: "Unidade", value: "épocas" }],
  },
  rerank_top_k: {
    tech: "Tamanho do Top-K usado no reranqueamento; controla quantos candidatos são reavaliados.",
    simple: "Quantos finalistas vão para a fase de desempate.",
    extra: [{ label: "Unidade", value: "candidatos" }],
  },
  contrastive_temperature: {
    tech: "Temperatura do contraste; regula a suavidade/rigidez da distribuição de similaridades.",
    simple: "O botão que deixa a comparação mais rígida ou mais flexível.",
    extra: [{ label: "Efeito", value: "menor = mais rígido" }],
  },
  num_global_negatives: {
    tech: "Quantidade de negativos globais usados por batch na perda contrastiva.",
    simple: "Quantos contraexemplos o modelo enfrenta por rodada.",
    extra: [{ label: "Unidade", value: "amostras" }],
  },
  kl_weight: {
    tech: "Peso do termo KL; controla regularização/pressão de compressão no espaço latente.",
    simple: "O aperto no funil de informação: mais alto, mais compressão.",
    extra: [{ label: "Unidade", value: "adimensional" }],
  },
  batch_size: {
    tech: "Número de triplas processadas por iteração; afeta throughput e convergência.",
    simple: "Quantos dados o modelo mastiga de cada vez.",
    extra: [{ label: "Impacto", value: "maior = mais rápido, mais memória" }],
  },
  ibp_alpha: {
    tech: "Parâmetro alpha do Interval Bound Propagation; controla a robustez adversarial.",
    simple: "O nível de blindagem contra perturbações nos dados.",
    extra: [{ label: "Faixa", value: "0.0 a 1.0" }],
  },
  kge_model: {
    tech: "Arquitetura base de KGE (Knowledge Graph Embedding) utilizada (ex: RotatE, TransE).",
    simple: "Qual modelo de IA é usado para mapear o grafo.",
    extra: [{ label: "Exemplos", value: "RotatE, TransE, ComplEx" }],
  },
  max_communities: {
    tech: "Número máximo de comunidades detectadas no grafo para agrupamento.",
    simple: "Em quantos grupos dividimos o mapa de relações.",
    extra: [{ label: "Unidade", value: "clusters" }],
  },
  metric_bounds: {
    tech: "Limites de normalização para métricas de avaliação (min/max).",
    simple: "A escala usada para padronizar as notas.",
    extra: [{ label: "Tipo", value: "booleano/config" }],
  },
  refresh_cache_on_val: {
    tech: "Se o cache de embeddings é recalculado antes de cada validação.",
    simple: "Se atualiza a memória antes de cada prova.",
    extra: [{ label: "Tipo", value: "booleano" }],
  },
  self_adversarial: {
    tech: "Ativa amostragem negativa self-adversarial (RotatE); pesos de negativos por softmax.",
    simple: "O modelo cria seus próprios desafios de dificuldade variável.",
    extra: [{ label: "Tipo", value: "booleano" }],
  },
  use_bert: {
    tech: "Ativa o uso de embeddings textuais BERT para enriquecer representações de entidades.",
    simple: "Usa IA de linguagem para entender melhor os nomes das coisas.",
    extra: [{ label: "Tipo", value: "booleano" }],
  },
  use_compile: {
    tech: "Ativa torch.compile para JIT-compilar o modelo e acelerar inferência.",
    simple: "Liga o turbo de otimização do PyTorch.",
    extra: [{ label: "Tipo", value: "booleano" }],
  },
};

/**
 * Expose param registry for dashboard usage.
 */
export const ParamRegistry = new BaseRegistry("Parameters", PARAMS);
