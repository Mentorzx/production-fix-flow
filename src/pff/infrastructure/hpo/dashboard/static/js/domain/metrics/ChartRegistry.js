import { BaseRegistry } from "./BaseRegistry.js";

const CHARTS = {
    convergence: {
        title: "Histórico de Otimização",
        tech: "Série temporal do objetivo por trial/época para verificar convergência e regressões.",
        simple: "A linha do tempo que mostra se estamos melhorando ou patinando.",
        extra: [{ label: "Eixos", value: "tempo (trials) × score" }]
    },
    params: {
        title: "Melhor Configuração",
        tech: "Hiperparâmetros do incumbent (melhor resultado até agora) com os valores atuais.",
        simple: "A receita que deu mais certo até este ponto.",
        extra: [{ label: "Fonte", value: "melhor trial filtrado" }]
    },
    matrix: {
        title: "Matriz de Confusão",
        tech: "Agregação de previsões por classe (VP, VN, FP, FN) para detectar viés.",
        simple: "Um raio‑X dos acertos e dos tropeços do modelo.",
        extra: [{ label: "Unidade", value: "contagem e porcentagem" }]
    },
    fanova: {
        title: "Importância fANOVA",
        tech: "Decomposição de variância para medir a contribuição global dos parâmetros.",
        simple: "Quem manda mais no resultado quando gira os botões.",
        extra: [{ label: "Saída", value: "importância relativa" }]
    },
    correlation: {
        title: "Matriz de Correlação",
        tech: "Correlação entre parâmetros/métricas (Pearson/Spearman) para identificar dependências.",
        simple: "Quem anda junto e quem anda na direção oposta.",
        extra: [{ label: "Escala", value: "-1 a 1" }]
    },
    interaction: {
        title: "Interaction Plot",
        tech: "Mostra como pares de parâmetros interagem para afetar o objetivo.",
        simple: "Dois ingredientes juntos mudam o sabor mais do que separados.",
        extra: [{ label: "Foco", value: "sinergia entre pares" }]
    },
    timeline: {
        title: "Timeline de Execução",
        tech: "Duração e sequência de trials para identificar gargalos.",
        simple: "A agenda do experimento: quem demorou mais.",
        extra: [{ label: "Unidade", value: "tempo por trial" }]
    },
    hypervolume: {
        title: "Hypervolume",
        tech: "Volume dominado no espaço multiobjetivo; maior é melhor.",
        simple: "Quanto território bom foi conquistado no mapa multiobjetivo.",
        extra: [{ label: "Interpretação", value: "maior = melhor" }]
    },
    edf: {
        title: "EDF Plot",
        tech: "Função de distribuição empírica para comparar desempenho acumulado.",
        simple: "A curva que mostra quantos trials ficam abaixo de cada nível.",
        extra: [{ label: "Eixo Y", value: "fração de trials" }]
    },
    detailed_history: {
        title: "Ranking de Trials",
        tech: "Tabela consolidada com todas as métricas por trial. Score e Loss usam barras proporcionais (DataBar): Score cresce de 0→1 (azul); Loss é invertida — menor loss = barra maior (vermelho). Duração usa barra relativa ao range dos dados (laranja). Métricas de classificação (MCC, Accuracy, etc.) e ranking (MRR, Hits@K) exibem pílulas com gradiente vermelho→amarelo→verde conforme o valor se aproxima de 1. O rodapé mostra estatísticas de duração: Média (soma÷N), Mediana (valor central quando ordenados) e Erro (desvio padrão÷√N, indica a confiança da média).",
        simple: "Um boletim completo de cada tentativa. As barras mostram quem foi bem (azul grande = bom score) e quem gastou menos (vermelho grande = menos perda). As pílulas coloridas funcionam como semáforo: vermelha = ruim, amarela = ok, verde = ótimo. No rodapé, a Média é o tempo típico, a Mediana é o tempo do trial 'do meio', e o Erro diz o quanto esses tempos variam.",
        extra: [
            { label: "Score", value: "Barra azul 0→1 (maior = melhor)" },
            { label: "Loss", value: "Barra vermelha invertida (menor loss = barra maior)" },
            { label: "Duração", value: "Barra laranja relativa (min→max dos dados)" },
            { label: "Pílulas", value: "Vermelho→Amarelo→Verde conforme 0→0.5→1" },
            { label: "Média", value: "Soma dos tempos ÷ quantidade" },
            { label: "Mediana", value: "Valor central (metade demora mais, metade menos)" },
            { label: "Erro (±)", value: "Desvio padrão ÷ √N — quanto menor, mais confiável a média" },
            { label: "★ MELHOR", value: "Trial com maior score (excluindo warmstart)" },
            { label: "Eficiência", value: "Score ÷ Duração × 100 — qualidade por segundo gasto" }
        ]
    },
    parallel: {
        title: "Coordenadas Paralelas",
        tech: "Visualização multidimensional de hiperparâmetros e objetivo por trial.",
        simple: "Um emaranhado de linhas que revela padrões escondidos.",
        extra: [{ label: "Cada linha", value: "um trial" }]
    },
    contour: {
        title: "Superfície de Resposta",
        tech: "Interpolação 2D/3D do objetivo em função de dois parâmetros.",
        simple: "Mapa de altitude do desempenho.",
        extra: [{ label: "Eixos", value: "parâmetros X/Y × score" }]
    },
    search_space_table: {
        title: "Espaço de Busca",
        tech: "Definição do domínio de busca: parâmetros, tipos e limites configurados.",
        simple: "O cardápio de ingredientes permitidos na receita.",
        extra: [{ label: "Origem", value: "configuração do HPO" }]
    },
    pareto_front: {
        title: "Fronteira de Pareto",
        tech: "Soluções não dominadas no trade‑off entre qualidade e custo.",
        simple: "Os melhores sem desperdício de tempo ou qualidade.",
        extra: [{ label: "Eixos", value: "score (↑) × duração (↓)" }]
    },
    hardware_monitor: {
        title: "Monitor de Hardware",
        tech: "Telemetria de uso de CPU/GPU/RAM/VRAM para detectar gargalos.",
        simple: "O painel do motor dizendo se está no limite.",
        extra: [{ label: "Unidade", value: "% e GB" }]
    },
    pc_comparison: {
        title: "Comparação PC",
        tech: "Compara variantes/configurações do módulo PC por score e latência.",
        simple: "Uma batalha entre versões do cérebro de regras.",
        extra: [{ label: "Critério", value: "qualidade e custo" }]
    },
    structural_metrics: {
        title: "Métricas Estruturais",
        tech: "Relaciona custo estrutural (complexidade) e performance obtida.",
        simple: "Quanto estrutura custa para ganhar pontos.",
        extra: [{ label: "Leitura", value: "complexidade × score" }]
    },
    latency_pareto: {
        title: "Latência x Qualidade",
        tech: "Trade‑off entre tempo de resposta e qualidade do modelo.",
        simple: "Rápido ou preciso? Este gráfico mostra o equilíbrio.",
        extra: [{ label: "Eixos", value: "latência (↓) × score (↑)" }]
    },
    metrics_evolution: {
        title: "Evolução de Métricas",
        tech: "Acompanha múltiplas métricas ao longo dos trials para detectar ganhos e regressões.",
        simple: "Várias notas no boletim, não só a final.",
        extra: [{ label: "Visão", value: "comparação multi‑métrica" }]
    },
    learning_curve: {
        title: "Curvas de Aprendizado",
        tech: "Séries de treino/validação por época para monitorar convergência e overfitting.",
        simple: "Mostra se o modelo aprende ou só decora.",
        extra: [{ label: "Eixos", value: "época × loss/score" }]
    },
    scatter_plot: {
        title: "Gráfico de Dispersão",
        tech: "Relação bivariada entre variáveis por trial; evidencia tendência e outliers.",
        simple: "Um céu de pontos para achar padrões.",
        extra: [{ label: "Cada ponto", value: "um trial" }]
    },
    gradient_health: {
        title: "Saúde do Gradiente",
        tech: "Normas e estatísticas do gradiente para detectar explosão/vanishing.",
        simple: "Se o motor patina ou dispara, aqui aparece.",
        extra: [{ label: "Sinal", value: "estabilidade do treino" }]
    },
    early_stopping: {
        title: "Early Stopping",
        tech: "Heurística baseada em estagnação recente da loss para sugerir parada.",
        simple: "O semáforo dizendo se vale continuar treinando.",
        extra: [{ label: "Base", value: "melhora recente" }]
    },
    loss_projection: {
        title: "Extrapolação de Perda",
        tech: "Projeção simples da loss para antecipar ganhos nas próximas épocas.",
        simple: "Uma previsão de para onde a curva deve ir se nada mudar.",
        extra: [{ label: "Uso", value: "planejamento de tempo" }]
    },
    optimization_velocity: {
        title: "Velocidade de Otimização",
        tech: "Inclinação média do objetivo ao longo dos trials.",
        simple: "Quão rápido a busca acelera.",
        extra: [{ label: "Interpretação", value: "inclinação da curva" }]
    },
    estimated_score: {
        title: "Estimativa de Score Final",
        tech: "Projeção do score esperado ao fim do orçamento de trials.",
        simple: "A aposta informada de onde a busca deve chegar.",
        extra: [{ label: "Horizonte", value: "fim do orçamento" }]
    },
    elbo_breakdown: {
        title: "ELBO Breakdown",
        tech: "Decompõe ELBO em reconstrução e KL para diagnosticar VAE/variacionais.",
        simple: "Mostra de onde vem a conta do treino.",
        extra: [{ label: "Partes", value: "reconstrução + regularização" }]
    },
    pc2_metrics: {
        title: "PC2 Metrics",
        tech: "Métricas de regras e latência do módulo PC2 durante a execução.",
        simple: "Indicadores do cérebro de regras enquanto trabalha.",
        extra: [{ label: "Foco", value: "regras e custo" }]
    },
    raw_config: {
        title: "Configuração Bruta",
        tech: "Dump do dicionário de parâmetros/configuração do trial selecionado.",
        simple: "O receituário completo do experimento.",
        extra: [{ label: "Formato", value: "chave → valor" }]
    },
    terminal_log: {
        title: "Logs de Execução",
        tech: "Stream de logs para depuração em tempo real; útil para erros e warnings.",
        simple: "A conversa do sistema dizendo o que está acontecendo.",
        extra: [{ label: "Uso", value: "diagnóstico rápido" }]
    },
    full_metrics_log: {
        title: "Log de Métricas por Época",
        tech: "Tabela granular com métricas registradas a cada época de treino. Loss usa barra invertida (vermelho) relativa ao range observado. Duração diferencia épocas de avaliação (amarelo, com métricas de classificação/ranking) de épocas de treino puro (laranja), cada grupo com seu próprio range min/max para comparação justa. Pílulas de classificação e ranking seguem o gradiente HSL vermelho(0)→amarelo(0.5)→verde(1). O rodapé agrega Média, Mediana e Erro Padrão da duração de todas as épocas.",
        simple: "O diário de bordo do treino, época por época. A barra de perda mostra se o modelo está melhorando (barra vermelha crescendo = perda caindo). As barras de tempo são separadas por tipo: épocas de avaliação (amarelas, mais lentas porque calculam métricas) e de treino (laranjas, mais rápidas). Cada tipo é comparado só com os seus — justo. No rodapé, a Média mostra o tempo típico por época, a Mediana é o tempo 'do meio da fila', e o Erro diz se os tempos são estáveis ou muito variados.",
        extra: [
            { label: "Loss", value: "Barra vermelha invertida — menor loss = progresso" },
            { label: "Dur. Avaliação", value: "Barra amarela — épocas com métricas clf/ranking (range próprio)" },
            { label: "Dur. Treino", value: "Barra laranja — épocas sem avaliação (range próprio)" },
            { label: "Pílulas", value: "Semáforo: vermelho (≈0) → amarelo (≈0.5) → verde (≈1)" },
            { label: "Média", value: "Tempo total ÷ número de épocas" },
            { label: "Mediana", value: "Tempo da época que fica no meio quando ordenadas" },
            { label: "Erro (±)", value: "Desvio padrão ÷ √N — estabilidade do tempo entre épocas" },
            { label: "Eficiência", value: "Score ÷ Duração × 100 — rendimento por segundo" }
        ]
    },
    confusion_matrix: {
        title: "Matriz de Confusão",
        tech: "Distribuição de acertos/erros por classe para identificar viés e classes difíceis.",
        simple: "Mostra onde o modelo mais confunde as categorias.",
        extra: [{ label: "Células", value: "VP, VN, FP, FN" }]
    },
    regression_chart: {
        title: "Tendência de Score (Regressão)",
        tech: "Regressão linear sobre o score por trial para estimar tendência global.",
        simple: "A linha que mostra para onde o desempenho está indo.",
        extra: [{ label: "Eixos", value: "trial × score" }]
    },
    regression_insights: {
        title: "Resumo da Regressão",
        tech: "Resumo com R², inclinação e projeção do score para o fim do horizonte escolhido.",
        simple: "Um raio‑X rápido da tendência e do próximo patamar esperado.",
        extra: [{ label: "Saída", value: "R², slope, projeção" }]
    },
    trial_diff: {
        title: "Comparativo de Trials",
        tech: "Tabela transposta para comparar métricas e parâmetros entre trials-chave.",
        simple: "Um quadro lado a lado para ver quem está melhor e por quê.",
        extra: [{ label: "Colunas", value: "melhor, recente, pior" }]
    },
    trial_learning_metrics: {
        title: "Loss + MCC/MRR",
        tech: "Evolução por época da loss e das métricas (MCC/MRR) durante um trial.",
        simple: "Mostra se a dor diminui enquanto a qualidade sobe.",
        extra: [{ label: "Eixos", value: "época × loss/score" }]
    },
    fold_confusions: {
        title: "Matriz de Confusão (Folds)",
        tech: "Comparação de matrizes por fold/época para verificar estabilidade entre splits.",
        simple: "Vê se o modelo acerta de forma consistente em cada divisão.",
        extra: [{ label: "Comparação", value: "folds/épocas" }]
    }
};

export const ChartRegistry = new BaseRegistry("Charts", CHARTS);
