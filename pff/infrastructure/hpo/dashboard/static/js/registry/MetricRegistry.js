export const MetricHints = {
    score: {
        tech: "Métrica objetivo usada pelo otimizador; pode ser composta ou simples.",
        simple: "A nota geral do campeonato.",
        direction: 'up'
    },
    mcc: {
        tech: "Coeficiente de Matthews; combina VP, VN, FP e FN e lida bem com desbalanceamento.",
        simple: "A bússola da verdade.",
        extra: [{ label: "Faixa", value: "-1 a 1" }],
        direction: 'up'
    },
    mrr: {
        tech: "Média do inverso da posição correta no ranking; premia acertos no topo.",
        simple: "Quanto mais perto do topo, maior a nota.",
        extra: [{ label: "Faixa", value: "0 a 1" }],
        direction: 'up'
    },
    loss: {
        tech: "Função de perda usada na otimização; menor é melhor.",
        simple: "O nível de desconforto do modelo.",
        extra: [{ label: "Unidade", value: "adimensional" }],
        direction: 'down'
    },
    duration: {
        tech: "Tempo total de execução do trial/treino.",
        simple: "Quanto tempo o forno ficou ligado.",
        extra: [{ label: "Unidade", value: "segundos (s)" }],
        direction: 'down'
    },
    accuracy: {
        tech: "Proporção total de acertos no conjunto avaliado.",
        simple: "O placar bruto.",
        extra: [{ label: "Faixa", value: "0 a 1" }],
        direction: 'up'
    },
    precision: {
        tech: "Entre os positivos previstos, a fração correta.",
        simple: "Evita alarme falso.",
        extra: [{ label: "Faixa", value: "0 a 1" }],
        direction: 'up'
    },
    recall: {
        tech: "Entre os positivos reais, a fração detectada.",
        simple: "Não deixa o peixe escapar.",
        extra: [{ label: "Faixa", value: "0 a 1" }],
        direction: 'up'
    },
    f1: {
        tech: "Média harmônica entre precisão e recall.",
        simple: "Equilíbrio entre errar e não deixar passar.",
        extra: [{ label: "Faixa", value: "0 a 1" }],
        direction: 'up'
    },
    auc: {
        tech: "Área sob a curva ROC; mede separabilidade.",
        simple: "O quanto separa dois grupos.",
        extra: [{ label: "Faixa", value: "0 a 1" }],
        direction: 'up'
    },
    hits1: {
        tech: "Fração de casos com resposta correta em 1º lugar.",
        simple: "Acertou de primeira.",
        extra: [{ label: "Faixa", value: "0 a 1" }],
        direction: 'up'
    },
    hits3: {
        tech: "Fração de casos com resposta correta no Top-3.",
        simple: "Subiu ao pódio.",
        extra: [{ label: "Faixa", value: "0 a 1" }],
        direction: 'up'
    },
    hits10: {
        tech: "Fração de casos com resposta correta no Top-10.",
        simple: "Entrou no Top-10.",
        extra: [{ label: "Faixa", value: "0 a 1" }],
        direction: 'up'
    },
    inference_time: {
        tech: "Latência média por consulta de inferência.",
        simple: "Tempo de resposta.",
        extra: [{ label: "Unidade", value: "milissegundos (ms)" }],
        direction: 'down'
    },
    moving_average: {
        tech: "Suavização da série para reduzir ruído.",
        simple: "A linha amaciada da tendência.",
        direction: 'up'
    },
    trend: {
        tech: "Inclinação média da série ao longo do tempo.",
        simple: "A seta da direção geral.",
        direction: 'up'
    },
    incumbent: {
        tech: "Melhor valor acumulado até o ponto atual.",
        simple: "O recorde batido até agora.",
        direction: 'up'
    },
    objective: {
        tech: "Valor observado em um trial específico.",
        simple: "O placar daquele momento.",
        direction: 'up'
    },
    performance_dim: {
        tech: "Métrica em função da dimensão do modelo.",
        simple: "Quanto tamanho compra de performance.",
        extra: [{ label: "Eixo X", value: "dimensão" }],
        direction: 'up'
    },
    latency_tradeoff: {
        tech: "Trade‑off entre latência e qualidade; fronteira de Pareto.",
        simple: "Mapa de custo‑benefício sem desperdício.",
        extra: [{ label: "Eixos", value: "score (↑) × latência (↓)" }],
        direction: 'up'
    }
};
