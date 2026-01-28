export const DESIGN_TOKENS = {
    colors: {
        bg: "#09090b", card: "#18181b", border: "#27272a", text: "#a1a1aa",
        textHigh: "#e4e4e7", primary: "#f97316", success: "#a3e635",
        error: "#f43f5e", warning: "#f59e0b", grid: "#27272a", tooltip: "#09090b",
    }
};

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
    }
};

export const ParamHints = {
    embedding_dim: {
        tech: "Tamanho do vetor latente; define capacidade de representação.",
        simple: "A resolução do mapa do modelo.",
        extra: [{ label: "Impacto", value: "capacidade × custo" }]
    },
    dslfm_epochs: {
        tech: "Número máximo de passagens completas pelo dataset.",
        simple: "Quantas vezes o modelo relê o livro.",
        extra: [{ label: "Unidade", value: "épocas" }]
    },
    learning_rate: {
        tech: "Tamanho do passo do gradiente; controla velocidade e estabilidade.",
        simple: "O acelerador do aprendizado.",
        extra: [{ label: "Escala", value: "valores pequenos" }]
    }
};

export const CHART_CONTRACT = {
    convergence: {
        title: "Histórico de Otimização",
        tech: "Série temporal do objetivo para verificar convergência e regressões.",
        simple: "A linha do tempo que mostra se estamos melhorando.",
        extra: [{ label: "Eixos", value: "tempo × score" }]
    },
    params: {
        title: "Melhor Configuração",
        tech: "Hiperparâmetros do incumbent com os valores atuais.",
        simple: "A receita que deu mais certo até agora.",
        extra: [{ label: "Fonte", value: "melhor trial" }]
    },
    matrix: {
        title: "Matriz de Confusão",
        tech: "Agregação de acertos/erros por classe.",
        simple: "Um raio‑X dos acertos e dos tropeços.",
        extra: [{ label: "Células", value: "VP, VN, FP, FN" }]
    }
};
