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
    },
    negative_sample_size: {
        tech: "Quantidade de triplas negativas geradas por batch para contraste.",
        simple: "Quantos exemplos errados o modelo vê para aprender a dizer NÃO.",
        extra: [{ label: "Unidade", value: "triplas por batch" }]
    },
    regularization_weight: {
        tech: "Peso da penalidade L2; controla o tamanho dos parâmetros.",
        simple: "O freio que impede o modelo de exagerar nos números.",
        extra: [{ label: "Escala", value: "peso adimensional" }]
    },
    early_stopping_patience: {
        tech: "Número de épocas sem melhora antes de interromper o treino.",
        simple: "A paciência do treinador antes de dizer chega.",
        extra: [{ label: "Unidade", value: "épocas" }]
    }
};
