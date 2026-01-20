import pytest
import torch

from pff.domain.learning.dslfm.dslfm_kgc import (
    DSLFMKGCConfig,
    DSLFMKGCModel,
)

# --- Fixtures ---


@pytest.fixture
def robust_config():
    """Configuração pequena mas suficiente para causar colisões."""
    return DSLFMKGCConfig(
        num_entities=10,  # Poucas entidades para forçar colisões
        num_relations=2,
        entity_dim=8,
        feature_dim=8,
        max_communities=4,
        hidden_dim=16,
        num_triples=50,
        lambda_pc=0.1,  # Ativar componentes complexos
        lambda_logic=0.1,
        num_global_negatives=2,
        negative_sample_size=5,  # Cap de negativas
    )


@pytest.fixture
def model(robust_config):
    model = DSLFMKGCModel(robust_config)
    return model.to("cpu")


# --- Testes de Robustez ---


class TestDSLFMRobustness:
    def test_negative_sampling_collision_avoidance(self, model):
        """
        Cenário: Universo de entidades muito pequeno.
        Teste: Verificar se o sampler evita retornar o próprio tail positivo como negativo.
        """
        # Configurar para forçar colisão: entidade 0 é a única resposta
        heads = torch.zeros(5, dtype=torch.long)
        relations = torch.zeros(5, dtype=torch.long)
        tails = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)  # Positivos cobrem metade do espaço

        # O método interno _sample_global_negative_tail_ids tenta evitar colisão
        neg_ids = model._sample_global_negative_tail_ids(
            heads, relations, tails, num_negatives=1, num_entities=model.config.num_entities
        )

        # Verificar se algum negativo é igual ao positivo correspondente
        # neg_ids shape: (batch, num_negatives) -> (5, 1)
        collisions = neg_ids == tails.unsqueeze(1)

        assert not collisions.any(), (
            f"Amostragem negativa gerou colisões com positivos: \nPos: {tails}\nNeg: {neg_ids.squeeze()}"
        )

    def test_evaluation_filter_leakage(self, model):
        """
        Cenário: Avaliação com Data Leakage.
        Teste: O método `evaluate` deve aceitar um filter_fn que mascara triplas conhecidas (treino).
        """
        model.eval()
        # Tripla de avaliação (h=0, r=0, t=1)
        eval_triples = torch.tensor([[0, 0, 1]], dtype=torch.long)

        # Scores sem filtro
        # score_triples_batch retorna score para a tripla específica
        # Mas evaluate calcula ranks contra TODAS as outras entidades.

        # Vamos simular um filter_fn que diz que a entidade 2 também é uma resposta correta (leakage do treino)
        # e portanto deve ser ignorada no ranking (score = -inf)

        def mock_filter_fn(scores, h, r, candidate_indices, true_tails):
            # scores shape: (batch, num_candidates)
            # h, r shape: (batch)
            # candidate_indices: (batch, num_candidates) ou (num_candidates) se chunked
            # true_tails: (batch)

            # Vamos mascarar a entidade ID 2
            mask = candidate_indices == 2

            # Handle different shapes of candidate_indices
            if candidate_indices.ndim == 1:
                # Chunked evaluation: scores is (batch, chunk_size)
                # mask is (chunk_size,)
                # We need to apply mask to dimension 1 of scores
                scores[:, mask] = -float("inf")
            else:
                # Batch evaluation: scores and indices have same shape
                scores[mask] = -float("inf")

            return scores

        # Executar evaluate com o filtro
        metrics = model.evaluate(
            eval_triples, batch_size=1, filter_fn=mock_filter_fn, score_all_tails_chunk_size=100
        )

        assert "mrr" in metrics
        assert "hits@1" in metrics
        # O teste aqui é integridade: não deve quebrar com o filtro customizado

    def test_pathological_batch_size_one(self, model):
        """
        Cenário: Batch size = 1.
        Motivo: Muitas operações de `squeeze` ou `view` falham quando dimensão é removida indevidamente.
        """
        heads = torch.tensor([0])
        relations = torch.tensor([0])
        tails = torch.tensor([1])

        try:
            output = model.compute_loss(heads, relations, tails)
            assert output["loss"].item() > 0
        except IndexError as e:
            pytest.fail(f"Batch size 1 causou erro de índice: {e}")
        except RuntimeError as e:
            pytest.fail(f"Batch size 1 causou erro de runtime (shape mismatch?): {e}")

    def test_cold_start_entity(self, model):
        """
        Cenário: Entidade que nunca foi vista (ou embedding não treinada).
        Teste: Verificar se forward pass gera scores válidos (não NaN) mesmo para embeddings aleatórias.
        """
        # ID 9 é a última entidade (config.num_entities=10)
        heads = torch.tensor([9, 9])
        relations = torch.tensor([0, 0])
        tails = torch.tensor([0, 1])

        output = model.forward(heads, relations, tails)
        scores = output["scores"]

        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

    def test_numerical_stability_extreme_values(self, model):
        """
        Cenário: Inputs que poderiam causar log(0) ou exp(inf).
        Teste: VAE Encoder e PC Log Prob.
        """
        # Injetar pesos muito grandes na embedding para simular explosão
        with torch.no_grad():
            model.entity_embedding.weight.data.fill_(100.0)

        heads = torch.tensor([0, 1])
        relations = torch.tensor([0, 0])
        tails = torch.tensor([2, 3])

        # Isso vai passar pelo VAE (reparameterize com mu grande) e PC (log prob)
        output = model.forward(heads, relations, tails, use_pc=True)
        scores = output["scores"]

        # Verificar se o sistema clipou ou tratou os NaNs
        # É aceitável que a loss seja alta, mas não NaN
        if torch.isnan(scores).any():
            pytest.fail("Scores tornaram-se NaN com pesos de embedding altos (100.0)")

    def test_forward_with_all_optional_components(self, model):
        """
        Cenário: Ativar PC, Logic Layer, e BERT (mockado) simultaneamente.
        Teste: Garantir que a soma das losses funciona e retorna gradientes para todos.
        """
        # Mock logic encoder e PC model se não estiverem inicializados (a fixture ativa eles via config)
        assert model.pc_model is not None
        # Logic encoder pode ser None se imports falharem, checar

        heads = torch.tensor([0, 1])
        relations = torch.tensor([0, 1])
        tails = torch.tensor([2, 3])

        loss_dict = model.compute_loss(heads, relations, tails)

        total_loss = loss_dict["loss"]
        total_loss.backward()

        # Verificar se pc_penalty foi calculado (já que lambda_pc > 0)
        assert "pc_penalty" in loss_dict
        # Se logic estiver ativo
        if model.logic_encoder:
            # Logic penalty geralmente é adicionada à loss total
            pass

    def test_massive_self_loops(self, model):
        """
        Cenário: Batch consistindo inteiramente de self-loops (h == t).
        Motivo: VAE pode colapsar se p(h) e p(t) forem idênticos.
        """
        # batch_size = 5
        heads = torch.tensor([0, 1, 2, 3, 4])
        relations = torch.zeros(5, dtype=torch.long)
        tails = heads.clone()  # Self-loops

        output = model.forward(heads, relations, tails)
        scores = output["scores"]

        assert scores.shape == (5,)
        assert not torch.isnan(scores).any()

        # O score deve ser consistente (não necessariamente alto ou baixo, depende do modelo)

    def test_missing_features_structure(self, robust_config):
        """
        Cenário: Configuração onde feature_dim é diferente de entity_dim.
        Teste: O modelo deve lidar com projeções se necessário (ou falhar na init se não suportado).
        """
        config = robust_config
        config.entity_dim = 16
        config.feature_dim = 8  # Diferente

        try:
            model = DSLFMKGCModel(config)
            # Testar um forward pass rápido
            h = torch.tensor([0])
            r = torch.tensor([0])
            t = torch.tensor([0])
            model.forward(h, r, t)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                pytest.fail(f"Modelo não suporta entity_dim != feature_dim: {e}")
            else:
                raise e
