"""
SOTA Negative Sampling Strategies for Knowledge Graph Embeddings.

Implements advanced negative sampling techniques:
- Type-constrained sampling: sample negatives of same entity type
- Relation-aware sampling: sample based on relation patterns
- Self-adversarial sampling: sample weighted by current model scores
- Uniform sampling: baseline random sampling

References:
- RotatE (Sun et al., 2019): Self-adversarial negative sampling
- Type-constrained (Krompaß et al., 2015): Type-aware corruption
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import torch
from torch import Tensor


class NegativeSamplingStrategy(Enum):
    """Estratégias de amostragem negativa disponíveis."""

    UNIFORM = "uniform"
    SELF_ADVERSARIAL = "self_adversarial"
    TYPE_CONSTRAINED = "type_constrained"
    RELATION_AWARE = "relation_aware"


class NegativeSampler(ABC):
    """Interface abstrata para amostragem negativa."""

    @abstractmethod
    def sample(
        self,
        heads: Tensor,
        relations: Tensor,
        tails: Tensor,
        num_negatives: int,
        corrupt_head: bool = True,
    ) -> Tensor:
        """
        Gera amostras negativas corrompendo head ou tail.

        Args:
            heads: Tensor de IDs de entidades cabeça [batch_size]
            relations: Tensor de IDs de relações [batch_size]
            tails: Tensor de IDs de entidades cauda [batch_size]
            num_negatives: Número de amostras negativas por tripla
            corrupt_head: Se True, corrompe head; senão, corrompe tail

        Returns:
            Tensor de IDs negativos [batch_size, num_negatives]
        """
        pass


class UniformNegativeSampler(NegativeSampler):
    """Amostragem negativa uniforme (baseline)."""

    def __init__(self, num_entities: int, device: torch.device):
        self.num_entities = num_entities
        self.device = device

    def sample(
        self,
        heads: Tensor,
        relations: Tensor,
        tails: Tensor,
        num_negatives: int,
        corrupt_head: bool = True,
    ) -> Tensor:
        """Amostra uniformemente de todas as entidades."""
        batch_size = heads.size(0)
        negatives = torch.randint(
            0,
            self.num_entities,
            (batch_size, num_negatives),
            device=self.device,
        )
        return negatives


class SelfAdversarialNegativeSampler(NegativeSampler):
    """
    Self-adversarial negative sampling (RotatE).

    Pesa amostras pela probabilidade do modelo atual,
    focando em negativos mais difíceis.
    """

    def __init__(
        self,
        num_entities: int,
        device: torch.device,
        temperature: float = 1.0,
    ):
        self.num_entities = num_entities
        self.device = device
        self.temperature = temperature
        self._score_fn: Optional[callable] = None

    def set_score_function(self, score_fn: callable) -> None:
        """Define função de scoring do modelo para pesar amostras."""
        self._score_fn = score_fn

    def sample(
        self,
        heads: Tensor,
        relations: Tensor,
        tails: Tensor,
        num_negatives: int,
        corrupt_head: bool = True,
    ) -> Tensor:
        """
        Amostra com pesos adversariais baseados no modelo.

        Se score_fn não definida, faz amostragem uniforme.
        """
        batch_size = heads.size(0)

        # Gera candidatos uniformes primeiro
        candidates = torch.randint(
            0,
            self.num_entities,
            (batch_size, num_negatives * 2),  # oversample para seleção
            device=self.device,
        )

        if self._score_fn is None:
            # Fallback para uniforme
            return candidates[:, :num_negatives]

        # Calcula scores para candidatos
        with torch.no_grad():
            if corrupt_head:
                # Cria triplas com heads corrompidas
                expanded_relations = relations.unsqueeze(1).expand(-1, num_negatives * 2)
                expanded_tails = tails.unsqueeze(1).expand(-1, num_negatives * 2)
                scores = self._score_fn(candidates, expanded_relations, expanded_tails)
            else:
                # Cria triplas com tails corrompidas
                expanded_heads = heads.unsqueeze(1).expand(-1, num_negatives * 2)
                expanded_relations = relations.unsqueeze(1).expand(-1, num_negatives * 2)
                scores = self._score_fn(expanded_heads, expanded_relations, candidates)

            # Calcula pesos softmax com temperatura
            weights = torch.softmax(scores / self.temperature, dim=-1)

            # Amostra índices baseado nos pesos
            indices = torch.multinomial(weights, num_negatives, replacement=False)

            # Seleciona negativos pelos índices
            negatives = torch.gather(candidates, 1, indices)

        return negatives


class TypeConstrainedNegativeSampler(NegativeSampler):
    """
    Type-constrained negative sampling.

    Só corrompe com entidades do mesmo tipo da entidade original,
    gerando negativos mais plausíveis e difíceis.
    """

    def __init__(
        self,
        num_entities: int,
        device: torch.device,
        entity_types: Optional[Tensor] = None,
        type_to_entities: Optional[dict[int, Tensor]] = None,
    ):
        """
        Args:
            num_entities: Total de entidades no KG
            device: Device para tensores
            entity_types: Tensor mapping entity_id -> type_id [num_entities]
            type_to_entities: Dict mapping type_id -> tensor de entity_ids
        """
        self.num_entities = num_entities
        self.device = device
        self.entity_types = entity_types
        self.type_to_entities = type_to_entities or {}

        # Fallback sampler se types não configurados
        self._fallback = UniformNegativeSampler(num_entities, device)

    def sample(
        self,
        heads: Tensor,
        relations: Tensor,
        tails: Tensor,
        num_negatives: int,
        corrupt_head: bool = True,
    ) -> Tensor:
        """Amostra negativos do mesmo tipo da entidade corrompida."""
        if self.entity_types is None or not self.type_to_entities:
            return self._fallback.sample(
                heads, relations, tails, num_negatives, corrupt_head
            )

        batch_size = heads.size(0)
        target_entities = heads if corrupt_head else tails

        # Obtém tipos das entidades alvo
        target_types = self.entity_types[target_entities]  # [batch_size]

        negatives = torch.zeros(
            (batch_size, num_negatives),
            dtype=torch.long,
            device=self.device,
        )

        # Group by type for batch sampling (vectorized optimization)
        unique_types = target_types.unique()
        for entity_type in unique_types:
            type_val = entity_type.item()
            mask = target_types == entity_type
            count = mask.sum().item()
            
            if type_val in self.type_to_entities:
                type_entities = self.type_to_entities[type_val]
                num_type_entities = len(type_entities)
                
                if num_type_entities > 0:
                    # Batch sample for all entities of this type
                    indices = torch.randint(
                        0, num_type_entities, (count, num_negatives), device=self.device
                    )
                    negatives[mask] = type_entities[indices]
                else:
                    negatives[mask] = torch.randint(
                        0, self.num_entities, (count, num_negatives), device=self.device
                    )
            else:
                # Unknown type, use uniform
                negatives[mask] = torch.randint(
                    0, self.num_entities, (count, num_negatives), device=self.device
                )

        return negatives


class RelationAwareNegativeSampler(NegativeSampler):
    """
    Relation-aware negative sampling.

    Mantém estatísticas de quais entidades tipicamente aparecem
    como head/tail para cada relação, e amostra dessas distribuições.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        device: torch.device,
    ):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.device = device

        # Histogramas de frequência por relação
        # relation_head_freq[r][e] = frequência de e como head em r
        self.relation_head_freq: dict[int, Tensor] = {}
        self.relation_tail_freq: dict[int, Tensor] = {}

        self._fallback = UniformNegativeSampler(num_entities, device)

    def build_frequency_tables(self, triples: Tensor) -> None:
        """
        Constrói tabelas de frequência a partir das triplas de treino.

        Args:
            triples: Tensor de triplas [num_triples, 3] (head, rel, tail)
        """
        for r in range(self.num_relations):
            # Triplas com relação r
            mask = triples[:, 1] == r
            rel_triples = triples[mask]

            if len(rel_triples) > 0:
                heads = rel_triples[:, 0]
                tails = rel_triples[:, 2]

                # Conta frequências
                head_counts = torch.bincount(heads, minlength=self.num_entities).float()
                tail_counts = torch.bincount(tails, minlength=self.num_entities).float()

                # Normaliza para probabilidades
                head_counts = head_counts / (head_counts.sum() + 1e-8)
                tail_counts = tail_counts / (tail_counts.sum() + 1e-8)

                self.relation_head_freq[r] = head_counts.to(self.device)
                self.relation_tail_freq[r] = tail_counts.to(self.device)

    def sample(
        self,
        heads: Tensor,
        relations: Tensor,
        tails: Tensor,
        num_negatives: int,
        corrupt_head: bool = True,
    ) -> Tensor:
        """Amostra baseado em frequências head/tail por relação."""
        if not self.relation_head_freq:
            return self._fallback.sample(
                heads, relations, tails, num_negatives, corrupt_head
            )

        batch_size = heads.size(0)
        negatives = torch.zeros(
            (batch_size, num_negatives),
            dtype=torch.long,
            device=self.device,
        )

        freq_dict = self.relation_head_freq if corrupt_head else self.relation_tail_freq

        # Group by relation for batch sampling (vectorized optimization)
        unique_relations = relations.unique()
        for rel_tensor in unique_relations:
            rel = rel_tensor.item()
            mask = relations == rel_tensor
            count = mask.sum().item()

            if rel in freq_dict:
                probs = freq_dict[rel]
                # Batch sample for all entries with this relation
                negatives[mask] = torch.multinomial(
                    probs + 1e-8,  # smoothing
                    count * num_negatives,
                    replacement=True,
                ).view(count, num_negatives)
            else:
                # Unknown relation, use uniform
                negatives[mask] = torch.randint(
                    0, self.num_entities, (count, num_negatives), device=self.device
                )

        return negatives


class NegativeSamplerFactory:
    """Factory for creating negative samplers based on strategy.
    
    Pattern: Factory Method
    
    Creates appropriate NegativeSampler instances based on the 
    specified sampling strategy, encapsulating the creation logic.
    """

    @staticmethod
    def create(
        strategy: NegativeSamplingStrategy,
        num_entities: int,
        device: torch.device,
        num_relations: int = 0,
        temperature: float = 1.0,
        entity_types: Optional[Tensor] = None,
        type_to_entities: Optional[dict[int, Tensor]] = None,
    ) -> NegativeSampler:
        """
        Cria um sampler baseado na estratégia especificada.

        Args:
            strategy: Estratégia de amostragem
            num_entities: Total de entidades
            device: Device para tensores
            num_relations: Total de relações (para relation-aware)
            temperature: Temperatura para self-adversarial
            entity_types: Mapeamento entidade -> tipo
            type_to_entities: Mapeamento tipo -> entidades

        Returns:
            Instância do sampler apropriado
        """
        factories = {
            NegativeSamplingStrategy.UNIFORM: lambda: UniformNegativeSampler(
                num_entities, device
            ),
            NegativeSamplingStrategy.SELF_ADVERSARIAL: lambda: SelfAdversarialNegativeSampler(
                num_entities, device, temperature
            ),
            NegativeSamplingStrategy.TYPE_CONSTRAINED: lambda: TypeConstrainedNegativeSampler(
                num_entities, device, entity_types, type_to_entities
            ),
            NegativeSamplingStrategy.RELATION_AWARE: lambda: RelationAwareNegativeSampler(
                num_entities, num_relations, device
            ),
        }

        if strategy not in factories:
            raise ValueError(f"Estratégia desconhecida: {strategy}")

        return factories[strategy]()


class CompositeNegativeSampler(NegativeSampler):
    """
    Combina múltiplas estratégias de amostragem.

    Pode misturar, por exemplo, 50% type-constrained + 50% self-adversarial.
    """

    def __init__(
        self,
        samplers: list[tuple[NegativeSampler, float]],
        num_entities: int,
        device: torch.device,
    ):
        """
        Args:
            samplers: Lista de (sampler, peso) onde pesos somam 1.0
            num_entities: Total de entidades
            device: Device para tensores
        """
        self.samplers = samplers
        self.num_entities = num_entities
        self.device = device

        # Normaliza pesos
        total_weight = sum(w for _, w in samplers)
        self.samplers = [(s, w / total_weight) for s, w in samplers]

    def sample(
        self,
        heads: Tensor,
        relations: Tensor,
        tails: Tensor,
        num_negatives: int,
        corrupt_head: bool = True,
    ) -> Tensor:
        """Combina amostras de múltiplos samplers."""
        batch_size = heads.size(0)
        all_negatives = []

        remaining = num_negatives
        for sampler, weight in self.samplers:
            n = int(num_negatives * weight)
            if n > 0:
                neg = sampler.sample(heads, relations, tails, n, corrupt_head)
                all_negatives.append(neg)
                remaining -= n

        # Adiciona amostras restantes do primeiro sampler
        if remaining > 0 and self.samplers:
            neg = self.samplers[0][0].sample(
                heads, relations, tails, remaining, corrupt_head
            )
            all_negatives.append(neg)

        return torch.cat(all_negatives, dim=1)[:, :num_negatives]
