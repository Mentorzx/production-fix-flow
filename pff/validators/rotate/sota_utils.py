"""
SOTA KGE Training Utilities.

Implements advanced techniques for Knowledge Graph Embedding training:
- Label Smoothing: reduces overconfidence in predictions
- Reciprocal Relations: doubles training data with inverse triples
- Embedding Regularization: L2, L3, dropout-based

Design Patterns:
- Strategy Pattern: RegularizationType enum selects regularization strategy at runtime
- Decorator Pattern: Regularization classes wrap loss computation with additional penalty terms

References:
- Label smoothing (Szegedy et al., 2016): Rethinking Inception
- Reciprocal relations (Lacroix et al., 2018): ComplEx-N3
"""

from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RegularizationType(Enum):
    """Tipos de regularização para embeddings."""

    NONE = "none"
    L2 = "l2"  # Frobenius norm
    L3 = "l3"  # N3 regularization (Lacroix et al., 2018)
    DROPOUT = "dropout"


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss com label smoothing.

    Em vez de targets hard (0 ou 1), usa soft labels:
    - positive: 1 - smoothing + smoothing/num_classes
    - negative: smoothing/num_classes

    Isso previne overconfidence e melhora generalização.
    """

    def __init__(self, smoothing: float = 0.1, num_classes: int = 2):
        """
        Args:
            smoothing: Fator de suavização [0, 1). Default 0.1.
            num_classes: Número de classes para distribuição uniforme.
        """
        super().__init__()

        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing deve estar em [0, 1), recebeu: {smoothing}")

        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Calcula loss com label smoothing.

        Args:
            logits: Predições do modelo [batch_size] ou [batch_size, num_classes]
            targets: Labels reais [batch_size] (valores 0 ou 1 para binário)

        Returns:
            Loss escalar
        """
        if logits.dim() == 1:
            # Binário: usa BCE com smoothed targets
            smooth_targets = targets * self.confidence + (1 - targets) * self.smoothing
            return F.binary_cross_entropy_with_logits(logits, smooth_targets)

        # Multi-classe: distribui smoothing uniformemente
        log_probs = F.log_softmax(logits, dim=-1)

        # Cria targets suaves
        smooth_targets = torch.full_like(log_probs, self.smoothing / self.num_classes)
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)

        return -(smooth_targets * log_probs).sum(dim=-1).mean()


class MarginRankingLossWithSmoothing(nn.Module):
    """
    Margin ranking loss com soft margins via label smoothing.

    Reduz o gap efetivo entre positivos e negativos proporcionalmente
    ao fator de smoothing, tornando o modelo mais conservador.
    """

    def __init__(self, margin: float = 1.0, smoothing: float = 0.1):
        super().__init__()
        self.base_margin = margin
        self.smoothing = smoothing
        # Reduz margem efetiva pelo smoothing
        self.effective_margin = margin * (1.0 - smoothing)

    def forward(self, pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
        """
        Calcula margin ranking loss suavizada.

        Args:
            pos_scores: Scores de triplas positivas [batch_size]
            neg_scores: Scores de triplas negativas [batch_size, num_neg]

        Returns:
            Loss escalar
        """
        # Expande pos_scores para broadcasting
        if neg_scores.dim() > pos_scores.dim():
            pos_scores = pos_scores.unsqueeze(-1)

        # margin_loss = max(0, margin - pos + neg)
        loss = F.relu(self.effective_margin - pos_scores + neg_scores)

        return loss.mean()


class ReciprocalRelationAugmenter:
    """
    Augmenta dados com relações recíprocas.

    Para cada tripla (h, r, t), adiciona (t, r_inv, h) onde r_inv = r + num_relations.
    Isso dobra efetivamente o dataset e melhora modelagem de relações simétricas.

    Reference: Lacroix et al. (2018) - Tensor Factorization for KB Completion
    """

    def __init__(self, num_relations: int):
        """
        Args:
            num_relations: Número original de relações no KG
        """
        self.num_relations = num_relations

    def augment(self, triples: Tensor) -> Tensor:
        """
        Adiciona triplas recíprocas ao dataset.

        Args:
            triples: Tensor [num_triples, 3] com (head, relation, tail)

        Returns:
            Tensor [2 * num_triples, 3] com originais + recíprocas
        """
        # Triplas originais: (h, r, t)
        heads = triples[:, 0]
        relations = triples[:, 1]
        tails = triples[:, 2]

        # Triplas recíprocas: (t, r + num_rel, h)
        reciprocal_relations = relations + self.num_relations
        reciprocal_triples = torch.stack(
            [tails, reciprocal_relations, heads],
            dim=1,
        )

        # Concatena originais + recíprocas
        augmented = torch.cat([triples, reciprocal_triples], dim=0)

        return augmented

    def get_extended_num_relations(self) -> int:
        """Retorna número de relações após augmentation."""
        return self.num_relations * 2


class EmbeddingRegularizer(nn.Module):
    """
    Regularização para embeddings de entidades e relações.

    Suporta múltiplos tipos de regularização:
    - L2: ||e||²  (Frobenius)
    - L3: ||e||³  (N3 regularization)
    - Dropout: aplicado durante treino

    N3 é especialmente efetivo para modelos ComplEx/RotatE.
    """

    def __init__(
        self,
        reg_type: RegularizationType = RegularizationType.L2,
        weight: float = 0.0001,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.reg_type = reg_type
        self.weight = weight
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        entity_embeddings: Optional[Tensor] = None,
        relation_embeddings: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calcula regularização para embeddings.

        Args:
            entity_embeddings: Embeddings de entidades [batch, dim]
            relation_embeddings: Embeddings de relações [batch, dim]

        Returns:
            Loss de regularização escalar
        """
        if self.weight == 0.0 or self.reg_type == RegularizationType.NONE:
            return torch.tensor(0.0, device=self._get_device(entity_embeddings, relation_embeddings))

        reg_loss = torch.tensor(0.0, device=self._get_device(entity_embeddings, relation_embeddings))

        embeddings = []
        if entity_embeddings is not None:
            embeddings.append(entity_embeddings)
        if relation_embeddings is not None:
            embeddings.append(relation_embeddings)

        for emb in embeddings:
            if self.reg_type == RegularizationType.L2:
                reg_loss = reg_loss + torch.norm(emb, p=2) ** 2
            elif self.reg_type == RegularizationType.L3:
                # N3 regularization: sum of cubes of absolute values
                reg_loss = reg_loss + torch.sum(torch.abs(emb) ** 3)

        return self.weight * reg_loss

    def apply_dropout(self, embeddings: Tensor) -> Tensor:
        """Aplica dropout se configurado."""
        if self.dropout is not None and self.training:
            return self.dropout(embeddings)
        return embeddings

    def _get_device(
        self,
        entity_embeddings: Optional[Tensor],
        relation_embeddings: Optional[Tensor],
    ) -> torch.device:
        """Obtém device dos embeddings."""
        if entity_embeddings is not None:
            return entity_embeddings.device
        if relation_embeddings is not None:
            return relation_embeddings.device
        return torch.device("cpu")


class GradientScaling:
    """
    Utilities para scaling de gradientes durante treino.

    Implementa técnicas SOTA:
    - Gradient accumulation: acumula gradientes antes de update
    - Gradient clipping: previne explosão de gradientes
    - Mixed precision scaling: para treino FP16
    """

    def __init__(
        self,
        clip_value: float = 1.0,
        accumulation_steps: int = 1,
        use_amp: bool = False,
    ):
        self.clip_value = clip_value
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp

        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self._step_count = 0

    def scale_loss(self, loss: Tensor) -> Tensor:
        """Escala loss para accumulation e AMP."""
        scaled = loss / self.accumulation_steps

        if self.scaler is not None:
            scaled = self.scaler.scale(scaled)

        return scaled

    def clip_gradients(self, model: nn.Module) -> float:
        """
        Aplica gradient clipping.

        Returns:
            Norma do gradiente antes do clipping
        """
        if self.scaler is not None:
            self.scaler.unscale_(model)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.clip_value,
        )

        return grad_norm.item()

    def should_update(self) -> bool:
        """Verifica se deve executar optimizer step."""
        self._step_count += 1
        return self._step_count % self.accumulation_steps == 0

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Executa optimizer step com suporte a AMP."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()


class WarmupScheduler:
    """
    Learning rate scheduler com warmup linear.

    Aumenta LR linearmente de 0 até target durante warmup_steps,
    depois aplica decay (linear, cosine, ou constant).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
        decay_type: str = "linear",  # linear, cosine, constant
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.decay_type = decay_type

        # Guarda LRs iniciais
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self) -> None:
        """Atualiza learning rate."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Warmup: linear increase
            warmup_factor = self.current_step / self.warmup_steps
            lrs = [lr * warmup_factor for lr in self.base_lrs]
        else:
            # Decay phase
            decay_steps = self.current_step - self.warmup_steps
            total_decay_steps = self.total_steps - self.warmup_steps

            if self.decay_type == "linear":
                decay_factor = 1.0 - (decay_steps / total_decay_steps)
            elif self.decay_type == "cosine":
                import math

                decay_factor = 0.5 * (1 + math.cos(math.pi * decay_steps / total_decay_steps))
            else:  # constant
                decay_factor = 1.0

            lrs = [
                max(self.min_lr, lr * decay_factor)
                for lr in self.base_lrs
            ]

        # Aplica novos LRs
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr

    def get_lr(self) -> list[float]:
        """Retorna LRs atuais."""
        return [pg["lr"] for pg in self.optimizer.param_groups]
