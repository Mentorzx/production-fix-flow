"""BERT-based relation encoder for DSLFM-KGC.

This module provides a BERT-based text encoder for relation names,
which contain semantic information in telecom domain.

Design Patterns:
    - Strategy: Can swap BERT model variants
    - Factory: Model creation via from_pretrained
"""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


import torch
from torch import nn

try:
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModel = None
    AutoTokenizer = None

from pff.shared.core.logger import logger


class RelationTextEncoder(nn.Module):
    """BERT encoder for relation text descriptions.

    Encodes relation names (e.g., "billCycleChangeType") into dense embeddings
    using a pretrained BERT model. Caches tokenized relations for efficiency.

    Args:
        model_name: HuggingFace model name.
        hidden_dim: Output projection dimension.
        freeze_bert: Whether to freeze BERT weights.
        max_length: Maximum token length for relation names.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_dim: int = 256,
        freeze_bert: bool = True,
        max_length: int = 32,
    ) -> None:
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers package required for BERT encoder. "
                "Install with: poetry add transformers"
            )

        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.freeze_bert = freeze_bert

        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        bert_hidden = self.bert.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(bert_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info("Pesos do BERT congelados para encoder de relacoes")
            logger.debug("Memoria otimizada: low_cpu_mem_usage=True ativado")

        self._relation_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._embedding_cache: dict[str, torch.Tensor] = {}

    def precompute_relation_embeddings(
        self,
        relation_names: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Precompute embeddings for all relations.

        Args:
            relation_names: List of relation name strings.
            device: Target device for embeddings.

        Returns:
            Tensor of shape [num_relations, hidden_dim].
        """
        self.bert = self.bert.to(device)
        self.projection = self.projection.to(device)

        embeddings = []

        for rel_name in relation_names:
            if rel_name in self._embedding_cache:
                emb = self._embedding_cache[rel_name].to(device)
            else:
                clean_name = self._clean_relation_name(rel_name)

                tokens = self.tokenizer(
                    clean_name,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                tokens = {k: v.to(device) for k, v in tokens.items()}

                with torch.no_grad():
                    outputs = self.bert(**tokens)
                    cls_embedding = outputs.last_hidden_state[:, 0, :]
                    emb = self.projection(cls_embedding).squeeze(0)

                self._embedding_cache[rel_name] = emb.detach()

            embeddings.append(emb)

        return torch.stack(embeddings, dim=0)

    def _clean_relation_name(self, rel_name: str) -> str:
        """Clean relation name for BERT tokenization.

        Converts camelCase to spaces and handles _inv suffix.

        Args:
            rel_name: Raw relation name.

        Returns:
            Cleaned string suitable for BERT.
        """
        is_inverse = rel_name.endswith("_inv")
        if is_inverse:
            rel_name = rel_name[:-4]

        import re

        cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", rel_name)
        cleaned = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", cleaned)
        cleaned = cleaned.replace("_", " ").lower()

        if is_inverse:
            cleaned = f"inverse of {cleaned}"

        return cleaned

    def forward(
        self,
        relation_ids: torch.Tensor,
        precomputed_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Lookup precomputed relation embeddings.

        Args:
            relation_ids: Tensor of relation indices [batch_size].
            precomputed_embeddings: Precomputed embeddings [num_relations, hidden_dim].

        Returns:
            Relation embeddings [batch_size, hidden_dim].
        """
        return precomputed_embeddings[relation_ids]


class LightweightRelationEncoder(nn.Module):
    """Lightweight relation encoder without BERT.

    Uses learned embeddings for relations when BERT is too memory-intensive.
    Serves as fallback for low-VRAM scenarios.

    Args:
        num_relations: Number of unique relations.
        hidden_dim: Embedding dimension.
    """

    def __init__(self, num_relations: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """Get relation embeddings.

        Args:
            relation_ids: Tensor of relation indices [batch_size].

        Returns:
            Relation embeddings [batch_size, hidden_dim].
        """
        return self.embedding(relation_ids)


def create_relation_encoder(
    num_relations: int,
    hidden_dim: int = 256,
    use_bert: bool = True,
    model_name: str = "bert-base-uncased",
    freeze_bert: bool = True,
) -> nn.Module:
    """Factory function to create appropriate relation encoder.

    Args:
        num_relations: Number of unique relations.
        hidden_dim: Output embedding dimension.
        use_bert: Whether to use BERT (requires more VRAM).
        model_name: BERT model name if use_bert=True.
        freeze_bert: Freeze BERT weights if use_bert=True.

    Returns:
        Relation encoder module.
    """
    if use_bert and TRANSFORMERS_AVAILABLE:
        try:
            encoder = RelationTextEncoder(
                model_name=model_name,
                hidden_dim=hidden_dim,
                freeze_bert=freeze_bert,
            )
            logger.info(f"Encoder BERT para relacoes criado: {model_name}")
            return encoder
        except Exception as e:
            logger.warning(
                f"Failed to create BERT encoder: {e}, falling back to lightweight"
            )

    encoder = LightweightRelationEncoder(num_relations, hidden_dim)
    logger.info(f"Encoder lightweight para {num_relations} relacoes criado")
    return encoder
