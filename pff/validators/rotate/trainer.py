"""RotatE Trainer Module.

Implements the training pipeline for RotatE models using Template Method pattern.
Extends BaseTrainer with RotatE-specific training, validation, and loss computation.

Design Patterns Applied:
    - **Template Method:** Extends BaseTrainer with RotatE-specific hooks.
    - **Strategy Pattern:** Uses RotatEStrategy for model operations.
    - **Observer Pattern:** Notifies TrainingObserver of training events.
    - **Dependency Injection:** Accepts model, optimizer, and observers.

Mathematical Foundation:
    RotatE loss with self-adversarial negative sampling:
        L = -log σ(γ - d_r(h,t)) - Σ p(h',r,t') log σ(d_r(h',t') - γ)
    
    Where:
        - d_r(h,t) = ||h ∘ r - t|| is the scoring function
        - p(h',r,t') is the self-adversarial weight
        - γ is the fixed margin

Example:
    >>> from pff.validators.rotate.trainer import RotatETrainer
    >>> from pff.validators.rotate.core import RotatEModel, RotatEDataset
    >>> trainer = RotatETrainer(model, config)
    >>> metrics = trainer.train(train_dataset, val_dataset, num_epochs=100)

Author: PFF Team
Date: 2025-11-25
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pff.utils import FileManager, logger
from pff.utils.ml.base_trainer import BaseTrainer, TrainerConfig
from pff.utils.performance.training_observer import (
    TrainingEvent,
    TrainingObserver,
    NullObserver,
)
from pff.validators.rotate.core import RotatEModel, RotatEDataset
from pff.validators.rotate.config import RotatEConfig


@dataclass
class RotatETrainerConfig(TrainerConfig):
    """Configuration for RotatE trainer.
    
    Extends TrainerConfig with RotatE-specific parameters.
    
    Attributes:
        gamma: Fixed margin for scoring (RotatE uses fixed gamma).
        adversarial_temperature: Temperature for self-adversarial sampling.
        regularization_weight: L2 regularization weight on embeddings.
        gradient_clip_val: Maximum gradient norm for clipping.
        use_self_adversarial: Enable self-adversarial negative sampling.
        warmup_steps: Number of warmup steps for learning rate scheduler.
        log_every: Log training metrics every N batches.
    """
    
    gamma: float = 12.0
    adversarial_temperature: float = 1.0
    regularization_weight: float = 0.0
    gradient_clip_val: float = 1.0
    use_self_adversarial: bool = True
    warmup_steps: int = 1000
    log_every: int = 100


class RotatETrainer(BaseTrainer):
    """Trainer for RotatE knowledge graph embedding model.
    
    Design Pattern: Template Method
        Extends BaseTrainer with RotatE-specific implementation of:
        - _setup_model(): Initialize RotatE model and optimizer
        - _train_epoch(): Single epoch training with self-adversarial loss
        - _validate(): Compute ranking metrics (MRR, Hits@k)
        - _compute_loss(): Self-adversarial negative sampling loss
    
    The trainer handles:
        1. Complex embedding management (real + imaginary parts)
        2. Self-adversarial negative sampling with temperature
        3. Gradient clipping for stable training
        4. Learning rate warmup and scheduling
        5. Metric computation (MRR, Hits@1/3/10)
    
    Attributes:
        model: RotatEModel instance.
        rotate_config: RotatE-specific configuration.
        scaler: GradScaler for automatic mixed precision.
        global_step: Global training step counter.
    
    Example:
        >>> model = RotatEModel(num_entities=5000, num_relations=50, ...)
        >>> config = RotatETrainerConfig(num_epochs=100, gamma=12.0)
        >>> trainer = RotatETrainer(model, config)
        >>> metrics = trainer.train(train_dataset, val_dataset)
    """
    
    def __init__(
        self,
        model: RotatEModel,
        config: RotatETrainerConfig | None = None,
        rotate_config: RotatEConfig | None = None,
        observer: TrainingObserver | None = None,
        file_manager: FileManager | None = None,
    ) -> None:
        """Initialize RotatE trainer.
        
        Args:
            model: RotatEModel to train.
            config: Trainer configuration.
            rotate_config: RotatE model configuration.
            observer: Training observer for events.
            file_manager: FileManager for I/O.
        """
        super().__init__(config or RotatETrainerConfig(), observer, file_manager)
        
        self.model = model
        self.rotate_config = rotate_config or RotatEConfig()
        self.global_step = 0
        
        # AMP scaler for mixed precision
        self.scaler = torch.amp.GradScaler(
            device=str(self.device),
            enabled=self.config.use_amp and self.device.type == "cuda"
        )
        
        # Move model to device
        self.model.to(self.device)
        
        logger.info(
            f"RotatE Trainer inicializado: {self.model.num_entities:,} entidades, "
            f"{self.model.num_relations} relacoes, device={self.device}"
        )
    
    def setup(self, train_data: Any, val_data: Any | None = None) -> None:
        """Setup training components.
        
        Initializes optimizer and learning rate scheduler with warmup.
        
        Args:
            train_data: Training dataset.
            val_data: Optional validation dataset.
        """
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
        )
        
        # Create scheduler with warmup
        total_steps = self.config.num_epochs * len(train_data) // self.config.batch_size
        warmup_steps = min(self.config.extra.get("warmup_steps", 1000), total_steps // 10)
        
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Notify observer with event
        self.observer.on_event(TrainingEvent(
            event_type="training_start",
            epoch=0,
            metrics={
                "model": "RotatE",
                "num_entities": self.model.num_entities,
                "num_relations": self.model.num_relations,
                "embedding_dim": self.model.embedding_dim,
                "gamma": self.config.extra.get("gamma", self.rotate_config.gamma),
                "device": str(self.device),
            },
        ))
    
    def _setup_model(self, train_data: Any) -> None:
        """Initialize model for training.
        
        Implements the abstract method from BaseTrainer.
        For RotatETrainer, the model is already provided in __init__,
        so this method just ensures it's on the correct device.
        
        Args:
            train_data: Training data (used for any data-dependent setup).
        """
        # Model is already set in __init__, just ensure device placement
        if self.model is not None:
            self.model.to(self.device)
        else:
            raise RuntimeError("RotatE model must be provided in __init__")
    
    def _train_epoch(self, train_data: Any, epoch: int) -> dict[str, float]:
        """Train for one epoch.
        
        Implements RotatE training with self-adversarial negative sampling.
        
        Args:
            train_data: Training dataset (RotatEDataset).
            epoch: Current epoch number.
            
        Returns:
            Dictionary with epoch metrics (loss, regularization, etc.).
        """
        self.model.train()
        
        # Create dataloader
        dataloader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )
        
        total_loss = 0.0
        total_reg_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Unpack batch: positive triples and negative samples
            pos_triples = batch["positive"].to(self.device)
            neg_heads = batch["neg_heads"].to(self.device)
            neg_tails = batch["neg_tails"].to(self.device)
            
            # Forward with AMP
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.config.use_amp and self.device.type == "cuda"
            ):
                loss, reg_loss = self._compute_loss(pos_triples, neg_heads, neg_tails)
                total_batch_loss = loss + reg_loss
            
            # Backward
            self.scaler.scale(total_batch_loss).backward()
            
            # Gradient clipping
            if self.config.extra.get("gradient_clip_val", 1.0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.extra.get("gradient_clip_val", 1.0)
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log progress
            log_every = self.config.extra.get("log_every", 100)
            if batch_idx % log_every == 0 and batch_idx > 0:
                avg_loss = total_loss / num_batches
                logger.debug(
                    f"Epoch {epoch} batch {batch_idx}/{len(dataloader)}: "
                    f"loss={avg_loss:.4f}"
                )
        
        metrics = {
            "loss": total_loss / max(num_batches, 1),
            "reg_loss": total_reg_loss / max(num_batches, 1),
            "lr": self.optimizer.param_groups[0]["lr"],
            "batches": num_batches,
        }
        
        return metrics
    
    def _compute_loss(
        self,
        pos_triples: torch.Tensor,
        neg_heads: torch.Tensor,
        neg_tails: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RotatE loss with self-adversarial negative sampling.
        
        Loss function:
            L = -log σ(γ - d_r(h,t)) - Σ p_i * log σ(d_r(h'_i,t'_i) - γ)
        
        Where p_i is the self-adversarial weight computed as softmax of scores.
        
        Args:
            pos_triples: Positive triples [batch, 3].
            neg_heads: Negative head samples [batch, num_neg].
            neg_tails: Negative tail samples [batch, num_neg].
            
        Returns:
            Tuple of (main_loss, regularization_loss).
        """
        batch_size = pos_triples.size(0)
        gamma = self.config.extra.get("gamma", self.rotate_config.gamma)
        adv_temp = self.config.extra.get("adversarial_temperature", 1.0)
        
        h_idx = pos_triples[:, 0]
        r_idx = pos_triples[:, 1]
        t_idx = pos_triples[:, 2]
        
        # Positive scores
        pos_scores = self.model.score_triples_batch(pos_triples)
        pos_loss = -torch.nn.functional.logsigmoid(gamma - pos_scores)
        
        # Negative head scores
        neg_h_triples = torch.stack([
            neg_heads.flatten(),
            r_idx.unsqueeze(1).expand(-1, neg_heads.size(1)).flatten(),
            t_idx.unsqueeze(1).expand(-1, neg_heads.size(1)).flatten(),
        ], dim=1)
        neg_h_scores = self.model.score_triples_batch(neg_h_triples)
        neg_h_scores = neg_h_scores.view(batch_size, -1)
        
        # Negative tail scores
        neg_t_triples = torch.stack([
            h_idx.unsqueeze(1).expand(-1, neg_tails.size(1)).flatten(),
            r_idx.unsqueeze(1).expand(-1, neg_tails.size(1)).flatten(),
            neg_tails.flatten(),
        ], dim=1)
        neg_t_scores = self.model.score_triples_batch(neg_t_triples)
        neg_t_scores = neg_t_scores.view(batch_size, -1)
        
        # Self-adversarial weights
        use_self_adv = self.config.extra.get("use_self_adversarial", True)
        if use_self_adv:
            neg_h_weights = torch.softmax(neg_h_scores * adv_temp, dim=1).detach()
            neg_t_weights = torch.softmax(neg_t_scores * adv_temp, dim=1).detach()
        else:
            neg_h_weights = torch.ones_like(neg_h_scores) / neg_h_scores.size(1)
            neg_t_weights = torch.ones_like(neg_t_scores) / neg_t_scores.size(1)
        
        # Negative losses
        neg_h_loss = (neg_h_weights * -torch.nn.functional.logsigmoid(neg_h_scores - gamma)).sum(dim=1)
        neg_t_loss = (neg_t_weights * -torch.nn.functional.logsigmoid(neg_t_scores - gamma)).sum(dim=1)
        
        # Total loss
        main_loss = (pos_loss + (neg_h_loss + neg_t_loss) / 2).mean()
        
        # Regularization
        reg_weight = self.config.extra.get("regularization_weight", 0.0)
        if reg_weight > 0:
            reg_loss = self.model.regularization_loss() * reg_weight
        else:
            reg_loss = torch.tensor(0.0, device=self.device)
        
        return main_loss, reg_loss
    
    def _validate(self, val_data: Any) -> dict[str, float]:
        """Validate model and compute ranking metrics.
        
        Computes filtered ranking metrics:
            - MRR (Mean Reciprocal Rank)
            - Hits@1, Hits@3, Hits@10
        
        Args:
            val_data: Validation dataset.
            
        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        
        ranks = []
        
        with torch.no_grad():
            dataloader = DataLoader(
                val_data,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
            )
            
            for batch in dataloader:
                pos_triples = batch["positive"].to(self.device)
                
                for triple in pos_triples:
                    h, r, t = triple.tolist()
                    
                    # Tail prediction
                    all_tails = torch.arange(
                        self.model.num_entities,
                        device=self.device
                    )
                    test_triples = torch.stack([
                        torch.full((self.model.num_entities,), h, device=self.device),
                        torch.full((self.model.num_entities,), r, device=self.device),
                        all_tails,
                    ], dim=1)
                    
                    scores = self.model.score_triples_batch(test_triples)
                    
                    # Get rank of true tail (lower score = better)
                    target_score = scores[t]
                    rank = (scores <= target_score).sum().item()
                    ranks.append(rank)
        
        ranks = np.array(ranks)
        
        metrics = {
            "mrr": float(np.mean(1.0 / ranks)),
            "hits@1": float(np.mean(ranks <= 1)),
            "hits@3": float(np.mean(ranks <= 3)),
            "hits@10": float(np.mean(ranks <= 10)),
            "mean_rank": float(np.mean(ranks)),
        }
        
        return metrics
    
    def _on_epoch_start(self, epoch: int) -> None:
        """Hook called at epoch start."""
        self.observer.on_event(TrainingEvent(
            event_type="epoch_start",
            epoch=epoch,
            data={"model": "RotatE"},
        ))
    
    def _on_epoch_end(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """Hook called at epoch end."""
        all_metrics = {**train_metrics, **val_metrics}
        
        self.observer.on_event(TrainingEvent(
            event_type="epoch_end",
            epoch=epoch,
            metrics=all_metrics,
        ))
        
        logger.info(
            f"Epoch {epoch}: loss={train_metrics.get('loss', 0):.4f}, "
            f"MRR={val_metrics.get('mrr', 0):.4f}, "
            f"Hits@10={val_metrics.get('hits@10', 0):.4f}"
        )
        
        self.training_history.append({
            "epoch": epoch,
            **all_metrics,
        })
    
    def _maybe_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Save checkpoint if metrics improved."""
        score = metrics.get("mrr", 0) + metrics.get("hits@10", 0) * 0.5
        
        if score > self.best_score:
            self.best_score = score
            self.patience_counter = 0
            
            checkpoint_path = self.config.checkpoint_dir / "best_model.pt"
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": {
                    "embedding_dim": self.model.embedding_dim,
                    "gamma": self.rotate_config.gamma,
                    "epsilon": self.rotate_config.epsilon,
                },
            }, checkpoint_path)
            
            logger.info(f"Checkpoint salvo: {checkpoint_path} (score={score:.4f})")
        else:
            self.patience_counter += 1
    
    def _check_early_stopping(self) -> bool:
        """Check if training should stop early.
        
        Returns:
            True if training should stop.
        """
        if self.patience_counter >= self.config.patience:
            logger.info(
                f"Early stopping: sem melhoria por {self.config.patience} epocas"
            )
            return True
        return False
    
    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        """Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file.
            
        Returns:
            Checkpoint dictionary with metadata.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.current_epoch = checkpoint.get("epoch", 0)
        
        logger.info(f"Checkpoint carregado: {path}")
        return checkpoint
