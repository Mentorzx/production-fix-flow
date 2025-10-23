from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Dataset

from pff import settings
from pff.utils import FileManager, logger
from pff.utils.global_interrupt_manager import get_interrupt_manager, should_stop
from pff.validators.kg.config import KGConfig
from pff.validators.kg.pipeline import MetricsCalculator

"""
TransE Core Implementation

This module implements the TransE (Translating Embeddings) model for knowledge graph
completion. TransE represents entities and relations as vectors in a continuous space
where relations are modeled as translations.
"""


class TransEModel(nn.Module):
    """
    TransE model implementation.

    TransE represents entities and relations as embeddings in the same space,
    where relations are interpreted as translations: h + r ≈ t
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 128,
        margin: float = 2.0,
        norm: int = 2,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize TransE model.

        Args:
            num_entities: Number of entities in the knowledge graph
            num_relations: Number of relations in the knowledge graph
            embedding_dim: Dimension of embeddings
            margin: Margin for the ranking loss
            norm: Norm to use for distance calculation (1 or 2)
            config: Additional configuration parameters
        """
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.norm = norm
        self.config = config or {}

        # Entity and relation embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Initialize embeddings
        self._initialize_embeddings()

        logger.info(
            f"✅ TransE Model initialized: "
            f"{num_entities:,} entities, {num_relations} relations, "
            f"dim={embedding_dim}"
        )

    def _initialize_embeddings(self) -> None:
        """Initialize embeddings using Xavier uniform initialization."""
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

        # Normalize entity embeddings
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )

    def forward(
        self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for TransE.

        Args:
            heads: Head entity indices [batch_size]
            relations: Relation indices [batch_size]
            tails: Tail entity indices [batch_size]

        Returns:
            Scores for the triples (negative distances)
        """
        # Clamp indices to valid range
        heads = torch.clamp(heads, 0, self.num_entities - 1)
        relations = torch.clamp(relations, 0, self.num_relations - 1)
        tails = torch.clamp(tails, 0, self.num_entities - 1)

        # Get embeddings
        head_emb = self.entity_embeddings(heads)
        rel_emb = self.relation_embeddings(relations)
        tail_emb = self.entity_embeddings(tails)

        # TransE scoring: ||h + r - t||
        scores = head_emb + rel_emb - tail_emb
        distances = torch.norm(scores, p=self.norm, dim=1)

        return -distances  # Negative because lower distance = higher score

    def score_triple(self, head_idx: int, rel_idx: int, tail_idx: int) -> float:
        """
        Score a single triple.

        Args:
            head_idx: Head entity index
            rel_idx: Relation index
            tail_idx: Tail entity index

        Returns:
            Score for the triple
        """
        with torch.no_grad():
            heads = torch.tensor([head_idx], dtype=torch.long)
            relations = torch.tensor([rel_idx], dtype=torch.long)
            tails = torch.tensor([tail_idx], dtype=torch.long)

            # Move to device if needed
            device = next(self.parameters()).device
            heads = heads.to(device)
            relations = relations.to(device)
            tails = tails.to(device)

            score = self.forward(heads, relations, tails)
            return score.item()

    def normalize_embeddings(self) -> None:
        """Normalize entity embeddings to unit length."""
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )


class TransEDataset(Dataset):
    """
    Dataset for TransE training with negative sampling.
    """

    def __init__(
        self,
        triples: np.ndarray,
        num_entities: int,
        num_negatives: int = 1,
        seed: int = 42,
    ):
        """
        Initialize dataset.

        Args:
            triples: Array of triples [num_triples, 3]
            num_entities: Total number of entities
            num_negatives: Number of negative samples per positive
            seed: Random seed
        """
        self.triples = torch.from_numpy(triples).long()
        self.num_entities = num_entities
        self.num_negatives = num_negatives
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        """Return number of triples."""
        return len(self.triples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a training sample with negative sampling.

        Returns:
            Dictionary with positive and negative samples
        """
        # Positive triple
        positive = self.triples[idx]
        head, rel, tail = positive

        # Generate negative samples
        negatives = []
        for _ in range(self.num_negatives):
            # Corrupt head or tail with 50% probability
            if self.rng.random() < 0.5:
                # Corrupt head
                neg_head = self.rng.randint(0, self.num_entities)
                negatives.append(torch.tensor([neg_head, rel, tail]))
            else:
                # Corrupt tail
                neg_tail = self.rng.randint(0, self.num_entities)
                negatives.append(torch.tensor([head, rel, neg_tail]))

        return {"positive": positive, "negatives": torch.stack(negatives)}


class TransEManager:
    """
    Manager for TransE model training, evaluation and inference.

    This class handles the complete lifecycle of TransE models including
    data preparation, training, checkpointing, and evaluation.
    """

    def __init__(self, transe_config_path: Path, kg_config_path: Path | None = None):
        """
        Initialize TransE manager.

        Args:
            transe_config_path: Path to TransE configuration file
            kg_config_path: Optional path to KG configuration file
        """
        self.file_manager = FileManager()
        self.transe_config_path = transe_config_path
        self.config = self.file_manager.read(transe_config_path)

        # KG configuration
        self.kg_config = KGConfig(kg_config_path) if kg_config_path else None

        # Device setup
        self.device = self._setup_device()
        self.seed = self.config["training"].get("seed", 42)
        self._set_seeds(self.seed)

        # Model and data
        self.model: TransEModel | None = None
        self.train_triples: np.ndarray | None = None
        self.val_triples: np.ndarray | None = None
        self.test_triples: np.ndarray | None = None

        # Mappings
        self.entity_to_idx: dict[str, int] = {}
        self.idx_to_entity: dict[int, str] = {}
        self.relation_to_idx: dict[str, int] = {}
        self.idx_to_relation: dict[int, str] = {}

        # Training components
        self.optimizer = None
        self.scheduler = None
        self.best_val_score = -float("inf")
        self.patience_counter = 0
        self.current_epoch = 0
        self.last_val_metrics = {"mrr": 0.0, "hits@1": 0.0, "hits@10": 0.0}

        # Checkpoint directory
        self.checkpoint_dir = Path(self.config["checkpointing"]["save_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metrics calculator
        self._metrics_calculator = None

        # Interrupt handling
        self.interrupt_manager = get_interrupt_manager()
        self._register_interrupt_handler()

        logger.info(f"✅ TransE Manager initialized with seed {self.seed}")

    def _setup_device(self) -> torch.device:
        """Setup and return the device for computation."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Usando GPU: {gpu_name}")
        else:
            device = torch.device("cpu")
            logger.info("Usando CPU")

        return device

    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _register_interrupt_handler(self) -> None:
        """Register cleanup callback for interrupt handling."""

        def cleanup_callback():
            logger.info("🧹 TransE: Iniciando limpeza por interrupção...")
            if self.model is not None:
                try:
                    emergency_path = self.checkpoint_dir / "emergency_checkpoint.pt"
                    self._save_checkpoint(emergency_path, is_best=False)
                    logger.info(f"💾 Checkpoint de emergência salvo: {emergency_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao salvar checkpoint de emergência: {e}")

        self.interrupt_manager.register_callback(cleanup_callback)
        logger.info("✅ TransEManager integrado ao GlobalInterruptManager")

    @property
    def metrics_calculator(self) -> MetricsCalculator:
        """Lazy loading of MetricsCalculator."""
        if self._metrics_calculator is None:
            self._metrics_calculator = MetricsCalculator(top_k=10)
        return self._metrics_calculator

    def _setup_data(self) -> None:
        """Load and prepare data for training."""
        if self.kg_config is None:
            raise ValueError("KGConfig required for data setup")

        logger.info("🔄 Configurando dados para TransE...")

        # Load mappings
        maps_path = settings.OUTPUTS_DIR / "transe"
        entity_map_path = maps_path / "transe_entity_map.parquet"
        relation_map_path = maps_path / "transe_relation_map.parquet"

        if not entity_map_path.exists() or not relation_map_path.exists():
            raise FileNotFoundError(
                f"Mapeamentos não encontrados em {maps_path}. "
                "Execute o pré-processamento primeiro."
            )

        # Load mappings
        from pff.validators.transe.mapping_utils import load_mappings

        (
            self.entity_to_idx,
            self.idx_to_entity,
            self.relation_to_idx,
            self.idx_to_relation,
        ) = load_mappings(entity_map_path, relation_map_path)

        # Load indexed data
        self.train_triples = np.load(maps_path / "train_indexed.npy")
        self.val_triples = np.load(maps_path / "valid_indexed.npy")

        # Try to load test data
        test_path = maps_path / "test_indexed.npy"
        if test_path.exists():
            self.test_triples = np.load(test_path)

        logger.info(
            f"✅ Dados carregados: "
            f"train={len(self.train_triples) if self.train_triples is not None else 0:,}, "
            f"val={len(self.val_triples) if self.val_triples is not None else 0:,}"
        )

    def _setup_model(self) -> None:
        """Initialize the TransE model."""
        if not self.entity_to_idx:
            self._setup_data()

        model_config = self.config["model"]

        self.model = TransEModel(
            num_entities=len(self.entity_to_idx),
            num_relations=len(self.relation_to_idx),
            embedding_dim=model_config["embedding_dim"],
            margin=model_config["margin"],
            norm=model_config["norm"],
            config=self.config,
        ).to(self.device)

        logger.info("✅ Modelo TransE criado e movido para dispositivo")

    def _setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        if self.model is None:
            raise RuntimeError("Model must be initialized before optimizer")

        train_config = self.config["training"]
        optimizer_cfg = train_config["optimizer"]
        if isinstance(optimizer_cfg, dict):
            optimizer_type = optimizer_cfg.get("type", "adam")
            optimizer_params = optimizer_cfg.get("params", {})
        else:
            optimizer_type = str(optimizer_cfg).lower()
            optimizer_params = {}

        # Create optimizer
        if optimizer_type == "adam":
            self.optimizer = Adam(self.model.parameters(), **optimizer_params)
        elif optimizer_type == "adamw":
            self.optimizer = AdamW(self.model.parameters(), **optimizer_params)
        elif optimizer_type == "sgd":
            self.optimizer = SGD(self.model.parameters(), **optimizer_params)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Create scheduler
        scheduler_config = train_config.get("scheduler", {})
        if isinstance(scheduler_config, dict):
            scheduler_type = scheduler_config.get("type", "none")
            scheduler_params = scheduler_config.get("params", {})
        else:
            scheduler_type = str(scheduler_config).lower()
            scheduler_params = {}

        if scheduler_type == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="max", **scheduler_params
            )
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=train_config["epochs"],
                **scheduler_params,
            )
        elif scheduler_type == "step":
            self.scheduler = StepLR(
                self.optimizer, **scheduler_params
            )

        logger.info(
            f"✅ Otimizador {optimizer_type} e scheduler {scheduler_type} configurados"
        )

    def train(
        self,
        train_triples: np.ndarray | None = None,
        val_triples: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Train the TransE model.

        Args:
            train_triples: Optional training triples (uses loaded data if None)
            val_triples: Optional validation triples (uses loaded data if None)

        Returns:
            Dictionary with training statistics
        """
        # Setup model if needed
        if self.model is None:
            logger.info("🔧 Inicializando modelo...")
            self._setup_model()
        if self.model is None:
            raise RuntimeError(
                "TransE model is not initialized. Please check model setup."
            )

        # Use provided data or loaded data
        if train_triples is None:
            if self.train_triples is None:
                self._setup_data()
            train_triples = self.train_triples

        if train_triples is None:
            raise ValueError(
                "train_triples cannot be None when initializing TransEDataset."
            )

        if val_triples is None:
            val_triples = self.val_triples

        # Training configuration
        train_config = self.config["training"]
        num_epochs = train_config["epochs"]
        batch_size = train_config["batch_size"]

        # Check for interruption
        if should_stop():
            logger.warning("🛑 Treinamento cancelado antes de iniciar")
            return {"status": "cancelled"}

        # Create dataset and dataloader
        dataset = TransEDataset(
            train_triples,
            num_entities=self.model.num_entities,
            num_negatives=train_config.get("num_negatives", 1),
            seed=self.seed,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=train_config.get("num_workers", 0),
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # Setup optimizer
        self._setup_optimizer()

        # Load checkpoint if exists
        _checkpoint_loaded = self._load_checkpoint()

        # Training statistics
        training_stats = {
            "epochs_trained": 0,
            "best_epoch": 0,
            "best_val_mrr": self.best_val_score,
            "training_time": 0.0,
            "final_metrics": {},
        }

        # Start MLflow run if configured
        if self.config.get("mlflow", {}).get("enabled", False):
            self._start_mlflow_run()

        logger.info("🚀 Iniciando treinamento do TransE...")
        logger.info(f"   Épocas: {self.current_epoch} → {num_epochs}")
        logger.info(f"   Batch size: {batch_size}")
        if self.optimizer is not None:
            logger.info(f"   Learning rate: {self.optimizer.param_groups[0]['lr']}")
        else:
            logger.info("   Learning rate: optimizer not initialized")

        start_time = time.time()

        # Training loop
        for epoch in range(self.current_epoch, num_epochs):
            # Check for interruption
            if should_stop():
                logger.warning(f"🛑 Treinamento interrompido na época {epoch}")
                self._save_checkpoint(
                    self.checkpoint_dir / "interrupted_checkpoint.pt", is_best=False
                )
                break

            self.current_epoch = epoch

            # Train one epoch
            epoch_loss = self._train_epoch(dataloader, epoch)

            # Validation
            if (
                val_triples is not None
                and epoch % train_config.get("validate_every", 5) == 0
            ):
                val_metrics = self._validate(val_triples)
                self.last_val_metrics = val_metrics

                # Log metrics
                logger.info(
                    f"Época {epoch}: Loss = {epoch_loss:.4f}, "
                    f"Val MRR = {val_metrics['mrr']:.4f}, "
                    f"Hits@1 = {val_metrics.get('hits@1', 0.0):.4f}, "
                    f"Hits@10 = {val_metrics.get('hits@10', 0.0):.4f}"
                )

                # Check for improvement
                if val_metrics["mrr"] > self.best_val_score:
                    self.best_val_score = val_metrics["mrr"]
                    self.patience_counter = 0
                    training_stats["best_epoch"] = epoch
                    training_stats["best_val_mrr"] = self.best_val_score

                    # Save best model
                    self._save_checkpoint(
                        self.checkpoint_dir / "best_model.pt", is_best=True
                    )
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= train_config.get("patience", 10):
                    logger.info(f"🛑 Early stopping triggered at epoch {epoch}")
                    break

                # Learning rate scheduling
                if self.scheduler and isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["mrr"])
            else:
                logger.info(f"Época {epoch}: Loss = {epoch_loss:.4f}")

            # Step scheduler if not ReduceLROnPlateau
            if self.scheduler and not isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step()

            training_stats["epochs_trained"] = epoch + 1

        # Training completed
        training_time = time.time() - start_time
        training_stats["training_time"] = training_time
        training_stats["final_metrics"] = self.last_val_metrics

        logger.success(
            f"✅ Treinamento concluído em {training_time:.1f}s "
            f"({training_stats['epochs_trained']} épocas)"
        )

        # End MLflow run
        if mlflow.active_run():
            mlflow.end_run()

        return training_stats

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train one epoch."""
        if self.model is None:
            raise RuntimeError(
                "TransE model is not initialized. Please check model setup before training."
            )
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        if self.model is None:
            raise RuntimeError(
                "TransE model is not initialized. Please check model setup before training."
            )
        for batch in dataloader:
            # Get positive and negative samples
            positives = batch["positive"].to(self.device)
            negatives = batch["negatives"].to(self.device)

            # Forward pass for positives
            if self.model is not None:
                pos_scores = self.model(
                    positives[:, 0],  # heads
                    positives[:, 1],  # relations
                    positives[:, 2],  # tails
                ) # type: ignore
            else:
                raise RuntimeError(
                    "TransE model is not initialized. Please check model setup before training."
                )

            # Forward pass for negatives
            batch_size, num_neg, _ = negatives.shape
            neg_scores = []

            for i in range(num_neg):
                neg_batch = negatives[:, i, :]
                neg_score = self.model(
                    neg_batch[:, 0], neg_batch[:, 1], neg_batch[:, 2]
                )
                neg_scores.append(neg_score)

            neg_scores = torch.stack(neg_scores, dim=1)

            # Margin ranking loss
            margin = self.config["model"]["margin"]
            losses = torch.relu(margin - pos_scores.unsqueeze(1) + neg_scores)
            loss = losses.mean()

            # Backward pass
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            else:
                raise RuntimeError(
                    "Optimizer is not initialized. Please check optimizer setup before training."
                )
            loss.backward()

            # Gradient clipping
            max_grad_norm = self.config["training"].get("max_grad_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            if self.optimizer is not None:
                self.optimizer.step()
            else:
                raise RuntimeError(
                    "Optimizer is not initialized. Please check optimizer setup before training."
                )

            # Normalize embeddings
            if self.config["training"].get("normalize_embeddings", True):
                self.model.normalize_embeddings()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Log to MLflow
        if mlflow.active_run():
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        return avg_loss

    def _validate(self, val_triples: np.ndarray) -> dict[str, float]:
        """Validate model on validation set."""
        if self.model is None:
            raise RuntimeError(
                "TransE model is not initialized. Please check model setup before validation."
            )
        self.model.eval()

        mrr_sum = 0.0
        hits_at_1 = 0
        hits_at_10 = 0

        with torch.no_grad():
            for i in range(len(val_triples)):
                head, rel, tail = val_triples[i]

                # Get all entity scores for this (head, relation, ?)
                all_entities = torch.arange(self.model.num_entities).to(self.device)
                heads = torch.full_like(all_entities, head)
                relations = torch.full_like(all_entities, rel)

                scores = self.model(heads, relations, all_entities)

                # Rank entities
                sorted_indices = torch.argsort(scores, descending=True)
                rank = (sorted_indices == tail).nonzero(as_tuple=True)[0].item() + 1

                # Update metrics
                mrr_sum += 1.0 / rank
                if rank == 1:
                    hits_at_1 += 1
                if rank <= 10:
                    hits_at_10 += 1

        num_samples = len(val_triples)
        metrics = {
            "mrr": mrr_sum / num_samples,
            "hits@1": hits_at_1 / num_samples,
            "hits@10": hits_at_10 / num_samples,
        }

        # Log to MLflow
        if mlflow.active_run():
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"val_{metric_name}", value, step=self.current_epoch)

        return metrics

    def _save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict()
            if self.model is not None
            else None,
            "optimizer_state_dict": self.optimizer.state_dict()
            if self.optimizer
            else None,
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "best_val_score": self.best_val_score,
            "config": self.config,
            "entity_to_idx": self.entity_to_idx,
            "relation_to_idx": self.relation_to_idx,
            "last_val_metrics": self.last_val_metrics,
        }

        torch.save(checkpoint, path)

        if is_best:
            logger.info(f"💾 Melhor modelo salvo: {path}")
        else:
            logger.info(f"💾 Checkpoint salvo: {path}")

    def _load_checkpoint(self, path: Path | None = None) -> bool:
        """Load model checkpoint."""
        if path is None:
            path = self.checkpoint_dir / "best_model.pt"

        if not path.exists():
            return False

        logger.info(f"🔄 Carregando checkpoint: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Ensure model is initialized before loading state dict
        if self.model is None:
            self._setup_model()
        if self.model is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise RuntimeError(
                "TransE model could not be initialized before loading state dict."
            )

        # Load optimizer state
        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.best_val_score = checkpoint.get("best_val_score", -float("inf"))
        self.last_val_metrics = checkpoint.get("last_val_metrics", {})

        # Load mappings if available
        if "entity_to_idx" in checkpoint:
            self.entity_to_idx = checkpoint["entity_to_idx"]
            self.idx_to_entity = {v: k for k, v in self.entity_to_idx.items()}

        if "relation_to_idx" in checkpoint:
            self.relation_to_idx = checkpoint["relation_to_idx"]
            self.idx_to_relation = {v: k for k, v in self.relation_to_idx.items()}

        logger.success(f"✅ Checkpoint carregado (época {self.current_epoch})")
        return True

    def _load_best_model(self) -> bool:
        """Load the best model checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pt"
        if not best_path.exists():
            logger.warning(f"⚠️ Melhor modelo não encontrado em {best_path}")
            return False

        return self._load_checkpoint(best_path)

    def extract_embeddings_for_lightgbm(self) -> dict[str, np.ndarray]:
        """
        Extract entity embeddings for use with LightGBM.

        Returns:
            Dictionary with entity embeddings
        """
        if self.model is None:
            logger.warning("⚠️ Modelo não encontrado, tentando carregar...")
            self._load_best_model()
            if self.model is None:
                raise RuntimeError(
                    "Modelo TransE não está treinado! Execute o treinamento primeiro."
                )

        logger.info("🔄 Extraindo embeddings para LightGBM...")

        with torch.no_grad():
            entity_embeddings = self.model.entity_embeddings.weight.cpu().numpy()

        # Save embeddings
        embeddings_path = settings.OUTPUTS_DIR / "transe" / "node_embeddings.pkl"
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings_dict = {"entity": entity_embeddings}
        self.file_manager.save(embeddings_dict, embeddings_path)
        logger.info(
            f"✅ Embeddings extraídos: "
            f"{entity_embeddings.shape} salvo em {embeddings_path}"
        )

        return embeddings_dict

    def _start_mlflow_run(self) -> None:
        """Start MLflow run for experiment tracking."""
        mlflow_config = self.config.get("mlflow", {})

        if mlflow_config.get("tracking_uri"):
            mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

        experiment_name = mlflow_config.get("experiment_name", "TransE_KGC")
        mlflow.set_experiment(experiment_name)

        # Start run with tags
        tags = {
            "model": "TransE",
            "embedding_dim": str(self.config["model"]["embedding_dim"]),
            "margin": str(self.config["model"]["margin"]),
            "optimizer": self.config["training"]["optimizer"]["type"],
        }

        mlflow.start_run(tags=tags)

        # Log parameters
        mlflow.log_params(
            {
                "num_entities": len(self.entity_to_idx),
                "num_relations": len(self.relation_to_idx),
                "embedding_dim": self.config["model"]["embedding_dim"],
                "margin": self.config["model"]["margin"],
                "norm": self.config["model"]["norm"],
                "batch_size": self.config["training"]["batch_size"],
                "learning_rate": self.config["training"]["optimizer"]["params"]["lr"],
                "epochs": self.config["training"]["epochs"],
            }
        )

        logger.info("📊 MLflow run iniciado")

    def generate_clean_consistent_splits(self) -> None:
        """
        Generate clean train/valid/test splits from optimized data.

        This method ensures no data leakage between splits and saves
        them in the appropriate format for TransE training.
        """
        logger.info("🔧 Gerando splits consistentes sem data leakage...")

        try:
            # Load optimized data
            if self.kg_config is None:
                raise ValueError("KGConfig required for split generation")

            base_path = self.kg_config.graph_directory
            train_opt_path = base_path / "train_optimized.parquet"

            if not train_opt_path.exists():
                logger.warning("Dados otimizados não encontrados, usando dados brutos")
                train_opt_path = base_path / "train.parquet"

            # Load data
            df = pl.read_parquet(train_opt_path)

            # Remove duplicates
            df_unique = df.unique(subset=["s", "p", "o"])
            duplicate_count = len(df) - len(df_unique)

            if duplicate_count > 0:
                logger.warning(f"⚠️ {duplicate_count} triplas duplicadas removidas!")

            logger.info(f"📊 Dados limpos: {len(df_unique)} triplas")

            # Convert to pandas for sklearn
            df_pd = df_unique.to_pandas()

            # Create splits
            train_val, test = train_test_split(
                df_pd, test_size=0.15, random_state=42, shuffle=True
            )

            train, val = train_test_split(
                train_val, test_size=0.15, random_state=42, shuffle=True
            )

            logger.info("✅ Splits criados:")
            logger.info(f"   Treino: {len(train)} triplas")
            logger.info(f"   Validação: {len(val)} triplas")
            logger.info(f"   Teste: {len(test)} triplas")

            # Verify no overlap
            train_set = set(
                train["s"].astype(str)
                + "|"
                + train["p"].astype(str)
                + "|"
                + train["o"].astype(str)
            )
            val_set = set(
                val["s"].astype(str)
                + "|"
                + val["p"].astype(str)
                + "|"
                + val["o"].astype(str)
            )
            test_set = set(
                test["s"].astype(str)
                + "|"
                + test["p"].astype(str)
                + "|"
                + test["o"].astype(str)
            )

            overlap_stats = {
                "train_val": len(train_set & val_set),
                "train_test": len(train_set & test_set),
                "val_test": len(val_set & test_set),
            }

            logger.info(f"🔍 Verificação de vazamento: {overlap_stats}")

            if any(overlap_stats.values()):
                raise RuntimeError(f"🚨 DATA LEAKAGE: {overlap_stats}")

            logger.success("✅ VERIFICAÇÃO PASSOU: Splits completamente limpos!")

            # Save splits
            for name, data in [("train", train), ("valid", val), ("test", test)]:
                path = base_path / f"{name}_optimized.parquet"
                pl.from_pandas(data).write_parquet(path)

            logger.success("✅ Splits consistentes salvos")

        except Exception as e:
            logger.error(f"❌ Erro ao gerar splits: {e}")
            raise


def compare_mlflow_experiments(
    experiment_name: str, metric: str = "val_mrr"
) -> pl.DataFrame | None:
    """
    Compare MLflow experiments and return sorted results.

    Args:
        experiment_name: Name of the MLflow experiment
        metric: Metric to sort by

    Returns:
        DataFrame with experiment results or None
    """
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if not experiment:
            logger.warning(f"Experimento '{experiment_name}' não encontrado")
            return None

        # Get all runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
        )

        if not runs:
            logger.warning("Nenhum run encontrado")
            return None

        # Extract data
        data = []
        for run in runs:
            row = {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or "unnamed",
                "status": run.info.status,
                "start_time": run.info.start_time,
                "duration_min": (run.info.end_time - run.info.start_time) / 60000
                if run.info.end_time
                else None,
            }

            # Add metrics
            for metric_key, metric_value in run.data.metrics.items():
                row[metric_key] = metric_value

            # Add key parameters
            for param_key in ["embedding_dim", "margin", "batch_size", "learning_rate"]:
                if param_key in run.data.params:
                    row[param_key] = run.data.params[param_key]

            data.append(row)

        # Create DataFrame
        df = pl.DataFrame(data)

        return df

    except Exception as e:
        logger.error(f"Erro ao comparar experimentos: {e}")
        return None
