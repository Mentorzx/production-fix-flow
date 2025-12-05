"""RotatE Manager Implementation.

This module provides the RotatEManager class for orchestrating RotatE model
training, evaluation, and inference. It follows the same structure as
TransEManager for consistency.

Design Patterns Applied:
    - **Template Method:** Training follows fixed flow with customizable steps.
    - **Strategy Pattern:** Uses RotatEStrategy for model-specific operations.
    - **Observer Pattern:** Metrics observed via TrainingObserver/CompositeObserver.
    - **Factory Pattern:** Model creation via ModelFactory.
    - **SRP Components:** CheckpointManager, DataLoader, MetricsReporter.

Author: PFF Team
Date: 2025-11-25
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from pff import settings
from pff.utils import FileManager, logger
from pff.utils.global_interrupt_manager import get_interrupt_manager, should_stop
from pff.utils.determinism import set_global_seed
from pff.utils.performance.performance import apply_sota_optimizations
from pff.utils.performance.observability import ObservabilityManager
from pff.utils.performance.training_observer import (
    TrainingObserver,
    ConsoleObserver,
    CompositeObserver,
)
from pff.utils.system.hardware_detector import HardwareDetector
from pff.utils.ml.model_factory import ModelFactory, ModelType
from pff.validators.kg.config import KGConfig
from pff.validators.kg.pipeline import MetricsCalculator
from pff.validators.rotate.core import RotatEModel, RotatEDataset
from pff.validators.rotate.config import RotatEConfig
from pff.validators.rotate.checkpoint_manager import RotatECheckpointManager
from pff.validators.rotate.metrics_reporter import RotatEMetricsReporter
from pff.validators.rotate.contrastive import ContrastiveLossFactory, LossType


# Global flag to track CUDA availability after first initialization attempt.
# Once CUDA fails during a process lifetime, we should never try again to avoid segfaults.
_CUDA_AVAILABLE: bool | None = None
_CUDA_DEVICE: torch.device | None = None


def _get_optimal_num_workers() -> int:
    """Get optimal number of DataLoader workers based on hardware detection.

    Uses HardwareDetector from utils layer to determine CPU cores.
    Returns a conservative value to avoid memory issues.

    Returns:
        Optimal number of workers (1-4 range, capped at physical cores - 1).
    """
    try:
        profile = HardwareDetector.detect()
        # Use at most 4 workers, or physical_cores - 1 (leave 1 for main thread)
        max_workers = max(1, min(4, profile.cpu_cores - 1))
        return max_workers
    except Exception as exc:  # noqa: BLE001 - defensive fallback
        logger.debug(f"Worker detection fallback: {exc}")
        return 2


class RotatEManager:
    """Manager for RotatE model training, evaluation and inference.

    This class handles the complete lifecycle of RotatE models including
    data preparation, training, checkpointing, and evaluation.

    Design Pattern: Template Method
        - Fixed training flow: setup → train epochs → validate → checkpoint
        - Customizable steps via _train_epoch(), _validate(), etc.

    Attributes:
        config: RotatE configuration dictionary.
        rotate_config: Parsed RotatEConfig dataclass.
        model: RotatE model instance.
        device: Computation device (CUDA/CPU).
        entity_to_idx: Entity name to index mapping.
        relation_to_idx: Relation name to index mapping.
        checkpoint_manager: SRP component for checkpoint operations.
        metrics_reporter: SRP component for metrics computation/reporting.
        training_observer: Composite observer for training events.

    Example:
        >>> manager = RotatEManager(Path("config/models/rotate.yaml"))
        >>> manager.train()
        >>> scores = manager.model.score_triples_batch(triples)
    """

    def __init__(
        self,
        rotate_config_path: Path,
        kg_config_path: Path | None = None,
    ) -> None:
        """Initialize RotatE manager.

        Args:
            rotate_config_path: Path to RotatE configuration file.
            kg_config_path: Optional path to KG configuration file.
        """
        self.file_manager = FileManager()
        self.rotate_config_path = rotate_config_path
        self.config = self.file_manager.read(rotate_config_path)
        self.rotate_config = RotatEConfig.from_yaml(rotate_config_path)
        self.kg_config = KGConfig(kg_config_path) if kg_config_path else None
        training_cfg = self.config.get("training", {})
        self.use_self_adversarial = bool(
            training_cfg.get("self_adversarial_negative_sampling", True)
        )
        self.adversarial_temperature = float(
            training_cfg.get("adversarial_temperature", 1.0)
        )

        if self.config.get("training", {}).get("use_sota_optimizations", True):
            apply_sota_optimizations()

        self.obs_manager = ObservabilityManager(
            experiment_name="rotate_training",
            enable_debugging=self.config.get("observability", {}).get(
                "enable_debugging", False
            ),
            model_name="rotate",
        )
        self.device = self._setup_device()
        self.seed = training_cfg.get("seed", 42)
        self._set_seeds(self.seed)
        self.model: RotatEModel | None = None
        self.train_triples: np.ndarray | None = None
        self.val_triples: np.ndarray | None = None
        self.test_triples: np.ndarray | None = None
        self.entity_to_idx: dict[str, int] = {}
        self.idx_to_entity: dict[int, str] = {}
        self.relation_to_idx: dict[str, int] = {}
        self.idx_to_relation: dict[int, str] = {}
        self.optimizer = None
        self.scheduler = None
        self.best_val_score = -float("inf")
        self.patience_counter = 0
        self.current_epoch = 0
        self.last_val_metrics: dict[str, float] = {"mrr": 0.0, "hits@1": 0.0, "hits@10": 0.0}
        self.checkpoint_dir = Path(
            self.config.get("checkpointing", {}).get("save_dir", "checkpoints/rotate")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_calculator = None
        self.interrupt_manager = get_interrupt_manager()

        # SRP Components
        self.checkpoint_manager = RotatECheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            file_manager=self.file_manager,
            keep_top_k=self.config.get("checkpointing", {}).get("save_top_k", 3),
        )
        self.metrics_reporter = RotatEMetricsReporter(
            output_dir=self.checkpoint_dir,
            file_manager=self.file_manager,
        )

        # Observer Pattern: Setup composite observer for training events
        self.training_observer = CompositeObserver([
            ConsoleObserver(verbose=False, log_every_n_batches=100),
        ])
        self.metrics_reporter.add_observer(self.training_observer)

        # Contrastive loss (enabled via config)
        self._contrastive_loss = None
        contrastive_cfg = self.config.get("contrastive", {})
        if contrastive_cfg.get("enabled", False):
            loss_type_str = contrastive_cfg.get("loss_type", "kg_loss").upper()
            try:
                loss_type = LossType[loss_type_str]
                self._contrastive_loss = ContrastiveLossFactory.create(
                    loss_type=loss_type,
                    margin=contrastive_cfg.get("margin", 1.0),
                    temperature=contrastive_cfg.get("temperature", 0.07),
                )
                logger.info(f"Loss contrastiva habilitada: {loss_type_str}")
            except (KeyError, ValueError) as e:
                logger.warning(f"Invalid contrastive loss type, using default: {e}")

        self._register_interrupt_handler()

        logger.info("RotatEManager inicializado")
        logger.debug(f"RotatEManager details: seed={self.seed}, device={self.device}")

    def _setup_device(self) -> torch.device:
        """Setup and return the device for computation.
        
        Handles CUDA initialization errors gracefully by falling back to CPU.
        Uses global state to remember CUDA availability across trials to avoid
        segfaults from repeated CUDA init attempts after failure.
        """
        global _CUDA_AVAILABLE, _CUDA_DEVICE
        
        # If we already determined device in this process, reuse it
        if _CUDA_DEVICE is not None:
            device_type = "GPU CUDA" if _CUDA_DEVICE.type == "cuda" else (
                "Intel XPU" if _CUDA_DEVICE.type == "xpu" else "CPU"
            )
            logger.info(f"Dispositivo de treinamento: {device_type}")
            return _CUDA_DEVICE
        
        # Try Intel XPU first
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                _CUDA_DEVICE = torch.device("xpu")
                gpu_name = torch.xpu.get_device_name(0)
                logger.debug(f"Intel XPU detected: {gpu_name}")
                logger.info(f"Dispositivo de treinamento: Intel XPU")
                return _CUDA_DEVICE
        except Exception as e:
            logger.warning(f"Intel XPU detection failed: {e}")

        # Try CUDA - but only if we haven't already failed
        if _CUDA_AVAILABLE is None:
            try:
                if torch.cuda.is_available():
                    # Test actual CUDA initialization with a small tensor
                    test_tensor = torch.zeros(1, device="cuda")
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                    _CUDA_AVAILABLE = True
                    _CUDA_DEVICE = torch.device("cuda")
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.debug(f"CUDA device detected: {gpu_name}")
                    logger.info(f"Dispositivo de treinamento: GPU CUDA")
                    return _CUDA_DEVICE
                else:
                    _CUDA_AVAILABLE = False
            except (RuntimeError, AssertionError) as e:
                # Handle CUDA allocator config mismatch and other CUDA init errors
                # Mark CUDA as permanently unavailable for this process
                _CUDA_AVAILABLE = False
                # Use debug level - this is expected when CUDA allocator was configured after init
                logger.debug(f"CUDA initialization issue (falling back to CPU): {e}")
        elif _CUDA_AVAILABLE is False:
            logger.debug("CUDA skipped: previously failed initialization")

        # Fallback to CPU
        _CUDA_DEVICE = torch.device("cpu")
        logger.info("Dispositivo de treinamento: CPU")
        return _CUDA_DEVICE

    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        global _CUDA_AVAILABLE
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            # Only try CUDA seeds if CUDA is known to work
            if _CUDA_AVAILABLE is True and torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        except (RuntimeError, AssertionError) as e:
            logger.debug(f"CUDA seed setting skipped: {e}")

    def _register_interrupt_handler(self) -> None:
        """Register cleanup callback for interrupt handling."""

        def cleanup_callback():
            logger.info("Iniciando limpeza por interrupcao no RotatE...")
            if self.model is not None:
                try:
                    emergency_path = self.checkpoint_dir / "emergency_checkpoint.pt"
                    self._save_checkpoint(emergency_path, is_best=False)
                    logger.info(f"Checkpoint de emergencia salvo em: {emergency_path}")
                except Exception as e:
                    logger.warning(f"Error saving emergency checkpoint: {e}")

        self.interrupt_manager.register_callback(cleanup_callback)
        logger.info("RotatEManager registrado no gerenciador de interrupcoes")

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

        logger.info("Configurando dados para o RotatE...")

        # Try multiple paths in order of preference
        possible_paths = [
            settings.OUTPUTS_DIR / "rotate",
            settings.OUTPUTS_DIR / "pyclause",
            settings.OUTPUTS_DIR / "kg",
        ]
        if self.kg_config is not None:
            # Prefer explicit paths from KGConfig (e.g., outputs/kg/pyclause)
            possible_paths.insert(0, self.kg_config.pyclause_directory)
            possible_paths.insert(1, self.kg_config.graph_directory)

        maps_path = None
        entity_map_path = None
        relation_map_path = None

        for path in possible_paths:
            if not path.exists():
                continue
            
            # Try different naming conventions
            entity_candidates = [
                path / "rotate_entity_map.parquet",
                path / "entity_map.parquet",
            ]
            relation_candidates = [
                path / "rotate_relation_map.parquet",
                path / "relation_map.parquet",
            ]
            
            for ent_path in entity_candidates:
                if ent_path.exists():
                    entity_map_path = ent_path
                    break
            
            for rel_path in relation_candidates:
                if rel_path.exists():
                    relation_map_path = rel_path
                    break
            
            if entity_map_path and relation_map_path:
                maps_path = path
                break

        # Fallback: direct paths from KGConfig if provided
        if (not entity_map_path or not relation_map_path) and self.kg_config is not None:
            kg_entity_map = self.kg_config.get_entity_map_path()
            kg_relation_map = self.kg_config.get_relation_map_path()
            if kg_entity_map.exists() and kg_relation_map.exists():
                entity_map_path = kg_entity_map
                relation_map_path = kg_relation_map
                maps_path = kg_entity_map.parent
                logger.info(
                    f"Mapeamentos encontrados via KGConfig: {entity_map_path}, {relation_map_path}"
                )

        if not entity_map_path or not relation_map_path:
            raise FileNotFoundError(
                f"Mapeamentos nao encontrados em {possible_paths}. "
                "Execute o pre-processamento primeiro."
            )

        logger.info(f"Carregando mapeamentos de {maps_path}")

        # Load mappings directly using FileManager
        entity_df = self.file_manager.read(entity_map_path)
        relation_df = self.file_manager.read(relation_map_path)

        # Detect column naming convention and build mappings
        # Common conventions: (id, label), (idx, entity), (index, name)
        if "label" in entity_df.columns and "id" in entity_df.columns:
            self.entity_to_idx = dict(zip(entity_df["label"], entity_df["id"]))
            self.idx_to_entity = dict(zip(entity_df["id"], entity_df["label"]))
        elif "entity" in entity_df.columns and "idx" in entity_df.columns:
            self.entity_to_idx = dict(zip(entity_df["entity"], entity_df["idx"]))
            self.idx_to_entity = dict(zip(entity_df["idx"], entity_df["entity"]))
        else:
            # Assume first two columns are (idx, name)
            cols = entity_df.columns.tolist()
            self.entity_to_idx = dict(zip(entity_df[cols[1]], entity_df[cols[0]]))
            self.idx_to_entity = dict(zip(entity_df[cols[0]], entity_df[cols[1]]))

        if "label" in relation_df.columns and "id" in relation_df.columns:
            self.relation_to_idx = dict(zip(relation_df["label"], relation_df["id"]))
            self.idx_to_relation = dict(zip(relation_df["id"], relation_df["label"]))
        elif "relation" in relation_df.columns and "idx" in relation_df.columns:
            self.relation_to_idx = dict(zip(relation_df["relation"], relation_df["idx"]))
            self.idx_to_relation = dict(zip(relation_df["idx"], relation_df["relation"]))
        else:
            cols = relation_df.columns.tolist()
            self.relation_to_idx = dict(zip(relation_df[cols[1]], relation_df[cols[0]]))
            self.idx_to_relation = dict(zip(relation_df[cols[0]], relation_df[cols[1]]))

        # Load indexed triples
        train_path = maps_path / "train_indexed.npy"
        val_path = maps_path / "valid_indexed.npy"
        test_path = maps_path / "test_indexed.npy"

        if train_path.exists():
            self.train_triples = self.file_manager.read(train_path)
        else:
            # Try to load from parquet and convert
            train_parquet = maps_path / "train.homogenized.parquet"
            if train_parquet.exists():
                self.train_triples = self._convert_parquet_to_indexed(train_parquet)
            else:
                raise FileNotFoundError(f"Training data not found in {maps_path}")

        if val_path.exists():
            self.val_triples = self.file_manager.read(val_path)
        else:
            val_parquet = maps_path / "valid.homogenized.parquet"
            if val_parquet.exists():
                self.val_triples = self._convert_parquet_to_indexed(val_parquet)
            else:
                self.val_triples = None

        if test_path.exists():
            self.test_triples = self.file_manager.read(test_path)
        else:
            test_parquet = maps_path / "test.homogenized.parquet"
            if test_parquet.exists():
                self.test_triples = self._convert_parquet_to_indexed(test_parquet)
            else:
                self.test_triples = None

        logger.info(
            f"Dados carregados: "
            f"train={len(self.train_triples) if self.train_triples is not None else 0:,}, "
            f"val={len(self.val_triples) if self.val_triples is not None else 0:,}"
        )

    def _convert_parquet_to_indexed(self, parquet_path: Path) -> np.ndarray:
        """Convert parquet triples to indexed numpy array.
        
        Args:
            parquet_path: Path to parquet file with columns [s, p, o] or similar.
            
        Returns:
            Numpy array of shape [n_triples, 3] with indexed entities/relations.
        """
        df = self.file_manager.read(parquet_path)
        
        # Detect column names
        cols = df.columns
        if "s" in cols:
            head_col, rel_col, tail_col = "s", "p", "o"
        elif "head" in cols:
            head_col, rel_col, tail_col = "head", "relation", "tail"
        elif "subject" in cols:
            head_col, rel_col, tail_col = "subject", "predicate", "object"
        else:
            # Assume first 3 columns
            head_col, rel_col, tail_col = cols[0], cols[1], cols[2]
        
        # Convert to pandas for iteration or use polars directly
        try:
            # Try polars API first
            heads = df[head_col].to_list()
            rels = df[rel_col].to_list()
            tails = df[tail_col].to_list()
        except AttributeError:
            # Fallback for pandas
            heads = df[head_col].tolist()
            rels = df[rel_col].tolist()
            tails = df[tail_col].tolist()
        
        indexed = []
        for h, r, t in zip(heads, rels, tails):
            h_idx = self.entity_to_idx.get(str(h), 0)
            r_idx = self.relation_to_idx.get(str(r), 0)
            t_idx = self.entity_to_idx.get(str(t), 0)
            indexed.append([h_idx, r_idx, t_idx])
        
        return np.array(indexed, dtype=np.int64)

    def _setup_model(self) -> None:
        """Initialize the RotatE model using ModelFactory."""
        if not self.entity_to_idx:
            self._setup_data()

        model_config = self.config.get("model", {})
        model_type_raw = str(model_config.get("type", "dslfm")).lower()
        model_type = ModelType.ROTATE if model_type_raw == "rotate" else ModelType.DSLFM

        # Use ModelFactory for centralized model creation
        factory = ModelFactory(file_manager=self.file_manager)
        self.model = factory.create(
            model_type,
            num_entities=len(self.entity_to_idx),
            num_relations=len(self.relation_to_idx),
            embedding_dim=model_config.get("embedding_dim", 256),
            gamma=model_config.get("gamma", 12.0),
            config=self.rotate_config,
        )
        self.model = self.model.to(self.device)

        if getattr(torch, "compile", None) is not None:
            try:
                # max-autotune favors steady-state throughput at the cost of a longer warmup; dynamic=True keeps shape flexibility.
                self.model = torch.compile(self.model, mode="max-autotune", dynamic=True)
                logger.success("Modelo RotatE compilado com torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        logger.info(
            f"Modelo RotatE criado: entities={len(self.entity_to_idx):,}, "
            f"relations={len(self.relation_to_idx):,}, dim={model_config.get('embedding_dim', 256)}"
        )

    def _setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        if self.model is None:
            raise RuntimeError("Model must be initialized before optimizer")

        train_config = self.config.get("training", {})
        lr = train_config.get("learning_rate", 0.00005)
        weight_decay = train_config.get("weight_decay", 0.0)
        optimizer_type = str(train_config.get("optimizer", "adam")).lower()

        # SOTA: Use fused optimizer on CUDA for 10-15% speedup
        use_fused = self.device.type == "cuda" and hasattr(torch, "__version__")
        fused_kwargs = {"fused": True} if use_fused else {}

        if optimizer_type == "adam":
            self.optimizer = Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay, **fused_kwargs
            )
        elif optimizer_type == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay, **fused_kwargs
            )
        elif optimizer_type == "sgd":
            self.optimizer = SGD(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        scheduler_type = str(train_config.get("scheduler", "warmup_linear")).lower()

        if scheduler_type == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", patience=5)
        elif scheduler_type in ("cosine", "cosine_annealing"):
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=train_config.get("epochs", 200)
            )
        elif scheduler_type == "step":
            self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)
        else:
            self.scheduler = None

        fused_status = "com fused=True" if use_fused else "sem fused"
        logger.info(
            f"Otimizador {optimizer_type} ({fused_status}) e scheduler {scheduler_type} configurados"
        )

    def train(
        self,
        train_triples: np.ndarray | None = None,
        val_triples: np.ndarray | None = None,
        force_retrain: bool = False,
    ) -> dict[str, Any]:
        """Train the RotatE model.

        Args:
            train_triples: Optional training triples (uses loaded data if None).
            val_triples: Optional validation triples (uses loaded data if None).
            force_retrain: If True, ignore existing checkpoints and retrain from scratch.

        Returns:
            Dictionary with training statistics.
        """
        if self.model is None:
            self._setup_model()

        if self.model is None:
            raise RuntimeError("RotatE model is not initialized")

        # Check if training was already completed using CheckpointManager
        train_config = self.config.get("training", {})
        num_epochs = train_config.get("epochs", 200)
        
        if not force_retrain:
            is_complete, completion_info = self.checkpoint_manager.has_completed_training(num_epochs)
            
            if is_complete:
                completed_epochs = completion_info.get("epochs_trained", 0)
                best_val_mrr = completion_info.get("best_val_mrr", 0.0)
                
                logger.info(
                    f"Treinamento RotatE ja concluido "
                    f"({completed_epochs} epocas, MRR={best_val_mrr:.4f})"
                )
                self._load_checkpoint()
                return {
                    "status": "skipped",
                    "epochs_trained": completed_epochs,
                    "best_epoch": completion_info.get("best_epoch", 0),
                    "best_val_mrr": best_val_mrr,
                    "training_time": completion_info.get("training_time", 0.0),
                    "final_metrics": completion_info.get("final_metrics", {}),
                }

        if train_triples is None:
            if self.train_triples is None:
                self._setup_data()
            train_triples = self.train_triples

        if train_triples is None:
            raise ValueError("train_triples cannot be None")

        if val_triples is None:
            val_triples = self.val_triples

        batch_size = train_config.get("batch_size", 1024)

        if should_stop():
            logger.warning("Training cancelled before starting")
            return {"status": "cancelled"}

        dataset = RotatEDataset(
            train_triples,
            num_entities=self.model.num_entities,
            num_negatives=train_config.get("negative_samples", 256),
            seed=self.seed,
        )
        
        # Get optimal num_workers from config or hardware detection
        config_workers = train_config.get("num_workers")
        if config_workers is not None and config_workers > 0:
            num_workers = config_workers
        else:
            num_workers = _get_optimal_num_workers()
        # Ambiente restrito: evitar multiprocessing para DataLoader
        num_workers = 0
        
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=self.device.type == "cuda",
            )
        except PermissionError as exc:
            logger.warning(
                f"Permission denied for DataLoader workers ({exc}); using num_workers=0"
            )
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=self.device.type == "cuda",
            )

        self._setup_optimizer()

        checkpoint_loaded = self._load_checkpoint()
        initial_best_epoch = max(0, self.current_epoch - 1) if checkpoint_loaded else 0

        training_stats: dict[str, Any] = {
            "epochs_trained": 0,
            "best_epoch": initial_best_epoch,
            "best_val_mrr": self.best_val_score,
            "training_time": 0.0,
            "final_metrics": {},
        }

        if self.config.get("mlflow", {}).get("enabled", False):
            self._start_mlflow_run()

        logger.info(f"Iniciando treinamento RotatE: {num_epochs} epocas, {len(train_triples):,} triplas")
        logger.debug(f"Training config: batch_size={batch_size}, neg_samples={train_config.get('negative_samples', 256)}")
        if self.optimizer is not None:
            logger.debug(f"Optimizer: lr={self.optimizer.param_groups[0]['lr']}")

        start_time = time.time()
        validate_every = train_config.get("validate_every_n_epochs", 5)
        patience = train_config.get("early_stopping_patience", 30)
        memory_cleanup_interval = 10

        for epoch in range(self.current_epoch, num_epochs):
            if should_stop():
                logger.warning(f"Training interrupted at epoch {epoch}")
                self._save_checkpoint(
                    self.checkpoint_dir / "interrupted_checkpoint.pt", is_best=False
                )
                break

            self.current_epoch = epoch
            self.metrics_reporter.report_epoch_start(epoch)

            try:
                epoch_loss = self._train_epoch(dataloader, epoch)
            except PermissionError as exc:
                if dataloader.num_workers > 0:
                    logger.warning(
                        f"Permission denied criando DataLoader workers ({exc}); reconfigurando para num_workers=0"
                    )
                    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=self.device.type == "cuda",
                    )
                    epoch_loss = self._train_epoch(dataloader, epoch)
                else:
                    raise

            if val_triples is not None and epoch % validate_every == 0:
                val_metrics = self._validate(val_triples)
                self.last_val_metrics = val_metrics

                # Report epoch metrics via Observer pattern
                current_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else None
                self.metrics_reporter.report_epoch_end(
                    epoch=epoch,
                    train_loss=epoch_loss,
                    val_metrics=val_metrics,
                    learning_rate=current_lr,
                )

                self.obs_manager.record_training_metrics(
                    epoch=epoch, loss=epoch_loss, val_metrics=val_metrics
                )

                if val_metrics["mrr"] > self.best_val_score:
                    self.best_val_score = val_metrics["mrr"]
                    self.patience_counter = 0
                    training_stats["best_epoch"] = epoch
                    training_stats["best_val_mrr"] = self.best_val_score
                    self._save_checkpoint(
                        self.checkpoint_dir / "best_model.pt", is_best=True
                    )
                else:
                    self.patience_counter += 1

                if self.patience_counter >= patience:
                    logger.info(f"Early stopping na epoca {epoch}: patience={patience}")
                    break

                if self.scheduler and isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["mrr"])

            if self.scheduler and not isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step()

            # Periodic memory cleanup to prevent gradual memory leaks
            if epoch > 0 and epoch % memory_cleanup_interval == 0:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            training_stats["epochs_trained"] = epoch + 1

        training_time = time.time() - start_time
        training_stats["training_time"] = training_time
        training_stats["final_metrics"] = self.last_val_metrics
        training_stats["status"] = "completed"

        logger.success(
            f"Treinamento RotatE concluido: {training_stats['epochs_trained']} epocas, "
            f"{training_time:.1f}s, MRR={training_stats['best_val_mrr']:.4f}"
        )

        # Save completion marker using CheckpointManager
        self.checkpoint_manager.mark_training_completed(
            epochs_trained=training_stats["epochs_trained"],
            target_epochs=num_epochs,
            best_epoch=training_stats["best_epoch"],
            best_val_mrr=training_stats["best_val_mrr"],
            training_time=training_time,
            final_metrics=self.last_val_metrics,
        )

        # Report training end to observers
        self.metrics_reporter.report_training_end(
            final_metrics=self.last_val_metrics,
            epochs_trained=training_stats["epochs_trained"],
            training_time=training_time,
        )

        if mlflow.active_run():
            mlflow.end_run()

        return training_stats

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train one epoch with self-adversarial negative sampling.

        Args:
            dataloader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Average loss for the epoch.
        """
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model and optimizer must be initialized")

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        training_cfg = self.config.get("training", {})
        use_amp = (
            self.device.type == "cuda"
            and training_cfg.get("mixed_precision", True)
            and hasattr(torch.cuda.amp, "autocast")
        )
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        max_grad_norm = training_cfg.get("gradient_clip_val", 1.0)

        for batch in dataloader:
            positives = batch["positive"].to(self.device)
            negatives = batch["negatives"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = self.model.compute_loss(positives, negatives)
                    reg_loss = self.model.regularization_loss()
                    total = loss + reg_loss

                scaler.scale(total).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss = self.model.compute_loss(positives, negatives)
                reg_loss = self.model.regularization_loss()
                total = loss + reg_loss

                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        if mlflow.active_run():
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        return avg_loss

    def _validate(self, val_triples: np.ndarray) -> dict[str, float]:
        """Validate model using vectorized batch scoring.

        Args:
            val_triples: Validation triples array of shape (n_triples, 3).

        Returns:
            Dictionary with 'mrr', 'hits@1', 'hits@3', 'hits@10', and 'mean_rank' metrics.
        """
        if self.model is None:
            raise RuntimeError("Model must be initialized")

        self.model.eval()
        num_samples = len(val_triples)
        num_entities = self.model.num_entities

        # Adaptive batch size based on entity count to prevent OOM
        # Memory usage ≈ batch_size * num_entities * 4 bytes (float32)
        # Target max ~200MB for validation scores tensor
        max_memory_bytes = 200 * 1024 * 1024  # 200MB
        max_batch_by_memory = max(1, max_memory_bytes // (num_entities * 4))
        val_batch_size = min(128, num_samples, max_batch_by_memory)

        all_mrr = []
        all_hits1 = []
        all_hits3 = []
        all_hits10 = []
        all_ranks: list[torch.Tensor] = []

        with torch.no_grad():
            all_entities = torch.arange(num_entities, device=self.device)

            for batch_start in range(0, num_samples, val_batch_size):
                batch_end = min(batch_start + val_batch_size, num_samples)
                batch_triples = val_triples[batch_start:batch_end]
                batch_size = len(batch_triples)

                heads_batch = torch.tensor(
                    batch_triples[:, 0], dtype=torch.long, device=self.device
                )
                rels_batch = torch.tensor(
                    batch_triples[:, 1], dtype=torch.long, device=self.device
                )
                tails_batch = torch.tensor(
                    batch_triples[:, 2], dtype=torch.long, device=self.device
                )
                heads_expanded = heads_batch.unsqueeze(1).expand(-1, num_entities)
                rels_expanded = rels_batch.unsqueeze(1).expand(-1, num_entities)
                all_tails = all_entities.unsqueeze(0).expand(batch_size, -1)

                scores = self.model.forward(
                    heads_expanded.reshape(-1),
                    rels_expanded.reshape(-1),
                    all_tails.reshape(-1),
                ).reshape(batch_size, num_entities)

                true_scores = scores[
                    torch.arange(batch_size, device=self.device), tails_batch
                ]

                ranks = (scores > true_scores.unsqueeze(1)).sum(dim=1) + 1

                all_mrr.append((1.0 / ranks.float()).cpu())
                all_hits1.append((ranks == 1).cpu())
                all_hits3.append((ranks <= 3).cpu())
                all_hits10.append((ranks <= 10).cpu())
                all_ranks.append(ranks.cpu())

        mrr_tensor = torch.cat(all_mrr)
        hits1_tensor = torch.cat(all_hits1)
        hits3_tensor = torch.cat(all_hits3)
        hits10_tensor = torch.cat(all_hits10)
        ranks_tensor = torch.cat(all_ranks)

        metrics = {
            "mrr": mrr_tensor.mean().item(),
            "hits@1": hits1_tensor.float().mean().item(),
            "hits@3": hits3_tensor.float().mean().item(),
            "hits@10": hits10_tensor.float().mean().item(),
            "mean_rank": ranks_tensor.float().mean().item(),
        }

        if mlflow.active_run():
            for metric_name, value in metrics.items():
                safe_name = metric_name.replace("@", "_at_")
                mlflow.log_metric(f"val_{safe_name}", value, step=self.current_epoch)

        return metrics

    def evaluate(self, test_triples: np.ndarray | None = None) -> dict[str, float]:
        """Evaluate model on test set and return metrics.

        This method provides a public interface for evaluation, loading test data
        if not provided and computing link prediction metrics.

        Args:
            test_triples: Optional test triples array. Uses loaded test data if None.

        Returns:
            Dictionary with 'mrr', 'hits@1', 'hits@10' and other metrics.

        Raises:
            ValueError: If no test data available.
        """
        if test_triples is None:
            if self.test_triples is None:
                if self.val_triples is not None:
                    logger.info("Usando dados de validacao para avaliacao final")
                    test_triples = self.val_triples
                else:
                    raise ValueError(
                        "No test triples available. Provide test_triples or ensure "
                        "test data exists in the data directory."
                    )
            else:
                test_triples = self.test_triples

        logger.info(f"Avaliando modelo RotatE em {len(test_triples):,} triplas...")

        metrics = self._validate(test_triples)

        logger.success(
            f"Avaliacao concluida: MRR={metrics['mrr']:.4f}, "
            f"Hits@1={metrics['hits@1']:.4f}, Hits@10={metrics['hits@10']:.4f}"
        )

        # Save evaluation metrics to file
        eval_path = self.checkpoint_dir / "evaluation_metrics.json"
        self.file_manager.save(metrics, eval_path)
        logger.info(f"Metricas de avaliacao salvas em: {eval_path}")

        return metrics

    def extract_embeddings(self) -> dict[str, np.ndarray]:
        """Extract embeddings from RotatE model in TransE-compatible format.

        This ensures the Ensemble pipeline can consume RotatE embeddings
        in the same format as TransE embeddings.

        Returns:
            Dictionary with 'entity_embeddings', 'relation_embeddings',
            'entity' (alias), and 'relation' (alias) numpy arrays.

        Raises:
            RuntimeError: If the model is not initialized.
        """
        logger.info("Extraindo embeddings do modelo RotatE...")

        if self.model is None:
            raise RuntimeError("Modelo RotatE nao esta carregado!")

        with torch.no_grad():
            # RotatE uses complex embeddings (real, imag parts)
            # We concatenate them for compatibility with downstream models
            entity_real, entity_imag = self.model.get_entity_embeddings()
            entity_embeddings = torch.cat([entity_real, entity_imag], dim=-1).cpu().numpy()
            
            # Relation embeddings are phase angles in RotatE
            # Convert to cos/sin format to match entity embedding dimensions
            relation_phases = self.model.get_relation_phases().cpu().numpy()
            relation_real = np.cos(relation_phases)
            relation_imag = np.sin(relation_phases)
            relation_embeddings = np.concatenate([relation_real, relation_imag], axis=1)

        logger.success(
            f"Embeddings extraidos: entities={entity_embeddings.shape}, "
            f"relations={relation_embeddings.shape}"
        )

        # Save in RotatE output directory
        embeddings_path = settings.OUTPUTS_DIR / "rotate" / "node_embeddings.pkl"
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)

        embeddings = {
            "entity_embeddings": entity_embeddings,
            "relation_embeddings": relation_embeddings,
            "entity": entity_embeddings,
            "relation": relation_embeddings,
        }

        self.file_manager.save(embeddings, embeddings_path)
        logger.info(f"Embeddings RotatE salvos em: {embeddings_path}")

        return embeddings

    def _save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Save model checkpoint using CheckpointManager.

        Args:
            path: Path to save checkpoint.
            is_best: Whether this is the best model so far.
        """
        if self.model is None:
            logger.warning("Cannot save checkpoint: model is None")
            return

        # Use CheckpointManager for SRP compliance
        saved_path = self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            metrics=self.last_val_metrics,
            is_best=is_best,
            filename=path.name if path else None,
        )

        # Notify observers of checkpoint event
        self.metrics_reporter.report_checkpoint(
            epoch=self.current_epoch,
            path=saved_path,
            is_best=is_best,
        )

    def _load_checkpoint(self, path: Path | None = None) -> bool:
        """Load model checkpoint using CheckpointManager.

        Args:
            path: Path to checkpoint. Uses best_model.pt if None.

        Returns:
            True if checkpoint was loaded, False otherwise.
        """
        if path is None:
            path = self.checkpoint_manager.get_best_checkpoint()

        if path is None or not path.exists():
            return False

        if self.model is None:
            self._setup_model()

        if self.model is None:
            raise RuntimeError("Model could not be initialized before loading state")

        # Use CheckpointManager for loading
        checkpoint_info = self.checkpoint_manager.load(
            model=self.model,
            optimizer=self.optimizer,
            path=path,
            device=self.device,
        )

        self.current_epoch = checkpoint_info.get("epoch", 0) + 1
        self.last_val_metrics = checkpoint_info.get("metrics", {})
        self.best_val_score = self.last_val_metrics.get("mrr", -float("inf"))

        logger.success(
            f"Checkpoint carregado: epoca {checkpoint_info.get('epoch', 0)}, "
            f"MRR={self.best_val_score:.4f}"
        )
        return True

    def _start_mlflow_run(self) -> None:
        """Start MLflow run for experiment tracking."""
        mlflow_config = self.config.get("mlflow", {})
        experiment_name = mlflow_config.get("experiment_name", "rotate_training")

        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        mlflow.log_params(
            {
                "embedding_dim": self.rotate_config.embedding_dim,
                "gamma": self.rotate_config.gamma,
                "epsilon": self.rotate_config.epsilon,
                "learning_rate": self.rotate_config.learning_rate,
                "batch_size": self.rotate_config.batch_size,
                "num_negatives": self.rotate_config.num_negatives,
                "use_self_adversarial": self.rotate_config.use_self_adversarial,
                "adversarial_temperature": self.rotate_config.adversarial_temperature,
            }
        )
