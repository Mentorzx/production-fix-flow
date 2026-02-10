"""DSLFM-KGC Training Manager.

This module provides the training pipeline for DSLFM-KGC with:
- Gradient accumulation for effective large batches on limited VRAM
- KL annealing for stable VAE training
- In-batch negative sampling
- Mixed precision training

Design Patterns:
    - Template Method: Training loop structure
    - Strategy: Different loss functions
    - Observer: Training callbacks
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pff.domain.learning.dslfm.time_estimator import (
    TimeBudgetConfig,
    TimeBudgetEstimator,
)
from pff.domain.learning.ml.training_observer import TrainingObserver
from pff.domain.ports.persistence.model_persistence import ModelPersistencePort
from pff.shared.acceleration.concurrency import progress_bar
from pff.shared.acceleration.numba_kernels import (
    TripleStoreSoA,
    find_unique_triples_mask_numba,
)
from pff.shared.core.config import settings
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logging import logger
from pff.shared.ops.global_interrupt_manager import check_interruption, should_stop
from pff.shared.system.cuda import is_cuda_available
from pff.shared.system.resource_manager import (
    get_auto_dataloader_workers,
    get_memory_safe_workers,
)

from .dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel


def _configure_scheduler_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message="The epoch parameter in `scheduler.step()`",
        category=UserWarning,
    )


def _bind_evaluate(model: DSLFMKGCModel) -> DSLFMKGCModel:
    """Ensure evaluate stays bound on the instance for downstream access."""
    try:
        bound = DSLFMKGCModel.evaluate.__get__(model, DSLFMKGCModel)
    except AttributeError as exc:
        raise AttributeError("DSLFMKGCModel is missing evaluate()") from exc
    model.evaluate = bound
    return model


class _CompiledModelWrapper(nn.Module):
    """Wrapper to preserve evaluate/utility methods when using torch.compile."""

    def __init__(self, base_model: DSLFMKGCModel, compiled_model: Any) -> None:
        super().__init__()
        self.base_model = base_model
        self.compiled_model = compiled_model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.compiled_model(*args, **kwargs)

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model.evaluate(*args, **kwargs)

    def score_triples_batch(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_model.score_triples_batch(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in {"base_model", "compiled_model"}:
            return super().__getattr__(name)
        if hasattr(self.base_model, name):
            return getattr(self.base_model, name)
        return super().__getattr__(name)

    @property
    def config(self) -> DSLFMKGCConfig:
        return self.base_model.config


@dataclass
class KGCTrainingConfig:
    """Configuration for DSLFM-KGC training."""

    epochs: int = 200
    batch_size: int = 256
    effective_batch_size: int = 1024
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    kl_warmup_epochs: int = 10
    min_kl_weight: float = 0.0
    max_kl_weight: float = 0.1
    temperature: float = 0.5
    temperature_anneal: float = 0.99
    min_temperature: float = 0.1
    validate_every: int = 5
    early_stopping_patience: int = 10
    min_delta: float = 0.0002
    mixed_precision: bool = True
    debug_checks: bool = False
    regularization_warmup_epochs: int = 8
    regularization_start_scale: float = 0.0
    use_compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_dynamic: bool = True
    compile_fullgraph: bool = False
    compile_backend: str | None = None
    tf32: bool = True
    optimizer_8bit: bool = False
    schedule_free: bool = False
    checkpoint_dir: Path = field(
        default_factory=lambda: settings.OUTPUTS_DIR / "dslfm_kgc" / "checkpoints"
    )
    num_workers: int = 0
    num_workers_heuristic: dict[str, Any] = field(default_factory=dict)
    pin_memory: bool = True
    dataloader_prefetch_factor: int = 4
    dataloader_persistent_workers: bool = True
    eval_batch_size: int = 256
    rerank_top_k: int | None = 256
    refresh_cache_on_val: bool = True
    pruning_burn_in_epochs: int = 10
    train_heartbeat_interval_s: float = 60.0
    score_all_tails_chunk_size: int = 20_000
    max_grad_norm: float | None = None
    time_budget: dict[str, Any] = field(default_factory=dict)
    adaptive_batch_size: bool = False
    min_batch_size: int = 128
    max_batch_size: int = 1024
    oom_backoff_factor: float = 0.5
    batch_growth_factor: float = 1.2
    target_gpu_mem_util: float = 0.7
    max_oom_retries: int = 3
    cuda_cache_flush_steps: int = 0
    cuda_cache_flush_enabled: bool = True
    cuda_cache_flush_free_ratio_low: float = 0.15
    cuda_cache_flush_free_ratio_high: float = 0.4
    use_faiss_eval: bool = False
    faiss_candidate_k: int = 1024
    allow_tf32: bool = True
    matmul_precision: str = "high"
    optimizer_fused: bool | None = None
    optimizer_foreach: bool | None = None
    mask_dense_max_entries: int = 5_000_000


class KGCTrainingConfigBuilder:
    """Fluent builder for KGCTrainingConfig."""

    def __init__(self, config: KGCTrainingConfig | None = None) -> None:
        self._config = config or KGCTrainingConfig()

    def with_epochs(self, value: int) -> KGCTrainingConfigBuilder:
        self._config.epochs = int(value)
        return self

    def with_batch_size(self, value: int) -> KGCTrainingConfigBuilder:
        self._config.batch_size = int(value)
        return self

    def with_effective_batch_size(self, value: int) -> KGCTrainingConfigBuilder:
        self._config.effective_batch_size = int(value)
        return self

    def with_learning_rate(self, value: float) -> KGCTrainingConfigBuilder:
        self._config.learning_rate = float(value)
        return self

    def with_validate_every(self, value: int) -> KGCTrainingConfigBuilder:
        self._config.validate_every = int(value)
        return self

    def with_early_stopping(
        self,
        *,
        patience: int | None = None,
        min_delta: float | None = None,
    ) -> KGCTrainingConfigBuilder:
        if patience is not None:
            self._config.early_stopping_patience = int(patience)
        if min_delta is not None:
            self._config.min_delta = float(min_delta)
        return self

    def with_mixed_precision(self, value: bool) -> KGCTrainingConfigBuilder:
        self._config.mixed_precision = bool(value)
        return self

    def with_time_budget(self, value: dict[str, Any]) -> KGCTrainingConfigBuilder:
        self._config.time_budget = dict(value)
        return self

    def apply_overrides(self, overrides: dict[str, Any]) -> KGCTrainingConfigBuilder:
        for key, value in overrides.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        return self

    def build(self) -> KGCTrainingConfig:
        if self._config.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self._config.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self._config.effective_batch_size <= 0:
            raise ValueError("effective_batch_size must be > 0")
        return self._config


def build_dslfm_configs(
    *,
    num_entities: int,
    num_relations: int,
    num_triples: int,
    raw_settings: dict[str, Any],
    overrides: dict[str, Any],
    checkpoint_dir: Path,
    use_bert: bool | None = None,
    relation_names: list[str] | None = None,
) -> tuple[DSLFMKGCConfig, KGCTrainingConfig]:
    """Build DSLFMKGCConfig and KGCTrainingConfig from YAML settings + overrides.

    Centralizes config construction for both production training and HPO trials,
    ensuring parameter parity and eliminating config drift.  Every configurable
    field is resolved through a three-level cascade:

        override dict  →  YAML section  →  dataclass default

    Args:
        num_entities: Total entity count in the KG.
        num_relations: Total relation count in the KG.
        num_triples: Total training triple count (critical for NSCachingSampler).
        raw_settings: Output of ``load_dslfm_kgc_settings()`` (full YAML dict).
        overrides: Parameter overrides (kwargs in production, HPO trial params).
        checkpoint_dir: Directory for model checkpoints.
        use_bert: Whether to use BERT relation encoder (None → read from config).
        relation_names: Relation names for BERT encoder (None disables BERT).

    Returns:
        Tuple of (DSLFMKGCConfig, KGCTrainingConfig).
    """
    if not isinstance(raw_settings, dict):
        raw_settings = {}

    kgc_cfg = raw_settings.get("kgc", {})
    if not isinstance(kgc_cfg, dict):
        kgc_cfg = {}
    m_cfg = kgc_cfg.get("model", {})
    t_cfg = kgc_cfg.get("training", {})
    if not isinstance(m_cfg, dict):
        m_cfg = {}
    if not isinstance(t_cfg, dict):
        t_cfg = {}
    logic_cfg = raw_settings.get("logic", {})
    pc_cfg = raw_settings.get("pc", {})
    compile_cfg = raw_settings.get("compile", {})
    if not isinstance(logic_cfg, dict):
        logic_cfg = {}
    if not isinstance(pc_cfg, dict):
        pc_cfg = {}
    if not isinstance(compile_cfg, dict):
        compile_cfg = {}

    def _get(section: dict[str, Any], key: str, fallback: Any) -> Any:
        if key in overrides:
            return overrides[key]
        return section.get(key, fallback)

    cuda_cache_cfg = t_cfg.get("cuda_cache_flush", {})
    if not isinstance(cuda_cache_cfg, dict):
        cuda_cache_cfg = {}
    num_workers_heuristic = t_cfg.get("num_workers_heuristic", {})
    if not isinstance(num_workers_heuristic, dict):
        num_workers_heuristic = {}

    use_bert_relations = (
        _resolve_use_bert_setting(use_bert, m_cfg) and relation_names is not None
    )

    pc_max_depth_raw = int(_get(pc_cfg, "max_circuit_depth", 0))

    model_config = DSLFMKGCConfig(
        num_entities=num_entities,
        num_relations=num_relations,
        num_triples=num_triples,
        entity_dim=int(_get(m_cfg, "entity_dim", 256)),
        feature_dim=int(_get(m_cfg, "feature_dim", 256)),
        max_communities=int(_get(m_cfg, "max_communities", 128)),
        hidden_dim=int(_get(m_cfg, "hidden_dim", 512)),
        ibp_alpha=float(_get(m_cfg, "ibp_alpha", 1.0)),
        temperature=float(_get(m_cfg, "temperature", 0.5)),
        stochastic_latents=bool(_get(m_cfg, "stochastic_latents", False)),
        encoder_dropout_p=float(_get(m_cfg, "encoder_dropout_p", 0.0)),
        kl_weight=float(_get(m_cfg, "kl_weight", 0.1)),
        free_bits=float(_get(m_cfg, "free_bits", 0.125)),
        sparsity_weight=float(_get(m_cfg, "sparsity_weight", 0.01)),
        use_bert_relations=use_bert_relations,
        bert_model=str(_get(m_cfg, "bert_model", "bert-base-uncased")),
        use_checkpointing=bool(_get(m_cfg, "use_checkpointing", False)),
        sampler_type=str(_get(m_cfg, "sampler_type", "degree_based")),
        sampler_temperature=float(_get(m_cfg, "sampler_temperature", 1.0)),
        learnable_temperature=bool(_get(m_cfg, "learnable_temperature", False)),
        contrastive_temperature=float(_get(m_cfg, "contrastive_temperature", 0.07)),
        negative_sample_size=int(_get(m_cfg, "negative_sample_size", 0)),
        num_global_negatives=int(_get(m_cfg, "num_global_negatives", 0)),
        cache_global_negatives=bool(_get(m_cfg, "cache_global_negatives", False)),
        global_negatives_refresh_steps=int(
            _get(m_cfg, "global_negatives_refresh_steps", 50)
        ),
        logvar_clip_min=float(_get(m_cfg, "logvar_clip_min", -20.0)),
        logvar_clip_max=float(_get(m_cfg, "logvar_clip_max", 10.0)),
        community_weight=float(_get(m_cfg, "community_weight", 1.0)),
        feature_weight=float(_get(m_cfg, "feature_weight", 0.0)),
        nsc_cache_size=int(_get(m_cfg, "nsc_cache_size", 64)),
        nsc_sample_ratio=float(_get(m_cfg, "nsc_sample_ratio", 0.5)),
        lambda_logic=float(_get(logic_cfg, "lambda_logic", 0.0)),
        t_norm=str(_get(logic_cfg, "t_norm", "product")),
        smoothing_epsilon=float(_get(logic_cfg, "smoothing_epsilon", 1e-6)),
        lambda_pc=float(_get(pc_cfg, "lambda_pc", 0.0)),
        pc_pruning_threshold=float(_get(pc_cfg, "pruning_threshold", 0.01)),
        pc_grow_noise=float(_get(pc_cfg, "grow_noise", 0.01)),
        pc_rebuild_every=int(_get(pc_cfg, "rebuild_every", 0)),
        pc_max_depth=pc_max_depth_raw if pc_max_depth_raw > 0 else None,
        triton_min_entities=int(_get(t_cfg, "triton_min_entities", 1024)),
    )

    time_budget = _get(t_cfg, "time_budget", {})
    if not isinstance(time_budget, dict):
        time_budget = {}

    train_config = KGCTrainingConfig(
        epochs=int(_get(t_cfg, "epochs", 200)),
        batch_size=int(_get(t_cfg, "batch_size", 256)),
        effective_batch_size=int(_get(t_cfg, "effective_batch_size", 1024)),
        learning_rate=float(_get(t_cfg, "learning_rate", 1e-4)),
        warmup_steps=int(_get(t_cfg, "warmup_steps", 1000)),
        kl_warmup_epochs=int(_get(t_cfg, "kl_warmup_epochs", 10)),
        min_kl_weight=float(_get(t_cfg, "min_kl_weight", 0.0)),
        max_kl_weight=float(_get(t_cfg, "max_kl_weight", 0.1)),
        temperature=float(_get(t_cfg, "temperature", model_config.temperature)),
        temperature_anneal=float(_get(t_cfg, "temperature_anneal", 0.99)),
        min_temperature=float(_get(t_cfg, "min_temperature", 0.1)),
        validate_every=int(_get(t_cfg, "validate_every", 5)),
        early_stopping_patience=int(_get(t_cfg, "early_stopping_patience", 10)),
        min_delta=float(_get(t_cfg, "min_delta", 0.0002)),
        train_heartbeat_interval_s=float(
            _get(t_cfg, "train_heartbeat_interval_s", 60.0)
        ),
        score_all_tails_chunk_size=int(
            _get(t_cfg, "score_all_tails_chunk_size", 20_000)
        ),
        mixed_precision=bool(_get(t_cfg, "mixed_precision", True)),
        use_compile=bool(_get(t_cfg, "use_compile", False)),
        compile_mode=str(_get(compile_cfg, "mode", "reduce-overhead")),
        compile_dynamic=bool(_get(compile_cfg, "dynamic", True)),
        compile_fullgraph=bool(_get(compile_cfg, "fullgraph", False)),
        compile_backend=_get(compile_cfg, "backend", None),
        optimizer_fused=_get(t_cfg, "optimizer_fused", None),
        optimizer_foreach=_get(t_cfg, "optimizer_foreach", None),
        num_workers=int(_get(t_cfg, "num_workers", 0)),
        num_workers_heuristic=dict(num_workers_heuristic),
        pin_memory=bool(_get(t_cfg, "pin_memory", True)),
        dataloader_prefetch_factor=int(_get(t_cfg, "dataloader_prefetch_factor", 4)),
        dataloader_persistent_workers=bool(
            _get(t_cfg, "dataloader_persistent_workers", True)
        ),
        eval_batch_size=int(_get(t_cfg, "eval_batch_size", 256)),
        regularization_warmup_epochs=int(
            _get(t_cfg, "regularization_warmup_epochs", 8)
        ),
        regularization_start_scale=float(
            _get(t_cfg, "regularization_start_scale", 0.0)
        ),
        rerank_top_k=_get(t_cfg, "rerank_top_k", 256),
        refresh_cache_on_val=bool(_get(t_cfg, "refresh_cache_on_val", True)),
        max_grad_norm=_get(t_cfg, "max_grad_norm", None),
        adaptive_batch_size=bool(_get(t_cfg, "adaptive_batch_size", False)),
        min_batch_size=int(_get(t_cfg, "min_batch_size", 128)),
        max_batch_size=int(_get(t_cfg, "max_batch_size", 1024)),
        oom_backoff_factor=float(_get(t_cfg, "oom_backoff_factor", 0.5)),
        batch_growth_factor=float(_get(t_cfg, "batch_growth_factor", 1.2)),
        target_gpu_mem_util=float(_get(t_cfg, "target_gpu_mem_util", 0.7)),
        max_oom_retries=int(_get(t_cfg, "max_oom_retries", 3)),
        cuda_cache_flush_steps=int(_get(t_cfg, "cuda_cache_flush_steps", 0)),
        cuda_cache_flush_enabled=bool(_get(cuda_cache_cfg, "enabled", True)),
        cuda_cache_flush_free_ratio_low=float(
            _get(cuda_cache_cfg, "free_ratio_low", 0.15)
        ),
        cuda_cache_flush_free_ratio_high=float(
            _get(cuda_cache_cfg, "free_ratio_high", 0.4)
        ),
        use_faiss_eval=bool(_get(t_cfg, "use_faiss_eval", False)),
        faiss_candidate_k=int(_get(t_cfg, "faiss_candidate_k", 1024)),
        allow_tf32=bool(_get(t_cfg, "allow_tf32", True)),
        matmul_precision=str(_get(t_cfg, "matmul_precision", "high")),
        mask_dense_max_entries=int(_get(t_cfg, "mask_dense_max_entries", 5_000_000)),
        checkpoint_dir=checkpoint_dir,
        time_budget=time_budget,
    )

    return model_config, train_config


class TripleDataset(Dataset):
    """Simple dataset for triples with indices."""

    def __init__(self, triples: np.ndarray) -> None:
        triples_arr = np.asarray(triples, dtype=np.int64)
        if not triples_arr.flags.writeable:
            triples_arr = np.array(triples_arr, copy=True)
        self.triples = torch.from_numpy(triples_arr).long()

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.triples[idx], idx


def _debug_check(tensor: Any, name: str) -> None:
    """Check tensors for NaN/Inf values."""
    if not isinstance(tensor, torch.Tensor):
        return
    if is_cuda_available() and tensor.is_cuda:
        torch.cuda.synchronize()
    if torch.isnan(tensor).any():
        logger.error(f"NaN detected in {name}!")
        raise RuntimeError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        logger.error(f"Inf detected in {name}!")
        raise RuntimeError(f"Inf detected in {name}")


class DSLFMKGCManager:
    """Training manager for DSLFM-KGC model."""

    def __init__(
        self,
        model_config: DSLFMKGCConfig,
        training_config: KGCTrainingConfig,
        persistence_port: ModelPersistencePort,
        relation_names: list[str] | None = None,
        device: torch.device | None = None,
        observers: list[TrainingObserver] | None = None,
        seed: int | None = None,
    ) -> None:
        self.model_config = model_config
        self.training_config = training_config
        self.persistence_port = persistence_port
        self.observers = observers or []
        self.device = device or torch.device("cuda" if is_cuda_available() else "cpu")
        self.rng = np.random.default_rng(seed)

        if self.device.type == "cuda":
            allow_tf32 = bool(self.training_config.allow_tf32)
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision(
                    self.training_config.matmul_precision
                )

        self._update_accumulation_steps()

        base_model = _bind_evaluate(
            DSLFMKGCModel(model_config, relation_names=relation_names).to(self.device)
        )

        if training_config.use_compile and hasattr(torch, "compile"):
            try:
                compile_dynamic = bool(training_config.compile_dynamic)
                if training_config.adaptive_batch_size and not compile_dynamic:
                    logger.warning(
                        "Forcing dynamic=True for torch.compile with adaptive batch size"
                    )
                    compile_dynamic = True

                compile_kwargs = {
                    "mode": str(training_config.compile_mode),
                    "dynamic": compile_dynamic,
                    "fullgraph": bool(training_config.compile_fullgraph),
                }
                if training_config.compile_backend:
                    compile_kwargs["backend"] = str(training_config.compile_backend)

                compiled = cast(nn.Module, torch.compile(base_model, **compile_kwargs))

                self.model = _CompiledModelWrapper(base_model, compiled)
                logger.debug("Model compiled with torch.compile")
            except Exception as e:
                logger.warning("torch.compile failed, using eager mode", error=str(e))
                self.model = base_model
        else:
            self.model = base_model

        if self.model.use_bert_relations:
            self.model.precompute_relation_embeddings(self.device)

        is_cuda = self.device.type == "cuda"
        optimizer_fused = training_config.optimizer_fused
        optimizer_foreach = training_config.optimizer_foreach
        fused = bool(optimizer_fused) if optimizer_fused is not None else is_cuda
        if not is_cuda:
            fused = False
        foreach = (
            bool(optimizer_foreach) if optimizer_foreach is not None else not is_cuda
        )
        if fused and foreach:
            logger.warning(
                "AdamW fused=True is incompatible with foreach=True; disabling foreach"
            )
            foreach = False
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=1e-5,
            fused=fused,
            foreach=foreach,
        )

        self.scheduler = self._create_scheduler()
        use_scaler = training_config.mixed_precision and is_cuda
        self.scaler = torch.amp.GradScaler("cuda") if use_scaler else None  # type: ignore[attr-defined]

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_mrr = float("-inf")
        self.best_val_mcc = float("-inf")
        self.patience_counter = 0
        self.current_temperature = training_config.temperature
        self._filter_arrays: dict[tuple[int, int], np.ndarray] = {}
        self._filter_tensors: dict[tuple[int, int], torch.Tensor] = {}
        self._entity_cache_ready = False

        tb_conf = TimeBudgetConfig.from_dict(training_config.time_budget)
        self.time_estimator = TimeBudgetEstimator(
            tb_conf,
            total_epochs=training_config.epochs,
            validate_every=training_config.validate_every,
        )

        bert_status = (
            "BERT nas relacoes"
            if self.model.use_bert_relations
            else "relacoes aprendidas"
        )
        logger.info(
            "Gerente DSLFM-KGC inicializado",
            batch=training_config.batch_size,
            effective=training_config.effective_batch_size,
            acumulacao=self.accumulation_steps,
            bert_status=bert_status,
        )

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.training_config.warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.training_config.epochs * 100,
            eta_min=1e-6,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.training_config.warmup_steps],
        )

    def _get_kl_weight(self, epoch: int) -> float:
        if epoch >= self.training_config.kl_warmup_epochs:
            return self.training_config.max_kl_weight
        progress = epoch / self.training_config.kl_warmup_epochs
        return self.training_config.min_kl_weight + progress * (
            self.training_config.max_kl_weight - self.training_config.min_kl_weight
        )

    def _get_regularization_scale(self, epoch: int) -> float:
        warmup = max(0, self.training_config.regularization_warmup_epochs)
        start = self.training_config.regularization_start_scale
        if warmup == 0:
            return 1.0
        progress = min(1.0, epoch / warmup)
        return start + (1.0 - start) * progress

    def _collect_elbo_metrics(self, loss_dict: dict, epoch: int) -> dict[str, float]:
        """Extract ELBO breakdown from loss dict for dashboard.

        Args:
            loss_dict: Loss dictionary from model.compute_loss().
            epoch: Current epoch number.

        Returns:
            Dict with elbo_recon, elbo_kl, kl_weight metrics.
        """
        contrastive = loss_dict.get("contrastive_loss")
        kl_gaussian = loss_dict.get("kl_gaussian")
        kl_ibp = loss_dict.get("kl_ibp")

        elbo_recon = 0.0
        if contrastive is not None:
            elbo_recon = (
                float(contrastive.item())
                if hasattr(contrastive, "item")
                else float(contrastive)
            )

        elbo_kl = 0.0
        if kl_gaussian is not None:
            kl_g = (
                float(kl_gaussian.item())
                if hasattr(kl_gaussian, "item")
                else float(kl_gaussian)
            )
            elbo_kl += kl_g
        if kl_ibp is not None:
            kl_i = float(kl_ibp.item()) if hasattr(kl_ibp, "item") else float(kl_ibp)
            elbo_kl += kl_i

        return {
            "elbo_recon": elbo_recon,
            "elbo_kl": elbo_kl,
            "kl_weight": float(self._get_kl_weight(epoch)),
        }

    def _collect_pc2_metrics(self, losses: dict) -> dict[str, float]:
        """Collect PC2 circuit metrics for dashboard.

        Args:
            losses: Loss dictionary that may contain PC-related info.

        Returns:
            Dict with pc2_rules, pc2_contexts, pc2_latency, pc2_density metrics.
        """
        pc_model = getattr(self.model, "pc_model", None)
        if pc_model is None:
            return {}

        num_rules = getattr(pc_model, "num_rules", 0)
        if num_rules == 0:
            num_rules = getattr(pc_model, "rule_count", 0)

        num_contexts = self.model_config.num_relations

        pc_density = 0.0
        sparsity = losses.get("sparsity_loss")
        if sparsity is not None:
            sp_val = (
                float(sparsity.item()) if hasattr(sparsity, "item") else float(sparsity)
            )
            pc_density = 1.0 - min(1.0, sp_val)

        return {
            "pc2_rules": int(num_rules),
            "pc2_contexts": int(num_contexts),
            "pc2_latency": getattr(self.model, "last_pc2_latency", 0.0),
            "pc2_density": pc_density,
        }

    def _collect_structural_metrics(self) -> dict[str, float]:
        """Collect structural metrics for dashboard.

        Returns:
            Dict with graphDensity and estimated numClusters.
        """
        num_entities = self.model_config.num_entities
        num_relations = self.model_config.num_relations

        num_triples = getattr(self, "_train_triples_count", 0)
        if num_triples == 0:
            num_triples = getattr(self.model_config, "num_triples", 0)

        max_possible = num_entities * num_entities * max(1, num_relations)
        graph_density = float(num_triples / max_possible) if max_possible > 0 else 0.0

        num_clusters = getattr(self.model_config, "max_communities", 0)
        if num_clusters == 0:
            num_clusters = getattr(self.model_config, "num_communities", 128)

        return {
            "graphDensity": graph_density,
            "numClusters": num_clusters,
            "latentEntropy": 0.0,
            "communityOverlap": 0.0,
        }

    def _anneal_temperature(self) -> None:
        self.current_temperature = max(
            self.training_config.min_temperature,
            self.current_temperature * self.training_config.temperature_anneal,
        )

    def _update_accumulation_steps(self) -> None:
        if self.training_config.tf32 and is_cuda_available():
            torch.set_float32_matmul_precision("medium")
            logger.debug("TF32 precision enabled for matmuls")
        self.accumulation_steps = max(
            1,
            self.training_config.effective_batch_size
            // self.training_config.batch_size,
        )

    def _resolve_adaptive_batch_size(self) -> None:
        if (
            not self.training_config.adaptive_batch_size
            or not torch.cuda.is_available()
        ):
            return
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024**3)
            current = self.training_config.batch_size
            min_bs, max_bs = (
                self.training_config.min_batch_size,
                self.training_config.max_batch_size,
            )

            if free_gb < 4:
                target = min_bs
            elif free_gb < 8:
                target = max(min_bs, current // 2)
            elif free_gb < 12:
                target = current
            else:
                target = max_bs

            target = max(min_bs, min(max_bs, target))
            if target != current:
                logger.debug(
                    "Adaptive batch size adjustment",
                    current=current,
                    target=target,
                    vram_gb=free_gb,
                )
                self.training_config.batch_size = target
                self._update_accumulation_steps()
        except Exception:
            pass

    def _maybe_grow_batch_size(self) -> None:
        if (
            not self.training_config.adaptive_batch_size
            or not torch.cuda.is_available()
        ):
            return
        try:
            free, total = torch.cuda.mem_get_info()
            used_ratio = 1.0 - (free / total)
            if used_ratio < self.training_config.target_gpu_mem_util:
                current = self.training_config.batch_size
                max_bs = self.training_config.max_batch_size
                new_bs = min(
                    max_bs, int(current * self.training_config.batch_growth_factor)
                )
                if new_bs > current:
                    logger.debug(
                        "Increasing batch size",
                        current=current,
                        new_bs=new_bs,
                        usage=used_ratio,
                    )
                    self.training_config.batch_size = new_bs
                    self._update_accumulation_steps()
        except Exception:
            pass

    def _build_train_loader(self, dataset: TripleDataset) -> DataLoader:
        num_workers = self.training_config.num_workers

        if num_workers == -1:
            num_workers = get_memory_safe_workers(
                get_auto_dataloader_workers(
                    len(dataset),
                    self.training_config.batch_size,
                    **self.training_config.num_workers_heuristic,
                )
            )
            logger.info(f"Auto-detected DataLoader workers: {num_workers}")

        has_workers = num_workers > 0

        prefetch_factor = (
            self.training_config.dataloader_prefetch_factor if has_workers else None
        )
        persistent_workers = (
            self.training_config.dataloader_persistent_workers if has_workers else False
        )
        return DataLoader(
            dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.training_config.pin_memory,
            drop_last=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    def _build_filter_dict(
        self, train_triples: np.ndarray, valid_triples: np.ndarray
    ) -> None:
        self._filter_arrays = {}
        if train_triples.size == 0 and valid_triples.size == 0:
            return

        combined = np.concatenate([train_triples, valid_triples], axis=0)
        store = TripleStoreSoA(combined.shape[0])
        store.load_from_triples(combined.astype(np.int32, copy=False))

        order = store.spo_index
        h_sorted = store.subjects[order]
        r_sorted = store.predicates[order]
        t_sorted = store.objects[order]

        mask = find_unique_triples_mask_numba(h_sorted, r_sorted, t_sorted)
        h_unique = h_sorted[mask].astype(np.int64)
        r_unique = r_sorted[mask].astype(np.int64)
        t_unique = t_sorted[mask]

        u_keys = (h_unique << 32) | r_unique
        unique_keys, first_idx, counts = np.unique(
            u_keys, return_index=True, return_counts=True
        )

        for key, start, count in zip(unique_keys, first_idx, counts):
            h, r = int(key >> 32), int(key & 0xFFFFFFFF)
            tails = t_unique[start : start + count]
            self._filter_arrays[(h, r)] = tails

    def _build_inbatch_known_positive_mask(
        self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        batch_size = len(h)
        mask = torch.zeros(
            (batch_size, batch_size), dtype=torch.bool, device=self.device
        )
        keys = torch.stack([h, r], dim=1)
        u_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)

        for idx, (h_id, r_id) in enumerate(u_keys):
            key = (int(h_id), int(r_id))
            known_arr = self._filter_arrays.get(key)
            if known_arr is None:
                continue

            if key not in self._filter_tensors:
                self._filter_tensors[key] = torch.from_numpy(known_arr).to(self.device)

            known_t = self._filter_tensors[key]
            match = torch.isin(t, known_t)

            if match.any():
                rows = (inverse == idx).nonzero().flatten()
                mask[rows.unsqueeze(1), match.nonzero().flatten().unsqueeze(0)] = True
        return mask

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> dict[str, float]:
        """Run one training epoch and return metrics.

        Args:
            train_loader: DataLoader for training triples.
            epoch: Current epoch number.

        Returns:
            Dict with loss and ELBO/PC2 metrics for dashboard.
        """
        self.model.train()
        total_loss = torch.zeros((), device=self.device)
        num_batches = 0
        hb_interval = self.training_config.train_heartbeat_interval_s
        last_hb = time.perf_counter()

        kl_weight = self._get_kl_weight(epoch)
        self.model.config.kl_weight = kl_weight
        reg_scale = self._get_regularization_scale(epoch)

        self.optimizer.zero_grad(set_to_none=True)
        pending_step = False

        last_losses: dict[str, Any] = {}

        for batch_idx, (batch_cpu, indices_cpu) in enumerate(train_loader):
            check_interruption()
            kp_mask = None
            batch = batch_cpu.to(self.device, non_blocking=True)
            indices = indices_cpu.to(self.device, non_blocking=True)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            if self.scaler:
                with torch.amp.autocast(device_type=self.device.type):  # type: ignore[attr-defined]
                    losses = self.model.compute_loss(
                        h,
                        r,
                        t,
                        use_inbatch_negatives=True,
                        entity_temperature=self.current_temperature,
                        regularization_scale=reg_scale,
                        known_positive_mask=kp_mask,
                        triple_indices=indices,
                    )
                    loss = losses["loss"] / self.accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                losses = self.model.compute_loss(
                    h,
                    r,
                    t,
                    use_inbatch_negatives=True,
                    entity_temperature=self.current_temperature,
                    regularization_scale=reg_scale,
                    known_positive_mask=kp_mask,
                    triple_indices=indices,
                )
                loss = losses["loss"] / self.accumulation_steps
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            total_loss += losses["loss"].detach()
            num_batches += 1
            pending_step = True
            last_losses = losses

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self._optimizer_step()
                pending_step = False

            now = time.perf_counter()
            if now - last_hb >= hb_interval:
                avg = total_loss.item() / num_batches
                logger.info(
                    "Progresso do treinamento",
                    epoch=epoch + 1,
                    batch=batch_idx + 1,
                    total_batches=len(train_loader),
                    loss=avg,
                )
                last_hb = now

            for obs in self.observers:
                obs.on_batch_end(epoch, batch_idx, losses["loss"].item())

        if pending_step:
            self._optimizer_step()

        avg_loss = total_loss.item() / num_batches if num_batches > 0 else 0.0

        result: dict[str, float] = {"loss": avg_loss}
        if last_losses:
            result.update(self._collect_elbo_metrics(last_losses, epoch))
            result.update(self._collect_pc2_metrics(last_losses))
        result.update(self._collect_structural_metrics())

        return result

    def _validate(self, valid_triples: torch.Tensor) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            metrics = self.model.evaluate(
                valid_triples,
                batch_size=self.training_config.eval_batch_size,
                refresh_cache=self.training_config.refresh_cache_on_val
                or not self._entity_cache_ready,
                filter_fn=self._mask_known_tails,
                rerank_top_k=self.training_config.rerank_top_k,
                use_faiss_eval=self.training_config.use_faiss_eval,
                faiss_candidate_k=self.training_config.faiss_candidate_k,
                score_all_tails_chunk_size=self.training_config.score_all_tails_chunk_size,
            )
            self._entity_cache_ready = True
        return metrics

    def _compute_binary_metrics_internal(
        self, val_triples: np.ndarray
    ) -> dict[str, float]:
        """Compute MCC and other binary metrics for HPO pruning."""
        if val_triples is None or len(val_triples) == 0:
            return {"mcc": 0.0}

        try:
            from sklearn.metrics import matthews_corrcoef
        except ImportError:
            return {"mcc": 0.0}

        num_negatives = 5
        max_samples = 2000

        n_pos = len(val_triples)
        if n_pos > max_samples:
            indices = self.rng.choice(n_pos, max_samples, replace=False)
            pos_triples = val_triples[indices]
        else:
            pos_triples = val_triples

        n_pos = len(pos_triples)
        n_neg = n_pos * num_negatives

        neg_triples = np.repeat(pos_triples, num_negatives, axis=0)
        mask = self.rng.random(n_neg) < 0.5
        rand_entities = self.rng.integers(0, self.model_config.num_entities, n_neg)
        neg_triples[mask, 0] = rand_entities[mask]
        neg_triples[~mask, 2] = rand_entities[~mask]
        if self._filter_arrays:
            max_attempts = 5
            for idx in range(neg_triples.shape[0]):
                h, r, t = neg_triples[idx]
                tails = self._filter_arrays.get((int(h), int(r)))
                if tails is None:
                    continue
                if np.any(tails == t):
                    for _ in range(max_attempts):
                        replacement = self.rng.integers(
                            0, self.model_config.num_entities
                        )
                        if mask[idx]:
                            h = replacement
                        else:
                            t = replacement
                        tails = self._filter_arrays.get((int(h), int(r)))
                        if tails is None or not np.any(tails == t):
                            neg_triples[idx, 0] = h
                            neg_triples[idx, 2] = t
                            break

        pos_tensor = torch.from_numpy(pos_triples).long().to(self.device)
        neg_tensor = torch.from_numpy(neg_triples).long().to(self.device)

        self.model.eval()
        with torch.no_grad():
            pos_scores_t = self.model.score_triples_batch(pos_tensor)
            neg_scores_t = self.model.score_triples_batch(neg_tensor)

            if hasattr(pos_scores_t, "cpu"):
                pos_scores = pos_scores_t.cpu().numpy()
            else:
                pos_scores = np.array(pos_scores_t)

            if hasattr(neg_scores_t, "cpu"):
                neg_scores = neg_scores_t.cpu().numpy()
            else:
                neg_scores = np.array(neg_scores_t)

        pos_scores = np.atleast_1d(pos_scores).flatten()
        neg_scores = np.atleast_1d(neg_scores).flatten()

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return {"mcc": 0.0}

        all_scores = np.concatenate([pos_scores, neg_scores])
        all_labels = np.concatenate(
            [np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
        )

        thresholds = np.percentile(all_scores, np.linspace(0, 100, 20))
        best_mcc = -1.0

        for t in thresholds:
            preds = (all_scores > t).astype(int)
            mcc = matthews_corrcoef(all_labels, preds)
            if mcc > best_mcc:
                best_mcc = mcc

        return {"mcc": float(best_mcc)}

    def _load_triples_optimized(self, path: str | Path) -> np.ndarray:
        """Load triples from Arrow/Parquet with zero-copy mapping when possible."""
        path_obj = Path(path)
        fm = FileManager()
        df_raw = fm.read(path_obj, return_native=True)

        import polars as pl

        if hasattr(df_raw, "to_native"):
            df = df_raw.to_native()
        else:
            df = df_raw

        if not isinstance(df, pl.DataFrame):
            df = pl.DataFrame(df)

        if df.width > 3:
            df = df.select(["h", "r", "t"])
        return df.to_numpy().astype(np.int64)

    def train(
        self,
        train_triples: np.ndarray | str | Path,
        valid_triples: np.ndarray | str | Path,
        *,
        trial: Any | None = None,
    ) -> dict[str, Any]:
        _configure_scheduler_warnings()
        import optuna

        if isinstance(train_triples, (str, Path)):
            train_triples = self._load_triples_optimized(train_triples)
        if isinstance(valid_triples, (str, Path)):
            valid_triples = self._load_triples_optimized(valid_triples)

        self._resolve_adaptive_batch_size()
        train_loader = self._build_train_loader(TripleDataset(train_triples))
        valid_tensor = torch.from_numpy(valid_triples).long().to(self.device)
        self._build_filter_dict(train_triples, valid_triples)

        self._train_triples_count = len(train_triples)

        logger.info(
            "Iniciando treinamento DSLFM-KGC",
            epocas=self.training_config.epochs,
            treino=len(train_triples),
            validacao=len(valid_triples),
        )

        for obs in self.observers:
            obs.on_training_start(self.training_config)

        stats = {
            "epochs_trained": 0,
            "best_epoch": 0,
            "best_val_mrr": 0.0,
            "best_val_mcc": 0.0,
            "training_losses": [],
        }
        start_time = time.time()

        try:
            for epoch in progress_bar(
                range(self.training_config.epochs),
                total=self.training_config.epochs,
                desc="DSLFM Training",
            ):
                if should_stop():
                    logger.warning(
                        "Training interrupted by stop signal",
                        stop_reason="user_interrupted",
                        epoch=epoch,
                    )
                    break

                self.current_epoch = epoch
                for obs in self.observers:
                    obs.on_epoch_start(epoch)

                train_metrics = self._train_epoch(train_loader, epoch)
                epoch_loss = train_metrics.get("loss", 0.0)
                stats["training_losses"].append(epoch_loss)

                val_metrics = {}
                if (epoch + 1) % self.training_config.validate_every == 0 or epoch == 0:
                    val_metrics = self._validate(valid_tensor)

                    binary_metrics = self._compute_binary_metrics_internal(
                        valid_triples
                    )
                    val_metrics.update(binary_metrics)

                    mcc = val_metrics.get("mcc", 0.0)
                    mrr = val_metrics.get("mrr", 0.0)
                    improved_mcc = (
                        mcc > self.best_val_mcc + self.training_config.min_delta
                    )
                    if improved_mcc:
                        self.best_val_mcc = mcc
                        stats["best_val_mcc"] = mcc

                    improved_mrr = (
                        mrr > self.best_val_mrr + self.training_config.min_delta
                    )
                    if improved_mrr:
                        self.best_val_mrr = mrr
                        stats["best_val_mrr"] = mrr

                    if improved_mcc or improved_mrr:
                        stats["best_epoch"] = epoch + 1
                        stats["best_metrics"] = val_metrics.copy()
                        self.patience_counter = 0
                        self._save_checkpoint("best_model.pt")
                    else:
                        self.patience_counter += 1

                    if trial:
                        trial.report(mrr, epoch)
                        if trial.should_prune():
                            logger.info(
                                "Trial podado pelo Optuna",
                                stop_reason="pruning",
                                epoch=epoch + 1,
                                mrr=mrr,
                            )
                            raise optuna.TrialPruned()

                epoch_metrics = {**train_metrics, **val_metrics}
                for obs in self.observers:
                    obs.on_epoch_end(epoch, epoch_metrics)

                if (
                    self.patience_counter
                    >= self.training_config.early_stopping_patience
                ):
                    logger.info(
                        "Parada antecipada por paciencia",
                        stop_reason="early_stopping",
                        epoch=epoch + 1,
                        patience=self.training_config.early_stopping_patience,
                    )
                    break
                if self.time_estimator.check_budget(epoch, loss=epoch_loss):
                    logger.warning(
                        "Time budget exceeded",
                        stop_reason="time_budget",
                        epoch=epoch,
                        budget_config=self.training_config.time_budget,
                    )

                    if trial:
                        raise optuna.TrialPruned("Time budget exceeded")
                    break
                self._anneal_temperature()
                self._maybe_grow_batch_size()

        except optuna.TrialPruned:
            stats["epochs_trained"] = self.current_epoch + 1
            stats["training_time"] = time.time() - start_time

            if "best_metrics" not in stats:
                stats["best_metrics"] = {
                    "mcc": self.best_val_mcc,
                    "mrr": self.best_val_mrr,
                }
            raise

        stats["epochs_trained"] = self.current_epoch + 1
        stats["training_time"] = time.time() - start_time

        self.model.negative_sampler.save_persistence()

        for obs in self.observers:
            obs.on_training_end(stats)
        return stats

    def _save_checkpoint(self, filename: str) -> None:
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.current_epoch,
            "best_val_mrr": self.best_val_mrr,
            "best_val_mcc": self.best_val_mcc,
            "global_step": self.global_step,
        }
        self.persistence_port.save_checkpoint(payload, filename)

    def _load_checkpoint(self, filename: str) -> None:
        ckpt = self.persistence_port.load_checkpoint(filename, map_location=self.device)
        if ckpt:
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.current_epoch = ckpt.get("epoch", 0)
            self.best_val_mrr = ckpt.get("best_val_mrr", 0.0)
            self.best_val_mcc = ckpt.get("best_val_mcc", 0.0)
            self.global_step = ckpt.get("global_step", 0)

    def _optimizer_step(self) -> None:
        if not self.scaler:
            for param in self.model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise RuntimeError("Non-finite gradient norm detected")

        if self.training_config.max_grad_norm:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.training_config.max_grad_norm
            )

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        self._step_scheduler()
        self.global_step += 1
        self._maybe_flush_cuda_cache()

    def _maybe_flush_cuda_cache(self) -> None:
        """Flush CUDA cache if memory pressure is high."""
        if self.device.type != "cuda":
            return
        if not self.training_config.cuda_cache_flush_enabled:
            return

        flush_steps = self.training_config.cuda_cache_flush_steps
        if flush_steps > 0 and self.global_step % flush_steps != 0:
            return

        try:
            free, total = torch.cuda.mem_get_info(self.device)
            free_ratio = free / total if total > 0 else 1.0
            if free_ratio < self.training_config.cuda_cache_flush_free_ratio_low:
                torch.cuda.empty_cache()
        except RuntimeError:
            pass

    def _step_scheduler(self) -> None:
        if self.scheduler:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

    def _mask_known_tails(
        self,
        scores: torch.Tensor,
        h: torch.Tensor,
        r: torch.Tensor,
        candidates: torch.Tensor,
        t: torch.Tensor,
        correction_only: bool = False,
    ) -> torch.Tensor:
        if not self._filter_arrays:
            return (
                scores
                if not correction_only
                else torch.zeros(len(h), dtype=torch.int32, device=scores.device)
            )

        if correction_only:
            correction = torch.zeros(len(h), device=scores.device, dtype=torch.int32)
            keys = torch.stack([h, r], dim=1)
            u_keys, inv = torch.unique(keys, dim=0, return_inverse=True)
            for i, (h_id, r_id) in enumerate(u_keys):
                key = (int(h_id), int(r_id))
                known = self._filter_arrays.get(key)
                if known is None:
                    continue
                rows = (inv == i).nonzero().flatten()
                k_scores = self.model.score_triples_batch(
                    torch.stack(
                        [
                            h[rows[0]].expand(len(known)),
                            r[rows[0]].expand(len(known)),
                            torch.from_numpy(known).to(scores.device),
                        ],
                        dim=1,
                    )
                )
                mask = torch.from_numpy(known).to(scores.device).unsqueeze(0) != t[
                    rows
                ].unsqueeze(1)
                correction[rows] = (
                    ((k_scores.unsqueeze(0) > scores[rows].unsqueeze(1)) & mask)
                    .sum(dim=1)
                    .to(torch.int32)
                )
            return correction

        device = scores.device
        if scores.numel() == 0 or candidates.numel() == 0:
            return scores

        if candidates.ndim == 1:
            offset = candidates[0].item()
            expected = torch.arange(offset, offset + scores.shape[1], device=device)
            use_contiguous = candidates.numel() == scores.shape[1] and torch.equal(
                candidates, expected
            )
        else:
            offset = 0
            use_contiguous = False

        keys = torch.stack([h, r], dim=1)
        unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)

        if use_contiguous:
            for idx, (h_id, r_id) in enumerate(unique_keys):
                key = (int(h_id), int(r_id))
                known = self._filter_arrays.get(key)
                if known is None:
                    continue

                if key not in self._filter_tensors:
                    self._filter_tensors[key] = torch.from_numpy(known).to(device)

                known_t = self._filter_tensors[key]
                rows = (inverse == idx).nonzero(as_tuple=True)[0]

                local_indices = known_t - offset
                valid_mask = (local_indices >= 0) & (local_indices < scores.shape[1])

                if not valid_mask.any():
                    continue

                valid_local_indices = local_indices[valid_mask]
                valid_known_t = known_t[valid_mask]

                for row_idx in rows:
                    true_tail = t[row_idx].item()

                    mask_to_apply = valid_known_t != true_tail
                    if mask_to_apply.any():
                        indices_to_mask = valid_local_indices[mask_to_apply]
                        scores[row_idx, indices_to_mask] = float("-inf")
            return scores

        if candidates.ndim == 1:
            candidates_matrix = candidates.unsqueeze(0).expand(scores.shape[0], -1)
        else:
            candidates_matrix = candidates

        for idx, (h_id, r_id) in enumerate(unique_keys):
            key = (int(h_id), int(r_id))
            known = self._filter_arrays.get(key)
            if known is None:
                continue

            if key not in self._filter_tensors:
                self._filter_tensors[key] = torch.from_numpy(known).to(device)

            known_t = self._filter_tensors[key]
            rows = (inverse == idx).nonzero(as_tuple=True)[0]
            if len(rows) == 0:
                continue

            cand_rows = candidates_matrix[rows]
            mask = torch.isin(cand_rows, known_t)
            if not mask.any():
                continue

            true_tails = t[rows].unsqueeze(1)
            mask = mask & (cand_rows != true_tails)
            scores[rows] = scores[rows].masked_fill(mask, float("-inf"))

        return scores


def _resolve_use_bert_setting(
    use_bert: bool | None, model_defaults: dict[str, Any]
) -> bool:
    """Resolve whether to use BERT relations based on explicit args and config defaults."""
    if use_bert is not None:
        return bool(use_bert)
    default = model_defaults.get("use_bert_relations")
    if default is None:
        default = True
    return bool(default)


def train_dslfm_kgc(
    train_triples: np.ndarray,
    valid_triples: np.ndarray,
    num_entities: int,
    num_relations: int,
    output_dir: Path | str = settings.OUTPUTS_DIR / "dslfm_kgc",
    relation_names: list[str] | None = None,
    use_bert: bool | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    file_manager = FileManager()
    from .dslfm_kgc import load_dslfm_kgc_settings

    cfg = load_dslfm_kgc_settings(file_manager, kwargs.get("config_path"))

    model_config, train_config = build_dslfm_configs(
        num_entities=num_entities,
        num_relations=num_relations,
        num_triples=len(train_triples),
        raw_settings=cfg,
        overrides=kwargs,
        checkpoint_dir=output_dir / "checkpoints",
        use_bert=use_bert,
        relation_names=relation_names,
    )

    persistence_port = kwargs.get("persistence_port")
    if persistence_port is None:
        raise ValueError("persistence_port is required for train_dslfm_kgc")

    manager = DSLFMKGCManager(
        model_config,
        train_config,
        persistence_port=persistence_port,
        relation_names=relation_names,
    )
    return manager.train(train_triples, valid_triples)
