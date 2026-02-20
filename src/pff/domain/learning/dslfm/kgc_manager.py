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
from pff_rust import TripleStoreSoA, fast_mcc_sweep, find_unique_triples_mask
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

try:
    _dynamo_disable = torch._dynamo.disable  # type: ignore[attr-defined]
except Exception:

    def _dynamo_disable(fn: Any) -> Any:
        return fn


def _configure_scheduler_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*epoch parameter in `scheduler\.step\(\)`.*",
        category=UserWarning,
        module=r"torch\.optim\.lr_scheduler",
    )


def _bind_evaluate(model: DSLFMKGCModel) -> DSLFMKGCModel:
    """Ensure evaluate stays bound on the instance for downstream access."""
    try:
        bound = DSLFMKGCModel.evaluate.__get__(model, DSLFMKGCModel)
    except AttributeError as exc:
        raise AttributeError("DSLFMKGCModel is missing evaluate()") from exc
    model.evaluate = bound  # type: ignore[method-assign]
    return model


class _CompiledModelWrapper(nn.Module):
    """Wrapper to preserve evaluate/utility methods when using torch.compile."""

    def __init__(self, base_model: DSLFMKGCModel, compiled_model: Any) -> None:
        """Execute init.



        Args:

            base_model: Input value used by this callable.

            compiled_model: Input value used by this callable.

        """

        super().__init__()
        self.base_model = base_model
        self.compiled_model = compiled_model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Execute forward.



        Args:

            *args: Additional positional arguments.

            **kwargs: Additional keyword arguments.



        Returns:

            Return value produced by the callable.

        """

        return self.compiled_model(*args, **kwargs)

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Execute evaluate.



        Args:

            *args: Additional positional arguments.

            **kwargs: Additional keyword arguments.



        Returns:

            Return value produced by the callable.

        """

        return self.base_model.evaluate(*args, **kwargs)

    def score_triples_batch(self, *args: Any, **kwargs: Any) -> Any:
        """Execute score triples batch.



        Args:

            *args: Additional positional arguments.

            **kwargs: Additional keyword arguments.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        return self.base_model.score_triples_batch(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name in {"base_model", "compiled_model"}:
            return super().__getattr__(name)
        if hasattr(self.base_model, name):
            return getattr(self.base_model, name)
        return super().__getattr__(name)

    @property
    def config(self) -> DSLFMKGCConfig:
        """Execute config.



        Returns:

            Return value produced by the callable.

        """

        return self.base_model.config


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _should_keep_compiled_model(
    *,
    eager_ms: float,
    compiled_ms: float,
    min_speedup_ratio: float,
) -> bool:
    """Return whether compiled path should be kept after probe benchmark."""
    if eager_ms <= 0.0 or compiled_ms <= 0.0:
        return False
    speedup_ratio = (eager_ms - compiled_ms) / eager_ms
    return speedup_ratio >= min_speedup_ratio


@_dynamo_disable
def _benchmark_forward_ms(
    *,
    model: DSLFMKGCModel,
    heads: torch.Tensor,
    relations: torch.Tensor,
    tails: torch.Tensor,
    warmup_steps: int,
    timed_steps: int,
    device: torch.device,
) -> float:
    """Measure mean forward latency for model(head, rel, tail)."""
    warmup = max(0, int(warmup_steps))
    steps = max(1, int(timed_steps))
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(heads, relations, tails)
        _sync_if_cuda(device)
        start = time.perf_counter()
        for _ in range(steps):
            _ = model(heads, relations, tails)
        _sync_if_cuda(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms / float(steps)


def _make_compile_probe_inputs(
    *,
    model_config: DSLFMKGCConfig,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic probe tensors for compile benchmark."""
    n = max(8, int(batch_size))
    heads = torch.arange(n, device=device, dtype=torch.long) % max(1, model_config.num_entities)
    relations = torch.arange(n, device=device, dtype=torch.long) % max(
        1, model_config.num_relations
    )
    tails = (heads + 1) % max(1, model_config.num_entities)
    return heads, relations, tails


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
    compile_auto_gate: bool = True
    compile_probe_batch_size: int = 128
    compile_probe_warmup_steps: int = 2
    compile_probe_timed_steps: int = 5
    compile_probe_min_speedup_ratio: float = 0.03
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
        """Execute init.



        Args:

            config: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self._config = config or KGCTrainingConfig()

    def with_epochs(self, value: int) -> KGCTrainingConfigBuilder:
        """Execute with epochs.



        Args:

            value: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self._config.epochs = int(value)
        return self

    def with_batch_size(self, value: int) -> KGCTrainingConfigBuilder:
        """Execute with batch size.



        Args:

            value: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self._config.batch_size = int(value)
        return self

    def with_effective_batch_size(self, value: int) -> KGCTrainingConfigBuilder:
        """Execute with effective batch size.



        Args:

            value: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self._config.effective_batch_size = int(value)
        return self

    def with_learning_rate(self, value: float) -> KGCTrainingConfigBuilder:
        """Execute with learning rate.



        Args:

            value: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self._config.learning_rate = float(value)
        return self

    def with_validate_every(self, value: int) -> KGCTrainingConfigBuilder:
        """Execute with validate every.



        Args:

            value: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self._config.validate_every = int(value)
        return self

    def with_early_stopping(
        self,
        *,
        patience: int | None = None,
        min_delta: float | None = None,
    ) -> KGCTrainingConfigBuilder:
        """Execute with early stopping.



        Args:

            patience: Optional input value.

            min_delta: Optional input value.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        if patience is not None:
            self._config.early_stopping_patience = int(patience)
        if min_delta is not None:
            self._config.min_delta = float(min_delta)
        return self

    def with_mixed_precision(self, value: bool) -> KGCTrainingConfigBuilder:
        """Execute with mixed precision.



        Args:

            value: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self._config.mixed_precision = bool(value)
        return self

    def with_time_budget(self, value: dict[str, Any]) -> KGCTrainingConfigBuilder:
        """Execute with time budget.



        Args:

            value: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self._config.time_budget = dict(value)
        return self

    def apply_overrides(self, overrides: dict[str, Any]) -> KGCTrainingConfigBuilder:
        """Execute apply overrides.



        Args:

            overrides: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        for key, value in overrides.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        return self

    def build(self) -> KGCTrainingConfig:
        """Execute build.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

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

    def _safe_dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    kgc_cfg = _safe_dict(raw_settings.get("kgc", {}))
    m_cfg = _safe_dict(kgc_cfg.get("model", {}))
    t_cfg = _safe_dict(kgc_cfg.get("training", {}))
    logic_cfg = _safe_dict(raw_settings.get("logic", {}))
    pc_cfg = _safe_dict(raw_settings.get("pc", {}))
    compile_cfg = _safe_dict(raw_settings.get("compile", {}))

    def _get(section: dict[str, Any], key: str, fallback: Any) -> Any:
        return overrides[key] if key in overrides else section.get(key, fallback)

    cuda_cache_cfg = t_cfg.get("cuda_cache_flush", {})
    if not isinstance(cuda_cache_cfg, dict):
        cuda_cache_cfg = {}
    num_workers_heuristic = t_cfg.get("num_workers_heuristic", {})
    if not isinstance(num_workers_heuristic, dict):
        num_workers_heuristic = {}

    use_bert_relations = _resolve_use_bert_setting(use_bert, m_cfg) and relation_names is not None

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
        global_negatives_refresh_steps=int(_get(m_cfg, "global_negatives_refresh_steps", 50)),
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
        train_heartbeat_interval_s=float(_get(t_cfg, "train_heartbeat_interval_s", 60.0)),
        score_all_tails_chunk_size=int(_get(t_cfg, "score_all_tails_chunk_size", 20_000)),
        mixed_precision=bool(_get(t_cfg, "mixed_precision", True)),
        use_compile=bool(_get(t_cfg, "use_compile", False)),
        compile_mode=str(_get(compile_cfg, "mode", "reduce-overhead")),
        compile_dynamic=bool(_get(compile_cfg, "dynamic", True)),
        compile_fullgraph=bool(_get(compile_cfg, "fullgraph", False)),
        compile_backend=_get(compile_cfg, "backend", None),
        compile_auto_gate=bool(_get(compile_cfg, "auto_gate", True)),
        compile_probe_batch_size=int(_get(compile_cfg, "probe_batch_size", 128)),
        compile_probe_warmup_steps=int(_get(compile_cfg, "probe_warmup_steps", 2)),
        compile_probe_timed_steps=int(_get(compile_cfg, "probe_timed_steps", 5)),
        compile_probe_min_speedup_ratio=float(_get(compile_cfg, "probe_min_speedup_ratio", 0.03)),
        optimizer_fused=_get(t_cfg, "optimizer_fused", None),
        optimizer_foreach=_get(t_cfg, "optimizer_foreach", None),
        num_workers=int(_get(t_cfg, "num_workers", 0)),
        num_workers_heuristic=dict(num_workers_heuristic),
        pin_memory=bool(_get(t_cfg, "pin_memory", True)),
        dataloader_prefetch_factor=int(_get(t_cfg, "dataloader_prefetch_factor", 4)),
        dataloader_persistent_workers=bool(_get(t_cfg, "dataloader_persistent_workers", True)),
        eval_batch_size=int(_get(t_cfg, "eval_batch_size", 256)),
        regularization_warmup_epochs=int(_get(t_cfg, "regularization_warmup_epochs", 8)),
        regularization_start_scale=float(_get(t_cfg, "regularization_start_scale", 0.0)),
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
        cuda_cache_flush_free_ratio_low=float(_get(cuda_cache_cfg, "free_ratio_low", 0.15)),
        cuda_cache_flush_free_ratio_high=float(_get(cuda_cache_cfg, "free_ratio_high", 0.4)),
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
        """Execute init.



        Args:

            triples: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        triples_arr = np.asarray(triples, dtype=np.int64)
        if not triples_arr.flags.writeable:
            triples_arr = np.array(triples_arr, copy=True)
        self.triples = torch.from_numpy(triples_arr).long()

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.triples[idx], idx


def _should_enable_fused_adamw(
    *,
    is_cuda: bool,
    optimizer_fused: bool | None,
    param_signatures: set[tuple[str, str]],
) -> bool:
    """Return whether fused AdamW is safe for the current parameter signatures."""
    # Keep fused AdamW opt-in only; default eager AdamW is more robust with mixed stacks.
    fused = bool(optimizer_fused) if optimizer_fused is not None else False
    if not is_cuda:
        return False
    if not fused:
        return False
    if len(param_signatures) > 1:
        return False
    return True


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
        """Execute init.



        Args:

            model_config: Input value used by this callable.

            training_config: Input value used by this callable.

            persistence_port: Input value used by this callable.

            relation_names: Optional input value.

            device: Optional input value.

            observers: Optional input value.

            seed: Optional input value.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

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
                torch.set_float32_matmul_precision(self.training_config.matmul_precision)

        self._update_accumulation_steps()

        base_model = _bind_evaluate(
            DSLFMKGCModel(model_config, relation_names=relation_names).to(self.device)
        )

        compile_requested = bool(training_config.use_compile and hasattr(torch, "compile"))
        if compile_requested and float(model_config.lambda_pc) > 0.0:
            logger.warning(
                "Disabling torch.compile: PC2 path contains dynamic control flow not supported by torch.compile"
            )
            compile_requested = False

        if compile_requested:
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

                compiled = cast(nn.Module, torch.compile(base_model, **compile_kwargs))  # type: ignore[call-overload]
                keep_compiled = True
                if bool(training_config.compile_auto_gate):
                    probe_batch = min(
                        int(training_config.batch_size),
                        int(training_config.compile_probe_batch_size),
                    )
                    heads, relations, tails = _make_compile_probe_inputs(
                        model_config=model_config,
                        batch_size=probe_batch,
                        device=self.device,
                    )
                    eager_ms = _benchmark_forward_ms(
                        model=base_model,
                        heads=heads,
                        relations=relations,
                        tails=tails,
                        warmup_steps=training_config.compile_probe_warmup_steps,
                        timed_steps=training_config.compile_probe_timed_steps,
                        device=self.device,
                    )
                    compiled_ms = _benchmark_forward_ms(
                        model=cast(DSLFMKGCModel, compiled),
                        heads=heads,
                        relations=relations,
                        tails=tails,
                        warmup_steps=training_config.compile_probe_warmup_steps,
                        timed_steps=training_config.compile_probe_timed_steps,
                        device=self.device,
                    )
                    keep_compiled = _should_keep_compiled_model(
                        eager_ms=eager_ms,
                        compiled_ms=compiled_ms,
                        min_speedup_ratio=float(training_config.compile_probe_min_speedup_ratio),
                    )
                    if keep_compiled:
                        logger.info(
                            f"torch.compile mantido: eager={eager_ms:.3f}ms "
                            f"compiled={compiled_ms:.3f}ms batch={probe_batch}"
                        )
                    else:
                        logger.warning(
                            f"torch.compile disabled by auto-gate (eager={eager_ms:.3f}ms, "
                            f"compiled={compiled_ms:.3f}ms, min_speedup={training_config.compile_probe_min_speedup_ratio:.3f})"
                        )
                if keep_compiled:
                    self.model = _CompiledModelWrapper(base_model, compiled)
                    logger.debug("Model compiled with torch.compile")
                else:
                    self.model = base_model  # type: ignore[assignment]
            except Exception as e:
                logger.warning("torch.compile failed, using eager mode", error=str(e))
                self.model = base_model  # type: ignore[assignment]
        else:
            self.model = base_model  # type: ignore[assignment]

        if self.model.use_bert_relations:
            self.model.precompute_relation_embeddings(self.device)

        is_cuda = self.device.type == "cuda"
        optimizer_fused = training_config.optimizer_fused
        optimizer_foreach = training_config.optimizer_foreach
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        param_signatures = {(param.device.type, str(param.dtype)) for param in trainable_params}
        fused_requested = bool(optimizer_fused) if optimizer_fused is not None else False
        fused = _should_enable_fused_adamw(
            is_cuda=is_cuda,
            optimizer_fused=optimizer_fused,
            param_signatures=param_signatures,
        )
        if fused_requested and not fused and is_cuda and len(param_signatures) > 1:
            logger.warning(
                "Disabling fused AdamW due to mixed parameter device/dtype signatures: "
                f"{sorted(param_signatures)}"
            )
            fused = False
        foreach = bool(optimizer_foreach) if optimizer_foreach is not None else not is_cuda
        if fused and foreach:
            logger.warning("AdamW fused=True is incompatible with foreach=True; disabling foreach")
            foreach = False
        self.optimizer = torch.optim.AdamW(
            trainable_params,
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
            "BERT nas relacoes" if self.model.use_bert_relations else "relacoes aprendidas"
        )
        logger.info(
            "Gerente DSLFM-KGC inicializado",
            batch=training_config.batch_size,
            effective=training_config.effective_batch_size,
            acumulacao=self.accumulation_steps,
            bert_status=bert_status,
        )

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Execute create scheduler.



        Returns:

            Return value produced by the callable.

        """

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
        """Execute get kl weight.



        Args:

            epoch: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if epoch >= self.training_config.kl_warmup_epochs:
            return self.training_config.max_kl_weight
        progress = epoch / self.training_config.kl_warmup_epochs
        return self.training_config.min_kl_weight + progress * (
            self.training_config.max_kl_weight - self.training_config.min_kl_weight
        )

    def _get_regularization_scale(self, epoch: int) -> float:
        """Execute get regularization scale.



        Args:

            epoch: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

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
                float(contrastive.item()) if hasattr(contrastive, "item") else float(contrastive)
            )

        elbo_kl = 0.0
        if kl_gaussian is not None:
            kl_g = float(kl_gaussian.item()) if hasattr(kl_gaussian, "item") else float(kl_gaussian)
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
            sp_val = float(sparsity.item()) if hasattr(sparsity, "item") else float(sparsity)
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
        """Execute update accumulation steps."""

        if self.training_config.tf32 and is_cuda_available():
            torch.set_float32_matmul_precision("medium")
            logger.debug("TF32 precision enabled for matmuls")
        self.accumulation_steps = max(
            1,
            self.training_config.effective_batch_size // self.training_config.batch_size,
        )

    def _resolve_adaptive_batch_size(self) -> None:
        """Execute resolve adaptive batch size."""

        if not self.training_config.adaptive_batch_size or not torch.cuda.is_available():
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
        """Execute maybe grow batch size."""

        if not self.training_config.adaptive_batch_size or not torch.cuda.is_available():
            return
        try:
            free, total = torch.cuda.mem_get_info()
            used_ratio = 1.0 - (free / total)
            if used_ratio < self.training_config.target_gpu_mem_util:
                current = self.training_config.batch_size
                max_bs = self.training_config.max_batch_size
                new_bs = min(max_bs, int(current * self.training_config.batch_growth_factor))
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
        """Execute build train loader.



        Args:

            dataset: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

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

        prefetch_factor = self.training_config.dataloader_prefetch_factor if has_workers else None
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

    def _build_filter_dict(self, train_triples: np.ndarray, valid_triples: np.ndarray) -> None:
        """Execute build filter dict.



        Args:

            train_triples: Input value used by this callable.

            valid_triples: Input value used by this callable.

        """

        self._filter_arrays = {}
        if train_triples.size == 0 and valid_triples.size == 0:
            return

        combined = np.concatenate([train_triples, valid_triples], axis=0)
        arr = combined.astype(np.int32, copy=False)
        store = TripleStoreSoA()
        store.load_from_arrays(
            np.ascontiguousarray(arr[:, 0]),
            np.ascontiguousarray(arr[:, 1]),
            np.ascontiguousarray(arr[:, 2]),
        )

        order = store.get_spo_index()
        h_sorted = store.get_subjects()[order]
        r_sorted = store.get_predicates()[order]
        t_sorted = store.get_objects()[order]

        mask = find_unique_triples_mask(
            h_sorted.astype(np.int64),
            r_sorted.astype(np.int64),
            t_sorted.astype(np.int64),
        )
        h_unique = h_sorted[mask].astype(np.int64)
        r_unique = r_sorted[mask].astype(np.int64)
        t_unique = t_sorted[mask]

        u_keys = (h_unique << 32) | r_unique
        unique_keys, first_idx, counts = np.unique(u_keys, return_index=True, return_counts=True)

        for key, start, count in zip(unique_keys, first_idx, counts):
            h, r = int(key >> 32), int(key & 0xFFFFFFFF)
            tails = t_unique[start : start + count]
            self._filter_arrays[(h, r)] = tails

    def _build_inbatch_known_positive_mask(
        self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Execute build inbatch known positive mask.



        Args:

            h: Input value used by this callable.

            r: Input value used by this callable.

            t: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        batch_size = len(h)
        mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=self.device)
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
        """Execute validate.



        Args:

            valid_triples: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        check_interruption()
        self.model.eval()
        with torch.no_grad():
            metrics: dict[str, float] = self.model.evaluate(
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

    def _compute_binary_metrics_internal(self, val_triples: np.ndarray) -> dict[str, float]:
        """Compute MCC and other binary metrics for HPO pruning."""
        check_interruption()
        if val_triples is None or len(val_triples) == 0:
            return {"mcc": 0.0}

        pos_triples = self._sample_binary_metric_positives(val_triples, max_samples=2000)
        neg_triples, mask = self._build_negative_triples(pos_triples, num_negatives=5)
        self._repair_negative_collisions(neg_triples, mask, max_attempts=5)
        pos_scores, neg_scores = self._score_binary_metric_triples(pos_triples, neg_triples)
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return {"mcc": 0.0}

        all_scores = np.concatenate([pos_scores, neg_scores])
        all_labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        best_mcc = self._compute_best_mcc_from_scores(all_scores, all_labels)

        return {"mcc": float(best_mcc)}

    def _sample_binary_metric_positives(
        self,
        val_triples: np.ndarray,
        *,
        max_samples: int,
    ) -> np.ndarray:
        """Execute sample binary metric positives.



        Args:

            val_triples: Input value used by this callable.

            max_samples: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        n_pos = len(val_triples)
        if n_pos <= max_samples:
            return val_triples
        indices = self.rng.choice(n_pos, max_samples, replace=False)
        return val_triples[indices]

    def _build_negative_triples(
        self,
        pos_triples: np.ndarray,
        *,
        num_negatives: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute build negative triples.



        Args:

            pos_triples: Input value used by this callable.

            num_negatives: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        n_neg = len(pos_triples) * num_negatives
        neg_triples = np.repeat(pos_triples, num_negatives, axis=0)
        mask = self.rng.random(n_neg) < 0.5
        rand_entities = self.rng.integers(0, self.model_config.num_entities, n_neg)
        neg_triples[mask, 0] = rand_entities[mask]
        neg_triples[~mask, 2] = rand_entities[~mask]
        return neg_triples, mask

    def _repair_negative_collisions(
        self,
        neg_triples: np.ndarray,
        head_mask: np.ndarray,
        *,
        max_attempts: int,
    ) -> None:
        """Execute repair negative collisions.



        Args:

            neg_triples: Input value used by this callable.

            head_mask: Input value used by this callable.

            max_attempts: Input value used by this callable.

        """

        if not self._filter_arrays:
            return

        for idx in range(neg_triples.shape[0]):
            h, r, t = neg_triples[idx]
            tails = self._filter_arrays.get((int(h), int(r)))
            if tails is None or not np.any(tails == t):
                continue

            for _ in range(max_attempts):
                replacement = self.rng.integers(0, self.model_config.num_entities)
                if head_mask[idx]:
                    h = replacement
                else:
                    t = replacement
                tails = self._filter_arrays.get((int(h), int(r)))
                if tails is None or not np.any(tails == t):
                    neg_triples[idx, 0] = h
                    neg_triples[idx, 2] = t
                    break

    def _score_binary_metric_triples(
        self,
        pos_triples: np.ndarray,
        neg_triples: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Execute score binary metric triples.



        Args:

            pos_triples: Input value used by this callable.

            neg_triples: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        pos_tensor = torch.from_numpy(pos_triples).long().to(self.device)
        neg_tensor = torch.from_numpy(neg_triples).long().to(self.device)

        self.model.eval()
        with torch.no_grad():
            pos_scores_t = self.model.score_triples_batch(pos_tensor)
            neg_scores_t = self.model.score_triples_batch(neg_tensor)

        return self._to_flat_numpy(pos_scores_t), self._to_flat_numpy(neg_scores_t)

    @staticmethod
    def _to_flat_numpy(scores: Any) -> np.ndarray:
        """Execute to flat numpy.



        Args:

            scores: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if hasattr(scores, "cpu"):
            arr = scores.cpu().numpy()
        else:
            arr = np.array(scores)
        return np.atleast_1d(arr).flatten()

    @staticmethod
    def _compute_best_mcc_from_scores(
        all_scores: np.ndarray,
        all_labels: np.ndarray,
    ) -> float:
        """Execute compute best mcc from scores.



        Args:

            all_scores: Input value used by this callable.

            all_labels: Input value used by this callable.

        Returns:

            Return value produced by the callable.

        """

        scores_arr = np.asarray(all_scores, dtype=np.float64).reshape(-1)
        labels_arr = np.asarray(all_labels, dtype=np.int64).reshape(-1)
        if scores_arr.size == 0 or labels_arr.size == 0:
            return 0.0
        if scores_arr.size != labels_arr.size:
            raise ValueError("scores and labels must have the same length")

        # Generate thresholds for MCC sweep: use unique score values
        unique_scores = np.unique(scores_arr)
        if len(unique_scores) > 1000:
            # Subsample thresholds if too many unique values
            thresholds = np.percentile(unique_scores, np.linspace(0, 100, 1000))
        else:
            thresholds = unique_scores
        thresholds = np.asarray(thresholds, dtype=np.float64)

        result = fast_mcc_sweep(labels_arr, scores_arr, thresholds)
        return float(result[0])  # Return best MCC value

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
        return np.asarray(df.to_numpy(), dtype=np.int64)

    def train(
        self,
        train_triples: np.ndarray | str | Path,
        valid_triples: np.ndarray | str | Path,
        *,
        trial: Any | None = None,
    ) -> dict[str, Any]:
        """Execute train.



        Args:

            train_triples: Input value used by this callable.

            valid_triples: Input value used by this callable.

            trial: Optional input value.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        _configure_scheduler_warnings()
        import optuna

        train_triples = self._coerce_triples_input(train_triples)
        valid_triples = self._coerce_triples_input(valid_triples)
        train_loader, valid_tensor = self._initialize_training_inputs(train_triples, valid_triples)
        stats = self._initialize_training_stats()
        start_time = time.time()

        try:
            for epoch in progress_bar(
                range(self.training_config.epochs),
                total=self.training_config.epochs,
                desc="DSLFM Training",
            ):
                if self._handle_training_stop_signal(epoch):
                    break

                train_metrics = self._run_train_epoch_with_observers(train_loader, epoch)
                epoch_loss = train_metrics.get("loss", 0.0)
                stats["training_losses"].append(epoch_loss)

                val_metrics = self._maybe_validate_epoch(
                    epoch=epoch,
                    valid_tensor=valid_tensor,
                    valid_triples=valid_triples,
                    stats=stats,
                    trial=trial,
                    optuna_module=optuna,
                )

                epoch_metrics = {**train_metrics, **val_metrics}
                for obs in self.observers:
                    obs.on_epoch_end(epoch, epoch_metrics)

                if self._should_stop_after_epoch(
                    epoch=epoch,
                    epoch_loss=epoch_loss,
                    trial=trial,
                    optuna_module=optuna,
                ):
                    break
                self._anneal_temperature()
                self._maybe_grow_batch_size()

        except optuna.TrialPruned:
            self._finalize_pruned_stats(stats, start_time)
            raise

        return self._finalize_training_stats(stats, start_time)

    def _coerce_triples_input(self, triples: np.ndarray | str | Path) -> np.ndarray:
        """Execute coerce triples input.



        Args:

            triples: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if isinstance(triples, (str, Path)):
            return self._load_triples_optimized(triples)
        return triples

    def _initialize_training_inputs(
        self,
        train_triples: np.ndarray,
        valid_triples: np.ndarray,
    ) -> tuple[DataLoader, torch.Tensor]:
        """Execute initialize training inputs.



        Args:

            train_triples: Input value used by this callable.

            valid_triples: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

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
        for observer in self.observers:
            observer.on_training_start(self.training_config)
        return train_loader, valid_tensor

    @staticmethod
    def _initialize_training_stats() -> dict[str, Any]:
        return {
            "epochs_trained": 0,
            "best_epoch": 0,
            "best_val_mrr": 0.0,
            "best_val_mcc": 0.0,
            "training_losses": [],
        }

    def _handle_training_stop_signal(self, epoch: int) -> bool:
        """Execute handle training stop signal.



        Args:

            epoch: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if not should_stop():
            return False
        logger.warning(
            "Training interrupted by stop signal",
            stop_reason="user_interrupted",
            epoch=epoch,
        )
        return True

    def _run_train_epoch_with_observers(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Execute run train epoch with observers.



        Args:

            train_loader: Input value used by this callable.

            epoch: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        self.current_epoch = epoch
        for observer in self.observers:
            observer.on_epoch_start(epoch)
        return self._train_epoch(train_loader, epoch)

    def _maybe_validate_epoch(
        self,
        *,
        epoch: int,
        valid_tensor: torch.Tensor,
        valid_triples: np.ndarray,
        stats: dict[str, Any],
        trial: Any | None,
        optuna_module: Any,
    ) -> dict[str, float]:
        """Execute maybe validate epoch.



        Args:

            epoch: Input value used by this callable.

            valid_tensor: Input value used by this callable.

            valid_triples: Input value used by this callable.

            stats: Input value used by this callable.

            trial: Input value used by this callable.

            optuna_module: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        should_validate = (epoch + 1) % self.training_config.validate_every == 0 or epoch == 0
        if not should_validate:
            return {}

        val_metrics = self._validate(valid_tensor)
        val_metrics.update(self._compute_binary_metrics_internal(valid_triples))
        self._update_best_validation_metrics(val_metrics, stats, epoch)
        self._maybe_prune_trial(trial, val_metrics.get("mrr", 0.0), epoch, optuna_module)
        return val_metrics

    def _update_best_validation_metrics(
        self,
        val_metrics: dict[str, float],
        stats: dict[str, Any],
        epoch: int,
    ) -> None:
        """Execute update best validation metrics.



        Args:

            val_metrics: Input value used by this callable.

            stats: Input value used by this callable.

            epoch: Input value used by this callable.

        """

        mcc = val_metrics.get("mcc", 0.0)
        mrr = val_metrics.get("mrr", 0.0)
        improved_mcc = mcc > self.best_val_mcc + self.training_config.min_delta
        improved_mrr = mrr > self.best_val_mrr + self.training_config.min_delta

        if improved_mcc:
            self.best_val_mcc = mcc
            stats["best_val_mcc"] = mcc
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

    def _maybe_prune_trial(
        self,
        trial: Any | None,
        mrr: float,
        epoch: int,
        optuna_module: Any,
    ) -> None:
        """Execute maybe prune trial.



        Args:

            trial: Input value used by this callable.

            mrr: Input value used by this callable.

            epoch: Input value used by this callable.

            optuna_module: Input value used by this callable.



        Raises:

            Exception: Propagates domain-specific failures with context.

        """

        if not trial:
            return
        trial.report(mrr, epoch)
        if trial.should_prune():
            logger.info(
                "Trial podado pelo Optuna",
                stop_reason="pruning",
                epoch=epoch + 1,
                mrr=mrr,
            )
            raise optuna_module.TrialPruned()

    def _should_stop_after_epoch(
        self,
        *,
        epoch: int,
        epoch_loss: float,
        trial: Any | None,
        optuna_module: Any,
    ) -> bool:
        """Execute should stop after epoch.



        Args:

            epoch: Input value used by this callable.

            epoch_loss: Input value used by this callable.

            trial: Input value used by this callable.

            optuna_module: Input value used by this callable.



        Returns:

            Return value produced by the callable.



        Raises:

            Exception: Propagates domain-specific failures with context.

        """

        if self.patience_counter >= self.training_config.early_stopping_patience:
            logger.info(
                "Parada antecipada por paciencia",
                stop_reason="early_stopping",
                epoch=epoch + 1,
                patience=self.training_config.early_stopping_patience,
            )
            return True

        if not self.time_estimator.check_budget(epoch, loss=epoch_loss):
            return False

        logger.warning(
            "Time budget exceeded",
            stop_reason="time_budget",
            epoch=epoch,
            budget_config=self.training_config.time_budget,
        )
        if trial:
            raise optuna_module.TrialPruned("Time budget exceeded")
        return True

    def _finalize_pruned_stats(self, stats: dict[str, Any], start_time: float) -> None:
        """Execute finalize pruned stats.



        Args:

            stats: Input value used by this callable.

            start_time: Input value used by this callable.

        """

        stats["epochs_trained"] = self.current_epoch + 1
        stats["training_time"] = time.time() - start_time
        if "best_metrics" not in stats:
            stats["best_metrics"] = {
                "mcc": self.best_val_mcc,
                "mrr": self.best_val_mrr,
            }

    def _finalize_training_stats(self, stats: dict[str, Any], start_time: float) -> dict[str, Any]:
        """Execute finalize training stats.



        Args:

            stats: Input value used by this callable.

            start_time: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        stats["epochs_trained"] = self.current_epoch + 1
        stats["training_time"] = time.time() - start_time
        self.model.negative_sampler.save_persistence()
        for observer in self.observers:
            observer.on_training_end(stats)
        return stats

    def _save_checkpoint(self, filename: str) -> None:
        """Execute save checkpoint.



        Args:

            filename: Input value used by this callable.

        """

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
        """Execute load checkpoint.



        Args:

            filename: Input value used by this callable.

        """

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
        """Execute optimizer step.



        Raises:

            Exception: Propagates domain-specific failures with context.

        """

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

        if self.device.type == "cuda":
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    grad = param.grad
                    if grad is None:
                        continue
                    if grad.device != param.device or grad.dtype != param.dtype:
                        param.grad = grad.to(
                            device=param.device,
                            dtype=param.dtype,
                            non_blocking=True,
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
        """Execute step scheduler."""

        if self.scheduler:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*epoch parameter in `scheduler\.step\(\)`.*",
                    category=UserWarning,
                    module=r"torch\.optim\.lr_scheduler",
                )
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
        """Execute mask known tails.



        Args:

            scores: Input value used by this callable.

            h: Input value used by this callable.

            r: Input value used by this callable.

            candidates: Input value used by this callable.

            t: Input value used by this callable.

            correction_only: Optional input value.



        Returns:

            Return value produced by the callable.

        """

        if not self._filter_arrays:
            return (
                scores
                if not correction_only
                else torch.zeros(len(h), dtype=torch.int32, device=scores.device)
            )

        if correction_only:
            return self._mask_known_tails_correction(scores, h, r, t)

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
            self._mask_known_tails_contiguous(
                scores=scores,
                t=t,
                unique_keys=unique_keys,
                inverse=inverse,
                device=device,
                offset=int(offset),
            )
            return scores

        if candidates.ndim == 1:
            candidates_matrix = candidates.unsqueeze(0).expand(scores.shape[0], -1)
        else:
            candidates_matrix = candidates

        self._mask_known_tails_general(
            scores=scores,
            t=t,
            candidates_matrix=candidates_matrix,
            unique_keys=unique_keys,
            inverse=inverse,
            device=device,
        )
        return scores

    def _mask_known_tails_correction(
        self,
        scores: torch.Tensor,
        h: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Execute mask known tails correction.



        Args:

            scores: Input value used by this callable.

            h: Input value used by this callable.

            r: Input value used by this callable.

            t: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        correction = torch.zeros(len(h), device=scores.device, dtype=torch.int32)
        keys = torch.stack([h, r], dim=1)
        unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
        for idx, (h_id, r_id) in enumerate(unique_keys):
            key = (int(h_id), int(r_id))
            known = self._filter_arrays.get(key)
            if known is None:
                continue
            rows = (inverse == idx).nonzero().flatten()
            known_t = torch.from_numpy(known).to(scores.device)
            k_scores = self.model.score_triples_batch(
                torch.stack(
                    [
                        h[rows[0]].expand(len(known)),
                        r[rows[0]].expand(len(known)),
                        known_t,
                    ],
                    dim=1,
                )
            )
            mask = known_t.unsqueeze(0) != t[rows].unsqueeze(1)
            correction[rows] = (
                ((k_scores.unsqueeze(0) > scores[rows].unsqueeze(1)) & mask)
                .sum(dim=1)
                .to(torch.int32)
            )
        return correction

    def _get_known_tails_tensor(
        self,
        key: tuple[int, int],
        known: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        """Execute get known tails tensor.



        Args:

            key: Input value used by this callable.

            known: Input value used by this callable.

            device: Input value used by this callable.



        Returns:

            Return value produced by the callable.

        """

        if key not in self._filter_tensors:
            self._filter_tensors[key] = torch.from_numpy(known).to(device)
        return self._filter_tensors[key]

    def _mask_known_tails_contiguous(
        self,
        *,
        scores: torch.Tensor,
        t: torch.Tensor,
        unique_keys: torch.Tensor,
        inverse: torch.Tensor,
        device: torch.device,
        offset: int,
    ) -> None:
        """Execute mask known tails contiguous.



        Args:

            scores: Input value used by this callable.

            t: Input value used by this callable.

            unique_keys: Input value used by this callable.

            inverse: Input value used by this callable.

            device: Input value used by this callable.

            offset: Input value used by this callable.

        """

        for idx, (h_id, r_id) in enumerate(unique_keys):
            key = (int(h_id), int(r_id))
            known = self._filter_arrays.get(key)
            if known is None:
                continue
            known_t = self._get_known_tails_tensor(key, known, device)
            rows = (inverse == idx).nonzero(as_tuple=True)[0]
            if len(rows) == 0:
                continue
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

    def _mask_known_tails_general(
        self,
        *,
        scores: torch.Tensor,
        t: torch.Tensor,
        candidates_matrix: torch.Tensor,
        unique_keys: torch.Tensor,
        inverse: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Execute mask known tails general.



        Args:

            scores: Input value used by this callable.

            t: Input value used by this callable.

            candidates_matrix: Input value used by this callable.

            unique_keys: Input value used by this callable.

            inverse: Input value used by this callable.

            device: Input value used by this callable.

        """

        for idx, (h_id, r_id) in enumerate(unique_keys):
            key = (int(h_id), int(r_id))
            known = self._filter_arrays.get(key)
            if known is None:
                continue
            known_t = self._get_known_tails_tensor(key, known, device)
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


def _resolve_use_bert_setting(use_bert: bool | None, model_defaults: dict[str, Any]) -> bool:
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
    """Execute train dslfm kgc.



    Args:

        train_triples: Input value used by this callable.

        valid_triples: Input value used by this callable.

        num_entities: Input value used by this callable.

        num_relations: Input value used by this callable.

        output_dir: Optional input value.

        relation_names: Optional input value.

        use_bert: Optional input value.

        **kwargs: Additional keyword arguments.



    Returns:

        Return value produced by the callable.



    Raises:

        Exception: Propagates domain-specific failures with context.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

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
