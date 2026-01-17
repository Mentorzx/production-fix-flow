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

import io
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pff.shared.core.config import settings
from pff.domain.learning.ml.training_observer import TrainingObserver
from pff.domain.learning.dslfm.time_estimator import (
    TimeBudgetConfig,
    TimeBudgetEstimator,
)
from pff.shared.acceleration.concurrency import progress_bar
from pff.shared.acceleration.numba_kernels import (
    TripleStoreSoA,
    find_unique_triples_mask_numba,
)
from pff.shared.core.file_manager import FileManager
from pff.shared.core.logger import logger
from pff.shared.ops.global_interrupt_manager import check_interruption
from pff.shared.system.cuda import is_cuda_available
from pff.shared.system.resource_manager import (
    get_auto_dataloader_workers,
    get_memory_safe_workers,
)

from .dslfm_kgc import DSLFMKGCModel, DSLFMKGCConfig

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
    model.evaluate = bound  # type: ignore[method-assign]
    return model


class _CompiledModelWrapper(nn.Module):
    """Wrapper to preserve evaluate/utility methods when using torch.compile."""

    def __init__(self, base_model: DSLFMKGCModel, compiled_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        self.compiled_model = compiled_model

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401 - delegated
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


class TripleDataset(Dataset):
    """Simple dataset for triples with indices."""

    def __init__(self, triples: np.ndarray) -> None:
        triples_arr = np.asarray(triples)
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
        relation_names: list[str] | None = None,
        device: torch.device | None = None,
        observers: list[TrainingObserver] | None = None,
        seed: int | None = None,
    ) -> None:
        self.model_config = model_config
        self.training_config = training_config
        self.observers = observers or []
        self.device = device or torch.device("cuda" if is_cuda_available() else "cpu")
        self.rng = np.random.default_rng(seed)

        if self.device.type == "cuda":
            allow_tf32 = bool(self.training_config.allow_tf32)
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision(self.training_config.matmul_precision)

        self.file_manager = FileManager()
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

                compiled = torch.compile(base_model, **compile_kwargs)
                # Suppress LSP error: torch.compile returns a Callable that behaves like a Module but isn't strictly typed as one
                self.model = _CompiledModelWrapper(base_model, compiled)  # type: ignore
                logger.info("Modelo compilado com torch.compile")
            except Exception as e:
                logger.warning("Compilacao torch.compile falhou, usando modo eager", error=str(e))
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
        foreach = bool(optimizer_foreach) if optimizer_foreach is not None else not is_cuda
        if fused and foreach:
            logger.warning("AdamW fused=True is incompatible with foreach=True; disabling foreach")
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
        self.scaler = torch.cuda.amp.GradScaler() if use_scaler else None

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_mrr = float("-inf")
        self.best_val_mcc = float("-inf")
        self.patience_counter = 0
        self.current_temperature = training_config.temperature
        self._filter_arrays: dict[tuple[int, int], np.ndarray] = {}
        self._filter_tensors: dict[tuple[int, int], torch.Tensor] = {}
        self._entity_cache_ready = False

        self.checkpoint_dir = training_config.checkpoint_dir
        self.file_manager.ensure_dir(self.checkpoint_dir)

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
            f"Gerente DSLFM-KGC inicializado: batch={training_config.batch_size}, "
            f"efetivo={training_config.effective_batch_size}, "
            f"acumulacao={self.accumulation_steps}, {bert_status}"
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
            "pc2_latency": 0.0,
            "pc2_density": pc_density,
        }

    def _collect_structural_metrics(self) -> dict[str, float]:
        """Collect structural metrics for dashboard.

        Returns:
            Dict with graphDensity and estimated numClusters.
        """
        num_entities = self.model_config.num_entities
        num_relations = self.model_config.num_relations

        # Get num_triples from training data if available
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
        self.accumulation_steps = max(
            1,
            self.training_config.effective_batch_size // self.training_config.batch_size,
        )

    def _resolve_adaptive_batch_size(self) -> None:
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
                logger.info(
                    f"Ajuste adaptativo de batch: {current} -> {target} (VRAM livre={free_gb:.1f}GB)"
                )
                self.training_config.batch_size = target
                self._update_accumulation_steps()
        except Exception:
            pass

    def _maybe_grow_batch_size(self) -> None:
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
                    logger.info(f"Aumentando batch: {current} -> {new_bs} (uso={used_ratio:.2f})")
                    self.training_config.batch_size = new_bs
                    self._update_accumulation_steps()
        except Exception:
            pass

    def _build_train_loader(self, dataset: TripleDataset) -> DataLoader:
        num_workers = self.training_config.num_workers
        if num_workers == 0:
            num_workers = get_memory_safe_workers(
                get_auto_dataloader_workers(
                    len(dataset),
                    self.training_config.batch_size,
                    **self.training_config.num_workers_heuristic,
                )
            )
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
        unique_keys, first_idx, counts = np.unique(u_keys, return_index=True, return_counts=True)

        for key, start, count in zip(unique_keys, first_idx, counts):
            h, r = int(key >> 32), int(key & 0xFFFFFFFF)
            tails = t_unique[start : start + count]
            self._filter_arrays[(h, r)] = tails

    def _build_inbatch_known_positive_mask(
        self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        batch_size = len(h)
        mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=self.device)
        keys = torch.stack([h, r], dim=1)
        u_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)

        for idx, (h_id, r_id) in enumerate(u_keys):
            key = (int(h_id), int(r_id))
            known_arr = self._filter_arrays.get(key)
            if known_arr is None:
                continue

            # Use cached GPU tensor if available
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
                with torch.cuda.amp.autocast():
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
                    f"Epoca {epoch + 1}: {batch_idx + 1}/{len(train_loader)} lotes, loss={avg:.4f}"
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

    def _compute_binary_metrics_internal(self, val_triples: np.ndarray) -> dict[str, float]:
        """Compute MCC and other binary metrics for HPO pruning."""
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

        # Negative sampling
        neg_triples = np.repeat(pos_triples, num_negatives, axis=0)
        mask = self.rng.random(n_neg) < 0.5
        rand_entities = self.rng.integers(0, self.model_config.num_entities, n_neg)
        neg_triples[mask, 0] = rand_entities[mask]
        neg_triples[~mask, 2] = rand_entities[~mask]

        pos_tensor = torch.from_numpy(pos_triples).long().to(self.device)
        neg_tensor = torch.from_numpy(neg_triples).long().to(self.device)

        self.model.eval()
        with torch.no_grad():
            pos_scores = self.model.score_triples_batch(pos_tensor).cpu().numpy()
            neg_scores = self.model.score_triples_batch(neg_tensor).cpu().numpy()

        all_scores = np.concatenate([pos_scores, neg_scores])
        all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

        thresholds = np.percentile(all_scores, np.linspace(0, 100, 20))
        best_mcc = -1.0

        for t in thresholds:
            preds = (all_scores > t).astype(int)
            mcc = matthews_corrcoef(all_labels, preds)
            if mcc > best_mcc:
                best_mcc = mcc

        return {"mcc": float(best_mcc)}

    def train(
        self,
        train_triples: np.ndarray,
        valid_triples: np.ndarray,
        *,
        trial: Any | None = None,
    ) -> dict[str, Any]:
        import optuna

        self._resolve_adaptive_batch_size()
        train_loader = self._build_train_loader(TripleDataset(train_triples))
        valid_tensor = torch.from_numpy(valid_triples).long().to(self.device)
        self._build_filter_dict(train_triples, valid_triples)

        self._train_triples_count = len(train_triples)

        logger.info(
            f"Iniciando treinamento DSLFM-KGC: epocas={self.training_config.epochs}, "
            f"treino={len(train_triples):,}, validacao={len(valid_triples):,}"
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
                self.current_epoch = epoch
                for obs in self.observers:
                    obs.on_epoch_start(epoch)

                train_metrics = self._train_epoch(train_loader, epoch)
                epoch_loss = train_metrics.get("loss", 0.0)
                stats["training_losses"].append(epoch_loss)

                val_metrics = {}
                if (epoch + 1) % self.training_config.validate_every == 0 or epoch == 0:
                    val_metrics = self._validate(valid_tensor)

                    binary_metrics = self._compute_binary_metrics_internal(valid_triples)
                    val_metrics.update(binary_metrics)

                    mcc = val_metrics.get("mcc", 0.0)
                    if mcc > self.best_val_mcc + self.training_config.min_delta:
                        self.best_val_mcc = mcc
                        stats["best_val_mcc"] = mcc
                        stats["best_epoch"] = epoch + 1
                        stats["best_metrics"] = val_metrics.copy()
                        self.patience_counter = 0
                        self._save_checkpoint("best_model.pt")
                    else:
                        self.patience_counter += 1

                    if trial:
                        trial.report(mcc, epoch)
                        if trial.should_prune():
                            logger.info(
                                "Trial pruned by Optuna",
                                stop_reason="pruning",
                                epoch=epoch + 1,
                                mcc=mcc,
                            )
                            raise optuna.TrialPruned()

                epoch_metrics = {**train_metrics, **val_metrics}
                for obs in self.observers:
                    obs.on_epoch_end(epoch, epoch_metrics)

                if self.patience_counter >= self.training_config.early_stopping_patience:
                    logger.info(
                        "Parada antecipada por paciencia",
                        stop_reason="early_stopping",
                        epoch=epoch + 1,
                        patience=self.training_config.early_stopping_patience,
                    )
                    break
                if self.time_estimator.check_budget(epoch, loss=epoch_loss):
                    logger.warning(
                        "Orcamento de tempo excedido",
                        stop_reason="time_budget",
                        epoch=epoch,
                        budget_config=self.training_config.time_budget,
                    )
                    break
                self._anneal_temperature()
                self._maybe_grow_batch_size()

        except optuna.TrialPruned:
            # Clean up before re-raising
            stats["epochs_trained"] = self.current_epoch + 1
            stats["training_time"] = time.time() - start_time
            # Try to grab final metrics if we have any
            if "best_metrics" not in stats:
                # Fallback if pruned extremely early
                stats["best_metrics"] = {
                    "mcc": self.best_val_mcc,
                    "mrr": self.best_val_mrr,
                }
            raise

        stats["epochs_trained"] = self.current_epoch + 1
        stats["training_time"] = time.time() - start_time

        # Persist NSCaching tensor if active
        self.model.negative_sampler.save_persistence()

        for obs in self.observers:
            obs.on_training_end(stats)
        return stats

    def _save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.current_epoch,
            "best_val_mrr": self.best_val_mrr,
            "best_val_mcc": self.best_val_mcc,
            "global_step": self.global_step,
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        self.file_manager.save(buffer.getvalue(), path)
        logger.info(f"Checkpoint salvo em {path}")

    def _load_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        if self.file_manager.exists(path):
            raw = self.file_manager.read_bytes(path)
            ckpt = torch.load(io.BytesIO(raw), map_location=self.device, weights_only=False)
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
            logger.info(f"Checkpoint carregado: {filename}")

    def _optimizer_step(self) -> None:
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
            # mem_get_info not available on some systems
            pass

    def _step_scheduler(self) -> None:
        if self.scheduler:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

    def _mask_known_tails(
        self, scores: torch.Tensor, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if not self._filter_arrays:
            return scores

        device = scores.device

        keys = torch.stack([h, r], dim=1)
        unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)

        for idx, (h_id, r_id) in enumerate(unique_keys):
            key = (int(h_id), int(r_id))
            known = self._filter_arrays.get(key)
            if known is None:
                continue

            if key not in self._filter_tensors:
                self._filter_tensors[key] = torch.from_numpy(known).to(device)

            known_t = self._filter_tensors[key]
            rows = (inverse == idx).nonzero(as_tuple=True)[0]

            true_tails_batch = t[rows]
            keep_mask = known_t.unsqueeze(0) != true_tails_batch.unsqueeze(1)
            mask_to_apply = keep_mask.all(dim=0).logical_not()

            if mask_to_apply.any():
                scores[rows.unsqueeze(1), known_t[mask_to_apply].unsqueeze(0)] = float("-inf")

        return scores


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
    output_dir = Path(output_dir)
    file_manager = FileManager()
    from .dslfm_kgc import load_dslfm_kgc_settings

    cfg = load_dslfm_kgc_settings(file_manager, kwargs.get("config_path"))

    m_cfg = cfg.get("kgc", {}).get("model", {})
    t_cfg = cfg.get("kgc", {}).get("training", {})

    use_bert_relations = _resolve_use_bert_setting(use_bert, m_cfg) and relation_names is not None
    model_config = DSLFMKGCConfig(
        num_entities=num_entities,
        num_relations=num_relations,
        num_triples=len(train_triples),
        entity_dim=kwargs.get("entity_dim", m_cfg.get("entity_dim", 256)),
        feature_dim=kwargs.get("feature_dim", m_cfg.get("feature_dim", 256)),
        max_communities=kwargs.get("max_communities", m_cfg.get("max_communities", 128)),
        use_bert_relations=use_bert_relations,
    )

    train_config = KGCTrainingConfig(
        epochs=kwargs.get("epochs", t_cfg.get("epochs", 200)),
        batch_size=kwargs.get("batch_size", t_cfg.get("batch_size", 256)),
        checkpoint_dir=output_dir / "checkpoints",
    )

    manager = DSLFMKGCManager(model_config, train_config, relation_names=relation_names)
    return manager.train(train_triples, valid_triples)
