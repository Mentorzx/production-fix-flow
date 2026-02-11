"""DSLFM Checkpoint Manager.

Handles checkpoint save/load/cleanup operations for DSLFM training.
Extracted from DSLFMManager for Single Responsibility Principle (SRP).

Design Patterns Applied:
    - **Strategy Pattern:** Checkpoint format can be swapped (PyTorch, ONNX).
    - **Factory Pattern:** Creates checkpoint dictionaries.
    - **Memento Pattern:** Captures and restores model state.

Author: PFF Team
Date: 2025-11-26
"""

from __future__ import annotations

import io
import re
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from pff.shared import FileManager, logger
from pff.shared.core.config import settings
from pff.shared.core.file_manager import ParquetBundle

try:
    from safetensors.torch import load as load_safetensors
    from safetensors.torch import save as save_safetensors

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


_EPOCH_RE = re.compile(r"checkpoint_epoch_(?P<epoch>\d+)\.pt$")


def _resolve_checkpoint_dir(path: Path) -> Path:
    """Resolve a checkpoint directory under `settings.OUTPUTS_DIR`.

    Args:
        path: Candidate checkpoint directory.

    Returns:
        Resolved checkpoint directory path anchored under `settings.OUTPUTS_DIR`.
    """
    candidate = path
    if not candidate.is_absolute():
        candidate = settings.OUTPUTS_DIR / candidate
    candidate = candidate.resolve()
    if candidate.is_relative_to(settings.OUTPUTS_DIR):
        return candidate
    logger.warning(
        f"Checkpoint directory outside OUTPUTS_DIR; redirecting requested={path} resolved={candidate}"
    )
    return (settings.OUTPUTS_DIR / candidate.name).resolve()


def _parse_checkpoint_epoch(path: Path) -> int:
    """Extract epoch number from checkpoint filename.

    Args:
        path: Checkpoint path.

    Returns:
        Epoch number when parseable, otherwise -1.
    """
    match = _EPOCH_RE.search(path.name)
    if match is None:
        return -1
    return int(match.group("epoch"))


class DSLFMCheckpointManager:
    """Manager for DSLFM model checkpoints.

    Pattern: Manager/Facade

    Handles saving, loading, and cleaning up model checkpoints.
    Supports versioned checkpoints and best-model tracking.

    Attributes:
        checkpoint_dir: Directory for storing checkpoints.
        file_manager: FileManager instance for I/O operations.
        keep_top_k: Number of recent checkpoints to keep.
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        file_manager: FileManager | None = None,
        keep_top_k: int = 3,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for storing checkpoints.
            file_manager: FileManager instance for I/O operations.
            keep_top_k: Number of recent checkpoints to keep (default: 3).
        """
        self.file_manager = file_manager or FileManager()
        self.checkpoint_dir = _resolve_checkpoint_dir(Path(checkpoint_dir))
        self.file_manager.ensure_dir(self.checkpoint_dir)
        self.keep_top_k = keep_top_k

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool = False,
        filename: str | None = None,
        extra_state: dict[str, Any] | None = None,
        use_safetensors: bool = False,
    ) -> Path:
        """Save model checkpoint.

        Args:
            model: PyTorch model to save.
            optimizer: Optimizer state to save (optional).
            epoch: Current epoch number.
            metrics: Dictionary of metrics at checkpoint time.
            is_best: Whether this is the best model so far.
            filename: Custom filename (auto-generated if None).
            extra_state: Optional extra state (e.g., NPC) to persist.
            use_safetensors: Also persist model/optimizer tensors as safetensors for faster loads.

        Returns:
            Path to saved checkpoint file.
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if extra_state:
            checkpoint["extra_state"] = extra_state

        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        self.file_manager.save(buffer.getvalue(), checkpoint_path)
        logger.info(f"Checkpoint salvo: {checkpoint_path}")

        if use_safetensors and SAFETENSORS_AVAILABLE:
            weights_path = checkpoint_path.with_suffix(".safetensors")
            tensor_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            raw = save_safetensors(tensor_state)
            self.file_manager.write_bytes(raw, weights_path)
            logger.info(f"Safetensors salvo: {weights_path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            self.file_manager.save(buffer.getvalue(), best_path)
            logger.success(f"Melhor modelo salvo: {best_path}")

        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        path: Path | str | None = None,
        device: torch.device | str = "cpu",
        prefer_safetensors: bool = False,
    ) -> dict[str, Any]:
        """Load model checkpoint.

        Args:
            model: Model to load state into.
            optimizer: Optimizer to load state into (optional).
            path: Path to checkpoint file (uses best_model.pt if None).
            device: Device to load tensors to.
            prefer_safetensors: If True and .safetensors exists, load weights from it for faster init.

        Returns:
            Dictionary with checkpoint metadata.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        if path is None:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = Path(path)

        if not self.file_manager.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint_bytes = self.file_manager.read_bytes(path)
        checkpoint = torch.load(
            io.BytesIO(checkpoint_bytes), map_location=device, weights_only=False
        )

        if prefer_safetensors and SAFETENSORS_AVAILABLE:
            weights_path = Path(path).with_suffix(".safetensors")
            if self.file_manager.exists(weights_path):
                raw = self.file_manager.read_bytes(weights_path)
                tensor_state = load_safetensors(raw)
                model.load_state_dict(tensor_state)
            else:
                model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Checkpoint carregado: {path} (epoca {checkpoint.get('epoch', '?')})")

        return {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
            "timestamp": checkpoint.get("timestamp", ""),
            "extra_state": checkpoint.get("extra_state", {}),
        }

    def get_latest_checkpoint(self) -> Path | None:
        """Get path to the most recent checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist.
        """
        checkpoints = self.file_manager.glob(self.checkpoint_dir, "checkpoint_epoch_*.pt")
        if not checkpoints:
            return None
        return max(checkpoints, key=_parse_checkpoint_epoch)

    def get_best_checkpoint(self) -> Path | None:
        """Get path to the best model checkpoint.

        Returns:
            Path to best checkpoint, or None if it doesn't exist.
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        return best_path if self.file_manager.exists(best_path) else None

    def has_completed_training(self, target_epochs: int) -> tuple[bool, dict[str, Any]]:
        """Check if training was previously completed.

        Args:
            target_epochs: Expected number of training epochs.

        Returns:
            Tuple of (completed: bool, completion_info: dict).
        """
        marker_path = self.checkpoint_dir / "training_completed.json"

        if not self.file_manager.exists(marker_path):
            return False, {}

        try:
            payload = self.file_manager.read(marker_path)
            completion_info = payload.to_native() if isinstance(payload, ParquetBundle) else payload
            completed_epochs = completion_info.get("epochs_trained", 0)
            saved_target = completion_info.get("target_epochs", 0)

            is_complete = completed_epochs >= saved_target and saved_target == target_epochs

            return is_complete, completion_info
        except Exception:
            return False, {}

    def mark_training_completed(
        self,
        epochs_trained: int,
        target_epochs: int,
        best_epoch: int,
        best_val_mrr: float,
        training_time: float,
        final_metrics: dict[str, float],
    ) -> None:
        """Mark training as completed.

        Args:
            epochs_trained: Number of epochs actually trained.
            target_epochs: Target number of epochs.
            best_epoch: Epoch with best validation score.
            best_val_mrr: Best validation MRR achieved.
            training_time: Total training time in seconds.
            final_metrics: Final validation metrics.
        """
        marker_path = self.checkpoint_dir / "training_completed.json"

        completion_info = {
            "epochs_trained": epochs_trained,
            "target_epochs": target_epochs,
            "best_epoch": best_epoch,
            "best_val_mrr": best_val_mrr,
            "training_time": training_time,
            "final_metrics": final_metrics,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.file_manager.save(completion_info, marker_path)
        logger.info(f"Marcador de conclusao salvo: {marker_path}")

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = self.file_manager.glob(self.checkpoint_dir, "checkpoint_epoch_*.pt")
        checkpoints = sorted(checkpoints, key=_parse_checkpoint_epoch, reverse=True)
        for old_checkpoint in checkpoints[self.keep_top_k :]:
            try:
                self.file_manager.delete_file(old_checkpoint, ignore_errors=False)
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")

    def clear_all(self) -> None:
        """Remove all checkpoints and markers.

        Warning: This is destructive and cannot be undone.
        """
        for file in self.file_manager.glob(self.checkpoint_dir, "*.pt"):
            self.file_manager.delete_file(file, ignore_errors=True)
        for file in self.file_manager.glob(self.checkpoint_dir, "*.json"):
            self.file_manager.delete_file(file, ignore_errors=True)
        logger.info(f"Todos os checkpoints removidos de {self.checkpoint_dir}")
