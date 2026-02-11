import io
import torch
from pathlib import Path
from typing import Any
from pff.domain.ports.persistence.model_persistence import ModelPersistencePort
from pff.shared.core.file_manager import FileManager
from pff.shared import logger


class FileSystemModelPersistence(ModelPersistencePort):
    """FileSystem implementation of ModelPersistencePort."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.file_manager = FileManager()
        self.file_manager.ensure_dir(self.checkpoint_dir)

    def save_checkpoint(self, checkpoint_data: dict[str, Any], filename: str) -> None:
        path = self.checkpoint_dir / filename
        buffer = io.BytesIO()
        torch.save(checkpoint_data, buffer)
        self.file_manager.save(buffer.getvalue(), path)

        logger.info("Checkpoint salvo", path=str(path))

    def load_checkpoint(self, filename: str, map_location: Any = None) -> dict[str, Any] | None:
        path = self.checkpoint_dir / filename
        if self.file_manager.exists(path):
            raw = self.file_manager.read_bytes(path)
            ckpt = torch.load(io.BytesIO(raw), map_location=map_location, weights_only=False)
            logger.info("Checkpoint carregado", filename=filename)
            return ckpt
        return None
