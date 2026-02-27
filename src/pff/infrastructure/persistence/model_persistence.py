"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/infrastructure/persistence/model_persistence.py

"""

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
        """Execute init.



        Args:

            checkpoint_dir: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        self.checkpoint_dir = checkpoint_dir
        self.file_manager = FileManager()
        self.file_manager.ensure_dir(self.checkpoint_dir)

    def save_checkpoint(self, checkpoint_data: dict[str, Any], filename: str) -> None:
        """Execute save checkpoint.



        Args:

            checkpoint_data: Input value used by this callable.

            filename: Input value used by this callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        path = self.checkpoint_dir / filename
        buffer = io.BytesIO()
        torch.save(checkpoint_data, buffer)
        self.file_manager.save(buffer.getvalue(), path)

        logger.info("Checkpoint salvo", path=str(path))

    def load_checkpoint(
        self, filename: str, map_location: Any = None
    ) -> dict[str, Any] | None:
        """Execute load checkpoint.



        Args:

            filename: Input value used by this callable.

            map_location: Optional input value.



        Returns:

            Return value produced by the callable.



        Notes:

            Keep behavior deterministic and free of hidden side effects.

        """

        path = self.checkpoint_dir / filename
        if self.file_manager.exists(path):
            raw = self.file_manager.read_bytes(path)
            ckpt = torch.load(
                io.BytesIO(raw), map_location=map_location, weights_only=False
            )
            logger.info("Checkpoint carregado", filename=filename)
            return ckpt
        return None
