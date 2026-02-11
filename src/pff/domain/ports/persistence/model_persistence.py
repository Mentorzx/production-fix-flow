from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class ModelPersistencePort(Protocol):
    """Port for persisting model checkpoints."""

    def save_checkpoint(self, checkpoint_data: dict[str, Any], filename: str) -> None:
        """Save a checkpoint dictionary to storage.

        Args:
            checkpoint_data: Dictionary containing model state, optimizer state, etc.
            filename: Name of the checkpoint file.
        """
        ...

    def load_checkpoint(
        self, filename: str, map_location: Any = None
    ) -> dict[str, Any] | None:
        """Load a checkpoint dictionary from storage.

        Args:
            filename: Name of the checkpoint file.
            map_location: Optional device/location mapping for torch.load.

        Returns:
            The checkpoint dictionary if found, None otherwise.
        """
        ...
