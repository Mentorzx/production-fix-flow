from datetime import datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, ValidationError

from pff.shared import FileManager, logger
from pff.shared.core.config import settings


class TaskModel(BaseModel):
    """
    Represents a task with associated metadata.
    Attributes:
        msisdn (str): The MSISDN (Mobile Station International Subscriber Directory Number) associated with the task.
        sequence (str): The sequence identifier for the task.
        payload (dict[str, Any] | None): Optional dictionary containing additional data relevant to the task.
    """

    msisdn: str
    sequence: str
    payload: dict[str, Any] | None = None


class ManifestModel(BaseModel):
    """
    Represents a manifest containing execution metadata and a list of tasks.
    Attributes:
        execution_id (str): Unique identifier for the execution, auto-generated with a timestamp if not provided.
        resource_usage (float | None): Percentage of system resources to use (0-100). Default: 90 (leaves 10% margin).
        max_workers (int | None): [DEPRECATED] Maximum number of worker threads/processes. Use resource_usage instead.
        tasks (list[TaskModel]): List of tasks to be executed as part of the manifest.
    """

    execution_id: str = Field(
        default_factory=lambda: f"exec-{datetime.now().strftime('%Y%m%d%H%M')}"
    )
    resource_usage: float | None = Field(default=90.0, ge=1.0, le=100.0)
    max_workers: int | None = Field(default=None, gt=0)
    tasks: list[TaskModel]


class ManifestParser:
    """Parses and validates manifest YAML files with custom tag support.

    Design Patterns Applied:
        - **Builder Pattern (implicit):** Incrementally constructs ManifestModel
          from parsed YAML data with validation at each step.
        - **Factory Method:** Custom YAML constructors (!file tag) act as
          factories for loading external file contents.
        - **Adapter Pattern:** Adapts YAML structure to Pydantic ManifestModel
          schema, handling type conversions and validations.

    Performance Optimizations:
        - FileManager used for all file I/O (AGENTS.md compliance).
        - Pydantic validation for schema enforcement.

    Attributes:
        file_manager: Utility for reading files from disk.

    Methods:
        parse: Parses manifest file and returns validated ManifestModel.
    """

    def __init__(self):
        self.file_manager = FileManager()

    def _file_constructor(
        self, loader: yaml.SafeLoader, node: yaml.Node
    ) -> dict[str, Any]:
        """
        Constructs a dictionary from a YAML node by loading the contents of a file specified in the node.
        This method resolves the file path using the application's data directory, checks if the file exists,
        and reads its contents using the file manager. If the file does not exist, a FileNotFoundError is raised.
        Args:
            loader (yaml.SafeLoader): The YAML loader instance.
            node (yaml.Node): The YAML node containing the file path as its value.
        Returns:
            dict[str, Any]: The contents of the file as a dictionary.
        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        file_path = settings.DATA_DIR / str(node.value)
        if not file_path.is_file():
            raise FileNotFoundError(f"Arquivo de payload não encontrado: {file_path}")

        logger.debug(f"Loading payload from file: {file_path}")
        return self.file_manager.read(file_path, return_native=True)  # type: ignore[no-any-return]

    def parse(self, manifest_path: Path) -> ManifestModel:
        """
        Parses a manifest file and returns a validated ManifestModel instance.
        Args:
            manifest_path (Path): The path to the manifest file.
        Returns:
            ManifestModel: The validated manifest model parsed from the file.
        Raises:
            FileNotFoundError: If the manifest file does not exist or cannot be found.
            yaml.YAMLError: If there is an error parsing the YAML content.
            ValidationError: If the parsed data does not conform to the ManifestModel schema.
            Exception: For any other unexpected errors during parsing.
        Logs:
            - Info: When starting to read the manifest.
            - Success: When the manifest is successfully validated.
            - Error: For file not found, validation, or unexpected errors.
        """
        logger.debug(f"Reading manifest from: {manifest_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Arquivo de manifesto não encontrado: {manifest_path}"
            )

        try:
            custom_yaml_tags = {"!file": self._file_constructor}
            data = self.file_manager.read(
                manifest_path, custom_tags=custom_yaml_tags, return_native=True
            )
            manifest = ManifestModel.model_validate(data)
            logger.success(
                f"Manifesto '{manifest.execution_id}' validado com sucesso com {len(manifest.tasks)} tarefas."
            )

            return manifest
        except FileNotFoundError as e:
            logger.error(f"Error reading file: {e}")
            raise
        except (yaml.YAMLError, ValidationError) as e:
            logger.error(f"Validation or format error in manifest file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing manifest: {e}")
            raise
