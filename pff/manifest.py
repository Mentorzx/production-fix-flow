from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from pff import settings
from pff.utils import FileManager, logger


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
    max_workers: int | None = Field(default=None, gt=0)  # Legacy, deprecated
    tasks: list[TaskModel]


class ManifestParser:
    """
    ManifestParser is responsible for parsing and validating manifest YAML files for the application.
    This class provides methods to:
    - Load and parse manifest files from disk.
    - Support custom YAML tags (e.g., !file) to include external file contents within the manifest.
    - Validate the parsed manifest data against the ManifestModel schema.
    - Log the parsing process, including successes and errors.
    Attributes:
        file_manager (FileManager): Utility for reading files from disk.
    Methods:
        _file_constructor(loader, node):
            Custom YAML constructor for the !file tag. Loads and returns the contents of a referenced file as a dictionary.
            Raises FileNotFoundError if the file does not exist.
        parse(manifest_path):
            Parses the manifest file at the given path, validates it, and returns a ManifestModel instance.
            Handles and logs errors related to file access, YAML parsing, and schema validation.
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

        logger.debug(f"Carregando payload do arquivo: {file_path}")
        return self.file_manager.read(file_path)

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
        logger.debug(f"Lendo manifesto de: {manifest_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Arquivo de manifesto não encontrado: {manifest_path}"
            )

        try:
            custom_yaml_tags = {"!file": self._file_constructor}
            data = self.file_manager.read(manifest_path, custom_tags=custom_yaml_tags)
            manifest = ManifestModel.model_validate(data)
            logger.success(
                f"Manifesto '{manifest.execution_id}' validado com sucesso com {len(manifest.tasks)} tarefas."
            )

            return manifest
        except FileNotFoundError as e:
            logger.error(f"Erro ao ler arquivo: {e}")
            raise
        except (yaml.YAMLError, ValidationError) as e:
            logger.error(f"Erro de validação ou formato no arquivo de manifesto: {e}")
            raise
        except Exception as e:
            logger.error(f"Erro inesperado ao parsear o manifesto: {e}")
            raise
