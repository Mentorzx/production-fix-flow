import os
from abc import ABC, abstractmethod
from pathlib import Path


class Options:
    """Placeholder Options (PyClause removido no modo DSLFM/PC)."""

    def __init__(self, *args, **kwargs):
        pass

    def set(self, *args, **kwargs):
        return None


from pff.config import settings  # noqa: E402
from pff.shared import FileManager, logger  # noqa: E402
from pff.shared.core.file_manager import ParquetBundle  # noqa: E402

"""
Configuration module for the Knowledge Graph Completion pipeline.

This module provides interfaces and implementations for managing
all configuration aspects of the KGC pipeline.
"""


class ConfigurationInterface(ABC):
    """Interface for pipeline configuration management."""

    @abstractmethod
    def validate(self) -> bool:
        """Validate configuration completeness and correctness."""
        pass

    @abstractmethod
    def get_split_path(self, split_name: str) -> Path:
        """Get path for a specific data split."""
        pass

    @abstractmethod
    def get_preprocessing_parameters(self) -> dict[str, float | int | bool]:
        """Get preprocessing stage parameters."""
        pass

    @abstractmethod
    def get_entity_map_path(self) -> Path:
        """Get the path to the entity-to-id mapping file."""
        pass

    @abstractmethod
    def get_relation_map_path(self) -> Path:
        """Get the path to the relation-to-id mapping file."""
        pass

    @abstractmethod
    def get_max_chunk_size(self) -> int:
        """Get maximum chunk size for ranking to prevent OOM."""
        pass

    @abstractmethod
    def get_mappings_directory(self) -> Path:
        """Get mappings directory path."""
        pass

    @abstractmethod
    def get_calibration_config(self) -> dict:
        """Get calibration configuration parameters."""
        pass

    @abstractmethod
    def get_dask_configuration(self) -> dict:
        """Get Dask configuration parameters."""
        pass


class PathResolver:
    """Resolve relative paths against a base directory."""

    def __init__(self, base_directory: Path):
        """
        Initialize path resolver.

        Args:
            base_directory: Base directory for relative paths
        """
        self.base_directory = base_directory

    def resolve(self, path: str | Path) -> Path:
        """
        Resolve a path relative to base directory.

        Args:
            path: Path to resolve

        Returns:
            Resolved absolute path
        """
        path_object = Path(path)

        if path_object.is_absolute():
            return path_object.resolve()

        return (self.base_directory / path_object).resolve()


class KGConfig(ConfigurationInterface):
    """
    Centralized configuration for Knowledge Graph Completion pipeline.

    This class manages:
    - Data paths (train, validation, test, rules, outputs)
    - Legacy rule-mining parameters (AnyBURL/PyClause, deprecated)
    - Ray configuration
    - Pipeline settings
    """

    def __init__(self, configuration_path: str | Path):
        """
        Initialize configuration from a YAML file.

        Args:
            configuration_path: Path to configuration file

        Raises:
            FileNotFoundError: If configuration file not found
        """
        self.configuration_path = Path(configuration_path).resolve()
        fm = FileManager()

        if not fm.exists(self.configuration_path):
            raise FileNotFoundError(
                f"Arquivo de configuração não encontrado: {self.configuration_path}"
            )

        # Load configuration data
        payload = fm.read(self.configuration_path)
        self._configuration_data = (
            payload.to_native() if isinstance(payload, ParquetBundle) else payload
        )

        # Initialize path resolver - use project root, not config directory
        # This ensures paths like "./data" resolve to "<project>/data" not "<config>/data"
        self.path_resolver = PathResolver(settings.ROOT_DIR)

        # Initialize all paths
        self._initialize_paths()

    def _initialize_paths(self) -> None:
        """
        Resolve and create all directories used by the pipeline.

        Priority order for each path:

        1. Explicit value provided in ``self._configuration_data["paths"]``.
        2. Project-wide default from :pydata:`pff.settings`.
        3. Hard-coded fallback (kept only for backward-compatibility).

        Every directory is created with ``exist_ok=True`` so the method may be
        called repeatedly without raising an error.
        """

        def _ensure(path: Path) -> Path:
            """Create *path* (and parents) if it does not exist, then return it."""
            FileManager().ensure_dir(path)
            return path

        def _resolve_output_path(raw: str | Path) -> Path:
            """Resolve output paths relative to OUTPUTS_DIR to avoid root pollution."""
            candidate = Path(raw)
            if candidate.is_absolute():
                return candidate
            resolved = (settings.OUTPUTS_DIR / candidate).resolve()
            # Guard: if candidate already points to ROOT, force outputs/
            if not resolved.is_relative_to(settings.OUTPUTS_DIR):
                return (settings.OUTPUTS_DIR / candidate.name).resolve()
            return resolved

        paths_cfg: dict[str, str] = self._configuration_data.get("paths", {})

        self.data_directory: Path = _ensure(
            self.path_resolver.resolve(paths_cfg.get("data_dir", settings.DATA_DIR))
        )
        raw_output = paths_cfg.get("output_dir", settings.OUTPUTS_DIR)
        self.output_directory: Path = _ensure(_resolve_output_path(raw_output))
        graph_subdir = paths_cfg.get("graph_subdir", "kg")
        graph_candidate = self.output_directory / graph_subdir
        if graph_candidate.name == self.output_directory.name:
            graph_candidate = self.output_directory
        if not graph_candidate.is_relative_to(settings.OUTPUTS_DIR):
            graph_candidate = settings.OUTPUTS_DIR / Path(graph_subdir).name
        self.graph_directory: Path = _ensure(graph_candidate)
        self.train_path: Path = self.graph_directory / "train.parquet"
        self.valid_path: Path = self.graph_directory / "valid.parquet"
        self.test_path: Path = self.graph_directory / "test.parquet"
        mappings_subdir = paths_cfg.get("mappings_subdir", "mappings")
        self.mappings_directory: Path = _ensure(self.output_directory / mappings_subdir)
        self.entity_map_path: Path = self.mappings_directory / "entity_map.parquet"
        self.relation_map_path: Path = self.mappings_directory / "relation_map.parquet"
        self.train_numpy_path: Path = self.mappings_directory / "train.npy"
        self.valid_numpy_path: Path = self.mappings_directory / "valid.npy"
        self.test_numpy_path: Path = self.mappings_directory / "test.npy"
        self.rules_path: Path = self.mappings_directory / "rules.tsv"
        self.ranking_path: Path = self.mappings_directory / "ranking.json"
        self.checkpoint_dir: Path = _ensure(self.output_directory / "checkpoints")

    def validate(self) -> bool:
        """
        Validate that all required files exist.

        Returns:
            True if valid, False otherwise
        """
        required_files = [self.train_path, self.valid_path, self.test_path]

        fm = FileManager()
        missing_files = [
            file_path for file_path in required_files if not fm.exists(file_path)
        ]

        if missing_files:
            logger.warning(f"Arquivos obrigatórios ausentes: {missing_files}")
            return False

        return True

    def get_split_path(self, split_name: str) -> Path:
        """
        Get path for a specific data split.

        Args:
            split_name: Name of split ('train', 'valid', 'test')

        Returns:
            Path to the split file

        Raises:
            ValueError: If split name is invalid
        """
        split_mapping = {
            "train": self.train_path,
            "valid": self.valid_path,
            "test": self.test_path,
        }

        if split_name not in split_mapping:
            raise ValueError(f"Invalid split name: {split_name}")

        return split_mapping[split_name]

    def get_mappings_directory(self) -> Path:
        """Get mappings directory (entity/relation maps and indexed splits)."""
        return self.mappings_directory

    def get_builder_config(self) -> dict:
        """
        Get builder configuration parameters.

        Returns:
            Dictionary with builder configuration
        """
        builder_config = self._configuration_data.get(
            "builder",
            {
                "source_path": "data/source.parquet",
                "parallel": True,
                "disk_cache": False,
            },
        )
        if "max_members" not in builder_config:
            builder_config["max_members"] = None

        return builder_config

    def get_pipeline_configuration(self) -> dict[str, int | bool]:
        """
        Get general pipeline configuration.

        Returns:
            Dictionary with pipeline configuration
        """
        default_config = {
            "chunk_size": 100_000,
            "num_workers": os.cpu_count() or 4,
            "max_rules_per_chunk": 10_000,
            "enable_caching": True,
        }

        pipeline_config = self._configuration_data.get("pipeline", {})
        return {**default_config, **pipeline_config}

    def get_max_chunk_size(self) -> int:
        """
        Get maximum chunk size for ranking.

        Returns:
            Maximum chunk size, defaults to 1000 if not specified
        """
        pipeline_config = self._configuration_data.get("pipeline", {})
        return pipeline_config.get("max_chunk_size", 1000)

    def get_calibration_config(self) -> dict:
        """
        Get calibration configuration parameters.

        Returns:
            Dictionary with calibration configuration
        """
        pipeline_config = self._configuration_data.get("pipeline", {})
        calibration_config = pipeline_config.get("calibration", {})

        return {
            "enabled": calibration_config.get("enabled", True),
            "method": calibration_config.get("method", "platt"),
            "cross_validation_folds": calibration_config.get(
                "cross_validation_folds", 5
            ),
            "optimize_threshold": calibration_config.get("optimize_threshold", True),
            "optimization_metric": calibration_config.get("optimization_metric", "f1"),
        }

    def get_preprocessing_parameters(self) -> dict[str, float | int | bool]:
        """
        Get preprocessing stage parameters.

        Returns:
            Dictionary with preprocessing parameters
        """
        pipeline_config = self._configuration_data.get("pipeline", {})

        default_preprocessing = {
            "enabled": False,
            "homogeneity_level": 0.5,
            "min_support": 3,
        }

        preprocessing_config = pipeline_config.get("preprocess", default_preprocessing)

        return preprocessing_config

    def get_step_outputs(self, step_name: str) -> list[Path]:
        """
        Returns a list of critical output files for a given pipeline step.
        """
        mappings_dir = self.get_mappings_directory()
        if step_name == "preprocess":
            return [
                mappings_dir / "train.npy",
                mappings_dir / "valid.npy",
                mappings_dir / "test.npy",
                mappings_dir / "entity_map.parquet",
                mappings_dir / "relation_map.parquet",
            ]
        return []

    def get_config_with_overrides(self, override_config: dict | None) -> dict:
        import copy  # noqa: PLC0415

        config_data = copy.deepcopy(self._configuration_data)

        if override_config:
            if "pipeline" in override_config:
                config_data.setdefault("pipeline", {}).update(
                    override_config["pipeline"]
                )

        return config_data

    def get_output_directory(self) -> Path:
        """Get the output directory path."""
        return self.output_directory

    def get_entity_map_path(self) -> Path:
        """Returns the resolved path to the entity map file."""
        return self.entity_map_path

    def get_relation_map_path(self) -> Path:
        """Returns the resolved path to the relation map file."""
        return self.relation_map_path

    def get_dask_configuration(self) -> dict:
        """Returns Dask configuration parameters from the YAML file."""
        return self._configuration_data.get("dask", {})

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"KnowledgeGraphConfiguration(\n"
            f"  config_path={self.configuration_path},\n"
            f"  data_dir={self.data_directory},\n"
            f"  output_dir={self.output_directory}\n"
            f")"
        )
