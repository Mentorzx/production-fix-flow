import os
from abc import ABC, abstractmethod
from pathlib import Path

from clause import Options

from pff.config import settings
from pff.utils import FileManager

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
    def get_rules_path(self) -> Path:
        """Get path for rules file."""
        pass

    @abstractmethod
    def get_anyburl_parameters(self) -> dict[str, int | float | str | list]:
        """Get AnyBURL configuration parameters."""
        pass

    @abstractmethod
    def get_pyclause_options_dictionary(self) -> dict[str, dict]:
        """Get PyClause options as dictionary."""
        pass

    @abstractmethod
    def get_ray_configuration(self) -> dict[str, int | str | dict]:
        """Get Ray configuration parameters."""
        pass

    @abstractmethod
    def get_test_numpy_path(self) -> Path:
        """Get the path to the .npy file containing test indices."""
        pass

    @abstractmethod
    def get_ranking_path(self) -> Path:
        """Get the path to the ranking output file."""
        pass

    @abstractmethod
    def get_scores_path(self) -> Path:
        """Get the path to the detailed scores file."""
        pass

    @abstractmethod
    def get_metadata_path(self) -> Path:
        """Get the path to the execution metadata file."""
        pass

    @abstractmethod
    def get_pyclause_directory(self) -> Path:
        """Get the PyClause working directory."""
        pass

    @abstractmethod
    def get_pipeline_configuration(self) -> dict[str, int | bool]:
        """Get general pipeline configuration."""
        pass

    @abstractmethod
    def get_preprocessing_parameters(self) -> dict[str, float | int | bool]:
        """Get preprocessing stage parameters."""
        pass

    @abstractmethod
    def get_builder_config(self) -> dict:
        """Get builder configuration parameters."""
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
    - AnyBURL parameters
    - PyClause parameters
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

        if not self.configuration_path.exists():
            raise FileNotFoundError(
                f"Arquivo de configuração não encontrado: {self.configuration_path}"
            )

        # Load configuration data
        self._configuration_data = FileManager().read(self.configuration_path)

        # Initialize path resolver
        self.path_resolver = PathResolver(self.configuration_path.parents[1])

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
            path.mkdir(parents=True, exist_ok=True)
            return path

        paths_cfg: dict[str, str] = self._configuration_data.get("paths", {})

        self.data_directory: Path = _ensure(
            self.path_resolver.resolve(paths_cfg.get("data_dir", settings.DATA_DIR))
        )
        graph_subdir = paths_cfg.get("graph_subdir", "models/kg")
        self.graph_directory: Path = _ensure(self.data_directory / graph_subdir)
        self.train_path: Path = self.graph_directory / "train.parquet"
        self.valid_path: Path = self.graph_directory / "valid.parquet"
        self.test_path: Path = self.graph_directory / "test.parquet"
        self.output_directory: Path = _ensure(
            self.path_resolver.resolve(
                paths_cfg.get("output_dir", settings.OUTPUTS_DIR)
            )
        )
        pyclause_subdir = paths_cfg.get("pyclause_subdir", "pyclause")
        self.pyclause_directory: Path = _ensure(self.output_directory / pyclause_subdir)
        self.entity_map_path: Path = self.pyclause_directory / "entity_map.parquet"
        self.relation_map_path: Path = self.pyclause_directory / "relation_map.parquet"
        self.train_numpy_path: Path = self.pyclause_directory / "train.npy"
        self.valid_numpy_path: Path = self.pyclause_directory / "valid.npy"
        self.test_numpy_path: Path = self.pyclause_directory / "test.npy"
        self.rules_path: Path = self.pyclause_directory / "rules_anyburl.tsv"
        self.ranking_path: Path = self.pyclause_directory / "ranking.json"

    def validate(self) -> bool:
        """
        Validate that all required files exist.

        Returns:
            True if valid, False otherwise
        """
        required_files = [self.train_path, self.valid_path, self.test_path]

        missing_files = [
            file_path for file_path in required_files if not file_path.exists()
        ]

        if missing_files:
            print(f"Arquivos faltando: {missing_files}")
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

    def get_rules_path(self) -> Path:
        """Get path for rules file."""
        return self.rules_path

    def get_ranking_path(self) -> Path:
        """Get path for ranking output file."""
        return self.ranking_path

    def get_scores_path(self) -> Path:
        """Get path for detailed scores file."""
        return self.pyclause_directory / "ranking_scores.parquet"

    def get_metadata_path(self) -> Path:
        """Get path for execution metadata file."""
        return self.pyclause_directory / "execution_metadata.json"

    def get_pyclause_directory(self) -> Path:
        """Get PyClause working directory."""
        return self.pyclause_directory

    def get_test_numpy_path(self) -> Path:
        """Get path for test NumPy indices."""
        return self.test_numpy_path

    def get_anyburl_parameters(self) -> dict[str, int | float | str | list]:
        """
        Get AnyBURL configuration parameters.

        Returns:
            Dictionary with AnyBURL parameters
        """
        default_parameters = {
            "SNAPSHOTS_AT": [30, 60],
            "WORKER_THREADS": os.cpu_count() or 8,
            "THRESHOLD_CORRECT_PREDICTIONS": 2,
            "THRESHOLD_CONFIDENCE": 0.0,
            "MAX_LENGTH_CYCLIC": 3,
            "MAX_LENGTH_ACYCLIC": 1,
        }

        configured_parameters = self._configuration_data.get("anyburl", {})
        return {**default_parameters, **configured_parameters}

    def get_pyclause_options(self) -> Options:
        """
        Create and return configured PyClause Options object.

        Returns:
            Configured Options object
        """
        options = Options()
        pyclause_config = self._configuration_data.get("pyclause", {})

        # Configure loader options
        loader_config = pyclause_config.get("loader", {})
        for key, value in loader_config.items():
            options.set(f"loader.{key}", value)

        # Configure ranking handler options
        default_ranking_config = {
            "disc_at_k": 10,
            "aggregation_function": "maxplus",
            "num_threads": -1,  # Use all cores
            "filter_w_data": True,
            "tie_handling": "frequency",
        }

        ranking_config = pyclause_config.get("ranking_handler", default_ranking_config)

        for key, value in ranking_config.items():
            options.set(f"ranking_handler.{key}", value)

        return options

    def get_pyclause_options_dictionary(self) -> dict[str, dict]:
        """
        Get PyClause options as serializable dictionary.

        Returns:
            Dictionary with PyClause options
        """
        pyclause_config = self._configuration_data.get("pyclause", {})

        return {
            "loader": pyclause_config.get("loader", {}),
            "ranking_handler": pyclause_config.get(
                "ranking_handler",
                {
                    "disc_at_k": 10,
                    "aggregation_function": "maxplus",
                    "num_threads": -1,
                    "filter_w_data": True,
                    "tie_handling": "frequency",
                },
            ),
        }

    def get_builder_config(self) -> dict:
        """
        Get builder configuration parameters.

        Returns:
            Dictionary with builder configuration
        """
        builder_config = self._configuration_data.get(
            "builder",
            {"source_path": "data/source.zip", "parallel": True, "disk_cache": False},
        )
        if "max_members" not in builder_config:
            builder_config["max_members"] = None

        return builder_config

    def get_ray_configuration(self) -> dict[str, int | str | dict]:
        """
        Get Ray configuration parameters.

        Returns:
            Dictionary with Ray configuration
        """
        ray_config = self._configuration_data.get("ray", {})

        return {
            "num_cpus": ray_config.get("num_cpus", None),
            "address": ray_config.get("address", None),  # None = local
            "logging_config": {"encoding": "JSON", "log_to_driver": False},
            "runtime_env": ray_config.get("runtime_env", {}),
        }

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
        pyclause_dir = self.get_pyclause_directory()
        if step_name == "preprocess":
            return [
                pyclause_dir / "train.npy",
                pyclause_dir / "valid.npy",
                pyclause_dir / "test.npy",
                pyclause_dir / "entity_map.parquet",
                pyclause_dir / "relation_map.parquet",
            ]
        if step_name == "learn_rules":
            return [self.get_rules_path()]
        if step_name == "ranking":
            output_dir = self.get_output_directory()
            return [
                output_dir / "ranking.json",
                output_dir / "metrics.json",
            ]
        return []

    def get_config_with_overrides(self, override_config: dict | None) -> dict:
        import copy

        config_data = copy.deepcopy(self._configuration_data)

        if override_config:
            if "anyburl" in override_config:
                config_data.setdefault("anyburl", {}).update(override_config["anyburl"])
            if "pyclause" in override_config:
                config_data.setdefault("pyclause", {}).update(
                    override_config["pyclause"]
                )
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
        ray_config = self.get_ray_configuration()

        return (
            f"KnowledgeGraphConfiguration(\n"
            f"  config_path={self.configuration_path},\n"
            f"  data_dir={self.data_directory},\n"
            f"  output_dir={self.output_directory},\n"
            f"  ray_workers={ray_config['num_cpus']}\n"
            f")"
        )
