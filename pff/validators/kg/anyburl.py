import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TYPE_CHECKING

from clause import Learner, Options
import polars as pl
from pff.utils import CacheManager, FileManager, logger, progress_bar

if TYPE_CHECKING:
    from pff.db.repositories.kg_rules import KGRulesRepository

from .config import ConfigurationInterface

"""
AnyBURL rule learning module.

This module provides interfaces and implementations for rule learning
using the AnyBURL system, including conversion utilities and rule parsing.
"""

# Initialize utility managers
file_manager = FileManager()
cache_manager = CacheManager()


class RuleLearnerInterface(ABC):
    """Interface for rule learning systems."""

    @abstractmethod
    async def learn_rules(self, configuration: ConfigurationInterface) -> Path:
        """Learn rules from the configured data."""
        pass


class TripleFormatConverter:
    """Convert between different triple data formats."""

    def convert_parquet_to_tsv(self, parquet_path: Path, tsv_path: Path) -> bool:
        """Convert a Parquet dataset to TSV and detect identifier safety requirements.

        This helper ensures the dataset matches AnyBURL expectations and returns
        whether ``SAFE_PREFIX_MODE`` must be enabled to accommodate numeric or
        short identifiers.

        Args:
            parquet_path: Source Parquet file containing triples with columns ``s``, ``p`` and ``o``.
            tsv_path: Destination TSV path expected by AnyBURL.

        Returns:
            bool: ``True`` when ``SAFE_PREFIX_MODE`` should be enabled.

        Raises:
            ValueError: If the Parquet file lacks the required triple columns.
        """
        logger.info(f"Convertendo {parquet_path} para TSV...")

        dataframe = file_manager.read(parquet_path)
        required_columns = ["s", "p", "o"]

        if not all(column in dataframe.columns for column in required_columns):
            raise ValueError(
                f"Arquivo {parquet_path} deve conter colunas {required_columns}"
            )

        needs_safe_prefix = False
        for column in required_columns:
            normalized = dataframe[column].cast(pl.Utf8).fill_null("")
            valid_mask = normalized.str.contains(r"^[A-Za-z].{1,}$", literal=False)
            valid_mask = valid_mask.fill_null(False)
            has_short = normalized.str.len_chars() < 2
            invalid_mask = (~valid_mask & (normalized != "")) | has_short
            if invalid_mask.any():
                needs_safe_prefix = True
                break

        file_manager.save(dataframe.select(*required_columns), tsv_path, separator="\t")

        logger.info(f"Conversão concluída: {len(dataframe)} triplas escritas")
        return needs_safe_prefix


class AnyBURLOptionsBuilder:
    """Build AnyBURL options from configuration."""

    def __init__(self) -> None:
        self._last_parameters: dict[str, Any] | None = None

    def build_options(
        self,
        configuration: ConfigurationInterface,
        train_tsv_path: Path,
        rules_output_path: Path,
        *,
        requires_safe_prefix: bool = False,
    ) -> Options:
        """
        Build Options object for AnyBURL learner.

        Args:
            configuration: Configuration object
            train_tsv_path: Path to training TSV file
            rules_output_path: Path for output rules
            requires_safe_prefix: Whether SAFE_PREFIX_MODE must be enforced

        Returns:
            Configured Options object
        """
        options = Options()
        options.set("learner.mode", "anyburl")

        # Set file paths
        options.set("learner.anyburl.raw.PATH_TRAIN", train_tsv_path.as_posix())
        options.set("learner.anyburl.raw.PATH_OUTPUT", rules_output_path.as_posix())

        # Apply AnyBURL parameters
        anyburl_parameters = self._normalize_parameters(
            configuration.get_anyburl_parameters(), requires_safe_prefix
        )
        self._last_parameters = dict(anyburl_parameters)

        for key, value in anyburl_parameters.items():
            formatted = ",".join(map(str, value)) if isinstance(value, list) else value
            if key.upper() == "TIME":
                options.set("learner.anyburl.time", str(formatted))
            else:
                options.set(f"learner.anyburl.raw.{key}", str(formatted))

        java_heap = str(anyburl_parameters.get("JAVA_HEAP", "8G"))
        java_options_list = [f"-Xmx{java_heap}", "-Dfile.encoding=UTF-8"]
        java_options_as_string_literal = str(java_options_list)
        options.set("learner.anyburl.java_options", java_options_as_string_literal)

        return options

    def get_last_parameters(self) -> dict[str, Any]:
        """Return the normalized parameter set used in the last build call."""
        return dict(self._last_parameters or {})

    def _normalize_parameters(
        self, parameters: dict[str, Any], requires_safe_prefix: bool
    ) -> dict[str, Any]:
        """Validate and normalize AnyBURL parameters before building Options."""
        normalized = dict(parameters)

        time_value = normalized.get("TIME", 300)
        try:
            time_int = max(1, int(float(time_value)))
        except (TypeError, ValueError):
            time_int = 300
        normalized["TIME"] = time_int

        worker_threads = normalized.get("WORKER_THREADS") or 0
        try:
            cpu_count = max(1, (os.cpu_count() or 1) - 1)
            worker_threads = int(worker_threads)
        except (TypeError, ValueError):
            worker_threads = 0
        normalized["WORKER_THREADS"] = max(1, min(worker_threads or cpu_count, cpu_count))

        snapshots = normalized.get("SNAPSHOTS_AT", [])
        if isinstance(snapshots, (int, float, str)):
            snapshots_list = [int(float(snapshots))]
        elif isinstance(snapshots, (list, tuple, set)):
            snapshots_list = [int(float(v)) for v in snapshots]
        else:
            snapshots_list = []
        snapshots_list = sorted({s for s in snapshots_list if s > 0})
        if snapshots_list:
            normalized["SNAPSHOTS_AT"] = snapshots_list
        else:
            normalized.pop("SNAPSHOTS_AT", None)

        if requires_safe_prefix:
            normalized["SAFE_PREFIX_MODE"] = True

        return normalized


class AnyBURLLearner(RuleLearnerInterface):
    """AnyBURL implementation of rule learner."""

    def __init__(self, rules_repository: "KGRulesRepository | None" = None):
        """Initialize the AnyBURL learner."""
        self.format_converter = TripleFormatConverter()
        self.options_builder = AnyBURLOptionsBuilder()
        self.rules_repository = rules_repository

    async def learn_rules(self, configuration: ConfigurationInterface) -> Path:
        """
        Execute rule learning using AnyBURL.

        Args:
            configuration: Configuration object

        Returns:
            Path to the generated rules file

        Raises:
            RuntimeError: If rule learning fails
        """
        logger.info("Iniciando aprendizado de regras com AnyBURL")

        # Prepare paths
        pyclause_dir = configuration.get_pyclause_directory()
        train_tsv_path = pyclause_dir / "train.tsv"

        try:
            #  Apply SOTA performance optimizations FIRST
            from .performance_optimizer import AnyBURLPerformanceOptimizer

            optimizer = AnyBURLPerformanceOptimizer()
            anyburl_config = configuration.get_anyburl_parameters()

            # Get train parquet path for optimization
            homogenized_train_path = pyclause_dir / "train.homogenized.parquet"
            if homogenized_train_path.exists():
                train_parquet_path = homogenized_train_path
                logger.info("Usando dados homogeneizados (filtrados) para AnyBURL")
            else:
                train_parquet_path = configuration.get_split_path("train")
                logger.warning("Homogenized data not found; using original splits")

            optimized_config = optimizer.optimize_parameters(
                anyburl_config,
                train_parquet_path,
            )

            # Check if any key has changed
            has_changes = any(
                optimized_config.get(k) != anyburl_config.get(k)
                for k in set(optimized_config.keys()) | set(anyburl_config.keys())
            )

            if has_changes:
                logger.info(" Aplicando parâmetros otimizados AnyBURL...")
                config_data = getattr(configuration, "_configuration_data", None)
                if isinstance(config_data, dict):
                    config_data.setdefault("anyburl", {}).update(optimized_config)
            else:
                logger.info(" Configuração AnyBURL já otimizada")

            # Convert training data to TSV
            requires_safe_prefix = self.format_converter.convert_parquet_to_tsv(
                train_parquet_path, train_tsv_path
            )

            if requires_safe_prefix:
                config_data = getattr(configuration, "_configuration_data", None)
                if isinstance(config_data, dict):
                    config_data.setdefault("anyburl", {})["SAFE_PREFIX_MODE"] = True
                logger.debug(
                    "SAFE_PREFIX_MODE ativado automaticamente após detectar identificadores incompatíveis"
                )

            # Build options with optimized config
            rules_path = configuration.get_rules_path()
            options = self.options_builder.build_options(
                configuration,
                train_tsv_path,
                rules_path,
                requires_safe_prefix=requires_safe_prefix,
            )
            normalized_parameters = self.options_builder.get_last_parameters()

            # Execute AnyBURL
            await self._execute_anyburl(
                options,
                train_tsv_path,
                rules_path,
                normalized_parameters,
            )

            # Clean up temporary files
            logger.info(f"Limpando arquivo temporário: {train_tsv_path.name}")
            train_tsv_path.unlink()

            return rules_path

        except Exception as error:
            logger.error(f"Rule learning failed: {error}")
            raise

    def _prepare_tsv_directory(self, configuration: ConfigurationInterface) -> Path:
        """Prepare temporary directory for TSV files."""
        tsv_directory = configuration.get_pyclause_directory() / "tsv_temp"
        tsv_directory.mkdir(exist_ok=True)
        return tsv_directory

    async def _execute_anyburl(
        self,
        options: Options,
        train_tsv_path: Path,
        rules_path: Path,
        parameters: dict[str, Any],
    ) -> None:
        """Execute AnyBURL learner and validate output."""
        learner = Learner(options=options.get("learner"))

        logger.info("Executando AnyBURL...")
        train_path_posix = train_tsv_path.as_posix()
        output_path_posix = rules_path.as_posix()
        learner.learn_rules(train_path_posix, output_path_posix)

        self._finalize_rules_output(rules_path, parameters)

        if not rules_path.exists():
            raise RuntimeError("AnyBURL executou mas não gerou arquivo de regras.")

        rule_count = await FileManager.count_lines(rules_path)
        logger.info(f" Aprendizado concluído: {rule_count} regras geradas")

        try:
            if self.rules_repository is None:
                from pff.db.repositories.kg_rules import KGRulesRepository

                self.rules_repository = KGRulesRepository()

            saved_count = await self.rules_repository.save_rules_from_file(
                rules_path.as_posix(),
                source="anyburl",
                iteration=parameters.get("ITERATION"),
            )
            logger.info(f"Regras sincronizadas com PostgreSQL: {saved_count} inseridas")
        except Exception as exc:
            logger.error(f"Failed to persist rules to PostgreSQL: {exc}")
            # Don't fail the whole process if DB save fails, but log it.


    def _finalize_rules_output(
        self, rules_path: Path, parameters: dict[str, Any]
    ) -> None:
        """Ensure the canonical rules file exists, promoting snapshots if required."""
        if rules_path.exists():
            return

        snapshot_candidates: list[tuple[int, Path]] = []
        snapshots = parameters.get("SNAPSHOTS_AT", [])

        if isinstance(snapshots, (list, tuple, set)):
            numeric_snapshots = []
            for value in snapshots:
                try:
                    numeric_snapshots.append(int(float(value)))
                except (TypeError, ValueError):
                    continue
        elif isinstance(snapshots, (int, float, str)):
            try:
                numeric_snapshots = [int(float(snapshots))]
            except (TypeError, ValueError):
                numeric_snapshots = []
        else:
            numeric_snapshots = []

        for snapshot in numeric_snapshots:
            candidate = Path(f"{rules_path.as_posix()}-{snapshot}")
            if candidate.exists():
                snapshot_candidates.append((snapshot, candidate))

        if not snapshot_candidates:
            for candidate in rules_path.parent.glob(f"{rules_path.name}-*"):
                snapshot_value = self._parse_snapshot_suffix(candidate.name, rules_path.name)
                if snapshot_value is not None:
                    snapshot_candidates.append((snapshot_value, candidate))

        if not snapshot_candidates:
            return

        snapshot_candidates.sort(key=lambda item: item[0])
        _, source_path = snapshot_candidates[-1]

        try:
            shutil.copy2(source_path, rules_path)
            logger.info(
                f"Snapshot {source_path.name} promovido para {rules_path.name}"
            )
        except Exception as exc:
            logger.warning(f"Failed to promote AnyBURL snapshot {source_path}: {exc}")

    @staticmethod
    def _parse_snapshot_suffix(filename: str, base_name: str) -> int | None:
        """Extract the numeric snapshot suffix from ``filename`` relative to ``base_name``."""
        if not filename.startswith(base_name):
            return None
        suffix = filename[len(base_name):].lstrip("-")
        if not suffix:
            return None
        try:
            return int(float(suffix))
        except ValueError:
            return None

    def _cleanup_temporary_files(self, tsv_file: Path, tsv_directory: Path) -> None:
        """Clean up temporary TSV files."""
        tsv_file.unlink()
        tsv_directory.rmdir()


class RuleParser:
    """Parse and analyze AnyBURL rule files."""

    @cache_manager.disk_cache(ttl=24 * 3600)
    def parse_rules_file(
        self, rules_path: Path
    ) -> tuple[list[str], list[dict[str, float | int | str]]]:
        """Parses a rules file from AnyBURL and extracts rules and their metadata.
        This function reads a TSV file containing rules generated by AnyBURL, processes each row,
        and separates the rules from their associated metadata (predictions, support, confidence).
        Args:
            rules_path (Path): Path to the TSV file containing the AnyBURL rules.
        Returns:
            tuple[list[str], list[dict[str, float | int | str]]]: A tuple containing:
                - list[str]: List of rule strings
                - list[dict]: List of dictionaries containing metadata for each rule with keys:
                    - num_predictions (int): Number of predictions for the rule
                    - support (int): Support value for the rule
                    - confidence (float): Confidence score for the rule
                    - rule (str): The rule string itself
        Example:
            rules_path = Path("rules.tsv")
            rules, metadata = parse_rules_file(rules_path)
        """
        logger.info(
            f"Analisando (ou carregando do cache) o arquivo de regras: {rules_path}"
        )
        rules = []
        metadata = []
        rules_df = file_manager.read(rules_path, has_header=False, separator="\t")
        if rules_df.shape[1] == 4:
            # Formato completo do AnyBURL (com estatísticas)
            df_renamed = rules_df.rename(
                {
                    "column_1": "num_predictions", "column_2": "support",
                    "column_3": "confidence", "column_4": "rule",
                }
            )
        elif rules_df.shape[1] == 1:
            # Formato simples (apenas a string da regra)
            logger.info("Arquivo de regras em formato simples detectado. Usando metadados padrão.")
            df_renamed = rules_df.rename({"column_1": "rule"})
            # Adiciona colunas de metadados com valores padrão
            df_renamed = df_renamed.with_columns(
                pl.lit(100, dtype=pl.Int64).alias("num_predictions"),
                pl.lit(100, dtype=pl.Int64).alias("support"),
                pl.lit(1.0, dtype=pl.Float64).alias("confidence"),
            )
        else:
            raise ValueError(f"Formato de arquivo de regras inesperado com {rules_df.shape[1]} colunas.")

        for row in progress_bar(
            df_renamed.iter_rows(named=True),
            desc="Processando regras do DataFrame",
            total=len(df_renamed),
        ):
            parsed_data = {
                "num_predictions": int(row["num_predictions"]),
                "support": int(row["support"]),
                "confidence": float(row["confidence"]),
                "rule": str(row["rule"]),
            }
            rules.append(parsed_data["rule"])
            metadata.append(parsed_data)

        logger.info(f"Parseadas {len(rules)} regras do arquivo")
        return rules, metadata
