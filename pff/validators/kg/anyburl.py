from abc import ABC, abstractmethod
from pathlib import Path

from clause import Learner, Options
import polars as pl
from pff.utils import CacheManager, FileManager, logger, progress_bar

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

    def convert_parquet_to_tsv(self, parquet_path: Path, tsv_path: Path) -> None:
        """
        Convert a Parquet file to TSV format expected by AnyBURL.

        Args:
            parquet_path: Path to the input Parquet file
            tsv_path: Path for the output TSV file

        Raises:
            ValueError: If required columns are missing
        """
        logger.info(f"Convertendo {parquet_path} para TSV...")

        dataframe = file_manager.read(parquet_path)
        required_columns = ["s", "p", "o"]

        # Validate columns
        if not all(column in dataframe.columns for column in required_columns):
            raise ValueError(
                f"Arquivo {parquet_path} deve conter colunas {required_columns}"
            )

        # Write TSV format
        file_manager.save(dataframe.select(*required_columns), tsv_path, separator="\t")

        logger.info(f"Conversão concluída: {len(dataframe)} triplas escritas")


class AnyBURLOptionsBuilder:
    """Build AnyBURL options from configuration."""

    def build_options(
        self,
        configuration: ConfigurationInterface,
        train_tsv_path: Path,
        rules_output_path: Path,
    ) -> Options:
        """
        Build Options object for AnyBURL learner.

        Args:
            configuration: Configuration object
            train_tsv_path: Path to training TSV file
            rules_output_path: Path for output rules

        Returns:
            Configured Options object
        """
        options = Options()
        options.set("learner.mode", "anyburl")

        # Set file paths
        options.set("learner.anyburl.raw.PATH_TRAIN", train_tsv_path.as_posix())
        options.set("learner.anyburl.raw.PATH_OUTPUT", rules_output_path.as_posix())

        # Apply AnyBURL parameters
        anyburl_parameters = configuration.get_anyburl_parameters()

        for key, value in anyburl_parameters.items():
            if isinstance(value, list):
                value = ",".join(map(str, value))
            if key.upper() == 'TIME':
                options.set(f"learner.anyburl.{key.lower()}", str(value))
            else:
                options.set(f"learner.anyburl.raw.{key}", str(value))

        java_heap = anyburl_parameters.get("JAVA_HEAP", "8G")
        java_options_list = [f"-Xmx{java_heap}", "-Dfile.encoding=UTF-8"]
        java_options_as_string_literal = str(java_options_list)
        options.set("learner.anyburl.java_options", java_options_as_string_literal)

        return options


class AnyBURLLearner(RuleLearnerInterface):
    """AnyBURL implementation of rule learner."""

    def __init__(self):
        """Initialize the AnyBURL learner."""
        self.format_converter = TripleFormatConverter()
        self.options_builder = AnyBURLOptionsBuilder()

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

        # Prepare temporary directory for TSV files
        pyclause_dir = configuration.get_pyclause_directory()
        train_tsv_path = pyclause_dir / "train.tsv"

        try:
            # Convert training data to TSV
            homogenized_train_path = pyclause_dir / "train.homogenized.parquet"
            if homogenized_train_path.exists():
                train_parquet_path = homogenized_train_path
                logger.info("Usando dados homogeneizados (filtrados) para AnyBURL")
            else:
                train_parquet_path = configuration.get_split_path("train")
                logger.warning("Dados homogeneizados não encontrados, usando originais")
            self.format_converter.convert_parquet_to_tsv(
                train_parquet_path, train_tsv_path
            )

            # Build options and execute learner
            rules_path = configuration.get_rules_path()
            options = self.options_builder.build_options(
                configuration, train_tsv_path, rules_path
            )

            # Execute AnyBURL
            await self._execute_anyburl(options, train_tsv_path, rules_path)

            # Clean up temporary files
            logger.info(f"Limpando arquivo temporário: {train_tsv_path.name}")
            train_tsv_path.unlink()

            return rules_path

        except Exception as error:
            logger.error(f"Erro no aprendizado de regras: {error}")
            raise

    def _prepare_tsv_directory(self, configuration: ConfigurationInterface) -> Path:
        """Prepare temporary directory for TSV files."""
        tsv_directory = configuration.get_pyclause_directory() / "tsv_temp"
        tsv_directory.mkdir(exist_ok=True)
        return tsv_directory

    async def _execute_anyburl(
        self, options: Options, train_tsv_path: Path, rules_path: Path
    ) -> None:
        """Execute AnyBURL learner and validate output."""
        learner = Learner(options=options.get("learner"))

        logger.info("Executando AnyBURL...")
        train_path_posix = train_tsv_path.as_posix()
        output_path_posix = rules_path.as_posix()
        learner.learn_rules(train_path_posix, output_path_posix)

        if not rules_path.exists():
            logger.debug(
                "Arquivo de regras principal não encontrado. Verificando arquivo temporário..."
            )
            learner_opts_dict = options.get("learner")
            if not isinstance(learner_opts_dict, dict):
                raise TypeError(
                    f"Esperava um dicionário para as opções do learner, mas recebi {type(learner_opts_dict)}"
                )
            anyburl_opts = learner_opts_dict.get("anyburl", {})
            if not isinstance(anyburl_opts, dict):
                raise TypeError(
                    f"Esperava um dicionário para as opções do anyburl, mas recebi {type(anyburl_opts)}"
                )
            anyburl_raw_params = anyburl_opts.get("raw", {})
            if not isinstance(anyburl_raw_params, dict):
                raise TypeError(
                    f"Esperava um dicionário para as opções 'raw', mas recebi {type(anyburl_raw_params)}"
                )
            time_limit = anyburl_raw_params.get("TIME")
            if not time_limit or int(str(time_limit)) == 0:
                raise RuntimeError("AnyBURL executou mas não gerou arquivo de regras.")
            temp_output_path = Path(f"{rules_path.as_posix()}-{time_limit}")
            if temp_output_path.exists():
                logger.info(
                    f"Arquivo temporário {temp_output_path.name} encontrado. Renomeando para {rules_path.name}"
                )
                temp_output_path.replace(rules_path)
            else:
                raise RuntimeError(
                    f"AnyBURL executou mas não gerou o arquivo de regras final nem o temporário ({temp_output_path.name})."
                )

        rule_count = await FileManager.count_lines(rules_path)
        logger.info(f"✅ Aprendizado concluído: {rule_count} regras geradas")

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
