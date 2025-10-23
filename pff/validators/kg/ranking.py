import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from c_clause import Loader, RankingHandler
from clause import Options

"""
Parallel ranking worker module for the KGC pipeline.

This module implements the distributed ranking computation
using Ray for parallel processing.
"""


class SuppressOutput:
    """
    A context manager to temporarily suppress stdout and stderr.
    """

    def __init__(self, suppress: bool = True):
        self.suppress = suppress
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        if not self.suppress:
            return

        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.suppress:
            return

        if self._stdout:
            sys.stdout.close()
            sys.stdout = self._stdout
        if self._stderr:
            sys.stderr.close()
            sys.stderr = self._stderr


class RankingWorkerInterface(ABC):
    """Interface for ranking worker implementations."""

    @abstractmethod
    def process_chunk(
        self,
        chunk: np.ndarray,
        rules_path: str,
        filter_triples: np.ndarray,
        train_triples: np.ndarray,
        worker_id: int,
        aggregation_function: str,
        tie_handling: str,
        filter_w_data: bool,
        num_threads: int,
        verbose: bool,
        entity_map_path: Path,
        relation_map_path: Path,
    ) -> dict[str, list | int | str | dict]:
        """Process a chunk of triples for ranking."""
        pass


class KGRankingWorker(RankingWorkerInterface):
    """PyClause-based implementation of ranking worker."""

    def __init__(self):
        """Initializes the worker without loading the logger yet."""
        self._logger = None

    @property
    def logger(self):
        """
        A lazy-loading property for the logger.
        It imports and caches the logger on its first use within the instance.
        """
        if self._logger is None:
            from pff.utils import logger as _logger

            self._logger = _logger
        return self._logger

    def process_chunk(
        self,
        chunk: np.ndarray,
        rules_path: str,
        filter_triples: np.ndarray,
        train_triples: np.ndarray,
        worker_id: int,
        aggregation_function: str,
        tie_handling: str,
        filter_w_data: bool,
        num_threads: int,
        verbose: bool,
        entity_map_path: Path,
        relation_map_path: Path,
    ) -> dict[str, list | int | str | dict]:
        """
        Processes a chunk of triples for ranking evaluation using specified rules and options.
        This method configures ranking options, initializes data loaders, executes the ranking process,
        and formats the results. It is designed to be run in a parallel worker context.
        Args:
            chunk (np.ndarray): The chunk of triples to process.
            rules (list[str]): List of rules to apply during ranking.
            filter_triples (np.ndarray): Triples used for filtering during evaluation.
            train_triples (np.ndarray): Training triples for context.
            worker_id (int): Identifier for the worker process.
            aggregation_function (str): Aggregation function for ranking (e.g., 'mean', 'max').
            tie_handling (str): Strategy for handling ties in ranking.
            filter_w_data (bool): Whether to filter with additional data.
            num_threads (int): Number of threads to use for computation.
            verbose (bool): If True, enables verbose logging.
            entity_map_path (Path): Path to the entity mapping file.
            relation_map_path (Path): Path to the relation mapping file.
        Returns:
            dict[str, list | int | str | dict]: A dictionary containing the results of the ranking process,
            including success or error information, worker ID, and processed data.
        Raises:
            Exception: Any exception encountered during processing is caught and returned in the result dictionary.
        """

        self.logger.info(f"Worker {worker_id} started (PID: {os.getpid()})")

        with SuppressOutput(suppress=(not verbose)):
            try:
                pyclause_options = Options()
                pyclause_options.set(
                    "ranking_handler.aggregation_function", aggregation_function
                )
                pyclause_options.set("ranking_handler.tie_handling", tie_handling)
                pyclause_options.set("ranking_handler.filter_w_data", filter_w_data)
                pyclause_options.set("ranking_handler.num_threads", num_threads)
                loader = self._initialize_loader(
                    pyclause_options,
                    train_triples,
                    filter_triples,
                    chunk,
                    entity_map_path,
                    relation_map_path,
                )
                ranking_results = self._execute_ranking(
                    pyclause_options, loader, worker_id, chunk, rules_path
                )
                return self._format_success_result(
                    worker_id, ranking_results, len(chunk)
                )
            except Exception as error:
                return self._format_error_result(worker_id, error)

    def _initialize_loader(
        self,
        options: Options,
        train_triples: np.ndarray,
        filter_triples: np.ndarray,
        chunk: np.ndarray,
        entity_map_path: Path,
        relation_map_path: Path,
    ) -> Loader:
        """Initialize loader, set entity/relation indices, then load data."""
        from pff.utils import FileManager

        fm = FileManager()

        self.logger.info("Criando Loader local...")
        loader = Loader(options=options.get("loader"))

        self.logger.info("Carregando mapas de índice...")
        entity_map = fm.read(entity_map_path)
        relation_map = fm.read(relation_map_path)

        entity_list = entity_map.sort("id")["label"].to_list()
        relation_list = relation_map.sort("id")["label"].to_list()

        self.logger.info(
            f"Definindo índice com {len(entity_list)} entidades e {len(relation_list)} relações..."
        )
        loader.set_entity_index(entity_list)
        loader.set_relation_index(relation_list)

        self.logger.info("Carregando dados numéricos no Loader...")
        loader.load_data(
            data=train_triples.tolist(),
            filter=filter_triples.tolist(),
            target=chunk.tolist(),
        )

        return loader

    def _execute_ranking(
        self,
        options: Options,
        loader: Loader,
        worker_id: int,
        chunk: np.ndarray,
        rules_path: str,
    ) -> dict[str, dict | list]:
        """Execute ranking computation."""
        chunk_set = set()
        for triple in chunk:
            h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
            chunk_set.add((h, r, t))
        
        self.logger.info(f"Worker {worker_id}: Chunk contém {len(chunk_set)} triplas únicas")
        chunk_size = len(chunk)
        self.logger.info(f"Worker {worker_id}: Criando RankingHandler...")
        handler = RankingHandler(options=options.get("ranking_handler"))
        self.logger.info(f"Worker {worker_id}: Carregando regras de {rules_path}...")
        loader.load_rules(rules=rules_path)
        self.logger.info(
            f"Worker {worker_id}: Calculando rankings para {chunk_size} triplas..."
        )
        handler.calculate_ranking(loader=loader)
        head_ranking_dict = handler.get_ranking("head", as_string=True)
        tail_ranking_dict = handler.get_ranking("tail", as_string=True)
        ranking_lines = []
        for direction_dict in [head_ranking_dict, tail_ranking_dict]:
            if isinstance(direction_dict, dict):
                for relation, entities_dict in direction_dict.items():
                    for entity, candidates in entities_dict.items():
                        if candidates:
                            candidates_str = "\t".join(str(c) for c in candidates)
                            ranking_lines.append(
                                f"{relation}\t{entity}\t{candidates_str}"
                            )
                        else:
                            ranking_lines.append(f"{relation}\t{entity}")
            elif isinstance(direction_dict, list):
                ranking_lines.extend(direction_dict)
        detailed_scores = self._collect_detailed_scores(handler, chunk)
        self.logger.info(
            f"Worker {worker_id}: Processamento concluído - {len(ranking_lines)} linhas de ranking"
        )

        return {"ranking_lines": ranking_lines, "detailed_scores": detailed_scores}

    def _collect_detailed_scores(
        self, handler: RankingHandler, test_chunk: np.ndarray
    ) -> list[dict[str, float | int | str]]:
        """
        Collect detailed scores for metrics calculation.

        Args:
            handler: The RankingHandler with calculated rankings
            test_chunk: The chunk of test triples being evaluated

        Returns:
            List of score dictionaries with ground truth labels
        """
        detailed_scores = []
        test_triples_debug = set()
        for triple in test_chunk:
            h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
            test_triples_debug.add((h, r, t))
        self.logger.info(f"Chunk de teste contém {len(test_triples_debug)} triplas únicas")
        test_set = set()
        for triple in test_chunk:
            test_set.add((int(triple[0]), int(triple[1]), int(triple[2])))
        # Build a set of true triples for quick lookup
        true_triples = set()
        for triple in test_chunk:
            h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
            true_triples.add(("head", r, t, h))
            true_triples.add(("tail", r, h, t))
        self.logger.info(f"Conjunto de triplas verdadeiras criado com {len(true_triples)} entradas")

        for direction in ["head", "tail"]:
            ranking = handler.get_ranking(as_string=False, direction=direction)
            for relation_id, source_dictionary in ranking.items():
                for source_id, candidate_scores in source_dictionary.items():
                    if not candidate_scores:
                        continue
                    for candidate_id, score in candidate_scores:
                        if direction == "tail":
                            triple = (
                                int(source_id),
                                int(relation_id),
                                int(candidate_id),
                            )
                        else:  # direction == "head"
                            triple = (
                                int(candidate_id),
                                int(relation_id),
                                int(source_id),
                            )
                        is_true = 1 if (direction, relation_id, source_id, candidate_id) in true_triples else 0
                        detailed_scores.append(
                            {
                                "direction": direction,
                                "rel_id": relation_id,
                                "src_id": source_id,
                                "cand_id": candidate_id,
                                "score": float(score),
                                "is_true": is_true,
                            }
                        )
        true_count = sum(1 for s in detailed_scores if s["is_true"] == 1)
        self.logger.info(
            f"Coletados {len(detailed_scores)} scores, {true_count} verdadeiros positivos"
        )

        return detailed_scores

    def _format_success_result(
        self,
        worker_id: int,
        ranking_results: dict[str, dict | list],
        triples_processed: int,
    ) -> dict[str, list | int | str | dict]:
        """Format successful result dictionary."""
        return {
            "worker_id": worker_id,
            "ranking_lines": ranking_results["ranking_lines"],
            "detailed_scores": ranking_results["detailed_scores"],
            "triples_processed": triples_processed,
            "status": "success",
        }

    def _format_error_result(
        self, worker_id: int, error: Exception
    ) -> dict[str, list | int | str | dict]:
        """Format error result dictionary."""
        self.logger.error(f"Worker {worker_id}: Erro durante processamento: {error}")
        return {
            "worker_id": worker_id,
            "status": "error",
            "error": str(error),
            "ranking_lines": [],
            "detailed_scores": [],
        }

def load_if_path(x):
    if isinstance(x, (str, Path)):
        x = Path(x)
        if x.suffix == ".npy" and x.exists():
            return np.load(x)
    return x

def parallel_ranking_worker(
    shared_data: dict, worker_id: int, chunk: np.ndarray
) -> dict:
    """
    Processes a chunk of data for knowledge graph ranking in parallel.
    This function is intended to be used as a worker in a parallel processing setup.
    It retrieves necessary shared data and parameters, initializes a KGRankingWorker,
    and processes the given chunk of triples for ranking evaluation.
    Args:
        shared_data (dict): A dictionary containing shared parameters and data required for ranking,
            including rules, filter triples, training triples, aggregation function, tie handling,
            filter_w_data, number of threads, verbosity, and paths to entity and relation maps.
        worker_id (int): The unique identifier for the worker process.
        chunk (np.ndarray): The chunk of triples to be processed by this worker.
    Returns:
        dict: The result of processing the chunk, typically containing ranking metrics or evaluation results.
    """
    rules_path = shared_data["rules_path"]
    filter_triples = load_if_path(shared_data.get("filter_triples"))
    train_triples = load_if_path(shared_data.get("train_triples"))

    worker = KGRankingWorker()
    return worker.process_chunk(
        chunk=chunk,
        rules_path=rules_path,
        filter_triples=filter_triples,
        train_triples=train_triples,
        worker_id=worker_id,
        aggregation_function=shared_data["aggregation_function"],
        tie_handling=shared_data["tie_handling"],
        filter_w_data=shared_data["filter_w_data"],
        num_threads=shared_data["num_threads"],
        verbose=shared_data["verbose"],
        entity_map_path=shared_data["entity_map_path"],
        relation_map_path=shared_data["relation_map_path"],
    )


def create_test_data_chunks(
    test_triples: np.ndarray,
    chunk_size: int,
    num_workers: int | None = None,
    max_chunk_size: int = 1000,
) -> list[np.ndarray]:
    """
    Splits test triples into chunks for parallel processing.

    Args:
        test_triples: Array of test triples to split
        chunk_size: Desired chunk size from configuration
        num_workers: Number of workers to use
        max_chunk_size: Maximum allowed chunk size to prevent OOM

    Returns:
        List of numpy arrays, each containing a chunk of test data
    """
    from pff.utils import logger

    if num_workers and num_workers > 0:
        total_chunks_needed = (len(test_triples) + max_chunk_size - 1) // max_chunk_size

        chunks = [
            test_triples[i * max_chunk_size : (i + 1) * max_chunk_size]
            for i in range(total_chunks_needed)
        ]

        logger.info(
            f"Dados de teste divididos em {len(chunks)} chunks pequenos (tamanho: ~{max_chunk_size})"
        )
    else:
        effective_chunk_size = min(chunk_size, max_chunk_size)
        number_of_chunks = (
            len(test_triples) + effective_chunk_size - 1
        ) // effective_chunk_size
        chunks = [
            test_triples[i * effective_chunk_size : (i + 1) * effective_chunk_size]
            for i in range(number_of_chunks)
        ]

        logger.info(
            f"Dados de teste divididos em {len(chunks)} chunks (tamanho: {effective_chunk_size})"
        )

    return chunks
