# Sprint 16.5: Removed stdlib json import, using FileManager.json_loads() instead
from pff.utils import FileManager
import pickle
from pathlib import Path
from typing import Any

from c_clause import QAHandler, PredictionHandler, RankingHandler, Loader
from clause import Options

"""
Module for scoring and querying the knowledge graph.

This module defines the KnowledgeGraphScorer class, which coordinates loading data,
rules, and manual rules, and provides methods to answer queries, score facts, and
explain the ranking of facts.
"""


class KGScorer:
    """
    Coordinates handlers to answer queries, score facts, and explain rankings.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize handlers and load graph data, clauses, stats, and manual rules.

        Args:
            config: Configuration mapping containing:
                - 'graph': {'path': str}
                - 'pyclause': {'out_dir': str}
                - 'manual_rules': {'path': str}

        Raises:
            FileNotFoundError: If the clauses pickle file is not found.
        """
        self.options = Options()
        self.qa = QAHandler(options=self.options.get("qa_handler"))
        self.predictor = PredictionHandler(
            options=self.options.get("prediction_handler")
        )
        self.ranking_handler = RankingHandler(
            options=self.options.get("ranking_handler")
        )
        self.loader = Loader(options=self.options.get("loader"))

        self._load_graph_data(config["graph"])
        clauses = self._load_clauses(config["pyclause"]["out_dir"])
        stats = self._load_stats(config["pyclause"]["out_dir"])
        self.loader.load_rules(rules=clauses, stats=stats)
        self.manual_rules = self._load_manual_rules(config["manual_rules"]["path"])

    def _load_graph_data(self, graph_cfg: dict[str, str]) -> None:
        """
        Carrega *somente* train/valid/test ao invés de tudo de uma vez.

        Args:
            graph_cfg: dict com chaves 'path', 'valid' e 'test',
                    apontando para os arquivos .txt gerados pelo builder.
        """
        data_path  = Path(graph_cfg["train"])
        valid_path = graph_cfg.get("valid")
        test_path  = graph_cfg.get("test")

        loader_kwargs: dict[str, str] = {"data": str(data_path)}
        if valid_path:
            loader_kwargs["filter"] = str(Path(valid_path))
        if test_path:
            loader_kwargs["target"] = str(Path(test_path))

        self.loader.load_data(**loader_kwargs)

    def _load_clauses(self, output_dir: str) -> list[Any]:
        """
        Load clauses from 'clauses.pkl' in the specified directory.

        Args:
            output_dir: Directory containing 'clauses.pkl'.

        Returns:
            The list of clauses.

        Raises:
            FileNotFoundError: If the pickle file does not exist.
        """
        clauses_file = Path(output_dir) / "clauses.pkl"
        if not clauses_file.exists():
            raise FileNotFoundError(f"Não achei {clauses_file}")
        return FileManager().read(clauses_file)

    def _load_stats(self, output_dir: str) -> dict[str, Any] | None:
        """
        Load statistics from 'stats.json' if present.

        Args:
            output_dir: Directory containing 'stats.json'.

        Returns:
            Parsed statistics dict, or None if the file is missing.
        """
        stats_file = Path(output_dir) / "stats.json"
        if stats_file.exists():
            return FileManager().read(stats_file)
        return None

    def _load_manual_rules(self, rules_path: str) -> list[dict[str, Any]]:
        """
        Load manual rules from a JSON file if it exists.

        Args:
            rules_path: Path to JSON file with manual rules.

        Returns:
            List of manual rule dictionaries.
        """
        manual_file = Path(rules_path)
        if manual_file.exists():
            return FileManager().read(manual_file)
        return []


class KGScorerFactory:
    """Factory for KGScorer construction.

    Pattern: Factory
    - Centralizes KGScorer creation from config mappings or explicit paths.
    - Keeps KGScorer __init__ stable while offering higher-level builders.
    """

    @staticmethod
    def from_paths(
        train_path: str | Path,
        *,
        valid_path: str | Path | None = None,
        test_path: str | Path | None = None,
        pyclause_dir: str | Path,
        manual_rules_path: str | Path | None = None,
    ) -> KGScorer:
        config = {
            "graph": {
                "train": str(train_path),
                "valid": str(valid_path) if valid_path else None,
                "test": str(test_path) if test_path else None,
            },
            "pyclause": {"out_dir": str(pyclause_dir)},
            "manual_rules": {"path": str(manual_rules_path or "")},
        }
        return KGScorer(config)

    @staticmethod
    def from_config(config: dict[str, Any]) -> KGScorer:
        """Build a KGScorer from an existing config mapping."""
        return KGScorer(config)

    def answer(self, query: dict[str, Any]) -> list[Any]:
        """
        Return top-k answers for the given query via QAHandler.

        Args:
            query: Parameters for the QA handler.

        Returns:
            Sequence of answers.
        """
        return self.qa.query(self.loader, query)

    def score(self, fact: dict[str, Any]) -> float:
        """
        Calculate a score for the given fact via PredictionHandler.

        Args:
            fact: Fact data to be scored.

        Returns:
            The computed score.
        """
        return self.predictor.score(self.loader, fact)

    def explain(self, fact: dict[str, Any]) -> list[str]:
        """
        Generate textual explanations for the given fact via RankingHandler.

        Args:
            fact: Fact data to explain.

        Returns:
            Sequence of explanation strings.
        """
        return self.ranking_handler.explain(self.loader, fact)
