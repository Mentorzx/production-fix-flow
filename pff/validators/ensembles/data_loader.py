import time
from typing import Any

import numpy as np
import polars as pl

from pff.config import settings
from pff.utils import CacheManager, FileManager, logger

cache = CacheManager()
file_manager = FileManager()


class EnsembleDataLoader:
    """
    Data loader for ensemble training and evaluation.
    """

    def __init__(self):
        self.file_manager = FileManager()
        self.cache_manager = CacheManager()

    def load_ensemble_data(self) -> tuple[list[Any], np.ndarray, list[Any], np.ndarray]:
        """
        Return X_train, y_train, X_test, y_test for ensemble training.

        Data are cached in-memory via ``CacheManager`` for 1 h to avoid
        re-parsing the Parquet files on every run.
        """
        logger.info("📊 Carregando dados para ensemble…")
        cache_key = "ensemble_data"
        try:
            cached_val, expiry = self.cache_manager[cache_key]
            if expiry > time.time():
                logger.info("✅ Dados carregados do cache")
                return cached_val
        except KeyError:
            pass
        logger.info("🔄 Gerando dados de treino/teste…")
        graph_path = settings.DATA_DIR / "models" / "kg"
        train_path = graph_path / "train_optimized.parquet"
        if not train_path.exists():
            train_path = graph_path / "train.parquet"
        test_path = graph_path / "test_optimized.parquet"
        if not test_path.exists():
            test_path = graph_path / "test.parquet"
        train_df: pl.DataFrame = self.file_manager.read(train_path)
        test_df: pl.DataFrame = self.file_manager.read(test_path)
        X_train, y_train = self._process_split(train_df, "train")
        X_test, y_test = self._process_split(test_df, "test")
        result = (X_train, y_train, X_test, y_test)
        ttl_seconds = 3600
        self.cache_manager[cache_key] = (result, time.time() + ttl_seconds)

        logger.success("✅ Ensemble data ready")
        return result

    def _process_split(
        self, df: pl.DataFrame, split_name: str
    ) -> tuple[list, np.ndarray]:
        """Process a data split into ensemble format."""
        logger.info(f"   Processando split: {split_name}")
        if "sample_id" in df.columns:
            grouped = df.group_by("sample_id", maintain_order=True)
            X = []
            y = []
            for group in grouped:
                triples = [
                    (row["s"], row["p"], row["o"])
                    for row in group[1].iter_rows(named=True)
                ]
                X.append(triples)
                y.append(1)
        else:
            X = [[(row["s"], row["p"], row["o"])] for row in df.iter_rows(named=True)]
            y = np.ones(len(X))
        X_neg, y_neg = self._generate_negatives(df, len(X))
        X_all = X + X_neg
        y_all = np.concatenate([y, y_neg])
        indices = np.random.permutation(len(X_all))
        X_shuffled = [X_all[i] for i in indices]
        y_shuffled = y_all[indices]

        return X_shuffled, y_shuffled

    def _generate_negatives(
        self, df: pl.DataFrame, num_negatives: int
    ) -> tuple[list, np.ndarray]:
        """
        Generate negative samples by corrupting existing positive triples.

        This standard negative sampling technique replaces either the head or
        the tail of a positive triple with a random entity from the vocabulary.
        """
        if df.is_empty() or num_negatives == 0:
            return [], np.array([])
        all_entities = pl.concat([df["s"], df["o"]]).unique().to_numpy()
        positive_triples_array = df.to_numpy()
        positive_triples_set = {tuple(row) for row in positive_triples_array}
        X_neg = []
        attempts = 0
        max_attempts = num_negatives * 5
        while len(X_neg) < num_negatives and attempts < max_attempts:
            attempts += 1
            base_triple_idx = np.random.randint(0, len(positive_triples_array))
            s, p, o = positive_triples_array[base_triple_idx]
            new_entity = np.random.choice(all_entities)
            if np.random.rand() < 0.5:
                if new_entity == s:  # Avoid replacing with the same entity
                    continue
                negative_triple = (new_entity, p, o)
            else:
                if new_entity == o:
                    continue
                negative_triple = (s, p, new_entity)
            if negative_triple not in positive_triples_set:
                X_neg.append([negative_triple])
        if len(X_neg) < num_negatives:
            logger.warning(
                f"Não foi possível gerar o número solicitado de amostras negativas. "
                f"Gerado: {len(X_neg)}, Solicitado: {num_negatives}. "
                "Isso pode acontecer em grafos densos ou se max_attempts for atingido."
            )
        y_neg = np.zeros(len(X_neg))

        return X_neg, y_neg
