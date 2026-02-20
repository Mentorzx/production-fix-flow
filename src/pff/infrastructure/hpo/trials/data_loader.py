"""Provide module-level functionality for the PFF codebase.



Notes:

    File: src/pff/infrastructure/hpo/trials/data_loader.py

"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sized
from typing import Any, cast

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from pff.shared import load_config, logger
from pff.shared.acceleration.asyncio_runner import (
    run_coroutine_in_new_loop,
    run_coroutine_sync,
)
from pff.shared.core.config import (
    KG_PIPELINE_CONFIG_PATH,
    OPTIMIZATION_CONFIG_PATH,
    settings,
)
from pff.shared.core.file_manager import FileManager, ParquetBundle

try:
    from pff.domain.kg.preprocessing import (
        KGPreprocessingPipeline,
        PreprocessingConfig,
        filter_attribute_relations,
    )

    HAS_PREPROCESSING_MODULE = True
except ImportError:
    HAS_PREPROCESSING_MODULE = False


INVERSE_POLICY_KEEP = "keep"
INVERSE_POLICY_DROP_SUFFIX = "drop_suffix"
ALLOWED_INVERSE_POLICIES = frozenset({INVERSE_POLICY_KEEP, INVERSE_POLICY_DROP_SUFFIX})


def _count_unique_arrow(*series: pl.Series) -> int:
    """Zero-copy unique count using PyArrow chunks (30x faster concat)."""
    chunks = []
    for s in series:
        arr = s.to_arrow()
        if isinstance(arr, pa.ChunkedArray):
            chunks.extend(arr.chunks)
        else:
            chunks.append(arr)

    if not chunks:
        return 0

    combined = pa.chunked_array(chunks)
    unique_fn = getattr(pc, "unique", None)
    if callable(unique_fn):
        unique_values = unique_fn(combined)
        if isinstance(unique_values, (pa.Array, pa.ChunkedArray, list, tuple, set)):
            return len(cast(Sized, unique_values))
        return len(set(combined.to_pylist()))
    return len(set(combined.to_pylist()))


def _infer_id_upper_bound(*series: pl.Series) -> int:
    """Infer maximum integer ID across series; returns -1 when unavailable."""
    max_id = -1
    for s in series:
        if not s.dtype.is_integer():
            continue
        if len(s) == 0:
            continue
        current = s.max()
        if current is None:
            continue
        if isinstance(current, bool):
            continue
        if isinstance(current, int):
            current_int = current
        else:
            current_int = int(cast(Any, current))
        max_id = max(max_id, current_int)
    return max_id


def _as_lazy_frame(bundle: Any) -> pl.LazyFrame:
    """Execute as lazy frame.



    Args:

        bundle: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Raises:

        Exception: Propagates domain-specific failures with context.

    """

    if isinstance(bundle, ParquetBundle):
        if bundle.parsed_kind != "tabular":
            raise ValueError("Expected tabular data for lazy loading")
        return bundle.lazyframe()
    if isinstance(bundle, pl.LazyFrame):
        return bundle
    if isinstance(bundle, pl.DataFrame):
        return bundle.lazy()
    raise ValueError(f"Expected tabular data, got {type(bundle)}")


def _as_dataframe(bundle: Any) -> pl.DataFrame:
    """Execute as dataframe.



    Args:

        bundle: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Raises:

        Exception: Propagates domain-specific failures with context.

    """

    if isinstance(bundle, ParquetBundle):
        if bundle.parsed_kind != "tabular":
            raise ValueError("Expected tabular data for DataFrame loading")
        return bundle.lazyframe().collect(engine="streaming")
    if isinstance(bundle, pl.LazyFrame):
        return bundle.collect(engine="streaming")
    if isinstance(bundle, pl.DataFrame):
        return bundle
    raise ValueError(f"Expected tabular data, got {type(bundle)}")


def compute_entity_quality_scores(train_df: pl.DataFrame, valid_df: pl.DataFrame) -> dict[str, Any]:
    """Compute simple entity quality scores based on degree frequency (lightweight, deterministic)."""
    entities = pl.concat([train_df["s"], train_df["o"], valid_df["s"], valid_df["o"]])
    degree_counts = entities.value_counts().rename({"s": "entity", "count": "degree"})
    max_degree = max(1, int(degree_counts["degree"].max()))  # type: ignore
    degree_counts = degree_counts.with_columns((pl.col("degree") / max_degree).alias("degree_norm"))
    return {
        "degree": degree_counts,
        "max_degree": max_degree,
        "n_entities_with_degree": int(len(degree_counts)),
    }


def _load_inverse_filter_settings(file_manager: FileManager) -> tuple[str, str]:
    """Load inverse-relation filtering policy from HPO defaults."""
    try:
        payload = file_manager.read(OPTIMIZATION_CONFIG_PATH)
        cfg = payload.to_native() if isinstance(payload, ParquetBundle) else payload or {}
        defaults_cfg = cfg.get("defaults", {}) if isinstance(cfg, dict) else {}

        raw_policy = str(defaults_cfg.get("inverse_relation_policy", INVERSE_POLICY_KEEP)).strip()
        policy = raw_policy.lower()
        if policy not in ALLOWED_INVERSE_POLICIES:
            logger.warning(
                f"Invalid inverse_relation_policy={raw_policy!r}; "
                f"falling back to {INVERSE_POLICY_KEEP!r}."
            )
            policy = INVERSE_POLICY_KEEP

        raw_suffix = str(defaults_cfg.get("inverse_suffix", "_inv")).strip()
        suffix = raw_suffix or "_inv"
        return policy, suffix
    except Exception as exc:
        logger.warning(f"Failed to load inverse filter settings; using defaults: {exc}")
        return INVERSE_POLICY_KEEP, "_inv"


def _iter_relation_map_candidates(
    preprocessing_config: PreprocessingConfig | None,
) -> list[Path]:
    """Build candidate relation-map paths used to resolve inverse IDs."""
    from pff_rust import stable_hash

    hashed_name = f"relation_map_{stable_hash('splits')}.parquet"
    candidates: list[Path] = []
    if preprocessing_config is not None:
        candidates.append(Path(preprocessing_config.output_dir) / hashed_name)
    candidates.append(Path("outputs/preprocessing") / hashed_name)
    candidates.append(settings.OUTPUTS_DIR / "kg" / "mappings" / "relation_map.parquet")
    return candidates


def _resolve_inverse_relation_ids(
    *,
    file_manager: FileManager,
    preprocessing_config: PreprocessingConfig | None,
    inverse_suffix: str,
) -> tuple[set[int], list[str]]:
    """Resolve integer relation IDs whose semantic label ends with inverse suffix."""
    if not inverse_suffix:
        return set(), []

    seen: set[Path] = set()
    for candidate in _iter_relation_map_candidates(preprocessing_config):
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not file_manager.exists(resolved):
            continue
        try:
            relation_map = _as_dataframe(file_manager.read(resolved))
        except Exception as exc:
            logger.debug(f"Failed reading relation map candidate {resolved}: {exc}")
            continue

        id_col = "relation_id" if "relation_id" in relation_map.columns else "id"
        label_col = "relation" if "relation" in relation_map.columns else "label"
        if id_col not in relation_map.columns or label_col not in relation_map.columns:
            continue

        try:
            normalized = relation_map.select(
                [
                    pl.col(id_col).cast(pl.Int64).alias("id"),
                    pl.col(label_col).cast(pl.Utf8).alias("label"),
                ]
            )
        except Exception as exc:
            logger.debug(f"Failed normalizing relation map {resolved}: {exc}")
            continue

        inverse_rows = normalized.filter(pl.col("label").str.ends_with(inverse_suffix))
        if inverse_rows.height == 0:
            continue
        inverse_ids = {int(value) for value in inverse_rows["id"].to_list()}
        inverse_labels = [str(value) for value in inverse_rows["label"].to_list()]
        return inverse_ids, inverse_labels

    return set(), []


def _drop_inverse_relations_for_split(
    df: pl.DataFrame | None,
    *,
    inverse_suffix: str,
    inverse_relation_ids: set[int],
) -> tuple[pl.DataFrame | None, int]:
    """Drop inverse relations from a split and return removed count."""
    if df is None:
        return None, 0
    if "p" not in df.columns:
        return df, 0

    relation_dtype = df.schema.get("p")
    if relation_dtype is not None and relation_dtype.is_integer():
        if not inverse_relation_ids:
            return df, 0
        filtered = df.filter(~pl.col("p").is_in(sorted(inverse_relation_ids)))
        return filtered, int(len(df) - len(filtered))

    if not inverse_suffix:
        return df, 0

    filtered = df.filter(~pl.col("p").cast(pl.Utf8).str.ends_with(inverse_suffix))
    return filtered, int(len(df) - len(filtered))


def _apply_inverse_relation_policy(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    test_df: pl.DataFrame | None,
    *,
    policy: str,
    inverse_suffix: str,
    file_manager: FileManager,
    preprocessing_config: PreprocessingConfig | None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None, dict[str, Any]]:
    """Apply inverse relation filtering policy to loaded splits."""
    stats: dict[str, Any] = {
        "policy": policy,
        "suffix": inverse_suffix,
        "removed": 0,
        "removed_by_split": {"train": 0, "valid": 0, "test": 0},
        "filtered_relation_ids": [],
        "filtered_relation_labels": [],
    }
    if policy != INVERSE_POLICY_DROP_SUFFIX:
        return train_df, valid_df, test_df, stats

    inverse_relation_ids: set[int] = set()
    inverse_labels: list[str] = []
    relation_dtype = train_df.schema.get("p")
    if relation_dtype is not None and relation_dtype.is_integer():
        inverse_relation_ids, inverse_labels = _resolve_inverse_relation_ids(
            file_manager=file_manager,
            preprocessing_config=preprocessing_config,
            inverse_suffix=inverse_suffix,
        )
        if not inverse_relation_ids:
            logger.warning(
                "Inverse relation filter requested but no inverse IDs were resolved from relation maps. "
                "Skipping filter for integer relation IDs."
            )
            stats["skip_reason"] = "inverse_ids_not_resolved"
            return train_df, valid_df, test_df, stats

    train_filtered, train_removed = _drop_inverse_relations_for_split(
        train_df,
        inverse_suffix=inverse_suffix,
        inverse_relation_ids=inverse_relation_ids,
    )
    valid_filtered, valid_removed = _drop_inverse_relations_for_split(
        valid_df,
        inverse_suffix=inverse_suffix,
        inverse_relation_ids=inverse_relation_ids,
    )
    test_filtered, test_removed = _drop_inverse_relations_for_split(
        test_df,
        inverse_suffix=inverse_suffix,
        inverse_relation_ids=inverse_relation_ids,
    )

    total_removed = train_removed + valid_removed + test_removed
    stats["removed"] = int(total_removed)
    stats["removed_by_split"] = {
        "train": int(train_removed),
        "valid": int(valid_removed),
        "test": int(test_removed),
    }
    stats["filtered_relation_ids"] = sorted(inverse_relation_ids)
    stats["filtered_relation_labels"] = sorted(inverse_labels)

    if total_removed > 0:
        logger.info(
            "Remocao de relacoes inversas concluida: "
            f"total={total_removed:,}, train={train_removed:,}, "
            f"valid={valid_removed:,}, test={test_removed:,}"
        )
    else:
        logger.info("Nenhuma relacao inversa encontrada para remocao")

    assert train_filtered is not None
    assert valid_filtered is not None
    return train_filtered, valid_filtered, test_filtered, stats


def validate_split_consistency(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    source: str,
) -> dict[str, Any]:
    """Validate split consistency with deterministic hashing.

    Computes a stable hash of s/p/o columns and counts to detect when different
    data sources return inconsistent splits across executions.

    Args:
        train_df: Training split DataFrame with s, p, o columns.
        valid_df: Validation split DataFrame with s, p, o columns.
        source: Data source identifier (e.g., 'postgresql', 'parquet').

    Returns:
        Dictionary with hash, counts, and source info for logging/verification.
    """
    from pff_rust import hash_bytes, stable_hash

    combined = pl.concat([train_df, valid_df]).sort(["s", "p", "o"])
    selected = combined.select(["s", "p", "o"])

    is_integer_frame = (
        selected["s"].dtype.is_integer()
        and selected["p"].dtype.is_integer()
        and selected["o"].dtype.is_integer()
    )
    has_nulls = (
        selected["s"].null_count() > 0
        or selected["p"].null_count() > 0
        or selected["o"].null_count() > 0
    )

    if is_integer_frame and not has_nulls:
        data_hash_int = hash_bytes(selected.to_numpy().tobytes())
    else:
        data_hash_int = stable_hash(selected.write_json())

    data_hash = format(data_hash_int, "016x")

    n_entities = _count_unique_arrow(combined["s"], combined["o"])
    n_relations = _count_unique_arrow(combined["p"])

    stats = {
        "hash": data_hash,
        "hash_short": data_hash[:16],
        "source": source,
        "train_triples": len(train_df),
        "valid_triples": len(valid_df),
        "total_triples": len(combined),
        "entities": n_entities,
        "relations": n_relations,
    }

    logger.info(
        f"Consistencia de splits validada: source={source}, "
        f"hash={data_hash[:12]}..., train={stats['train_triples']:,}, "
        f"valid={stats['valid_triples']:,}, entidades={stats['entities']:,}"
    )

    return stats


def _pick_existing_path(candidates: list[Path]) -> Path:
    """Return the first existing path from candidates or raise a detailed error."""
    for path in candidates:
        if FileManager.exists(path):
            return path
    raise FileNotFoundError(
        f"Training/validation data not found in candidates: {', '.join(str(p) for p in candidates)}"
    )


def _get_preprocessed_parquet_baseline(
    file_manager: FileManager,
) -> dict[str, float] | None:
    """Execute get preprocessed parquet baseline.



    Args:

        file_manager: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    outputs_dir = settings.OUTPUTS_DIR / "kg" / "mappings"
    train_path = outputs_dir / "train.preprocessed.parquet"
    valid_path = outputs_dir / "valid.preprocessed.parquet"

    if not file_manager.exists(train_path) or not file_manager.exists(valid_path):
        return None

    try:
        train_rows = float(pq.ParquetFile(train_path).metadata.num_rows)
        valid_rows = float(pq.ParquetFile(valid_path).metadata.num_rows)
    except Exception as exc:
        logger.debug(f"Failed to read preprocessed parquet baseline: {exc}")
        return None

    return {
        "train_len": train_rows,
        "valid_len": valid_rows,
    }


def _get_kg_paths(
    file_manager: FileManager | None = None, prefer_outputs: bool = True
) -> tuple[Path, Path]:
    """
    Resolve train and validation parquet paths.

    Args:
        file_manager: Optional FileManager instance
        prefer_outputs: If True, prefer outputs/kg (preprocessed) over data/models/kg (raw)

    Returns:
        Tuple of (train_path, valid_path)
    """
    kg_config = load_config(KG_PIPELINE_CONFIG_PATH) or {}
    paths_cfg = kg_config.get("paths", {})
    data_dir = Path(paths_cfg.get("data_dir", settings.DATA_DIR))
    graph_subdir = paths_cfg.get("graph_subdir", "models/kg")
    graph_dir = data_dir / graph_subdir
    outputs_graph_dir = settings.OUTPUTS_DIR / "kg"

    if prefer_outputs:
        train_candidates = [
            outputs_graph_dir / "train_optimized.parquet",
            outputs_graph_dir / "train.parquet",
            graph_dir / "train_optimized.parquet",
            graph_dir / "train.parquet",
        ]
        valid_candidates = [
            outputs_graph_dir / "valid_optimized.parquet",
            outputs_graph_dir / "valid.parquet",
            graph_dir / "valid_optimized.parquet",
            graph_dir / "valid.parquet",
        ]
    else:
        train_candidates = [
            graph_dir / "train_optimized.parquet",
            graph_dir / "train.parquet",
            outputs_graph_dir / "train_optimized.parquet",
            outputs_graph_dir / "train.parquet",
        ]
        valid_candidates = [
            graph_dir / "valid_optimized.parquet",
            graph_dir / "valid.parquet",
            outputs_graph_dir / "valid_optimized.parquet",
            outputs_graph_dir / "valid.parquet",
        ]

    train_path = _pick_existing_path(train_candidates)
    valid_path = _pick_existing_path(valid_candidates)

    return train_path, valid_path


def load_kg_data_lazy(
    file_manager: FileManager | None = None,
) -> tuple[pl.LazyFrame, pl.LazyFrame, dict[str, Any]]:
    """
    Load KG data as LazyFrames for memory-efficient processing.

    SOTA: Uses Polars lazy evaluation to defer computation until needed.
    Useful for large datasets where full materialization is expensive.

    Args:
        file_manager: Optional FileManager instance

    Returns:
        Tuple of (train_lazy, valid_lazy, data_info)
    """
    fm = file_manager or FileManager()
    train_path, valid_path = _get_kg_paths(fm)

    train_lazy: pl.LazyFrame = _as_lazy_frame(fm.read(train_path))
    valid_lazy: pl.LazyFrame = _as_lazy_frame(fm.read(valid_path))

    data_info = {
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "lazy": True,
    }

    logger.info(f"Dados KG carregados (lazy): train={train_path.name}, valid={valid_path.name}")

    return train_lazy, valid_lazy, data_info


def load_real_kg_data(
    file_manager: FileManager | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """
    Load real PFF Knowledge Graph data using the FileManager abstraction.

    Args:
        file_manager: Optional FileManager instance for I/O operations

    Returns:
        Tuple of (train_df, valid_df, data_info dict)

    Raises:
        FileNotFoundError: If training/validation data not found
        RuntimeError: If loaded data is not Polars DataFrame
    """
    fm = file_manager or FileManager()
    train_path, valid_path = _get_kg_paths(fm)

    train_df: pl.DataFrame = _as_dataframe(fm.read(train_path, return_native=True))
    valid_df: pl.DataFrame = _as_dataframe(fm.read(valid_path, return_native=True))

    n_entities = _count_unique_arrow(train_df["s"], train_df["o"], valid_df["s"], valid_df["o"])
    n_predicates = _count_unique_arrow(train_df["p"], valid_df["p"])

    entity_quality_scores = compute_entity_quality_scores(train_df, valid_df)

    data_info = {
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_entities": n_entities,
        "n_predicates": n_predicates,
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "entity_quality_scores": entity_quality_scores,
    }

    logger.info(
        f"Dados reais carregados (Polars): treino={data_info['n_train']}, "
        f"valid={data_info['n_valid']}, entidades={data_info['n_entities']}, "
        f"predicados={data_info['n_predicates']}"
    )

    return train_df, valid_df, data_info


def load_synthetic_kg_data(
    file_manager: FileManager | None = None,
    config_path: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """
    Generate a small deterministic synthetic KG dataset for smoke runs.

    Args:
        file_manager: Optional FileManager instance.
        config_path: Optional config path override for synthetic settings.

    Returns:
        Tuple of (train_df, valid_df, data_info dict).
    """
    fm = file_manager or FileManager()
    cfg_path = config_path or OPTIMIZATION_CONFIG_PATH
    cfg_payload = fm.read(cfg_path)
    cfg = (cfg_payload.to_native() if isinstance(cfg_payload, ParquetBundle) else cfg_payload) or {}
    defaults_cfg = cfg.get("synthetic_data", {}) if isinstance(cfg, dict) else {}

    n_entities = int(defaults_cfg.get("n_entities", 64))
    n_relations = int(defaults_cfg.get("n_relations", 8))
    train_size = int(defaults_cfg.get("train_size", 256))
    valid_size = int(defaults_cfg.get("valid_size", 64))
    seed = int(defaults_cfg.get("seed", 42))

    rng = np.random.default_rng(seed)
    train_df = pl.DataFrame(
        {
            "s": rng.integers(0, n_entities, train_size, dtype=np.int64),
            "p": rng.integers(0, n_relations, train_size, dtype=np.int64),
            "o": rng.integers(0, n_entities, train_size, dtype=np.int64),
        }
    )
    valid_df = pl.DataFrame(
        {
            "s": rng.integers(0, n_entities, valid_size, dtype=np.int64),
            "p": rng.integers(0, n_relations, valid_size, dtype=np.int64),
            "o": rng.integers(0, n_entities, valid_size, dtype=np.int64),
        }
    )

    data_info = {
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_entities": n_entities,
        "n_predicates": n_relations,
        "source": "synthetic",
    }

    logger.info(
        "Dados sinteticos gerados: "
        f"treino={data_info['n_train']}, valid={data_info['n_valid']}, "
        f"entidades={data_info['n_entities']}, relacoes={data_info['n_predicates']}"
    )

    return train_df, valid_df, data_info


def load_real_kg_data_with_preprocessing(
    file_manager: FileManager | None = None,
    use_centralized: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """
    Load real KG data with centralized preprocessing applied.

    This function ensures HPO uses the SAME preprocessing as the main pipeline,
    preventing inconsistencies between optimization and production.

    Args:
        file_manager: Optional FileManager instance
        use_centralized: Whether to use centralized preprocessing module

    Returns:
        Tuple of (train_df, valid_df, data_info dict)
    """
    if not use_centralized or not HAS_PREPROCESSING_MODULE:
        logger.info("Carregamento padrão de dados (preprocessamento centralizado desativado)")
        return load_real_kg_data(file_manager)

    fm = file_manager or FileManager()
    train_path, valid_path = _get_kg_paths(fm)

    config_path = settings.CONFIG_DIR / "preprocessing.yaml"
    if fm.exists(config_path):
        config = PreprocessingConfig.from_yaml(config_path)
        logger.info(f"HPO: Usando config de preprocessing de {config_path}")
    else:
        config = PreprocessingConfig()
        logger.info("HPO: Usando config de preprocessing padrao")

    train_df: pl.DataFrame = _as_dataframe(fm.read(train_path))
    valid_df: pl.DataFrame = _as_dataframe(fm.read(valid_path))

    pipeline = KGPreprocessingPipeline(config)

    logger.info("HPO: Aplicando preprocessing centralizado...")
    empty_test_df = pl.DataFrame({"s": [], "p": [], "o": []}).cast(
        {"s": pl.Utf8, "p": pl.Utf8, "o": pl.Utf8}
    )
    result = pipeline.preprocess_splits(train_df, valid_df, empty_test_df)
    try:
        saved_paths = pipeline.save_preprocessed(
            result,
            settings.OUTPUTS_DIR / "kg",
            suffix="_preprocessed",
        )
        logger.info(f"HPO: Splits preprocessados salvos em {saved_paths}")
    except Exception as save_exc:
        logger.warning(f"Failed to persist processed splits: {save_exc}")

    train_preprocessed = result.train
    valid_preprocessed = result.valid if result.valid is not None else valid_df

    n_entities = int(
        pl.concat(
            [
                train_preprocessed["s"],
                train_preprocessed["o"],
                valid_preprocessed["s"],
                valid_preprocessed["o"],
            ]
        )
        .unique()
        .len()
    )
    n_predicates = int(pl.concat([train_preprocessed["p"], valid_preprocessed["p"]]).unique().len())

    entity_quality_scores = compute_entity_quality_scores(train_preprocessed, valid_preprocessed)

    data_info = {
        "n_train": len(train_preprocessed),
        "n_valid": len(valid_preprocessed),
        "n_entities": n_entities,
        "n_predicates": n_predicates,
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "entity_quality_scores": entity_quality_scores,
        "preprocessing_applied": True,
        "preprocessing_stats": result.stats,
        "saved_paths": saved_paths if "saved_paths" in locals() else {},
    }

    logger.info(
        f"HPO: Dados preprocessados: treino={data_info['n_train']}, "
        f"valid={data_info['n_valid']}, entidades={data_info['n_entities']}, "
        f"predicados={data_info['n_predicates']}"
    )

    return train_preprocessed, valid_preprocessed, data_info


async def _load_from_postgres_preprocessed() -> tuple[
    pl.DataFrame | None, pl.DataFrame | None, dict
]:
    """
    Load preprocessed data directly from PostgreSQL.

    Returns:
        Tuple of (train_df, valid_df, metadata) or (None, None, {}) if not available
    """
    try:
        from pff.infrastructure.persistence.db.repositories import KGSplitsRepository

        repo = KGSplitsRepository()
        train_df, valid_df, _test_df, metadata = await repo.load_preprocessed_splits(
            fallback_to_raw=False,
            map_to_ints=False,
        )

        if (
            train_df is not None
            and valid_df is not None
            and len(train_df) > 0
            and len(valid_df) > 0
        ):
            return train_df, valid_df, metadata
        return None, None, {}

    except ImportError:
        logger.debug("KGSplitsRepository not available")
        return None, None, {}
    except Exception as e:
        logger.warning(f"Failed to load from PostgreSQL: {e}")
        return None, None, {}


async def _save_preprocessed_to_postgres(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
) -> bool:
    """
    Save preprocessed data to PostgreSQL for future fast loading.

    Args:
        train_df: Preprocessed training data
        valid_df: Preprocessed validation data

    Returns:
        True if successful
    """
    try:
        from pff.infrastructure.persistence.db.repositories import KGSplitsRepository

        repo = KGSplitsRepository()

        await repo.delete_preprocessed()

        await repo.save_preprocessed_splits(train_df, valid_df)

        return True

    except ImportError:
        logger.debug("KGSplitsRepository not available")
        return False
    except Exception as e:
        logger.warning(f"Failed to save processed to PostgreSQL: {e}")
        return False


async def _get_postgres_raw_baseline() -> dict[str, float] | None:
    """Load raw split counts from PostgreSQL to validate preprocessed size."""
    try:
        from pff.infrastructure.persistence.db.repositories import KGSplitsRepository

        repo = KGSplitsRepository()
        stats = await repo.get_statistics()
        train_raw = stats.get("train/raw", {}).get("count", 0)
        valid_raw = stats.get("valid/raw", {}).get("count", 0)
        if train_raw or valid_raw:
            return {
                "train_len": float(train_raw),
                "valid_len": float(valid_raw),
            }
    except ImportError:
        logger.debug("KGSplitsRepository not available")
    except Exception as exc:
        logger.debug(f"Failed to read PostgreSQL raw baseline: {exc}")
    return None


def _get_local_baseline_counts(
    file_manager: FileManager,
) -> dict[str, float] | None:
    """Execute get local baseline counts.



    Args:

        file_manager: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    try:
        train_path, valid_path = _get_kg_paths(file_manager, prefer_outputs=True)
        train_base = _as_dataframe(file_manager.read(train_path))
        valid_base = _as_dataframe(file_manager.read(valid_path))
        return {
            "train_len": float(len(train_base)),
            "valid_len": float(len(valid_base)),
            "relations": float(_count_unique_arrow(train_base["p"], valid_base["p"])),
        }
    except Exception as exc:
        logger.debug(f"Failed to compute local baseline: {exc}")
        return None


def _postprocess_preprocessed_splits(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    preprocessing_config: PreprocessingConfig | None,
    *,
    apply_id_mapping: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any] | None]:
    """Execute postprocess preprocessed splits.



    Args:

        train_df: Input value used by this callable.

        valid_df: Input value used by this callable.

        preprocessing_config: Input value used by this callable.

        apply_id_mapping: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    attr_stats: dict[str, Any] | None = None
    if apply_id_mapping:
        train_cast, valid_cast = _cast_preindexed_string_ids(train_df, valid_df)
        if train_cast is not None and valid_cast is not None:
            train_df, valid_df = train_cast, valid_cast
        elif HAS_PREPROCESSING_MODULE:
            pipeline = KGPreprocessingPipeline()
            mapped_train, mapped_valid, _ = pipeline._map_ids_for_splits(train_df, valid_df, None)
            if mapped_train is not None:
                train_df = mapped_train
            if mapped_valid is not None:
                valid_df = mapped_valid
    if preprocessing_config:
        filtered_train, filtered_valid, _, attr_stats = filter_attribute_relations(
            train_df, valid_df, None, preprocessing_config
        )
        if filtered_train is not None:
            train_df = filtered_train
        if filtered_valid is not None:
            valid_df = filtered_valid
    return train_df, valid_df, attr_stats


def _cast_preindexed_string_ids(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Cast pre-indexed s/p/o string IDs to integers without remapping labels."""

    required_cols = ("s", "p", "o")
    if not all(col in train_df.columns for col in required_cols):
        return None, None
    if not all(col in valid_df.columns for col in required_cols):
        return None, None

    def _is_string_numeric_frame(df: pl.DataFrame) -> bool:
        for col in required_cols:
            dtype = df.schema[col]
            if dtype != pl.Utf8:
                return False
            if df.select(pl.col(col).is_null().any()).item():
                return False
            numeric_mask = df.select(pl.col(col).str.contains(r"^\d+$").all()).item()
            if not bool(numeric_mask):
                return False
        return True

    if not (_is_string_numeric_frame(train_df) and _is_string_numeric_frame(valid_df)):
        return None, None

    cast_exprs = [pl.col(col).cast(pl.Int64).alias(col) for col in required_cols]
    return train_df.with_columns(cast_exprs), valid_df.with_columns(cast_exprs)


def _load_preprocessed_with_postprocessing(
    *,
    preprocessing_config: PreprocessingConfig | None,
    timeout_s: float,
    apply_id_mapping: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any], dict[str, Any] | None] | None:
    """Execute load preprocessed with postprocessing.



    Args:

        preprocessing_config: Input value used by this callable.

        timeout_s: Input value used by this callable.

        apply_id_mapping: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    train_df, valid_df, metadata = run_coroutine_sync(
        _load_from_postgres_preprocessed(),
        timeout_s=timeout_s,
    )
    if train_df is None or valid_df is None:
        return None
    train_df, valid_df, attr_stats = _postprocess_preprocessed_splits(
        train_df,
        valid_df,
        preprocessing_config,
        apply_id_mapping=apply_id_mapping,
    )
    return train_df, valid_df, metadata, attr_stats


def _is_below_baseline_threshold(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    baseline_counts: dict[str, float] | None,
    preprocessed_baseline: dict[str, float] | None = None,
) -> bool:
    """Execute is below baseline threshold.



    Args:

        train_df: Input value used by this callable.

        valid_df: Input value used by this callable.

        baseline_counts: Input value used by this callable.

        preprocessed_baseline: Optional input value.



    Returns:

        Return value produced by the callable.

    """

    baseline_train = baseline_counts["train_len"] if baseline_counts else None
    baseline_valid = baseline_counts["valid_len"] if baseline_counts else None
    baseline_relations = baseline_counts.get("relations") if baseline_counts else None
    if preprocessed_baseline:
        baseline_train = max(baseline_train or 0.0, preprocessed_baseline["train_len"])
        baseline_valid = max(baseline_valid or 0.0, preprocessed_baseline["valid_len"])
    if not (baseline_train or baseline_valid or baseline_relations):
        return False
    relation_count = _count_unique_arrow(train_df["p"], valid_df["p"])
    return bool(
        (baseline_train and len(train_df) < baseline_train * 0.5)
        or (baseline_valid and len(valid_df) < baseline_valid * 0.5)
        or (baseline_relations and relation_count < baseline_relations * 0.5)
    )


def _build_preprocessed_data_info(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    *,
    attr_stats: dict[str, Any] | None,
    inverse_stats: dict[str, Any] | None = None,
    populated_by: str | None = None,
) -> dict[str, Any]:
    """Execute build preprocessed data info.



    Args:

        train_df: Input value used by this callable.

        valid_df: Input value used by this callable.

        attr_stats: Input value used by this callable.

        populated_by: Optional input value.



    Returns:

        Return value produced by the callable.

    """

    entity_unique = _count_unique_arrow(train_df["s"], train_df["o"], valid_df["s"], valid_df["o"])
    relation_unique = _count_unique_arrow(train_df["p"], valid_df["p"])
    entity_upper_bound = _infer_id_upper_bound(
        train_df["s"], train_df["o"], valid_df["s"], valid_df["o"]
    )
    relation_upper_bound = _infer_id_upper_bound(train_df["p"], valid_df["p"])
    n_entities = max(entity_unique, entity_upper_bound + 1 if entity_upper_bound >= 0 else 0)
    n_predicates = max(
        relation_unique, relation_upper_bound + 1 if relation_upper_bound >= 0 else 0
    )
    entity_quality_scores = compute_entity_quality_scores(train_df, valid_df)
    data_info: dict[str, Any] = {
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_entities": n_entities,
        "n_predicates": n_predicates,
        "source": "postgresql_preprocessed",
        "entity_quality_scores": entity_quality_scores,
        "preprocessing_applied": True,
        "attribute_filter": attr_stats or {},
        "inverse_filter": inverse_stats
        or {
            "policy": INVERSE_POLICY_KEEP,
            "suffix": "_inv",
            "removed": 0,
            "removed_by_split": {"train": 0, "valid": 0, "test": 0},
            "filtered_relation_ids": [],
            "filtered_relation_labels": [],
        },
    }
    if populated_by is not None:
        data_info["populated_by"] = populated_by
    return data_info


def _try_load_existing_preprocessed(
    *,
    preprocessing_config: PreprocessingConfig | None,
    baseline_counts: dict[str, float] | None,
    preprocessed_baseline: dict[str, float] | None,
    config_path: Path | None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]] | None:
    """Execute try load existing preprocessed.



    Args:

        preprocessing_config: Input value used by this callable.

        baseline_counts: Input value used by this callable.

        preprocessed_baseline: Input value used by this callable.

        config_path: Input value used by this callable.



    Returns:

        Return value produced by the callable.

    """

    try:
        loaded = _load_preprocessed_with_postprocessing(
            preprocessing_config=preprocessing_config,
            timeout_s=30.0,
            apply_id_mapping=True,
        )
        if loaded is None:
            return None
        train_df, valid_df, _, attr_stats = loaded
        if _is_below_baseline_threshold(
            train_df,
            valid_df,
            baseline_counts,
            preprocessed_baseline,
        ):
            logger.warning(
                "Preprocessed PostgreSQL splits look stale (size/relations far below local baseline). "
                "Re-running preprocessing pipeline to repopulate Postgres."
            )
            if _populate_preprocessed_splits(config_path=config_path):
                try:
                    reloaded = _load_preprocessed_with_postprocessing(
                        preprocessing_config=preprocessing_config,
                        timeout_s=30.0,
                        apply_id_mapping=False,
                    )
                    if reloaded is not None:
                        train_df, valid_df, _, attr_stats = reloaded
                except Exception as retry_exc:
                    logger.warning(f"Reload after repopulation failed: {retry_exc}")
        logger.success("Dados preprocessados carregados do PostgreSQL (fonte única)")
        data_info = _build_preprocessed_data_info(
            train_df,
            valid_df,
            attr_stats=attr_stats,
        )
        logger.info(
            f"PostgreSQL: treino={data_info['n_train']:,}, "
            f"valid={data_info['n_valid']:,}, entidades={data_info['n_entities']:,}"
        )
        return train_df, valid_df, data_info
    except Exception as exc:
        logger.debug(f"PostgreSQL load failed: {exc}")
        return None


def _try_populate_and_reload_preprocessed(
    *,
    preprocessing_config: PreprocessingConfig | None,
    baseline_counts: dict[str, float] | None,
    config_path: Path | None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]] | None:
    """Execute try populate and reload preprocessed.



    Args:

        preprocessing_config: Input value used by this callable.

        baseline_counts: Input value used by this callable.

        config_path: Input value used by this callable.



    Returns:

        Return value produced by the callable.



    Raises:

        Exception: Propagates domain-specific failures with context.

    """

    logger.info(
        "Preprocessed splits ausentes no PostgreSQL. Executando build+preprocess (KG pipeline)..."
    )

    def _fallback_from_correct_parquet(
        reason: str,
    ) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]] | None:
        logger.bind(
            component="hpo_data_loader",
            stop_reason="fallback_to_correct_parquet",
            key_parameters={"reason": reason},
        ).warning("Falling back to correct.parquet materialization.")
        fallback_loaded = _load_from_parquet_and_push(
            preprocessing_config=preprocessing_config,
            file_manager=FileManager(),
            config_path=config_path,
        )
        if fallback_loaded is None:
            return None
        train_df, valid_df, test_df = fallback_loaded
        data_info = _build_preprocessed_data_info(
            train_df,
            valid_df,
            attr_stats=None,
            populated_by="correct_parquet",
        )
        data_info["source"] = "correct_parquet"
        if test_df is not None:
            data_info["n_test"] = int(len(test_df))
        return train_df, valid_df, data_info

    if not _populate_preprocessed_splits(config_path=config_path):
        return _fallback_from_correct_parquet("kg_pipeline_population_failed")
    try:
        reloaded = _load_preprocessed_with_postprocessing(
            preprocessing_config=preprocessing_config,
            timeout_s=30.0,
            apply_id_mapping=False,
        )
        if reloaded is None:
            return None
        train_df, valid_df, _, attr_stats = reloaded
        if _is_below_baseline_threshold(
            train_df,
            valid_df,
            baseline_counts,
        ):
            raise RuntimeError(
                "Repopulated splits are still below the local baseline. "
                "Fix KG preprocessing and rerun the pipeline."
            )
        data_info = _build_preprocessed_data_info(
            train_df,
            valid_df,
            attr_stats=attr_stats,
            populated_by="kg_pipeline",
        )
        logger.bind(
            component="hpo_data_loader",
            stop_reason="data_loaded",
            key_parameters={
                "treino": data_info["n_train"],
                "valid": data_info["n_valid"],
                "entidades": data_info["n_entities"],
            },
        ).success("Dados preprocessados carregados do PostgreSQL.")
        return train_df, valid_df, data_info
    except Exception as retry_exc:
        logger.bind(
            component="hpo_data_loader",
            stop_reason="reload_failed",
            key_parameters={"error": repr(retry_exc)},
        ).warning("Retry load after populate failed.")
        return _fallback_from_correct_parquet("reload_after_population_failed")


def _apply_hpo_inverse_filter_to_loaded_data(
    *,
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    data_info: dict[str, Any],
    file_manager: FileManager,
    preprocessing_config: PreprocessingConfig | None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Apply HPO inverse-relation policy after loading preprocessed splits."""
    policy, inverse_suffix = _load_inverse_filter_settings(file_manager)
    filtered_train, filtered_valid, _unused_test, inverse_stats = _apply_inverse_relation_policy(
        train_df,
        valid_df,
        None,
        policy=policy,
        inverse_suffix=inverse_suffix,
        file_manager=file_manager,
        preprocessing_config=preprocessing_config,
    )

    rebuilt_info = _build_preprocessed_data_info(
        filtered_train,
        filtered_valid,
        attr_stats=cast(dict[str, Any] | None, data_info.get("attribute_filter")),
        inverse_stats=inverse_stats,
        populated_by=cast(str | None, data_info.get("populated_by")),
    )
    for key, value in data_info.items():
        if key not in rebuilt_info:
            rebuilt_info[key] = value
    source = data_info.get("source")
    if source is not None:
        rebuilt_info["source"] = source
    if "n_test" in data_info:
        rebuilt_info["n_test"] = int(data_info["n_test"])
    return filtered_train, filtered_valid, rebuilt_info


def load_preprocessed_from_postgres(
    file_manager: FileManager | None = None,
    require_preprocessed: bool = True,
    auto_populate_if_missing: bool = True,
    config_path: Path | None = None,
    allow_fallback: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """
    Load preprocessed KG data from PostgreSQL (single source of truth).

    This is the PREFERRED method for HPO and pff learn to ensure consistency.
    Avoids file fallbacks unless explicitly enabled.

    Flow:
    1. Try loading preprocessed from PostgreSQL
    2. If not available, load raw + preprocess + save to PostgreSQL
    3. Return preprocessed data

    Args:
        file_manager: Optional FileManager instance
        allow_fallback: If True, allow parquet-based fallback when Postgres reload fails.

    Returns:
        Tuple of (train_df, valid_df, data_info dict)
    """
    preprocessing_config = PreprocessingConfig.from_yaml() if HAS_PREPROCESSING_MODULE else None
    fm = file_manager or FileManager()
    baseline_counts = _get_local_baseline_counts(fm)
    if baseline_counts is None:
        baseline_counts = run_coroutine_sync(_get_postgres_raw_baseline())
    preprocessed_baseline = _get_preprocessed_parquet_baseline(fm)

    existing = _try_load_existing_preprocessed(
        preprocessing_config=preprocessing_config,
        baseline_counts=baseline_counts,
        preprocessed_baseline=preprocessed_baseline,
        config_path=config_path,
    )
    if existing is not None:
        train_df, valid_df, data_info = existing
        return _apply_hpo_inverse_filter_to_loaded_data(
            train_df=train_df,
            valid_df=valid_df,
            data_info=data_info,
            file_manager=fm,
            preprocessing_config=preprocessing_config,
        )

    if auto_populate_if_missing:
        populated = _try_populate_and_reload_preprocessed(
            preprocessing_config=preprocessing_config,
            baseline_counts=baseline_counts,
            config_path=config_path,
        )
        if populated is not None:
            train_df, valid_df, data_info = populated
            return _apply_hpo_inverse_filter_to_loaded_data(
                train_df=train_df,
                valid_df=valid_df,
                data_info=data_info,
                file_manager=fm,
                preprocessing_config=preprocessing_config,
            )

    if allow_fallback:
        fallback_loaded = _load_from_parquet_and_push(
            preprocessing_config=preprocessing_config,
            file_manager=fm,
            config_path=config_path,
        )
        if fallback_loaded is not None:
            train_df, valid_df, test_df = fallback_loaded
            data_info = _build_preprocessed_data_info(
                train_df,
                valid_df,
                attr_stats=None,
                populated_by="parquet_fallback",
            )
            data_info["source"] = "parquet_fallback"
            if test_df is not None:
                data_info["n_test"] = int(len(test_df))
            return _apply_hpo_inverse_filter_to_loaded_data(
                train_df=train_df,
                valid_df=valid_df,
                data_info=data_info,
                file_manager=fm,
                preprocessing_config=preprocessing_config,
            )

    if require_preprocessed:
        raise RuntimeError(
            "Preprocessed KG splits not available in PostgreSQL. "
            "Populate via KGSplitsRepository.save_preprocessed_splits before running HPO."
        )

    logger.bind(
        component="hpo_data_loader",
        stop_reason="fallback_disabled",
        key_parameters={},
    ).error("Preprocessed KG splits unavailable and fallback disabled.")
    raise RuntimeError("Preprocessed KG splits unavailable and fallback disabled.")


def _populate_preprocessed_splits(config_path: Path | None = None) -> bool:
    """Run KG pipeline build+preprocess to populate PostgreSQL with preprocessed splits."""
    try:
        from pff.domain.kg.config import KGConfig
        from pff.domain.kg.pipeline import KGPipeline
        from pff.infrastructure.persistence.db.repositories import KGSplitsRepository
        from pff.infrastructure.persistence.db.repositories.pipeline_checkpoints import (
            PipelineCheckpointsRepository,
        )

        cfg = KGConfig(config_path or KG_PIPELINE_CONFIG_PATH)
        splits_repo = KGSplitsRepository()

        async def _raw_exists() -> bool:
            """Check whether raw splits are already present in PostgreSQL."""
            train_exists = await splits_repo.split_exists("train", "raw")
            valid_exists = await splits_repo.split_exists("valid", "raw")
            test_exists = await splits_repo.split_exists("test", "raw")
            return bool(train_exists and valid_exists and test_exists)

        has_raw = bool(run_coroutine_sync(_raw_exists(), timeout_s=20.0))
        if not has_raw:
            logger.info(
                "Raw splits ausentes no PostgreSQL. Gerando train/valid/test a partir de correct.parquet..."
            )
            if not _materialize_raw_splits_from_correct_parquet(config_path=config_path):
                raise RuntimeError("Failed to materialize raw splits from correct.parquet")

        pipeline = KGPipeline(
            cfg,
            checkpoints_repo=PipelineCheckpointsRepository(),
            splits_repo=splits_repo,  # type: ignore[arg-type]
        )
        run_coroutine_in_new_loop(pipeline.run_build_and_preprocess(), drain_pending_tasks=True)

        logger.bind(
            component="hpo_data_loader",
            stop_reason="population_complete",
            key_parameters={},
        ).success("Splits preprocessados populados no PostgreSQL via KG pipeline.")
        return True
    except Exception as exc:
        logger.bind(
            component="hpo_data_loader",
            stop_reason="population_failed",
            key_parameters={"error": repr(exc)},
        ).error("Failed to populate preprocessed splits via KG pipeline.")
        return False


def _load_from_parquet_and_push(
    preprocessing_config: PreprocessingConfig | None,
    file_manager: FileManager,
    config_path: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None] | None:
    """Fallback: load local parquets (preprocessed or raw splits) and push to PostgreSQL."""
    from pff.infrastructure.persistence.db.repositories import KGSplitsRepository

    outputs_dir = settings.OUTPUTS_DIR / "kg" / "mappings"
    train_path = outputs_dir / "train.preprocessed.parquet"
    valid_path = outputs_dir / "valid.preprocessed.parquet"
    test_path = outputs_dir / "test.preprocessed.parquet"

    if not file_manager.exists(train_path) or not file_manager.exists(valid_path):
        train_path = settings.OUTPUTS_DIR / "kg" / "train.parquet"
        valid_path = settings.OUTPUTS_DIR / "kg" / "valid.parquet"
        test_path = settings.OUTPUTS_DIR / "kg" / "test.parquet"
        if not file_manager.exists(train_path) or not file_manager.exists(valid_path):
            if not _materialize_raw_splits_from_correct_parquet(config_path=config_path):
                return None
            train_path = settings.OUTPUTS_DIR / "kg" / "train.parquet"
            valid_path = settings.OUTPUTS_DIR / "kg" / "valid.parquet"
            test_path = settings.OUTPUTS_DIR / "kg" / "test.parquet"
        if not file_manager.exists(train_path) or not file_manager.exists(valid_path):
            return None

    train_df = _as_dataframe(file_manager.read(train_path))
    valid_df = _as_dataframe(file_manager.read(valid_path))
    test_df = (
        _as_dataframe(file_manager.read(test_path)) if file_manager.exists(test_path) else None
    )

    if preprocessing_config:
        train_df, valid_df, test_df, _ = filter_attribute_relations(  # type: ignore
            train_df, valid_df, test_df, preprocessing_config
        )
        assert train_df is not None and valid_df is not None

    try:
        repo = KGSplitsRepository()

        async def _persist() -> None:
            """Execute persist."""

            await repo.delete_preprocessed()
            await repo.save_preprocessed_splits(train_df, valid_df, test_df)

        run_coroutine_sync(_persist(), timeout_s=90.0)
        logger.success("Parquets preprocessados materializados no PostgreSQL (modo_alternativo)")
    except Exception as exc:
        logger.warning(f"Failed to persist fallback parquets to Postgres: {exc}")

    _mirror_preprocessed_to_lance(train_df, valid_df, test_df)

    return train_df, valid_df, test_df


def _materialize_raw_splits_from_correct_parquet(
    config_path: Path | None = None,
) -> bool:
    """Build raw train/valid/test splits from correct.parquet and persist to PostgreSQL."""
    try:
        from pff.domain.kg.builder import KGBuilder
        from pff.domain.kg.config import KGConfig
        from pff.infrastructure.persistence.db.repositories import KGSplitsRepository

        cfg = KGConfig(config_path or KG_PIPELINE_CONFIG_PATH)
        builder_config = cfg.get_builder_config()

        builder = KGBuilder(
            source_path=builder_config.get("source_path"),
            output_dir=settings.OUTPUTS_DIR / "kg",
            max_members=builder_config.get("max_members"),
            parallel=builder_config.get("parallel", True),
            disk_cache=builder_config.get("disk_cache", False),
            splits_repo=KGSplitsRepository(),  # type: ignore[arg-type]
            seed=42,
        )
        run_coroutine_in_new_loop(builder.run(), drain_pending_tasks=True)
        logger.bind(
            component="hpo_data_loader",
            stop_reason="raw_built",
            key_parameters={},
        ).success("Splits raw gerados do correct.parquet.")
        return True
    except Exception as exc:
        logger.warning(f"Failed to materialize raw splits from correct.parquet: {exc}")
        return False


def _mirror_preprocessed_to_lance(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    test_df: pl.DataFrame | None,
) -> None:
    """Best-effort mirror of preprocessed splits to LanceDB."""
    try:
        from pff.infrastructure.persistence.db.repositories.kg_splits import KGSplitsRepositoryLance

        repo = KGSplitsRepositoryLance()

        async def _persist() -> None:
            await repo.delete_preprocessed()
            await repo.save_preprocessed_splits(train_df, valid_df, test_df)

        run_coroutine_sync(_persist(), timeout_s=90.0)
        logger.success("Splits preprocessados espelhados no LanceDB")
    except ImportError:
        logger.warning("LanceDB unavailable; skipping preprocessed mirror")
    except Exception as exc:
        logger.warning(f"Failed to mirror preprocessed splits to LanceDB: {exc}")
