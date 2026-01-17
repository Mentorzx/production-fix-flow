from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pff import settings
from pff.shared.core.config import KG_PIPELINE_CONFIG_PATH, OPTIMIZATION_CONFIG_PATH
from pff.shared import logger
from pff.shared.acceleration.asyncio_runner import (
    run_coroutine_in_new_loop,
    run_coroutine_sync,
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


def _as_lazy_frame(bundle: Any) -> pl.LazyFrame:
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
    if isinstance(bundle, ParquetBundle):
        if bundle.parsed_kind != "tabular":
            raise ValueError("Expected tabular data for DataFrame loading")
        return bundle.lazyframe().collect(engine="streaming")
    if isinstance(bundle, pl.LazyFrame):
        return bundle.collect(engine="streaming")
    if isinstance(bundle, pl.DataFrame):
        return bundle
    raise ValueError(f"Expected tabular data, got {type(bundle)}")


def compute_entity_quality_scores(
    train_df: pl.DataFrame, valid_df: pl.DataFrame
) -> dict[str, Any]:
    """Compute simple entity quality scores based on degree frequency (lightweight, deterministic)."""
    combined = pl.concat(
        [
            train_df[["s", "o"]].rename({"s": "e1", "o": "e2"}),
            valid_df[["s", "o"]].rename({"s": "e1", "o": "e2"}),
        ]
    )
    entities = pl.concat([combined["e1"], combined["e2"]])
    degree_counts = entities.value_counts().rename({"e1": "entity", "count": "degree"})
    max_degree = max(1, int(degree_counts["degree"].max()))
    degree_counts = degree_counts.with_columns(
        (pl.col("degree") / max_degree).alias("degree_norm")
    )
    return {
        "degree": degree_counts,
        "max_degree": max_degree,
        "n_entities_with_degree": int(len(degree_counts)),
    }


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
    from pff.shared.hash import hash_bytes, stable_hash

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

    all_entities = pl.concat([combined["s"], combined["o"]]).unique()
    all_relations = combined["p"].unique()

    stats = {
        "hash": data_hash,
        "hash_short": data_hash[:16],
        "source": source,
        "train_triples": len(train_df),
        "valid_triples": len(valid_df),
        "total_triples": len(combined),
        "entities": len(all_entities),
        "relations": len(all_relations),
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
    fm = file_manager or FileManager()
    kg_payload = fm.read(KG_PIPELINE_CONFIG_PATH)
    kg_config = (
        kg_payload.to_native() if isinstance(kg_payload, ParquetBundle) else kg_payload
    ) or {}
    paths_cfg = kg_config.get("paths", {}) if isinstance(kg_config, dict) else {}
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

    logger.info(
        f"Dados KG carregados (lazy): train={train_path.name}, valid={valid_path.name}"
    )

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

    train_df: pl.DataFrame = _as_dataframe(fm.read(train_path))
    valid_df: pl.DataFrame = _as_dataframe(fm.read(valid_path))

    n_entities = int(
        pl.concat([train_df["s"], train_df["o"], valid_df["s"], valid_df["o"]])
        .unique()
        .len()
    )
    n_predicates = int(pl.concat([train_df["p"], valid_df["p"]]).unique().len())

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
    cfg = (
        cfg_payload.to_native()
        if isinstance(cfg_payload, ParquetBundle)
        else cfg_payload
    ) or {}
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
        logger.info(
            "Carregamento padrão de dados (preprocessamento centralizado desativado)"
        )
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
    except Exception as save_exc:  # noqa: BLE001
        logger.warning(f"Failed to persist preprocessed splits: {save_exc}")

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
    n_predicates = int(
        pl.concat([train_preprocessed["p"], valid_preprocessed["p"]]).unique().len()
    )

    entity_quality_scores = compute_entity_quality_scores(
        train_preprocessed, valid_preprocessed
    )

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


async def _load_from_postgres_preprocessed() -> (
    tuple[pl.DataFrame | None, pl.DataFrame | None, dict]
):
    """
    Load preprocessed data directly from PostgreSQL.

    Returns:
        Tuple of (train_df, valid_df, metadata) or (None, None, {}) if not available
    """
    try:
        from pff.infrastructure.persistence.db.repositories import KGSplitsRepository

        repo = KGSplitsRepository()
        train_df, valid_df, test_df, metadata = await repo.load_preprocessed_splits(
            fallback_to_raw=False,
            map_to_ints=False,
        )

        train_valid = train_df is not None and valid_df is not None
        train_has_data = train_valid and len(train_df) > 0
        valid_has_data = train_valid and len(valid_df) > 0

        if train_has_data and valid_has_data:
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
        logger.warning(f"Failed to save preprocessed to PostgreSQL: {e}")
        return False


def load_preprocessed_from_postgres(
    file_manager: FileManager | None = None,
    require_preprocessed: bool = True,
    auto_populate_if_missing: bool = True,
    config_path: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """
    Load preprocessed KG data from PostgreSQL (single source of truth).

    This is the PREFERRED method for HPO and pff learn to ensure consistency.
    Falls back to file-based loading + preprocessing if PostgreSQL unavailable.

    Flow:
    1. Try loading preprocessed from PostgreSQL
    2. If not available, load raw + preprocess + save to PostgreSQL
    3. Return preprocessed data

    Args:
        file_manager: Optional FileManager instance

    Returns:
        Tuple of (train_df, valid_df, data_info dict)
    """
    preprocessing_config = (
        PreprocessingConfig.from_yaml() if HAS_PREPROCESSING_MODULE else None
    )
    attr_stats: dict[str, Any] | None = None
    baseline_counts: dict[str, float] | None = None
    fm = file_manager or FileManager()

    try:
        train_path, valid_path = _get_kg_paths(fm, prefer_outputs=True)
        train_base = _as_dataframe(fm.read(train_path))
        valid_base = _as_dataframe(fm.read(valid_path))
        baseline_counts = {
            "train_len": float(len(train_base)),
            "valid_len": float(len(valid_base)),
            "relations": float(
                len(pl.concat([train_base["p"], valid_base["p"]]).unique())
            ),
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Failed to compute local baseline: {exc}")
        baseline_counts = None

    try:
        train_df, valid_df, metadata = run_coroutine_sync(
            _load_from_postgres_preprocessed(),
            timeout_s=30.0,
        )

        if train_df is not None and valid_df is not None:
            if HAS_PREPROCESSING_MODULE:
                pipeline = KGPreprocessingPipeline()
                train_df, valid_df, _ = pipeline._map_ids_for_splits(
                    train_df, valid_df, None
                )

            if preprocessing_config:
                train_df, valid_df, _, attr_stats = filter_attribute_relations(
                    train_df, valid_df, None, preprocessing_config
                )
            if baseline_counts:
                rels = len(pl.concat([train_df["p"], valid_df["p"]]).unique())
                too_small = (
                    len(train_df) < baseline_counts["train_len"] * 0.5
                    or len(valid_df) < baseline_counts["valid_len"] * 0.5
                    or rels < baseline_counts["relations"] * 0.5
                )
                if too_small:
                    logger.warning(
                        "Preprocessed PostgreSQL splits look stale (size/relations far below local baseline). "
                        "Re-running preprocessing pipeline to repopulate Postgres."
                    )
                    populated = _populate_preprocessed_splits(config_path=config_path)
                    if populated:
                        try:
                            train_df, valid_df, metadata = run_coroutine_sync(
                                _load_from_postgres_preprocessed(),
                                timeout_s=30.0,
                            )
                            if preprocessing_config:
                                train_df, valid_df, _, attr_stats = (
                                    filter_attribute_relations(
                                        train_df, valid_df, None, preprocessing_config
                                    )
                                )
                        except Exception as retry_exc:  # noqa: BLE001
                            logger.warning(
                                f"Reload after repopulation failed: {retry_exc}"
                            )
            logger.success(
                "Dados preprocessados carregados do PostgreSQL (fonte única)"
            )

            n_entities = int(
                pl.concat([train_df["s"], train_df["o"], valid_df["s"], valid_df["o"]])
                .unique()
                .len()
            )
            n_predicates = int(pl.concat([train_df["p"], valid_df["p"]]).unique().len())

            entity_quality_scores = compute_entity_quality_scores(train_df, valid_df)

            data_info = {
                "n_train": len(train_df),
                "n_valid": len(valid_df),
                "n_entities": n_entities,
                "n_predicates": n_predicates,
                "source": "postgresql_preprocessed",
                "entity_quality_scores": entity_quality_scores,
                "preprocessing_applied": True,
                "attribute_filter": attr_stats or {},
            }

            logger.info(
                f"PostgreSQL: treino={data_info['n_train']:,}, "
                f"valid={data_info['n_valid']:,}, entidades={n_entities:,}"
            )

            return train_df, valid_df, data_info

    except Exception as e:
        logger.debug(f"PostgreSQL load failed: {e}")

    if auto_populate_if_missing:
        logger.info(
            "Preprocessed splits ausentes no PostgreSQL. Executando build+preprocess (KG pipeline)..."
        )
        populated = _populate_preprocessed_splits(config_path=config_path)
        if populated:
            try:
                train_df, valid_df, metadata = run_coroutine_sync(
                    _load_from_postgres_preprocessed(),
                    timeout_s=30.0,
                )
                if train_df is not None and valid_df is not None:
                    if preprocessing_config:
                        train_df, valid_df, _, attr_stats = filter_attribute_relations(
                            train_df, valid_df, None, preprocessing_config
                        )
                    if baseline_counts:
                        rels = len(pl.concat([train_df["p"], valid_df["p"]]).unique())
                        too_small = (
                            len(train_df) < baseline_counts["train_len"] * 0.5
                            or len(valid_df) < baseline_counts["valid_len"] * 0.5
                            or rels < baseline_counts["relations"] * 0.5
                        )
                        if too_small:
                            logger.warning(
                                "Repopulated splits are still below the local baseline. "
                                "Reloading from parquet and repopulating Postgres."
                            )
                            parquet_loaded = _load_from_parquet_and_push(
                                preprocessing_config, fm
                            )
                            if parquet_loaded:
                                train_df, valid_df, _ = parquet_loaded
                            else:
                                logger.warning(
                                    "Parquet fallback failed; keeping current splits."
                                )
                    entity_quality_scores = compute_entity_quality_scores(
                        train_df, valid_df
                    )
                    n_entities = int(
                        pl.concat(
                            [train_df["s"], train_df["o"], valid_df["s"], valid_df["o"]]
                        )
                        .unique()
                        .len()
                    )
                    n_predicates = int(
                        pl.concat([train_df["p"], valid_df["p"]]).unique().len()
                    )
                    data_info = {
                        "n_train": len(train_df),
                        "n_valid": len(valid_df),
                        "n_entities": n_entities,
                        "n_predicates": n_predicates,
                        "source": "postgresql_preprocessed",
                        "entity_quality_scores": entity_quality_scores,
                        "preprocessing_applied": True,
                        "populated_by": "kg_pipeline",
                        "attribute_filter": attr_stats or {},
                    }
                    logger.success(
                        f"PostgreSQL: treino={data_info['n_train']:,}, valid={data_info['n_valid']:,}, entidades={n_entities:,}"
                    )
                    return train_df, valid_df, data_info
            except Exception as retry_exc:  # noqa: BLE001
                logger.warning(f"Retry load after populate failed: {retry_exc}")

    if auto_populate_if_missing:
        parquet_loaded = _load_from_parquet_and_push(preprocessing_config, file_manager)
        if parquet_loaded:
            train_df, valid_df, _ = parquet_loaded
            entity_quality_scores = compute_entity_quality_scores(train_df, valid_df)
            n_entities = int(
                pl.concat([train_df["s"], train_df["o"], valid_df["s"], valid_df["o"]])
                .unique()
                .len()
            )
            n_predicates = int(pl.concat([train_df["p"], valid_df["p"]]).unique().len())
            data_info = {
                "n_train": len(train_df),
                "n_valid": len(valid_df),
                "n_entities": n_entities,
                "n_predicates": n_predicates,
                "source": "parquet_fallback",
                "entity_quality_scores": entity_quality_scores,
                "preprocessing_applied": True,
                "attribute_filter": attr_stats or {},
            }
            logger.debug(
                "Using local preprocessed parquets as fallback (Postgres reload failed)"
            )
            return train_df, valid_df, data_info

    if require_preprocessed:
        raise RuntimeError(
            "Preprocessed KG splits not available in PostgreSQL. "
            "Populate via KGSplitsRepository.save_preprocessed_splits before running HPO."
        )

    logger.info("Modo alternativo desativado por configuracao; retornando erro.")
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
        pipeline = KGPipeline(
            cfg,
            checkpoints_repo=PipelineCheckpointsRepository(),
            splits_repo=KGSplitsRepository(),
        )
        run_coroutine_in_new_loop(
            pipeline.run_build_and_preprocess(), drain_pending_tasks=True
        )

        logger.success("Splits preprocessados populados no PostgreSQL via KG pipeline")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to populate preprocessed splits via KG pipeline: {exc}")
        return False


def _load_from_parquet_and_push(
    preprocessing_config: PreprocessingConfig | None,
    file_manager: FileManager,
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
            return None

    train_df = _as_dataframe(file_manager.read(train_path))
    valid_df = _as_dataframe(file_manager.read(valid_path))
    test_df = (
        _as_dataframe(file_manager.read(test_path))
        if file_manager.exists(test_path)
        else None
    )

    if preprocessing_config:
        train_df, valid_df, test_df, _ = filter_attribute_relations(
            train_df, valid_df, test_df, preprocessing_config
        )

    try:
        repo = KGSplitsRepository()

        async def _persist() -> None:
            await repo.delete_preprocessed()
            await repo.save_preprocessed_splits(train_df, valid_df, test_df)

        run_coroutine_sync(_persist(), timeout_s=90.0)
        logger.success(
            "Parquets preprocessados materializados no PostgreSQL (modo_alternativo)"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to persist fallback parquets to Postgres: {exc}")

    return train_df, valid_df, test_df
