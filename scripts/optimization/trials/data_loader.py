from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from pff import settings
from pff.config import KG_PIPELINE_CONFIG_PATH
from pff.utils import logger
from pff.utils.core.file_manager import FileManager

from .helpers import compute_entity_quality_scores


def _pick_existing_path(candidates: list[Path]) -> Path:
    """Return the first existing path from candidates or raise a detailed error."""
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Training/validation data not found in candidates: {', '.join(str(p) for p in candidates)}")


def _get_kg_paths(file_manager: FileManager | None = None) -> tuple[Path, Path]:
    """
    Resolve train and validation parquet paths.
    
    Args:
        file_manager: Optional FileManager instance
        
    Returns:
        Tuple of (train_path, valid_path)
    """
    fm = file_manager or FileManager()
    kg_config = fm.read(KG_PIPELINE_CONFIG_PATH) or {}
    paths_cfg = kg_config.get("paths", {}) if isinstance(kg_config, dict) else {}
    data_dir = Path(paths_cfg.get("data_dir", settings.DATA_DIR))
    graph_subdir = paths_cfg.get("graph_subdir", "models/kg")
    graph_dir = data_dir / graph_subdir
    outputs_graph_dir = settings.OUTPUTS_DIR / "kg"

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


def load_kg_data_lazy(file_manager: FileManager | None = None) -> tuple[pl.LazyFrame, pl.LazyFrame, dict[str, Any]]:
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
    
    # Use lazy=True for deferred execution
    train_lazy: pl.LazyFrame = fm.read(train_path, lazy=True)
    valid_lazy: pl.LazyFrame = fm.read(valid_path, lazy=True)
    
    # Compute minimal stats lazily
    # Note: These will be computed when accessed
    data_info = {
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "lazy": True,
    }
    
    logger.info(
        f"Dados KG carregados (lazy): train={train_path.name}, valid={valid_path.name}"
    )
    
    return train_lazy, valid_lazy, data_info


def load_real_kg_data(file_manager: FileManager | None = None) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
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

    train_df: pl.DataFrame = fm.read(train_path)
    valid_df: pl.DataFrame = fm.read(valid_path)

    if not isinstance(train_df, pl.DataFrame) or not isinstance(valid_df, pl.DataFrame):
        raise RuntimeError(f"Expected Polars DataFrames, got train={type(train_df)}, valid={type(valid_df)}")

    n_entities = int(pl.concat([train_df["s"], train_df["o"], valid_df["s"], valid_df["o"]]).unique().len())
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

