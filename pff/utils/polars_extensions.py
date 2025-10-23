from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import polars as pl

from pff.utils import logger, FileManager

# Type alias for Parquet compression (replaces deprecated polars.type_aliases)
ParquetCompression = Literal[
    "uncompressed", "snappy", "gzip", "lzo", "brotli", "lz4", "zstd"
]

"""
Polars extensions for high-performance JSON processing and DataFrame operations.

This module provides utilities for converting JSON responses to Polars DataFrames,
optimized searching, and efficient data manipulation while maintaining compatibility
with the existing PFF architecture.
"""


class ResponseToDataFrameConverter:
    """
    Converts JSON API responses to Polars DataFrames with automatic schema inference
    and nested structure handling.
    """

    @staticmethod
    def is_tabular_response(data: Any) -> bool:
        """
        Determine if a JSON response has tabular structure suitable for DataFrame conversion.

        Args:
            data: JSON response data (dict or list)

        Returns:
            bool: True if data can be efficiently represented as DataFrame
        """
        if isinstance(data, list) and data:
            return all(isinstance(item, dict) for item in data)
        if isinstance(data, dict):
            for _key, value in data.items():
                if isinstance(value, list) and value:
                    if all(isinstance(item, dict) for item in value):
                        return True
            if "contract" in data and isinstance(data["contract"], list):
                return True
            if "product" in data and isinstance(data["product"], list):
                return True

        return False

    @staticmethod
    def extract_tabular_data(data: Any) -> tuple[Any, str]:
        """
        Extract the main tabular component from a response.

        Args:
            data: JSON response data

        Returns:
            tuple: (tabular_data, data_type) where data_type helps identify the structure
        """
        if isinstance(data, list):
            return data, "list"

        if isinstance(data, dict):
            priority_keys = ["contract", "product", "customer", "party", "enquiry"]
            for key in priority_keys:
                if key in data and isinstance(data[key], list):
                    return data[key], key
            for key, value in data.items():
                if (
                    isinstance(value, list)
                    and len(value) > 0
                    and isinstance(value[0], dict)
                ):
                    return value, key

        return data, "unknown"

    @staticmethod
    def json_to_dataframe(
        json_data: str | dict | list, flatten_nested: bool = True, max_depth: int = 3
    ) -> pl.DataFrame | None:
        """
        Convert JSON data to Polars DataFrame with intelligent schema handling.

        Args:
            json_data: JSON string, dict, or list to convert
            flatten_nested: Whether to flatten nested structures
            max_depth: Maximum depth for flattening

        Returns:
            pl.DataFrame or None if conversion not suitable
        """
        try:
            if isinstance(json_data, str):
                # Sprint 16.5: Use FileManager for 2-3x faster JSON parsing (msgspec)
                data = FileManager.json_loads(json_data)
            else:
                data = json_data
            if not ResponseToDataFrameConverter.is_tabular_response(data):
                return None
            tabular_data, data_type = ResponseToDataFrameConverter.extract_tabular_data(
                data
            )
            if isinstance(tabular_data, list) and tabular_data:
                df = pl.DataFrame(tabular_data)
                df = df.with_columns(pl.lit(data_type).alias("_source_type"))
                if flatten_nested:
                    df = ResponseToDataFrameConverter._flatten_dataframe(df, max_depth)

                return df
        except Exception as e:
            logger.debug(f"Could not convert to DataFrame: {e}")
            return None

    @staticmethod
    def _flatten_dataframe(df: pl.DataFrame, max_depth: int = 3) -> pl.DataFrame:
        """
        Intelligently flatten nested structures in DataFrame.

        Args:
            df: Input DataFrame
            max_depth: Maximum nesting depth to flatten

        Returns:
            Flattened DataFrame
        """
        depth = 0

        while depth < max_depth:
            struct_cols = [col for col in df.columns if df[col].dtype == pl.Struct]

            if not struct_cols:
                break

            for col in struct_cols:
                # Unnest struct columns
                df = df.unnest(col)

            depth += 1

        return df


class PolarsResearch:
    """
    High-performance search implementation using Polars expressions.
    Maintains compatibility with existing Research interface.
    """

    def __init__(self):
        self._cache: dict[str, pl.DataFrame] = {}

    def search_dataframe(
        self, df: pl.DataFrame, criteria: dict[str, Any], limit: int | None = None
    ) -> pl.DataFrame:
        """
        Search DataFrame using criteria dict compatible with existing Research.

        Args:
            df: DataFrame to search
            criteria: Search criteria in Research format
            limit: Maximum results to return

        Returns:
            Filtered DataFrame
        """
        filters = []

        for key, value in criteria.items():
            col_name = key
            if "." in key:
                col_name = key.replace(".", "_")
            if isinstance(value, list):
                filters.append(pl.col(col_name).is_in(value))
            elif isinstance(value, dict):
                filters.extend(self._build_complex_filter(col_name, value))
            else:
                filters.append(pl.col(col_name) == value)
        if filters:
            result = df.filter(pl.all_horizontal(filters))
        else:
            result = df
        if limit is not None:
            result = result.limit(limit)

        return result

    def _build_complex_filter(self, col_name: str, criteria: dict) -> list:
        """Build complex filter expressions from nested criteria."""
        filters = []
        comparison_ops = {
            "gt": lambda col, val: col > val,
            "gte": lambda col, val: col >= val,
            "lt": lambda col, val: col < val,
            "lte": lambda col, val: col <= val,
            "ne": lambda col, val: col != val,
        }
        string_ops = {
            "contains": lambda col, val: col.str.contains(val),
            "startswith": lambda col, val: col.str.starts_with(val),
            "endswith": lambda col, val: col.str.ends_with(val),
        }
        for op, value in criteria.items():
            col = pl.col(col_name)
            if op in comparison_ops:
                filters.append(comparison_ops[op](col, value))
            elif op in string_ops:
                filters.append(string_ops[op](col, value))

        return filters


class DataFrameCache:
    """
    Specialized cache for Polars DataFrames with Parquet persistence.
    Integrates with existing CacheManager infrastructure.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or Path(".cache/dataframes")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Generate cache file path for a key."""
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.parquet"

    def save_dataframe(
        self,
        df: pl.DataFrame,
        key: str,
        compression: str = "snappy",
        statistics: bool = True,
    ) -> None:
        """
        Save DataFrame to Parquet with optimizations.

        Args:
            df: DataFrame to save
            key: Cache key
            compression: Compression algorithm (snappy, lz4, zstd, gzip)
            statistics: Whether to write column statistics
        """
        try:
            path = self._get_cache_path(key)

            if not isinstance(compression, str):
                compression = "zstd"

            df.write_parquet(
                path,
                compression=compression,  # type: ignore[arg-type]
                statistics=statistics,
                row_group_size=50_000,
            )

            logger.debug(f"Cached DataFrame to {path} ({df.shape} shape)")

        except Exception as e:
            logger.warning(f"Failed to cache DataFrame: {e}")
            logger.warning(f"Failed to cache DataFrame: {e}")

    def load_dataframe(self, key: str) -> pl.DataFrame | None:
        """
        Load DataFrame from cache.

        Args:
            key: Cache key

        Returns:
            Cached DataFrame or None if not found
        """
        try:
            path = self._get_cache_path(key)

            if not path.exists():
                return None

            # Load with lazy evaluation for memory efficiency
            df = pl.read_parquet(path)
            logger.debug(f"Loaded DataFrame from cache: {key}")
            return df

        except Exception as e:
            logger.warning(f"Failed to load cached DataFrame: {e}")
            return None

    def invalidate(self, key: str) -> None:
        """Remove DataFrame from cache."""
        try:
            path = self._get_cache_path(key)
            if path.exists():
                path.unlink()
                logger.debug(f"Invalidated cache: {key}")
        except Exception as e:
            logger.warning(f"Failed to invalidate cache: {e}")


class PolarsContextManager:
    """
    Enhanced context manager that supports both dict and DataFrame storage.
    Maintains backward compatibility while adding DataFrame capabilities.
    """

    def __init__(self):
        self._contexts: dict[str, dict[str, Any]] = {}
        self._df_cache = DataFrameCache()

    def get_context(self, msisdn: str) -> dict[str, Any]:
        """Get or create context for MSISDN."""
        if msisdn not in self._contexts:
            self._contexts[msisdn] = {}
        return self._contexts[msisdn]

    def set_value(self, msisdn: str, key: str, value: Any) -> None:
        """
        Set value in context, with special handling for DataFrames.

        Args:
            msisdn: MSISDN identifier
            key: Context key
            value: Value to store (dict, DataFrame, etc.)
        """
        ctx = self.get_context(msisdn)

        # Special handling for DataFrames
        if isinstance(value, pl.DataFrame):
            # Store DataFrame reference
            df_key = f"{msisdn}_{key}"
            self._df_cache.save_dataframe(value, df_key)

            # Also store as dict for compatibility
            ctx[key] = value.to_dicts()

            # Store DataFrame reference for efficient access
            ctx[f"{key}_df"] = value
        else:
            ctx[key] = value

    def get_value(self, msisdn: str, key: str) -> Any:
        """
        Get value from context with DataFrame support.

        Args:
            msisdn: MSISDN identifier
            key: Context key

        Returns:
            Stored value (dict, DataFrame, etc.)
        """
        ctx = self.get_context(msisdn)

        # Check for DataFrame version first
        df_key = f"{key}_df"
        if df_key in ctx:
            return ctx[df_key]

        return ctx.get(key)

    def clear_context(self, msisdn: str) -> None:
        """Clear context for MSISDN."""
        if msisdn in self._contexts:
            # Clear any cached DataFrames
            ctx = self._contexts[msisdn]
            for key in list(ctx.keys()):
                if key.endswith("_df"):
                    df_cache_key = f"{msisdn}_{key[:-3]}"
                    self._df_cache.invalidate(df_cache_key)

            del self._contexts[msisdn]


# Utility functions for common operations
def json_response_to_parquet(
    json_data: str | dict, output_path: Path, compression: str = "snappy"
) -> Path | None:
    """
    Convert JSON response directly to Parquet file.

    Args:
        json_data: JSON response data
        output_path: Output file path
        compression: Parquet compression algorithm

    Returns:
        Path to saved file or None if conversion failed
    """
    converter = ResponseToDataFrameConverter()
    df = converter.json_to_dataframe(json_data)

    if df is not None:
        # Validate compression parameter
        if not isinstance(compression, str):
            compression = "zstd"
        df.write_parquet(output_path, compression=compression)  # type: ignore[arg-type]
        return output_path

    return None


def optimize_dataframe_for_search(df: pl.DataFrame) -> pl.DataFrame:
    """
    Optimize DataFrame for search operations.

    Args:
        df: Input DataFrame

    Returns:
        Optimized DataFrame with appropriate data types and indexes
    """
    # Convert string columns that look like numbers
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            # Try to convert to numeric if possible
            try:
                # Check if all non-null values are numeric
                if df[col].drop_nulls().str.contains(r"^\d+$").all():
                    df = df.with_columns(pl.col(col).cast(pl.Int64))
            except Exception:
                pass

    # Sort by commonly searched columns for better performance
    common_keys = ["msisdn", "customer_id", "contract_id", "id"]
    sort_cols = [col for col in common_keys if col in df.columns]

    if sort_cols:
        df = df.sort(sort_cols)

    return df
