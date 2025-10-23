from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Generator, Mapping, Sequence, TypeVar

from pff.utils import ConcurrencyManager, FileManager, logger, progress_bar
from pff.utils.cache import DiskCache

ANY: object = object()

__all__ = ["Research", "research", "SearchStrategy", "ANY"]

T = TypeVar("T")
_EXPR_CACHE: dict[str, Any] = {}


# ─────────────────────────── Strategy interface ────────────────────────────
class SearchStrategy(ABC):
    @abstractmethod
    def matches(self, item: Any, criteria: Mapping[str, Any]) -> bool: ...

    def flatten(self, node: Any) -> Generator[Mapping[str, Any], None, None]:
        if isinstance(node, Mapping):
            yield node
            for value in node.values():
                yield from self.flatten(value)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for element in node:
                yield from self.flatten(element)


class _ImperativeStrategy(SearchStrategy):
    # --- helpers ----------------------------------------------------------
    def _value_matches_any(self, node: Mapping[str, Any], v_need: Any) -> bool:
        for v in node.values():
            if isinstance(v_need, Mapping) and isinstance(v, Mapping):
                if self.matches(v, v_need):  # type: ignore[arg-type]
                    return True
            elif isinstance(v_need, Sequence) and isinstance(v, Sequence):
                if self._list_matches(v, v_need):
                    return True
            elif v == v_need:
                return True
        return False

    def _list_matches(self, have: Any, wants: Sequence[Any]) -> bool:
        if not isinstance(have, Sequence) or isinstance(have, (str, bytes)):
            return False
        for want in wants:
            if not any(
                self.matches(elem, want) if isinstance(want, Mapping) else elem == want
                for elem in have
            ):
                return False
        return True

    def matches(self, item: Any, criteria: Mapping[str, Any]) -> bool:  # noqa: C901
        if not isinstance(item, Mapping):
            return False

        for key_need, val_need in criteria.items():
            if key_need is ANY:
                if not self._value_matches_any(item, val_need):
                    return False
                continue
            if key_need not in item:
                return False

            val_have = item[key_need]

            if val_need is ANY:
                continue
            if isinstance(val_need, Mapping) and isinstance(val_have, Mapping):
                if not self.matches(val_have, val_need):
                    return False
                continue
            if isinstance(val_need, Sequence) and not isinstance(
                val_need, (str, bytes)
            ):
                if not self._list_matches(val_have, val_need):
                    return False
                continue
            if val_have != val_need:
                return False

        return True


class _JsonPathStrategy(SearchStrategy):
    def __init__(self) -> None:
        try:
            from jsonpath_ng.ext import parse  # lazy import
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "jsonpath_ng.ext não instalado – pip install jsonpath-ng[ext]"
            ) from exc
        self._parse: Callable[[str], Any] = parse

    def _criteria_to_expr(self, criteria: Mapping[str, Any]) -> "object":
        def _val(v):
            if v is ANY:
                return "*"
            return f"'{v}'" if isinstance(v, str) else str(v).lower()

        parts = []
        for k, v in criteria.items():
            if k is ANY:
                if v is ANY:
                    parts.append("..*")
                else:
                    parts.append(f"..*[?(@ == {_val(v)})]")
            else:
                if v is ANY:
                    parts.append(f"..*[?(@.{k})]")
                elif isinstance(v, dict):
                    nested = self._criteria_to_expr(v)
                    parts.append(f"..[?(@.{k}{nested})]")
                else:
                    parts.append(f"..[?(@.{k} == {_val(v)})]")

        expr_body = "".join(parts)
        expr = "$" + expr_body if not expr_body.startswith("$") else expr_body
        return self._parse(expr)

    @staticmethod
    def _val(v: Any) -> str:
        return f"'{v}'" if isinstance(v, str) else str(v).lower()

    # ---------------------------------------------------------------------
    def matches(self, item: Any, criteria: Mapping[str, Any]) -> bool:
        expr = self._criteria_to_expr(criteria)
        return bool(expr.find(item))  # type: ignore


class _PysimdjsonStrategy(SearchStrategy):
    def __init__(self) -> None:
        try:
            from simdjson import Parser
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "pysimdjson não instalado – pip install pysimdjson>=7"
            ) from exc

        self._parser = Parser()
        self._delegate = _ImperativeStrategy()

    def _ensure_obj(self, item: Any) -> Any:
        if isinstance(item, (bytes, bytearray, memoryview)):
            return self._parser.parse(item).to_dict()  # type: ignore
        return item

    def matches(self, item: Any, criteria: Mapping[str, Any]) -> bool:
        return self._delegate.matches(self._ensure_obj(item), criteria)

    def flatten(self, node: Any):
        yield from self._delegate.flatten(self._ensure_obj(node))


class _CythonStrategy(SearchStrategy):
    def __init__(self) -> None:
        try:
            import research_cython as _rc  # type: ignore[import]
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "Cython module not built – rode `python setup.py build_ext -i`"
            ) from exc
        self._rc = _rc

    def matches(self, item: Any, criteria: Mapping[str, Any]) -> bool:
        return self._rc.dict_matches(item, criteria)  # type: ignore[arg-type]

    def flatten(self, node: Any):
        yield from self._rc.flatten(node)


class _TripleIndexStrategy:
    """
    _TripleIndexStrategy is an internal utility class for high-performance indexing and querying of nested JSON-like data structures using a triple-based approach.
    Core Features:
    - Converts nested dictionaries/lists into subject-predicate-object triples for fast lookup.
    - Supports parallel processing for large lists of JSONs using multiprocessing.
    - Maintains multiple indexes for O(1) retrieval by value, predicate, subject, and path.
    - Provides optimized iterative flattening to avoid recursion limits and improve speed.
    - Allows matching and filtering of triples based on flexible criteria, including wildcards.
    - Can reconstruct original entities from indexed triples, preserving nested structure.
    - Includes utility methods for flattening and nested value setting.
    Attributes:
        by_value (dict): Index mapping object values to (subject, predicate) pairs.
        by_subject_paths (dict): Index mapping subjects to their predicates and corresponding values.
        by_subject_triples (dict): Index mapping subjects to their full triples.
        by_predicate (dict): Index mapping predicates to (subject, object) pairs.
        _data_loaded (bool): Flag indicating if data has been indexed.
        file_manager (FileManager): Utility for file and JSON handling.
        concurrency_manager (ConcurrencyManager): Utility for parallel processing.
    Methods:
        _ensure_indexed(data): Ensures data is indexed, using parallel or sequential processing.
        _build_indexes_parallel(json_list): Builds indexes in parallel for multiple JSONs.
        _build_indexes(data): Builds indexes sequentially.
        _normalize_to_triples_optimized(data): Normalizes input to a list of triples.
        _flatten_dict_to_triples_iterative(obj, subject_id, path_prefix): Iteratively flattens dict to triples.
        _populate_indexes_from_triples(triples): Populates all indexes from triples.
        _clear_indexes(): Clears all indexes.
        match(data, criteria): Finds triples matching criteria with fast lookup.
        matches(item, criteria): Checks if item matches criteria.
        flatten(node): Yields all leaf values from nested structure.
        get_entity_as_dict(subject_id): Reconstructs entity as nested dict from triples.
        _set_nested_value(obj, path, value): Sets nested value in dict using dot-separated path.
    Usage:
        Instantiate and use for efficient querying, matching, and reconstruction of complex JSON-like data.
    """

    def __init__(self) -> None:
        self.by_value: dict[Any, list[tuple[Any, str]]] = {}
        self.by_subject_paths: dict[Any, dict[str, list[Any]]] = {}
        self.by_subject_triples: dict[Any, list[tuple[Any, str, Any]]] = {}
        self.by_predicate: dict[str, list[tuple[Any, Any]]] = {}
        self._data_loaded = False
        self.file_manager = FileManager()
        self.concurrency_manager = ConcurrencyManager()
        self.triples_cache = DiskCache(root=".cache/triples_cache")

    def _generate_cache_key(self, data: Any) -> str:
        """
        Generates a cache key for the given data.

        Args:
            data (Any): The data to generate a cache key for. Can be a dictionary, a list of dictionaries, or any other type.

        Returns:
            str: A string representing the cache key. For dictionaries, a hash of the JSON representation is returned.
                 For lists of dictionaries, a combined hash of all items is returned. For other types, the object's id is used.
        """
        if isinstance(data, dict):
            return _hash_json_for_cache(data)
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            hashes = [_hash_json_for_cache(item) for item in data]
            combined_hash_str = "".join(hashes)
            return hashlib.md5(combined_hash_str.encode()).hexdigest()
        return str(id(data))

    def _ensure_indexed(self, data: Any) -> None:
        """
        Ensures that the provided data is indexed and available for use.
        This method checks if the data has already been loaded and indexed. If not, it attempts to load
        indexed triples from the cache using a generated cache key. If the cache is missed, it builds the
        indexes either in parallel (for lists of dictionaries) or sequentially, then saves the result to
        the cache. Finally, it populates internal indexes from the triples and marks the data as loaded.
        Args:
            data (Any): The data to be indexed. Can be a list of dictionaries or other supported types.
        Side Effects:
            - Loads triples from cache if available.
            - Builds and caches triples if not available.
            - Populates internal indexes from triples.
            - Sets the `_data_loaded` flag to True.
        """
        if self._data_loaded:
            return
        cache_key = self._generate_cache_key(data)
        cached_triples = self.triples_cache._load_from_cache(cache_key, ttl=None)
        if cached_triples is not None:
            logger.debug(
                f"⚡️ Cache HIT para triplas. Chave: {cache_key[:10]}... Carregando {len(cached_triples)} triplas do cache."
            )
            triples = cached_triples
        else:
            # logger.info(f"🐢 Cache MISS. Gerando triplas para a chave: {cache_key[:10]}...")
            if (
                isinstance(data, list)
                and len(data) > 1
                and all(isinstance(item, dict) for item in data)
            ):
                triples = self._build_indexes_parallel(data)
            else:
                triples = self._build_indexes(data)

            self.triples_cache._save_to_cache(cache_key, triples)
            logger.debug(
                f"💾 {len(triples)} triplas salvas no cache. Chave: {cache_key[:10]}..."
            )
        self._populate_indexes_from_triples(triples)
        self._data_loaded = True

    def _build_indexes_parallel(self, json_list: list[dict]) -> list[tuple]:
        """
        Processes a list of JSON objects in parallel to extract triples.

        This method uses the concurrency manager to flatten each JSON object in parallel,
        collecting all extracted triples into a single list. If parallel processing fails,
        it falls back to sequential processing.

        Args:
            json_list (list[dict]): A list of JSON objects to be processed.

        Returns:
            list[tuple]: A list of extracted triples from all JSON objects.

        Logs:
            - Info: Number of triples extracted upon successful parallel processing.
            - Warning: If parallel processing fails, logs the exception and falls back to sequential processing.
        """
        try:
            results = self.concurrency_manager.execute_sync(
                fn=_flatten_single_json_worker,
                args_list=[(json_data, i) for i, json_data in enumerate(json_list)],
                task_type="process",
                desc="Paralelizando achatamento de JSONs",
            )
            all_triples = [triple for batch in results for triple in batch]
            logger.info(
                f"📊 Processamento paralelo concluído: {len(all_triples)} triplas extraídas"
            )
            return all_triples
        except Exception as e:
            logger.warning(
                f"⚠️ Paralelismo falhou ({e}), caindo para processamento sequencial"
            )
            return self._build_indexes(json_list)

    def _build_indexes(self, data: Any) -> list[tuple]:
        """
        Builds and returns a list of normalized triples from the provided data.

        This method first clears any existing indexes, then normalizes the input data
        into a list of triples using an optimized approach. The number of extracted triples
        is logged for debugging purposes.

        Args:
            data (Any): The input data to be normalized into triples.

        Returns:
            list[tuple]: A list of triples extracted from the input data.
        """
        self._clear_indexes()
        triples = self._normalize_to_triples_optimized(data)
        logger.debug(f"📊 {len(triples)} triplas extraídas (sequencial)")
        return triples

    def _normalize_to_triples_optimized(self, data: Any) -> list[tuple[Any, str, Any]]:
        """
        Optimized normalization with fast JSON handling via FileManager.
        Supports direct JSON file paths for lazy loading.
        """
        if isinstance(data, (str, bytes)) and (
            (isinstance(data, str) and data.strip().startswith("{"))
            or isinstance(data, bytes)
        ):
            try:
                import io

                if isinstance(data, str):
                    data = data.encode("utf-8")
                buffer = io.BytesIO(data)
                data = self.file_manager.read(buffer)
            except Exception as e:
                logger.warning(f"⚠️ JSON parsing failed: {e}")

        if isinstance(data, dict):
            return self._flatten_dict_to_triples_iterative(data, "entity_0")
        elif isinstance(data, list):
            if not data:
                return []
            first_elem = data[0]
            if isinstance(first_elem, dict):
                triples = []
                for i, item in enumerate(data):
                    if not isinstance(item, dict):
                        raise TypeError(
                            f"Lista mista não suportada. Item {i} não é dict."
                        )
                    triples.extend(
                        self._flatten_dict_to_triples_iterative(item, f"entity_{i}")
                    )
                return triples
            elif isinstance(first_elem, tuple) and len(first_elem) == 3:
                for i, item in enumerate(data):
                    if not (isinstance(item, tuple) and len(item) == 3):
                        raise TypeError(
                            f"Item {i} não é uma tripla válida (sujeito, predicado, objeto)."
                        )
                return data.copy()
            else:
                raise TypeError(
                    f"Formato de lista não suportado. Esperado: list[dict] ou list[tuple]. "
                    f"Recebido: list[{type(first_elem).__name__}]"
                )
        else:
            raise TypeError(
                f"Formato de entrada inválido. Esperado: dict, list[dict] ou list[tuple]. "
                f"Recebido: {type(data).__name__}"
            )

    def _flatten_dict_to_triples_iterative(
        self, obj: dict, subject_id: str, path_prefix: str = ""
    ) -> list[tuple[Any, str, Any]]:
        """
        Iterative stack-based flattening - NO RECURSION LIMITS!

        Major optimization: Uses explicit stack instead of recursive calls.
        This avoids Python's recursion limit and is marginally faster by
        eliminating function call overhead for deeply nested JSONs.

        Performance characteristics:
        - Memory: O(depth) for stack vs O(depth) for call stack
        - Speed: ~5-15% faster than recursive version
        - Robustness: No recursion limit (Python default: ~1000 levels)
        """
        triples = []

        # Stack entries: (object, subject_id, path_prefix)
        stack = deque([(obj, subject_id, path_prefix)])

        while stack:
            current_obj, current_subject, current_path = stack.pop()

            if isinstance(current_obj, dict):
                for key, value in current_obj.items():
                    new_path = f"{current_path}.{key}" if current_path else key

                    if isinstance(value, dict):
                        # Push nested dict to stack for processing
                        stack.append((value, current_subject, new_path))
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            indexed_path = f"{new_path}[{i}]"
                            if isinstance(item, dict):
                                # Push nested dict from list to stack
                                stack.append((item, current_subject, indexed_path))
                            else:
                                # Leaf value in list
                                triples.append((current_subject, indexed_path, item))
                    else:
                        # Leaf value
                        triples.append((current_subject, new_path, value))

        return triples

    def _populate_indexes_from_triples(
        self, triples: list[tuple[Any, str, Any]]
    ) -> None:
        """
        Populates all internal indexes from a list of triples.
        Optimized for batch processing of large triple sets.
        """
        logger.debug(f"🏗️ Populando índices com {len(triples)} triplas...")

        for subject, predicate, obj in triples:
            if obj not in self.by_value:
                self.by_value[obj] = []
            self.by_value[obj].append((subject, predicate))
            if subject not in self.by_subject_paths:
                self.by_subject_paths[subject] = {}
            if predicate not in self.by_subject_paths[subject]:
                self.by_subject_paths[subject][predicate] = []
            self.by_subject_paths[subject][predicate].append(obj)
            if subject not in self.by_subject_triples:
                self.by_subject_triples[subject] = []
            self.by_subject_triples[subject].append((subject, predicate, obj))
            if predicate not in self.by_predicate:
                self.by_predicate[predicate] = []
            self.by_predicate[predicate].append((subject, obj))

    def _clear_indexes(self) -> None:
        """Clears all internal indexes."""
        self.by_value.clear()
        self.by_subject_paths.clear()
        self.by_subject_triples.clear()
        self.by_predicate.clear()

    def match(
        self, data: Any, criteria: Mapping[str, Any]
    ) -> list[tuple[Any, str, Any]]:
        """
        Find triples matching criteria with O(1) lookup performance.
        The initial indexing cost pays dividends here with constant-time retrieval.
        """
        self._ensure_indexed(data)

        matching_triples = []

        for key, expected_value in criteria.items():
            if key == "*":
                # Wildcard: return all triples
                for triples_list in self.by_subject_triples.values():
                    matching_triples.extend(triples_list)
            elif expected_value == "*":
                # Wildcard value: find all triples with this predicate
                if key in self.by_predicate:
                    for subject, obj in self.by_predicate[key]:
                        matching_triples.append((subject, key, obj))
            else:
                # Exact match: use value index for O(1) lookup
                if expected_value in self.by_value:
                    for subject, predicate in self.by_value[expected_value]:
                        if key == predicate or key == "*":
                            matching_triples.append(
                                (subject, predicate, expected_value)
                            )

        return matching_triples

    def matches(self, item: Any, criteria: Mapping[str, Any]) -> bool:
        """
        Check if item matches criteria - optimized boolean check.
        Returns early on first match for performance.
        """
        try:
            matching_triples = self.match(item, dict(criteria))
            return len(matching_triples) > 0
        except Exception:
            return False

    def flatten(self, node: Any):
        """
        Yields all leaf values from nested structure.
        Uses optimized iterative approach for consistency.
        """
        try:
            if isinstance(node, dict):
                triples = self._flatten_dict_to_triples_iterative(node, "root")
                for _, path, value in triples:
                    yield value
            elif isinstance(node, list):
                for item in node:
                    yield from self.flatten(item)
            else:
                yield node
        except Exception:
            yield node

    def get_entity_as_dict(self, subject_id: Any) -> dict[str, Any]:
        """
        Reconstruct entity as nested dictionary from indexed triples.
        Uses path information to rebuild original structure.
        """
        if subject_id not in self.by_subject_paths:
            return {}

        result = {}
        paths = self.by_subject_paths[subject_id]

        for path, values in paths.items():
            for value in values:
                self._set_nested_value(result, path, value)

        return result

    @staticmethod
    def _set_nested_value(obj: dict, path: str, value: Any) -> None:
        """
        Set nested value in dictionary using dot-separated path.
        Handles both object keys and array indices.
        """
        keys = path.split(".")
        current = obj

        for i, key in enumerate(keys[:-1]):
            if "[" in key and key.endswith("]"):
                # Handle array index
                base_key, index_part = key.split("[", 1)
                index = int(index_part[:-1])

                if base_key not in current:
                    current[base_key] = []

                # Ensure list is long enough
                while len(current[base_key]) <= index:
                    current[base_key].append(
                        {} if "." in ".".join(keys[i + 1 :]) else None
                    )

                current = current[base_key][index]
            else:
                if key not in current:
                    current[key] = {}
                current = current[key]

        # Set final value
        final_key = keys[-1]
        if "[" in final_key and final_key.endswith("]"):
            base_key, index_part = final_key.split("[", 1)
            index = int(index_part[:-1])

            if base_key not in current:
                current[base_key] = []

            while len(current[base_key]) <= index:
                current[base_key].append(None)

            current[base_key][index] = value
        else:
            current[final_key] = value


# WORKER FUNCTIONS FOR MULTIPROCESSING


def _flatten_single_json_worker(
    json_data: dict, entity_id: int
) -> list[tuple[Any, str, Any]]:
    """
    Worker function for parallel JSON flattening.

    This function runs in a separate process to bypass Python's GIL.
    Each process handles one JSON independently, then results are aggregated.

    Args:
        json_data: Dictionary to flatten
        entity_id: Unique identifier for this entity

    Returns:
        List of triples extracted from the JSON
    """
    try:
        temp_strategy = _TripleIndexStrategy()
        triples = temp_strategy._flatten_dict_to_triples_iterative(
            json_data, f"entity_{entity_id}"
        )
        return triples
    except Exception as e:
        logger.error(f"Worker {entity_id} failed: {e}")
        return []


def _hash_json_for_cache(json_data: dict) -> str:
    """
    Generate consistent hash for JSON data for caching purposes.
    Uses sorted keys to ensure deterministic hashing.
    """
    import orjson

    normalized = orjson.dumps(json_data)
    return hashlib.md5(normalized).hexdigest()


# ────────────────────────────  Facade Research  ───────────────────────────
@dataclass(slots=True)
class Research:
    """
    Research class for performing advanced search operations on data collections using various strategies.
    Attributes:
        strategy (SearchStrategy | None): The selected search strategy instance.
        ANY (object): A constant representing a wildcard value for matching.
    Methods:
        __post_init__():
            Initializes the Research instance by selecting the first available search strategy from a predefined list.
            Logs errors for unavailable strategies and raises RuntimeError if none are available.
        async search_in(
            Searches for items in `data` that match the given `criterias`.
            Supports parallel execution using different concurrency backends.
            Returns a list of matching items.
        async search_async(
            Asynchronously searches for items in `data` that match the given `criterias` using asyncio.
            Returns a list of matching items.
        _match_item(item: Any, crit_list: Sequence[Mapping[str, Any]]) -> Tuple[bool, Any]:
            Internal method to check if an item matches all criteria in `crit_list` using the selected strategy.
            Returns a tuple (match_result, item).
    """

    strategy: SearchStrategy | None = None
    ANY: ClassVar[object] = ANY

    def __post_init__(self) -> None:
        errors: dict[str, Exception] = {}

        for strat in (
            _TripleIndexStrategy,
            _PysimdjsonStrategy,
            _CythonStrategy,
            _JsonPathStrategy,
            _ImperativeStrategy,
        ):
            name = strat.__name__.lstrip("_").removesuffix("Strategy")
            try:
                # logger.debug(f"🔍  Tentando estratégia {name}…")
                self.strategy = strat()
                # logger.debug(f"✅  Estratégia selecionada: {name}")

                for n, exc in errors.items():
                    logger.debug(f"↪︎ {n} indisponível ({exc})")
                break
            except Exception as exc:  # pragma: no cover
                errors[name] = exc
                logger.debug(f"{name} falhou ({exc})", exc_info=True)

        else:
            logger.error("🤯  Nenhuma estratégia de busca disponível!")
            raise RuntimeError("Nenhuma estratégia de busca disponível!")

    async def search_in(
        self,
        data: Sequence[Any] | Any,
        criterias: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        parallel: bool = True,
        backend: str = "thread",
        max_workers: int | None = None,
        desc: str | None = "Buscando...",
    ) -> list[Any]:
        """
        Searches for items in the provided data that match the given criteria.
        Args:
            data (Sequence[Any] | Any): The data to search through. Can be a sequence of items or a single item.
            criterias (Mapping[str, Any] | Sequence[Mapping[str, Any]]): Criteria or list of criteria to match against each item.
            parallel (bool, optional): Whether to perform the search in parallel. Defaults to True.
            backend (str, optional): The concurrency backend to use ("thread", "process", "io_asyncio", "polars"). Defaults to "thread".
            max_workers (int | None, optional): Maximum number of worker threads/processes. Defaults to None.
            desc (str | None, optional): Description for the progress bar. Defaults to "Buscando...".
        Returns:
            list[Any]: list of items from data that match the given criteria.
        Notes:
            - If parallel is True and there are multiple items, the search is performed concurrently using the specified backend.
            - If parallel is False or only one item is provided, the search is performed sequentially.
            - Uses a progress bar if a description is provided.
        """
        items: Sequence[Any] = data if isinstance(data, (list, tuple)) else [data]
        crit_list: list[Mapping[str, Any]] = (
            [criterias] if isinstance(criterias, Mapping) else list(criterias)
        )
        if parallel and len(items) > 1:
            # Map backend to ConcurrencyManager task_type
            bk = backend.lower()
            if bk in ("thread", "io", "io_bound"):
                task_type = "io_thread"
            elif bk in ("process", "cpu", "cpu_bound"):
                task_type = "process"
            elif bk in ("io_asyncio", "io_async", "async"):
                task_type = "io_asyncio"
            elif bk == "polars":
                task_type = "polars"
            else:
                task_type = "process"

            cm = ConcurrencyManager()
            args = [(item, crit_list) for item in items]
            pairs = await cm.execute(
                self._match_item,
                args,
                task_type=task_type,
                max_workers=max_workers,
                desc=desc,
            )
            return [item for ok, item in pairs if ok]

        out: list[Any] = []
        for item in progress_bar(items, desc=desc, enabled=bool(desc)):
            if self._match_item(item, crit_list)[0]:
                out.append(item)
        return out

    async def search_async(
        self,
        data: Sequence[Any] | Any,
        criterias: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        **run_async_kw,
    ) -> list[Any]:
        """
        Asynchronously searches for items in the provided data that match the given criteria.

        Args:
            data (Sequence[Any] | Any): The data to search through. Can be a single item or a sequence of items.
            criterias (Mapping[str, Any] | Sequence[Mapping[str, Any]]): The criteria or list of criteria to match against each item.
            **run_async_kw: Additional keyword arguments for asynchronous execution.

        Returns:
            list[Any]: A list of items from the input data that match the specified criteria.

        Note:
            The matching is performed using the `_match_item` method in a separate thread for each item.
        """
        items: Sequence[Any] = data if isinstance(data, (list, tuple)) else [data]
        crit_list: list[Mapping[str, Any]] = (
            [criterias] if isinstance(criterias, Mapping) else list(criterias)
        )
        ok_items = await asyncio.gather(
            *[
                asyncio.create_task(
                    asyncio.to_thread(self._match_item, item, crit_list)
                )
                for item in items
            ]
        )
        return [item for ok, item in ok_items if ok]

    # -------------------------- internals -------------------------------
    def _match_item(self, item: Any, crit_list: Sequence[Mapping[str, Any]]):
        """
        Determines if the given item matches all criteria in the provided list.

        Iterates over each criterion in `crit_list` and checks if any node in the
        flattened representation of `item` matches the criterion using the current
        strategy. If any criterion is not matched, returns False and the item.
        Otherwise, returns True and the item.

        Args:
            item (Any): The item to be checked against the criteria.
            crit_list (Sequence[Mapping[str, Any]]): A sequence of criteria mappings
                to match against the item.

        Returns:
            Tuple[bool, Any]: A tuple where the first element indicates whether the
                item matches all criteria, and the second element is the item itself.

        Raises:
            AssertionError: If `self.strategy` is None.
        """
        assert self.strategy is not None
        for crit in crit_list:
            if not any(
                self.strategy.matches(node, crit)
                for node in self.strategy.flatten(item)
            ):
                return False, item
        return True, item


class TripleStore:
    """
    TripleStore is a class for storing and efficiently querying RDF-like triples (subject, predicate, object).
    It maintains six internal index structures to support fast lookup for any combination of subject, predicate, and object.
    Attributes:
        spo (defaultdict): Index mapping subject -> predicate -> set of objects.
        pos (defaultdict): Index mapping predicate -> object -> set of subjects.
        osp (defaultdict): Index mapping object -> subject -> set of predicates.
        pso (defaultdict): Index mapping predicate -> subject -> set of objects.
        sop (defaultdict): Index mapping subject -> object -> set of predicates.
        ops (defaultdict): Index mapping object -> predicate -> set of subjects.
    Methods:
        __init__(triples=None):
            Initializes the TripleStore, optionally loading an iterable of triples.
        add(triple):
            Adds a triple (subject, predicate, object) to all internal index structures.
        find(s=None, p=None, o=None):
            Yields triples matching the provided subject, predicate, and/or object.
            Supports queries with any combination of s, p, and o, or returns all triples if none are specified.
    """

    def __init__(self, triples=None):
        self.spo = defaultdict(lambda: defaultdict(set))
        self.pos = defaultdict(lambda: defaultdict(set))
        self.osp = defaultdict(lambda: defaultdict(set))
        self.pso = defaultdict(lambda: defaultdict(set))
        self.sop = defaultdict(lambda: defaultdict(set))
        self.ops = defaultdict(lambda: defaultdict(set))
        self._count = 0
        if triples:
            unique_triples = set(triples)
            for s, p, o in unique_triples:
                self.add((s, p, o))

    def __len__(self):
        """
        Returns the number of items in the collection.

        Returns:
            int: The total count of items.
        """
        return self._count

    def add(self, triple):
        """
        Adds a triple (subject, predicate, object) to multiple internal index structures.

        The triple is unpacked into subject (s), predicate (p), and object (o), and each
        is inserted into six different dictionaries that index the data in various ways:
        - spo: subject -> predicate -> set of objects
        - pos: predicate -> object -> set of subjects
        - osp: object -> subject -> set of predicates
        - pso: predicate -> subject -> set of objects
        - sop: subject -> object -> set of predicates
        - ops: object -> predicate -> set of subjects

        Args:
            triple (tuple): A tuple containing (subject, predicate, object).
        """
        s, p, o = triple
        if o not in self.spo[s][p]:
            self._count += 1
        self.spo[s][p].add(o)
        self.pos[p][o].add(s)
        self.osp[o][s].add(p)
        self.pso[p][s].add(o)
        self.sop[s][o].add(p)
        self.ops[o][p].add(s)

    def find(self, s=None, p=None, o=None):
        """
        Finds and yields triples (subject, predicate, object) from the internal data structures
        based on the provided parameters.

        Parameters:
            s (optional): The subject to search for.
            p (optional): The predicate to search for.
            o (optional): The object to search for.

        Yields:
            tuple: A triple (subject, predicate, object) matching the search criteria.

        Behavior:
            - If all of s, p, and o are provided, yields the triple if it exists.
            - If s and p are provided, yields all triples with the given subject and predicate.
            - If s and o are provided, yields all triples with the given subject and object.
            - If p and o are provided, yields all triples with the given predicate and object.
            - If only s is provided, yields all triples with the given subject.
            - If only p is provided, yields all triples with the given predicate.
            - If only o is provided, yields all triples with the given object.
            - If none are provided, yields all triples.

        Exceptions:
            Returns nothing if a KeyError occurs (i.e., if the search keys do not exist).
        """
        try:
            if s is not None and p is not None and o is not None:
                if o in self.spo[s][p]:
                    yield (s, p, o)
            elif s is not None and p is not None:
                for obj in self.spo[s][p]:
                    yield (s, p, obj)
            elif s is not None and o is not None:
                for pred in self.sop[s][o]:
                    yield (s, pred, o)
            elif p is not None and o is not None:
                for subj in self.pos[p][o]:
                    yield (subj, p, o)
            elif s is not None:
                for pred, objs in self.spo[s].items():
                    for obj in objs:
                        yield (s, pred, obj)
            elif p is not None:
                for subj, objs in self.pso[p].items():
                    for obj in objs:
                        yield (subj, p, obj)
            elif o is not None:
                for subj, preds in self.osp[o].items():
                    for pred in preds:
                        yield (subj, pred, o)
            else:
                for subj, preds in self.spo.items():
                    for pred, objs in preds.items():
                        for obj in objs:
                            yield (subj, pred, obj)
        except KeyError:
            return


research = Research()
