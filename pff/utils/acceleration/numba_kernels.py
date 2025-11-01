"""
Numba-compiled kernels for performance-critical operations.

Sprint 17: Hot loop optimization using Numba JIT compilation.

This module contains Numba-compiled functions that accelerate the rule validation
process by compiling Python to native machine code.

Performance Impact:
    - Target: 50-70% speedup on rule validation (2min34s → 46-70s)
    - Compiles hot loops processing 128K rules × 1.1K triples (144M operations)
    - Expected 10-100x speedup on unification operations

Author: PFF Team
Date: 2025-10-23
"""

from typing import Any
import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit, types
    from numba.typed import Dict, List
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator


# --- Robust input normalizers to avoid ambiguous truth-value of numpy arrays ---
def _to_py_scalar(val):
    """
    Convert numpy scalars/1-element arrays/lists to Python scalars (str/int).
    For multi-element arrays/lists returns a python tuple of scalars.
    Always returns a Python object (never np.ndarray).
    """
    if val is None:
        return None

    # numpy ndarray
    if isinstance(val, np.ndarray):
        # 0-dim scalar (np.str_, np.int32, etc.)
        if val.shape == ():
            return val.item()
        # single-element array
        if val.size == 1:
            return val.flat[0]
        # multi-element: convert to tuple of python scalars
        return tuple(_to_py_scalar(x) for x in val.tolist())

    # numpy scalar types (np.str_, np.int32, etc.)
    try:
        if hasattr(val, "item") and callable(getattr(val, "item")):
            return val.item()
    except Exception:
        pass

    # python list/tuple -> map recursively
    if isinstance(val, (list, tuple)):
        if len(val) == 1:
            return _to_py_scalar(val[0])
        return tuple(_to_py_scalar(x) for x in val)

    # fallback: keep as-is (likely already a python scalar)
    return val


def _to_py_str(val) -> str:
    """
    Convert val to Python string in a safe way (handles numpy types).
    Returns '' if val is None.
    """
    if val is None:
        return ""
    v = _to_py_scalar(val)
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


class VocabularyEncoder:
    """
    Converts strings (entities, relations, predicates) to integer indices for Numba.

    Sprint 17: Required for Numba because it doesn't handle Python strings well.
    Provides O(1) string→int and int→string mappings.
    """

    def __init__(self):
        # Entity vocabulary
        self.entity_to_idx: dict[str, int] = {}
        self.idx_to_entity: dict[int, str] = {}
        self.next_entity_idx = 0

        # Relation/predicate vocabulary
        self.relation_to_idx: dict[str, int] = {}
        self.idx_to_relation: dict[int, str] = {}
        self.next_relation_idx = 0

        # Special tokens
        self.WILDCARD_IDX = -1  # For "*" predicates
        self.VARIABLE_START = 1_000_000  # Variables use indices >= this

    def encode_entity(self, entity: Any) -> int:
        """Encode entity to integer index. O(1) average case."""
        entity_str = _to_py_str(entity)
        if entity_str not in self.entity_to_idx:
            idx = self.next_entity_idx
            self.entity_to_idx[entity_str] = idx
            self.idx_to_entity[idx] = entity_str
            self.next_entity_idx += 1
            return idx
        return self.entity_to_idx[entity_str]

    def encode_relation(self, relation: Any) -> int:
        """Encode relation/predicate to integer index. O(1) average case."""
        relation_str = _to_py_str(relation)
        if relation_str == "*":
            return self.WILDCARD_IDX
        if relation_str not in self.relation_to_idx:
            idx = self.next_relation_idx
            self.relation_to_idx[relation_str] = idx
            self.idx_to_relation[idx] = relation_str
            self.next_relation_idx += 1
            return idx
        return self.relation_to_idx[relation_str]

    def decode_entity(self, idx: int) -> str:
        """Decode integer index to entity string. O(1) average case."""
        return self.idx_to_entity.get(idx, f"<unknown_entity_{idx}>")

    def decode_relation(self, idx: int) -> str:
        """Decode integer index to relation string. O(1) average case."""
        if idx == self.WILDCARD_IDX:
            return "*"
        return self.idx_to_relation.get(idx, f"<unknown_relation_{idx}>")

    def encode_triples(self, triples: list[tuple[Any, str, Any]]) -> NDArray[np.int32]:
        """
        Convert list/array of (subject, predicate, object) triples to NumPy array of indices.

        Accepts numpy arrays and lists. Returns ndarray shape (n_triples,3) dtype int32.
        """
        # If numpy passed, convert rows to python tuples safely
        if isinstance(triples, np.ndarray):
            try:
                rows = [tuple(row.tolist()) for row in triples]
            except Exception:
                # fallback: stringify elements
                rows = [tuple(map(_to_py_str, row)) for row in triples]
        else:
            rows = list(triples)

        n_triples = len(rows)
        encoded = np.zeros((n_triples, 3), dtype=np.int32)

        for i, triple in enumerate(rows):
            try:
                s, p, o = triple
            except Exception:
                s = triple
                p = ""
                o = ""
            s = _to_py_str(s)
            p = _to_py_str(p)
            o = _to_py_str(o)
            encoded[i, 0] = self.encode_entity(s)
            encoded[i, 1] = self.encode_relation(p)
            encoded[i, 2] = self.encode_entity(o)

        return encoded

    def encode_pattern(self, pattern: dict[str, Any]) -> tuple[int, int, int, int, int]:
        """
        Encode a pattern dictionary to (pred_idx, arg0_idx, arg0_is_var, arg1_idx, arg1_is_var).

        Normalizes args so numpy inputs don't break .isupper() checks.
        """
        pred = pattern.get("predicate", "")
        pred_idx = self.encode_relation(pred)

        args = pattern.get("args", [])
        if len(args) < 2:
            # Invalid pattern, return dummy values
            return (pred_idx, 0, 0, 0, 0)

        # Normalize args to python scalars/strings
        arg0_raw = _to_py_scalar(args[0])
        arg1_raw = _to_py_scalar(args[1])

        arg0_is_var = 0
        arg1_is_var = 0

        if isinstance(arg0_raw, str) and arg0_raw.isupper():
            arg0_idx = self.VARIABLE_START + (hash(arg0_raw) % 1000)
            arg0_is_var = 1
        else:
            arg0_idx = self.encode_entity(arg0_raw)

        if isinstance(arg1_raw, str) and arg1_raw.isupper():
            arg1_idx = self.VARIABLE_START + (hash(arg1_raw) % 1000)
            arg1_is_var = 1
        else:
            arg1_idx = self.encode_entity(arg1_raw)

        return (pred_idx, arg0_idx, arg0_is_var, arg1_idx, arg1_is_var)


@njit(cache=True)
def _unify_pattern_triple_numba(
    pattern_pred: int,
    pattern_arg0: int,
    pattern_arg0_is_var: int,
    pattern_arg1: int,
    pattern_arg1_is_var: int,
    triple_s: int,
    triple_p: int,
    triple_o: int,
    wildcard_idx: int,
) -> int:
    """
    Numba-compiled unification of a single pattern against a single triple.
    """
    # Check predicate match (or wildcard)
    if pattern_pred != wildcard_idx and pattern_pred != triple_p:
        return 0

    # Check arg0 (subject) match
    if pattern_arg0_is_var == 0:  # Constant
        if pattern_arg0 != triple_s:
            return 0

    # Check arg1 (object) match
    if pattern_arg1_is_var == 0:  # Constant
        if pattern_arg1 != triple_o:
            return 0

    return 1


@njit(cache=True, parallel=True)
def unify_batch_numba(
    patterns: NDArray[np.int32],  # Shape: (n_patterns, 5) - pred, arg0, is_var0, arg1, is_var1
    triples: NDArray[np.int32],   # Shape: (n_triples, 3) - s, p, o
    wildcard_idx: int,
) -> NDArray[np.int8]:
    """
    Vectorized unification of multiple patterns against multiple triples.
    """
    n_patterns = patterns.shape[0]
    n_triples = triples.shape[0]

    matches = np.zeros((n_patterns, n_triples), dtype=np.int8)

    for i in range(n_patterns):
        pattern_pred = patterns[i, 0]
        pattern_arg0 = patterns[i, 1]
        pattern_arg0_is_var = patterns[i, 2]
        pattern_arg1 = patterns[i, 3]
        pattern_arg1_is_var = patterns[i, 4]

        for j in range(n_triples):
            triple_s = triples[j, 0]
            triple_p = triples[j, 1]
            triple_o = triples[j, 2]

            matches[i, j] = _unify_pattern_triple_numba(
                pattern_pred,
                pattern_arg0,
                pattern_arg0_is_var,
                pattern_arg1,
                pattern_arg1_is_var,
                triple_s,
                triple_p,
                triple_o,
                wildcard_idx,
            )

    return matches


def find_matching_triples_accelerated(
    pattern: dict[str, Any],
    triples: list[tuple[Any, str, Any]],
    encoder: VocabularyEncoder,
) -> list[int]:
    """
    Find indices of triples that match the given pattern using Numba acceleration.
    """
    if not NUMBA_AVAILABLE:
        # Fallback to Python implementation
        return _find_matching_triples_python(pattern, triples)

    # Encode pattern (encoder normalizes inputs)
    pattern_encoded = encoder.encode_pattern(pattern)
    pattern_array = np.array([pattern_encoded], dtype=np.int32)

    # Normalize triples into python rows, tolerant to numpy inputs
    if isinstance(triples, np.ndarray):
        try:
            triple_rows = [tuple(row.tolist()) for row in triples]
        except Exception:
            triple_rows = [tuple(map(_to_py_str, row)) for row in triples]
    else:
        try:
            triple_rows = [tuple(map(_to_py_scalar, t)) for t in triples]
        except Exception:
            triple_rows = [tuple(map(_to_py_str, t)) for t in triples]

    # Encode triples (encoder will further normalize)
    triples_encoded = encoder.encode_triples(triple_rows)

    # Run Numba kernel
    matches = unify_batch_numba(
        pattern_array,
        triples_encoded,
        encoder.WILDCARD_IDX,
    )

    # Extract matching indices
    matching_indices = np.where(matches[0] == 1)[0].tolist()
    return matching_indices


def _find_matching_triples_python(
    pattern: dict[str, Any],
    triples: list[tuple[Any, str, Any]],
) -> list[int]:
    """
    Python fallback for finding matching triples (used if Numba unavailable).
    """
    matching_indices = []
    predicate = pattern["predicate"]
    args = pattern.get("args", [])

    if len(args) < 2:
        return matching_indices

    arg0, arg1 = args[0], args[1]
    arg0_is_var = isinstance(arg0, str) and arg0.isupper()
    arg1_is_var = isinstance(arg1, str) and arg1.isupper()

    for i, (s, p, o) in enumerate(triples):
        # Normalize to python strings for safe comparisons
        s = _to_py_str(s)
        p = _to_py_str(p)
        o = _to_py_str(o)

        # Check predicate
        if predicate != "*" and predicate != p:
            continue

        # Check arg0 (subject)
        if not arg0_is_var and arg0 != s:
            continue

        # Check arg1 (object)
        if not arg1_is_var and arg1 != o:
            continue

        matching_indices.append(i)

    return matching_indices


# Export public API
__all__ = [
    "VocabularyEncoder",
    "unify_batch_numba",
    "find_matching_triples_accelerated",
    "NUMBA_AVAILABLE",
]
