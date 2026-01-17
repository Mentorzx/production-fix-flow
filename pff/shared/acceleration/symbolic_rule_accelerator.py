"""
Symbolic Rule Acceleration - Numba-optimized rule violation checking.

This module provides Numba-accelerated functions for checking rule violations
in symbolic/logic-based validation. It adapts the generic LoopAccelerator for
the specific case of Prolog-like rules.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray

from pff.config import ACCELERATION_CONFIG_PATH
from pff.shared.core.file_manager import FileManager
from ..core.logger import logger
from ..hash import stable_hash
from .loop_accelerator import LoopAccelerator, AcceleratorConfig, AcceleratorBackend

try:
    from numba import njit, prange, types
    from numba.typed import Dict

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if args and callable(args[0]) else decorator

    prange = range


def _load_symbolic_acceleration_settings() -> dict[str, Any]:
    try:
        cfg = FileManager().read(ACCELERATION_CONFIG_PATH, return_native=True) or {}
        symbolic_cfg = cfg.get("symbolic_rule_accelerator", {})
        return symbolic_cfg if isinstance(symbolic_cfg, dict) else {}
    except Exception as exc:  # pragma: no cover
        logger.warning(
            f"Failed to load symbolic acceleration config from {ACCELERATION_CONFIG_PATH}: {exc}",
        )
        return {}


_SYMBOLIC_ACCEL_SETTINGS = _load_symbolic_acceleration_settings()
VECTORIZED_BATCH_SIZE = int(_SYMBOLIC_ACCEL_SETTINGS.get("vectorized_batch_size", 500))


class RuleEncoder:
    """
    Encodes symbolic rules (Prolog-like) to Numba-compatible integer arrays.
    """

    def __init__(self):
        self.predicate_to_idx: dict[str, int] = {}
        self.idx_to_predicate: dict[int, str] = {}
        self.next_predicate_idx = 0

        self.entity_to_idx: dict[str, int] = {}
        self.idx_to_entity: dict[int, str] = {}
        self.next_entity_idx = 0

        self.VARIABLE_START = 1_000_000

        self._vocabulary_built = False

    def build_vocabulary_from_rules(self, rules: list[dict]) -> None:
        """
        Pre-build vocabulary from all rules to ensure deterministic encoding.

        Args:
            rules: List of rules to extract vocabulary from
        """
        all_predicates = set()
        all_entities = set()

        for rule in rules:
            if "head" in rule:
                head = rule["head"]
                if isinstance(head, dict):
                    all_predicates.add(head.get("predicate", ""))
                    subj = head.get("subject", "")
                    obj = head.get("object", "")
                    if subj and not (subj[0].isupper() if subj else False):
                        all_entities.add(subj)
                    if obj and not (obj[0].isupper() if obj else False):
                        all_entities.add(obj)

            if "body" in rule:
                for atom in rule["body"]:
                    if isinstance(atom, dict):
                        all_predicates.add(atom.get("predicate", ""))
                        subj = atom.get("subject", "")
                        obj = atom.get("object", "")
                        if subj and not (subj[0].isupper() if subj else False):
                            all_entities.add(subj)
                        if obj and not (obj[0].isupper() if obj else False):
                            all_entities.add(obj)

        sorted_predicates = sorted(all_predicates)
        sorted_entities = sorted(all_entities)

        for pred in sorted_predicates:
            if pred and pred not in self.predicate_to_idx:
                idx = self.next_predicate_idx
                self.predicate_to_idx[pred] = idx
                self.idx_to_predicate[idx] = pred
                self.next_predicate_idx += 1

        for entity in sorted_entities:
            if entity and entity not in self.entity_to_idx:
                idx = self.next_entity_idx
                self.entity_to_idx[entity] = idx
                self.idx_to_entity[idx] = entity
                self.next_entity_idx += 1

        self._vocabulary_built = True

        from pff.shared.core.logger import logger

        logger.debug(
            f"Built deterministic vocabulary: {len(self.predicate_to_idx)} predicates, "
            f"{len(self.entity_to_idx)} entities"
        )

    def encode_predicate(self, predicate: str) -> int:
        """Encode predicate to integer. O(1) average case."""
        if self._vocabulary_built:
            if predicate in self.predicate_to_idx:
                return self.predicate_to_idx[predicate]
            else:
                from pff.shared.core.logger import logger

                if not hasattr(self, "_logged_new_predicates"):
                    self._logged_new_predicates = 0

                if self._logged_new_predicates < 10:
                    logger.debug(
                        f"New predicate '{predicate}' not in pre-built vocabulary, adding dynamically"
                    )
                    self._logged_new_predicates += 1
                elif self._logged_new_predicates == 10:
                    self._logged_new_predicates += 1

        if predicate not in self.predicate_to_idx:
            idx = self.next_predicate_idx
            self.predicate_to_idx[predicate] = idx
            self.idx_to_predicate[idx] = predicate
            self.next_predicate_idx += 1
            return idx
        return self.predicate_to_idx[predicate]

    def encode_entity(self, entity: str) -> int:
        """
        Encode entity to integer with deterministic variable encoding.

        Args:
            entity: Entity string to encode

        Returns:
            Integer encoding (>= VARIABLE_START for variables)
        """
        if entity and entity[0].isupper():
            var_id = stable_hash(entity) % 100000
            return self.VARIABLE_START + var_id

        if self._vocabulary_built:
            if entity in self.entity_to_idx:
                return self.entity_to_idx[entity]
            else:
                from pff.shared.core.logger import logger

                if not hasattr(self, "_logged_new_entities"):
                    self._logged_new_entities = 0

                if self._logged_new_entities < 10:
                    logger.debug(
                        f"New entity '{entity}' not in pre-built vocabulary, adding dynamically"
                    )
                    self._logged_new_entities += 1
                elif self._logged_new_entities == 10:
                    self._logged_new_entities += 1

        if entity not in self.entity_to_idx:
            idx = self.next_entity_idx
            self.entity_to_idx[entity] = idx
            self.idx_to_entity[idx] = entity
            self.next_entity_idx += 1
            return idx
        return self.entity_to_idx[entity]

    def is_variable(self, entity_idx: int) -> bool:
        """Check if encoded entity is a variable."""
        return entity_idx >= self.VARIABLE_START

    def encode_atom(self, atom: dict) -> tuple[int, int, int]:
        """
        Encode atom (predicate, subject, object) to integer triple.

        Returns:
            (pred_idx, subj_idx, obj_idx)
        """
        pred_idx = self.encode_predicate(str(atom.get("predicate", "")))
        subj_idx = self.encode_entity(str(atom.get("subject", "")))
        obj_idx = self.encode_entity(str(atom.get("object", "")))
        return (pred_idx, subj_idx, obj_idx)

    def encode_rule(self, rule: dict) -> NDArray[np.int32]:
        """
        Encode rule to flat integer array.

        Args:
            rule: Rule dict with "head" and "body" keys

        Returns:
            NumPy array of int32
        """
        head = rule.get("head", {})
        body = rule.get("body", [])

        head_p, head_s, head_o = self.encode_atom(head)

        body_encoded = []
        for atom in body:
            p, s, o = self.encode_atom(atom)
            body_encoded.extend([p, s, o])

        flat = [len(body), head_p, head_s, head_o] + body_encoded

        return np.array(flat, dtype=np.int32)

    def encode_rules(self, rules: list[dict]) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """
        Encode multiple rules to padded arrays for Numba.

        Returns:
            (rules_array, lengths_array)
        """
        encoded_rules = [self.encode_rule(rule) for rule in rules]

        max_length = max(len(r) for r in encoded_rules) if encoded_rules else 0

        n_rules = len(encoded_rules)
        rules_array = np.full((n_rules, max_length), -1, dtype=np.int32)
        lengths_array = np.zeros(n_rules, dtype=np.int32)

        for i, rule in enumerate(encoded_rules):
            length = len(rule)
            rules_array[i, :length] = rule
            lengths_array[i] = length

        return rules_array, lengths_array

    def encode_triples(self, triples: list[tuple]) -> NDArray[np.int32]:
        """
        Encode triples to NumPy array.

        Args:
            triples: List of (subject, predicate, object) tuples

        Returns:
            Array of shape (n_triples, 3) with int32 indices
        """
        n_triples = len(triples)
        encoded = np.zeros((n_triples, 3), dtype=np.int32)

        for i, triple in enumerate(triples):
            s, p, o = triple
            encoded[i, 0] = self.encode_entity(str(s))
            encoded[i, 1] = self.encode_predicate(str(p))
            encoded[i, 2] = self.encode_entity(str(o))

        return encoded


@njit(cache=True)
def _check_atom_match_numba(
    atom_p: int,
    atom_s: int,
    atom_o: int,
    triples_dict: dict,
    variable_start: int,
) -> int:
    if atom_s < variable_start and atom_o < variable_start:
        if (atom_s, atom_p, atom_o) in triples_dict:
            return 1
        return 0
    return 0


@njit(cache=True)
def _check_rule_violation_numba(
    rule: NDArray[np.int32],
    rule_length: int,
    triples_dict: dict,
    variable_start: int,
) -> int:
    if rule_length < 4:
        return 0

    n_body = rule[0]
    head_p = rule[1]
    head_s = rule[2]
    head_o = rule[3]

    body_start = 4
    for i in range(n_body):
        offset = body_start + i * 3
        if offset + 2 >= rule_length:
            break

        body_p = rule[offset]
        body_s = rule[offset + 1]
        body_o = rule[offset + 2]

        if _check_atom_match_numba(body_p, body_s, body_o, triples_dict, variable_start) == 0:
            return 0

    head_satisfied = _check_atom_match_numba(head_p, head_s, head_o, triples_dict, variable_start)
    return 1 if head_satisfied == 0 else 0


@njit(cache=True, parallel=True)
def check_violations_batch_numba(
    rules: NDArray[np.int32],
    rule_lengths: NDArray[np.int32],
    triples_dict: dict,
    variable_start: int,
) -> NDArray[np.int8]:
    n_rules = rules.shape[0]
    violations = np.zeros(n_rules, dtype=np.int8)

    for i in prange(n_rules):
        violations[i] = _check_rule_violation_numba(
            rules[i, :],
            rule_lengths[i],
            triples_dict,
            variable_start,
        )

    return violations


class SymbolicRuleAccelerator:
    """
    High-level interface for accelerated rule violation checking.
    """

    def __init__(self, rules: list[dict], enable_numba: bool = True):
        """
        Initialize accelerator with rules.

        Args:
            rules: List of rule dicts with "head" and "body" keys
            enable_numba: Use Numba acceleration if available
        """
        self.rules = rules
        self.encoder = RuleEncoder()
        self.enable_numba = enable_numba and NUMBA_AVAILABLE

        logger.debug(f" Building deterministic vocabulary from {len(rules)} rules...")
        self.encoder.build_vocabulary_from_rules(rules)

        logger.info(f" Encoding {len(rules)} rules for Numba...")
        self.encoded_rules, self.rule_lengths = self.encoder.encode_rules(rules)

        logger.success(
            f" Rules encoded: shape={self.encoded_rules.shape}, "
            f"vocabulary={len(self.encoder.predicate_to_idx)} predicates, "
            f"{len(self.encoder.entity_to_idx)} entities (deterministic)"
        )

    def check_violations(
        self, sample_triples: list[tuple], validate: bool = False
    ) -> NDArray[np.int8]:
        if not self.enable_numba:
            return self._check_violations_python(sample_triples)

        # Use local imports to satisfy LSP and avoid top-level issues
        from numba.typed import Dict as NumbaDict  # noqa: PLC0415
        from numba import types  # noqa: PLC0415

        encoded_triples = self.encoder.encode_triples(sample_triples)

        triples_dict = NumbaDict.empty(
            key_type=types.UniTuple(types.int32, 3),
            value_type=types.int8,
        )
        for i in range(len(encoded_triples)):
            triples_dict[(encoded_triples[i, 0], encoded_triples[i, 1], encoded_triples[i, 2])] = 1

        violations = check_violations_batch_numba(
            self.encoded_rules,
            self.rule_lengths,
            triples_dict,
            self.encoder.VARIABLE_START,
        )

        if validate and len(self.rules) > 10:
            mismatch_rate = self._validate_numba_results(violations, sample_triples)
            if mismatch_rate > 0.05:
                return self._check_violations_python(sample_triples)

        return violations

    def _validate_numba_results(
        self, numba_violations: NDArray[np.int8], sample_triples: list[tuple]
    ) -> float:
        from pff.application.services.business_service import RuleValidator

        n_rules = len(self.rules)
        sample_size = max(10, n_rules // 10)
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(n_rules, min(sample_size, n_rules), replace=False)
        validator = RuleValidator()
        mismatch = 0
        for idx in sample_indices:
            try:
                business_rule = self._convert_to_business_rule(self.rules[idx], idx)
                violations_found = validator.validate_rules([business_rule], list(sample_triples))
                business_result = 1 if len(violations_found) > 0 else 0
                if numba_violations[idx] != business_result:
                    mismatch += 1
            except Exception:
                pass
        return mismatch / len(sample_indices) if len(sample_indices) > 0 else 0.0

    def _check_violations_python(self, sample_triples: list[tuple]) -> NDArray[np.int8]:
        from pff.application.services.business_service import RuleValidator

        validator = RuleValidator()
        violations = np.zeros(len(self.rules), dtype=np.int8)
        for idx, rule in enumerate(self.rules):
            try:
                business_rule = self._convert_to_business_rule(rule, idx)
                violations_found = validator.validate_rules([business_rule], list(sample_triples))
                violations[idx] = 1 if len(violations_found) > 0 else 0
            except Exception:
                violations[idx] = 0
        return violations

    def _convert_to_business_rule(self, rule: dict, rule_id: int) -> Any:
        from pff.application.services.business_service import Rule

        head = rule.get("head", {})
        body = rule.get("body", [])
        return Rule(
            id=f"numba_rule_{rule_id}",
            confidence=rule.get("confidence", 0.0),
            head=head,
            body=body,
            source="numba_accelerator",
        )

    def check_violations_vectorized(
        self,
        samples: list[list[tuple]],
    ) -> list[NDArray[np.int8]]:
        if not self.enable_numba:
            return [self.check_violations(sample) for sample in samples]

        all_violations = []
        for sample in samples:
            all_violations.append(self.check_violations(sample))
        return all_violations

    def check_violations_batch(
        self,
        samples: list[list[tuple]],
        use_parallel: bool = True,
        batch_size: int = 1000,
    ) -> list[NDArray[np.int8]]:
        """
        Check violations for multiple samples with optimized batching.

        Args:
            samples: List of samples, each is a list of triples
            use_parallel: Use parallel execution via LoopAccelerator
            batch_size: Size of batches for processing to reduce overhead

        Returns:
            List of violation arrays, one per sample
        """
        if not samples:
            return []

        if len(samples) < 50:
            return [self.check_violations(sample) for sample in samples]

        if self.enable_numba and NUMBA_AVAILABLE and len(samples) > 200:
            try:
                logger.debug(f"Using vectorized processing for {len(samples)} samples")
                return self.check_violations_vectorized(samples)
            except Exception as e:
                logger.debug(f"Vectorized processing failed: {e}")

        if self.enable_numba and NUMBA_AVAILABLE:
            try:
                config = AcceleratorConfig(
                    backend=AcceleratorBackend.NUMBA,
                    parallel=True,
                    fastmath=True,
                    cache=True,
                )
                accelerator = LoopAccelerator(config=config)
                return accelerator.map(self.check_violations, samples)
            except Exception as e:
                logger.debug(f"Numba backend failed, falling back to PARALLEL: {e}")

        if use_parallel and len(samples) > 100:
            all_results = []

            for i in range(0, len(samples), batch_size):
                batch = samples[i : i + batch_size]
                logger.debug(f"Processing batch {i // batch_size + 1}: {len(batch)} samples")

                config = AcceleratorConfig(
                    backend=AcceleratorBackend.PARALLEL,
                    parallel=True,
                )
                accelerator = LoopAccelerator(config=config)

                try:
                    batch_results = accelerator.map(self.check_violations, batch)
                    all_results.extend(batch_results)
                except Exception as e:
                    logger.warning(
                        f"Parallel batch failed: {e}, falling back to sequential for this batch"
                    )
                    batch_results = [self.check_violations(sample) for sample in batch]
                    all_results.extend(batch_results)

            return all_results
        else:
            return [self.check_violations(sample) for sample in samples]

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about encoded rules."""
        return {
            "n_rules": len(self.rules),
            "n_predicates": len(self.encoder.predicate_to_idx),
            "n_entities": len(self.encoder.entity_to_idx),
            "encoded_shape": self.encoded_rules.shape,
            "numba_enabled": self.enable_numba,
        }
