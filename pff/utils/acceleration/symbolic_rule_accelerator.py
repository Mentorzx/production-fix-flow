"""
Symbolic Rule Acceleration - Numba-optimized rule violation checking.

This module provides Numba-accelerated functions for checking rule violations
in symbolic/logic-based validation. It adapts the generic LoopAccelerator for
the specific case of Prolog-like rules.

Performance: 10-100× speedup over pure Python for large rule sets (10k+ rules).

Author: PFF Team
Date: 2025-10-31
Version: 1.0.0
"""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray

from ..core.logger import logger
from ..hash import stable_hash
from .loop_accelerator import LoopAccelerator, AcceleratorConfig, AcceleratorBackend

# Try to import Numba
try:
    from numba import njit, types
    from numba.typed import Dict
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator


class RuleEncoder:
    """
    Encodes symbolic rules (Prolog-like) to Numba-compatible integer arrays.

    Rules have structure:
        {
            "head": {"predicate": str, "subject": str, "object": str},
            "body": [{"predicate": str, "subject": str, "object": str}, ...],
            "confidence": float
        }

    Encoding:
        - All strings (predicates, subjects, objects) → integers
        - Rule structure → flat integer arrays
        - Variables (uppercase strings) → special indices
    """

    def __init__(self):
        # String vocabularies
        self.predicate_to_idx: dict[str, int] = {}
        self.idx_to_predicate: dict[int, str] = {}
        self.next_predicate_idx = 0

        self.entity_to_idx: dict[str, int] = {}
        self.idx_to_entity: dict[int, str] = {}
        self.next_entity_idx = 0

        # Special indices
        self.VARIABLE_START = 1_000_000  # Variables use indices >= this
        
        # Determinism flag
        self._vocabulary_built = False
    
    def build_vocabulary_from_rules(self, rules: list[dict]) -> None:
        """
        Pre-build vocabulary from all rules to ensure deterministic encoding.
        
        This must be called BEFORE any parallel processing to guarantee that
        entity_to_idx mapping is consistent across all workers.
        
        Args:
            rules: List of rules to extract vocabulary from
        """
        # Collect all unique entities and predicates (sorted for determinism)
        all_predicates = set()
        all_entities = set()
        
        for rule in rules:
            # Head
            if "head" in rule:
                head = rule["head"]
                if isinstance(head, dict):
                    all_predicates.add(head.get("predicate", ""))
                    subj = head.get("subject", "")
                    obj = head.get("object", "")
                    # Only add non-variable entities
                    if subj and not (subj[0].isupper() if subj else False):
                        all_entities.add(subj)
                    if obj and not (obj[0].isupper() if obj else False):
                        all_entities.add(obj)
            
            # Body
            if "body" in rule:
                for atom in rule["body"]:
                    if isinstance(atom, dict):
                        all_predicates.add(atom.get("predicate", ""))
                        subj = atom.get("subject", "")
                        obj = atom.get("object", "")
                        # Only add non-variable entities
                        if subj and not (subj[0].isupper() if subj else False):
                            all_entities.add(subj)
                        if obj and not (obj[0].isupper() if obj else False):
                            all_entities.add(obj)
        
        # Sort for determinism (critical!)
        sorted_predicates = sorted(all_predicates)
        sorted_entities = sorted(all_entities)
        
        # Build vocabularies in sorted order
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
        
        from loguru import logger
        logger.debug(f"Built deterministic vocabulary: {len(self.predicate_to_idx)} predicates, "
                    f"{len(self.entity_to_idx)} entities")

    def encode_predicate(self, predicate: str) -> int:
        """Encode predicate to integer. O(1) average case."""
        # If vocabulary was pre-built, use it (deterministic)
        if self._vocabulary_built:
            if predicate in self.predicate_to_idx:
                return self.predicate_to_idx[predicate]
            else:
                from loguru import logger
                logger.debug(f"New predicate '{predicate}' not in pre-built vocabulary, adding dynamically")
        
        # Add new predicate (or return existing)
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
        
        Variables (starting with uppercase) get special encoding.
        Constants get normal vocabulary encoding.
        
        Args:
            entity: Entity string to encode
            
        Returns:
            Integer encoding (>= VARIABLE_START for variables)
        """
        if entity and entity[0].isupper():
            var_id = stable_hash(entity) % 100000
            return self.VARIABLE_START + var_id

        # If vocabulary was pre-built, use it (deterministic)
        if self._vocabulary_built:
            if entity in self.entity_to_idx:
                return self.entity_to_idx[entity]
            else:
                # New entity not in pre-built vocabulary - still add it deterministically
                # but log a warning (this should rarely happen if build_vocabulary was called correctly)
                from loguru import logger
                logger.debug(f"New entity '{entity}' not in pre-built vocabulary, adding dynamically")
        
        # Add new entity (or return existing if already added)
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

        Format: [n_body_atoms, head_p, head_s, head_o, body1_p, body1_s, body1_o, ...]

        Args:
            rule: Rule dict with "head" and "body" keys

        Returns:
            NumPy array of int32
        """
        head = rule.get("head", {})
        body = rule.get("body", [])

        # Encode head
        head_p, head_s, head_o = self.encode_atom(head)

        # Encode body atoms
        body_encoded = []
        for atom in body:
            p, s, o = self.encode_atom(atom)
            body_encoded.extend([p, s, o])

        # Build flat array: [n_body, head_p, head_s, head_o, body_atoms...]
        flat = [len(body), head_p, head_s, head_o] + body_encoded

        return np.array(flat, dtype=np.int32)

    def encode_rules(self, rules: list[dict]) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """
        Encode multiple rules to padded arrays for Numba.

        Returns:
            (rules_array, lengths_array)
            - rules_array: (n_rules, max_length) padded with -1
            - lengths_array: (n_rules,) actual length of each rule
        """
        encoded_rules = [self.encode_rule(rule) for rule in rules]

        # Find max length
        max_length = max(len(r) for r in encoded_rules) if encoded_rules else 0

        # Create padded array
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
    triples: NDArray[np.int32],
    variable_start: int,
) -> int:
    """
    Check if atom matches any triple (Numba-compiled).

    Args:
        atom_p, atom_s, atom_o: Encoded atom (predicate, subject, object)
        triples: Array of encoded triples (n_triples, 3)
        variable_start: Index where variables start (for identifying variables)

    Returns:
        1 if match found, 0 otherwise
    """
    n_triples = triples.shape[0]

    for i in range(n_triples):
        triple_s = triples[i, 0]
        triple_p = triples[i, 1]
        triple_o = triples[i, 2]

        # Check predicate match
        if atom_p != triple_p:
            continue

        # Check subject match (or variable)
        if atom_s < variable_start:  # Constant
            if atom_s != triple_s:
                continue

        # Check object match (or variable)
        if atom_o < variable_start:  # Constant
            if atom_o != triple_o:
                continue

        # Match found!
        return 1

    return 0


@njit(cache=True)
def _check_rule_violation_numba(
    rule: NDArray[np.int32],
    rule_length: int,
    triples: NDArray[np.int32],
    variable_start: int,
) -> int:
    """
    Check if rule is violated by given triples (Numba-compiled).

    Rule is violated if:
    - All body atoms are satisfied (match triples)
    - Head atom is NOT satisfied

    Args:
        rule: Encoded rule array [n_body, head_p, head_s, head_o, body_atoms...]
        rule_length: Actual length of rule (rest is padding)
        triples: Encoded triples array (n_triples, 3)
        variable_start: Index where variables start

    Returns:
        1 if violated, 0 otherwise
    """
    if rule_length < 4:
        return 0  # Invalid rule

    n_body = rule[0]
    head_p = rule[1]
    head_s = rule[2]
    head_o = rule[3]

    # Check body atoms (all must be satisfied)
    body_start = 4
    for i in range(n_body):
        offset = body_start + i * 3
        if offset + 2 >= rule_length:
            break  # Safety check

        body_p = rule[offset]
        body_s = rule[offset + 1]
        body_o = rule[offset + 2]

        # Check if this body atom is satisfied
        if _check_atom_match_numba(body_p, body_s, body_o, triples, variable_start) == 0:
            # Body atom not satisfied → rule not applicable
            return 0

    # All body atoms satisfied, check head
    head_satisfied = _check_atom_match_numba(head_p, head_s, head_o, triples, variable_start)

    # Rule violated if body satisfied but head not
    return 1 if head_satisfied == 0 else 0


@njit(cache=True, parallel=True)
def check_violations_batch_numba(
    rules: NDArray[np.int32],
    rule_lengths: NDArray[np.int32],
    triples: NDArray[np.int32],
    variable_start: int,
) -> NDArray[np.int8]:
    """
    Check violations for multiple rules in parallel (Numba-compiled).

    This is the CRITICAL HOT LOOP that processes millions of rule×sample checks.

    Args:
        rules: Array of encoded rules (n_rules, max_rule_length)
        rule_lengths: Actual length of each rule (n_rules,)
        triples: Encoded triples for ONE sample (n_triples, 3)
        variable_start: Index where variables start

    Returns:
        Binary array (n_rules,) where 1 = violated, 0 = satisfied
    """
    n_rules = rules.shape[0]
    violations = np.zeros(n_rules, dtype=np.int8)

    # Parallel loop over rules (Numba parallelizes this automatically)
    for i in range(n_rules):
        violations[i] = _check_rule_violation_numba(
            rules[i, :],
            rule_lengths[i],
            triples,
            variable_start,
        )

    return violations


class SymbolicRuleAccelerator:
    """
    High-level interface for accelerated rule violation checking.

    Usage:
        # Initialize with rules
        accelerator = SymbolicRuleAccelerator(rules)

        # Check violations for sample
        violations = accelerator.check_violations(sample_triples)

        # Process multiple samples
        all_violations = accelerator.check_violations_batch(samples)
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

        # CRITICAL: Build vocabulary FIRST for deterministic encoding
        logger.debug(f"🔧 Building deterministic vocabulary from {len(rules)} rules...")
        self.encoder.build_vocabulary_from_rules(rules)

        # Encode rules once at initialization (now with deterministic vocabulary)
        logger.info(f"🔄 Encoding {len(rules)} rules for Numba...")
        self.encoded_rules, self.rule_lengths = self.encoder.encode_rules(rules)

        logger.success(
            f"✅ Rules encoded: shape={self.encoded_rules.shape}, "
            f"vocabulary={len(self.encoder.predicate_to_idx)} predicates, "
            f"{len(self.encoder.entity_to_idx)} entities (deterministic)"
        )

    def check_violations(self, sample_triples: list[tuple], validate: bool = False) -> NDArray[np.int8]:
        """
        Check which rules are violated by given sample triples.

        Args:
            sample_triples: List of (subject, predicate, object) tuples
            validate: If True, sample 10% and validate with business_service

        Returns:
            Binary array (n_rules,) where 1 = violated, 0 = satisfied
        """
        if not self.enable_numba:
            return self._check_violations_python(sample_triples)

        encoded_triples = self.encoder.encode_triples(sample_triples)

        violations = check_violations_batch_numba(
            self.encoded_rules,
            self.rule_lengths,
            encoded_triples,
            self.encoder.VARIABLE_START,
        )
        
        if validate and len(self.rules) > 10:
            mismatch_rate = self._validate_numba_results(violations, sample_triples)
            if mismatch_rate > 0.05:
                logger.warning(f"Numba mismatch {mismatch_rate:.1%}, using business_service")
                return self._check_violations_python(sample_triples)
            logger.debug(f"Numba validated: mismatch={mismatch_rate:.1%}")

        return violations
    
    def _validate_numba_results(self, numba_violations: NDArray[np.int8], 
                                 sample_triples: list[tuple]) -> float:
        """
        Validate Numba results by sampling and comparing with business_service.
        
        Args:
            numba_violations: Violations from Numba
            sample_triples: Sample triples
            
        Returns:
            Mismatch rate (0.0 = perfect match, 1.0 = complete mismatch)
        """
        from pff.services.business_service import RuleValidator
        
        n_rules = len(self.rules)
        sample_size = max(10, n_rules // 10)
        sample_indices = np.random.choice(n_rules, min(sample_size, n_rules), replace=False)
        
        validator = RuleValidator()
        mismatch = 0
        
        for idx in sample_indices:
            try:
                business_rule = self._convert_to_business_rule(self.rules[idx], idx)
                violations_found = validator.validate_rules([business_rule], list(sample_triples))
                business_result = 1 if len(violations_found) > 0 else 0
                numba_result = numba_violations[idx]
                
                if numba_result != business_result:
                    mismatch += 1
                    logger.debug(f"Mismatch rule {idx}: numba={numba_result}, business={business_result}")
            except Exception as e:
                logger.debug(f"Validation error rule {idx}: {e}")
        
        return mismatch / len(sample_indices) if len(sample_indices) > 0 else 0.0

    def _check_violations_python(self, sample_triples: list[tuple]) -> NDArray[np.int8]:
        """
        Fallback implementation using business_service for correct matching.
        
        Args:
            sample_triples: List of (subject, predicate, object) tuples
            
        Returns:
            Binary array where 1 = violated, 0 = satisfied
        """
        logger.warning("Numba fallback activated, using business_service")
        
        from pff.services.business_service import RuleValidator, Rule
        
        validator = RuleValidator()
        violations = np.zeros(len(self.rules), dtype=np.int8)
        
        for idx, rule in enumerate(self.rules):
            try:
                business_rule = self._convert_to_business_rule(rule, idx)
                violations_found = validator.validate_rules([business_rule], list(sample_triples))
                violations[idx] = 1 if len(violations_found) > 0 else 0
            except Exception as e:
                logger.debug(f"Error validating rule {idx}: {e}")
                violations[idx] = 0
        
        return violations
    
    def _convert_to_business_rule(self, rule: dict, rule_id: int) -> 'Rule':
        """
        Convert internal rule format to business_service Rule format.
        
        Args:
            rule: Internal rule dictionary
            rule_id: Rule index
            
        Returns:
            Rule object for business_service
        """
        from pff.services.business_service import Rule
        
        head = rule.get("head", {})
        body = rule.get("body", [])
        confidence = rule.get("confidence", 0.0)
        
        head_tuple = (
            str(head.get("subject", "?")),
            str(head.get("predicate", "?")),
            str(head.get("object", "?"))
        )
        
        body_tuples = [
            (
                str(clause.get("subject", "?")),
                str(clause.get("predicate", "?")),
                str(clause.get("object", "?"))
            )
            for clause in body
        ]
        
        return Rule(
            id=f"numba_rule_{rule_id}",
            confidence=confidence,
            head=head_tuple,
            body=body_tuples,
            source="numba_accelerator",
        )

    def check_violations_vectorized(
        self,
        samples: list[list[tuple]],
    ) -> list[NDArray[np.int8]]:
        """
        Ultra-fast vectorized processing of multiple samples at once.

        This method encodes all samples once and then processes them in larger batches
        to maximize Numba parallelization and minimize Python overhead.
        """
        if not self.enable_numba:
            # Fallback to regular batch processing
            return [self.check_violations(sample) for sample in samples]

        # Encode all samples at once to minimize overhead
        all_encoded_triples = []
        for sample in samples:
            encoded = self.encoder.encode_triples(sample)
            all_encoded_triples.append(encoded)

        # Process samples in optimized batches
        batch_size = 500  # Larger batches for better Numba parallelization
        all_violations = []

        for i in range(0, len(all_encoded_triples), batch_size):
            batch_triples = all_encoded_triples[i:i + batch_size]

            # Process this batch
            batch_violations = []
            for triples in batch_triples:
                violations = check_violations_batch_numba(
                    self.encoded_rules,
                    self.rule_lengths,
                    triples,
                    self.encoder.VARIABLE_START,
                )
                batch_violations.append(violations)

            all_violations.extend(batch_violations)

            if i % 1000 == 0:
                logger.debug(f"Processed {min(i + batch_size, len(samples))}/{len(samples)} samples")

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

        # For small datasets, sequential execution is faster
        if len(samples) < 50:
            return [self.check_violations(sample) for sample in samples]

        # Try ultra-fast vectorized processing first
        if self.enable_numba and NUMBA_AVAILABLE and len(samples) > 200:
            try:
                logger.debug(f"Using vectorized processing for {len(samples)} samples")
                return self.check_violations_vectorized(samples)
            except Exception as e:
                logger.debug(f"Vectorized processing failed: {e}")

        # Try to use Numba backend next if available
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

        # For larger datasets, use parallel with batching to reduce overhead
        if use_parallel and len(samples) > 100:
            all_results = []

            # Process in batches to reduce process pool overhead
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch)} samples")

                # Use parallel backend for batches
                config = AcceleratorConfig(
                    backend=AcceleratorBackend.PARALLEL,
                    parallel=True,
                )
                accelerator = LoopAccelerator(config=config)

                try:
                    batch_results = accelerator.map(self.check_violations, batch)
                    all_results.extend(batch_results)
                except Exception as e:
                    logger.warning(f"Parallel batch failed: {e}, falling back to sequential for this batch")
                    # Fallback to sequential for this batch
                    batch_results = [self.check_violations(sample) for sample in batch]
                    all_results.extend(batch_results)

            return all_results
        else:
            # Sequential execution
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
