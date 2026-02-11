//! Symbolic rule acceleration — Rust replacement for symbolic_rule_accelerator.py.
//!
//! Provides:
//! - RuleEncoder (Prolog-like rule → integer encoding)
//! - Batch violation checking with Rayon parallelism

use ahash::{AHashMap, AHashSet};
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Encodes symbolic rules (Prolog-like) to integer arrays.
#[pyclass]
pub struct RuleEncoder {
    predicate_to_idx: AHashMap<String, i32>,
    idx_to_predicate: AHashMap<i32, String>,
    next_predicate_idx: i32,

    entity_to_idx: AHashMap<String, i32>,
    idx_to_entity: AHashMap<i32, String>,
    next_entity_idx: i32,

    variable_start: i32,
    vocabulary_built: bool,
}

#[pymethods]
impl RuleEncoder {
    #[new]
    fn new() -> Self {
        Self {
            predicate_to_idx: AHashMap::new(),
            idx_to_predicate: AHashMap::new(),
            next_predicate_idx: 0,
            entity_to_idx: AHashMap::new(),
            idx_to_entity: AHashMap::new(),
            next_entity_idx: 0,
            variable_start: 1_000_000,
            vocabulary_built: false,
        }
    }

    /// Pre-build vocabulary from rules for deterministic encoding.
    fn build_vocabulary_from_rules(
        &mut self,
        rules: Vec<Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        let mut all_predicates: Vec<String> = Vec::new();
        let mut all_entities: Vec<String> = Vec::new();

        for rule in &rules {
            if let Some(head) = rule.get_item("head")?
                && let Ok(head_dict) = head.downcast::<pyo3::types::PyDict>()
            {
                if let Some(pred) = head_dict.get_item("predicate")? {
                    let p: String = pred.extract()?;
                    all_predicates.push(p);
                }
                for key in &["subject", "object"] {
                    if let Some(val) = head_dict.get_item(*key)? {
                        let s: String = val.extract()?;
                        if !s.is_empty() && !s.starts_with(char::is_uppercase) {
                            all_entities.push(s);
                        }
                    }
                }
            }
            if let Some(body) = rule.get_item("body")?
                && let Ok(body_list) = body.downcast::<pyo3::types::PyList>()
            {
                for atom in body_list.iter() {
                    if let Ok(atom_dict) = atom.downcast::<pyo3::types::PyDict>() {
                        if let Some(pred) = atom_dict.get_item("predicate")? {
                            let p: String = pred.extract()?;
                            all_predicates.push(p);
                        }
                        for key in &["subject", "object"] {
                            if let Some(val) = atom_dict.get_item(*key)? {
                                let s: String = val.extract()?;
                                if !s.is_empty() && !s.starts_with(char::is_uppercase) {
                                    all_entities.push(s);
                                }
                            }
                        }
                    }
                }
            }
        }

        all_predicates.sort();
        all_predicates.dedup();
        all_entities.sort();
        all_entities.dedup();

        for pred in all_predicates {
            if !pred.is_empty() && !self.predicate_to_idx.contains_key(&pred) {
                let idx = self.next_predicate_idx;
                self.predicate_to_idx.insert(pred.clone(), idx);
                self.idx_to_predicate.insert(idx, pred);
                self.next_predicate_idx += 1;
            }
        }

        for entity in all_entities {
            if !entity.is_empty() && !self.entity_to_idx.contains_key(&entity) {
                let idx = self.next_entity_idx;
                self.entity_to_idx.insert(entity.clone(), idx);
                self.idx_to_entity.insert(idx, entity);
                self.next_entity_idx += 1;
            }
        }

        self.vocabulary_built = true;
        Ok(())
    }

    /// Encode a predicate string to an integer index.
    fn encode_predicate(&mut self, predicate: &str) -> i32 {
        if let Some(&idx) = self.predicate_to_idx.get(predicate) {
            return idx;
        }
        let idx = self.next_predicate_idx;
        self.predicate_to_idx.insert(predicate.to_string(), idx);
        self.idx_to_predicate.insert(idx, predicate.to_string());
        self.next_predicate_idx += 1;
        idx
    }

    /// Encode an entity string to an integer index.
    /// Variables (uppercase-starting) are hashed via BLAKE3.
    fn encode_entity(&mut self, entity: &str) -> i32 {
        if !entity.is_empty() && entity.starts_with(char::is_uppercase) {
            let hash = blake3::hash(entity.as_bytes());
            let bytes = hash.as_bytes();
            let var_id = u64::from_be_bytes(bytes[..8].try_into().unwrap()) % 100_000;
            return self.variable_start + var_id as i32;
        }

        if let Some(&idx) = self.entity_to_idx.get(entity) {
            return idx;
        }
        let idx = self.next_entity_idx;
        self.entity_to_idx.insert(entity.to_string(), idx);
        self.idx_to_entity.insert(idx, entity.to_string());
        self.next_entity_idx += 1;
        idx
    }

    /// Check if an encoded entity index represents a variable.
    fn is_variable(&self, entity_idx: i32) -> bool {
        entity_idx >= self.variable_start
    }

    /// Encode triples [(s, p, o), ...] to an (N, 3) i32 array.
    fn encode_triples<'py>(
        &mut self,
        py: Python<'py>,
        triples: Vec<(String, String, String)>,
    ) -> Bound<'py, PyArray1<i32>> {
        let n = triples.len();
        let mut flat = Vec::with_capacity(n * 3);
        for (s, p, o) in triples {
            flat.push(self.encode_entity(&s));
            flat.push(self.encode_predicate(&p));
            flat.push(self.encode_entity(&o));
        }
        Array1::from_vec(flat).into_pyarray(py)
    }

    /// Get statistics about the encoder state.
    fn get_stats(&self) -> (usize, usize, bool) {
        (
            self.predicate_to_idx.len(),
            self.entity_to_idx.len(),
            self.vocabulary_built,
        )
    }
}

/// Check rule violations for a batch of rules against a triple set.
///
/// Args:
///     rules: (n_rules, max_len) int32 array of encoded rules.
///     rule_lengths: (n_rules,) int32 array of actual rule lengths.
///     triple_subjects: (n_triples,) int32 array.
///     triple_predicates: (n_triples,) int32 array.
///     triple_objects: (n_triples,) int32 array.
///     variable_start: Integer threshold for variable detection.
///
/// Returns:
///     (n_rules,) int8 array where 1 = violation, 0 = no violation.
#[pyfunction]
pub fn check_violations_batch<'py>(
    py: Python<'py>,
    rules: PyReadonlyArray2<i32>,
    rule_lengths: PyReadonlyArray1<i32>,
    triple_subjects: PyReadonlyArray1<i32>,
    triple_predicates: PyReadonlyArray1<i32>,
    triple_objects: PyReadonlyArray1<i32>,
    variable_start: i32,
) -> Bound<'py, PyArray1<i8>> {
    let rules_arr = rules.as_array();
    let lengths = crate::shared::to_vec(&rule_lengths);
    let ts = crate::shared::to_vec(&triple_subjects);
    let tp = crate::shared::to_vec(&triple_predicates);
    let to = crate::shared::to_vec(&triple_objects);

    let rules_owned: Vec<Vec<i32>> = (0..rules_arr.nrows())
        .map(|i| rules_arr.row(i).iter().copied().collect())
        .collect();

    let violations = py.allow_threads(|| {
        let mut triple_set: AHashSet<(i32, i32, i32)> = AHashSet::with_capacity(ts.len());
        for i in 0..ts.len() {
            triple_set.insert((ts[i], tp[i], to[i]));
        }

        let n_rules = lengths.len();

        let violations: Vec<i8> = (0..n_rules)
            .into_par_iter()
            .map(|i| {
                let rule_len = lengths[i] as usize;
                if rule_len < 4 {
                    return 0i8;
                }

                let row = &rules_owned[i];
                let n_body = row[0] as usize;
                let head_p = row[1];
                let head_s = row[2];
                let head_o = row[3];

                let body_start = 4usize;
                for b in 0..n_body {
                    let offset = body_start + b * 3;
                    if offset + 2 >= rule_len {
                        break;
                    }
                    let bp = row[offset];
                    let bs = row[offset + 1];
                    let bo = row[offset + 2];

                    if bs < variable_start
                        && bo < variable_start
                        && !triple_set.contains(&(bs, bp, bo))
                    {
                        return 0i8;
                    }
                }

                if head_s < variable_start
                    && head_o < variable_start
                    && !triple_set.contains(&(head_s, head_p, head_o))
                {
                    return 1i8;
                }

                0i8
            })
            .collect();

        violations
    });

    Array1::from_vec(violations).into_pyarray(py)
}
