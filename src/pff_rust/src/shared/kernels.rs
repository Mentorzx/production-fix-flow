//! Numerical kernels for KGC.
//!
//! Consolidates kernels and similarity functions from Python into a single module.

use ahash::AHashMap;
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use super::to_vec;

/// O(1) string↔int vocabulary encoder for entities and relations.
#[pyclass]
pub struct VocabularyEncoder {
    entity_to_idx: AHashMap<String, i64>,
    idx_to_entity: AHashMap<i64, String>,
    next_entity_idx: i64,

    relation_to_idx: AHashMap<String, i64>,
    idx_to_relation: AHashMap<i64, String>,
    next_relation_idx: i64,

    variable_start: i64,
}

#[pymethods]
impl VocabularyEncoder {
    #[new]
    fn new() -> Self {
        Self {
            entity_to_idx: AHashMap::new(),
            idx_to_entity: AHashMap::new(),
            next_entity_idx: 0,
            relation_to_idx: AHashMap::new(),
            idx_to_relation: AHashMap::new(),
            next_relation_idx: 0,
            variable_start: 1_000_000,
        }
    }

    /// Encode an entity string to a deterministic integer index.
    fn encode_entity(&mut self, entity: &str) -> i64 {
        if let Some(&idx) = self.entity_to_idx.get(entity) {
            return idx;
        }
        let idx = self.next_entity_idx;
        self.entity_to_idx.insert(entity.to_string(), idx);
        self.idx_to_entity.insert(idx, entity.to_string());
        self.next_entity_idx += 1;
        idx
    }

    /// Decode an entity index back to string.
    fn decode_entity(&self, idx: i64) -> Option<String> {
        self.idx_to_entity.get(&idx).cloned()
    }

    /// Encode a relation string to a deterministic integer index.
    fn encode_relation(&mut self, relation: &str) -> i64 {
        if let Some(&idx) = self.relation_to_idx.get(relation) {
            return idx;
        }
        let idx = self.next_relation_idx;
        self.relation_to_idx.insert(relation.to_string(), idx);
        self.idx_to_relation.insert(idx, relation.to_string());
        self.next_relation_idx += 1;
        idx
    }

    /// Decode a relation index back to string.
    fn decode_relation(&self, idx: i64) -> Option<String> {
        self.idx_to_relation.get(&idx).cloned()
    }

    /// Encode triples [(s, p, o), ...] to an (N, 3) int64 array.
    fn encode_triples(&mut self, triples: Vec<(String, String, String)>) -> Vec<[i64; 3]> {
        triples
            .into_iter()
            .map(|(s, p, o)| {
                let si = self.encode_entity(&s);
                let pi = self.encode_relation(&p);
                let oi = self.encode_entity(&o);
                [si, pi, oi]
            })
            .collect()
    }

    /// Encode a pattern (subject, predicate, object) where uppercase-starting
    /// strings are treated as variables and hashed via BLAKE3.
    fn encode_pattern(&mut self, subject: &str, predicate: &str, object: &str) -> [i64; 3] {
        let s = self.encode_pattern_term(subject, true);
        let p = self.encode_pattern_term(predicate, false);
        let o = self.encode_pattern_term(object, true);
        [s, p, o]
    }

    /// Number of encoded entities.
    fn num_entities(&self) -> i64 {
        self.next_entity_idx
    }

    /// Number of encoded relations.
    fn num_relations(&self) -> i64 {
        self.next_relation_idx
    }
}

impl VocabularyEncoder {
    fn encode_pattern_term(&mut self, term: &str, is_entity: bool) -> i64 {
        if term.starts_with(char::is_uppercase) {
            let hash = blake3::hash(term.as_bytes());
            let bytes = hash.as_bytes();
            let var_id = u64::from_be_bytes(bytes[..8].try_into().unwrap()) % 100_000;
            self.variable_start + var_id as i64
        } else if is_entity {
            self.encode_entity(term)
        } else {
            self.encode_relation(term)
        }
    }
}

/// Structure-of-Arrays triple store with SPO/POS/OSP sorted indexes.
#[pyclass]
pub struct TripleStoreSoA {
    subjects: Vec<i32>,
    predicates: Vec<i32>,
    objects: Vec<i32>,
    spo_index: Vec<usize>,
    pos_index: Vec<usize>,
    osp_index: Vec<usize>,
}

#[pymethods]
impl TripleStoreSoA {
    #[new]
    fn new() -> Self {
        Self {
            subjects: Vec::new(),
            predicates: Vec::new(),
            objects: Vec::new(),
            spo_index: Vec::new(),
            pos_index: Vec::new(),
            osp_index: Vec::new(),
        }
    }

    /// Load triples from parallel arrays of subjects, predicates, objects.
    fn load_from_arrays(
        &mut self,
        subjects: PyReadonlyArray1<i32>,
        predicates: PyReadonlyArray1<i32>,
        objects: PyReadonlyArray1<i32>,
    ) {
        let s = to_vec(&subjects);
        let p = to_vec(&predicates);
        let o = to_vec(&objects);

        self.subjects = s;
        self.predicates = p;
        self.objects = o;
        self.build_indexes();
    }

    /// Number of triples stored.
    fn len(&self) -> usize {
        self.subjects.len()
    }

    /// Get subjects array.
    fn get_subjects<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        Array1::from_vec(self.subjects.clone()).into_pyarray(py)
    }

    /// Get predicates array.
    fn get_predicates<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        Array1::from_vec(self.predicates.clone()).into_pyarray(py)
    }

    /// Get objects array.
    fn get_objects<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        Array1::from_vec(self.objects.clone()).into_pyarray(py)
    }

    /// Get SPO-sorted index.
    fn get_spo_index<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let idx: Vec<i64> = self.spo_index.iter().map(|&i| i as i64).collect();
        Array1::from_vec(idx).into_pyarray(py)
    }

    /// Get POS-sorted index.
    fn get_pos_index<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let idx: Vec<i64> = self.pos_index.iter().map(|&i| i as i64).collect();
        Array1::from_vec(idx).into_pyarray(py)
    }

    /// Get OSP-sorted index.
    fn get_osp_index<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let idx: Vec<i64> = self.osp_index.iter().map(|&i| i as i64).collect();
        Array1::from_vec(idx).into_pyarray(py)
    }

    /// Find triples matching a pattern (s, p, o) where -1 means wildcard.
    ///
    /// Uses binary search on sorted SPO/POS/OSP indices for O(log n + k)
    /// instead of O(n) linear scan.
    fn find_matching<'py>(
        &self,
        py: Python<'py>,
        s: i32,
        p: i32,
        o: i32,
    ) -> Bound<'py, PyArray2<i32>> {
        let mut results: Vec<[i32; 3]> = Vec::new();

        if !self.subjects.is_empty() {
            if s >= 0 {
                let start = self.spo_index.partition_point(|&i| self.subjects[i] < s);
                let end =
                    start + self.spo_index[start..].partition_point(|&i| self.subjects[i] <= s);
                for &i in &self.spo_index[start..end] {
                    if (p < 0 || self.predicates[i] == p) && (o < 0 || self.objects[i] == o) {
                        results.push([self.subjects[i], self.predicates[i], self.objects[i]]);
                    }
                }
            } else if p >= 0 {
                let start = self.pos_index.partition_point(|&i| self.predicates[i] < p);
                let end =
                    start + self.pos_index[start..].partition_point(|&i| self.predicates[i] <= p);
                for &i in &self.pos_index[start..end] {
                    if (s < 0 || self.subjects[i] == s) && (o < 0 || self.objects[i] == o) {
                        results.push([self.subjects[i], self.predicates[i], self.objects[i]]);
                    }
                }
            } else if o >= 0 {
                let start = self.osp_index.partition_point(|&i| self.objects[i] < o);
                let end =
                    start + self.osp_index[start..].partition_point(|&i| self.objects[i] <= o);
                for &i in &self.osp_index[start..end] {
                    if (s < 0 || self.subjects[i] == s) && (p < 0 || self.predicates[i] == p) {
                        results.push([self.subjects[i], self.predicates[i], self.objects[i]]);
                    }
                }
            } else {
                for i in 0..self.subjects.len() {
                    results.push([self.subjects[i], self.predicates[i], self.objects[i]]);
                }
            }
        }

        let n = results.len();
        if n == 0 {
            Array2::zeros((0, 3)).into_pyarray(py)
        } else {
            let flat: Vec<i32> = results.into_iter().flatten().collect();
            Array2::from_shape_vec((n, 3), flat)
                .unwrap()
                .into_pyarray(py)
        }
    }
}

impl TripleStoreSoA {
    fn build_indexes(&mut self) {
        let n = self.subjects.len();
        let mut indices: Vec<usize> = (0..n).collect();

        indices.sort_unstable_by(|&a, &b| {
            self.subjects[a]
                .cmp(&self.subjects[b])
                .then(self.predicates[a].cmp(&self.predicates[b]))
                .then(self.objects[a].cmp(&self.objects[b]))
        });
        self.spo_index = indices.clone();

        indices.sort_unstable_by(|&a, &b| {
            self.predicates[a]
                .cmp(&self.predicates[b])
                .then(self.objects[a].cmp(&self.objects[b]))
                .then(self.subjects[a].cmp(&self.subjects[b]))
        });
        self.pos_index = indices.clone();

        indices.sort_unstable_by(|&a, &b| {
            self.objects[a]
                .cmp(&self.objects[b])
                .then(self.subjects[a].cmp(&self.subjects[b]))
                .then(self.predicates[a].cmp(&self.predicates[b]))
        });
        self.osp_index = indices;
    }
}

/// Probabilistic membership test for triple filtering.
#[pyclass]
pub struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: u32,
}

#[pymethods]
impl BloomFilter {
    #[new]
    #[pyo3(signature = (expected_items, fp_rate=0.01))]
    fn new(expected_items: usize, fp_rate: f64) -> Self {
        let num_bits =
            (-(expected_items as f64 * fp_rate.ln()) / (2.0_f64.ln().powi(2))).ceil() as usize;
        let num_bits = num_bits.max(64);
        let num_hashes = ((num_bits as f64 / expected_items as f64) * 2.0_f64.ln()).ceil() as u32;
        let num_hashes = num_hashes.clamp(1, 16);
        let words = num_bits.div_ceil(64);

        Self {
            bits: vec![0u64; words],
            num_bits,
            num_hashes,
        }
    }

    /// Add an item (as string repr) to the filter.
    fn add(&mut self, item: &Bound<'_, PyAny>) -> PyResult<()> {
        let repr = item.repr()?.to_string();
        let data = repr.as_bytes();
        for i in 0..self.num_hashes {
            let pos = self.hash_pos(data, i);
            let word = pos / 64;
            let bit = pos % 64;
            self.bits[word] |= 1u64 << bit;
        }
        Ok(())
    }

    /// Test if an item might be in the filter.
    fn might_contain(&self, item: &Bound<'_, PyAny>) -> PyResult<bool> {
        let repr = item.repr()?.to_string();
        let data = repr.as_bytes();
        for i in 0..self.num_hashes {
            let pos = self.hash_pos(data, i);
            let word = pos / 64;
            let bit = pos % 64;
            if self.bits[word] & (1u64 << bit) == 0 {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl BloomFilter {
    #[inline]
    fn hash_pos(&self, data: &[u8], seed: u32) -> usize {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&seed.to_be_bytes());
        hasher.update(data);
        let hash = hasher.finalize();
        let bytes = hash.as_bytes();
        let val = u64::from_be_bytes(bytes[..8].try_into().unwrap());
        (val as usize) % self.num_bits
    }
}

/// PCG-based negative sample generation (tail corruption).
///
/// Deterministic: same seed → same output across platforms.
#[pyfunction]
#[pyo3(signature = (heads, rels, tails, num_entities, num_negatives=1, seed=42))]
pub fn generate_negative_samples<'py>(
    py: Python<'py>,
    heads: PyReadonlyArray1<i64>,
    rels: PyReadonlyArray1<i64>,
    tails: PyReadonlyArray1<i64>,
    num_entities: i64,
    num_negatives: usize,
    seed: u64,
) -> Bound<'py, PyArray2<i64>> {
    let h = to_vec(&heads);
    let r = to_vec(&rels);
    let t = to_vec(&tails);

    let out = py.detach(|| {
        let n = h.len();
        let total = n * num_negatives;
        let mut out = Array2::<i64>::zeros((total, 3));

        for i in 0..n {
            let mut state: u64 = seed.wrapping_add((i as u64).wrapping_mul(193939));
            let base = i * num_negatives;
            for j in 0..num_negatives {
                state = 6364136223846793005u64
                    .wrapping_mul(state)
                    .wrapping_add(1442695040888963407u64);
                let mut rand_ent = ((state >> 32) % (num_entities as u64)) as i64;
                if rand_ent == t[i] {
                    rand_ent = (rand_ent + 1) % num_entities;
                }
                out[[base + j, 0]] = h[i];
                out[[base + j, 1]] = r[i];
                out[[base + j, 2]] = rand_ent;
            }
        }
        out
    });

    out.into_pyarray(py)
}

/// Batch negative sample generation with Rayon parallelism.
#[pyfunction]
#[pyo3(signature = (heads, rels, tails, num_negatives, num_entities, seed))]
pub fn batch_generate_negative_samples<'py>(
    py: Python<'py>,
    heads: PyReadonlyArray1<i64>,
    rels: PyReadonlyArray1<i64>,
    tails: PyReadonlyArray1<i64>,
    num_negatives: usize,
    num_entities: i64,
    seed: u64,
) -> Bound<'py, PyArray2<i64>> {
    let h = to_vec(&heads);
    let r = to_vec(&rels);
    let t = to_vec(&tails);

    let out = py.detach(|| {
        let n = h.len();
        let total = n * num_negatives;

        let results: Vec<[i64; 3]> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                let mut state: u64 = seed.wrapping_add((i as u64).wrapping_mul(193939));
                let mut local: Vec<[i64; 3]> = Vec::with_capacity(num_negatives);
                for _j in 0..num_negatives {
                    state = 6364136223846793005u64
                        .wrapping_mul(state)
                        .wrapping_add(1442695040888963407u64);
                    let mut rand_ent = ((state >> 32) % (num_entities as u64)) as i64;
                    if rand_ent == t[i] {
                        rand_ent = (rand_ent + 1) % num_entities;
                    }
                    local.push([h[i], r[i], rand_ent]);
                }
                local
            })
            .collect();

        let flat: Vec<i64> = results.into_iter().flatten().collect();
        Array2::from_shape_vec((total, 3), flat).unwrap()
    });

    out.into_pyarray(py)
}

/// Degree-weighted negative sampling.
#[pyfunction]
#[pyo3(signature = (heads, rels, tails, degree_weights, num_entities, num_negatives=1, seed=42))]
#[allow(clippy::too_many_arguments)]
pub fn degree_weighted_negative_sampling<'py>(
    py: Python<'py>,
    heads: PyReadonlyArray1<i64>,
    rels: PyReadonlyArray1<i64>,
    tails: PyReadonlyArray1<i64>,
    degree_weights: PyReadonlyArray1<f64>,
    num_entities: i64,
    num_negatives: usize,
    seed: u64,
) -> Bound<'py, PyArray2<i64>> {
    let h = to_vec(&heads);
    let r = to_vec(&rels);
    let t = to_vec(&tails);
    let weights = to_vec(&degree_weights);

    let out = py.detach(|| {
        let n = h.len();
        let total = n * num_negatives;

        let w_len = weights.len().min(num_entities as usize);
        let sum: f64 = weights[..w_len].iter().sum();
        let mut cumulative = Vec::with_capacity(w_len);
        let mut acc = 0.0;
        for &w in &weights[..w_len] {
            acc += w / sum;
            cumulative.push(acc);
        }

        let mut out = Array2::<i64>::zeros((total, 3));

        for i in 0..n {
            let mut state: u64 = seed.wrapping_add((i as u64).wrapping_mul(193939));
            let base = i * num_negatives;
            for j in 0..num_negatives {
                state = 6364136223846793005u64
                    .wrapping_mul(state)
                    .wrapping_add(1442695040888963407u64);
                let uniform = (state >> 11) as f64 / (1u64 << 53) as f64;
                let mut rand_ent = match cumulative.binary_search_by(|v| {
                    v.partial_cmp(&uniform).unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    Ok(pos) => pos as i64,
                    Err(pos) => pos.min(w_len - 1) as i64,
                };
                if rand_ent == t[i] {
                    rand_ent = (rand_ent + 1) % num_entities;
                }
                out[[base + j, 0]] = h[i];
                out[[base + j, 1]] = r[i];
                out[[base + j, 2]] = rand_ent;
            }
        }
        out
    });

    out.into_pyarray(py)
}

/// Generate EMU noise using Box-Muller transform.
#[pyfunction]
#[pyo3(signature = (embedding_dim, num_samples, perturbation_scale=0.1, seed=42))]
pub fn generate_emu_noise<'py>(
    py: Python<'py>,
    embedding_dim: usize,
    num_samples: usize,
    perturbation_scale: f64,
    seed: u64,
) -> Bound<'py, PyArray2<f32>> {
    use rand::SeedableRng;
    use rand_pcg::Pcg64;

    let out = py.detach(|| {
        let mut rng = Pcg64::seed_from_u64(seed);
        let mut out = Array2::<f32>::zeros((num_samples, embedding_dim));

        for i in 0..num_samples {
            for j in (0..embedding_dim).step_by(2) {
                let u1 = pcg_uniform(&mut rng);
                let u2 = pcg_uniform(&mut rng);
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                out[[i, j]] = (r * theta.cos() * perturbation_scale) as f32;
                if j + 1 < embedding_dim {
                    out[[i, j + 1]] = (r * theta.sin() * perturbation_scale) as f32;
                }
            }
        }
        out
    });

    out.into_pyarray(py)
}

#[inline]
fn pcg_uniform(rng: &mut rand_pcg::Pcg64) -> f64 {
    use rand::RngCore;
    let bits = rng.next_u64();
    (bits >> 11) as f64 / (1u64 << 53) as f64
}

/// Compute Expected Calibration Error.
#[pyfunction]
#[pyo3(signature = (probs, labels, n_bins=15))]
pub fn compute_ece<'py>(
    py: Python<'py>,
    probs: PyReadonlyArray1<f64>,
    labels: PyReadonlyArray1<f64>,
    n_bins: usize,
) -> f64 {
    let p = to_vec(&probs);
    let l = to_vec(&labels);

    py.detach(|| {
        let n = p.len();
        if n == 0 {
            return 0.0;
        }

        let bin_width = 1.0 / n_bins as f64;
        let mut bin_sums = vec![0.0f64; n_bins];
        let mut label_sums = vec![0.0f64; n_bins];
        let mut bin_counts = vec![0usize; n_bins];

        for i in 0..n {
            let clamped = p[i].clamp(0.0, 1.0);
            let b = ((clamped / bin_width) as usize).min(n_bins - 1);
            bin_sums[b] += clamped;
            label_sums[b] += l[i];
            bin_counts[b] += 1;
        }

        let mut ece = 0.0;
        for b in 0..n_bins {
            if bin_counts[b] > 0 {
                let acc = label_sums[b] / bin_counts[b] as f64;
                let conf = bin_sums[b] / bin_counts[b] as f64;
                ece += (bin_counts[b] as f64 / n as f64) * (acc - conf).abs();
            }
        }
        ece
    })
}

/// Sweep thresholds to find the best MCC.
///
/// Returns: (best_mcc, tp, tn, fp, fn, best_threshold)
#[pyfunction]
pub fn fast_mcc_sweep<'py>(
    py: Python<'py>,
    y_true: PyReadonlyArray1<i64>,
    y_score: PyReadonlyArray1<f64>,
    thresholds: PyReadonlyArray1<f64>,
) -> (f64, i64, i64, i64, i64, f64) {
    let yt = to_vec(&y_true);
    let ys = to_vec(&y_score);
    let ts = to_vec(&thresholds);

    py.detach(|| {
        let n = yt.len();

        let results: Vec<(f64, i64, i64, i64, i64, f64)> = ts
            .par_iter()
            .map(|&t| {
                let (mut tp, mut tn, mut fp, mut fn_) = (0i64, 0i64, 0i64, 0i64);
                for j in 0..n {
                    let pred = ys[j] > t;
                    let actual = yt[j] == 1;
                    match (pred, actual) {
                        (true, true) => tp += 1,
                        (false, false) => tn += 1,
                        (true, false) => fp += 1,
                        (false, true) => fn_ += 1,
                    }
                }
                let num = (tp as f64) * (tn as f64) - (fp as f64) * (fn_ as f64);
                let den =
                    ((tp + fp) as f64 * (tp + fn_) as f64 * (tn + fp) as f64 * (tn + fn_) as f64)
                        .sqrt();
                let mcc = if den > 0.0 { num / den } else { 0.0 };
                (mcc, tp, tn, fp, fn_, t)
            })
            .collect();

        results
            .into_iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0.0, 0, 0, 0, 0, 0.0))
    })
}

/// Fast ROC-AUC computation.
#[pyfunction]
pub fn fast_roc_auc_score<'py>(
    py: Python<'py>,
    y_true: PyReadonlyArray1<i64>,
    y_score: PyReadonlyArray1<f64>,
) -> f64 {
    let yt = to_vec(&y_true);
    let ys = to_vec(&y_score);

    py.detach(|| {
        let n = yt.len();
        if n == 0 {
            return 0.5;
        }

        let n_pos: i64 = yt.iter().sum();
        let n_neg = n as i64 - n_pos;
        if n_pos == 0 || n_neg == 0 {
            return 0.5;
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_unstable_by(|&a, &b| {
            ys[b]
                .partial_cmp(&ys[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut tps = 0i64;
        let mut fps = 0i64;
        let mut prev_tpr = 0.0f64;
        let mut prev_fpr = 0.0f64;
        let mut auc = 0.0f64;

        for &idx in indices.iter() {
            if yt[idx] == 1 {
                tps += 1;
            } else {
                fps += 1;
            }
            let tpr = tps as f64 / n_pos as f64;
            let fpr = fps as f64 / n_neg as f64;
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
            prev_tpr = tpr;
            prev_fpr = fpr;
        }

        auc.abs()
    })
}

/// Fast Matthews Correlation Coefficient.
#[pyfunction]
pub fn fast_matthews_corrcoef<'py>(
    py: Python<'py>,
    y_true: PyReadonlyArray1<i64>,
    y_pred: PyReadonlyArray1<i64>,
) -> f64 {
    let yt = to_vec(&y_true);
    let yp = to_vec(&y_pred);

    py.detach(|| {
        let (mut tp, mut tn, mut fp, mut fn_) = (0f64, 0f64, 0f64, 0f64);
        for i in 0..yt.len() {
            match (yp[i] == 1, yt[i] == 1) {
                (true, true) => tp += 1.0,
                (false, false) => tn += 1.0,
                (true, false) => fp += 1.0,
                (false, true) => fn_ += 1.0,
            }
        }

        let denom = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        if denom == 0.0 {
            0.0
        } else {
            (tp * tn - fp * fn_) / denom
        }
    })
}

/// Fast Average Precision computation.
#[pyfunction]
pub fn fast_average_precision_score<'py>(
    py: Python<'py>,
    y_true: PyReadonlyArray1<i64>,
    y_score: PyReadonlyArray1<f64>,
) -> f64 {
    let yt = to_vec(&y_true);
    let ys = to_vec(&y_score);

    py.detach(|| {
        let n_pos: i64 = yt.iter().sum();
        if n_pos == 0 {
            return 0.0;
        }

        let mut indices: Vec<usize> = (0..yt.len()).collect();
        indices.sort_unstable_by(|&a, &b| {
            ys[b]
                .partial_cmp(&ys[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut tps = 0i64;
        let mut ap = 0.0;
        for (rank, &idx) in indices.iter().enumerate() {
            if yt[idx] == 1 {
                tps += 1;
                ap += tps as f64 / (rank + 1) as f64;
            }
        }
        ap / n_pos as f64
    })
}

/// Find mask for unique (h, r, t) triples in sorted arrays.
#[pyfunction]
pub fn find_unique_triples_mask<'py>(
    py: Python<'py>,
    h: PyReadonlyArray1<i64>,
    r: PyReadonlyArray1<i64>,
    t: PyReadonlyArray1<i64>,
) -> Bound<'py, PyArray1<bool>> {
    let h = to_vec(&h);
    let r = to_vec(&r);
    let t = to_vec(&t);

    let mask = py.detach(|| {
        let n = h.len();
        let mut mask = vec![false; n];
        if n > 0 {
            mask[0] = true;
            for i in 1..n {
                mask[i] = h[i] != h[i - 1] || r[i] != r[i - 1] || t[i] != t[i - 1];
            }
        }
        mask
    });

    Array1::from_vec(mask).into_pyarray(py)
}

/// Return type for precision-recall curve: (precisions, recalls, thresholds).
type PrCurveResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// Fast Precision-Recall curve computation.
///
/// Returns (precisions, recalls, thresholds).
#[pyfunction]
pub fn fast_precision_recall_curve<'py>(
    py: Python<'py>,
    y_true: PyReadonlyArray1<i64>,
    y_score: PyReadonlyArray1<f64>,
) -> PrCurveResult<'py> {
    let yt = to_vec(&y_true);
    let ys = to_vec(&y_score);
    let n = yt.len();
    let n_pos: i64 = yt.iter().sum();

    if n_pos == 0 {
        return (
            Array1::from_vec(vec![1.0]).into_pyarray(py),
            Array1::from_vec(vec![0.0]).into_pyarray(py),
            Array1::from_vec(vec![]).into_pyarray(py),
        );
    }

    let (precisions, recalls, thresholds) = py.detach(|| {
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_unstable_by(|&a, &b| {
            ys[b]
                .partial_cmp(&ys[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut precisions = Vec::with_capacity(n + 1);
        let mut recalls = Vec::with_capacity(n + 1);
        let mut thresholds = Vec::with_capacity(n);

        let mut tps = 0i64;
        for (rank, &idx) in indices.iter().enumerate() {
            if yt[idx] == 1 {
                tps += 1;
            }
            let fps = (rank + 1) as i64 - tps;
            precisions.push(tps as f64 / (tps + fps) as f64);
            recalls.push(tps as f64 / n_pos as f64);
            thresholds.push(ys[idx]);
        }

        precisions.push(1.0);
        recalls.push(0.0);

        (precisions, recalls, thresholds)
    });

    (
        Array1::from_vec(precisions).into_pyarray(py),
        Array1::from_vec(recalls).into_pyarray(py),
        Array1::from_vec(thresholds).into_pyarray(py),
    )
}
/// Compute Jaccard similarity between two sorted unique integer arrays.
///
/// Uses a merge-based O(n+m) algorithm on pre-sorted inputs.
#[pyfunction]
pub fn sorted_jaccard_similarity<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<i64>,
    b: PyReadonlyArray1<i64>,
) -> f64 {
    let a_slice = to_vec(&a);
    let b_slice = to_vec(&b);

    py.detach(|| {
        let n_a = a_slice.len();
        let n_b = b_slice.len();

        if n_a == 0 || n_b == 0 {
            return 0.0;
        }

        let mut intersection = 0usize;
        let mut i = 0usize;
        let mut j = 0usize;

        while i < n_a && j < n_b {
            let va = a_slice[i];
            let vb = b_slice[j];
            if va == vb {
                intersection += 1;
                i += 1;
                j += 1;
            } else if va < vb {
                i += 1;
            } else {
                j += 1;
            }
        }

        let union_size = n_a + n_b - intersection;
        if union_size == 0 {
            return 0.0;
        }

        intersection as f64 / union_size as f64
    })
}

/// Convert a string to sorted unique BLAKE3 hashes of its character n-grams.
///
/// ASCII fast path uses zero-alloc byte windows; non-ASCII uses a reusable buffer.
#[pyfunction]
#[pyo3(signature = (s, n=3))]
pub fn string_to_ngram_hashes<'py>(
    py: Python<'py>,
    s: &str,
    n: usize,
) -> Bound<'py, PyArray1<i64>> {
    let lower = s.to_lowercase();

    if lower.len() < n {
        let hash = blake3::hash(lower.as_bytes());
        let bytes = hash.as_bytes();
        let val = i64::from_be_bytes(bytes[..8].try_into().unwrap());
        return Array1::from_vec(vec![val]).into_pyarray(py);
    }

    let mut hashes: Vec<i64> = if lower.is_ascii() {
        let bytes = lower.as_bytes();
        let mut h = Vec::with_capacity(bytes.len() - n + 1);
        for window in bytes.windows(n) {
            let hash = blake3::hash(window);
            let hb = hash.as_bytes();
            h.push(i64::from_be_bytes(hb[..8].try_into().unwrap()));
        }
        h
    } else {
        let chars: Vec<char> = lower.chars().collect();
        if chars.len() < n {
            let hash = blake3::hash(lower.as_bytes());
            let bytes = hash.as_bytes();
            let val = i64::from_be_bytes(bytes[..8].try_into().unwrap());
            return Array1::from_vec(vec![val]).into_pyarray(py);
        }
        let mut h = Vec::with_capacity(chars.len() - n + 1);
        let mut buf = String::with_capacity(n * 4);
        for i in 0..=(chars.len() - n) {
            buf.clear();
            for &c in &chars[i..i + n] {
                buf.push(c);
            }
            let hash = blake3::hash(buf.as_bytes());
            let hb = hash.as_bytes();
            h.push(i64::from_be_bytes(hb[..8].try_into().unwrap()));
        }
        h
    };

    hashes.sort_unstable();
    hashes.dedup();

    Array1::from_vec(hashes).into_pyarray(py)
}
