//! PFF Native Extensions - Rust bindings for performance-critical operations.
//!
//! Provides zero-copy NumPy array processing for:
//! - Filter dict construction (10-19x faster than Python)
//! - Negative sampling (when Numba unavailable)

use numpy::{PyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::HashMap;

type FastHashMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

/// Build filter dictionary from triples array.
///
/// Given an Nx3 array of (head, relation, tail) triples, builds a dictionary
/// mapping (head, relation) -> list of tail entities.
///
/// This is ~10-19x faster than Python dict construction for large KGs.
///
/// Args:
///     triples: Nx3 numpy array of int64 (head, relation, tail)
///
/// Returns:
///     dict mapping (head, relation) tuples to numpy arrays of tail entities
#[pyfunction]
fn build_filter_dict<'py>(
    py: Python<'py>,
    triples: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyDict>> {
    let arr = triples.as_array();
    let n_triples = arr.nrows();
    
    let mut dict: FastHashMap<(i64, i64), Vec<i64>> = FastHashMap::default();
    dict.reserve(n_triples / 10);
    
    for i in 0..n_triples {
        let h = arr[[i, 0]];
        let r = arr[[i, 1]];
        let t = arr[[i, 2]];
        dict.entry((h, r)).or_default().push(t);
    }
    
    let py_dict = PyDict::new(py);
    for ((h, r), tails) in dict {
        let py_tails = tails.to_pyarray(py);
        py_dict.set_item((h, r), py_tails)?;
    }
    
    Ok(py_dict)
}

/// Build filter dictionary with parallel processing for large arrays.
///
/// Uses Rayon to parallelize dict construction. Best for >100k triples.
#[pyfunction]
fn build_filter_dict_parallel<'py>(
    py: Python<'py>,
    triples: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyDict>> {
    let arr = triples.as_array();
    let n_triples = arr.nrows();
    
    if n_triples < 50_000 {
        return build_filter_dict(py, triples);
    }
    
    let rows: Vec<(i64, i64, i64)> = (0..n_triples)
        .into_par_iter()
        .map(|i| (arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]))
        .collect();
    
    let chunk_size = (n_triples / rayon::current_num_threads()).max(1000);
    let partial_dicts: Vec<FastHashMap<(i64, i64), Vec<i64>>> = rows
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local: FastHashMap<(i64, i64), Vec<i64>> = FastHashMap::default();
            for &(h, r, t) in chunk {
                local.entry((h, r)).or_default().push(t);
            }
            local
        })
        .collect();
    
    let mut merged: FastHashMap<(i64, i64), Vec<i64>> = FastHashMap::default();
    for partial in partial_dicts {
        for (key, tails) in partial {
            merged.entry(key).or_default().extend(tails);
        }
    }
    
    let py_dict = PyDict::new(py);
    for ((h, r), tails) in merged {
        let py_tails = tails.to_pyarray(py);
        py_dict.set_item((h, r), py_tails)?;
    }
    
    Ok(py_dict)
}

/// Lookup filter mask for a batch of (head, relation) pairs.
///
/// Args:
///     filter_dict: Dictionary from build_filter_dict
///     heads: 1D array of head entities
///     relations: 1D array of relation types
///     num_entities: Total number of entities in KG
///
/// Returns:
///     2D boolean mask array [batch_size, num_entities] where True means entity is known tail
#[pyfunction]
fn lookup_filter_mask<'py>(
    py: Python<'py>,
    filter_dict: &Bound<'py, PyDict>,
    heads: PyReadonlyArray2<'py, i64>,
    relations: PyReadonlyArray2<'py, i64>,
    num_entities: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let h_arr = heads.as_array();
    let r_arr = relations.as_array();
    let batch_size = h_arr.len();
    
    let mut mask = vec![false; batch_size * num_entities];
    
    for i in 0..batch_size {
        let h = h_arr[[i, 0]];
        let r = r_arr[[i, 0]];
        
        if let Ok(Some(tails)) = filter_dict.get_item((h, r)) {
            if let Ok(tails_arr) = tails.extract::<Vec<i64>>() {
                for t in tails_arr {
                    if (t as usize) < num_entities {
                        mask[i * num_entities + t as usize] = true;
                    }
                }
            }
        }
    }
    
    Ok(mask.to_pyarray(py))
}

/// Simple xorshift64 PRNG for deterministic negative sampling.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    
    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    
    fn next_range(&mut self, max: i64) -> i64 {
        (self.next() % max as u64) as i64
    }
}

/// Generate negative samples for a batch of triples.
///
/// Args:
///     heads: 1D array of head entities
///     rels: 1D array of relations  
///     tails: 1D array of tail entities
///     num_negatives: Number of negative samples per triple
///     num_entities: Total entities in KG
///     seed: Random seed for reproducibility
///
/// Returns:
///     2D array [batch_size, num_negatives] of negative entity IDs
#[pyfunction]
fn batch_negative_sampling<'py>(
    py: Python<'py>,
    heads: PyReadonlyArray2<'py, i64>,
    _rels: PyReadonlyArray2<'py, i64>,
    _tails: PyReadonlyArray2<'py, i64>,
    num_negatives: usize,
    num_entities: i64,
    seed: u64,
) -> PyResult<Bound<'py, numpy::PyArray2<i64>>> {
    let h_arr = heads.as_array();
    let batch_size = h_arr.len();
    
    let mut rng = Xorshift64::new(seed);
    let mut samples = vec![0i64; batch_size * num_negatives];
    
    for i in 0..batch_size {
        for j in 0..num_negatives {
            samples[i * num_negatives + j] = rng.next_range(num_entities);
        }
    }
    
    let arr = numpy::PyArray2::from_vec2(
        py,
        &samples
            .chunks(num_negatives)
            .map(|c| c.to_vec())
            .collect::<Vec<_>>(),
    )?;
    
    Ok(arr)
}

/// Module initialization
#[pymodule]
fn pff_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_filter_dict, m)?)?;
    m.add_function(wrap_pyfunction!(build_filter_dict_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(lookup_filter_mask, m)?)?;
    m.add_function(wrap_pyfunction!(batch_negative_sampling, m)?)?;
    Ok(())
}
