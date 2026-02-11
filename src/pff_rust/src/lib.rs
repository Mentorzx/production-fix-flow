mod builder;
mod rules;
mod shared;

use pyo3::prelude::*;

#[pymodule]
fn _pff_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(shared::hash::stable_hash, m)?)?;
    m.add_function(wrap_pyfunction!(shared::hash::hash_tuple, m)?)?;
    m.add_function(wrap_pyfunction!(shared::hash::hash_64bit, m)?)?;
    m.add_function(wrap_pyfunction!(shared::hash::hash_bytes, m)?)?;

    m.add_class::<shared::kernels::VocabularyEncoder>()?;
    m.add_class::<shared::kernels::TripleStoreSoA>()?;
    m.add_class::<shared::kernels::BloomFilter>()?;
    m.add_function(wrap_pyfunction!(
        shared::kernels::generate_negative_samples,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        shared::kernels::batch_generate_negative_samples,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        shared::kernels::degree_weighted_negative_sampling,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(shared::kernels::generate_emu_noise, m)?)?;
    m.add_function(wrap_pyfunction!(shared::kernels::compute_ece, m)?)?;
    m.add_function(wrap_pyfunction!(shared::kernels::fast_mcc_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(shared::kernels::fast_roc_auc_score, m)?)?;
    m.add_function(wrap_pyfunction!(
        shared::kernels::fast_matthews_corrcoef,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        shared::kernels::fast_average_precision_score,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        shared::kernels::fast_precision_recall_curve,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        shared::kernels::find_unique_triples_mask,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        shared::kernels::sorted_jaccard_similarity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        shared::kernels::string_to_ngram_hashes,
        m
    )?)?;

    m.add_class::<rules::RuleEncoder>()?;
    m.add_function(wrap_pyfunction!(rules::check_violations_batch, m)?)?;

    m.add_function(wrap_pyfunction!(builder::convert_to_triples, m)?)?;

    Ok(())
}
