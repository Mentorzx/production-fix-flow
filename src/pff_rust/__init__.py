"""PFF Rust acceleration package."""

from ._pff_rust import (
    BloomFilter,
    RuleEncoder,
    TripleStoreSoA,
    VocabularyEncoder,
    batch_generate_negative_samples,
    check_violations_batch,
    compute_ece,
    convert_to_triples,
    degree_weighted_negative_sampling,
    fast_average_precision_score,
    fast_matthews_corrcoef,
    fast_mcc_sweep,
    fast_precision_recall_curve,
    fast_roc_auc_score,
    find_unique_triples_mask,
    generate_emu_noise,
    generate_negative_samples,
    hash_64bit,
    hash_bytes,
    hash_tuple,
    sorted_jaccard_similarity,
    stable_hash,
    string_to_ngram_hashes,
)

try:
    from ._pff_rust import fast_spearman_corr
except ImportError:  # pragma: no cover - backward compatibility with stale local build
    fast_spearman_corr = None

__all__ = [
    "BloomFilter",
    "RuleEncoder",
    "TripleStoreSoA",
    "VocabularyEncoder",
    "batch_generate_negative_samples",
    "check_violations_batch",
    "compute_ece",
    "convert_to_triples",
    "degree_weighted_negative_sampling",
    "fast_average_precision_score",
    "fast_matthews_corrcoef",
    "fast_mcc_sweep",
    "fast_precision_recall_curve",
    "fast_spearman_corr",
    "fast_roc_auc_score",
    "find_unique_triples_mask",
    "generate_emu_noise",
    "generate_negative_samples",
    "hash_64bit",
    "hash_bytes",
    "hash_tuple",
    "sorted_jaccard_similarity",
    "stable_hash",
    "string_to_ngram_hashes",
]
