"""
Jaccard similarity kernels for entity resolution.

Delegates to Rust-accelerated pff_rust (replaces Numba).
"""

from __future__ import annotations

from pff_rust import sorted_jaccard_similarity, string_to_ngram_hashes

__all__ = ["sorted_jaccard_similarity", "string_to_ngram_hashes"]
