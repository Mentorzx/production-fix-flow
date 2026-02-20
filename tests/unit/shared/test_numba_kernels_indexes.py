"""Provide module-level functionality for the PFF codebase.



Notes:

    File: tests/unit/shared/test_numba_kernels_indexes.py

"""

import numpy as np

from pff_rust import TripleStoreSoA


def test_triple_store_spo_index_lexsort_order() -> None:
    """Execute test triple store spo index lexsort order.



    Notes:

        Keep behavior deterministic and free of hidden side effects.

    """

    triples = np.array(
        [
            [5, 1, 0],
            [5, 0, 1_000_000],
            [4, 2, 3],
        ],
        dtype=np.int32,
    )
    store = TripleStoreSoA()
    store.load_from_arrays(
        triples[:, 0].copy(),
        triples[:, 1].copy(),
        triples[:, 2].copy(),
    )

    expected = np.lexsort((triples[:, 2], triples[:, 1], triples[:, 0])).astype(np.int32)

    spo = store.get_spo_index()
    assert np.array_equal(spo, expected)
