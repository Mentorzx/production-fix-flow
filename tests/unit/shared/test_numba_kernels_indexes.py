import numpy as np

from pff.shared.acceleration.numba_kernels import TripleStoreSoA


def test_triple_store_spo_index_lexsort_order() -> None:
    triples = np.array(
        [
            [5, 1, 0],
            [5, 0, 1_000_000],
            [4, 2, 3],
        ],
        dtype=np.int32,
    )
    store = TripleStoreSoA(triples.shape[0])
    store.load_from_triples(triples)

    expected = np.lexsort((triples[:, 2], triples[:, 1], triples[:, 0])).astype(np.int32)

    assert np.array_equal(store.spo_index, expected)
