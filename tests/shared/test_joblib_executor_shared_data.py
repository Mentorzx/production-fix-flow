import numpy as np

from pff.shared.acceleration.concurrency import JoblibExecutor


def _select(shared: np.ndarray, idx: int) -> int:
    return int(shared[idx])


def test_joblib_executor_shared_data_memmap() -> None:
    shared = np.arange(10, dtype=np.int64)
    executor = JoblibExecutor(n_jobs=1, mmap_threshold=1)
    try:
        results = executor.map(
            _select,
            [(i,) for i in range(len(shared))],
            shared_data=shared,
        )
    finally:
        executor.shutdown()

    assert results == [int(x) for x in shared]
