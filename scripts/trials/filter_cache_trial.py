import numpy as np
import time
import torch
import polars as pl
from pathlib import Path
from pff.shared.core.file_manager import FileManager
from pff.shared.hash import stable_hash


def run_trial():
    fm = FileManager()
    triples = np.random.randint(0, 100000, (500000, 3), dtype=np.int32)

    # 1. Generate Cache Key
    # Section 4.10: Using stable_hash from shared
    t_hash = stable_hash(triples)
    cache_path = Path(f".cache/filter_{t_hash}.arrow")
    fm.ensure_dir(cache_path.parent)

    # 2. Simulate Save (Prematerialization)
    if not cache_path.exists():
        # Pack everything into a single Polars DataFrame for Arrow storage
        # packed_keys, hr_unique, tails_sorted, hr_ranges
        # For simplicity in trial, just saving the biggest one: tails_sorted
        df = pl.DataFrame({"tails": triples[:, 2].astype(np.int64)})
        fm.save(df, cache_path, compression="uncompressed")

    # 3. Measure Load (TRIAL)
    start = time.perf_counter()
    bundle = fm.read(cache_path, memory_map=True)
    # Access Polars dataframe from bundle
    loaded_df = bundle.to_polars()
    torch.from_numpy(loaded_df["tails"].to_numpy())
    t_trial = (time.perf_counter() - start) * 1000

    print(f"FILTER_MMAP_LOAD_MS: {t_trial:.2f}")

    if cache_path.exists():
        cache_path.unlink()


if __name__ == "__main__":
    run_trial()
