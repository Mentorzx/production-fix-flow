import time
import json
import gc
import os
import statistics
import mmap
import tempfile
from pathlib import Path

# Imports condicionais seguros
try:
    import orjson
except ImportError:
    orjson = None
try:
    import msgspec
except ImportError:
    msgspec = None


def generate_data(n_rows=500_000):
    """Gera ~100MB de JSON complexo."""
    return [
        {
            "id": i,
            "guid": f"guid-{i}-{i * 2}",
            "tags": ["tag1", "tag2", "tag3"] * 5,
            "meta": {"a": 1, "b": 2.5, "nested": [1, 2, 3] * 10},
        }
        for i in range(n_rows)
    ]


def benchmark_strategy(name, func, path, n_iter=10):
    """Executa benchmark com warmup e coleta de GC."""
    # Warmup
    try:
        func(path)
    except Exception as e:
        return f"Falha: {e}"

    times = []
    for _ in range(n_iter):
        gc.collect()
        start = time.perf_counter()
        func(path)
        times.append(time.perf_counter() - start)

    return statistics.median(times)


def run_suite():
    preload = os.environ.get("LD_PRELOAD", "")
    mode = "Mimalloc" if "libmimalloc" in preload else "Standard Allocator"
    print(f"--- Benchmark Ambiente: {mode} ---")

    # 1. Setup Dados
    data = generate_data()
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".json") as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(orjson.dumps(data) if orjson else json.dumps(data).encode())

    size_mb = tmp_path.stat().st_size / (1024 * 1024)
    print(f"Dataset: {size_mb:.2f} MB")

    strategies = []

    # Estratégia 1: Msgspec (Atual)
    if msgspec:

        def run_msgspec(p):
            with open(p, "rb") as f:
                return msgspec.json.decode(f.read())

        strategies.append(("msgspec_current", run_msgspec))

    # Estratégia 2: Orjson (Zero-Copy mmap)
    if orjson:

        def run_orjson_mmap(p):
            with open(p, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    return orjson.loads(memoryview(mm))

        strategies.append(("orjson_mmap", run_orjson_mmap))

    # Estratégia 3: Orjson (Bytes - load direto)
    if orjson:

        def run_orjson_bytes(p):
            with open(p, "rb") as f:
                return orjson.loads(f.read())

        strategies.append(("orjson_bytes", run_orjson_bytes))

    # Execução
    results = {}
    for name, func in strategies:
        med = benchmark_strategy(name, func, tmp_path)
        print(f"{name:<20}: {med:.4f}s")
        results[name] = med

    # Cleanup
    os.unlink(tmp_path)
    return results


if __name__ == "__main__":
    run_suite()
