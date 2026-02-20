from __future__ import annotations

import argparse
import random
import string
import time

import numpy as np

from pff_rust import hash_tuple, sorted_jaccard_similarity, stable_hash, string_to_ngram_hashes


def _make_strings(count: int, length: int, seed: int, unicode: bool = False) -> list[str]:
    rng = random.Random(seed)
    alphabet = string.ascii_letters
    extras = "ÁÉÍÓÚÜçΔβλ性能" if unicode else ""
    symbols = alphabet + extras
    return ["".join(rng.choice(symbols) for _ in range(length)) for _ in range(count)]


def _make_sorted_ints(size: int, step: int) -> list[int]:
    return list(range(0, size * step, step))


def bench_string_to_ngram(strings: list[str], n: int, iters: int) -> float:
    start = time.perf_counter()
    for i in range(iters):
        string_to_ngram_hashes(strings[i % len(strings)], n)
    return time.perf_counter() - start


def bench_sorted_jaccard(a: np.ndarray, b: np.ndarray, iters: int) -> float:
    start = time.perf_counter()
    acc = 0.0
    for _ in range(iters):
        acc += sorted_jaccard_similarity(a, b)
    if acc < 0:
        raise RuntimeError("unexpected")
    return time.perf_counter() - start


def bench_stable_hash(strings: list[str], iters: int) -> float:
    start = time.perf_counter()
    for i in range(iters):
        stable_hash(strings[i % len(strings)], truncate=16)
    return time.perf_counter() - start


def bench_hash_tuple(iters: int) -> float:
    start = time.perf_counter()
    for i in range(iters):
        hash_tuple([i, i + 1, i + 2, i + 3], truncate=16)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--ngram-iters", type=int, default=2000)
    parser.add_argument("--jaccard-iters", type=int, default=1000)
    args = parser.parse_args()

    ascii_strings = _make_strings(200, 24, args.seed, unicode=False)
    unicode_strings = _make_strings(200, 24, args.seed + 1, unicode=True)

    a = np.asarray(_make_sorted_ints(2048, 3), dtype=np.int64)
    b = np.asarray(_make_sorted_ints(2048, 4), dtype=np.int64)

    ascii_ngram = bench_string_to_ngram(ascii_strings, 3, args.ngram_iters)
    unicode_ngram = bench_string_to_ngram(unicode_strings, 3, args.ngram_iters)
    jaccard = bench_sorted_jaccard(a, b, args.jaccard_iters)
    stable = bench_stable_hash(ascii_strings, args.iters)
    tuple_hash = bench_hash_tuple(args.iters)

    print("Benchmark results (seconds):")
    print(f"  string_to_ngram_hashes/ascii:   {ascii_ngram:.6f} / {args.ngram_iters}")
    print(f"  string_to_ngram_hashes/unicode: {unicode_ngram:.6f} / {args.ngram_iters}")
    print(f"  sorted_jaccard_similarity:      {jaccard:.6f} / {args.jaccard_iters}")
    print(f"  stable_hash:                    {stable:.6f} / {args.iters}")
    print(f"  hash_tuple:                     {tuple_hash:.6f} / {args.iters}")


if __name__ == "__main__":
    main()
