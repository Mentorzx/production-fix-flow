use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::collections::HashSet;
use xxhash_rust::xxh3::xxh3_64;

use _pff_rust::shared::hash::{hash_i64_slice_for_bench, stable_hash_bytes_for_bench};
use _pff_rust::shared::kernels::{
    sorted_jaccard_similarity_for_bench, string_to_ngram_hashes_for_bench,
};

fn xxh3_ngram_hashes(s: &str, n: usize) -> Vec<i64> {
    let lower = s.to_lowercase();
    if lower.len() < n {
        return vec![xxh3_64(lower.as_bytes()) as i64];
    }

    let mut hashes: Vec<i64> = if lower.is_ascii() {
        let bytes = lower.as_bytes();
        let mut h = Vec::with_capacity(bytes.len() - n + 1);
        for window in bytes.windows(n) {
            h.push(xxh3_64(window) as i64);
        }
        h
    } else {
        let mut h = Vec::with_capacity(lower.chars().count().saturating_sub(n) + 1);
        let mut buffer: Vec<u8> = Vec::new();
        let chars: Vec<char> = lower.chars().collect();
        for window in chars.windows(n) {
            buffer.clear();
            for ch in window {
                let mut temp = [0u8; 4];
                buffer.extend_from_slice(ch.encode_utf8(&mut temp).as_bytes());
            }
            h.push(xxh3_64(&buffer) as i64);
        }
        h
    };

    hashes.sort_unstable();
    hashes.dedup();
    hashes
}

fn report_ngram_collision_estimate(sample: &[&str], n: usize) {
    let mut ngram_set = HashSet::new();
    let mut xxh3_set = HashSet::new();

    for value in sample {
        let lower = value.to_lowercase();
        if lower.len() < n {
            ngram_set.insert(lower.clone());
            xxh3_set.insert(xxh3_64(lower.as_bytes()));
            continue;
        }
        if lower.is_ascii() {
            let bytes = lower.as_bytes();
            for window in bytes.windows(n) {
                ngram_set.insert(String::from_utf8_lossy(window).to_string());
                xxh3_set.insert(xxh3_64(window));
            }
        } else {
            let chars: Vec<char> = lower.chars().collect();
            let mut buffer: Vec<u8> = Vec::new();
            for window in chars.windows(n) {
                buffer.clear();
                for ch in window {
                    let mut temp = [0u8; 4];
                    buffer.extend_from_slice(ch.encode_utf8(&mut temp).as_bytes());
                }
                ngram_set.insert(String::from_utf8_lossy(&buffer).to_string());
                xxh3_set.insert(xxh3_64(&buffer));
            }
        }
    }

    if !ngram_set.is_empty() {
        println!(
            "XXH3 collision estimate: unique_ngrams={}, unique_hashes={}, collision_rate={:.6}%",
            ngram_set.len(),
            xxh3_set.len(),
            100.0 * (1.0 - (xxh3_set.len() as f64 / ngram_set.len() as f64))
        );
    }
}

fn bench_string_to_ngram_hashes(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_to_ngram_hashes");
    let ascii = "PerformanceHashingExampleString";
    let unicode = "PerfÓtimoΔTeste性能";
    report_ngram_collision_estimate(&[ascii, unicode, "CollisionTestSample"], 3);

    for (label, s) in [("ascii", ascii), ("unicode", unicode)] {
        group.throughput(Throughput::Bytes(s.len() as u64));
        group.bench_function(BenchmarkId::new("ngram_3", label), |b| {
            b.iter(|| {
                let _ = string_to_ngram_hashes_for_bench(black_box(s), 3);
            })
        });
        group.bench_function(BenchmarkId::new("ngram_3_xxh3", label), |b| {
            b.iter(|| {
                let _ = xxh3_ngram_hashes(black_box(s), 3);
            })
        });
    }
    group.finish();
}

fn bench_sorted_jaccard_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorted_jaccard_similarity");

    let a_values: Vec<i64> = (0..4096).step_by(3).map(|v| v as i64).collect();
    let b_values: Vec<i64> = (0..4096).step_by(4).map(|v| v as i64).collect();

    group.throughput(Throughput::Elements(
        (a_values.len() + b_values.len()) as u64,
    ));
    group.bench_function("merge", |bch| {
        bch.iter(|| {
            let _ = sorted_jaccard_similarity_for_bench(black_box(&a_values), black_box(&b_values));
        })
    });
    group.finish();
}

fn bench_hashes(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashes");

    let s = b"stable-hash-input-0123456789";
    let tuple = [1i64, 2i64, 3i64, 4i64, 5i64];

    group.bench_function("stable_hash_bytes", |b| {
        b.iter(|| {
            let _ = stable_hash_bytes_for_bench(black_box(s), 16);
        })
    });

    group.bench_function("hash_i64_slice", |b| {
        b.iter(|| {
            let _ = hash_i64_slice_for_bench(black_box(&tuple), 16);
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_string_to_ngram_hashes,
    bench_sorted_jaccard_similarity,
    bench_hashes
);
criterion_main!(benches);
