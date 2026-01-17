//! HPO-specific benchmarks for DSLFM-KGC acceleration.
//!
//! Benchmarks negative sampling and filter mask construction to establish
//! Rust performance ceiling vs Python/Numba implementations.
//!
//! Run: cargo run --release --bin bench_hpo

use std::{
    collections::HashMap,
    hint::black_box,
    time::Instant,
};

use mimalloc::MiMalloc;
use rayon::prelude::*;
use serde::Serialize;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// Use faster hasher for integer keys
type FastHashMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

#[derive(Serialize)]
struct BenchResult {
    name: String,
    batch_size: usize,
    num_negatives: usize,
    num_entities: i64,
    runs: usize,
    median_time_us: f64,
    mean_time_us: f64,
    min_time_us: f64,
    max_time_us: f64,
    throughput_samples_per_sec: f64,
}

#[derive(Serialize)]
struct FilterMaskResult {
    name: String,
    num_triples: usize,
    num_entities: usize,
    runs: usize,
    build_median_us: f64,
    lookup_median_us: f64,
    lookup_batch_size: usize,
}

#[derive(Serialize)]
struct HpoBenchOutput {
    negative_sampling: Vec<BenchResult>,
    filter_mask: Vec<FilterMaskResult>,
    system_info: SystemInfo,
}

#[derive(Serialize)]
struct SystemInfo {
    num_cpus: usize,
    rayon_threads: usize,
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Simple xorshift64 RNG for deterministic, fast random generation.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    
    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    
    fn next_range(&mut self, max: i64) -> i64 {
        (self.next() % max as u64) as i64
    }
}

/// Generate batch of negative samples (sequential).
fn batch_negative_sampling_seq(
    batch_size: usize,
    num_negatives: usize,
    num_entities: i64,
    seed: u64,
) -> Vec<i64> {
    let mut rng = Xorshift64::new(seed);
    let total = batch_size * num_negatives;
    let mut result = Vec::with_capacity(total);
    
    for _ in 0..total {
        result.push(rng.next_range(num_entities));
    }
    
    result
}

/// Generate batch of negative samples (parallel with Rayon).
fn batch_negative_sampling_par(
    batch_size: usize,
    num_negatives: usize,
    num_entities: i64,
    seed: u64,
) -> Vec<i64> {
    (0..batch_size)
        .into_par_iter()
        .flat_map(|i| {
            let mut rng = Xorshift64::new(seed.wrapping_add(i as u64));
            (0..num_negatives)
                .map(move |_| rng.next_range(num_entities))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Build filter mask dictionary (head, relation) -> [tails].
fn build_filter_dict(
    heads: &[i64],
    relations: &[i64],
    tails: &[i64],
) -> FastHashMap<(i64, i64), Vec<i64>> {
    let mut dict: FastHashMap<(i64, i64), Vec<i64>> = FastHashMap::default();
    
    for i in 0..heads.len() {
        let key = (heads[i], relations[i]);
        dict.entry(key)
            .or_insert_with(Vec::new)
            .push(tails[i]);
    }
    
    dict
}

/// Lookup filter mask for a batch of (h, r) pairs.
fn lookup_filter_mask(
    dict: &FastHashMap<(i64, i64), Vec<i64>>,
    heads: &[i64],
    relations: &[i64],
    num_entities: usize,
) -> Vec<Vec<bool>> {
    heads.iter().zip(relations.iter())
        .map(|(&h, &r)| {
            let mut mask = vec![false; num_entities];
            if let Some(tails) = dict.get(&(h, r)) {
                for &t in tails {
                    mask[t as usize] = true;
                }
            }
            mask
        })
        .collect()
}

/// Parallel lookup filter mask.
fn lookup_filter_mask_par(
    dict: &FastHashMap<(i64, i64), Vec<i64>>,
    heads: &[i64],
    relations: &[i64],
    num_entities: usize,
) -> Vec<Vec<bool>> {
    heads.par_iter().zip(relations.par_iter())
        .map(|(&h, &r)| {
            let mut mask = vec![false; num_entities];
            if let Some(tails) = dict.get(&(h, r)) {
                for &t in tails {
                    mask[t as usize] = true;
                }
            }
            mask
        })
        .collect()
}

fn bench_negative_sampling() -> Vec<BenchResult> {
    let configs = [
        (256, 128, 10_000i64),    // Small KG
        (512, 256, 50_000i64),    // Medium KG
        (1024, 512, 100_000i64),  // Large KG
        (2048, 1024, 500_000i64), // Very large KG
    ];
    
    let runs = 100;
    let seed = 42u64;
    let mut results = Vec::new();
    
    for &(batch_size, num_negatives, num_entities) in &configs {
        // Sequential benchmark
        let mut times_seq: Vec<f64> = Vec::with_capacity(runs);
        for _ in 0..runs {
            let start = Instant::now();
            let samples = batch_negative_sampling_seq(batch_size, num_negatives, num_entities, seed);
            black_box(&samples);
            times_seq.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        }
        
        let total_samples = (batch_size * num_negatives) as f64;
        let median_us = median(&mut times_seq);
        
        results.push(BenchResult {
            name: "rust_neg_sampling_seq".into(),
            batch_size,
            num_negatives,
            num_entities,
            runs,
            median_time_us: median_us,
            mean_time_us: times_seq.iter().sum::<f64>() / runs as f64,
            min_time_us: times_seq.iter().cloned().fold(f64::INFINITY, f64::min),
            max_time_us: times_seq.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            throughput_samples_per_sec: total_samples / (median_us / 1_000_000.0),
        });
        
        // Parallel benchmark
        let mut times_par: Vec<f64> = Vec::with_capacity(runs);
        for _ in 0..runs {
            let start = Instant::now();
            let samples = batch_negative_sampling_par(batch_size, num_negatives, num_entities, seed);
            black_box(&samples);
            times_par.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        }
        
        let median_us = median(&mut times_par);
        
        results.push(BenchResult {
            name: "rust_neg_sampling_par".into(),
            batch_size,
            num_negatives,
            num_entities,
            runs,
            median_time_us: median_us,
            mean_time_us: times_par.iter().sum::<f64>() / runs as f64,
            min_time_us: times_par.iter().cloned().fold(f64::INFINITY, f64::min),
            max_time_us: times_par.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            throughput_samples_per_sec: total_samples / (median_us / 1_000_000.0),
        });
    }
    
    results
}

fn bench_filter_mask() -> Vec<FilterMaskResult> {
    let configs = [
        (10_000, 1_000),    // Small
        (50_000, 5_000),    // Medium
        (200_000, 20_000),  // Large
    ];
    
    let runs = 50;
    let seed = 42u64;
    let num_relations = 100;
    let lookup_batch = 256;
    let mut results = Vec::new();
    
    for &(num_triples, num_entities) in &configs {
        // Generate synthetic triples
        let mut rng = Xorshift64::new(seed);
        let heads: Vec<i64> = (0..num_triples).map(|_| rng.next_range(num_entities as i64)).collect();
        let relations: Vec<i64> = (0..num_triples).map(|_| rng.next_range(num_relations)).collect();
        let tails: Vec<i64> = (0..num_triples).map(|_| rng.next_range(num_entities as i64)).collect();
        
        // Benchmark build
        let mut build_times: Vec<f64> = Vec::with_capacity(runs);
        let mut dict = FastHashMap::default();
        for _ in 0..runs {
            let start = Instant::now();
            dict = build_filter_dict(&heads, &relations, &tails);
            black_box(&dict);
            build_times.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        }
        
        // Prepare lookup batch
        let lookup_heads: Vec<i64> = (0..lookup_batch).map(|_| rng.next_range(num_entities as i64)).collect();
        let lookup_rels: Vec<i64> = (0..lookup_batch).map(|_| rng.next_range(num_relations)).collect();
        
        // Benchmark lookup (sequential)
        let mut lookup_times: Vec<f64> = Vec::with_capacity(runs);
        for _ in 0..runs {
            let start = Instant::now();
            let masks = lookup_filter_mask(&dict, &lookup_heads, &lookup_rels, num_entities);
            black_box(&masks);
            lookup_times.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        }
        
        results.push(FilterMaskResult {
            name: "rust_filter_mask_seq".into(),
            num_triples,
            num_entities,
            runs,
            build_median_us: median(&mut build_times),
            lookup_median_us: median(&mut lookup_times),
            lookup_batch_size: lookup_batch,
        });
        
        // Benchmark lookup (parallel)
        let mut lookup_times_par: Vec<f64> = Vec::with_capacity(runs);
        for _ in 0..runs {
            let start = Instant::now();
            let masks = lookup_filter_mask_par(&dict, &lookup_heads, &lookup_rels, num_entities);
            black_box(&masks);
            lookup_times_par.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        }
        
        results.push(FilterMaskResult {
            name: "rust_filter_mask_par".into(),
            num_triples,
            num_entities,
            runs,
            build_median_us: median(&mut build_times.clone()),
            lookup_median_us: median(&mut lookup_times_par),
            lookup_batch_size: lookup_batch,
        });
    }
    
    results
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running HPO benchmarks...");
    
    let neg_sampling_results = bench_negative_sampling();
    println!("Negative sampling: {} configs done", neg_sampling_results.len() / 2);
    
    let filter_mask_results = bench_filter_mask();
    println!("Filter mask: {} configs done", filter_mask_results.len() / 2);
    
    let output = HpoBenchOutput {
        negative_sampling: neg_sampling_results,
        filter_mask: filter_mask_results,
        system_info: SystemInfo {
            num_cpus: num_cpus::get(),
            rayon_threads: rayon::current_num_threads(),
        },
    };
    
    let out_path = std::path::PathBuf::from("outputs/benches/rust_hpo_benchmark.json");
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out_path, serde_json::to_string_pretty(&output)?)?;
    
    println!("\nResults saved to: {}", out_path.display());
    
    // Print summary
    println!("\n=== NEGATIVE SAMPLING (samples/sec) ===");
    for r in &output.negative_sampling {
        println!("{:<25} batch={:<5} neg={:<4} entities={:<6} | {:.2e} samples/sec | {:.1}us median",
            r.name, r.batch_size, r.num_negatives, r.num_entities,
            r.throughput_samples_per_sec, r.median_time_us);
    }
    
    println!("\n=== FILTER MASK (us) ===");
    for r in &output.filter_mask {
        println!("{:<25} triples={:<6} entities={:<5} | build={:.1}us | lookup={:.1}us (batch={})",
            r.name, r.num_triples, r.num_entities,
            r.build_median_us, r.lookup_median_us, r.lookup_batch_size);
    }
    
    Ok(())
}
