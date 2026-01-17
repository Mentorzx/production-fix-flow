use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;
use mimalloc::MiMalloc;
use serde::Serialize;
use polars::prelude::*;

use pff_rust_bench::file_manager_core::{FileManager as FileManagerCore, FileFormat as FormatCore};
use pff_rust_bench::file_manager_opt::{FileManager as FileManagerOpt, FileFormat as FormatOpt};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Serialize, Clone)]
struct RunStats {
    load_s: f64,
    parse_s: f64,
    total_s: f64,
}

#[derive(Serialize)]
struct BenchResult {
    name: String,
    iterations: usize,
    mean: RunStats,
    p50: RunStats,
    p95: RunStats,
    p99: RunStats,
    throughput_mbs: f64,
}

fn calculate_stats(mut times: Vec<RunStats>, file_size_mb: f64) -> BenchResult {
    let n = times.len();
    let name = "unnamed".to_string();
    
    // Sort by total_s for percentiles
    times.sort_by(|a, b| a.total_s.partial_cmp(&b.total_s).unwrap());
    
    let sum_load: f64 = times.iter().map(|t| t.load_s).sum();
    let sum_parse: f64 = times.iter().map(|t| t.parse_s).sum();
    let sum_total: f64 = times.iter().map(|t| t.total_s).sum();
    
    let mean = RunStats {
        load_s: sum_load / n as f64,
        parse_s: sum_parse / n as f64,
        total_s: sum_total / n as f64,
    };
    
    BenchResult {
        name,
        iterations: n,
        mean,
        p50: times[n / 2].clone(),
        p95: times[(n as f64 * 0.95) as usize].clone(),
        p99: times[(n as f64 * 0.99) as usize].clone(),
        throughput_mbs: file_size_mb / (sum_total / n as f64),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path_str = "data/models/correct.parquet";
    let path = PathBuf::from(path_str);
    let file_size_mb = std::fs::metadata(&path)?.len() as f64 / 1024.0 / 1024.0;
    let iterations = 100;

    println!("Starting Rust Benchmark ({} iterations) on {}", iterations, path_str);

    // 1. Core FileManager
    let fm_core = FileManagerCore::new(".")?;
    let mut core_runs = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        // Core read_df includes load + parse
        let df = fm_core.read_df(&path, FormatCore::Parquet)?;
        black_box(df);
        let elapsed = start.elapsed().as_secs_f64();
        core_runs.push(RunStats {
            load_s: 0.0, // Internal breakdown not exposed easily without editing source
            parse_s: 0.0,
            total_s: elapsed,
        });
    }
    let mut core_res = calculate_stats(core_runs, file_size_mb);
    core_res.name = "Rust_Core_FileManager".to_string();

    // 2. Optimized FileManager
    let fm_opt = FileManagerOpt::new(".")?;
    let mut opt_runs = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let df = fm_opt.read_df(&path, FormatOpt::Parquet)?;
        black_box(df);
        let elapsed = start.elapsed().as_secs_f64();
        opt_runs.push(RunStats {
            load_s: 0.0,
            parse_s: 0.0,
            total_s: elapsed,
        });
    }
    let mut opt_res = calculate_stats(opt_runs, file_size_mb);
    opt_res.name = "Rust_Optimized_FileManager".to_string();

    // 3. Raw Polars (Speed of light)
    let mut raw_runs = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        // Mimic what the managers do but at lowest level
        let file = std::fs::File::open(&path)?;
        let df = ParquetReader::new(file).finish()?;
        black_box(df);
        let elapsed = start.elapsed().as_secs_f64();
        raw_runs.push(RunStats {
            load_s: 0.0,
            parse_s: 0.0,
            total_s: elapsed,
        });
    }
    let mut raw_res = calculate_stats(raw_runs, file_size_mb);
    raw_res.name = "Rust_Raw_Polars_Native".to_string();

    let results = vec![core_res, opt_res, raw_res];
    let json = serde_json::to_string_pretty(&results)?;
    std::fs::create_dir_all("outputs/benches")?;
    std::fs::write("outputs/benches/rust_parquet_results.json", json)?;
    
    println!("Rust benchmarks completed. Results saved to outputs/benches/rust_parquet_results.json");
    Ok(())
}
