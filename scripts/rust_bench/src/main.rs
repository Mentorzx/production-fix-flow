use std::{
    hint::black_box,
    path::PathBuf,
    time::Instant,
};

use mimalloc::MiMalloc;
use serde::Serialize;

use pff_rust_bench::file_manager::{
    FileManager,
    mmap_file,
    parse_json_buffers,
    process_zip_json_bytes,
    read_zip_entry_buffers,
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Serialize)]
struct BenchSection {
    load_avg_s: f64,
    read_avg_s: f64,
    parse_avg_s: f64,
    total_avg_s: f64,
    total_full_avg_s: f64,
    runs: usize,
}

#[derive(Serialize)]
struct BenchOutput {
    zip: BenchSection,
    zstd: BenchSection,
    notes: serde_json::Value,
}

fn avg(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn zip_load_parse(zip_path: &PathBuf) -> pff_rust_bench::file_manager::Result<(f64, f64, f64)> {
    let read_start = Instant::now();
    let mmap = mmap_file(zip_path)?;
    let mut buffers = read_zip_entry_buffers(&mmap)?;
    let read_time = read_start.elapsed().as_secs_f64();

    let parse_start = Instant::now();
    let parsed = parse_json_buffers(&mut buffers)?;
    black_box(parsed);
    let parse_time = parse_start.elapsed().as_secs_f64();

    Ok((read_time, parse_time, read_time + parse_time))
}

fn zip_total_full(zip_path: &PathBuf) -> pff_rust_bench::file_manager::Result<f64> {
    let start = Instant::now();
    let mmap = mmap_file(zip_path)?;
    let parsed = process_zip_json_bytes(&mmap)?;
    black_box(parsed);
    Ok(start.elapsed().as_secs_f64())
}

fn zstd_load_parse(
    fm: &FileManager,
    zstd_path: &PathBuf,
) -> pff_rust_bench::file_manager::Result<(f64, f64, f64)> {
    let read_start = Instant::now();
    let zip_bytes = fm.read_bytes(zstd_path)?;
    let mut buffers = read_zip_entry_buffers(&zip_bytes)?;
    let read_time = read_start.elapsed().as_secs_f64();

    let parse_start = Instant::now();
    let parsed = parse_json_buffers(&mut buffers)?;
    black_box(parsed);
    let parse_time = parse_start.elapsed().as_secs_f64();

    Ok((read_time, parse_time, read_time + parse_time))
}

fn zstd_total_full(fm: &FileManager, zstd_path: &PathBuf) -> pff_rust_bench::file_manager::Result<f64> {
    let start = Instant::now();
    let zip_bytes = fm.read_bytes(zstd_path)?;
    let parsed = process_zip_json_bytes(&zip_bytes)?;
    black_box(parsed);
    Ok(start.elapsed().as_secs_f64())
}

fn main() -> pff_rust_bench::file_manager::Result<()> {
    let zip_path = PathBuf::from("data/models/correct.zip");
    let zstd_path = PathBuf::from("outputs/benches/correct.zip.zst");
    let fm = FileManager::new(".")?;

    let runs = 3usize;
    let mut zip_read = Vec::with_capacity(runs);
    let mut zip_parse = Vec::with_capacity(runs);
    let mut zip_total = Vec::with_capacity(runs);
    let mut zip_total_full_times = Vec::with_capacity(runs);

    let mut zstd_read = Vec::with_capacity(runs);
    let mut zstd_parse = Vec::with_capacity(runs);
    let mut zstd_total = Vec::with_capacity(runs);
    let mut zstd_total_full_times = Vec::with_capacity(runs);

    for _ in 0..runs {
        let (read_s, parse_s, total_s) = zip_load_parse(&zip_path)?;
        zip_read.push(read_s);
        zip_parse.push(parse_s);
        zip_total.push(total_s);
        zip_total_full_times.push(zip_total_full(&zip_path)?);
    }

    for _ in 0..runs {
        let (read_s, parse_s, total_s) = zstd_load_parse(&fm, &zstd_path)?;
        zstd_read.push(read_s);
        zstd_parse.push(parse_s);
        zstd_total.push(total_s);
        zstd_total_full_times.push(zstd_total_full(&fm, &zstd_path)?);
    }

    let output = BenchOutput {
        zip: BenchSection {
            load_avg_s: avg(&zip_read),
            read_avg_s: avg(&zip_read),
            parse_avg_s: avg(&zip_parse),
            total_avg_s: avg(&zip_total),
            total_full_avg_s: avg(&zip_total_full_times),
            runs,
        },
        zstd: BenchSection {
            load_avg_s: avg(&zstd_read),
            read_avg_s: avg(&zstd_read),
            parse_avg_s: avg(&zstd_parse),
            total_avg_s: avg(&zstd_total),
            total_full_avg_s: avg(&zstd_total_full_times),
            runs,
        },
        notes: serde_json::json!({
            "zip_path": zip_path.display().to_string(),
            "zstd_path": zstd_path.display().to_string(),
            "total_full_avg_s": "Full SOTA pipeline (read+parse) via process_zip_json_bytes.",
        }),
    };

    let out_path = PathBuf::from("outputs/benches/rust_file_manager_zip_vs_zstd_benchmark.json");
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out_path, serde_json::to_string_pretty(&output).unwrap())?;
    println!("{}", out_path.display());
    Ok(())
}
