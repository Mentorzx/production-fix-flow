//! optimized_io.rs
//!
//! SOTA-optimized I/O utilities based on benchmark results:
//! - sonic-rs for JSON parsing (3x faster than serde_json)
//! - rkyv for zero-copy cached data access (16x faster deserialize)
//! - memchr for SIMD-accelerated line scanning
//! - LZ4 for fast compression when speed > ratio
//!
//! Benchmark source: benches/optimizations_benchmark.rs

use std::io::{Read, Write};
use std::path::Path;

use polars::prelude::*;
use thiserror::Error;

/// Error types for optimized I/O operations
#[derive(Debug, Error)]
pub enum OptimizedIoError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json parse error: {0}")]
    JsonParse(String),

    #[error("lz4 error: {0}")]
    Lz4(String),

    #[error("rkyv error: {0}")]
    Rkyv(String),

    #[error("polars error: {0}")]
    Polars(#[from] PolarsError),
}

pub type Result<T> = std::result::Result<T, OptimizedIoError>;

// ============================================================================
// MEMCHR UTILITIES - SIMD-accelerated byte searching
// ============================================================================

/// Count lines using SIMD-accelerated memchr (2x faster than iterator)
#[inline]
pub fn count_lines_fast(data: &[u8]) -> usize {
    memchr::memchr_iter(b'\n', data).count()
}

/// Find line ranges for parallel processing
/// Returns Vec<(start, end)> for each line
#[inline]
pub fn find_line_ranges(data: &[u8]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::with_capacity(data.len() / 100); // ~100 bytes per line estimate
    let mut start = 0;

    for pos in memchr::memchr_iter(b'\n', data) {
        if pos > start {
            ranges.push((start, pos));
        }
        start = pos + 1;
    }

    // Handle last line without trailing newline
    if start < data.len() {
        ranges.push((start, data.len()));
    }

    ranges
}

/// Split data into lines efficiently using memchr
#[inline]
pub fn split_lines_fast(data: &[u8]) -> impl Iterator<Item = &[u8]> {
    LineIterator { data, pos: 0 }
}

struct LineIterator<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for LineIterator<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }

        let remaining = &self.data[self.pos..];
        match memchr::memchr(b'\n', remaining) {
            Some(idx) => {
                let line = &remaining[..idx];
                self.pos += idx + 1;
                Some(line)
            }
            None => {
                let line = remaining;
                self.pos = self.data.len();
                if line.is_empty() {
                    None
                } else {
                    Some(line)
                }
            }
        }
    }
}

// ============================================================================
// SONIC-RS JSON PARSING - 3x faster than serde_json
// ============================================================================

/// Parse JSON using sonic-rs (fastest JSON parser)
/// Returns a dynamic Value that can be accessed lazily
#[inline]
pub fn parse_json_fast(data: &[u8]) -> Result<sonic_rs::Value> {
    sonic_rs::from_slice(data).map_err(|e| OptimizedIoError::JsonParse(e.to_string()))
}

/// Parse JSON lazily - only parses accessed fields (best for large JSON)
#[inline]
pub fn parse_json_lazy(data: &[u8]) -> Result<sonic_rs::LazyValue<'_>> {
    sonic_rs::from_slice(data).map_err(|e| OptimizedIoError::JsonParse(e.to_string()))
}

/// Parse JSON into a typed struct using sonic-rs
#[inline]
pub fn parse_json_typed<'a, T: serde::Deserialize<'a>>(data: &'a [u8]) -> Result<T> {
    sonic_rs::from_slice(data).map_err(|e| OptimizedIoError::JsonParse(e.to_string()))
}

/// Serialize to JSON using sonic-rs
#[inline]
pub fn to_json_fast<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
    sonic_rs::to_vec(value).map_err(|e| OptimizedIoError::JsonParse(e.to_string()))
}

/// Parse NDJSON (newline-delimited JSON) in parallel using sonic-rs
/// Uses memchr for SIMD line splitting + rayon for parallel parsing
pub fn parse_ndjson_parallel<T: serde::de::DeserializeOwned + Send>(
    data: &[u8],
) -> Vec<Result<T>> {
    use rayon::prelude::*;

    let line_ranges = find_line_ranges(data);

    line_ranges
        .par_iter()
        .filter_map(|&(start, end)| {
            let line = &data[start..end];
            // Skip empty or whitespace-only lines
            if line.iter().all(|&b| b == b' ' || b == b'\t' || b == b'\r') {
                return None;
            }
            Some(parse_json_typed::<T>(line))
        })
        .collect()
}

/// Parse NDJSON into LazyValue for deferred parsing (fastest)
pub fn parse_ndjson_lazy(data: &[u8]) -> Vec<Result<sonic_rs::LazyValue<'_>>> {
    find_line_ranges(data)
        .iter()
        .filter_map(|&(start, end)| {
            let line = &data[start..end];
            if line.iter().all(|&b| b == b' ' || b == b'\t' || b == b'\r') {
                return None;
            }
            Some(parse_json_lazy(line))
        })
        .collect()
}

// ============================================================================
// LZ4 COMPRESSION - 5x faster compression than ZSTD
// ============================================================================

/// Compress data using LZ4 (fastest compression, ~4000 MB/s decompress)
/// Best for: Hot paths where speed > compression ratio
#[inline]
pub fn compress_lz4(data: &[u8]) -> Vec<u8> {
    lz4_flex::compress_prepend_size(data)
}

/// Decompress LZ4 data
#[inline]
pub fn decompress_lz4(data: &[u8]) -> Result<Vec<u8>> {
    lz4_flex::decompress_size_prepended(data)
        .map_err(|e| OptimizedIoError::Lz4(e.to_string()))
}

/// Compress with streaming LZ4 (for large data)
pub fn compress_lz4_stream<R: Read, W: Write>(reader: &mut R, writer: &mut W) -> Result<u64> {
    let mut encoder = lz4_flex::frame::FrameEncoder::new(writer);
    let bytes = std::io::copy(reader, &mut encoder)?;
    encoder.finish().map_err(|e| OptimizedIoError::Lz4(e.to_string()))?;
    Ok(bytes)
}

/// Decompress streaming LZ4
pub fn decompress_lz4_stream<R: Read, W: Write>(reader: &mut R, writer: &mut W) -> Result<u64> {
    let mut decoder = lz4_flex::frame::FrameDecoder::new(reader);
    let bytes = std::io::copy(&mut decoder, writer)?;
    Ok(bytes)
}

// ============================================================================
// SNAPPY COMPRESSION - Alternative fast compression
// ============================================================================

/// Compress data using Snappy (500 MB/s decompress, smaller than LZ4)
#[inline]
pub fn compress_snappy(data: &[u8]) -> Vec<u8> {
    snap::raw::Encoder::new().compress_vec(data).unwrap_or_default()
}

/// Decompress Snappy data
#[inline]
pub fn decompress_snappy(data: &[u8]) -> Result<Vec<u8>> {
    snap::raw::Decoder::new()
        .decompress_vec(data)
        .map_err(|e| OptimizedIoError::Lz4(format!("snappy: {}", e)))
}

// ============================================================================
// RKYV ZERO-COPY SERIALIZATION - 16x faster than serde_json deserialize
// ============================================================================

// Note: rkyv 0.8's complex trait bounds make generic functions impractical.
// Use the rkyv crate directly for zero-copy access:
//
// Example usage:
// ```rust
// use rkyv::{Archive, Deserialize, Serialize};
//
// #[derive(Archive, Deserialize, Serialize)]
// struct MyData { value: i32 }
//
// // Serialize
// let bytes = rkyv::to_bytes::<rkyv::rancor::Failure>(&data).unwrap();
//
// // Zero-copy access (fastest - 16x faster than deserialize)
// let archived = rkyv::access::<ArchivedMyData, rkyv::rancor::Failure>(&bytes).unwrap();
// println!("value = {}", archived.value);
//
// // Full deserialize (when needed)
// let data = rkyv::from_bytes::<MyData, rkyv::rancor::Failure>(&bytes).unwrap();
// ```

/// Check if data has rkyv magic header (for cached data detection)
#[inline]
pub fn is_rkyv_data(data: &[u8]) -> bool {
    // rkyv doesn't have a magic header, but archived data typically starts
    // with the root object's first field. Check for alignment.
    data.len() >= 4 && data.as_ptr() as usize % 4 == 0
}

/// Serialize Vec<u8> to rkyv format (for caching raw byte arrays)
#[inline]
pub fn serialize_bytes_rkyv(data: &Vec<u8>) -> Result<rkyv::util::AlignedVec> {
    rkyv::to_bytes::<rkyv::rancor::Failure>(data)
        .map_err(|e| OptimizedIoError::Rkyv(format!("{:?}", e)))
}

/// Deserialize bytes from rkyv format
#[inline]
pub fn deserialize_bytes_rkyv(data: &[u8]) -> Result<Vec<u8>> {
    rkyv::from_bytes::<Vec<u8>, rkyv::rancor::Failure>(data)
        .map_err(|e| OptimizedIoError::Rkyv(format!("{:?}", e)))
}

// ============================================================================
// POLARS INTEGRATION - Fast DataFrame I/O
// ============================================================================

/// Read NDJSON directly into DataFrame using optimized parsing
pub fn read_ndjson_to_dataframe(data: &[u8]) -> Result<DataFrame> {
    let cursor = std::io::Cursor::new(data);
    let df = polars::prelude::JsonLineReader::new(cursor)
        .infer_schema_len(std::num::NonZeroUsize::new(100))
        .finish()?;
    Ok(df)
}

/// Read CSV with SIMD-optimized parsing
pub fn read_csv_fast(data: &[u8], delimiter: u8) -> Result<DataFrame> {
    use polars::prelude::{CsvParseOptions, CsvReadOptions};
    let cursor = std::io::Cursor::new(data);
    let parse_opts = CsvParseOptions::default().with_separator(delimiter);
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_parse_options(parse_opts)
        .into_reader_with_file_handle(cursor)
        .finish()?;
    Ok(df)
}

// ============================================================================
// FILE FORMAT DETECTION
// ============================================================================

/// Compression format detection
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressionFormat {
    None,
    Zstd,
    Lz4,
    Gzip,
    Snappy,
}

/// Detect compression format from file extension
pub fn detect_compression(path: &Path) -> CompressionFormat {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase());

    match ext.as_deref() {
        Some("zst" | "zstd") => CompressionFormat::Zstd,
        Some("lz4") => CompressionFormat::Lz4,
        Some("gz" | "gzip") => CompressionFormat::Gzip,
        Some("snappy" | "sz") => CompressionFormat::Snappy,
        _ => CompressionFormat::None,
    }
}

/// Detect compression format from magic bytes
pub fn detect_compression_magic(data: &[u8]) -> CompressionFormat {
    if data.len() < 4 {
        return CompressionFormat::None;
    }

    match &data[..4] {
        // ZSTD magic: 0x28B52FFD
        [0x28, 0xB5, 0x2F, 0xFD] => CompressionFormat::Zstd,
        // LZ4 frame magic: 0x184D2204
        [0x04, 0x22, 0x4D, 0x18] => CompressionFormat::Lz4,
        // Gzip magic: 0x1F8B
        [0x1F, 0x8B, _, _] => CompressionFormat::Gzip,
        _ => CompressionFormat::None,
    }
}

/// Auto-decompress based on detected format
pub fn auto_decompress(data: &[u8]) -> Result<Vec<u8>> {
    match detect_compression_magic(data) {
        CompressionFormat::Zstd => {
            let expected = zstd_safe::get_frame_content_size(data)
                .map_err(|e| OptimizedIoError::Io(std::io::Error::other(e.to_string())))?;
            
            if let Some(len) = expected {
                let mut out = vec![0u8; len as usize];
                let res = zstd_safe::decompress(&mut out, data)
                    .map_err(|code| OptimizedIoError::Io(std::io::Error::other(
                        zstd_safe::get_error_name(code).to_string()
                    )))?;
                out.truncate(res);
                Ok(out)
            } else {
                let mut decoder = zstd::stream::read::Decoder::new(std::io::Cursor::new(data))?;
                let mut out = Vec::new();
                decoder.read_to_end(&mut out)?;
                Ok(out)
            }
        }
        CompressionFormat::Lz4 => decompress_lz4(data),
        CompressionFormat::Gzip => {
            let mut decoder = flate2::read::GzDecoder::new(data);
            let mut out = Vec::new();
            decoder.read_to_end(&mut out)?;
            Ok(out)
        }
        CompressionFormat::Snappy => decompress_snappy(data),
        CompressionFormat::None => Ok(data.to_vec()),
    }
}

// ============================================================================
// BENCHMARKING UTILITIES
// ============================================================================

/// Quick benchmark helper - returns (result, duration_secs)
#[inline]
pub fn bench<F, T>(f: F) -> (T, f64)
where
    F: FnOnce() -> T,
{
    let start = std::time::Instant::now();
    let result = f();
    let duration = start.elapsed().as_secs_f64();
    (result, duration)
}

/// Benchmark multiple iterations and return median
pub fn bench_median<F, T>(iterations: usize, mut f: F) -> (T, f64)
where
    F: FnMut() -> T,
{
    let mut times: Vec<f64> = Vec::with_capacity(iterations);
    let mut last_result = None;

    for _ in 0..iterations {
        let start = std::time::Instant::now();
        last_result = Some(f());
        times.push(start.elapsed().as_secs_f64());
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[iterations / 2];

    (last_result.unwrap(), median)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_lines_fast() {
        let data = b"line1\nline2\nline3\n";
        assert_eq!(count_lines_fast(data), 3);
    }

    #[test]
    fn test_find_line_ranges() {
        let data = b"line1\nline2\nline3";
        let ranges = find_line_ranges(data);
        assert_eq!(ranges.len(), 3);
        assert_eq!(&data[ranges[0].0..ranges[0].1], b"line1");
        assert_eq!(&data[ranges[1].0..ranges[1].1], b"line2");
        assert_eq!(&data[ranges[2].0..ranges[2].1], b"line3");
    }

    #[test]
    fn test_split_lines_fast() {
        let data = b"a\nb\nc";
        let lines: Vec<_> = split_lines_fast(data).collect();
        assert_eq!(lines, vec![b"a".as_slice(), b"b", b"c"]);
    }

    #[test]
    fn test_parse_json_fast() {
        let data = br#"{"key": "value", "num": 42}"#;
        let value = parse_json_fast(data).unwrap();
        assert_eq!(value["key"].as_str(), Some("value"));
        assert_eq!(value["num"].as_i64(), Some(42));
    }

    #[test]
    fn test_lz4_roundtrip() {
        let original = b"Hello, world! This is test data for LZ4 compression.";
        let compressed = compress_lz4(original);
        let decompressed = decompress_lz4(&compressed).unwrap();
        assert_eq!(original.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_snappy_roundtrip() {
        let original = b"Hello, world! This is test data for Snappy compression.";
        let compressed = compress_snappy(original);
        let decompressed = decompress_snappy(&compressed).unwrap();
        assert_eq!(original.as_slice(), decompressed.as_slice());
    }

    #[test]
    fn test_detect_compression() {
        assert_eq!(detect_compression(Path::new("file.zst")), CompressionFormat::Zstd);
        assert_eq!(detect_compression(Path::new("file.lz4")), CompressionFormat::Lz4);
        assert_eq!(detect_compression(Path::new("file.gz")), CompressionFormat::Gzip);
        assert_eq!(detect_compression(Path::new("file.txt")), CompressionFormat::None);
    }

    #[test]
    fn test_detect_compression_magic() {
        // ZSTD magic bytes
        assert_eq!(
            detect_compression_magic(&[0x28, 0xB5, 0x2F, 0xFD]),
            CompressionFormat::Zstd
        );
        // LZ4 frame magic
        assert_eq!(
            detect_compression_magic(&[0x04, 0x22, 0x4D, 0x18]),
            CompressionFormat::Lz4
        );
        // Gzip magic
        assert_eq!(
            detect_compression_magic(&[0x1F, 0x8B, 0x08, 0x00]),
            CompressionFormat::Gzip
        );
    }
}
