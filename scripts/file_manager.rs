//! file_manager.rs
//!
//! High-throughput file manager for ETL workloads.
//! - Uses ConcurrencyEngine (Rayon + IoEngine)
//! - Thread-per-core io_uring on Linux (feature="linux_uring")
//! - Buffer reuse via IoEngine
//! - Plugin-style format handlers
//! - Streaming DataFrame output

use std::{
    cell::RefCell,
    collections::HashMap,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};

use memmap2::MmapOptions;
use polars::prelude::*;
use rayon::prelude::*;
use thiserror::Error;

use crate::concurrency::{
    ConcurrencyConfig, ConcurrencyEngine, ConcurrencyError, IoPayload, ReadMode, ReadRequest,
};

pub type Result<T> = std::result::Result<T, FileManagerError>;

#[derive(Debug, Error)]
pub enum FileManagerError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    #[error("polars error: {0}")]
    Polars(#[from] PolarsError),

    #[error("zstd error: {0}")]
    Zstd(String),

    #[error("zip error: {0}")]
    Zip(String),

    #[error("json error: {0}")]
    Json(String),

    #[error("unsupported extension: {0}")]
    UnsupportedExtension(String),

    #[error("dispatch error")]
    Dispatch,

    #[error("concurrency error: {0}")]
    Concurrency(#[from] ConcurrencyError),
}

/// File format selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileFormat {
    Auto,
    Csv,
    Tsv,
    Parquet,
    Jsonl,
    Ndjson,
}

/// Handler trait for pluggable formats.
pub trait FormatHandler: Send + Sync {
    fn read_df(&self, bytes: &[u8]) -> Result<DataFrame>;
    fn write_df(&self, path: &Path, df: &DataFrame) -> Result<()>;
}

/// A parsed DataFrame with its source path.
#[derive(Debug)]
pub struct DataFrameItem {
    pub path: PathBuf,
    pub df: DataFrame,
}

/// Stream of parsed DataFrames.
pub type DataFrameStream = flume::Receiver<Result<DataFrameItem>>;

/// Loader configuration.
#[derive(Clone, Debug)]
pub struct FileManagerConfig {
    pub concurrency: ConcurrencyConfig,
    pub small_decompress_threshold: usize,
    pub stream_capacity: usize,
}

impl Default for FileManagerConfig {
    fn default() -> Self {
        Self {
            concurrency: ConcurrencyConfig::default(),
            small_decompress_threshold: 1 * 1024 * 1024,
            stream_capacity: 128,
        }
    }
}

/// Main file manager.
pub struct FileManager {
    root: PathBuf,
    engine: ConcurrencyEngine,
    handlers: HashMap<String, Arc<dyn FormatHandler>>,
    cfg: FileManagerConfig,
}

impl FileManager {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        Self::with_config(root, FileManagerConfig::default())
    }

    pub fn with_config(root: impl Into<PathBuf>, cfg: FileManagerConfig) -> Result<Self> {
        let engine = ConcurrencyEngine::new(cfg.concurrency.clone())?;
        let mut handlers: HashMap<String, Arc<dyn FormatHandler>> = HashMap::new();

        handlers.insert("csv".into(), Arc::new(CsvHandler { delim: b',' }));
        handlers.insert("tsv".into(), Arc::new(CsvHandler { delim: b'\t' }));
        handlers.insert("parquet".into(), Arc::new(ParquetHandler));
        handlers.insert("jsonl".into(), Arc::new(JsonLinesHandler));
        handlers.insert("ndjson".into(), Arc::new(JsonLinesHandler));

        Ok(Self {
            root: root.into(),
            engine,
            handlers,
            cfg,
        })
    }

    pub fn register_handler(&mut self, ext: &str, handler: Arc<dyn FormatHandler>) {
        self.handlers.insert(ext.to_ascii_lowercase(), handler);
    }

    pub fn resolve_path(&self, p: impl AsRef<Path>) -> PathBuf {
        let p = p.as_ref();
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.root.join(p)
        }
    }

    pub fn exists(&self, p: impl AsRef<Path>) -> bool {
        self.resolve_path(p).exists()
    }

    pub fn list(&self, dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
        let dir = self.resolve_path(dir);
        let mut out = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            out.push(entry.path());
        }
        Ok(out)
    }

    pub fn glob(&self, pattern: &str) -> Result<Vec<PathBuf>> {
        let mut out = Vec::new();
        let abs_pattern = if Path::new(pattern).is_absolute() {
            pattern.to_string()
        } else {
            self.root.join(pattern).to_string_lossy().to_string()
        };
        for entry in glob::glob(&abs_pattern)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?
        {
            if let Ok(p) = entry {
                out.push(p);
            }
        }
        Ok(out)
    }

    pub fn mkdir(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = self.resolve_path(path);
        std::fs::create_dir_all(path)?;
        Ok(())
    }

    pub fn delete(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = self.resolve_path(path);
        if path.is_dir() {
            std::fs::remove_dir_all(path)?;
        } else {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    /// Read a single file into a DataFrame.
    pub fn read_df(&self, path: impl AsRef<Path>, format: FileFormat) -> Result<DataFrame> {
        let path = self.resolve_path(path);
        let (ext, mode) =
            resolve_format_and_mode(&path, format, self.cfg.small_decompress_threshold)?;

        let io = self.engine.io_engine();
        let (result_tx, result_rx) = flume::bounded(1);
        io.distributor()
            .dispatch(ReadRequest {
                path: path.clone(),
                size_hint: None,
                mode,
                result_tx,
            })
            .map_err(|_| FileManagerError::Dispatch)?;

        let res = result_rx.recv().map_err(|_| FileManagerError::Dispatch)?;
        let bytes = resolve_payload_bytes(res.payload, mode)?;

        let handler = self
            .handlers
            .get(&ext)
            .ok_or_else(|| FileManagerError::UnsupportedExtension(ext.clone()))?;

        handler.read_df(&bytes)
    }

    /// Read raw bytes (auto-decompresses .zst/.zstd).
    pub fn read_bytes(&self, path: impl AsRef<Path>) -> Result<Vec<u8>> {
        let path = self.resolve_path(path);
        let (ext, mode) =
            resolve_format_and_mode(&path, FileFormat::Auto, self.cfg.small_decompress_threshold)?;
        let _ = ext;

        let io = self.engine.io_engine();
        let (result_tx, result_rx) = flume::bounded(1);
        io.distributor()
            .dispatch(ReadRequest {
                path: path.clone(),
                size_hint: None,
                mode,
                result_tx,
            })
            .map_err(|_| FileManagerError::Dispatch)?;

        let res = result_rx.recv().map_err(|_| FileManagerError::Dispatch)?;
        resolve_payload_bytes(res.payload, mode)
    }

    /// Write a DataFrame using a registered handler.
    pub fn write_df(
        &self,
        path: impl AsRef<Path>,
        df: &DataFrame,
        format: FileFormat,
    ) -> Result<()> {
        let path = self.resolve_path(path);
        let ext = resolve_format(&path, format)?;
        if ext == "zst" || ext == "zstd" {
            return Err(FileManagerError::UnsupportedExtension(ext));
        }

        let handler = self
            .handlers
            .get(&ext)
            .ok_or_else(|| FileManagerError::UnsupportedExtension(ext.clone()))?;
        handler.write_df(&path, df)
    }

    /// Stream DataFrames from a list of paths.
    pub fn stream_dataframes(&self, paths: Vec<PathBuf>) -> Result<DataFrameStream> {
        let (out_tx, out_rx) = flume::bounded(self.cfg.stream_capacity);
        let io = self.engine.io_engine();

        for p in paths {
            let path = self.resolve_path(p);
            let (ext, mode) = resolve_format_and_mode(
                &path,
                FileFormat::Auto,
                self.cfg.small_decompress_threshold,
            )?;

            let (result_tx, result_rx) = flume::bounded(1);
            io.distributor()
                .dispatch(ReadRequest {
                    path: path.clone(),
                    size_hint: None,
                    mode,
                    result_tx,
                })
                .map_err(|_| FileManagerError::Dispatch)?;

            let handler = self
                .handlers
                .get(&ext)
                .ok_or_else(|| FileManagerError::UnsupportedExtension(ext.clone()))?
                .clone();
            let out_tx = out_tx.clone();

            // Parsing on Rayon to avoid blocking I/O workers.
            let pool = self.engine.cpu_pool();
            pool.spawn(move || {
                let res = (|| -> Result<DataFrameItem> {
                    let read = result_rx.recv().map_err(|_| FileManagerError::Dispatch)?;
                    let bytes = resolve_payload_bytes(read.payload, mode)?;
                    let df = handler.read_df(&bytes)?;
                    Ok(DataFrameItem { path, df })
                })();
                let _ = out_tx.send(res);
            });
        }

        Ok(out_rx)
    }
}

fn should_parse_entry(entry: &piz::read::FileMetadata<'_>) -> bool {
    entry.is_file()
}

fn pad_simd_json(buffer: &mut Vec<u8>) {
    buffer.reserve(simd_json::SIMDJSON_PADDING);
    buffer.extend(std::iter::repeat(0).take(simd_json::SIMDJSON_PADDING));
}

// -----------------------------------------------------------------------------
// Hot-path helpers: thread-local reuse to kill per-file/per-line setup overhead.
// -----------------------------------------------------------------------------
//
// - ZSTD: reuse a DCtx per thread (no context init + fewer allocator hits).
//   Zstd explicitly recommends reusing contexts for repeated decompressions.
// - JSON (simd-json): reuse parsing Buffers + a scratch input Vec per thread to
//   avoid allocating a new tape/buffers for every tiny payload.
//
// These optimizations matter most for “many small files/lines” workloads.

thread_local! {
    static ZSTD_DCTX: RefCell<zstd_safe::DCtx<'static>> =
        RefCell::new(zstd_safe::DCtx::create());

    static SIMD_BYTES: RefCell<Vec<u8>> = RefCell::new(Vec::new());
    static SIMD_BUFFERS: RefCell<simd_json::Buffers> = RefCell::new(simd_json::Buffers::new(0));
}

#[inline]
fn load_and_pad(bytes: &mut Vec<u8>, input: &[u8]) {
    bytes.clear();
    // Allocate space for input + padding zeros for safe SIMD reads.
    bytes.reserve(input.len() + simd_json::SIMDJSON_PADDING);
    bytes.extend_from_slice(input);
    // Write padding zeros into capacity (not visible to parser, but safe for SIMD reads).
    let orig_len = bytes.len();
    unsafe {
        let ptr = bytes.as_mut_ptr().add(orig_len);
        std::ptr::write_bytes(ptr, 0, simd_json::SIMDJSON_PADDING);
    }
    // len stays at orig_len; simd-json reads beyond but only parses up to len.
}

fn parse_json_or_lines(buffer: &[u8]) -> Result<usize> {
    // Parse buffer as one or more JSON documents (handles concatenated JSON).
    // Use serde_json's streaming parser which is robust and handles multi-document buffers.
    let stream = serde_json::Deserializer::from_slice(buffer).into_iter::<serde_json::Value>();
    let mut count = 0usize;
    for result in stream {
        result.map_err(|e| FileManagerError::Json(format!("invalid JSON: {e}")))?;
        count += 1;
    }
    if count == 0 {
        return Err(FileManagerError::Json("empty JSON buffer".to_string()));
    }
    Ok(count)
}

fn trim_ascii(mut input: &[u8]) -> &[u8] {
    while let Some((&first, rest)) = input.split_first() {
        if first.is_ascii_whitespace() {
            input = rest;
        } else {
            break;
        }
    }
    while let Some((&last, rest)) = input.split_last() {
        if last.is_ascii_whitespace() {
            input = rest;
        } else {
            break;
        }
    }
    input
}

pub fn read_zip_entry_buffers(zip_bytes: &[u8]) -> Result<Vec<Vec<u8>>> {
    let archive =
        piz::ZipArchive::new(zip_bytes).map_err(|e| FileManagerError::Zip(e.to_string()))?;
    let entries = archive.entries();

    let buffers = entries
        .par_iter()
        .filter(|entry| should_parse_entry(entry))
        .map(|entry| -> Result<Vec<u8>> {
            let mut reader = archive
                .read(entry)
                .map_err(|e| FileManagerError::Zip(e.to_string()))?;
            let mut buffer = Vec::with_capacity(entry.size);
            reader.read_to_end(&mut buffer)?;
            Ok(buffer)
        })
        .collect::<Result<Vec<Vec<u8>>>>()?;

    Ok(buffers)
}

pub fn parse_json_buffers(buffers: &mut [Vec<u8>]) -> Result<usize> {
    let count = buffers
        .par_iter_mut()
        .map(|buffer| -> Result<usize> { parse_json_or_lines(buffer) })
        .try_reduce(|| 0usize, |a, b| Ok(a + b))?;

    Ok(count)
}

pub fn process_zip_json_bytes(zip_bytes: &[u8]) -> Result<usize> {
    let archive =
        piz::ZipArchive::new(zip_bytes).map_err(|e| FileManagerError::Zip(e.to_string()))?;
    let entries = archive.entries();

    let count = entries
        .par_iter()
        .filter(|entry| should_parse_entry(entry))
        .map(|entry| -> Result<usize> {
            let mut reader = archive
                .read(entry)
                .map_err(|e| FileManagerError::Zip(e.to_string()))?;
            let mut buffer = Vec::with_capacity(entry.size);
            reader.read_to_end(&mut buffer)?;
            parse_json_or_lines(&buffer)
        })
        .try_reduce(|| 0usize, |a, b| Ok(a + b))?;

    Ok(count)
}

pub fn mmap_file(path: &Path) -> Result<memmap2::Mmap> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    Ok(mmap)
}

pub fn decompress_zstd_bytes(input: &[u8]) -> Result<Vec<u8>> {
    decompress_zstd(input)
}

fn resolve_payload_bytes(payload: IoPayload, mode: ReadMode) -> Result<Vec<u8>> {
    match payload {
        IoPayload::Decompressed(buf) => Ok(buf),
        IoPayload::Raw(recycled) => {
            if matches!(mode, ReadMode::Zstd { .. }) {
                decompress_zstd(recycled.as_slice())
            } else {
                Ok(recycled.as_slice().to_vec())
            }
        }
    }
}

fn resolve_format_and_mode(
    path: &Path,
    format: FileFormat,
    small_threshold: usize,
) -> Result<(String, ReadMode)> {
    let path_ext = extension_lower(path);
    let ext = resolve_format(path, format)?;
    let is_zstd_path = matches!(path_ext.as_deref(), Some("zst") | Some("zstd"));

    if is_zstd_path {
        let inner = if format == FileFormat::Auto {
            inner_extension(path).unwrap_or_else(|| "zst".to_string())
        } else {
            ext.clone()
        };
        Ok((
            inner,
            ReadMode::Zstd {
                small_decompress_threshold: small_threshold,
            },
        ))
    } else {
        Ok((ext, ReadMode::Raw))
    }
}

fn resolve_format(path: &Path, format: FileFormat) -> Result<String> {
    let ext = match format {
        FileFormat::Auto => extension_lower(path)
            .ok_or_else(|| FileManagerError::UnsupportedExtension(path.display().to_string()))?,
        FileFormat::Csv => "csv".into(),
        FileFormat::Tsv => "tsv".into(),
        FileFormat::Parquet => "parquet".into(),
        FileFormat::Jsonl => "jsonl".into(),
        FileFormat::Ndjson => "ndjson".into(),
    };
    Ok(ext)
}

fn extension_lower(p: &Path) -> Option<String> {
    p.extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
}

fn inner_extension(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_string_lossy().to_string();
    extension_lower(Path::new(&stem))
}

fn decompress_zstd(input: &[u8]) -> Result<Vec<u8>> {
    ZSTD_DCTX.with(|cell| -> Result<Vec<u8>> {
        let mut dctx = cell.borrow_mut();
        dctx.reset(zstd_safe::ResetDirective::SessionOnly)
            .map_err(|e| FileManagerError::Zstd(format!("zstd reset: {e}")))?;

        let expected = zstd_safe::get_frame_content_size(input);
        let expected = match expected {
            Ok(size) => size,
            Err(_) => None,
        };

        let Some(len_u64) = expected else {
            // Unknown size: fall back to streaming decoder.
            let mut decoder = zstd::stream::read::Decoder::new(input)?;
            let mut out = Vec::with_capacity(input.len().saturating_mul(4));
            decoder.read_to_end(&mut out)?;
            return Ok(out);
        };

        let len: usize = len_u64
            .try_into()
            .map_err(|_| FileManagerError::Zstd("zstd frame too large".to_string()))?;

        let mut out = vec![0u8; len];
        let written = dctx
            .decompress(&mut out, input)
            .map_err(|code| FileManagerError::Zstd(zstd_safe::get_error_name(code).to_string()))?;
        out.truncate(written);
        Ok(out)
    })
}

struct CsvHandler {
    delim: u8,
}

impl FormatHandler for CsvHandler {
    fn read_df(&self, bytes: &[u8]) -> Result<DataFrame> {
        use polars::prelude::{CsvParseOptions, CsvReadOptions};
        let cursor = std::io::Cursor::new(bytes);
        let parse_opts = CsvParseOptions::default().with_separator(self.delim);
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .with_parse_options(parse_opts)
            .into_reader_with_file_handle(cursor)
            .finish()?;
        Ok(df)
    }

    fn write_df(&self, path: &Path, df: &DataFrame) -> Result<()> {
        let mut f = std::fs::File::create(path)?;
        let mut df_clone = df.clone();
        CsvWriter::new(&mut f).finish(&mut df_clone)?;
        Ok(())
    }
}

struct ParquetHandler;

impl FormatHandler for ParquetHandler {
    fn read_df(&self, bytes: &[u8]) -> Result<DataFrame> {
        let cursor = std::io::Cursor::new(bytes);
        let df = ParquetReader::new(cursor).finish()?;
        Ok(df)
    }

    fn write_df(&self, path: &Path, df: &DataFrame) -> Result<()> {
        let mut f = std::fs::File::create(path)?;
        let mut df_clone = df.clone();
        ParquetWriter::new(&mut f).finish(&mut df_clone)?;
        Ok(())
    }
}

struct JsonLinesHandler;

impl FormatHandler for JsonLinesHandler {
    fn read_df(&self, bytes: &[u8]) -> Result<DataFrame> {
        let cursor = std::io::Cursor::new(bytes);
        let df = polars::prelude::JsonLineReader::new(cursor)
            .infer_schema_len(std::num::NonZeroUsize::new(100))
            .finish()?;
        Ok(df)
    }

    fn write_df(&self, path: &Path, df: &DataFrame) -> Result<()> {
        let mut f = std::fs::File::create(path)?;
        let mut df_clone = df.clone();
        JsonWriter::new(&mut f)
            .with_json_format(JsonFormat::JsonLines)
            .finish(&mut df_clone)?;
        Ok(())
    }
}
