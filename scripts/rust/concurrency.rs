//! concurrency.rs
//!
//! Concurrency orchestrator with CPU, IO, and GPU backends.
//! - CPU: Rayon thread pool
//! - IO: Monoio TPC (io_uring) on Linux, fallback to blocking threads
//! - GPU: pluggable backend (stub by default)
//!
//! High-throughput file I/O is delegated to the IoEngine to avoid head-of-line
//! blocking when a single request stalls; CPU parsing runs on Rayon.

use std::{
    alloc::{alloc, dealloc, Layout},
    io::Read,
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
};

use thiserror::Error;

use crate::logger;

#[cfg(feature = "linux_uring")]
use monoio::fs::File as MonoioFile;

/// Errors for the concurrency engine.
#[derive(Debug, Error)]
pub enum ConcurrencyError {
    #[error("invalid config: {0}")]
    InvalidConfig(&'static str),

    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    #[error("channel closed")]
    ChannelClosed,

    #[error("gpu backend unavailable")]
    GpuUnavailable,
}

pub type Result<T> = std::result::Result<T, ConcurrencyError>;

/// Execution backend selection.
#[derive(Clone, Copy, Debug)]
pub enum TaskKind {
    Auto,
    Cpu,
    Io,
    Gpu,
}

/// Engine config.
#[derive(Clone, Debug)]
pub struct ConcurrencyConfig {
    pub cpu_threads: usize,
    pub io: IoEngineConfig,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        let physical = num_cpus::get_physical().max(1);
        Self {
            cpu_threads: physical,
            io: IoEngineConfig::default(),
        }
    }
}

/// Main orchestrator: CPU + IO + optional GPU.
pub struct ConcurrencyEngine {
    cpu_pool: Arc<rayon::ThreadPool>,
    io_engine: Arc<IoEngine>,
    gpu: Arc<dyn GpuExecutor>,
}

impl ConcurrencyEngine {
    pub fn new(cfg: ConcurrencyConfig) -> Result<Self> {
        if cfg.cpu_threads == 0 {
            return Err(ConcurrencyError::InvalidConfig("cpu_threads cannot be 0"));
        }

        let cpu_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.cpu_threads)
            .build()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        Ok(Self {
            cpu_pool: Arc::new(cpu_pool),
            io_engine: Arc::new(IoEngine::new(cfg.io)?),
            gpu: Arc::new(NoopGpuExecutor),
        })
    }

    pub fn io_engine(&self) -> Arc<IoEngine> {
        self.io_engine.clone()
    }

    pub fn cpu_pool(&self) -> Arc<rayon::ThreadPool> {
        self.cpu_pool.clone()
    }

    pub fn set_gpu_backend(&mut self, backend: Arc<dyn GpuExecutor>) {
        self.gpu = backend;
    }

    /// Execute a task on the selected backend. IO here is a blocking task;
    /// use IoEngine for zero-copy reads.
    pub fn execute_task<F, R>(&self, kind: TaskKind, task: F) -> Result<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let selected = match kind {
            TaskKind::Auto => TaskKind::Cpu,
            other => other,
        };

        match selected {
            TaskKind::Cpu => Ok(self.cpu_pool.install(task)),
            TaskKind::Io => {
                let handle = thread::spawn(task);
                Ok(handle.join().map_err(|_| {
                    ConcurrencyError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "io task panicked",
                    ))
                })?)
            }
            TaskKind::Gpu => {
                let boxed = Box::new(move || Box::new(task()) as Box<dyn std::any::Any + Send>);
                let out = self.gpu.execute_boxed(boxed)?;
                out.downcast::<R>().map(|b| *b).map_err(|_| {
                    ConcurrencyError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "gpu result type mismatch",
                    ))
                })
            }
            TaskKind::Auto => Ok(self.cpu_pool.install(task)),
        }
    }

    /// Create a progress span for a long-running task.
    pub fn progress(task_name: &str, total: Option<u64>, unit: Option<&str>) -> tracing::Span {
        logger::progress_span(task_name, total, unit)
    }
}

/// GPU executor trait (optional backends).
pub trait GpuExecutor: Send + Sync {
    fn execute_boxed(
        &self,
        task: Box<dyn FnOnce() -> Box<dyn std::any::Any + Send> + Send>,
    ) -> Result<Box<dyn std::any::Any + Send>>;
}

struct NoopGpuExecutor;

impl GpuExecutor for NoopGpuExecutor {
    fn execute_boxed(
        &self,
        _task: Box<dyn FnOnce() -> Box<dyn std::any::Any + Send> + Send>,
    ) -> Result<Box<dyn std::any::Any + Send>> {
        Err(ConcurrencyError::GpuUnavailable)
    }
}

/// Runtime + worker configuration for I/O engine.
#[derive(Clone, Debug)]
pub struct IoEngineConfig {
    pub workers: usize,
    pub queue_capacity: usize,
    pub direct_io_block: usize,
    pub read_buffer_bytes: usize,
    pub direct_io: bool,
    pub use_uring: bool,
    pub pin_threads: bool,
}

impl Default for IoEngineConfig {
    fn default() -> Self {
        let workers = num_cpus::get().max(1);
        let direct_io_block = 4096usize;
        Self {
            workers,
            queue_capacity: (workers * 2).max(64),
            direct_io_block,
            read_buffer_bytes: 4 * 1024 * 1024,
            direct_io: true,
            use_uring: true,
            pin_threads: true,
        }
    }
}

/// A read request dispatched to an I/O worker.
#[derive(Debug)]
pub struct ReadRequest {
    pub path: PathBuf,
    pub size_hint: Option<u64>,
    pub mode: ReadMode,
    pub result_tx: flume::Sender<IoReadResult>,
}

/// Processing mode for a read request.
#[derive(Clone, Copy, Debug)]
pub enum ReadMode {
    Raw,
    Zstd { small_decompress_threshold: usize },
    Lz4,  // Added: LZ4 decompression (5x faster than ZSTD)
}

/// Output of a read request.
#[derive(Debug)]
pub struct IoReadResult {
    pub path: PathBuf,
    pub bytes: usize,
    pub payload: IoPayload,
}

/// Payload returned from the I/O thread.
#[derive(Debug)]
pub enum IoPayload {
    Raw(RecycledBuffer),
    Decompressed(Vec<u8>),
}

/// Lock-free work distributor with round-robin routing.
pub struct WorkDistributor {
    senders: Vec<flume::Sender<ReadRequest>>,
    next: AtomicUsize,
    shutdown: Arc<AtomicBool>,
}

impl WorkDistributor {
    fn new(senders: Vec<flume::Sender<ReadRequest>>, shutdown: Arc<AtomicBool>) -> Self {
        Self {
            senders,
            next: AtomicUsize::new(0),
            shutdown,
        }
    }

    pub fn dispatch(&self, req: ReadRequest) -> Result<()> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(ConcurrencyError::ChannelClosed);
        }
        let idx = self
            .next
            .fetch_add(1, Ordering::Relaxed)
            % self.senders.len().max(1);
        self.senders[idx]
            .send(req)
            .map_err(|_| ConcurrencyError::ChannelClosed)
    }
}

/// I/O engine with a fixed runtime per worker thread.
pub struct IoEngine {
    cfg: IoEngineConfig,
    distributor: WorkDistributor,
    handles: Vec<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
}

impl IoEngine {
    pub fn new(cfg: IoEngineConfig) -> Result<Self> {
        if cfg.workers == 0 {
            return Err(ConcurrencyError::InvalidConfig("workers cannot be 0"));
        }
        if cfg.queue_capacity == 0 {
            return Err(ConcurrencyError::InvalidConfig("queue_capacity cannot be 0"));
        }
        if cfg.direct_io_block == 0 {
            return Err(ConcurrencyError::InvalidConfig("direct_io_block cannot be 0"));
        }
        if cfg.read_buffer_bytes == 0 {
            return Err(ConcurrencyError::InvalidConfig("read_buffer_bytes cannot be 0"));
        }

        let shutdown = Arc::new(AtomicBool::new(false));
        let mut senders = Vec::with_capacity(cfg.workers);
        let mut handles = Vec::with_capacity(cfg.workers);

        for worker_id in 0..cfg.workers {
            let (tx, rx) = flume::bounded::<ReadRequest>(cfg.queue_capacity);
            senders.push(tx);

            let cfg_clone = cfg.clone();
            let shutdown_clone = shutdown.clone();
            let handle = thread::Builder::new()
                .name(format!("io-worker-{worker_id}"))
                .spawn(move || {
                    if cfg_clone.pin_threads {
                        pin_thread_to_core(worker_id);
                    }
                    run_worker(worker_id, cfg_clone, shutdown_clone, rx);
                })
                .map_err(|e| ConcurrencyError::Io(e))?;

            handles.push(handle);
        }

        let distributor = WorkDistributor::new(senders, shutdown.clone());

        Ok(Self {
            cfg,
            distributor,
            handles,
            shutdown,
        })
    }

    pub fn distributor(&self) -> &WorkDistributor {
        &self.distributor
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    pub fn join(self) {
        let handles = self.handles;
        drop(self.distributor);
        for h in handles {
            let _ = h.join();
        }
    }
}

fn run_worker(
    worker_id: usize,
    cfg: IoEngineConfig,
    shutdown: Arc<AtomicBool>,
    rx: flume::Receiver<ReadRequest>,
) {
    if cfg.use_uring {
        #[cfg(feature = "linux_uring")]
        {
            monoio::start::<monoio::IoUringDriver, _>(async move {
                let pool = BufferPool::new(cfg.read_buffer_bytes, cfg.direct_io_block);
                loop {
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }

                    let req = match rx.recv_async().await {
                        Ok(r) => r,
                        Err(_) => break,
                    };

                    let read = read_file_aligned(&cfg, &pool, &req.path, req.size_hint).await;

                    match read {
                        Ok((buf, bytes)) => {
                            let payload = match req.mode {
                                ReadMode::Zstd {
                                    small_decompress_threshold,
                                } if bytes <= small_decompress_threshold => {
                                    match decompress_zstd(buf.as_slice()) {
                                        Ok(out) => {
                                            pool.release(buf);
                                            IoPayload::Decompressed(out)
                                        }
                                        Err(_) => {
                                            IoPayload::Raw(RecycledBuffer::new(
                                                buf,
                                                pool.clone(),
                                            ))
                                        }
                                    }
                                }
                                ReadMode::Lz4 => {
                                    match decompress_lz4(buf.as_slice()) {
                                        Ok(out) => {
                                            pool.release(buf);
                                            IoPayload::Decompressed(out)
                                        }
                                        Err(_) => {
                                            IoPayload::Raw(RecycledBuffer::new(
                                                buf,
                                                pool.clone(),
                                            ))
                                        }
                                    }
                                }
                                _ => IoPayload::Raw(RecycledBuffer::new(buf, pool.clone())),
                            };

                            let _ = req.result_tx.send(IoReadResult {
                                path: req.path,
                                bytes,
                                payload,
                            });
                        }
                        Err(e) => {
                            let _ = req.result_tx.send(IoReadResult {
                                path: req.path,
                                bytes: 0,
                                payload: IoPayload::Raw(RecycledBuffer::new(
                                    pool.acquire(),
                                    pool.clone(),
                                )),
                            });
                            eprintln!("worker {worker_id} read error: {e}");
                        }
                    }
                }
            });
            return;
        }
    }

    run_worker_blocking(worker_id, cfg, shutdown, rx);
}

fn run_worker_blocking(
    worker_id: usize,
    cfg: IoEngineConfig,
    shutdown: Arc<AtomicBool>,
    rx: flume::Receiver<ReadRequest>,
) {
    let pool = BufferPool::new(cfg.read_buffer_bytes, cfg.direct_io_block);
    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        let req = match rx.recv() {
            Ok(r) => r,
            Err(_) => break,
        };

        let read = read_file_blocking(&cfg, &pool, &req.path, req.size_hint);
        match read {
            Ok((buf, bytes)) => {
                let payload = match req.mode {
                    ReadMode::Zstd {
                        small_decompress_threshold,
                    } if bytes <= small_decompress_threshold => {
                        match decompress_zstd(buf.as_slice()) {
                            Ok(out) => {
                                pool.release(buf);
                                IoPayload::Decompressed(out)
                            }
                            Err(_) => IoPayload::Raw(RecycledBuffer::new(buf, pool.clone())),
                        }
                    }
                    ReadMode::Lz4 => {
                        match decompress_lz4(buf.as_slice()) {
                            Ok(out) => {
                                pool.release(buf);
                                IoPayload::Decompressed(out)
                            }
                            Err(_) => IoPayload::Raw(RecycledBuffer::new(buf, pool.clone())),
                        }
                    }
                    _ => IoPayload::Raw(RecycledBuffer::new(buf, pool.clone())),
                };

                let _ = req.result_tx.send(IoReadResult {
                    path: req.path,
                    bytes,
                    payload,
                });
            }
            Err(e) => {
                let _ = req.result_tx.send(IoReadResult {
                    path: req.path,
                    bytes: 0,
                    payload: IoPayload::Raw(RecycledBuffer::new(pool.acquire(), pool.clone())),
                });
                eprintln!("worker {worker_id} read error: {e}");
            }
        }
    }
}

#[cfg(feature = "linux_uring")]
async fn read_file_aligned(
    cfg: &IoEngineConfig,
    pool: &Arc<BufferPool>,
    path: &Path,
    size_hint: Option<u64>,
) -> Result<(AlignedBuffer, usize)> {
    let size = match size_hint {
        Some(n) => n as usize,
        None => std::fs::metadata(path)?.len() as usize,
    };
    if size == 0 {
        return Ok((pool.acquire(), 0));
    }

    // O_DIRECT requires size and offset to be aligned; fallback for odd sizes.
    let use_direct = cfg.direct_io && size % cfg.direct_io_block == 0;
    let file = open_file(path, use_direct).await?;

    let buf = pool.acquire_with_size(size);
    let (res, mut buf) = file.read_at(buf, 0).await;
    let bytes = res?;

    unsafe {
        buf.set_init(bytes);
    }

    Ok((buf, bytes))
}

#[cfg(feature = "linux_uring")]
async fn open_file(path: &Path, direct: bool) -> Result<MonoioFile> {
    if direct {
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::OpenOptionsExt;
            let std_file = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(path)?;
            return Ok(MonoioFile::from_std(std_file)?);
        }
    }
    Ok(MonoioFile::open(path).await?)
}

fn read_file_blocking(
    _cfg: &IoEngineConfig,
    pool: &Arc<BufferPool>,
    path: &Path,
    size_hint: Option<u64>,
) -> Result<(AlignedBuffer, usize)> {
    let size = match size_hint {
        Some(n) => n as usize,
        None => std::fs::metadata(path)?.len() as usize,
    };
    if size == 0 {
        return Ok((pool.acquire(), 0));
    }

    let mut buf = pool.acquire_with_size(size);
    let mut file = std::fs::File::open(path)?;
    let dst = buf.as_mut_slice();
    let bytes = std::io::Read::read(&mut file, dst)?;
    unsafe {
        buf.set_init(bytes);
    }
    Ok((buf, bytes))
}

/// Buffer pool with fixed alignment for O_DIRECT.
#[derive(Clone, Debug)]
pub struct BufferPool {
    tx: flume::Sender<AlignedBuffer>,
    rx: flume::Receiver<AlignedBuffer>,
    align: usize,
    default_len: usize,
}

impl BufferPool {
    pub fn new(default_len: usize, align: usize) -> Arc<Self> {
        let (tx, rx) = flume::bounded::<AlignedBuffer>(64);
        Arc::new(Self {
            tx,
            rx,
            align: align.max(1),
            default_len: default_len.max(1),
        })
    }

    pub fn acquire(&self) -> AlignedBuffer {
        self.rx.try_recv().unwrap_or_else(|_| {
            AlignedBuffer::new(self.default_len, self.align)
                .expect("aligned buffer alloc failed")
        })
    }

    pub fn acquire_with_size(&self, len: usize) -> AlignedBuffer {
        if len <= self.default_len {
            self.acquire()
        } else {
            AlignedBuffer::new(len, self.align).expect("aligned buffer alloc failed")
        }
    }

    pub fn release(&self, mut buf: AlignedBuffer) {
        buf.reset_init();
        let _ = self.tx.try_send(buf);
    }
}

/// A buffer that returns itself to the pool when dropped.
#[derive(Debug)]
pub struct RecycledBuffer {
    buf: Option<AlignedBuffer>,
    pool: Arc<BufferPool>,
}

impl RecycledBuffer {
    pub fn new(buf: AlignedBuffer, pool: Arc<BufferPool>) -> Self {
        Self {
            buf: Some(buf),
            pool,
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        self.buf.as_ref().map(|b| b.as_slice()).unwrap_or(&[])
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buf
            .as_mut()
            .map(|b| b.as_mut_slice())
            .unwrap_or(&mut [])
    }

    pub fn into_inner(mut self) -> AlignedBuffer {
        self.buf.take().expect("buffer already taken")
    }
}

impl Drop for RecycledBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            self.pool.release(buf);
        }
    }
}

/// Aligned buffer for O_DIRECT.
#[derive(Debug)]
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    cap: usize,
    len: usize,
    align: usize,
}

impl AlignedBuffer {
    pub fn new(len: usize, align: usize) -> Result<Self> {
        let cap = len.max(1);
        let align = align.max(1);
        let layout = Layout::from_size_align(cap, align)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Other, "aligned alloc failed")
        })?;
        Ok(Self {
            ptr,
            cap,
            len: 0,
            align,
        })
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.cap) }
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn reset_init(&mut self) {
        self.len = 0;
    }

    pub unsafe fn set_init(&mut self, len: usize) {
        self.len = len.min(self.cap);
    }
}

#[cfg(feature = "linux_uring")]
unsafe impl monoio::buf::IoBuf for AlignedBuffer {
    fn read_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    fn bytes_init(&self) -> usize {
        self.len
    }
}

#[cfg(feature = "linux_uring")]
unsafe impl monoio::buf::IoBufMut for AlignedBuffer {
    fn write_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    fn bytes_total(&mut self) -> usize {
        self.cap
    }

    unsafe fn set_init(&mut self, pos: usize) {
        self.len = pos.min(self.cap);
    }
}

unsafe impl Send for AlignedBuffer {}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if let Ok(layout) = Layout::from_size_align(self.cap, self.align) {
            unsafe { dealloc(self.ptr.as_ptr(), layout) };
        }
    }
}

fn pin_thread_to_core(idx: usize) {
    #[cfg(target_os = "linux")]
    {
        let cpus = num_cpus::get().max(1);
        let core = idx % cpus;
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            libc::CPU_SET(core, &mut set);
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = idx;
    }
}

fn decompress_zstd(input: &[u8]) -> Result<Vec<u8>> {
    let expected = zstd_safe::get_frame_content_size(input).map_err(|e| {
        ConcurrencyError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })?;
    let Some(len) = expected else {
        let mut decoder = zstd::stream::read::Decoder::new(std::io::Cursor::new(input))
            .map_err(ConcurrencyError::Io)?;
        let mut out = Vec::new();
        decoder.read_to_end(&mut out).map_err(ConcurrencyError::Io)?;
        return Ok(out);
    };

    let mut out = vec![0u8; len as usize];
    let res = zstd_safe::decompress(&mut out, input).map_err(|code| {
        ConcurrencyError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            zstd_safe::get_error_name(code),
        ))
    })?;
    out.truncate(res);
    Ok(out)
}

/// LZ4 decompression (5x faster than ZSTD for hot paths)
fn decompress_lz4(input: &[u8]) -> Result<Vec<u8>> {
    lz4_flex::decompress_size_prepended(input).map_err(|e| {
        ConcurrencyError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })
}
