//! logger.rs
//!
//! Structured logging stack (tracing-based).
//! - Console + file layers
//! - Optional JSON output
//! - Optional async/non-blocking writer
//! - Optional progress integration via tracing-indicatif (feature="progress")

use std::{env, path::PathBuf};

use tracing_subscriber::{
    fmt,
    layer::{Layer, SubscriberExt},
    util::SubscriberInitExt,
    EnvFilter, Registry,
};

#[cfg(feature = "progress")]
use tracing_indicatif::IndicatifLayer;
#[cfg(feature = "progress")]
use tracing_indicatif::span_ext::IndicatifSpanExt;

pub struct LoggerGuard {
    _guards: Vec<tracing_appender::non_blocking::WorkerGuard>,
}

impl LoggerGuard {
    fn new() -> Self {
        Self { _guards: Vec::new() }
    }

    fn push_guard(&mut self, guard: tracing_appender::non_blocking::WorkerGuard) {
        self._guards.push(guard);
    }
}

#[derive(Clone, Debug)]
pub struct LoggerConfig {
    pub level: String,
    pub log_dir: Option<PathBuf>,
    pub console: bool,
    pub file: bool,
    pub json: bool,
    pub async_logging: bool,
    pub progress: bool,
}

impl Default for LoggerConfig {
    fn default() -> Self {
        let level = env::var("LOG_LEVEL").unwrap_or_else(|_| "INFO".to_string());
        let log_dir = env::var("LOG_DIR").ok().map(PathBuf::from);
        let json = env::var("LOG_JSON")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let async_logging = env::var("LOG_ASYNC")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);
        let console = env::var("LOG_CONSOLE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);
        let file = env::var("LOG_FILE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);
        let progress = env::var("LOG_PROGRESS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);

        Self {
            level,
            log_dir,
            console,
            file,
            json,
            async_logging,
            progress,
        }
    }
}

pub fn init_logger(cfg: LoggerConfig) -> LoggerGuard {
    let filter = EnvFilter::try_new(cfg.level).unwrap_or_else(|_| EnvFilter::new("info"));
    let mut guard = LoggerGuard::new();

    let console_layer: Option<Box<dyn Layer<Registry> + Send + Sync>> = if cfg.console {
        let layer = if cfg.async_logging {
            let (nb, g) = tracing_appender::non_blocking(std::io::stderr());
            guard.push_guard(g);
            if cfg.json {
                fmt::layer()
                    .with_target(false)
                    .with_thread_ids(true)
                    .with_thread_names(true)
                    .json()
                    .with_writer(nb)
                    .boxed()
            } else {
                fmt::layer()
                    .with_target(false)
                    .with_thread_ids(true)
                    .with_thread_names(true)
                    .with_writer(nb)
                    .boxed()
            }
        } else if cfg.json {
            fmt::layer()
                .with_target(false)
                .with_thread_ids(true)
                .with_thread_names(true)
                .json()
                .with_writer(std::io::stderr)
                .boxed()
        } else {
            fmt::layer()
                .with_target(false)
                .with_thread_ids(true)
                .with_thread_names(true)
                .with_writer(std::io::stderr)
                .boxed()
        };
        Some(layer)
    } else {
        None
    };

    let file_layer: Option<Box<dyn Layer<Registry> + Send + Sync>> = if cfg.file {
        let dir = cfg.log_dir.unwrap_or_else(|| PathBuf::from("logs"));
        if std::fs::create_dir_all(&dir).is_ok() {
            let appender = tracing_appender::rolling::RollingFileAppender::new(
                tracing_appender::rolling::Rotation::DAILY,
                dir,
                "pff-rust.log",
            );
            let (nb, g) = tracing_appender::non_blocking(appender);
            guard.push_guard(g);
            let layer = if cfg.json {
                fmt::layer()
                    .with_writer(nb)
                    .with_ansi(false)
                    .with_target(true)
                    .with_thread_ids(true)
                    .with_thread_names(true)
                    .json()
                    .boxed()
            } else {
                fmt::layer()
                    .with_writer(nb)
                    .with_ansi(false)
                    .with_target(true)
                    .with_thread_ids(true)
                    .with_thread_names(true)
                    .boxed()
            };
            Some(layer)
        } else {
            None
        }
    } else {
        None
    };

    #[cfg(feature = "progress")]
    let progress_layer: Option<Box<dyn Layer<Registry> + Send + Sync>> = if cfg.progress {
        Some(IndicatifLayer::new().boxed())
    } else {
        None
    };
    #[cfg(not(feature = "progress"))]
    let progress_layer: Option<Box<dyn Layer<Registry> + Send + Sync>> = None;

    let mut combined: Box<dyn Layer<Registry> + Send + Sync> =
        Box::new(tracing_subscriber::layer::Identity::default());
    if let Some(layer) = console_layer {
        combined = combined.and_then(layer).boxed();
    }
    if let Some(layer) = file_layer {
        combined = combined.and_then(layer).boxed();
    }
    if let Some(layer) = progress_layer {
        combined = combined.and_then(layer).boxed();
    }

    tracing_subscriber::registry()
        .with(combined)
        .with(filter)
        .init();

    guard
}

/// Create a progress span that can be used by tracing-indicatif.
/// When the feature is disabled, this returns a normal span.
pub fn progress_span(name: &str, total: Option<u64>, unit: Option<&str>) -> tracing::Span {
    let span = tracing::info_span!("progress", task = name);

    #[cfg(feature = "progress")]
    {
        if let Some(total) = total {
            span.pb_set_length(total);
        }
        if let Some(unit) = unit {
            span.pb_set_message(unit);
        }
        span.pb_start();
    }

    span
}

/// Increment a progress span by `delta`.
pub fn progress_inc(span: &tracing::Span, delta: u64) {
    #[cfg(feature = "progress")]
    {
        span.pb_inc(delta);
    }
    let _ = delta;
}

/// Finish a progress span.
pub fn progress_finish(span: &tracing::Span, message: Option<&str>) {
    #[cfg(feature = "progress")]
    {
        if let Some(msg) = message {
            span.pb_set_finish_message(msg);
        }
    }
    let _ = message;
}
