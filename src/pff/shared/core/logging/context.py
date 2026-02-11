from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from loguru import logger as _loguru_logger

if TYPE_CHECKING:
    from types import ModuleType

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    otel_trace: ModuleType | None = None  # type: ignore[no-redef]
    TracerProvider: type | None = None  # type: ignore[no-redef,misc]
    Resource: type | None = None  # type: ignore[no-redef,misc]


class TraceContext:
    """Manages trace context using contextvars for async safety."""

    _trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "trace_id", default=None
    )
    _span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "span_id", default=None
    )

    @classmethod
    def get(cls) -> dict[str, str | None]:
        return {"trace_id": cls._trace_id.get(), "span_id": cls._span_id.get()}

    @classmethod
    def set(cls, trace_id: str | None, span_id: str | None):
        if trace_id:
            cls._trace_id.set(trace_id)
        if span_id:
            cls._span_id.set(span_id)


_tracer: Any | None = None


def _init_tracer() -> Any | None:
    global _tracer
    if _tracer is not None:
        return _tracer

    if not OTEL_AVAILABLE or otel_trace is None:
        return None

    try:
        trace_provider = otel_trace.get_tracer_provider()

        if (
            OTEL_AVAILABLE
            and TracerProvider is not None
            and not isinstance(trace_provider, TracerProvider)
        ):
            resource_attrs = {
                "service.name": "pff",
                "service.version": os.getenv("PFF_VERSION", "6.0.0"),
                "deployment.environment": os.getenv("ENVIRONMENT", "development"),
            }
            if Resource is not None:
                provider = TracerProvider(resource=Resource.create(resource_attrs))
                otel_trace.set_tracer_provider(provider)
                trace_provider = provider

        _tracer = trace_provider.get_tracer("pff")
        return _tracer
    except Exception:
        return None


@contextmanager
def start_span(name: str, **attributes: Any) -> Any:
    """Starts an OpenTelemetry span and injects context into logs."""
    tracer = _init_tracer()
    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span(name, attributes=attributes) as span:
        ctx = span.get_span_context()
        trace_id = f"{ctx.trace_id:032x}"
        span_id = f"{ctx.span_id:016x}"

        token_t = TraceContext._trace_id.set(trace_id)
        token_s = TraceContext._span_id.set(span_id)
        try:
            yield span
        finally:
            TraceContext._trace_id.reset(token_t)
            TraceContext._span_id.reset(token_s)


def bind_trace_id(trace_id: str | None) -> Any:
    """Bind a specific trace ID to the logger context (Legacy shim)."""
    if trace_id:
        TraceContext.set(trace_id, None)
    return _loguru_logger.bind(trace_id=trace_id)
