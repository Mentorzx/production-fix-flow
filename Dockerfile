# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12
ARG PFF_ACCELERATOR=cpu

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG PFF_ACCELERATOR=cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CARGO_HOME=/root/.cargo \
    POETRY_VERSION=2.3.3 \
    POETRY_NO_INTERACTION=1 \
    RUSTUP_HOME=/root/.rustup \
    PATH="/root/.cargo/bin:${PATH}" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_CREATE=true

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    mold \
    curl \
    git \
    libpq-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable

RUN pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock poetry.toml ./

RUN case "${PFF_ACCELERATOR}" in \
      cpu|cuda) ;; \
      *) echo "Unsupported PFF_ACCELERATOR=${PFF_ACCELERATOR}" >&2; exit 1 ;; \
    esac

RUN poetry install --without dev --no-root --no-ansi

COPY . .

RUN poetry install --without dev --no-ansi
RUN rm -f /app/src/pff_rust/_pff_rust*.so
RUN /app/.venv/bin/pip install "maturin==1.11.5"
RUN . /app/.venv/bin/activate && maturin build --release --manifest-path /app/src/pff_rust/Cargo.toml --out /tmp/pff-rust-dist
RUN /app/.venv/bin/pip install /tmp/pff-rust-dist/pff_rust-*.whl

RUN case "${PFF_ACCELERATOR}" in \
      cpu) \
        /app/.venv/bin/pip uninstall -y torch triton || true && \
        /app/.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.0+cpu ;; \
      cuda) \
        /app/.venv/bin/pip uninstall -y torch triton || true && \
        /app/.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0+cu128 triton==3.3.0 ;; \
      *) \
        echo "Unsupported PFF_ACCELERATOR=${PFF_ACCELERATOR}" >&2; exit 1 ;; \
    esac

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime-base

ARG PFF_ACCELERATOR=cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}" \
    PFF_ENV=production \
    PFF_ACCELERATOR=${PFF_ACCELERATOR}

WORKDIR /app

RUN groupadd -r pff && useradd -r -g pff pff && usermod -d /app pff

COPY --from=builder --chown=pff:pff /app/.venv /app/.venv
COPY --from=builder --chown=pff:pff /app/src/pff /app/src/pff
COPY --from=builder --chown=pff:pff /app/config /app/config
COPY --from=builder --chown=pff:pff /app/README.md /app/pyproject.toml /app/poetry.lock /app/poetry.toml /app/

RUN mkdir -p /app/data /app/logs /app/outputs && chown pff:pff /app/data /app/logs /app/outputs

USER pff

ENTRYPOINT ["pff"]
CMD ["--help"]

FROM runtime-base AS runtime

FROM runtime-base AS runtime-cpu
ENV PFF_ACCELERATOR=cpu

FROM runtime-base AS runtime-cuda
ENV PFF_ACCELERATOR=cuda
