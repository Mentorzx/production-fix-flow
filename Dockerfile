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
  POETRY_CACHE_DIR=/tmp/pff-poetry-cache \
  RUSTUP_HOME=/root/.rustup \
  PATH="/usr/local/bin:/root/.cargo/bin:${PATH}" \
  POETRY_VIRTUALENVS_IN_PROJECT=true \
  POETRY_VIRTUALENVS_CREATE=true

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  mold \
  curl \
  git \
  libpq-dev \
  nodejs \
  npm \
  pkg-config \
  && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable
RUN rustup target add wasm32-unknown-unknown
RUN toolchain_dir="$(find /root/.rustup/toolchains -mindepth 1 -maxdepth 1 -type d | sort | head -n1)" \
  && ln -sf "${toolchain_dir}/bin/cargo" /usr/local/bin/cargo \
  && ln -sf "${toolchain_dir}/bin/rustc" /usr/local/bin/rustc \
  && ln -sf "${toolchain_dir}/bin/rustdoc" /usr/local/bin/rustdoc

RUN pip install "poetry==${POETRY_VERSION}"
RUN cargo install wasm-bindgen-cli --version 0.2.113 --locked

COPY pyproject.toml poetry.lock poetry.toml ./

RUN case "${PFF_ACCELERATOR}" in \
  cpu|cuda) ;; \
  *) echo "Unsupported PFF_ACCELERATOR=${PFF_ACCELERATOR}" >&2; exit 1 ;; \
  esac

RUN poetry install --without dev --no-root --no-ansi \
  && rm -rf /tmp/pff-poetry-cache /root/.cache/pypoetry

COPY . .

RUN bash /app/src/pff/infrastructure/hpo/dashboard/build_dashboard.sh
RUN rm -rf /app/src/pff/infrastructure/hpo/dashboard/node_modules

RUN find /app -path /app/.venv -prune -o -exec chmod a+rX {} +

RUN poetry install --without dev --no-ansi \
  && rm -rf /tmp/pff-poetry-cache /root/.cache/pypoetry
RUN rm -f /app/src/pff_rust/_pff_rust*.so
RUN /app/.venv/bin/pip install "maturin==1.11.5"
RUN . /app/.venv/bin/activate && maturin build --release --manifest-path /app/src/pff_rust/Cargo.toml --out /tmp/pff-rust-dist
RUN /app/.venv/bin/pip install /tmp/pff-rust-dist/pff_rust-*.whl
RUN find /app/.venv/lib -path '*/site-packages/pff_rust/_pff_rust*.so' -exec cp {} /app/src/pff_rust/ \;

RUN case "${PFF_ACCELERATOR}" in \
  cpu) \
  /app/.venv/bin/python -c "import torch; raise SystemExit(0 if torch.version.cuda is None else 1)" ;; \
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
  HOME=/tmp/pff-home \
  PATH="/app/.venv/bin:${PATH}" \
  PFF_ENV=production \
  PFF_ACCELERATOR=${PFF_ACCELERATOR} \
  TRITON_CACHE_DIR=/tmp/pff-home/.cache/triton \
  XDG_CACHE_HOME=/tmp/pff-home/.cache

WORKDIR /app

RUN groupadd -r pff && useradd -r -g pff pff && usermod -d /app pff

COPY --from=builder --chown=pff:pff /app/.venv /app/.venv
COPY --from=builder --chown=pff:pff /app/src/pff /app/src/pff
COPY --from=builder --chown=pff:pff /app/config /app/config
COPY --from=builder --chown=pff:pff /app/README.md /app/pyproject.toml /app/poetry.lock /app/poetry.toml /app/

RUN mkdir -p /app/data /app/logs /app/outputs /tmp/pff-home \
  && chown pff:pff /app/data /app/logs /app/outputs \
  && chmod 1777 /tmp/pff-home

USER pff

ENTRYPOINT ["pff"]
CMD ["--help"]

FROM runtime-base AS runtime

FROM runtime-base AS runtime-cpu
ENV PFF_ACCELERATOR=cpu

FROM runtime-base AS runtime-cuda
USER root
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
  && rm -rf /var/lib/apt/lists/*
USER pff
ENV CC=gcc \
  PFF_ACCELERATOR=cuda

FROM builder AS tools

ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PATH="/app/.venv/bin:${PATH}" \
  PFF_ENV=development \
  PYTHONPATH=/app:/app/src

RUN poetry install --with dev --no-ansi \
  && rm -rf /tmp/pff-poetry-cache /root/.cache/pypoetry

ENTRYPOINT ["bash"]
CMD ["-lc", "pff --help"]

FROM tools AS test

ARG DOCKER_CLI_VERSION=28.0.4
ARG DOCKER_COMPOSE_VERSION=2.39.2

ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PATH="/app/.venv/bin:${PATH}" \
  PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
  PFF_ENV=test \
  PYTHONPATH=/app:/app/src

RUN curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_CLI_VERSION}.tgz" \
  -o /tmp/docker.tgz \
  && tar -xzf /tmp/docker.tgz -C /tmp \
  && mv /tmp/docker/docker /usr/local/bin/docker \
  && rm -rf /tmp/docker /tmp/docker.tgz

RUN mkdir -p /usr/local/lib/docker/cli-plugins \
  && curl -fsSL "https://github.com/docker/compose/releases/download/v${DOCKER_COMPOSE_VERSION}/docker-compose-linux-x86_64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose \
  && chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

RUN mkdir -p "${PLAYWRIGHT_BROWSERS_PATH}" \
  && /app/.venv/bin/python -m playwright install --with-deps chromium \
  && chmod -R a+rX "${PLAYWRIGHT_BROWSERS_PATH}"

ENTRYPOINT ["pytest"]
CMD ["-q"]
