# PFF - Production Fix Flow - Multi-Stage Dockerfile
# Build: docker build -t pff:latest .
# Run: docker-compose up -d

# ============================================================================
# Stage 1: Builder - Install dependencies and build
# ============================================================================
FROM python:3.13-slim AS builder

LABEL maintainer="PFF Team"
LABEL description="PFF - Production Fix Flow AI/ML Pipeline (Builder Stage)"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.8.5 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_CREATE=true

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install "poetry==${POETRY_VERSION}"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock poetry.toml ./

# Install dependencies (without dev deps)
RUN poetry install --only main --no-root --no-interaction --no-ansi

# Copy application code
COPY . .

# Install the application
RUN poetry install --only main --no-interaction --no-ansi

# ============================================================================
# Stage 2: Runtime - Lightweight production image
# ============================================================================
FROM python:3.13-slim AS runtime

LABEL maintainer="PFF Team"
LABEL description="PFF - Production Fix Flow AI/ML Pipeline (Runtime)"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PFF_ENV=production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    libmimalloc2.0 \
    && ln -s /usr/lib/x86_64-linux-gnu/libmimalloc.so.2 /usr/lib/libmimalloc.so \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r pff && useradd -r -g pff pff

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PFF_ENV=production \
    LD_PRELOAD=/usr/lib/libmimalloc.so \
    MIMALLOC_LARGE_OS_PAGES=1


# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command: Run API server
CMD ["uvicorn", "pff.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
