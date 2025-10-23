# PFF - Deployment Guide

**Version:** 1.0.0
**Last Updated:** 2025-10-22
**Target:** Production deployment with Docker + PostgreSQL + Redis

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (Local Development)](#quick-start-local-development)
3. [Production Deployment](#production-deployment)
4. [Environment Configuration](#environment-configuration)
5. [Database Migrations](#database-migrations)
6. [Health Checks](#health-checks)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### System Requirements

| Component | Minimum | Recommended (Production) |
|-----------|---------|--------------------------|
| **OS** | Linux (Ubuntu 20.04+) | Linux (Ubuntu 22.04 LTS) |
| **Docker** | 20.10+ | 24.0+ |
| **Docker Compose** | 2.0+ | 2.20+ |
| **RAM** | 8 GB | 16 GB (mid_spec) or 32 GB (high_spec) |
| **CPU** | 4 cores | 12+ cores |
| **Disk** | 50 GB SSD | 200 GB NVMe SSD |
| **GPU** | None (optional) | NVIDIA RTX 3070 Ti (CUDA 11.8+) |

### Software Dependencies

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Verify installation
docker --version  # Docker version 24.0+
docker compose version  # Docker Compose version 2.20+
```

---

## 🚀 Quick Start (Local Development)

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/pff.git
cd pff
```

### Step 2: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Generate secure secrets
python -c "import secrets; print(f'SECRET_KEY={secrets.token_urlsafe(32)}')" >> .env
python -c "import secrets; print(f'API_KEY={secrets.token_urlsafe(32)}')" >> .env

# Edit .env file with your configuration
nano .env
```

**Minimum `.env` Configuration:**

```bash
# Database
POSTGRES_USER=pff_user
POSTGRES_PASSWORD=<generate-secure-password>
POSTGRES_DB=pff_production

# Security
SECRET_KEY=<generated-in-step-above>
API_KEY=<generated-in-step-above>

# Application
PFF_ENV=production
LOG_LEVEL=INFO
HARDWARE_PROFILE=mid_spec  # low_spec, mid_spec, high_spec

# Redis
USE_REDIS=true
```

### Step 3: Start Services

```bash
# Build and start all services
docker compose up -d

# Check logs
docker compose logs -f

# Expected output:
# ✅ pff-postgres   | database system is ready to accept connections
# ✅ pff-redis      | Ready to accept connections
# ✅ pff-api        | Application startup complete
```

### Step 4: Run Database Migrations

```bash
# Access API container
docker compose exec api bash

# Run migrations
alembic upgrade head

# Verify migrations
alembic current
# Expected: head (5 migrations applied)

# Exit container
exit
```

### Step 5: Verify Deployment

```bash
# Basic health check
curl http://localhost:8000/health
# Expected: {"status":"ok"}

# Detailed health check
curl http://localhost:8000/health/detailed
# Expected: {"status":"healthy","checks":{"postgres":{"status":"healthy"},...}}

# API root
curl http://localhost:8000/
# Expected: {"message":"PFF API","version":"..."}
```

---

## 🏭 Production Deployment

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Load Balancer                        │
│                      (nginx/Traefik)                         │
└────────────────┬────────────────────────────────────────────┘
                 │
     ┌───────────┴───────────┬─────────────────┐
     │                       │                 │
┌────▼─────┐         ┌───────▼────┐     ┌─────▼────┐
│  PFF API │         │  PFF API   │     │ PFF API  │
│ (Docker) │         │  (Docker)  │     │ (Docker) │
└────┬─────┘         └──────┬─────┘     └────┬─────┘
     │                      │                 │
     └──────────────────────┴─────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
    ┌─────▼────┐      ┌────▼─────┐    ┌─────▼────┐
    │PostgreSQL│      │  Redis   │    │  Celery  │
    │   16     │      │    7     │    │  Worker  │
    └──────────┘      └──────────┘    └──────────┘
```

### Production docker-compose.yml

**Already created at:** `docker-compose.yml`

**Features:**
- ✅ Multi-stage Dockerfile (builder + runtime)
- ✅ PostgreSQL 16 with pgvector
- ✅ Redis 7 with LRU eviction
- ✅ Celery worker for background tasks
- ✅ Health checks on all services
- ✅ Resource limits (CPU/RAM)
- ✅ Persistent volumes
- ✅ Network isolation

### Production Deployment Steps

**On Production Server:**

```bash
# 1. Clone repository
git clone https://github.com/your-org/pff.git
cd pff

# 2. Configure production .env (use strong passwords!)
cp .env.example .env
nano .env

# 3. Build images (takes ~5 minutes)
docker compose build --no-cache

# 4. Start services
docker compose up -d

# 5. Run migrations
docker compose exec api alembic upgrade head

# 6. Verify deployment
curl http://localhost:8000/health/detailed

# 7. Check logs
docker compose logs -f api
```

---

## ⚙️ Environment Configuration

### Required Environment Variables

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `SECRET_KEY` | JWT secret key | `secrets.token_urlsafe(32)` | ✅ Yes |
| `API_KEY` | API authentication key | `secrets.token_urlsafe(32)` | ✅ Yes |
| `POSTGRES_USER` | PostgreSQL username | `pff_user` | ✅ Yes |
| `POSTGRES_PASSWORD` | PostgreSQL password | `<secure-password>` | ✅ Yes |
| `POSTGRES_DB` | PostgreSQL database name | `pff_production` | ✅ Yes |

### Optional Environment Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `PFF_ENV` | Environment mode | `production` | `development`, `production` |
| `LOG_LEVEL` | Logging level | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `HARDWARE_PROFILE` | Hardware configuration | `mid_spec` | `low_spec`, `mid_spec`, `high_spec` |
| `USE_REDIS` | Enable Redis | `true` | `true`, `false` |
| `REDIS_URL` | Redis connection URL | `redis://redis:6379/0` | URL string |

### Hardware Profiles

**Automatically detected by `hardware_detector.py`, but can be overridden:**

```bash
# Low spec (8GB RAM, no GPU)
HARDWARE_PROFILE=low_spec

# Mid spec (16GB RAM, optional GPU)
HARDWARE_PROFILE=mid_spec  # DEFAULT

# High spec (32GB RAM, RTX 3070 Ti)
HARDWARE_PROFILE=high_spec
```

---

## 🗄️ Database Migrations

### Initial Setup

```bash
# Initialize alembic (already done in repo)
alembic init migrations

# Generate first migration
alembic revision --autogenerate -m "initial schema"

# Apply migrations
alembic upgrade head
```

### Common Migration Commands

```bash
# Check current migration
docker compose exec api alembic current

# Upgrade to latest
docker compose exec api alembic upgrade head

# Downgrade one version
docker compose exec api alembic downgrade -1

# Show migration history
docker compose exec api alembic history

# Create new migration
docker compose exec api alembic revision --autogenerate -m "add new table"
```

### Database Backup/Restore

```bash
# Backup
docker compose exec postgres pg_dump -U pff_user pff_production > backup_$(date +%Y%m%d).sql

# Restore
docker compose exec -T postgres psql -U pff_user pff_production < backup_20251022.sql
```

---

## 🏥 Health Checks

### Basic Health Check

```bash
curl http://localhost:8000/health

# Response:
{
  "status": "ok"
}
```

### Detailed Health Check

```bash
curl http://localhost:8000/health/detailed

# Response:
{
  "status": "healthy",
  "timestamp": "2025-10-22T01:50:00.000Z",
  "checks": {
    "postgres": {
      "status": "healthy",
      "message": "Connected"
    },
    "redis": {
      "status": "healthy",
      "message": "Connected"
    }
  },
  "response_time_ms": 45.2
}
```

### Docker Health Checks

```bash
# Check container health
docker compose ps

# Expected:
# pff-api        healthy
# pff-postgres   healthy
# pff-redis      healthy
# pff-celery     running
```

---

## 📊 Monitoring

### Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api

# Last 100 lines
docker compose logs --tail=100 api

# Since timestamp
docker compose logs --since 2025-10-22T00:00:00 api
```

### Metrics

**Container Stats:**

```bash
# Real-time stats
docker stats

# Expected output:
# CONTAINER      CPU %    MEM USAGE / LIMIT    MEM %    NET I/O
# pff-api        5.2%     2.1GiB / 8GiB       26%      100MB / 50MB
# pff-postgres   1.8%     800MiB / 4GiB       20%      50MB / 100MB
# pff-redis      0.5%     200MiB / 2GiB       10%      10MB / 5MB
```

### Prometheus + Grafana (Future)

**TODO (Sprint 11):**
- `/metrics` endpoint (Prometheus format)
- Grafana dashboards
- Alert rules

---

## 🔍 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs api

# Common issues:
# 1. Missing .env file
cp .env.example .env

# 2. Port already in use
sudo lsof -i :8000  # Find process using port
sudo kill -9 <PID>  # Kill process

# 3. Database not ready
docker compose up postgres  # Start postgres first
docker compose up -d  # Then start all
```

### Database Connection Failed

```bash
# Test PostgreSQL connection
docker compose exec postgres psql -U pff_user -d pff_production

# If fails:
# 1. Check credentials in .env
# 2. Wait for postgres to be ready (30s+)
docker compose logs postgres | grep "ready to accept connections"
```

### Redis Connection Failed

```bash
# Test Redis connection
docker compose exec redis redis-cli ping
# Expected: PONG

# If USE_REDIS=false, ignore Redis errors (fallback to in-memory)
```

### Build Fails

```bash
# Clear Docker cache
docker compose down
docker system prune -a --volumes
docker compose build --no-cache
```

### Performance Issues

```bash
# Check resource usage
docker stats

# If high CPU/RAM:
# 1. Adjust HARDWARE_PROFILE in .env
HARDWARE_PROFILE=low_spec  # Reduces workers

# 2. Limit Docker resources in docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

---

## 📝 Next Steps

**After deployment:**

1. ✅ Configure reverse proxy (nginx/Traefik)
2. ✅ Setup SSL/TLS certificates (Let's Encrypt)
3. ✅ Configure monitoring (Prometheus + Grafana)
4. ✅ Setup automated backups
5. ✅ Configure CI/CD pipelines (GitHub Actions)

**Reference:**
- CI/CD: `.github/workflows/ci.yml`
- Monitoring: Sprint 11 (Prometheus + Grafana)
- Backups: `scripts/backup.sh` (TODO)

---

**Support:** https://github.com/your-org/pff/issues
**Documentation:** https://docs.pff.ai
**Version:** 1.0.0 (Sprint 10 - DevOps)
