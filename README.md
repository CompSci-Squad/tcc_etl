# tcc_etl

> **High-performance Python ETL pipeline** - Extract data from any source,
> transform it with Numba-accelerated kernels, and load it into AWS S3.

[![CI](https://github.com/CompSci-Squad/tcc_etl/actions/workflows/ci.yml/badge.svg)](https://github.com/CompSci-Squad/tcc_etl/actions/workflows/ci.yml)
[![CD](https://github.com/CompSci-Squad/tcc_etl/actions/workflows/cd.yml/badge.svg)](https://github.com/CompSci-Squad/tcc_etl/actions/workflows/cd.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Development](#development)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Configuration](#configuration)

---

## Features

| Feature | Details |
|---------|---------|
| **ETL pipeline** | Extract -> Transform -> Load pattern with clean abstractions |
| **Performance** | Numba JIT-compiled numeric kernels (falls back to NumPy on PyPy) |
| **S3 output** | Parallel uploads with `ThreadPoolExecutor`, Parquet or CSV format |
| **Type safety** | Fully typed with `pyrefly` |
| **Linting** | `ruff` for linting and formatting |
| **Testing** | `pytest` + `moto` S3 mock, 80%+ coverage enforced |
| **Task runner** | `taskipy` one-liners for all common tasks |
| **Docker** | Multi-stage CPython (amd64 + arm64) and PyPy arm64 images |
| **CI/CD** | GitHub Actions - lint, typecheck, test, build, push |

---

## Project Structure

```
tcc_etl/
- src/tcc_etl/
  - config/           # Pydantic-settings configuration
  - extract/          # BaseExtractor, HttpExtractor, FileExtractor
  - transform/        # BaseTransformer, DataFrameTransformer, Numba kernels
  - load/             # BaseLoader, S3Loader (parallel uploads)
  - pipeline.py       # Pipeline orchestrator (extract -> transform -> load)
  - main.py           # CLI entry point
- tests/              # pytest test suite (moto S3 mock)
- docker/
  - Dockerfile        # CPython multi-stage (amd64 + arm64)
  - Dockerfile.pypy   # PyPy slim (arm64)
- docker-compose.yml  # Local dev with LocalStack
- pyproject.toml      # uv, ruff, pyrefly, taskipy, pytest config
- .github/workflows/
  - ci.yml            # Lint + typecheck + test
  - cd.yml            # Build + push Docker images to GHCR
```

---

## Getting Started

### Prerequisites

- [uv](https://docs.astral.sh/uv/) >= 0.5
- Python 3.12+

### Installation

```bash
# Clone the repo
git clone https://github.com/CompSci-Squad/tcc_etl.git
cd tcc_etl

# Install all dependencies (including dev)
uv sync

# Run the ETL pipeline
uv run tcc-etl
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region |
| `AWS_ACCESS_KEY_ID` | `` | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | `` | AWS credentials |
| `AWS_ENDPOINT_URL` | `None` | Custom endpoint (e.g. LocalStack) |
| `S3_BUCKET_NAME` | `tcc-etl-bucket` | Target S3 bucket |
| `S3_PREFIX` | `data/` | Key prefix |
| `SOURCE_URL` | *(README URL)* | Data source URL |
| `LOG_LEVEL` | `INFO` | Log level |
| `BATCH_SIZE` | `10000` | Records per batch |
| `MAX_WORKERS` | `4` | Parallel upload threads |

---

## Development

All development tasks are managed via **taskipy**:

```bash
# Format code
uv run task fmt

# Lint (ruff check)
uv run task lint

# Check formatting without modifying files
uv run task fmt-check

# Type check (pyrefly)
uv run task typecheck

# Run tests with coverage
uv run task test

# Run tests quickly (no coverage, stop on first failure)
uv run task test-fast

# Full CI check: lint + format-check + test
uv run task ci

# Format + lint + typecheck + test all in one
uv run task all
```

---

## Docker

### CPython (amd64 + arm64)

```bash
# Build
docker build -f docker/Dockerfile -t tcc-etl .

# Run
docker run --rm \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e S3_BUCKET_NAME=my-bucket \
  tcc-etl
```

### PyPy (arm64 only)

```bash
docker buildx build \
  --platform linux/arm64 \
  -f docker/Dockerfile.pypy \
  -t tcc-etl-pypy .
```

### Local Development with LocalStack

```bash
docker compose up
```

This starts LocalStack (S3 mock) and the ETL container connected to it.

---

## CI/CD

### CI (`ci.yml`) - runs on every push and PR

1. **Lint** - `ruff check` and `ruff format --check`
2. **Type check** - `pyrefly check`
3. **Test** - `pytest` on Python 3.12 and 3.13 with coverage report

### CD (`cd.yml`) - runs on push to `main` or version tags

1. Build CPython image for `linux/amd64` and `linux/arm64`
2. Build PyPy image for `linux/arm64`
3. Push both images to GHCR (`ghcr.io/compsci-squad/tcc_etl`)

Images are tagged with branch name, semantic version, and commit SHA.

---

## Architecture

```
Source (HTTP / File)
        |
        v
  [BaseExtractor]
        |
  raw data (list[dict] / bytes)
        |
        v
  [BaseTransformer]  <-- Numba JIT kernels for numeric columns
        |
  pd.DataFrame / bytes / JSON
        |
        v
  [S3Loader]  <-- parallel uploads via ThreadPoolExecutor
        |
        v
  s3://bucket/prefix/output.parquet
```
