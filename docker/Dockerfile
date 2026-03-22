# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – build / install dependencies with uv
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install uv (pinned version to prevent supply-chain drift)
COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency manifests first (cache-friendly layer)
COPY pyproject.toml uv.lock ./
# uv needs the package source to resolve the project metadata
COPY src/ ./src/

# Install only production dependencies into a virtual environment
RUN uv sync --no-dev --frozen --no-cache

# Pre-compile bytecode so the runtime image starts faster
RUN .venv/bin/python -m compileall src/

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – lean runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="tcc-etl" \
      org.opencontainers.image.description="High-performance Python ETL pipeline" \
      org.opencontainers.image.licenses="MIT"

# Non-root user for security
RUN groupadd --gid 1001 appgroup \
 && useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy the pre-built venv and compiled source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Make sure the venv is on PATH
# PYTHONDONTWRITEBYTECODE is intentionally omitted: bytecode is pre-compiled
# in the builder stage, allowing fast startup in the immutable container.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser

# Lightweight liveness check: verify the package imports correctly
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import tcc_etl; print('ok')" || exit 1

ENTRYPOINT ["tcc-etl"]
