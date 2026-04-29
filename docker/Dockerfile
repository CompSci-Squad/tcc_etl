ARG LAMBDA_ARCH=arm64
FROM public.ecr.aws/lambda/python:3.12-${LAMBDA_ARCH}

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies declared in pyproject.toml (no package build, no dist-info needed).
COPY pyproject.toml ${LAMBDA_TASK_ROOT}/
RUN uv pip install --system --no-cache \
    polars \
    httpx \
    numpy \
    "jax[cpu]>=0.4" \
    "jaxlib>=0.4" \
    statsmodels \
    "pyarrow>=15.0" \
    boto3 \
    blake3 \
    "pandera[polars]>=0.20"

# Copy only the package source — no build step needed.
COPY src/tcc_etl/ ${LAMBDA_TASK_ROOT}/tcc_etl/

# hadolint ignore=DL3002
CMD ["tcc_etl.main.handler"]

