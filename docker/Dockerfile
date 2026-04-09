# AWS Lambda ARM64 (Graviton) container image
# Triggered by EventBridge Scheduler: cron(0 22 ? * MON-FRI *) UTC
#
# Build for ARM64:
#   docker buildx build --platform linux/arm64 -f docker/Dockerfile -t tcc-etl-lambda .
#
# Local smoke test via Lambda Runtime Interface Emulator:
#   docker run --platform linux/arm64 -p 9000:8080 --env-file .env tcc-etl-lambda
#   curl -XPOST http://localhost:9000/2015-03-31/functions/function/invocations -d '{}'
# ─────────────────────────────────────────────────────────────────────────────
FROM public.ecr.aws/lambda/python:3.12-arm64

# Install production dependencies
# boto3 is included in the Lambda base image but pinned here for reproducibility
RUN pip install --no-cache-dir \
    polars \
    httpx \
    numpy \
    "jax[cpu]>=0.4" \
    "jaxlib>=0.4" \
    statsmodels \
    "pyarrow>=15.0" \
    boto3 \
    "pandera[polars]>=0.20"

# Copy only the package source — no tests, no config files
COPY src/tcc_etl/ ${LAMBDA_TASK_ROOT}/

# Lambda invokes: main.handler (main.py → handler function)
CMD ["main.handler"]

