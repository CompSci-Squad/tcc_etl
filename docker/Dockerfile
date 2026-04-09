FROM public.ecr.aws/lambda/python:3.12-arm64

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

RUN useradd --no-create-home --shell /sbin/nologin appuser
USER appuser

CMD ["main.handler"]

