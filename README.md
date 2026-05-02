# tcc_etl

FRED-MD ETL pipeline used by the TCC (undergraduate thesis) project. In production
it runs as an AWS Lambda container and writes parquet artifacts to S3 on a monthly
schedule. This README focuses on running the same pipeline **locally**, on your
machine, with no AWS account, no Docker, and no LocalStack, so you can inspect
the resulting data with any parquet reader.

---

## What the pipeline does

On every run the pipeline:

1. **Extracts** the latest monthly FRED-MD CSV from the St. Louis Fed website
   (`https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/<YYYY-MM>-md.csv`).
   The CSV is roughly 130 series of US macroeconomic data observed monthly since
   1959. The pipeline always asks for the **previous** calendar month's vintage
   (e.g. running on 2026-05-02 fetches the `2026-04-md.csv` file).
2. **Removes outliers**: replaces values outside `median plus or minus 10 * IQR`
   with NaN, per series.
3. **Applies McCracken-Ng `tcodes`** to make every series stationary
   (levels, first/second differences, logs, log-differences, etc.). The first
   two rows of the panel are dropped because they are consumed by the
   second-difference transforms.
4. **Imputes** the remaining missing values with **EM-PCA** (Stock and Watson,
   2002). Series with more than `IMPUTE_MAX_MISSING_FRAC` (default 0.5) of
   missing observations are dropped. The imputer also returns a Boolean **mask**
   of the same shape as the panel, where `True` means "this cell was originally
   NaN and was filled in by the imputer".
5. **Builds a per-series data card** (first/last observation date, missing
   counts split into leading vs internal, drop reason if any) and a **balanced
   subpanel** containing only series whose first observation is on or before
   `BALANCED_CUTOFF_DATE` (default 1965-01-01).
6. **Validates** every output dataframe against a Pandera schema and writes
   parquet files (zstd compression).

---

## Output files

When you run the local script, the pipeline writes the following files into
the output directory (default `./out/`):

| File                                    | What it is                                                                      |
|-----------------------------------------|----------------------------------------------------------------------------------|
| `fred_md_raw.parquet`                   | Outlier-cleaned panel **before** tcode transforms (one row per month).           |
| `fred_md_transformed.parquet`           | Stationary, imputed panel ready for modelling. One column per kept series.       |
| `fred_md_mask.parquet`                  | Boolean sidecar, same shape as the transformed panel. `True` = imputed cell.     |
| `fred_md_data_card.parquet`             | One row per input series with audit metadata (first/last obs, missing counts).   |
| `fred_md_validation.parquet`            | Per-series stationarity report (ADF p-value, null rate, sparsity flags).         |
| `fred_md_transformed_balanced.parquet`  | Subset of `fred_md_transformed.parquet` restricted to "early-start" series.      |
| `fred_md_mask_balanced.parquet`         | Mask matching the balanced transformed file.                                     |
| `fred_md_imputation.json`               | Imputation report: kept/dropped series, EM iterations, fractions imputed, etc.   |

The two `_balanced` files are only written if at least one series qualifies
under `BALANCED_CUTOFF_DATE`.

---

## Prerequisites

- Python 3.12 or newer.
- [`uv`](https://docs.astral.sh/uv/) version 0.5 or newer.
- An internet connection (the pipeline downloads the FRED-MD CSV at runtime;
  it is roughly 1 MB).

You do **not** need:

- An AWS account, AWS CLI, or AWS credentials.
- Docker, AWS SAM, or LocalStack.
- A pre-existing S3 bucket.

The local entry point never calls `boto3` and never opens a network connection
to AWS.

---

## First-time setup

From the repository root:

```bash
uv sync
```

This installs the runtime dependencies (`polars`, `numpy`, `httpx`, `pandera`,
`statsmodels`, `pyarrow`, `boto3`, `blake3`) and the dev dependencies
(`pytest`, `ruff`, `taskipy`, etc.) into a `.venv/` managed by `uv`.

You only need to do this once, or after pulling changes to `pyproject.toml` or
`uv.lock`.

---

## Running the pipeline locally

There is one command:

```bash
uv run task run-local
```

Internally this runs:

```bash
S3_BUCKET=local uv run python -c \
  'from tcc_etl.main import run_local; import json, sys; \
   sys.stdout.write(json.dumps(run_local("out"), indent=2, default=str) + "\n")'
```

The `S3_BUCKET=local` value is a placeholder. The Lambda module reads
`S3_BUCKET` at import time, but the local entry point ignores it and writes
directly to the filesystem instead.

The script:

1. Creates `./out/` if it does not exist (existing files in that directory
   are **overwritten**, but never deleted, so old artifacts from a previous run
   may stick around if their filenames change).
2. Downloads the latest FRED-MD vintage.
3. Runs extract -> outlier removal -> tcode transforms -> EM-PCA imputation ->
   data card -> validation.
4. Validates every dataframe against its Pandera schema. If validation fails,
   the script exits with a non-zero status and **no parquet files are written**
   for the failing artifact.
5. Writes the parquet files and `fred_md_imputation.json` listed above into
   `./out/`.
6. Prints a JSON summary to stdout, for example:

```json
{
  "output_dir": "/abs/path/to/tcc_etl/out",
  "rows": 798,
  "series_input": 127,
  "series_kept": 124,
  "series_dropped": 3,
  "balanced_n_series": 100,
  "frac_imputed": 0.0142,
  "frac_imputed_leading": 0.0091,
  "frac_imputed_internal": 0.0051,
  "em_iter": 17,
  "em_converged": true,
  "files": ["fred_md_data_card.parquet", "fred_md_imputation.json", ...]
}
```

A typical run on a laptop takes 10-30 seconds, dominated by the HTTPS download
and the per-series ADF stationarity tests in the validation step.

---

## Tuning the run

Override these environment variables on the same command line if you want
non-default behaviour:

| Variable                  | Default      | Effect                                                                                            |
|---------------------------|--------------|----------------------------------------------------------------------------------------------------|
| `FRED_MD_VINTAGE`         | `2026-03` (via the `run-local` task) | Pin the FRED-MD CSV vintage as `YYYY-MM`. Unset in production; the Lambda uses the previous calendar month. |
| `IMPUTE_K`                | `8`          | Number of latent factors used by EM-PCA imputation.                                                |
| `IMPUTE_MAX_MISSING_FRAC` | `0.5`        | Series with more than this fraction of missing observations are dropped before imputation.         |
| `BALANCED_CUTOFF_DATE`    | `1965-01-01` | Series whose first observation is on or before this date go into the `_balanced` parquet outputs.  |
| `PIPELINE_VERSION`        | `v2`         | Stamped into `fred_md_imputation.json` for downstream traceability.                                |

Example:

```bash
IMPUTE_K=12 IMPUTE_MAX_MISSING_FRAC=0.33 uv run task run-local
```

Pin a different vintage:

```bash
FRED_MD_VINTAGE=2026-02 uv run task run-local
```

To write to a different directory, call the function directly:

```bash
S3_BUCKET=local uv run python -c \
  'from tcc_etl.main import run_local; print(run_local("/tmp/fred_md_out"))'
```

---

## Inspecting the output

The output is plain Parquet, readable from any tool that understands Arrow.
Two quick options:

**Polars (recommended, since it is already installed):**

```bash
uv run python -c \
  'import polars as pl; print(pl.read_parquet("out/fred_md_transformed.parquet").head())'
```

**Pandas:**

```bash
uv run python -c \
  'import pandas as pd; print(pd.read_parquet("out/fred_md_transformed.parquet").head())'
```

**DuckDB CLI** (if installed):

```bash
duckdb -c "SELECT * FROM 'out/fred_md_transformed.parquet' LIMIT 5;"
```

The mask file uses Boolean columns. To find which transformed cells were
imputed:

```python
import polars as pl
panel = pl.read_parquet("out/fred_md_transformed.parquet")
mask = pl.read_parquet("out/fred_md_mask.parquet")
imputed_count_per_series = mask.drop("date").sum()
print(imputed_count_per_series)
```

To see why a particular series was dropped, look in the data card:

```python
import polars as pl
card = pl.read_parquet("out/fred_md_data_card.parquet")
print(card.filter(~pl.col("kept")).select("series_id", "drop_reason", "frac_missing"))
```

---

## Troubleshooting

**`KeyError: 'S3_BUCKET'` on import.** The Lambda module reads `S3_BUCKET` at
import time. You must prefix the command with `S3_BUCKET=local` (the
`run-local` task already does this).

**`httpx.HTTPStatusError: 404` or schema/parsing errors on the CSV.** The
St. Louis Fed has not yet published the vintage you asked for (they release
on the 20th of each month). Either wait a few days or pin an older one
explicitly with `FRED_MD_VINTAGE=YYYY-MM`. The local task already pins
`2026-03` by default for this reason.

**Schema validation fails on one of the parquet outputs.** This means the
upstream FRED-MD CSV violated an assumption (e.g. duplicate `sasdate`). The
script aborts before writing the offending file. Re-run later or open an
issue.

**Out of memory.** Unlikely on a laptop. The full panel is roughly
800 rows by 130 columns of `float64`, well under 10 MB in memory. If it does
happen, lower `IMPUTE_K`, which reduces SVD memory.

---

## Other tasks

These are unrelated to local data generation but are listed for convenience:

```bash
uv run task test         # full test suite with coverage gate (80%)
uv run task test-fast    # tests only, no coverage, stop on first failure
uv run task lint         # ruff check
uv run task fmt          # ruff format
uv run task typecheck    # pyrefly
uv run task all          # fmt + lint + typecheck + test
```

The AWS-related tasks (`sam-*`, `local-down`, `local-logs`) and the `Makefile`
targets exist for production deployment of the Lambda and are not needed for
local data generation.

---

## Project layout

```
src/tcc_etl/
  extract.py      Async download and parsing of the FRED-MD CSV.
  transform.py    Outlier removal and McCracken-Ng tcode transforms.
  imputation.py   EM-PCA imputer and the LazyFrame wrapper.
  data_card.py    Per-series audit table and balanced-subpanel selection.
  loader.py       Pandera schemas, parquet writers, ADF validation report.
  main.py         Lambda handler, plus the local run_local() entry point.
tests/            pytest suite (uses moto to mock S3).
docker/           Lambda container image (production deployment only).
template.yaml,
template.local.yaml,
samconfig.toml    AWS SAM templates (production deployment only).
docker-compose.yml LocalStack for SAM-based local Lambda invocation only.
```

## License

MIT. See [LICENSE](LICENSE).
