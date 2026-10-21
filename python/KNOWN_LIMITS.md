# Known limits for Veridist 2.1

This document defines the 2.1 release boundary for package version `2.1.0`.

## What these limits mean in common applications

- Reliability, health, credit, insurance, digital-product, and operations teams can model one clearly defined time-to-event outcome when the documented assumptions apply. The current models do not adjust that outcome for customer, patient, machine, or environmental characteristics.
- Fraud and cybersecurity teams can use supported scalar distribution calculations to create a signal under an already specified reference model. Veridist does not train a classifier, select an alert threshold, process feedback labels, or supply a production event-stream adapter.
- Finance, insurance, manufacturing, and supply-chain teams can fit any of the six registered families to their observations. A family still has to be justified separately: fitting and uncertainty reporting do not rank families, and goodness-of-fit tests and adequacy-gated model selection cover only finite, uncensored samples (`INFERENCE-GOF`).
- In every field, model output still needs domain validation, appropriate sampling, decision-cost analysis, and any required legal, clinical, safety, or regulatory review.

- `FIT-CSV-EXP`: the strict CSV path fits only a fixed-location, rate-only
  exponential model over exact and independently right-censored lifetimes.
  The other families (Weibull-minimum, lognormal, gamma, normal, and right-Gumbel) are fitted over typed observations held in memory through `fit` and the per-family functions, not through the CSV path; for exact data that is too large for memory, `veridist.scale` fits all six families (`SCALE-ENGINE`). `frequency_weights` are supported in memory only. Analytic weights, covariates,
  truncation, left censoring, interval censoring, and free location parameters
  remain unsupported.
- `CSV-STRICT`: the bundled file adapter accepts only UTF-8 CSV with exactly
  `time,event_observed`, where `1` is an exact event and `0` is independent
  right censoring. It is not a general CSV or spreadsheet reader. A blank
  record is tolerated only when it is the last thing in the file (for
  example a trailing blank line an editor or spreadsheet added); a blank
  record anywhere else is still a `blank_record` failure.
- `SCALAR-FAMILIES`: normal, gamma, Weibull-minimum, lognormal, right-Gumbel,
  and exponential expose log-density, CDF, survival, quantile, and sampling
  operations. `logpdf`, `cdf`, `sf`, and `ppf` also evaluate numpy arrays,
  broadcasting the point against array-valued parameters; scalar input still
  returns a Python `float`. The exponential, Weibull-minimum, and right-Gumbel
  families use numpy-native kernels. The normal, lognormal, and gamma families
  wrap the verified scalar kernels element by element: the results equal the
  scalar path exactly, but large arrays are slow, because numpy has no `erfc`
  or incomplete gamma function and Veridist has no scipy runtime dependency.
  Goodness-of-fit tests and model selection are described under `INFERENCE-GOF`.
- `STREAM-SOURCE`: `IterableDataSource` adapts caller-owned chunk iterables.
  Apart from the CSV adapter, the package bundles no network adapter, and the
  sources of `veridist.scale` are described under `SCALE-ENGINE`. Durable
  resume is limited to the strict lifetime CSV path and the partition
  checkpoints of `veridist.scale`, both on local SQLite on one host; it is not
  a distributed checkpoint store. For
  `fit_exponential_checkpointed_csv`, the source revision must equal the CSV
  file's current SHA-256 hex digest, checked against the file on disk before
  any row is read; a changed file, a different public source identifier, or a
  different stored schema returns a typed mismatch instead of resuming.
  `fit_exponential_checkpointed_chunks` should be called with the offset form
  `(row_start, payload)` so a replayed chunk is recognized by its row range
  and skipped instead of being applied a second time; the legacy bare-`bytes`
  form is deprecated, emits a warning, and cannot generally detect a replay.
- `INFERENCE-GOF`: refit Monte Carlo KS/AD/CvM goodness-of-fit
  (`refit_monte_carlo_gof`) and adequacy-gated selection among families
  (`assess_families`) cover all six registered families, for finite, exactly
  observed samples held in memory: strictly positive values for the lifetime
  families, any real value for the normal and right-Gumbel families, and at
  least three observations for every family except the exponential, which needs
  one. Censored observations are rejected with `TypeError`; goodness-of-fit for
  censored data is not available. If the observed sample cannot be fitted,
  `GofFitError` names the failure code and no p-value is computed; a replicate
  whose refit fails is counted in `failed_replicates`, not retried. A p-value is
  a Monte Carlo estimate with a binomial standard error, and the cost is the
  number of replicates times the cost of one fit (an assessment adds one such
  run per family). Both accept at most 100000 observations by default: a
  larger sample raises `RefitGofRefused` (code `SCALE_REFIT_REFUSED`) before
  anything is fitted or drawn, and the limit can be raised with the
  `max_observations` keyword. Passing the adequacy gate means a family was not rejected,
  not that it is the true model, and AIC ranks only the families compared. The
  calibration evidence is a seeded simulation on a declared grid only
  (sample sizes 30 and 100, one or two parameter settings for each
  non-exponential family, nominal level 0.10); nothing is claimed outside it.
  There is no bootstrap selection stability.
- `FIT-UNCERTAINTY`: every fit success reports `result.uncertainty()`: the
  covariance and standard errors from the observed information at the estimate,
  Wald and profile-likelihood confidence intervals (and, for uncensored
  exponential data, the exact chi-square interval), and the mean, quantiles
  (B-lives) and survival probability with intervals. The numbers are
  large-sample results and assume independent right censoring, an interior
  maximum-likelihood estimate and a positive-definite information matrix;
  Wald intervals also need a sample large enough for the likelihood to be
  roughly quadratic. A profile interval can be unbounded on one side (reported
  as `inf`, or `0` for a positive parameter, with a flag). When the information
  is singular or not positive definite, or the Weibull shape was fixed,
  `uncertainty` is an `UncertaintyUnavailable` value with a reason instead of
  numbers. Profile intervals for derived quantities exist only for the
  exponential and Weibull families. Seeded simulations (n = 30 uncensored and
  n = 60 with about 30% right censoring, at least 400 replicates per family and
  setting) gave coverage of the 95% intervals between about 92% and 96%;
  nothing is claimed outside that grid. A fit keeps its observed values (the
  exponential fit only its sufficient statistics) so that intervals can be
  computed on demand.
- `LL-CENSORED`: `reduce_lifetime_log_likelihood_chunks` and
  `reduce_value_log_likelihood_chunks` support independent right censoring
  only. The log-survival terms of censored observations are accurate to about
  `1e-12` relative in their tail expansions and are not covered by the
  log-density oracle envelope. A term that binary64 cannot represent, or a
  gamma shape so large (roughly `1e5` and up) that the incomplete-gamma
  expansion does not converge, is a typed failure rather than a guessed value.
- `DEPRECATED-FORMS`: the mapping form of `cdf`, `sf`, and `ppf`, the
  `sample(family, size, parameters, rng)` form, and bare `bytes` chunks for
  `fit_exponential_checkpointed_chunks` still work, emit
  `DeprecationWarning`, and are removed in 3.0.
- `MEMORY-BOUND`: the delivery bound covers queued payload and active consumer
  leases until explicit release. It is a logical retained-payload bound, not a
  portable RSS ceiling.
- `SCALE-ENGINE`: `veridist.scale` fits the six registered families to exact,
  uncensored, finite observations that are read in bounded batches. Right
  censoring, frequency and analytic weights, truncation, covariates, and free
  location parameters are not supported at scale; use the in-memory fits for
  censored data. The sources are numpy arrays, Parquet files, Arrow tables,
  readers, and streams (`veridist[arrow]`), Polars frames (`veridist[polars]`),
  and DuckDB relations (`veridist[duckdb]`); a source class of your own is not
  part of the public API. Only sources whose partitions have a fingerprint
  (`NumpySource` and `ParquetSource`) can be checkpointed and resumed; an Arrow,
  Polars, or DuckDB source is not resumable (`SOURCE_REVISION_UNAVAILABLE`), and
  a reader or a stream can be read once, so a multi-pass family refuses it
  without a spool (`SPOOL_REQUIRED`). `weibull_min` and `gumbel_right` read the
  source at least 3 and at most 17 times, within a `pass_budget` of 8 by
  default; profile-likelihood intervals, only for those two families, read at
  least 2 and at most 9 further passes within their own budget (6 by default);
  the large-n goodness-of-fit report is one pass and an enclosure of the
  Kolmogorov-Smirnov distance, never a p-value. Checkpoints are local SQLite on
  one host and are tied to the numeric environment; there is no distributed
  executor. A batch holds at most `max_rows` values (262144 by default) and a
  state does not grow with the number of rows; this is a logical bound, not an
  RSS ceiling. Within one numeric environment the results are bit-identical for
  every batch size, partition layout, executor, worker count, spool, and resume
  point; across platforms, Python builds, and numpy versions only agreement
  within the rounding error of `log`, `exp`, and `pow` is expected, and no
  numeric tolerance is declared. Retained measurements of this engine are
  described under `SCALE-EVIDENCE`, and the refit Monte Carlo tests of
  `INFERENCE-GOF` refuse every scale source.
- `SCALE-EVIDENCE`: measurements apply only to their exact adapter, family,
  workload, platform, Python version, chunk limit, and candidate SHA. They do
  not establish universal throughput, generic big-data support, or a broad
  out-of-core capability. The retained 2.2.0 measurements are in the source
  repository's `python/evidence/` directory (they are not part of the
  distribution archives). They were taken on 2026-10-21 from candidate commit
  `e316d125e0ae434beaf0c69ebeee3238c1076f6b` with one measurement worker, on a
  GitHub Actions Linux runner (CPython 3.11.17) and a Windows runner (CPython
  3.11.9), both with 4 vCPUs, and are accepted by
  `tools/check_scale_engine_evidence.py`,
  `tools/check_scale_csv_exponential_evidence.py`, and
  `tools/check_log_likelihood_scale_evidence.py`. For 1,000,000 rows, the strict
  CSV exponential fit took 14.4 to 19.7 seconds on Linux and 15.2 to 21.2
  seconds on Windows across the 32, 64 and 128 KiB chunk limits, with peak
  process memory (RSS) of 40.2 to 43.8 MiB on Linux and 41.3 MiB on Windows. The
  exact-state log-likelihood reducer took 3.1 to 9.2 seconds on Linux and 3.4 to
  11.3 seconds on Windows for one million observations, depending on the family
  (the lognormal family is the slowest), with a peak RSS of 25.98 MiB on Linux
  and 26.25 MiB on Windows. For the scale engine (`veridist.scale`), which
  fitted exponential observations in Parquet files with `max_rows` 262144 and a
  `pass_budget` of 8, the ladder is 1,000,000, 10,000,000, and 100,000,000 rows
  on both runners and 1,000,000,000 rows on Linux only, for the four one-pass
  families only. At 100,000,000 rows on Linux, the one-pass families took 1.3 to
  5.4 seconds with the serial executor and 1.8 to 3.7 seconds with four
  processes, `weibull_min` took 57.9 and 26.4 seconds in 6 passes, and
  `gumbel_right` took 39.8 and 18.7 seconds in 4 passes; on Windows the one-pass
  families took 1.9 to 7.6 seconds serially and 1.8 to 4.5 seconds with four
  processes, `weibull_min` 83.1 and 45.6 seconds, and `gumbel_right` 59.4 and
  32.8 seconds. At 1,000,000,000 rows on Linux, the one-pass families took 13.1
  to 57.5 seconds serially and 12.3 to 30.1 seconds with four processes. The
  peak RSS of the measuring process (the workers are not included) was 221.6 MiB
  in every Linux cell at 100,000,000 and 1,000,000,000 rows, and 78.4 to 97.8
  MiB (serial) and 56.1 to 59.7 MiB (four processes) on Windows at 100,000,000
  rows. For every family and row count, the result, state, and pass digests were
  identical across the serial and process executors, a second partition layout,
  and an interrupted and resumed run. These are descriptive timings of those
  runs, not guarantees.
- `LICENSE`: the package uses BUSL-1.1, which is source-available and not an
  open-source license. The additional use grant in `LICENSE` permits production
  use only for non-commercial purposes (personal use, academic research and
  teaching, and non-profit organisations' non-commercial activities); any other
  production use, including internal business use, needs a commercial license
  from the licensor. The license changes to Apache License, Version 2.0
  (Apache-2.0) on 2030-09-05.
- `SOURCE-MUTATION-STAT`: a CSV execution's `VERIFIED_UNCHANGED` mutation
  status compares the file's OS-reported identity (device, inode, size, and
  modification time) before and after the read. It is not a content hash and
  cannot detect every in-place rewrite that preserves those four values.
- `CONTEXT-REDACTION`: failure-context redaction is a fixed key-name
  allowlist split on `_`; it rejects a key whose parts match a forbidden
  list but never inspects values. A key that happens to avoid those parts
  (for example `filepath` instead of `file_path`) is not screened, so this
  is not general data redaction. Exception text shows numbers and short
  code-like tokens from the context and replaces any other string, such as a
  path or URI, with `<redacted>`; the context mapping itself is unchanged.

[فارسی](KNOWN_LIMITS.fa.md) | [Deutsch](KNOWN_LIMITS.de.md)
