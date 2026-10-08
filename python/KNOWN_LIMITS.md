# Known limits for Veridist 1.0

This document defines the 1.0 release boundary for package version `1.0.1`.

## What these limits mean in common applications

- Reliability, health, credit, insurance, digital-product, and operations teams can model one clearly defined time-to-event outcome when the documented assumptions apply. The current models do not adjust that outcome for customer, patient, machine, or environmental characteristics.
- Fraud and cybersecurity teams can use supported scalar distribution calculations to create a signal under an already specified reference model. Veridist does not train a classifier, select an alert threshold, process feedback labels, or supply a production event-stream adapter.
- Finance, insurance, manufacturing, and supply-chain teams should not assume every registered family has a fitting API. Scalar calculations require a family and parameters that were justified separately unless a documented fitting path exists.
- In every field, model output still needs domain validation, appropriate sampling, decision-cost analysis, and any required legal, clinical, safety, or regulatory review.

- `FIT-CSV-EXP`: the strict CSV path fits only a fixed-location, rate-only
  exponential model over exact and independently right-censored lifetimes.
  Weibull-minimum and lognormal fits are callable over typed lifetime objects,
  not through a general file-fitting API. Analytic weights, covariates,
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
  Inference is not available for every registered family.
- `STREAM-SOURCE`: `IterableDataSource` adapts caller-owned chunk iterables.
  The package bundles no Parquet, Arrow, dataframe, database, or network
  adapter. Durable resume is limited to the strict lifetime CSV path and local
  SQLite on one host; it is not a distributed checkpoint store. For
  `fit_exponential_checkpointed_csv`, the source revision must equal the CSV
  file's current SHA-256 hex digest, checked against the file on disk before
  any row is read; a changed file, a different public source identifier, or a
  different stored schema returns a typed mismatch instead of resuming.
  `fit_exponential_checkpointed_chunks` should be called with the offset form
  `(row_start, payload)` so a replayed chunk is recognized by its row range
  and skipped instead of being applied a second time; the legacy bare-`bytes`
  form is deprecated, emits a warning, and cannot generally detect a replay.
- `INFERENCE-EXP`: refit Monte Carlo KS/AD/CvM and adequacy-gated selection are
  limited to finite positive uncensored exponential samples. There is no
  bootstrap selection stability or calibration claim outside the tested grid.
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
- `MEMORY-BOUND`: the delivery bound covers queued payload and active consumer
  leases until explicit release. It is a logical retained-payload bound, not a
  portable RSS ceiling.
- `SCALE-EVIDENCE`: measurements apply only to their exact adapter, family,
  workload, platform, Python version, chunk limit, and candidate SHA. They do
  not establish universal throughput, generic big-data support, or a broad
  out-of-core capability.
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
