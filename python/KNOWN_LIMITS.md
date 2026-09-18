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
  right censoring. It is not a general CSV or spreadsheet reader.
- `SCALAR-FAMILIES`: normal, gamma, Weibull-minimum, lognormal, and
  right-Gumbel expose scalar log-density, CDF, survival, quantile, and sampling
  operations. They do not expose array evaluation, a uniform fitting API, or
  inference for every registered family.
- `STREAM-SOURCE`: `IterableDataSource` adapts caller-owned chunk iterables.
  The package bundles no Parquet, Arrow, dataframe, database, or network
  adapter. Durable resume is limited to the strict lifetime CSV path and local
  SQLite; it is not a distributed checkpoint store.
- `INFERENCE-EXP`: refit Monte Carlo KS/AD/CvM and adequacy-gated selection are
  limited to finite positive uncensored exponential samples. There is no
  bootstrap selection stability or calibration claim outside the tested grid.
- `MEMORY-BOUND`: the delivery bound covers queued payload and active consumer
  leases until explicit release. It is a logical retained-payload bound, not a
  portable RSS ceiling.
- `SCALE-EVIDENCE`: measurements apply only to their exact adapter, family,
  workload, platform, Python version, chunk limit, and candidate SHA. They do
  not establish universal throughput, generic big-data support, or a broad
  out-of-core capability.
- `LICENSE`: the package uses BUSL-1.1 with the Apache-2.0 additional-use grant
  stated in `LICENSE`; it changes to Apache-2.0 on 2030-09-05.

[فارسی](KNOWN_LIMITS.fa.md) | [Deutsch](KNOWN_LIMITS.de.md)
