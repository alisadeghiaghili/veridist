# Migrating to Veridist 2.0

Veridist 2.0 turns the narrow exponential CSV package into a small, uniform
fitting and distribution API. Most 1.0 code keeps working. This page lists what
changed, what is deprecated (and removed in 3.0), and what to do about it.
The complete list is in the [changelog](../CHANGELOG.md).

## Summary

| Area | 1.0 | 2.0 |
| --- | --- | --- |
| Families | five scalar families; one exponential fit | six families (`EXPONENTIAL` added), a fit for each |
| Calling form | `cdf("gamma", x, {"shape": 2.0})` | `cdf("gamma", x, shape=2.0)` (mapping form deprecated) |
| Arrays | scalars only | scalars and numpy arrays, broadcasting |
| Uncertainty | none | `result.uncertainty()` |
| Censored likelihood | not available | `reduce_lifetime_log_likelihood_chunks`, `reduce_value_log_likelihood_chunks` |
| Top level | CSV entry points only | the fits, operations, observations and errors |
| License | modified template | Business Source License 1.1 with a non-commercial grant |

## One calling form for the distribution operations

Every scalar operation now takes the family first and the parameters as
keywords. `family` is a `FamilyId` or its string value (a declared alias such as
`"weibull"` also resolves).

| 1.0 (mapping form, deprecated) | 2.0 |
| --- | --- |
| `cdf("gamma", 2.0, {"shape": 2.0, "scale": 1.0})` | `cdf("gamma", 2.0, shape=2.0, scale=1.0)` |
| `sf("gamma", 2.0, {"shape": 2.0, "scale": 1.0})` | `sf("gamma", 2.0, shape=2.0, scale=1.0)` |
| `ppf("gamma", 0.5, {"shape": 2.0, "scale": 1.0})` | `ppf("gamma", 0.5, shape=2.0, scale=1.0)` |
| `sample("gamma", 5, {"shape": 2.0, "scale": 1.0}, rng)` | `sample("gamma", 5, rng=rng, shape=2.0, scale=1.0)` |
| `evaluate_log_density(FamilyId.GAMMA, x, shape=2.0, scale=1.0)` | unchanged, or `logpdf("gamma", x, shape=2.0, scale=1.0)` |

The deprecated forms return exactly what the keyword forms return but emit a
`DeprecationWarning`. **They are removed in 3.0.** To find every use, run your
tests with `python -W error::DeprecationWarning`.

`logpdf` is new. It returns `-inf` outside the support (where
`evaluate_log_density` returns a typed `support_violation`), raises `ValueError`
for a non-finite point and `ArithmeticError` when the value is not representable.

## Arrays and numpy scalars

The same operations accept numpy scalars and arrays for the point and for every
parameter; they are broadcast against each other. Scalar input still returns a
Python `float`; array input returns a `float64` array. numpy is imported only
when you pass an array, so the package still imports without it.

```python
import numpy as np
from veridist import cdf

cdf("weibull_min", np.array([100.0, 200.0]), shape=1.5, scale=500.0)
```

The exponential, Weibull-minimum and right-Gumbel families use numpy-native
kernels. The normal, lognormal and gamma families evaluate the verified scalar
kernels element by element, so they equal the scalar results exactly but are slow
on very large arrays (numpy has no `erfc` or incomplete gamma function and scipy
is not a runtime dependency).

`lifetimes_from_arrays(time, event)` and `values_from_arrays(value, event)` build
the observation tuples the fits take from two equal-length columns.

## New fits and the `FamilyId` on results

`fit_normal`, `fit_gamma` and `fit_gumbel_right` join `fit_exponential`,
`fit_weibull` and `fit_lognormal`, and `fit(family, observations, /, **options)`
dispatches on a `FamilyId` or its name. Every family supports right censoring and
`frequency_weights`. The two real-line families (`NORMAL`, `GUMBEL_RIGHT`) take
`ExactValue` and `RightCensoredValue`; the four lifetime families take
`ExactLifetime` and `RightCensoredLifetime`. Passing the other pair raises
`TypeError`.

Every result now has a `family` attribute that is a `FamilyId`. It still compares
equal to its string value, so `result.family == "weibull_min"` keeps working;
code that used `is`/`isinstance` on a plain string should switch to the enum.
Every success also exposes a read-only `parameters` mapping with the canonical
registry names, and the `FitSuccess` and `FitFailure` protocols (importable from
`veridist`) describe the common surface. The family-specific attributes
(`rate`, `shape`, `scale`, `mu_log`, `sigma_log`, `mu`, `sigma`, `location`) are
unchanged.

`fit_normal` without censoring reports the maximum-likelihood `sigma` (divisor
`n`), not the unbiased estimate (divisor `n - 1`).

## Uncertainty

`result.uncertainty()` is a **method**, called on a fit success. It returns a
`FitUncertainty` with `covariance`, `standard_errors` and
`confidence_intervals(level=0.95, method="wald")`, plus the derived quantities
`mean()`, `quantile(p)` (for example B10 is `quantile(0.1)`) and `survival(t)`.
When the observed information is singular, a Weibull shape was fixed, or the
result holds no data, it returns an `UncertaintyUnavailable` value with a stable
`reason` instead of raising. Check the type before reading intervals.

```python
from veridist import ExactLifetime, RightCensoredLifetime, fit

times = (120.0, 340.0, 560.0, 800.0, 1250.0, 1700.0)
observations = [ExactLifetime(t) for t in times] + [RightCensoredLifetime(2000.0)]
result = fit("weibull_min", observations)
uncertainty = result.uncertainty()
print(uncertainty.confidence_intervals())
print(uncertainty.quantile(0.1).lower)
```

## Checkpoint contract changes

These are fixes to results that could be silently wrong; if you resume runs, read
them.

- **`source_revision` is the file's SHA-256.** For
  `fit_exponential_checkpointed_csv` the revision must equal the CSV file's
  current SHA-256 digest (lowercase hex), stream-hashed before any row is read,
  and the checkpoint's own `source_id` and `source_schema` must match the call.
  A reused label, or a stale digest for a file that has since changed, returns
  `SOURCE_REVISION_MISMATCH` (or `SOURCE_ID_MISMATCH` / `SOURCE_SCHEMA_MISMATCH`)
  before any reducer call instead of mixing rows from two files. Build the store
  with `create_checkpointed_csv_store(store_path, csv_path=..., source_id=...)`;
  it computes the right revision and binds the store to that one file.
- **Offset chunk form.** Pass each chunk of `fit_exponential_checkpointed_chunks`
  as `(row_start, payload)`. A replayed, already-committed chunk is then
  recognised and skipped instead of being counted twice. Bare `bytes` chunks still
  work but are deprecated (removed in 3.0), emit `DeprecationWarning`, and can
  only recognise a replay of the single most recent chunk.

```python
# 1.0
fit_exponential_checkpointed_chunks(store=store, source_revision=revision, chunks=[b"[[1.5,true]]"])
# 2.0
fit_exponential_checkpointed_chunks(store=store, source_revision=revision, chunks=[(0, b"[[1.5,true]]")])
```

## Fits no longer report a boundary as converged

`fit_weibull` and `fit_lognormal` previously could return an estimate sitting on
the edge of the internal search range with `converged=True`. They now widen the
range, and when no interior optimum exists they return a typed failure
(`DEGENERATE_SAMPLE` or `BOUNDARY_SOLUTION`). Code that assumed every
`WeibullFitSuccess` could be unpacked blindly must check the result type first.
`fit_exponential` reports a vanishing total time as `NUMERICAL_OVERFLOW`
instead of raising `ValueError`.

## Removed APIs

Nothing that appeared in the 1.0 public documentation is removed in 2.0. Engine
internals reviewed for removal (`PartialOutcome`, `to_canonical_json_bytes`,
`AdapterCapabilities`/`OrderingGuarantee`, `SourceHash`/`SourceHashAlgorithm`)
are kept because the accepted design records describe the concepts they
implement (partial-result labelling, provenance hash-or-redaction, adapter
ordering declarations). They stay importable from `veridist.engine` and are not
part of the top-level API.

## The top-level API

Everything below is importable from `veridist`: the six fits and `fit`;
`FamilyId`, `logpdf`, `cdf`, `sf`, `ppf`, `sample`; the observation types and
`lifetimes_from_arrays` / `values_from_arrays`; the `FitSuccess` / `FitFailure`
protocols; `VeridistError`, `CapabilityError`, `EngineContractError`; the CSV and
checkpoint entry points including `create_checkpointed_csv_store` and
`fit_exponential_checkpointed_csv`; and the three log-likelihood reducers.
Engine internals (stores, buffers, provenance, outcomes) stay in
`veridist.engine`. `CapabilityError` is also exported from `veridist.engine`.

## License change

Veridist 2.0 is licensed under the Business Source License 1.1 (BUSL-1.1) with an
Additional Use Grant that permits **non-commercial use only**: personal use,
academic research and teaching, and the non-commercial activities of non-profit
organisations. Any other production use, including internal business use, needs
a commercial license. On the change date (2030-09-05) the work converts to the
Apache License, Version 2.0. The text is the canonical BUSL-1.1 template; read
the [`LICENSE`](../LICENSE) file for the exact parameters. This replaces the 1.0
text, whose grant referred to Apache-2.0 terms and allowed internal business
analytics. BUSL-1.1 is not an OSI-approved open-source license.
