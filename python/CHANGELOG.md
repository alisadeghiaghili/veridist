# Veridist changelog

This changelog covers only the nested `veridist` package. The repository-root
legacy changelog describes the frozen `distfit_pro` history and is not a
Veridist release record.

## [Unreleased]

### Fixed

- Corrected the gamma family's `cdf`/`sf`/`ppf` for the continued-fraction
  branch (`x / scale >= shape + 1`), which a sign-destroying clamp
  (`max(abs(...), tiny)`) on the modified Lentz recurrence's intermediate
  terms made silently wrong. `_regularized_gamma` is replaced by
  `_regularized_gamma_pq`, which returns `(P, Q)` directly from whichever
  branch is numerically stable, so `sf` no longer computes `1 - P` and loses
  the right tail to underflow. Both the series and continued-fraction loops
  now raise `ArithmeticError` instead of returning silently if they exhaust
  their iteration budget. The gamma quantile (`ppf`) bisects on `sf` for the
  upper half and in log-space rather than over a fixed linear range, so it no
  longer collapses to the search bound for small upper-tail probabilities.
  Examples of the previous error, now corrected: `cdf("gamma", 6.5, {"shape":
  5.0, "scale": 1.0})` returned `0.0` instead of `~0.77633`; `sf("gamma",
  10.0, {"shape": 2.0, "scale": 1.0})` returned `9.08e-4` instead of
  `~4.994e-4`; `sf("gamma", 50.0, {"shape": 2.0, "scale": 1.0})` returned
  `0.0` instead of `~9.837e-21`; `ppf("gamma", 0.95, {"shape": 2.0, "scale":
  1.0})` returned `~5.3696` instead of `~4.74386`. The gamma log-density
  evaluator is unaffected; it does not share this code path.
- `fit_exponential_checkpointed_csv` no longer accepts a resume whose
  `source_revision` is just a reused label: it must equal the CSV file's
  current SHA-256 digest, stream-hashed before any row is read, and the
  checkpoint's own `source_id` and `source_schema` must match the call's.
  Resuming against a changed file with a stale revision string, or pairing
  that stale revision with a different public source id, previously
  advanced the checkpoint and silently mixed rows from two different files;
  it now returns `SOURCE_REVISION_MISMATCH`, `SOURCE_ID_MISMATCH`, or
  `SOURCE_SCHEMA_MISMATCH` before any reducer call.
- `fit_exponential_checkpointed_chunks` no longer double-counts a replayed
  chunk. `row_start` was always read from the live cursor, so resending an
  already-committed chunk always looked like a new operation at a new
  offset and its rows were counted twice; the offset form `(row_start,
  payload)` now lets a replay be recognized and skipped instead of applied
  again.
- `SQLiteCheckpointStore.create` now creates its schema and initial row in a
  single transaction instead of two separate autocommit statements. A crash
  between the two previously could leave a store file that `read()` reported
  as missing and a later `create()` rejected as already existing; a failed
  `create()` now leaves either no file or a fully valid record.

### Added

- `veridist.execution.create_checkpointed_csv_store`, which builds a
  checkpoint store already bound to one CSV file's current SHA-256 revision,
  a public source id, and the exponential reducer contract, so callers no
  longer hand-write the checkpoint record.

### Deprecated

- Passing bare `bytes` chunks to `fit_exponential_checkpointed_chunks` is
  deprecated in favor of the offset form `(row_start, payload)`. The legacy
  form still works but emits `DeprecationWarning`, and it can only recognize
  a replay of the single most recently committed chunk.

### Changed

- Rebuilt the English, Persian, and German repository and package READMEs as
  progressive tutorials, from distribution modelling to a runnable lifetime
  analysis, result interpretation, supported APIs, and adoption guidance.
- Explained distribution-derived machine-learning features, the distinction
  between current fitting APIs and future multi-model ranking, and the planned
  review and migration of 25 legacy distributions.
- Distinguished the static coverage requirement badge from live CI status and
  retained explicit citation, support, and conditional-license guidance.

## [1.0.1] - 2026-09-12

### Changed

- Reorganized the English, Persian, and German repository and package landing
  pages around installation, first success, workflow selection, validation,
  production boundaries, and task-specific documentation.
- Added a live `coverage >=95%` badge backed by the maintained-branch CI
  workflow and its enforced global line and branch coverage contract.
- Corrected the published 1.0 known-limits and capability labels and aligned
  package, citation, Zenodo, documentation, and conda-forge release metadata.

### Quality

- Python 3.11 through 3.14, documentation, RTL browser, package, coverage,
  mutation, reproducible-build, and release-artifact gates remain mandatory.

## [1.0.0] - 2026-09-11

### Added

- Candidate-bound execution evidence for complete, retry-resume, and
  cancellation scenarios at 10k, 100k, and 1m rows on Linux, macOS, and
  Windows. The release gate validates all 27 collected cells and their source
  and result digests.
- Reproducible release validation that builds artifacts twice, canonicalizes
  archive metadata, verifies byte equality, and retains the validated wheel
  and source distribution.
- Isolated PyPI Trusted Publishing through GitHub OIDC and the protected
  `pypi` environment. Published GitHub Release assets are the only artifacts
  eligible for upload.

### Changed

- Checkpointed CSV execution persists bounded adapter chunks in one
  transaction and retains a committed prefix on cancellation for nonzero-
  cursor resume.

### Quality

- Python 3.11 through 3.14, documentation, RTL browser, package, 95%
  coverage, and fail-closed mutation gates remain release requirements.

## [0.9.1] - 2026-09-11

### Changed

- Checkpointed CSV execution commits bounded adapter chunks instead of opening
  a SQLite transaction for every row. Cancellation persists the completed
  prefix for nonzero-cursor resume.
- Candidate-bound 1.0 evidence collection verifies complete, resumed, and
  cancelled 10k, 100k, and 1m-row runs on Linux, macOS, and Windows and rejects
  digest drift.
- Citation File Format, Zenodo, and conda-forge release metadata is
  version-aligned and validated in CI.

### Quality

- The 95% global line and branch coverage contract remains enforced; the
  execution-module denominator is frozen at 179 statements and 70 branches.
- Competitive source-lock validation remains outside the package release gate
  while ADR-0014 is Proposed; competitive drafts remain non-publishable.

## [0.9.0] - 2026-09-11

### Added

- Durable local SQLite checkpoints with generation-based compare-and-swap,
  corruption detection, source-revision checks, and resumable exponential CSV
  and canonical-chunk execution.
- Fixed-location Weibull-minimum and lognormal MLE cells for exact and
  independently right-censored lifetimes, including typed failures and
  frequency-weight contracts.
- Scalar CDF, survival, quantile, and caller-owned RNG sampling operations for
  the declared continuous-family registry.
- AIC/BIC, adequacy-gated model selection, and refit Monte Carlo KS, AD, and
  CvM goodness-of-fit for the uncensored exponential cell. Monte Carlo output
  reports requested, successful, and failed refits plus sampling uncertainty.

### Quality

- Python 3.11 through 3.14 CI, package, documentation, RTL browser, coverage,
  and fail-closed mutation gates pass on the release line.
- The inference module is registered with 129 statements and 46 branches and
  is fully exercised by its contract and calibration tests.

### Known limits

See [Known limits](KNOWN_LIMITS.md). Inference remains limited to uncensored
exponential samples, checkpoints remain local SQLite, operations are scalar,
and no general performance, RSS, or universal best-fit claim is made.

## [0.5.0] - 2026-09-10

### Available in the candidate scope

- A strict UTF-8 CSV adapter for fixed-location, rate-only exponential MLE
  with exact and independently right-censored lifetimes.
- Scalar log-density evaluation for normal, gamma, Weibull-minimum,
  lognormal, and right-Gumbel families.
- Exact-state streaming log-likelihood reduction through the public
  `IterableDataSource` contract.
- Bounded delivery that charges queued payload and active consumer leases.
- English, Persian, and German package documentation with Persian RTL checks.
- Candidate-bound coverage, mutation, scale-evidence, and release-validation
  workflows.

### Release status

The package version is `0.5.0`. Its candidate-specific ADR-0020 gates passed
on the exact release candidate; retained CI and scale artifacts bind to that
candidate SHA.

### Known limits

See [Known limits](KNOWN_LIMITS.md). These limits are part of the release
contract and constrain every public 0.5 claim.
