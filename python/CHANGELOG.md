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
- `fit_weibull` and `fit_lognormal` no longer report a point at the edge of
  their internal search range as a converged estimate. `bounded_maximize` now
  detects when its result sits on a bound, and a new bracket-expansion step
  widens that bound (up to a hard limit) before giving up, so an estimate
  that genuinely exists just outside the old fixed range is now found rather
  than clipped. Previously, `fit_weibull([ExactLifetime(5.0)])`,
  `fit_weibull([ExactLifetime(5.0)] * 5)`, and a 50-point sample with
  standard deviation `1e-4` around 5 all reported `shape ~= 403.43` (`e**6`,
  the old hard-coded bound) with `converged=True`; a heavily right-censored
  Lognormal sample (1 event out of 500) reported `mu_log` at exactly
  `center + 8`, also with `converged=True`. The first two Weibull cases have
  no finite-shape MLE at all (every exact time is tied, with no censored
  time beyond it) and are now declared non-estimable immediately, before any
  search; the noisier 50-point sample and the heavy-censoring Lognormal case
  now either find a genuine (if large) interior optimum or, if the hard
  limit is reached, report a boundary failure instead of a false success.
  `WeibullFitSuccess`/`LognormalFitSuccess` are now only ever constructed
  for an interior, converged optimum, so their `converged` field is
  truthful; `restart_failures` stays `0`, documented as such, because
  neither fit restarts.
- The Weibull shape search now divides every time by the sample's geometric
  mean before optimizing and rescales the reported scale and log-likelihood
  back afterward, so the fit no longer depends on the time unit. It
  previously computed `exp(shape * log(time))` directly, which could lose
  precision or overflow once times were recorded on a very different scale
  (verified invariant to within relative `1e-9` for the same sample recorded
  at a factor of `1e-8`, `1`, and `1e12`).
- The Lognormal right-censoring term (`_log_sf`) raised `ValueError`
  whenever `erfc` underflowed to exactly `0.0`, which aborted the entire fit
  with `OPTIMIZER_EXHAUSTED` even when the true optimum was fine elsewhere.
  It is replaced by `_log_normal_sf`, which falls back to the asymptotic
  Mills-ratio expansion of the standard-normal upper tail once the direct
  formula would underflow, and never raises for a finite argument (checked
  against `mpmath` up to `z = 1e3` at relative error `1e-12`).
- Censored Lognormal fitting is substantially faster. It grouped identical
  censoring times and summed them individually rather than by count, and
  recomputed the exact-observation sum terms from scratch on every one of
  the roughly 6,400 nested golden-section likelihood evaluations. Censored
  observations are now grouped by identical time (a single shared censoring
  time is the common case), and the exact-observation sums are hoisted out
  of the inner loop, so each evaluation is `O(distinct censored times)`
  instead of `O(n)`. A 2,000-observation censored sample (1,403 events) went
  from roughly 10.6s to roughly 0.02s on the reference machine; semantics
  are unchanged (verified against an independent reference within relative
  `1e-6`).
- `fit_exponential` and `fit_exponential_csv` no longer raise a raw
  `ValueError` for a sample whose rate estimate cannot be represented as a
  finite number. A single vanishingly small exact lifetime (for example
  `1e-320`) drives `events / total_time` past the top of `float` range, and
  the resulting infinite rate used to escape the `ExponentialFitSuccess`
  constructor as `ValueError: rate must be finite and positive` instead of a
  typed outcome. `fit_exponential_reduction_state` now checks the rate, its
  reciprocal, its logarithm, and the log-likelihood for finiteness before
  constructing a success, and reports a typed `NUMERICAL_OVERFLOW` failure
  instead; the CSV execution path already treats this as a complete run
  carrying a non-estimate, not an execution failure.
- `fit_exponential_source`'s terminal-coverage check could never actually
  fail: `expected_row_stop` and `expected_chunk_count` were re-derived from
  the very same chunk envelopes `DeliveryValidator` had just accepted, so a
  chunk silently lost between parsing and delivery would pass unnoticed.
  The CSV adapter now records, independently of what it hands to a caller,
  how many records it parsed through EOF (`CsvLifetimeAdapter
  .terminal_record_count`), and `finish()` compares delivery against that
  instead of against itself.
- Execution provenance's reported pass count was always `1`, even when the
  adapter's own source was never actually read, because
  `fit_exponential_source` wrapped a fresh `PassEnforcer` around its own
  internal generator instead of observing the adapter's own single-pass
  enforcer (`CsvLifetimeAdapter` already tracked this separately). Reported
  provenance now comes from the adapter's own enforcer
  (`CsvLifetimeAdapter.passes`), so it reflects what actually happened.
- `BoundedChunkBuffer._release` subtracted a released chunk's bytes from the
  in-flight tally and only then checked for underflow, so a corrupted,
  briefly-negative tally could be observed before the call raised. The check
  now runs before the subtraction.
- `ChunkEnvelope`, `DeliveryValidator`, and `DataSourceMetadata` now raise
  `TypeError` for a non-`str` id instead of letting `str.strip()` raise a
  bare `AttributeError`.

### Added

- `veridist.execution.create_checkpointed_csv_store`, which builds a
  checkpoint store already bound to one CSV file's current SHA-256 revision,
  a public source id, and the exponential reducer contract, so callers no
  longer hand-write the checkpoint record.
- `WeibullFitFailureCode.DEGENERATE_SAMPLE` and
  `WeibullFitFailureCode.BOUNDARY_SOLUTION`, and
  `LognormalFitFailureCode.BOUNDARY_SOLUTION`: declared reasons for the
  previously-silent boundary and non-existence cases described above.

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
- `EngineContractError.__str__` and `__repr__` now include the failure's
  sorted context (`CODE (k1=v1, k2=v2)`) instead of only the bare code, when
  that context is non-empty; an empty context still renders as just the
  code. Numbers and short code-like tokens are shown as-is; any other
  string, such as a path or URI, is rendered as `<redacted>`, because
  context keys are screened but values are not (see the new
  `CONTEXT-REDACTION` entry in `KNOWN_LIMITS.md`).
- `SQLiteCheckpointStore` now adds `sqlite_errorname` (for example
  `SQLITE_BUSY` or `SQLITE_FULL`) to the context of a
  `CHECKPOINT_STORAGE_FAILED` failure raised from a `sqlite3.Error`, without
  exposing that exception's free-text message.
- Documented, in docstrings and in `KNOWN_LIMITS.md`/the capability guide,
  that `SourceMutationStatus.VERIFIED_UNCHANGED` compares a file's
  OS-reported identity (device, inode, size, modification time) rather than
  its content, and that failure-context redaction is a key-name allowlist,
  not general data redaction.
- The PyPI publish workflow now rebuilds the published wheel and sdist from
  the tagged commit with the same reproducible-build procedure and epoch as
  release validation, refuses to publish unless both rebuilt artifacts are
  byte-identical to the ones attached to the release, and runs the release
  metadata, artifact-payload, and legacy-isolation checks against them before
  calling the publish action. It also now requires a successful CI run and a
  successful mutation-evidence run for the exact tagged commit, and fails
  with an explicit message telling the maintainer to dispatch the missing
  run. Third-party actions it uses are pinned to a commit SHA. The mutation
  workflow now also runs on every push to `main`, so a squash-merged commit
  gets evidence instead of relying on its (now-superseded) pull-request run.
- `check_release_artifacts.py` now compares every `*.py`/`py.typed` file
  under `src/veridist` byte-for-byte against the packaged wheel and sdist
  payload and rejects any missing or unexpected file under the package root.
  Previously it only checked a handful of top-level documents and the
  package's presence by name, so a wheel containing nothing but a rewritten
  `veridist/__init__.py` (plus the real `METADATA` and `LICENSE`) passed
  validation.
- `check_coverage.py` now rejects a coverage exception once its `expiry`
  date has passed, and rejects one whose `adr` does not name a document that
  actually exists under `docs/adr`. Previously an exception's expiry and ADR
  reference were only checked for well-formedness, not for being true.
- `build_reproducible.py` now normalizes file permission bits in both
  archives (`0o644` for ordinary files, `0o755` for directories and files
  that were executable in the source build) instead of carrying over
  whatever the local build toolchain happened to produce. The release
  workflow pins `setuptools` and `wheel` to exact versions for the same
  reason: `build --no-isolation` otherwise depends on whatever is already
  installed.
- `ci_scope.py` no longer classifies `.github/workflows/ci.yml` as
  Veridist-only; edits to the legacy workflow itself now correctly route
  through the legacy test lane instead of skipping it.
- The mutation evidence tool's list of required non-test input files
  (`pyproject.toml`, the mutation manifest, the three mutation tool
  scripts, and the mutation workflow file) is now enforced: a missing file
  previously was silently dropped from the expected set instead of failing
  the gate.

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
