# Veridist changelog

This changelog covers only the nested `veridist` package. The repository-root
legacy changelog describes the frozen `distfit_pro` history and is not a
Veridist release record.

## [Unreleased]

### Fixed

- `cdf`, `sf` and `ppf` no longer raise `OverflowError` when an intermediate
  `exp` or power overflows, for example the Gumbel CDF far in the left tail
  (`cdf("gumbel_right", -800.0, location=0.0, scale=1.0)`), the Weibull CDF far
  in the right tail, or a Weibull or lognormal quantile beyond the largest
  float. They return the limiting value (`0`, `1` or `inf`), and scalar and
  array evaluation agree.
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
- The strict CSV adapter no longer fails a whole run over a single trailing
  blank record (for example a final `\r\n\r\n` an editor or spreadsheet
  appended). A blank record is now tolerated only when it is the last thing
  in the file; a blank record anywhere else, including one followed by
  another data record, still fails with `SOURCE_ROW_INVALID`/`blank_record`
  at that record's offset, exactly as before.
- A CSV time literal with an astronomically large exponent (beyond roughly
  `1e18` digits of exponent, for example `1e9999999999999999999999`) escaped
  `fit_exponential_csv` and the checkpointed CSV fit as a raw
  `decimal.InvalidOperation`, because `Decimal` rejects such an exponent
  before any range check runs. The literal is now an ordinary typed outcome:
  `SOURCE_ROW_INVALID`/`invalid_time` when it overflows to infinity or a
  nonzero mantissa underflows to zero, and a valid zero for a zero mantissa.
- `IterableDataSource.iter_chunks` could hand the single allowed pass of a
  single-pass source to more than one thread when they acquired it at the same
  time. Acquisition is now serialized with a lock: exactly one caller receives
  the iterator and the others get `PASS_BUDGET_EXCEEDED`.
- `cdf`, `sf`, `ppf` and `sample` called with a declared family alias
  (`"gaussian"`, `"weibull"`, `"gumbel"`) passed registry resolution and then
  failed with an `AssertionError`; the aliases now work like the canonical
  names.

### Added

- `veridist.execution.create_checkpointed_csv_store`, which builds a
  checkpoint store already bound to one CSV file's current SHA-256 revision,
  a public source id, and the exponential reducer contract, so callers no
  longer hand-write the checkpoint record.
- `WeibullFitFailureCode.DEGENERATE_SAMPLE` and
  `WeibullFitFailureCode.BOUNDARY_SOLUTION`, and
  `LognormalFitFailureCode.BOUNDARY_SOLUTION`: declared reasons for the
  previously-silent boundary and non-existence cases described above.
- `CsvLifetimeLimits.default()`, returning `CsvLifetimeLimits(65_536,
  65_536)`; its docstring states that the unit is CPython retained
  object-graph bytes (as measured by `retained_object_graph_bytes`), not
  file bytes.
- `RefitMonteCarloGof.primary_statistic`, naming which requested
  `GofStatistic` the reported standard error and interval actually describe.
  Before this field, a caller requesting more than one statistic had no way
  to tell which one `monte_carlo_standard_error`/`interval` referred to (it
  was always the alphabetically first of the requested statistics).
- `veridist.engine.VeridistError`, the common base of every exception class the
  package defines (`EngineContractError` and its subclasses, `CapabilityError`
  and `CheckpointCommitUncertain`). It is exported from `veridist.engine`, not
  from the top-level package.
- `veridist.statistics.reduce_lifetime_log_likelihood_chunks`, an exact-state
  streaming log-likelihood for exact and right-censored lifetimes under the
  fixed-location `WEIBULL_MIN`, `LOGNORMAL` and `GAMMA` families: an
  `ExactLifetime` contributes its log-density and a `RightCensoredLifetime` its
  log-survival. Terms are accumulated in the same exact integer units as
  `reduce_log_likelihood_chunks`, so the total does not depend on chunking or
  order. The gamma log-survival is evaluated in the log domain, so a survival
  probability below the smallest binary64 stays a finite term instead of
  becoming `-inf`; a term that is not representable, or an incomplete-gamma
  expansion that does not converge (very large shape), is a typed failure.
  Checked against `mpmath` and against the Weibull and lognormal fits'
  reported log-likelihood. It is not exported from the top-level package.
- `FamilyId.EXPONENTIAL` (parameter `rate`, fixed location zero), registered
  with every operation: log-density, `cdf`, `sf`, `ppf`, `sample` and `fit`.
  `Operation` gains `CDF`, `SF`, `PPF`, `SAMPLE` and `FIT`, which every family now
  advertises, and `FamilySpec.support` (with `FamilySpec.contains`) declares the
  log-density support per family: `gamma`, `weibull_min` and `lognormal` keep
  the open support `(0, inf)`, so `x <= 0` is a `support_violation` (and `-inf`
  in `logpdf`), while the exponential has the closed support `[0, inf)`
  (`Support.NON_NEGATIVE`), so its log-density at zero is `log(rate)` and only
  `x < 0` is outside it. A zero time is therefore valid for the exponential
  reducer, which reproduces `fit_exponential`'s log-likelihood on samples that
  contain `ExactLifetime(0.0)` or `RightCensoredLifetime(0.0)`. For every
  fixed-location family `cdf` is `0` and `sf` is `1` for `x <= 0`.
  `evaluate_log_density`,
  `reduce_log_likelihood_chunks` and `reduce_lifetime_log_likelihood_chunks`
  all accept it.
- One calling convention for the scalar operations in
  `veridist.statistics.distributions`: `logpdf(family, x, /, **parameters)`,
  `cdf`, `sf`, `ppf(family, q, /, **parameters)` and
  `sample(family, size, /, *, rng, **parameters)`. `family` is a `FamilyId` or
  its string value (a declared alias such as `"weibull"` also resolves);
  anything else raises `TypeError` and an unknown name `ValueError`. `logpdf`
  returns `-inf` outside the support, raises `ValueError` for a non-finite `x`
  and `ArithmeticError` when the value is not representable.
- `ExactValue` and `RightCensoredValue` in `veridist.domain`: finite real
  observations (negative values allowed; `bool`, non-finite and non-real values
  are rejected) for the families on the whole real line. The lifetime fits
  (`EXPONENTIAL`, `WEIBULL_MIN`, `LOGNORMAL`, `GAMMA`) take only the lifetime
  types and the real-line fits (`NORMAL`, `GUMBEL_RIGHT`) only the real-valued
  types; the other pair raises `TypeError`.
- `fit_normal`, `fit_gamma` and `fit_gumbel_right`, each with right censoring,
  `frequency_weights`, and the same guarantees as the Weibull and lognormal
  fits: bracket expansion with hard limits, `BOUNDARY_SOLUTION`,
  `DEGENERATE_SAMPLE` and `OPTIMIZER_EXHAUSTED` failures instead of a boundary
  reported as converged, and equivariance under location-scale (normal,
  Gumbel) or scale (gamma) changes. `fit_normal` without censoring is closed
  form and reports the maximum-likelihood `sigma` (divisor `n`), not the
  unbiased estimate (divisor `n - 1`). Agreement with `scipy` references is
  better than a relative `1e-6`.
- `veridist.statistics.reduce_value_log_likelihood_chunks`, the censored
  log-likelihood reducer for `NORMAL` and `GUMBEL_RIGHT` over `ExactValue` and
  `RightCensoredValue`; `reduce_lifetime_log_likelihood_chunks` now also admits
  `EXPONENTIAL`. At each fit's parameters the matching reducer reproduces the
  fit's `log_likelihood` to a relative `1e-12`.
- A common fit-result interface: the `FitSuccess` and `FitFailure`
  `runtime_checkable` protocols in `veridist.families`. Every success exposes
  `family: FamilyId`, a read-only `parameters` mapping with the canonical
  registry names, `log_likelihood`, `observation_count`, `event_count`,
  `censored_count` and `converged`; every failure exposes `family`, `code` and
  the counts. The family-specific attributes (`rate`, `shape`, `scale`,
  `mu_log`, `sigma_log`, `mu`, `sigma`, `location`) are unchanged, and the
  `family` attribute of the existing results is now a `FamilyId`, which still
  compares equal to its string value.
- `veridist.families.fit(family, observations, /, **options)`, a dispatcher over
  all six families; an option the family does not accept raises `TypeError`
  naming it.
- `logpdf`, `cdf`, `sf` and `ppf` evaluate arrays: the point and every
  parameter may be a Python or numpy scalar, a 0-d array, or an array-like, and
  are broadcast against each other. Scalar input still returns a Python
  `float`; any array input returns a `float64` array of the broadcast shape
  (empty shapes included). For arrays, `logpdf` is `-inf` element-wise outside
  the support, a non-finite point raises `ValueError`, `ppf` needs every
  probability strictly inside `(0, 1)`, and an invalid parameter element raises
  `ValueError` naming the parameter and its first invalid flat index; a log-density
  that binary64 cannot represent raises `ArithmeticError` naming the first flat
  index. The exponential, Weibull-minimum and right-Gumbel families use
  numpy-native kernels that mirror the scalar formulas (agreeing with the scalar
  path to within a few ulp); the normal, lognormal and gamma families wrap the
  verified scalar kernels element by element, so they equal the scalar results
  exactly but are slow on large arrays (numpy has no `erfc` or incomplete gamma
  function, and scipy is not a runtime dependency).
- `lifetimes_from_arrays(time, event, /)` and `values_from_arrays(value, event, /)`
  in `veridist.domain`: build the tuple of `ExactLifetime`/`RightCensoredLifetime`
  (or `ExactValue`/`RightCensoredValue`) from two equal-length one-dimensional
  array-likes, where `event` is boolean or integer 0/1 (true means the event was
  observed). The columns are validated as arrays first, and the error names the
  first bad row; the result equals the row-by-row construction.
- `FitSuccess.uncertainty()`, on every fit success (all six families, with right
  censoring): a `FitUncertainty` with the `covariance` (the inverse observed
  information at the estimate, in the canonical parameter order), the
  `standard_errors`, and `confidence_intervals(level=0.95, method=...)`.
  `method="wald"` forms intervals for positive parameters on the log scale, so
  they never contain a non-positive value; `method="profile"` inverts the
  profile likelihood by bounded root finding and reports a side that never
  crosses as `inf` (or `0`) with an `interval_details` flag; `method="exact"`
  gives the chi-square interval for the exponential rate on uncensored data
  and raises `ValueError` with censoring. The result is computed on first
  access and cached. When the observed information is singular or not
  positive definite, a Weibull shape was fixed, or the result carries no data,
  `uncertainty` is an `UncertaintyUnavailable` value with a stable `reason`
  instead of an exception. The Hessians are analytic (exponential, Weibull,
  normal, lognormal, right Gumbel, and the exact part of gamma); the gamma
  shape derivative of the censored term is a Richardson-extrapolated central
  difference. A fit now keeps its observed values (the exponential fit only its
  sufficient statistics) to support this.
- `FitUncertainty.mean()`, `.quantile(p)` and `.survival(t)`: the mean (MTTF),
  the quantile (the B-life: `quantile(0.1)` is B10) and the survival
  probability, each as a frozen `DerivedEstimate` with `estimate`, `lower`,
  `upper`, `method` and `level`. Wald intervals use the delta method on the log
  scale (positive quantities), the logit scale (survival) or the identity scale
  (real-line quantities); `method="profile"` is available for the exponential
  and Weibull families and raises `NotImplementedError` naming
  `method="wald"` elsewhere.

### Deprecated

- Passing bare `bytes` chunks to `fit_exponential_checkpointed_chunks` is
  deprecated in favor of the offset form `(row_start, payload)`. The legacy
  form still works but emits `DeprecationWarning`, and it can only recognize
  a replay of the single most recently committed chunk.
- Passing the parameters of `cdf`, `sf`, `ppf` as a mapping
  (`cdf("gamma", x, {"shape": 2.0, "scale": 1.0})`) and the
  `sample(family, size, parameters, rng)` form are deprecated in favour of the
  keyword form (`cdf("gamma", x, shape=2.0, scale=1.0)`,
  `sample(family, size, rng=rng, shape=2.0, scale=1.0)`). The old forms return
  the identical result but emit `DeprecationWarning`; they will be removed in
  3.0.

### Changed

- Replaced `LICENSE` with the canonical Business Source License 1.1 text and
  set its parameters: Licensed Work veridist 2.0.0, a non-commercial-only
  additional use grant (personal use, academic research and teaching, and
  non-profit organisations' non-commercial activities; any other production
  use, including internal business use, needs a commercial license), change
  date 2030-09-05 and change license Apache License, Version 2.0. The previous
  text was a modified template whose grant referred to Apache-2.0 terms and
  allowed internal business analytics. The READMEs, `KNOWN_LIMITS`, and
  `CONTRIBUTING.md` (en/fa/de where applicable) now state the grant
  accurately and no longer imply that the package is open source or that
  business use is free. The root and package copies of `LICENSE` are
  byte-identical, and a test now enforces that.
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
- `CsvLifetimeAdapter`'s running byte tally no longer calls
  `retained_object_graph_bytes` on every parsed observation. A
  `ExactLifetime`/`RightCensoredLifetime` built from a `float` has a
  retained-graph size that cannot depend on the float's value (asserted at
  construction over a spread of sample values), so that size is now
  measured once per observation *type* instead of two to three times per
  row; the exact measurement is still taken at every chunk emission, and
  every existing chunk-boundary and byte-limit test passes unchanged. The
  conservative byte estimate used while a chunk is still being filled is
  also now a tighter bound (still proven never to under-count), which cuts
  down how often a near-boundary row needs a real rebuild-and-measure
  check to confirm it. On a 200,000-row file (random exponential times
  formatted `%.6f`, random `0`/`1` events, `CsvLifetimeLimits(32768,
  32768)`), `fit_exponential_csv` went from roughly 33.9s (about 5.9k
  rows/s) to roughly 6.9s (about 29.1k rows/s) on the reference machine.
- `fit_exponential_csv`, `fit_exponential_checkpointed_csv`,
  `create_checkpointed_csv_store`, and `CsvLifetimeAdapter` now accept
  `str | os.PathLike[str]` for their path arguments, converting to `Path`
  internally, instead of rejecting anything that is not already a
  `pathlib.Path`. A non-path-like argument (for example an `int`) still
  raises `TypeError`.
- `compare_models`'s docstring now states that passing the adequacy gate is
  not evidence that a candidate is an adequate model in any absolute sense,
  and that every candidate's `aic` must come from the same data as every
  other candidate's.
- The Persian exponential report label for the fitted rate's reciprocal is
  now "میانگین محاسبه‌شده" ("calculated mean"); the previous "میانگین
  مشتق‌شده" ("derived mean") read as a direct, slightly awkward calque.
- Package metadata now declares search keywords and trove classifiers
  (audience, topic, Python 3 only, Python 3.11 to 3.14, OS independent,
  typed). No `License ::` or `Development Status ::` classifier is declared.
- `check_release_metadata.py` now also requires `veridist.__version__`
  (read from the source with `ast`, never imported) to equal the project
  version, and the conda-forge recipe's `license` to equal the project
  license.
- The `veridist-ci`, scale-evidence, v1-release-evidence and release
  workflows now pin every third-party action to a commit SHA with the tag in
  a trailing comment, and a test requires it for every workflow except the
  hash-pinned legacy `ci.yml`. `veridist-ci` also runs the test suite on
  Windows and macOS (Python 3.12) and its aggregate gate waits for both.
- `check_coverage.py` now enforces a pragma budget: the coverage manifest's
  new `pragma_budget` records, per production file, how many `# pragma: no
  cover` and `# pragma: no branch` comments it may contain, and the gate
  rejects a file that exceeds its budget or an unlisted file that contains
  any such comment. The only pragmas today are the five in
  `statistics/distributions.py`.
- Mutation evidence moves to schema 3. It counts mutants rejected by mutmut's
  type checker (`type_check`, a subset of `killed`) separately per file,
  module and in the totals, and reports `score_excluding_type_check` next to
  the unchanged `score`; the 0.8 gate still applies to `score`.
- Scale evidence moves to schema 3 (CSV/exponential) and schema 5
  (log-likelihood). Each cell now measures elapsed time in a pass without
  `tracemalloc` and memory in a separate pass, the artifact records the
  worker count and the methodology, and the checkers reject timing evidence
  from more than one measurement worker. `run_scale_csv_exponential_evidence.py`
  now defaults to one worker. Process memory is read by one shared helper
  (`tools/process_memory.py`), which also corrects the log-likelihood runner's
  macOS `ru_maxrss` unit (bytes there, not KiB). `artifact_sha256` is
  documented as an integrity digest, not a signature. Retained evidence files
  are unchanged and remain rejected by the current checkers.
- v1 execution evidence moves to schema 2 and six scenarios per row count:
  the previous three plus `source_mutated` (a changed source is refused with
  `SOURCE_REVISION_MISMATCH` and the checkpoint is untouched), `chunk_replay`
  (replaying committed offset chunks changes neither state nor generation) and
  `process_killed` (a child process is terminated after its first commit and
  the parent resumes it to the uninterrupted result). The assembled matrix has
  54 cells, enforced by the checker, the assembler and the release-evidence
  workflow.
- `CheckpointCommitUncertain` is now also a `VeridistError` (it remains a
  `RuntimeError`, so existing handlers keep working), and `EngineContractError`
  and `CapabilityError` derive from `VeridistError` instead of directly from
  `Exception`.
- `fit_exponential_checkpointed_csv` and `fit_exponential_checkpointed_chunks`
  no longer re-read the checkpoint before every chunk: the record returned by
  their own last commit is reused, and `apply_pure_update` still re-reads and
  revalidates the store at the compare-and-swap boundary, so a concurrent
  writer is still rejected (as `RANGE_MISMATCH`, `SOURCE_REVISION_MISMATCH` or
  `CHECKPOINT_CONFLICT`) rather than merged. For a 100,000-row CSV in 111
  chunks the store reads fall from 224 to 114. `SQLiteCheckpointStore` also
  requests `PRAGMA synchronous = FULL` only on its write paths.
- The strict CSV adapter converts a validated time literal with `float`
  instead of building a `Decimal` first; both round correctly, so every literal
  that parsed before still yields the same value, and the rejection of a
  positive literal that underflows to zero is unchanged.
- Public entry points accept numpy real scalars (`numpy.float32`, `numpy.int64`
  and so on) wherever they accepted a Python `int` or `float`, and still reject
  `bool` and `numpy.bool_`: the point and parameters of `logpdf`/`cdf`/`sf`/
  `ppf`/`sample` and of `evaluate_log_density`, the `size` of `sample`, the
  arguments of `information_criteria`, `compare_models`, `summarize_calibration`
  and `refit_monte_carlo_gof`, `fixed_shape` and the integer `frequency_weights`
  of the fits, and the values of the lifetime observation types. The error text of
  these checks no longer says "built-in". As before, `information_criteria` and
  `compare_models` take their likelihood, `aic`, `p_value` and probabilities as
  floats, not integers.
- A list or tuple passed as a point or parameter of `logpdf`/`cdf`/`sf`/`ppf` is
  now an array-like and is evaluated element-wise instead of raising
  `TypeError`; a probability that is not a real number still raises
  `ValueError`.

### Documentation

- The repository-root and package READMEs (en/fa/de) no longer claim that
  "release evidence covers declared CSV/Exponential paths at 10k, 100k, and
  1m rows"; the retained `scale-csv-exponential-v1.json` snapshot predates
  the schema the current checker requires and does not pass it, so the
  sentence now says a historical snapshot is retained, that it does not pass
  the current checker, and that no throughput or scale claim follows from it.
- Added dated amendment notes to `docs/adr/ADR-0005-out-of-core-backends-and-
  scale-tiers.md` and `docs/adr/ADR-0021-streaming-log-likelihood-reducer.md`
  recording that their retained scale-evidence artifacts (schema version 1
  and schema version 2, respectively) are both now rejected by their current
  checkers (which now require schema versions 3 and 5); the log-likelihood
  artifact's recorded `git_sha` is also not part of this repository's
  history. Neither ADR's body is rewritten.
- `python/docs/source/api.md` (and its Persian/German counterparts and
  `.po` catalogs) now documents the `fit_exponential_checkpointed_csv`
  revision contract (it must equal the CSV file's current SHA-256 hex
  digest) and the `create_checkpointed_csv_store` helper.
- `KNOWN_LIMITS.md`/`.fa.md`/`.de.md` now describe the checkpoint-resume
  contract under `STREAM-SOURCE`: the SHA-256 revision check, the offset
  chunk form, and that durable resume is local to one host.
- Rewrote the repository-root `CONTRIBUTING.md` for `veridist`: it previously
  described the legacy `distfit_pro` tooling (black/isort, a `py-distfit-pro`
  clone URL). It now covers the `python/` layout, the ruff/mypy/pytest/
  `check_coverage.py` gates, the coverage-manifest denominator rule, the
  en/fa/de documentation-parity rule, Conventional Commits, and that mutation
  testing runs only on Linux CI.
- The scale-evidence notes in `docs/evidence`, `docs/v1-readiness.md`,
  `docs/v1-test-plan.md` and the two scale-evidence ADR amendments name the
  schema versions the current checkers require (3 and 5) and the single-worker
  timing rule.
- `fit_exponential`, `fit_exponential_chunks` and `reduce_log_likelihood_chunks`
  now document that an already exhausted one-shot iterator is
  indistinguishable from an empty input (an `EMPTY_SAMPLE` failure for the
  exponential fits, a zero-count success for the log-likelihood reducer), and
  point to `IterableDataSource` for enforced single-pass semantics.

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
