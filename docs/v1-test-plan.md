# veridist v1 test plan

Every production change starts with a failing test that fails for the intended
reason.  Expected values come from a derivation, primary source, or independent
versioned oracle--never from legacy implementation behaviour.

## Taxonomy

| Suite | Purpose | Default CI |
| --- | --- | --- |
| `unit` | pure math, error paths, diagnostics | required |
| `property` | support, monotonicity, invariance, reduction laws | required |
| `conformance` | every registered family | required |
| `reference` | committed data/golden files with provenance | required |
| `statistical` | calibration, recovery, interval coverage, selection | nightly/release |
| `scale` | streaming, pass count, state/memory integration | required from first slice; extended nightly |
| `docs` | Sphinx/MyST links, examples, localized rendering | required from first slice |

## Core and numerical tests

- Conformance covers `log_density`, support, CDF/SF/PPF identities, discrete
  mass, declared free-parameter count, undefined moments, and sample support.
- Test integer frequency weights against replicated observations within the
  documented numerical tolerance.  Test zero censoring against ordinary MLE.
- Test optimizer non-convergence, invalid support, all-censored data and failed
  restarts: diagnostics/failure counts must be returned or raised, never
  substituted.
- Tolerances state their source: high-precision oracle error, published
  precision, convergence tolerance, or Monte-Carlo error.  No tuned tolerance.

## Out-of-core tests

For each advertised tier, test empty chunks, one chunk, ragged chunks, changing
boundaries, stable chunk IDs/row offsets, deterministic shuffled schedule,
non-finite input, duplicate/lost/out-of-order chunks, source mutation and
restartable/non-restartable errors. Test planning rejects multipass work on
`single_pass`; test explicit spool opt-in separately. Assert state bound,
`chunk_bytes`, `max_inflight_bytes`, backpressure, cancellation, no retained
chunk, no full-source materialization and declared `max_passes` with an
instrumented pass-enforcer. Retry tests prove idempotent checkpoint recovery or
return a labelled incomplete/failure result.

Compare to an in-memory canonical reduction under the floating-point
reproducibility contract: tolerance-based across ordinary schedules; bitwise
only under the pinned reproducible schedule.  Integration benchmarks measure
peak memory and pass count outside ordinary PR CI; no portable RSS ratio or
parallel speedup is a universal unit-test assertion.

## DataSource contract-to-test matrix

This is the binding test design for the DataSource rules in
[conventions.md](conventions.md) and ADR-0005. IDs are stable: an ADR or
implementation may add tests, but may not weaken or rename an ID without a
superseding ADR. The contract fixtures below are implemented using in-memory
sources and test doubles in commits `d846fc8` through `1345666`. This does not
imply an actual file/database adapter, persistent checkpoint backend,
production orchestrator or scale result.

| ID | Contract | Contract test fixture | Observable assertion |
| --- | --- | --- | --- |
| DS-01 | Identity, schema and redaction | metadata-contract fixture | Source ID is stable; schema/provenance versions are non-empty and supported; either a source hash is present or an explicit redaction reason is present. |
| DS-02 | Replayability plan | one `single_pass`, one `replayable`, one `checkpoint_replayable` fake source | Planner accepts within-budget plans only; it rejects a second pass on `single_pass` before iteration; checkpoint replay is accepted only with compatible checkpoint metadata. |
| DS-03 | Explicit spooling | single-pass fixture plus opt-in/off plans | A multipass plan with spooling off returns `SPOOL_REQUIRED`/capability error without reading; opt-in records disk budget, retention and cleanup outcome in provenance. |
| DS-04 | Adapter ordering, offsets and boundaries | CSV, Parquet, Arrow, pandas, Polars, Dask and database adapter contract fakes | Each adapter declares replayability and ordering; emitted chunk IDs are stable, offsets are contiguous and non-overlapping, and boundary changes preserve row identity. |
| DS-05 | Empty, duplicate, lost and out-of-order delivery | adversarial chunk scheduler | Empty chunks complete normally; undeclared duplicate/lost/out-of-order delivery returns its distinct typed code and neither double-counts nor silently reorders input. |
| DS-06 | Byte bounds, inflight bound, backpressure and cancellation | instrumented producer/consumer with oversized and blocked chunks | Observed buffered bytes never exceed `max_inflight_bytes`; each chunk honours `chunk_bytes`; producer blocks under backpressure; cancellation stops new reads, releases resources and reports a typed cancelled outcome. |
| DS-07 | Pass budget | counting source plus pass-enforcer | Iteration count never exceeds declared `max_passes`; attempt `max_passes + 1` fails before the next source read and provenance records the actual pass count. |
| DS-08 | Retry, idempotency, checksum and source mutation | fault-injecting checkpointed source | Retry only replays an idempotency-keyed update; checksum mismatch and source mutation receive distinct typed errors; no accumulator update is applied twice. |
| DS-09 | Checkpoint resume | interrupted run with compatible and incompatible accumulator/source schema | Compatible resume equals the declared canonical reduction within its contract; incompatible schema/version/range rejects without reuse and records the reason. |
| DS-10 | Typed failure surface | parameterized invalid-delivery/source faults | Every documented failure returns/raises the declared stable error code; no generic string-only failure is accepted. |
| DS-11 | Partial-result labelling | failure after a known offset range | A returned partial result has `complete=False`, exact missing ranges and a typed cause; a complete result cannot contain missing ranges or a partial cause. |
| DS-12 | Full provenance | completed, cancelled, retried and redacted runs | Provenance validates schema/version, source ID/hash-or-redaction, row ranges/count, chunk/pass/budget settings, adapter/library versions, estimator/RNG/approximation and checkpoint metadata; raw input is absent under redaction. |

The executable contract matrix is a prerequisite, not evidence that a
DataSource adapter or scale tier exists. Before either is advertised, the same
matrix must run against the actual adapter/orchestrator and retain fixture
version, observed pass/byte traces and the typed-result/provenance record needed
to audit each assertion. `SCALE-CSV-EXP-01` preserves a historical strict CSV
lifetime adapter plus one-pass exponential-orchestrator snapshot at
`python/evidence/scale-csv-exponential-v1.json`. Schema v1 is quarantined by
the current checker because its timing lacks valid provenance. It supplies
historical source/Decimal/one-pass/logical-payload facts, not current candidate
or performance evidence. Any replacement must be schema v3, bind the reviewed
candidate SHA, declare paired timing provenance, record a single measurement
worker, and cover the fixed
10k/100k/1m-row by three-budget matrix before it can support a current claim.

## Statistical evidence

For each supported family/estimator/test scenario, simulate a declared grid of
parameters and sample sizes.  Fit every simulated sample again before GoF.
With nominal alpha `a` and `R` replicates, record rejection rate and binomial
standard error `sqrt(a*(1-a)/R)`; at `a=.05, R=2000`, this is .00487 and a
three-standard-error descriptive band is .05 +/- .0146.  Publish confidence
intervals and the full grid; this is evidence for that grid, not proof for all
families.

Bootstrap coverage tests declare the data-generating model, parameter regime,
sample size, interval type and failures.  At 95% nominal coverage, report its
binomial uncertainty; do not promise BCa/CRB behaviour in boundary or
nonregular models.  Selection tests measure algorithmic stability and recovery
over a stated candidate set; they never interpret resampling frequency as the
probability that a model is true.

## Localization and documentation tests

- Verify English/Persian/German key parity, permitted explicit fallback rules,
  placeholder/format compatibility, Unicode normalization, and locale-aware
  numeral/date formatting.
- Render Persian pages in RTL and perform a visual smoke test for direction,
  code blocks, formulae, tables and mixed Latin identifiers.
- Build every locale with warnings treated as errors, run link checking, and
  execute every public code example from a canonical source or parity harness.

## Legacy migration governance tests

`ML-01` validates the Draft 2020-12 ledger shape, target namespace and allowed
dispositions. `ML-02` recomputes the source SHA-256 and validates the recorded
commit/blob pair against `origin/main`; a mismatch is a hard stale-evidence
failure. `LI-01` parses every `veridist` Python module and rejects static,
dynamic and constant-composed imports of `distfit_pro`. `LI-02` inspects each
built wheel/sdist for legacy payload paths. These checks do not prove an
approved component is statistically correct, portable, or ready to ship.

The narrow `I18N-RTL-EXP-01` contract has an opt-in Persian **HTML report** browser
gate covering success and failure facts, computed document/report direction,
right alignment and LTR/isolate handling of mixed Latin API identifiers. It
requires exactly two nonempty screenshots. `I18N-RTL-DOC-01` builds the Farsi
and German tutorial pages and independently requires the exact `code.literal`,
`.highlight pre`, `table.docutils` and `.math` selectors: Farsi root/body are
RTL/right, German is LTR, and each Farsi machine-readable exemplar is
LTR/`unicode-bidi:isolate`. The rendered page must contain no network MathJax
runtime. Both HTML gates pass locally with Edge; the pinned Playwright 1.62.0
matched-Chromium CI run remains unverified. PDF is not covered by either gate
and must not be inferred from HTML evidence.

## CI tiers and checker contract

The branch workflow configures Ruff, strict source type checking,
DS-01--DS-12 contracts, branch coverage, package build/clean-wheel smoke and
EN/FA/DE documentation checks on Linux. Tests run on Python 3.11, 3.12, 3.13
and 3.14. The current branch adds the exponential, documentation and browser
lanes; their remote execution remains **UNVERIFIED**. A supported Windows/macOS
release matrix is not implemented.

The implemented coverage checker contract is exact: its input must identify every
production file and report separate line and branch percentages; it fails when
global line <95%, global branch <95%, a numerical production path in
`domain`, `statistics`, `families` or `engine` has line or branch <98%, or any
production file has line or branch <90% without an accepted exact-path ADR
exception. It must reject unlisted files and denominator changes. Its mutation
input must enumerate every eligible mutant in those same critical numerical
paths and fail unless `killed / (killed + survived) >= 0.80`; timeout, harness
error, unclassified result, excluded mutant without an accepted exact-path
expiry, or a missing module report fails the run. Generated code, `pragma: no
cover`, import-only execution, test-only helpers, and configuration exclusions
cannot reduce a denominator unless an accepted ADR supplies an exact path,
rationale, owner and expiry.

The documentation job is configured to build EN, FA and DE with warnings fatal,
check links, execute canonical examples/parity tests, validate direction and
retain rendered HTML. The complete local gettext/three-locale HTML/linkcheck
run passes. Local Edge browser evidence passes for Persian report HTML and the
Farsi/German built Sphinx tutorial contract; remote matched-Chromium CI and any
PDF rendering evidence remain unverified. Eight
targeted manual mutant probes were killed, but they are not a formal mutation
score. The formal mutation runner and its versioned GitHub Linux workflow are
implemented; no Linux execution, retained evidence, score, or release PASS
exists yet. The narrow
`SCALE-CSV-EXP-01` retained-artifact checker is implemented; broader adapter,
RSS, calibration, docs/i18n/example and rendered-RTL artifacts remain required
continuous gates, not skipped placeholders or late-release work.

`LLR-06` validates `python/evidence/scale-log-likelihood-v1.json`: generated
Normal(0,1) streams at 10k, 100k and 1m rows by three chunk sizes, independent
exact-unit reconstruction, one actual outer iterator acquisition, every
observation yield, returned-total bit equality, and the 2162-bit algorithmic
bound. It does not claim to observe private reducer state. Elapsed and
tracemalloc fields are descriptive only; this is not a process-memory,
throughput, or out-of-core claim.
