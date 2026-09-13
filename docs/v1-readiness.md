# veridist v1 readiness ledger

This is a planning ledger, not a release checklist for `distfit_pro`. v1 uses
the `veridist` namespace. Legacy assets count only after an explicit,
evidence-backed disposition and validation under the v1 contracts.

ADR-0020 records the completed `veridist` 0.5 release contract. Version 1.0.0
adds local SQLite resume, Weibull-minimum and Lognormal reliability fits,
scalar family operations, and the declared exponential inference/selection
cell. This ledger tracks the remaining evidence required for 1.0; completion
of a milestone does not expand any capability beyond the matrix.

## Evidence already obtained

- The reviewed legacy package demonstrated the value of explicit failure,
  canonical parameter counts, log-domain likelihoods, injected RNGs, and
  parameter-refit Monte-Carlo GoF.
- Independent reproduction described in the supplied review is consistent with
  the mechanism that a normal-only AD correction can miscalibrate another
  family and that a known-parameter KS null is invalid after fitting.
- Current public documentation establishes that SciPy offers refit
  Monte-Carlo GoF; fitdistrplus and SurPyval establish that censoring,
  truncation, bootstrap and broad estimator support are table stakes in parts
  of the market.  These are positioning inputs, not v1 implementation evidence.

## Static verified legacy defects

The supplied read-only review reports: discrete likelihood dispatch failures;
duplicate parameter aliases corrupting `k`; duplicated/miscalibrated AD and KS
paths; silent bootstrap substitution; global RNG reseeding; unsupported
censoring and method-name drift; weak test/CI gates; and documentation claims
that did not execute, including six syntax-invalid legacy examples. Treat these
as regression blockers/scenarios to specify, **not**
as a statement about uninspected current files.

## Current verified and unverified status

- `LLR-06` preserves `python/evidence/scale-log-likelihood-v1.json` as a
  historical 10k/100k/1m generated-stream snapshot. Its schema-v2 artifact
  lacks candidate binding and a public-source-route fact, so the current
  checker rejects it as candidate evidence. A current schema-v3 run must bind
  the candidate SHA, use public `IterableDataSource` single-pass acquisition,
  and verify the returned total bitwise against independently reconstructed
  `Fraction` oracle units and the algorithmic 2162-bit bound. Elapsed/tracemalloc
  facts are descriptive; no process-memory, throughput, out-of-core, fitting,
  or general cross-platform claim follows from this historical artifact.

### Historical snapshot: `bfb496d` (preserved verbatim)

- The family-registry tranche adds one metadata-only production file. The
  CI-shaped local run collected 282 items (280 passed and two opt-in browser
  tests skipped); the deterministic checker accepted all 23 enumerated
  production files with 2,113 statements and 736 branches at 100% observed
  coverage, with no accepted exceptions. The registry itself is not a
  numerical evaluator, fit, inference, or large-data capability claim.

### Current unmerged family-kernel candidate

- On 2026-08-30, the CI-shaped local command
  `python -m pytest --cov=veridist --cov-branch --ignore=tests/docs/test_docs_toolchain.py --cov-report=json:coverage.json`
  collected 351 items: 349 passed and the two opt-in browser tests skipped.
  The deterministic checker accepted all 26 enumerated production files with
  2,518 statements and 854 branches at 100% observed line and branch coverage,
  with no accepted exceptions. The generated coverage JSON is deliberately not
  retained as a release artifact, so this ledger does not assert a mutable file
  hash. The current candidate's five-family scalar evaluator has independent
  exact-binary `mpmath==1.3.0` references, a Gamma large-shape/scale/delta grid
  with more than 90 finite cases plus far-left subnormal direct-log regressions,
  a Lognormal magnitude/adjacent-center/sigma grid, and a fixed-seed ordinary
  smoke sweep. All reference acceptance uses
  `max(8 ULP, 2e-14 relative, 2e-14 absolute)`; the sweep is not a mutation
  score or a generalized numerical guarantee. This evidence is limited to
  scalar log-density correctness; it is not fitting, inference, censoring,
  array, or large-data evidence.  At exact candidate `31a015f`, the remote
  Linux CI and configured pinned-browser gates passed; mutation run
  `33804554520` is retained separately below.  Those PR milestone runs are
  neither a tag/release proof nor Windows/portable performance evidence.
- ADR-0017's first callable cell is implemented on the development branch: a
  fixed-location, rate-only exponential MLE for exact and independently
  right-censored lifetimes. It returns a finite point estimate or a typed
  statistical failure; `inference=not_provided`. Reference and contract tests for
  constructor-boundary, reducer, merge, underflow/overflow and report contracts
  pass. This is not a 0.1.0a1 release claim.
- The DS-01--DS-12 contract suite now exercises in-memory sources, bounded
  delivery, pass enforcement, retry/checkpoint boundaries, typed failures,
  outcome classification and closed redacted provenance. The implementation
  span is `d846fc8` through `1345666`.
- A branch-local PEP 517 sdist/wheel build and Twine check pass. A clean wheel
  environment outside the checkout passes dependency/import/version/`py.typed`
  checks and calls the 0.5-rate exponential fit plus a Persian RTL report.
- PR #31 (`csv-exponential-source`) was merged to `main` at
  `1b01385a5d707cfff1dc59c22a6e2e2c5f8eaf14`; its reviewed head was
  `b57041701f3c6d25e99f968fcb07a810dc97e14b`. The remote Linux Python
  3.11--3.14 CI lanes, package build/clean-wheel checks, and browser gates for
  that merged vertical are verified. This verifies only the narrow CSV/
  exponential vertical, not a broad distribution-fitting release.
- Local Sphinx gettext, EN/FA/DE HTML and English linkcheck complete with
  warnings fatal. Exact real POT/catalog parity, translated rendered semantics,
  locale direction and canonical examples pass.  In addition to the historical
  PR #31 evidence, exact candidate `31a015f` has passing remote Linux CI for
  its family-kernel changes.  This is execution evidence for configured gates,
  not a release documentation or visual-review claim.
- The narrow `I18N-RTL-EXP-01` opt-in browser contract passes locally with Edge 151 and exactly two
  nonempty Persian HTML screenshots; computed RTL/right alignment and
  LTR/isolate facts pass for success and failure reports. Exact Playwright
  1.62.0 bundled-Chromium execution is verified for the merged PR #31 vertical.
  `I18N-RTL-DOC-01` additionally passes locally with Edge 151 against
  Sphinx-built Persian and German `exponential-right-censoring` and
  `families-log-density-likelihood` pages: the Farsi root/body are
  `fa`/RTL/right, German is `de`/LTR, and independently selected inline code,
  highlighted Python block, table and static math nodes are LTR with
  `unicode-bidi:isolate`. The local browser scope is those two pages in each
  locale plus the existing report success/failure gate; no documentation
  screenshots are retained locally, while the CI policy retains only the two
  report screenshots. The pages use a local static math asset rather than a
  network MathJax fetch.  Exact candidate `31a015f` has remote pinned-browser
  execution evidence for its configured gate.  PDF rendering, network-font
  rendering, retained independent visual review, and pixel-baseline comparison
  remain unverified and are not claimed.
- Eight scratch-only manual patches were each killed by one targeted unittest:
  `fit_exponential` materialized `tuple(observations)` / `test_exp14_memory_growth_is_bounded_for_unique_generated_observations`;
  `merge` used the raw two totals / `test_exp11_merge_preserves_compensation_and_declares_only_tolerance_across_partitions`;
  time validation became only `float(value)` / `test_positive_decimal_underflow_is_rejected_not_silently_changed_to_zero`;
  success `__post_init__` became a no-op / `test_exp14_success_constructor_rejects_each_inconsistent_derived_fact`;
  the FA catalog was replaced by EN / `test_i18n_exp07_catalogs_are_closed_nfc_translated_and_immutable`;
  the final `REPORT_KEYS` item was removed / `test_i18n_exp04_has_exact_stable_keys_and_semantic_facts_for_every_result`;
  every machine fact became `mutated` / `test_i18n_exp13_machine_values_are_bound_to_success_and_failure_facts`;
  and Persian `dir="rtl"` became `dir="ltr"` / `test_i18n_exp02_farsi_report_has_rtl_root_and_ltr_isolates`.
  This is diagnostic evidence only.  Later immutable candidate `31a015f` has
  retained GitHub Actions run `33804554520` baseline-passing mutation evidence:
  2,217 generated, 1,776 killed, 441 survived, zero unresolved, and score
  `0.801082543978349`.  That milestone evidence is not a 0.5 release proof and
  does not establish scale portability or broader statistical support.
- `SCALE-CSV-EXP-01` preserves a historical strict CSV/exponential snapshot at
  `python/evidence/scale-csv-exponential-v1.json`. Its schema v1 timing fields
  lack valid provenance, so the current checker deliberately rejects it: it is
  not candidate evidence and cannot support a current scale or performance
  claim. Its independently reproducible source bytes, Decimal fit facts,
  one-pass counts and logical-payload observations remain historical context.
  A current candidate requires a clean schema-v2 run that binds both run and
  candidate SHA and declares paired timing provenance. Parquet/Arrow/dataframe/
  database adapters, cancellation orchestration, broad streaming
  equivalence and production-scale RSS bounds remain **NOT IMPLEMENTED**.
  A local SQLite checkpoint backend, its resume/retry contracts, and an
  end-to-end exponential chunk reducer are now implemented, with scope limited
  to one host and local filesystems. The public helper consumes canonical JSON
  chunks and retains only O(1) sufficient state; it does not yet claim a
  checkpoint-replayable CSV adapter or a general orchestration API.
- The package top level now exposes the narrow strict CSV/exponential API:
  `fit_exponential_csv`, `CsvLifetimeSchema`, `CsvLifetimeLimits`,
  `PublicSourceId`, and `ExponentialSourceFitResult`. It is deliberately not
  a broad distribution-fitting API. Version 1.0.0 additionally exposes the
  bounded milestone modules documented in the capability matrix; formal v1
  publication evidence and release exits remain incomplete.
- ADR-0016 now provides an evidence-gated migration ledger, a dependency-free
  fail-closed semantic checker and a separate immutable source-lock policy that
  require the exact frozen source commit, `commit:path` blob, SHA-256, and
  ordered in-bounds source ranges, plus AST/dynamic-import isolation checks and
  built-artifact payload inspection in the configured package job. LM-002 records the
  independent exponential rewrite and reviewed statistical evidence but remains
  `review_pending`; no legacy runtime code, compatibility surface or numerical
  oracle is used. LM-003 remains pending and does not claim legacy phrase reuse.
- Every exact numeric defect magnitude, benchmark, release status, competitor
  feature count, name/trademark availability, and SQL-export novelty remains
  unverified unless a current command or primary source is attached.
- The declared exponential refit Monte Carlo calibration smoke contract is
  implemented. Retained full calibration grids, bootstrap coverage,
  production-adapter streaming equivalence, production-scale memory bounds,
  and PyPI publication remain unverified.

## Remaining v1 work and release evidence

| Area | Required v1 evidence |
| --- | --- |
| Core | Keep the current immutable exponential result/capability facts; extend the [capability guide](capability-guide.md) only with cited conformance evidence |
| First vertical | Exponential, Weibull-minimum, and Lognormal point-estimation cells exist; add portable process-memory and retained reliability/scale evidence before calling a broad reliability + big-data vertical complete |
| Families | Preserve the tested scalar operations; any added family must pass support/CDF/PPF/log-density/reference tests with cited specifications |
| Estimation | Retain MLE applicability, convergence/restart diagnostics, and visible failures; complete release evidence for the declared cells |
| Scale | Keep the retained strict CSV/exponential matrix; add RSS/process-memory evidence and promote other advertised adapters/orchestrators only with their own measured traces |
| Inference | Retain implemented exponential refit Monte-Carlo GoF; publish the full calibration scope and add no analytic p-value without a family/estimator source |
| Uncertainty | Bootstrap failure accounting and coverage evidence only for declared regular scenarios |
| Censoring | Explicit likelihood semantics and reference tests; unsupported family combinations fail loudly |
| Localization | Keep implemented EN/FA/DE parity, examples, local `-W`/linkcheck and `I18N-RTL-DOC-01` Sphinx HTML evidence; require remote pinned-Chromium evidence and separately specify PDF before claiming it |
| Migration | Keep ADR-0016 ledger hashes current; require independent specs/RED tests per candidate and retain `distfit_pro` import/package isolation |
| Quality | Global >=95% line and branch; numerical `domain`/`statistics`/`families`/`engine` paths >=98% line and branch; every production file >=90% of both absent ADR; executable mutation >=80% for that scope |
| Competitive evidence | Every public `supported`/`not_supported` claim cell source-locked at 100% coverage; source-lock registry/checker currently NOT IMPLEMENTED |
| Release | Reproducible PyPI artifact, conda-forge publication readiness, valid `CITATION.cff`, Zenodo DOI release metadata, all CI tiers, changelog, security/license review, and public limitations/calibration report |

## Explicitly not done

Legacy API compatibility, an R implementation, 25+ families, KLL-based GoF,
Bag of Little Bootstraps, arbitrary SQL export, and a universal "best fit"
claim are not v1-complete merely because they appear in older plans.  They need
separate accepted ADRs and evidence. Telemetry is not a v1 feature: collection,
upload and auto-reporting are prohibited.
