# ADR-0020: Veridist 0.5 release contract and fail-closed publication

Status: Accepted

Owner: Ali Sadeghi Aghili

## Context

The repository contains both a frozen legacy package and the new `veridist`
package.  A root legacy workflow previously reacted to GitHub release events
and could publish the repository-root legacy artifact.  That is incompatible
with the binding rule that `veridist` is the sole new runtime namespace and
release artifact.  It is also unsafe to attach a 0.5 tag to a development
version while the current evidence covers only narrow callable cells.

Earlier ADRs set v1 ambitions.  They do not by themselves define what a real,
honest 0.5 release may claim.  This record establishes a deliberately narrow
0.5 contract, a fail-closed release posture until a dedicated publisher exists,
and measurable gates that must pass before the version changes from
`0.0.0.dev0`.

## Decision

The legacy `.github/workflows/ci.yml` must neither subscribe to GitHub release
events nor contain a publication job, publication action, upload command, or
release-credential environment.  It may validate legacy code only.  No GitHub release may
publish `distfit_pro` from this repository.

No publishing workflow is authorized by this ADR.  Publication remains
fail-closed until a separate dedicated `veridist` workflow is implemented and
tested.  That workflow must build only from `python/`, verify the tag equals
the package version, inspect both sdist and wheel metadata and payloads, prove
the installed artifact imports `veridist` rather than legacy code, and publish
only after every required gate succeeds.  A repository release event alone is
never sufficient authority to publish.

The authoritative 0.5 statistical scope is limited to these selected cells:

1. The selected five scalar log-density families under their closed parameter,
   support, and typed-failure contracts.
2. Exact-state streaming likelihood reduction of those scalar binary64 outputs.
3. The strict CSV, independently right-censored, fixed-location exponential
   point-estimation vertical.
4. A public reusable streaming `DataSource`/adapter abstraction used end to end
   by every in-scope streaming fit and likelihood operation.

The current CSV/exponential vertical and reducer are development evidence, not
a sufficient 0.5 scale boundary.  A tag cannot rest on one narrow CSV cell.
Before tagging, the reusable abstraction must support strict CSV and a generic
iterable/stream adapter (or an Accepted ADR must justify an equally reusable
public equivalent), with evidence for each supported adapter/family operation.

`0.5.0` may describe only behavior backed by retained, reproducible evidence
for those cells.  It must not imply a general distribution-fitting package,
general out-of-core engine, calibrated inference, broad censoring, or a
performance result beyond the declared measurement.

The package version remains `0.0.0.dev0` until every 0.5 exit criterion below
passes on the candidate being tagged.  An ADR, local test, branch name, or
green historical run is not a substitute for that evidence.

## Scope

Before a `0.5.0` tag, all of the following are mandatory:

- A public reusable streaming `DataSource`/adapter abstraction is used end to
  end by every in-scope streaming fit and likelihood operation across the
  selected families.  It supports strict CSV and a generic iterable/stream
  adapter, unless an Accepted ADR supplies an equally reusable alternative.
  Each adapter/family combination has retained evidence and explicit limits.
- The delivery bound accounts for queued payload bytes **and active consumer
  leases**; a producer cannot exceed the stated hard byte bound by handing a
  chunk to a consumer.  The implementation, contracts, and retained evidence
  must use the corrected definition consistently.
- Every execution path, including cancellation, retry exhaustion, parser
  failure, consumer failure, and normal completion, has exhaustive resource
  cleanup evidence.  No claim of cleanup may rely solely on garbage
  collection.  Backpressure and cancellation are deterministic, one-pass
  semantics are verified, and no terminal path releases a partial result.
- Reproducible 1m-or-greater-row measurements for every supported
  adapter/family operation record platform, Python version, input generation,
  chunk limit, elapsed time, throughput, and process RSS.  Their checkers must
  reject altered facts.  Linux and at least one Windows run are required for
  each supported operation.  The reports state their exact platform and
  workload limits; they establish neither universal throughput nor a generic
  out-of-core claim.
- EN, FA, and DE documentation have semantic parity for the supported cells.
  The Persian rendered documentation and reports have retained RTL screenshots
  covering ordinary prose plus tables, inline/code blocks, and formulae, with
  required LTR isolates where applicable.  This ADR is English canonical and
  does not falsely assert translated-ADR parity.
- Candidate coverage is at least 95% line and 95% branch globally, and at
  least 98% line and 98% branch for the binding critical scope.  The manifest,
  denominator, and checker are fail-closed.
- Mutation evidence for the binding critical scope has a score of at least
  80% and zero unresolved mutants.  No exclusion, pragma, timeout
  reclassification, or threshold reduction may substitute for a killed mutant.
- The dedicated `veridist` release workflow, tag/version/artifact checks, and
  installed-artifact smoke contract pass.  The legacy workflow remains
  incapable of publication.
- `REL-META-01` passes: tag, package version, wheel and sdist metadata agree;
  the BUSL-1.1 text and Apache-2.0 additional-use grant are included exactly;
  `SECURITY.md`, the 0.5
  changelog and known limits, and root repository landing content are present
  and mutually consistent.
- ADR-0018, ADR-0019, and ADR-0021, together with every other behavioral ADR
  underlying a release cell, are Accepted before tag.  ADR-0018 and ADR-0019
  are currently Proposed, so satisfying their own acceptance criteria is a
  blocking gate rather than a documentation edit.
- Final independent role-based audits cover code/statistics, release safety,
  artifacts, localization, and public claims; all blocking findings are fixed
  and rerun on the exact tag candidate.

Inference, goodness-of-fit, model ranking, broad family catalogues,
conda-forge publication, Zenodo metadata, and claims of superiority are
explicitly deferred to later milestones or v1.  They are not backfilled into
0.5 by documentation or naming.

## Evidence

The release-safety RED contract reads the legacy workflow and fails on a
release trigger; publication-like job; `id-token: write` or `packages: write`;
release-credential environment; secret reference; or known and generic publication
actions/commands.  Its unsafe fixtures cover PyPI action, Twine, uv, Hatch,
Poetry, Flit, and generic publish/upload forms.  The corresponding green
change removes the legacy release trigger and publisher.  This proves only
that the legacy path is disabled; it does not prove a future publisher safe.

Evidence for each remaining gate is candidate-specific and must be retained
with its command, immutable source revision, environment, checker result, and
limits.  A passing result from another revision is historical context only.

## Test implications

- `REL-LEGACY-01`: legacy CI cannot trigger or publish a release artifact.
- `REL-STREAM-01`: every supported streaming operation uses the public
  reusable adapter abstraction and records its adapter/family limits.
- `REL-BOUND-01`: active leases participate in the byte-bound invariant.
- `REL-CLEANUP-01`: each terminal execution path closes its owned resources;
  deterministic backpressure, cancellation, and one-pass behavior are tested.
- `REL-SCALE-01`: every supported adapter/family million-row-or-greater
  evidence checker rejects altered RSS, throughput, platform, workload, and
  limit facts on Linux and Windows.
- `REL-I18N-01`: EN/FA/DE rendered semantics and Persian RTL/LTR-isolate
  screenshot evidence cover prose, tables, code, and formulae.
- `REL-QUALITY-01`: global and critical line/branch coverage gates are
  fail-closed, as is mutation score with zero unresolved mutants.
- `REL-PUBLISH-01`: the future dedicated publisher rejects a nonmatching tag,
  version, metadata, payload, or installed-artifact namespace.
- `REL-META-01`: version consistency, BUSL-1.1 text and additional-use grant,
  known limits, and root landing content are checked as one release contract.

These are acceptance tests, not placeholders.  They are written before the
corresponding production or workflow changes and remain required on the exact
release candidate.

## Dependencies

ADR-0005, ADR-0006, ADR-0010, ADR-0011, ADR-0013, ADR-0015, ADR-0016,
ADR-0017, ADR-0018, ADR-0019, and ADR-0021.  This record narrows the
release claim without superseding their individual behavioral contracts.

## Consequences

The next release cannot be rushed by creating a GitHub release or changing a
badge.  The temporary absence of any publishing workflow is intentional: it
prevents an artifact-name error while the new publisher is designed.  The 0.5
scope is a reusable foundation for a broader future catalogue, but its claims
can be tested and falsified.  It remains no general out-of-core claim beyond
the supported adapter/family cells.  Feature breadth without the listed
evidence is progress toward later work, not eligibility for a 0.5 tag.

## Exit criteria and effort class

All scope bullets are binary release gates for the exact tagged revision, with
the retained evidence and final independent role-based audit results available
for review.
Until then, version `0.0.0.dev0` is mandatory.  Effort class: milestone;
estimated work spans release engineering, portability, measurement,
documentation, and adversarial quality review.

### Amendment (2026-10)

The license described above has been replaced. `LICENSE` is now the canonical
Business Source License 1.1 text with a non-commercial-only Additional Use
Grant, a Change Date of 2030-09-05 and Apache License, Version 2.0 as the
Change License. The references to an "Apache-2.0 additional-use grant" in
this record describe the earlier license and no longer apply.
