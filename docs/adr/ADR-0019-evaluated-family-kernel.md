# ADR-0019: Evaluated-family kernel and parameter contracts

Status: Accepted

Owner: Ali Sadeghi Aghili

## Context

The completed exponential vertical is a deliberately narrow fitted-family
operation.  Adding several distribution names without a closed description of
their parameters, operations, and boundary rules would invite ambiguous
aliases, parameter-count mistakes, mutable fitted state, and accidental
claims about fitting or inference.  The next increment must create a small,
locale-neutral kernel that can later host independently tested likelihood
evaluators without becoming a fitting API.

Frozen legacy source contains useful naming and scenario cues, but it also
mixes mutable fitted objects, presentation strings, broad exception handling,
and SciPy wrappers.  It is therefore migration evidence only, never a runtime
dependency, numerical oracle, fallback, or specification.

## Decision

Create an immutable `veridist.families.registry` domain kernel with exactly
five canonical families, in this deterministic order:

1. `normal`: `Normal(mu, sigma)`.
2. `gamma`: zero-location `Gamma(shape, scale)`.
3. `weibull_min`: zero-location `Weibull-min(shape, scale)`.
4. `lognormal`: zero-location `Lognormal(mu_log, sigma_log)`.
5. `gumbel_right`: `Gumbel-right(location, scale)`.

The registry retains `LOGPDF` in its planned-operation roadmap and marks it
available only as a closed all-family capability with a separate
exact-observation evaluator and dispatch-parity tests for all five families.
It does not provide fitting, CDF, PPF, SF, model selection, censored
likelihood, weights, raw-data ingestion, reporting, or inference.

Each family has an immutable canonical parameter tuple and an explicit free
parameter count derived from that tuple, never from a mapping length.  Normal,
Gumbel-right, and Lognormal respectively use `mu`/`location` and positive
scale-like parameters as named above.  Gamma, Weibull-min, and Lognormal are
explicitly fixed at location zero; no implicit or fitted location parameter is
accepted.  Finite locations are valid.  Shape and scale-like parameters must
be finite real built-in numbers greater than zero; booleans, non-numbers,
NaN, infinities, zero, and negative values are rejected.  These parameter
contract violations are programmer misuse and raise only `TypeError` or
`ValueError`; typed evaluation failures belong to the later evaluator tranche.

Canonical IDs and explicitly declared aliases resolve deterministically.
Tokens are lowercase snake-case only. Alias collisions, including their
separator-free collision key, fail registry construction; resolution itself
performs no normalization.  The registry, family
specifications, operation sets, aliases, and parameter specifications are
immutable.  The kernel has no fitted object, localized string, I/O, reporting,
raw-data retention, NumPy, SciPy, global warning policy, or legacy import.
Dependency direction is `domain -> families registry`; later evaluator code
may depend on the registry, not vice versa.

## Scope

This record authorizes the registry and, through its scalar-log-density
addendum, exact-observation evaluators for the five closed families.  It does
not change the existing exponential fit API or authorize fitting, ranking, or
likelihood reduction.

The kernel is metadata-sized and says nothing about end-to-end process memory
or throughput.  In particular, it makes no blanket O(1)-memory, out-of-core,
or high-throughput claim.  Streaming likelihood reduction remains a future,
separately evidenced operation.

## Evidence

The five family names and zero-location intent are independently specified in
this ADR and in RED contracts.  The migration ledger pins the inspected legacy
blob and its SHA-256, records the useful scenario cues, and rejects its mutable
state, aliases, catch-all behavior, and SciPy wrapper as production code.
Registry behavior is demonstrated by deterministic contract and mutation-probe
tests, not legacy output.

## Test implications

- `FAM-REG-01`: canonical IDs and listing order are exact and deterministic.
- `FAM-REG-02`: aliases resolve only when explicitly declared; invalid tokens
  and direct or separator-free collisions fail.
- `FAM-REG-03`: mappings, specs, and operation sets cannot be mutated.
- `FAM-PAR-01`: canonical tuples and free counts are fixed for every family.
- `FAM-PAR-02`: positive finite shape/scale contracts reject bool, NaN,
  infinity, zero, and negatives; finite locations remain valid.
- `FAM-ISO-01`: the registry contains no legacy import, fitted object, or
  localized string.

The tests are written before implementation.  Mutation probes must cover
alias collision, mapping-length parameter counts, bool-as-number, nonpositive
shape/scale, mutation, and listing order.

## Dependencies

ADR-0001, ADR-0003, ADR-0005, ADR-0006, ADR-0010, ADR-0011, ADR-0016,
and the future likelihood/reproducibility decisions that will govern evaluator
and streaming-reducer work.

## Consequences

The first multi-family surface is intentionally less convenient than broad
wrapper packages.  The limitation is valuable: callers can see exactly what
exists, and later fitting/model-selection work cannot silently inherit a
parameterization or alias policy.

## Addendum: scalar log-density tranche

This addendum specifies the first numerical operation authorized by this ADR.
It is intentionally a scalar, exact-observation primitive; it is not a fit,
CDF/SF/PPF, array, censoring, ranking, or streaming-likelihood API.

`veridist.statistics.log_density.evaluate_log_density` accepts exactly a
`FamilyId`, one built-in finite scalar observation, and the exact canonical
parameter names declared by `FamilySpec`.  A bool is not a scalar observation.
The public dispatcher accepts no family aliases or string IDs.  Missing,
extra, and alias parameter names are programmer misuse and raise `TypeError`;
parameter validation continues to use the immutable registry and raises its
documented `TypeError`/`ValueError` boundary errors.

The only success is a finite binary64 log-density.  Positive-support Gamma,
Weibull-min, and Lognormal require **strictly** `x > 0`: `x == 0` is a typed
`SUPPORT_VIOLATION`, rather than a limiting value.  A non-finite or bool
observation is `NONFINITE_OBSERVATION`. A mathematically valid calculation
whose required representable intermediate, scaled difference, product, or
exponential term overflows is `NUMERICAL_OVERFLOW`. Negligible
exponential-tail underflow is acceptable when the log-density remains finite.
`NONFINITE_LOG_DENSITY` is reserved for a postcondition breach after a
calculation otherwise completed. The evaluator never returns NaN or either
infinity.
It does not use a broad catch-all, and failures never retain or serialize the
observation, parameter values, paths, or localized text.

Successes and failures are frozen, slotted, locale-neutral data objects.
Their JSON serialization is deterministic (sorted compact keys,
`allow_nan=False`), and success serialization contains only family ID and its
finite computed log-density.  Failure serialization contains only family ID
and a closed failure code.  This makes diagnostics stable without presenting
raw input values.

The canonical formulas are evaluated in log space using the standard library:

- Normal: `-0.5 log(2 pi) - log(sigma) - 0.5 z^2`, `z=(x-mu)/sigma`.
- Gamma: for `shape < 8` the canonical
  `(shape-1) log(x) - x/scale - lgamma(shape) - shape log(scale)` form; for
  `shape >= 8`, the cancellation-safe deviance form
  `-log(scale)-0.5log(2pi shape)-stirlerr(shape)+shape*log1pmx(delta)-log1p(delta)`,
  where `delta=x/(shape*scale)-1` is formed from exact binary64 ratios. When a
  strictly positive exact ratio rounds so far left that `delta == -1` in
  binary64, the evaluator uses the finite direct-log form instead of calling
  `log1p(-1)`; this is not an overflow.
- Weibull-min: `log(shape)-log(scale)+(shape-1)log(x/scale)-(x/scale)^shape`,
  with an exact-binary adjacent-ratio `log1p(delta)` path and `expm1` near
  zero; a required exponential above the binary64 bound is typed overflow.
- Lognormal: `-log(x)-log(sigma_log)-0.5log(2pi)-0.5z^2`,
  `z=(log(x)-mu_log)/sigma_log`.
- Gumbel-right: `-log(scale)-z-exp(-z)`, `z=(x-location)/scale`, evaluated
  with exact-binary scaled differences and the `expm1` near-zero identity.

The stable Gamma form follows the positive-real Stirling expansion and its
first-neglected-term remainder discipline in NIST DLMF §5.11, with terms
through `1/(156 shape^13)`. At the selected `shape >= 8` threshold, the next
term is below the ordinary scalar acceptance envelope. The implementation is
an independent derivation under the binding migration rules in
`docs/conventions.md`; no legacy source is a behavioural oracle or runtime
dependency. The test oracle pins
`mpmath==1.3.0`, reconstructs every input from `float.as_integer_ratio()`, and
uses `max(100, 200 + max(abs(floor(log10(abs(input))))))` decimal digits;
the 1e308 tests therefore use at least 508 digits. All ordinary reference
grids retain `max(8 ULP, 2e-14 relative, 2e-14 absolute)`. The fixed-seed
extreme sweep is a scoped smoke probe with the same ordinary envelope; it is
neither a mutation score nor a general numerical guarantee.

The original scalar-log-density golden vectors were corrected after the
exact-binary, high-precision regression exposed a false Gamma mode result and
a false Weibull finite success. The committed RED test is retained as evidence
of the correction, rather than silently rewriting the prior values.

Primary mathematical source:

- [NIST DLMF §5.11](https://dlmf.nist.gov/5.11)

The metadata registry keeps `LOGPDF` planned.  It advertises it as available
only as a closed all-family capability.  `statistics.log_density` validates at
module import that its dispatch table is exactly the registry's canonical
family set and that each advertised family supports `LOGPDF`; an absent or
extra evaluator is a deterministic import failure.  The registry does not
import numerical code, preserving the `statistics -> families` dependency
direction.

Independent frozen reference vectors and metamorphic identities are required:
Normal location/scale, Gamma exponential and special-shape cases,
Weibull exponential, Lognormal Jacobian, and Gumbel location/scale plus
`z=0`.  They are not copied from production formulas or legacy output.

## Exit criteria and effort class

This ADR may move from Proposed to Accepted only when all five registry
contracts and mutation probes are green, critical coverage meets the binding
threshold, legacy isolation is green, and the ADR is reconciled with the
capability matrix.  Acceptance of this registry does not accept numerical
evaluators, fitting, censoring, or large-data claims.  Effort class: medium,
as the first tranche of the family-evaluation-kernel milestone.

### Amendment (2026-10)

The registry described above has since grown, and its scalar surface is no
longer log-density only:

- It now declares six canonical families: `exponential` (parameter `rate`,
  fixed location zero) joins the five above, appended after `gumbel_right`.
  The support of the log-density is declared per family. `gamma`,
  `weibull_min` and `lognormal` keep the open support `(0, inf)`, because
  their density at zero is `0` or infinite depending on the shape, while
  `exponential` has the closed support `[0, inf)`, because its density at
  zero is the finite `rate`. The cumulative operations remain defined on the
  whole real line: `cdf` is `0` and `sf` is `1` for `x <= 0`.
- It declares and dispatches `cdf`, `sf`, `ppf`, `sample` and `fit` for every
  family, in addition to `logpdf`, through one keyword calling convention:
  `logpdf/cdf/sf/ppf(family, x, /, **parameters)` and
  `sample(family, size, /, *, rng, **parameters)`, with the family given as a
  `FamilyId` or its string value.
- The earlier forms that passed the parameters as a mapping, and
  `sample(family, size, parameters, rng)`, are deprecated; they return the
  identical result, emit `DeprecationWarning`, and are planned for removal in
  3.0.

The statements above that the registry provides no CDF, PPF, SF or fitting,
and that it has exactly five families, describe the original tranche and no
longer apply.
