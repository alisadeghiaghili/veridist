# Veridist capability guide

[English](capability-matrix.md) | [فارسی](capability-matrix.fa.md) | [Deutsch](capability-matrix.de.md)

This matrix records the callable scope of `veridist` 1.0.1. A listed cell is
supported only within its stated data and execution contracts. Historical
evidence remains useful context, but release claims require checks on the exact
candidate revision.

| Capability | Admitted scope | Result and failure contract | Evidence boundary |
| --- | --- | --- | --- |
| Exponential MLE | Fixed `loc=0`, rate-only; exact and independent right-censoring; typed objects or strict UTF-8 lifetime CSV | Finite point estimate or typed statistical/execution failure | Reference, contract, CSV, checkpoint, coverage, and mutation tests |
| Weibull-minimum MLE | Fixed `loc=0`; exact and independent right-censoring; optional frequency weights and optional fixed shape | Finite shape/scale estimate or typed failure with convergence facts | Independent reference and censoring/weight contract tests |
| Lognormal MLE | Fixed `loc=0`; exact and independent right-censoring; optional frequency weights | Finite log-location/log-scale estimate or typed failure with convergence facts | Independent reference and censoring/weight contract tests |
| Scalar family operations | Normal, Gamma, Weibull-minimum, Lognormal, and right-Gumbel | Log-density, CDF, survival, quantile, and caller-owned RNG sampling | Conformance, identity, boundary, and coverage tests; scalar only |
| Streaming likelihood | Exact-state reduction of successful binary64 scalar log-density terms | One final binary64 rounding; explicit unsigned-64 count limit | Contract and generated-stream evidence; no generic RSS or throughput claim |
| Durable resume | Sequential exponential reduction over canonical chunks or strict lifetime CSV, one host, local filesystem | SQLite generation compare-and-swap, checksum validation, source revision and range checks | End-to-end interruption, replay, corruption, contention, and cancellation tests |
| Inference and selection | Finite positive uncensored exponential samples | Refit Monte Carlo KS/AD/CvM, AIC/BIC, calibration summary, lowest-AIC adequate selection or `NONE_ADEQUATE` | Caller-owned NumPy generator; requested/successful/failed refits and Monte Carlo uncertainty reported |

The fixed O(1) reducer state and logical bounded-delivery budget are
algorithmic contracts. They do not establish a portable process RSS ceiling,
universal throughput, distributed execution, or a general out-of-core claim.

Unsupported combinations fail explicitly. The 1.0 line has no left or interval
censoring, truncation, covariates, analytic weights, free location parameters,
array API, distributed checkpoint store, generic dataframe/database adapter,
bootstrap selection stability, or inference for every registered family.

English, Persian, and German landing pages describe the same release boundary.
The strict CSV example and rendered Persian RTL reports are executable CI
contracts. The detailed exclusions are in
[`python/KNOWN_LIMITS.md`](../python/KNOWN_LIMITS.md).

The release line is tested on Python 3.11 through 3.14. The deterministic
coverage manifest requires at least 95% global line and branch coverage and
stricter numerical-module thresholds. The critical statistical core also has a
fail-closed mutation gate. Candidate-specific scale and release validation
workflows bind retained evidence to a full commit SHA.
