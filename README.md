# Veridist

**Evidence-first distribution fitting for inspectable statistical results and reproducible decisions.**

[![PyPI](https://img.shields.io/pypi/v/veridist.svg)](https://pypi.org/project/veridist/)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-3776AB)](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md)
[![CI](https://img.shields.io/github/actions/workflow/status/alisadeghiaghili/veridist/v1-ci.yml?branch=main&label=CI)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)
[![Coverage ≥95%](https://img.shields.io/badge/coverage%20requirement-%E2%89%A595%25-blue)](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md)
[![License](https://img.shields.io/badge/license-BUSL--1.1-purple)](https://github.com/alisadeghiaghili/veridist/blob/main/LICENSE)

[English](README.md) | [فارسی](README.fa.md) | [Deutsch](README.de.md)

## Why model a probability distribution?

Data contains patterns: typical outcomes, variation, and rare events. Fitting a probability distribution gives us a way to describe those patterns, estimate event probabilities, and reason about uncertainty.

When there is too little data to train and evaluate complex models such as deep neural networks reliably, statistical models with fewer parameters can be useful, provided their assumptions suit the problem. They can offer an interpretable account of statistical behaviour with limited observations. Data volume alone does not determine the right method: the analytical goal, data structure, and need for explanation matter too. Distribution modelling remains useful for large datasets.

A reference distribution can also help identify unusual observations or changes in the pattern of new data. These ideas underpin **anomaly detection** and **distribution drift monitoring**. Reliable applications also need model validation, decision thresholds, and control of false alarms.

Distribution models can later contribute to deep learning. Estimated parameters, quantiles, and threshold-exceedance probabilities can become input features alongside the original data. Evaluate whether these features help, and estimate them without future observations or test-set information to avoid leakage.

## When you do not know the distribution

Distribution fitting can involve fitting several candidate models, estimating their parameters, and comparing how well they describe the observations. A multi-model workflow can return ranked candidates with their parameters and evaluation measures.

The best-ranked fit depends on the candidates, comparison criterion, and assumptions. It need not be the true data-generating distribution; none of the candidates may be adequate.

Veridist currently provides separate Exponential, Weibull-minimum, and Lognormal fitting APIs. Automatic ranking across these three families is a future direction, not a current feature. Existing inference and selection have a narrower Exponential scope.

## What Veridist helps you do

Veridist is a Python library for **lifetime data and reliability analysis**. It helps engineers and researchers turn observations into estimates whose assumptions and execution facts can be inspected.

The name **Veridist** combines *verified* and *distribution*: distribution fitting designed for verification.

- Fit Exponential, Weibull-minimum, and Lognormal lifetime models.
- Include observations that ended before the event occurred.
- Inspect estimates, observation counts, and calculation assumptions.
- Use scalar distribution operations and chunked likelihood reduction.
- Resume compatible interrupted computations locally on supported paths.

Anomaly detection, drift monitoring, and deep learning are broader applications of distribution modelling. Veridist does not currently supply ready-made systems for these tasks.

## From an observation to a model

Imagine following a fleet of pumps. For each pump, you know how long it was observed and whether it failed.

| Pump status | What the record tells you |
| --- | --- |
| Failed during the study | Its failure time is known. |
| Still operating when observation ended | Its lifetime exceeds the recorded observation time. |

The second record is **right-censored**: observation ended before failure was seen. It still contributes information.

A statistical model is a simplified description of these times. Veridist fits the declared model; checking whether its assumptions describe your equipment remains part of the analysis.

## Your first analysis

We start with an Exponential model. It assumes a constant failure rate over time: a simple learning example, but not necessarily a suitable description of ageing equipment.

### Install

Use Python 3.11 through 3.14:

```console
python -m pip install veridist
```

### Understand the data

| time | event_observed | Meaning |
| --- | --- | --- |
| 1 | 1 | Failure was observed at time 1. |
| 1 | 0 | The pump was still operating at time 1. |

Choose one time unit, such as hours or months, and use it throughout. These two records illustrate the calculation; they are not sufficient evidence for a real reliability decision.

### Fit the model

The example creates its own CSV and runs after installation.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from veridist import (
    CsvLifetimeLimits,
    CsvLifetimeSchema,
    PublicSourceId,
    fit_exponential_csv,
)
from veridist.families import ExponentialFitSuccess

with TemporaryDirectory() as directory:
    path = Path(directory) / "lifetimes.csv"
    path.write_text("time,event_observed\n1,1\n1,0\n", encoding="utf-8")
    fit = fit_exponential_csv(
        path,
        schema=CsvLifetimeSchema("time", "event_observed"),
        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
        limits=CsvLifetimeLimits(32768, 32768),
    ).fit
assert isinstance(fit, ExponentialFitSuccess)
assert fit.rate == 0.5
assert fit.inference == "not_provided"
assert fit.censoring_assumption == "independent_right_censoring"
print(f"rate={fit.rate}; events={fit.event_count}; censored={fit.censored_count}")
```

```text
rate=0.5; events=1; censored=1
```

### Interpret the result

There is one failure across two observed time units, so the estimated rate is 1 / 2 = 0.5. If the times are months, this is 0.5 failures per pump-month of observation. It is a rate, not a 50% probability of failure within a month.

The operating pump contributes one observed time unit without failure. The model assumes censoring is independent of the unobserved failure time. Removing pumps early because they appear likely to fail would require examining that assumption.

A successful fit confirms the calculation completed; it does not prove model adequacy. Continue with the [right-censoring walkthrough](python/docs/source/exponential-right-censoring.md).

## Use your own data

Pass your CSV path to the fitting function. The public CSV entry point is Exponential-only and accepts strict UTF-8 lifetime CSV.

| Setting | Purpose |
| --- | --- |
| CsvLifetimeSchema | Names the time and event-indicator columns. |
| PublicSourceId | Supplies a non-secret source identifier in execution provenance. |
| CsvLifetimeLimits | Declares input byte budgets. |

The [API reference](python/docs/source/api.md) explains accepted input, result types, and typed failures.

## Models and tools available today

### Lifetime fitting

| Model | Pattern it can describe |
| --- | --- |
| Exponential | A constant failure rate. |
| Weibull-minimum | A decreasing, constant, or increasing failure rate, depending on shape. |
| Lognormal | Positive lifetimes whose logarithms are modelled by a Normal distribution. |

These fits use fixed location zero and support exact and independently right-censored lifetimes. Weibull-minimum and Lognormal use their model APIs; the CSV example fits Exponential only.

### Probability calculations

Scalar operations for Normal, Gamma, Weibull-minimum, Lognormal, and right-Gumbel include log-density, CDF, survival, quantiles, and caller-owned RNG sampling. An available distribution operation does not imply an available fitting API. See the [families and likelihood guide](python/docs/source/families-log-density-likelihood.md).

### Model assessment
The lower-level API exposes `FAMILY_REGISTRY`, `evaluate_log_density`, and `reduce_log_likelihood_chunks` for family lookup, scalar log-density, and chunked likelihood reduction.


Finite positive uncensored Exponential samples support refit Monte Carlo KS/AD/CvM, AIC/BIC, and adequacy-gated selection with a caller-owned generator. Inference is narrower than fitting; the [capability matrix](docs/capability-matrix.md) records the exact scope.

## When data grows

Likelihood tools can reduce caller-supplied chunks. Your application owns how data is split and delivered.

SQLiteCheckpointStore retains local restart state for compatible Exponential reductions, including the supported CSV path. Keep the source revision stable and follow the [checkpoint and resume recipe](python/examples/checkpoint_resume.py).

Release evidence covers declared CSV/Exponential paths at 10k, 100k, and 1m rows under recorded conditions. It does not establish universal throughput or a portable process-memory ceiling. Current durable resume runs on one machine with a local filesystem.

## How quality is checked

| Check | What it establishes |
| --- | --- |
| Statistical reference and API contract tests | Numerical results, boundaries, and explicit failure behaviour. |
| Coverage ≥95% | Required global line and branch coverage; numerical modules have stricter thresholds. |
| Mutation testing of the statistical core | Whether tests detect deliberately faulty code changes. |
| Package build and installation checks | Whether release artifacts can be built and installed. |
| Executable examples and multilingual documentation checks | Whether examples run and documentation builds. |

The coverage badge states the required threshold, not a measured current percentage. The CI badge reports the main workflow status.

## Before adopting Veridist

Review three things: whether the statistical assumptions match data collection, whether the needed fitting and inference path exists, and whether local processing and recovery meet your workload requirements.

The [known limits](python/KNOWN_LIMITS.md) describe exclusions such as left and interval censoring, covariates, and distributed execution. The historical distfit_pro source is tracked in the [migration ledger](docs/migration/README.md); it is not a runtime compatibility promise.

## Future plans

Development directions focus on broader model coverage, easier analysis, and statistical correctness:

- **Review and migrate the 25 legacy distributions:** 20 continuous and 5 discrete distributions, with numerical tests, documentation, and explicit capability boundaries for each migrated model.
- **Multi-model fitting and comparison:** ranked candidate results with fitted parameters, comparison measures, adequacy information, and an explicit outcome when no model is suitable.
- **Broader statistical assessment:** extend goodness-of-fit and uncertainty tools to more families and observation settings.
- **More efficient large-data processing:** measure and improve runtime and memory with reproducible experiments, alongside chunked processing.
- **More practical vignettes:** walk from a real question through data to interpretation, then explore distribution-derived features for anomaly detection, drift monitoring, and machine learning.

These are development directions, not currently supported features or promised release dates. Published capabilities remain in the [capability matrix](docs/capability-matrix.md), and delivered changes in the [changelog](python/CHANGELOG.md).

## Guides, help, and contributions

| Your goal | Where to go |
| --- | --- |
| Read the standalone package guide | [Package README](python/README.md) |
| Learn the censoring example | [Exponential walkthrough](python/docs/source/exponential-right-censoring.md) |
| Inspect inputs, outputs, and failures | [API reference](python/docs/source/api.md) |
| Report a reproducible defect | [GitHub Issues](https://github.com/alisadeghiaghili/veridist/issues) |
| Contribute | [Contribution guide](CONTRIBUTING.md) and [engineering conventions](docs/conventions.md) |
| Report a vulnerability | [Security policy](SECURITY.md) |
| Follow releases | [Changelog](python/CHANGELOG.md) |

## Cite Veridist

Cite the version that produced your result. The [citation guide](docs/citing-veridist.md) provides IEEE, APA 7, Chicago, MLA 9, Harvard, Vancouver, BibTeX, RIS, EndNote XML, and CSL-JSON formats. [CITATION.cff](CITATION.cff) is the canonical machine-readable record.

## Author and research profiles

Veridist is maintained by [Seyed Ali Sadeghi Aghili](https://zil.ink/thedatascientist).
Research and publication profiles:

- [Google Scholar](https://scholar.google.com/citations?user=BDD_JUIAAAAJ&hl=en&authuser=1)
- [ResearchGate](https://www.researchgate.net/profile/Seyed-Ali-Sadeghi-Aghili)
- [PeerJ](https://peerj.com/AliSadeghiAghili/)
- [ORCID](https://orcid.org/0000-0002-5938-3291)

## Terminology

Key terms used in this guide are given with their English names at first use:
Distribution Fitting, Likelihood, Right Censoring, Anomaly Detection, and
Distribution Drift.

## License

Veridist is distributed under **Business Source License 1.1 (BUSL-1.1)**. The [LICENSE](LICENSE) specifies the conditional Apache-2.0 additional-use grant and the change date. The BUSL-1.1 badge does not mean the current release is unconditionally licensed under Apache-2.0.
