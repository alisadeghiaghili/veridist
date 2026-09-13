# Veridist

**Evidence-first distribution fitting for inspectable statistical results and reproducible decisions.**

[![PyPI](https://img.shields.io/pypi/v/veridist.svg)](https://pypi.org/project/veridist/)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-3776AB)](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md)
[![CI](https://img.shields.io/github/actions/workflow/status/alisadeghiaghili/veridist/v1-ci.yml?branch=main&label=CI)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)
[![Coverage ≥95%](https://img.shields.io/badge/coverage%20requirement-%E2%89%A595%25-blue)](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md)
[![License](https://img.shields.io/badge/license-BUSL--1.1-purple)](https://github.com/alisadeghiaghili/veridist/blob/main/LICENSE)

[English](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.md) | [فارسی](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.fa.md) | [Deutsch](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.de.md)

## What Veridist is

Veridist is a Python library for **lifetime data and reliability analysis**. It turns failure and survival observations into estimates whose assumptions, observation counts, and execution facts can be reviewed.

The name **Veridist** combines *verified* and *distribution*: distribution fitting designed for verification. It makes a calculation inspectable; it does not declare a model true simply because fitting completed.

Use Veridist today when you need to:

- fit fixed-location Exponential, Weibull-minimum, or Lognormal lifetime models;
- retain independent right-censored observations instead of discarding them;
- inspect the estimate alongside the declared modelling assumptions; or
- reduce supported likelihood calculations in chunks and resume a compatible local Exponential CSV run.

The first file-based workflow is deliberately narrow: strict UTF-8 CSV fits a rate-only Exponential model. Weibull-minimum and Lognormal fitting use typed lifetime objects. See the [capability matrix](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md) for the release boundary.

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

A successful fit confirms the calculation completed; it does not prove model adequacy. Continue with the [right-censoring walkthrough](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/exponential-right-censoring.md).

## Why model a probability distribution?

Data has patterns: typical outcomes, variation, and rare events. A fitted probability distribution gives those patterns a compact statistical description, supports probability calculations, and makes uncertainty explicit.

When there is too little data to train and evaluate complex models such as deep neural networks reliably, lower-parameter statistical models can be useful if their assumptions suit the problem. Data volume alone does not choose the method: the analytical goal, data structure, and need for explanation matter too. Distribution modelling remains useful for large datasets.

A reference distribution can support **anomaly detection** and **distribution drift monitoring** by making unusual observations or changed patterns visible. These uses still require validation, decision thresholds, and control of false alarms.

Distribution-derived parameters, quantiles, and threshold-exceedance probabilities can later become features for deep-learning models. Evaluate those features separately and estimate them without future or test-set information to avoid leakage.

## When you do not know the distribution

Distribution fitting can mean fitting several candidate models, estimating their parameters, and comparing how well they describe the observations. A multi-model workflow can return ranked candidates with evaluation measures. The best-ranked candidate is not necessarily the true data-generating distribution, and none may be adequate.

Automatic ranking across the currently supported fitting families is a future direction for Veridist. Existing inference and adequacy-gated selection have a narrower scope: finite, positive, uncensored Exponential samples. The [capability matrix](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md) records the exact contract.

## Use your own data

Pass your CSV path to the fitting function. The public CSV entry point is Exponential-only and accepts strict UTF-8 lifetime CSV.

| Setting | Purpose |
| --- | --- |
| CsvLifetimeSchema | Names the time and event-indicator columns. |
| PublicSourceId | Supplies a non-secret source identifier in execution provenance. |
| CsvLifetimeLimits | Declares input byte budgets. |

The [API reference](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/api.md) explains accepted input, result types, and typed failures.

## Models and tools available today

### Lifetime fitting

| Model | Pattern it can describe |
| --- | --- |
| Exponential | A constant failure rate. |
| Weibull-minimum | A decreasing, constant, or increasing failure rate, depending on shape. |
| Lognormal | Positive lifetimes whose logarithms are modelled by a Normal distribution. |

These fits use fixed location zero and support exact and independently right-censored lifetimes. Weibull-minimum and Lognormal use their model APIs; the CSV example fits Exponential only.

### Probability calculations

Scalar operations for Normal, Gamma, Weibull-minimum, Lognormal, and right-Gumbel include log-density, CDF, survival, quantiles, and caller-owned RNG sampling. An available distribution operation does not imply an available fitting API. See the [families and likelihood guide](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/families-log-density-likelihood.md).

### Model assessment
The lower-level API exposes `FAMILY_REGISTRY`, `evaluate_log_density`, and `reduce_log_likelihood_chunks` for family lookup, scalar log-density, and chunked likelihood reduction.


Finite positive uncensored Exponential samples support refit Monte Carlo KS/AD/CvM, AIC/BIC, and adequacy-gated selection with a caller-owned generator. Inference is narrower than fitting; the [capability matrix](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md) records the exact scope.

## When data grows

Likelihood tools can reduce caller-supplied chunks. Your application owns how data is split and delivered.

SQLiteCheckpointStore retains local restart state for compatible Exponential reductions, including the supported CSV path. Keep the source revision stable and follow the [checkpoint and resume recipe](https://github.com/alisadeghiaghili/veridist/blob/main/python/examples/checkpoint_resume.py).

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

The [known limits](https://github.com/alisadeghiaghili/veridist/blob/main/python/KNOWN_LIMITS.md) describe exclusions such as left and interval censoring, covariates, and distributed execution. The historical distfit_pro source is tracked in the [migration ledger](https://github.com/alisadeghiaghili/veridist/blob/main/docs/migration/README.md); it is not a runtime compatibility promise.

## Future plans

Development directions focus on broader model coverage, easier analysis, and statistical correctness:

- **Review and migrate the 25 legacy distributions:** 20 continuous and 5 discrete distributions, with numerical tests, documentation, and explicit capability boundaries for each migrated model.
- **Multi-model fitting and comparison:** ranked candidate results with fitted parameters, comparison measures, adequacy information, and an explicit outcome when no model is suitable.
- **Broader statistical assessment:** extend goodness-of-fit and uncertainty tools to more families and observation settings.
- **More efficient large-data processing:** measure and improve runtime and memory with reproducible experiments, alongside chunked processing.
- **More practical vignettes:** walk from a real question through data to interpretation, then explore distribution-derived features for anomaly detection, drift monitoring, and machine learning.

These are development directions, not currently supported features or promised release dates. Published capabilities remain in the [capability matrix](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md), and delivered changes in the [changelog](https://github.com/alisadeghiaghili/veridist/blob/main/python/CHANGELOG.md).

## Guides, help, and contributions

| Your goal | Where to go |
| --- | --- |
| Read the standalone package guide | [Package README](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.md) |
| Learn the censoring example | [Exponential walkthrough](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/exponential-right-censoring.md) |
| Inspect inputs, outputs, and failures | [API reference](https://github.com/alisadeghiaghili/veridist/blob/main/python/docs/source/api.md) |
| Report a reproducible defect | [GitHub Issues](https://github.com/alisadeghiaghili/veridist/issues) |
| Contribute | [Contribution guide](https://github.com/alisadeghiaghili/veridist/blob/main/CONTRIBUTING.md) and [engineering conventions](https://github.com/alisadeghiaghili/veridist/blob/main/docs/conventions.md) |
| Report a vulnerability | [Security policy](https://github.com/alisadeghiaghili/veridist/blob/main/SECURITY.md) |
| Follow releases | [Changelog](https://github.com/alisadeghiaghili/veridist/blob/main/python/CHANGELOG.md) |

## Cite Veridist

Cite the version that produced your result. The [citation guide](https://github.com/alisadeghiaghili/veridist/blob/main/docs/citing-veridist.md) provides IEEE, APA 7, Chicago, MLA 9, Harvard, Vancouver, BibTeX, RIS, EndNote XML, and CSL-JSON formats. [CITATION.cff](https://github.com/alisadeghiaghili/veridist/blob/main/CITATION.cff) is the canonical machine-readable record.

## Author and research profiles

Maintainer: [Seyed Ali Sadeghi Aghili](https://linktr.ee/aliaghili).

[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=BDD_JUIAAAAJ&hl=en&authuser=1)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?logo=researchgate&logoColor=white)](https://www.researchgate.net/profile/Seyed-Ali-Sadeghi-Aghili)
[![PeerJ](https://img.shields.io/badge/PeerJ-00A4A6?logo=peerj&logoColor=white)](https://peerj.com/AliSadeghiAghili/)
[![ORCID](https://img.shields.io/badge/ORCID-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0002-5938-3291)

## Terminology

Important terms are named in English at first use: Distribution Fitting,
Likelihood, Right Censoring, Anomaly Detection, and Distribution Drift.

## License

Veridist is distributed under **Business Source License 1.1 (BUSL-1.1)**. The [LICENSE](https://github.com/alisadeghiaghili/veridist/blob/main/LICENSE) specifies the conditional Apache-2.0 additional-use grant and the change date. The BUSL-1.1 badge does not mean the current release is unconditionally licensed under Apache-2.0.
