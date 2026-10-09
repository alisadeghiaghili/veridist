# Veridist

**Evidence-first distribution fitting for inspectable statistical results and reproducible decisions.**

[![PyPI](https://img.shields.io/pypi/v/veridist.svg)](https://pypi.org/project/veridist/)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-3776AB)](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-guide.md)
[![CI](https://img.shields.io/github/actions/workflow/status/alisadeghiaghili/veridist/v1-ci.yml?branch=main&label=CI)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)
[![Coverage ≥95%](https://img.shields.io/badge/coverage%20requirement-%E2%89%A595%25-blue)](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-guide.md)
[![License](https://img.shields.io/badge/license-BUSL--1.1-purple)](https://github.com/alisadeghiaghili/veridist/blob/main/LICENSE)

[English](README.md) | [فارسی](README.fa.md) | [Deutsch](README.de.md)

## What Veridist is

Veridist is a Python library for **lifetime data and reliability analysis**. It turns failure and survival observations into estimates whose assumptions, observation counts, and execution facts can be reviewed.

The name **Veridist** combines *verified* and *distribution*: distribution fitting designed for verification. It makes a calculation inspectable; it does not declare a model true simply because fitting completed.

Use Veridist today when you need to:

- fit Exponential, Weibull-minimum, Lognormal, Gamma, Normal, or right-Gumbel models;
- retain independent right-censored observations instead of discarding them;
- report standard errors, confidence intervals, and B-lives for a fitted model; or
- reduce supported likelihood calculations in chunks and resume a compatible local Exponential CSV run.

The first file-based workflow is deliberately narrow: strict UTF-8 CSV fits a rate-only Exponential model. Every family also fits typed observation objects held in memory. See the [capability guide](docs/capability-guide.md) for the release boundary.

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
low, high = fit.uncertainty().confidence_intervals()["rate"]
assert low < fit.rate < high
print(f"rate={fit.rate}; events={fit.event_count}; censored={fit.censored_count}")
```

```text
rate=0.5; events=1; censored=1
```

### Interpret the result

There is one failure across two observed time units, so the estimated rate is 1 / 2 = 0.5. If the times are months, this is 0.5 failures per pump-month of observation. It is a rate, not a 50% probability of failure within a month.

The operating pump contributes one observed time unit without failure. The model assumes censoring is independent of the unobserved failure time. Removing pumps early because they appear likely to fail would require examining that assumption.

A successful fit confirms the calculation completed; it does not prove model adequacy. Continue with the [right-censoring walkthrough](python/docs/source/exponential-right-censoring.md).

## Why model a probability distribution?

Data has patterns: typical outcomes, variation, and rare events. A fitted probability distribution gives those patterns a compact statistical description, supports probability calculations, and makes uncertainty explicit.

When there is too little data to train and evaluate complex models such as deep neural networks reliably, lower-parameter statistical models can be useful if their assumptions suit the problem. Data volume alone does not choose the method: the analytical goal, data structure, and need for explanation matter too. Distribution modelling remains useful for large datasets.

A reference distribution can support **anomaly detection** and **distribution drift monitoring** by making unusual observations or changed patterns visible. These uses still require validation, decision thresholds, and control of false alarms.

Distribution-derived parameters, quantiles, and threshold-exceedance probabilities can later become features for deep-learning models. Evaluate those features separately and estimate them without future or test-set information to avoid leakage.

## Where the same ideas can help

In statistics, a “lifetime” can mean the time until any clearly defined event, not only the life of a machine. The event, time unit, population, and reason observation ended must be defined before fitting.

| Field | Example question | How Veridist can contribute today |
| --- | --- | --- |
| Reliability and manufacturing | How long until a pump, bearing, battery, or component fails? | Fit supported lifetime models while retaining units that were still operating when observation ended. |
| Health and survival research | How long until relapse, readmission, or another recorded event? | Analyse exact and independently right-censored times with the supported lifetime models; clinical interpretation and covariate adjustment remain outside the current package. |
| Credit and insurance | How long until default, early repayment, or the first claim? | Represent customers with no event by the study end as right-censored observations and fit a supported time-to-event model. |
| Fraud detection and cybersecurity | Is a transaction amount, time gap, or login latency unusual relative to a defensible reference distribution? | Use scalar log density, tail probability, or a quantile as one signal in a separately validated detection system. Veridist is not an end-to-end fraud classifier. |
| Operations and supply chains | What delivery, repair, waiting, or service time should we expect? | Describe positive durations with a supported model and calculate probabilities or quantiles when the required family and parameters are available. |
| Digital products and customer analytics | How long until churn, conversion, or another product event? | Treat users who remain active at the observation cutoff as right-censored, provided the censoring assumption is defensible. |

These examples share statistical structure, not identical business meaning. Domain validation, sampling design, costs, decision thresholds, and legal or safety requirements remain part of the application.

## When you do not know the distribution

Distribution fitting can mean fitting several candidate models, estimating their parameters, and comparing how well they describe the observations. A multi-model workflow can return ranked candidates with evaluation measures. The best-ranked candidate is not necessarily the true data-generating distribution, and none may be adequate.

Automatic ranking across the currently supported fitting families is a future direction for Veridist. Existing inference and adequacy-gated selection cover all six families, but only for finite, uncensored samples held in memory; a family that passes the adequacy check has not been rejected, which does not make it the true model. The [capability guide](docs/capability-guide.md) records the exact contract.

## Use your own data

Pass your CSV path to the fitting function. The public CSV entry point is Exponential-only and accepts strict UTF-8 lifetime CSV.

| Setting | Purpose |
| --- | --- |
| CsvLifetimeSchema | Names the time and event-indicator columns. |
| PublicSourceId | Supplies a non-secret source identifier in execution provenance. |
| CsvLifetimeLimits | Declares input byte budgets. |

The [API reference](python/docs/source/api.md) explains accepted input, result types, and typed failures.

## Models and tools available today

### Fitting

| Model | Pattern it can describe |
| --- | --- |
| Exponential | A constant failure rate. |
| Weibull-minimum | A decreasing, constant, or increasing failure rate, depending on shape. |
| Lognormal | Positive lifetimes whose logarithms are modelled by a Normal distribution. |
| Gamma | Positive lifetimes with a flexible, right-skewed shape. |
| Normal | Real-valued measurements around a mean. |
| Right-Gumbel | Real-valued maxima and other extreme measurements. |

The lifetime families (Exponential, Weibull-minimum, Lognormal, and Gamma) use fixed location zero and take exact and independently right-censored lifetimes; Normal and right-Gumbel take exact and right-censored real values. `fit(family, observations)` dispatches to a family, and each family also has its own function such as `fit_weibull`. The CSV example above fits Exponential only. Every successful fit has an `uncertainty()` method that returns standard errors, confidence intervals, and derived quantities such as a mean, a B-life, or a survival probability.

### Probability calculations

`logpdf`, `cdf`, `sf`, `ppf`, and `sample` take the family first and the parameters as keywords. They cover all six families and accept scalars and numpy arrays, and `lifetimes_from_arrays` and `values_from_arrays` build observations from columns. See the [families and likelihood guide](python/docs/source/families-log-density-likelihood.md); when moving from version 1.0, read the [migration guide](python/docs/migration-2.0.md).

### Model assessment
The lower-level API exposes `FAMILY_REGISTRY`, `evaluate_log_density`, and `reduce_log_likelihood_chunks` for family lookup, scalar log-density, and chunked likelihood reduction.


Finite, uncensored samples support refit Monte Carlo KS/AD/CvM for all six families (`refit_monte_carlo_gof`), and AIC/BIC with adequacy-gated selection across families (`assess_families`), with a caller-owned generator. Inference is narrower than fitting: censored observations are not supported. The [capability guide](docs/capability-guide.md) records the exact scope.

## When data grows

Likelihood tools can reduce caller-supplied chunks, with or without right censoring (`reduce_lifetime_log_likelihood_chunks` and `reduce_value_log_likelihood_chunks`). Your application owns how data is split and delivered.

SQLiteCheckpointStore retains local restart state for compatible Exponential reductions, including the supported CSV path. Keep the source revision stable and follow the [checkpoint and resume recipe](python/examples/checkpoint_resume.py).

Retained scale evidence for the strict CSV/Exponential path covers 10k, 100k, and 1m rows at 32, 64, and 128 KiB chunk limits. It comes from one run each on a Linux and a Windows runner (candidate commit `19ecf10`, 2026-10-09, CPython 3.11): 1m rows took 14 to 20 seconds, with peak process memory of about 40 to 45 MiB. These figures describe those runs only; they are not a speed or memory guarantee for other machines or data. Current durable resume runs on one machine with a local filesystem.

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

The [known limits](python/KNOWN_LIMITS.md) describe exclusions such as left and interval censoring, covariates, and distributed execution. The historical distfit_pro code is not part of Veridist and is not a runtime compatibility promise.

## Future plans

Development directions focus on broader model coverage, easier analysis, and statistical correctness:

- **Review and migrate the 25 legacy distributions:** 20 continuous and 5 discrete distributions, with numerical tests, documentation, and explicit capability boundaries for each migrated model.
- **Multi-model fitting and comparison:** ranked candidate results with fitted parameters, comparison measures, adequacy information, and an explicit outcome when no model is suitable.
- **Broader statistical assessment:** extend goodness-of-fit tools to censored observations and further observation settings.
- **More efficient large-data processing:** measure and improve runtime and memory with reproducible experiments, alongside chunked processing.
- **More practical vignettes:** walk from a real question through data to interpretation, then explore distribution-derived features for anomaly detection, drift monitoring, and machine learning.

These are development directions, not currently supported features or promised release dates. Published capabilities remain in the [capability guide](docs/capability-guide.md), and delivered changes in the [changelog](python/CHANGELOG.md).

## Guides, help, and contributions

| Your goal | Where to go |
| --- | --- |
| Read the standalone package guide | [Package README](python/README.md) |
| Learn the censoring example | [Exponential walkthrough](python/docs/source/exponential-right-censoring.md) |
| Inspect inputs, outputs, and failures | [API reference](python/docs/source/api.md) |
| Upgrade from version 1.0 | [Migration guide](python/docs/migration-2.0.md) |
| Report a reproducible defect | [GitHub Issues](https://github.com/alisadeghiaghili/veridist/issues) |
| Contribute | [Contribution guide](CONTRIBUTING.md) |
| Report a vulnerability | [Security policy](SECURITY.md) |
| Follow releases | [Changelog](python/CHANGELOG.md) |

## Cite Veridist

Cite the version that produced your result. The [citation guide](docs/citing-veridist.md) provides IEEE, APA 7, Chicago, MLA 9, Harvard, Vancouver, BibTeX, RIS, EndNote XML, and CSL-JSON formats. [CITATION.cff](CITATION.cff) is the canonical machine-readable record.

## Author and research profiles

Veridist is maintained by [Seyed Ali Sadeghi Aghili](https://linktr.ee/aliaghili).
Research and publication profiles:

[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=BDD_JUIAAAAJ&hl=en&authuser=1)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?logo=researchgate&logoColor=white)](https://www.researchgate.net/profile/Seyed-Ali-Sadeghi-Aghili)
[![PeerJ](https://img.shields.io/badge/PeerJ-00A4A6?logo=peerj&logoColor=white)](https://peerj.com/AliSadeghiAghili/)
[![ORCID](https://img.shields.io/badge/ORCID-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0002-5938-3291)

## Terminology

Key terms used in this guide are given with their English names at first use:
Distribution Fitting, Likelihood, Right Censoring, Anomaly Detection, and
Distribution Drift.

## License

Veridist is distributed under **Business Source License 1.1 (BUSL-1.1)**. It is source-available, not open source. The [LICENSE](LICENSE) grants production use only for non-commercial purposes: personal use, academic research and teaching, and use by non-profit organisations for their non-commercial activities. Any other production use, including use by or on behalf of a for-profit entity and internal business use, requires a commercial license from the licensor (alisadeghiaghili@gmail.com). On the change date, 2030-09-05, or on the fourth anniversary of a version's first public distribution if that comes first, the license changes to Apache License, Version 2.0. The BUSL-1.1 badge does not mean the current release is licensed under Apache-2.0 today.
