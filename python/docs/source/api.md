(veridist-api)=
# Veridist API Guide

<a href="api.md">English</a> | <a href="api.fa.md">فارسی</a> | <a href="api.de.md">Deutsch</a>

This guide explains the current public API in Veridist 2.0. For fitting a distribution<sup id="fnref-fitting"><a href="#fn-fitting">1</a></sup> to lifetime or measurement data, the usual path is to pass your observations, or one CSV file, to a fit function and read the returned result. Six distribution families have a fit, every distribution operation also accepts numpy arrays, and every successful fit can report the uncertainty of its estimate<sup id="fnref-uncertainty"><a href="#fn-uncertainty">22</a></sup>. If a calculation is long and may be interrupted, you can save progress and continue later. Scalar tools<sup id="fnref-scalar"><a href="#fn-scalar">2</a></sup> and data streams managed by the caller<sup id="fnref-caller"><a href="#fn-caller">3</a></sup> are also available for more technical use. If you are moving from version 1.0, read the <a href="../migration-2.0.md">migration guide</a>.

## Use the same event-time contract in different fields

The two CSV columns describe a general time-to-event record. `time` says how long the item was observed. `event_observed` says whether the chosen event happened during that time. The code does not decide what the event means; your analysis must define it consistently.

| Application field | What `time` can mean | What event `1` can mean | What event `0` can mean |
| --- | --- | --- | --- |
| Reliability record | Hours a component was observed | The component failed | It was still operating when observation ended |
| Health-research record | Days a participant was followed | The defined health event occurred | The event had not occurred by the last follow-up |
| Credit or insurance record | Months an account or policy was followed | Default or the first claim occurred | No such event had occurred by the study end |
| Digital-product record | Days since signup or campaign entry | Churn or conversion occurred | The user remained active without the event at the cutoff |
| Operations record | Hours since a repair, order, or case opened | The process completed | The process was still open when data collection ended |

For fraud detection or cybersecurity, the time-to-event CSV may fit questions such as time until an alert, but it is not a general fraud-data format. The lower-level distribution tools can also compare a transaction amount, time gap, or latency with an already specified reference distribution. Such a value is one input to a separately tested detection system, not a fraud decision by itself.

## Choose an execution path

<table width="100%">
  <thead>
    <tr>
      <th width="36%" align="center"><p align="center">What you want to do</p></th>
      <th width="32%" align="center"><p align="center">Function</p></th>
      <th width="32%" align="center"><p align="center">Limit of this path</p></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Read a CSV file and fit one exponential model in one run</td>
      <td><code class="literal">fit_exponential_csv</code></td>
      <td>It does not provide resume, cancellation, or automatic model selection.</td>
    </tr>
    <tr>
      <td>Continue a calculation from chunked JSON data</td>
      <td><code class="literal">fit_exponential_checkpointed_chunks</code></td>
      <td>Your program must prepare and read the data chunks.</td>
    </tr>
    <tr>
      <td>Process a CSV file with cancellation and resume from the last saved row</td>
      <td><code class="literal">fit_exponential_checkpointed_csv</code></td>
      <td>It is only for a local file on one machine; it does not provide distributed recovery<sup id="fnref-distributed-recovery"><a href="#fn-distributed-recovery">4</a></sup>.</td>
    </tr>
    <tr>
      <td>Fit one of six families to observations held in memory</td>
      <td><code class="literal">fit</code>, <code class="literal">fit_weibull</code>, <code class="literal">fit_gamma</code>, and the other per-family fits</td>
      <td>All observations must fit in memory; there is no automatic model selection.</td>
    </tr>
    <tr>
      <td>Calculate density, CDF, survival, quantiles, or samples for a value or an array</td>
      <td><code class="literal">logpdf</code>, <code class="literal">cdf</code>, <code class="literal">sf</code>, <code class="literal">ppf</code>, <code class="literal">sample</code></td>
      <td>They use a distribution you already specified; they do not fit parameters.</td>
    </tr>
    <tr>
      <td>Calculate log density for one value</td>
      <td><code class="literal">evaluate_log_density</code></td>
      <td>It does not fit parameters.</td>
    </tr>
    <tr>
      <td>Calculate likelihood for several data chunks, with or without right censoring</td>
      <td><code class="literal">reduce_log_likelihood_chunks</code>, <code class="literal">reduce_lifetime_log_likelihood_chunks</code>, <code class="literal">reduce_value_log_likelihood_chunks</code></td>
      <td>They do not choose the best distribution.</td>
    </tr>
  </tbody>
</table>

Run the complete example below for the usual path:

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
    result = fit_exponential_csv(
        path,
        schema=CsvLifetimeSchema("time", "event_observed"),
        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
        limits=CsvLifetimeLimits(32768, 32768),
    )

fit = result.fit
assert isinstance(fit, ExponentialFitSuccess)
print(f"rate={fit.rate}; events={fit.event_count}; censored={fit.censored_count}")
```

```text
rate=0.5; events=1; censored=1
```

This example has two observations. In the first observation, the event happens at time 1. In the second observation, no event has been observed by time 1, so all we know is that the real lifetime is greater than 1. That gives one observed event and 2 total units of observed time. The estimated rate<sup id="fnref-rate"><a href="#fn-rate">5</a></sup> is `1 / 2 = 0.5` events per unit of time. This means that the model estimates, on average, half an event per unit of observed time across comparable units. Calculating an event probability also requires a specified time interval.

## What should the CSV file look like?

`fit_exponential_csv(path, *, schema, source_id, limits)` accepts a UTF-8 file with two columns, `time,event_observed`, in exactly that order. `time` must be a finite non-negative number. In `event_observed`, `1` means the event happened at the recorded time; `0` means the event had not yet been observed by the recorded time. The second case is independent right censoring<sup id="fnref-right-censoring"><a href="#fn-right-censoring">6</a></sup>.

`CsvLifetimeSchema` names the two expected columns. `PublicSourceId` is a public, non-secret identifier for recording data provenance<sup id="fnref-provenance"><a href="#fn-provenance">7</a></sup>; the local file path is not placed in the returned result. `CsvLifetimeLimits` sets the maximum size of each data chunk and the maximum amount of data kept in the processing queue at the same time<sup id="fnref-byte-limits"><a href="#fn-byte-limits">8</a></sup>. Both values must be positive.

Veridist does not guess column names, delimiters, encoding, missing data, or the meaning of zero and one. If the file does not follow the format above, processing stops with a clear error; Veridist does not automatically change questionable values.

## How should you read the result?

`fit_exponential_csv` always returns an `ExponentialSourceFitResult`. If the file is processed successfully and the data is sufficient to estimate a rate, the fitted result is stored in `result.fit`. If the file is processed correctly but a statistically valid rate cannot be estimated, for example because the file is empty or no event was observed, the same field returns a typed statistical non-estimate<sup id="fnref-typed-non-estimate"><a href="#fn-typed-non-estimate">9</a></sup> so the reason remains explicit.

If reading or processing the file fails, `result.fit` is `None`. In that case, `result.execution` is a typed outcome<sup id="fnref-typed-outcome"><a href="#fn-typed-outcome">10</a></sup> that records the stage and reason for the failure. Check the result type before using model parameters.

The CSV model keeps the location parameter fixed at zero. The fit it returns can report the uncertainty of its estimate (see the uncertainty section below), but this path does not provide goodness-of-fit tests<sup id="fnref-goodness-of-fit"><a href="#fn-goodness-of-fit">12</a></sup>, weights<sup id="fnref-weight"><a href="#fn-weight">13</a></sup>, covariates<sup id="fnref-covariate"><a href="#fn-covariate">14</a></sup>, data truncation<sup id="fnref-truncation"><a href="#fn-truncation">15</a></sup>, left censoring<sup id="fnref-left-censoring"><a href="#fn-left-censoring">16</a></sup>, interval censoring<sup id="fnref-interval-censoring"><a href="#fn-interval-censoring">17</a></sup>, a free location parameter<sup id="fnref-free-location"><a href="#fn-free-location">18</a></sup>, or automatic model selection. The statistical assumption and non-estimate cases are explained in <a href="exponential-right-censoring.md">the right-censoring tutorial</a>.

## Fit any of the six families

`fit(family, observations)` fits one of six families by maximum likelihood to observations held in memory. `family` is a `FamilyId` or its string value: `exponential`, `weibull_min`, `lognormal`, `gamma`, `normal`, or `gumbel_right`. Each family also has its own function (`fit_exponential`, `fit_weibull`, `fit_lognormal`, `fit_gamma`, `fit_normal`, and `fit_gumbel_right`) that takes the same observations and options.

The four lifetime families (`exponential`, `weibull_min`, `lognormal`, and `gamma`) take `ExactLifetime` and `RightCensoredLifetime` observations. The two families on the whole real line (`normal` and `gumbel_right`) take `ExactValue` and `RightCensoredValue`, and a value may be negative. Passing the other pair raises `TypeError`. Every family supports independent right censoring and `frequency_weights`; `fit_weibull` also accepts `fixed_shape`.

```python
from veridist import ExactLifetime, RightCensoredLifetime, fit

times = (120.0, 340.0, 560.0, 800.0, 1250.0, 1700.0)
observations = [ExactLifetime(t) for t in times] + [RightCensoredLifetime(2000.0)]
result = fit("weibull_min", observations)

shape, scale = result.parameters["shape"], result.parameters["scale"]
print(f"{result.family.value} shape={shape:.3f} scale={scale:.1f}")
print(result.observation_count, result.event_count, result.censored_count)
```

```text
weibull_min shape=1.216 scale=1154.2
7 6 1
```

A data problem, such as a sample with no observed event, is returned as a typed failure and is not raised. A success exposes `family`, a read-only `parameters` mapping with the canonical parameter names, `log_likelihood`, `observation_count`, `event_count`, `censored_count`, and `converged`, together with the family's own attributes such as `rate`, `shape`, `scale`, `mu`, or `sigma`. A failure exposes `family`, `code`, and the counts. The `FitSuccess` and `FitFailure` protocols describe this common surface, so `isinstance(result, FitSuccess)` is the check to make before reading parameters. A fit never reports a point at the edge of its search range as a converged estimate; it returns a failure such as `BOUNDARY_SOLUTION` or `DEGENERATE_SAMPLE` instead.

## Evaluate distributions on scalars and arrays

`logpdf`, `cdf`, `sf`, `ppf`, and `sample` evaluate a distribution you have already specified. The family comes first and the parameters follow as keywords, for example `cdf("gamma", 6.5, shape=5.0, scale=1.0)`. `logpdf` is the log density, `sf` the survival function, and `ppf` the quantile function; `sample(family, size, rng=rng, ...)` draws from the family with a numpy generator that you provide.

The point and every parameter may be a Python or numpy scalar, or an array, and they are broadcast<sup id="fnref-broadcasting"><a href="#fn-broadcasting">23</a></sup> against each other. A scalar call returns a Python `float`, and any array input returns a `float64` array of the broadcast shape. `logpdf` is `-inf` outside the support, and `ppf` needs every probability strictly between 0 and 1. numpy is imported only when you pass an array.

```python
import numpy as np

from veridist import cdf, logpdf, ppf

x = np.array([100.0, 200.0, 400.0])
print(np.round(cdf("weibull_min", x, shape=1.5, scale=500.0), 4))
print(round(logpdf("normal", 0.0, mu=0.0, sigma=1.0), 6))
print(round(ppf("gamma", 0.5, shape=2.0, scale=1.0), 6))
```

```text
[0.0856 0.2235 0.5111]
-0.918939
1.678347
```

The exponential, Weibull-minimum, and right-Gumbel families use numpy-native kernels. The normal, lognormal, and gamma families evaluate the verified scalar kernels element by element, so they give the same values as a scalar call but are slow on very large arrays. The older form that passes the parameters as a mapping, `cdf("gamma", x, {"shape": 2.0, "scale": 1.0})`, still works but is deprecated and will be removed in 3.0.

## Build observations from arrays

`lifetimes_from_arrays(time, event)` builds the lifetime observations from two one-dimensional columns of equal length, and `values_from_arrays(value, event)` does the same for real values. `event` is a boolean array or an integer array of 0 and 1; true means the event was observed, and false means the observation is right censored at that time. The columns are validated as whole arrays first, and an error names the first bad row.

```python
import numpy as np

from veridist import fit, lifetimes_from_arrays

time = np.array([120.0, 340.0, 560.0, 800.0, 2000.0])
event = np.array([1, 1, 1, 1, 0])
result = fit("exponential", lifetimes_from_arrays(time, event))
print(round(result.rate, 8))
```

```text
0.00104712
```

## Report the uncertainty of an estimate

Every successful fit has an `uncertainty()` method; call it on the result, because it is not an attribute. It returns a `FitUncertainty` with the parameter `standard_errors`, the `covariance` matrix<sup id="fnref-covariance"><a href="#fn-covariance">24</a></sup> in the canonical parameter order, and `confidence_intervals(level=0.95, method="wald")`<sup id="fnref-confidence-interval"><a href="#fn-confidence-interval">11</a></sup>. `method="wald"` forms intervals for positive parameters on the log scale, so they never contain a non-positive value. `method="profile"` inverts the profile likelihood<sup id="fnref-profile-likelihood"><a href="#fn-profile-likelihood">25</a></sup>, which is more reliable for small samples; a side that never crosses is reported as `inf` (or `0`). `method="exact"` gives the chi-square interval for the exponential rate on uncensored data.

The same object gives derived quantities with intervals: `mean()` (the mean time to failure), `quantile(p)` (a B-life<sup id="fnref-b-life"><a href="#fn-b-life">26</a></sup>; `quantile(0.1)` is B10), and `survival(t)`. Each returns a `DerivedEstimate` with `estimate`, `lower`, `upper`, `method`, and `level`. Wald intervals use the delta method on a scale that respects the range of the quantity, and `method="profile"` is available for the exponential and Weibull families.

```python
from veridist import ExactLifetime, RightCensoredLifetime, fit

times = (120.0, 340.0, 560.0, 800.0, 1250.0, 1700.0)
observations = [ExactLifetime(t) for t in times] + [RightCensoredLifetime(2000.0)]
uncertainty = fit("weibull_min", observations).uncertainty()

intervals = uncertainty.confidence_intervals()
b10 = uncertainty.quantile(0.1)
print(tuple(round(value, 3) for value in intervals["shape"]))
print(round(b10.estimate, 1), round(b10.lower, 1), round(b10.upper, 1))
```

```text
(0.621, 2.382)
181.4 40.9 804.5
```

When the observed information is singular, a Weibull shape was fixed, or the result carries no data, `uncertainty()` returns an `UncertaintyUnavailable` value with a stable `reason` instead of raising. Check the type before reading intervals.

## Saving progress and continuing a calculation

For data that your program has already split into small JSON chunks, import `fit_exponential_checkpointed_chunks` from the `veridist` package. The function processes each chunk separately and stores the sufficient statistics<sup id="fnref-sufficient-statistics"><a href="#fn-sufficient-statistics">19</a></sup> needed to continue the calculation; it does not store a copy of the original data rows in the state file.

For CSV files, `veridist.execution.fit_exponential_checkpointed_csv` does the same work with cancellation support. You provide the file, the column definition, the size limits, the state storage location, the data revision identifier, and, if needed, a `cancel(cursor)` callback. The revision identifier is not a free-form label: it must equal the CSV file's current SHA-256 hex digest, computed by streaming the file before any row is read. If the run is cancelled, the completed part is saved first; the next run can continue from the last recorded row.

The data file and its revision identifier must remain stable between runs: a changed file, a different public source identifier, or a different stored schema returns a typed mismatch instead of resuming. Use `veridist.execution.create_checkpointed_csv_store` to build the initial local store for one CSV file instead of constructing the checkpoint record by hand. The checkpoint mechanism<sup id="fnref-checkpoint"><a href="#fn-checkpoint">20</a></sup> is designed to continue a calculation on the same machine. The current implementation uses local SQLite, but it does not copy raw CSV rows into that store. The file is not automatically encrypted or authenticated, so keep it in a safe location like other working files. The [save-and-resume example](../../examples/checkpoint_resume.py) shows the two-step flow.

## Low-level tools<sup id="fnref-low-level-tools"><a href="#fn-low-level-tools">21</a></sup> for prepared data

If your own program generates or chunks the data, `IterableDataSource` accepts those chunks with immutable `DataSourceMetadata` and an explicit `Replayability` declaration. In `SINGLE_PASS` mode, the data is read once. In `REPLAYABLE` mode, you must provide a function that creates a fresh traversal of the data each time. `CHECKPOINT_REPLAYABLE` is not implemented in this adapter yet; using it raises `CHECKPOINT_REQUIRED`.

`FAMILY_REGISTRY` and `FamilyId` hold the metadata for the six evaluated statistical families. `evaluate_log_density` calculates the log density of one value with supplied parameters. `reduce_log_likelihood_chunks` sums the same calculation across several data chunks and returns a result that does not depend on how the data was chunked. These functions do not estimate model parameters or rank distributions. Their exact contract is documented in <a href="families-log-density-likelihood.md">evaluated families and log likelihood</a>.

For censored data, `reduce_lifetime_log_likelihood_chunks` (families `exponential`, `weibull_min`, `lognormal`, and `gamma`, with `ExactLifetime` and `RightCensoredLifetime`) and `reduce_value_log_likelihood_chunks` (families `normal` and `gumbel_right`, with `ExactValue` and `RightCensoredValue`) add the log density of every exact observation and the log survival of every right-censored one. They take a `FamilyId` and the canonical parameters, accumulate exactly like `reduce_log_likelihood_chunks`, and at a fit's parameters reproduce that fit's `log_likelihood`.

```python
from veridist import (
    ExactLifetime,
    FamilyId,
    RightCensoredLifetime,
    reduce_lifetime_log_likelihood_chunks,
)

chunks = [[ExactLifetime(120.0), ExactLifetime(340.0)], [RightCensoredLifetime(2000.0)]]
result = reduce_lifetime_log_likelihood_chunks(
    FamilyId.WEIBULL_MIN, chunks, shape=1.2, scale=1100.0
)
print(result.observation_count, round(result.total_log_likelihood, 6))
```

```text
3 -16.682975
```

## What the top-level package exports

Import the public API from `veridist`: the six fits and `fit`; `FamilyId`, `logpdf`, `cdf`, `sf`, `ppf`, and `sample`; the observation types with `lifetimes_from_arrays` and `values_from_arrays`; the `FitSuccess` and `FitFailure` protocols; the CSV and checkpoint entry points, including `create_checkpointed_csv_store` and `fit_exponential_checkpointed_csv`; the three log-likelihood reducers; and the error classes `VeridistError`, `CapabilityError`, and `EngineContractError`. Stores, buffers, provenance, and outcome types stay in `veridist.engine`, which also exports `CapabilityError`.

## Upgrading from version 1.0

Existing 1.0 code keeps working. Two forms are deprecated and will be removed in 3.0: the mapping form of `cdf`, `sf`, `ppf`, and `sample`, and bare `bytes` chunks for `fit_exponential_checkpointed_chunks`. The <a href="../migration-2.0.md">migration guide</a> lists every change, including the license change to the Business Source License 1.1 with a non-commercial grant.

<details>
<summary>Technical details and limits of the current version</summary>

The CSV file is read in one pass. The size limits only control the volume of data chunks retained by the adapter itself; they are not a ceiling on total process memory or execution speed. General support for all data sources, distributed execution, and recovery across multiple machines is not available in the current version; planned items and exact boundaries are recorded in <a href="../../KNOWN_LIMITS.md">known limits</a>. A successful calculation also does not prove that the exponential distribution is appropriate for the data.

</details>

## Terms Used On This Page

<p id="fn-fitting"><strong>1.</strong> <bdi>Distribution fitting</bdi> — estimating the parameters of a distribution from data and checking its compatibility with the observations. <a href="#fnref-fitting" aria-label="Back to text">↩</a></p>
<p id="fn-scalar"><strong>2.</strong> <bdi>Scalar operation</bdi> — an operation on a single numeric value; the same operations also accept arrays. <a href="#fnref-scalar" aria-label="Back to text">↩</a></p>
<p id="fn-caller"><strong>3.</strong> <bdi>Caller</bdi> — the code or program that calls the library function and supplies its inputs. <a href="#fnref-caller" aria-label="Back to text">↩</a></p>
<p id="fn-distributed-recovery"><strong>4.</strong> <bdi>Distributed recovery</bdi> — continuing a calculation on another computer or service with shared state. <a href="#fnref-distributed-recovery" aria-label="Back to text">↩</a></p>
<p id="fn-rate"><strong>5.</strong> <bdi>Rate</bdi> — the expected number of events per unit of time; a rate is not the same as event probability. <a href="#fnref-rate" aria-label="Back to text">↩</a></p>
<p id="fn-right-censoring"><strong>6.</strong> <bdi>Independent right censoring</bdi> — the event has not been observed by the end of observation, and the reason observation ended is assumed independent of the event time. <a href="#fnref-right-censoring" aria-label="Back to text">↩</a></p>
<p id="fn-provenance"><strong>7.</strong> <bdi>Provenance</bdi> — recorded information about the data source and execution so the result can be inspected later. <a href="#fnref-provenance" aria-label="Back to text">↩</a></p>
<p id="fn-byte-limits"><strong>8.</strong> <bdi>Byte limits</bdi> — limits on how much data each chunk and the processing queue can hold at the same time; this is not a limit on total program RAM. <a href="#fnref-byte-limits" aria-label="Back to text">↩</a></p>
<p id="fn-typed-non-estimate"><strong>9.</strong> <bdi>Typed statistical non-estimate</bdi> — a structured result saying that the calculation ran but the data did not allow a valid estimate, with the reason preserved. <a href="#fnref-typed-non-estimate" aria-label="Back to text">↩</a></p>
<p id="fn-typed-outcome"><strong>10.</strong> <bdi>Typed outcome</bdi> — a result with a defined type and code that software can inspect without parsing free text. <a href="#fnref-typed-outcome" aria-label="Back to text">↩</a></p>
<p id="fn-confidence-interval"><strong>11.</strong> <bdi>Confidence interval</bdi> — an interval that shows uncertainty in a parameter estimate under a specified statistical method. <a href="#fnref-confidence-interval" aria-label="Back to text">↩</a></p>
<p id="fn-goodness-of-fit"><strong>12.</strong> <bdi>Goodness-of-fit test</bdi> — a statistical check of how well the chosen distribution shape matches the data. <a href="#fnref-goodness-of-fit" aria-label="Back to text">↩</a></p>
<p id="fn-weight"><strong>13.</strong> <bdi>Weight</bdi> — a number that makes an observation contribute more or less to the calculation. <a href="#fnref-weight" aria-label="Back to text">↩</a></p>
<p id="fn-covariate"><strong>14.</strong> <bdi>Covariate</bdi> — a variable such as temperature or pressure that may affect the event time. <a href="#fnref-covariate" aria-label="Back to text">↩</a></p>
<p id="fn-truncation"><strong>15.</strong> <bdi>Truncation</bdi> — part of the population is excluded from the sample because of the entry or observation process, not merely because the event time is unknown. <a href="#fnref-truncation" aria-label="Back to text">↩</a></p>
<p id="fn-left-censoring"><strong>16.</strong> <bdi>Left censoring</bdi> — we only know that the event happened before a specified time. <a href="#fnref-left-censoring" aria-label="Back to text">↩</a></p>
<p id="fn-interval-censoring"><strong>17.</strong> <bdi>Interval censoring</bdi> — we only know that the event happened between two times. <a href="#fnref-interval-censoring" aria-label="Back to text">↩</a></p>
<p id="fn-free-location"><strong>18.</strong> <bdi>Free location parameter</bdi> — a distribution shift parameter that is estimated from data instead of being fixed. <a href="#fnref-free-location" aria-label="Back to text">↩</a></p>
<p id="fn-sufficient-statistics"><strong>19.</strong> <bdi>Sufficient statistics</bdi> — numeric summaries needed to estimate the parameter, replacing the need to keep all raw rows in this calculation. <a href="#fnref-sufficient-statistics" aria-label="Back to text">↩</a></p>
<p id="fn-checkpoint"><strong>20.</strong> <bdi>Checkpoint</bdi> — saved intermediate state that lets a compatible run continue from the last recorded chunk. <a href="#fnref-checkpoint" aria-label="Back to text">↩</a></p>
<p id="fn-low-level-tools"><strong>21.</strong> <bdi>Low-level tools</bdi> — more basic functions for cases where your program performs data preparation, data chunking, or parameter selection itself. These tools are usually not the quick-start path for end users. <a href="#fnref-low-level-tools" aria-label="Back to text">↩</a></p>
<p id="fn-uncertainty"><strong>22.</strong> <bdi>Uncertainty</bdi> — how far an estimate computed from a limited sample may be from the true value, usually reported as standard errors and confidence intervals. <a href="#fnref-uncertainty" aria-label="Back to text">↩</a></p>
<p id="fn-broadcasting"><strong>23.</strong> <bdi>Broadcasting</bdi> — numpy's rule for combining arrays of different but compatible shapes element by element. <a href="#fnref-broadcasting" aria-label="Back to text">↩</a></p>
<p id="fn-covariance"><strong>24.</strong> <bdi>Covariance matrix</bdi> — a table of the variances of the parameter estimates and the covariances between pairs of them; the square roots of its diagonal are the standard errors. <a href="#fnref-covariance" aria-label="Back to text">↩</a></p>
<p id="fn-profile-likelihood"><strong>25.</strong> <bdi>Profile likelihood</bdi> — the likelihood maximized over the other parameters for each value of the one of interest; intervals built from it follow the true shape of the likelihood. <a href="#fnref-profile-likelihood" aria-label="Back to text">↩</a></p>
<p id="fn-b-life"><strong>26.</strong> <bdi>B-life</bdi> — the time by which a given percentage of units has failed; B10 is the time by which 10 percent have failed. <a href="#fnref-b-life" aria-label="Back to text">↩</a></p>
