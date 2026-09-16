(veridist-api)=
# Veridist API Guide

This guide explains the current public API in Veridist. For fitting an exponential distribution<sup id="fnref-fitting"><a href="#fn-fitting">1</a></sup> to lifetime data, the usual path is to pass one CSV file to the main function and read the returned result. If a calculation is long and may be interrupted, you can save progress and continue later. Scalar tools<sup id="fnref-scalar"><a href="#fn-scalar">2</a></sup> and data streams managed by the caller<sup id="fnref-caller"><a href="#fn-caller">3</a></sup> are also available for more technical use.

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
      <td><code>fit_exponential_csv</code></td>
      <td>It does not provide resume, cancellation, or automatic model selection.</td>
    </tr>
    <tr>
      <td>Continue a calculation from chunked JSON data</td>
      <td><code>fit_exponential_checkpointed_chunks</code></td>
      <td>Your program must prepare and read the data chunks.</td>
    </tr>
    <tr>
      <td>Process a CSV file with cancellation and resume from the last saved row</td>
      <td><code>fit_exponential_checkpointed_csv</code></td>
      <td>It is only for a local file on one machine; it does not provide distributed recovery<sup id="fnref-distributed-recovery"><a href="#fn-distributed-recovery">4</a></sup>.</td>
    </tr>
    <tr>
      <td>Calculate log density for one value</td>
      <td><code>evaluate_log_density</code></td>
      <td>It does not fit parameters.</td>
    </tr>
    <tr>
      <td>Calculate likelihood for several data chunks</td>
      <td><code>reduce_log_likelihood_chunks</code></td>
      <td>It does not choose the best distribution.</td>
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

This example has two observations. In the first observation, the event happens at time 1. In the second observation, no event has been observed by time 1, so all we know is that the real lifetime is greater than 1. That gives one observed event and 2 total units of observed time. The estimated rate<sup id="fnref-rate"><a href="#fn-rate">5</a></sup> is `1 / 2 = 0.5` events per unit of time. This is not a 50 percent event probability; its unit depends on the unit of the `time` column.

## What should the CSV file look like?

`fit_exponential_csv(path, *, schema, source_id, limits)` accepts a UTF-8 file with two columns, `time,event_observed`, in exactly that order. `time` must be a finite non-negative number. In `event_observed`, `1` means the event happened at the recorded time; `0` means the event had not yet been observed by the recorded time. The second case is independent right censoring<sup id="fnref-right-censoring"><a href="#fn-right-censoring">6</a></sup>.

`CsvLifetimeSchema` names the two expected columns. `PublicSourceId` is a public, non-secret identifier for recording data provenance<sup id="fnref-provenance"><a href="#fn-provenance">7</a></sup>; the local file path is not placed in the returned result. `CsvLifetimeLimits` sets the maximum size of each data chunk and the maximum amount of data kept in the processing queue at the same time<sup id="fnref-byte-limits"><a href="#fn-byte-limits">8</a></sup>. Both values must be positive.

Veridist does not guess column names, delimiters, encoding, missing data, or the meaning of zero and one. If the file does not match the contract above, it reports the problem explicitly and does not silently rewrite the data.

## How should you read the result?

`fit_exponential_csv` always returns an `ExponentialSourceFitResult`. If the file is processed successfully and the data is sufficient to estimate a rate, the fitted result is stored in `result.fit`. If the file is processed correctly but a statistically valid rate cannot be estimated, for example because the file is empty or no event was observed, the same field returns a typed statistical non-estimate<sup id="fnref-typed-non-estimate"><a href="#fn-typed-non-estimate">9</a></sup> so the reason remains explicit.

If reading or processing the file fails, `result.fit` is `None`. In that case, `result.execution` is a typed outcome<sup id="fnref-typed-outcome"><a href="#fn-typed-outcome">10</a></sup> that records the stage and reason for the failure. Check the result type before using model parameters.

The current model keeps the location parameter fixed at zero. This path does not yet provide confidence intervals<sup id="fnref-confidence-interval"><a href="#fn-confidence-interval">11</a></sup>, goodness-of-fit tests<sup id="fnref-goodness-of-fit"><a href="#fn-goodness-of-fit">12</a></sup>, weights<sup id="fnref-weight"><a href="#fn-weight">13</a></sup>, covariates<sup id="fnref-covariate"><a href="#fn-covariate">14</a></sup>, data truncation<sup id="fnref-truncation"><a href="#fn-truncation">15</a></sup>, left censoring<sup id="fnref-left-censoring"><a href="#fn-left-censoring">16</a></sup>, interval censoring<sup id="fnref-interval-censoring"><a href="#fn-interval-censoring">17</a></sup>, a free location parameter<sup id="fnref-free-location"><a href="#fn-free-location">18</a></sup>, or automatic model selection. The statistical assumption and non-estimate cases are explained in the [right-censoring tutorial](exponential-right-censoring.md).

## Saving progress and continuing a calculation

For data that your program has already split into small JSON chunks, import `fit_exponential_checkpointed_chunks` from the `veridist` package. The function processes each chunk separately and stores the sufficient statistics<sup id="fnref-sufficient-statistics"><a href="#fn-sufficient-statistics">19</a></sup> needed to continue the calculation; it does not store a copy of the original data rows in the state file.

For CSV files, `veridist.execution.fit_exponential_checkpointed_csv` does the same work with cancellation support. You provide the file, the column definition, the size limits, the state storage location, the data revision identifier, and, if needed, a `cancel(cursor)` callback. If the run is cancelled, the completed part is saved first; the next run can continue from the last recorded row.

The data file and its revision identifier must remain stable between runs. The checkpoint mechanism<sup id="fnref-checkpoint"><a href="#fn-checkpoint">20</a></sup> is designed to continue a calculation on the same machine. The current implementation uses local SQLite, but it does not copy raw CSV rows into that store. The file is not automatically encrypted or authenticated, so keep it in a safe location like other working files. The [save-and-resume example](../../examples/checkpoint_resume.py) shows the two-step flow.

## Low-level tools<sup id="fnref-low-level-tools"><a href="#fn-low-level-tools">21</a></sup> for prepared data

If your own program generates or chunks the data, `IterableDataSource` accepts those chunks with source metadata. In `SINGLE_PASS` mode, the data is read once. In `REPLAYABLE` mode, you must provide a function that creates a fresh traversal of the data each time. `CHECKPOINT_REPLAYABLE` is not implemented in this adapter yet; using it raises `CHECKPOINT_REQUIRED`.

`FAMILY_REGISTRY` and `FamilyId` hold the metadata for the five evaluated statistical families. `evaluate_log_density` calculates the log density of one value with supplied parameters. `reduce_log_likelihood_chunks` sums the same calculation across several data chunks and returns a result that does not depend on how the data was chunked. These functions do not estimate model parameters, rank distributions, or build likelihoods for censored data. Their exact contract is documented in [evaluated families and log likelihood](families-log-density-likelihood.md).

<details>
<summary>Technical details and limits of the current version</summary>

The CSV file is read in one pass. The size limits only control the volume of data chunks retained by the adapter itself; they are not a ceiling on total process memory or execution speed. General support for all data sources, distributed execution, and recovery across multiple machines is not available in the current version; planned items and exact boundaries are recorded in <a href="../../KNOWN_LIMITS.md">known limits</a>. A successful calculation also does not prove that the exponential distribution is appropriate for the data.

</details>

## Terms Used On This Page

<p id="fn-fitting"><strong>1.</strong> <bdi>Distribution fitting</bdi> — estimating the parameters of a distribution from data and checking its compatibility with the observations. <a href="#fnref-fitting" aria-label="Back to text">↩</a></p>
<p id="fn-scalar"><strong>2.</strong> <bdi>Scalar operation</bdi> — an operation that works on one numeric value at a time, not on a full array. <a href="#fnref-scalar" aria-label="Back to text">↩</a></p>
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
