(veridist-exponential-right-censoring)=
# Exponential right-censoring tutorial

<a href="exponential-right-censoring.md">English</a> | <a href="exponential-right-censoring.fa.md">فارسی</a> | <a href="exponential-right-censoring.de.md">Deutsch</a>

This tutorial shows the first lifetime-data path in Veridist: fitting an exponential distribution<sup id="fnref-fitting"><a href="#fn-fitting">1</a></sup> when some observations are right censored<sup id="fnref-right-censoring"><a href="#fn-right-censoring">2</a></sup>. Use it when each row says either “the event happened at this time” or “the event had not happened yet by this time”.

## The data idea

A lifetime row has two fields:

<table class="docutils" width="100%">
  <thead>
    <tr>
      <th width="30%" align="center"><p align="center">CSV field</p></th>
      <th width="70%" align="center"><p align="center">Meaning</p></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code class="literal">time</code></td>
      <td>A finite, non-negative amount of observed time.</td>
    </tr>
    <tr>
      <td><code class="literal">event_observed</code></td>
      <td><code class="literal">1</code> means the event happened at <code class="literal">time</code>; <code class="literal">0</code> means the event had not happened by <code class="literal">time</code>.</td>
    </tr>
  </tbody>
</table>

<!-- Markdown table contract: | CSV field | Meaning | -->

For example, a row `1,1` means the event happened at time 1. A row `1,0` means the unit was observed until time 1 and was still alive, working, or event-free at that point. That second row is still useful: it tells the model the lifetime is longer than 1. This is independent right censoring.

## Run the example

The schema declaration below names the only accepted header pair and keeps machine-readable identifiers left to right.

```python
from veridist import CsvLifetimeSchema

schema = CsvLifetimeSchema("time", "event_observed")
```

The complete runnable example uses that schema with a small CSV file.

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

## Interpret the result

The example has one observed event and two total units of observed time: one unit from the exact event row, and one unit from the right-censored row. The exponential maximum-likelihood estimate<sup id="fnref-mle"><a href="#fn-mle">3</a></sup> is therefore:

```{math}
\widehat{rate} = r / \tau = 1 / 2 = 0.5
```

The rate<sup id="fnref-rate"><a href="#fn-rate">4</a></sup> is measured in events per unit of time. It is not a 50 percent event probability. If your time unit is hours, the rate is events per hour; if your time unit is days, it is events per day.

## What Veridist assumes

The model assumes that censoring is independent of the event time. In plain language, the reason observation stopped should not itself reveal hidden information about when the event would happen. Veridist records and documents this assumption; it cannot prove it from the CSV file.

The current public path estimates a fixed-location exponential model from strict UTF-8 CSV. It does not guess column names, delimiters, encoding, missing data, or the meaning of zero and one. If the file does not match the contract, the adapter reports the problem instead of silently rewriting the data.

## Failure cases and current limits

A valid statistical estimate is not returned when the sample is empty, no event is observed, an event appears with zero total observed time, or numeric overflow occurs. These cases return typed non-estimates<sup id="fnref-non-estimate"><a href="#fn-non-estimate">5</a></sup> so calling software can distinguish them from file-reading failures.

This path has no confidence interval today and does not currently provide confidence intervals<sup id="fnref-confidence-interval"><a href="#fn-confidence-interval">6</a></sup>, goodness-of-fit tests<sup id="fnref-goodness-of-fit"><a href="#fn-goodness-of-fit">7</a></sup>, truncation<sup id="fnref-truncation"><a href="#fn-truncation">8</a></sup>, left censoring<sup id="fnref-left-censoring"><a href="#fn-left-censoring">9</a></sup>, interval censoring<sup id="fnref-interval-censoring"><a href="#fn-interval-censoring">10</a></sup>, weights<sup id="fnref-weight"><a href="#fn-weight">11</a></sup>, covariates<sup id="fnref-covariate"><a href="#fn-covariate">12</a></sup>, a free location parameter<sup id="fnref-free-location"><a href="#fn-free-location">13</a></sup>, or automatic model selection.

<details>
<summary>Technical scale and evidence notes</summary>

The CSV adapter consumes one iterator pass. Its chunk byte budget is a logical bound on retained payload, not a portable process-memory or RSS limit and not a throughput claim.

The historical `SCALE-CSV-EXP-01` evidence keeps a 10k, 100k, and 1m row snapshot with 32 KiB, 64 KiB, and 128 KiB chunk sizes for this strict adapter and estimator only. That older schema does not have enough provenance<sup id="fnref-provenance"><a href="#fn-provenance">14</a></sup> to support a current performance claim. A current performance claim needs a clean run tied to the reviewed source revision and documented environment.

The historical snapshot does not establish general big-data support, another adapter, cancellation, retry, or checkpointing. For save-and-resume behavior, use the API guide and the dedicated checkpoint example.

</details>

## Terms used in this tutorial

<p id="fn-fitting"><strong>1.</strong> <bdi>Distribution fitting</bdi> — estimating distribution parameters from data and checking whether the distribution is a defensible description of the observations. <a href="#fnref-fitting" aria-label="Back to text">↩</a></p>
<p id="fn-right-censoring"><strong>2.</strong> <bdi>Right censoring</bdi> — an observation where the event has not happened by the last observed time, so the exact lifetime is known only to be longer than that time. <a href="#fnref-right-censoring" aria-label="Back to text">↩</a></p>
<p id="fn-mle"><strong>3.</strong> <bdi>Maximum-likelihood estimate</bdi> — the parameter value that makes the observed data most compatible with the chosen statistical model. <a href="#fnref-mle" aria-label="Back to text">↩</a></p>
<p id="fn-rate"><strong>4.</strong> <bdi>Rate</bdi> — the expected number of events per unit of time; it is not the same as event probability. <a href="#fnref-rate" aria-label="Back to text">↩</a></p>
<p id="fn-non-estimate"><strong>5.</strong> <bdi>Typed non-estimate</bdi> — a structured result saying the input was processed, but the data did not allow a valid statistical estimate. <a href="#fnref-non-estimate" aria-label="Back to text">↩</a></p>
<p id="fn-confidence-interval"><strong>6.</strong> <bdi>Confidence interval</bdi> — an interval that expresses uncertainty around an estimate under a specified statistical method. <a href="#fnref-confidence-interval" aria-label="Back to text">↩</a></p>
<p id="fn-goodness-of-fit"><strong>7.</strong> <bdi>Goodness-of-fit test</bdi> — a statistical check of how well the selected distribution shape matches the data. <a href="#fnref-goodness-of-fit" aria-label="Back to text">↩</a></p>
<p id="fn-truncation"><strong>8.</strong> <bdi>Truncation</bdi> — a sampling process where part of the population cannot enter the observed sample. <a href="#fnref-truncation" aria-label="Back to text">↩</a></p>
<p id="fn-left-censoring"><strong>9.</strong> <bdi>Left censoring</bdi> — a case where the event is known to have happened before a specified time. <a href="#fnref-left-censoring" aria-label="Back to text">↩</a></p>
<p id="fn-interval-censoring"><strong>10.</strong> <bdi>Interval censoring</bdi> — a case where the event is known to have happened between two times. <a href="#fnref-interval-censoring" aria-label="Back to text">↩</a></p>
<p id="fn-weight"><strong>11.</strong> <bdi>Weight</bdi> — a number that makes an observation contribute more or less to a calculation. <a href="#fnref-weight" aria-label="Back to text">↩</a></p>
<p id="fn-covariate"><strong>12.</strong> <bdi>Covariate</bdi> — an additional variable, such as temperature or pressure, that may help explain lifetime. <a href="#fnref-covariate" aria-label="Back to text">↩</a></p>
<p id="fn-free-location"><strong>13.</strong> <bdi>Free location parameter</bdi> — a shift parameter estimated from data instead of being fixed in advance. <a href="#fnref-free-location" aria-label="Back to text">↩</a></p>
<p id="fn-provenance"><strong>14.</strong> <bdi>Provenance</bdi> — recorded information about source revision, environment, and execution so a claim can be checked later. <a href="#fnref-provenance" aria-label="Back to text">↩</a></p>
