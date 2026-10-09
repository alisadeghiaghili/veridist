(veridist-home)=
# Veridist documentation

Evidence-first, deliberately narrow distribution primitives.

## Applications across fields

The same distribution tools can support reliability, survival research, credit and insurance, fraud and cybersecurity, operations, supply chains, digital products, and environmental risk when the data and assumptions match the documented contract. “Lifetime” can mean time until any defined event, such as failure, relapse, default, a first claim, churn, or process completion. Scalar distribution calculations can also contribute signals for anomaly detection, but Veridist does not replace domain validation or provide an end-to-end fraud detector.

## First CSV vertical

The first public vertical fits a fixed-location exponential lifetime model from
a UTF-8 CSV source with an explicit input format. Its canonical parameter is a positive rate; the
reported mean is derived as its reciprocal.

## Limits

The CSV schema is exactly `time,event_observed`; event `1` is exact and `0` is
independent right censoring. The adapter makes one iterator pass and retains at
most its declared logical payload budget. `inference=not_provided` means the
result object itself carries no inference output; request standard errors and
confidence intervals with `uncertainty()`. The CSV path supplies no
goodness-of-fit result, truncation, weights, covariates, or free location
parameter.

## Executable example

The example below has one canonical executable source and reports only stable
machine facts. Use an opaque public source identifier; file paths are not part
of the returned provenance.

Translations for this vertical are owner-reviewed provisional text; no external
native-speaker review is claimed.

## Six evaluated families

The public kernel has an immutable six-family registry, a maximum-likelihood
fit for each family with right censoring and uncertainty reporting,
distribution operations that accept scalars and numpy arrays, and exact-state
streaming log-likelihood reducers with and without right censoring. It does
not provide goodness-of-fit tests or model ranking.

## Generic stream sources and delivery leases

`IterableDataSource` adapts caller-owned chunk iterables with immutable source
metadata. A `single_pass` source is acquired once and a second acquisition
raises a typed pass-budget failure; replayable declarations require an explicit
iterator factory. The CSV lifetime adapter with the documented input format remains the only bundled file
adapter. `BoundedChunkBuffer` charges a chunk while queued and while held by a
consumer, until `BufferedChunk.release()`; consumers must release received
chunks, normally in `finally`. This is not a generic CSV, Parquet, Arrow,
dataframe, database, RSS, throughput, or broad out-of-core claim.

```{literalinclude} examples/quickstart.py
:language: python
:caption: Canonical executable example
```

```{toctree}
:maxdepth: 2

api
exponential-right-censoring
families-log-density-likelihood
```
