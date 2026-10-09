(veridist-families-log-density-likelihood)=
# Evaluated families, scalar log density, and streaming log likelihood

## Closed six-family registry

`FAMILY_REGISTRY` is immutable metadata for exactly six evaluated families:
`exponential`, `normal`, `gamma`, `weibull_min`, `lognormal`, and
`gumbel_right`. It fixes canonical names, aliases, parameter names, validation
roles, the support, and the operations each family offers: log density, CDF,
survival function, quantile, sampling, and fit. It is not a generic
distribution registry, and it does not rank distributions.

For the contract-level API surface, see
<a class="veridist-api-url" href="api.html#veridist-api">veridist.statistics.log_density</a>.

| Family | Canonical parameters |
| --- | --- |
| `exponential` | `rate > 0` |
| `normal` | `mu`, `sigma > 0` |
| `gamma` | `shape > 0`, `scale > 0` |
| `weibull_min` | `shape > 0`, `scale > 0` |
| `lognormal` | `mu_log`, `sigma_log > 0` |
| `gumbel_right` | `location`, `scale > 0` |

## Where these calculations can help

These tools answer a narrow question: how compatible is an individual value, or a collection of values, with a specified family and parameter set? They do not estimate those parameters (the fits do) and they do not turn a low-density value into a business decision.

| Application area | Example value | Defensible use of this API |
| --- | --- | --- |
| Finance and insurance example | A return, loss amount, or claim size | Evaluate log density under an already justified reference model or combine log likelihood across chunks. |
| Fraud and cybersecurity example | A transaction amount, time gap, or login latency | Produce one anomaly signal under a reference model; low density is not itself a probability of fraud. |
| Manufacturing-quality example | A dimension, load, or cycle measurement | Compare measurements with a validated process model and identify values that need review. |
| Operations and supply-chain example | A delivery, repair, waiting, or service duration | Accumulate likelihood for a specified duration model without keeping every value in one array. |
| Environmental-risk example | A high load, loss, or extreme measurement | Evaluate the right-Gumbel log density when that family and its parameters have been justified independently. |
| Machine-learning example | A distribution-based feature | Supply a model feature only when its parameters are estimated without future or test-set information. |

Different fields have different sampling rules, costs, thresholds, and regulatory requirements. Validate the reference model and the downstream decision process in the field where they will be used.

## Scalar log-density evaluation

`evaluate_log_density` evaluates one built-in finite scalar observation after
canonical parameter validation. A successful result contains a finite binary64
log-density. Data-domain failures are closed values: `nonfinite_observation`,
`support_violation`, `nonfinite_log_density`, or `numerical_overflow`.
Programmer misuse of a family identity or parameter set raises rather than
being converted into a data result. It evaluates one scalar and has no
censoring; arrays, the CDF, the survival function, quantiles, and sampling are
provided by `logpdf`, `cdf`, `sf`, `ppf`, and `sample`, and fitting by `fit`, as
described in the API guide. It is not a goodness-of-fit test or a model ranking.

```python
from veridist.families.registry import FAMILY_REGISTRY, FamilyId
from veridist.statistics.log_density import LogDensityFailure, evaluate_log_density

assert FAMILY_REGISTRY.resolve("gaussian").id is FamilyId.NORMAL
assert evaluate_log_density(FamilyId.NORMAL, 0.0, mu=0.0, sigma=1.0).log_density < 0.0
failure = evaluate_log_density(FamilyId.GAMMA, 0.0, shape=2.0, scale=1.0)
assert isinstance(failure, LogDensityFailure)
assert failure.code.value == "support_violation"
```

## Exact-state streaming log likelihood

`reduce_log_likelihood_chunks` consumes an iterable of ragged observation
chunks once. It evaluates each scalar through the same closed evaluator and
accumulates each successful binary64 result as an exact integer number of
subnormal units. The final sum is rounded to binary64 once, at finalization;
it is not a chunk-wise floating-point sum.

```{math}
\operatorname{LL} = \operatorname{round}_{binary64}\left(\sum_i \log f(x_i)\right)
```

The immutable `LogLikelihoodState` supports compatible exact-state merging.
One state has an explicit unsigned-64 observation limit (`2^64 - 1`); its
maximum absolute exact integer total has a documented 2162-bit bound. A scalar
failure stops reduction and reports only the number of successful observations
before it. An empty chunk sequence succeeds with count zero and total `0.0`.

```python
from veridist.families.registry import FamilyId
from veridist.statistics.log_likelihood import reduce_log_likelihood_chunks

success = reduce_log_likelihood_chunks(
    FamilyId.NORMAL, ((0.0, 1.0), (), (-1.0,)), mu=0.0, sigma=1.0
)
assert success.observation_count == 3
empty = reduce_log_likelihood_chunks(FamilyId.NORMAL, ((),), mu=0.0, sigma=1.0)
assert empty.observation_count == 0
failed = reduce_log_likelihood_chunks(FamilyId.GAMMA, ((1.0,), (0.0,)), shape=2.0, scale=1.0)
assert failed.code.value == "scalar_evaluation_failure"
assert failed.processed_count == 1
```

Right-censored observations have their own reducers,
`reduce_lifetime_log_likelihood_chunks` and `reduce_value_log_likelihood_chunks`.
They add the log survival of each censored observation to the same exact
accumulation, so the total is again independent of chunking and order.

## Evidence and limits

`LLR-06` retains generated-stream measurements of the exact-state
log-likelihood reducer for each of the five scalar families (`normal`,
`gamma`, `weibull_min`, `lognormal`, `gumbel_right`) at 10k/100k/1m
observations and three chunk sizes (1,024, 8,192, and 65,536). They were taken
on 2026-10-09 from candidate commit
`19ecf1062978fe0f894625e65b5a113ba1b68166`, one run each on a Linux runner
(CPython 3.11.17) and a Windows runner (CPython 3.11.9), and a schema-v5
checker accepts them. Each run used the public `IterableDataSource`
single-pass route, bound the candidate SHA, recorded one outer acquisition and
every yielded observation, and agreed bitwise with the independent exact-unit
oracle and 2162-bit algorithmic bound. At 1m observations, elapsed time was
3.2 to 9.2 seconds on Linux and 3.3 to 10.6 seconds on Windows depending on
the family, with peak process memory of 25.8 MiB on Linux and 25.3 MiB on
Windows. Elapsed time and `tracemalloc` values remain descriptive only.

This evidence is scoped to the tested fixed streams of the five scalar
families and the exact reducer, on the two measured runners. It does not
establish fitting, inference, goodness-of-fit, ranking, arrays, censoring,
generic out-of-core processing, a throughput or process-memory/RSS guarantee,
another data adapter, or performance on other machines.
