# Competitive feature matrix

**Status: Internal Draft / not publishable.** The URLs and compressed cells are
working evidence notes, not a public comparison. Publication requires 100%
source-lock coverage of every `supported` and `not_supported` CSV claim under
[competitive-evidence-policy.md](competitive-evidence-policy.md).

The CSV is authoritative for the long-form schema: `capability_status` is
`supported`, `partial`, `not_supported`, `unverified`, or `not_applicable`;
`evidence_quality` is `primary`, `official_secondary`, `paper`, or
`unverified`. Versioned/tagged source URLs are preferred. Every mutable source
has `retrieved_at`; no retained snapshot/hash or source-lock registry exists
yet for this draft.
`GoF-refit` requires refitting parameters in every null simulation. Links are
evidence, not independently audited statistical-quality claims.

For compact tables below, **V** maps to `supported`, **P** to `partial`, and
**U** to `unverified`; `not_supported` is written out. The CSV, not these
compressed cells, is the publication/export record.

## Current veridist branch boundary

This is a project-status note, not a competitor claim and not a new
`supported` CSV cell. Release 1.0.0 has the bounded reliability, scalar-family,
checkpoint, and exponential-inference cells listed in the project capability
matrix. It does not establish an external-memory, distributed, universal
best-fit, or broad out-of-core claim. The complete project matrix is
[capability-guide.md](capability-guide.md).

## Python and Julia

| Tool (version/date) | Fit and data semantics | Inference/selection | Scale/interface | Status/source |
| --- | --- | --- | --- | --- |
| SciPy `stats` current docs (retrieved 2026-08-20) | MLE/MM; continuous censored MLE supported; mixture representation partial | KS/AD/CvM refit Monte-Carlo for finite uncensored data supported | NumPy; external-memory/GPU GoF contract unverified | [source](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html) |
| statsmodels stable (2026 census) | duration/survival/regression adjacent, not generic selector P | model-specific inference P | pandas/NumPy oriented P | [P](https://www.statsmodels.org/stable/duration.html) |
| OpenTURNS 1.27 docs (2026) | factories, MLE/MoM, uni/multivariate distributions V | Lilliefors/KS selection and estimator distribution V | C++ core; external-memory contract U | [V](https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.DistributionFactory.html) |
| SurPyval latest docs (2026) | MLE/MPP/MSE/MoM/MPS; censoring/truncation; mixtures P/V | confidence bounds/plots P; GoF-refit U | array API; streaming U | [P](https://surpyval.readthedocs.io/) |
| reliability latest docs (2026) | reliability-family fit and right censoring P | probability plots/CI P; GoF-refit U | in-memory P | [P](https://reliability.readthedocs.io/_/downloads/en/latest/pdf/) |
| fitter 1.8.0 docs (2026) | scans SciPy distributions P | SSE/AIC/BIC/KS displayed; calibrated GoF U | full scans noted slow; streaming U | [P](https://fitter.readthedocs.io/en/latest/references.html) |
| distfit latest docs (2026) | multi-family fitting P | tests/ranking P; GoF-refit U | no verified streaming contract | [P](https://erdogant.github.io/distfit/) |
| Phitter JOSS 2025 | broad continuous/discrete fitting P | three GoF tests P; refit calibration U | benchmark/scale contract U | [P](https://joss.theoj.org/papers/10.21105/joss.07625) |
| lifelines latest docs (2026) | survival/covariate models, not general selection P | survival diagnostics/inference P | dataframe-oriented P | [P](https://lifelines.readthedocs.io/) |
| lmoments3 1.0.8 (2024-10-18) | sample L-moments + selected fits V | GoF/censoring/streaming U | arrays P | [V](https://lmoments3.readthedocs.io/stable/api.html) |
| pyextremes latest docs (2026) | EVT workflow P | EVT diagnostics P | external-memory U | [P](https://georgebv.github.io/pyextremes/) |
| PreliZ/PyMC current docs (2026) | prior elicitation/Bayesian modelling, not automatic general fitting P | posterior predictive checking P | backend dependent P | [P](https://preliz.readthedocs.io/), [P](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/posterior_predictive.html) |
| Julia Distributions.jl (generated 2026-06-26) | `fit`, weighted `fit_mle`, `suffstats` for listed families V | MAP/conjugate support P; generic GoF U | arrays; external-memory U | [V](https://juliastats.org/Distributions.jl/latest/fit/) |

## R

| Tool (version/date) | Fit/data semantics | Inference/selection | Scale/interface | Status/source |
| --- | --- | --- | --- | --- |
| fitdistrplus docs 2026-06-30 | custom d/p/q; MLE/MME/QME/MGE; weights; L/R/interval censored MLE | bootstrap, AIC/BIC/plots; censored `gofstat` not supported | R/S3; parallel bootstrap partial | [source](https://lbbe-software.github.io/fitdistrplus/articles/fitdistrplus_vignette.html) |
| MASS `fitdistr` R-devel | specified-family MLE P | SE/logLik P | R API | [P](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/fitdistr.html) |
| survival `survreg` R-devel | parametric survival regression P | model inference P | R API | [P](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survreg.html) |
| flexsurv current CRAN docs | parametric survival and covariates V | survival inference P | R API | [V](https://stat.ethz.ch/CRAN/web/packages/flexsurv/refman/flexsurv.html) |
| lmomco 2025 citation | L-, censored/trimmed L-moments supported | hydrology/extremes tools partial | R API | [source](https://cran.r-project.org/web/packages/lmomco/citation.html) |
| actuar/gamlss/EnvStats | actuarial, distributional-regression and environmental niches P | family-specific P | R APIs | [P](https://cran.r-project.org/package=actuar), [P](https://cran.r-project.org/package=gamlss), [P](https://cran.r-project.org/package=EnvStats) |
| evd/extRemes | EVT distributions and modelling P | EVT diagnostics/inference P | R APIs | [P](https://cran.r-project.org/package=evd), [P](https://cran.r-project.org/package=extRemes) |

## Commercial, GUI and other platforms

| Tool (version/date) | Verified capability | Important boundary | Status/source |
| --- | --- | --- | --- |
| MATLAB Statistics Toolbox R2026a | `fitdist`/`mle`, custom distributions, frequency, grouping, censoring/truncation/CI; GPU/codegen subsets V | family/data support is function-specific | [V](https://www.mathworks.com/help/stats/fitdist.html), [V](https://www.mathworks.com/help/stats/mle.html) |
| Wolfram Language current docs | `FindDistribution` univariate forms; `EstimatedDistribution` parameter estimation V | censoring/GoF calibration not established here | [P](https://reference.wolfram.com/language/ref/FindDistribution.html) |
| SAS PROC SEVERITY / HPSEVERITY | custom continuous models, MLE, selection, L/R censoring/truncation, BY, covariates, threading/scoring V | estimated-parameter GoF calibration U | [V](https://support.sas.com/documentation/cdl/en/etsug/68148/HTML/default/etsug_severity_overview.htm) |
| SAS PROC LIFEREG 2024 guide | parametric AFT/lifetime regression P | not a general candidate selector | [P](https://documentation.sas.com/api/collections/pgmsascdc/v_051/docsets/statug/content/lifereg.pdf?locale=da) |
| JMP 19 Life Distribution (2025) | right/interval censoring, Bayesian fits, mixtures, competing risks, comparisons V | public refit-GoF calibration U | [V](https://www.jmp.com/support/help/en/19.0/jmp/life-distribution.shtml) |
| Minitab Individual Distribution ID | 14 distributions, probability plots and GoF documented; estimator scope partial | custom families/censoring/refit GoF unverified | [source](https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/quality-tools/how-to/individual-distribution-identification/before-you-start/overview/) |
| SPSS Statistics | generic native distribution-selection workflow U | extensions may exist; absence is not claimed | [U](https://www.ibm.com/docs/en/spss-statistics) |
| Stata | parametric survival/regression P | generic selector not established | [P](https://www.stata.com/manuals/ststreg.pdf) |
| EasyFit, ExpertFit, Stat::Fit, Arena, @RISK, Crystal Ball, Statgraphics, XLSTAT | product/version-specific fitting claims U | no release comparison before primary manual captured | [U](https://www.mathwave.com/easyfit-distribution-fitting.html) |

## Evaluation-only versus fitting

PDF/CDF/random-number libraries, hypothesis-test functions, histogram tools and
mixture *representations* are not automatically fitting products. Count a tool
as a fitter only when its source documents parameter estimation from data;
count calibrated GoF only when refitting/simulation or an applicable cited
correction is documented.

## Cross-cutting platform dimensions

| Dimension | Source-backed observation | Census conclusion |
| --- | --- | --- |
| Out-of-core / exact streaming | Julia documents sufficient-statistics APIs but array inputs; direct fitter docs above do not document a portable external-memory equality contract | **U** for a broad category-wide absence claim; a veridist contract must be published and benchmarked |
| Parallel / GPU | SAS documents multithreaded severity fitting; MATLAB R2026a documents GPU support for specified fit paths | **V/P**; performance is feature- and hardware-specific |
| Failure / convergence accounting | SciPy returns fit result status; fitdistrplus `bootdistcens` removes `NA` failed fits and reports convergence count | **V** for these local behaviours, not a quality ranking |
| API / GUI / web | Python/R/Julia APIs and MATLAB/JMP/Minitab GUIs are documented; Phitter has a public project but web workflow scope is not audited | **P**; ergonomics needs task-based evaluation |
| Data formats / adapters | MATLAB imports workspace data; R/Python tools generally accept in-memory objects | **P**; no evidence here for Arrow/Dask/Polars/DuckDB semantics |
| Export / SQL | MATLAB exports a fitted object to workspace; SAS documents scoring functions | **P**; generic dialect-safe SQL export is **U** in this census |
| i18n / documentation | product documentation languages exist unevenly, but complete fitting-workflow i18n was not audited | **U**; veridist EN/FA/DE is a product requirement, not a proven differentiator |
| License / platform | open-source and commercial tools coexist, but exact current license tiers were not uniformly audited | **U** except where a release-specific source is captured |
