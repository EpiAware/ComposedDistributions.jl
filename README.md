# ComposedDistributions <img src="docs/src/assets/logo.svg" width="150" alt="ComposedDistributions logo" align="right">

<!-- badges:start -->
| **Documentation** | **Build Status** | **Code Quality** | **License & DOI** | **Downloads** |
|:-----------------:|:----------------:|:----------------:|:-----------------:|:-------------:|
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://composeddistributions.epiaware.org/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://composeddistributions.epiaware.org/dev/) | [![Test](https://github.com/EpiAware/ComposedDistributions.jl/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/EpiAware/ComposedDistributions.jl/actions/workflows/test.yaml) [![codecov](https://codecov.io/gh/EpiAware/ComposedDistributions.jl/graph/badge.svg)](https://codecov.io/gh/EpiAware/ComposedDistributions.jl) [![AD](https://github.com/EpiAware/ComposedDistributions.jl/actions/workflows/ad.yaml/badge.svg?branch=main)](https://github.com/EpiAware/ComposedDistributions.jl/actions/workflows/ad.yaml) | [![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![JET](https://img.shields.io/badge/%E2%9C%88%EF%B8%8F%20tested%20with%20-%20JET.jl%20-%20red)](https://github.com/aviatesk/JET.jl) | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) | [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FComposedDistributions&query=total_requests&label=Downloads)](https://juliapkgstats.com/pkg/ComposedDistributions) [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FComposedDistributions&query=total_requests&suffix=%2Fmonth&label=Downloads)](https://juliapkgstats.com/pkg/ComposedDistributions) |

| ForwardDiff | ReverseDiff (tape) | Enzyme forward | Enzyme reverse | Mooncake reverse | Mooncake forward |
|:---:|:---:|:---:|:---:|:---:|:---:|
| [![cov ForwardDiff](https://codecov.io/gh/EpiAware/ComposedDistributions.jl/graph/badge.svg?flag=ad-forwarddiff)](https://app.codecov.io/gh/EpiAware/ComposedDistributions.jl?flags%5B0%5D=ad-forwarddiff) | [![cov ReverseDiff](https://codecov.io/gh/EpiAware/ComposedDistributions.jl/graph/badge.svg?flag=ad-reversediff)](https://app.codecov.io/gh/EpiAware/ComposedDistributions.jl?flags%5B0%5D=ad-reversediff) | [![cov Enzyme forward](https://codecov.io/gh/EpiAware/ComposedDistributions.jl/graph/badge.svg?flag=ad-enzyme-forward)](https://app.codecov.io/gh/EpiAware/ComposedDistributions.jl?flags%5B0%5D=ad-enzyme-forward) | [![cov Enzyme reverse](https://codecov.io/gh/EpiAware/ComposedDistributions.jl/graph/badge.svg?flag=ad-enzyme-reverse)](https://app.codecov.io/gh/EpiAware/ComposedDistributions.jl?flags%5B0%5D=ad-enzyme-reverse) | [![cov Mooncake reverse](https://codecov.io/gh/EpiAware/ComposedDistributions.jl/graph/badge.svg?flag=ad-mooncake-reverse)](https://app.codecov.io/gh/EpiAware/ComposedDistributions.jl?flags%5B0%5D=ad-mooncake-reverse) | [![cov Mooncake forward](https://codecov.io/gh/EpiAware/ComposedDistributions.jl/graph/badge.svg?flag=ad-mooncake-forward)](https://app.codecov.io/gh/EpiAware/ComposedDistributions.jl?flags%5B0%5D=ad-mooncake-forward) |
<!-- badges:end -->

A verb grammar for n-ary composition over any `Distributions.jl` distribution.

## Why ComposedDistributions?

- A natural history is usually a hand-rolled convolution or simulation loop
  per project; ComposedDistributions gives it one small vocabulary — chains,
  branches, and one_of outcomes — so the model is built by naming its
  structure, not by re-deriving the maths.
- The same composed object scores an observed record and simulates a new one,
  so a model built once serves calibration and forward simulation alike.
- Parameter uncertainty is an ordinary leaf, not a separate PPL-specific
  layer, so a delay's literature uncertainty is written once and reused
  whether it is sampled or just drawn from.
- Every leaf is an ordinary Distributions.jl distribution, so nothing needs
  reimplementing to compose it, and the composed result is itself a
  distribution that drops into code expecting one.
- A composed tree is inspectable, editable data — a parameter table, a nested
  prior, a rendered tree — rather than an opaque function, so its structure
  and priors can be read and changed without rereading the model code.
- Convolution, quadrature and automatic differentiation across four backends
  come with the package, so a composed delay is fit-ready with no extra
  plumbing.

## Getting started

See [documentation](https://composeddistributions.epiaware.org/stable/) for a full walkthrough.

A hospital pathway: an admission delay with literature uncertainty, then a
death-versus-discharge split where the death probability is the case-fatality
ratio, alongside a separate delay to public reporting.

```julia
using ComposedDistributions, Distributions, Random

cfr = 0.12   # case-fatality ratio among admitted cases

admission = compose((
    path = sequential(
        :onset_admit => uncertain(LogNormal(1.5, 0.4); mu = Normal(1.5, 0.2)),
        :admit_outcome => resolve(:death => (Gamma(1.5, 1.0), cfr),
            :discharge => Gamma(2.0, 1.5))),
    onset_report = Gamma(1.5, 1.0)))
```

`admission` prints as the tree it is: two branches off the onset, the
admission branch itself a two-step chain ending in the death/discharge split.

```julia
admission
```

The same object simulates a structured record and scores one straight back.

```julia
record = rand(Xoshiro(1), admission)
```

```julia
logpdf(admission, record)
```

The [getting started guide](https://composeddistributions.epiaware.org/stable/getting-started/)
carries this same tree further: reading its parameter table, editing a
node, collapsing a chain to its observed total, and fitting it.

## Relationship to Distributions.jl

ComposedDistributions builds on Distributions.jl rather than replacing it.
Every leaf is a Distributions.jl `UnivariateDistribution`, and a composed object is itself a `Distribution`, so `logpdf`, `rand`, `mean`, `var` and the rest of the interface work unchanged.

| Aspect | Distributions.jl | ComposedDistributions |
|--------|------------------|-----------------------|
| **Scope** | one distribution | many delays wired into an event tree |
| **Question** | "what is this delay?" | "how do these events relate?" |
| **Builds on** | — | any Distributions.jl `UnivariateDistribution` as a leaf |
| **Adds** | — | `compose`, the five composers, a parameter table and structural edits |

Because a composed object is a `Distribution`, it also works with `truncated()` from Distributions.jl and drops into any code that expects a distribution.

## What packages work well with ComposedDistributions?

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) supplies the leaf distributions and the interface a composed object implements.
- [ConvolvedDistributions.jl](https://github.com/EpiAware/ConvolvedDistributions.jl) is re-exported, so convolution (`convolved`, `convolve_series`, `difference`) and quadrature are reachable through ComposedDistributions alone.
- [Tables.jl](https://github.com/JuliaData/Tables.jl) sources build a composer through `compose`, and `params_table` returns a Tables.jl table.
- [Turing.jl](https://github.com/TuringLang/Turing.jl) and the wider probabilistic-programming ecosystem, where automatic-differentiation-friendly scoring lets a composed distribution drop into a Bayesian fit.

## Where to learn more

- Want to get started running code? See the [getting started guide](https://composeddistributions.epiaware.org/stable/getting-started/).
- Want the right verb by intent? See the [Concepts](https://composeddistributions.epiaware.org/stable/getting-started/concepts) page.
- Want to understand the API? See the [API reference](https://composeddistributions.epiaware.org/stable/lib/public).
- Want to see the code? Check out our [GitHub repository](https://github.com/EpiAware/ComposedDistributions.jl).

## Getting help

For usage questions, ask on the [Julia Discourse](https://discourse.julialang.org)
(the SciML or usage categories) or the [epinowcast community forum](https://community.epinowcast.org),
our home for epidemiological modelling questions.
Please use [GitHub issues](https://github.com/EpiAware/ComposedDistributions.jl/issues)
for bug reports and feature requests only.

## Contributing

We welcome contributions and new contributors! This package follows [ColPrac](https://github.com/SciML/ColPrac) and the [SciML style](https://github.com/SciML/SciMLStyle).

## Supporting and citing

If you would like to support ComposedDistributions, please star the repository — such metrics help secure future funding.

If you use ComposedDistributions in your work, please cite it:

```bibtex
@software{ComposedDistributions_jl,
  author       = {Sam Abbott and EpiAware contributors},
  title        = {ComposedDistributions.jl},
  year         = {2026},
  doi          = {10.5281/zenodo.XXXXXXX}, # replace once released
  url          = {https://github.com/EpiAware/ComposedDistributions.jl}
}
```

## Code of conduct

Please note that the ComposedDistributions project is released with a [Contributor Code of Conduct](https://github.com/EpiAware/.github/blob/main/CODE_OF_CONDUCT.md). By contributing, you agree to abide by its terms.
